"""AUTOENCODIX-native downstream evaluation of sampled VARIX latents."""

from __future__ import annotations

from pathlib import Path

from autoencodix.varix import Varix
from autoencodix.evaluate._general_evaluator import GeneralEvaluator
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass(frozen=True)
class DownstreamMetricConfig:
    n_draws: int = 10
    sampling_seed: int = 31

    def __post_init__(self):
        if type(self.n_draws) is not int or (self.n_draws != 0 and self.n_draws < 2):
            raise ValueError("n_draws must be zero or at least two")
        if type(self.sampling_seed) is not int or not 0 <= self.sampling_seed < 2**32:
            raise ValueError("sampling_seed must be a nonnegative 32-bit integer")


METRIC_COLUMNS = (
    "metric_name", "value", "status", "warning", "n_samples", "split",
)
TARGETS = ("CANCER_TYPE_ACRONYM", "AJCC_PATHOLOGIC_TUMOR_STAGE_SHORT", "SEX", "AGE")


class ClassAUCEvaluator(GeneralEvaluator):
    """Record per-class OvR AUC using the classifier fitted by AUTOENCODIX."""

    def _single_ml_presplit(self, sample_split, df, clin_data, task_param,
                           sklearn_ml, metric, ml_type, top_k_classes=20):
        # AUTOENCODIX fills absent annotations with the string "missing".
        # Exclude placeholders per task, without mutating other tasks' cohorts.
        eligible = _clean(clin_data[task_param]).dropna().index.intersection(df.index)
        sample_split = sample_split.loc[sample_split.SAMPLE_ID.isin(eligible)]
        results = super()._single_ml_presplit(
            sample_split, df, clin_data, task_param, sklearn_ml,
            metric, ml_type, top_k_classes,
        )
        if ml_type != "classification" or results.empty:
            return results
        if not hasattr(self, "class_auc_records"):
            self.class_auc_records = []
        train_ids = sample_split.loc[sample_split.SPLIT == "train", "SAMPLE_ID"]
        train_labels = clin_data.loc[train_ids, task_param]
        selected_classes = None
        if top_k_classes is not None and train_labels.nunique() > top_k_classes:
            selected_classes = train_labels.value_counts().nlargest(top_k_classes).index
            train_labels = train_labels.where(train_labels.isin(selected_classes), "other")
        for split in ("valid", "test"):
            ids = sample_split.loc[sample_split.SPLIT == split, "SAMPLE_ID"]
            if len(ids) == 0:
                continue
            labels = clin_data.loc[ids, task_param]
            if selected_classes is not None:
                labels = labels.where(labels.isin(selected_classes), "other")
            labels = labels[labels.isin(sklearn_ml.classes_)]
            if labels.empty:
                continue
            probabilities = sklearn_ml.predict_proba(df.loc[labels.index])
            for column, label in enumerate(sklearn_ml.classes_):
                positive = np.asarray(labels == label)
                n_positive = int(positive.sum())
                n_negative = int(len(positive) - n_positive)
                defined = n_positive > 0 and n_negative > 0
                self.class_auc_records.append({
                    "CLINIC_PARAM": task_param, "score_split": split,
                    "class": label, "metric": "roc_auc_ovr",
                    "value": float(roc_auc_score(positive, probabilities[:, column])) if defined else np.nan,
                    "n_positive": n_positive, "n_negative": n_negative,
                    "n_train_positive": int((train_labels == label).sum()),
                    "status": "ok" if defined else "undefined_single_class",
                    "representation": "sampled_z_from_result_latentspaces",
                })
        return results


def compute_z_logistic_regression_probes_use_split(
    varix: Varix,
    run_name: str,
    targets: tuple[str, ...] = TARGETS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return scalar probe scores and the detailed AUTOENCODIX evaluation table.

    Input representation:
        Sampled VARIX ``z`` representations read from AUTOENCODIX's existing
        train, validation, and test latent-space entries.
    """
    if isinstance(getattr(varix, "evaluator", None), ClassAUCEvaluator):
        varix.evaluator.class_auc_records = []
    varix.evaluate(params=list(targets), reference_methods=[], split_type="use-split",
                   top_k_classes=None, n_downsample=None)
    details = varix.result.embedding_evaluation.copy().reset_index(drop=True)
    details = details.loc[
        details["metric"].isin(["roc_auc_ovo", "r2"])
        & details["CLINIC_PARAM"].isin(targets)
    ].copy()
    identity_columns = ["score_split", "CLINIC_PARAM", "metric"]
    duplicate_mask = details.duplicated(identity_columns, keep=False)
    if duplicate_mask.any():
        conflicting = (
            details.loc[duplicate_mask]
            .groupby(identity_columns, dropna=False)["value"]
            .nunique(dropna=False)
            .gt(1)
        )
        if conflicting.any():
            raise ValueError(
                "AUTOENCODIX returned conflicting downstream scores for the "
                "same target and split"
            )
        details = details.drop_duplicates(identity_columns, keep="last")
    details = details.reset_index(drop=True)
    details.insert(0, "run_name", run_name)
    details["representation"] = "sampled_z_from_result_latentspaces"

    records: list[dict[str, object]] = []
    for row in details.itertuples(index=False):
        split = str(row.score_split)
        if split not in {"valid", "test"}:
            continue
        score = float(row.value)
        finite = np.isfinite(score)
        records.append({
            "metric_name": f"downstream_{row.CLINIC_PARAM}_z_{split}_{row.metric}",
            "value": score if finite else np.nan,
            "status": "ok" if finite else "degenerate_input",
            "warning": None if finite else "AUTOENCODIX returned a non-finite score.",
            "n_samples": None,
            "split": split,
        })
    return pd.DataFrame.from_records(records, columns=METRIC_COLUMNS), details


def run_downstream_metrics_pipeline(
    varix: Varix,
    run_name: str,
    dataset_family: str | None = None,
    output_dir: Path | None = None,
    targets: tuple[str, ...] = TARGETS,
    config: DownstreamMetricConfig | None = None,
) -> pd.DataFrame:
    """Run downstream probes, save their detail table, and return scalar values.

    Input representation:
        Sampled VARIX ``z`` representations stored on the trained ``Varix``.
    """
    if dataset_family == "synthetic_manifold" or not targets:
        return pd.DataFrame(columns=METRIC_COLUMNS)

    downstream, details = compute_z_logistic_regression_probes_use_split(
        varix, run_name, targets=targets
    )
    if output_dir is not None:
        directory = Path(output_dir) / "absolute_core_metrics_v1" / "downstream" / run_name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "z_logistic_regression_use_split.csv"
        details.to_csv(path, index=False)
        print(path)
        class_records = getattr(getattr(varix, "evaluator", None), "class_auc_records", [])
        if class_records:
            class_auc = pd.DataFrame.from_records(class_records)
            class_auc.insert(0, "run_name", run_name)
            class_auc = class_auc.sort_values(
                ["CLINIC_PARAM", "score_split", "value"], na_position="last"
            )
            class_path = directory / "z_class_auc_use_split.csv"
            class_auc.to_csv(class_path, index=False)
            print(class_path)
            for (target, split), group in class_auc.groupby(["CLINIC_PARAM", "score_split"]):
                print(f"Lowest class AUCs ({target}, {split}; one-vs-rest):")
                print(group.head(5)[["class", "value", "n_positive", "status"]].to_string(index=False))
    config = config or DownstreamMetricConfig()
    if config.n_draws:
        if output_dir is None:
            raise ValueError('Repeated z evaluation requires output_dir with saved final posteriors')
        repeated = run_native_z_draws(varix, run_name, output_dir, targets, config)
        downstream = pd.concat([downstream, repeated], ignore_index=True)
    return downstream


def _clean(series):
    missing = series.astype(str).str.strip().str.lower().isin(
        ('', 'unknown', 'missing', 'nan', 'none', 'na', 'n/a', 'not reported',
         '[not available]', '[not applicable]'))
    return series.mask(missing)


def mixed_sex_cancers(train, min_class_train=5, min_sex_fraction=0.05):
    """Train-defined mixed cancers: >=5 per sex and >=5% each by default.

    The fraction is an explicit cohort convention, not a biological threshold.
    It excludes near-single-sex types such as BRCA despite a few male patients.
    Validation/test labels never determine membership.
    """
    if not 0 <= min_sex_fraction <= 0.5:
        raise ValueError('min_sex_fraction must lie in [0, 0.5]')
    counts = pd.crosstab(_clean(train['CANCER_TYPE_ACRONYM']), _clean(train['SEX']))
    counts = counts.reindex(columns=['Female', 'Male'], fill_value=0)
    fractions = counts.div(counts.sum(axis=1), axis=0)
    return counts.index[((counts >= min_class_train) & (fractions >= min_sex_fraction)).all(axis=1)].tolist()


def sex_task_mask(metadata, cancers):
    """Mark known-sex patients in the train-defined mixed-cancer cohort."""
    return (metadata['CANCER_TYPE_ACRONYM'].isin(cancers)
            & metadata['SEX'].isin(['Female', 'Male'])).to_numpy()


def snapshot_tasks(labels, min_class_train=5):
    """Build train-defined clinical cohorts or synthetic ground-truth probes."""
    train = labels['train']
    tasks = []

    def add(name, target, masks=None, regression=False):
        masks = masks or {s: np.ones(len(labels[s]), dtype=bool) for s in labels}
        y = {s: (_clean(labels[s][target]) if target in labels[s]
                 else pd.Series(np.nan, index=labels[s].index)) for s in labels}
        if regression:
            y = {s: pd.to_numeric(v, errors='coerce').replace([np.inf, -np.inf], np.nan)
                 for s, v in y.items()}
        keep = {s: np.asarray(masks[s], dtype=bool) & y[s].notna().to_numpy() for s in labels}
        eligible = {s: v.copy() for s, v in keep.items()}
        if not regression:
            y = {s: v.astype(str) for s, v in y.items()}
            counts = y['train'][keep['train']].value_counts()
            classes = counts[counts >= min_class_train].index
            keep = {s: keep[s] & y[s].isin(classes).to_numpy() for s in labels}
        tasks.append(dict(name=name, target=target, regression=regression, y=y, keep=keep,
                          cohort=masks, eligible=eligible))

    if 'CANCER_TYPE_ACRONYM' in train:
        cancers = {s: _clean(labels[s]['CANCER_TYPE_ACRONYM']) for s in labels}
        for name, target in (('cancer_type', 'CANCER_TYPE_ACRONYM'),
                             ('cancer_group', 'CANCER_TYPE'),
                             ('histology', 'CANCER_TYPE_DETAILED'),
                             ('histology_icd', 'ICD_O_3_HISTOLOGY')):
            add(name, target)
        mixed = mixed_sex_cancers(train, min_class_train) if 'SEX' in train else []
        add('sex_mixed_cancers', 'SEX', {s: cancers[s].isin(mixed).to_numpy()
                                       & labels[s].get('SEX', pd.Series(index=labels[s].index, dtype=object)).isin(['Female', 'Male']).to_numpy()
                                       for s in labels})
        for cancer in ('BRCA', 'UCEC'):
            add(f'subtype_{cancer}', 'SUBTYPE',
                {s: cancers[s].eq(cancer).to_numpy() for s in labels})
        for cancer in sorted(cancers['train'].dropna().unique()):
            masks = {s: cancers[s].eq(cancer).to_numpy() for s in labels}
            if cancer in mixed:
                add(f'sex_{cancer}', 'SEX', masks)
            add(f'stage_{cancer}', 'AJCC_PATHOLOGIC_TUMOR_STAGE_SHORT', masks)
        add('age', 'AGE', regression=True)
    else:
        # Circular coordinates are represented by sin/cos to avoid a wrap discontinuity.
        for split, frame in labels.items():
            if 'ground_truth_angle' in frame:
                frame['circle_sin'], frame['circle_cos'] = np.sin(frame.ground_truth_angle), np.cos(frame.ground_truth_angle)
            if {'ground_truth_polar', 'ground_truth_azimuth'} <= set(frame):
                polar, azimuth = frame.ground_truth_polar, frame.ground_truth_azimuth
                frame['sphere_x'], frame['sphere_y'], frame['sphere_z'] = np.sin(polar)*np.cos(azimuth), np.sin(polar)*np.sin(azimuth), np.cos(polar)
        for target in ('circle_sin', 'circle_cos', 'sphere_x', 'sphere_y', 'sphere_z', 'ground_truth_t'):
            if target in train:
                add(target, target, regression=True)
    return tasks



def evaluate_epoch_downstream(folder, epoch, posteriors, labels, effective_beta):
    """Save AUTOENCODIX-native Train/Validation scores and fitted-model predictions.

    Reuse its presplit evaluator and default model specifications; no test data.
    Use stored native z when supplied, otherwise a fixed local posterior draw.
    """
    from sklearn.linear_model import LogisticRegression, LinearRegression
    output = Path(folder) / f'epoch_{epoch:05d}_downstream'
    output.mkdir(parents=True, exist_ok=True)
    scores, predictions, class_scores = [], [], []
    tasks = snapshot_tasks(labels)
    representation = 'z'
    features = {}
    for split_index, (split, posterior) in enumerate(posteriors.items()):
        values = posterior['mu']
        if 'z' in posterior:
            values = posterior['z']
        else:
            rng = np.random.default_rng(np.random.SeedSequence([31, split_index]))
            values = values + np.exp(.5 * posterior['logvar']) * rng.standard_normal(values.shape)
        features[split] = pd.DataFrame(values, index=labels[split].index)
    for task in tasks:
        frames, annotations, partitions = [], [], []
        for split in ('train', 'valid'):
            keep = task['keep'][split]
            frame = features[split].loc[keep]
            frames.append(frame)
            annotations.append(task['y'][split].loc[keep].rename(task['target']))
            partitions.append(pd.DataFrame(dict(SAMPLE_ID=frame.index, SPLIT=split)))
        frame = pd.concat(frames)
        annotation = pd.concat(annotations).to_frame()
        partition = pd.concat(partitions, ignore_index=True)
        regression = task['regression']
        model = LinearRegression() if regression else LogisticRegression(max_iter=1000)
        metric = 'r2' if regression else 'roc_auc_ovo'
        evaluator = ClassAUCEvaluator()
        try:
            result = evaluator._single_ml_presplit(partition, frame, annotation,
                task['target'], model, metric,
                'regression' if regression else 'classification', top_k_classes=None)
            result = result.assign(status=np.where(np.isfinite(result.value), 'ok', 'unavailable'), warning=None)
            if result.empty:
                raise ValueError('No evaluable training classes')
        except ValueError as exc:
            result = pd.DataFrame([dict(score_split='valid', CLINIC_PARAM=task['target'],
                metric=metric, value=np.nan, status='unavailable', warning=str(exc))])
        scores.append(result.assign(task=task['name'], representation=representation))
        records = getattr(evaluator, 'class_auc_records', [])
        if records:
            class_scores.append(pd.DataFrame(records).assign(task=task['name'], representation=representation))
        # The model is fitted by AUTOENCODIX, including when only scoring fails.
        if hasattr(model, 'coef_'):
            valid = frames[1]
            if len(valid):
                prediction = pd.DataFrame(dict(sample_id=valid.index,
                    truth=annotation.loc[valid.index, task['target']].to_numpy(),
                    prediction=model.predict(valid), task=task['name'], representation=representation))
                if not regression:
                    probabilities = model.predict_proba(valid)
                    for column, label in enumerate(model.classes_):
                        prediction[f'probability:{label}'] = probabilities[:, column]
                predictions.append(prediction)
    for name, frames in (('scores', scores), ('predictions', predictions), ('class_auc', class_scores)):
        table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        table.assign(epoch=epoch, beta=effective_beta).to_csv(output / f'{name}.csv', index=False)


def run_native_z_draws(varix, run_name, output_dir, targets, config):
    """Evaluate fresh posterior draws with AUTOENCODIX and take score medians."""
    from autoencodix.utils._traindynamics import TrainingDynamics
    directory = Path(output_dir) / 'absolute_core_metrics_v1' / 'downstream' / run_name
    posterior_dir = Path(output_dir) / 'posterior_snapshots' / run_name
    posteriors = {}
    for split in ('train', 'valid', 'test'):
        with np.load(posterior_dir / f'posterior_{split}.npz', allow_pickle=False) as saved:
            posteriors[split] = {key: saved[key].copy() for key in ('mu', 'logvar', 'sample_ids')}
        epoch = -1 if split == 'test' else varix.result.model.config.epochs - 1
        ids = varix.result.get_latent_df(epoch=epoch, split=split).index.astype(str)
        if not np.array_equal(ids, posteriors[split]['sample_ids'].astype(str)):
            raise ValueError(f'Posterior sample order differs from AUTOENCODIX: {split}')
    original_latents = varix.result.latentspaces
    original_scores = varix.result.embedding_evaluation
    original_classes = getattr(varix.evaluator, 'class_auc_records', [])
    draws = []
    try:
        for draw in range(config.n_draws):
            varix.result.latentspaces = TrainingDynamics()
            varix.result.embedding_evaluation = pd.DataFrame()
            for split_index, (split, posterior) in enumerate(posteriors.items()):
                rng = np.random.default_rng(np.random.SeedSequence([config.sampling_seed, draw, split_index]))
                z = posterior['mu'] + np.exp(.5 * posterior['logvar']) * rng.standard_normal(posterior['mu'].shape)
                if not np.isfinite(z).all():
                    raise ValueError('Nonfinite posterior draw')
                epoch = -1 if split == 'test' else varix.result.model.config.epochs - 1
                varix.result.latentspaces.add(epoch=epoch, split=split, data=z)
            _, details = compute_z_logistic_regression_probes_use_split(varix, run_name, targets)
            draws.append(details.assign(draw=draw))
    finally:
        varix.result.latentspaces = original_latents
        varix.result.embedding_evaluation = original_scores
        varix.evaluator.class_auc_records = original_classes
    details = pd.concat(draws, ignore_index=True)
    directory.mkdir(parents=True, exist_ok=True)
    details.to_csv(directory / 'z_mc_draws.csv', index=False)
    keys = ['score_split', 'CLINIC_PARAM', 'metric']
    summary = details.groupby(keys, dropna=False).value.agg(['median', 'count']).reset_index()
    complete = summary['count'].eq(config.n_draws)
    summary.loc[~complete, 'median'] = np.nan
    summary['n_draws'] = config.n_draws
    summary['sampling_seed'] = config.sampling_seed
    summary.to_csv(directory / 'z_mc_median.csv', index=False)
    return pd.DataFrame([dict(metric_name=f'downstream_{r.CLINIC_PARAM}_z_mc_{r.score_split}_{r.metric}_median',
        value=r.median, status='ok' if np.isfinite(r.median) else 'unavailable',
        warning=None if np.isfinite(r.median) else 'Incomplete draw scores',
        n_samples=None, split=r.score_split) for r in summary.itertuples(index=False)], columns=METRIC_COLUMNS)
