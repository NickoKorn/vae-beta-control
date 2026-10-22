"""Observe information retention on frozen train/validation posteriors.

Probes are fitted on train only with fixed sklearn hyperparameters. Cohorts
and retained classes are defined on train; test posteriors are never loaded.
These observations do not feed back into beta. Seeds match samples/noise
across epochs; score quantiles or draw variation are not confidence intervals.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.manifold import trustworthiness

from .metrics.dimensionality import compute_effective_rank_mu
from .metrics.geometry import compute_zadu_geometry
from .metrics.information_theoretic import compute_conditional_kl_matrix


@dataclass(frozen=True)
class ObserverConfig:
    enabled: bool = False
    max_samples: int = 512
    k: int = 20
    seed: int = 31
    z_draws: int = 3
    max_iter: int = 1000
    min_class_train: int = 5

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('observer.enabled must be boolean')
        for key in ('max_samples', 'k', 'z_draws', 'max_iter', 'min_class_train'):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f'observer.{key} must be a positive integer')
        if self.max_samples < 2*self.k + 1:
            raise ValueError('observer.max_samples must be at least 2*k+1')
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError('observer.seed must be a nonnegative 32-bit integer')


def aligned_annotations(metadata, sample_ids):
    """Unwrap AUTOENCODIX's annotation mapping and align by IDs, never position."""
    ids = pd.Index(np.asarray(sample_ids).astype(str))
    frames = [metadata] if isinstance(metadata, pd.DataFrame) else (
        list(metadata.values()) if isinstance(metadata, dict) else [])
    if not frames:
        return pd.DataFrame(index=ids)
    columns = {}
    for frame in frames:
        if not isinstance(frame, pd.DataFrame):
            raise ValueError('Annotation mappings must contain DataFrames')
        frame = frame.copy()
        frame.index = frame.index.astype(str)
        if not frame.index.is_unique or not frame.columns.is_unique:
            raise ValueError('Annotation IDs and columns must be unique')
        if not ids.isin(frame.index).all():
            raise ValueError('Annotations do not cover snapshot sample IDs')
        for column in frame:
            values = frame.loc[ids, column]
            if column in columns and not columns[column].equals(values):
                raise ValueError(f'Conflicting annotation column: {column}')
            columns[column] = values
    return pd.DataFrame(columns, index=ids)


def _clean(series):
    missing = series.astype(str).str.strip().str.lower().isin(
        ('', 'unknown', 'missing', 'nan', 'none', 'na', 'n/a', 'not reported',
         '[not available]', '[not applicable]'))
    return series.mask(missing)


def cohort_tasks(labels, config):
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
            classes = counts[counts >= config.min_class_train].index
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
        sex = _clean(train['SEX']) if 'SEX' in train else pd.Series(np.nan, index=train.index)
        counts = pd.crosstab(cancers['train'], sex).reindex(columns=['Female', 'Male'], fill_value=0)
        mixed = counts.index[(counts >= config.min_class_train).all(axis=1)]
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


def evaluate_probes(posteriors, labels, config):
    """Return individual validation scores, per-class AUCs and cohort membership."""
    tasks = cohort_tasks(labels, config)
    rows, classes, membership = [], [], []
    for task in tasks:
        for split in labels:
            ids = labels[split].index[task['keep'][split]]
            membership.extend(dict(task=task['name'], split=split, sample_id=i) for i in ids)
    for draw in range(-1, config.z_draws):
        features = {}
        for split_index, split in enumerate(('train', 'valid')):
            mu, lv = posteriors[split]['mu'], posteriors[split]['logvar']
            if draw == -1:
                features[split] = mu
            else:
                rng = np.random.default_rng(np.random.SeedSequence([config.seed, draw, split_index]))
                with np.errstate(over='ignore', invalid='ignore'):
                    features[split] = mu + np.exp(.5*lv)*rng.standard_normal(mu.shape)
        for task in tasks:
            train_keep, valid_keep = (task['keep'][s] for s in ('train', 'valid'))
            y_train, truth = (task['y'][s][task['keep'][s]].to_numpy() for s in ('train', 'valid'))
            row = dict(task=task['name'], target=task['target'], representation='mu' if draw == -1 else 'z',
                draw=draw, split='valid', metric='r2' if task['regression'] else 'roc_auc_ovo',
                value=np.nan, status='unavailable', warning=None, n_train=len(y_train), n_valid=len(truth),
                n_train_cohort=int(np.sum(task['cohort']['train'])), n_valid_cohort=int(np.sum(task['cohort']['valid'])),
                n_train_missing=int(np.sum(task['cohort']['train'])-task['eligible']['train'].sum()),
                n_valid_missing=int(np.sum(task['cohort']['valid'])-task['eligible']['valid'].sum()),
                n_train_rare_class_excluded=int(task['eligible']['train'].sum()-train_keep.sum()),
                n_valid_unseen_or_rare_class_excluded=int(task['eligible']['valid'].sum()-valid_keep.sum()),
                n_train_classes=0 if task['regression'] else len(np.unique(y_train)))
            try:
                if min(len(y_train), len(truth)) < 2:
                    raise ValueError('Insufficient labeled cohort samples')
                if not task['regression'] and len(np.unique(y_train)) < 2:
                    raise ValueError('Insufficient training classes')
                model = LinearRegression() if task['regression'] else LogisticRegression(
                    max_iter=config.max_iter, random_state=config.seed)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    model.fit(features['train'][train_keep], y_train)
                row['warning'] = '; '.join(dict.fromkeys(str(w.message) for w in caught)) or None
                if task['regression']:
                    if np.ptp(truth) == 0:
                        raise ValueError('Constant validation target; R2 undefined')
                    row['value'] = float(r2_score(truth, model.predict(features['valid'][valid_keep])))
                else:
                    probabilities = model.predict_proba(features['valid'][valid_keep])
                    aucs = []
                    for j, label in enumerate(model.classes_):
                        positive = truth == label
                        defined = positive.any() and (~positive).any()
                        auc = float(roc_auc_score(positive, probabilities[:, j])) if defined else np.nan
                        aucs.append(auc)
                        classes.append(dict(task=task['name'], representation=row['representation'], draw=draw,
                            split='valid', label=label, value=auc, status='ok' if defined else 'undefined_single_class',
                            n_positive=int(positive.sum()), n_negative=int((~positive).sum()),
                            n_train_positive=int((y_train == label).sum())))
                    if not np.isfinite(aucs).all():
                        raise ValueError('Validation cohort missing training classes; macro AUC withheld')
                    row['value'] = float(roc_auc_score(truth == model.classes_[1], probabilities[:, 1])
                        if len(model.classes_) == 2 else roc_auc_score(truth, probabilities,
                            multi_class='ovo', average='macro', labels=model.classes_))
                if not np.isfinite(row['value']):
                    raise ValueError('Nonfinite probe score')
                row['status'] = 'warning' if row['warning'] else 'ok'
            except (ValueError, FloatingPointError) as exc:
                row.update(value=np.nan, status='unavailable', warning=str(exc))
            rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(classes), pd.DataFrame(membership, columns=['task', 'split', 'sample_id'])


def summarize_probes(probes, z_draws):
    """Keep mu scores and summarize z only when every requested draw succeeded."""
    rows = []
    if probes.empty:
        return pd.DataFrame()
    for (task, representation, metric), group in probes.groupby(['task', 'representation', 'metric']):
        finite = np.isfinite(group.value)
        expected = 1 if representation == 'mu' else z_draws
        complete = len(group) == expected and finite.all()
        rows.append(dict(task=task, representation=representation, metric=metric,
            value=float(group.value.mean()) if complete else np.nan,
            draw_std=float(group.value.std(ddof=1)) if complete and expected > 1 else np.nan,
            n_draws=expected, n_success=int(finite.sum()),
            status=('warning' if group.status.eq('warning').any() else 'ok') if complete else 'unavailable',
            warning='; '.join(dict.fromkeys(group.warning.dropna().astype(str))) or None,
            n_train=int(group.n_train.iloc[0]), n_valid=int(group.n_valid.iloc[0]), split='valid'))
    return pd.DataFrame(rows)


def observe_snapshot(folder, epoch, settings, effective_beta):
    """Evaluate saved train/valid snapshots; input API has no test-data path."""
    config = ObserverConfig(**settings)
    started = time.perf_counter()
    folder = Path(folder)
    posteriors, labels, seen = {}, {}, set()
    for split in ('train', 'valid'):
        with np.load(folder / f'epoch_{epoch:05d}_{split}.npz', allow_pickle=False) as saved:
            posterior = {k: saved[k].copy() for k in ('mu', 'logvar', 'sample_ids')}
        ids = posterior['sample_ids'].astype(str)
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError('Snapshot IDs must be unique and train/valid disjoint')
        seen.update(ids)
        if posterior['mu'].shape != posterior['logvar'].shape or len(posterior['mu']) != len(ids):
            raise ValueError('Misaligned posterior parameters')
        posteriors[split] = posterior
        labels[split] = aligned_annotations(pd.read_parquet(folder / f'labels_{split}.parquet'), ids)
    probes, class_auc, cohorts = evaluate_probes(posteriors, labels, config)
    output = folder / 'observations' / f'epoch_{epoch:05d}'
    output.mkdir(parents=True, exist_ok=True)
    metrics, dimensions = [], []
    for split, posterior in posteriors.items():
        kl = compute_conditional_kl_matrix(posterior['mu'], posterior['logvar'])
        dimensions.extend(dict(split=split, latent_dimension=j, kl_mean=float(v)) for j, v in enumerate(kl.mean(axis=0)))
        metrics.append(dict(metric='kl_total_per_sample', value=float(kl.sum(axis=1).mean()), split=split, status='ok'))
        for row in compute_effective_rank_mu(posterior['mu']):
            metrics.append(dict(metric=row['metric_name'], value=row['value'], split=split, status=row['status']))
    with np.load(folder / 'input_valid.npz', allow_pickle=False) as saved:
        x, ids = saved['x'], saved['sample_ids'].astype(str)
    if not np.array_equal(ids, posteriors['valid']['sample_ids'].astype(str)):
        raise ValueError('Input X and posterior mu IDs differ')
    selected = np.sort(np.random.default_rng(config.seed).choice(len(x), min(len(x), config.max_samples), replace=False))
    geometry_metadata = {}
    if len(selected) > 2*config.k:
        # sklearn excludes each sample's own index explicitly. Keep this output
        # separate from the audited ZADU backend, which withholds duplicate rows.
        # Equal-distance ranks still use the library's tie conventions.
        sx, smu = x[selected], posteriors['valid']['mu'][selected]
        tied = len(np.unique(sx, axis=0)) < len(sx) or len(np.unique(smu, axis=0)) < len(smu)
        for name, a, b in (('trustworthiness_sklearn', sx, smu), ('continuity_sklearn', smu, sx)):
            try:
                if min(len(np.unique(a, axis=0)), len(np.unique(b, axis=0))) < 2:
                    raise ValueError('Constant representation; neighborhood preservation undefined')
                value = trustworthiness(a, b, n_neighbors=config.k)
                metrics.append(dict(metric=name, value=float(value), split='valid',
                    status='warning' if tied else 'ok', backend='sklearn.manifold.trustworthiness',
                    warning='Duplicate rows; sklearn tie conventions apply, no jitter' if tied else None))
            except ValueError as exc:
                metrics.append(dict(metric=name, value=np.nan, split='valid', status='unavailable', warning=str(exc)))
        result = compute_zadu_geometry(x[selected], posteriors['valid']['mu'][selected], config.k,
                                       ids[selected], random_state=config.seed)
        for name in ('trustworthiness', 'continuity', 'lcmc', 'steadiness', 'cohesiveness', 'scale_normalized_stress'):
            metrics.append(dict(metric=name, value=result.global_metrics[name], split='valid',
                status=result.metric_statuses[name], warning=result.metric_warnings.get(name), backend='zadu==0.4.2'))
        geometry_metadata = dict(metadata=result.metadata, failed_measures=result.failed_measures, warnings=result.warnings)
    else:
        for name in ('trustworthiness', 'continuity', 'lcmc', 'steadiness', 'cohesiveness', 'scale_normalized_stress',
                     'trustworthiness_sklearn', 'continuity_sklearn'):
            metrics.append(dict(metric=name, value=np.nan, split='valid', status='unavailable', warning='Too few samples for configured k'))
    for name, frame in (('probes.csv', probes), ('probe_summary.csv', summarize_probes(probes, config.z_draws)), ('class_auc.csv', class_auc),
                        ('kl_per_dimension.csv', pd.DataFrame(dimensions)), ('label_free.csv', pd.DataFrame(metrics))):
        frame.assign(epoch=epoch, effective_beta=effective_beta).to_csv(output / name, index=False)
    cohorts.to_parquet(folder / 'observation_cohorts.parquet', index=False)
    (output / 'metadata.json').write_text(json.dumps(dict(config=vars(config), epoch=epoch,
        effective_beta=effective_beta, test_used=False, seed_policy='fixed pairs/noise across epochs',
        selected_valid_ids=ids[selected].tolist(), geometry=geometry_metadata,
        runtime_seconds=time.perf_counter()-started), indent=2, default=str) + '\n')
    print(f'Annealing observation saved: epoch={epoch}, seconds={time.perf_counter()-started:.2f}', flush=True)
