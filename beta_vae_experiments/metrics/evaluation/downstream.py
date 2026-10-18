"""AUTOENCODIX-native downstream evaluation of sampled VARIX latents."""

from __future__ import annotations

from pathlib import Path

from autoencodix.varix import Varix
from autoencodix.evaluate._general_evaluator import GeneralEvaluator
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from .sampled_z import DownstreamMetricConfig, run_sampled_z_metrics


METRIC_COLUMNS = (
    "metric_name", "value", "status", "warning", "n_samples", "split",
)
TARGETS = ("CANCER_TYPE_ACRONYM", "AJCC_PATHOLOGIC_TUMOR_STAGE_SHORT", "SEX", "AGE")


class ClassAUCEvaluator(GeneralEvaluator):
    """Record per-class OvR AUC using the classifier fitted by AUTOENCODIX."""

    def _single_ml_presplit(self, sample_split, df, clin_data, task_param,
                           sklearn_ml, metric, ml_type, top_k_classes=20):
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
    varix.evaluate(params=list(targets), reference_methods=[], split_type="use-split", top_k_classes=None)
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
        repeated = run_sampled_z_metrics(varix, run_name, output_dir, targets, config)
        downstream = pd.concat([downstream, repeated], ignore_index=True)
    return downstream
