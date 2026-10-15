"""Tests for AUTOENCODIX-native sampled-z downstream probes."""

from types import SimpleNamespace

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from beta_vae_experiments.metrics.evaluation.downstream import (
    TARGETS,
    ClassAUCEvaluator,
    compute_z_logistic_regression_probes_use_split,
)


def test_class_auc_uses_native_fitted_classifier_and_aligned_ids():
    ids = [f"s{i}" for i in range(36)]
    labels = pd.Series(["A", "B", "C"] * 12, index=ids)
    rng = np.random.default_rng(12)
    features = pd.DataFrame(
        np.eye(3)[np.tile(np.arange(3), 12)] + rng.normal(0, .2, (36, 3)),
        index=ids,
    )
    features.iloc[24:27] = features.iloc[24:27].to_numpy()[::-1]
    splits = pd.DataFrame({"SAMPLE_ID": ids, "SPLIT": ["train"] * 18 + ["valid"] * 6 + ["test"] * 12})
    model = LogisticRegression(max_iter=1000)
    evaluator = ClassAUCEvaluator()
    native = evaluator._single_ml_presplit(
        splits, features.iloc[::-1], labels.to_frame("target"), "target",
        model, "roc_auc_ovo", "classification",
    )
    assert len(native) == 3
    records = pd.DataFrame(evaluator.class_auc_records)
    assert len(records) == 6
    test = records[records.score_split == "test"].set_index("class")
    probabilities = model.predict_proba(features.loc[ids[24:]])
    for column, label in enumerate(model.classes_):
        assert test.loc[label, "value"] == roc_auc_score(labels.loc[ids[24:]] == label, probabilities[:, column])
        assert test.loc[label, "n_positive"] == 4
        assert test.loc[label, "n_train_positive"] == 6


class FakeVarix:
    def __init__(self) -> None:
        self.config = SimpleNamespace(global_seed=2)
        self.result = SimpleNamespace(embedding_evaluation=pd.DataFrame())
        metadata = pd.DataFrame(
            {
                "SEX": ["A", "B"] * 6,
                "CANCER_TYPE": ["A", "B"] * 6,
                "CANCER_TYPE_ACRONYM": ["A", "B"] * 6,
            },
            index=[f"sample-{index}" for index in range(12)],
        )
        self.visualizer = SimpleNamespace(
            _collect_all_metadata=lambda result: metadata
        )
        self.calls: list[dict[str, object]] = []

    def evaluate(self, **kwargs):
        self.calls.append(kwargs)
        rows = []
        values = {"train": 0.95, "valid": 0.75, "test": 0.70}
        for target in kwargs["params"]:
            for split, value in values.items():
                rows.append(
                    {
                        "score_split": split,
                        "CLINIC_PARAM": target,
                        "metric": "r2" if target == "AGE" else "roc_auc_ovo",
                        "value": value,
                        "ML_TASK": "Latent",
                        "ML_ALG": "LinearRegression" if target == "AGE" else "LogisticRegression",
                        "ML_TYPE": "regression" if target == "AGE" else "classification",
                        "ML_SUBTASK": "Latent",
                    }
                )
        self.result.embedding_evaluation = pd.concat(
            [self.result.embedding_evaluation, pd.DataFrame(rows)], ignore_index=True
        )
        return self.result


def test_downstream_uses_autoencodix_sampled_z_and_existing_splits() -> None:
    varix = FakeVarix()
    probe, table = compute_z_logistic_regression_probes_use_split(
        varix, "probe-test"
    )

    assert len(varix.calls) == 1
    call = varix.calls[0]
    assert call == {"params": list(TARGETS), "reference_methods": [], "split_type": "use-split", "top_k_classes": None}
    assert len(probe) == 8
    assert set(probe["split"]) == {"valid", "test"}
    assert probe["metric_name"].str.contains("_z_").all()
    assert probe.loc[~probe["metric_name"].str.contains("_AGE_"), "metric_name"].str.endswith("_roc_auc_ovo").all()
    assert probe.loc[probe["metric_name"].str.contains("_AGE_"), "metric_name"].str.endswith("_r2").all()
    assert not probe["metric_name"].str.contains("_mu_|cv5").any()
    assert set(table["score_split"]) == {"train", "valid", "test"}
    assert set(table["representation"]) == {
        "sampled_z_from_result_latentspaces"
    }


def test_downstream_removes_identical_rows_accumulated_by_autoencodix() -> None:
    varix = FakeVarix()
    compute_z_logistic_regression_probes_use_split(varix, "first")
    probe, table = compute_z_logistic_regression_probes_use_split(varix, "second")
    assert varix.calls[0]["reference_methods"] is not varix.calls[1]["reference_methods"]
    assert len(probe) == 8
    assert len(table) == 12


def test_explicit_task_selection_excludes_previous_results():
    varix = FakeVarix()
    compute_z_logistic_regression_probes_use_split(varix, "all")
    probe, table = compute_z_logistic_regression_probes_use_split(
        varix, "age-only", targets=("AGE",),
    )
    assert varix.calls[-1]["params"] == ["AGE"]
    assert len(probe) == 2
    assert set(table.CLINIC_PARAM) == {"AGE"}
    assert probe.metric_name.str.endswith("_r2").all()
