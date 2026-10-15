"""Tests for public, per-measure ZADU execution and failure isolation."""

from pathlib import Path

import numpy as np
import pytest

import beta_vae_experiments.metrics.geometry as geometry


def test_one_public_zadu_object_is_created_per_measure(monkeypatch) -> None:
    calls = {"construct": 0, "measure": 0}

    class FakeZADU:
        def __init__(self, specs, original, **kwargs):
            calls["construct"] += 1
            assert len(specs) == 1
            assert original.shape == (25, 4)
            assert kwargs["return_local"] is True

        def measure(self, embedding):
            calls["measure"] += 1
            assert embedding.shape == (25, 2)
            return [{"score": 0.5}], [{"local_score": np.arange(25)}]

    monkeypatch.setattr(geometry, "ZADU", FakeZADU)
    result, local = geometry._run_zadu_measure(
        {"id": "metric_a", "params": {}},
        np.zeros((25, 4)),
        np.zeros((25, 2)),
    )
    assert result == {"score": 0.5}
    assert local is not None and len(local["local_score"]) == 25
    assert calls == {"construct": 1, "measure": 1}


def test_failure_in_one_object_does_not_block_later_measures(monkeypatch) -> None:
    calls: list[str] = []

    def fake_run(spec, _x, _mu):
        measure = str(spec["id"])
        calls.append(measure)
        if measure == "neighbor_dissimilarity":
            raise FloatingPointError("undefined neighbors")
        outputs = {
            "trustworthiness_continuity": {"trustworthiness": 0.8, "continuity": 0.7},
            "local_continuity_meta_criteria": {"lcmc": -0.4},
            "mean_relative_rank_error": {"mrre_false": 0.85, "mrre_missing": 0.9},
            "non_metric_stress": {"non_metric_stress": 0.5},
            "procrustes": {"procrustes": 0.1},
            "spearman_rho": {"spearman_rho": 0.9},
            "scale_normalized_stress": {"scale_normalized_stress": 0.6},
            "steadiness_cohesiveness": {"steadiness": 0.7, "cohesiveness": 0.8},
            "topographic_product": {"topographic_product": 0.2},
            "distance_to_measure": {"distance_to_measure": 0.3},
        }
        if measure == "kl_divergence":
            return {"kl_divergence": spec["params"]["sigma"]}, None
        return outputs[measure], None

    monkeypatch.setattr(geometry, "_run_zadu_measure", fake_run)
    result = geometry.compute_zadu_geometry(
        np.zeros((25, 4)), np.ones((25, 2)), 10
    )
    assert len(calls) == 14
    assert "mean_relative_rank_error" in calls
    assert result.global_metrics["mrre_false"] == 0.85
    assert result.global_metrics["mrre_missing"] == 0.9
    assert np.isnan(result.global_metrics["neighbor_dissimilarity"])
    assert result.global_metrics["lcmc"] == -0.4
    assert result.global_metrics["kl_divergence_sigma_1p0"] == 1.0
    assert result.failed_measures[0]["measure"] == "neighbor_dissimilarity"


def test_public_zadu_0p4p2_call_returns_lcmc_and_local_values() -> None:
    spec = geometry.make_spec(geometry.MEASURE.LCMC, k=10)
    result, local = geometry._run_zadu_measure(
        spec, np.zeros((25, 3)), np.ones((25, 2))
    )
    assert set(result) == {"lcmc"}
    assert set(local or {}) == {"local_lcmc"}


def test_zadu_mrre_identical_rankings_score_one() -> None:
    x = np.random.default_rng(19).normal(size=(25, 3))
    result, _ = geometry._run_zadu_measure(geometry.make_spec(geometry.MEASURE.MRRE, k=5), x, x)
    assert result == {"mrre_false": 1.0, "mrre_missing": 1.0}


def test_zadu_dependency_is_pinned_to_0p4p2() -> None:
    requirements = Path("beta_vae_experiments/requirements.txt").read_text(
        encoding="utf-8"
    )
    assert "zadu==0.4.2" in requirements.splitlines()


def test_different_sample_counts_are_rejected_before_zadu() -> None:
    with pytest.raises(ValueError, match="same aligned samples"):
        geometry.compute_zadu_geometry(np.zeros((25, 3)), np.zeros((24, 2)), 10)
