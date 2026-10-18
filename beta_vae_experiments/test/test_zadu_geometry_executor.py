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
        np.random.default_rng(1).normal(size=(25, 4)),
        np.random.default_rng(2).normal(size=(25, 2)), 10
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


@pytest.mark.parametrize('kwargs', [
    {'selected_k': True}, {'selected_k': 2.5}, {'selected_k': 0},
    {'max_samples': 3.2}, {'random_state': -1}, {'random_state': 2**32},
])
def test_geometry_config_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        geometry.GeometryMetricConfig(**kwargs)


def test_primary_identity_matches_definitions_and_sklearn():
    from sklearn.manifold import trustworthiness
    x = np.random.default_rng(9).normal(size=(45, 4))
    mu = x[:, :2]
    result = geometry.compute_zadu_geometry(x, mu, 5)
    assert result.global_metrics['trustworthiness'] == pytest.approx(trustworthiness(x, mu, n_neighbors=5))
    assert result.global_metrics['continuity'] == pytest.approx(trustworthiness(mu, x, n_neighbors=5))
    identity = geometry.compute_zadu_geometry(x, x, 5)
    assert identity.global_metrics['trustworthiness'] == pytest.approx(1)
    assert identity.global_metrics['continuity'] == pytest.approx(1)
    assert identity.global_metrics['lcmc'] == pytest.approx(1-5/44)
    assert identity.global_metrics['mrre_false'] == pytest.approx(1)


def test_duplicate_points_do_not_report_arbitrary_neighbor_scores():
    x = np.random.default_rng(19).normal(size=(35, 3))
    mu = x[:, :2].copy()
    mu[1] = mu[0]
    result = geometry.compute_zadu_geometry(x, mu, 5)
    for name in ('trustworthiness','continuity','lcmc','mrre_false','mrre_missing',
                 'steadiness','cohesiveness','neighbor_dissimilarity','procrustes','topographic_product'):
        assert np.isnan(result.global_metrics[name]), name
    assert np.isfinite(result.global_metrics['scale_normalized_stress'])
    assert result.metadata['spaces']['mu']['duplicate_rows'] == 1
    assert result.local_distortions.shape[0] == len(x)
    assert any(f['stage'] == 'input_guard' for f in result.failed_measures)


def test_collapsed_space_never_produces_perfect_rank_scores(monkeypatch):
    monkeypatch.setattr(geometry, '_run_zadu_measure', lambda *_: pytest.fail('collapsed measure executed'))
    x = np.random.default_rng(10).normal(size=(20, 3))
    result = geometry.compute_zadu_geometry(x, np.zeros((20, 2)), 5)
    assert all(np.isnan(v) for v in result.global_metrics.values())
    assert result.metadata['spaces']['mu']['diameter'] == 0


def test_float32_collisions_are_recorded_without_jitter():
    x = np.random.default_rng(3).normal(size=(30, 3))
    mu = 1e9 + x[:, :2]
    result = geometry.compute_zadu_geometry(x, mu, 5)
    assert result.metadata['spaces']['mu']['unique_rows'] == 30
    assert result.metadata['spaces']['mu']['float32_unique_rows'] < 30
    assert np.isnan(result.global_metrics['lcmc'])


def test_snc_seed_is_reproducible_and_preserves_callers_rng():
    x = np.random.default_rng(4).normal(size=(40, 3))
    mu = x[:, :2]
    before = np.random.get_state()
    first = geometry.compute_zadu_geometry(x, mu, 5, random_state=47)
    after = np.random.get_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    second = geometry.compute_zadu_geometry(x, mu, 10, random_state=47)
    assert first.metadata['snc_effective_k'] == 6
    for key in ('steadiness','cohesiveness'):
        assert np.isfinite(first.global_metrics[key])
        # Changing nominal k does not change SNC in the pinned backend.
        assert first.global_metrics[key] == second.global_metrics[key]


def test_kernel_outputs_are_scale_invariant_with_the_pinned_normalization():
    x = np.random.default_rng(8).normal(size=(45, 3))
    mu = x[:, :2]
    for measure in (geometry.MEASURE.KL_DIV, geometry.MEASURE.DTM):
        spec = geometry.make_spec(measure, sigma=.1)
        first, _ = geometry._run_zadu_measure(spec, x, mu)
        scaled, _ = geometry._run_zadu_measure(spec, x*2, mu*7)
        assert first == pytest.approx(scaled, abs=1e-12)


def test_runner_provenance_records_actual_k_and_local_alignment(tmp_path):
    import json
    from types import SimpleNamespace
    x = np.random.default_rng(2).normal(size=(25, 4))
    mu = x[:, :2].copy()
    ids = np.array([f'p{i}' for i in range(25)])
    inputs = SimpleNamespace(x_test=x, mu_test=mu, sample_ids_test=ids, run_name='run')
    indices = np.arange(15)
    frame = geometry.run_geometry_metrics_pipeline(inputs, indices, x[indices], mu[indices],
        geometry.GeometryMetricConfig(selected_k=20, random_state=47), tmp_path)
    path = tmp_path/'absolute_core_metrics_v1/geometry/run/geometry_provenance.json'
    facts = json.loads(path.read_text())
    assert facts['requested_k'] == 20
    assert facts['effective_k'] == 7
    assert facts['k_was_clipped'] is True
    assert facts['snc_effective_k'] == 3
    assert facts['snc_seed'] == 47
    assert facts['sample_ids'] == list(ids[:15])
    assert facts['versions']['zadu'] == '0.4.2'
    assert frame.metric_name.str.contains('trustworthiness_zadu_k7').any()
    with pytest.raises(ValueError, match='Selected mu'):
        geometry.run_geometry_metrics_pipeline(inputs, indices, x[indices], mu[indices][::-1],
            geometry.GeometryMetricConfig(), None)


def test_near_collapse_does_not_mean_same_as_exact_collapse():
    x = np.random.default_rng(7).normal(size=(35, 3))
    small = geometry.compute_zadu_geometry(x, x*1e-8, 5)
    assert small.global_metrics['trustworthiness'] == pytest.approx(1)
    assert small.global_metrics['lcmc'] == pytest.approx(1-5/34)
