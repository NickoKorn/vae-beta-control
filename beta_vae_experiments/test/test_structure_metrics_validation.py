"""Scientific edge cases and real-library integration for TDA and graphs."""
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest

import beta_vae_experiments.metrics.topology.preservation as tda
from beta_vae_experiments.metrics.topology._diagrams import _finite_diagram
from beta_vae_experiments.metrics.graph.graph_builder import build_graph
from beta_vae_experiments.metrics.graph.graph_metrics_pipeline import GraphMetricConfig, run_graph_metrics_pipeline
from beta_vae_experiments.metrics.graph.label_diagnostics import compute_label_diagnostics


def inputs(x, mu, name="run"):
    return SimpleNamespace(x_test=x, mu_test=mu, run_name=name,
        sample_ids_test=np.array([f"patient{i}" for i in range(len(x))]), metadata={})


def circle(n=32):
    angle = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([np.cos(angle), np.sin(angle)])


@pytest.mark.parametrize("field,value", [("max_samples", True), ("max_samples", 2.5),
    ("diameter_epsilon", np.nan), ("diameter_epsilon", np.inf), ("random_state", -1)])
def test_invalid_topology_config(field, value):
    with pytest.raises(ValueError):
        tda.TopologyMetricConfig(**{field: value})


def test_diagram_single_bar_has_known_cost():
    diagrams = {f"topology_{space}_h{dim}_persistence_diagram": (
        np.array([[0.2, 0.8]]) if space == "input" else np.empty((0, 2)))
        for space in ("input", "latent") for dim in (0, 1)}
    assert tda.compute_normalized_wasserstein_h0_h1(diagrams, 2).value.tolist() == pytest.approx([.6/np.sqrt(2)]*2)
    assert tda.compute_normalized_bottleneck_h0_h1(diagrams, 2).value.tolist() == pytest.approx([.3]*2)


@pytest.mark.parametrize("bad", [np.array([[0., np.nan]]), np.array([[np.inf, np.inf]]),
    np.array([[1., .2]]), np.array([0., 1.])])
def test_invalid_diagram_is_not_silently_an_empty_diagram(bad):
    with pytest.raises(ValueError):
        _finite_diagram(bad)


@pytest.mark.parametrize("constant_space", ["input", "latent", "both"])
def test_collapse_preserves_scale_artifacts(tmp_path, constant_space):
    x = circle()
    mu = x.copy()
    if constant_space in ("input", "both"):
        x[:] = 0
    if constant_space in ("latent", "both"):
        mu[:] = 0
    result = tda.run_topology_metrics_pipeline(inputs(x, mu), np.arange(len(x)), x, mu,
                                              tda.TopologyMetricConfig(), tmp_path)
    assert result.status.eq("zero_diameter").all()
    assert result.value.isna().all()
    directory = tmp_path / "absolute_core_metrics_v1/topology/run"
    facts = json.loads((directory / "provenance.json").read_text())
    assert facts["distance_scale_diagnostics"]["latent_diameter"] == pytest.approx(float(np.linalg.norm(mu[:, None]-mu, axis=2).max()))
    assert (directory / "distance_scale_diagnostics.csv").exists()


def test_alignment_is_checked_locally():
    x = circle()
    with pytest.raises(ValueError, match="Selected mu"):
        tda.run_topology_metrics_pipeline(inputs(x, x), np.arange(len(x)), x, x[::-1], tda.TopologyMetricConfig())


def test_cache_reuses_only_input_and_invalidates_on_ids(tmp_path, monkeypatch):
    actual = tda.ripser
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return actual(*args, **kwargs)
    monkeypatch.setattr(tda, "ripser", counted)
    x = circle()
    data = inputs(x, x)
    config = tda.TopologyMetricConfig()
    for name in ("a", "b"):
        data.run_name = name
        result = tda.run_topology_metrics_pipeline(data, np.arange(len(x)), x, x, config, tmp_path)
        assert result.value.to_numpy() == pytest.approx(np.zeros(5), abs=1e-7)
    assert len(calls) == 3  # first X, first mu, second mu
    data.run_name = "c"
    data.sample_ids_test = data.sample_ids_test[::-1]
    tda.run_topology_metrics_pipeline(data, np.arange(len(x)), x, x, config, tmp_path)
    assert len(calls) == 5
    assert len(list((tmp_path / "metrics_artifacts/topology_input_cache").glob("*.npz"))) == 2


def test_circle_scaling_and_destroyed_loop(tmp_path):
    x = circle()
    config = tda.TopologyMetricConfig()
    good = tda.run_topology_metrics_pipeline(inputs(x, 7*x, "good"), np.arange(len(x)), x, 7*x, config, tmp_path)
    destroyed = x.copy()
    destroyed[:, 1] = 0
    bad = tda.run_topology_metrics_pipeline(inputs(x, destroyed, "bad"), np.arange(len(x)), x, destroyed, config, tmp_path)
    name = "topology_h1_normalized_wasserstein"
    assert good.set_index("metric_name").loc[name, "value"] == pytest.approx(0, abs=1e-6)
    assert bad.set_index("metric_name").loc[name, "value"] > .1
    assert len(good) == 5
    directory = tmp_path / "absolute_core_metrics_v1/topology/good"
    assert (directory / "exploratory_entropy.csv").exists()
    assert json.loads((directory / "betti_convention.json").read_text())["h0"].startswith("reduced")


def test_missing_persim_does_not_disable_ph_or_bottleneck(tmp_path, monkeypatch):
    monkeypatch.setattr(tda, "wasserstein", None)
    x = circle()
    result = tda.run_topology_metrics_pipeline(inputs(x, x), np.arange(len(x)), x, x,
                                              tda.TopologyMetricConfig(), tmp_path)
    assert result.loc[result.metric_name.str.contains("bottleneck"), "value"].eq(0).all()
    assert result.loc[result.metric_name.str.contains("wasserstein"), "status"].eq("missing_dependency").all()
    assert list(tmp_path.rglob("*persistence_diagram.npy"))


def test_diagnostic_failure_does_not_destroy_main_outputs(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("diagnostic unavailable")
    monkeypatch.setattr(tda, "run_landscape_metrics", broken)
    x = circle()
    with pytest.warns(RuntimeWarning, match="landscape"):
        result = tda.run_topology_metrics_pipeline(inputs(x, x), np.arange(len(x)), x, x,
            tda.TopologyMetricConfig(landscapes_enabled=True), tmp_path)
    assert result.status.eq("ok").all()
    directory = tmp_path / "absolute_core_metrics_v1/topology/run"
    assert (directory / "scalar_metrics.csv").exists()
    assert json.loads((directory / "provenance.json").read_text())["diagnostic_failures"]


def test_graph_core_and_labels_are_separate(tmp_path):
    x = np.random.default_rng(12).normal(size=(36, 3))
    data = inputs(x, x)
    data.sample_metadata_test = pd.DataFrame({"SEX": ["a"]*18+["b"]*17+[None]}, index=data.sample_ids_test)
    geometry = SimpleNamespace(selected_k=5, random_state=31, distance_metric="euclidean")
    result = run_graph_metrics_pipeline(data, np.arange(len(x)), x, x,
        GraphMetricConfig(k_values=(3, 5)), geometry, tmp_path)
    assert result.metric_name.is_unique
    assert not result.metric_name.str.contains("betweenness|local_id|global_id|assortativity|SEX").any()
    assert result.loc[result.metric_name.str.contains("partition_ami"), "value"].eq(1).all()
    assert len(list(tmp_path.rglob("*_assortativity.csv"))) == 4


def test_label_alignment_and_missingness():
    x = np.arange(8.)[:, None]
    g = build_graph(x, np.array([str(i) for i in range(8)]), 2)
    labels = pd.DataFrame({"SEX": ["a"]*4+["b"]*3+[None]}, index=g.sample_ids)
    a, mixing = compute_label_diagnostics(g, labels.iloc[::-1], ("SEX",))
    assert a.iloc[0].n_missing == 1
    assert a.iloc[0].n_nodes == 7
    assert mixing.edge_end_fraction.sum() == pytest.approx(1)
    b, _ = compute_label_diagnostics(g, labels, ("SEX",))
    pd.testing.assert_frame_equal(a, b)


def test_constant_graph_not_reported_as_valid_preservation():
    x = circle()
    data = inputs(x, np.zeros_like(x))
    geometry = SimpleNamespace(selected_k=5, random_state=31, distance_metric="euclidean")
    result = run_graph_metrics_pipeline(data, np.arange(len(x)), x, data.mu_test, GraphMetricConfig(), geometry)
    assert result.loc[result.metric_name.str.startswith("graph_latent"), "value"].isna().all()
    assert result.loc[result.metric_name.str.contains("partition_ami"), "value"].isna().all()


def test_visualization_flag_creates_diagram_artifact(tmp_path):
    x = circle()
    tda.run_topology_metrics_pipeline(inputs(x, x), np.arange(len(x)), x, x,
        tda.TopologyMetricConfig(ph_visualizations_enabled=True), tmp_path)
    assert (tmp_path / "absolute_core_metrics_v1/topology/run/persistence_diagrams.png").exists()


def test_invalid_distances_have_separate_status(tmp_path):
    x = circle()
    x[0, 0] = np.nan
    result = tda.run_topology_metrics_pipeline(inputs(x, x), np.arange(len(x)), x, x,
        tda.TopologyMetricConfig(), tmp_path)
    assert result.status.eq("invalid_distances").all()


def test_missing_ripser_preserves_scale_facts(tmp_path, monkeypatch):
    monkeypatch.setattr(tda, "ripser", None)
    x = circle()
    result = tda.run_topology_metrics_pipeline(inputs(x, x), np.arange(len(x)), x, x,
        tda.TopologyMetricConfig(), tmp_path)
    assert result.status.eq("missing_dependency").all()
    assert (tmp_path / "absolute_core_metrics_v1/topology/run/distance_scale_diagnostics.csv").exists()
