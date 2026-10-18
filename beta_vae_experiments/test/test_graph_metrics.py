"""Library-backed graph/ID behavior and integration; no estimator replicas."""
from types import SimpleNamespace
import json
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_swiss_roll
from sklearn.metrics import pairwise_distances
from beta_vae_experiments.metrics.graph.graph_builder import build_graph, RepresentationGraph
from beta_vae_experiments.metrics.dimensionality.intrinsic_dimension import compute_global_id, compute_local_id
from beta_vae_experiments.metrics.graph.hubness_metrics import compute_hubness
from beta_vae_experiments.metrics.graph.graph_neighborhood_metrics import compute_neighborhood
from beta_vae_experiments.metrics.graph.graph_community_metrics import compute_communities
from beta_vae_experiments.metrics.graph.graph_connectivity_metrics import compute_connectivity
from beta_vae_experiments.metrics.graph.graph_path_metrics import compute_paths
from beta_vae_experiments.metrics.graph.input_latent_graph_comparison import compare_graphs
from beta_vae_experiments.metrics.graph.graph_export import export_graph
from beta_vae_experiments.metrics.graph.graph_metrics_pipeline import GraphMetricConfig, run_graph_metrics_pipeline


@pytest.fixture
def cloud():
    x, _ = make_swiss_roll(n_samples=80, random_state=31)
    return x, build_graph(x, np.array([f"s{i}" for i in range(len(x))]), 5)


def test_builder(cloud):
    x, g = cloud
    assert all(d == 5 for _, d in g.directed.out_degree())
    assert nx.number_of_selfloops(g.directed) == 0
    pre = build_graph(pairwise_distances(x), g.sample_ids, 5, "precomputed")
    assert set(pre.directed.edges) == set(g.directed.edges)
    assert all(d['mutual'] == (g.directed.has_edge(i,j) and g.directed.has_edge(j,i)) for i,j,d in g.undirected.edges(data=True))
    with pytest.raises(ValueError): build_graph(x, np.repeat('same', len(x)), 5)
    with pytest.raises(ValueError): build_graph(x, g.sample_ids, len(x))
    with pytest.raises(ValueError): build_graph(x[:-1], g.sample_ids, 5)


def test_zero_distance_edges():
    g = build_graph(np.zeros((6, 2)), np.arange(6), 2)
    assert g.directed.number_of_edges() == 12
    assert all(d['distance'] == 0 for _,_,d in g.directed.edges(data=True))


def test_id(cloud):
    x, _ = cloud
    assert np.isfinite(compute_global_id(x)['global_id'])
    metrics, nodes = compute_local_id(x, 5)
    assert nodes['local_id'].shape == (80,)
    assert all(np.isfinite(list(metrics.values())))
    assert metrics['local_id_q10'] <= metrics['local_id_median']


def test_hubness(cloud):
    _, g = cloud
    metrics, nodes = compute_hubness(g)
    np.testing.assert_array_equal(nodes['k_occurrence'], [d for _, d in g.directed.in_degree()])
    assert 0 <= metrics['antihub_fraction'] <= 1


def test_neighborhood(cloud):
    m = compute_neighborhood(cloud[1])
    assert 0 <= m['reciprocity'] <= 1
    assert 0 <= m['clustering'] <= 1


def test_leiden(cloud):
    a, n = compute_communities(cloud[1])
    b, m = compute_communities(cloud[1])
    assert a == b
    np.testing.assert_array_equal(n['community'], m['community'])
    assert len(n['community']) == 80


def test_disconnected_paths():
    g = nx.disjoint_union(nx.path_graph(4), nx.complete_graph(3))
    graph = RepresentationGraph(g.to_directed(), g, None, np.arange(7), 1)
    connectivity = compute_connectivity(graph)
    assert connectivity['components'] == 2
    assert connectivity['lambda2'] == 0
    assert connectivity['giant_component_fraction'] == 4/7
    metrics, nodes, edges = compute_paths(graph)
    assert metrics['lcc_diameter'] == 3
    assert len(nodes['betweenness']) == 7
    assert len(edges['edge_betweenness']) == g.number_of_edges()
    with pytest.raises(ValueError): compute_paths(graph, max_nodes=3)


def test_curvature_and_export(cloud, tmp_path):
    from beta_vae_experiments.metrics.complementary.curvature.graph_curvature_metrics import compute_curvature
    _, g = cloud
    metrics, nodes, edges = compute_curvature(g)
    assert all(np.isfinite(list(metrics.values())))
    assert len(edges['forman_curvature']) == g.undirected.number_of_edges()
    nodes = compute_communities(g)[1]
    annotation = pd.DataFrame({'cancer_type': ['A']*80}, index=g.sample_ids)
    export_graph(g, nodes, edges, tmp_path, graphml=True, sample_metadata={'paired': annotation.iloc[::-1]})
    assert nx.read_graphml(tmp_path/'union.graphml').number_of_nodes() == 80
    assert pd.read_csv(tmp_path/'nodes.csv').sample_id.tolist() == g.sample_ids.tolist()
    directed = nx.read_graphml(tmp_path/'directed.graphml')
    assert directed.nodes['0']['external_cancer_type'] == 'A'
    assert 'community' in directed.nodes['0']


def test_comparison(cloud):
    _, g = cloud
    m, n = compute_hubness(g)
    n.update(compute_communities(g)[1])
    r = compare_graphs(g, g, m, m, n, n)
    assert r['partition_ari'] == 1
    assert 'directed_edge_jaccard_lcmc_related' not in r
    assert r["partition_nmi"] == 1
    assert r["partition_ami"] == 1
    assert r["hub_identity_spearman"] == pytest.approx(1)
    assert r['delta_hubness_skewness'] == 0
    bad = build_graph(cloud[0], g.sample_ids[::-1], 5)
    with pytest.raises(ValueError): compare_graphs(g, bad, m, m, n, n)


def test_pipeline(cloud, tmp_path):
    x, g = cloud
    inputs = SimpleNamespace(run_name='test', x_test=x, sample_ids_test=g.sample_ids, metadata={'beta': .1})
    geometry = SimpleNamespace(selected_k=5, random_state=31, distance_metric='euclidean')
    result = run_graph_metrics_pipeline(inputs, np.arange(80), x, x, GraphMetricConfig(), geometry, tmp_path)
    assert result.status.eq('ok').all()
    assert result.metric_name.is_unique
    metadata = json.loads((tmp_path/'metrics_artifacts/test/graph_metrics/metadata.json').read_text())
    assert metadata['run_metadata']['beta'] == .1
    assert metadata['package_versions']['scikit-hubness']
    assert not list(tmp_path.rglob('*.graphml'))
    result = run_graph_metrics_pipeline(inputs, np.arange(80), x, x,
        GraphMetricConfig(max_path_nodes=3, families=("paths", "community")), geometry)
    failed = result.loc[result.metric_name.str.contains('lcc_average_path_length')]
    assert len(failed) == 3 and failed.status.eq('unavailable').all()
    assert not result.metric_name.str.contains('forman|ollivier').any()
    with pytest.raises(ValueError, match='Selected X'):
        run_graph_metrics_pipeline(inputs, np.arange(80), x[::-1], x, GraphMetricConfig(), geometry)
    invalid = run_graph_metrics_pipeline(inputs, np.arange(80), x, x, GraphMetricConfig(k_values=(80,)), geometry)
    assert invalid.value.isna().all()
    assert 'graph_delta_modularity_k80' in set(invalid.metric_name)
