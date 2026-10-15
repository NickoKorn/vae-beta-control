import numpy as np
import pytest
from scipy.stats import spearmanr
from beta_vae_experiments.metrics import graph


def test_identity_and_repeatability():
    x = np.random.default_rng(7).normal(size=(30, 3))
    a = graph.compute_geodesic_preservation(x, x.copy())
    assert a == graph.compute_geodesic_preservation(x, x.copy())
    assert a['geodesic_spearman'] == pytest.approx(1.)
    assert a['x_graph_connected'] and a['mu_graph_connected']
    assert a['n_total_pairs'] == a['n_valid_pairs'] == 435
    assert a['valid_pair_fraction'] == 1


def test_distorted_connected_representation():
    x = np.arange(20., dtype=float)[:, None]
    mu = x[np.random.default_rng(8).permutation(len(x))]
    result = graph.compute_geodesic_preservation(x, mu, 2)
    assert result['x_graph_connected'] and result['mu_graph_connected']
    assert np.isfinite(result['geodesic_spearman'])
    assert result['geodesic_spearman'] < .8


def test_mismatched_samples():
    with pytest.raises(ValueError, match='same number of samples'):
        graph.compute_geodesic_preservation(np.ones((12, 2)), np.ones((11, 2)))


def test_disconnected_intersection_only(monkeypatch):
    x = np.array([0., 1., 10., 12., 30., 33.])[:, None]
    mu = np.array([0., 1., 3., 20., 22., 25.])[:, None]
    seen = []
    def checked(a, b):
        assert np.isfinite(a).all() and np.isfinite(b).all()
        seen.append(len(a))
        return spearmanr(a, b)
    monkeypatch.setattr(graph, 'spearmanr', checked)
    r = graph.compute_geodesic_preservation(x, mu, 1)
    assert not r['x_graph_connected'] and not r['mu_graph_connected']
    assert r['n_total_pairs'] == 15 and r['n_valid_pairs'] == 2
    assert r['valid_pair_fraction'] == pytest.approx(2/15)
    assert seen == [2]


def test_zero_distance_edges_and_undefined_correlation():
    x = np.zeros((4, 2))
    r = graph.compute_geodesic_preservation(x, x.copy(), 3)
    assert r['x_graph_connected'] and r['mu_graph_connected']
    assert r['n_valid_pairs'] == 6 and np.isnan(r['geodesic_spearman'])


def test_union_preserves_one_sided_edge_weight():
    d = graph._graph_distances(np.array([[0.], [1.], [3.]]), 1)
    np.testing.assert_allclose(d, [[0, 1, 3], [1, 0, 2], [3, 2, 0]])


@pytest.mark.parametrize('k', [0, 4, 1.5, True])
def test_invalid_neighbors(k):
    with pytest.raises(ValueError, match='n_neighbors'):
        graph.compute_geodesic_preservation(np.ones((4, 2)), np.ones((4, 1)), k)


@pytest.mark.parametrize('bad', [np.ones(4), np.ones((4, 0)), np.full((4, 1), np.nan), np.full((4, 1), np.inf)])
def test_invalid_arrays(bad):
    with pytest.raises(ValueError):
        graph.compute_geodesic_preservation(bad, np.ones((4, 1)), 1)


def test_no_shared_connected_pairs():
    x = np.array([0., 1., 10., 12.])[:, None]
    mu = np.array([0., 10., 1., 12.])[:, None]
    r = graph.compute_geodesic_preservation(x, mu, 1)
    assert r['n_valid_pairs'] == 0 and r['valid_pair_fraction'] == 0
    assert np.isnan(r['geodesic_spearman'])
