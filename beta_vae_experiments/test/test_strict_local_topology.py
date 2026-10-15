"""Behavior of the public local-homology adapter, not a custom local estimator."""
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest

from beta_vae_experiments.metrics.topology import local_topology as local


def fixture_inputs(mu=None):
    x = np.random.default_rng(8).normal(size=(18, 3))
    return SimpleNamespace(run_name='local', x_test=x,
        mu_test=x.copy() if mu is None else mu,
        sample_ids_test=np.array([f'id-{i}' for i in range(len(x))]))


def run(inputs, output=None, indices=None, config=None):
    indices = np.arange(len(inputs.x_test)) if indices is None else indices
    return local.run_local_topology_metrics_pipeline(inputs, indices,
        inputs.x_test[indices], inputs.mu_test[indices],
        config or local.LocalTopologyMetricConfig(), output)


def test_aligned_subset_and_identical_spaces(tmp_path):
    inputs = fixture_inputs()
    indices = np.array([13, 2, 4, 9, 0, 7, 11, 17, 5])
    result = run(inputs, tmp_path, indices)
    assert result.empty and tuple(result.columns) == local.SCALAR_COLUMNS
    folder = tmp_path / 'absolute_core_metrics_v1/local_topology/local'
    rows = pd.read_csv(folder / 'local_topology_per_anchor.csv')
    assert rows.bottleneck_distance.eq(0).all()
    assert rows.status.eq('ok').all()
    assert rows.selected_test_index.tolist() == np.repeat(indices, 2).tolist()
    assert rows.sample_id.tolist() == np.repeat(inputs.sample_ids_test[indices], 2).tolist()
    np.testing.assert_array_equal(np.load(folder / 'selected_indices.npy'), indices)
    params = json.loads((folder / 'parameters.json').read_text())
    assert params['giotto_tda_version'] == '0.6.2'
    assert params['coefficient_field'] == 2 and params['reduced_homology'] is True
    assert params['distance_normalization'] == 'none'
    assert (folder / 'local_topology_diagrams.csv').exists()
    assert (folder / 'local_topology_raw_mu.npy').exists()


def test_direct_transformer_parity_no_scaling():
    from gtda.local_homology import KNeighborsLocalVietorisRips
    inputs = fixture_inputs()
    inputs.mu_test = inputs.x_test[:, :2] * 5
    artifacts = run(inputs).attrs['local_topology']
    for space, points in [('x', inputs.x_test), ('mu', inputs.mu_test)]:
        expected = KNeighborsLocalVietorisRips(n_neighbors=(1, 7),
            homology_dimensions=(0, 1), n_jobs=1).fit_transform(points)
        np.testing.assert_array_equal(artifacts['raw_outputs'][space], expected)


@pytest.mark.parametrize('mutation', ['length', 'reordered', 'nan', 'indices'])
def test_bad_alignment_is_rejected(mutation):
    inputs = fixture_inputs()
    indices, x, mu = np.arange(18), inputs.x_test.copy(), inputs.mu_test.copy()
    if mutation == 'length': mu = mu[:-1]
    elif mutation == 'reordered': mu = mu[::-1]
    elif mutation == 'nan': x[0, 0] = np.nan
    else: indices = np.arange(18, dtype=float)
    with pytest.raises(ValueError):
        local.run_local_topology_metrics_pipeline(inputs, indices, x, mu, local.LocalTopologyMetricConfig())


@pytest.mark.parametrize('duplicate', [False, True])
def test_collapse_and_duplicates_are_not_zero_preservation(duplicate):
    inputs = fixture_inputs()
    if duplicate: inputs.mu_test[1] = inputs.mu_test[0]
    else: inputs.mu_test[:] = 0
    data = run(inputs).attrs['local_topology']
    assert data['per_anchor'].bottleneck_distance.isna().all()
    assert data['per_anchor'].status.eq('degenerate_input').all()
    assert set(data['raw_outputs']) == {'x'}
    assert data['descriptions'].query("space == 'x'").status.eq('ok').all()


def test_too_few_samples_are_not_automatically_reparameterized():
    data = run(fixture_inputs(), indices=np.arange(5)).attrs['local_topology']
    assert data['per_anchor'].status.eq('degenerate_input').all()
    assert data['raw_outputs'] == {}


def test_padding_empty_and_essential_bars():
    triples = np.array([[0., 0., 0.], [2., 2., 1.], [0., np.inf, 1.], [1., 3., 1.]])
    empty = local.diagram_in_dimension(triples, 0)
    assert empty.shape == (0, 2)
    assert local.compare_local_diagrams(empty, empty) == 0
    diagram = local.diagram_in_dimension(triples, 1)
    assert len(diagram) == 2 and np.isposinf(diagram[0, 1])
    assert local.compare_local_diagrams(diagram, diagram) == 0
    assert np.isposinf(local.compare_local_diagrams(diagram, empty))
    assert local.compare_local_diagrams(np.array([[0., 4.]]), empty) == 2


def test_empty_library_output_is_preserved_as_valid_empty(monkeypatch):
    from gtda.local_homology import KNeighborsLocalVietorisRips
    def empty_transform(self, points):
        triples = np.zeros((len(points), 2, 3))
        triples[:, 1, 2] = 1
        return triples
    monkeypatch.setattr(KNeighborsLocalVietorisRips, 'transform', empty_transform)
    data = run(fixture_inputs()).attrs['local_topology']
    assert data['diagrams'].empty
    assert data['descriptions'].status.eq('ok').all()
    assert data['per_anchor'].bottleneck_distance.eq(0).all()


def test_reproducibility():
    first = run(fixture_inputs()).attrs['local_topology']
    second = run(fixture_inputs()).attrs['local_topology']
    pd.testing.assert_frame_equal(first['per_anchor'], second['per_anchor'])
    pd.testing.assert_frame_equal(first['diagrams'], second['diagrams'])


def test_unverified_version_does_not_fall_back(monkeypatch):
    original = local._version
    monkeypatch.setattr(local, '_version', lambda name: '9.0' if name == 'giotto-tda' else original(name))
    data = run(fixture_inputs()).attrs['local_topology']
    assert data['per_anchor'].status.eq('failed_source_verification').all()
    assert data['per_anchor'].bottleneck_distance.isna().all()


def test_sensitivity_cannot_overwrite_another_setting(tmp_path):
    run(fixture_inputs(), tmp_path)
    with pytest.raises(ValueError, match='distinct artifact location'):
        run(fixture_inputs(), tmp_path, config=local.LocalTopologyMetricConfig(n_neighbors=(1, 8)))


def test_disabled_does_not_write(tmp_path):
    result = run(fixture_inputs(), tmp_path, config=local.LocalTopologyMetricConfig(enabled=False))
    assert result.empty and list(tmp_path.iterdir()) == []
