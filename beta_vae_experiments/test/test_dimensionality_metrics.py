from types import SimpleNamespace
import json

import numpy as np
import pytest
import skdim

from beta_vae_experiments.metrics.dimensionality import (
    DimensionalityMetricConfig, compute_effective_rank_mu,
    _estimate_intrinsic_dimension, run_dimensionality_metrics_pipeline,
)


def test_effective_rank_known_spectrum_and_scale_invariance():
    points = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    for scale in (1., 1e-30, 1e30):
        rows = compute_effective_rank_mu(points * scale)
        assert rows[0]['value'] == pytest.approx(2.)
        assert rows[1]['value'] == pytest.approx(1.)
    line = points.copy()
    line[:, 1] = 0
    assert compute_effective_rank_mu(line)[0]['value'] == pytest.approx(1.)
    assert compute_effective_rank_mu(np.full((100, 3), .1))[0]['value'] == 0


def test_id_matches_official_estimators():
    points = np.random.default_rng(13).normal(size=(150, 3))
    config = DimensionalityMetricConfig()
    for estimator, expected in [
        ('twonn', skdim.id.TwoNN(discard_fraction=.1).fit(points).dimension_),
        ('mle', skdim.id.MLE().fit(points, n_neighbors=20).dimension_),
    ]:
        result = _estimate_intrinsic_dimension(points, representation='mu', estimator_name=estimator, config=config)
        assert result['value'] == pytest.approx(expected)
        assert result['status'] == 'ok'


@pytest.mark.parametrize('estimator', ['twonn', 'mle'])
def test_duplicate_points_are_unavailable_without_repair(estimator):
    points = np.random.default_rng(13).normal(size=(50, 3))
    points[1] = points[0]
    result = _estimate_intrinsic_dimension(points, representation='mu', estimator_name=estimator,
                                           config=DimensionalityMetricConfig())
    assert np.isnan(result['value'])
    assert 'Duplicate' in result['warning']


def test_small_sample_mle_is_not_silently_clipped():
    points = np.random.default_rng(13).normal(size=(10, 3))
    result = _estimate_intrinsic_dimension(points, representation='mu', estimator_name='mle',
                                           config=DimensionalityMetricConfig())
    assert np.isnan(result['value'])
    assert 'no automatic clipping' in result['warning']


def test_alignment_and_sample_basis_provenance(tmp_path):
    points = np.random.default_rng(13).normal(size=(80, 3))
    inputs = SimpleNamespace(x_test=points, mu_test=points, z_test=points, run_name='example')
    indices = np.arange(40)
    frame = run_dimensionality_metrics_pipeline(inputs, indices, points[indices], points[indices],
                                                DimensionalityMetricConfig(), tmp_path)
    assert len(frame) == 8
    path = tmp_path / 'absolute_core_metrics_v1' / 'dimensionality' / 'example' / 'provenance.json'
    meta = json.loads(path.read_text())
    assert meta['effective_rank_n'] == 80
    assert meta['intrinsic_dimension_n'] == 40
    with pytest.raises(ValueError, match='subset'):
        run_dimensionality_metrics_pipeline(inputs, indices, points[indices][::-1], points[indices],
                                            DimensionalityMetricConfig())


@pytest.mark.parametrize('kwargs', [{'mle_n_neighbors': 2}, {'mle_n_neighbors': True},
                                    {'mle_n_neighbors': 3.5}, {'twonn_discard_fraction': np.nan},
                                    {'twonn_discard_fraction': True}])
def test_parameter_validation(kwargs):
    with pytest.raises(ValueError):
        DimensionalityMetricConfig(**kwargs)
