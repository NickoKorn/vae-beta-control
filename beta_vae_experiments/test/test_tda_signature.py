"""Scientific checks for the additional correspondence-sensitive H0 score."""
import numpy as np
import pytest
from beta_vae_experiments.metrics.topology import (
    compute_h0_signature_preservation, SIGNATURE_METRIC_NAME,
)


def score(x, y, directory=None):
    row = compute_h0_signature_preservation(x, y, directory).iloc[0]
    assert row.metric_name == SIGNATURE_METRIC_NAME
    assert row.status == "ok", row.warning
    return row.value


def test_signature_identity_and_uniform_scaling():
    x = np.array([[0., 0.], [1., 0.], [3., 1.], [8., 2.]])
    assert score(x, x) == pytest.approx(0, abs=1e-12)
    assert score(x, x*7) == pytest.approx(0, abs=1e-12)


def test_signature_correspondence_and_symmetry():
    x = np.array([[0.], [1.], [3.], [8.]])
    # Exactly the same unlabelled cloud and diagram, but different point identities.
    y = x[[0, 2, 1, 3]]
    assert score(x, y) > 0
    assert score(x, y) == pytest.approx(score(y, x), abs=1e-12)


def test_signature_matches_saved_pairing_formula(tmp_path):
    x = np.array([[0.], [1.], [3.], [8.]])
    y = np.array([[0., 0.], [1., 1.], [2., 0.], [4., 2.]])
    value = score(x, y, tmp_path)
    dx = np.linalg.norm(x[:, None]-x[None, :], axis=-1)
    dy = np.linalg.norm(y[:, None]-y[None, :], axis=-1)
    delta = dx/dx.max() - dy/dy.max()
    edges_x = np.load(tmp_path/'signature_h0_input_pairings.npy')[:, 1:]
    edges_y = np.load(tmp_path/'signature_h0_latent_pairings.npy')[:, 1:]
    expected = .5 * sum(np.linalg.norm(delta[e[:, 0], e[:, 1]]) for e in (edges_x, edges_y))
    assert len(edges_x) == len(x)-1
    assert value == pytest.approx(expected, abs=1e-12)


def test_signature_collapse_is_explicit():
    row = compute_h0_signature_preservation(np.arange(4.)[:, None], np.zeros((4, 2))).iloc[0]
    assert row.status == 'zero_diameter'
    assert np.isnan(row.value)
