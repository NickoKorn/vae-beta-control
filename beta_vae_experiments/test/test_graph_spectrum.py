"""Tests for the graph-frequency module metrics/graph_signals/graph_spectrum.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_swiss_roll

# Load the module by file path: importing the package beta_vae_experiments.metrics currently
# fails on an unrelated circular import in metrics/topology (landscape ↔ __init__).
_PATH = Path(__file__).resolve().parents[1] / "metrics" / "graph_signals" / "graph_spectrum.py"
_spec = importlib.util.spec_from_file_location("graph_spectrum", _PATH)
gs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gs)


def _circle(n: int = 1000, seed: int = 0):
    theta = np.sort(np.random.default_rng(seed).uniform(0, 2 * np.pi, n))
    return np.column_stack([np.cos(theta), np.sin(theta)]), theta


@pytest.fixture(scope="module")
def circle_basis():
    X, theta = _circle()
    _, L = gs.build_graph(X, k=10)  # defaults: binary weights, normalized Laplacian
    U, lam = gs.fourier_basis(L)
    return X, theta, U, lam


@pytest.fixture(scope="module")
def swiss_roll_basis():
    X, t = make_swiss_roll(n_samples=1500, noise=0.0, random_state=0)
    _, L = gs.build_graph(X, k=12)
    U, lam = gs.fourier_basis(L, n_eig=40)
    return X, t, U, lam


def test_circle_first_pair_explains_cos_sin(circle_basis):
    X, theta, U, _ = circle_basis
    target = np.column_stack([np.cos(theta), np.sin(theta)])
    res = gs.structure_band_preservation(U, target, k=2, seeds=(0,))
    assert res["S"] > 0.95
    out = gs.structure_k(U, target, max_m=6, kind="reg", seeds=(0,))
    assert out["k_eig"] == 2 and out["curve"][1] > 0.95


def test_circle_has_two_unique_directions(circle_basis):
    """Intrinsic dimension 1, but two unique eigendirections: a closed loop cannot be written as a
    function of one coordinate (cos θ does not determine the sign of sin θ), so φ₂ is no harmonic
    of φ₁. From φ₃ = cos 2θ on, every eigenvector is a function of (φ₁, φ₂)."""
    _, _, U, lam = circle_basis
    uniq = gs.unique_eigendirections(U, lam, max_m=8)
    assert uniq["k_dim"] == 2 and uniq["unique"][:2].all() and (uniq["residuals"][2:] < 0.2).all()


def test_eigengap_finds_eight_blobs():
    X, _ = make_blobs(n_samples=800, centers=8, cluster_std=0.3, center_box=(-40, 40), random_state=3)
    with pytest.warns(UserWarning, match="connected components"):
        _, L = gs.build_graph(X, k=10)
    _, lam = gs.fourier_basis(L, n_eig=20)
    assert gs.eigengap_k(lam, max_c=15) == 8


def test_swiss_roll_two_unique_directions(swiss_roll_basis):
    """The roll (length ≈ 90) is much longer than high (21): the first modes are harmonics along
    the roll, so several eigenvectors are needed (k_eig > 2) but only two are unique (k_dim = 2)."""
    X, t, U, lam = swiss_roll_basis
    out = gs.structure_k(U, np.column_stack([t, X[:, 1]]), max_m=30, kind="reg", seeds=(0,), thresh=0.9)
    assert out["k_eig"] > 2 and out["k_dim"] == 2


def test_select_indices_matches_topology_selection():
    n = 3000
    expected = np.sort(np.random.default_rng(31).choice(n, 1000, replace=False))  # metric_pipeline._select_test_rows
    assert np.array_equal(gs.select_indices(n), expected)
    assert np.array_equal(gs.select_indices(707), np.arange(707))


def test_band_filter_with_all_frequencies_is_identity(circle_basis):
    _, _, U, _ = circle_basis
    mu = np.random.default_rng(1).normal(size=(U.shape[0], 4))
    assert np.allclose(gs.band_filter(U, mu, np.arange(U.shape[1])), mu)


def test_frequency_response_on_circle(circle_basis):
    X, theta, U, lam = circle_basis
    mu = np.column_stack([np.cos(theta), np.sin(theta)]) + 0.05 * np.random.default_rng(2).normal(size=(len(theta), 2))
    bands = [np.array([0]), np.array([1, 2]), np.arange(3, 50), np.arange(50, U.shape[1])]
    S_of = lambda X_, mu_: gs.structure_band_preservation(U, mu_, k=2, seeds=(0,))["S"]
    resp = gs.frequency_response(U, X, mu, bands, {"S": S_of})
    names = list(resp["conditions"])
    assert resp["values"][0, names.index("bandstop_1")] < 0.5
    assert resp["values"][0, names.index("bandstop_3")] > 0.95


def test_bands_zero_first_and_energies_sum_to_one(circle_basis):
    _, theta, U, lam = circle_basis
    bands = gs.make_bands(lam, 4)
    assert 0 in bands[0] and len(bands[0]) == 1
    E = gs.band_energy(gs.gft(U, np.cos(theta)), bands)
    assert np.isclose(E.sum(), 1.0)


def test_ladder_and_gamma():
    g = gs.gamma_from_k(k=3, d=8, beta_min=1e-5, beta_safe=1e-4, beta_max=1e-3)
    assert g == pytest.approx(1.9, abs=0.05)
    b = gs.ladder(8, 1e-5, 1e-3, g)
    assert b[2] == pytest.approx(1e-4, rel=1e-6) and b[0] == 1e-5 and b[-1] == pytest.approx(1e-3)
    assert gs.ordering_score(b) == pytest.approx(1.0)


def test_basis_cache_roundtrip(tmp_path, circle_basis):
    _, _, U, lam = circle_basis
    idx = gs.select_indices(len(lam), None)
    path = gs.save_basis(gs.basis_cache_path(tmp_path, "circle", "test", 10, "normalized"), U[:, :5], lam[:5], idx, k=10)
    back = gs.load_basis(path)
    assert np.array_equal(back["indices"], idx) and np.allclose(back["U"], U[:, :5])
