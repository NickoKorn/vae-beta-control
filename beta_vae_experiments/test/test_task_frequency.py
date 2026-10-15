"""Tests for metrics/graph_signals/task_frequency.py (Part A: task frequencies and bounds)."""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

_DIR = Path(__file__).resolve().parents[1] / "metrics" / "graph_signals"
for _name in ("graph_spectrum", "task_frequency"):
    if _name not in sys.modules:
        _spec = importlib.util.spec_from_file_location(_name, _DIR / f"{_name}.py")
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules[_name] = _mod
        _spec.loader.exec_module(_mod)
gs, tf = sys.modules["graph_spectrum"], sys.modules["task_frequency"]


@pytest.fixture(scope="module")
def blobs_with_noise_class():
    """3 separated Gaussian blobs plus a high-frequency extra class: random labels inside the blobs."""
    X, blob = make_blobs(n_samples=600, centers=3, cluster_std=1.0, center_box=(-20, 20), random_state=4)
    rand = np.random.default_rng(0).integers(0, 2, len(X))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, L = gs.build_graph(X, k=15)
    U, lam = gs.fourier_basis(L)
    return X, blob, rand, U, lam


def test_band0_energy_vanishes_after_u0_projection(blobs_with_noise_class):
    _, _, _, U, _ = blobs_with_noise_class
    S = np.random.default_rng(1).normal(size=(U.shape[0], 3))
    chk = tf.check_band0(U, S)
    assert np.all(chk["u0_projected"] < 1e-20)


def test_class_cutoff_ranks_random_class_higher(blobs_with_noise_class):
    _, blob, rand, U, lam = blobs_with_noise_class
    cb = tf.class_cutoff(U, lam, blob.astype(str), n_boot=20)
    cr = tf.class_cutoff(U, lam, np.where(rand == 1, "noise", "other"), n_boot=20)
    assert cr["lambda_c"][list(cr["classes"]).index("noise")] > cb["lambda_c"].max()


def test_band_predictive_does_not_put_random_class_in_low_band(blobs_with_noise_class):
    _, blob, rand, U, lam = blobs_with_noise_class
    bands = tf.structural_bands(lam, k_eig=3)
    tasks = {"blob0": {"rows": np.arange(len(blob)), "y": (blob == 0).astype(int)}, "noise": {"rows": np.arange(len(blob)), "y": rand}}
    P = tf.band_predictive(U, tasks, bands)
    assert P["blob0"]["band"][1] > 0.95 and P["noise"]["band"][1] < 0.65


@pytest.fixture(scope="module")
def circle_lowpass():
    """Circle; μ = progressively low-pass filtered copy of the first 40 eigenvectors; classes = sign of sin(jθ)."""
    theta = np.sort(np.random.default_rng(0).uniform(0, 2 * np.pi, 600))
    X = np.column_stack([np.cos(theta), np.sin(theta)])
    _, L = gs.build_graph(X, k=10)
    U, lam = gs.fourier_basis(L)
    Z = U[:, 1:41]
    m_steps = np.array([40, 24, 12, 6, 2])  # decreasing number of kept non-trivial frequencies = increasing "β"
    mus = [gs.band_filter(U, Z, gs.lowpass(m + 1)) for m in m_steps]
    bands = [np.array([0])] + [np.arange(a, b) for a, b in [(1, 3), (3, 5), (5, 9), (9, 17), (17, 33), (33, 600)]]
    return theta, U, lam, mus, bands


def test_model_cutoff_falls_monotonically(circle_lowpass):
    _, U, lam, mus, bands = circle_lowpass
    S = np.vstack([tf.band_preservation(U, mu, bands) for mu in mus])
    ls = tf.model_cutoff(S, tf.band_upper_lambda(lam, bands), thr=0.5)
    assert np.all(np.diff(ls) <= 0) and ls[0] > ls[-1]


def test_validate_bounds_on_simulated_lowpass(circle_lowpass):
    theta, U, lam, mus, bands = circle_lowpass
    js = [1, 2, 4, 8]
    lam_c = np.array([tf.class_cutoff(U, lam, np.where(np.sin(j * theta) > 0, "pos", "neg"), n_boot=10)["lambda_c"][1] for j in js])
    S = np.vstack([tf.band_preservation(U, mu, bands) for mu in mus])
    ls = tf.model_cutoff(S, tf.band_upper_lambda(lam, bands), thr=0.5)
    steps = np.arange(len(mus), dtype=float)
    pred = tf.predicted_loss_beta(lam_c, ls, steps)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    auc = np.array([[roc_auc_score(y, cross_val_predict(LogisticRegression(max_iter=2000), mu, y, cv=cv, method="predict_proba")[:, 1])
                     for y in [(np.sin(j * theta) > 0).astype(int) for j in js]] for mu in mus])
    meas = tf.measured_loss_beta(auc, steps, thr=0.9)
    res = tf.validate_bounds(pred[None, :], meas[None, :])
    assert res["rho_median"] > 0.8


def test_structural_bands_and_enrichment():
    lam = np.linspace(0, 1, 200)
    bands = tf.structural_bands(lam, k_eig=6, edges=(30, 150))
    assert [len(b) for b in bands] == [1, 5, 24, 120, 50]
    E = np.array([0.0, 0.5, 0.2, 0.2, 0.1])
    en = tf.enrichment(E, bands)
    assert np.isnan(en[0]) and en[1] == pytest.approx(0.5 / (5 / 199))


def test_sentinel_lead_counts_steps():
    S = np.array([1.0, 1.0, 0.99, 0.80, 0.60, 0.40])
    auc = np.array([0.97, 0.97, 0.97, 0.97, 0.90, 0.70])
    assert tf.sentinel_lead(S, auc) == 1.0
    assert tf.sentinel_band(0.2, np.array([0.0, 0.05, 0.15, 0.3, 0.6])) == 3


def test_predictive_cutoff_rises_with_comb_frequency():
    theta = np.sort(np.random.default_rng(1).uniform(0, 2 * np.pi, 400))
    X = np.column_stack([np.cos(theta), np.sin(theta)])
    lam_c = []
    for j in (1, 3, 6):
        cc = tf.class_cutoff_predictive(X, np.where(np.sin(j * theta) > 0, "pos", "neg"), k=10, m_grid=(1, 2, 4, 6, 8, 12, 16, 20), n_boot=3)
        lam_c.append(cc["lambda_c"][1])
        assert cc["ci_low"][1] <= cc["ci_high"][1]
    assert lam_c[0] < lam_c[1] < lam_c[2]


def test_energy_cutoff_falls_under_lowpass(circle_lowpass):
    _, U, lam, mus, _ = circle_lowpass
    ls = np.array([tf.model_cutoff_energy(U, lam, mu, q=0.8) for mu in mus])
    assert np.all(np.diff(ls) <= 1e-12) and ls[0] > ls[-1]


def test_k_task_excludes_read_and_ucs():
    table = {"task": np.array(["cancer:BRCA", "cancer:BRCA", "cancer:UCS", "cancer:UCS"]), "band": np.array([1, 2, 1, 2]),
             "n_vectors": np.array([5, 24, 5, 24]), "class": np.array(["relevant", "label_only", "label_only", "relevant"], dtype=object)}
    assert tf.k_task(table) == 5 and tf.k_task(table, exclude=()) == 29
