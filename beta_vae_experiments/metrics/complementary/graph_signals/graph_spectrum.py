"""Graph-frequency analysis of latent representations (graph signal processing).

A kNN graph is built once on a fixed set of input samples X. Its Laplacian
eigenvectors U are the graph "frequencies" (low = smooth over the data
manifold, high = local detail). Latent coordinates, metrics and task labels
are then described by how much of their energy sits in which frequency band.

Conventions
    - Pure functions, numpy arrays in and out; results are dicts of arrays
      that can be stored with ``np.savez``.
    - U is only defined on the samples the graph was built on. Every latent
      matrix ``mu`` must be evaluated on exactly the same, stored index set.
    - Eigenvectors are only defined up to sign and, for repeated eigenvalues,
      up to rotation (e.g. the sine/cosine pair on a circle). Only subspace
      quantities are used: energies, R², never single signs.
    - Labels are used for calibration only (``structure_k``, ``task_spectrum``),
      never in a training loop.
    - Eigendecompositions run in float64; every random step takes a seed.

PyGSP 0.6.1 (``pygsp.graphs.NNGraph(Xin, NNtype, use_flann, center, rescale,
k, sigma, epsilon, symmetrize_type, dist_type, ...)``) is optional. The
default path uses scikit-learn for the kNN search and SciPy for the
Laplacian, so the graph does not depend on the PyGSP version.

Defaults: symmetric normalized Laplacian I − D^-½ W D^-½ on a binary kNN
graph, evaluated on the same fixed test-sample subset as the geometry and
topology metrics (``metric_pipeline._select_test_rows``).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, laplacian as csgraph_laplacian
from scipy.sparse.linalg import eigsh
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors

DENSE_MAX_SAMPLES = 5000


# --------------------------------------------------------------------- indices and caching
TOPOLOGY_MAX_SAMPLES = 1000   # TopologyMetricConfig.max_samples
TOPOLOGY_RANDOM_STATE = 31    # TopologyMetricConfig.random_state (= GeometryMetricConfig.random_state)


def select_indices(n_samples: int, max_samples: int | None = TOPOLOGY_MAX_SAMPLES, seed: int = TOPOLOGY_RANDOM_STATE) -> np.ndarray:
    """Fixed, sorted evaluation index set (m,).

    Defaults reproduce ``metric_pipeline._select_test_rows`` (limit = min(geometry 3000,
    topology 1000) = 1000, ``np.random.default_rng(31).choice(n, 1000, replace=False)``,
    sorted; all rows if n ≤ 1000, e.g. the 707 TCGA test samples). All samples if
    max_samples is None.
    """
    if max_samples is None or max_samples >= n_samples:
        return np.arange(n_samples)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_samples, size=max_samples, replace=False))


def basis_cache_path(root: str | Path, dataset: str, split: str, k: int, laplacian: str) -> Path:
    """Cache file for one (dataset, split, k, Laplacian type)."""
    return Path(root) / f"graph_basis_{dataset}_{split}_k{k}_{laplacian}.npz"


def save_basis(path: str | Path, U: np.ndarray, lam: np.ndarray, indices: np.ndarray, **meta) -> Path:
    """Store U (n, M), lam (M,), the sample index set (n,) and scalar metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, U=U, lam=lam, indices=np.asarray(indices), **{f"meta_{k}": np.asarray(v) for k, v in meta.items()})
    return path


def load_basis(path: str | Path) -> dict[str, np.ndarray]:
    """Inverse of save_basis: dict with U, lam, indices and meta_* entries."""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


# --------------------------------------------------------------------- graph and basis
def _pygsp_library():
    try:
        import pygsp as library  # the installed PyGSP, not this module
    except ImportError:
        return None
    return library if hasattr(library, "graphs") else None


def build_graph(
    X: np.ndarray,
    k: int = 20,
    laplacian: str = "normalized",
    weights: str = "binary",
    use_pygsp: bool = False,
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Symmetric kNN graph on X (n, p) and its Laplacian.

    weights "binary" (default): w_ij = 1 for kNN edges, the same neighbourhood
    the kNN metrics use; robust to uneven sampling density. "gaussian":
    w_ij = exp(-d_ij² / σ²) with σ = median distance to the k-th neighbour
    (a smaller σ nearly cuts the graph at sampling gaps). The directed kNN
    graph is symmetrised by averaging, (W + Wᵀ) / 2.
    laplacian: "normalized" (default, I − D^-½ W D^-½) or "combinatorial" (L = D − W).
    Returns W (n, n), L (n, n) as CSR float64. Warns if the graph has more
    than one connected component.
    """
    if laplacian not in ("combinatorial", "normalized"):
        raise ValueError("laplacian must be 'combinatorial' or 'normalized'")
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if not 0 < k < n:
        raise ValueError("k must lie in [1, n_samples - 1]")
    dist, idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    dist, idx = dist[:, 1:], idx[:, 1:]  # drop self
    if weights not in ("binary", "gaussian"):
        raise ValueError("weights must be 'binary' or 'gaussian'")
    sigma = float(np.median(dist[:, -1]))
    if sigma <= 0:
        raise ValueError("median k-th neighbour distance is zero; duplicate samples dominate")
    library = _pygsp_library() if (use_pygsp and weights == "gaussian") else None
    if library is not None:
        graph = library.graphs.NNGraph(X, NNtype="knn", k=k, sigma=sigma**2, center=False, rescale=False, symmetrize_type="average")
        W = sp.csr_matrix(graph.W, dtype=np.float64)
    else:
        if use_pygsp:
            warnings.warn("PyGSP path only for gaussian weights and an installed PyGSP; using scikit-learn/SciPy.")
        rows = np.repeat(np.arange(n), k)
        vals = np.ones(n * k) if weights == "binary" else np.exp(-(dist.ravel() / sigma) ** 2)
        W = sp.csr_matrix((vals, (rows, idx.ravel())), shape=(n, n))
        W = ((W + W.T) * 0.5).tocsr()
    n_comp, _ = connected_components(W, directed=False)
    if n_comp > 1:
        warnings.warn(f"kNN graph has {n_comp} connected components; Laplacian has {n_comp} zero eigenvalues.")
    L = csgraph_laplacian(W, normed=(laplacian == "normalized")).tocsr().astype(np.float64)
    return W, L


def fourier_basis(L: sp.spmatrix | np.ndarray, n_eig: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvectors U (n, M) and eigenvalues lam (M,) of L, ascending.

    Dense numpy.linalg.eigh when n ≤ 5000 or n_eig is None (all n vectors);
    otherwise scipy.sparse.linalg.eigsh for the first n_eig vectors.
    """
    n = L.shape[0]
    if n_eig is None or n <= DENSE_MAX_SAMPLES:
        dense = L.toarray() if sp.issparse(L) else np.asarray(L)
        lam, U = np.linalg.eigh(dense.astype(np.float64))
        if n_eig is not None:
            lam, U = lam[:n_eig], U[:, :n_eig]
    else:
        lam, U = eigsh(sp.csr_matrix(L, dtype=np.float64), k=n_eig, which="SA", v0=np.ones(n))
    order = np.argsort(lam)
    lam = np.clip(lam[order], 0.0, None)
    return U[:, order], lam


def make_bands(lam: np.ndarray, n_bands: int, scheme: str = "log", zero_tol: float = 1e-9) -> list[np.ndarray]:
    """Index arrays of frequency bands.

    Band 0 holds the (near-)zero eigenvalues, i.e. the constant vector(s).
    The remaining indices are split into n_bands bands with edges spaced
    "log" (geometric in λ), "linear" (in λ) or "index" (equal counts).
    Empty bands are dropped with a warning. Returns 1 + n_bands lists at most.
    """
    if scheme not in ("log", "linear", "index"):
        raise ValueError("scheme must be 'log', 'linear' or 'index'")
    lam = np.asarray(lam, dtype=np.float64)
    tol = zero_tol * max(float(lam.max()), 1.0)
    zero = np.flatnonzero(lam <= tol)
    rest = np.flatnonzero(lam > tol)
    bands = [zero]
    if len(rest) == 0:
        return bands
    if scheme == "index":
        parts = np.array_split(rest, n_bands)
    else:
        lo, hi = lam[rest].min(), lam[rest].max()
        edges = np.geomspace(lo, hi, n_bands + 1) if scheme == "log" else np.linspace(lo, hi, n_bands + 1)
        which = np.clip(np.searchsorted(edges, lam[rest], side="right") - 1, 0, n_bands - 1)
        parts = [rest[which == b] for b in range(n_bands)]
    kept = [p for p in parts if len(p)]
    if len(kept) < len(parts):
        warnings.warn(f"{len(parts) - len(kept)} empty frequency band(s) dropped.")
    return bands + kept


# --------------------------------------------------------------------- transforms and energies
def _as_2d(S: np.ndarray) -> np.ndarray:
    S = np.asarray(S, dtype=np.float64)
    return S[:, None] if S.ndim == 1 else S


def gft(U: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Graph Fourier coefficients C (M, s) of the column-centred signals S (n, s)."""
    S = _as_2d(S)
    return U.T @ (S - S.mean(axis=0))


def band_energy(C: np.ndarray, bands: Sequence[np.ndarray], normalize: bool = True) -> np.ndarray:
    """Energy per band (s, n_bands) from coefficients C (M, s).

    normalize=True divides by the total energy over all M coefficients
    (with a truncated basis: the captured energy, not ‖S‖²).
    """
    P = _as_2d(C) ** 2
    E = np.stack([P[b].sum(axis=0) for b in bands], axis=1)
    if normalize:
        E = E / np.maximum(P.sum(axis=0), np.finfo(float).tiny)[:, None]
    return E


def smoothness(U: np.ndarray, S: np.ndarray, k: int, lam: np.ndarray | None = None, L=None) -> dict[str, np.ndarray]:
    """Low-frequency share and Dirichlet energy per signal.

    low_share (s,): share of the energy in the first k frequencies.
    dirichlet (s,): xᵀLx / xᵀx of the centred signal; exact with L, spectral
    Σλc² / Σc² with lam (exact only for a full basis). NaN if neither given.
    """
    S = _as_2d(S)
    C = gft(U, S)
    P = C**2
    total = np.maximum(P.sum(axis=0), np.finfo(float).tiny)
    out = {"low_share": P[:k].sum(axis=0) / total}
    if L is not None:
        Sc = S - S.mean(axis=0)
        out["dirichlet"] = np.einsum("ij,ij->j", Sc, L @ Sc) / np.maximum(np.einsum("ij,ij->j", Sc, Sc), np.finfo(float).tiny)
    elif lam is not None:
        out["dirichlet"] = (np.asarray(lam)[:, None] * P).sum(axis=0) / total
    else:
        out["dirichlet"] = np.full(S.shape[1], np.nan)
    return out


# --------------------------------------------------------------------- structure size
def _cv_score(F: np.ndarray, y: np.ndarray, kind: str, seed: int, folds: int) -> float:
    if kind == "clf":
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        pred = cross_val_predict(LogisticRegression(max_iter=2000), F, y, cv=cv)
        return float(np.mean(pred == y))
    cv = KFold(n_splits=folds, shuffle=True, random_state=seed)
    pred = cross_val_predict(LinearRegression(), F, y, cv=cv)
    return float(r2_score(y, pred, multioutput="variance_weighted"))


def structure_k(
    U: np.ndarray,
    target: np.ndarray,
    max_m: int,
    kind: str = "clf",
    seeds: Sequence[int] = (0, 1, 2),
    thresh: float = 0.98,
    folds: int = 5,
    residual_threshold: float = 0.3,
) -> dict[str, np.ndarray | int]:
    """Size of the structure in eigenvectors (k_eig) and in unique directions (k_dim).

    k_eig: smallest m such that the non-constant eigenvectors U[:, 1:m+1] explain
    the target (curve ≥ thresh · max(curve)). kind "clf": cross-validated accuracy
    of a logistic regression (labels (n,)); kind "reg": cross-validated R² of a
    linear regression (target (n,) or (n, t)). Curve (max_m,) averaged over seeds.
    k_dim: number of unique eigendirections (not harmonics) among these k_eig
    eigenvectors, see unique_eigendirections. Uses labels: calibration only.
    """
    y = np.asarray(target)
    curve = np.array([np.mean([_cv_score(U[:, 1 : m + 1], y, kind, s, folds) for s in seeds]) for m in range(1, max_m + 1)])
    k_eig = int(np.flatnonzero(curve >= thresh * curve.max())[0] + 1)
    uniq = unique_eigendirections(U, None, k_eig, threshold=residual_threshold, seed=int(seeds[0]))
    return {"k_eig": k_eig, "k_dim": int(uniq["k_dim"]), "curve": curve, "residuals": uniq["residuals"]}


def unique_eigendirections(
    U: np.ndarray,
    lam: np.ndarray | None,
    max_m: int,
    threshold: float = 0.3,
    eps_scale: float = 3.0,
    n_subsample: int | None = 1000,
    seed: int = 0,
    chunk: int = 256,
) -> dict[str, np.ndarray | int]:
    """Separate unique eigendirections from harmonics by local linear regression.

    Dsilva, Talmon, Coifman & Kevrekidis (2018), "Parsimonious representation of
    nonlinear dynamical systems through manifold learning", Appl. Comput. Harmon.
    Anal. 44: 759–773; reference implementation: datafold
    ``LocalRegressionSelection`` (eps_med_scale = 3).
    For the non-trivial eigenvectors φ_1 … φ_max_m (U without the (near-)zero
    eigenvalues; without lam only column 0 is skipped), φ_k is predicted from
    φ_1 … φ_{k−1} by leave-one-out kernel-weighted local linear regression,
    kernel exp(−‖Φ_i − Φ_j‖² / ε²), ε = median pairwise distance / eps_scale.
    residuals (max_m,): r_k = sqrt(Σ(φ_k − φ̂_k)² / Σφ_k²), r_1 = 1 by definition.
    Small r_k: φ_k is a function of earlier ones (harmonic); r_k > threshold: unique.
    threshold 0.3 is our choice: on circle and Swiss roll the harmonics stay ≤ 0.18 and
    the unique directions are ≥ 0.55 (datafold's 'threshold' default 0.9 would drop the
    Swiss-roll height; its default strategy 'dim' fixes the number instead).
    Rows are subsampled (seeded) to n_subsample for speed. Returns residuals,
    unique (max_m,) bool, k_dim, columns (indices into U).
    """
    U = np.asarray(U, dtype=np.float64)
    if lam is None:
        cols = np.arange(1, U.shape[1])
    else:
        lam = np.asarray(lam, dtype=np.float64)
        cols = np.flatnonzero(lam > 1e-9 * max(float(lam.max()), 1.0))
    cols = cols[:max_m]
    Phi = U[:, cols]
    if n_subsample is not None and len(Phi) > n_subsample:
        Phi = Phi[np.sort(np.random.default_rng(seed).choice(len(Phi), n_subsample, replace=False))]
    n = len(Phi)
    res = np.ones(len(cols))
    for k in range(1, len(cols)):  # loop over eigenvectors, not samples
        X, y = Phi[:, :k], Phi[:, k]
        D2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
        eps = np.median(np.sqrt(D2[np.triu_indices(n, 1)])) / eps_scale
        W = np.exp(-D2 / max(eps**2, np.finfo(float).tiny))
        np.fill_diagonal(W, 0.0)  # leave-one-out
        Xt = np.column_stack([np.ones(n), X])
        pred = np.empty(n)
        ridge = 1e-10 * np.eye(k + 1)
        for start in range(0, n, chunk):  # vectorised within each chunk of rows
            Wc = W[start : start + chunk]
            A = np.einsum("cj,jp,jq->cpq", Wc, Xt, Xt) + ridge
            b = np.einsum("cj,jp,j->cp", Wc, Xt, y)
            beta = np.linalg.solve(A, b[..., None])[..., 0]
            pred[start : start + chunk] = np.einsum("cp,cp->c", Xt[start : start + chunk], beta)
        res[k] = np.sqrt(((y - pred) ** 2).sum() / max((y**2).sum(), np.finfo(float).tiny))
    unique = res > threshold
    return {"residuals": res, "unique": unique, "k_dim": int(unique.sum()), "columns": cols}


def eigengap_k(lam: np.ndarray, max_c: int) -> int:
    """Label-free number of clusters: position of the largest gap among the first max_c + 1 eigenvalues."""
    gaps = np.diff(np.asarray(lam, dtype=np.float64)[: max_c + 1])
    return int(np.argmax(gaps) + 1)


def structure_band_preservation(
    U: np.ndarray, mu: np.ndarray, k: int, seeds: Sequence[int] = (0, 1, 2), folds: int = 5
) -> dict[str, np.ndarray | float]:
    """How well μ (n, d) linearly reproduces the structure band U[:, 1:k+1].

    S: subspace R² = 1 − Σ residual² / Σ total² over the k targets (invariant
    to sign flips and to rotations inside the band, scale-free).
    r2_per_eig (k,): cross-validated R² per eigenvector (not rotation-invariant
    inside degenerate pairs; report for inspection only). Averaged over seeds.
    """
    T = U[:, 1 : k + 1]
    X = _as_2d(mu)
    S_vals, per = [], []
    for seed in seeds:
        pred = cross_val_predict(LinearRegression(), X, T, cv=KFold(n_splits=folds, shuffle=True, random_state=seed))
        resid, tot = ((T - pred) ** 2).sum(axis=0), ((T - T.mean(axis=0)) ** 2).sum(axis=0)
        S_vals.append(1.0 - resid.sum() / tot.sum())
        per.append(1.0 - resid / tot)
    return {"S": float(np.mean(S_vals)), "r2_per_eig": np.mean(per, axis=0)}


# --------------------------------------------------------------------- per-dimension profile
def dim_profile(
    U: np.ndarray,
    mu: np.ndarray,
    kl_per_dim: np.ndarray,
    k: int,
    bands: Sequence[np.ndarray],
    metric_fn: Callable[[np.ndarray], float] | None = None,
) -> dict[str, np.ndarray]:
    """Per latent dimension of μ (n, d).

    smooth (d,): energy share in the first k frequencies;
    band_energy (d, n_bands); kl (d,) as given;
    importance (d,): metric_fn(μ) − metric_fn(μ with dimension j set to 0,
    the prior mean); NaN without metric_fn.
    """
    mu = _as_2d(mu)
    C = gft(U, mu)
    P = C**2
    total = np.maximum(P.sum(axis=0), np.finfo(float).tiny)
    out = {"smooth": P[:k].sum(axis=0) / total, "band_energy": band_energy(C, bands), "kl": np.asarray(kl_per_dim, dtype=np.float64)}
    if metric_fn is None:
        out["importance"] = np.full(mu.shape[1], np.nan)
    else:
        base = float(metric_fn(mu))
        imp = []
        for j in range(mu.shape[1]):  # loop over latent dimensions, not samples
            ablated = mu.copy()
            ablated[:, j] = 0.0
            imp.append(base - float(metric_fn(ablated)))
        out["importance"] = np.asarray(imp)
    return out


# --------------------------------------------------------------------- filtering
def lowpass(m: int) -> np.ndarray:
    """Indices of the first m frequencies."""
    return np.arange(m)


def bandpass(band: np.ndarray) -> np.ndarray:
    """Indices of one band."""
    return np.asarray(band)


def bandstop(band: np.ndarray, n_freq: int) -> np.ndarray:
    """All frequency indices 0 … n_freq − 1 except those of one band."""
    return np.setdiff1d(np.arange(n_freq), band)


def band_filter(U: np.ndarray, mu: np.ndarray, keep_idx: np.ndarray) -> np.ndarray:
    """μ' = U_keep U_keepᵀ (μ − mean) + mean, (n, d). Identity if keep_idx covers a full basis."""
    mu = _as_2d(mu)
    mean = mu.mean(axis=0)
    Uk = U[:, np.asarray(keep_idx)]
    return Uk @ (Uk.T @ (mu - mean)) + mean


def frequency_response(
    U: np.ndarray,
    X: np.ndarray,
    mu: np.ndarray,
    bands: Sequence[np.ndarray],
    metric_fns: Mapping[str, Callable[[np.ndarray, np.ndarray], float]],
) -> dict[str, np.ndarray]:
    """Metric values after filtering μ, relative to the unfiltered value.

    Conditions: low-pass up to the end of each band, band-stop per band,
    band-pass per band (in this order). values (n_metrics, n_conditions)
    = metric(X, μ') / metric(X, μ).
    """
    n_freq = U.shape[1]
    conds, names = [], []
    for b, band in enumerate(bands):
        conds.append(lowpass(int(np.max(band)) + 1)); names.append(f"lowpass_to_band{b}")
    for b, band in enumerate(bands):
        conds.append(bandstop(band, n_freq)); names.append(f"bandstop_{b}")
    for b, band in enumerate(bands):
        conds.append(bandpass(band)); names.append(f"bandpass_{b}")
    metric_names = list(metric_fns)
    base = np.array([float(metric_fns[m](X, mu)) for m in metric_names])
    vals = np.array([[float(metric_fns[m](X, band_filter(U, mu, keep))) for keep in conds] for m in metric_names])
    return {"values": vals / base[:, None], "base": base, "conditions": np.array(names), "metrics": np.array(metric_names)}


# --------------------------------------------------------------------- tasks and checkpoints
def task_spectrum(U: np.ndarray, labels: np.ndarray, bands: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """Band energy (n_classes, n_bands) of the one-hot class indicators; mean over classes in 'mean'."""
    labels = np.asarray(labels)
    classes = np.unique(labels)
    onehot = (labels[:, None] == classes[None, :]).astype(np.float64)
    E = band_energy(gft(U, onehot), bands)
    return {"classes": classes, "energy": E, "mean": E.mean(axis=0)}


def band_energy_over_checkpoints(U: np.ndarray, list_of_mu: Sequence[np.ndarray], bands: Sequence[np.ndarray]) -> np.ndarray:
    """Band energy of μ summed over latent dimensions, per checkpoint: (n_ckpt, n_bands)."""
    rows = []
    for mu in list_of_mu:  # loop over checkpoints, not samples
        P = gft(U, mu) ** 2
        tot = P.sum(axis=1)
        rows.append(np.array([tot[b].sum() for b in bands]) / max(tot.sum(), np.finfo(float).tiny))
    return np.vstack(rows)


def correlate(
    band_energy_table: np.ndarray,
    metric_table: np.ndarray,
    trajectory: np.ndarray,
    beta: np.ndarray,
) -> dict[str, np.ndarray]:
    """Spearman of values and of drops per log-β step, as in the deck.

    band_energy_table (n_states, n_bands), metric_table (n_states, n_metrics),
    trajectory (n_states,) ids (one d and seed), beta (n_states,) > 0.
    Per trajectory: ρ(values) = Spearman(e, m); ρ(drops) = Spearman of
    Δe/Δlog β and Δm/Δlog β. Returns medians over trajectories,
    each (n_bands, n_metrics).
    """
    E, M = _as_2d(band_energy_table), _as_2d(metric_table)
    traj, beta = np.asarray(trajectory), np.asarray(beta, dtype=np.float64)
    lev, drp = [], []
    for t in np.unique(traj):
        idx = np.flatnonzero(traj == t)
        idx = idx[np.argsort(beta[idx])]
        if len(idx) < 4:
            continue
        lb = np.log(beta[idx])
        e, m = E[idx], M[idx]
        de, dm = np.diff(e, axis=0) / np.diff(lb)[:, None], np.diff(m, axis=0) / np.diff(lb)[:, None]
        lev.append([[spearmanr(e[:, i], m[:, j])[0] for j in range(M.shape[1])] for i in range(E.shape[1])])
        drp.append([[spearmanr(de[:, i], dm[:, j])[0] for j in range(M.shape[1])] for i in range(E.shape[1])])
    return {"values": np.nanmedian(np.array(lev), axis=0), "drops": np.nanmedian(np.array(drp), axis=0), "n_trajectories": np.array(len(lev))}


# --------------------------------------------------------------------- β ladder per latent dimension
def ladder(d: int, beta_min: float, beta_max: float, gamma: float) -> np.ndarray:
    """β_j = β_min + (β_max − β_min) · (j / (d − 1))^γ for j = 0 … d − 1, shape (d,)."""
    if d == 1:
        return np.array([beta_min], dtype=np.float64)
    t = np.arange(d, dtype=np.float64) / (d - 1)
    return beta_min + (beta_max - beta_min) * t**gamma


def gamma_from_k(k: int, d: int, beta_min: float, beta_safe: float, beta_max: float) -> float:
    """γ such that the k-th dimension (index j = k − 1) gets β_safe: requires 2 ≤ k ≤ d − 1 and β_min < β_safe < β_max."""
    if not 2 <= k <= d - 1:
        raise ValueError("gamma_from_k needs 2 <= k <= d - 1")
    if not beta_min < beta_safe < beta_max:
        raise ValueError("need beta_min < beta_safe < beta_max")
    return float(np.log((beta_safe - beta_min) / (beta_max - beta_min)) / np.log((k - 1) / (d - 1)))


def ordering_score(values_per_dim: np.ndarray) -> float:
    """Spearman between the dimension index j and the given values (d,)."""
    v = np.asarray(values_per_dim, dtype=np.float64)
    return float(spearmanr(np.arange(len(v)), v)[0])


__all__ = [
    "select_indices", "basis_cache_path", "save_basis", "load_basis",
    "build_graph", "fourier_basis", "make_bands", "gft", "band_energy", "smoothness",
    "structure_k", "unique_eigendirections", "eigengap_k", "structure_band_preservation", "dim_profile",
    "lowpass", "bandpass", "bandstop", "band_filter", "frequency_response",
    "task_spectrum", "band_energy_over_checkpoints", "correlate",
    "ladder", "gamma_from_k", "ordering_score",
]
