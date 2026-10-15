"""Which graph frequencies carry which task, and which ones a model still keeps.

Builds on ``graph_spectrum`` (same directory). All signals live on the fixed
sample set the graph was built on (the topology/geometry evaluation subset).
Labels are used for calibration only.

Note on band 0 (A1): with the normalized Laplacian the first eigenvector is
u₀ ∝ D^½·1, not the constant vector. Mean-centring (``graph_spectrum.gft``)
therefore leaves a small component along u₀ (≈ 0.1 % of the energy on TCGA).
Here signals are instead centred by projecting out u₀ (``center_u0``), which
makes the band-0 energy exactly zero; band 0 is excluded everywhere.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict


def _load_graph_spectrum():
    name = "graph_spectrum"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("graph_spectrum.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GS = _load_graph_spectrum()
RELATED_PAIRS = (("COAD", "READ"), ("UCEC", "UCS"))  # hardly separable in the x graph: "related", not a compression loss


# --------------------------------------------------------------------- A1 centring and band 0
def center_u0(U: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Remove the component along u₀ (first eigenvector) from signals S (n, s)."""
    S = GS._as_2d(S)
    u0 = U[:, :1]
    return S - u0 @ (u0.T @ S)


def check_band0(U: np.ndarray, S: np.ndarray) -> dict[str, np.ndarray]:
    """Band-0 energy share (s,) after mean-centring (graph_spectrum.gft) and after projecting out u₀."""
    P_mean = GS.gft(U, S) ** 2
    P_u0 = (U.T @ center_u0(U, S)) ** 2
    tiny = np.finfo(float).tiny
    return {"mean_centered": P_mean[0] / np.maximum(P_mean.sum(0), tiny), "u0_projected": P_u0[0] / np.maximum(P_u0.sum(0), tiny)}


def structural_bands(lam: np.ndarray, k_eig: int, edges: Sequence[int] = (30, 150)) -> list[np.ndarray]:
    """Bands from the data structure: [0], [1 … k_eig−1] (below the eigengap), then index edges, then the rest."""
    n = len(lam)
    cuts = [1, k_eig, *[e for e in edges if k_eig < e < n], n]
    return [np.array([0])] + [np.arange(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


def band_energy_u0(U: np.ndarray, S: np.ndarray, bands: Sequence[np.ndarray]) -> np.ndarray:
    """Energy share per band (s, n_bands) of u₀-centred signals S (n, s)."""
    return GS.band_energy(U.T @ center_u0(U, S), bands)


def enrichment(energy: np.ndarray, bands: Sequence[np.ndarray], n_total: int | None = None) -> np.ndarray:
    """Energy share divided by the share a white-noise signal would get (band size / non-trivial vectors)."""
    sizes = np.array([len(b) for b in bands], dtype=np.float64)
    n_total = n_total or (sizes.sum() - sizes[0])
    expected = sizes / n_total
    expected[0] = np.nan
    return energy / expected


# --------------------------------------------------------------------- A2 preservation per band
def band_preservation(U: np.ndarray, mu: np.ndarray, bands: Sequence[np.ndarray], seeds: Sequence[int] = (0,), folds: int = 5) -> np.ndarray:
    """S_b per band (n_bands,): joint cross-validated R² of the linear regression μ (n, d) → U[:, band]. Band 0 → NaN."""
    mu = GS._as_2d(mu)
    out = np.full(len(bands), np.nan)
    for b, idx in enumerate(bands):
        if b == 0 or len(idx) == 0:
            continue
        T = U[:, idx]
        vals = []
        for s in seeds:
            pred = cross_val_predict(LinearRegression(), mu, T, cv=KFold(folds, shuffle=True, random_state=s))
            vals.append(1.0 - ((T - pred) ** 2).sum() / ((T - T.mean(0)) ** 2).sum())
        out[b] = float(np.mean(vals))
    return out


# --------------------------------------------------------------------- A3 three levels per task
def _cv_auc(F: np.ndarray, y: np.ndarray, seed: int, folds: int, C: float = 1.0, penalty: str = "l2") -> float:
    if len(np.unique(y)) < 2 or min(np.bincount(y.astype(int))) < folds:
        return np.nan
    clf = LogisticRegression(max_iter=3000, C=C, penalty=penalty, solver="liblinear" if penalty == "l1" else "lbfgs")
    p = cross_val_predict(clf, F, y, cv=StratifiedKFold(folds, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def task_signals(cancer: np.ndarray, sex: np.ndarray, mixed_types: Sequence[str] = ("COAD", "LUAD", "LUSC", "READ")) -> dict[str, dict]:
    """Task definitions on the fixed sample set (n,).

    cancer one-vs-rest per class: signal = indicator, rows = all;
    sex within type: signal = male − share of men in the own type (unknown → 0, all rows),
    probe rows = known sex within the mixed types, label = male.
    """
    cancer, sex = np.asarray(cancer).astype(str), np.asarray(sex).astype(str)
    out = {}
    for c in np.unique(cancer):
        y = (cancer == c).astype(int)
        out[f"cancer:{c}"] = {"signal": y.astype(float), "rows": np.arange(len(y)), "y": y, "related": any(c in p for p in RELATED_PAIRS)}
    known = np.isin(sex, ["Female", "Male"])
    male = (sex == "Male").astype(float)
    share = {c: male[(cancer == c) & known].mean() if ((cancer == c) & known).any() else 0.0 for c in np.unique(cancer)}
    within = np.where(known, male - np.array([share[c] for c in cancer]), 0.0)
    rows = np.flatnonzero(known & np.isin(cancer, list(mixed_types)))
    out["sex:within_type"] = {"signal": within, "rows": rows, "y": male[rows].astype(int), "related": False}
    return out


def label_energy(U: np.ndarray, tasks: dict[str, dict], bands: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """(a) energy share per band of each task signal: dict task → (n_bands,)."""
    return {t: band_energy_u0(U, d["signal"], bands)[0] for t, d in tasks.items()}


def band_predictive(
    U: np.ndarray, tasks: dict[str, dict], bands: Sequence[np.ndarray], cumulative_m: Sequence[int] = (), seeds: Sequence[int] = (0,), folds: int = 5, max_band_size: int = 200
) -> dict[str, dict[str, np.ndarray]]:
    """(b) cross-validated AUC of a logistic probe on the eigenvectors of one band (band-pass) and on the first m eigenvectors.

    Bands larger than max_band_size use their first max_band_size eigenvectors (cost; noted in 'truncated').
    Returns task → {'band': (n_bands,), 'cumulative': (len(cumulative_m),), 'truncated': (n_bands,) bool}.
    """
    res = {}
    for t, d in tasks.items():
        rows, y = d["rows"], d["y"]
        band_auc = np.full(len(bands), np.nan)
        trunc = np.zeros(len(bands), bool)
        for b, idx in enumerate(bands):
            if b == 0:
                continue
            trunc[b] = len(idx) > max_band_size
            F = U[np.ix_(rows, idx[:max_band_size])]
            band_auc[b] = np.nanmean([_cv_auc(F, y, s, folds) for s in seeds])
        cum = np.array([np.nanmean([_cv_auc(U[np.ix_(rows, np.arange(1, m + 1))], y, s, folds) for s in seeds]) for m in cumulative_m])
        res[t] = {"band": band_auc, "cumulative": cum, "truncated": trunc}
    return res


def l1_importance(U: np.ndarray, rows: np.ndarray, y: np.ndarray, max_m: int, C: float = 0.1) -> np.ndarray:
    """Optional: |coefficients| (max_m,) of an L1 logistic regression on the standardised first max_m non-trivial eigenvectors."""
    F = U[np.ix_(rows, np.arange(1, max_m + 1))]
    F = (F - F.mean(0)) / np.maximum(F.std(0), np.finfo(float).tiny)
    return np.abs(LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=3000).fit(F, y).coef_[0])


def band_used(
    U: np.ndarray, mu: np.ndarray, tasks: dict[str, dict], bands: Sequence[np.ndarray], seeds: Sequence[int] = (0,), folds: int = 5
) -> dict[str, np.ndarray]:
    """(c) AUC loss per band (n_bands,) when the band is removed from a fixed μ (n, d): AUC(μ) − AUC(bandstop_b μ)."""
    n_freq = U.shape[1]
    filtered = [None] + [GS.band_filter(U, mu, GS.bandstop(idx, n_freq)) for idx in bands[1:]]
    out = {}
    for t, d in tasks.items():
        rows, y = d["rows"], d["y"]
        base = np.nanmean([_cv_auc(GS._as_2d(mu)[rows], y, s, folds) for s in seeds])
        loss = np.full(len(bands), np.nan)
        for b in range(1, len(bands)):
            loss[b] = base - np.nanmean([_cv_auc(filtered[b][rows], y, s, folds) for s in seeds])
        out[t] = loss
    return out


def classify_bands(
    label_enrichment: np.ndarray, band_auc: np.ndarray, used_loss: np.ndarray, enrich_thr: float = 1.0, auc_thr: float = 0.6, loss_thr: float = 0.01
) -> np.ndarray:
    """Per band: 'relevant' (label ∧ predictive ∧ used), 'shortcut' (used, little label),
    'label_only' (label, not used), 'predictive_only', 'none'; band 0 → 'trivial'."""
    L = np.nan_to_num(label_enrichment, nan=0.0) >= enrich_thr
    P = np.nan_to_num(band_auc, nan=0.5) >= auc_thr
    Uu = np.nan_to_num(used_loss, nan=0.0) >= loss_thr
    cls = np.where(L & P & Uu, "relevant", np.where(Uu & ~L, "shortcut", np.where(L & ~Uu, "label_only", np.where(P & ~L & ~Uu, "predictive_only", "none"))))
    cls = cls.astype(object)
    cls[0] = "trivial"
    return cls


def task_band_table(U, mu, tasks, bands, seeds=(0,), folds=5, cumulative_m=(), **thr) -> dict[str, np.ndarray]:
    """A3 table, long format: task, related, band, n_vectors, label_energy, label_enrichment, band_auc, used_loss, class."""
    E = label_energy(U, tasks, bands)
    P = band_predictive(U, tasks, bands, cumulative_m=cumulative_m, seeds=seeds, folds=folds)
    Us = band_used(U, mu, tasks, bands, seeds=seeds, folds=folds)
    rows = {k: [] for k in ["task", "related", "band", "n_vectors", "label_energy", "label_enrichment", "band_auc", "used_loss", "class"]}
    for t in tasks:
        en = enrichment(E[t], bands)
        cls = classify_bands(en, P[t]["band"], Us[t], **thr)
        for b, idx in enumerate(bands):
            for k, v in [("task", t), ("related", tasks[t]["related"]), ("band", b), ("n_vectors", len(idx)), ("label_energy", E[t][b]), ("label_enrichment", en[b]),
                         ("band_auc", P[t]["band"][b]), ("used_loss", Us[t][b]), ("class", cls[b])]:
                rows[k].append(v)
    out = {k: np.asarray(v) for k, v in rows.items()}
    out["cumulative_m"] = np.asarray(cumulative_m)
    out["cumulative_auc"] = np.vstack([P[t]["cumulative"] for t in tasks]) if len(cumulative_m) else np.empty((len(tasks), 0))
    out["task_order"] = np.asarray(list(tasks))
    return out


# --------------------------------------------------------------------- A4 structure size
def k_task(table: dict[str, np.ndarray], protected_prefix: str = "cancer:", exclude: Sequence[str] = ("READ", "UCS"), class_col: str = "class") -> int:
    """Number of eigenvectors in bands that are 'relevant' (all three levels) for any protected task.

    exclude: protected classes left out (default READ and UCS: small, merged with COAD/UCEC in the graph).
    """
    tasks = table["task"].astype(str)
    keep = np.char.startswith(tasks, protected_prefix) & ~np.isin(tasks, [protected_prefix + e for e in exclude])
    m = keep & (table[class_col] == "relevant")
    bands = np.unique(table["band"][m])
    sizes = {int(b): int(n) for b, n in zip(table["band"], table["n_vectors"])}
    return int(sum(sizes[int(b)] for b in bands))


# --------------------------------------------------------------------- A5 bounds
def class_cutoff(U: np.ndarray, lam: np.ndarray, labels: np.ndarray, q: float = 0.8, n_boot: int = 200, seed: int = 0) -> dict[str, np.ndarray]:
    """λ_c per class (n_classes,): lowest λ below which a share q of the class-indicator energy lies (u₀ excluded).

    Bootstrap CI (2.5 %, 97.5 %): class members redrawn with replacement (indicator → counts),
    an approximation, since U itself is fixed on the sample set.
    """
    labels = np.asarray(labels).astype(str)
    classes = np.unique(labels)
    rng = np.random.default_rng(seed)
    lam = np.asarray(lam)

    def cutoff(S):
        P = (U.T @ center_u0(U, S)) ** 2
        P[0] = 0.0
        cum = np.cumsum(P, axis=0) / np.maximum(P.sum(0), np.finfo(float).tiny)
        return lam[np.argmax(cum >= q, axis=0)]

    ind = (labels[:, None] == classes[None, :]).astype(float)
    point = cutoff(ind)
    boots = np.empty((n_boot, len(classes)))
    for i in range(n_boot):  # loop over bootstrap draws, vectorised over classes
        counts = np.zeros_like(ind)
        for j, c in enumerate(classes):
            members = np.flatnonzero(labels == c)
            np.add.at(counts[:, j], rng.choice(members, len(members), replace=True), 1.0)
        boots[i] = cutoff(counts)
    return {"classes": classes, "lambda_c": point, "ci_low": np.percentile(boots, 2.5, axis=0), "ci_high": np.percentile(boots, 97.5, axis=0)}


def band_upper_lambda(lam: np.ndarray, bands: Sequence[np.ndarray]) -> np.ndarray:
    """Upper λ of each band (n_bands,)."""
    return np.array([np.max(np.asarray(lam)[b]) for b in bands])


def model_cutoff(S_b: np.ndarray, lam_bands: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """λ*(β) per checkpoint (n_ckpt,): upper λ of the highest band with S_b ≥ thr (band 0 ignored); 0 if none."""
    S_b = np.atleast_2d(S_b)
    ok = np.nan_to_num(S_b, nan=-np.inf) >= thr
    ok[:, 0] = False
    lam_b = np.asarray(lam_bands)
    return np.array([lam_b[np.flatnonzero(r)].max() if r.any() else 0.0 for r in ok])


def predicted_loss_beta(lambda_c: np.ndarray, lambda_star: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """β_c per class (n_classes,): first β (ascending) with λ*(β) < λ_c; inf if never."""
    order = np.argsort(betas)
    b, ls = np.asarray(betas)[order], np.asarray(lambda_star)[order]
    out = np.full(len(lambda_c), np.inf)
    for j, lc in enumerate(np.asarray(lambda_c)):  # loop over classes
        hit = np.flatnonzero(ls < lc)
        if len(hit):
            out[j] = b[hit[0]]
    return out


def measured_loss_beta(class_auc: np.ndarray, betas: np.ndarray, thr: float = 0.9) -> np.ndarray:
    """First β (ascending) at which the one-vs-rest AUC drops below thr, per class; class_auc (n_beta, n_classes); inf if never."""
    order = np.argsort(betas)
    A, b = np.asarray(class_auc)[order], np.asarray(betas)[order]
    low = np.nan_to_num(A, nan=1.0) < thr
    return np.array([b[np.flatnonzero(col)[0]] if col.any() else np.inf for col in low.T])


def validate_bounds(pred_beta: np.ndarray, meas_beta: np.ndarray) -> dict[str, np.ndarray]:
    """Spearman between predicted and measured loss β across classes, per trajectory (n_traj,) and median.

    pred_beta, meas_beta: (n_traj, n_classes); inf is ranked as the largest value.
    """
    P, M = np.atleast_2d(pred_beta), np.atleast_2d(meas_beta)
    big = np.nanmax(np.where(np.isfinite(np.r_[P, M]), np.r_[P, M], np.nan)) * 10 if np.isfinite(np.r_[P, M]).any() else 1.0
    rho = np.array([spearmanr(np.where(np.isfinite(p), p, big), np.where(np.isfinite(m), m, big))[0] for p, m in zip(P, M)])
    return {"rho_per_trajectory": rho, "rho_median": np.array(np.nanmedian(rho))}


def model_cutoff_energy(U: np.ndarray, lam: np.ndarray, mu: np.ndarray, q: float = 0.8) -> float:
    """λ*(β) for one model: lowest λ below which a share q of the energy of μ (n, d) lies (u₀-centred, summed over dims); NaN if μ ≡ 0."""
    P = ((U.T @ center_u0(U, mu)) ** 2).sum(axis=1)
    P[0] = 0.0
    if P.sum() <= 1e-24:
        return np.nan
    return float(np.asarray(lam)[np.argmax(np.cumsum(P) / P.sum() >= q)])


def _cumulative_curve(U: np.ndarray, y: np.ndarray, m_grid: Sequence[int], seed: int, folds: int) -> np.ndarray:
    return np.array([_cv_auc(U[:, 1 : m + 1], y, seed, folds) for m in m_grid])


def class_cutoff_predictive(
    X: np.ndarray,
    labels: np.ndarray,
    k: int = 20,
    laplacian: str = "normalized",
    m_grid: Sequence[int] = (1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 60, 80),
    frac: float = 0.95,
    n_boot: int = 50,
    subsample: float = 0.8,
    seed: int = 0,
    folds: int = 5,
) -> dict[str, np.ndarray]:
    """λ_c per class from predictive power: smallest m in m_grid at which the cross-validated AUC of a
    one-vs-rest probe on the first m non-trivial eigenvectors reaches frac · max over m_grid; λ_c = λ_m.

    Bootstrap: subsamples of `subsample` · n samples WITHOUT replacement, graph rebuilt each time
    (eigenvalues of the subsample graph). Returns classes, m, lambda_c, ci_low, ci_high (2.5 / 97.5 %), m_boot.
    """
    gs = GS
    labels = np.asarray(labels).astype(str)
    classes = np.unique(labels)

    def one(Xs, ls, s):
        _, L = gs.build_graph(Xs, k=k, laplacian=laplacian)
        U, lam = gs.fourier_basis(L, n_eig=max(m_grid) + 1)
        m_out, l_out = [], []
        for c in classes:  # loop over classes
            y = (ls == c).astype(int)
            curve = _cumulative_curve(U, y, m_grid, s, folds)
            if not np.isfinite(curve).any():
                m_out.append(np.nan); l_out.append(np.nan); continue
            i = int(np.argmax(np.nan_to_num(curve, nan=-1) >= frac * np.nanmax(curve)))
            m_out.append(m_grid[i]); l_out.append(lam[m_grid[i]])
        return np.array(m_out, dtype=float), np.array(l_out)

    m_pt, l_pt = one(np.asarray(X, dtype=np.float64), labels, seed)
    rng = np.random.default_rng(seed)
    n = len(labels)
    boot_l, boot_m = [], []
    for b in range(n_boot):  # loop over bootstrap subsamples
        idx = np.sort(rng.choice(n, int(round(subsample * n)), replace=False))
        mb, lb = one(np.asarray(X, dtype=np.float64)[idx], labels[idx], seed + b + 1)
        boot_m.append(mb); boot_l.append(lb)
    boot_l = np.array(boot_l)
    return {"classes": classes, "m": m_pt, "lambda_c": l_pt, "ci_low": np.nanpercentile(boot_l, 2.5, axis=0),
            "ci_high": np.nanpercentile(boot_l, 97.5, axis=0), "m_boot": np.array(boot_m)}


def null_distribution(
    U: np.ndarray, mu: np.ndarray, tasks: dict[str, dict], bands: Sequence[np.ndarray], n_draw: int = 200, seed: int = 0, folds: int = 5, max_band_size: int = 200
) -> dict[str, np.ndarray]:
    """Null for band_auc and used_loss: random sets of non-trivial eigenvectors of the same size as each band.

    Returns auc (n_draw, n_tasks, n_bands) and loss (n_draw, n_tasks, n_bands); band 0 → NaN.
    """
    rng = np.random.default_rng(seed)
    n_freq = U.shape[1]
    pool = np.arange(1, n_freq)
    names = list(tasks)
    mu = GS._as_2d(mu)
    base = {t: _cv_auc(mu[tasks[t]["rows"]], tasks[t]["y"], seed, folds) for t in names}
    auc = np.full((n_draw, len(names), len(bands)), np.nan)
    loss = np.full_like(auc, np.nan)
    for i in range(n_draw):  # loop over random draws
        for b, idx in enumerate(bands):
            if b == 0:
                continue
            rand = np.sort(rng.choice(pool, len(idx), replace=False))
            filt = GS.band_filter(U, mu, GS.bandstop(rand, n_freq))
            for j, t in enumerate(names):
                rows, y = tasks[t]["rows"], tasks[t]["y"]
                auc[i, j, b] = _cv_auc(U[np.ix_(rows, rand[:max_band_size])], y, seed + i, folds)
                loss[i, j, b] = base[t] - _cv_auc(filt[rows], y, seed + i, folds)
    return {"auc": auc, "loss": loss, "task_order": np.asarray(names)}


def classify_bands_null(label_enrichment, band_auc, used_loss, auc_null_p95, loss_null_p95, enrich_thr: float = 1.0) -> np.ndarray:
    """As classify_bands, but 'predictive' / 'used' only above the 95th percentile of the random-set null."""
    L = np.nan_to_num(label_enrichment, nan=0.0) >= enrich_thr
    P = np.nan_to_num(band_auc, nan=0.5) > np.nan_to_num(auc_null_p95, nan=np.inf)
    Uu = np.nan_to_num(used_loss, nan=0.0) > np.nan_to_num(loss_null_p95, nan=np.inf)
    cls = np.where(L & P & Uu, "relevant", np.where(Uu & ~L, "shortcut", np.where(L & ~Uu, "label_only", np.where(P & ~L & ~Uu, "predictive_only", "none")))).astype(object)
    cls[0] = "trivial"
    return cls


# --------------------------------------------------------------------- A6 sentinel band
def sentinel_band(lambda_protect: float, lam_bands: np.ndarray) -> int:
    """Index of the first band whose upper λ lies above the protection limit (band 0 excluded)."""
    lam_bands = np.asarray(lam_bands)
    above = np.flatnonzero(lam_bands > lambda_protect)
    above = above[above > 0]
    return int(above[0]) if len(above) else int(len(lam_bands) - 1)


def first_lasting_drop(values: np.ndarray, tau: float) -> int:
    """First index after which values stay more than tau (relative) below the start (mean of the first two); −1 if never."""
    v = np.asarray(values, dtype=np.float64)
    start = np.nanmean(v[:2])
    lost = np.nan_to_num(v, nan=-np.inf) < (1 - tau) * start
    ok = [i for i in range(len(v)) if lost[i:].all()]
    return ok[0] if ok else -1


def sentinel_lead(S_sentinel: np.ndarray, auc: np.ndarray, tau_sentinel: float = 0.05, tau_auc: float = 0.02) -> int | float:
    """Lead in β steps of the sentinel over the AUC within one trajectory (ascending β); NaN if either never drops."""
    i_s, i_a = first_lasting_drop(S_sentinel, tau_sentinel), first_lasting_drop(auc, tau_auc)
    return float(i_a - i_s) if (i_s >= 0 and i_a >= 0) else np.nan


__all__ = [
    "RELATED_PAIRS", "center_u0", "check_band0", "structural_bands", "band_energy_u0", "enrichment",
    "band_preservation", "task_signals", "label_energy", "band_predictive", "l1_importance", "band_used",
    "classify_bands", "task_band_table", "k_task", "class_cutoff", "band_upper_lambda", "model_cutoff",
    "predicted_loss_beta", "measured_loss_beta", "model_cutoff_energy", "class_cutoff_predictive", "null_distribution", "classify_bands_null", "validate_bounds", "sentinel_band", "first_lasting_drop", "sentinel_lead",
]
