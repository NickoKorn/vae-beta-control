"""Descriptive spectral attribution, using the existing graph_spectrum basis.

Sample centring is intentional: a constant posterior mean has zero energy,
including for a normalized Laplacian. All component-null modes remain in B0.
No controller, no training changes. A reference test graph is transductive.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from . import graph_spectrum as gs


def align_rows(values, ids, reference_ids):
    ids, ref = np.asarray(ids).astype(str), np.asarray(reference_ids).astype(str)
    if len(set(ids)) != len(ids) or len(set(ref)) != len(ref):
        raise ValueError('Duplicate sample IDs')
    if len(values) != len(ids):
        raise ValueError('Data and sample IDs have different lengths')
    lookup = {v: i for i, v in enumerate(ids)}
    if not set(ref).issubset(lookup):
        raise ValueError('Missing reference sample IDs')
    return np.asarray(values)[[lookup[v] for v in ref]]


def reference_bands(lam):
    """Reuse F0's six logarithmic bands, then merge any split eigenvalue blocks.

    Near degeneracy uses |delta lambda| <= 1e-8 * max(1, lambda_max).
    Band labels retain the lowest merged band number. B0 contains all zeros.
    """
    lam = np.asarray(lam, float)
    if lam.ndim != 1 or len(lam) == 0 or np.any(np.diff(lam) < -1e-10):
        raise ValueError('Eigenvalues must be nonempty and ascending')
    bands = gs.make_bands(lam, 6, scheme='log')
    tol = 1e-8 * max(1., lam.max())
    merged, names = [], []
    for b, idx in enumerate(bands):
        if len(idx) == 0:
            continue
        if merged and b > 1 and abs(lam[idx[0]] - lam[merged[-1][-1]]) <= tol:
            merged[-1] = np.r_[merged[-1], idx]
        else:
            merged.append(idx); names.append(f'B{b}')
    return merged, names


def fingerprints(U, lam, mu, logvar, bands, *, validate_basis=True):
    """Energy fractions use total centred signal energy, including uncaptured energy."""
    U, lam, mu, lv = [np.asarray(v, float) for v in (U, lam, mu, logvar)]
    if mu.ndim != 2 or lv.shape != mu.shape or U.ndim != 2 or U.shape[0] != len(mu) or U.shape[1] != len(lam):
        raise ValueError('Inconsistent basis/posterior shapes')
    if not all(np.isfinite(v).all() for v in (U, lam, mu, lv)):
        raise ValueError('Nonfinite basis/posterior')
    if validate_basis and not np.allclose(U.T @ U, np.eye(U.shape[1]), atol=1e-8):
        raise ValueError('Basis must be orthonormal')
    indices = np.concatenate(bands)
    if len(indices) != U.shape[1] or not np.array_equal(np.sort(indices), np.arange(U.shape[1])):
        raise ValueError('Bands must partition every captured mode exactly once')
    centred = mu - mu.mean(0)
    C = U.T @ centred; P = C**2
    total = np.einsum('ij,ij->j', centred, centred)
    energy = np.stack([P[b].sum(0) for b in bands], axis=1)
    missing = total - energy.sum(1)
    if np.any(missing < -1e-8 * np.maximum(total, 1)):
        raise ValueError('Captured energy exceeds signal energy')
    missing = np.maximum(missing, 0)
    energy = np.column_stack([energy, missing])
    fraction = np.divide(energy, total[:, None], out=np.full_like(energy, np.nan), where=total[:, None] > 1e-24)
    captured = P.sum(0)
    centroid = np.divide((lam[:, None]*P).sum(0), captured, out=np.full(mu.shape[1], np.nan), where=captured > 1e-24)
    # Entropy over band subspaces, invariant to rotation within a band.
    entropy = -np.sum(np.where(fraction > 0, fraction * np.log(np.maximum(fraction, 1e-300)), 0), axis=1)
    entropy[total <= 1e-24] = np.nan
    kl = (.5*(mu**2 + np.exp(lv)-1-lv)).mean(0)
    return dict(energy=energy, fraction=fraction, total=total, KL=kl,
                variance=mu.var(0), active=mu.var(0) > .01,
                centroid=centroid, entropy=entropy)


def match_dimensions(reference_mu, mu):
    """Descriptive Hungarian alignment of |activation correlation| to beta=0.

    Test activation alignment is not an inductive procedure. Degenerate or
    collapsed coordinates remain unaligned (-1), rather than inventing matches.
    """
    if reference_mu.shape != mu.shape:
        raise ValueError('Matching requires equal sample count and latent dimension')
    a, b = reference_mu-reference_mu.mean(0), mu-mu.mean(0)
    na, nb = np.linalg.norm(a, axis=0), np.linalg.norm(b, axis=0)
    r = np.divide(a.T@b, na[:, None]*nb[None, :], out=np.zeros((a.shape[1], b.shape[1])), where=(na[:, None]*nb[None, :]) > 1e-24)
    rows, cols = linear_sum_assignment(-abs(r))
    matches = np.full(b.shape[1], -1, int); quality = np.full(b.shape[1], np.nan)
    for i, j in zip(rows, cols):
        if na[i] > 1e-12 and nb[j] > 1e-12:
            matches[j] = i; quality[j] = abs(r[i,j])
    return matches, quality


def probe_ablation(train, evaluation, ytrain, yeval, groups, retrain=True):
    """Same logistic probe specification as native AUTOENCODIX, fit on train only.

    Fixed ablation replaces coordinates by their TRAIN means. Retrained
    ablation also refits on ablated train. Groups permit redundancy checks.
    """
    clf = LogisticRegression(max_iter=1000).fit(train, ytrain)
    if set(np.unique(yeval)) != set(clf.classes_):
        raise ValueError('Train/evaluation classes differ; no silent label dropping')
    def score(model, x):
        p=model.predict_proba(x)
        return roc_auc_score(yeval,p,multi_class='ovo',labels=model.classes_) if len(model.classes_)>2 else roc_auc_score(yeval==model.classes_[1],p[:,1])
    full = score(clf,evaluation); mean = train.mean(0); rows=[]
    for name, idx in groups:
        idx=np.asarray(idx,int); ev=evaluation.copy();ev[:,idx]=mean[idx]
        fixed=score(clf,ev)
        row=dict(group=name, dimensions=','.join(map(str,idx)),AUC_full=full,
                 AUC_fixed=fixed,delta_auc_fixed=full-fixed)
        if retrain:
            tr=train.copy();tr[:,idx]=mean[idx]
            fresh=LogisticRegression(max_iter=1000).fit(tr,ytrain);re=score(fresh,ev)
            row.update(AUC_retrained=re,delta_auc_retrained=full-re)
        rows.append(row)
    return rows


def ridge_reconstruct(U, mu, bands, seed=31, folds=5):
    """Nested RidgeCV: scalers and alpha selection are fitted inside each outer fold.

    Fold train and evaluation are disjoint reference TEST graph nodes. This is
    out-of-fold transductive R2, not a new-patient inductive score.
    """
    model=make_pipeline(StandardScaler(),RidgeCV(alphas=[.001,.01,.1,1.,10.]))
    pred=cross_val_predict(model,mu,U,cv=KFold(folds,shuffle=True,random_state=seed))
    residual=((U-pred)**2).sum(0);denom=((U-U.mean(0))**2).sum(0)
    per=np.divide(residual,denom,out=np.full_like(denom,np.nan),where=denom>1e-24)
    band=[]
    for idx in bands:
        total=denom[idx].sum();band.append(1-residual[idx].sum()/total if total>1e-24 else np.nan)
    return 1-per,np.array(band)
