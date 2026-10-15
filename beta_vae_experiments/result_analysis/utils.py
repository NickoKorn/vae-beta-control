"""Shared artifact readers and descriptive statistics for saved experiments."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SWEEP = ROOT / 'beta_vae_experiments/new_results_2026-09-29/biology_breakpoints_2026_09_29'
DEFAULT_X = ROOT / 'archive_06_10_26/results/biology_no_annealing_2026-09-28/biology/input_data/x_splits_split_seed2.npz'

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def posterior(folder,split):
    with np.load(folder/f'posterior_{split}.npz',allow_pickle=False) as a:
        mu,lv,ids=a['mu'].astype(float),a['logvar'].astype(float),a['sample_ids'].astype(str)
    if mu.shape!=lv.shape or len(mu)!=len(ids) or len(set(ids))!=len(ids):raise ValueError(f'Invalid posterior {folder}, {split}')
    return mu,lv,ids


def correlation(a, b, method='spearman'):
    pair = pd.concat([a, b], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 4 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return np.nan
    return pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method)

def breakpoint(beta, rate):
    good = np.isfinite(beta) & np.isfinite(rate)
    b, y = np.asarray(beta)[good], np.asarray(rate)[good]
    if len(b) < 7 or np.ptp(y) <= 1e-12 or not np.any(b > 0):
        return np.nan, np.nan
    x = np.log1p(b / b[b > 0].min())
    baseline = np.column_stack([np.ones(len(x)), x])
    sse0 = np.sum((y - baseline @ np.linalg.lstsq(baseline, y, rcond=None)[0])**2)
    fits = []
    for k in range(2, len(x)-2):
        design = np.column_stack([baseline, np.maximum(0, x-x[k])])
        sse = np.sum((y-design @ np.linalg.lstsq(design,y,rcond=None)[0])**2)
        fits.append((sse,b[k]))
    sse,beta_knee = min(fits)
    return beta_knee, 1-sse/sse0 if sse0 > 1e-20 else np.nan
