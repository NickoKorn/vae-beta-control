"""Official scikit-dimension TwoNN and pointwise MLE on unchanged features.

Global TwoNN and local MLE describe each space separately, without labels.
See graph_metrics.md; Bac et al. (2021), Facco et al. (2017), Levina & Bickel.

Duplicate points give NaN: TwoNN divides by the nearest-neighbour distance
r1 and MLE takes log(r_k/r_j), so a zero distance is undefined. On Varix
``mu`` this is common at small d, because means below 1e-7 are set to 0 and
many samples share the origin. A single non-finite pointwise MLE value makes
all local_id summaries NaN (see below), which is why all 14 graph_*_local_id
columns are missing together (about a third of non-collapsed runs). Not
repaired on purpose; see DimensionalityMetricConfig for details.
"""
import numpy as np
from skdim.id import MLE
from skdim.id import TwoNN


def compute_global_id(values):
    return {"global_id": float(TwoNN(discard_fraction=0.1).fit(values).dimension_)}


def compute_local_id(values, k):
    if not 3 <= k < len(values):
        raise ValueError("Local MLE requires 3 <= k < n; no automatic k change")
    fitted = MLE().fit(values, n_neighbors=k)
    local = np.asarray(fitted.dimension_pw_, dtype=float)
    if local.shape != (len(values),):
        raise ValueError("Unexpected pointwise MLE shape")
    # Never silently discard invalid pointwise values in summaries.
    metrics = {"local_id_"+name: float(func(local)) for name, func in
               {"mean": np.mean, "median": np.median, "std": np.std,
                "q10": lambda a: np.quantile(a, .1), "q25": lambda a: np.quantile(a, .25),
                "q75": lambda a: np.quantile(a, .75), "q90": lambda a: np.quantile(a, .9)}.items()}
    if not np.isfinite(local).all():
        metrics = {name: float("nan") for name in metrics}
        local = np.where(np.isfinite(local), local, np.nan)
    return metrics, {"local_id": local}
