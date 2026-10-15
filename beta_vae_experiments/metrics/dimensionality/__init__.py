"""Library-backed dimensionality diagnostics for aligned VARIX representations.

No intrinsic-dimension estimator is reimplemented in this module. TwoNN and
MLE delegate to scikit-dimension 0.3.4.

Software reference:
    Bac, Mirkes, Gorban, Tyukin, and Zinovyev, Scikit-Dimension: A Python
    Package for Intrinsic Dimension Estimation, arXiv:2109.02596 (2021).

"""

from __future__ import annotations

from dataclasses import dataclass
import warnings
from importlib.metadata import version

import pandas as pd
import numpy as np
import skdim
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from beta_vae_experiments.metric_pipeline import CoreMetricInputs


@dataclass(frozen=True)
class DimensionalityMetricConfig:
    """Parameters passed to the official TwoNN and MLE estimators.

    Known limitation on ``mu`` (checked on biology_breakpoints_2026_09_29):
        AUTOENCODIX Varix sets posterior means below 1e-7 to 0, which acts like
        a ReLU on the encoder output. At small latent sizes many samples land on
        exactly the same point, mostly the origin (median 16 % of test samples
        in affected runs). TwoNN uses r2/r1 and MLE uses log(r_k/r_j); a
        zero nearest-neighbour distance makes both undefined, so the estimate
        is NaN. All 69 non-collapsed runs with NaN ``dimensionality_mu_mle``
        contain duplicate ``mu`` rows; no run without duplicates is NaN.
        Mostly d = 2 (28 of 40 runs), rarely d = 16 (5 of 40).
        On ``z`` the sampled noise (sigma^2 >= 1) removes ties, so
        ``dimensionality_z_*`` is complete and is the version used downstream.
        Deliberately not repaired here (no deduplication, no jitter): the point
        mass is an architecture artifact, and a repaired value would hide it.
    """

    twonn_discard_fraction: float = 0.1
    mle_n_neighbors: int = 20

    def __post_init__(self) -> None:
        if not 0 <= self.twonn_discard_fraction < 1:
            raise ValueError("TwoNN discard_fraction must lie in [0, 1)")
        if self.mle_n_neighbors < 2:
            raise ValueError("MLE n_neighbors must be at least two")

def compute_effective_rank_mu(mu: np.ndarray) -> list[dict[str, object]]:
    """Compute Roy--Vetterli effective rank of centered posterior means.

    Input representation:
        Complete test-set posterior means ``mu_test``.
    Metric family:
        Linear spectral/effective dimensionality.
    Mathematical definition:
        After centering ``mu``, let ``p_i=s_i/sum_j(s_j)`` for its positive
        singular values. Effective rank is ``exp(-sum_i p_i log(p_i))`` and
        its fraction is effective rank divided by the configured latent size.
    Scientific source:
        Roy and Vetterli, The Effective Rank: A Measure of Effective
        Dimensionality, EUSIPCO 2007.
    Repository provenance:
        Transparent NumPy reduction using ``numpy.linalg.svd``; no intrinsic-
        dimension estimator is reimplemented here.
    Interpretation:
        Continuous measure of how broadly posterior-mean variation is spread
        across linear latent directions.
    Limitations:
        It is sensitive only to linear singular-value structure and is not an
        intrinsic manifold-dimension estimator or an explicit collapse test.
    Direction:
        Descriptive; a larger fraction means less spectrally concentrated
        posterior-mean variation.
    In core metrics included:
        Yes; the normalized effective-rank fraction is the primary value.
    """
    values = np.asarray(mu, dtype=np.float64)
    centered = values - np.mean(values, axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    total = float(np.sum(singular_values))
    if total <= np.finfo(np.float64).eps:
        warning = "Centered posterior means have zero numerical rank."
        return [
            {"metric_name": "dimensionality_mu_effective_rank", "value": 0.0, "status": "degenerate_input", "warning": warning, "n_samples": len(values), "split": "test"},
            {"metric_name": "dimensionality_mu_effective_rank_fraction", "value": 0.0, "status": "degenerate_input", "warning": warning, "n_samples": len(values), "split": "test"},
        ]
    probabilities = singular_values[singular_values > 0] / total
    effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    return [
        {"metric_name": "dimensionality_mu_effective_rank", "value": effective_rank, "status": "ok", "warning": None, "n_samples": len(values), "split": "test"},
        {"metric_name": "dimensionality_mu_effective_rank_fraction", "value": effective_rank / values.shape[1], "status": "ok", "warning": None, "n_samples": len(values), "split": "test"},
    ]


def _estimate_intrinsic_dimension(
    values: np.ndarray,
    *,
    representation: str,
    estimator_name: str,
    config: DimensionalityMetricConfig,
) -> dict[str, object]:
    """Fit one public scikit-dimension estimator and expose its scalar result."""
    if estimator_name == "twonn":
        estimator = skdim.id.TwoNN(discard_fraction=config.twonn_discard_fraction)
    elif estimator_name == "mle":
        estimator = skdim.id.MLE()
    else:
        raise ValueError(
            f"Unsupported intrinsic-dimension estimator {estimator_name!r}"
        )

    metric_name = f"dimensionality_{representation}_{estimator_name}"
    captured: list[warnings.WarningMessage]
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            if estimator_name == "mle":
                fitted = estimator.fit(
                    values,
                    n_neighbors=config.mle_n_neighbors,
                )
            else:
                fitted = estimator.fit(values)
            estimate = float(fitted.dimension_)
    except (FloatingPointError, ValueError, ZeroDivisionError) as exc:
        warning = f"scikit-dimension {estimator_name} failed: {exc}"
        return {"metric_name": metric_name, "value": float("nan"), "status": "degenerate_input", "warning": warning, "n_samples": len(values), "split": "test"}

    warning_messages = [str(item.message) for item in captured]
    if not np.isfinite(estimate):
        warning_messages.append(
            f"scikit-dimension {estimator_name} returned a non-finite estimate"
        )
    warning = "; ".join(dict.fromkeys(warning_messages)) or None
    return {"metric_name": metric_name, "value": estimate if np.isfinite(estimate) else float("nan"), "status": "ok" if np.isfinite(estimate) else "degenerate_input", "warning": warning, "n_samples": len(values), "split": "test"}


def run_dimensionality_metrics_pipeline(
    inputs: CoreMetricInputs,
    selected_indices: np.ndarray,
    x_selected: np.ndarray,
    mu_selected: np.ndarray,
    config: DimensionalityMetricConfig,
) -> pd.DataFrame:
    """Evaluate library-backed intrinsic and linear spectral dimensionality.

    Input representation:
        Complete ``mu_test`` for effective rank; the shared, sample-aligned
        subsets of ``x_test`` and ``mu_test`` for intrinsic dimensionality.
        Stored ``z_test`` is used only for the explicitly named sampled-z
        sensitivity outputs and never replaces posterior means.

    TwoNN follows Facco et al., Scientific Reports 2017,
    DOI:10.1038/s41598-017-11873-y. MLE follows Levina and Bickel, NeurIPS
    2004. Both implementations are provided by scikit-dimension 0.3.4 and are
    applied to the same deterministic, sample-aligned test subset used by the
    geometry and topology pipelines. Effective rank is computationally cheap
    and intentionally uses the complete test split; this distinct sample basis
    is reported explicitly in the result artifacts.

    """
    spaces = {
        "x": np.asarray(x_selected, dtype=np.float64),
        "mu": np.asarray(mu_selected, dtype=np.float64),
    }
    if inputs.z_test is not None:
        indices = np.asarray(selected_indices, dtype=int)
        spaces["z"] = np.asarray(inputs.z_test, dtype=np.float64)[indices]

    values = compute_effective_rank_mu(inputs.mu_test)
    for representation, space in spaces.items():
        for estimator_name in ("twonn", "mle"):
            values.append(_estimate_intrinsic_dimension(
                space,
                representation=representation,
                estimator_name=estimator_name,
                config=config,
            ))
    return pd.DataFrame.from_records(values)


__all__ = ["compute_effective_rank_mu", "run_dimensionality_metrics_pipeline"]
