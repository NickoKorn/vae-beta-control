"""Library-backed dimensionality diagnostics for aligned VARIX representations.

No intrinsic-dimension estimator is reimplemented in this module. TwoNN and
MLE delegate to scikit-dimension 0.3.4.

Software reference:
    Bac, Mirkes, Gorban, Tyukin, and Zinovyev, Scikit-Dimension: A Python
    Package for Intrinsic Dimension Estimation, arXiv:2109.02596 (2021).

Quellen: https://zenodo.org/records/40328 (Roy & Vetterli, Singularwertentropie),
https://arxiv.org/abs/1803.06992 (Facco et al., TwoNN),
https://papers.nips.cc/paper/2577-maximum-likelihood-estimation-of-intrinsic-dimension
(Levina & Bickel, MLE-Grundlage). Die konkrete MLE-Variante folgt skdim 0.3.4;
deren Haro-Parameter sind keine zusätzlich geschätzte Posterior-Noise-Korrektur:
https://scikit-dimension.readthedocs.io/en/latest/skdim.id.MLE.html

"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
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
        Sampled z can remove exact ties, but measures a different geometry.
        Its ID is only a sensitivity output; μ remains the primary point cloud.
        Posterior variance is not generally constrained to be at least one.
        Deliberately not repaired here (no deduplication, no jitter): the point
        mass is an architecture artifact, and a repaired value would hide it.
    """

    twonn_discard_fraction: float = 0.1
    mle_n_neighbors: int = 20

    def __post_init__(self) -> None:
        if (isinstance(self.twonn_discard_fraction, bool)
                or not np.isfinite(self.twonn_discard_fraction)
                or not 0 <= self.twonn_discard_fraction < 1):
            raise ValueError("TwoNN discard_fraction must lie in [0, 1)")
        if type(self.mle_n_neighbors) is not int or self.mle_n_neighbors < 3:
            raise ValueError("MLE n_neighbors must be an integer of at least three")


def _validate_points(values):
    points = np.asarray(values)
    if (points.ndim != 2 or min(points.shape) == 0
            or not np.issubdtype(points.dtype, np.number) or np.iscomplexobj(points)
            or not np.isfinite(points).all()):
        raise ValueError('Expected a nonempty finite real point matrix')
    return np.asarray(points, dtype=np.float64)

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
    values = _validate_points(mu)
    # Global rescaling changes no singular-value proportions. It prevents the
    # previous absolute eps threshold from classifying tiny nonzero clouds as 0.
    scale = np.max(np.abs(values))
    normalized = values / scale if scale > 0 else values
    centered = normalized - np.mean(normalized, axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    total = float(np.sum(singular_values))
    if np.all(values == values[0]) or total == 0:
        warning = "Constant posterior means: effective rank is set to 0 by project convention."
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
        estimator = skdim.id.TwoNN(discard_fraction=config.twonn_discard_fraction, dist=False)
    elif estimator_name == "mle":
        estimator = skdim.id.MLE(dnoise=None, sigma=0, n=None,
            integral_approximation='Haro', unbiased=False, neighborhood_based=True, K=5)
    else:
        raise ValueError(
            f"Unsupported intrinsic-dimension estimator {estimator_name!r}"
        )

    metric_name = f"dimensionality_{representation}_{estimator_name}"
    captured: list[warnings.WarningMessage]
    try:
        values = _validate_points(values)
        if len(np.unique(values, axis=0)) != len(values):
            raise ValueError('Duplicate points: nearest-neighbor ID is undefined; no jitter/deduplication')
        if values.shape[1] < 2:
            raise ValueError('scikit-dimension 0.3.4 requires at least two input features')
        if len(values) < 3:
            raise ValueError('Two nearest neighbors require at least three samples')
        if estimator_name == 'twonn' and int(len(values) * (1-config.twonn_discard_fraction)) < 2:
            raise ValueError('Too few points retained by TwoNN discard_fraction')
        if estimator_name == 'mle' and config.mle_n_neighbors >= len(values):
            raise ValueError('MLE requires n_neighbors < n_samples; no automatic clipping')
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            if estimator_name == "mle":
                fitted = estimator.fit(
                    values,
                    n_neighbors=config.mle_n_neighbors,
                    comb='mle', n_jobs=1,
                )
            else:
                fitted = estimator.fit(values)
            estimate = float(fitted.dimension_)
    except (FloatingPointError, ValueError, ZeroDivisionError) as exc:
        warning = f"scikit-dimension {estimator_name} failed: {exc}"
        return {"metric_name": metric_name, "value": float("nan"), "status": "degenerate_input", "warning": warning, "n_samples": len(values), "split": "test"}

    warning_messages = [str(item.message) for item in captured]
    if not np.isfinite(estimate) or estimate <= 0:
        warning_messages.append(
            f"scikit-dimension {estimator_name} returned a nonpositive/nonfinite estimate"
        )
    warning = "; ".join(dict.fromkeys(warning_messages)) or None
    valid = np.isfinite(estimate) and estimate > 0
    return {"metric_name": metric_name, "value": estimate if valid else float("nan"), "status": "ok" if valid else "degenerate_input", "warning": warning, "n_samples": len(values), "split": "test"}


def run_dimensionality_metrics_pipeline(
    inputs: CoreMetricInputs,
    selected_indices: np.ndarray,
    x_selected: np.ndarray,
    mu_selected: np.ndarray,
    config: DimensionalityMetricConfig,
    output_dir: Path | None = None,
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
    indices = np.asarray(selected_indices)
    if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
            or len(indices) != len(np.unique(indices)) or (indices < 0).any()
            or (indices >= len(inputs.x_test)).any()):
        raise ValueError('Selected indices must be unique valid test indices')
    if not np.array_equal(np.asarray(inputs.x_test)[indices], x_selected, equal_nan=True):
        raise ValueError('X subset does not match selected test indices')
    if not np.array_equal(np.asarray(inputs.mu_test)[indices], mu_selected, equal_nan=True):
        raise ValueError('μ subset does not match selected test indices')
    spaces = {
        "x": np.asarray(x_selected, dtype=np.float64),
        "mu": np.asarray(mu_selected, dtype=np.float64),
    }
    if inputs.z_test is not None:
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
    frame = pd.DataFrame.from_records(values)
    provenance = dict(config=asdict(config), skdim_version=version('scikit-dimension'),
        numpy_version=version('numpy'), sklearn_version=version('scikit-learn'),
        selected_indices=indices.tolist(), effective_rank_n=len(inputs.mu_test),
        intrinsic_dimension_n=len(indices), effective_rank_normalization='latent dimension d',
        mle_parameters=dict(dnoise=None, sigma=0, n=None, integral_approximation='Haro',
                            unbiased=False, neighborhood_based=True, K=5, comb='mle', n_jobs=1),
        twonn_parameters=dict(discard_fraction=config.twonn_discard_fraction, dist=False),
        effective_rank_spectrum='singular values of centered mu, not covariance eigenvalues',
        zero_rank_convention=0, representation_roles=dict(mu='primary', z='single stored draw sensitivity'),
        spaces={name: dict(n_samples=len(space), n_features=space.shape[1],
                           n_unique_rows=len(np.unique(space, axis=0))) for name, space in spaces.items()})
    frame.attrs['dimensionality_metadata'] = provenance
    if output_dir is not None:
        directory = Path(output_dir) / 'absolute_core_metrics_v1' / 'dimensionality' / inputs.run_name
        directory.mkdir(parents=True, exist_ok=True)
        frame.to_csv(directory / 'scalar_metrics.csv', index=False)
        (directory / 'provenance.json').write_text(json.dumps(provenance, indent=2, allow_nan=False)+'\n')
    return frame


__all__ = ["compute_effective_rank_mu", "run_dimensionality_metrics_pipeline"]
