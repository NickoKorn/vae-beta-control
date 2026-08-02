"""Library-backed dimensionality diagnostics for aligned VARIX representations.

No intrinsic-dimension estimator or spectral decomposition is reimplemented in
this module. TwoNN and MLE delegate to scikit-dimension 0.3.4, and the centered
posterior-mean spectrum delegates to scikit-learn PCA with its full SVD solver.

Software reference:
    Bac, Mirkes, Gorban, Tyukin, and Zinovyev, Scikit-Dimension: A Python
    Package for Intrinsic Dimension Estimation, arXiv:2109.02596 (2021).

PCA reference:
    Jolliffe and Cadima, Principal component analysis: a review and recent
    developments, Philosophical Transactions A 374 (2016), 20150202.
"""

from __future__ import annotations

import warnings
from importlib.metadata import version

import numpy as np
import pandas as pd
import skdim
from sklearn.decomposition import PCA

from .contracts import (
    CoreMetricInputs,
    DimensionalityMetricConfig,
    MetricGroupResult,
    MetricValue,
    SharedMetricContext,
)


def _estimate_intrinsic_dimension(
    values: np.ndarray,
    *,
    representation: str,
    estimator_name: str,
    config: DimensionalityMetricConfig,
) -> MetricValue:
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
        return MetricValue(
            metric_name,
            float("nan"),
            status="degenerate_input",
            warning=warning,
            n_samples=len(values),
            split="test",
        )

    warning_messages = [str(item.message) for item in captured]
    if not np.isfinite(estimate):
        warning_messages.append(
            f"scikit-dimension {estimator_name} returned a non-finite estimate"
        )
    warning = "; ".join(dict.fromkeys(warning_messages)) or None
    return MetricValue(
        metric_name,
        estimate if np.isfinite(estimate) else float("nan"),
        status="ok" if np.isfinite(estimate) else "degenerate_input",
        warning=warning,
        n_samples=len(values),
        split="test",
    )


def _posterior_mean_spectrum(mu: np.ndarray) -> pd.DataFrame:
    """Fit scikit-learn's full PCA and return its library-provided spectrum.

    The table is descriptive rather than a new scalar metric. Its singular
    values and explained variances expose the linear concentration of centered
    ``mu`` without selecting a project-defined dimensionality threshold.
    """
    estimator = PCA(svd_solver="full").fit(mu)
    explained = np.asarray(estimator.explained_variance_ratio_, dtype=np.float64)
    table = pd.DataFrame(
        {
            "singular_direction": np.arange(len(estimator.singular_values_)),
            "singular_value": estimator.singular_values_,
            "explained_variance": estimator.explained_variance_,
            "explained_variance_ratio": explained,
            "cumulative_explained_variance_ratio": np.cumsum(explained),
            "scientific_source": (
                "Jolliffe and Cadima, Philosophical Transactions A 2016, "
                "DOI:10.1098/rsta.2015.0202"
            ),
            "implementation": "sklearn.decomposition.PCA(svd_solver='full')",
        }
    )
    return table


def run_dimensionality_metrics_pipeline(
    inputs: CoreMetricInputs,
    context: SharedMetricContext,
    config: DimensionalityMetricConfig,
) -> MetricGroupResult:
    """Evaluate library-backed intrinsic and linear spectral dimensionality.

    TwoNN follows Facco et al., Scientific Reports 2017,
    DOI:10.1038/s41598-017-11873-y. MLE follows Levina and Bickel, NeurIPS
    2004. Both implementations are provided by scikit-dimension 0.3.4 and are
    applied to the same deterministic, sample-aligned test subset used by the
    geometry and topology pipelines.

    The posterior-mean spectrum is taken directly from scikit-learn PCA fitted
    with ``svd_solver='full'``. It is retained as a table, not converted into a
    project-defined aggregate score.
    """
    spaces = {
        "x": np.asarray(context.x_test_selected, dtype=np.float64),
        "mu": np.asarray(context.mu_test_selected, dtype=np.float64),
    }
    if inputs.z_test is not None:
        indices = np.asarray(context.selected_test_indices, dtype=int)
        spaces["z"] = np.asarray(inputs.z_test, dtype=np.float64)[indices]

    values = [
        _estimate_intrinsic_dimension(
            space,
            representation=representation,
            estimator_name=estimator_name,
            config=config,
        )
        for representation, space in spaces.items()
        for estimator_name in ("twonn", "mle")
    ]
    spectrum = _posterior_mean_spectrum(spaces["mu"])
    metric_warnings = sorted(
        {value.warning for value in values if value.warning is not None}
    )
    return MetricGroupResult(
        group_name="representation_dimensionality",
        status=(
            "ok"
            if all(value.status == "ok" for value in values)
            else "degenerate_input"
        ),
        values=values,
        tables={"dimensionality_mu_spectrum": spectrum},
        artifacts={
            "scikit_dimension_version": version("scikit-dimension"),
            "scikit_learn_version": version("scikit-learn"),
            "scikit_learn_pca_solver": "full",
            "twonn_discard_fraction": config.twonn_discard_fraction,
            "mle_n_neighbors": config.mle_n_neighbors,
        },
        warnings=metric_warnings,
    )


__all__ = ["run_dimensionality_metrics_pipeline"]
