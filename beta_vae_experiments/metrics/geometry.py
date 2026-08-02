"""Geometry-preservation pipeline backed exclusively by official ZADU metrics.

The reference space is the sample-ID-aligned, preprocessed test input selected
in :mod:`shared_context`.  The two evaluated embeddings are the posterior mean
``mu_test`` and, when available, the persisted posterior draw ``z_test``.

Metric implementation is delegated to :func:`compute_zadu_geometry`, which
uses only ``ZADU(...).measure(...)`` from ZADU 0.4.2.  This orchestration layer
does not construct distances, neighbors, rankings, co-ranking matrices, or
metric formulas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import (
    CoreMetricInputs,
    GeometryMetricConfig,
    MetricGroupResult,
    MetricValue,
    SharedMetricContext,
)
from .geometry_zadu import ZADUGeometryResult, compute_zadu_geometry


def compute_effective_rank_mu(mu: np.ndarray) -> MetricValue:
    """Compute entropy effective rank of centered posterior means.

    Metric family:
        Latent geometry and effective dimensionality.
    Space comparison:
        Posterior-mean latent space only.
    Exact spaces:
        Column-centered empirical posterior means mu_test.
    Representations:
        Singular-value spectrum of centered mu_test.
    Mathematical definition:
        erank=exp(-sum_i p_i log(p_i)), p_i=sigma_i/sum_j sigma_j.
    Scientific source:
        Roy and Vetterli, The Effective Rank: A Measure of Effective
        Dimensionality, EUSIPCO 2007.
    Repository provenance:
        NumPy singular-value implementation of the cited definition after
        explicitly documented column centering.
    Interpretation:
        Continuous number of substantially represented variation directions.
    Limitations:
        Describes spectral dimensionality, not semantic usefulness,
        disentanglement, or input-geometry preservation.
    Direction:
        Descriptive; a sharp decline can indicate dimensional collapse.
    """
    values = np.asarray(mu, dtype=np.float64)
    centered = values - np.mean(values, axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    total = float(np.sum(singular_values))
    if total <= np.finfo(np.float64).eps:
        warning = "Centered posterior means have zero numerical rank."
        return MetricValue(
            "geometry_effective_rank_mu",
            0.0,
            "degenerate_input",
            warning,
            len(values),
            "test",
        )
    probabilities = singular_values[singular_values > 0] / total
    effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    return MetricValue(
        "geometry_effective_rank_mu",
        effective_rank,
        n_samples=len(values),
        split="test",
    )


def _metric_values(
    result: ZADUGeometryResult,
    *,
    representation: str,
    k: int,
    n_samples: int,
) -> list[MetricValue]:
    """Map one typed ZADU result to stable, representation-explicit columns."""
    prefix = f"geometry_{representation}"
    named_values = (
        (f"{prefix}_trustworthiness_zadu_k{k}", result.trustworthiness),
        (f"{prefix}_continuity_zadu_k{k}", result.continuity),
        (f"{prefix}_mrre_false_zadu_k{k}", result.mrre_false),
        (f"{prefix}_mrre_missing_zadu_k{k}", result.mrre_missing),
        (f"{prefix}_lcmc_zadu_k{k}", result.lcmc),
        (
            f"{prefix}_neighbor_dissimilarity_zadu_k{k}",
            result.neighbor_dissimilarity,
        ),
        (f"{prefix}_scale_normalized_stress_zadu", result.scale_normalized_stress),
        (f"{prefix}_non_metric_stress_zadu", result.non_metric_stress),
    )
    values = []
    for name, raw_value in named_values:
        is_finite = bool(np.isfinite(raw_value))
        values.append(
            MetricValue(
                name,
                float(raw_value) if is_finite else float("nan"),
                status="ok" if is_finite else "degenerate_input",
                warning=(
                    None
                    if is_finite
                    else (
                        "ZADU metric is undefined because the latent distance "
                        f"matrix for {representation} is degenerate."
                    )
                ),
                n_samples=n_samples,
                split="test",
            )
        )
    return values


def _table_row(
    result: ZADUGeometryResult,
    *,
    run_name: str,
    representation: str,
    k: int,
    n_samples: int,
) -> dict[str, object]:
    """Create a readable long-form row without collapsing metric families."""
    raw_values = (
        result.trustworthiness,
        result.continuity,
        result.mrre_false,
        result.mrre_missing,
        result.lcmc,
        result.neighbor_dissimilarity,
        result.scale_normalized_stress,
        result.non_metric_stress,
    )
    is_degenerate = not all(np.isfinite(value) for value in raw_values)
    return {
        "run_name": run_name,
        "split": "test",
        "input_representation": "preprocessed_model_input_x_test",
        "latent_representation": representation,
        "n_samples": n_samples,
        "k": k,
        "trustworthiness": result.trustworthiness,
        "continuity": result.continuity,
        "mrre_false": result.mrre_false,
        "mrre_missing": result.mrre_missing,
        "lcmc": result.lcmc,
        "neighbor_dissimilarity": result.neighbor_dissimilarity,
        "scale_normalized_stress": result.scale_normalized_stress,
        "non_metric_stress": result.non_metric_stress,
        "status": "degenerate_input" if is_degenerate else "ok",
        "warning": (
            "At least one ZADU metric is undefined for the degenerate latent space."
            if is_degenerate
            else None
        ),
        "implementation": "zadu.ZADU 0.4.2 public class API",
    }


def run_geometry_metrics_pipeline(
    inputs: CoreMetricInputs,
    context: SharedMetricContext,
    config: GeometryMetricConfig,
) -> MetricGroupResult:
    """Evaluate ZADU's label-free geometry core for ``mu`` and stored ``z``.

    Args:
        inputs:
            Validated run data.  ``z_test`` is optional; ``mu_test`` is always
            required by the absolute-core contract.
        context:
            Shared deterministic test subset.  Its ``selected_test_indices``
            align the input, posterior mean, and realized latent draw by the
            same sample rows.
        config:
            Geometry configuration. ``selected_k`` is used for the ZADU
            neighborhood measures. ``distance_metric`` is currently required
            to be ``"euclidean"`` because that is the distance used internally
            by the selected official ZADU measures.

    Returns:
        A successful ``MetricGroupResult`` with eight independent scalars for
        ``mu`` and, if present, eight for ``z``. ``geometry_summary`` contains
        one row per representation; despite the historical filename, these are
        selected-k summaries rather than project-reimplemented k-curves.

    Scientific interpretation:
        ``mu`` measures deterministic encoder geometry.  Comparing ``z`` with
        ``mu`` reveals additional degradation from posterior sampling.  High
        TNC/MRRE/LCMC and low ND/stress are favorable, but no metric establishes
        biological validity and none should be optimized against the held-out
        test split during model selection.
    """
    x = np.asarray(context.x_test_selected, dtype=np.float64)
    mu = np.asarray(context.mu_test_selected, dtype=np.float64)
    n_samples = x.shape[0]
    if n_samples < 3:
        return MetricGroupResult(
            "geometry",
            "degenerate_input",
            [],
            warnings=["ZADU geometry requires at least three selected samples."],
        )
    selected_k = min(config.selected_k, (n_samples - 1) // 2)

    mu_result = compute_zadu_geometry(x, mu, selected_k)
    values = [compute_effective_rank_mu(inputs.mu_test)] + _metric_values(
        mu_result,
        representation="mu",
        k=selected_k,
        n_samples=n_samples,
    )
    rows = [
        _table_row(
            mu_result,
            run_name=inputs.run_name,
            representation="posterior_mean_mu_test",
            k=selected_k,
            n_samples=n_samples,
        )
    ]

    if inputs.z_test is not None:
        z_all = np.asarray(inputs.z_test, dtype=np.float64)
        z = z_all[np.asarray(context.selected_test_indices, dtype=int)]
        z_result = compute_zadu_geometry(x, z, selected_k)
        values.extend(
            _metric_values(
                z_result,
                representation="z",
                k=selected_k,
                n_samples=n_samples,
            )
        )
        rows.append(
            _table_row(
                z_result,
                run_name=inputs.run_name,
                representation="persisted_sampled_latent_z_test",
                k=selected_k,
                n_samples=n_samples,
            )
        )

    group_status = (
        "degenerate_input" if any(value.status != "ok" for value in values) else "ok"
    )
    metric_warnings = sorted(
        {value.warning for value in values if value.warning is not None}
    )
    return MetricGroupResult(
        group_name="geometry",
        status=group_status,
        values=values,
        tables={"geometry_summary": pd.DataFrame(rows)},
        artifacts={
            "implementation": "zadu.ZADU",
            "zadu_version": "0.4.2",
            "measure_families": (
                "TNC",
                "MRRE",
                "LCMC",
                "ND",
                "SN_STRESS",
                "NM_STRESS",
            ),
            "selected_test_indices": np.asarray(
                context.selected_test_indices, dtype=int
            ),
            "sampling_metadata": dict(context.sampling_metadata),
        },
        warnings=metric_warnings,
    )


__all__ = ["run_geometry_metrics_pipeline"]
