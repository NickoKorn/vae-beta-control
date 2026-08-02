"""Top-level absolute core-metrics orchestration."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from .contracts import (
    AbsoluteCoreMetricConfig,
    AbsoluteCoreMetricResults,
    CoreMetricInputs,
    MetricGroupResult,
    MetricValue,
)
from .dimensionality import run_dimensionality_metrics_pipeline
from .geometry import run_geometry_metrics_pipeline
from .information_theoretic import run_information_theoretic_metrics_pipeline
from .io import save_absolute_core_metric_results
from .probabilistic import run_probabilistic_metrics_pipeline
from .reconstruction import run_reconstruction_metrics_pipeline
from .registry import get_metric_definition, metric_catalog_frame
from .shared_context import build_shared_metric_context
from .topology import run_topology_metrics_pipeline
from .validation import validate_metric_inputs


def _merge_per_dimension(
    run_name: str,
    latent_dimension: int,
    groups: Iterable[MetricGroupResult],
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {"run_name": run_name, "latent_dimension": np.arange(latent_dimension)}
    )
    table_names = (
        "conditional_kl_per_dimension",
        "posterior_entropy_per_dimension",
        "active_units_per_dimension",
    )
    all_tables = {key: value for group in groups for key, value in group.tables.items()}
    for name in table_names:
        if name in all_tables:
            frame = frame.merge(
                all_tables[name], on=["run_name", "latent_dimension"], how="left"
            )
    required = [
        "run_name",
        "latent_dimension",
        "conditional_kl_mean",
        "conditional_kl_median",
        "conditional_kl_p95",
        "posterior_entropy_mean",
        "posterior_variance_mean",
        "posterior_standard_deviation_mean",
        "posterior_active_units_variance",
        "is_active_eps_0p001",
        "is_active_eps_0p01",
        "is_active_eps_0p1",
    ]
    for column in required:
        if column not in frame:
            frame[column] = np.nan
    return frame[required]


def _long_row(run_name: str, value: MetricValue) -> dict[str, object]:
    definition = get_metric_definition(value.name)
    return {
        "run_name": run_name,
        "metric_name": value.name,
        "value": value.value,
        "group": definition.group,
        "space_comparison": definition.space_comparison.value,
        "exact_spaces": " | ".join(definition.exact_spaces),
        "representations": " | ".join(
            item.value for item in definition.representations
        ),
        "scientific_source": definition.scientific_source,
        "repository_provenance": definition.repository_provenance,
        "estimator_name": definition.estimator_name,
        "unit": definition.unit,
        "higher_is_better": definition.higher_is_better,
        "interpretation": definition.interpretation,
        "limitations": definition.limitations,
        "status": value.status,
        "warning": value.warning,
        "n_samples": value.n_samples,
        "split": value.split,
    }


def assemble_absolute_core_metric_results(
    inputs: CoreMetricInputs,
    groups: list[MetricGroupResult],
    context_metadata: dict[str, object],
) -> AbsoluteCoreMetricResults:
    values = [value for group in groups for value in group.values]
    wide: dict[str, object] = {
        "run_name": inputs.run_name,
        "n_train_samples": len(inputs.x_train),
        "n_val_samples": len(inputs.x_val),
        "n_test_samples": len(inputs.x_test),
        "input_dimension": inputs.x_test.shape[1],
        "latent_dimension": inputs.mu_test.shape[1],
    }
    wide.update({value.name: value.value for value in values})
    for group in groups:
        wide[f"{group.group_name}_status"] = group.status
        for key, item in group.artifacts.items():
            if isinstance(item, (str, int, float, bool, np.generic)):
                wide[key] = item.item() if isinstance(item, np.generic) else item
    long = pd.DataFrame([_long_row(inputs.run_name, value) for value in values])
    long = long.sort_values(["group", "metric_name", "run_name"]).reset_index(drop=True)
    per_dimension = _merge_per_dimension(
        inputs.run_name, inputs.mu_test.shape[1], groups
    )
    tables = {key: value for group in groups for key, value in group.tables.items()}
    arrays = {
        key: value
        for group in groups
        for key, value in group.artifacts.items()
        if isinstance(value, np.ndarray)
    }
    warnings = [warning for group in groups for warning in group.warnings]
    return AbsoluteCoreMetricResults(
        run_name=inputs.run_name,
        scalar_metrics_wide=pd.DataFrame([wide]).reindex(
            columns=["run_name"] + sorted(key for key in wide if key != "run_name")
        ),
        scalar_metrics_long=long,
        per_dimension_metrics=per_dimension,
        metric_catalog=metric_catalog_frame(),
        artifacts={"tables": tables, "arrays": arrays},
        warnings=warnings,
        metadata={
            "run_name": inputs.run_name,
            "input_contract_metadata": dict(inputs.metadata),
            "shared_metric_context": context_metadata,
            "metric_groups": {
                group.group_name: {"status": group.status, "warnings": group.warnings}
                for group in groups
            },
        },
    )


def run_absolute_core_metrics_pipeline(
    inputs: CoreMetricInputs,
    config: AbsoluteCoreMetricConfig,
    varix: Any,
) -> AbsoluteCoreMetricResults:
    """Run all scientifically sourced absolute core metrics.

    Metric family:
        Composite orchestration pipeline.
    Space comparison:
        Mixed; delegates to explicitly scoped group pipelines.
    Exact spaces:
        Input, reconstruction, posterior, prior, and posterior-mean latent spaces.
    Representations:
        X, X_hat, mu, logvar, q(z|x), q(z), and p(z).
    Mathematical definition:
        No direct metric is calculated; six scientifically defined group pipelines are orchestrated.
    Scientific source:
        Sources are documented by every delegated leaf metric.
    Repository provenance:
        Project orchestration code.
    Interpretation:
        Unified result without merging distinct metrics into an aggregate score.
    Limitations:
        No single representation-quality score is created.
    Direction:
        Descriptive only.
    """
    validated = validate_metric_inputs(inputs)
    context = build_shared_metric_context(validated, config.geometry, config.topology)
    reconstruction = run_reconstruction_metrics_pipeline(
        validated, config.reconstruction
    )
    probabilistic = run_probabilistic_metrics_pipeline(
        validated, config.probabilistic, varix
    )
    information = run_information_theoretic_metrics_pipeline(
        validated, config.information, varix
    )
    dimensionality = run_dimensionality_metrics_pipeline(
        validated, context, config.dimensionality
    )
    geometry = run_geometry_metrics_pipeline(validated, context, config.geometry)
    topology = run_topology_metrics_pipeline(validated, context, config.topology)
    results = assemble_absolute_core_metric_results(
        validated,
        [
            reconstruction,
            probabilistic,
            information,
            dimensionality,
            geometry,
            topology,
        ],
        context.sampling_metadata,
    )
    save_absolute_core_metric_results(results, config)
    return results
