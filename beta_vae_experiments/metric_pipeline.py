"""Run metric families and concatenate their scalar DataFrames."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import numpy as np

import pandas as pd


@dataclass(frozen=True)
class CoreMetricInputs:
    """Aligned arrays and identifiers produced by one completed VARIX run."""

    run_name: str
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    x_recon_train: np.ndarray
    x_recon_val: np.ndarray
    x_recon_test: np.ndarray
    mu_train: np.ndarray
    mu_val: np.ndarray | None
    mu_test: np.ndarray
    logvar_train: np.ndarray | None = None
    logvar_val: np.ndarray | None = None
    logvar_test: np.ndarray | None = None
    z_train: np.ndarray | None = None
    z_val: np.ndarray | None = None
    z_test: np.ndarray | None = None
    sample_ids_train: np.ndarray | None = None
    sample_ids_val: np.ndarray | None = None
    sample_ids_test: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    sample_metadata_test: pd.DataFrame | Mapping[str, pd.DataFrame] | None = None

from .metrics.dimensionality import (
    DimensionalityMetricConfig,
    run_dimensionality_metrics_pipeline,
)
from .metrics.evaluation.downstream import TARGETS, run_downstream_metrics_pipeline
from .metrics.evaluation.reconstruction import run_reconstruction_metrics_pipeline
from .metrics.evaluation.native_losses import run_native_loss_metrics_pipeline
from .metrics.geometry import GeometryMetricConfig, run_geometry_metrics_pipeline
from .metrics.curvature.encoder_geometry import EncoderGeometryConfig, run_encoder_geometry_metrics_pipeline
from .metrics.information_theoretic import run_information_theoretic_metrics_pipeline
from .metrics.latent_usage import (
    LatentUsageMetricConfig,
    run_latent_usage_metrics_pipeline,
)
from .metrics.topology import TopologyMetricConfig, run_topology_metrics_pipeline
from beta_vae_experiments.metrics.topology.local_topology import LocalTopologyMetricConfig, run_local_topology_metrics_pipeline
from beta_vae_experiments.metrics.graph.graph_metrics_pipeline import GraphMetricConfig, run_graph_metrics_pipeline
@dataclass(frozen=True)
class AbsoluteCoreMetricConfig:
    """Configuration passed to the metric-family runners."""

    output_dir: Path
    graph: GraphMetricConfig = field(default_factory=GraphMetricConfig)
    latent_usage: LatentUsageMetricConfig = field(default_factory=LatentUsageMetricConfig)
    dimensionality: DimensionalityMetricConfig = field(
        default_factory=DimensionalityMetricConfig
    )
    geometry: GeometryMetricConfig = field(default_factory=GeometryMetricConfig)
    encoder_geometry: EncoderGeometryConfig = field(default_factory=EncoderGeometryConfig)
    topology: TopologyMetricConfig = field(default_factory=TopologyMetricConfig)
    local_topology: LocalTopologyMetricConfig = field(default_factory=LocalTopologyMetricConfig)


def validate_metric_inputs(inputs: CoreMetricInputs) -> CoreMetricInputs:
    """Normalize metric arrays and verify split-wise row alignment."""
    array_fields = (
        "x_train", "x_val", "x_test",
        "x_recon_train", "x_recon_val", "x_recon_test",
        "mu_train", "mu_val", "mu_test",
        "logvar_train", "logvar_val", "logvar_test",
        "z_train", "z_val", "z_test",
        "sample_ids_train", "sample_ids_val", "sample_ids_test",
    )
    normalized: dict[str, np.ndarray | None] = {}
    for name in array_fields:
        value = getattr(inputs, name)
        if value is None or (name == "mu_val" and np.asarray(value).size == 0):
            normalized[name] = None
        else:
            normalized[name] = np.asarray(value)

    for split in ("train", "val", "test"):
        x = normalized[f"x_{split}"]
        reconstruction = normalized[f"x_recon_{split}"]
        mu = normalized[f"mu_{split}"]
        required = {"x": x, "reconstruction": reconstruction}
        if split != "val" or mu is not None:
            required["mu"] = mu
        for label, values in required.items():
            if values is None or values.ndim != 2:
                raise ValueError(f"{label}_{split} must be a two-dimensional array")
        expected_rows = len(x)
        for label, values in (
            ("reconstruction", reconstruction), ("mu", mu),
            ("logvar", normalized[f"logvar_{split}"]),
            ("z", normalized[f"z_{split}"]),
            ("sample_ids", normalized[f"sample_ids_{split}"]),
        ):
            if values is not None and len(values) != expected_rows:
                raise ValueError(
                    f"{label}_{split} and x_{split} must contain the same samples"
                )
        logvar = normalized[f"logvar_{split}"]
        if logvar is not None and mu is not None and logvar.shape != mu.shape:
            raise ValueError(f"mu_{split} and logvar_{split} must have equal shapes")

    return replace(inputs, **normalized)


def _select_test_rows(
    inputs: CoreMetricInputs,
    geometry: GeometryMetricConfig,
    topology: TopologyMetricConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select one deterministic aligned subset for structural metrics."""
    if geometry.random_state != topology.random_state:
        raise ValueError("geometry and topology must use the same random_state")
    limit = min(geometry.max_samples, topology.max_samples)
    if len(inputs.x_test) <= limit:
        indices = np.arange(len(inputs.x_test), dtype=int)
    else:
        rng = np.random.default_rng(geometry.random_state)
        indices = np.sort(rng.choice(len(inputs.x_test), limit, replace=False))
    return indices, inputs.x_test[indices], inputs.mu_test[indices]

def run_absolute_core_metrics_pipeline(
    inputs: CoreMetricInputs,
    config: AbsoluteCoreMetricConfig,
    varix: Any,
) -> pd.DataFrame:
    """Run every metric family, save details locally, and return scalar metrics."""
    validated = validate_metric_inputs(inputs)
    selected_indices, x_selected, mu_selected = _select_test_rows(
        validated, config.geometry, config.topology
    )
    output_dir = Path(config.output_dir)

    encoder_model = None
    if config.encoder_geometry.enabled:
        trainer = getattr(varix, '_trainer', None)
        if trainer is not None:
            encoder_model = trainer.get_model()
    frames = [
        run_encoder_geometry_metrics_pipeline(
            validated, selected_indices, x_selected, mu_selected,
            config.encoder_geometry, output_dir, encoder_model),
        run_graph_metrics_pipeline(
            validated, selected_indices, x_selected, mu_selected,
            config.graph, config.geometry, output_dir,
        ),
        run_native_loss_metrics_pipeline(validated, varix),
        run_reconstruction_metrics_pipeline(validated),
        run_latent_usage_metrics_pipeline(
            validated, config.latent_usage, varix, output_dir
        ),
        run_information_theoretic_metrics_pipeline(validated, varix),
        run_dimensionality_metrics_pipeline(
            validated, selected_indices, x_selected, mu_selected,
            config.dimensionality,
        ),
        run_geometry_metrics_pipeline(
            validated, selected_indices, x_selected, mu_selected,
            config.geometry, output_dir,
        ),
        run_topology_metrics_pipeline(
            validated, selected_indices, x_selected, mu_selected,
            config.topology, output_dir
        ),
        run_local_topology_metrics_pipeline(
            validated, selected_indices, x_selected, mu_selected,
            config.local_topology, output_dir,
        ),
        run_downstream_metrics_pipeline(
            varix,
            validated.run_name,
            dataset_family=(
                str(validated.metadata.get("dataset_family", "")) or None
            ),
            output_dir=output_dir,
            targets=tuple(validated.metadata.get("downstream_targets", TARGETS)),
        ),
    ]
    metrics = pd.concat(frames, ignore_index=True)
    metrics.insert(0, "run_name", validated.run_name)

    directory = output_dir / "metrics_artifacts" / validated.run_name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "scalar_metrics.csv"
    metrics.to_csv(path, index=False)
    print(path)
    return metrics


__all__ = [
    "AbsoluteCoreMetricConfig",
    "CoreMetricInputs",
    "validate_metric_inputs",
    "run_absolute_core_metrics_pipeline",
]
