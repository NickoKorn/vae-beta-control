"""Run metric families and concatenate their scalar DataFrames."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from .metrics.dimensionality import (
    DimensionalityMetricConfig,
    run_dimensionality_metrics_pipeline,
)
from .metrics.evaluation.downstream import TARGETS, run_downstream_metrics_pipeline
from .metrics.evaluation.downstream import DownstreamMetricConfig
from .metrics.evaluation.reconstruction import run_reconstruction_metrics_pipeline
from .metrics.evaluation.native_losses import run_native_loss_metrics_pipeline
from .metrics.preservation import GeometryMetricConfig, run_geometry_metrics_pipeline
from .metrics.information_theoretic import run_information_theoretic_metrics_pipeline
from .metrics.latent_usage import (
    LatentUsageMetricConfig,
    run_latent_usage_metrics_pipeline,
)
from .metrics.topology import TopologyMetricConfig, run_topology_metrics_pipeline
from .metrics.complementary.topology_annealing import SigmaTopologyConfig, run_sigma_unit_topology_pipeline
from .metrics.graph.neighbor_mixing_pipeline import NeighborMixingConfig, run_neighbor_mixing_pipeline
from beta_vae_experiments.metrics.graph.graph_metrics_pipeline import GraphMetricConfig, run_graph_metrics_pipeline

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
    sample_metadata_train: pd.DataFrame | Mapping[str, pd.DataFrame] | None = None

@dataclass(frozen=True)
class AbsoluteCoreMetricConfig:
    """Configuration passed to the metric-family runners."""

    output_dir: Path
    downstream: DownstreamMetricConfig = field(default_factory=DownstreamMetricConfig)
    graph: GraphMetricConfig = field(default_factory=GraphMetricConfig)
    latent_usage: LatentUsageMetricConfig = field(default_factory=LatentUsageMetricConfig)
    dimensionality: DimensionalityMetricConfig = field(
        default_factory=DimensionalityMetricConfig
    )
    geometry: GeometryMetricConfig = field(default_factory=GeometryMetricConfig)
    topology: TopologyMetricConfig = field(default_factory=TopologyMetricConfig)
    neighbor_mixing: NeighborMixingConfig = field(default_factory=NeighborMixingConfig)
    sigma_topology: SigmaTopologyConfig = field(default_factory=SigmaTopologyConfig)
    preservation_only: bool = False

def get_testsplit_numpy_rows(
    inputs: CoreMetricInputs,
    geometry: GeometryMetricConfig,
    topology: TopologyMetricConfig,
    sigma_topology: SigmaTopologyConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select the same reproducible test subset for geometry, graph and topology."""
    if geometry.random_state != topology.random_state:
        raise ValueError("geometry and topology must use the same random_state")
    if inputs.mu_test is None or len(inputs.x_test) != len(inputs.mu_test):
        raise ValueError("x_test and mu_test must contain the same aligned rows")
    if inputs.sample_ids_test is not None:
        sample_ids = np.asarray(inputs.sample_ids_test).astype(str)
        if len(np.unique(sample_ids)) != len(sample_ids):
            raise ValueError("sample_ids_test must be unique for structural metrics")
    limit = min(geometry.max_samples, topology.max_samples)
    if sigma_topology is not None and sigma_topology.enabled:
        limit = min(limit, sigma_topology.max_samples)
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
    selected_indices, x_selected, mu_selected = get_testsplit_numpy_rows(
        inputs, config.geometry, config.topology, config.sigma_topology
    )
    output_dir = Path(config.output_dir)

    frames = [
        run_sigma_unit_topology_pipeline(inputs, config.sigma_topology, output_dir,
                                         selected_indices=selected_indices),
        *([] if config.preservation_only else [run_graph_metrics_pipeline(
            inputs, selected_indices, x_selected, mu_selected,
            config.graph, config.geometry, output_dir,
        )]),
        run_native_loss_metrics_pipeline(inputs, varix),
        run_reconstruction_metrics_pipeline(inputs),
        run_latent_usage_metrics_pipeline(
            inputs, config.latent_usage, varix, output_dir
        ),
        run_information_theoretic_metrics_pipeline(
            inputs, varix, include_total_correlation=not config.preservation_only),
        run_neighbor_mixing_pipeline(inputs, config.neighbor_mixing, output_dir),
        *([] if config.preservation_only else [run_dimensionality_metrics_pipeline(
            inputs, selected_indices, x_selected, mu_selected,
            config.dimensionality,
            output_dir=output_dir,
        )]),
        run_geometry_metrics_pipeline(
            inputs, selected_indices, x_selected, mu_selected,
            config.geometry, output_dir,
        ),
        *([] if config.preservation_only else [run_topology_metrics_pipeline(
            inputs, selected_indices, x_selected, mu_selected,
            config.topology, output_dir
        )]),
        run_downstream_metrics_pipeline(
            varix,
            inputs.run_name,
            dataset_family=(
                str(inputs.metadata.get("dataset_family", "")) or None
            ),
            output_dir=output_dir,
            targets=tuple(inputs.metadata.get("downstream_targets", TARGETS)),
            config=config.downstream,
        ),
    ]
    metrics = pd.concat(frames, ignore_index=True)
    metrics.insert(0, "run_name", inputs.run_name)

    directory = output_dir / "metrics_artifacts" / inputs.run_name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "scalar_metrics.csv"
    metrics.to_csv(path, index=False)
    print(path)
    return metrics
