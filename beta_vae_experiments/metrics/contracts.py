"""Typed contracts and validated configuration for absolute core metrics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class SpaceComparison(str, Enum):
    INPUT_VS_RECONSTRUCTION = "input_vs_reconstruction"
    CROSS_SPLIT_RECONSTRUCTION = "cross_split_reconstruction"
    POSTERIOR_VS_PRIOR = "posterior_vs_prior"
    POSTERIOR_ONLY = "posterior_only"
    LATENT_ONLY = "latent_only"
    INPUT_VS_LATENT = "input_vs_latent"
    AGGREGATED_POSTERIOR_ONLY = "aggregated_posterior_only"
    POSTERIOR_MARGINALS_VS_PRIOR = "posterior_marginals_vs_prior"
    INPUT_ONLY = "input_only"
    LATENT_TOPOLOGY_ONLY = "latent_topology_only"
    INPUT_TOPOLOGY_VS_LATENT_TOPOLOGY = "input_topology_vs_latent_topology"


class Representation(str, Enum):
    INPUT_X = "x"
    RECONSTRUCTION_X_HAT = "x_hat"
    POSTERIOR_MEAN_MU = "mu"
    SAMPLED_LATENT_Z = "z"
    POSTERIOR_LOGVAR = "logvar"
    CONDITIONAL_POSTERIOR = "q_z_given_x"
    AGGREGATED_POSTERIOR = "q_z"
    AGGREGATED_POSTERIOR_MARGINALS = "q_zj"
    PRIOR = "p_z"
    INPUT_DISTANCE_MATRIX = "input_distance_matrix"
    LATENT_DISTANCE_MATRIX = "latent_distance_matrix"
    INPUT_PERSISTENCE_DIAGRAM = "input_persistence_diagram"
    LATENT_PERSISTENCE_DIAGRAM = "latent_persistence_diagram"


@dataclass(frozen=True)
class CoreMetricInputs:
    run_name: str
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    x_recon_train: np.ndarray
    x_recon_val: np.ndarray
    x_recon_test: np.ndarray
    mu_train: np.ndarray
    mu_val: np.ndarray
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


@dataclass(frozen=True)
class PosteriorCollapseConfig:
    delta: float = 0.01
    epsilon_values: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
    primary_epsilon: float = 1e-2

    def __post_init__(self) -> None:
        # Contstraints so that Posterior Collapse detection have right numbers according to paper
        if not 0 <= self.delta < 1:
            raise ValueError("collapse delta must satisfy 0 <= delta < 1")
        if not self.epsilon_values or any(value <= 0 for value in self.epsilon_values):
            raise ValueError("collapse epsilon values must be positive")
        if self.primary_epsilon not in self.epsilon_values:
            raise ValueError("primary_epsilon must occur in epsilon_values")


@dataclass(frozen=True)
class ReconstructionMetricConfig:
    # For val_mse / max(train_mse, epsilon), if train_mse == 0.
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.epsilon <= 0:
            raise ValueError("reconstruction epsilon must be positive")


@dataclass(frozen=True)
class ProbabilisticMetricConfig:
    collapse: PosteriorCollapseConfig = field(default_factory=PosteriorCollapseConfig)


@dataclass(frozen=True)
class InformationMetricConfig:
    # Computational cap, not a value prescribed by the beta-TCVAE paper.
    # Example: n_test=6200 and max_samples=5000 selects 5000 observations;
    # n_test=900 and max_samples=5000 uses all 900 observations.
    max_samples: int = 5000

    # Memory cap for the query axis of the finite-mixture calculation.
    # Example with max_samples=5000, chunk_size=256, and latent_dim=8:
    # ceil(5000 / 256) = 20 chunks. The largest temporary density tensor has
    # 256 * 5000 * 8 = 10,240,000 float64 entries
    chunk_size: int = 256

    # Reproducibility seed, not a scientifically value. It controls
    # the 5000-of-6200 subsample above and posterior resampling if stored z is
    # unavailable. Identical inputs and seed reproduce the same selection.
    random_state: int = 31

    def __post_init__(self) -> None:
        if self.max_samples < 2:
            raise ValueError("max_samples must be at least 2")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")


@dataclass(frozen=True)
class DimensionalityMetricConfig:
    twonn_discard_fraction: float = 0.1
    mle_n_neighbors: int = 20

    def __post_init__(self) -> None:
        if not 0 <= self.twonn_discard_fraction < 1:
            raise ValueError("TwoNN discard_fraction must lie in [0, 1)")
        if self.mle_n_neighbors < 2:
            raise ValueError("MLE n_neighbors must be at least two")


@dataclass(frozen=True)
class GeometryMetricConfig:
    distance_metric: str = "euclidean"
    selected_k: int = 10

    # Computational cap, not a paper-prescribed value. Important: geometry and
    # topology currently share one aligned subset, whose effective size is
    # min(n_test, geometry.max_samples, topology.max_samples). With the default
    # values 3000 and 1000, both pipelines therefore use at most 1000 samples.
    max_samples: int = 3000

    # Reproducibility seed for deterministic subsampling without replacement.
    # The value 31 has no scientific significance.
    random_state: int = 31

    def __post_init__(self) -> None:
        if self.distance_metric != "euclidean":
            raise ValueError("geometry distance_metric must be 'euclidean'")
        if self.selected_k < 1:
            raise ValueError("geometry selected_k must be positive")
        if self.max_samples < 3:
            raise ValueError("geometry max_samples must be at least three")


@dataclass(frozen=True)
class TopologyMetricConfig:
    max_samples: int = 1000
    betti_grid_points: int = 256
    random_state: int = 31

    def __post_init__(self) -> None:
        if self.max_samples < 2:
            raise ValueError("topology max_samples must be at least two")
        if self.betti_grid_points < 2:
            raise ValueError("topology betti_grid_points must be at least two")


@dataclass(frozen=True)
class AbsoluteCoreMetricConfig:
    output_dir: Path
    reconstruction: ReconstructionMetricConfig = field(
        default_factory=ReconstructionMetricConfig
    )
    probabilistic: ProbabilisticMetricConfig = field(
        default_factory=ProbabilisticMetricConfig
    )
    information: InformationMetricConfig = field(
        default_factory=InformationMetricConfig
    )
    dimensionality: DimensionalityMetricConfig = field(
        default_factory=DimensionalityMetricConfig
    )
    geometry: GeometryMetricConfig = field(default_factory=GeometryMetricConfig)
    topology: TopologyMetricConfig = field(default_factory=TopologyMetricConfig)
    replace_existing_run: bool = True


@dataclass
class MetricValue:
    name: str
    value: float | int | str | bool
    status: str = "ok"
    warning: str | None = None
    n_samples: int | None = None
    split: str | None = None


@dataclass
class MetricResult:
    values: list[MetricValue]
    vectors: dict[str, np.ndarray] = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class MetricGroupResult:
    group_name: str
    status: str
    values: list[MetricValue]
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class SharedMetricContext:
    selected_test_indices: np.ndarray
    x_test_selected: np.ndarray
    mu_test_selected: np.ndarray
    input_distance_matrix: np.ndarray | None
    latent_distance_matrix: np.ndarray | None
    input_diameter: float | None
    latent_diameter: float | None
    sampling_metadata: dict[str, Any]


@dataclass
class AbsoluteCoreMetricResults:
    run_name: str
    scalar_metrics_wide: pd.DataFrame
    scalar_metrics_long: pd.DataFrame
    per_dimension_metrics: pd.DataFrame
    metric_catalog: pd.DataFrame
    artifacts: dict[str, Any]
    warnings: list[str]
    metadata: dict[str, Any]
