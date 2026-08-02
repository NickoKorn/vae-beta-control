"""Shared deterministic geometry and topology comparison context."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import pairwise_distances

from .contracts import (
    CoreMetricInputs,
    GeometryMetricConfig,
    SharedMetricContext,
    TopologyMetricConfig,
)


def _deterministic_indices(n: int, limit: int, random_state: int) -> np.ndarray:
    if n <= limit:
        return np.arange(n, dtype=int)
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(n, size=limit, replace=False))


def build_shared_metric_context(
    inputs: CoreMetricInputs,
    geometry_config: GeometryMetricConfig,
    topology_config: TopologyMetricConfig,
) -> SharedMetricContext:
    """Build one aligned subset and one pair of distance matrices for geometry/TDA."""
    effective_limit = min(
        geometry_config.max_samples,
        topology_config.max_samples,
    )
    if geometry_config.random_state != topology_config.random_state:
        raise ValueError("geometry and topology must use the same random_state")
    indices = _deterministic_indices(
        len(inputs.x_test), effective_limit, geometry_config.random_state
    )
    x_selected = inputs.x_test[indices]
    mu_selected = inputs.mu_test[indices]
    input_distances = pairwise_distances(
        x_selected, metric=geometry_config.distance_metric
    )
    latent_distances = pairwise_distances(
        mu_selected, metric=geometry_config.distance_metric
    )
    input_diameter = float(np.max(input_distances)) if len(indices) else None
    latent_diameter = float(np.max(latent_distances)) if len(indices) else None
    return SharedMetricContext(
        selected_test_indices=indices,
        x_test_selected=x_selected,
        mu_test_selected=mu_selected,
        input_distance_matrix=input_distances,
        latent_distance_matrix=latent_distances,
        input_diameter=input_diameter,
        latent_diameter=latent_diameter,
        sampling_metadata={
            "n_test_total": len(inputs.x_test),
            "n_selected": len(indices),
            "geometry_max_samples": int(geometry_config.max_samples),
            "topology_max_samples": int(topology_config.max_samples),
            "effective_shared_max_samples": int(effective_limit),
            "random_state": int(geometry_config.random_state),
            "selection": (
                "all"
                if len(indices) == len(inputs.x_test)
                else "sorted_random_without_replacement"
            ),
        },
    )
