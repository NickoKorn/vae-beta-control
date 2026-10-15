"""Focused structural comparison of aligned input data ``X`` and posterior means ``mu``.

The scalar core exposes normalized H0/H1 Wasserstein/Bottleneck distances,
persistence entropy, and Landscape distances. SRTD/NTS is archived outside the active pipeline. Shared-grid Betti curves are saved as
artifacts; the historical normalized scalar error is archived outside this module.
This block compares X -> mu, independently of posterior-distribution topology.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import json
import numpy as np
import pandas as pd
from ._diagrams import _finite_diagram
from .landscape import LANDSCAPE_NAMES, LandscapeConfig, run_landscape_metrics, NORMALIZATION
from gudhi import bottleneck_distance
from gudhi.representations import BettiCurve, Entropy
from sklearn.metrics import pairwise_distances
from typing import TYPE_CHECKING

try:
    from ripser import ripser
    from persim import wasserstein
except ImportError:  # pragma: no cover
    ripser = None
    wasserstein = None

if TYPE_CHECKING:
    from beta_vae_experiments.metric_pipeline import CoreMetricInputs

WASSERSTEIN_METRIC_NAMES = (
    "topology_h0_normalized_wasserstein",
    "topology_h1_normalized_wasserstein",
)
BOTTLENECK_METRIC_NAMES = (
    "topology_h0_normalized_bottleneck",
    "topology_h1_normalized_bottleneck",
)
ENTROPY_METRIC_NAMES = tuple(
    f"topology_h{dim}_persistence_entropy_{space}"
    for dim in (0, 1) for space in ("x", "mu")
)
PERSISTENCE_METRIC_NAMES = WASSERSTEIN_METRIC_NAMES + BOTTLENECK_METRIC_NAMES + ENTROPY_METRIC_NAMES + LANDSCAPE_NAMES
TOPOLOGY_METRIC_NAMES = PERSISTENCE_METRIC_NAMES

@dataclass(frozen=True)
class TopologyMetricConfig:
    """Controls for the focused structural and persistence topology block."""

    max_samples: int = 1000
    random_state: int = 31
    diameter_epsilon: float = 1e-12
    # Accepted only for compatibility with historical configurations; unused.
    srtd_quantile: float = 0.90
    ph_visualizations_enabled: bool = False
    landscape_num_landscapes: int = 5
    landscape_resolution: int = 100

    def __post_init__(self) -> None:
        LandscapeConfig(self.landscape_num_landscapes, self.landscape_resolution)
        if self.max_samples < 2:
            raise ValueError("topology max_samples must be at least two")
        if self.diameter_epsilon <= 0:
            raise ValueError("topology diameter_epsilon must be positive")
        if not 0 < self.srtd_quantile <= 1:
            raise ValueError("topology srtd_quantile must lie in (0, 1]")
        if type(self.ph_visualizations_enabled) is not bool:
            raise TypeError("ph_visualizations_enabled must be a boolean")



def validate_paired_dissimilarity_matrices(
    input_distances: np.ndarray,
    latent_distances: np.ndarray,
    *,
    atol: float = 1e-10,
    rtol: float = 1e-8,
    negative_tolerance: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and canonically clean paired dissimilarity matrices."""
    matrices = (
        np.asarray(input_distances, dtype=np.float64),
        np.asarray(latent_distances, dtype=np.float64),
    )
    for name, matrix in zip(("input_distances", "latent_distances"), matrices):
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{name} must be a square two-dimensional matrix")
        if not np.isfinite(matrix).all():
            raise ValueError(f"{name} must contain only finite values")
        if float(np.min(matrix)) < -negative_tolerance:
            raise ValueError(f"{name} contains materially negative dissimilarities")
        if not np.allclose(matrix, matrix.T, atol=atol, rtol=rtol):
            raise ValueError(f"{name} is materially asymmetric")
        if not np.allclose(np.diag(matrix), 0.0, atol=atol, rtol=0.0):
            raise ValueError(f"{name} must have an approximately zero diagonal")
    if matrices[0].shape != matrices[1].shape:
        raise ValueError("Input and latent dissimilarity matrices must have equal shapes")

    canonical = []
    for matrix in matrices:
        cleaned = np.maximum(0.5 * (matrix + matrix.T), 0.0)
        np.fill_diagonal(cleaned, 0.0)
        canonical.append(cleaned)
    return canonical[0], canonical[1]


def _dependency_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unavailable"


def _undefined_values(
    status: str,
    warning: str,
    n_samples: int,
    names: tuple[str, ...] = TOPOLOGY_METRIC_NAMES,
) -> list[dict[str, object]]:
    return [
        {"metric_name": name, "value": float("nan"), "status": status, "warning": warning, "n_samples": n_samples, "split": "test"}
        for name in names
    ]


def compute_persistence_diagrams_h0_h1(
    input_distances: np.ndarray,
    latent_distances: np.ndarray,
    input_diameter: float,
    latent_diameter: float,
) -> dict[str, np.ndarray] | None:
    """Compute normalized X/mu H0/H1 diagrams exactly once with Ripser.

    Input representation:
        Paired Euclidean dissimilarity matrices from aligned ``X_test`` and
        ``mu_test`` rows, normalized separately by their maximum distances.
    """
    if ripser is None:
        warning = "ripser is required for persistent-homology artifacts."
        print(warning)
        return None
    input_diagrams = ripser(
        input_distances / input_diameter, distance_matrix=True, maxdim=1
    )["dgms"]
    latent_diagrams = ripser(
        latent_distances / latent_diameter, distance_matrix=True, maxdim=1
    )["dgms"]
    return {
            "topology_input_h0_persistence_diagram": np.asarray(
                input_diagrams[0], dtype=np.float64
            ),
            "topology_input_h1_persistence_diagram": np.asarray(
                input_diagrams[1], dtype=np.float64
            ),
            "topology_latent_h0_persistence_diagram": np.asarray(
                latent_diagrams[0], dtype=np.float64
            ),
            "topology_latent_h1_persistence_diagram": np.asarray(
                latent_diagrams[1], dtype=np.float64
            ),
        }


def compute_normalized_wasserstein_h0_h1(
    diagrams: dict[str, np.ndarray],
    n_samples: int,
) -> pd.DataFrame:
    """Compare cached normalized H0/H1 diagrams with Wasserstein distance.

    Input representation:
        Ripser persistence diagrams already computed from the separately
        diameter-normalized paired ``X_test`` and ``mu_test`` metric spaces.
    """
    if wasserstein is None:
        warning = "persim is required for topology Wasserstein metrics."
        return pd.DataFrame.from_records(_undefined_values(
                "failed_source_verification", warning, n_samples,
                WASSERSTEIN_METRIC_NAMES,
            ))
    values: list[dict[str, object]] = []
    for dimension, name in enumerate(WASSERSTEIN_METRIC_NAMES):
        input_finite = _finite_diagram(
            diagrams[f"topology_input_h{dimension}_persistence_diagram"]
        )
        latent_finite = _finite_diagram(
            diagrams[f"topology_latent_h{dimension}_persistence_diagram"]
        )
        distance = (
            0.0
            if len(input_finite) == len(latent_finite) == 0
            else float(wasserstein(input_finite, latent_finite, matching=False))
        )
        values.append({"metric_name": name, "value": distance, "status": "ok", "warning": None, "n_samples": n_samples, "split": "test"})
    return pd.DataFrame.from_records(values)


def compute_normalized_bottleneck_h0_h1(
    diagrams: dict[str, np.ndarray], n_samples: int,
) -> pd.DataFrame:
    """GUDHI bottleneck on cached diameter-normalized finite diagram points.

    Input representation: cached finite diagrams of separately normalized X/mu."""
    rows = []
    for dimension, name in enumerate(BOTTLENECK_METRIC_NAMES):
        try:
            dx = _finite_diagram(diagrams[f"topology_input_h{dimension}_persistence_diagram"])
            dm = _finite_diagram(diagrams[f"topology_latent_h{dimension}_persistence_diagram"])
            value = 0.0 if len(dx) == len(dm) == 0 else float(bottleneck_distance(dx, dm, e=0.0))
            if not np.isfinite(value):
                raise ValueError("Non-finite bottleneck distance")
            rows.append(dict(metric_name=name, value=value, status="ok", warning=None,
                             n_samples=n_samples, split="test"))
        except (ValueError, RuntimeError) as error:
            rows.extend(_undefined_values("degenerate_input", f"Bottleneck failed: {error}", n_samples, (name,)))
    return pd.DataFrame.from_records(rows)




def compute_persistence_entropy(diagrams, n_samples):
    """GUDHI scalar Shannon entropy of positive finite bar lengths (natural log).

    Atienza et al., https://arxiv.org/abs/1803.08304. Empty diagrams are
    undefined rather than equated to a one-bar diagram (entropy zero).
    No extra division by log(feature count), no cross-space difference.
    """
    records = []
    for dimension in (0, 1):
        for space, key in [('x', 'input'), ('mu', 'latent')]:
            name = f'topology_h{dimension}_persistence_entropy_{space}'
            diagram = _finite_diagram(diagrams[f'topology_{key}_h{dimension}_persistence_diagram'])
            diagram = diagram[diagram[:, 1] > diagram[:, 0]]
            if not len(diagram):
                records.extend(_undefined_values('degenerate_input',
                    'Persistence entropy requires positive finite lifetimes.', n_samples, (name,)))
            else:
                value = float(Entropy(mode='scalar', normalized=False).fit_transform([diagram])[0, 0])
                records.append(dict(metric_name=name, value=value, status='ok',
                                    warning=None, n_samples=n_samples, split='test'))
    return pd.DataFrame(records)


def compute_betti_curve_artifacts(
    diagrams: dict[str, np.ndarray],
) -> dict[str, np.ndarray | str]:
    """Derive shared-grid H0/H1 Betti curves from cached Ripser diagrams.

    Input representation:
        The same cached, normalized Ripser H0/H1 diagrams consumed by the
        Wasserstein comparison; essential infinite H0 bars are excluded.
    """
    artifacts: dict[str, np.ndarray | str] = {
        "topology_betti_curve_backend": "gudhi.representations.BettiCurve",
    }
    for dimension in (0, 1):
        input_diagram = _finite_diagram(
            diagrams[f"topology_input_h{dimension}_persistence_diagram"]
        )
        latent_diagram = _finite_diagram(
            diagrams[f"topology_latent_h{dimension}_persistence_diagram"]
        )
        if len(input_diagram) == len(latent_diagram) == 0:
            grid = np.empty(0, dtype=np.float64)
            curves = np.empty((2, 0), dtype=np.int64)
        else:
            transformer = BettiCurve(resolution=None, predefined_grid=None)
            curves = transformer.fit_transform([input_diagram, latent_diagram])
            grid = np.asarray(transformer.grid_, dtype=np.float64)
        artifacts[f"topology_betti_h{dimension}_grid"] = grid
        artifacts[f"topology_betti_h{dimension}_x"] = np.asarray(curves[0])
        artifacts[f"topology_betti_h{dimension}_mu"] = np.asarray(curves[1])
    return artifacts


def run_topology_metrics_pipeline(
    inputs: CoreMetricInputs,
    selected_indices: np.ndarray,
    x_selected: np.ndarray,
    mu_selected: np.ndarray,
    config: TopologyMetricConfig,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Compare the same aligned test observations in ``X`` and deterministic ``mu``.

    Input representation:
        One shared deterministic subset of aligned ``X_test`` and ``mu_test``
        rows together with its cached Euclidean dissimilarity matrices.
    """
    x = np.asarray(x_selected, dtype=np.float64)
    mu = np.asarray(mu_selected, dtype=np.float64)
    n_samples = len(x)
    if n_samples < 2:
        warning = "Topology comparison requires at least two aligned observations."
        return pd.DataFrame.from_records(
            _undefined_values("degenerate_input", warning, n_samples)
        )
    if len(x) != len(mu):
        raise ValueError("X_test and mu_test must describe the same rows")
    from sklearn.metrics import pairwise_distances

    input_distances, latent_distances = validate_paired_dissimilarity_matrices(
        pairwise_distances(x), pairwise_distances(mu)
    )
    input_diameter = float(np.max(input_distances))
    latent_diameter = float(np.max(latent_distances))
    if (
        input_diameter <= config.diameter_epsilon
        or latent_diameter <= config.diameter_epsilon
    ):
        warning = "Input or posterior-mean metric space has negligible diameter."
        return pd.DataFrame.from_records(
            _undefined_values("degenerate_input", warning, n_samples)
        )

    persistence = compute_persistence_diagrams_h0_h1(
        input_distances,
        latent_distances,
        input_diameter,
        latent_diameter,
    )
    if persistence is None:
        warning = "ripser is required for persistent-homology artifacts."
        persistence_result = pd.DataFrame.from_records(_undefined_values(
            "failed_source_verification", warning, n_samples,
            PERSISTENCE_METRIC_NAMES,
        ))
    else:
        diagrams = persistence
        wasserstein_result = compute_normalized_wasserstein_h0_h1(
            diagrams, n_samples
        )
        bottleneck_result = compute_normalized_bottleneck_h0_h1(diagrams, n_samples)
        betti_curves = compute_betti_curve_artifacts(diagrams)
        ids = getattr(inputs, 'sample_ids_test', None)
        provenance = dict(run_name=inputs.run_name, split='test', representation='mu',
            selected_indices=np.asarray(selected_indices).tolist(),
            sample_ids=None if ids is None else np.asarray(ids).astype(str)[selected_indices].tolist(),
            n_samples=n_samples, input_diameter=input_diameter, latent_diameter=latent_diameter,
            normalization=NORMALIZATION, sample_alignment='shared _select_test_rows indices')
        landscape_dir = None if output_dir is None else Path(output_dir)/'metrics_artifacts'/inputs.run_name/'landscape'
        landscape = run_landscape_metrics(diagrams, n_samples, landscape_dir, provenance,
            LandscapeConfig(config.landscape_num_landscapes, config.landscape_resolution))
        persistence_result = pd.concat([wasserstein_result, bottleneck_result,
            compute_persistence_entropy(diagrams, n_samples), landscape], ignore_index=True)
        if output_dir is not None:
            directory = Path(output_dir) / "absolute_core_metrics_v1" / "topology" / inputs.run_name
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
            for name, diagram in diagrams.items():
                path = directory / f"{name}.npy"
                np.save(path, diagram)
                print(path)
            for name, values in betti_curves.items():
                if isinstance(values, np.ndarray):
                    path = directory / f"{name}.npy"
                    np.save(path, values)
                    print(path)
    return persistence_result.reset_index(drop=True)
