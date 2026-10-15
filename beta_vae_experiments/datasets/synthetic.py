"""Controlled synthetic-manifold inputs for the shared beta-VAE pipeline.

The synthetic manifold datasets provide controlled ground-truth benchmarks for
validating geometry and topology diagnostics used in the beta-VAE experiments.

The circle S1 provides a simple known topological structure with one connected
component and one one-dimensional loop. The sphere S2 adds a closed
two-dimensional surface with one two-dimensional void and no loop, so H2 and
the absence of H1 can be checked. The Swiss roll provides a nonlinear
two-dimensional manifold embedded in three-dimensional space and retains its
intrinsic generator coordinate t for later ground-truth geometry analyses.

The same training and evaluation pipeline used for real datasets is applied to
these synthetic benchmarks. This module only generates data, identifiers,
ground-truth metadata, and deterministic split membership; it defines no new
geometry, topology, downstream, or phase-transition metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Any, Mapping

from autoencodix.data.datapackage import DataPackage
import numpy as np
import pandas as pd
from scipy.stats import uniform_direction
from sklearn.datasets import make_swiss_roll

@dataclass(frozen=True)
class SyntheticManifoldDataset:
    """One immutable generated point cloud and its sample-aligned ground truth."""

    dataset_id: str
    x_raw: pd.DataFrame
    sample_metadata: pd.DataFrame
    metadata: dict[str, Any]

    def to_data_package(self) -> DataPackage:
        """Return the standard AUTOENCODIX numeric/annotation input contract."""
        return DataPackage(
            multi_bulk={"SYNTHETIC": self.x_raw.copy()},
            annotation={"paired": self.sample_metadata.copy()},
        )


def _validated_config(dataset_id: str, config: Mapping[str, Any] | None) -> dict[str, Any]:
    values = dict(config or {})
    common = {"n_samples", "seed", "dataset_seed", "split_seed", "correlated_copies", "observation_noise"}
    specific = {"radius"} if dataset_id in {"circle", "sphere"} else {"noise"}
    unexpected = set(values).difference(common | specific)
    if unexpected:
        raise ValueError(
            f"Unsupported {dataset_id} dataset keys: {sorted(unexpected)}"
        )
    n_samples = int(values.get("n_samples", 1000))
    dataset_seed = int(values.get("dataset_seed", values.get("seed", 42)))
    split_seed = int(values.get("split_seed", 42))
    if n_samples < 20:
        raise ValueError("Synthetic datasets require at least 20 samples")
    return {
        **values,
        "n_samples": n_samples,
        "dataset_seed": dataset_seed,
        "split_seed": split_seed,
    }


def _circle(n_samples: int, dataset_seed: int, radius: float) -> np.ndarray:
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Circle radius must be positive and finite")
    # GUDHI 3.11 points.sphere has no seed API and its C++ RNG is unaffected by
    # NumPy seeding. Reproducibility is mandatory here, so use SciPy's official
    # uniform-direction generator with an explicit random_state instead.
    generated = uniform_direction.rvs(
        dim=2,
        size=n_samples,
        random_state=dataset_seed,
    ) * float(radius)
    return np.asarray(generated, dtype=np.float64)


def _sphere(n_samples: int, dataset_seed: int, radius: float) -> np.ndarray:
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Sphere radius must be positive and finite")
    # Same seeded SciPy generator as the circle, on the 2-sphere in R^3.
    generated = uniform_direction.rvs(
        dim=3,
        size=n_samples,
        random_state=dataset_seed,
    ) * float(radius)
    return np.asarray(generated, dtype=np.float64)


def _generate_synthetic_manifold(
    dataset_id: str,
    config: Mapping[str, Any] | None = None,
) -> SyntheticManifoldDataset:
    """Generate Circle S1, Sphere S2 or Swiss Roll with stable IDs and ground truth."""
    dataset_id = str(dataset_id)
    values = _validated_config(dataset_id, config)
    n_samples = values["n_samples"]
    dataset_seed = values["dataset_seed"]
    split_seed = values["split_seed"]

    if dataset_id == "circle":
        radius = float(values.get("radius", 1.0))
        x = _circle(n_samples, dataset_seed, radius)
        sample_ids = [f"circle_{index:06d}" for index in range(n_samples)]
        columns = ["circle_x", "circle_y"]
        sample_metadata = pd.DataFrame(
            {
                "ground_truth_angle": np.mod(
                    np.arctan2(x[:, 1], x[:, 0]), 2.0 * np.pi
                )
            },
            index=sample_ids,
        )
        metadata: dict[str, Any] = {
            "dataset_family": "synthetic_manifold",
            "manifold_name": "circle",
            "topology": "S1",
            "ambient_dimension": 2,
            "intrinsic_dimension": 1,
            "betti_0": 1,
            "betti_1": 1,
            "known_betti_0": 1,
            "known_betti_1": 1,
            "radius": radius,
            "ground_truth_coordinate": "angle",
            "ground_truth_angle_available": True,
        }
    elif dataset_id == "sphere":
        radius = float(values.get("radius", 1.0))
        x = _sphere(n_samples, dataset_seed, radius)
        sample_ids = [f"sphere_{index:06d}" for index in range(n_samples)]
        columns = ["sphere_x", "sphere_y", "sphere_z"]
        sample_metadata = pd.DataFrame(
            {
                # Polar angle in [0, pi] and azimuth in [0, 2 pi).
                "ground_truth_polar": np.arccos(
                    np.clip(x[:, 2] / radius, -1.0, 1.0)
                ),
                "ground_truth_azimuth": np.mod(
                    np.arctan2(x[:, 1], x[:, 0]), 2.0 * np.pi
                ),
            },
            index=sample_ids,
        )
        metadata = {
            "dataset_family": "synthetic_manifold",
            "manifold_name": "sphere",
            "topology": "S2",
            "ambient_dimension": 3,
            "intrinsic_dimension": 2,
            "betti_0": 1,
            "betti_1": 0,
            "betti_2": 1,
            "known_betti_0": 1,
            "known_betti_1": 0,
            "known_betti_2": 1,
            "radius": radius,
            "ground_truth_coordinate": "polar, azimuth",
            "ground_truth_angles_available": True,
        }
    else:
        noise = float(values.get("noise", 0.0))
        if not np.isfinite(noise) or noise < 0:
            raise ValueError("Swiss-roll noise must be non-negative and finite")
        x, t = make_swiss_roll(
            n_samples=n_samples,
            noise=noise,
            random_state=dataset_seed,
            hole=False,
        )
        x = np.asarray(x, dtype=np.float64)
        sample_ids = [f"swiss_roll_{index:06d}" for index in range(n_samples)]
        columns = ["swiss_roll_x", "swiss_roll_y", "swiss_roll_z"]
        sample_metadata = pd.DataFrame(
            {"ground_truth_t": np.asarray(t, dtype=np.float64)},
            index=sample_ids,
        )
        metadata = {
            "dataset_family": "synthetic_manifold",
            "manifold_name": "swiss_roll",
            "ambient_dimension": 3,
            "intrinsic_dimension": 2,
            "ground_truth_coordinate": "t",
            "ground_truth_t_available": True,
            "noise": noise,
        }

    if x.shape != (n_samples, int(metadata["ambient_dimension"])):
        raise RuntimeError(f"Generator returned unexpected shape {x.shape}")
    copies = values.get("correlated_copies", 1)
    observation_noise = float(values.get("observation_noise", 0.0))
    if isinstance(copies, bool) or not isinstance(copies, int) or copies < 1:
        raise ValueError("correlated_copies must be a positive integer")
    if not np.isfinite(observation_noise) or observation_noise < 0:
        raise ValueError("observation_noise must be finite and nonnegative")
    if copies > 1 or observation_noise > 0:
        # Ground truth still describes the underlying noiseless manifold.
        base_dimension = x.shape[1]
        x = np.tile(x, copies)
        scale = x.std(axis=0, keepdims=True)
        rng = np.random.default_rng(dataset_seed + 104729)
        x = x + observation_noise * scale * rng.standard_normal(x.shape)
        columns = [f"observed_{i}" for i in range(x.shape[1])]
        metadata.update(base_ambient_dimension=base_dimension, ambient_dimension=x.shape[1],
                        correlated_copies=copies, observation_noise=observation_noise,
                        topology_describes="underlying_noiseless_manifold")
    frame = pd.DataFrame(x, index=sample_ids, columns=columns)
    metadata.update(
        {
            "dataset_id": dataset_id,
            "dataset_seed": dataset_seed,
            "split_seed": split_seed,
            "n_samples": n_samples,
        }
    )
    return SyntheticManifoldDataset(dataset_id, frame, sample_metadata, metadata)


def generate_circle(
    config: Mapping[str, Any] | None = None,
) -> SyntheticManifoldDataset:
    """Generate the Circle dataset selected by JSON ID ``circle``."""
    return _generate_synthetic_manifold("circle", config)


def generate_sphere(
    config: Mapping[str, Any] | None = None,
) -> SyntheticManifoldDataset:
    """Generate the Sphere dataset selected by JSON ID ``sphere``."""
    return _generate_synthetic_manifold("sphere", config)


def generate_swiss_roll(
    config: Mapping[str, Any] | None = None,
) -> SyntheticManifoldDataset:
    """Generate the Swiss-roll dataset selected by JSON ID ``swiss_roll``."""
    return _generate_synthetic_manifold("swiss_roll", config)


SYNTHETIC_DATASET_GENERATORS: dict[
    str, Callable[[Mapping[str, Any] | None], SyntheticManifoldDataset]
] = {
    "circle": generate_circle,
    "sphere": generate_sphere,
    "swiss_roll": generate_swiss_roll,
}


def is_synthetic_dataset(dataset_id: str) -> bool:
    """Return whether a JSON dataset ID has a registered synthetic generator."""
    return dataset_id in SYNTHETIC_DATASET_GENERATORS


def generate_synthetic_dataset(
    dataset_id: str,
    config: Mapping[str, Any] | None = None,
) -> SyntheticManifoldDataset:
    """Dispatch a JSON dataset ID to its registered generator function."""
    try:
        generator = SYNTHETIC_DATASET_GENERATORS[dataset_id]
    except KeyError as error:
        raise ValueError(f"Unknown synthetic dataset_id {dataset_id!r}") from error
    return generator(config)


def save_synthetic_ground_truth(
    splits: dict[str, dict[str, Any]],
    output_dir: Path,
    run_name: str,
) -> Path | None:
    """Persist sample-aligned synthetic ground truth after training."""
    rows: list[pd.DataFrame] = []
    for split in ("train", "valid", "test"):
        metadata = splits[split].get("sample_metadata")
        if not isinstance(metadata, pd.DataFrame):
            continue
        ground_truth_columns = [
            column for column in metadata if str(column).startswith("ground_truth_")
        ]
        if not ground_truth_columns:
            continue
        frame = metadata[ground_truth_columns].copy()
        frame.insert(0, "sample_id", splits[split]["sample_ids"])
        frame.insert(1, "split", split)
        rows.append(frame.reset_index(drop=True))
    if not rows:
        return None
    destination = Path(output_dir) / "metrics_artifacts" / run_name
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "synthetic_ground_truth.csv"
    pd.concat(rows, ignore_index=True).to_csv(path, index=False)
    return path


def synthetic_split_indices(
    sample_ids: np.ndarray | list[str],
    *,
    split_seed: int,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
) -> dict[str, np.ndarray]:
    """Mirror the project's deterministic ID-based train/valid/test split."""
    identifiers = np.asarray([str(value) for value in sample_ids], dtype=object)
    if len(set(identifiers.tolist())) != len(identifiers):
        raise ValueError("Synthetic sample IDs must be unique")
    if not 0 < train_ratio < 1 or not 0 <= valid_ratio < 1 - train_ratio:
        raise ValueError("Invalid train/validation ratios")
    id_to_index = {sample_id: index for index, sample_id in enumerate(identifiers)}
    ordered = np.asarray(sorted(identifiers.tolist()), dtype=object)
    shuffled = np.random.default_rng(int(split_seed)).permutation(ordered)
    n_train = int(len(shuffled) * train_ratio)
    n_valid = int(len(shuffled) * valid_ratio)
    memberships = {
        "train": shuffled[:n_train],
        "valid": shuffled[n_train : n_train + n_valid],
        "test": shuffled[n_train + n_valid :],
    }
    return {
        split: np.asarray(sorted(id_to_index[value] for value in values), dtype=int)
        for split, values in memberships.items()
    }


__all__ = [
    "SYNTHETIC_DATASET_GENERATORS",
    "SyntheticManifoldDataset",
    "generate_circle",
    "generate_sphere",
    "generate_swiss_roll",
    "generate_synthetic_dataset",
    "is_synthetic_dataset",
    "save_synthetic_ground_truth",
    "synthetic_split_indices",
]
