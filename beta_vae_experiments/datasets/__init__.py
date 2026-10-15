"""Dataset adapters used by the shared beta-VAE experiment pipeline."""

from .synthetic import (
    SYNTHETIC_DATASET_GENERATORS,
    SyntheticManifoldDataset,
    generate_circle,
    generate_sphere,
    generate_swiss_roll,
    generate_synthetic_dataset,
    is_synthetic_dataset,
    save_synthetic_ground_truth,
    synthetic_split_indices,
)

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
