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
from .gudhi_data import load_gudhi_dataset

__all__ = [
    "load_gudhi_dataset",
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
