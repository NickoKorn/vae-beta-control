"""GUDHI point clouds as sample-aligned Pandas tables and AUTOENCODIX inputs.

GUDHI returns NumPy coordinates. DataFrames add stable sample/feature IDs;
activity labels remain annotations, never input features. Downloaded data are
cached locally. Source: https://gudhi.inria.fr/python/latest/datasets.html
"""

from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd

from .synthetic import SyntheticManifoldDataset

GUDHI_DATASET_IDS = ("gudhi_torus", "gudhi_spiral", "gudhi_bunny", "gudhi_activities")
_GENERATOR_LOCK = Lock()


def load_gudhi_dataset(dataset_id, config=None) -> SyntheticManifoldDataset:
    """Load/generate coordinates, select a seeded cohort and attach annotations.

    Options: n_samples (default 5000), dataset_seed (42), cache_dir
    (data/raw/gudhi), accept_license (False); torus additionally supports dim
    (2). A remote dataset is downloaded in full before optional subsampling.
    Original row numbers identify downloaded points across cohort selections.
    Use the returned object's to_data_package() for native AUTOENCODIX input.
    """
    if dataset_id not in GUDHI_DATASET_IDS:
        raise ValueError(f"Unknown GUDHI dataset: {dataset_id}")
    options = dict(config or {})
    allowed = {"n_samples", "dataset_seed", "cache_dir", "accept_license"}
    if dataset_id == "gudhi_torus":
        allowed.add("dim")
    if set(options) - allowed:
        raise ValueError(f"Unknown GUDHI options: {sorted(set(options) - allowed)}")
    n = options.get("n_samples", 5000)
    seed = options.get("dataset_seed", 42)
    if isinstance(n, bool) or not isinstance(n, int) or n < 20:
        raise ValueError("n_samples must be an integer >= 20")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("dataset_seed must be an integer in [0, 2**32)")
    accept = options.get("accept_license", False)
    if not isinstance(accept, bool):
        raise ValueError("accept_license must be a boolean")
    metadata = dict(dataset_id=dataset_id, dataset_seed=seed,
                    source="https://gudhi.inria.fr/python/latest/datasets.html")
    labels = None
    if dataset_id == "gudhi_torus":
        from gudhi.datasets.generators import points

        dim = options.get("dim", 2)
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 1:
            raise ValueError("dim must be a positive integer")
        # GUDHI 3.11 torus uses NumPy's legacy global RNG. Restore its state
        # so generating data does not reseed subsequent training/splitting.
        with _GENERATOR_LOCK:
            state = np.random.get_state()
            try:
                np.random.seed(seed)
                raw = points.torus(n_samples=n, dim=dim, sample="random")
            finally:
                np.random.set_state(state)
        indices = np.arange(n)
        metadata.update(intrinsic_dimension=dim, geometry="flat_torus")
    else:
        from gudhi.datasets import remote

        cache = Path(options.get("cache_dir", "data/raw/gudhi")).expanduser()
        cache.mkdir(parents=True, exist_ok=True)
        path = str((cache / f"{dataset_id}.npy").resolve())
        if dataset_id == "gudhi_spiral":
            raw = remote.fetch_spiral_2d(file_path=path)
        else:
            if not accept:
                raise ValueError("Review the dataset license and set accept_license=true")
            fetch = remote.fetch_bunny if dataset_id == "gudhi_bunny" else remote.fetch_daily_activities
            raw = fetch(file_path=path, accept_license=True)
        raw = np.asarray(raw)
        if dataset_id == "gudhi_activities":
            if raw.ndim != 2 or raw.shape[1] != 4:
                raise ValueError("Expected three activity coordinates plus a label column")
            labels = raw[:, 3]
            raw = raw[:, :3]
        if n > len(raw):
            raise ValueError(f"Requested {n} points but the dataset has only {len(raw)}")
        indices = np.sort(np.random.default_rng(seed).choice(len(raw), n, replace=False))
        metadata.update(cache_file=path, source_n_samples=len(raw))
        raw = raw[indices]
    coordinates = np.asarray(raw, dtype=np.float64)
    if coordinates.ndim != 2 or not np.isfinite(coordinates).all():
        raise ValueError("GUDHI coordinates must be a finite samples-by-features matrix")
    ids = pd.Index([f"{dataset_id}_{i:08d}" for i in indices], name="sample_id")
    frame = pd.DataFrame(coordinates, index=ids,
                         columns=[f"coordinate_{i}" for i in range(coordinates.shape[1])])
    annotations = pd.DataFrame(index=ids)
    if labels is not None:
        names = {14: "cross_training", 18: "jumping", 13: "stepper", 9: "walking"}
        annotations["ACTIVITY"] = pd.Series(labels[indices], index=ids).map(names)
        if annotations["ACTIVITY"].isna().any():
            raise ValueError("Unrecognized activity label")
    metadata.update(n_samples=len(frame), ambient_dimension=frame.shape[1])
    return SyntheticManifoldDataset(dataset_id, frame, annotations, metadata)
