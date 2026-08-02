"""Data loading, final-snapshot extraction, alignment, and split provenance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from autoencodix.configs.default_config import DataConfig, DataInfo
from autoencodix.utils._result import Result

try:
    from beta_vae_experiments.metrics import CoreMetricInputs
except ModuleNotFoundError:
    from metrics import CoreMetricInputs  # type: ignore[no-redef]


def build_data_config(project_root: Path) -> DataConfig:
    """Build the real multi-bulk DataConfig from repository-local parquet files."""
    tutorial_root = project_root / "data" / "raw" / "OntixTutorialData"
    paths = {
        "RNA": tutorial_root / "combined_rnaseq_formatted.parquet",
        "METH": tutorial_root / "combined_meth_formatted.parquet",
        "ANNO": tutorial_root / "combined_clin_formatted.parquet",
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Required input data files are missing:\n{formatted}")
    return DataConfig(
        data_info={
            "RNA": DataInfo(
                file_path=str(paths["RNA"]),
                data_type="NUMERIC",
                filtering="VAR",
            ),
            "METH": DataInfo(
                file_path=str(paths["METH"]),
                data_type="NUMERIC",
                filtering="VAR",
            ),
            "ANNO": DataInfo(
                file_path=str(paths["ANNO"]),
                data_type="ANNOTATION",
            ),
        }
    )


def _to_numpy_2d(value: Any, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "toarray"):
        value = value.toarray()
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _validate_sample_ids(values: Any, expected: int, name: str) -> np.ndarray:
    if values is None:
        raise ValueError(f"{name} sample IDs are unavailable")
    ids = np.asarray(values, dtype=object).ravel()
    if len(ids) != expected:
        raise ValueError(f"{name} has {len(ids)} sample IDs, expected {expected}")
    ids = np.asarray([str(value).strip() for value in ids], dtype=object)
    if any(not value for value in ids):
        raise ValueError(f"{name} contains empty sample IDs")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError(f"{name} contains duplicate sample IDs")
    return ids


def _analysis_matrix(
    value: Any,
    name: str,
    expected_rows: int,
    *,
    optional: bool = False,
) -> np.ndarray | None:
    if isinstance(value, dict) and not value:
        value = None
    if isinstance(value, np.ndarray) and value.size == 0:
        value = None
    if value is None:
        if optional:
            return None
        raise ValueError(f"{name} is unavailable")
    array = _to_numpy_2d(value, name)
    if len(array) != expected_rows:
        raise ValueError(f"{name} has {len(array)} rows, expected {expected_rows}")
    return array


def prepare_analysis_data(result: Result, epoch: int = -1) -> dict[str, dict[str, Any]]:
    """Prepare the matrices required by the analysis for every data split."""
    splits: dict[str, dict[str, Any]] = {}

    for split in ("train", "valid", "test"):
        dataset = getattr(result.datasets, split, None)
        if dataset is None:
            raise ValueError(f"Dataset split {split!r} is unavailable")

        x = _to_numpy_2d(dataset.data, f"x_{split}")
        sample_ids = _validate_sample_ids(
            result.sample_ids.get(split=split, epoch=epoch),
            len(x),
            f"sample_ids_{split}",
        )
        dataset_ids = _validate_sample_ids(
            dataset.sample_ids,
            len(x),
            f"dataset_ids_{split}",
        )
        for index, (dataset_id, result_id) in enumerate(
            zip(dataset_ids, sample_ids, strict=True)
        ):
            if dataset_id.lower() != result_id.lower():
                raise ValueError(
                    f"Sample-ID mismatch for split {split!r} at row {index}: "
                    f"{dataset_id!r} != {result_id!r}"
                )

        # GeneralTrainer._capture_dynamics uses the same ``indices_np`` to write
        # sample_ids, latentspaces, reconstructions, mus, and sigmas into equally
        # sized split buffers. Dataset data uses that same index space. Therefore
        # row i identifies the same sample in x and in every stored result matrix;
        # the positional ID check above verifies this invariant without reordering.
        split_data: dict[str, Any] = {"x": x, "sample_ids": sample_ids}
        for key in ("reconstructions", "mus", "sigmas", "latentspaces"):
            dynamics = getattr(result, key)
            split_data[key] = _analysis_matrix(
                dynamics.get(split=split, epoch=epoch),
                f"{key}_{split}",
                len(x),
                optional=key in {"sigmas", "latentspaces"},
            )
        splits[split] = split_data

    return splits


def validate_or_create_split_manifest(
    splits: dict[str, dict[str, Any]],
    output_dir: Path,
    split_seed: int,
) -> Path:
    """Persist the first deterministic split and reject later membership changes."""
    path = output_dir / "provenance" / "fixed_split_sample_ids.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    current = {
        split: sorted(str(value) for value in splits[split]["sample_ids"])
        for split in ("train", "valid", "test")
    }
    combined = [value for values in current.values() for value in values]
    if len(combined) != len(set(combined)):
        raise ValueError("A sample occurs in more than one split")
    manifest = {
        "data_split_seed": int(split_seed),
        "split_order_is_significant": False,
        "sample_ids": current,
    }
    if not path.exists():
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return path
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored_ids = stored.get("sample_ids", {})
    stored_membership = {
        split: sorted(str(value) for value in stored_ids.get(split, []))
        for split in ("train", "valid", "test")
    }
    stored_seed = stored.get("data_split_seed")
    if stored_seed != int(split_seed) or stored_membership != current:
        raise ValueError(
            "Train/valid/test sample IDs differ from the stored reference split"
        )
    if stored != manifest:
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def build_core_metric_inputs(
    run_name: str,
    splits: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
) -> CoreMetricInputs:
    """Convert aligned AUTOENCODIX split snapshots to the core metric contract."""
    train, valid, test = splits["train"], splits["valid"], splits["test"]
    return CoreMetricInputs(
        run_name=run_name,
        x_train=train["x"],
        x_val=valid["x"],
        x_test=test["x"],
        x_recon_train=train["reconstructions"],
        x_recon_val=valid["reconstructions"],
        x_recon_test=test["reconstructions"],
        mu_train=train["mus"],
        mu_val=valid["mus"],
        mu_test=test["mus"],
        # Result.sigmas contains posterior log-variance, not standard deviation.
        logvar_train=train["sigmas"],
        logvar_val=valid["sigmas"],
        logvar_test=test["sigmas"],
        z_train=train["latentspaces"],
        z_val=valid["latentspaces"],
        z_test=test["latentspaces"],
        sample_ids_train=train["sample_ids"],
        sample_ids_val=valid["sample_ids"],
        sample_ids_test=test["sample_ids"],
        metadata=metadata,
    )
