"""Translate data between the experiment, AUTOENCODIX, and metric contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from autoencodix.configs.default_config import DataConfig, DataInfo
from autoencodix.utils._result import Result

import numpy as np
import pandas as pd
import torch
import json
import pickle
import warnings
from beta_vae_experiments.metric_pipeline import CoreMetricInputs
from beta_vae_experiments.datasets import is_synthetic_dataset


def build_data_config(project_root: Path, dataset_id: str, options: dict | None = None) -> DataConfig:
    """Build the AUTOENCODIX data contract for the selected dataset."""
    if is_synthetic_dataset(dataset_id):
        return DataConfig(
            data_info={
                "SYNTHETIC": DataInfo(
                    data_type="NUMERIC",
                    filtering="NOFILT",
                    scaling="MINMAX",
                )
            }
        )

    """
    - TCGA: The Cancer Genome Atlas
    """
    if dataset_id != "tcga":
        raise ValueError(f"Unsupported dataset_id {dataset_id!r}")

    options = dict(options or {})
    unknown = set(options) - {"data_root", "rna_file", "methylation_file", "clinical_file"}
    if unknown:
        raise ValueError(f"Unknown TCGA dataset options: {sorted(unknown)}")
    tutorial_root = Path(options.get("data_root", project_root / "data" / "raw" / "TCGAPanCancerAutoencodix"))
    if not tutorial_root.is_absolute():
        tutorial_root = project_root / tutorial_root
    paths = {
        "RNA": tutorial_root / options.get("rna_file", "data_mrna_seq_v2_rsem_formatted.parquet"),
        "METH": tutorial_root / options.get("methylation_file", "data_methylation_per_gene_formatted.parquet"),
        "ANNO": tutorial_root / options.get("clinical_file", "data_clinical_formatted.parquet"),
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


def _to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if not value:
            return None
        if len(value) != 1:
            raise ValueError(
                "AUTOENCODIX returned multiple named arrays where one VARIX "
                f"matrix was expected: {tuple(value)}"
            )
        return _to_numpy(next(iter(value.values())))
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if hasattr(value, "toarray"):
        return np.asarray(value.toarray())
    return np.asarray(value)


def prepare_metric_inputs_from_varix_result(
    result: Result,
    epoch: int = -1,
    varix: Any | None = None,
) -> dict[str, dict[str, Any]]:
    """Extract metric data, retaining valid all-zero test posterior buffers.

    AUTOENCODIX currently omits ``mu``/``logvar`` from ``Result`` whenever the
    complete buffer sum equals zero. The trainer buffer still contains the
    actual prediction. For the final test snapshot only, use that buffer as a
    lossless compatibility fallback; no values are synthesized.
    """
    splits: dict[str, dict[str, Any]] = {}
    trainer = getattr(varix, "_trainer", None)
    buffer_names = {"mus": "_mu_buffer", "sigmas": "_sigma_buffer"}

    for split in ("train", "valid", "test"):
        dataset = getattr(result.datasets, split)
        split_data: dict[str, Any] = {
            "x": _to_numpy(dataset.data),
            "sample_metadata": getattr(dataset, "metadata", None),
            "sample_ids": _to_numpy(
                result.sample_ids.get(split=split, epoch=epoch)
            ),
        }

        for name in ("reconstructions", "mus", "sigmas", "latentspaces"):
            dynamics = getattr(result, name)
            value = _to_numpy(dynamics.get(split=split, epoch=epoch))
            source = "result"
            missing = value is None or value.size == 0
            #Missing posterior
            if missing and epoch == -1 and split == "test" and name in buffer_names:
                buffer = getattr(trainer, buffer_names[name], None)
                buffered = buffer.get("test") if isinstance(buffer, dict) else None
                candidate = _to_numpy(buffered)
                if (
                    candidate is not None
                    and candidate.ndim == 2
                    and len(candidate) == len(split_data["x"])
                ):
                    value = candidate
                    source = "autoencodix_trainer_buffer_fallback"
            if value is not None and value.size == 0:
                value = None
            split_data[name] = value
            split_data[f"{name}_source"] = source if value is not None else "missing"

        splits[split] = split_data

    return splits


def verify_test_posterior(splits: dict[str, dict[str, Any]], varix: Any) -> dict:
    """Check posterior consistency, recording small FP32 differences as warnings.

    The original rtol=1e-5/atol=1e-7 remains the diagnostic threshold.
    FP32 inference permits atol=1e-6 with the same rtol; larger discrepancies
    still fail. This absolute allowance is an engineering verification policy,
    not a collapse threshold. Stored posterior values are never altered.
    """
    test = splits["test"]
    trainer = varix._trainer
    model = trainer.get_model()
    parameter = next(model.parameters())
    means, logvars = [], []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode(), trainer._fabric.autocast():
            for start in range(0, len(test["x"]), trainer._config.batch_size):
                x = torch.as_tensor(
                    test["x"][start:start + trainer._config.batch_size],
                    device=parameter.device, dtype=parameter.dtype,
                )
                mu, logvar = model.encode(x)
                means.append(_to_numpy(mu))
                logvars.append(_to_numpy(logvar))
    finally:
        model.train(was_training)
    checks = {"verification_status": "ok", "warnings": [],
              "comparison_rtol": 1e-5, "diagnostic_atol": 1e-7}
    for key, batches in (("mus", means), ("sigmas", logvars)):
        fresh = np.concatenate(batches)
        stored = test[key]
        if stored is None or fresh.shape != stored.shape:
            raise ValueError(f"Fresh encoder {key} does not match stored posterior shape")
        if not np.isfinite(stored).all() or not np.isfinite(fresh).all():
            raise ValueError(f"Non-finite {key} in stored or fresh posterior")
        # Do not silently widen tolerances for float16/bfloat16 inference.
        atol = 1e-6 if fresh.dtype == np.float32 else 1e-7
        np.testing.assert_allclose(stored, fresh, rtol=1e-5, atol=atol,
                                   equal_nan=False,
                                   err_msg=f"Fresh encoder {key} disagrees with stored posterior")
        difference = np.abs(stored.astype(np.float64) - fresh.astype(np.float64))
        outside_strict = difference > (1e-7 + 1e-5 * np.abs(fresh.astype(np.float64)))
        checks[f"{key}_max_abs_difference"] = float(np.max(difference))
        checks[f"{key}_diagnostic_exceedance_count"] = int(outside_strict.sum())
        checks[f"{key}_comparison_atol"] = atol
        checks[f"{key}_inference_dtype"] = str(fresh.dtype)
        if outside_strict.any():
            message = (f"Fresh encoder {key}: {outside_strict.sum()} values exceed "
                       f"the strict diagnostic tolerance, but pass rtol=1e-5, atol={atol:g}; "
                       f"max absolute difference={difference.max():.9g}. Stored values retained.")
            checks["verification_status"] = "numerical_warning"
            checks["warnings"].append(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
    checks["fresh_encoder_verified"] = True
    return checks


def save_test_posterior(splits: dict[str, dict[str, Any]], output_dir: Path,
                        run_name: str, verification: dict | None = None) -> Path:
    """Persist real posterior arrays and collapse diagnostics before other metrics.

    This also keeps the outcome recoverable if a structural metric fails.
    Missing posterior values are never replaced with synthetic zeros.
    """
    from .metrics.information_theoretic import compute_conditional_kl_matrix
    from .metrics.latent_usage import (
        PosteriorCollapseConfig, compute_epsilon_delta_posterior_collapse,
    )

    test = splits["test"]
    kl = compute_conditional_kl_matrix(test["mus"], test["sigmas"])
    folder = Path(output_dir) / "posterior_snapshots" / run_name
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / "posterior_test.tmp.npz"
    np.savez_compressed(
        temporary, mu=test["mus"], logvar=test["sigmas"],
        sample_ids=np.asarray(test["sample_ids"], dtype=str),
    )
    temporary.replace(folder / "posterior_test.npz")
    (folder / "provenance.json").write_text(json.dumps({
        "run_name": run_name, "split": "test",
        "mu_source": test["mus_source"],
        "logvar_source": test["sigmas_source"],
        "verification": verification,
        "mu_abs_max": float(np.max(np.abs(test["mus"]))),
        "logvar_min": float(np.min(test["sigmas"])),
        "logvar_max": float(np.max(test["sigmas"])),
        "conditional_kl_mean": float(np.mean(kl)),
        "conditional_kl_max": float(np.max(kl)),
    }, indent=2))
    compute_epsilon_delta_posterior_collapse(
        kl, run_name, PosteriorCollapseConfig(), folder,
    ).to_csv(folder / "collapse_metrics.csv", index=False)
    return folder


def build_core_metric_inputs(
    run_name: str,
    splits: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
) -> CoreMetricInputs:
    """Convert extracted VARIX splits to the core metric contract."""
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
        # AUTOENCODIX Result.sigmas stores posterior log-variance.
        logvar_train=train["sigmas"],
        logvar_val=valid["sigmas"],
        logvar_test=test["sigmas"],
        z_train=train["latentspaces"],
        z_val=valid["latentspaces"],
        z_test=test["latentspaces"],
        sample_ids_train=train["sample_ids"],
        sample_ids_val=valid["sample_ids"],
        sample_ids_test=test["sample_ids"],
        sample_metadata_test=test.get("sample_metadata"),
        metadata=metadata,
    )

def load_file(file_path: str | Path):
    """Load a CSV, NumPy, JSON, pickle, text, or log artifact."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    extension = path.suffix.lower()
    if extension == ".csv":
        return pd.read_csv(path)
    if extension == ".npy":
        return np.load(path, allow_pickle=False)
    if extension == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}
    if extension == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if extension in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            return pickle.load(handle)
    if extension in {".txt", ".log"}:
        return path.read_text(encoding="utf-8")
    raise ValueError(f"Unsupported file format: {extension}")


__all__ = ["load_file"]
