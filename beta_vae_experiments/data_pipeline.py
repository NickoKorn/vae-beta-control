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


def to_numpy(value: Any) -> np.ndarray | None:
    """Convert arrays, tensors, sparse matrices or a single-array mapping to NumPy."""
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
        return to_numpy(next(iter(value.values())))
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
            "x": to_numpy(dataset.data),
            "sample_metadata": getattr(dataset, "metadata", None),
            "sample_ids": to_numpy(
                result.sample_ids.get(split=split, epoch=epoch)
            ),
        }

        for name in ("reconstructions", "mus", "sigmas", "latentspaces"):
            dynamics = getattr(result, name)
            value = to_numpy(dynamics.get(split=split, epoch=epoch))
            source = "result"
            missing = value is None or value.size == 0
            #Missing posterior
            if missing and epoch == -1 and split == "test" and name in buffer_names:
                buffer = getattr(trainer, buffer_names[name], None)
                buffered = buffer.get("test") if isinstance(buffer, dict) else None
                candidate = to_numpy(buffered)
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


def save_posteriors(splits: dict[str, dict[str, Any]], output_dir: Path,
                        run_name: str, varix: Any) -> Path:
    """Save final Train/Valid/Test posteriors once, plus test collapse diagnostics.

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
    # Test is already final-model inference; encode Train/Valid with that same model.
    trainer = varix._trainer
    model = trainer.get_model()
    parameter = next(model.parameters())
    was_training = model.training
    try:
        model.eval()
        with torch.inference_mode(), trainer._fabric.autocast():
            for split, values in splits.items():
                ids = np.asarray(values['sample_ids'], dtype=str)
                mu, logvar = values['mus'], values['sigmas']
                if split != 'test':
                    batches = []
                    for start in range(0, len(values['x']), trainer._config.batch_size):
                        batch = torch.as_tensor(values['x'][start:start + trainer._config.batch_size],
                                                device=parameter.device, dtype=parameter.dtype)
                        batches.append(tuple(to_numpy(v) for v in model.encode(batch)))
                    mu, logvar = (np.concatenate(parts) for parts in zip(*batches))
                if mu.shape != logvar.shape or len(mu) != len(ids) or not (
                        np.isfinite(mu).all() and np.isfinite(logvar).all()):
                    raise ValueError(f'Invalid final posterior: {split}')
                temporary = folder / f'posterior_{split}.tmp.npz'
                np.savez_compressed(temporary, mu=mu, logvar=logvar, sample_ids=ids)
                temporary.replace(folder / f'posterior_{split}.npz')
    finally:
        model.train(was_training)
    (folder / "provenance.json").write_text(json.dumps({
        "run_name": run_name, "splits": list(splits),
        "train_valid_source": "fresh_final_model_eval", "diagnostic_split": "test",
        "mu_source": test["mus_source"],
        "logvar_source": test["sigmas_source"],
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
        sample_metadata_train=train.get("sample_metadata"),
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


def save_epoch_snapshots(varix, run_folder, every=50, downstream=True, beta_schedule=None):
    """Save recorded Train/Validation arrays every N epochs and at the final epoch.

    Train arrays come from batch updates; Validation arrays from native eval mode.
    No model is re-encoded, and no historical model weights are exported.
    """
    from .metrics.evaluation.downstream import evaluate_epoch_downstream
    if type(every) is not int or every < 1:
        raise ValueError("Snapshot interval must be a positive integer")
    result, config = varix.result, varix.config
    folder = Path(run_folder) / "epoch_snapshots"
    folder.mkdir(parents=True, exist_ok=True)
    metadata = varix.visualizer._collect_all_metadata(result=result)
    metadata.index = metadata.index.astype(str)
    factors = None
    if beta_schedule:
        from .beta_annealing import schedule_from_settings
        factors = schedule_from_settings(beta_schedule, config.beta / config.latent_dim).factors()
    records = []
    for epoch in result.latentspaces.epochs():
        if epoch < 0 or ((epoch + 1) % every and epoch != config.epochs - 1):
            continue
        posteriors, labels = {}, {}
        for split in ("train", "valid"):
            path = folder / f"epoch_{epoch+1:05d}_{split}.npz"
            arrays = {}
            if path.exists():
                with np.load(path) as saved:
                    arrays = {key: saved[key] for key in saved.files}
            for key, name in (("mu", "mus"), ("logvar", "sigmas"), ("z", "latentspaces"),
                              ("sample_ids", "sample_ids")):
                value = to_numpy(getattr(result, name).get(epoch=epoch, split=split))
                if key not in arrays and value is not None and value.size:
                    arrays[key] = value.astype(str) if key == "sample_ids" else value
            if "z" not in arrays:
                continue
            path = folder / f"epoch_{epoch+1:05d}_{split}.npz"
            np.savez_compressed(path, **arrays)
            missing = sorted({"mu", "logvar", "z", "sample_ids"} - arrays.keys())
            records.append(dict(epoch=epoch+1, split=split, file=path.name,
                                status="missing_arrays" if missing else "ok", missing=",".join(missing)))
            if not missing:
                posteriors[split] = arrays
                labels[split] = metadata.loc[arrays["sample_ids"]].copy()
                label_path = folder / f"labels_{split}.parquet"
                if not label_path.exists():
                    labels[split].to_parquet(label_path)
        test_path = folder / f"epoch_{epoch+1:05d}_test.npz"
        if test_path.exists():
            with np.load(test_path) as saved:
                test_ids = saved["sample_ids"]
            metadata.loc[test_ids].to_parquet(folder / "labels_test.parquet")
            records.append(dict(epoch=epoch+1, split="test", file=test_path.name, status="ok", missing=""))
        if downstream:
            if set(posteriors) != {"train", "valid"}:
                print(f"Snapshot epoch {epoch+1}: native posterior arrays missing; downstream unavailable")
                continue
            realized_schedule = Path(run_folder) / "beta_schedule.csv"
            if realized_schedule.exists():
                realized = pd.read_csv(realized_schedule).set_index("epoch")
                effective_beta = float(realized.loc[epoch+1, "beta_config"])
                evaluate_epoch_downstream(folder, epoch+1, posteriors, labels, effective_beta)
                continue
            factor = 1.
            if factors is not None:
                factor = factors[epoch]
            elif config.anneal_function != "no-annealing":
                from autoencodix.utils._annealer import AnnealingScheduler
                factor = AnnealingScheduler().get_weight(epoch_current=epoch,
                    total_epoch=config.epochs, func=config.anneal_function)
            evaluate_epoch_downstream(folder, epoch+1, posteriors, labels, config.beta * factor)
    pd.DataFrame(records, columns=["epoch", "split", "file", "status", "missing"]).to_csv(
        folder / "snapshot_index.csv", index=False)
    (folder / "provenance.json").write_text(json.dumps(dict(
        source="checkpoint_inference" if (folder / "labels_test.parquet").exists() else "autoencodix_training_dynamics", every=every,
        train="checkpoint eval inference" if (folder / "labels_test.parquet").exists() else "captured during batch updates",
        valid="checkpoint eval inference" if (folder / "labels_test.parquet").exists() else "captured during validation",
        exported_after_training=True,
        splits=["train", "valid", "test"] if (folder / "labels_test.parquet").exists() else ["train", "valid"],
    ), indent=2) + "\n")
    return folder


def save_preprocessed_splits(splits, output_dir):
    """Persist exact model inputs, feature IDs and labels once per dataset/split."""
    import fcntl
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".export.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for split in ("train", "valid", "test"):
            dataset = getattr(splits, split)
            if dataset is None:
                continue
            arrays = dict(x=to_numpy(dataset.data),
                          sample_ids=np.asarray(dataset.sample_ids, dtype=str),
                          feature_ids=np.asarray(dataset.feature_ids, dtype=str))
            path = directory / f"input_{split}.npz"
            if path.exists():
                with np.load(path) as saved:
                    if any(not np.array_equal(saved[key], value) for key, value in arrays.items()):
                        raise ValueError(f"Preprocessed data differ from saved export: {path}")
            else:
                temporary = path.with_suffix(".tmp.npz")
                np.savez_compressed(temporary, **arrays)
                temporary.replace(path)
            metadata = getattr(dataset, "metadata", None)
            if isinstance(metadata, dict):
                metadata = metadata.get("paired")
            if isinstance(metadata, pd.DataFrame):
                metadata = metadata.copy()
                metadata.index = metadata.index.astype(str)
                metadata.loc[arrays["sample_ids"]].to_parquet(directory / f"labels_{split}.parquet")


def make_snapshot_trainer(splits, run_folder):
    """Export same-checkpoint posteriors without altering the native training steps.

    AUTOENCODIX records no historical test inference, and state_dict checkpoints
    are not deep-copied. Encode at the checkpoint itself rather than reconstruct
    historical test arrays from final weights. No labels or test scores are used.
    """
    from autoencodix.trainers._general_trainer import GeneralTrainer

    class PosteriorSnapshotTrainer(GeneralTrainer):
        def _store_checkpoint(self, epoch):
            super()._store_checkpoint(epoch)
            folder = Path(run_folder) / "epoch_snapshots"
            folder.mkdir(parents=True, exist_ok=True)
            model = self.get_model()
            parameter = next(model.parameters())
            was_training = model.training
            devices = [parameter.device.index] if parameter.is_cuda else []
            try:
                model.eval()
                with torch.random.fork_rng(devices=devices), torch.inference_mode(), self._fabric.autocast():
                    for split_index, split in enumerate(("train", "valid", "test")):
                        dataset = getattr(splits, split)
                        if dataset is None:
                            continue
                        x = to_numpy(dataset.data)
                        batches = []
                        for start in range(0, len(x), self._config.batch_size):
                            batch = torch.as_tensor(x[start:start+self._config.batch_size],
                                                    device=parameter.device, dtype=parameter.dtype)
                            batches.append(tuple(to_numpy(value) for value in model.encode(batch)))
                        mu, logvar = (np.concatenate(parts) for parts in zip(*batches))
                        if not (np.isfinite(mu).all() and np.isfinite(logvar).all()):
                            raise ValueError(f"Nonfinite snapshot posterior: {split}, epoch {epoch+1}")
                        rng = np.random.default_rng(np.random.SeedSequence([31, epoch, split_index]))
                        z = mu + np.exp(.5 * logvar) * rng.standard_normal(mu.shape)
                        path = folder / f"epoch_{epoch+1:05d}_{split}.npz"
                        temporary = path.with_suffix(".tmp.npz")
                        np.savez_compressed(temporary, mu=mu, logvar=logvar, z=z,
                                            sample_ids=np.asarray(dataset.sample_ids, dtype=str))
                        temporary.replace(path)
            finally:
                model.train(was_training)
    return PosteriorSnapshotTrainer
