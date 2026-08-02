"""Experiment-local persistent checkpoint support for the VARIX grid."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from autoencodix.trainers._general_trainer import GeneralTrainer


class ResumableExperimentTrainer(GeneralTrainer):
    """GeneralTrainer-compatible variant with lossless snapshots and resume support."""

    checkpoint_path: Path | None = None
    checkpoint_run_name: str | None = None
    checkpoint_format_version = 1

    @classmethod
    def configure_checkpoint(cls, path: Path, run_name: str) -> None:
        """Set the checkpoint target for the next sequential experiment run."""
        cls.checkpoint_path = Path(path)
        cls.checkpoint_run_name = run_name

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._resume_epoch = -1
        self._recomputing_completed_snapshot = False
        super().__init__(*args, **kwargs)
        self._restore_persistent_checkpoint()

    def _init_buffers(self, *args: Any, **kwargs: Any) -> None:
        super()._init_buffers(*args, **kwargs)
        self._observed_posterior_buffers = {
            split: {"mu": False, "logvar": False}
            for split in ("train", "valid", "test")
        }

    def _capture_dynamics(
        self,
        model_output: Any,
        split: str,
        indices: torch.Tensor,
        sample_ids: Any,
        **kwargs: Any,
    ) -> None:
        """Capture dynamics exactly as GeneralTrainer and record posterior presence."""
        super()._capture_dynamics(
            model_output,
            split,
            indices,
            sample_ids,
            **kwargs,
        )
        if model_output.latent_mean is not None:
            self._observed_posterior_buffers[split]["mu"] = True
        if model_output.latent_logvar is not None:
            self._observed_posterior_buffers[split]["logvar"] = True

    def _dynamics_to_result(self, epoch: int, split: str) -> None:
        """Store observed buffers without treating a zero sum as missing data."""
        self._result.latentspaces.add(
            epoch=epoch,
            split=split,
            data=self._latentspace_buffer[split].cpu().detach().numpy(),
        )
        self._result.reconstructions.add(
            epoch=epoch,
            split=split,
            data=self._reconstruction_buffer[split].cpu().detach().numpy(),
        )
        self._result.sample_ids.add(
            epoch=epoch,
            split=split,
            data=self._sample_ids_buffer[split],
        )
        posterior_targets = (
            ("mu", self._mu_buffer, self._result.mus),
            ("logvar", self._sigma_buffer, self._result.sigmas),
        )
        for name, buffer, target in posterior_targets:
            if buffer[split] is None:
                continue
            if not self._observed_posterior_buffers[split][name]:
                continue
            array = buffer[split].cpu().detach().numpy()
            target.add(epoch=epoch, split=split, data=array)

        provenance = getattr(self._result, "posterior_snapshot_provenance", {})
        source = (
            "recomputed_from_checkpoint"
            if self._recomputing_completed_snapshot
            else "observed"
        )
        provenance[split] = {
            name: {
                "source": source,
                "all_zero": bool(torch.count_nonzero(buffer[split]).item() == 0),
            }
            for name, buffer, _target in posterior_targets
            if buffer[split] is not None
            and self._observed_posterior_buffers[split][name]
        }
        self._result.posterior_snapshot_provenance = provenance

    def train(self, epochs_overwrite: int | None = None):
        """Train from the first epoch not present in the persistent checkpoint."""
        epochs = epochs_overwrite or self._config.epochs
        if self._resume_epoch >= epochs - 1:
            self._fabric.print(
                f"Training already complete through epoch {self._resume_epoch + 1}."
            )
            self._capture_completed_run_dynamics(self._resume_epoch)
        with self._fabric.autocast():
            for epoch in range(self._resume_epoch + 1, epochs):
                self._init_buffers()
                should_checkpoint = self._should_checkpoint(epoch)
                self._model.train()

                epoch_loss, epoch_sub_losses = self._train_epoch(
                    should_checkpoint=should_checkpoint,
                    epoch=epoch,
                )
                self._log_losses(epoch, "train", epoch_loss, epoch_sub_losses)

                if self._validset:
                    valid_loss, valid_sub_losses = self._validate_epoch(
                        should_checkpoint=should_checkpoint,
                        epoch=epoch,
                    )
                    self._log_losses(epoch, "valid", valid_loss, valid_sub_losses)

                if should_checkpoint:
                    self._store_checkpoint(epoch)
                else:
                    self._save_persistent_checkpoint(epoch)

        self._result.model = next(self._model.children())
        return self._result

    def _capture_completed_run_dynamics(self, epoch: int) -> None:
        """Recreate final train/valid snapshots omitted from persistent checkpoints."""
        self._init_buffers()
        self._model.eval()
        loaders = (("train", self._trainloader), ("valid", self._validloader))
        self._recomputing_completed_snapshot = True
        try:
            with torch.no_grad(), self._fabric.autocast():
                for split, loader in loaders:
                    if loader is None:
                        continue
                    for indices, features, sample_ids in loader:
                        model_outputs = self._model(self.maskix_hook(features))
                        self._capture_dynamics(
                            model_outputs,
                            split,
                            indices,
                            sample_ids,
                        )
                    self._dynamics_to_result(epoch, split)
        finally:
            self._recomputing_completed_snapshot = False

    def _store_checkpoint(self, epoch: int) -> None:
        super()._store_checkpoint(epoch)
        self._save_persistent_checkpoint(epoch)

    def _save_persistent_checkpoint(self, epoch: int) -> None:
        path = type(self).checkpoint_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": self.checkpoint_format_version,
            "run_name": type(self).checkpoint_run_name,
            "completed_epoch": epoch,
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "losses": self._result.losses._data,
            "sub_losses": self._result.sub_losses._losses,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "mps_random_state": (
                torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
            ),
            "cuda_random_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        }
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary_path)
        temporary_path.replace(path)
        self._fabric.print(
            f"Saved resumable checkpoint after epoch {epoch + 1}: {path}"
        )

    def _restore_persistent_checkpoint(self) -> None:
        path = type(self).checkpoint_path
        if path is None or not path.exists():
            return
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format_version") != self.checkpoint_format_version:
            raise ValueError(f"Unsupported training checkpoint format: {path}")
        if payload.get("run_name") != type(self).checkpoint_run_name:
            raise ValueError(f"Training checkpoint belongs to another run: {path}")

        self._model.load_state_dict(payload["model_state_dict"])
        self._optimizer.load_state_dict(payload["optimizer_state_dict"])
        self._result.losses._data = payload.get("losses", {})
        self._result.sub_losses._losses = payload.get("sub_losses", {})
        random.setstate(payload["python_random_state"])
        np.random.set_state(payload["numpy_random_state"])
        torch.set_rng_state(payload["torch_random_state"])
        if (
            payload.get("mps_random_state") is not None
            and torch.backends.mps.is_available()
        ):
            torch.mps.set_rng_state(payload["mps_random_state"])
        if (
            payload.get("cuda_random_state_all") is not None
            and torch.cuda.is_available()
        ):
            torch.cuda.set_rng_state_all(payload["cuda_random_state_all"])

        self._resume_epoch = int(payload["completed_epoch"])
        self._result.model_checkpoints.add(
            epoch=self._resume_epoch,
            data=payload["model_state_dict"],
        )
        self._fabric.print(
            f"Resuming training at epoch {self._resume_epoch + 2} from {path}"
        )
