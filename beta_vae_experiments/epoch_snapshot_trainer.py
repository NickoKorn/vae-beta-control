"""Frozen-model posterior snapshots using AUTOENCODIX's trainer extension point."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import torch
from autoencodix.trainers._general_trainer import GeneralTrainer


def make_epoch_snapshot_trainer(folder, every=50, save_models=False, observer=None, beta_schedule=None):
    """Store all train/validation posteriors; test remains for final evaluation.

    Runs after native validation, independent of checkpoint_interval. torch RNG
    state and model train/eval mode are restored, so observation does not alter
    stochastic training. Snapshots use one frozen model for every sample.
    """
    if not isinstance(every, int) or every < 1:
        raise ValueError('snapshot interval must be a positive integer')
    folder = Path(folder) / 'epoch_snapshots'
    from .annealing_observer import ObserverConfig, aligned_annotations, observe_snapshot
    observer_config = ObserverConfig(**(observer or {}))

    class PosteriorSnapshotTrainer(GeneralTrainer):
        def _validate_epoch(self, should_checkpoint, epoch):
            result = super()._validate_epoch(should_checkpoint, epoch)
            if epoch == 0 or (epoch + 1) % every == 0 or epoch + 1 == self._config.epochs:
                self._save_epoch_posteriors(epoch)
            return result

        def _save_epoch_posteriors(self, epoch):
            folder.mkdir(parents=True, exist_ok=True)
            model = self.get_model()
            parameter = next(model.parameters())
            was_training = model.training
            cuda_devices = [parameter.device.index] if parameter.is_cuda else []
            try:
                with torch.random.fork_rng(devices=cuda_devices), torch.inference_mode(), self._fabric.autocast():
                    model.eval()
                    for split, dataset in [('train', self._trainset), ('valid', self._validset)]:
                        ids = np.asarray(dataset.sample_ids, dtype=str)
                        x = np.asarray(dataset.data)
                        if len(ids) != len(set(ids)) or len(x) != len(ids):
                            raise ValueError('Invalid snapshot input IDs')
                        reference = folder / f'input_{split}.npz'
                        label_path = folder / f'labels_{split}.parquet'
                        if not label_path.exists():
                            aligned_annotations(dataset.metadata, ids).to_parquet(label_path)
                        if not reference.exists():
                            np.savez_compressed(reference, x=x, sample_ids=ids)
                        else:
                            with np.load(reference) as saved:
                                if not np.array_equal(saved['sample_ids'], ids):
                                    raise ValueError('Snapshot sample order changed')
                        means, logvars = [], []
                        for start in range(0, len(ids), self._config.batch_size):
                            batch = torch.as_tensor(x[start:start + self._config.batch_size],
                                                    device=parameter.device, dtype=parameter.dtype)
                            mu, logvar = model.encode(batch)
                            means.append(mu.detach().cpu().numpy())
                            logvars.append(logvar.detach().cpu().numpy())
                        mu, logvar = np.concatenate(means), np.concatenate(logvars)
                        if mu.shape != logvar.shape or len(mu) != len(ids) or not np.isfinite(mu).all() or not np.isfinite(logvar).all():
                            raise ValueError('Invalid posterior snapshot')
                        temporary = folder / f'epoch_{epoch+1:05d}_{split}.tmp.npz'
                        np.savez_compressed(temporary, mu=mu, logvar=logvar, sample_ids=ids)
                        temporary.replace(folder / f'epoch_{epoch+1:05d}_{split}.npz')
                    if save_models:
                        torch.save({key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                                   folder / f'epoch_{epoch+1:05d}_model.pt')
            finally:
                model.train(was_training)
            (folder / 'provenance.json').write_text(json.dumps({
                'source': 'frozen_model_after_validation', 'epoch_index_zero_based': epoch,
                'epoch': epoch + 1, 'every': every, 'splits': ['train', 'valid'],
                'test_during_training': False, 'model_states_saved': save_models,
            }, indent=2) + '\n')
            if observer_config.enabled:
                if beta_schedule:
                    from .beta_annealing import schedule_from_settings
                    factor = schedule_from_settings(beta_schedule,
                        self._config.beta / self._config.latent_dim).factors()[epoch]
                else:
                    from autoencodix.utils._annealer import AnnealingScheduler
                    factor = AnnealingScheduler().get_weight(epoch_current=epoch,
                        total_epoch=self._config.epochs, func=self._config.anneal_function)
                observe_snapshot(folder, epoch+1, vars(observer_config), self._config.beta * factor)

    return PosteriorSnapshotTrainer
