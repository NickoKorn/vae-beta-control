"""Final-model posterior snapshots and numerical epoch histories."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from autoencodix.utils._losses import VarixLoss

from beta_vae_experiments.data_pipeline import _to_numpy


class RecordedVarixLoss(VarixLoss):
    """Add diagnostics without changing the optimized loss or its gradients."""

    def forward(self, model_output, targets, **kwargs):
        loss, components = super().forward(model_output=model_output, targets=targets, **kwargs)
        with torch.no_grad():
            # The trainer sums batches and divides by sample count.
            components['reconstruction_mse_per_sample'] = (
                (model_output.reconstruction - targets).square().flatten(1).mean(1).sum()
            )
            if self.config.default_vae_loss == 'kl':
                mu, lv = model_output.latent_mean, model_output.latent_logvar
                components['kl_total_per_sample'] = .5 * (mu.square() + lv.exp() - lv - 1).sum()
        return loss, components


def save_epoch_history(result, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, dynamics in [('loss', result.losses), *list(result.sub_losses.losses())]:
        for epoch, splits in dynamics.get().items():
            for split, value in splits.items():
                value = _to_numpy(value)
                if value is None or value.size != 1:
                    raise ValueError(f'Expected scalar epoch loss: {name}, {epoch}, {split}')
                key = (int(epoch), split)
                record = records.setdefault(key, {'epoch_index': int(epoch), 'epoch': int(epoch)+1,
                                                 'split': split})
                record[name] = float(value.reshape(-1)[0])
    frame = pd.DataFrame(records.values()).sort_values(['epoch_index', 'split'])
    frame.to_csv(folder/'epoch_losses.csv', index=False)
    return frame


def save_final_posteriors(varix, result, folder):
    """Encode every split with the same final model in dataset/ID order."""
    folder = Path(folder)/'posteriors'
    folder.mkdir(parents=True, exist_ok=True)
    trainer = varix._trainer
    model = trainer.get_model()
    parameter = next(model.parameters())
    was_training = model.training
    manifest = {'source': 'fresh_final_model_eval', 'splits': {}}
    seen = set()
    rng = np.random.default_rng(31)
    model.eval()
    try:
        with torch.inference_mode(), trainer._fabric.autocast():
            for split in ('train', 'valid', 'test'):
                dataset = getattr(result.datasets, split)
                if dataset is None:
                    manifest['splits'][split] = {'status': 'absent'}
                    continue
                ids = np.asarray(dataset.sample_ids, dtype=str)
                x = _to_numpy(dataset.data)
                if len(ids) != len(x) or len(set(ids)) != len(ids) or seen.intersection(ids):
                    raise ValueError(f'Invalid or overlapping sample IDs in {split}')
                seen.update(ids)
                means, logvars = [], []
                for start in range(0, len(x), trainer._config.batch_size):
                    batch = torch.as_tensor(x[start:start+trainer._config.batch_size],
                                            device=parameter.device, dtype=parameter.dtype)
                    mu, logvar = model.encode(batch)
                    means.append(_to_numpy(mu)); logvars.append(_to_numpy(logvar))
                if not means:
                    raise ValueError(f'Empty dataset in {split}')
                mu, logvar = np.concatenate(means), np.concatenate(logvars)
                if mu.shape != logvar.shape or len(mu) != len(ids) or not (
                    np.isfinite(mu).all() and np.isfinite(logvar).all()
                ):
                    raise ValueError(f'Invalid final posterior in {split}')
                temporary = folder/f'posterior_{split}.tmp.npz'
                z_redrawn = mu + np.exp(.5 * logvar) * rng.standard_normal(mu.shape)
                np.savez_compressed(temporary, mu=mu, logvar=logvar, sample_ids=ids, z_redrawn=z_redrawn)
                parameters = pd.DataFrame({
                    **{f'mu_{j}': mu[:, j] for j in range(mu.shape[1])},
                    **{f'logvar_{j}': logvar[:, j] for j in range(mu.shape[1])},
                    **{f'z_redrawn_{j}': z_redrawn[:, j] for j in range(mu.shape[1])},
                }, index=pd.Index(ids, name='SAMPLE_ID'))
                parameters.to_parquet(folder / f'posterior_parameters_{split}.parquet')
                temporary.replace(folder/f'posterior_{split}.npz')
                manifest['splits'][split] = {'status': 'ok', 'n_samples':len(ids),
                                             'latent_dim':mu.shape[1]}
    finally:
        model.train(was_training)
    manifest['z_redrawn_seed'] = 31
    manifest['z_redrawn_is_original_training_draw'] = False
    torch.save({key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
               folder.parent / 'model_final.pt')
    (folder/'provenance.json').write_text(json.dumps(manifest, indent=2)+'\n')
    save_final_distribution_plots(folder)


def initialize_artifact_layout(output_dir, settings, runs):
    for name in ('plots', 'run_analysis', 'metric_analysis', 'posterior_snapshots',
                 'metrics_artifacts', 'absolute_core_metrics_v1'):
        (output_dir/name).mkdir(parents=True, exist_ok=True)
    config_path = output_dir/'experiment_config.json'
    if not config_path.exists():
        config_path.write_text(json.dumps(settings, default=str, indent=2)+'\n')
    grid = pd.DataFrame(runs)
    grid['beta_over_d'] = grid.beta / grid.latent_dim
    grid.to_csv(output_dir/'run_analysis'/'training_grid.csv', index=False)
    (output_dir/'run_analysis'/'README.md').write_text(
        '# Run artifacts\n\nEach run contains epoch_losses.csv and posteriors/posterior_{train,valid,test}.npz.\n'
        'Posteriors are freshly encoded with the final model in eval mode; IDs follow dataset order.\n'
        'Original AUTOENCODIX loss/recon_loss/var_loss retain their native aggregation; var_loss is beta-weighted.\n'
        'kl_total_per_sample and reconstruction_mse_per_sample use sample-weighted epoch averages, including beta=0.\n'
        'Train epoch diagnostics follow changing model weights; final posterior exports use fixed final weights.\n'
        'epoch_index is zero-based; epoch is one-based. Test epoch_index=-1 is a final evaluation, not a training epoch.\n'
    )


def verify_split_manifest(datasets, output_dir):
    """Reject resume on changed membership or order before training any run."""
    split = {name: list(map(str, getattr(datasets, name).sample_ids))
             for name in ('train', 'valid', 'test') if getattr(datasets, name) is not None}
    ids = [sample for values in split.values() for sample in values]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate or overlapping split sample IDs')
    path = Path(output_dir)/'split.json'
    if path.exists():
        if json.loads(path.read_text()) != split:
            raise ValueError('Split IDs/order differ from the saved experiment; refusing resume')
    else:
        path.write_text(json.dumps(split, indent=2)+'\n')


def save_final_distribution_plots(posterior_folder):
    """Plot empirical means and posterior standard deviations on the final TEST split."""
    import matplotlib.pyplot as plt
    folder = Path(posterior_folder)
    path = folder / 'posterior_test.npz'
    if not path.exists():
        return
    with np.load(path) as arrays:
        mu, logvar = arrays['mu'], arrays['logvar']
    standard_deviation = np.exp(.5 * logvar)
    rows = []
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for dimension in range(mu.shape[1]):
        axes[0].hist(mu[:, dimension], bins=35, histtype='step', label=f'dim {dimension}', density=True)
        axes[1].hist(standard_deviation[:, dimension], bins=35, histtype='step', density=True)
        rows.append(dict(neuron_id=dimension, var_mu=float(mu[:, dimension].var()),
                         mean_mu=float(mu[:, dimension].mean()),
                         mean_posterior_std=float(standard_deviation[:, dimension].mean())))
    axes[0].set(xlabel='Posterior mean μ', ylabel='Empirical density')
    axes[1].set(xlabel='Posterior standard deviation σ', ylabel='Empirical density')
    figure.suptitle('Final posterior parameters · test split · one curve per dimension')
    figure.tight_layout()
    figure.savefig(folder / 'final_posterior_distributions.png', dpi=170, bbox_inches='tight')
    plt.close(figure)
    pd.DataFrame(rows).to_parquet(folder / 'posterior_dimension_summary.parquet', index=False)
