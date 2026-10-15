"""Expand the reviewable suite JSON into native runner configs and beta previews.

This command NEVER starts training. Execute exported configs with the existing
new_beta_varix_experiments runner after data/preflight and resource review.
"""
from beta_vae_experiments.user_interface.config import load_config
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from beta_vae_experiments.beta_annealing import schedule_from_settings


def prepare(suite_path, output_dir):
    suite = load_config(suite_path)
    output_dir = Path(output_dir)
    configs = output_dir / 'configs'
    configs.mkdir(parents=True, exist_ok=True)
    overview = []
    figure, axes = plt.subplots(len(suite['datasets']), 1, figsize=(14, 3.2 * len(suite['datasets'])))
    for axis, dataset in zip(np.atleast_1d(axes), suite['datasets']):
        for kind in suite['schedule_kinds']:
            schedule_settings = dict(suite['schedule'], kind=kind, beta_low_over_d=dataset['beta_low_over_d'])
            name = f"{suite['experiment_name']}_{dataset['name']}_{kind}"
            high = max(dataset['beta_high_over_d'])
            schedule = schedule_from_settings(schedule_settings, high)
            axis.plot(np.arange(schedule.epochs) + 1, high * schedule.factors(), label=kind)
            settings = {
                'schema_version': 2, 'experiment_name': name, 'results_date': suite['results_date'],
                'output_dir': str((output_dir / name).resolve()),
                **({'results_root': suite['results_root']} if suite.get('results_root') else {}),
                **({'require_results_mount': suite['require_results_mount']} if suite.get('require_results_mount') else {}),
                'dataset': {key: dataset[key] for key in ['id', 'split_seed', 'options']},
                'training': {
                    'beta_over_d': dataset['beta_high_over_d'], 'learning_rates': suite['learning_rates'],
                    'latent_dims': suite['latent_dims'], 'seeds': suite['seeds'], 'epochs': schedule.epochs,
                    'losses': ['kl'], 'loss_reductions': ['mean'], 'device': suite['device'],
                    'checkpoint_interval': 100, 'anneal_function': '3phase-linear',
                    'beta_schedule': schedule_settings, 'snapshots': suite['snapshots'],
                    'downstream_targets': suite['analysis']['auc_targets'] if dataset['id'] == 'tcga' else [],
                },
                'metrics': {'absolute_core': {'enabled': True},
                            'topology': {'max_samples': 512},
                            'graph': {'k_values': suite['analysis'].get('k_values', [5, 10, 20]), 'max_path_nodes': 1000, 'export_graphml': False}},
                'plots': suite.get('plots', {'enabled': True, 'per_run': True}), 'metric_analysis': {'enabled': True},
                'execution': {'resume_existing': True, 'rerun_passed': False},
                'recovery_analysis': suite['analysis'],
            }
            # The two constant-low amplitudes describe the same trajectory: export once.
            if kind == 'constant_low':
                settings['training']['beta_over_d'] = [high]
            (configs / f'{name}.json').write_text(json.dumps(settings, indent=2) + '\n')
            for amplitude in settings['training']['beta_over_d']:
                actual = schedule_from_settings(schedule_settings, amplitude)
                frame = pd.DataFrame({'epoch': np.arange(actual.epochs) + 1,
                                      'beta_over_d': amplitude * actual.factors()})
                frame.to_csv(output_dir / f'{dataset["name"]}_{kind}_high{amplitude:g}.csv', index=False)
            runs = len(settings['training']['beta_over_d']) * len(suite['seeds']) * len(suite['latent_dims']) * len(suite['learning_rates'])
            overview.append(dict(dataset=dataset['name'], schedule=kind, runs=runs, epochs=schedule.epochs,
                                 training_epochs_total=runs * schedule.epochs, config=f'{name}.json'))
        axis.set(title=dataset['name'], xlabel='Epoch', ylabel='β/d')
        axis.set_yscale('symlog', linthresh=dataset['beta_low_over_d'])
        axis.legend(ncol=5, frameon=False)
        axis.grid(alpha=.2)
    figure.tight_layout()
    figure.savefig(output_dir / 'beta_schedule_preview.png', dpi=150)
    plt.close(figure)
    overview = pd.DataFrame(overview)
    overview.to_csv(output_dir / 'suite_inventory.csv', index=False)
    (output_dir / 'suite.json').write_text(json.dumps(suite, indent=2) + '\n')
    return overview


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.suite, args.output_dir)
    print(result.to_string(index=False))
    print('Runs:', result.runs.sum(), '| total epochs:', result.training_epochs_total.sum())
