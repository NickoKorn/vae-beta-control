"""Prepare the single-annealer experiment grid without starting training."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .beta_annealing import schedule_from_settings
from .user_interface.config import CATALOG, load_config


def prepare(suite_reference, output_dir):
    suite = load_config(suite_reference)
    seeds = suite['split_seeds']
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or seed < 0 for seed in seeds)):
        raise ValueError('split_seeds must be distinct nonnegative integers')
    if suite['analysis']['held_out_confirmation']['split_seed'] in seeds:
        raise ValueError('Development splits must exclude the confirmation split')
    directory = Path(output_dir).resolve()
    configs = directory / 'configs'
    if configs.exists() and any(configs.glob('*.json')):
        raise FileExistsError('Choose a fresh design directory; existing configs are not overwritten')
    configs.mkdir(parents=True, exist_ok=True)
    rows = []
    project = Path(__file__).resolve().parent.parent
    for dataset in suite['datasets']:
        schedule = schedule_from_settings(suite['schedule'], max(dataset['beta_over_d']))
        for split_seed in seeds:
            name = f"{suite['experiment_name']}_{dataset['name']}_split{split_seed}"
            options = copy.deepcopy(dataset['options'])
            if dataset['id'] == 'tcga':
                options['data_root'] = str((project / options['data_root']).resolve())
            config = dict(
                schema_version=2, experiment_name=name, results_date=suite['results_date'],
                results_root=str((project / suite['results_root']).resolve()),
                **({'require_results_mount': suite['require_results_mount']}
                   if suite.get('require_results_mount') else {}),
                output_dir=str(directory / name),
                dataset=dict(id=dataset['id'], split_seed=split_seed, options=options),
                training=dict(beta_over_d=dataset['beta_over_d'], learning_rates=suite['learning_rates'],
                    latent_dims=suite['latent_dims'], seeds=suite['seeds'], epochs=schedule.epochs,
                    losses=['kl'], loss_reductions=['mean'], device=suite['device'],
                    checkpoint_interval=100, anneal_function='3phase-linear',
                    beta_schedule=suite['schedule'], snapshots=suite['snapshots'],
                    downstream_targets=suite['analysis']['auc_targets'] if dataset['id'] == 'tcga' else []),
                metrics=dict(absolute_core=dict(enabled=True), topology=dict(max_samples=1000),
                             graph=dict(k_values=suite['analysis']['k_values'])),
                plots=suite['plots'], metric_analysis=dict(enabled=False),
                execution=dict(resume_existing=True, rerun_passed=False),
            )
            (configs / f'{name}.json').write_text(json.dumps(config, indent=2) + '\n')
            runs = (len(dataset['beta_over_d']) * len(suite['latent_dims'])
                    * len(suite['seeds']) * len(suite['learning_rates']))
            rows.append(dict(dataset=dataset['name'], split_seed=split_seed, runs=runs,
                             epochs=schedule.epochs, warmup_epochs=schedule.warmup_epochs,
                             training_epochs_total=runs * schedule.epochs, config=f'{name}.json'))
    inventory = pd.DataFrame(rows)
    inventory.to_csv(directory / 'suite_inventory.csv', index=False)
    pd.DataFrame(dict(epoch=np.arange(schedule.epochs) + 1, factor=schedule.factors())).to_csv(
        directory / 'beta_schedule.csv', index=False)
    (directory / 'suite.json').write_text(json.dumps(suite, indent=2) + '\n')
    return inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite', nargs='?', default=str(CATALOG) + '#annealing.warmup')
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parent / 'new_results_2026-10-22' / 'annealing_warmup_design')
    args = parser.parse_args()
    inventory = prepare(args.suite, args.output_dir)
    print(inventory.to_string(index=False))
    print(f'Runs: {inventory.runs.sum()} | total epochs: {inventory.training_epochs_total.sum()}')


if __name__ == '__main__':
    main()
