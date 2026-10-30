"""Offline rate/preservation analysis of the 106 saved fixed-beta models.

Uses the existing ZADU adapter; no training, label-based tuning or AUTOENCODIX
modification. Cache identity includes posterior/input hashes and adapter source.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from tqdm.auto import tqdm


SERIES = 'fixed_beta_106_epochs750_2026_10_27'
PRESERVATION = ('trustworthiness', 'continuity', 'lcmc', 'mrre_false', 'mrre_missing',
                'steadiness', 'cohesiveness', 'scale_normalized_stress', 'spearman_rho')


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _analyse_saved_run(arguments):
    job, series, output, max_samples, k, seed = arguments
    from .metrics.preservation import compute_zadu_geometry
    import inspect
    series, output = Path(series), Path(output)
    config_path = Path(job['config'])
    settings = json.loads(config_path.read_text())
    folder = Path(job['output'])
    overview = pd.read_csv(folder / 'experiment_metrics.csv')
    if len(overview) != 1 or not overview.status.eq('ok').all():
        raise ValueError(f'Unexpected run status: {job["name"]}')
    row = overview.iloc[0].to_dict()
    run_folder = folder / 'run_analysis' / row['run_name']
    posterior_path = folder / 'posterior_snapshots' / row['run_name'] / 'posterior_test.npz'
    input_path = Path(settings['preprocessed_data_dir']) / 'input_test.npz'
    cache = output / 'per_run' / (job['name'] + '.json')
    identity = dict(posterior_sha256=_digest(posterior_path), input_sha256=_digest(input_path),
                    adapter_sha256=_digest(inspect.getfile(compute_zadu_geometry)),
                    max_samples=max_samples, k=k, seed=seed)
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved['identity'] == identity:
            return {key.replace('_mse_native', '_loss_native'): value for key, value in saved['row'].items()}
    with np.load(posterior_path) as arrays, np.load(input_path) as inputs:
        mu, logvar, ids = arrays['mu'], arrays['logvar'], arrays['sample_ids'].astype(str)
        if not np.array_equal(ids, inputs['sample_ids'].astype(str)):
            raise ValueError(f'Input/posterior sample order differs: {job["name"]}')
        x = inputs['x']
    if mu.shape != logvar.shape or len(mu) != len(x):
        raise ValueError('Posterior/input shapes do not align')
    mu, logvar = mu.astype(np.float64), logvar.astype(np.float64)
    if not (np.isfinite(mu).all() and np.isfinite(logvar).all()):
        raise ValueError('Posterior contains nonfinite values')
    kl = .5 * (mu**2 + np.expm1(logvar) - logvar)
    collapsed = (kl < .01).mean(axis=0) >= .99
    indices = np.arange(len(x)) if len(x) <= max_samples else np.sort(
        np.random.default_rng(seed).choice(len(x), max_samples, replace=False))
    result = compute_zadu_geometry(x[indices], mu[indices], k,
                                  sample_ids=ids[indices], random_state=seed)
    row.update(job_name=job['name'], dataset=settings['dataset']['id'],
               split_seed=settings['dataset']['split_seed'], n_test=len(x), n_structure=len(indices),
               beta_over_d=row['beta']/row['latent_dim'], rate=kl.sum(axis=1).mean(),
               rate_per_dim=kl.mean(), active_units=int((mu.var(axis=0, ddof=1) > .01).sum()),
               collapsed_dimensions=int(collapsed.sum()), collapsed_fraction=collapsed.mean(),
               full_collapse=bool(collapsed.all()), k=k,
               rank_backend=result.metadata['rank_backend'],
               input_duplicate_rows=result.metadata['spaces']['x']['duplicate_rows'],
               mu_duplicate_rows=result.metadata['spaces']['mu']['duplicate_rows'],
               mu_float32_unique_fraction=result.metadata['spaces']['mu']['float32_unique_rows']/len(indices),
               **result.global_metrics)
    row['log_rate'] = float(np.log(max(row['rate'], 1e-12)))
    losses = pd.read_csv(run_folder / 'epoch_losses.csv')
    for split in ('train', 'valid'):
        last = losses[losses.split.eq(split)].sort_values('epoch').iloc[-1]
        row[f'recon_{split}_loss_native'] = float(last['recon_loss'])
    row['recon_generalization_gap'] = row['recon_valid_loss_native']-row['recon_train_loss_native']
    clean = {key: (value.item() if isinstance(value, np.generic) else value) for key, value in row.items()}
    clean = {key: (None if isinstance(value, float) and not np.isfinite(value) else value)
             for key, value in clean.items()}
    payload = dict(identity=identity, row=clean, selected_indices=indices.tolist(),
                   geometry_warnings=list(result.warnings), geometry_failures=result.failed_measures,
                   geometry_metadata=result.metadata)
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    temporary.replace(cache)
    return clean


def collect_large_scale_metrics(project_root, workers=4, max_samples=1000, k=20, seed=31):
    root = Path(project_root).resolve() / 'beta_vae_experiments'
    series = root / 'large_scale_results' / SERIES
    output = root / 'analysis_results/large_scale_fixed_beta'
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((series / 'launcher.json').read_text())
    if manifest['status'] != 'complete' or manifest['failed']:
        raise ValueError('The training series is not completely successful')
    tasks = [(job, str(series), str(output), max_samples, k, seed) for job in manifest['jobs']]
    rows = []
    if workers == 1:
        rows = [_analyse_saved_run(task) for task in tqdm(tasks, desc='Saved models', unit='model')]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_analyse_saved_run, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc='Saved models', unit='model'):
                rows.append(future.result())
    table = pd.DataFrame(rows).sort_values(['dataset', 'weight_decay', 'beta_over_d', 'seed'])
    if len(table) != 106 or table.job_name.duplicated().any():
        raise ValueError('Expected 106 distinct saved models')
    table.to_csv(output / 'run_metrics.csv', index=False)
    (output / 'provenance.json').write_text(json.dumps(dict(
        series=str(series), models=len(table), split='test', k=k, max_samples=max_samples,
        subset_seed=seed, collapsed_epsilon=.01, collapsed_delta=.01,
        reconstruction='native logged batch means summed then divided by n; not ordinary per-sample MSE or test reconstruction',
        missing_geometry='NaN/None with per-run warnings; never replaced by zero',
        groups='separate datasets and trajectories; wd=0 controls excluded from correlations',
        pending=['downstream AUC', 'multi-k/RNX', 'graph', 'TDA'],
    ), indent=2)+'\n')
    return table


def rate_correlations(table):
    """Per-seed correlations, plus slopes along log-beta; no pooled p-values."""
    metrics = [*PRESERVATION, 'rate_per_dim', 'active_units', 'collapsed_fraction',
               'recon_valid_loss_native', 'recon_generalization_gap']
    rows = []
    main = table[table.weight_decay.eq(.01)]
    for (dataset, dimension, seed), group in main.groupby(['dataset', 'latent_dim', 'seed']):
        for scope in ('all_models', 'before_full_collapse'):
            eligible = group if scope == 'all_models' else group[~group.full_collapse]
            for metric in metrics:
                for rate_name in ('rate', 'log_rate'):
                    paired = eligible.replace([np.inf, -np.inf], np.nan).dropna(subset=[metric, rate_name]).sort_values('beta_over_d')
                    for mode in ('values', 'slopes'):
                        frame = paired if mode == 'values' else paired[paired.beta_over_d.gt(0)]
                        a, b = frame[metric].to_numpy(), frame[rate_name].to_numpy()
                        if mode == 'slopes' and len(frame) > 1:
                            step = np.diff(np.log(frame.beta_over_d.to_numpy()))
                            a, b = np.diff(a)/step, np.diff(b)/step
                        valid = len(a) >= 4 and np.std(a) > 1e-12 and np.std(b) > 1e-12
                        rows.append(dict(dataset=dataset, latent_dim=dimension, seed=seed, scope=scope,
                            metric=metric, rate_reference=rate_name, mode=mode, n_pairs=len(a),
                            pearson=pearsonr(a,b).statistic if valid else np.nan,
                            spearman=spearmanr(a,b).statistic if valid else np.nan,
                            status='ok' if valid else 'insufficient_or_constant'))
    details = pd.DataFrame(rows)
    valid = details[details.status.eq('ok')]
    summary = valid.groupby(['dataset', 'scope', 'rate_reference', 'mode', 'metric']).agg(
        pearson=('pearson','median'), spearman=('spearman','median'),
        pearson_q25=('pearson',lambda s:s.quantile(.25)), pearson_q75=('pearson',lambda s:s.quantile(.75)),
        spearman_q25=('spearman',lambda s:s.quantile(.25)), spearman_q75=('spearman',lambda s:s.quantile(.75)),
        trajectories=('seed','size'), min_pairs=('n_pairs','min')).reset_index()
    return details, summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    table = collect_large_scale_metrics(root, workers=args.workers)
    details, summary = rate_correlations(table)
    output = root / 'beta_vae_experiments/analysis_results/large_scale_fixed_beta'
    details.to_csv(output / 'rate_correlations_by_trajectory.csv', index=False)
    summary.to_csv(output / 'rate_correlations_summary.csv', index=False)
    print(f'Analysed {len(table)} models; saved correlations in {output}')
