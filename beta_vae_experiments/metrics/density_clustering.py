"""Optional label-free ADP/HDBSCAN diagnostics, without changing training.

ADP: 2NN intrinsic dimension -> PAk density -> density-peak merging at Z=1.65.
https://dadapy.readthedocs.io/en/latest/clustering.html
HDBSCAN: min_cluster_size=5, min_samples=5, epsilon=0, EOM; a single cluster
is explicitly allowed. https://hdbscan.readthedocs.io/en/latest/api.html

Counts and noise describe density structure, not probabilistic posterior
intersection, biological classes or proof of collapse. Input values are used
without column standardisation, jitter or silent removal of duplicate samples.
"""
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DensityClusteringConfig:
    methods: tuple[str, ...] = ('adp', 'hdbscan')
    max_samples: int = 1000
    seed: int = 31
    adp_z: float = 1.65
    adp_maxk: int = 100
    adp_halo: bool = False
    min_cluster_size: int = 5
    min_samples: int = 5
    cluster_selection_epsilon: float = 0.
    cluster_selection_method: str = 'eom'
    allow_single_cluster: bool = True
    n_draws: int = 10
    min_cluster_sizes: tuple[int, ...] = (5, 20, 50)

    def __post_init__(self):
        object.__setattr__(self, 'methods', tuple(self.methods))
        object.__setattr__(self, 'min_cluster_sizes', tuple(self.min_cluster_sizes))
        if type(self.n_draws) is not int or self.n_draws < 1:
            raise ValueError('n_draws must be a positive integer')
        if (not self.min_cluster_sizes or len(set(self.min_cluster_sizes)) != len(self.min_cluster_sizes)
                or any(type(size) is not int or size < 2 for size in self.min_cluster_sizes)):
            raise ValueError('min_cluster_sizes must contain distinct integers >= 2')
        if not self.methods or len(set(self.methods)) != len(self.methods) or not set(self.methods) <= {'adp', 'hdbscan'}:
            raise ValueError('methods must be distinct selections from adp/hdbscan')
        for name, minimum in (('max_samples', 2), ('adp_maxk', 3), ('min_cluster_size', 2), ('min_samples', 1)):
            if type(getattr(self, name)) is not int or getattr(self, name) < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError('seed must be a nonnegative 32-bit integer')
        if not np.isfinite(self.adp_z) or self.adp_z <= 0:
            raise ValueError('adp_z must be positive and finite')
        if not np.isfinite(self.cluster_selection_epsilon) or self.cluster_selection_epsilon < 0:
            raise ValueError('cluster_selection_epsilon must be finite and nonnegative')
        if self.cluster_selection_method not in ('eom', 'leaf'):
            raise ValueError('cluster_selection_method must be eom or leaf')
        if type(self.allow_single_cluster) is not bool or type(self.adp_halo) is not bool:
            raise ValueError('allow_single_cluster/adp_halo must be boolean')


def fit_density_clusters(values, method='adp', config=None):
    """Return library labels and diagnostics, preserving every input row.

    Noise has label -1. ADP's continuous 2NN/PAk estimates require distinct
    coordinates; duplicates are reported as unavailable rather than deduplicated.
    HDBSCAN can handle duplicates. A fully constant representation is withheld
    for both methods; it is not assigned an artificial cluster count.
    """
    config = config or DensityClusteringConfig()
    if method not in ('adp', 'hdbscan'):
        raise ValueError('Unknown clustering method')
    if np.iscomplexobj(values):
        raise ValueError('Coordinates must be real')
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) == 0 or not np.isfinite(values).all():
        raise ValueError('Expected a finite nonempty samples-by-features matrix')
    unique = len(np.unique(values, axis=0))
    if unique == 1:
        raise ValueError('Constant representation: density structure is undefined')
    arrays, metadata = {}, dict(n_samples=len(values), n_unique=unique, distance='euclidean')
    if method == 'hdbscan':
        from hdbscan import HDBSCAN
        if len(values) < config.min_cluster_size or len(values) <= config.min_samples:
            raise ValueError('Too few points for the configured HDBSCAN neighborhood')
        model = HDBSCAN(min_cluster_size=config.min_cluster_size, min_samples=config.min_samples,
            cluster_selection_epsilon=config.cluster_selection_epsilon,
            cluster_selection_method=config.cluster_selection_method,
            allow_single_cluster=config.allow_single_cluster, metric='euclidean',
            core_dist_n_jobs=1, approx_min_span_tree=False)
        labels = model.fit_predict(values)
        arrays['membership_strength'] = model.probabilities_
        metadata.update(package='hdbscan', package_version=version('hdbscan'))
    else:
        if len(values) < 6 or config.adp_maxk < 3:
            raise ValueError('ADP requires at least six points and adp_maxk >= 3')
        if unique != len(values):
            raise ValueError('Duplicate coordinates: 2NN/PAk unavailable; no deduplication or jitter applied')
        from dadapy import Data
        maxk = min(config.adp_maxk, len(values)-1)
        model = Data(coordinates=values, maxk=maxk, verbose=False, n_jobs=1)
        # Explicit RNG controls possible library sampling; it is independent of VAE seeds.
        model.rng = np.random.default_rng(config.seed)
        model.compute_distances(maxk=maxk)
        model.compute_id_2NN()
        dimension = float(model.intrinsic_dim)
        if not np.isfinite(dimension) or dimension <= 0:
            raise ValueError('Invalid 2NN intrinsic dimension')
        model.compute_density_PAk()
        if not (np.isfinite(model.log_den).all() and np.isfinite(model.log_den_err).all()):
            raise ValueError('Nonfinite PAk density estimates')
        labels = model.compute_clustering_ADP(Z=config.adp_z, halo=config.adp_halo, v2=False)
        arrays.update(log_density=model.log_den, log_density_error=model.log_den_err,
                      density_k=model.kstar)
        metadata.update(package='dadapy', package_version=version('dadapy'),
                        intrinsic_dimension=dimension, effective_maxk=maxk, density='PAk', id_estimator='2NN')
    labels = np.asarray(labels, dtype=int)
    if labels.shape != (len(values),) or np.any(labels < -1):
        raise ValueError('Library returned invalid or unaligned labels')
    groups, sizes = np.unique(labels[labels >= 0], return_counts=True)
    summary = dict(n_clusters=len(groups), noise_fraction=float(np.mean(labels == -1)),
                   assigned_fraction=float(np.mean(labels >= 0)),
                   largest_cluster_fraction=float(sizes.max()/len(labels)) if len(sizes) else 0.)
    return labels, summary, arrays, metadata


def run_density_clustering_pipeline(inputs, config=None, output_dir=None, *, selected_indices=None, split='test'):
    """Run optional diagnostics on the same X/mu cohort; return repository scalar rows.

    Pass CoreMetricInputs and, optionally, its shared structural subset. Fits
    each space independently with fixed settings: cluster labels across beta
    are not identities and do not track biological classes or cluster mergers.
    Missing optional dependencies fail only the affected method.
    """
    config = config or DensityClusteringConfig()
    if split not in ('train', 'val', 'test'):
        raise ValueError('split must be train, val or test')
    x, mu = (np.asarray(getattr(inputs, prefix+'_'+split)) for prefix in ('x', 'mu'))
    ids = np.asarray(getattr(inputs, 'sample_ids_'+split))
    if x.ndim != 2 or mu.ndim != 2 or len(x) != len(mu) or ids.shape != (len(x),):
        raise ValueError('X, mu and real sample IDs must align')
    ids = ids.astype(str)
    if pd.isna(getattr(inputs, 'sample_ids_'+split)).any() or len(np.unique(ids)) != len(ids):
        raise ValueError('Sample IDs must be unique')
    if selected_indices is None:
        indices = np.arange(len(x)) if len(x) <= config.max_samples else np.sort(
            np.random.default_rng(config.seed).choice(len(x), config.max_samples, replace=False))
    else:
        indices = np.asarray(selected_indices)
        if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
                or len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= len(x))):
            raise ValueError('Selected indices must be unique valid integer row indices')
    directory = None if output_dir is None else Path(output_dir)/'metrics_artifacts'/inputs.run_name/'density_clustering'/split
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
    rows, fits = [], []
    names = ('n_clusters', 'noise_fraction', 'assigned_fraction', 'largest_cluster_fraction')
    for space, values in (('x', x[indices]), ('mu', mu[indices])):
        for method in config.methods:
            status, warning = 'ok', None
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    labels, summary, arrays, metadata = fit_density_clusters(values, method, config)
                warning = '; '.join(dict.fromkeys(str(item.message) for item in caught)) or None
                if directory is not None:
                    pd.DataFrame(dict(sample_id=ids[indices], cluster=labels, **arrays)).to_csv(
                        directory/f'{space}_{method}_patients.csv', index=False)
                fits.append(dict(space=space, method=method, status=status, warning=warning, **metadata))
            except Exception as error:
                status = 'dependency_unavailable' if isinstance(error, ImportError) else 'degenerate_input' if isinstance(error, ValueError) else 'failed'
                warning = f'{type(error).__name__}: {error}'
                summary = dict.fromkeys(names, np.nan)
                fits.append(dict(space=space, method=method, status=status, warning=warning))
            rows.extend(dict(metric_name=f'density_{space}_{method}_{name}', value=float(summary[name]),
                             status=status, warning=warning, n_samples=len(indices), split=split) for name in names)
    table = pd.DataFrame(rows)
    if directory is not None:
        table.to_csv(directory/'scalar_metrics.csv', index=False)
        (directory/'metadata.json').write_text(json.dumps(dict(config=asdict(config),
            selected_indices=indices.tolist(), sample_ids=ids[indices].tolist(), fits=fits), indent=2)+'\n')
    return table


def run_posterior_density_pipeline(mu, logvar, sample_ids, config=None, output_dir=None, *,
                                   saved_z=None, run_name='posterior', split='test'):
    """Cluster repeated z draws, with the same draw for ADP and all HDBSCAN sizes.

    Draw 0 uses saved_z when provided; other draws use stored mu/logvar. Standard
    deviations are exp(logvar/2), never replaced by one. Draw dispersion measures
    posterior sampling sensitivity, not patient-bootstrap uncertainty. There is
    no fallback to mu. X/mu-only diagnostics use the separate point-data runner.
    """
    config = config or DensityClusteringConfig()
    if split not in ('train', 'valid', 'val', 'test'):
        raise ValueError('Invalid posterior split')
    mu, logvar = np.asarray(mu, dtype=float), np.asarray(logvar, dtype=float)
    ids = np.asarray(sample_ids)
    if mu.ndim != 2 or mu.shape != logvar.shape or min(mu.shape) == 0:
        raise ValueError('mu/logvar must have matching nonempty matrix shapes')
    if not (np.isfinite(mu).all() and np.isfinite(logvar).all()):
        raise ValueError('Posterior parameters must be finite')
    if ids.shape != (len(mu),) or pd.isna(ids).any():
        raise ValueError('Sample IDs must align with posteriors')
    ids = ids.astype(str)
    if len(np.unique(ids)) != len(ids):
        raise ValueError('Sample IDs must be unique')
    with np.errstate(over='ignore', under='ignore'):
        sigma = np.exp(.5 * logvar)
    if not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError('Posterior standard deviations are outside numeric range; no clipping')
    if saved_z is not None:
        saved_z = np.asarray(saved_z, dtype=float)
        if saved_z.shape != mu.shape or not np.isfinite(saved_z).all():
            raise ValueError('saved_z must be finite and match the posterior shape')
    indices = np.arange(len(mu)) if len(mu) <= config.max_samples else np.sort(
        np.random.default_rng(config.seed).choice(len(mu), config.max_samples, replace=False))
    variants = []
    for method in config.methods:
        if method == 'adp':
            variants.append(('adp', method, config))
        else:
            variants.extend((f'hdbscan_mcs{size}', method, replace(config, min_cluster_size=size))
                            for size in config.min_cluster_sizes)
    directory = None if output_dir is None else Path(output_dir)/'metrics_artifacts'/run_name/'density_z'/split
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
    records, patients = [], []
    for draw in range(config.n_draws):
        source = 'saved_z' if draw == 0 and saved_z is not None else 'redrawn_from_mu_logvar'
        z = saved_z if source == 'saved_z' else mu + sigma * np.random.default_rng(
            np.random.SeedSequence([config.seed, draw, 1])).standard_normal(mu.shape)
        values = z[indices]
        for variant, method, options in variants:
            record = dict(draw=draw, z_source=source, variant=variant, method=method,
                          n_samples=len(indices), status='ok', warning=None)
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    labels, summary, arrays, metadata = fit_density_clusters(values, method, options)
                record.update(summary, **metadata)
                record['warning'] = '; '.join(dict.fromkeys(str(w.message) for w in caught)) or None
                if directory is not None:
                    patients.append(pd.DataFrame(dict(sample_id=ids[indices], draw=draw,
                                                      variant=variant, cluster=labels)))
                    np.savez_compressed(directory/f'{variant}_draw{draw:03d}.npz',
                                        sample_ids=ids[indices], labels=labels, **arrays)
            except Exception as error:
                record.update(status='dependency_unavailable' if isinstance(error, ImportError)
                              else 'degenerate_input' if isinstance(error, ValueError) else 'failed',
                              warning=f'{type(error).__name__}: {error}')
            records.append(record)
    draws = pd.DataFrame(records)
    rows = []
    for variant, group in draws.groupby('variant', sort=False):
        good = group[group.status.eq('ok')]
        status = 'ok' if len(good) == len(group) else 'partial' if len(good) else str(group.status.iloc[0])
        warning = '; '.join(dict.fromkeys(group.warning.dropna().astype(str))) or None
        for name in ('n_clusters', 'noise_fraction', 'assigned_fraction', 'largest_cluster_fraction', 'intrinsic_dimension'):
            values = good[name].dropna() if name in good else pd.Series(dtype=float)
            if name == 'intrinsic_dimension' and not variant.startswith('adp'):
                continue
            statistics = dict(mean=values.mean(), median=values.median(), std=values.std(ddof=1),
                              q25=values.quantile(.25), q75=values.quantile(.75))
            for statistic, value in statistics.items():
                rows.append(dict(metric_name=f'density_z_{variant}_{name}_{statistic}', value=float(value),
                    status=status, warning=warning, n_samples=len(indices), split=split,
                    n_draws_ok=len(good), n_draws_failed=len(group)-len(good)))
    table = pd.DataFrame(rows)
    if directory is not None:
        draws.to_csv(directory/'draw_metrics.csv', index=False)
        table.to_csv(directory/'scalar_metrics.csv', index=False)
        if patients:
            pd.concat(patients, ignore_index=True).to_csv(directory/'patient_clusters.csv', index=False)
        (directory/'metadata.json').write_text(json.dumps(dict(config=asdict(config),
            selected_indices=indices.tolist(), sample_ids=ids[indices].tolist(),
            saved_z_used=saved_z is not None, draw_protocol='saved z first if available, then reproducible Gaussian draws',
            uncertainty='posterior sampling variability, not patient bootstrap',
            scaling='original posterior coordinates; no standardisation'), indent=2)+'\n')
    return table, draws


if __name__ == '__main__':
    import argparse
    import hashlib
    parser = argparse.ArgumentParser(description='Offline ADP/HDBSCAN on repeated posterior z draws')
    parser.add_argument('--posterior', type=Path, required=True, help='Saved posterior/snapshot npz with mu, logvar, sample_ids; z optional')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--draws', type=int, default=10)
    parser.add_argument('--run-name', default=None)
    parser.add_argument('--split', choices=['train', 'valid', 'test'], default=None)
    args = parser.parse_args()
    split = args.split or next((s for s in ('train', 'valid', 'test') if args.posterior.stem.endswith('_'+s)), 'test')
    name = args.run_name or ('snapshot_'+args.posterior.stem+'_'+hashlib.sha256(str(args.posterior.resolve()).encode()).hexdigest()[:10]
                            if args.posterior.name.startswith('epoch_') else args.posterior.parent.name)
    with np.load(args.posterior) as posterior:
        table, draws = run_posterior_density_pipeline(posterior['mu'], posterior['logvar'],
            posterior['sample_ids'], DensityClusteringConfig(n_draws=args.draws), args.output_dir,
            saved_z=posterior['z'] if 'z' in posterior.files else None, run_name=name, split=split)
    print(draws.to_string(index=False))
