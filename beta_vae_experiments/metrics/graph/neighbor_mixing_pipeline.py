"""Active kNN group-mixing runner; optional density-overlap methods are complementary.

Neighbors and foreign-group fractions are implemented in graph.neighbor_mixing.
No BC integration, Hellinger distance matrix or reference clustering runs here.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .neighbor_mixing import foreign_neighbor_fraction


@dataclass(frozen=True)
class NeighborMixingConfig:
    """kNN group-mixing settings; z draws are averaged after measuring mixing."""

    seed: int = 31
    group_columns: tuple[str, ...] = ()
    sex_mixed_cancers_only: bool = False
    n_neighbors: int = 20
    n_draws: int = 50

    def __post_init__(self):
        for name in ('n_neighbors', 'n_draws'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError('seed must be a nonnegative 32-bit integer')


def _posterior_arrays(mu, logvar):
    mu, logvar = np.asarray(mu, dtype=float), np.asarray(logvar, dtype=float)
    if mu.ndim != 2 or min(mu.shape) == 0 or mu.shape != logvar.shape:
        raise ValueError('mu and logvar must have matching nonempty (patients, dimensions) shapes')
    if not np.isfinite(mu).all() or not np.isfinite(logvar).all():
        raise ValueError('Posterior parameters must be finite')
    with np.errstate(over='ignore', under='ignore'):
        sigma = np.exp(0.5 * logvar)
    if not np.isfinite(sigma).all() or np.any(sigma == 0):
        raise ValueError('Posterior standard deviations are outside numerical range')
    return mu, logvar, sigma


def run_neighbor_mixing_pipeline(inputs, config=None, output_dir=None, *, groups=None, split='test'):
    """Run foreign-neighbor mixing on mu, posterior W2 geometry and repeated z.

    Group memberships stay fixed. Configured annotation groups are label-based;
    a sample-ID-indexed `groups` Series can instead supply reference clusters.
    Euclidean neighbors on [mu, sigma] are exact W2 neighbors for diagonal
    Gaussian posteriors; this measures transport geometry, not density overlap.
    All eligible patients are used without additional subsampling.
    Draw variation is sampling sensitivity, not a population confidence interval.
    """
    config = config or NeighborMixingConfig()
    if split not in ('train', 'val', 'test'):
        raise ValueError("split must be 'train', 'val', or 'test'")
    mu, logvar = getattr(inputs, 'mu_' + split), getattr(inputs, 'logvar_' + split)
    ids = getattr(inputs, 'sample_ids_' + split)
    if mu is None or ids is None:
        raise ValueError('Neighbor mixing requires posterior means and sample IDs')
    mu, ids = np.asarray(mu), np.asarray(ids).astype(str)
    if len(ids) != len(mu) or len(np.unique(ids)) != len(ids):
        raise ValueError('sample_ids must be unique and align with posterior rows')
    sigma = _posterior_arrays(mu, logvar)[2] if logvar is not None else None
    group_sets = {}
    if groups is not None:
        if not isinstance(groups, pd.Series) or not groups.index.is_unique:
            raise ValueError('groups must be a uniquely sample-ID-indexed Series')
        group_sets['supplied'] = groups.reindex(ids)
    if config.group_columns:
        if split != 'test':
            raise ValueError('Configured annotation columns currently require test inputs')
        annotations = inputs.sample_metadata_test
        if isinstance(annotations, dict):
            annotations = annotations['paired']
        for column in config.group_columns:
            labels = annotations[column].reindex(ids)
            labels = labels.mask(labels.astype(str).str.strip().str.lower().isin(
                ['unknown', 'missing', 'nan', 'none', '', 'not reported']))
            if column == 'SEX' and config.sex_mixed_cancers_only:
                from ..evaluation.downstream import mixed_sex_cancers, sex_task_mask
                train = inputs.sample_metadata_train
                if isinstance(train, dict):
                    train = train['paired']
                labels = labels.where(sex_task_mask(annotations.reindex(ids), mixed_sex_cancers(train)))
            group_sets[column] = labels
    scalar_rows, summaries, patients, group_rows, draw_rows = [], [], [], [], []
    matrices = {}
    for grouping, labels in group_sets.items():
        for representation in ('mu', 'wasserstein', 'z'):
            results = []
            warning, status = None, 'ok'
            for draw in range(config.n_draws if representation == 'z' else 1):
                if representation in ('wasserstein', 'z') and sigma is None:
                    status, warning = 'missing_posterior', 'logvar unavailable'
                    break
                values = mu
                if representation == 'wasserstein':
                    values = np.concatenate((mu, sigma), axis=1)
                if representation == 'z':
                    rng = np.random.default_rng(np.random.SeedSequence([config.seed, draw]))
                    values = mu + sigma * rng.standard_normal(mu.shape)
                try:
                    result = foreign_neighbor_fraction(values, labels.to_numpy(), config.n_neighbors)
                except ValueError as error:
                    status, warning = 'degenerate_input', str(error)
                    break
                results.append(result)
                draw_rows.append(dict(grouping=grouping, representation=representation, draw=draw,
                    mean=result['per_patient'].mean(), normalized_mean=result['normalized'].mean()))
            mean, normalized_mean, draw_std = np.nan, np.nan, np.nan
            n = int((labels.notna() & labels.ne(-1)).sum())
            if results and status == 'ok':
                first = results[0]
                fractions = np.stack([r['per_patient'] for r in results])
                normalized = np.stack([r['normalized'] for r in results])
                mean, normalized_mean = fractions.mean(), normalized.mean()
                draw_std = fractions.mean(axis=1).std(ddof=1) if len(results) > 1 else 0.
                eligible_ids, names = ids[first['patient_mask']], first['groups']
                for i, sample_id in enumerate(eligible_ids):
                    patients.append(dict(grouping=grouping, representation=representation,
                        sample_id=sample_id, group=names[i], fraction=fractions[:, i].mean(),
                        normalized=normalized[:, i].mean(), random_expectation=first['random_mixing_expectation'][i],
                        forced_foreign_floor=first['forced_foreign_floor'][i]))
                unique = list(pd.unique(names))
                matrix = pd.DataFrame(0., index=unique, columns=unique)
                for name in unique:
                    mask = names == name
                    group_rows.append(dict(grouping=grouping, representation=representation, group=name,
                        n_patients=int(mask.sum()), fraction=fractions[:, mask].mean(),
                        normalized=normalized[:, mask].mean()))
                    for other in unique:
                        matrix.loc[name, other] = np.mean([
                            (names[r['neighbor_indices'][mask]] == other).mean() for r in results])
                matrices[(grouping, representation)] = matrix
            summaries.append(dict(grouping=grouping, representation=representation, mean=mean,
                normalized_mean=normalized_mean, draw_std=draw_std, n_patients=n,
                n_groups=labels.nunique(), n_neighbors=config.n_neighbors, n_draws=len(results),
                status=status, warning=warning,
                duplicate_fraction=results[0]['duplicate_fraction'] if results else np.nan))
            for name, value in [('foreign_neighbor_fraction', mean), ('foreign_neighbor_fraction_normalized', normalized_mean)]:
                scalar_rows.append(dict(metric_name=f'graph_neighbor_mixing_{grouping}_{representation}_{name}',
                    value=float(value), status=status, warning=warning, n_samples=n, split=split))
    if output_dir is not None:
        directory = Path(output_dir) / 'metrics_artifacts' / inputs.run_name / 'neighbor_mixing' / split
        directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summaries).to_csv(directory / 'neighbor_mixing_summary.csv', index=False)
        pd.DataFrame(patients).to_csv(directory / 'neighbor_mixing_patients.csv', index=False)
        pd.DataFrame(group_rows).to_csv(directory / 'neighbor_mixing_groups.csv', index=False)
        pd.DataFrame(draw_rows).to_csv(directory / 'neighbor_mixing_draws.csv', index=False)
        for (grouping, representation), matrix in matrices.items():
            matrix.to_csv(directory / f'neighbor_matrix_{grouping}_{representation}.csv')
        if group_sets:
            np.savez_compressed(directory / 'group_memberships.npz', sample_ids=ids,
                **{name + '_in_task': (labels.notna() & labels.ne(-1)).to_numpy() for name, labels in group_sets.items()},
                **{name: labels.fillna('').astype(str).to_numpy(dtype=str) for name, labels in group_sets.items()})
        (directory / 'settings.json').write_text(json.dumps(dict(
            **asdict(config), split=split, method='foreign_neighbor_fraction_only',
            self_neighbors_excluded=True, n_patients=len(mu), groupings=list(group_sets),
            random_baseline='(N-own_group_size)/(N-1)',
            wasserstein_representation='[mu, exp(logvar/2)]; exact diagonal-Gaussian W2 geometry; no column standardization',
        ), indent=2) + '\n')
        pd.DataFrame(scalar_rows).to_csv(directory / 'scalar_metrics.csv', index=False)
    return pd.DataFrame(scalar_rows, columns=['metric_name','value','status','warning','n_samples','split'])
