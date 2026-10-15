"""Observe posterior collapse and recovery without feeding metrics into training.

Read frozen train/valid posteriors, compute the existing ZADU geometry adapter
on a fixed validation subset, and use native AUTOENCODIX probes for class AUC.
Test samples are not used to select schedules or recovery thresholds.
"""
import argparse
import warnings
from sklearn.exceptions import ConvergenceWarning
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from autoencodix.evaluate._general_evaluator import GeneralEvaluator
from beta_vae_experiments.metrics.geometry import compute_zadu_geometry


def analyse(folder, k_values=(5, 10, 20), max_samples=512, targets=(), rate_thresholds=(.001, .01, .1)):
    k_values = tuple(k_values)
    if not k_values or any(not isinstance(k, int) or isinstance(k, bool) or k < 1 for k in k_values):
        raise ValueError("k_values must contain positive integers")
    if len(set(k_values)) != len(k_values):
        raise ValueError("k_values must be unique")
    folder = Path(folder)
    snapshots = folder / 'epoch_snapshots'
    with np.load(snapshots / 'input_valid.npz') as saved:
        x, reference_ids = saved['x'], saved['sample_ids']
    selected = np.sort(np.random.default_rng(31).choice(len(x), min(max_samples, len(x)), replace=False))
    train_labels, valid_labels = (snapshots / 'labels_train.parquet', snapshots / 'labels_valid.parquet')
    labels = pd.concat([pd.read_parquet(train_labels), pd.read_parquet(valid_labels)]) if train_labels.exists() and valid_labels.exists() else None
    records, auc_records, geometry_records = [], [], []
    for path in sorted(snapshots.glob('epoch_*_valid.npz')):
        epoch = int(path.name.split('_')[1])
        with np.load(path) as saved:
            mu, lv, ids = saved['mu'].astype(float), saved['logvar'].astype(float), saved['sample_ids']
        if not np.array_equal(ids, reference_ids):
            raise ValueError('Snapshot IDs differ from frozen reference')
        kl = .5 * (mu**2 + np.exp(lv) - lv - 1).sum(1).mean()
        singular = np.linalg.svd(mu - mu.mean(0), compute_uv=False)
        p = singular / singular.sum() if singular.sum() > 0 else np.array([])
        effective_rank = np.exp(-(p[p>0] * np.log(p[p>0])).sum()) if len(p) else 0.
        record = dict(epoch=epoch, kl_total=float(kl), active_units=int((mu.var(0) > .01).sum()),
                      effective_rank=float(effective_rank), var_mu_total=float(mu.var(0).sum()))
        for threshold in rate_thresholds:
            if not np.isfinite(threshold) or threshold < 0:
                raise ValueError('Rate thresholds must be finite and nonnegative')
            record[f'rate_below_{threshold:g}_nats'] = bool(kl <= threshold)
        for k in k_values:
            # Never silently replace k: all scales use the same frozen samples.
            names = ['trustworthiness', 'continuity', 'lcmc',
                     'scale_normalized_stress', 'steadiness', 'cohesiveness']
            enough_samples = k < len(selected) / 2
            values = {name: np.nan for name in names}
            if enough_samples:
                geometry = compute_zadu_geometry(x[selected], mu[selected], k, ids[selected])
                values.update({name: geometry.global_metrics.get(name, np.nan) for name in names})
            geometry_records.append(dict(epoch=epoch, k=k, n_samples=len(selected),
                                         status='ok' if enough_samples else 'insufficient_samples', **values))
            record.update({f'{name}_k{k}': value for name, value in values.items()})
            if k == 20:  # Preserve existing observer column names, explicitly the k=20 view.
                record.update(values)
        records.append(record)
        if labels is not None and targets:
            with np.load(snapshots / f'epoch_{epoch:05d}_train.npz') as train:
                train_mu, train_lv, train_ids = train['mu'], train['logvar'], train['sample_ids']
            rng = np.random.default_rng(31)
            train_z = train_mu + np.exp(.5*train_lv) * rng.standard_normal(train_mu.shape)
            valid_z = mu + np.exp(.5*lv) * rng.standard_normal(mu.shape)
            features = pd.concat([pd.DataFrame(train_z, index=train_ids), pd.DataFrame(valid_z, index=ids)])
            for target in targets:
                if target not in labels:
                    continue
                y_train = labels.reindex(train_ids)[target].astype('string')
                known = y_train.notna() & ~y_train.str.lower().isin(['unknown','nan','none',''])
                counts = y_train[known].value_counts()
                classes = set(counts[counts >= 20].index)
                y_valid = labels.reindex(ids)[target].astype('string')
                valid_keep = y_valid.isin(classes)
                if len(classes) < 2 or set(y_valid[valid_keep]) != classes:
                    auc_records.append(dict(epoch=epoch, task=target, status='missing_evaluation_classes', value=np.nan))
                    continue
                split = pd.DataFrame({'SAMPLE_ID': list(train_ids[y_train.isin(classes)]) + list(ids[valid_keep]),
                                      'SPLIT': ['train']*int(y_train.isin(classes).sum()) + ['valid']*int(valid_keep.sum())})
                with warnings.catch_warnings(record=True) as recorded:
                    warnings.simplefilter('always', ConvergenceWarning)
                    scores = GeneralEvaluator._single_ml_presplit(split, features, labels, target,
                             LogisticRegression(max_iter=10000), 'roc_auc_ovo', 'classification', top_k_classes=None)
                converged = not any(issubclass(w.category, ConvergenceWarning) for w in recorded)
                auc_records.append(dict(epoch=epoch, task=target, status='ok' if converged else 'probe_not_converged',
                                        value=float(scores.loc[scores.score_split.eq('valid'), 'value'].iloc[0]) if converged else np.nan))
    result = pd.DataFrame(records)
    result.to_parquet(folder / 'epoch_observer_metrics.parquet', index=False)
    result.to_csv(folder / 'epoch_observer_metrics.csv', index=False)
    pd.DataFrame(geometry_records).to_parquet(folder / 'epoch_geometry_by_k.parquet', index=False)
    pd.DataFrame(auc_records).to_parquet(folder / 'epoch_auc.parquet', index=False)
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    for axis, columns in zip(axes.flat, [['kl_total'], ['active_units','effective_rank'],
                                       [f'trustworthiness_k{k}' for k in k_values], [f'lcmc_k{k}' for k in k_values]]):
        for column in columns:
            axis.plot(result.epoch, result[column], marker='.', label=column)
        axis.set_xlabel('Epoch');axis.legend();axis.grid(alpha=.2)
    figure.tight_layout();figure.savefig(folder / 'epoch_observer_metrics.png', dpi=170);plt.close(figure)
    if auc_records:
        scores = pd.DataFrame(auc_records)
        figure, axis = plt.subplots(figsize=(11, 5))
        for task, group in scores.groupby('task'):
            axis.plot(group.epoch, group.value, marker='.', label=task)
        axis.set(xlabel='Epoch', ylabel='Validation macro-OvO AUC', ylim=(0, 1.02))
        axis.legend();axis.grid(alpha=.2);figure.tight_layout()
        figure.savefig(folder / 'epoch_auc.png', dpi=170);plt.close(figure)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_folder', type=Path)
    parser.add_argument('--max-samples', type=int, default=512)
    parser.add_argument('--k-values', type=int, nargs='+', default=[5, 10, 20])
    parser.add_argument('--targets', nargs='*', default=['CANCER_TYPE_ACRONYM','SEX'])
    args = parser.parse_args()
    print(analyse(args.run_folder, k_values=args.k_values, max_samples=args.max_samples, targets=args.targets).to_string(index=False))
