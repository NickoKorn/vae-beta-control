"""Labelabhängige Offline-Diagnostik für finale VAE-Repräsentationen.

Keine unüberwachten Controller-Features und keine Testsatz-Modellauswahl.
Silhouette: Rousseeuw (1987), https://doi.org/10.1016/0377-0427(87)90125-7.
Konfusionsmatrix: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
Nachbarn: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
"""
from pathlib import Path
import math

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, silhouette_samples
from sklearn.neighbors import NearestNeighbors


def confusion_rows(truth, predictions, trained_classes):
    """Wahre Klasse in Zeilen; unbekannte Testklassen bleiben sichtbar."""
    truth, predictions = np.asarray(truth, dtype=str), np.asarray(predictions, dtype=str)
    classes = sorted(set(trained_classes) | set(truth) | set(predictions))
    matrix = confusion_matrix(truth, predictions, labels=classes)
    support = matrix.sum(axis=1)
    return pd.DataFrame([
        dict(true_class=a, predicted_class=b, count=int(matrix[i, j]),
             fraction=float(matrix[i, j]/support[i]) if support[i] else np.nan,
             n_true=int(support[i]), trained_class=a in trained_classes,
             status='ok' if support[i] else 'absent_true_class')
        for i, a in enumerate(classes) for j, b in enumerate(classes)])


def summarize_confusions(frame, n_draws):
    rows = []
    for key, group in frame.groupby(['target', 'split', 'true_class', 'predicted_class'], sort=False):
        complete = len(group) == n_draws and np.isfinite(group.fraction).all()
        rows.append(dict(zip(('target', 'split', 'true_class', 'predicted_class'), key),
            count_mean=float(group['count'].mean()), count_std=float(group['count'].std(ddof=1)),
            fraction_mean=float(group.fraction.mean()) if complete else np.nan,
            fraction_std=float(group.fraction.std(ddof=1)) if complete else np.nan,
            fraction_median=float(group.fraction.median()) if complete else np.nan,
            n_true=int(group.n_true.iloc[0]), n_draws=n_draws,
            n_valid_draws=int(np.isfinite(group.fraction).sum()),
            trained_class=bool(group.trained_class.iloc[0]),
            status=('warning' if 'probe_warning' in group and group.probe_warning.notna().any()
                    else 'ok') if complete else 'unavailable'))
    return pd.DataFrame(rows)


def label_geometry(points, labels, sample_ids, *, k, max_samples, seed):
    """Silhouette auf festem, gleichverteiltem Subset; Reinheit auf allen Labels.

    Ein Subset pro Split/Target, unabhängig vom Draw. Kein Oversampling seltener
    Klassen. Fehlende Labels ausgeschlossen. k bleibt fest, Self wird über
    kneighbors(X=None) ausgeschlossen. Tie-/Duplikatwerte werden markiert.
    Reinheit = Anteil gleicher Labels unter k Nachbarn; chance baseline ist
    (n_klasse−1)/(n−1). Dies ist keine Cluster-Purity und kein Klassifikator.
    """
    points, ids = np.asarray(points, dtype=float), np.asarray(sample_ids, dtype=str)
    series = pd.Series(np.asarray(labels), index=ids)
    keep = series.notna().to_numpy()
    values, ids, y = points[keep], ids[keep], series[keep].astype(str).to_numpy()
    n = len(y)
    if n == 0:
        return pd.DataFrame(), pd.DataFrame([dict(metric='label_geometry', value=np.nan,
            status='unavailable', warning='No labeled samples', n_samples=0)])
    rng = np.random.default_rng(seed)
    subset = np.sort(rng.choice(n, min(n, max_samples), replace=False))
    output = pd.DataFrame(dict(sample_id=ids, **{'class': y},
        silhouette=np.nan, neighbor_purity=np.nan, purity_chance=np.nan,
        silhouette_selected=np.isin(np.arange(n), subset)))
    warnings = []
    unique = len(np.unique(values, axis=0))
    if unique == 1:
        warnings.append('Constant representation: distances do not define meaningful neighborhoods')
    else:
        if unique != n:
            warnings.append('Duplicate points: nearest-neighbor ties depend on row/backend order')
        selected_y = y[subset]
        n_classes = len(np.unique(selected_y))
        if 2 <= n_classes < len(subset):
            output.loc[subset, 'silhouette'] = silhouette_samples(values[subset], selected_y, metric='euclidean')
        else:
            warnings.append('Silhouette requires 2 <= n_classes < subset size')
        if k < n:
            neighbors = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=1).fit(values)
            distances, indices = neighbors.kneighbors(X=None)
            output['neighbor_purity'] = (y[indices] == y[:, None]).mean(axis=1)
            if np.any(distances == 0):
                warnings.append('Zero neighbor distances present; no jitter applied')
            counts = pd.Series(y).value_counts()
            output['purity_chance'] = [(counts[label]-1)/(n-1) for label in y]
        else:
            warnings.append('Neighbor purity requires k < labeled samples; no automatic clipping')
    summary = []
    for metric, selection in [('silhouette', output.silhouette_selected),
                               ('neighbor_purity', np.ones(n, dtype=bool)),
                               ('purity_chance', np.ones(n, dtype=bool))]:
        measured = output.loc[selection, metric]
        finite = np.isfinite(measured)
        complete = len(measured) > 0 and finite.all()
        summary.append(dict(metric=metric, value=float(measured.mean()) if complete else np.nan,
            status=('warning' if warnings else 'ok') if complete else 'unavailable',
            warning='; '.join(warnings) or None, n_samples=len(measured),
            n_valid=int(finite.sum()), k=k, n_labeled=n,
            n_classes_full=len(np.unique(y)), n_classes_subset=len(np.unique(y[subset]))))
    output['neighbor_purity_excess'] = output.neighbor_purity - output.purity_chance
    return output, pd.DataFrame(summary)


def distribution_tables(samples, annotations, targets, edges):
    """Histogrammgrenzen fest über β/Runs; Überlauf bleibt als eigener Bin."""
    summaries, histograms = [], []
    bins = np.r_[np.asarray(edges, float), np.inf]
    metrics = ['kl_total', 'kl_per_dimension', 'near_prior_dimension_fraction',
               'all_dimensions_near_prior', 'reconstruction_mse_mean']
    for split, split_samples in samples.groupby('split', sort=False):
        for target in targets:
            if target == 'AGE':
                continue
            labels = (annotations.reindex(split_samples.sample_id)[target].to_numpy()
                      if target in annotations else np.full(len(split_samples), None))
            grouped = split_samples.copy()
            grouped['_class'] = pd.Series(labels, index=grouped.index).astype('string')
            # Missing label is an explicit group, never a literal cancer class.
            for label, group in grouped.groupby('_class', dropna=False, sort=True):
                missing_label = bool(pd.isna(label))
                for metric in metrics:
                    raw = group[metric].to_numpy(float) if metric in group else np.full(len(group), np.nan)
                    finite = np.isfinite(raw)
                    values = raw[finite]
                    complete = finite.all()
                    common = dict(target=target, split=split, **{'class': label},
                        missing_label=missing_label, metric=metric, n_samples=len(raw),
                        n_valid=int(finite.sum()), n_invalid=int((~finite).sum()))
                    summaries.append(dict(common, mean=float(values.mean()) if complete else np.nan,
                        median=float(np.median(values)) if complete else np.nan,
                        q25=float(np.quantile(values, .25)) if complete else np.nan,
                        q75=float(np.quantile(values, .75)) if complete else np.nan,
                        status='ok' if complete else 'unavailable'))
                    metric_bins = np.linspace(0, 1, 11) if 'near_prior' in metric else bins
                    counts, _ = np.histogram(values, bins=metric_bins)
                    for index, count in enumerate(counts):
                        histograms.append(dict(common, bin=index, lower=float(metric_bins[index]),
                            upper=float(metric_bins[index+1]), count=int(count),
                            fraction=float(count/len(values)) if len(values) else np.nan,
                            status='ok' if complete else 'partial_or_unavailable'))
    return pd.DataFrame(summaries), pd.DataFrame(histograms)


def plot_diagnostics(directory, confusions, histograms):
    """Mean-Konfusionsmatrizen und Histogramme je Klasse, ohne Test-Tuning."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    confusion_groups = [] if confusions.empty else confusions.groupby(['target', 'split'], sort=False)
    for index, ((target, split), group) in enumerate(confusion_groups):
        matrix = group.pivot(index='true_class', columns='predicted_class', values='fraction_mean')
        fig, axis = plt.subplots(figsize=(max(6, len(matrix)*.26), max(5, len(matrix)*.25)))
        image = axis.imshow(np.ma.masked_invalid(matrix.to_numpy()), vmin=0, vmax=1, cmap='Blues')
        axis.set(xticks=np.arange(len(matrix.columns)), yticks=np.arange(len(matrix)),
                 xticklabels=matrix.columns, yticklabels=matrix.index,
                 xlabel='Vorhergesagte Klasse', ylabel='Wahre Klasse', title=f'{target}, {split}: Mittel über z-Draws')
        axis.tick_params(axis='x', rotation=90)
        fig.colorbar(image, ax=axis, label='Anteil je wahrer Klasse')
        fig.tight_layout()
        fig.savefig(directory / f'confusion_{index:02d}.png', dpi=130)
        plt.close(fig)
    if histograms.empty:
        return
    for index, ((target, split, metric), group) in enumerate(histograms.groupby(['target', 'split', 'metric'], sort=False)):
        if metric not in ('kl_total', 'reconstruction_mse_mean'):
            continue
        groups = list(group.groupby(['class', 'missing_label'], dropna=False, sort=True))
        columns = min(4, len(groups))
        fig, axes = plt.subplots(math.ceil(len(groups)/columns), columns,
            figsize=(columns*3.1, math.ceil(len(groups)/columns)*2.3), squeeze=False)
        for axis, ((label, missing), data) in zip(axes.flat, groups):
            axis.bar(data.bin, data.fraction, width=.9)
            axis.set(title=f'{"fehlendes Label" if missing else label} (n={data.n_samples.iloc[0]})',
                     xlabel='Fester Bin-Index (Grenzen in CSV)', ylabel='Probenanteil')
        for axis in list(axes.flat)[len(groups):]:
            axis.set_visible(False)
        fig.suptitle(f'{target}, {split}: {metric}')
        fig.tight_layout()
        fig.savefig(directory / f'histogram_{index:02d}.png', dpi=110)
        plt.close(fig)
