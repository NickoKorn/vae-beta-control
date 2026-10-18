"""Wiederholte Probes auf einzelnen posterioren z-Draws, keine Ensembles.

Kingma & Welling: https://arxiv.org/abs/1312.6114 (Reparametrisierung).
Bonheme & Grzes: https://jmlr.org/papers/v24/21-1145.html (μ versus z).
Zehn Draws sind eine Projektentscheidung, kein Literaturstandard. Die Streuung
misst Sampling-Empfindlichkeit, kein Patienten- oder Trainings-Konfidenzintervall.
"""
from dataclasses import dataclass, asdict
from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, r2_score
from .class_diagnostics import (
    confusion_rows, summarize_confusions, label_geometry, posterior_sample_table,
    distribution_tables, make_reconstruction_callback, plot_diagnostics,
)


@dataclass(frozen=True)
class DownstreamMetricConfig:
    n_draws: int = 10
    sampling_seed: int = 31
    probe_seed: int = 31
    max_iter: int = 1000
    diagnostics_enabled: bool = True
    label_geometry_enabled: bool = True
    diagnostic_k: int = 20
    silhouette_max_samples: int = 1000
    diagnostic_plots: bool = True
    near_prior_epsilon: float = .01
    histogram_edges: tuple[float, ...] = (0., 1e-6, 1e-5, 1e-4, 1e-3, .01, .1,
                                          1., 10., 100., 1000., 10000., 1e6)

    def __post_init__(self):
        if type(self.n_draws) is not int or (self.n_draws != 0 and self.n_draws < 2):
            raise ValueError('n_draws must be zero (legacy only) or at least two')
        for name in ('sampling_seed', 'probe_seed'):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < 2**32:
                raise ValueError(f'{name} must be a nonnegative 32-bit integer')
        if type(self.max_iter) is not int or self.max_iter < 1:
            raise ValueError('max_iter must be a positive integer')
        for name in ('diagnostics_enabled', 'label_geometry_enabled', 'diagnostic_plots'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be boolean')
        if type(self.diagnostic_k) is not int or self.diagnostic_k < 1:
            raise ValueError('diagnostic_k must be a positive integer')
        if type(self.silhouette_max_samples) is not int or self.silhouette_max_samples < 3:
            raise ValueError('silhouette_max_samples must be at least three')
        if (isinstance(self.near_prior_epsilon, bool) or not np.isfinite(self.near_prior_epsilon)
                or self.near_prior_epsilon < 0):
            raise ValueError('near_prior_epsilon must be finite and nonnegative')
        edges = np.asarray(self.histogram_edges, dtype=float)
        if edges.ndim != 1 or len(edges) < 2 or edges[0] != 0 or not np.isfinite(edges).all() or not (np.diff(edges) > 0).all():
            raise ValueError('histogram_edges must start at zero and be finite, strictly increasing')
        object.__setattr__(self, 'histogram_edges', tuple(edges.tolist()))


def load_final_posteriors(directory):
    """Lade unveränderte finale Encoder-Ausgaben einschließlich ihrer IDs."""
    arrays = {}
    hashes = {}
    for split in ('train', 'valid', 'test'):
        path = Path(directory) / f'posterior_{split}.npz'
        with np.load(path, allow_pickle=False) as saved:
            arrays[split] = {key: saved[key].copy() for key in ('mu', 'logvar', 'sample_ids')}
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    return arrays, hashes


def evaluate_sampled_z(posteriors, annotations, targets, config, *, diagnostics=None,
                       reconstruction_callback=None):
    """Fit je Draw auf Train; Valid/Test ausschließlich auswerten.

    Gleiche LogReg-Grundspezifikation wie AUTOENCODIX (C=1, lbfgs, max_iter=1000),
    keine zusätzliche Skalierung und keine Hyperparametersuche. AGE verwendet
    LinearRegression. Klassen werden nur auf Train gelernt, nicht zusammengelegt.
    Andere explizit angeforderte Targets werden als Klassifikation behandelt.
    Jeder Draw enthält einen unabhängigen Punkt pro Probe und Split. Die AUCs
    werden aggregiert, nicht die Vektoren oder Vorhersagewahrscheinlichkeiten.
    """
    if not isinstance(annotations, pd.DataFrame):
        raise ValueError('Annotations must be a DataFrame indexed by sample ID')
    if not targets or len(set(targets)) != len(targets) or config.n_draws < 2:
        raise ValueError('Repeated evaluation requires distinct targets and at least two draws')
    labels = annotations.copy()
    if labels.index.isna().any():
        raise ValueError('Annotation IDs must not be missing')
    labels.index = labels.index.astype(str)
    if not labels.index.is_unique or not labels.columns.is_unique:
        raise ValueError('Annotation IDs and columns must be unique')
    seen, checked, dimension = set(), {}, None
    for split in ('train', 'valid', 'test'):
        data = posteriors[split]
        mu, lv = (np.asarray(data[key], dtype=float) for key in ('mu', 'logvar'))
        raw_ids = np.asarray(data['sample_ids'])
        if raw_ids.ndim != 1 or pd.isna(raw_ids).any():
            raise ValueError(f'Invalid sample IDs: {split}')
        ids = raw_ids.astype(str)
        if (mu.ndim != 2 or mu.shape != lv.shape or mu.shape[1] == 0
                or len(mu) != len(ids) or len(mu) == 0
                or not np.isfinite(mu).all() or not np.isfinite(lv).all()):
            raise ValueError(f'Invalid posterior arrays: {split}')
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError(f'Duplicate or overlapping sample IDs: {split}')
        if not set(ids) <= set(labels.index):
            raise ValueError(f'Annotations do not cover posterior IDs: {split}')
        if dimension is not None and dimension != mu.shape[1]:
            raise ValueError('Posterior latent dimensions differ between splits')
        dimension = mu.shape[1]
        seen.update(ids)
        with np.errstate(over='raise', invalid='raise', under='ignore'):
            sigma = np.exp(.5 * lv)
        checked[split] = (mu, sigma, ids)

    records, class_records, draw_seeds = [], [], []
    collect = diagnostics is not None and config.diagnostics_enabled
    if collect:
        for key in ('confusions', 'geometry_samples', 'geometry_draws', 'reconstruction_draws'):
            diagnostics[key] = []
    for draw in range(config.n_draws):
        features = {}
        for split_index, (split, (mu, sigma, ids)) in enumerate(checked.items()):
            # Separate Streams: changes in split size do not change other draws.
            seed = np.random.SeedSequence([config.sampling_seed, draw, split_index])
            rng = np.random.default_rng(seed)
            z = mu + sigma * rng.standard_normal(mu.shape)
            if not np.isfinite(z).all():
                raise ValueError(f'Nonfinite sampled z: draw={draw}, split={split}')
            features[split] = z
            draw_seeds.append(dict(draw=draw, split=split,
                                   seed_entropy=[config.sampling_seed, draw, split_index]))
            if collect and split in ('valid', 'test'):
                error, mse = None, np.full(len(z), np.nan)
                try:
                    if reconstruction_callback is None:
                        raise ValueError('Final decoder unavailable; no reconstruction substituted')
                    mse = np.asarray(reconstruction_callback(split, z), dtype=float)
                    if mse.shape != (len(z),) or (mse < 0).any():
                        raise ValueError('Invalid per-sample reconstruction MSE')
                except (ValueError, RuntimeError, FloatingPointError) as exc:
                    error, mse = str(exc), np.full(len(z), np.nan)
                diagnostics['reconstruction_draws'].append(pd.DataFrame(dict(draw=draw,
                    split=split, sample_id=ids, value=mse,
                    status=np.where(np.isfinite(mse), 'ok', 'unavailable'), warning=error)))
        for target in targets:
            regression = target == 'AGE'
            metric = 'r2' if regression else 'roc_auc_ovo'
            y = {}
            for split, (_, _, ids) in checked.items():
                series = labels.loc[ids, target] if target in labels else pd.Series(np.nan, index=ids)
                y[split] = pd.to_numeric(series, errors='coerce') if regression else series
            if collect and config.label_geometry_enabled and not regression:
                for split in ('valid', 'test'):
                    samples, geometry = label_geometry(features[split], y[split].to_numpy(),
                        checked[split][2], k=config.diagnostic_k,
                        max_samples=config.silhouette_max_samples, seed=config.sampling_seed)
                    samples = samples.assign(draw=draw, target=target, split=split)
                    geometry = geometry.assign(draw=draw, target=target, split=split)
                    diagnostics['geometry_samples'].append(samples)
                    diagnostics['geometry_draws'].append(geometry)
            train_keep = y['train'].notna().to_numpy(copy=True)
            if regression:
                train_keep &= np.isfinite(y['train'].to_numpy(dtype=float, na_value=np.nan))
            train_y = y['train'].iloc[np.flatnonzero(train_keep)]
            model = (LinearRegression() if regression else
                     LogisticRegression(max_iter=config.max_iter, random_state=config.probe_seed))
            fit_error, fit_warning = None, None
            try:
                if len(train_y) < 2 or (not regression and train_y.nunique() < 2):
                    raise ValueError('Insufficient labeled training samples/classes')
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    model.fit(features['train'][train_keep], train_y if regression else train_y.astype(str))
                fit_warning = '; '.join(dict.fromkeys(str(w.message) for w in caught
                    if not issubclass(w.category, (DeprecationWarning, FutureWarning)))) or None
                if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                    fit_warning = 'Probe did not converge. ' + (fit_warning or '')
            except ValueError as exc:
                fit_error = str(exc)
            for split in ('valid', 'test'):
                series = y[split]
                if collect and not regression and fit_error is None:
                    labeled = series.notna().to_numpy()
                    if labeled.any():
                        confusion = confusion_rows(series[labeled].astype(str),
                            model.predict(features[split][labeled]), model.classes_)
                        diagnostics['confusions'].append(confusion.assign(draw=draw, target=target,
                            split=split, probe_warning=fit_warning))
                keep = series.notna().to_numpy(copy=True)
                if regression:
                    keep &= np.isfinite(series.to_numpy(dtype=float, na_value=np.nan))
                n_labeled = int(keep.sum())
                unseen = 0
                if not regression and fit_error is None:
                    known = series.astype(str).isin(model.classes_).to_numpy()
                    unseen = int((keep & ~known).sum())
                    keep &= known
                truth = series.iloc[np.flatnonzero(keep)]
                row = dict(draw=draw, target=target, split=split, metric=metric,
                           value=np.nan, status='unavailable', warning=fit_error or fit_warning,
                           n_samples=int(keep.sum()), n_missing=len(series)-n_labeled,
                           n_unseen_class=unseen, n_train=int(train_keep.sum()),
                           sampling_seed=config.sampling_seed, probe_seed=config.probe_seed)
                try:
                    if fit_error is not None:
                        raise ValueError(fit_error)
                    if len(truth) < 2:
                        raise ValueError('Insufficient labeled evaluation samples')
                    if regression:
                        row['value'] = float(r2_score(truth, model.predict(features[split][keep])))
                    else:
                        truth = truth.astype(str)
                        probabilities = model.predict_proba(features[split][keep])
                        for column, label in enumerate(model.classes_):
                            positive = np.asarray(truth == label)
                            defined = positive.any() and (~positive).any()
                            class_records.append(dict(draw=draw, target=target, split=split,
                                **{'class': label}, metric='roc_auc_ovr',
                                value=float(roc_auc_score(positive, probabilities[:, column])) if defined else np.nan,
                                status=('warning' if fit_warning else 'ok') if defined else 'undefined_single_class',
                                n_positive=int(positive.sum()), n_negative=int((~positive).sum()),
                                n_train_positive=int((train_y.astype(str) == label).sum())))
                        if len(model.classes_) == 2:
                            row['value'] = float(roc_auc_score(truth == model.classes_[1], probabilities[:, 1]))
                        else:
                            # Require all training classes, never silently reduce macro AUC.
                            if set(truth) != set(model.classes_):
                                raise ValueError('Evaluation split is missing training classes for macro OvO AUC')
                            row['value'] = float(roc_auc_score(truth, probabilities,
                                multi_class='ovo', average='macro', labels=model.classes_))
                    if not np.isfinite(row['value']):
                        raise ValueError('Undefined evaluation score')
                    row['status'] = 'warning' if fit_warning else 'ok'
                except ValueError as exc:
                    row.update(value=np.nan, status='unavailable', warning=str(exc))
                records.append(row)
    details = pd.DataFrame(records)
    details['warning'] = details['warning'].astype('string')
    classes = pd.DataFrame(class_records)
    summary = summarize_draws(details, config.n_draws, ['target', 'split', 'metric'])
    class_summary = (summarize_draws(classes, config.n_draws, ['target', 'split', 'class', 'metric'])
                     if not classes.empty else pd.DataFrame())
    provenance = dict(config=asdict(config), draw_streams=draw_seeds,
        representation='fresh_final_posterior_sampled_z',
        aggregation='mean of individual draw scores; sample SD (ddof=1); no prediction averaging',
        uncertainty='posterior sampling variability, not a patient/training confidence interval',
        probe='refit on train for every draw; no scaling or hyperparameter search',
        numpy_version=version('numpy'), sklearn_version=version('scikit-learn'))
    if collect:
        diagnostics.update({key: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
                            for key, frames in diagnostics.items()})
    return details, summary, classes, class_summary, provenance


def summarize_draws(frame, n_draws, keys):
    """Kein stilles Weglassen fehlgeschlagener Draws aus dem Mittelwert."""
    rows = []
    for key, group in frame.groupby(keys, dropna=False, sort=False):
        valid = np.isfinite(group.value.to_numpy(dtype=float))
        complete = len(group) == n_draws and valid.all()
        rows.append(dict(zip(keys, key if isinstance(key, tuple) else (key,)),
            mean=float(group.value.mean()) if complete else np.nan,
            std=float(group.value.std(ddof=1)) if complete else np.nan,
            median=float(group.value.median()) if complete else np.nan,
            q25=float(group.value.quantile(.25)) if complete else np.nan,
            q75=float(group.value.quantile(.75)) if complete else np.nan,
            n_draws=n_draws, n_valid_draws=int(valid.sum()),
            status=('warning' if (group.status != 'ok').any() else 'ok') if complete else 'unavailable'))
    return pd.DataFrame(rows)


def run_sampled_z_metrics(varix, run_name, output_dir, targets, config):
    directory = Path(output_dir) / 'absolute_core_metrics_v1' / 'downstream' / run_name
    posteriors, hashes = load_final_posteriors(Path(output_dir) / 'run_analysis' / run_name / 'posteriors')
    annotations = varix.visualizer._collect_all_metadata(varix.result)
    annotations = annotations.copy()
    annotations.index = annotations.index.astype(str)
    diagnostics, callback, decoder_warning = {}, None, None
    if config.diagnostics_enabled:
        try:
            callback = make_reconstruction_callback(varix, posteriors)
        except AttributeError:
            decoder_warning = 'Final decoder unavailable; reconstruction artifacts contain explicit NaN'
    details, summary, classes, class_summary, provenance = evaluate_sampled_z(
        posteriors, annotations, targets, config, diagnostics=diagnostics,
        reconstruction_callback=callback)
    provenance['posterior_file_sha256'] = hashes
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in [('draws', details), ('summary', summary),
                        ('class_draws', classes), ('class_summary', class_summary)]:
        frame.insert(0, 'run_name', run_name)
        frame.to_csv(directory / f'z_mc_{name}.csv', index=False)
    if config.diagnostics_enabled:
        export_class_diagnostics(directory, run_name, diagnostics, posteriors,
                                  annotations, targets, config, provenance)
    provenance['decoder_warning'] = decoder_warning
    (directory / 'z_mc_provenance.json').write_text(json.dumps(provenance, indent=2, allow_nan=False)+'\n')
    rows = []
    for row in summary.itertuples(index=False):
        for statistic in ('mean', 'std'):
            rows.append(dict(metric_name=f'downstream_{row.target}_z_mc_{row.split}_{row.metric}_{statistic}',
                             value=getattr(row, statistic), status=row.status,
                             warning=None if row.status == 'ok' else 'Inspect z_mc_draws.csv for unavailable scores or probe warnings',
                             n_samples=None, split=row.split))
    return pd.DataFrame(rows)


def export_class_diagnostics(directory, run_name, diagnostics, posteriors,
                             annotations, targets, config, provenance):
    """Label-Artefakte bleiben getrennt von den unüberwachten Core-Skalaren."""
    tables = dict(diagnostics)
    confusion = diagnostics['confusions']
    tables['confusion_summary'] = (summarize_confusions(confusion, config.n_draws)
                                    if not confusion.empty else pd.DataFrame())
    if not tables['confusion_summary'].empty:
        wrong = tables['confusion_summary'].query('true_class != predicted_class').copy()
        wrong = wrong.sort_values(['target', 'split', 'true_class', 'fraction_mean'],
                                   ascending=[True, True, True, False], na_position='last')
        wrong['confusion_rank'] = wrong.groupby(['target', 'split', 'true_class']).cumcount()+1
        tables['most_confused_classes'] = wrong
    geometry = diagnostics['geometry_draws']
    tables['geometry_summary'] = (summarize_draws(geometry, config.n_draws, ['target', 'split', 'metric'])
                                  if not geometry.empty else pd.DataFrame())
    geometry_samples = diagnostics['geometry_samples']
    if not geometry_samples.empty:
        long = geometry_samples.melt(id_vars=['draw', 'target', 'split', 'class', 'sample_id'],
            value_vars=['silhouette', 'neighbor_purity', 'neighbor_purity_excess'], value_name='value', var_name='metric')
        per_class = long.groupby(['draw', 'target', 'split', 'class', 'metric'], sort=False).agg(
            value=('value', 'mean'), n_valid=('value', 'count')).reset_index()
        per_class['status'] = np.where(np.isfinite(per_class.value), 'ok', 'unavailable')
        tables['geometry_class_draws'] = per_class
        tables['geometry_class_summary'] = summarize_draws(per_class, config.n_draws,
                                                            ['target', 'split', 'class', 'metric'])
    samples = posterior_sample_table(posteriors, annotations, config.near_prior_epsilon)
    reconstruction = summarize_draws(diagnostics['reconstruction_draws'], config.n_draws, ['split', 'sample_id'])
    reconstruction = reconstruction.rename(columns={'mean': 'reconstruction_mse_mean',
        'std': 'reconstruction_mse_std', 'median': 'reconstruction_mse_median',
        'q25': 'reconstruction_mse_q25', 'q75': 'reconstruction_mse_q75',
        'status': 'reconstruction_status'})
    samples = samples.merge(reconstruction, on=['split', 'sample_id'], how='left', validate='one_to_one')
    for target in targets:
        if target in annotations:
            samples[f'label_{target}'] = annotations.reindex(samples.sample_id)[target].to_numpy()
    tables['posterior_samples'] = samples
    tables['distribution_summary'], tables['histograms'] = distribution_tables(
        samples, annotations, targets, config.histogram_edges)
    for name, frame in tables.items():
        if 'run_name' not in frame:
            frame.insert(0, 'run_name', run_name)
        frame.to_csv(directory / f'z_mc_{name}.csv', index=False)
    provenance['diagnostics'] = dict(
        mode='offline labeled evaluation, not controller inputs',
        confusion='true rows, predicted columns; mean/SD of per-draw row fractions; absent rows NaN; unseen true classes retained',
        geometry='Euclidean sampled z; same uniform capped subset per split/target for all draws; purity uses all labeled split samples',
        silhouette_singletons='scikit-learn convention: singleton label silhouette 0, constant full cloud unavailable',
        histogram='fixed configured nonnegative edges plus overflow bin; near-prior fractions use ten equal bins in [0,1]',
        reconstruction='MSE(final_decoder(z_draw), preprocessed X), averaged per sample over draws; not training ELBO distortion',
        near_prior='per-dimension conditional KL <= epsilon; descriptive, not a significant-collapse decision',
        missing_labels='excluded from geometry/confusion; explicit missing-label distribution group')
    if config.diagnostic_plots:
        plot_diagnostics(directory / 'plots', tables['confusion_summary'], tables['histograms'])
