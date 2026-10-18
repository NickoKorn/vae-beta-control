"""Offline-Klassendiagnostik: Bibliotheksabgleich und vollständige Artefakte."""
from contextlib import nullcontext
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import silhouette_samples, confusion_matrix

from beta_vae_experiments.metrics.evaluation.class_diagnostics import (
    confusion_rows, summarize_confusions, label_geometry, posterior_sample_table,
    distribution_tables, make_reconstruction_callback,
)
from beta_vae_experiments.metrics.evaluation.sampled_z import (
    DownstreamMetricConfig, evaluate_sampled_z, run_sampled_z_metrics,
)


def example():
    rng = np.random.default_rng(7)
    posteriors, frames = {}, []
    for split in ('train', 'valid', 'test'):
        y = np.tile(['A', 'B', 'C'], 12)
        mu = np.eye(3)[np.tile(np.arange(3), 12)] * 2 + rng.normal(0, .2, (36, 3))
        ids = np.array([f'{split}-{i}' for i in range(36)])
        posteriors[split] = dict(mu=mu, logvar=np.full_like(mu, -1.), sample_ids=ids)
        frames.append(pd.DataFrame({'CANCER_TYPE_ACRONYM': y, 'SEX': np.tile(['F', 'M'], 18),
                                    'AGE': np.arange(36)}, index=ids))
    return posteriors, pd.concat(frames).iloc[::-1]


def test_confusion_matches_sklearn_and_keeps_unseen_and_absent_classes():
    truth, predictions = ['A', 'A', 'B', 'unseen'], ['B', 'A', 'B', 'A']
    table = confusion_rows(truth, predictions, ['A', 'B', 'absent'])
    classes = sorted({'A', 'B', 'absent', 'unseen'})
    np.testing.assert_array_equal(table.pivot(index='true_class', columns='predicted_class', values='count'),
                                  confusion_matrix(truth, predictions, labels=classes))
    assert table.query("true_class == 'absent'").fraction.isna().all()
    assert not table.query("true_class == 'unseen'").trained_class.any()
    table = table.assign(target='cancer', split='test', draw=0)
    second = table.copy().assign(draw=1)
    result = summarize_confusions(pd.concat([table, second]), 2)
    fractions = result.query("true_class != 'absent'").groupby('true_class').fraction_mean.sum()
    np.testing.assert_allclose(fractions, 1)
    assert result.query("true_class == 'absent'").fraction_mean.isna().all()


def test_silhouette_and_purity_match_manual_values_without_self_neighbors():
    x = np.array([[0., 0.], [.1, 0.], [.2, 0.], [5., 0.], [5.1, 0.], [5.2, 0.]])
    y = np.array(['A']*3 + ['B']*3)
    samples, summary = label_geometry(x, y, np.arange(6), k=2, max_samples=6, seed=31)
    np.testing.assert_allclose(samples.silhouette, silhouette_samples(x, y))
    np.testing.assert_allclose(samples.neighbor_purity, 1.)
    np.testing.assert_allclose(samples.purity_chance, .4)
    np.testing.assert_allclose(samples.neighbor_purity_excess, .6)
    assert (summary.status == 'ok').all()


def test_geometry_constant_duplicates_and_missing_labels():
    samples, summary = label_geometry(np.ones((6, 2)), ['A']*3+['B']*3,
                                      np.arange(6), k=2, max_samples=6, seed=31)
    assert summary.value.isna().all()
    assert samples.neighbor_purity.isna().all()
    x = np.array([[0., 0.], [0., 0.], [1., 0.], [2., 0.]])
    samples, summary = label_geometry(x, ['A', 'A', 'B', None], np.arange(4),
                                      k=2, max_samples=4, seed=31)
    assert len(samples) == 3
    assert summary.warning.str.contains('Duplicate').all()
    assert np.isfinite(samples.neighbor_purity).all()
    _, small = label_geometry(x, ['A', 'A', 'B', None], np.arange(4),
                              k=3, max_samples=4, seed=31)
    assert small.query("metric == 'neighbor_purity'").value.isna().all()


def test_geometry_subset_deterministic_and_counts_exposed():
    rng = np.random.default_rng(13)
    x = rng.normal(size=(100, 3))
    y = np.tile(['A', 'B'], 50)
    first, summary = label_geometry(x, y, np.arange(100), k=5, max_samples=20, seed=31)
    second, _ = label_geometry(x*2, y, np.arange(100), k=5, max_samples=20, seed=31)
    np.testing.assert_array_equal(first.silhouette_selected, second.silhouette_selected)
    assert first.silhouette.notna().sum() == 20
    assert summary.query("metric == 'silhouette'").n_samples.item() == 20


def test_analytic_kl_and_fixed_histograms_include_overflow_and_missing_labels():
    posteriors, labels = example()
    for data in posteriors.values():
        data['mu'][:] = 0
        data['logvar'][:] = 0
    samples = posterior_sample_table(posteriors, labels, .01)
    assert (samples.kl_total == 0).all()
    assert (samples.all_dimensions_near_prior == 1).all()
    samples['reconstruction_mse_mean'] = 0.
    samples.loc[0, 'reconstruction_mse_mean'] = 1e12
    labels.loc[samples.sample_id.iloc[0], 'CANCER_TYPE_ACRONYM'] = None
    summary, hist = distribution_tables(samples, labels, ['CANCER_TYPE_ACRONYM'], (0., .01, 1.))
    assert summary.missing_label.any()
    overflow = hist.query("metric == 'reconstruction_mse_mean' and missing_label").iloc[-1]
    assert np.isinf(overflow.upper) and overflow['count'] == 1
    groups = hist.groupby(['target', 'split', 'class', 'missing_label', 'metric'], dropna=False)
    np.testing.assert_array_equal(groups['count'].sum(), groups.n_valid.first())


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(3))

    def decode(self, z):
        return z @ self.weight


def varix_fixture(posteriors, labels):
    model = Decoder()
    datasets = SimpleNamespace(**{split: SimpleNamespace(sample_ids=data['sample_ids'],
        data=torch.as_tensor(data['mu'], dtype=torch.float32)) for split, data in posteriors.items()})
    trainer = SimpleNamespace(get_model=lambda: model, _config=SimpleNamespace(batch_size=7),
                               _fabric=SimpleNamespace(autocast=nullcontext))
    varix = SimpleNamespace(_trainer=trainer, result=SimpleNamespace(datasets=datasets),
        visualizer=SimpleNamespace(_collect_all_metadata=lambda result: labels))
    return varix, model


def test_decoder_mse_alignment_batching_and_training_state():
    posteriors, labels = example()
    varix, model = varix_fixture(posteriors, labels)
    reconstruct = make_reconstruction_callback(varix, posteriors)
    z = posteriors['test']['mu'] + 1
    result = reconstruct('test', z)
    np.testing.assert_allclose(result, 1., atol=1e-6)
    assert model.training
    varix.result.datasets.test.sample_ids = posteriors['test']['sample_ids'][::-1]
    with pytest.raises(ValueError, match='IDs differ'):
        make_reconstruction_callback(varix, posteriors)


def test_full_file_evaluation_with_torch_decoder_and_plot_exports(tmp_path):
    posteriors, labels = example()
    varix, _ = varix_fixture(posteriors, labels)
    posterior_dir = tmp_path / 'run_analysis' / 'example' / 'posteriors'
    posterior_dir.mkdir(parents=True)
    for split, data in posteriors.items():
        np.savez_compressed(posterior_dir / f'posterior_{split}.npz', **data)
    config = DownstreamMetricConfig(n_draws=3, diagnostic_k=5, silhouette_max_samples=25)
    metrics = run_sampled_z_metrics(varix, 'example', tmp_path, ('CANCER_TYPE_ACRONYM', 'SEX'), config)
    assert len(metrics) == 8
    directory = tmp_path / 'absolute_core_metrics_v1' / 'downstream' / 'example'
    for name in ['confusions', 'confusion_summary', 'most_confused_classes', 'geometry_draws',
                 'geometry_summary', 'geometry_class_summary', 'posterior_samples',
                 'distribution_summary', 'histograms', 'reconstruction_draws']:
        assert len(pd.read_csv(directory / f'z_mc_{name}.csv')) > 0
    samples = pd.read_csv(directory / 'z_mc_posterior_samples.csv')
    assert 'label_CANCER_TYPE_ACRONYM' in samples
    assert samples.reconstruction_mse_mean.notna().all()
    assert (samples.reconstruction_mse_mean > 0).all()
    details = pd.read_csv(directory / 'z_mc_reconstruction_draws.csv')
    for split in ('valid', 'test'):
        for draw in range(3):
            rng = np.random.default_rng(np.random.SeedSequence([31, draw, 1 if split == 'valid' else 2]))
            data = posteriors[split]
            z = data['mu'] + np.exp(.5*data['logvar'])*rng.standard_normal(data['mu'].shape)
            expected = np.mean((z.astype(np.float32).astype(float)-data['mu'].astype(np.float32))**2, axis=1)
            actual = details[(details.split == split) & (details.draw == draw)].value
            np.testing.assert_allclose(actual, expected, atol=1e-7)
    assert len(list((directory / 'plots').glob('confusion_*.png'))) == 4
    assert len(list((directory / 'plots').glob('histogram_*.png'))) == 8
    manifest = json.loads((directory / 'z_mc_provenance.json').read_text())
    assert manifest['decoder_warning'] is None
    assert 'offline' in manifest['diagnostics']['mode']


def test_disabled_diagnostics_do_not_change_scores():
    posteriors, labels = example()
    enabled = evaluate_sampled_z(posteriors, labels, ('SEX',), DownstreamMetricConfig(n_draws=2), diagnostics={})
    disabled = evaluate_sampled_z(posteriors, labels, ('SEX',),
        DownstreamMetricConfig(n_draws=2, diagnostics_enabled=False), diagnostics={})
    for a, b in zip(enabled[:4], disabled[:4]):
        pd.testing.assert_frame_equal(a, b)


def test_missing_decoder_and_regression_only_leave_explicit_unavailable_diagnostics(tmp_path):
    posteriors, labels = example()
    directory = tmp_path / 'run_analysis' / 'example' / 'posteriors'
    directory.mkdir(parents=True)
    for split, data in posteriors.items():
        np.savez_compressed(directory / f'posterior_{split}.npz', **data)
    varix = SimpleNamespace(result=object(),
        visualizer=SimpleNamespace(_collect_all_metadata=lambda result: labels))
    scores = run_sampled_z_metrics(varix, 'example', tmp_path, ('AGE',),
        DownstreamMetricConfig(n_draws=2, diagnostic_plots=True))
    assert np.isfinite(scores.value).all()
    output = tmp_path / 'absolute_core_metrics_v1' / 'downstream' / 'example'
    samples = pd.read_csv(output / 'z_mc_posterior_samples.csv')
    assert samples.reconstruction_mse_mean.isna().all()
    manifest = json.loads((output / 'z_mc_provenance.json').read_text())
    assert 'unavailable' in manifest['decoder_warning']


def test_catalog_exposes_validated_downstream_settings():
    from beta_vae_experiments.user_interface.config import load_config, CATALOG
    config = DownstreamMetricConfig(**load_config(CATALOG)['metrics']['downstream'])
    assert config.n_draws == 10 and config.diagnostic_k == 20
    assert config.diagnostics_enabled and config.diagnostic_plots


@pytest.mark.parametrize('kwargs', [{'diagnostic_k': 0}, {'silhouette_max_samples': 2},
    {'near_prior_epsilon': -1}, {'histogram_edges': [0, 1, 1]}, {'histogram_edges': [1, 2]},
    {'diagnostic_plots': 1}])
def test_diagnostic_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        DownstreamMetricConfig(**kwargs)
