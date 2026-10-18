"""Repeated posterior draws: ID alignment, train-only fit and honest aggregation."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from beta_vae_experiments.metrics.evaluation.sampled_z import (
    DownstreamMetricConfig, evaluate_sampled_z, load_final_posteriors,
    run_sampled_z_metrics,
)


def fixture_data():
    rng = np.random.default_rng(2)
    posteriors, labels = {}, []
    for split in ('train', 'valid', 'test'):
        ids = np.array([f'{split}-{i}' for i in range(60)])
        y = np.tile(['A', 'B', 'C'], 20)
        mu = np.eye(3)[np.tile(np.arange(3), 20)] + rng.normal(0, .2, (60, 3))
        posteriors[split] = dict(mu=mu, logvar=np.full_like(mu, -.5), sample_ids=ids)
        labels.append(pd.DataFrame({'CANCER_TYPE_ACRONYM': y, 'AGE': np.arange(60, dtype=float)}, index=ids))
    return posteriors, pd.concat(labels).iloc[::-1]


def test_draws_reproduce_manual_train_only_probe_and_sampling_variation():
    posterior, labels = fixture_data()
    config = DownstreamMetricConfig(n_draws=4)
    first = evaluate_sampled_z(posterior, labels, ('CANCER_TYPE_ACRONYM', 'AGE'), config)
    second = evaluate_sampled_z(posterior, labels, ('CANCER_TYPE_ACRONYM', 'AGE'), config)
    for a, b in zip(first[:4], second[:4]):
        pd.testing.assert_frame_equal(a, b)
    features = {}
    for index, (split, data) in enumerate(posterior.items()):
        rng = np.random.default_rng(np.random.SeedSequence([31, 0, index]))
        features[split] = data['mu'] + np.exp(.5 * data['logvar']) * rng.standard_normal(data['mu'].shape)
    model = LogisticRegression(max_iter=1000, random_state=31).fit(
        features['train'], labels.loc[posterior['train']['sample_ids'], 'CANCER_TYPE_ACRONYM'])
    expected = roc_auc_score(labels.loc[posterior['test']['sample_ids'], 'CANCER_TYPE_ACRONYM'],
                             model.predict_proba(features['test']), multi_class='ovo', labels=model.classes_)
    details, summary = first[:2]
    actual = details.query("draw == 0 and split == 'test' and target == 'CANCER_TYPE_ACRONYM'").value.item()
    assert actual == pytest.approx(expected)
    group = details.query("split == 'test' and target == 'CANCER_TYPE_ACRONYM'")
    aggregate = summary.query("split == 'test' and target == 'CANCER_TYPE_ACRONYM'").iloc[0]
    assert aggregate['mean'] == pytest.approx(group.value.mean())
    assert aggregate['std'] == pytest.approx(group.value.std(ddof=1))
    assert aggregate['std'] > 0
    changed = labels.copy()
    changed.loc[posterior['test']['sample_ids'], 'CANCER_TYPE_ACRONYM'] = 'A'
    third = evaluate_sampled_z(posterior, changed, ('CANCER_TYPE_ACRONYM', 'AGE'), config)
    pd.testing.assert_frame_equal(first[0].query("split == 'valid'").reset_index(drop=True),
                                  third[0].query("split == 'valid'").reset_index(drop=True))


def test_missing_classes_and_labels_are_explicit_not_successfully_averaged():
    posterior, labels = fixture_data()
    ids = posterior['test']['sample_ids']
    labels.loc[ids, 'CANCER_TYPE_ACRONYM'] = 'A'
    labels.loc[ids[0], 'CANCER_TYPE_ACRONYM'] = None
    labels.loc[ids[1], 'CANCER_TYPE_ACRONYM'] = 'unseen'
    details, summary, classes, _, _ = evaluate_sampled_z(
        posterior, labels, ('CANCER_TYPE_ACRONYM', 'ABSENT'), DownstreamMetricConfig(n_draws=2))
    test = details.query("split == 'test' and target == 'CANCER_TYPE_ACRONYM'")
    assert test.value.isna().all()
    assert (test.n_missing == 1).all() and (test.n_unseen_class == 1).all()
    assert summary.query("split == 'test'")['mean'].isna().all()
    assert (summary.query("split == 'test'").n_valid_draws == 0).all()
    assert classes.query("split == 'test'").value.isna().all()


@pytest.mark.parametrize('case', ['overlap', 'missing_id', 'nan', 'shape'])
def test_invalid_posteriors_rejected(case):
    posterior, labels = fixture_data()
    if case == 'overlap':
        posterior['test']['sample_ids'][0] = posterior['train']['sample_ids'][0]
    elif case == 'missing_id':
        labels = labels.drop(posterior['test']['sample_ids'][0])
    elif case == 'nan':
        posterior['test']['logvar'][0, 0] = np.nan
    else:
        posterior['test']['mu'] = posterior['test']['mu'][:-1]
    with pytest.raises(ValueError):
        evaluate_sampled_z(posterior, labels, ('AGE',), DownstreamMetricConfig(n_draws=2))


def test_final_posterior_files_and_output_contract(tmp_path):
    posterior, labels = fixture_data()
    directory = tmp_path / 'run_analysis' / 'example' / 'posteriors'
    directory.mkdir(parents=True)
    for split, data in posterior.items():
        np.savez_compressed(directory / f'posterior_{split}.npz', **data)
    loaded, hashes = load_final_posteriors(directory)
    np.testing.assert_array_equal(loaded['train']['mu'], posterior['train']['mu'])
    assert all(len(h) == 64 for h in hashes.values())
    varix = SimpleNamespace(result=object(), visualizer=SimpleNamespace(_collect_all_metadata=lambda result: labels))
    metrics = run_sampled_z_metrics(varix, 'example', tmp_path,
                                    ('CANCER_TYPE_ACRONYM',), DownstreamMetricConfig(n_draws=2))
    assert len(metrics) == 4
    assert not metrics.metric_name.duplicated().any()
    assert metrics.metric_name.str.contains('_z_mc_').all()
    output = tmp_path / 'absolute_core_metrics_v1' / 'downstream' / 'example'
    assert len(pd.read_csv(output / 'z_mc_draws.csv')) == 4
    assert len(pd.read_csv(output / 'z_mc_class_summary.csv')) == 6
    assert (output / 'z_mc_provenance.json').exists()


@pytest.mark.parametrize('kwargs', [{'n_draws': 1}, {'n_draws': True}, {'n_draws': -2},
                                    {'sampling_seed': -1}, {'probe_seed': 2**32}, {'max_iter': 0}])
def test_configuration_validation(kwargs):
    with pytest.raises(ValueError):
        DownstreamMetricConfig(**kwargs)
