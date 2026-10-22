"""Schedule boundaries, native loss coefficients and correlated-input invariants."""
import numpy as np
import pytest
import torch
from autoencodix.configs.varix_config import VarixConfig
from autoencodix.utils._model_output import ModelOutput
from beta_vae_experiments.beta_annealing import EpochBetaSchedule, make_scheduled_loss
from beta_vae_experiments.datasets import generate_synthetic_dataset


@pytest.mark.parametrize('kind', ['linear', 'cosine', 'step'])
def test_excursion_returns_to_low_and_preserves_plateaus(kind):
    schedule = EpochBetaSchedule(kind, .01, 3, 2, 4, 5, 4, 6)
    factors = schedule.factors()
    assert len(factors) == 41
    assert np.all(factors[:3] == .01)
    assert np.all(factors[7:12] == 1)
    assert factors[-1] == pytest.approx(.01)
    assert factors.min() >= .01 and factors.max() <= 1
    assert np.allclose(factors[3:22], factors[22:])


def test_native_loss_uses_low_high_and_recovery_coefficients():
    settings = dict(kind='cosine', beta_low_over_d=.001, initial_hold=2,
                    cycles=1, ramp_up=2, high_hold=2, ramp_down=2, low_hold=2)
    config = VarixConfig(latent_dim=4, beta=.4, anneal_function='3phase-linear',
                         epochs=10, loss_reduction='mean', batch_size=5, default_vae_loss='kl')
    loss = make_scheduled_loss(settings, config)(config)
    mu = torch.ones(5,4, requires_grad=True)
    out = ModelOutput(reconstruction=torch.zeros(5,3), latentspace=mu,
                      latent_mean=mu, latent_logvar=torch.zeros(5,4))
    for epoch, coefficient in [(0,.004), (4,.4), (9,.004), (None,.004)]:
        total, parts = loss(out, torch.zeros(5,3), epoch=epoch)
        assert parts['effective_beta_factor'] == pytest.approx(coefficient)
        assert total.item() == pytest.approx(coefficient * .5)
    total.backward()
    assert mu.grad is not None and torch.isfinite(mu.grad).all()


def test_duration_mismatch_fails_before_training():
    config = VarixConfig(latent_dim=4, beta=.4, anneal_function='3phase-linear', epochs=3, loss_reduction='mean', default_vae_loss='kl')
    with pytest.raises(ValueError, match='duration'):
        make_scheduled_loss(dict(kind='cosine', beta_low_over_d=.001, initial_hold=2,
                                 cycles=1, ramp_up=2, high_hold=2, ramp_down=2, low_hold=2), config)


def test_correlated_observations_reproducible_and_keep_ground_truth():
    options = dict(n_samples=100, dataset_seed=3, correlated_copies=8, observation_noise=.02)
    first = generate_synthetic_dataset('swiss_roll', options)
    second = generate_synthetic_dataset('swiss_roll', options)
    assert first.x_raw.shape == (100,24)
    assert np.array_equal(first.x_raw, second.x_raw)
    assert first.sample_metadata.equals(second.sample_metadata)
    assert np.corrcoef(first.x_raw.iloc[:,0], first.x_raw.iloc[:,3])[0,1] > .99
    noisy = generate_synthetic_dataset('swiss_roll', dict(options, observation_noise=1.))
    assert noisy.sample_metadata.equals(first.sample_metadata)
    assert noisy.x_raw.shape == first.x_raw.shape
    assert np.corrcoef(noisy.x_raw.iloc[:,0], noisy.x_raw.iloc[:,3])[0,1] < .8


def test_long_run_names_fit_filesystem_and_remain_unique():
    from beta_vae_experiments.new_beta_varix_experiments import _run_name
    first = _run_name('experiment_' * 30, {'seed': 13})
    second = _run_name('experiment_' * 30, {'seed': 17})
    assert len(first) <= 220 and first != second


def test_explicit_results_root_is_not_overwritten_by_date(tmp_path):
    import json
    from beta_vae_experiments.new_beta_varix_experiments import load_experiment_config
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(dict(results_date='2026-10-14', experiment_name='test',
                                    output_dir='ignored', results_root=str(tmp_path / 'drive'))))
    assert load_experiment_config(path)['output_dir'] == tmp_path / 'drive/new_results_2026-10-14/test'


def test_unmounted_results_location_fails_before_training(tmp_path):
    import json
    from beta_vae_experiments.new_beta_varix_experiments import load_experiment_config
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(dict(output_dir='ignored', require_results_mount=str(tmp_path))))
    with pytest.raises(FileNotFoundError, match='not mounted'):
        load_experiment_config(path)


def test_recovery_suite_crosses_splits_and_training_seeds(tmp_path):
    import json
    from beta_vae_experiments.user_interface.config import CATALOG, load_config
    from beta_vae_experiments.result_analysis.prepare_annealing_recovery import prepare
    from beta_vae_experiments.new_beta_varix_experiments import build_training_runs
    suite = load_config(str(CATALOG) + '#annealing.recovery')
    suite['datasets'] = [next(d for d in suite['datasets'] if d['name'] == 'circle')]
    source = tmp_path / 'suite.json'
    source.write_text(json.dumps(suite))
    destination = tmp_path / 'prepared'
    inventory = prepare(source, destination)
    assert len(inventory) == 3 * len(suite['schedule_kinds'])
    assert set(inventory.split_seed) == {2, 3, 5}
    assert inventory.config.is_unique
    assert inventory.epochs.eq(1000).all()
    names = set()
    for row in inventory.itertuples():
        config = json.loads((destination / 'configs' / row.config).read_text())
        assert config['dataset']['split_seed'] == row.split_seed
        assert config['dataset']['options']['dataset_seed'] == 42
        assert config['training']['seeds'] == [13, 17, 23, 31]
        assert config['experiment_name'] not in names
        names.add(config['experiment_name'])
        runs = build_training_runs(config['experiment_name'], config['training'])
        assert len(runs) == row.runs
        assert {run['seed'] for run in runs} == {13, 17, 23, 31}
    suite['split_seeds'] = [2, 7]
    source.write_text(json.dumps(suite))
    with pytest.raises(ValueError, match='held-out'):
        prepare(source, tmp_path / 'invalid')
