import numpy as np
import pytest
import torch
from types import SimpleNamespace

from beta_vae_experiments.data_pipeline import (
    _to_numpy,
    prepare_metric_inputs_from_varix_result,
    save_test_posterior,
)


def test_to_numpy_unwraps_one_autoencodix_named_matrix() -> None:
    expected = np.arange(12, dtype=float).reshape(3, 4)
    actual = _to_numpy({"VARIX": expected})
    assert np.array_equal(actual, expected)


def test_to_numpy_rejects_ambiguous_named_matrices() -> None:
    with pytest.raises(ValueError, match="multiple named arrays"):
        _to_numpy({"first": np.zeros((2, 2)), "second": np.ones((2, 2))})


@pytest.mark.parametrize("epoch", [-1, 5])
def test_missing_test_mu_uses_exact_autoencodix_trainer_buffer(epoch, tmp_path) -> None:
    class Dynamics:
        def __init__(self, value=None):
            self.value = value

        def get(self, **_kwargs):
            return self.value

    dataset = SimpleNamespace(
        data=np.ones((3, 4)), sample_ids=np.asarray(["a", "b", "c"])
    )
    result = SimpleNamespace(
        datasets=SimpleNamespace(train=dataset, valid=dataset, test=dataset),
        sample_ids=Dynamics(np.asarray(["a", "b", "c"])),
        reconstructions=Dynamics(np.ones((3, 4))),
        mus=Dynamics(None),
        sigmas=Dynamics(None),
        latentspaces=Dynamics(np.ones((3, 2))),
    )
    exact_zero_mu = torch.zeros((3, 2))
    varix = SimpleNamespace(
        _trainer=SimpleNamespace(
            _mu_buffer={"test": exact_zero_mu},
            _sigma_buffer={"test": torch.zeros((3, 2))},
        )
    )

    splits = prepare_metric_inputs_from_varix_result(result, varix=varix, epoch=epoch)
    if epoch != -1:
        assert splits["test"]["mus"] is None
        assert splits["test"]["sigmas"] is None
        return

    assert np.array_equal(splits["test"]["mus"], exact_zero_mu.numpy())
    assert splits["test"]["mus_source"] == (
        "autoencodix_trainer_buffer_fallback"
    )
    assert splits["valid"]["mus"] is None
    folder = save_test_posterior(splits, tmp_path, "fully_collapsed")
    with np.load(folder / "posterior_test.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["mu"], exact_zero_mu.numpy())
        np.testing.assert_array_equal(saved["logvar"], exact_zero_mu.numpy())
        np.testing.assert_array_equal(saved["sample_ids"], ["a", "b", "c"])
    import pandas as pd
    metrics = pd.read_csv(folder / "collapse_metrics.csv").set_index("metric_name")
    assert metrics.loc["posterior_collapsed_dimensions_fraction_eps_0p01_delta_0p01", "value"] == 1.0


def test_missing_posterior_is_not_saved_as_collapse(tmp_path):
    with pytest.raises(ValueError, match="two-dimensional"):
        save_test_posterior({"test": {"mus": None, "sigmas": None}}, tmp_path, "missing")
    assert not (tmp_path / "posterior_snapshots").exists()


@pytest.mark.parametrize('logvar, expected_collapse', [(0.01, 1.0), (2.0, 0.0)])
def test_real_autoencodix_zero_sum_omission_and_collapse(logvar, expected_collapse, tmp_path):
    from autoencodix.trainers._general_trainer import GeneralTrainer
    from autoencodix.utils._traindynamics import TrainingDynamics
    import pandas as pd

    dataset = SimpleNamespace(data=np.ones((3, 4)))
    result = SimpleNamespace(datasets=SimpleNamespace(train=dataset, valid=dataset, test=dataset))
    for key in ('mus', 'sigmas', 'reconstructions', 'latentspaces', 'sample_ids'):
        setattr(result, key, TrainingDynamics())
    # An earlier train snapshot causes get(test, -1) to return an empty array.
    result.mus.add(epoch=5, split='train', data=np.ones((3, 2)))
    result.sigmas.add(epoch=5, split='train', data=np.ones((3, 2)))
    trainer = SimpleNamespace(
        _result=result,
        _mu_buffer={'test': torch.zeros((3, 2))},
        _sigma_buffer={'test': torch.full((3, 2), logvar)},
        _latentspace_buffer={'test': torch.ones((3, 2))},
        _reconstruction_buffer={'test': torch.ones((3, 4))},
        _sample_ids_buffer={'test': np.array(['a', 'b', 'c'])},
    )
    # Exercise the installed code without changing it.
    GeneralTrainer._dynamics_to_result(trainer, epoch=-1, split='test')
    assert result.mus.get(epoch=-1, split='test').size == 0
    splits = prepare_metric_inputs_from_varix_result(result, varix=SimpleNamespace(_trainer=trainer))
    np.testing.assert_array_equal(splits['test']['mus'], np.zeros((3, 2)))
    folder = save_test_posterior(splits, tmp_path, 'real_omission')
    metrics = pd.read_csv(folder / 'collapse_metrics.csv').set_index('metric_name')
    assert metrics.loc['posterior_collapsed_dimensions_fraction_eps_0p01_delta_0p01', 'value'] == expected_collapse


@pytest.mark.parametrize('corrupt', [False, True])
def test_fresh_encoder_verification_detects_incorrect_buffers(corrupt):
    from contextlib import nullcontext
    from beta_vae_experiments.data_pipeline import verify_test_posterior

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def encode(self, x):
            return x[:, :2] * self.weight, torch.full_like(x[:, :2], 0.01)

    model = Encoder()
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    test = {'x': x, 'mus': x[:, :2].copy(), 'sigmas': np.full((3, 2), 0.01)}
    if corrupt:
        test['mus'][1, 1] += 1
    trainer = SimpleNamespace(
        get_model=lambda: model, _config=SimpleNamespace(batch_size=2),
        _fabric=SimpleNamespace(autocast=nullcontext),
    )
    varix = SimpleNamespace(_trainer=trainer)
    if corrupt:
        with pytest.raises(AssertionError, match='disagrees'):
            verify_test_posterior({'test': test}, varix)
    else:
        assert verify_test_posterior({'test': test}, varix)['fresh_encoder_verified']
    assert model.training


@pytest.mark.parametrize('delta, outcome', [(0., 'ok'), (2.5331974e-7, 'warning'), (1e-3, 'error'), (np.nan, 'invalid'), (np.inf, 'invalid')])
def test_posterior_comparison_roundoff_policy(delta, outcome, tmp_path):
    from contextlib import nullcontext
    import json
    from beta_vae_experiments.data_pipeline import verify_test_posterior

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
        def encode(self, x):
            return x[:, :2] * self.weight, torch.zeros_like(x[:, :2])

    model = Encoder()
    x = np.full((3, 2), .01, dtype=np.float32)
    stored = x.copy(); stored[0, 0] += delta
    test = dict(x=x, mus=stored, sigmas=np.zeros_like(x), sample_ids=np.array(['a','b','c']),
                mus_source='result', sigmas_source='result')
    original = stored.copy()
    trainer = SimpleNamespace(get_model=lambda:model, _config=SimpleNamespace(batch_size=2),
                              _fabric=SimpleNamespace(autocast=nullcontext))
    args = ({'test':test}, SimpleNamespace(_trainer=trainer))
    if outcome == 'error':
        with pytest.raises(AssertionError, match='disagrees'): verify_test_posterior(*args)
    elif outcome == 'invalid':
        with pytest.raises(ValueError, match='Non-finite'): verify_test_posterior(*args)
    else:
        if outcome == 'warning':
            with pytest.warns(RuntimeWarning, match='strict diagnostic tolerance'):
                result = verify_test_posterior(*args)
            assert result['verification_status'] == 'numerical_warning'
            assert result['mus_diagnostic_exceedance_count'] == 1
        else:
            result = verify_test_posterior(*args)
            assert result['verification_status'] == 'ok'
        folder = save_test_posterior({'test':test}, tmp_path, 'roundoff', result)
        assert json.loads((folder/'provenance.json').read_text())['verification'] == result
        np.testing.assert_array_equal(np.load(folder/'posterior_test.npz')['mu'], original)
    np.testing.assert_array_equal(stored, original)
    assert model.training
