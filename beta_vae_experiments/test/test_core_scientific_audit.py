import numpy as np
import pandas as pd
import pytest

from beta_vae_experiments.metrics.topology import compute_persistence_entropy


def test_persistence_entropy_known_barcodes():
    diagrams = {f'topology_{space}_h{dim}_persistence_diagram': diagram
                for space, diagram in [('input', np.array([[0., 1.], [0., 1.], [0., np.inf]])),
                                       ('latent', np.array([[0., 4.]]))]
                for dim in (0, 1)}
    result = compute_persistence_entropy(diagrams, 3).set_index('metric_name')
    assert result.loc['topology_h0_persistence_entropy_x', 'value'] == pytest.approx(np.log(2))
    assert result.loc['topology_h1_persistence_entropy_mu', 'value'] == pytest.approx(0)
    diagrams['topology_latent_h1_persistence_diagram'] = np.empty((0, 2))
    result = compute_persistence_entropy(diagrams, 3).set_index('metric_name')
    assert np.isnan(result.loc['topology_h1_persistence_entropy_mu', 'value'])


def test_native_losses_use_exact_epoch_and_no_mse_fallback():
    from types import SimpleNamespace
    from autoencodix.utils._result import Result
    from beta_vae_experiments.metrics.evaluation.native_losses import run_native_loss_metrics_pipeline
    result = Result()
    result.losses.add(epoch=1, split='valid', data=7.)
    result.sub_losses.add(epoch=1, split='valid', data={'recon_loss': .123456, 'var_loss': 6.876544})
    # A later stored epoch must not replace the configured final epoch.
    result.sub_losses.add(epoch=2, split='valid', data={'recon_loss': 99.})
    inputs = SimpleNamespace(metadata={'epochs': 2}, x_train=np.zeros((4, 2)), x_val=np.zeros((3, 2)))
    rows = run_native_loss_metrics_pipeline(inputs, SimpleNamespace(result=result)).set_index('metric_name')
    assert rows.loc['autoencodix_valid_recon_loss', 'value'] == .123456
    assert np.isnan(rows.loc['autoencodix_train_recon_loss', 'value'])
    assert rows.loc['autoencodix_train_recon_loss', 'status'] == 'unavailable'


def test_full_core_on_small_synthetic_data(tmp_path):
    from types import SimpleNamespace
    from beta_vae_experiments.metric_pipeline import (
        CoreMetricInputs, AbsoluteCoreMetricConfig, run_absolute_core_metrics_pipeline,
    )
    rng = np.random.default_rng(19)
    x = rng.normal(size=(40, 3))
    mu = x[:, :2]
    inputs = CoreMetricInputs(run_name='full', x_train=x, x_val=x, x_test=x,
        x_recon_train=x, x_recon_val=x, x_recon_test=x,
        mu_train=mu, mu_val=mu, mu_test=mu, logvar_test=np.zeros_like(mu),
        metadata={'dataset_family': 'synthetic_manifold'})
    result = run_absolute_core_metrics_pipeline(inputs,
        AbsoluteCoreMetricConfig(output_dir=tmp_path),
        SimpleNamespace())
    assert not result.metric_name.duplicated().any()
    assert not result.metric_name.str.startswith('dtm_').any()
    assert not result.metric_name.str.startswith('variance_by_task_').any()
    rows = result.set_index('metric_name')
    for name in ('topology_h0_persistence_entropy_mu',
                 'posterior_conditional_kl_total_mean',
                 'posterior_collapsed_dimensions_fraction_eps_0p01_delta_0p01',
                 'dimensionality_mu_twonn',
                 'topology_landscape_h0_l2', 'topology_landscape_h1_l2'):
        assert np.isfinite(rows.loc[name, 'value']), name
    assert not any(name.startswith('topology_cknn_') for name in rows.index)
    assert not (tmp_path / 'absolute_core_metrics_v1/cknn_topology').exists()
    assert not any(name.startswith('posterior_pairwise_gaussian_hellinger_') for name in rows.index)
    assert 'posterior_snr_mean' not in rows.index
    assert not any(name.startswith('local_topology_') for name in rows.index)
    folder = tmp_path / 'absolute_core_metrics_v1/local_topology/full'
    local_rows = pd.read_csv(folder / 'local_topology_per_anchor.csv')
    assert len(local_rows) == 2 * len(x)
    assert local_rows.status.eq('ok').all()
