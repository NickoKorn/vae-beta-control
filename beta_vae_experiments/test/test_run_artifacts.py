import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from autoencodix.configs.varix_config import VarixConfig
from autoencodix.utils._losses import VarixLoss
from autoencodix.utils._model_output import ModelOutput
from beta_vae_experiments.run_artifacts import RecordedVarixLoss, verify_split_manifest
from beta_vae_experiments.new_beta_varix_experiments import load_experiment_config


@pytest.mark.parametrize('beta', [0., .03])
def test_loss_and_gradients_unchanged_with_diagnostics(beta):
    config = VarixConfig(beta=beta, latent_dim=2, batch_size=3,
                         loss_reduction='mean', default_vae_loss='kl', anneal_function='no-annealing')
    mu = torch.ones(3,2, requires_grad=True)
    lv = torch.zeros(3,2, requires_grad=True)
    reconstruction = torch.ones(3,4, requires_grad=True)
    output = ModelOutput(reconstruction=reconstruction, latent_mean=mu,
                         latent_logvar=lv, latentspace=mu)
    target = torch.zeros(3,4)
    original,_ = VarixLoss(config)(model_output=output, targets=target)
    recorded,parts = RecordedVarixLoss(config)(model_output=output, targets=target)
    assert original.item() == recorded.item()
    for a,b in zip(torch.autograd.grad(original,(mu,lv,reconstruction),retain_graph=True),
                   torch.autograd.grad(recorded,(mu,lv,reconstruction))):
        torch.testing.assert_close(a,b)
    assert parts['kl_total_per_sample'].item() == 3
    assert parts['reconstruction_mse_per_sample'].item() == 3


def test_resume_rejects_changed_split(tmp_path):
    data=SimpleNamespace(**{k:SimpleNamespace(sample_ids=[k]) for k in ['train','valid','test']})
    verify_split_manifest(data,tmp_path)
    verify_split_manifest(data,tmp_path)
    data.test.sample_ids=['different']
    with pytest.raises(ValueError,match='refusing resume'):
        verify_split_manifest(data,tmp_path)


def test_dated_output_is_stable_and_explicit(tmp_path):
    path=tmp_path/'config.json'
    path.write_text(json.dumps({'results_date':'2026-09-29','experiment_name':'example','output_dir':'old'}))
    output=load_experiment_config(path)['output_dir']
    assert output.parts[-2:]==('new_results_2026-09-29','example')
