import numpy as np
import pytest
import torch
from types import SimpleNamespace
from beta_vae_experiments.metrics.curvature.encoder_geometry import (
    EncoderGeometryConfig, immersion_curvature, jacobian_spectrum,
    sampled_hyperbolicity, latent_geometry, encoder_geometry,
    run_encoder_geometry_metrics_pipeline)


@pytest.mark.parametrize('radius',[1.,2.])
def test_sphere_curvature_and_coordinate_invariance(radius):
    def sphere(t):
        u,v=t
        return radius*torch.stack((torch.cos(u)*torch.cos(v),torch.sin(u)*torch.cos(v),torch.sin(v)))
    point=torch.tensor([.3,.2],dtype=torch.float64)
    result=immersion_curvature(sphere,point,smooth=True)
    assert result['status']=='ok'
    assert result['sectional_01']==pytest.approx(1/radius**2)
    assert result['scalar_curvature']==pytest.approx(2/radius**2)
    transform=torch.tensor([[2.,.5],[.1,3.]],dtype=torch.float64)
    changed=immersion_curvature(lambda t:sphere(transform@t),torch.linalg.solve(transform,point),smooth=True)
    assert changed['scalar_curvature']==pytest.approx(result['scalar_curvature'])


def test_plane_cylinder_and_saddle_distinguish_intrinsic_curvature():
    p=torch.zeros(2,dtype=torch.float64)
    plane=lambda t:torch.stack((t[0],t[1],0*t[0]))
    cylinder=lambda t:torch.stack((torch.cos(t[0]),torch.sin(t[0]),t[1]))
    saddle=lambda t:torch.stack((t[0],t[1],(t[0]**2-t[1]**2)/2))
    assert immersion_curvature(plane,p,smooth=True)['scalar_curvature']==pytest.approx(0)
    c=immersion_curvature(cylinder,p,smooth=True)
    assert c['scalar_curvature']==pytest.approx(0)
    assert c['second_fundamental_form_squared']==pytest.approx(1)
    assert immersion_curvature(saddle,p,smooth=True)['scalar_curvature']==pytest.approx(-2)


def test_degenerate_and_nonsmooth_never_get_fake_curvature():
    f=lambda t:torch.stack((t[0],0*t[1]))
    p=torch.zeros(2,dtype=torch.float64)
    assert immersion_curvature(f,p,smooth=True)['reason']=='rank_deficient_pullback'
    assert immersion_curvature(f,p,smooth=False)['reason']=='C2_regularity_not_established'
    stats,_=jacobian_spectrum([[2.,0.,0.],[0.,0.,0.]])
    assert stats['rank']==1 and stats['input_nullity']==2
    assert not stats['full_input_metric_positive_definite']


def test_four_point_tree_cycle_and_scale():
    tree=np.abs(np.arange(4)[:,None]-np.arange(4)[None,:])
    cycle=np.minimum(tree,4-tree)
    q=np.array([[0,1,2,3]])
    assert sampled_hyperbolicity(tree,q)['delta_lower_bound']==0
    assert sampled_hyperbolicity(cycle,q)['delta_lower_bound']==1
    assert sampled_hyperbolicity(cycle*7,q)['delta_over_diameter']==sampled_hyperbolicity(cycle,q)['delta_over_diameter']


def test_radial_collapse_and_zero_angular_are_not_hidden():
    x=np.arange(30).reshape(10,3)
    mu=np.zeros((10,2))
    result,_=latent_geometry(x,mu,np.arange(10),EncoderGeometryConfig(max_samples=10,k_values=(5,),quadruples=20))
    table=result.set_index('metric_name')
    assert table.loc['mu_zero_norm_fraction','value']==1
    assert table.loc['mu_angular_preservation_spearman','status']=='unavailable'
    assert table.loc['mu_to_input_distance_ratio_median','value']==0


class ToyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.tensor([[2.,0.,0.],[0.,3.,0.]]))
    def encode(self,x):
        mu=x@self.weight.T
        return mu,torch.zeros_like(mu)
    def decode(self,*args):
        raise AssertionError('Decoder must never be called')


def test_encoder_spectra_and_posterior_metrics_preserve_model():
    model=ToyEncoder().train()
    x=np.ones((2,3))
    mu=np.array([[2.,3.],[2.,3.]])
    config=EncoderGeometryConfig()
    stats,_=encoder_geometry(model,x,['a','b'],config,mu,np.zeros_like(mu))
    assert model.training and model.weight.dtype==torch.float32
    assert (stats['rank']==2).all()
    assert (stats['frobenius_squared']==13).all()
    assert (stats['input_nullity']==1).all()
    with pytest.raises(ValueError,match='Checkpoint mu'):
        encoder_geometry(model,x,['a','b'],config,mu+1,np.zeros_like(mu))


def test_adapter_reports_missing_checkpoint(tmp_path):
    rng=np.random.default_rng(5)
    x=rng.normal(size=(30,3));mu=x[:,:2]
    inputs=SimpleNamespace(x_test=x,mu_test=mu,sample_ids_test=np.arange(30),
                           run_name='test',metadata={},logvar_test=None)
    cfg=EncoderGeometryConfig(enabled=True,max_samples=20,k_values=(5,),quadruples=10)
    result=run_encoder_geometry_metrics_pipeline(inputs,np.arange(30),x,mu,cfg,tmp_path)
    table=result.set_index('metric_name')
    assert table.loc['encoder_mu_operator_norm_median','status']=='missing_checkpoint'
    assert table.loc['encoder_intrinsic_scalar_curvature','status']=='not_defined'
    assert (tmp_path/'metrics_artifacts/test/encoder_geometry/provenance.json').exists()


def test_posterior_variance_changes_fisher_and_w2_differently():
    class VariableVariance(ToyEncoder):
        def encode(self,x):
            return x@self.weight.T, np.log(4)+torch.stack((x[:,0],2*x[:,1]),dim=1)
    stats,_=encoder_geometry(VariableVariance(),np.zeros((1,3)),['a'],EncoderGeometryConfig())
    table=stats.set_index('representation')
    assert table.loc['mu','frobenius_squared']==pytest.approx(13)
    assert table.loc['posterior_w2','frobenius_squared']==pytest.approx(18)
    assert table.loc['posterior_fisher','frobenius_squared']==pytest.approx(5.75)
