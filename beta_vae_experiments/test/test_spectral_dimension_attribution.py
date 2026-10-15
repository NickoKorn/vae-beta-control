"""Scientific invariants, not replicas of the calculation code."""
import numpy as np
import pytest
from beta_vae_experiments.metrics.graph_signals.dimension_attribution import (
    fingerprints,align_rows,reference_bands,match_dimensions,probe_ablation)


def fixture_basis(n=12):
    U=np.linalg.qr(np.random.default_rng(31).normal(size=(n,n)))[0]
    lam=np.r_[0,0,np.repeat(.2,3),np.linspace(.4,1.2,n-5)]
    return U,lam,[np.arange(2),np.arange(2,5),np.arange(5,n)]


def test_energy_full_and_truncated_parseval():
    U,lam,bands=fixture_basis();mu=np.random.default_rng(1).normal(size=(12,4));lv=np.zeros_like(mu)
    full=fingerprints(U,lam,mu,lv,bands)
    assert np.allclose(full['energy'].sum(1),((mu-mu.mean(0))**2).sum(0))
    assert np.allclose(full['fraction'].sum(1),1)
    truncated=fingerprints(U[:,:5],lam[:5],mu,lv,bands[:2])
    assert np.all(truncated['energy'][:,-1]>0)
    assert np.allclose(truncated['fraction'].sum(1),1)


def test_constant_mu_is_not_structure():
    U,lam,bands=fixture_basis();mu=np.ones((12,4))*3
    p=fingerprints(U,lam,mu,np.zeros_like(mu),bands)
    assert np.all(p['energy']==0) and np.isnan(p['fraction']).all()
    assert not p['active'].any() and np.isnan(p['centroid']).all()


def test_signs_and_degenerate_rotations_do_not_change_band_energy():
    U,lam,bands=fixture_basis();mu=np.random.default_rng(2).normal(size=(12,3))
    ref=fingerprints(U,lam,mu,np.zeros_like(mu),bands)
    V=U.copy();V[:,0]*=-1;V[:,2:5]=U[:,2:5]@np.linalg.qr(np.random.default_rng(3).normal(size=(3,3)))[0]
    actual=fingerprints(V,lam,mu,np.zeros_like(mu),bands)
    for key in ['energy','fraction','centroid','entropy']:assert np.allclose(ref[key],actual[key],equal_nan=True)


def test_id_alignment_and_failures():
    assert np.array_equal(align_rows(np.array([[2],[1]]),['b','a'],['a','b']),[[1],[2]])
    with pytest.raises(ValueError,match='Duplicate'):align_rows(np.ones((2,1)),['a','a'],['a'])
    with pytest.raises(ValueError,match='Missing'):align_rows(np.ones((1,1)),['a'],['b'])
    U,lam,bands=fixture_basis()
    with pytest.raises(ValueError,match='shapes'):fingerprints(U,lam,np.ones((11,2)),np.ones((11,2)),bands)
    with pytest.raises(ValueError,match='partition'):fingerprints(U,lam,np.ones((12,2)),np.ones((12,2)),[np.arange(12),np.array([0])])


def test_nullspace_and_degenerate_band_partition():
    lam=np.r_[0,0,1e-2,np.repeat(.1,4),1.]
    bands,names=reference_bands(lam)
    assert np.array_equal(bands[0],[0,1]) and names[0]=='B0'
    assert sorted(np.concatenate(bands))==list(range(len(lam)))
    assert sum(np.isin(np.arange(3,7),b).any() for b in bands)==1


def test_dimension_matching_handles_permutation_and_collapse():
    mu=np.random.default_rng(4).normal(size=(50,3));other=mu[:,[2,0,1]]*[-1,1,-1]
    matching,quality=match_dimensions(mu,other)
    assert np.array_equal(matching,[2,0,1]) and np.allclose(quality,1)
    other[:,0]=0;matching,quality=match_dimensions(mu,other)
    assert matching[0]==-1 and np.isnan(quality[0])


def test_fixed_probe_ablation_of_task_signal():
    rng=np.random.default_rng(5);x=rng.normal(size=(240,3));y=(x[:,0]>0).astype(int)
    rows=probe_ablation(x[:160],x[160:],y[:160],y[160:],[('signal',[0]),('noise',[1])])
    assert rows[0]['delta_auc_fixed']>.3 and rows[1]['delta_auc_fixed']<.05
    assert rows[0]['delta_auc_retrained']>.3


def test_sample_centered_nullspace_energy_is_retained():
    # Two disjoint components: a component contrast is constant on each
    # component but not over the whole sample set. Keep it in B0.
    a=np.array([1,1,0,0])/np.sqrt(2);b=np.array([0,0,1,1])/np.sqrt(2)
    U=np.column_stack([a,b,np.array([1,-1,0,0])/np.sqrt(2),np.array([0,0,1,-1])/np.sqrt(2)])
    mu=np.array([[1],[1],[-1],[-1]])
    p=fingerprints(U,np.array([0,0,1,1]),mu,np.zeros_like(mu),[np.array([0,1]),np.array([2,3])])
    assert np.isclose(p['fraction'][0,0],1) and np.isclose(p['energy'][0,1],0)


def test_ridge_band_reconstructability_is_rotation_invariant():
    from beta_vae_experiments.metrics.graph_signals.dimension_attribution import ridge_reconstruct
    rng=np.random.default_rng(10);mu=rng.normal(size=(60,4));U=np.linalg.qr(np.column_stack([mu[:,:2],rng.normal(size=(60,2))]))[0]
    bands=[np.array([0,1]),np.array([2,3])]
    rotation=np.array([[.6,-.8],[.8,.6]]);V=U.copy();V[:,:2]=U[:,:2]@rotation
    _,a=ridge_reconstruct(U,mu,bands);_,b=ridge_reconstruct(V,mu,bands)
    assert np.allclose(a,b,atol=1e-8)
