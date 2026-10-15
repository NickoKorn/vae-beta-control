"""Validate shared-grid Landscape before enabling the final experiment sweep."""
import json
import numpy as np
import pytest
from beta_vae_experiments.metrics.topology.landscape import (
    LandscapeConfig, compare_landscapes, run_landscape_metrics,
)
from beta_vae_experiments.metrics.topology import compute_persistence_diagrams_h0_h1
from sklearn.metrics import pairwise_distances


def test_identical_loop_and_shared_grid():
    x=np.array([[0.1,0.8],[0.2,0.4]])
    value,vectors,metadata=compare_landscapes(x,x.copy())
    assert value==pytest.approx(0,abs=1e-12)
    assert vectors.shape==(2,500)
    assert len(metadata['grid'])==100 and metadata['shared_fit']
    y=np.array([[0.1,0.9]])
    value,vectors,metadata=compare_landscapes(x,y)
    assert value>0
    assert metadata['sample_range']==pytest.approx([0.1,0.9])


def test_empty_and_infinite_classes():
    empty=np.empty((0,2))
    value,_,meta=compare_landscapes(empty,empty)
    assert np.isnan(value) and meta['status']=='degenerate_input'
    value,_,meta=compare_landscapes(empty,np.array([[0.,1.]]))
    assert np.isfinite(value) and value>0
    pd=np.array([[0.,np.inf],[0.,.5]])
    value,_,meta=compare_landscapes(pd,pd)
    assert value==pytest.approx(0)
    assert meta['removed_infinite']==[1,1] and meta['finite_points']==[1,1]


def test_zero_lifetimes_resolution_and_determinism():
    x=np.array([[0.,0.]])
    value,_,meta=compare_landscapes(x,x)
    assert value==0 and meta['reason']
    x=np.array([[0.,1.]]);y=np.array([[0.,.8]])
    a,va,ma=compare_landscapes(x,y)
    b,vb,mb=compare_landscapes(x,y)
    assert a==b and np.array_equal(va,vb)
    _,v,m=compare_landscapes(x,y,LandscapeConfig(resolution=200))
    assert v.shape==(2,1000) and m['config']['num_landscapes']==5
    assert ma['sample_range']==m['sample_range']
    with pytest.raises(ValueError):LandscapeConfig(resolution=0)


def test_homology_separate_and_metadata(tmp_path):
    diagrams={f'topology_{space}_h{k}_persistence_diagram':np.array([[0.,1.]]) for space in ['input','latent'] for k in [0,1]}
    diagrams['topology_latent_h1_persistence_diagram']=np.array([[0.,.6]])
    frame=run_landscape_metrics(diagrams,20,tmp_path,dict(sample_ids=[str(i) for i in range(20)]))
    values=frame.set_index('metric_name').value
    assert values['topology_landscape_h0_l2']==0
    assert values['topology_landscape_h1_l2']>0
    meta=json.loads((tmp_path/'metadata.json').read_text())
    assert set(meta['homology'])=={'0','1'}


def test_circle_normalization_preservation_vs_destroyed():
    theta=np.linspace(0,2*np.pi,40,endpoint=False)
    x=np.column_stack([np.cos(theta),np.sin(theta)])
    dx=pairwise_distances(x)
    # Existing VR builder normalizes each space by its own diameter.
    preserved=compute_persistence_diagrams_h0_h1(dx,dx*7,dx.max(),dx.max()*7)
    destroyed=np.column_stack([x[:,0],np.zeros(len(x))])
    dm=pairwise_distances(destroyed)
    damaged=compute_persistence_diagrams_h0_h1(dx,dm,dx.max(),dm.max())
    a=run_landscape_metrics(preserved,len(x)).set_index('metric_name').value
    b=run_landscape_metrics(damaged,len(x)).set_index('metric_name').value
    assert a['topology_landscape_h1_l2']==pytest.approx(0,abs=1e-10)
    assert b['topology_landscape_h1_l2']>a['topology_landscape_h1_l2']
