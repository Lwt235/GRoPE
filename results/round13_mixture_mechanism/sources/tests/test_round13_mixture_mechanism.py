import numpy as np
import torch
from scipy.special import ndtr

from experiments.round11_compression_diagnostic import component_bank,integrate
from experiments.round13_mixture_mechanism import (
    predicted_centers,content_scores,add_content,center_statistics,spatial_bounds,smooth_bank,QUERIES,
)


def test_exact_fixed_selection():
    np.testing.assert_array_equal(QUERIES,[0,6,13,20,26,33,40,47])


def test_predicted_center_uses_its_ray_and_bilinear_depth():
    y,x=np.mgrid[:144,:192];depth=1+.01*x+.02*y
    K=np.array([[100.,0,96],[0,100.,72],[0,0,1.]])
    trans=np.eye(4);trans[0,3]=.2
    inputs=dict(q_predicted_depth=depth,q_K=K,k_K=K,q_to_k_transform=trans)
    bounds=spatial_bounds(24)[QUERIES]
    center=(bounds[:,:2]+bounds[:,2:]-1)/2
    z=1+.01*center[:,0]+.02*center[:,1]
    expected=center.copy();expected[:,0]+=20/z
    actual,front,fov=predicted_centers(inputs,bounds)
    np.testing.assert_allclose(actual,expected,atol=1e-12)
    assert front.all()
    assert len(fov)==8


def test_center_validity_does_not_filter_distribution_rows():
    p=np.full((2,432),1/432.)
    rows=center_statistics(p,np.array([[3.5,3.5],[-8.,0.]]),np.array([True,False]))
    assert len(rows)==2 and rows[0]['center_bin']==0
    assert rows[1]['center_log_margin'] is None
    assert rows[1]['argmax']==0


def test_shared_parent_content_and_softmax_reference():
    rng=np.random.default_rng(12)
    q=rng.normal(size=(432,7));k=rng.normal(size=(432,7));bounds=spatial_bounds(24)[QUERIES]
    scores=content_scores(q,k,bounds)
    raster=q.reshape(18,24,7)
    manual=raster[:3,:3].mean((0,1));manual/=np.linalg.norm(manual)
    np.testing.assert_allclose(scores[0],manual@(k/np.linalg.norm(k,axis=1)[:,None]).T)
    p=np.full((8,432),1/432.)
    got=add_content(p,scores,5)
    ref=np.exp(5*scores-5*scores.max(-1,keepdims=True));ref/=ref.sum(-1,keepdims=True)
    np.testing.assert_allclose(got,ref)
    np.testing.assert_allclose(add_content(p,scores,0),p)


def test_identity_continuous_valley_does_not_imply_discrete_full4_failure():
    # Independent analytic marginal CDF, with the correct within-pixel moment.
    y,x=np.mgrid[:24,:24]
    xy=torch.tensor(np.stack([x,y],-1)/8,dtype=torch.float64)
    within=torch.eye(2,dtype=torch.float64).expand(24,24,2,2)/(12*64)
    box=np.array([[0,0,24,24]])
    full=component_bank(xy,within,box,'full4')
    quad=component_bank(xy,within,box,'quad4')
    keys=torch.tensor((np.array([[a,b,a+8,b+8] for b in range(-80,104,8) for a in range(-80,104,8)])-.5)/8,dtype=torch.float64)
    p,_=integrate(full,keys,.5)
    side=23;centers=(keys[:,:2]+keys[:,2:]).numpy()/2*8-11.5
    assert np.linalg.norm(centers[p[0].argmax()])==0
    q,_=integrate(quad,keys,.5)
    np.testing.assert_allclose(np.linalg.norm(centers[q[0].argmax()]),np.sqrt(128))
    variance=12+16
    assert 6**2>variance  # Continuous full4 center is nevertheless a local minimum.
    marginal=np.mean(ndtr((4-np.array([-6.,6.]))/np.sqrt(variance))-
                     ndtr((-4-np.array([-6.,6.]))/np.sqrt(variance)))
    np.testing.assert_allclose(p.max().item(),marginal**2,atol=1e-12)
    smoothed=smooth_bank(full)
    ps,_=integrate(smoothed,keys,.5)
    assert ps.max()>p.max()
