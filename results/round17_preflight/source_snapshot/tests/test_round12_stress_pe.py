import numpy as np
import pytest
from scipy.stats import _mvn
import torch

from models import round12_stress_pe as M
from models.round10_cell_integral import gaussian_cell_mass
from models.round10_partition_pe import prepare_support


def test_24px_tiled_integral_independent_scipy_and_robust():
    rng=np.random.default_rng(120024)
    means=rng.uniform(-2,5,(31,2))
    factors=rng.normal(size=(31,2,2))*rng.choice([0.,.01,.5,3.,16.],(31,1,1))
    covariance=factors@factors.transpose(0,2,1)
    # Actual pixel edges include the -0.5 offset; two 24px rectangles.
    native=np.array([[0,0,24,24],[24,0,48,24]])
    child,owner=M.split_cell_bounds(native)
    key={'cell_bounds':torch.tensor(child),'cell_owner':torch.tensor(owner),
         'mass':torch.tensor([576.,576.])}
    physical=(native-.5)/8
    for h in (.25,.5,1.,2.):
        got=M.tiled_cell_mass(torch.tensor(means),torch.tensor(covariance),key,h).numpy()
        reference=np.array([[_mvn.mvnun(b[:2],b[2:],mu,c+h*h*np.eye(2),100000,1.e-13)[0]
            for b in physical] for mu,c in zip(means,covariance)])
        np.testing.assert_allclose(got,reference,atol=2.e-12,rtol=2.e-8)
        robust=gaussian_cell_mass(torch.tensor(means),torch.tensor(covariance),torch.tensor(physical),bandwidth=h)
        np.testing.assert_allclose(got,robust.numpy(),atol=2.e-12,rtol=2.e-8)
    assert np.max(child[:,2:]-child[:,:2]) == 1.5


def test_24px_tiled_gradients():
    child,owner=M.split_cell_bounds([[0,0,24,24]])
    key={'cell_bounds':torch.tensor(child),'cell_owner':torch.tensor(owner),'mass':torch.tensor([576.])}
    mean=torch.tensor([[.4,.7]],dtype=torch.float64,requires_grad=True)
    factor=torch.tensor([[[.3,.1],[.05,.4]]],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda m,a:M.tiled_cell_mass(m,a@a.transpose(-1,-2),key,.25),
                                   (mean,factor),atol=1.e-6,rtol=1.e-4)


def test_partition_coverage_content_and_24_stratum_moments():
    parts=M.partitions();p=M.partition_tensors(parts,'cpu')['g24']
    assert len(p['mean']) == 48 and int(p['mass'].sum()) == 192*144
    assert bool((p['parent']==-1).all())
    assert np.array_equal(np.bincount(parts['g24']['atoms']),np.full(48,9))
    yy,xx=torch.meshgrid(torch.arange(144,dtype=torch.float64),torch.arange(192,dtype=torch.float64),indexing='ij')
    xy=torch.stack([xx,yy],-1).reshape(-1,2)
    pixel_cov=torch.eye(2,dtype=torch.float64).expand(len(xy),2,2)/12
    features=torch.arange(432*3,dtype=torch.float64).reshape(432,3)+1
    obs={'q_to_k_xy':xy,'q_to_k_pixel_cov':pixel_cov,'q_features':features}
    full=M.prepare_query(obs,p,'full4');quad=M.prepare_query(obs,p,'quad4')
    torch.testing.assert_close(full['nodes'],quad['nodes'],atol=0,rtol=0)
    torch.testing.assert_close(full['node_weights'],torch.full((48,4),.25,dtype=torch.float64))
    expected=torch.eye(2,dtype=torch.float64)*(12**2/12/64)
    torch.testing.assert_close(full['node_cov'],expected.expand(48,4,2,2))
    reference=prepare_support(xy/8,pixel_cov/64,p['owner'],p['bounds'],p['parent'])
    average=(full['nodes']*full['node_weights'][...,None]).sum(1)
    delta=full['nodes']-average[:,None]
    covariance=((full['node_cov']+delta[...,None]*delta[...,None,:])*full['node_weights'][...,None,None]).sum(1)
    torch.testing.assert_close(average,reference['mean'],atol=2.e-14,rtol=0)
    torch.testing.assert_close(covariance,reference['cov'],atol=2.e-14,rtol=0)


def synthetic_inputs():
    z=np.ones((144,192))*2
    K=np.array([[100.,0,96],[0,100.,72],[0,0,1]])
    pixels=M.D.pixel_centers();rays=np.c_[pixels,np.ones(len(pixels))]@np.linalg.inv(K).T
    points=rays*z.ravel()[:,None]
    a=np.linalg.inv(K)[:,:2];cov=np.broadcast_to(4*(a@a.T)/12,(len(pixels),3,3)).copy()
    T=np.eye(4);T[:3,3]=[.02,.03,.01]
    xy,cov2,_=M.D.project_finite_support(points+T[:3,3],cov,K)
    return dict(q_points3=points,q_pixel_cov3=cov,q_to_k_transform=T,k_K=K,
        q_to_k_xy=xy,q_to_k_pixel_cov=cov2,q_features=np.ones((432,2)),
        k_features=np.ones((432,2)),q_predicted_depth=z,k_predicted_depth=z*1.5)


def test_perturbations_reproject_points_and_pixel_footprints():
    inputs=synthetic_inputs()
    zero,diagnostic=M.perturbed_observations(inputs,'zero')
    assert diagnostic['zero_xy_max_error']==0
    assert diagnostic['zero_cov_max_error']==0
    for state in M.STATES[1:]:
        obs,diag=M.perturbed_observations(inputs,state)
        assert np.isfinite(obs['q_to_k_xy']).all()
        assert not np.array_equal(obs['q_to_k_xy'],zero['q_to_k_xy'])
        assert diag['key_depth_used_by_positional_operator'] is False
        if state.startswith('depth'):
            factor=.9 if state=='depth09' else 1.1
            assert diag['effective_depth_sha256']['k']==M.array_sha(inputs['k_predicted_depth']*factor)
        if state.startswith('ry'):
            R=np.asarray(diag['transform'])[:3,:3]
            np.testing.assert_allclose(np.asarray(diag['transform'])[:3,3],R@inputs['q_to_k_transform'][:3,3])


def test_unsupported_key_extent_rejected():
    with pytest.raises(ValueError):M.split_cell_bounds([[0,0,24,16]])
