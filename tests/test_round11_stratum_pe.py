"""Independent local moments and differentiable covariance closure checks."""
import numpy as np
import pytest
import torch

from models.round10_partition_pe import prepare_support
from models.round11_stratum_pe import prepare_gaussian_strata,gaussian_stratum_logits
from experiments.round10_cell_probe import geometry_logits


def fixture(dtype=torch.float64):
    y,x=np.mgrid[:4,:8]
    xy=np.stack([x/4+.03*y*y,y/4+.02*x*y],-1).reshape(-1,2)
    factor=np.stack([.02+0*x,.003+0*x,.01+0*x,.03+0*x],-1).reshape(-1,2,2)
    pc=factor@factor.transpose(0,2,1)
    owner=np.broadcast_to(np.arange(8)//4,(4,8)).reshape(-1)
    return (torch.tensor(xy,dtype=dtype),torch.tensor(pc,dtype=dtype),
            torch.tensor(owner),torch.tensor([[0,0,4,4],[4,0,8,4]]),torch.tensor([0,0]))


def reference(xy,pc,bounds):
    means=[];cov=[]
    xy=xy.reshape(4,8,2);pc=pc.reshape(4,8,2,2)
    for x0,y0,x1,y1 in bounds:
        local_means=[];local_cov=[]
        for dy in range(2):
            for dx in range(2):
                a=x0+dx*(x1-x0)//2;b=x0+(dx+1)*(x1-x0)//2
                c=y0+dy*(y1-y0)//2;d=y0+(dy+1)*(y1-y0)//2
                points=xy[c:d,a:b].reshape(-1,2);m=points.mean(0)
                local_means.append(m)
                local_cov.append(pc[c:d,a:b].mean((0,1))+(points-m).T@(points-m)/len(points))
        means.append(local_means);cov.append(local_cov)
    return np.asarray(means),np.asarray(cov)


def test_independent_rectangular_pixel_slice_moments():
    args=fixture();q=prepare_gaussian_strata(*args)
    mean,cov=reference(args[0].numpy(),args[1].numpy(),args[3].numpy())
    np.testing.assert_allclose(q['nodes'].numpy(),mean,rtol=2e-14,atol=2e-14)
    np.testing.assert_allclose(q['node_cov'].numpy(),cov,rtol=2e-14,atol=2e-14)
    torch.testing.assert_close(q['node_weights'],torch.full((2,4),.25,dtype=torch.float64),rtol=0,atol=0)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_same_quad4_means_weights_and_total_covariance_identity(dtype):
    args=fixture(dtype);q=prepare_gaussian_strata(*args)
    quad=prepare_support(*args,samples=4,include_full_moments=False)
    full=prepare_support(*args,samples=1,include_full_moments=True)
    for key in ('mean','mass','parent','nodes','node_weights'):
        torch.testing.assert_close(q[key],quad[key],rtol=0,atol=0)
    mean=(q['nodes']*q['node_weights'][...,None]).sum(1)
    delta=q['nodes']-mean[:,None]
    cov=((q['node_cov']+delta[...,None]*delta[...,None,:])*q['node_weights'][...,None,None]).sum(1)
    atol=3e-7 if dtype==torch.float32 else 2e-14
    torch.testing.assert_close(mean,full['mean'],rtol=atol,atol=atol)
    torch.testing.assert_close(cov,full['cov'],rtol=atol,atol=atol)


def test_zero_scatter_reduces_to_original_quad4_operator():
    args=list(fixture());args[0]=torch.ones_like(args[0])*.3
    q=prepare_gaussian_strata(*args)
    quad=prepare_support(*args,samples=4,include_full_moments=False)
    key=dict(cell_bounds=torch.tensor([[0.,0.,1.,1.],[1.,0.,2.,1.]],dtype=torch.float64),
             mass=torch.tensor([64.,64.],dtype=torch.float64))
    torch.testing.assert_close(q['node_cov'],quad['node_cov'],rtol=0,atol=0)
    torch.testing.assert_close(gaussian_stratum_logits(q,key,'full4',.25),
        geometry_logits(quad,key,'quad4',.25),rtol=0,atol=0)


def test_full4_is_same_mixture_operator_with_covariance_replacement_only():
    q=prepare_gaussian_strata(*fixture())
    key=dict(cell_bounds=torch.tensor([[0.,0.,1.,1.],[1.,0.,2.,1.]],dtype=torch.float64),
             mass=torch.tensor([64.,64.],dtype=torch.float64))
    torch.testing.assert_close(gaussian_stratum_logits(q,key,'full4',.25),
        geometry_logits(q,key,'quad4',.25),rtol=0,atol=0)
    iso=dict(q)
    iso['node_cov']=q['node_cov'].diagonal(dim1=-2,dim2=-1).mean(-1)[...,None,None]*torch.eye(2,dtype=torch.float64)
    torch.testing.assert_close(gaussian_stratum_logits(q,key,'iso4',.25),
        geometry_logits(iso,key,'quad4',.25),rtol=0,atol=0)


def test_position_scatter_and_pixel_covariance_gradients():
    xy,pc,owner,bounds,parent=fixture()
    xy=xy.requires_grad_();scale=torch.tensor(.8,dtype=torch.float64,requires_grad=True)
    def f(positions,s):
        q=prepare_gaussian_strata(positions,pc*s.square(),owner,bounds,parent)
        return q['node_cov']
    assert torch.autograd.gradcheck(f,(xy,scale),eps=1e-6,atol=1e-7,rtol=1e-5)
    value=f(xy,scale).square().sum();gx,gs=torch.autograd.grad(value,(xy,scale))
    assert torch.isfinite(gx).all() and gx.abs().sum()>0
    assert torch.isfinite(gs) and gs.abs()>0


def test_numerical_fallback_is_shared_background_density():
    q=prepare_gaussian_strata(*fixture());q['nodes']=q['nodes']+1000
    key=dict(cell_bounds=torch.tensor([[0.,0.,1.,1.],[1.,0.,3.,1.]],dtype=torch.float64),
             mass=torch.tensor([64.,128.],dtype=torch.float64))
    diagnostics={};p=gaussian_stratum_logits(q,key,'full4',.25,diagnostics=diagnostics).softmax(-1)
    torch.testing.assert_close(p,torch.tensor([[1/3,2/3],[1/3,2/3]],dtype=torch.float64),rtol=2e-14,atol=2e-14)
    assert diagnostics['raw_zero_fov_rows']==2
    with pytest.raises(ValueError):gaussian_stratum_logits(q,key,'adaptive8',.25)
