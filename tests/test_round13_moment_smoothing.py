import math

import pytest
import torch

from models.round13_moment_smoothing import smooth_gaussian_strata, RHO_SQUARED


def sample():
    generator=torch.Generator().manual_seed(611)
    nodes=torch.randn(3,4,2,dtype=torch.float64,generator=generator)
    a=torch.randn(3,4,2,2,dtype=torch.float64,generator=generator)
    cov=a@a.transpose(-1,-2)
    weights=torch.tensor([[.1,.2,.3,.4],[.25]*4,[0.,.4,.1,.5]],dtype=torch.float64)
    return dict(nodes=nodes,node_cov=cov,node_weights=weights)


def independent_moments(query):
    # E[XX^T]-E[X]E[X]^T, distinct from the centered implementation.
    x,c,w=(query[k] for k in ('nodes','node_cov','node_weights'))
    mean=torch.einsum('ns,nsd->nd',w,x)
    raw=torch.einsum('ns,nsij->nij',w,c+x[...,None]*x[...,None,:])
    return mean,raw-mean[...,None]*mean[...,None,:]


def test_weighted_moments_psd_and_no_input_mutation():
    query=sample(); before={k:v.clone() for k,v in query.items()}
    actual=smooth_gaussian_strata(query)
    for a,b in zip(independent_moments(query),independent_moments(actual)):
        torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
    assert torch.linalg.eigvalsh(actual['node_cov']).min()>=0
    for key in query:torch.testing.assert_close(query[key],before[key],rtol=0,atol=0)
    assert actual['node_weights'] is query['node_weights']


def test_affine_equivariance():
    query=sample();a=torch.tensor([[2.,.7],[-.3,1.2]],dtype=torch.float64);b=torch.tensor([4.,-3.])
    moved={**query,'nodes':query['nodes']@a.T+b,'node_cov':a@query['node_cov']@a.T}
    base=smooth_gaussian_strata(query); result=smooth_gaussian_strata(moved)
    torch.testing.assert_close(result['nodes'],base['nodes']@a.T+b)
    torch.testing.assert_close(result['node_cov'],a@base['node_cov']@a.T)


def test_gradients_through_means_covariances_and_weights():
    q=sample()
    nodes=q['nodes'].requires_grad_(); raw=torch.eye(2,dtype=torch.float64).expand(3,4,2,2).clone().requires_grad_()
    weight_logits=torch.zeros(3,4,dtype=torch.float64,requires_grad=True)
    def fn(x,a,w):
        out=smooth_gaussian_strata(dict(nodes=x,node_cov=a@a.transpose(-1,-2),node_weights=w.softmax(-1)))
        return torch.cat([out['nodes'].flatten(),out['node_cov'].flatten()])
    assert torch.autograd.gradcheck(fn,(nodes,raw,weight_logits))


def test_identical_components_stay_identical():
    x=torch.tensor([2.,3.],dtype=torch.float64).expand(1,4,2)
    c=torch.tensor([[2.,.2],[.2,1.]],dtype=torch.float64).expand(1,4,2,2)
    q=dict(nodes=x,node_cov=c,node_weights=torch.full((1,4),.25,dtype=torch.float64))
    out=smooth_gaussian_strata(q)
    torch.testing.assert_close(out['nodes'],x)
    torch.testing.assert_close(out['node_cov'],c)


def test_ideal_square_threshold_and_fixed_value():
    assert RHO_SQUARED==2/3
    for side in (8.,16.,24.,96.):
        a=side/4;c=side**2/48
        q=dict(nodes=torch.tensor([[[-a,-a],[a,-a],[-a,a],[a,a]]],dtype=torch.float64),
               node_cov=torch.eye(2,dtype=torch.float64).expand(1,4,2,2)*c,
               node_weights=torch.full((1,4),.25,dtype=torch.float64))
        out=smooth_gaussian_strata(q)
        torch.testing.assert_close(out['nodes'].abs(),torch.full((1,4,2),math.sqrt(2/3)*a,dtype=torch.float64))
        assert torch.all(out['node_cov'].diagonal(dim1=-2,dim2=-1)+1e-12>=out['nodes'].square())


@pytest.mark.parametrize('problem',['count','negative_weight','weight_sum','asymmetric','negative_cov','nonfinite','empty'])
def test_invalid_input_rejected(problem):
    q=sample()
    if problem=='count':q={k:v[:,:3] for k,v in q.items()}
    elif problem=='negative_weight':q['node_weights'][0]=torch.tensor([-.1,.3,.3,.5])
    elif problem=='weight_sum':q['node_weights']*=2
    elif problem=='asymmetric':q['node_cov'][0,0,0,1]+=1
    elif problem=='negative_cov':q['node_cov'][0,0]=-torch.eye(2)
    elif problem=='nonfinite':q['nodes'][0,0,0]=float('nan')
    elif problem=='empty':q={k:v[:0] for k,v in q.items()}
    with pytest.raises(ValueError):smooth_gaussian_strata(q)
