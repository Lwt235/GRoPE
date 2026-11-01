"""Compression controls and conditional-region semantics, without real labels."""
import numpy as np
import pytest
from scipy.stats import entropy
import torch

from experiments.round11_compression_diagnostic import component_bank,discrepancy,integrate,mixture_moments
from models.round10_cell_integral_fast import gaussian_cell_mass_fast


def synthetic_field():
    yy,xx=np.mgrid[:8,:16]
    # Two spatial strata with different geometry, retaining correlated scatter.
    xy=np.stack([xx/8+.3*(yy>=4),yy/8+.1*np.sin(xx)],-1)
    covariance=np.broadcast_to(np.array([[.003,.001],[.001,.01]]),(8,16,2,2)).copy()
    return torch.tensor(xy),torch.tensor(covariance),np.array([[0,0,8,8],[8,0,16,8]])


@pytest.mark.parametrize('method',['full1','full4','iso4','quad4','quad16'])
def test_component_compression_retains_declared_moments_and_all_input_mass(method):
    xy,within,bounds=synthetic_field()
    dense=component_bank(xy,within,bounds,'dense');compressed=component_bank(xy,within,bounds,method)
    dm,dc=mixture_moments(dense);cm,cc=mixture_moments(compressed)
    torch.testing.assert_close(cm,dm,atol=1e-14,rtol=1e-14)
    for i in range(2):assert compressed['weights'][compressed['owner']==i].sum()==1
    if method in ('full1','full4'):torch.testing.assert_close(cc,dc,atol=1e-14,rtol=1e-14)
    elif method=='iso4':
        torch.testing.assert_close(cc.diagonal(dim1=-2,dim2=-1).sum(-1),dc.diagonal(dim1=-2,dim2=-1).sum(-1))
        assert (cc-dc).abs().max()>.001
    else:
        assert torch.linalg.eigvalsh(dc-cc).min()>1e-4
    expected=1 if method=='full1' else 16 if method=='quad16' else 4
    assert torch.bincount(compressed['owner']).tolist()==[expected,expected]


def test_region_mixture_is_normalized_after_summing_components():
    means=torch.tensor([[-.3,0.],[2.,0.]],dtype=torch.float64)
    cov=torch.eye(2,dtype=torch.float64).expand(2,2,2)*.04
    bounds=torch.tensor([[-1.,-1.,0.,1.],[0.,-1.,1.,1.]],dtype=torch.float64)
    bank=dict(mean=means,cov=cov,weights=torch.tensor([.5,.5]),owner=torch.zeros(2,dtype=torch.long),queries=1)
    with torch.no_grad():
        probability,fov=integrate(bank,bounds,.25)
        raw=gaussian_cell_mass_fast(means,cov,bounds,bandwidth=.25)
    expected=raw.mean(0);expected=expected/expected.sum()
    torch.testing.assert_close(probability[0],expected)
    torch.testing.assert_close(fov[0],raw.mean(0).sum())
    separately_normalized=(raw/raw.sum(-1,keepdim=True)).mean(0)
    assert (separately_normalized-probability[0]).abs().max()>.1


def test_known_probability_discrepancies_match_scipy():
    p=np.array([[.75,.25],[.5,.5]]);q=np.array([[.5,.5],[.5,.5]])
    kl,js,l1=discrepancy(p,q)
    np.testing.assert_allclose(kl,[entropy(p[0],q[0]),0],atol=1e-15)
    mid=(p[0]+q[0])/2
    np.testing.assert_allclose(js,[.5*(entropy(p[0],mid)+entropy(q[0],mid)),0],atol=1e-15)
    np.testing.assert_allclose(l1,[.5,0],atol=0)
    with pytest.raises(ValueError):discrepancy(p,q*.9)


def test_unknown_representation_and_nonuniform_strata_rejected():
    xy,within,bounds=synthetic_field()
    with pytest.raises(ValueError):component_bank(xy,within,bounds,'selected_after_observing_scores')
    with pytest.raises(ValueError):component_bank(xy,within,np.array([[0,0,7,8]]),'quad4')
