"""Bounded-domain fast integral, independent of benchmark metrics."""
import numpy as np
import pytest
from scipy.stats import _mvn
import torch

from models.round10_cell_integral import gaussian_cell_mass
from models.round10_cell_integral_fast import gaussian_cell_mass_fast


def tensor(x):
    return torch.tensor(x, dtype=torch.float64)


def test_full_correlations_against_scipy_and_robust_operator():
    rng = np.random.default_rng(101032)
    m = rng.uniform(-2., 2., (25, 2))
    theta = rng.uniform(-np.pi, np.pi, 25)
    r = np.stack([np.cos(theta), -np.sin(theta), np.sin(theta), np.cos(theta)], -1).reshape(25, 2, 2)
    eigenvalues = np.exp(rng.uniform(np.log(1.e-8), np.log(256.), (25, 2)))
    c = (r*eigenvalues[:, None])@r.transpose(0, 2, 1)
    low = rng.uniform(-2., 2., (11, 2))
    bounds = np.concatenate([low, low+rng.choice([1., 2.], (11, 2))], -1)
    args = tuple(tensor(v) for v in (m, c, bounds))
    actual = gaussian_cell_mass_fast(*args)
    robust = gaussian_cell_mass(*args)
    expected = np.array([[_mvn.mvnun(b[:2], b[2:], mu, cov+.25**2*np.eye(2),
        100000, 1.e-13)[0] for b in bounds] for mu, cov in zip(m, c)])
    np.testing.assert_allclose(actual.numpy(), expected, rtol=2.e-8, atol=2.e-12)
    torch.testing.assert_close(actual, robust, rtol=2.e-8, atol=2.e-12)


@pytest.mark.parametrize('rho', [-1., -.999, 0., .999, 1.])
def test_rank_one_covariances_and_smallest_bandwidth(rho):
    m = tensor([[0., 0.], [.9, .1]])
    c = tensor([[[256., 256*rho], [256*rho, 256.]], [[1.e-10, 0.], [0., 1.e-10]]])
    b = tensor([[-1., -1., 1., 1.], [.01, .01, .02, .02], [-2., .3, 0., 2.3]])
    actual = gaussian_cell_mass_fast(m, c, b)
    robust = gaussian_cell_mass(m, c, b)
    torch.testing.assert_close(actual, robust, rtol=3.e-9, atol=2.e-12)


def test_actual_subdivision_conservation():
    m = tensor([[.2, -.7], [1., .3]])
    c = tensor([[[.03, .018], [.018, .02]], [[12., -8.], [-8., 8.]]])
    parent = tensor([[-1., -1., 1., 1.]])
    children = tensor([[-1., -1., 0., 0.], [-1., 0., 0., 1.], [0., -1., 1., 0.], [0., 0., 1., 1.]])
    a = gaussian_cell_mass_fast(m, c, parent)
    b = gaussian_cell_mass_fast(m, c, children).sum(-1, keepdim=True)
    torch.testing.assert_close(a, b, rtol=2.e-10, atol=2.e-12)


def test_no_global_tail_truncation_and_diagonal_fast_path():
    m = tensor([[0., 0.]])
    c = tensor([[[1., 0.], [0., 1.]]])
    b = tensor([[12., -.5, 12.1, .5]])
    mass = gaussian_cell_mass_fast(m, c, b)
    assert mass.item() > 0
    # Requires-grad forces quadrature; its value agrees with analytic fast path.
    q = gaussian_cell_mass_fast(m.clone().requires_grad_(), c, b)
    torch.testing.assert_close(mass, q, rtol=2.e-12, atol=0.)


def test_gradient_including_zero_correlation():
    m = tensor([[.2, -.4]]).requires_grad_()
    factor = tensor([[[.6, 0.], [0., .7]]]).requires_grad_()
    b = tensor([[-1., -1., .3, .6], [.3, -1., 1., .6]]).requires_grad_()
    width = tensor(.3).requires_grad_()
    def f(mean, a, bounds, h):
        return gaussian_cell_mass_fast(mean, a@a.transpose(-1,-2), bounds, bandwidth=h)
    assert torch.autograd.gradcheck(f, (m, factor, b, width), eps=1.e-5, atol=1.e-6, rtol=1.e-5)


def test_chunk_batch_and_float_contract():
    m = tensor([[[0.,0.],[.1,.2]],[[1.,1.],[-1.,-2.]]])
    c = tensor([[1.,.3],[.3,.5]]).expand(2,2,2,2)
    b = tensor([[-1.,-1.,1.,1.],[1.,-1.,3.,1.]])
    a = gaussian_cell_mass_fast(m,c,b,query_chunk_size=1,key_chunk_size=1)
    z = gaussian_cell_mass_fast(m,c,b,query_chunk_size=99,key_chunk_size=99)
    torch.testing.assert_close(a,z,rtol=0.,atol=0.)
    assert gaussian_cell_mass_fast(m.float(),c.float(),b.float()).dtype == torch.float64


@pytest.mark.parametrize('reason',['width','bandwidth','negative','nonsymmetric','nan','empty'])
def test_domain_rejections(reason):
    m=tensor([[0.,0.]])
    c=torch.eye(2,dtype=torch.float64)[None]
    b=tensor([[-1.,-1.,1.,1.]])
    kw={}
    if reason=='width':b[0,2]=1.01
    if reason=='bandwidth':kw['bandwidth']=.249
    if reason=='negative':c[0,0,0]=-1.
    if reason=='nonsymmetric':c[0,0,1]=.01
    if reason=='nan':m[0,0]=float('nan')
    if reason=='empty':m=m[:0];c=c[:0]
    with pytest.raises(ValueError):gaussian_cell_mass_fast(m,c,b,**kw)
