import math

import numpy as np
import pytest
from scipy.integrate import quad
import torch

from models.round15_quartic_pe import (quartic_cell_mass,
    quartic_cell_mass_cartesian, quartic_log_density)


torch.set_num_threads(1)


def reference(mu, cov, box):
    inv=np.linalg.inv(cov);norm=np.pi**2*np.sqrt(np.linalg.det(cov))
    def inner(y,x):
        d=np.array([x,y])-mu;q=d@inv@d
        return np.exp(-q*q/(4*np.pi))/norm
    return quad(lambda x:quad(lambda y:inner(y,x),box[1],box[3],
                             epsabs=1.e-13,epsrel=2.e-12)[0],
                box[0],box[2],epsabs=1.e-13,epsrel=2.e-12)[0]


def test_independent_scipy_rectangle_and_cartesian():
    means=np.array([[.2,.3],[-1.,.6],[1.,1.],[0.,0.]])
    covs=np.array([[[.3,.11],[.11,.5]],[[2.,1.8],[1.8,2.]],
                  [[0.,0.],[0.,0.]],[[9.,-.8],[-.8,.1]]])
    boxes=np.array([[0.,0.,1.,1.],[-.5,-.5,2.5,2.5],[3.,-2.,4.,0.]])
    args=tuple(torch.tensor(x) for x in (means,covs,boxes))
    got=quartic_cell_mass(*args,bandwidth=.25).numpy()
    ref=np.array([[reference(m,c+.25**2*np.eye(2),b) for b in boxes]
                  for m,c in zip(means,covs)])
    np.testing.assert_allclose(got,ref,atol=2.e-10,rtol=2.e-7)
    cart=quartic_cell_mass_cartesian(*args,bandwidth=.25,quadrature_order=96).numpy()
    np.testing.assert_allclose(got,cart,atol=2.e-10,rtol=2.e-7)


def test_normalization_symmetry_and_large_box():
    mean=torch.zeros((1,2),dtype=torch.float64)
    cov=torch.eye(2,dtype=torch.float64)[None]
    boxes=torch.tensor([[-12.,-12.,12.,12.],[0.,0.,12.,12.],[-12.,0.,0.,12.],
                        [-12.,-12.,0.,0.],[0.,-12.,12.,0.]],dtype=torch.float64)
    got=quartic_cell_mass(mean,cov,boxes,bandwidth=0.)[0]
    torch.testing.assert_close(got,torch.tensor([1.,.25,.25,.25,.25],dtype=torch.float64),atol=2.e-13,rtol=0)


def test_moments_from_independent_cartesian_density_grid():
    # The integration directly samples density, not the angular primitive.
    nodes,weights=np.polynomial.legendre.leggauss(160)
    z=5*nodes;w=5*weights
    yy,xx=np.meshgrid(z,z,indexing='ij')
    points=torch.tensor(np.stack((xx,yy),-1))
    mean=torch.tensor([.3,-.6]);factor=torch.tensor([[1.2,0.],[.4,.7]])
    transformed=points@factor.double().T+mean.double()
    cov=factor.double()@factor.double().T
    density=quartic_log_density(transformed,mean,cov).exp().numpy()*np.linalg.det(factor)
    measure=density*w[:,None]*w[None,:]
    np.testing.assert_allclose(measure.sum(),1.,atol=3.e-14)
    delta=transformed.numpy()-mean.numpy()
    np.testing.assert_allclose((measure[...,None]*delta).sum((0,1)),np.zeros(2),atol=5.e-14)
    moment=np.einsum('ij,ija,ijb->ab',measure,delta,delta)
    np.testing.assert_allclose(moment,cov.numpy(),atol=5.e-13)
    np.testing.assert_allclose((measure*xx**4).sum(),3*np.pi/4,atol=5.e-13)


def test_density_affine_equivariance_and_rectangle_axis_affine():
    points=torch.tensor([[.2,-.4],[1.,3.]],dtype=torch.float64)
    mean=torch.tensor([.3,.2],dtype=torch.float64)
    cov=torch.tensor([[.5,.1],[.1,2.]],dtype=torch.float64)
    A=torch.tensor([[2.,.7],[-.2,.4]],dtype=torch.float64);b=torch.tensor([-.6,.1])
    got=quartic_log_density(points@A.T+b,mean@A.T+b,A@cov@A.T)
    ref=quartic_log_density(points,mean,cov)-torch.log(torch.linalg.det(A).abs())
    torch.testing.assert_close(got,ref,atol=2.e-13,rtol=1.e-13)
    boxes=torch.tensor([[-1.,0.,2.,3.],[2.,-.5,3.,1.]],dtype=torch.float64)
    scale=torch.tensor([1.7,.3]);shift=torch.tensor([.2,-3.])
    transformed=boxes*scale.repeat(2)+shift.repeat(2)
    p=quartic_cell_mass(mean[None],cov[None],boxes,bandwidth=0.)
    q=quartic_cell_mass((mean*scale+shift)[None],(cov*scale[:,None]*scale[None])[None],transformed,bandwidth=0.)
    torch.testing.assert_close(p,q,atol=2.e-13,rtol=1.e-11)


@pytest.mark.parametrize('mu',[[.23,.61],[0.,.7],[0.,0.],[-5.,7.]])
def test_split_additivity_edges_corners_and_tails(mu):
    mean=torch.tensor([mu],dtype=torch.float64)
    cov=torch.tensor([[[.7,.25],[.25,.4]]],dtype=torch.float64)
    boxes=torch.tensor([[0.,0.,3.,3.],[0.,0.,1.5,1.5],[1.5,0.,3.,1.5],
                        [0.,1.5,1.5,3.],[1.5,1.5,3.,3.]],dtype=torch.float64)
    p=quartic_cell_mass(mean,cov,boxes,bandwidth=.25,quadrature_order=48)[0]
    assert bool(torch.isfinite(p).all()) and bool((p>=0).all())
    torch.testing.assert_close(p[0],p[1:].sum(),atol=2.e-12,rtol=2.e-8)


@pytest.mark.parametrize('mu',[[.21,.63],[0.,.7],[0.,0.]])
def test_gradcheck_mean_covariance_bandwidth_and_bounds(mu):
    mean=torch.tensor([mu],dtype=torch.float64,requires_grad=True)
    factor=torch.tensor([[[.5,.1],[-.2,.6]]],dtype=torch.float64,requires_grad=True)
    bounds=torch.tensor([[0.,0.,1.,1.]],dtype=torch.float64,requires_grad=True)
    width=torch.tensor(.5,dtype=torch.float64,requires_grad=True)
    fn=lambda m,a,b,h:quartic_cell_mass(m,a@a.transpose(-1,-2),b,bandwidth=h)
    assert torch.autograd.gradcheck(fn,(mean,factor,bounds,width),atol=3.e-6,rtol=2.e-4)


def test_representable_tail_no_erf_cancellation_and_underflow():
    mean=torch.zeros((1,2),dtype=torch.float64);cov=torch.eye(2,dtype=torch.float64)[None]
    boxes=torch.tensor([[4.,0.,4.1,.1],[20.,20.,21.,21.]],dtype=torch.float64)
    p=quartic_cell_mass(mean,cov,boxes,bandwidth=0.)[0]
    ref=reference(np.zeros(2),np.eye(2),boxes[0].numpy())
    assert 0 < p[0] < 1.e-10 and p[1] == 0
    np.testing.assert_allclose(p[0].item(),ref,rtol=3.e-9,atol=1.e-30)


def test_batch_chunks_and_point_is_new_kernel():
    mean=torch.tensor([[[0.,0.],[.2,.6]],[[1.,1.],[2.,2.]]])
    cov=torch.zeros(2,2,2,2);boxes=torch.tensor([[0.,0.,1.,1.],[1.,0.,2.,1.]])
    p=quartic_cell_mass(mean,cov,boxes,query_chunk_size=1,key_chunk_size=1)
    q=quartic_cell_mass(mean,cov,boxes,query_chunk_size=4,key_chunk_size=2)
    torch.testing.assert_close(p,q,atol=0,rtol=0)
    assert p.dtype==torch.float64 and p.shape==(2,2,2)
    logp=quartic_log_density(torch.zeros(2),torch.zeros(2),torch.eye(2)*.25)
    assert abs(logp.item()-math.log(1/(math.pi**2*.25)))<1.e-14
    assert abs(logp.item()-math.log(1/(2*math.pi*.25)))>.1


def test_invalid_inputs():
    m=torch.zeros(1,2);c=torch.zeros(1,2,2);b=torch.tensor([[0.,0.,1.,1.]])
    with pytest.raises(ValueError):quartic_cell_mass(m,c,b,bandwidth=0.)
    with pytest.raises(ValueError):quartic_cell_mass(m,c,b,bandwidth=-1.)
    with pytest.raises(ValueError):quartic_cell_mass(m,c,b[:,[2,1,0,3]])
    with pytest.raises(ValueError):quartic_cell_mass(m,torch.tensor([[[-1.,0.],[0.,1.]]]),b)
