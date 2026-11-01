import numpy as np
import pytest
from scipy.special import ndtr
from experiments import round13_mixture_mechanism_audit as A


def synthetic():
    y,x=np.mgrid[:144,:192];xy=np.stack([x,y],-1).reshape(-1,2).astype(float)
    cov=np.broadcast_to(np.eye(2)/12,(len(xy),2,2)).copy()
    return dict(q_to_k_xy=xy,q_to_k_pixel_cov=cov)


def test_literal_banks_and_ou_preserve_dense_moments():
    inputs=synthetic();bounds=A.grid(24)[A.QUERIES]
    dense=A.component_reference(inputs,bounds,'dense');dm,dc=A.moments(dense)
    full=A.component_reference(inputs,bounds,'full4');ou=A.smooth_reference(full)
    original=full['mean'].copy()
    for bank in (full,ou,A.component_reference(inputs,bounds,'full1')):
        m,c=A.moments(bank)
        np.testing.assert_allclose(m,dm,atol=2e-13,rtol=0);np.testing.assert_allclose(c,dc,atol=2e-13,rtol=0)
    np.testing.assert_array_equal(full['mean'],original)
    np.testing.assert_allclose(full['cov'],np.broadcast_to(np.eye(2)*.1875,full['cov'].shape),atol=1e-14,rtol=0)


def test_independent_scipy_rectangle_matches_separable_analytic_cdf():
    mean=np.array([[.5,.8]]);cov=np.array([[[.3,0],[0,.2]]])
    bank=dict(mean=mean,cov=cov,owner=np.array([0]),weights=np.array([1.]),queries=1)
    p,fov=A.integral_reference(bank);b=(A.grid(8)-.5)/8;sd=np.sqrt(np.array([.3,.2])+.25)
    mass=(ndtr((b[:,2:]-mean)/sd)-ndtr((b[:,:2]-mean)/sd)).prod(-1)
    np.testing.assert_allclose(fov,mass.sum(),atol=2e-15,rtol=0)
    np.testing.assert_allclose(p,(mass+1e-30)[None]/(mass.sum()+432e-30),atol=2e-15,rtol=0)


def test_center_uses_four_depth_samples_before_ray_projection():
    K=np.array([[100.,0,96],[0,100,72],[0,0,1.]])
    z=np.ones((144,192));z[11:13,11:13]=[[2,4],[6,8]]
    T=np.eye(4);T[0,3]=.1
    inputs=dict(q_predicted_depth=z,q_K=K,k_K=K,q_to_k_transform=T)
    xy,front,fov=A.predicted_center_reference(inputs,A.grid(24)[:1])
    np.testing.assert_allclose(xy,[[13.5,11.5]],atol=1e-13,rtol=0)
    assert front[0] and fov[0]


def test_predicted_center_mask_never_removes_distribution_discrepancy_rows():
    p=np.ones((2,432))/432;q=p.copy();q[1]=np.linspace(1,432,432);q[1]/=q[1].sum()
    rows=A.center_rows(q,np.array([[11.5,11.5],[1000.,1000.]]),np.array([True,False]))
    assert rows[1]['center_bin'] is None
    kl,js,l1=A.discrepancies(p,q)
    assert len(kl)==2 and kl[1]>0 and l1[1]>0
