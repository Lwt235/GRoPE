import math
import numpy as np
import pytest
from experiments import round15_quartic_numerical_audit as A


def test_cartesian_full_plane_normalization_covariance_and_fourth_moment():
    result=A.cartesian_moments()
    assert result['maximum_identity_error']<1e-10
    assert result['fourth_x']['value']==pytest.approx(3*math.pi/4,abs=1e-10)


def test_rectangle_quadratic_minimum_uses_correlated_edge_projection():
    c=np.array([[4.,1.5],[1.5,1.]])
    q,point=A.rectangle_minimum(np.array([2.,1.]),c,np.array([-.5,-.5,.5,.5]))
    assert q==pytest.approx(.5625,abs=1e-14)
    np.testing.assert_allclose(point,[.5,.4375],atol=1e-14,rtol=0)


def test_adaptive_native24_rectangle_equals_nine_native8_children():
    mean=np.array([.4,-.2]);c=A.rotated_covariance((.02,4.),35);box=np.array([-.0625,-.0625,2.9375,2.9375])
    parent=A.adaptive_rectangle(mean,c,box,.25)
    children=[A.adaptive_rectangle(mean,c,b,.25)['mass'] for b in A.subdivide(box,3)]
    assert parent['mass']==pytest.approx(sum(children),abs=2e-11)
    assert not parent['warnings']


def test_reference_affine_axis_scaling_preserves_probability_and_transpose():
    mean=np.array([.2,.4]);cov=np.array([[.7,.2],[.2,.5]]);bounds=np.array([-.5,-.3,1.,1.4]);h=.25
    reference=A.adaptive_rectangle(mean,cov,bounds,h)['mass']
    scale=np.diag([2.,1.5]);offset=np.array([3.,-2.]);newh=.1
    newcov=scale@(cov+h*h*np.eye(2))@scale.T-newh*newh*np.eye(2)
    newbounds=np.r_[scale@bounds[:2]+offset,scale@bounds[2:]+offset]
    transformed=A.adaptive_rectangle(scale@mean+offset,newcov,newbounds,newh)['mass']
    swapped=A.adaptive_rectangle(mean[::-1],cov[::-1,::-1],bounds[[1,0,3,2]],h)['mass']
    assert transformed==pytest.approx(reference,abs=2e-12)
    assert swapped==pytest.approx(reference,abs=2e-12)


def test_tiny_tail_mass_is_positive_and_underflow_floor_preserves_area():
    tail=A.adaptive_rectangle(np.array([-1.25,.4375]),np.zeros((2,2)),A.native_grid(8)[0],.25)
    assert 1e-30<tail['mass']<1e-15 and tail['log_mass'] is not None
    bounds=np.array([[0.,0.,1.,1.],[1.,0.,3.,1.]])
    np.testing.assert_allclose(A.normalize(np.zeros(2),bounds),[1/3,2/3],atol=0,rtol=0)
    extreme=A.adaptive_rectangle(np.array([-128.,8.]),np.zeros((2,2)),A.native_grid(8)[0],.25)
    assert extreme['underflow_bound'] and extreme['mass']==0


def test_point_covariance_uses_positive_floor_and_rejects_indefinite_input():
    assert A.density(np.zeros(2),np.zeros(2),np.zeros((2,2)),.5)==pytest.approx(4/math.pi**2)
    with pytest.raises(ValueError,match='PSD'):
        A.adaptive_rectangle(np.zeros(2),np.diag([-1.,0.]),np.array([0.,0.,1.,1.]),.25)
