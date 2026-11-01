"""R17 numerical audit development uses only fixed synthetic arrays."""
from copy import deepcopy
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr, softmax
import torch

from experiments import round17_trace_numerical_audit as N
from models import round17_trace_pe as P
from models.round10_cell_integral_fast import gaussian_cell_mass_fast


def test_locked_ratio_matches_independent_direct_mathematics_and_point_limit():
    cases=N.synthetic_cases();C=cases['covariance']
    for h in N.WIDTHS:
        for T in N.TEMPERATURES:
            tau=N.independent_temperature(C,h,T)
            direct=T*h*h/(h*h+C[:,0,0].astype(float)+C[:,1,1].astype(float))
            np.testing.assert_allclose(tau,direct,rtol=1e-14,atol=0)
            actual=P.trace_temperature(torch.from_numpy(C),h,T).numpy()
            np.testing.assert_array_equal(actual,tau.astype(np.float32))
            np.testing.assert_array_equal(actual[np.all(C==0,axis=(1,2))],np.float32(T))
            assert np.isfinite(tau).all() and np.all((tau>=0)&(tau<=T))
    big=np.eye(2)[None]*1e308
    np.testing.assert_allclose(N.independent_temperature(big,1.,5.),[2.5e-308],rtol=1e-14,atol=0)


def test_trace_units_rotation_and_spectral_response_bound():
    C=np.array([[[.3,.1],[.1,.7]],[[4.,2.],[2.,1.]]])
    theta=.731;R=np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])
    base=N.independent_temperature(C,.5,5.)
    np.testing.assert_allclose(N.independent_temperature(R@C@R.T,.5,5.),base,rtol=5e-16,atol=1e-16)
    np.testing.assert_allclose(N.independent_temperature(64*C,4.,5.),base,rtol=5e-16,atol=1e-16)
    largest=np.linalg.eigvalsh(C+.25*np.eye(2))[:,-1]
    assert np.all(base*largest<=1.25+1e-15)
    ratio=base/(1.25/largest)
    assert np.all((ratio>=.5)&(ratio<=1.+1e-15))
    assert ratio[1]==pytest.approx(1.)


def test_temperature_gradient_matches_analytic_diagonal_derivative():
    c=torch.tensor([[[.3,.1],[.1,.7]]],dtype=torch.float64,requires_grad=True)
    actual=P.trace_temperature(c,.5,5.)
    grad,=torch.autograd.grad(actual.sum(),c)
    expected=np.eye(2)[None]*(-5*.25/(.25+1.)**2)
    np.testing.assert_allclose(grad.numpy(),expected,rtol=1e-14,atol=1e-14)
    # Gradcheck parametrizes C=LL^T, so every finite-difference covariance
    # remains symmetric positive definite, not a one-off-diagonal mutation.
    L=torch.tensor([[[.7,.2],[.1,.8]]],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda factor:P.trace_temperature(factor@factor.transpose(-1,-2),.5,5.),
                                    (L,),eps=1e-6,atol=1e-6,rtol=1e-5)


def test_float32_bias_rounding_precedes_double_softmax_and_point_identity():
    content=np.array([[.12345679,.9876543,-.2]],np.float32)
    C=np.array([[[.2,.03],[.03,.1]]],np.float32)
    bias=N.independent_bias(content,C,.5,5.)
    np.testing.assert_array_equal(bias,P.content_bias(torch.from_numpy(C),.5,5.,torch.from_numpy(content)).numpy().astype(float))
    assert np.max(np.abs(bias-N.independent_bias(content,C,.5,5.,float32_bias=False)))>1e-10
    zero=np.zeros_like(C)
    np.testing.assert_array_equal(N.independent_bias(content,zero,.5,5.),(np.float32(5)*content).astype(float))
    boxes=np.array([[0.,0.,1.,1.],[1.,0.,2.,1.],[2.,0.,3.,1.]])
    raw=np.array([[.2,.3,.1]])
    geometry=torch.log(torch.from_numpy(raw)+1e-30)
    expected=N.independent_probability(raw,content,C,.5,5.,boxes)
    actual=P.probabilities(geometry,torch.from_numpy(content),torch.from_numpy(C),.5,5.).numpy()
    np.testing.assert_allclose(actual,expected,atol=2e-16,rtol=0)


def test_native24_gaussian_reference_matches_analytic_diagonal_cdf():
    boxes=np.array([[-.0625,-.0625,2.9375,2.9375],[2.9375,-.0625,5.9375,2.9375]])
    mean=np.array([[1.3,1.7],[-.7,1.1]])
    C=np.zeros((2,2,2));h=.25
    reference=N.independent_mass(mean,C,boxes,h,96)
    exact=np.empty_like(reference)
    for row,m in enumerate(mean):
        exact[row]=(ndtr((boxes[:,2]-m[0])/h)-ndtr((boxes[:,0]-m[0])/h))*(ndtr((boxes[:,3]-m[1])/h)-ndtr((boxes[:,1]-m[1])/h))
    np.testing.assert_allclose(reference,exact,atol=3e-15,rtol=1e-12)
    np.testing.assert_allclose(N.independent_mass(mean,C,boxes,h,64),reference,atol=3e-15,rtol=1e-12)


def test_rotated_native_reference_vs_adaptive_cartesian_integral_and_production():
    mean=np.array([[.13,-.17]]);C=np.array([[[.3,.12],[.12,.7]]]);h=.25
    bounds=np.array([[-.5,-.5,.5,.5],[.5,-.5,1.5,.5]])
    reference=N.independent_mass(mean,C,bounds,h,96)[0]
    S=C[0]+h*h*np.eye(2);precision=np.linalg.inv(S);normalizer=2*np.pi*np.sqrt(np.linalg.det(S))
    expected=[]
    for x0,y0,x1,y1 in bounds:
        def outer(x):
            def inner(y):
                d=np.array([x,y])-mean[0]
                return np.exp(-.5*d@precision@d)/normalizer
            return quad(inner,y0,y1,epsabs=1e-13,epsrel=1e-12)[0]
        expected.append(quad(outer,x0,x1,epsabs=1e-13,epsrel=1e-12)[0])
    np.testing.assert_allclose(reference,expected,atol=2e-14,rtol=0)
    actual=gaussian_cell_mass_fast(torch.from_numpy(mean),torch.from_numpy(C),torch.from_numpy(bounds),
                                  bandwidth=h,quadrature_order=32).numpy()[0]
    np.testing.assert_allclose(actual,reference,atol=2e-14,rtol=0)


def test_whole_native_rectangle_adds_to_literal_children_without_extra_floor():
    mean=np.array([[.13,-.17],[1.4,1.4]])
    C=np.array([[[.3,.12],[.12,.7]],[[.04,-.015],[-.015,.1]]])
    whole=np.array([[-.0625,-.0625,2.9375,2.9375]])
    split=np.array([[x,y,x+1.5,y+1.5] for y in (-.0625,1.4375) for x in (-.0625,1.4375)])
    np.testing.assert_allclose(N.independent_mass(mean,C,whole,.25,96)[:,0],
                               N.independent_mass(mean,C,split,.25,96).sum(1),atol=3e-14,rtol=0)
    outside=np.full((1,2),1000.);zero=np.zeros((1,2,2))
    np.testing.assert_array_equal(N.independent_mass(outside,zero,split,.25,96),0.)


def test_floor_is_added_once_by_native_area_and_preserves_tiny_rows():
    bounds=np.array([[0.,0.,1.,1.],[1.,0.,4.,3.]])
    raw=np.zeros((2,2));raw[1]=[1e-25,2e-25]
    C=np.zeros((2,2,2));content=np.zeros((2,2),np.float32)
    p=N.independent_probability(raw,content,C,.5,5.,bounds)
    np.testing.assert_allclose(p[0],[.1,.9],atol=1e-15)
    check=N.probability_check(p,p.copy(),p.copy(),raw)
    assert check['passed'] and check['zero_fov_rows']==1 and check['tiny_fov_rows']==2
    assert check['zero_raw_cells']==2


def test_probability_error_and_reference_error_fail_separately_even_with_same_argmax():
    raw=np.array([[.8,.2]])
    bad=np.array([[.8002,.1998]])
    report=N.probability_check(bad,raw,raw,raw)
    assert not report['passed'] and report['argmax_change_count']==0
    report=N.probability_check(raw,np.array([[.80000002,.19999998]]),raw,raw)
    assert not report['passed'] and report['max_reference_convergence_error']>1e-8
    near=np.array([[.5-1e-15,.5+1e-15]])
    report=N.probability_check(near,np.array([[.5,.5]]),np.array([[.5,.5]]),raw)
    assert report['passed'] and report['argmax_change_count']==1
    assert report['argmax_changes'][0]['exact_reference_tie']


def test_reduced_content_does_not_increase_geometry_penalty_away_from_ties():
    rng=np.random.default_rng(1701)
    geometry=rng.normal(size=(19,9));content=rng.uniform(-1,1,size=(19,9)).astype(np.float32)
    C=np.repeat(np.eye(2)[None],19,axis=0)*np.linspace(.01,4,19)[:,None,None]
    for T in (5.,10.):
        direct=geometry+(np.float32(T)*content).astype(float)
        reduced=geometry+N.independent_bias(content,C,.5,T)
        assert min(np.min(np.diff(np.sort(z,axis=1)[:,-2:],axis=1)) for z in (direct,reduced))>1e-4
        j0=geometry.argmax(1);j1=direct.argmax(1);j2=reduced.argmax(1);row=np.arange(19)
        assert np.all(geometry[row,j0]-geometry[row,j2]<=geometry[row,j0]-geometry[row,j1]+1e-14)
        assert np.all((j2!=j0).astype(int)<=(j1!=j0).astype(int))
    # This is an exact-envelope property, not a PCK assertion. Quantized
    # products at near ties are deliberately reported by probability_check.


def test_fixed_synthetic_case_domain_and_partitions():
    cases=N.synthetic_cases();parts=N.synthetic_partitions()
    assert len(cases['names'])==19 and len(set(cases['names']))==19
    assert cases['mean'].dtype==cases['covariance'].dtype==np.float32
    assert len(parts['g8']['bounds'])==432 and len(parts['g16']['bounds'])==108
    assert len(parts['g24']['bounds'])==48
    for part in parts.values():
        b=part['native_bounds'];owner=np.zeros((144,192),np.int64)
        for x0,y0,x1,y1 in b:owner[y0:y1,x0:x1]+=1
        np.testing.assert_array_equal(owner,1)
        assert np.prod(part['bounds'][:,2:]-part['bounds'][:,:2],axis=1).sum()==432
    assert len(cases['names'])*len(parts)*len(N.WIDTHS)*len(N.TEMPERATURES)==912


@pytest.mark.parametrize('kind',['nonsymmetric','not_psd','negative_trace','content64','bad_bounds','h_underflow'])
def test_reference_rejects_changed_numeric_domain(kind):
    cov=np.eye(2)[None];content=np.zeros((1,2),np.float32)
    if kind=='nonsymmetric':cov[0,0,1]=.1
    elif kind=='not_psd':cov[0]=[[1,2],[2,1]]
    elif kind=='negative_trace':cov[0]=np.eye(2)*-1e-14
    elif kind=='content64':content=content.astype(float)
    with pytest.raises(ValueError):
        if kind=='bad_bounds':N.independent_mass(np.zeros((1,2)),cov,[[0,0,0,1]],.5)
        else:N.independent_bias(content,cov,1e-200 if kind=='h_underflow' else .5,5.)


def test_synthetic_runner_refuses_to_overwrite_without_any_model_execution(tmp_path):
    with pytest.raises(FileExistsError):N.run_synthetic(tmp_path)


def test_synthetic_runner_archives_all_checks_including_failures(tmp_path,monkeypatch):
    # A deliberately reduced unit-test fixture exercises orchestration only;
    # it does not execute the predeclared full synthetic audit before freeze.
    cases=dict(names=['zero','full'],mean=np.array([[.2,.4],[.3,.5]],np.float32),
               covariance=np.array([np.zeros((2,2)),[[.4,.1],[.1,.6]]],np.float32))
    native=np.array([[0,0,8,8],[8,0,16,8]],np.int64)
    monkeypatch.setattr(N,'synthetic_cases',lambda:cases)
    monkeypatch.setattr(N,'synthetic_partitions',lambda:{'g8':dict(native_bounds=native,bounds=(native-.5)/8)})
    monkeypatch.setattr(N,'WIDTHS',(.5,))
    monkeypatch.setattr(N,'TEMPERATURES',(0.,5.))
    original=P.probabilities
    monkeypatch.setattr(P,'probabilities',lambda *args,**kwargs:original(*args,**kwargs).flip(-1))
    report=N.run_synthetic(tmp_path/'synthetic')
    assert not report['passed'] and not report['scientific_gate_passed']
    assert report['matrix_count']==2 and report['probability_rows']==4 and len(report['checks'])==2
    assert all(not r['passed'] for r in report['checks'])
    assert report['max_same_mass_probability_error']>N.SAME_MASS_TOLERANCE
    assert (tmp_path/'synthetic'/'verification.json').exists()
    with np.load(tmp_path/'synthetic'/'arrays.npz',allow_pickle=False) as archive:
        for ti in range(2):
            for field in ('actual','reference64','reference96','tau32','bias32'):
                assert f'g8__h0__t{ti}__{field}' in archive.files
