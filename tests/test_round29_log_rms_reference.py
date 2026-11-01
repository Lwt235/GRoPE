"""Distinct synthetic reference toys; never invokes a formal descriptor."""
import math
import time
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import erf

from experiments import round29_log_rms_reference as R

POLICY=dict(h=.4,floor_density=1e-30)


def test_log_interval_matches_scaled_density_integral_below_float_range():
    for a,b in ((2.,2.1),(45.,45.1),(-45.1,-45.)):
        value=float(R.log_normal_interval(a,b))
        lo,hi=(a,b) if a>=0 else (-b,-a)
        integral=quad(lambda t:math.exp(-lo*t-.5*t*t),0.,hi-lo,epsabs=1e-14,epsrel=1e-13)[0]
        expected=-.5*lo*lo-.5*math.log(2*math.pi)+math.log(integral)
        assert math.isclose(value,expected,rel_tol=0,abs_tol=3e-13)
        if lo>40:assert math.exp(value)==0 and math.isfinite(value)


def test_log_interval_reflection_and_crossing_zero():
    a=np.asarray([-2.,.1,5.]);b=a+.3
    np.testing.assert_allclose(R.log_normal_interval(a,b),R.log_normal_interval(-b,-a),rtol=0,atol=2e-15)
    expected=math.log(math.erf(.7/math.sqrt(2)))
    assert abs(float(R.log_normal_interval(-.7,.7))-expected)<1e-15


@pytest.mark.parametrize('a,b',((1.,1.),(2.,1.),(math.nan,2.),(-math.inf,0.)))
def test_invalid_log_interval_is_not_made_zero(a,b):
    with pytest.raises(ValueError):R.log_normal_interval(a,b)


def test_positive_but_unresolved_logcdf_difference_fails_explicitly():
    with pytest.raises(FloatingPointError):R.log_normal_interval(0.,1e-320)


def test_analytic_diagonal_constant_depth_matches_independent_erf_product():
    mu=np.asarray([.6,-.4,2.]);factor=np.diag([.8,.4,0.]);bounds=np.asarray([[-.7,-.5,.1,.2],[.1,-.5,1.3,.2]])
    result=R.integrate_analytic(mu,factor,bounds,np.zeros(2),policy=POLICY)
    mean=mu[:2]/2;sd=np.sqrt(np.asarray([.8,.4])**2/4+.4**2)
    expected=.25*np.prod(erf((bounds[:,2:]-mean)/(sd*math.sqrt(2)))-erf((bounds[:,:2]-mean)/(sd*math.sqrt(2))),axis=1)
    np.testing.assert_allclose(np.exp(result['log_raw_mass']),expected,rtol=2e-15,atol=0)
    assert result['work']==dict(cell_node_evaluations=0,analytic_cdf_evaluations=12)
    assert result['order'] is None and result['reference_route']=='analytic_single_call'
    assert result['passed'] and not result['tail_truncation_used']


@pytest.mark.parametrize('change',('stochastic_depth','negative_depth','cross_covariance'))
def test_analytic_reference_rejects_other_laws(change):
    mu=np.asarray([0.,0.,2.]);factor=np.diag([.8,.4,0.]);bounds=np.asarray([[-1.,-1.,1.,1.]])
    if change=='stochastic_depth':factor[2,0]=1e-15
    elif change=='negative_depth':mu[2]=-2.
    else:factor[1,0]=.1
    with pytest.raises(R.ReferenceFailure) as caught:R.integrate_analytic(mu,factor,bounds,policy=POLICY)
    assert caught.value.reference_work['cell_node_evaluations']==0


def test_analytic_deadline_failure_retains_explicit_zero_work():
    with pytest.raises(R.ReferenceFailure) as caught:
        R.integrate_analytic(np.asarray([0.,0.,2.]),np.zeros((3,3)),np.asarray([[-1.,-1.,1.,1.]]),policy=POLICY,deadline=time.perf_counter()-1)
    assert caught.value.reference_work==dict(cell_node_evaluations=0,analytic_cdf_evaluations=0)


def test_log_floor_stays_once_only_when_raw_is_subfloat_or_structural_zero():
    raw=np.asarray([-1200.,-np.inf]);bounds=np.asarray([[0.,0.,1.,1.],[1.,0.,3.,1.]])
    result=R.probabilities(raw,bounds,np.zeros(2),1e-30)
    np.testing.assert_allclose(result['probability'],np.asarray([1/3,2/3]),rtol=2e-14,atol=0)
    np.testing.assert_array_equal(result['log_floored_mass'],np.log(np.asarray([1e-30,2e-30])))


def test_old_reference_wrapper_forwards_original_parameters_and_preserves_raw(monkeypatch):
    calls=[];raw=np.asarray([.2,.3]);data=dict(raw_mass=raw,raw_fov_mass=.5,structural_zero=False,passed=True)
    def old(*args,**kwargs):calls.append((args,kwargs));return dict(data)
    monkeypatch.setattr(R.OLD,'integrate',old)
    budget=object();policy={'h':.4};reference_policy={'orders':[32,48]}
    result=R.integrate_old('mu','factor','bounds','bias',policy=policy,reference_policy=reference_policy,order=48,budget=budget,deadline=123.)
    assert calls[0][1]==dict(policy=policy,reference_policy=reference_policy,order=48,budget=budget,deadline=123.)
    assert result['raw_mass'] is raw
    np.testing.assert_array_equal(result['log_raw_mass'],np.log(raw))
    assert result['analytic_cdf_evaluations']==0


def test_old_wrapper_post_return_failure_keeps_charged_work_and_arrays(monkeypatch):
    raw=np.asarray([0.,.3]);work=dict(cell_node_evaluations=17)
    monkeypatch.setattr(R.OLD,'integrate',lambda *a,**k:dict(raw_mass=raw,raw_fov_mass=.3,structural_zero=False,work=work))
    with pytest.raises(R.ReferenceFailure) as caught:
        R.integrate_old('mu','factor','bounds',policy={},reference_policy={},order=32)
    assert caught.value.reference_work==work and caught.value.available_arrays['raw_mass'] is raw
