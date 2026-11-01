"""Distinct R31 numerical toys; none executes a declared formal descriptor."""
from copy import deepcopy
import math
import time
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp

from experiments import round31_log_conditional_reference as R
from experiments import round31_log_conditional_contract as C


def policies():
    return dict(C.POLICY,h=.3),deepcopy(C.REFERENCE_POLICY)


def run(mu,factor,bounds,order=32,**kwargs):
    policy,ref=policies()
    return R.integrate(np.asarray(mu,np.float64),np.asarray(factor,np.float64),np.asarray(bounds,np.float64),
        policy=policy,reference_policy=ref,order=order,**kwargs)


def test_constant_depth_full_x_far_tail_is_positive_below_float_range():
    # Standard X interval [45,45.2], distinct from the formal [39,41] cells.
    bounds=np.asarray([[13.5,-.18,13.56,.18]])
    result=run([0.,0.,2.],np.zeros((3,3)),bounds)
    a,b=45.,45.2
    scaled=quad(lambda t:math.exp(-a*t-.5*t*t),0,b-a,epsabs=1e-14,epsrel=1e-13)[0]
    logx=-.5*a*a-.5*math.log(2*math.pi)+math.log(scaled)
    expected=logx+math.log(math.erf(.6/math.sqrt(2)))
    assert result['passed'] and math.isfinite(result['log_raw_mass'][0])
    assert abs(result['log_raw_mass'][0]-expected)<1e-10 and math.exp(result['log_raw_mass'][0])==0.
    assert result['work']['depth_evaluations']==0 and result['scaled_outer_error_estimate'] is None
    assert 'pilot_zeta' not in result and not result['inner_x_truncation_used']
    assert result['truncation_bounds']['log_depth_omitted_mass'] is None


def test_wide_full_x_gaussian_has_no_old_cutoff_mass_loss():
    result=run([0.,0.,1.7],np.zeros((3,3)),[[-13.5,-.24,13.5,.24]])
    expected=math.log(math.erf(.8/math.sqrt(2)))
    assert result['passed'] and abs(result['log_raw_mass'][0]-expected)<2e-14
    assert result['maximum_full_x_panel_width']<=2.


@pytest.mark.parametrize('order',(32,48))
def test_variable_depth_pilots_and_scaled_integral_match_solid_angle_oracle(order):
    policy,ref=policies();policy['h']=.19
    factor=np.diag([.37,.52,1.1]);bounds=np.asarray([[-.7,-.6,.1,.8],[.1,-.6,.5,.8]])
    result=R.integrate(np.zeros(3),factor,bounds,policy=policy,reference_policy=ref,order=order)
    sx,sy=math.hypot(.37/1.1,.19),math.hypot(.52/1.1,.19)
    def f(x,y):
        x,y=x/sx,y/sy
        return math.atan2(x*y,math.sqrt(1+x*x+y*y))/(4*math.pi)
    expected=np.asarray([f(b[2],b[3])-f(b[0],b[3])-f(b[2],b[1])+f(b[0],b[1]) for b in bounds])
    assert result['passed'] and result['outer_convergence_success']
    np.testing.assert_allclose(np.exp(result['log_raw_mass']),expected,rtol=2e-12,atol=0)
    np.testing.assert_array_equal(result['pilot_zeta'],(result['initial_depth_edges'][:-1]+result['initial_depth_edges'][1:])/2)
    np.testing.assert_array_equal(result['outer_log_scale'],result['pilot_log_integrand'].max(axis=0))
    restored=np.log(result['outer_scaled_integral'])+result['outer_log_scale']
    np.testing.assert_array_equal(restored,np.r_[result['log_raw_mass'],result['log_raw_fov_mass']])
    work=result['work']
    assert work['pilot_depth_evaluations']==len(result['pilot_zeta'])
    assert work['depth_evaluations']==work['pilot_depth_evaluations']+work['quadrature_depth_evaluations']
    assert work['cell_node_evaluations']==work['inner_x_nodes']+work['narrow_y_pdf_nodes']


@pytest.mark.parametrize('mu,factor',(([0.,0.,0.],np.zeros((3,3))),([0.,0.,-2.],np.zeros((3,3)))))
def test_only_exact_structural_zero_has_negative_infinity_raw_and_zero_work(mu,factor):
    result=run(mu,factor,[[-.6,-.5,.4,.7]])
    assert result['passed'] and result['structural_zero'] and np.isneginf(result['log_raw_mass']).all()
    assert result['work']['cell_node_evaluations']==0 and result['work']['pilot_depth_evaluations']==0
    assert result['truncation_bounds']['raw_l1_over_reference_fov'] is None
    assert np.array_equal(result['probability'],np.ones(1))


def test_fixed_pilot_overflow_fails_without_reselection_and_keeps_evidence(monkeypatch):
    policy,ref=policies();calls=[]
    monkeypatch.setattr(R,'outer_breaks',lambda *a:[])
    def fake_rect(location,z,boxes,*args):return np.full(len(boxes),-1000. if z>1 else 0.)
    monkeypatch.setattr(R,'_rectangles',fake_rect)
    def fake_quad(function,*args,**kwargs):calls.append(True);function(0.);pytest.fail('overflow should have failed')
    monkeypatch.setattr(R,'quad_vec',fake_quad)
    with pytest.raises(R.ReferenceFailure,match='overflow') as caught:
        R.integrate(np.asarray([0.,0.,1.]),np.diag([.2,.2,.5]),np.asarray([[-.4,-.4,.4,.4]]),policy=policy,reference_policy=ref,order=32)
    error=caught.value
    assert len(calls)==1 and error.reference_work['pilot_depth_evaluations']==1
    assert error.reference_work['quadrature_depth_evaluations']==1
    assert 'pilot_log_integrand' in error.available_arrays and 'outer_log_scale' in error.available_arrays
    assert not error.reference_metadata['outer_exp_diagnostics']['scale_reselected']


def test_full_x_grid_excess_fails_before_allocating_or_charging_nodes():
    policy,ref=policies();budget=R.NodeBudget(16)
    with pytest.raises(R.ReferenceFailure,match='before allocation') as caught:
        R.integrate(np.asarray([0.,0.,2.]),np.zeros((3,3)),np.asarray([[-1e20,-.2,1e20,.2]]),
            policy=policy,reference_policy=ref,order=32,budget=budget)
    assert budget.used==0 and caught.value.reference_work['cell_node_evaluations']==0


def test_narrow_positive_interval_preserves_halfwidth_when_endpoints_round_equal():
    _,ref=policies();budget=R.NodeBudget(100);ledger=R._Ledger(ref,None,budget)
    value=R.normal_log_interval(np.asarray([2.]),np.asarray([1e-30]),reference_policy=ref,ledger=ledger)
    expected=math.log(2e-30)-2.-.5*math.log(2*math.pi)
    assert abs(value[0]-expected)<2e-14 and budget.used==16
    assert 2.-1e-30==2.+1e-30


def test_failure_after_narrow_node_charge_retains_actual_budget_delta():
    _,ref=policies();budget=R.NodeBudget(100);ledger=R._Ledger(ref,None,budget)
    with pytest.raises(FloatingPointError):
        R.normal_log_interval(np.asarray([1e308]),np.asarray([1e-309]),reference_policy=ref,ledger=ledger)
    assert budget.used==16 and ledger.work['narrow_y_pdf_nodes']==16


def test_physical_depth_tail_is_counted_once_and_blocks_tiny_raw():
    _,ref=policies()
    prepared=dict(structural_zero=False,normalized_depth_std=1.,normalized_depth_mean=0.)
    raw=np.asarray([-1000.,-1001.]);floor=np.log(np.asarray([1e-30,2e-30]));bias=np.zeros(2)
    bounds=R.truncation_bounds(prepared,raw,bias,floor,ref)
    expected=math.log(2.)+float(R.log_ndtr(-18.))
    assert abs(bounds['log_depth_omitted_mass']-expected)<1e-13
    assert bounds['log_raw_l1_over_reference_fov']>math.log(1e-10)
    assert not bounds['inner_x_truncation_used'] and bounds['native_union_bound']


def test_positive_front_outside_depth_cutoff_is_failed_not_structural_zero():
    with pytest.raises(R.ReferenceFailure,match='outside') as caught:
        run([0.,0.,-10.],np.diag([.1,.1,.3]),[[-.5,-.5,.5,.5]])
    assert caught.value.reference_metadata['structural_zero'] is False
    assert caught.value.reference_work['cell_node_evaluations']==0


def test_deadline_before_reference_work_is_retained():
    with pytest.raises(R.ReferenceFailure) as caught:
        run([0.,0.,2.],np.zeros((3,3)),[[-.5,-.5,.5,.5]],deadline=time.perf_counter()-1)
    assert caught.value.reference_work['cell_node_evaluations']==0


def test_independent_source_imports_no_producer_model_or_old_reference():
    import ast
    from pathlib import Path
    tree=ast.parse(Path(R.__file__).read_text());names=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    assert not any(n.startswith(('models','experiments')) for n in names)
