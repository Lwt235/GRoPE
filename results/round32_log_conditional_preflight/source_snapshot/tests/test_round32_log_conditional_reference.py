"""Distinct R32 numerical toys; none executes a declared formal descriptor."""
from copy import deepcopy
import math
import time
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp

from experiments import round32_log_conditional_reference as R
from experiments import round32_log_conditional_contract as C


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


@pytest.mark.parametrize('mean,multiplier',((0.,1.),(50.,2.)))
def test_physical_depth_tail_is_counted_once_and_blocks_tiny_raw(mean,multiplier):
    _,ref=policies()
    prepared=dict(structural_zero=False,normalized_depth_std=1.,normalized_depth_mean=mean)
    raw=np.asarray([-1000.,-1001.]);floor=np.log(np.asarray([1e-30,2e-30]));bias=np.zeros(2)
    bounds=R.truncation_bounds(prepared,raw,bias,floor,ref)
    expected=math.log(multiplier)+float(R.log_ndtr(-18.))
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


def coalesce(values):
    available={};metadata={}
    retained=R.coalesce_depth_edges(np.asarray(values,np.float64),available=available,metadata=metadata)
    return retained,available,metadata['depth_edge_coalescing']


def test_adjacent_float_boundary_is_removed_but_two_ulp_boundary_retained():
    one=1.;adjacent=np.nextafter(one,np.inf);two=np.nextafter(adjacent,np.inf)
    kept,arrays,report=coalesce([0.,one,adjacent,2.])
    np.testing.assert_array_equal(kept,[0.,1.,2.])
    np.testing.assert_array_equal(arrays['dropped_depth_edge_indices'],np.asarray([2],np.int64))
    np.testing.assert_array_equal(arrays['dropped_depth_edge_values'],np.asarray([adjacent]))
    assert report['dropped'][0]['reason']=='no_strict_midpoint_from_previous_retained'
    kept,arrays,report=coalesce([0.,one,two,2.])
    np.testing.assert_array_equal(kept,[0.,one,two,2.])
    assert report['dropped_count']==0


def test_upper_endpoint_backtracking_preserves_endpoint_bits_and_domain():
    upper=2.;near=np.nextafter(upper,-np.inf)
    original=np.asarray([-0.,1.,near,upper])
    kept,arrays,report=coalesce(original)
    np.testing.assert_array_equal(kept.view(np.uint64)[[0,-1]],original.view(np.uint64)[[0,-1]])
    np.testing.assert_array_equal(kept,np.asarray([-0.,1.,2.]))
    assert report['dropped']==[dict(original_index=2,value=float(near),reason='no_strict_midpoint_to_upper')]
    np.testing.assert_array_equal(arrays['original_depth_edges'],original)


def test_subnormal_chain_terminates_without_epsilon_and_is_idempotent():
    tiny=np.nextafter(0.,1.);original=np.asarray([0.,tiny,2*tiny,3*tiny,1.])
    kept,arrays,report=coalesce(original)
    np.testing.assert_array_equal(kept,[0.,2*tiny,1.])
    np.testing.assert_array_equal(arrays['dropped_depth_edge_indices'],np.asarray([1,3],np.int64))
    again,_,second=coalesce(kept)
    np.testing.assert_array_equal(again,kept);assert second['dropped_count']==0
    midpoints=(kept[:-1]+kept[1:])/2
    assert np.all(kept[:-1]<midpoints) and np.all(midpoints<kept[1:])


@pytest.mark.parametrize('raw',((0.,.5,.5,1.),(0.,.8,.2,1.),(0.,-1.,1.),(0.,2.,1.),(0.,math.nan,1.),(0.,math.inf,1.)))
def test_invalid_original_partition_is_rejected_with_original_evidence(raw):
    available={};metadata={};original=np.asarray(raw,np.float64)
    with pytest.raises(ValueError):R.coalesce_depth_edges(original,available=available,metadata=metadata)
    np.testing.assert_array_equal(available['original_depth_edges'],original)
    assert metadata['depth_edge_coalescing']['dropped_count']==0


def test_whole_domain_without_strict_midpoint_still_fails_and_keeps_endpoints():
    available={};metadata={};left=1.;right=np.nextafter(left,np.inf)
    original=np.asarray([left,right])
    with pytest.raises(FloatingPointError,match='whole physical depth domain'):
        R.coalesce_depth_edges(original,available=available,metadata=metadata)
    np.testing.assert_array_equal(available['initial_depth_edges'],original)
    assert not len(available['dropped_depth_edge_indices'])


def test_retained_partition_is_original_subset_and_full_domain_is_unchanged():
    from fractions import Fraction
    a=-1.;near=np.nextafter(a,np.inf)
    original=np.asarray([-3.,a,near,.4,np.nextafter(2.,-np.inf),2.])
    kept,arrays,report=coalesce(original)
    assert all(v in original for v in kept) and kept[0]==original[0] and kept[-1]==original[-1]
    covered=sum((Fraction(float(b))-Fraction(float(a)) for a,b in zip(kept[:-1],kept[1:])),Fraction(0))
    assert covered==Fraction(float(original[-1]))-Fraction(float(original[0]))
    assert report['original_interior_count']==len(original)-2
    assert report['retained_interior_count']+report['dropped_count']==report['original_interior_count']


def test_pilot_and_quad_share_identical_retained_edges_on_distinct_toy(monkeypatch):
    policy,reference_policy=policies();calls=[];original_quad=R.quad_vec
    monkeypatch.setattr(R,'outer_breaks',lambda *a:[1.,np.nextafter(1.,np.inf),3.])
    def quad(function,*args,**kwargs):
        calls.append(np.asarray(kwargs['points']).copy())
        return original_quad(function,*args,**kwargs)
    monkeypatch.setattr(R,'quad_vec',quad)
    result=R.integrate(np.asarray([.1,-.2,1.2]),np.diag([.2,.3,.45]),np.asarray([[-.4,-.3,.6,.5]]),
        policy=policy,reference_policy=reference_policy,order=32)
    assert result['passed'] and len(calls)==1
    edges=result['initial_depth_edges']
    np.testing.assert_array_equal(calls[0],edges[1:-1])
    np.testing.assert_array_equal(result['pilot_zeta'],(edges[:-1]+edges[1:])/2)
    assert result['depth_edge_coalescing']['dropped_count']==1
    assert result['work']['outer_geometry_breakpoints']==len(edges)-2
    assert result['work']['pilot_depth_evaluations']==len(edges)-1


def test_invalid_raw_breakpoints_fail_before_pilots_with_canonical_evidence(monkeypatch):
    monkeypatch.setattr(R,'outer_breaks',lambda *a:[2.,1.])
    with pytest.raises(R.ReferenceFailure,match='strictly sorted') as caught:
        run([.1,-.2,1.2],np.diag([.2,.3,.45]),[[-.4,-.3,.6,.5]])
    error=caught.value
    assert error.reference_work['pilot_depth_evaluations']==0 and error.reference_work['cell_node_evaluations']==0
    np.testing.assert_array_equal(error.available_arrays['original_depth_edges'][1:-1],[2.,1.])
    assert error.reference_metadata['depth_edge_coalescing']['dropped_count']==0
