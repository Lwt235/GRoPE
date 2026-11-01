"""Distinct R36 numerical toys; none executes a declared formal descriptor."""
from copy import deepcopy
import math
import time
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp

from experiments import round36_component_round_reference as R
from experiments import round36_component_round_contract as C


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
    assert abs(value[0]-expected)<2e-14 and budget.used==8
    assert 2.-1e-30==2.+1e-30


def test_failure_after_narrow_node_charge_retains_actual_budget_delta():
    _,ref=policies();budget=R.NodeBudget(100);ledger=R._Ledger(ref,None,budget)
    with pytest.raises(FloatingPointError):
        R.normal_log_interval(np.asarray([1e308]),np.asarray([1e-309]),reference_policy=ref,ledger=ledger)
    assert budget.used==8 and ledger.work['narrow_y_pdf_nodes']==8


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
    assert not any(n.startswith('models') for n in names)
    assert [n for n in names if n.startswith('experiments')] == ['experiments.round36_component_round_quadrature']


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


def test_local_remainder_formula_matches_independent_decimal_evaluation():
    from decimal import Decimal, localcontext
    with localcontext() as context:
        context.prec=80
        half=Decimal(1)/2
        terms=sum((half**(16-k)/(Decimal(2)**k*Decimal(math.factorial(k))*Decimal(math.factorial(16-2*k)))
                   for k in range(9)),Decimal(0))
        # Cancel 16! symbolically, and exp(.5)/exp(-.5) -> exp(1).
        independent=(Decimal(2)**16*Decimal(math.factorial(8))**4*Decimal(1).exp()*terms
                     /(17*Decimal(math.factorial(16))**2))
        assert independent<Decimal('1.88e-13')
    evidence=R.narrow_normal_local_bound()
    assert abs(evidence['computed_relative_remainder']-float(independent))<1e-27
    assert evidence==dict(order=8,branch_exponent_span=.25,proof_envelope=.5,
        computed_relative_remainder=evidence['computed_relative_remainder'],
        conservative_relative_remainder_bound=1.88e-13,exact_arithmetic_only=True,
        includes_floating_point_error=False,is_pipeline_error_bound=False)


@pytest.mark.parametrize('mid',(-6.25,-1.75,0.,2.125,5.5))
def test_gl8_narrow_mass_matches_centered_positive_adaptive_integral(mid):
    _,ref=policies();half=.2/(abs(mid)+1.)
    budget=R.NodeBudget(8);ledger=R._Ledger(ref,None,budget)
    actual=R.normal_log_interval(np.asarray([mid]),np.asarray([half]),reference_policy=ref,ledger=ledger)[0]
    a,b=mid*half,half*half/2
    normalized=quad(lambda u:math.exp(-a*u-b*u*u),-1.,1.,epsabs=1e-14,epsrel=1e-13)[0]
    expected=math.log(half)-.5*mid*mid-.5*math.log(2*math.pi)+math.log(normalized)
    assert abs(math.expm1(actual-expected))<3e-14
    assert budget.used==ledger.work['narrow_y_pdf_nodes']==8
    assert ledger.normal_logcdf_scalar_evaluations==0


def test_exact_span_boundary_uses_gl8_but_next_float_uses_cdf():
    _,ref=policies()
    for half,nodes in ((.5,8),(np.nextafter(.5,np.inf),0)):
        budget=R.NodeBudget(8);ledger=R._Ledger(ref,None,budget)
        value=R.normal_log_interval(np.asarray([0.]),np.asarray([half]),reference_policy=ref,ledger=ledger)[0]
        assert abs(value-math.log(math.erf(half/math.sqrt(2))))<3e-15
        assert budget.used==nodes


def test_large_midpoint_unresolved_endpoints_keep_positive_halfwidth():
    _,ref=policies();mid=64.;half=2.**-60
    assert mid-half==mid+half
    ledger=R._Ledger(ref,None,R.NodeBudget(8))
    value=R.normal_log_interval(np.asarray([mid]),np.asarray([half]),reference_policy=ref,ledger=ledger)[0]
    expected=math.log(2*half)-.5*mid*mid-.5*math.log(2*math.pi)
    assert math.isfinite(value) and abs(value-expected)<=np.spacing(abs(expected))
    assert ledger.work['narrow_y_pdf_nodes']==8


def test_gl8_node_budget_precharges_whole_narrow_batch():
    _,ref=policies();budget=R.NodeBudget(15);ledger=R._Ledger(ref,None,budget)
    with pytest.raises(RuntimeError,match='before allocation'):
        R.normal_log_interval(np.asarray([.3,-.7]),np.asarray([.01,.02]),reference_policy=ref,ledger=ledger)
    assert budget.used==0 and ledger.work['narrow_y_pdf_nodes']==0
    budget=R.NodeBudget(16);ledger=R._Ledger(ref,None,budget)
    value=R.normal_log_interval(np.asarray([.3,-.7]),np.asarray([.01,.02]),reference_policy=ref,ledger=ledger)
    assert np.isfinite(value).all() and budget.used==16 and ledger.work['narrow_y_pdf_nodes']==16


def test_frozen_policy_rejects_gl16_or_changed_span():
    _,ref=policies();R._check_policy(ref,32)
    for key,value in (('narrow_cdf_order',16),('narrow_cdf_exponent_span',.5)):
        changed=dict(ref,**{key:value})
        with pytest.raises(ValueError):R._check_policy(changed,32)
    with pytest.raises(ValueError):R._legendre(16)
    for order in (8,32,48):
        nodes,weights=R._legendre(order)
        assert len(nodes)==order and len(weights)==order and (weights>0).all()


def test_all_other_r34_math_ast_is_unchanged():
    import ast
    from pathlib import Path
    old=Path(R.__file__).with_name('round34_log_narrow_reference.py').read_text()
    new=Path(R.__file__).read_text()
    def definitions(source):
        return {n.name:n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    before,after=definitions(old),definitions(new)
    assert set(after)==set(before)
    for name in before:
        if name not in ('integrate','_check_policy'):
            assert ast.dump(before[name],include_attributes=False)==ast.dump(after[name],include_attributes=False),name


# The following driver tests use distinct analytic vectors or explicit mocks,
# never declared formal reference inputs or any stored input archive.
from experiments import round36_component_round_quadrature as Q


def drive(function, a=0., b=1., **kwargs):
    arrays, metadata, work = {}, {}, {}
    options = dict(epsabs=0., epsrel=1e-12, component_count=1,
                   available=arrays, metadata=metadata, work=work)
    options.update(kwargs)
    result = Q.quad_vec(function, a, b, **options)
    return result, arrays, metadata['outer_component_roundoff'], work


def test_gk21_rule_matches_original_scipy_constants_and_prefloor_arithmetic():
    from scipy.integrate._quad_vec import _quadrature_gk21
    function = lambda x: np.asarray([1.+.7*math.sin(45*x), math.exp(2*x)], np.float64)
    integral, truncation, rounding = Q.gk21(-.3, .8, function)
    old_integral, old_error, old_round = _quadrature_gk21(-.3, .8, function,
                                                       lambda a: float(np.max(np.abs(a))))
    np.testing.assert_array_equal(integral, old_integral)
    assert old_error == max(truncation, float(np.max(rounding)))
    assert old_round == float(np.max(rounding))


def test_initial_analytic_integral_checked_before_any_split():
    (integral, error, info), arrays, report, work = drive(
        lambda x: np.asarray([1.+x, 2.+x*x]), component_count=2)
    np.testing.assert_array_max_ulp(integral, np.asarray([1.5, 7/3]), maxulp=2)
    assert info.success and info.neval == 21 and report['split_count'] == 0
    assert report['initial_checked_before_split'] and report['complete']
    assert report['estimated_error'] == error < report['target']
    assert report['callback_evaluations'] == 21 and work['outer_intervals'] == 1
    assert arrays['outer_interval_integrals'].shape == (1, 2)


def test_disjoint_positive_vector_avoids_sum_of_max_roundoff_false_floor():
    from scipy.integrate import quad_vec as old_driver
    def function(x):
        result = np.zeros(16, np.float64)
        result[min(int(x), 15)] = 1.
        return result
    (integral, _, info), arrays, report, _ = drive(function, 0., 16.,
        points=list(range(1, 16)), component_count=16)
    _, _, old_info = old_driver(function, 0., 16., points=list(range(1, 16)),
        epsabs=0., epsrel=1e-12, norm='max', quadrature='gk21', limit=512, full_output=True)
    assert info.success and not old_info.success and old_info.status == 2
    np.testing.assert_array_max_ulp(integral, np.ones(16), maxulp=1)
    scalar_floor = math.fsum(float(np.max(v)) for v in arrays['outer_interval_roundoff'])
    assert scalar_floor > report['target'] > report['estimated_error']
    assert report['split_count'] == 0 and report['completed_interval_evaluations'] == 16


def test_historical_parent_roundoff_and_leftmost_ties_are_retained(monkeypatch):
    def exact_mock(left, right, function):
        for _ in range(21): function((left+right)/2)
        width = right-left
        return np.asarray([width]), float(width > .5), np.asarray([50*Q.EPSILON*width])
    monkeypatch.setattr(Q, 'gk21', exact_mock)
    (integral, _, info), arrays, report, _ = drive(lambda x: np.ones(1), 0., 2., points=[1.])
    assert info.success and report['split_count'] == 2
    np.testing.assert_array_equal(arrays['outer_interval_bounds'],
        [[0.,1.],[1.,2.],[0.,.5],[.5,1.],[1.,1.5],[1.5,2.]])
    np.testing.assert_array_equal(arrays['outer_active_interval_indices'], [2,3,4,5])
    assert integral[0] == 2.
    assert report['maximum_historical_roundoff'] == 200*Q.EPSILON
    assert report['callback_evaluations'] == 21*6


def test_current_leaf_integral_is_fsum_not_incremental_parent_replacement(monkeypatch):
    def mock(left, right, function):
        for _ in range(21): function((left+right)/2)
        # Deliberately ill-conditioned replacement bookkeeping, not a claimed
        # numerical quadrature rule: direct positive-leaf sum must give 3.
        value = 1e16 if (left,right)==(0.,1.) else (1. if left==0. else 2.)
        error = 1e16 if right-left>.5 else 0.
        return np.asarray([value]), error, np.asarray([0.])
    monkeypatch.setattr(Q, 'gk21', mock)
    (integral, _, info), arrays, report, _ = drive(lambda x: np.ones(1))
    assert info.success and integral[0] == 3.
    assert report['split_count'] == 1
    assert math.fsum(float(arrays['outer_interval_integrals'][i,0])
                     for i in arrays['outer_active_interval_indices']) == 3.


def test_roundoff_floor_is_a_failure_not_precision_relaxation():
    (_, _, info), _, report, _ = drive(lambda x: np.ones(1), epsrel=1e-15)
    assert not info.success and info.status == 'roundoff_floor'
    assert report['maximum_historical_roundoff'] >= report['target']
    assert report['complete'] and report['callback_evaluations'] == 21


def test_interval_limit_is_strict_and_does_not_evaluate_extra_children():
    (_, _, info), arrays, report, _ = drive(
        lambda x: np.asarray([1.+.7*math.sin(80*x)]), limit=1)
    assert not info.success and info.status == 'interval_limit'
    assert len(arrays['outer_interval_bounds']) == report['initial_interval_count'] == 1
    assert report['callback_evaluations'] == 21


@pytest.mark.parametrize('bad', (np.asarray([-1.]), np.asarray([math.nan]),
                                np.asarray([math.inf]), np.ones(1,np.float32)))
def test_invalid_integrand_preserves_empty_trace_and_callback(bad):
    arrays, metadata, work = {}, {}, {}
    with pytest.raises((ValueError,FloatingPointError)):
        Q.quad_vec(lambda x: bad, 0., 1., epsabs=0., epsrel=1e-12,
            component_count=1, available=arrays, metadata=metadata, work=work)
    report = metadata['outer_component_roundoff']
    assert report['callback_evaluations'] == 1 and report['completed_interval_evaluations'] == 0
    assert report['pending_interval_bounds'] == [0.,1.] and not report['complete']
    assert arrays['outer_interval_integrals'].shape == (0,1)
    assert report['estimated_error'] == report['target'] == 0.


def test_no_strict_midpoint_fails_without_any_callback_and_preserves_pending():
    arrays, metadata = {}, {}
    with pytest.raises(FloatingPointError, match='midpoint'):
        Q.quad_vec(lambda x: pytest.fail('no callback allowed'), 1., np.nextafter(1.,np.inf),
            epsabs=0., epsrel=1e-12, component_count=1, available=arrays, metadata=metadata)
    assert metadata['outer_component_roundoff']['callback_evaluations'] == 0
    assert not len(arrays['outer_interval_bounds'])


def test_failure_during_right_child_keeps_parent_active_and_left_child_history():
    arrays, metadata, work, calls = {}, {}, {}, []
    def function(x):
        calls.append(x)
        if len(calls)==47: raise RuntimeError('synthetic callback budget exhausted')
        return np.asarray([1.+.7*math.sin(80*x)])
    with pytest.raises(RuntimeError, match='synthetic callback budget'):
        Q.quad_vec(function,0.,1.,epsabs=0.,epsrel=1e-12,component_count=1,
            available=arrays,metadata=metadata,work=work)
    report=metadata['outer_component_roundoff']
    assert len(calls)==report['callback_evaluations']==47
    assert report['completed_interval_evaluations']==2 and report['split_count']==0
    assert report['pending_interval_bounds']==[.5,1.] and not report['complete']
    np.testing.assert_array_equal(arrays['outer_active_interval_indices'],[0])
    np.testing.assert_array_equal(arrays['outer_interval_bounds'],[[0.,1.],[0.,.5]])
    expected=math.fsum(float(v) for v in arrays['outer_interval_roundoff'][:,0])
    assert report['maximum_historical_roundoff']==expected


def test_partial_initial_partition_retains_prefix_without_filling_missing_interval():
    arrays, metadata, calls = {}, {}, []
    def function(x):
        calls.append(x)
        if len(calls)==26: raise RuntimeError('synthetic initial callback failure')
        return np.ones(1)
    with pytest.raises(RuntimeError, match='synthetic initial callback'):
        Q.quad_vec(function,0.,2.,points=[1.],epsabs=0.,epsrel=1e-12,
            component_count=1,available=arrays,metadata=metadata)
    report=metadata['outer_component_roundoff']
    assert not report['initial_checked_before_split'] and not report['complete']
    assert report['completed_interval_evaluations']==1 and report['callback_evaluations']==26
    np.testing.assert_array_equal(arrays['outer_active_interval_indices'],[0])
    assert report['pending_interval_bounds']==[1.,2.]


def test_reference_ledger_failure_keeps_outer_trace_and_exact_charged_work(monkeypatch):
    policy,ref=policies();budget=R.NodeBudget(10_000_000)
    monkeypatch.setattr(R,'outer_breaks',lambda *a:[])
    original_depth=R._Ledger.depth
    def limited(self,pilot=False):
        if not pilot and self.work['quadrature_depth_evaluations']==24:
            raise RuntimeError('synthetic outer ledger stop')
        original_depth(self,pilot=pilot)
    monkeypatch.setattr(R._Ledger,'depth',limited)
    with pytest.raises(R.ReferenceFailure,match='synthetic outer ledger stop') as caught:
        R.integrate(np.asarray([.2,-.1,1.3]),np.diag([.3,.4,.6]),np.asarray([[-.4,-.3,.6,.5]]),
            policy=policy,reference_policy=ref,order=32,budget=budget)
    failure=caught.value;report=failure.reference_metadata['outer_component_roundoff']
    assert report['callback_evaluations']==25
    assert failure.reference_work['quadrature_depth_evaluations']==24
    assert budget.used==failure.reference_work['cell_node_evaluations']>0
    assert report['completed_interval_evaluations']==1
    assert all(k in failure.available_arrays for k in Q.TRACE_FIELDS)


def test_outer_policy_strings_are_frozen():
    _,ref=policies()
    for key in ('outer_error_aggregation','outer_integral_accumulation','outer_refinement'):
        changed=dict(ref,**{key:'unregistered_change'})
        with pytest.raises(ValueError):R._check_policy(changed,32)
