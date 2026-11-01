"""Distinct pure toys for canonical log masses, guards and retained work."""
import ast
import inspect
import json
import math
import time

import numpy as np
import pytest
from scipy.special import log_ndtr, ndtr

from models import round26_rms_projective_pe as OLD
from models import round29_log_rms_projective_pe as M


def policy(**updates):
    result=dict(h=.29,floor_density=1e-30,quadrature_order=24,maximum_tile_side_in_h=2.,
        max_cell_node_evaluations=4_000_000,negative_tail_threshold=10.,negative_tail_series_terms=20,
        max_negative_tail_relative_remainder=1e-10,projection_regularization='rms_depth_xy_gaussian')
    result.update(updates)
    return result


def bounds():
    return np.asarray([[-.8,-.5,-.1,.7],[-.1,-.5,.6,.7]],np.float64)


def run(mu=None,factor=None,boxes=None,bias=None,**kwargs):
    return M.integrate(np.asarray([.17,-.28,1.3] if mu is None else mu,np.float64),
        np.asarray([[.21,0.,0.],[.04,.18,0.],[.03,-.02,.2]] if factor is None else factor,np.float64),
        bounds() if boxes is None else boxes,bias,policy=kwargs.pop('policy',policy()),**kwargs)


def only_metadata(result):
    return {name:value for name,value in result.items() if not isinstance(value,np.ndarray)}


@pytest.mark.parametrize('name',('validate','_prepare_validated','_positive_qr_factor','negative_log_k','integration_tiles'))
def test_original_admission_geometry_tail_and_tile_rules_are_unchanged(name):
    assert ast.dump(ast.parse(inspect.getsource(getattr(M,name))))==ast.dump(ast.parse(inspect.getsource(getattr(OLD,name))))


@pytest.mark.parametrize('factor',(
    [[.21,0.,0.],[.04,.18,0.],[0.,0.,0.]],
    [[.21,0.,0.],[.04,.18,0.],[.03,-.02,.2]],
))
def test_log_quadrature_matches_old_when_all_raw_values_are_representable(factor):
    mu=np.asarray([.17,-.28,1.3]);factor=np.asarray(factor,np.float64)
    bias=np.asarray([.23,-.17])
    new=M.integrate(mu,factor,bounds(),bias,policy=policy())
    old=OLD.integrate(mu,factor,bounds(),bias,policy=policy())
    assert new['passed'] and old['passed']
    np.testing.assert_allclose(np.exp(new['log_raw_mass']),old['raw_mass'],rtol=2e-13,atol=0)
    np.testing.assert_allclose(new['probability'],old['probability'],rtol=0,atol=1e-13)
    assert new['work']==old['work'] and new['ordinary_raw_unmaterialized_count']==0
    assert set(new['materialized_raw_indices'])=={0,1}
    json.dumps(only_metadata(new),allow_nan=False)


def test_local_lse_reports_underflow_without_claiming_pipeline_certificate():
    report=M.new_lse_diagnostics()
    value=M.positive_logsum(np.asarray([0.,-800.,-900.,-np.inf]),report,'cell_tiles')
    assert value==0
    loss=report['by_scope']['cell_tiles']
    assert loss['sum_count']==1 and loss['underflow_sum_count']==1 and loss['underflow_term_count']==2
    assert loss['maximum_log_scaled_omitted_bound']==math.log(2)-800
    assert loss['maximum_log_absolute_omitted_bound']==loss['maximum_log_relative_omitted_bound']==math.log(2)-800
    assert report['is_pipeline_error_bound'] is False
    assert report['excludes_addition_absorption_density_quadrature_and_roundoff'] is True


def test_vector_lse_keeps_empty_identity_distinct_from_positive_terms():
    report=M.new_lse_diagnostics()
    result=M.positive_logsum(np.asarray([[-np.inf,-np.inf],[-745.,-746.]]),report,'tile_nodes')
    assert np.isneginf(result[0]) and math.isfinite(result[1])
    assert report['by_scope']['tile_nodes']['sum_count']==2
    assert report['by_scope']['tile_nodes']['underflow_term_count']==0


@pytest.mark.parametrize('values',([0.,np.nan],[0.,np.inf],[1e308,-1e308]))
def test_nonfinite_terms_or_unrepresentable_relative_exponent_fail(values):
    with pytest.raises(M.NumericalFailure):
        M.positive_logsum(np.asarray(values),M.new_lse_diagnostics(),'tile_nodes')


def test_indexed_materialization_never_invents_positive_raw_zeros():
    indices,values,subnormal=M.materialize_raw(np.asarray([-2.,-710.,-744.,-800.]),False)
    np.testing.assert_array_equal(indices,[0,1,2])
    assert np.all(values>0) and subnormal==2
    zero_indices,zeros,sub=M.materialize_raw(np.full(3,-np.inf),True)
    np.testing.assert_array_equal(zero_indices,[0,1,2]);assert np.all(zeros==0) and sub==0


def log_positive_interval(lo,hi):
    left,right=log_ndtr(-lo),log_ndtr(-hi)
    return float(left+np.log(-np.expm1(right-left)))


def test_entire_far_fov_below_float64_keeps_independent_positive_log_mass():
    # Distinct from all R29 formal descriptors; homogeneous point at positive depth.
    b=np.asarray([[12.3,-.2,12.45,.15]],np.float64)
    result=run(mu=[0.,0.,1.2],factor=np.zeros((3,3)),boxes=b)
    assert result['passed'] and not result['structural_zero']
    expected=log_positive_interval(12.3/.29,12.45/.29)+math.log(ndtr(.15/.29)-ndtr(-.2/.29))
    assert abs(math.expm1(result['log_raw_mass'][0]-expected))<1e-8
    assert result['ordinary_raw_unmaterialized_count']==1 and len(result['materialized_raw_mass'])==0
    assert np.isfinite(result['log_raw_mass']).all() and result['zero_raw_cells']==0
    assert not result['all_raw_mass_zero'] and not result['area_prior_used']
    assert result['raw_mass_below_total_floor'] and result['probability'][0]==1
    assert result['lse_diagnostics']['by_scope']['floor']['underflow_term_count']==1
    json.dumps(only_metadata(result),allow_nan=False)


def test_visible_mass_survives_far_nodes_and_tiles_without_clipping():
    b=np.asarray([[0.,-.13,13.7,.21]],np.float64)
    mu=[.11,-.07,1.3]
    result=run(mu=mu,factor=np.zeros((3,3)),boxes=b)
    assert result['passed'] and result['ordinary_raw_unmaterialized_count']==0
    center=np.asarray(mu[:2])/mu[2]
    expected=float(np.prod(ndtr((b[0,2:]-center)/.29)-ndtr((b[0,:2]-center)/.29)))
    assert abs(math.expm1(result['log_raw_mass'][0]-math.log(expected)))<1e-10
    assert result['density_diagnostics']['minimum_log_density'] < -800
    assert result['lse_diagnostics']['by_scope']['cell_tiles']['underflow_term_count']>0


@pytest.mark.parametrize('z',(0.,-.8))
def test_exact_structural_zero_is_separate_and_json_metadata_remains_finite(z):
    result=run(mu=[.11,-.07,z],factor=[[.1,0,0],[0,.05,0],[0,0,0]],bias=np.asarray([.4,-.2]))
    assert result['passed'] and result['structural_zero']
    assert np.isneginf(result['log_raw_mass']).all() and np.isneginf(result['log_raw_fov_mass']).all()
    assert result['log_raw_mass_sum'] is None and result['all_raw_mass_zero']
    assert result['zero_raw_cells']==2 and not result['area_prior_used']
    assert result['ordinary_raw_unmaterialized_count']==0
    assert np.all(result['materialized_raw_mass']==0)
    assert result['work']['cell_node_evaluations']==0
    json.dumps(only_metadata(result),allow_nan=False)


def test_exact_area_prior_flag_only_for_zero_geometry_and_constant_bias():
    result=run(mu=[.11,-.07,-.8],factor=np.zeros((3,3)))
    assert result['passed'] and result['area_prior_used'] and result['all_raw_mass_zero']


def test_budget_failure_keeps_completed_native_and_unfinished_direct_fov_distinct():
    b=np.asarray([[.1,.2,.35,.45]],np.float64)
    result=run(boxes=b,policy=policy(max_cell_node_evaluations=576))
    assert not result['passed'] and result['work']['cell_node_evaluations']==576
    assert np.isfinite(result['partial_log_raw_mass']).all()
    assert result['partial_completed_cells'].tolist()==[True]
    assert result['partial_completed_fov'].tolist()==[False]
    assert result['partial_accumulated_tiles'].tolist()==[1]
    assert result['partial_accumulated_fov_tiles'].tolist()==[0]
    assert np.isneginf(result['partial_log_raw_fov_mass']).all()


def test_partial_cell_logs_are_not_marked_complete():
    # Total 35 native tiles fit allocation cap; first batch is charged, next fails.
    b=np.asarray([[0.,-.1,20.,.1]],np.float64)
    result=run(boxes=b,policy=policy(max_cell_node_evaluations=35*576))
    assert not result['passed']
    assert result['work']['cell_node_evaluations']==35*576
    assert result['partial_completed_cells'].tolist()==[True]
    assert result['partial_completed_fov'].tolist()==[False]
    # A direct manual accumulator failure tests an incomplete native cell.
    p=M.prepare(np.asarray([.17,-.28,1.3]),np.eye(3)*.1,b,None,policy())
    tiles,owners=M.integration_tiles(b,.58,100)
    work=dict(cell_node_evaluations=0,native_tiles_completed=0,fov_tiles_completed=0)
    accumulator=np.asarray([-np.inf]);counts=np.zeros(1,np.int64)
    diag=dict(constant_depth_nodes=0,positive_branch_nodes=0,negative_branch_nodes=0,
        negative_erfcx_nodes=0,negative_series_nodes=0,maximum_log_density=None,minimum_log_density=None,
        maximum_log_negative_tail_relative_remainder=None,minimum_scaled_negative_tail_series_lower_bound=None,
        maximum_scaled_negative_tail_series_upper_bound=None)
    with pytest.raises(M.NumericalFailure,match='budget'):
        M._integrate_log_tiles(tiles,owners,p,policy(max_cell_node_evaluations=32*576),None,work,diag,
            M.new_lse_diagnostics(),accumulator,counts,'native')
    assert counts.tolist()==[32] and np.isfinite(accumulator).all()


def test_deadline_retains_no_computed_mass():
    result=run(deadline=time.perf_counter()-1)
    assert not result['passed'] and not result['partial_completed_cells'].any()
    assert not result['partial_accumulated_tiles'].any() and np.isneginf(result['partial_log_raw_mass']).all()
    assert result['work']['cell_node_evaluations']==0


def test_front_mass_admission_is_not_extended_to_log_front():
    result=run(mu=[.1,-.2,-50.],factor=np.diag([.1,.1,1.]))
    assert not result['passed'] and 'front mass' in result['failure_reason']
    assert result['work']['cell_node_evaluations']==0


def test_original_floor_product_underflow_still_fails():
    result=run(boxes=np.asarray([[.1,.2,.3,.4]],np.float64),
               policy=policy(floor_density=np.nextafter(0.,1.)))
    assert not result['passed'] and 'area floor' in result['failure_reason']


def test_nonfinite_log_density_geometry_is_not_repaired():
    p=M.prepare(np.asarray([.1,.2,1.]),np.zeros((3,3)),bounds(),None,policy())
    diag=dict(constant_depth_nodes=0,maximum_log_density=None,minimum_log_density=None)
    with pytest.raises(M.NumericalFailure,match='log-density'):
        M.log_density(np.asarray([[1e308,0.]]),p,policy(),diag)
