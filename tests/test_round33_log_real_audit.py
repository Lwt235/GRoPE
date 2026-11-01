"""R33 pure saved-array, identity and fail-fast mocks; no integral or real archive."""
import ast


import copy


import json


import math


from pathlib import Path


from unittest.mock import patch


import numpy as np


import pytest


from scipy.special import logsumexp


from experiments import round33_log_real_audit as A


def work(**changes):
    value={k:0 for k in A.WORK_KEYS};value.update(changes);return value


def toy(zero=False):
    bank=dict(input__mu=np.array([.3,-.2,0. if zero else 2.]),input__factor=np.diag([.6,.4,0.]),
        input__bounds=np.array([[-1.,-1.,0.,1.],[0.,-1.,1.,1.]]),input__content_bias=np.array([.2,-.1]))
    raw=np.full(2,-np.inf) if zero else np.log(np.array([.2,.3]))
    norm=A.probabilities(raw,bank['input__bounds'],bank['input__content_bias'],A.C.POLICY['floor_density'])
    bank.update(log_raw_mass=raw,log_raw_fov_mass=np.array([float(logsumexp(raw))]),
        **{k:norm[k] for k in A.SUCCESS_ARRAYS if k not in ('log_raw_mass','log_raw_fov_mass')})
    info=A.independent_depth(bank['input__mu'],bank['input__factor'])
    scopes={k:A.local_lse_loss(np.array([],float)) for k in A.LSE_SCOPES}
    floor=norm['log_floor_mass'];scopes.update(native_total=A.local_lse_loss(raw),
        floor=A.combine_loss_reports([A.local_lse_loss(np.array([x,y])) for x,y in zip(raw,floor)]),
        floor_total=A.local_lse_loss(floor),floored_total=A.local_lse_loss(bank['log_floored_mass']),
        content_normalization=A.local_lse_loss(bank['log_floored_mass']+bank['input__content_bias']))
    row=dict(id='toy',order=32,call_id='toy__order32',status='complete',passed=True,attempted=True,
        work=work(),charged_cell_node_evaluations=0,reference_work_consistent=True,wall_seconds=.1,
        call_and_input_load_seconds=.1,peak_rss_kib_process=10,
        real_inputs_read=True,independent_real_reference=True,**A.SCOPE,**info,
        method='independent-SVD-fullX-log-conditional-pilot',reference_algorithm_is_unchanged_R27=False,
        inner_x_truncation_used=False,quadrature_error_bound_available=False,tail_truncation_used=False,
        maximum_full_x_panel_width=1.5,normal_logcdf_scalar_evaluations=0,
        scaled_outer_error_estimate=None,outer_convergence_success=True,
        outer_exp_diagnostics=dict(evaluation_vectors=0,component_evaluations=0,total_underflow_terms=0,
            local_node_omission_only=True,is_continuous_integral_or_pipeline_error_bound=False,
            scale_reselected=False,scale_is_certified_upper_bound=False),
        lse_diagnostics=dict(by_scope=scopes,is_pipeline_error_bound=False,exact_arithmetic_local_omission_only=True,
            excludes_addition_absorption_density_quadrature_and_roundoff=True),
        log_raw_mass_sum=None if zero else float(logsumexp(raw)),log_total_area_floor=float(logsumexp(floor)),
        log_normalizer=float(logsumexp(bank['log_floored_mass'])),log_content_normalizer=norm['log_content_normalizer'],
        normalization_error=abs(math.fsum(map(float,bank['probability']))-1.),
        direct_fov_log_absolute_error=None,direct_fov_absolute_error_materialized_diagnostic=0.)
    tail=dict(native_union_bound=True,inner_x_truncation_used=False,structural_zero_exemption=zero,
        excludes_quadrature_and_roundoff=True,raw_relative_denominator='computed retained FOV mass, not an interval enclosure')
    for k in ('depth_omitted_mass','native_raw_l1_absolute','direct_fov_absolute','raw_l1_over_reference_fov','probability_max_absolute'):
        tail[k]=None if zero and k=='raw_l1_over_reference_fov' else 0.;tail['log_'+k]=None
    row['truncation_bounds']=tail
    return row,bank


def check(row,bank):return A.reference_check(row,bank,A.C.POLICY,A.C.REFERENCE_POLICY,A.C.GATES)


@pytest.mark.parametrize('zero',[False,True])
def test_constant_saved_arithmetic_and_analytic_structural_zero(zero):
    row,bank=toy(zero);prepared={k:bank[k] for k in A.INPUT_FIELDS}
    assert A.validate_call(row,bank,prepared,A.C.REFERENCE_POLICY)['delivery_integrity_passed']
    assert check(row,bank)['passed']


def test_saved_probability_is_gated_not_replaced_with_recomputed_probability():
    row,b=toy();a=copy.deepcopy(b);a['probability']=a['probability'][::-1].copy()
    result=A.compare_saved(a,b,b,A.C.POLICY,A.C.GATES,False)
    assert not result['passed'] and result['maximum_probability_error']>.01


def test_subfloat_raw_relative_gate_survives_floor_dominance():
    bounds=np.array([[0.,0.,1.,1.],[1.,0.,2.,1.]]);bias=np.zeros(2)
    ref=np.array([-1000.,-1001.]);actual=ref+.01
    result=A.compare_log_raw(actual,ref,bounds,bias,1e-30,A.C.GATES,reference_comparison=True)
    assert result['maximum_probability_error']==0. and not result['passed']
    assert result['raw_fov_relative']>.01


def test_structural_zero_requires_exact_negative_infinity():
    with pytest.raises(ValueError):A.validate_logs(np.array([-10000.,-np.inf]),True)


def test_direct_fov_gate_does_not_disappear_when_probability_is_normalized():
    row,bank=toy();bank['log_raw_fov_mass']+=.02
    assert not check(row,bank)['passed']


def test_honest_failed_nonfinite_partial_keeps_bytes_but_cannot_pass():
    row,bank=toy();prepared={k:bank[k] for k in A.INPUT_FIELDS}
    row.update(passed=False,status='failed',failure_reason='nonfinite scaled result',work=work(outer_intervals=513))
    bank=dict(prepared,partial_log_raw_mass=np.array([-np.inf,-np.inf]),partial_log_raw_fov_mass=np.array([-np.inf]),
        partial_raw_computed=np.array([False,False,False]),outer_scaled_integral=np.array([np.nan,np.inf,-1.]))
    assert A.validate_call(row,bank,prepared,A.C.REFERENCE_POLICY)['delivery_integrity_passed']
    assert not check(row,bank)['passed']
    row.update(passed=True,status='complete')
    with pytest.raises(ValueError):A.validate_call(row,bank,prepared,A.C.REFERENCE_POLICY)


def test_partial_pilots_allow_only_uncomputed_negative_infinity():
    row,bank=toy();prepared={k:bank[k] for k in A.INPUT_FIELDS}
    row.update(passed=False,status='failed',failure_reason='pilot interrupted')
    bank=dict(prepared,pilot_zeta=np.array([0.,1.]),pilot_completed=np.array([True,False]),
        pilot_log_integrand=np.array([[-2.,-3.,-1.],[-np.inf,-np.inf,-np.inf]]),last_conditional_log_mass=np.array([-1.,-2.,-.5]))
    assert A.validate_call(row,bank,prepared,A.C.REFERENCE_POLICY)['delivery_integrity_passed']
    bank['pilot_log_integrand'][1,0]=0.
    with pytest.raises(ValueError):A.validate_call(row,bank,prepared,A.C.REFERENCE_POLICY)


def test_partial_raw_mask_cannot_claim_uncomputed_positive_mass():
    row,b=toy();prepared={k:b[k] for k in A.INPUT_FIELDS};row.update(passed=False,status='failed',failure_reason='interrupted')
    b=dict(prepared,partial_log_raw_mass=np.array([-1.,-np.inf]),partial_log_raw_fov_mass=np.array([-np.inf]),partial_raw_computed=np.array([False]*3))
    with pytest.raises(ValueError):A.validate_call(row,b,prepared,A.C.REFERENCE_POLICY)


def test_unattempted_row_cannot_invent_work_or_reference():
    row,b=toy();prepared={k:b[k] for k in A.INPUT_FIELDS}
    row.update(attempted=False,passed=False,status='failed',failure_reason='deadline',real_inputs_read=False,independent_real_reference=False)
    assert A.validate_call(row,{},prepared,A.C.REFERENCE_POLICY)['delivery_integrity_passed']
    row['work']['inner_x_nodes']=1
    with pytest.raises(ValueError):A.validate_call(row,{},prepared,A.C.REFERENCE_POLICY)


def test_depth_tail_union_bound_and_no_extra_front_multiplier():
    info=A.independent_depth(np.array([0.,0.,-5.]),np.diag([1.,1.,1.]))
    assert math.isclose(A.physical_depth_tail(info,18.),float(A.log_ndtr(-18.)),abs_tol=1e-12)
    central=A.independent_depth(np.zeros(3),np.eye(3))
    assert central['front_mass']==.5 and A.physical_depth_tail(central,18.)==A.log_ndtr(-18.)


def test_fixed_pilot_scales_cannot_be_reselected_or_front_normalized():
    row,b=toy();b['input__mu']=np.zeros(3);b['input__factor']=np.eye(3)
    row['work']=work(pilot_depth_evaluations=1,quadrature_depth_evaluations=21,depth_evaluations=22,outer_intervals=1)
    row.update(scaled_outer_error_estimate=1e-14,outer_convergence_success=True)
    b.update(initial_depth_edges=np.array([0.,18.]),pilot_zeta=np.array([9.]),pilot_log_integrand=np.array([[-3.,-2.,-1.]]),
        outer_log_scale=np.array([-3.,-2.,-1.]),outer_scaled_integral=np.exp(np.r_[b['log_raw_mass'],b['log_raw_fov_mass']]+np.array([3.,2.,1.])))
    b['original_depth_edges']=b['initial_depth_edges'].copy()
    partition=A.expected_depth_partition(b['original_depth_edges'])
    b.update({k:partition[k] for k in A.COALESCING_ARRAYS if k!='original_depth_edges'})
    row['depth_edge_coalescing']=partition['depth_edge_coalescing']
    info=A.independent_depth(b['input__mu'],b['input__factor'])
    assert A.scale_check(row,b,info,A.C.REFERENCE_POLICY,2e-12)['passed']
    b['outer_scaled_integral']*=.5
    assert not A.scale_check(row,b,info,A.C.REFERENCE_POLICY,2e-12)['passed']
    b['outer_log_scale'][0]+=.1
    with pytest.raises(ValueError):A.scale_check(row,b,info,A.C.REFERENCE_POLICY,2e-12)


def test_outer_underflow_diagnostics_are_local_and_counted():
    row,b=toy();b['input__mu']=np.zeros(3);b['input__factor']=np.eye(3)
    row['work']=work(quadrature_depth_evaluations=4,depth_evaluations=4)
    row['outer_exp_diagnostics'].update(evaluation_vectors=4,component_evaluations=12,total_underflow_terms=2)
    b.update(outer_underflow_counts=np.array([2,0,0],np.int64),outer_max_dropped_log_shift=np.array([-800.,-np.inf,-np.inf]),outer_max_log_shift=np.zeros(3))
    info=A.independent_depth(b['input__mu'],b['input__factor'])
    assert A.outer_loss_check(row,b,info)['passed']
    row['outer_exp_diagnostics']['is_continuous_integral_or_pipeline_error_bound']=True
    with pytest.raises(ValueError):A.outer_loss_check(row,b,info)


def test_local_lse_loss_does_not_count_addition_absorption_as_underflow():
    result=A.local_lse_loss(np.array([0.,-40.,-800.]))
    assert result['underflow_term_count']==1 and result['maximum_log_scaled_omitted_bound']==-800.


def test_archive_hash_and_exact_four_field_decode(tmp_path):
    row,b=toy();path=tmp_path/'toy.npz';np.savez(path,**b);h={k:A.C.array_sha(b[k]) for k in A.INPUT_FIELDS};calls=[]
    with patch.object(np,'load',wraps=np.load) as load:
        out=A.read_arrays(path,h,A.INPUT_FIELDS,lambda:calls.append(True))
    assert set(out)==set(A.INPUT_FIELDS) and len(calls)==4 and load.call_count==1
    with pytest.raises(ValueError):A.read_arrays(path,h)


def test_all_scope_flags_are_exact_false_and_no_numerical_imports():
    value=dict(real_inputs_read=False,independent_real_reference=False,**A.SCOPE);A.scope_check(value)
    for key in A.SCOPE:
        bad=dict(value);bad[key]=0
        with pytest.raises(ValueError):A.scope_check(bad)
    tree=ast.parse(Path(A.__file__).read_text())
    imports=[ast.unparse(n) for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
    assert not any('preflight' in n or 'models' in n or 'reference' in n for n in imports)


def test_work_ledger_arithmetic_and_success_budget():
    value=work(cell_node_evaluations=2,elementary_inner_nodes=2,inner_x_nodes=2,outer_intervals=513)
    assert A.work_check(value,A.C.REFERENCE_POLICY,require_budget=False)
    with pytest.raises(ValueError):A.work_check(value,A.C.REFERENCE_POLICY)
    value['inner_x_nodes']=1
    with pytest.raises(ValueError):A.work_check(value,A.C.REFERENCE_POLICY,require_budget=False)


def partition_toy(edges):
    row,bank=toy();bank['input__mu']=np.zeros(3);bank['input__factor']=np.eye(3)
    original=np.asarray(edges,np.float64);result=A.expected_depth_partition(original)
    bank['original_depth_edges']=original
    for key in ('initial_depth_edges','dropped_depth_edge_indices','dropped_depth_edge_values'):bank[key]=result[key]
    row['depth_edge_coalescing']=result['depth_edge_coalescing']
    row['work']['outer_geometry_breakpoints']=len(result['initial_depth_edges'])-2
    return row,bank,A.independent_depth(bank['input__mu'],bank['input__factor'])


def test_adjacent_interior_cut_removed_but_two_ulp_interval_retained():
    one=1.;adj=np.nextafter(one,2.);two=np.nextafter(adj,2.)
    result=A.expected_depth_partition(np.array([0.,one,adj,two,18.]))
    assert np.array_equal(result['initial_depth_edges'],[0.,one,two,18.])
    assert result['dropped_depth_edge_indices'].tolist()==[2]
    assert result['depth_edge_coalescing']['dropped'][0]['reason']=='no_strict_midpoint_from_previous_retained'


def test_negative_and_subnormal_chain_preserves_full_domain():
    tiny=np.nextafter(0.,1.);left=-1.;adj=np.nextafter(left,0.)
    original=np.array([-2.,left,adj,-2*tiny,-tiny,0.,tiny,2*tiny,18.])
    result=A.expected_depth_partition(original);retained=result['initial_depth_edges']
    assert retained[0]==original[0] and retained[-1]==original[-1]
    assert np.all(np.diff(retained)>0)
    mids=(retained[:-1]+retained[1:])/2
    assert np.all(retained[:-1]<mids) and np.all(mids<retained[1:])
    assert np.isin(retained,original).all()


def test_upper_adjacent_cut_backtracks_with_original_index_and_reason():
    upper=18.;near=np.nextafter(upper,0.)
    result=A.expected_depth_partition(np.array([0.,1.,near,upper]))
    assert np.array_equal(result['initial_depth_edges'],[0.,1.,18.])
    assert result['depth_edge_coalescing']['dropped']==[dict(original_index=2,value=near,reason='no_strict_midpoint_to_upper')]


def test_already_valid_partition_byte_identity_and_idempotence():
    original=np.array([-18.,-2.,0.,1.,18.]);result=A.expected_depth_partition(original)
    assert A.C.array_sha(result['initial_depth_edges'])==A.C.array_sha(original)
    assert result['dropped_depth_edge_indices'].size==0
    again=A.expected_depth_partition(result['initial_depth_edges'])
    assert A.C.array_sha(again['initial_depth_edges'])==A.C.array_sha(original)


def test_whole_domain_adjacent_endpoints_still_fails_strict_midpoint():
    original=np.array([1.,np.nextafter(1.,2.)]);result=A.expected_depth_partition(original)
    assert not result['strict_midpoints_exist']
    assert A.C.array_sha(result['initial_depth_edges'])==A.C.array_sha(original)
    assert not result['depth_edge_coalescing']['dropped']


def test_independent_coalescing_rejects_tampered_records_and_preserves_endpoint():
    row,b,info=partition_toy([0.,1.,np.nextafter(1.,2.),18.])
    assert A.coalescing_check(row,b,info,A.C.REFERENCE_POLICY)['passed']
    row['depth_edge_coalescing']['dropped'][0]['reason']='no_strict_midpoint_to_upper'
    with pytest.raises(ValueError):A.coalescing_check(row,b,info,A.C.REFERENCE_POLICY)
    row,b,info=partition_toy([0.,1.,np.nextafter(1.,2.),18.])
    b['initial_depth_edges'][-1]=np.nextafter(18.,0.)
    with pytest.raises(ValueError):A.coalescing_check(row,b,info,A.C.REFERENCE_POLICY)


def test_invalid_original_partition_can_only_be_retained_failure_evidence():
    row,b=toy();prepared={k:b[k] for k in A.INPUT_FIELDS}
    row.update(passed=False,status='failed',failure_reason='original partition not finite')
    partial=dict(prepared,original_depth_edges=np.array([0.,np.nan,18.]),initial_depth_edges=np.array([18.]),
        dropped_depth_edge_indices=np.array([],np.int64),dropped_depth_edge_values=np.array([],np.float64))
    row['depth_edge_coalescing']=dict(rule=A.COALESCING_RULE,original_interior_count=1,retained_interior_count=0,dropped_count=0,dropped=[])
    assert A.validate_call(row,partial,prepared,A.C.REFERENCE_POLICY)['delivery_integrity_passed']
    assert not check(row,partial)['passed']
    with pytest.raises(ValueError):A.expected_depth_partition(partial['original_depth_edges'])


def test_wrong_original_domain_and_dropped_values_are_rejected():
    row,b,info=partition_toy([0.,1.,np.nextafter(1.,2.),18.])
    b['dropped_depth_edge_values'][0]=1.
    with pytest.raises(ValueError):A.coalescing_check(row,b,info,A.C.REFERENCE_POLICY)
    row,b,info=partition_toy([.5,1.,18.])
    with pytest.raises(ValueError):A.coalescing_check(row,b,info,A.C.REFERENCE_POLICY)



def real_inputs(index=0,size=8):
    b=dict(input__mu=np.array([1.,-2.,3.]),input__factor=np.diag([.2,.3,.4]),
        input__bounds=A.literal_bounds(),input__content_bias=np.zeros(432))
    x=(index%(192//size))*size;y=(index//(192//size))*size
    spec=dict(id='mock',scene='mock_scene',pair=[30,45],pair_name='30_45',query_size=size,query_index=index,
        query_bounds=[x,y,x+size,y+size],array_sha256={k:A.C.array_sha(v) for k,v in b.items()})
    return b,spec


def test_real_input_identity_uses_four_canonical_fields_without_fixture_or_remapping():
    b,spec=real_inputs(13,24)
    result=A.input_check(b,copy.deepcopy(b),spec)
    assert result['passed'] and result['original_geometry_provenance_inherited']
    assert not result['new_geometric_transformation_performed']
    bad=copy.deepcopy(b);bad['input__mu'][0]=np.nextafter(1.,2.)
    with pytest.raises(ValueError):A.input_check(b,bad,spec)
    badspec=copy.deepcopy(spec);badspec['query_bounds'][0]+=1
    with pytest.raises(ValueError):A.input_check(b,b,badspec)


@pytest.mark.parametrize('field',['input__bounds','input__content_bias'])
def test_rehashed_changed_native_grid_or_bias_is_still_rejected(field):
    b,spec=real_inputs()
    b[field].flat[0]+=1e-5
    spec['array_sha256'][field]=A.C.array_sha(b[field])
    with pytest.raises(ValueError):A.input_check(b,b,spec)


def test_216_comparisons_are_exactly_72_dual_and_144_old_production_new():
    specs=[dict(id='query'+str(i)) for i in range(72)]
    plan=A.comparison_plan(specs)
    assert len(plan)==216
    assert sum(v['kind']=='new_orders' for v in plan)==72
    assert sum(v['kind']=='old_production_vs_new' for v in plan)==144
    assert len({(v['id'],v['kind'],v.get('new_order')) for v in plan})==216


def prefix_rows(count=6,first_fail=2,stage='reference'):
    rows=[];trigger=None
    for i in range(count):
        skip=trigger is not None;failed=i==first_fail or skip
        attempted=not skip and (not failed or stage=='reference')
        row=dict(call_id='mock__'+str(i),call_index=i,passed=not failed,attempted=attempted,
            real_inputs_read=not skip and stage!='phase_deadline',independent_real_reference=attempted,
            failure_stage='fail_fast' if skip else (stage if failed else None),
            not_attempted_reason='fail_fast' if skip else (stage if failed and not attempted else None),
            stop_trigger=copy.deepcopy(trigger),failure_reason='mock failure' if failed else None,
            archive_sha256='hash'+str(i),array_sha256={} if skip else {'input':'h'},work={},
            charged_cell_node_evaluations=0,reference_work_consistent=True)
        rows.append(row)
        if i==first_fail:trigger=dict(call_id=row['call_id'],index=i,reason=row['failure_reason'],
            archive_sha256=row['archive_sha256'],stage=stage)
    seal=dict(stop_trigger=trigger,executed_failed_calls=sum(v['attempted'] and not v['passed'] for v in rows),
        fail_fast_skipped_calls=sum(v['not_attempted_reason']=='fail_fast' for v in rows),
        unattempted_failure_calls=sum(not v['attempted'] and v['not_attempted_reason']!='fail_fast' for v in rows))
    return rows,seal


@pytest.mark.parametrize('stage',['reference','input_decode','phase_deadline'])
def test_first_failure_triggers_one_bound_empty_unread_zero_work_suffix(stage):
    rows,seal=prefix_rows(stage=stage)
    assert A.fail_fast_check(rows,seal)==seal
    encoded=json.dumps(A.fail_fast_check(rows,seal),allow_nan=False)
    assert 'mock__2' in encoded


@pytest.mark.parametrize('field,value',[('attempted',True),('real_inputs_read',True),('array_sha256',{'fake':'hash'}),
    ('charged_cell_node_evaluations',1),('stop_trigger',None),('work',{'cell_node_evaluations':1})])
def test_fail_fast_suffix_cannot_claim_read_call_arrays_work_or_wrong_trigger(field,value):
    rows,seal=prefix_rows();rows[-1][field]=value
    with pytest.raises(ValueError):A.fail_fast_check(rows,seal)


def test_global_trigger_is_first_failure_exact_archive_hash_and_none_for_all_success():
    rows,seal=prefix_rows();seal['stop_trigger']['archive_sha256']='wrong'
    with pytest.raises(ValueError):A.fail_fast_check(rows,seal)
    rows,seal=prefix_rows(first_fail=-1)
    assert A.fail_fast_check(rows,seal)['stop_trigger'] is None


def test_new_coalescing_boolean_is_native_json_bool_in_success_and_failure(tmp_path):
    for edges,expected in [(np.array([0.,1.,18.]),True),(np.array([1.,np.nextafter(1.,2.)]),False)]:
        result=A.expected_depth_partition(edges)
        assert type(result['strict_midpoints_exist']) is bool and result['strict_midpoints_exist'] is expected
        target=tmp_path/('valid.json' if expected else 'invalid.json')
        A.C.write(target,result['depth_edge_coalescing']|dict(strict_midpoints_exist=result['strict_midpoints_exist']))
        assert A.C.read(target)['strict_midpoints_exist'] is expected


def test_full_saved_arithmetic_json_is_serializable_without_default_handler(tmp_path):
    row,bank=toy();result=check(row,bank)
    A.C.write(tmp_path/'check.json',result)
    assert A.C.read(tmp_path/'check.json')['passed'] is True
    assert json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('stage,attempted,delivery_failed',[
    ('input_decode',False,True),('phase_deadline',False,False),
    ('reference',True,False),('fail_fast',False,False)])
def test_input_decode_failure_is_delivery_failure_not_numerical_counterexample(stage,attempted,delivery_failed):
    row,bank=toy();prepared={k:bank[k] for k in A.INPUT_FIELDS}
    row.update(status='failed',passed=False,attempted=attempted,failure_stage=stage,
        failure_reason='mock '+stage,real_inputs_read=attempted,independent_real_reference=attempted)
    payload=prepared if attempted else {}
    result=A.validate_call(row,payload,prepared,A.C.REFERENCE_POLICY)
    assert result['input_delivery_failed'] is delivery_failed
    assert result['delivery_integrity_passed'] is (not delivery_failed)
    assert check(row,payload)['status']=='reference_failed'
    assert json.dumps(result,allow_nan=False)


def test_metadata_freeze_forbids_decode_and_binds_72_ids_144_calls(tmp_path):
    run=tmp_path/'run';out=tmp_path/'audit';run.mkdir()
    specs=[dict(id='metadata'+str(i)) for i in range(72)];bindings={'old.json':'hash'}
    A.C.write(run/'query_specs.json',specs)
    lock=dict(version=1,created_unix=0.,runtime={},source_hashes={},policy=A.C.POLICY,
        reference_policy=A.C.REFERENCE_POLICY,gates=A.C.GATES,bindings=bindings,fail_fast=A.C.FAIL_FAST,
        case_ids=[s['id'] for s in specs],case_count=72,ordered_call_ids=[v[2] for v in A.ordered_calls(specs)],call_count=144,
        query_specs_sha256=A.C.sha(run/'query_specs.json'),audit_root=str(out),real_inputs_read=False,
        independent_real_reference=False,**A.SCOPE)
    A.C.write(run/'lock.json',lock);(run/'lock.sha256').write_text(A.C.sha(run/'lock.json')+'\n')
    with patch.object(np,'load',side_effect=AssertionError('metadata freeze must not decode')),\
         patch.object(A.C,'inherited_evidence',return_value=(bindings,specs)),\
         patch.object(A.C,'source_names',return_value=[]),patch.object(A.C,'check_runtime',return_value={}):
        A.freeze(run,out);assert A.joint_locks(run,out)[0]['call_count']==144
    design=A.C.read(out/'lock.json')
    assert design['real_inputs_read'] is False and design['independent_real_reference'] is False
    assert all(design[k] is False for k in A.SCOPE)


@pytest.mark.parametrize('failure_stage',['reference','input_decode'])
def test_post_seal_audit_retains_all_144_and_216_after_first_failure(tmp_path,failure_stage):
    run=tmp_path/'run';out=tmp_path/'audit';inherited=tmp_path/'old';run.mkdir();out.mkdir();inherited.mkdir()
    template,bank=toy();specs=[];preps=[];rows=[];banks={}
    prefix,failinfo=prefix_rows(144,2,stage=failure_stage)
    for i in range(72):
        name='toy'+str(i)
        identity=dict(id=name,scene='mock_scene',pair=[30,45],pair_name='30_45',query_size=8,query_index=i,query_bounds=[0,0,8,8])
        spec=dict(identity,policy_overrides={},production_archive_path='old/'+name,production_archive_sha256='hash',
            production_array_sha256={k:'hash' for k in ('log_raw_mass','log_raw_fov_mass','probability')})
        specs.append(spec);preps.append(dict(id=name,archive_path='prepared/'+name,array_sha256={k:'hash' for k in A.INPUT_FIELDS}))
        banks[str(run/'prepared'/name)]={k:bank[k] for k in A.INPUT_FIELDS}
        banks['old/'+name]={k:bank[k] for k in ('log_raw_mass','log_raw_fov_mass','probability')}
        for order in (32,48):
            index=len(rows);row=copy.deepcopy(template);control=prefix[index]
            row.update(identity,order=order,call_id=name+'__order'+str(order),call_index=index,
                archive_path='forward/'+name+str(order),array_sha256={k:'hash' for k in bank},
                failure_stage=control['failure_stage'],not_attempted_reason=control['not_attempted_reason'],stop_trigger=control['stop_trigger'])
            value=copy.deepcopy(bank)
            if index>=2:
                executed=index==2 and failure_stage=='reference'
                row.update(status='failed',passed=False,failure_reason='mock '+failure_stage+' failure',attempted=executed,
                    real_inputs_read=executed,independent_real_reference=executed)
                value={k:bank[k] for k in A.INPUT_FIELDS} if executed else {}
                row['array_sha256']={k:'hash' for k in value}
            rows.append(row);banks[str(run/row['archive_path'])]=value
    A.C.write(inherited/'identities.json',[dict(passed=True) for _ in range(6)])
    A.C.write(inherited/'near_constant_limit.json',dict(passed=True))
    lock=dict(policy=A.C.POLICY,reference_policy=A.C.REFERENCE_POLICY,gates=A.C.GATES)
    attempted=3 if failure_stage=='reference' else 2
    seal=dict(finished_unix=0,attempted_calls=attempted,reference_calls_completed=False,phase_wall_seconds=.1,
        total_cell_node_evaluations=0,phase_wall_passed=True,phase_node_budget_passed=True,**failinfo)
    sealed={'checked':False};old_decodes=[]
    def forward(*args):sealed['checked']=True;return lock,seal,rows,preps,specs
    def read(path,hashes,fields=None,on_decode=None):
        if str(path).startswith('old/'):
            assert sealed['checked'];old_decodes.append(str(path))
        result=copy.deepcopy(banks[str(path)])
        if on_decode:
            for _ in result:on_decode()
        return result
    with patch.object(A,'verify_forward',side_effect=forward),patch.object(A,'joint_locks'),\
         patch.object(A,'read_arrays',side_effect=read),patch.object(A.C,'sha',return_value='hash'),\
         patch.object(A.C,'OLD_AUDIT',inherited):
        result=A.audit(run,out)
    assert result['delivery_integrity_passed'] is (failure_stage!='input_decode')
    assert not result['numerical_preflight_passed']
    checks=A.C.read(out/'checks.json');comparisons=A.C.read(out/'comparisons.json');summary=A.C.read(out/'summary.json')
    assert len(checks)==144 and len(comparisons)==216 and len(old_decodes)==72
    assert checks[3]['producer_real_inputs_read'] is False and checks[3]['real_inputs_read'] is False
    assert checks[4]['producer_real_inputs_read'] is False and checks[4]['real_inputs_read'] is True
    assert summary['actual_reference_calls']==attempted and summary['calls_not_attempted']==144-attempted
    assert summary['input_delivery_failure_count']==int(failure_stage=='input_decode')
    assert checks[2]['input_delivery_failed'] is (failure_stage=='input_decode')
    assert all(checks[i]['input_delivery_failed'] is False for i in range(144) if i!=2)
    assert all(v['real_inputs_read'] is False and v['real_arrays_used'] is True for v in comparisons)
    assert all(v[k] is False for v in checks+comparisons+[summary,result] for k in A.SCOPE)
    assert 'independent_probability_reference' not in summary
    assert A.C.read(out/'completion.json')['verification_sha256']=='hash'
    with pytest.raises(ValueError):A.audit(run,out)
