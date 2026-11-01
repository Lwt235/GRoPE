"""Distinct saved-array/toy and mocked-population checks; never calls an integrator."""
import ast
import copy
import json
import math
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from scipy.special import logsumexp
from experiments import round36_component_round_audit as A


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
    scopes={k:dict(A.local_lse_loss(np.array([],float)),sum_count=0) for k in A.LSE_SCOPES}
    floor=norm['log_floor_mass'];scopes.update(native_total=A.local_lse_loss(raw),
        floor=A.combine_loss_reports([A.local_lse_loss(np.array([x,y])) for x,y in zip(raw,floor)]),
        floor_total=A.local_lse_loss(floor),floored_total=A.local_lse_loss(bank['log_floored_mass']),
        content_normalization=A.local_lse_loss(bank['log_floored_mass']+bank['input__content_bias']))
    row=dict(id='toy',order=32,call_id='toy__order32',status='complete',passed=True,attempted=True,
        work=work(),charged_cell_node_evaluations=0,reference_work_consistent=True,wall_seconds=.1,
        call_and_input_load_seconds=.1,peak_rss_kib_process=10,
        synthetic_inputs_read=True,independent_synthetic_reference=True,**A.SCOPE,**info,
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
    row['narrow_normal_quadrature_bound']=dict(order=8,branch_exponent_span=.25,proof_envelope=.5,
        computed_relative_remainder=1.8779378947512937e-13,conservative_relative_remainder_bound=1.88e-13,
        exact_arithmetic_only=True,includes_floating_point_error=False,is_pipeline_error_bound=False)
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
    row.update(attempted=False,passed=False,status='failed',failure_reason='deadline',synthetic_inputs_read=False,independent_synthetic_reference=False)
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


def test_input_hash_and_descriptor_are_both_required():
    row,b=toy();inputs={k:b[k] for k in A.INPUT_FIELDS}
    fixture=dict(id='toy',kind='base',**{k.removeprefix('input__'):v.tolist() for k,v in inputs.items()})
    spec=dict(id='toy',fixture=fixture,array_sha256={k:A.C.array_sha(v) for k,v in inputs.items()})
    assert A.input_check(inputs,inputs,spec)['passed']
    spec['fixture']['mu'][0]=1.
    with pytest.raises(ValueError):A.input_check(inputs,inputs,spec)


def test_archive_hash_and_exact_four_field_decode(tmp_path):
    row,b=toy();path=tmp_path/'toy.npz';np.savez(path,**b);h={k:A.C.array_sha(b[k]) for k in A.INPUT_FIELDS};calls=[]
    with patch.object(np,'load',wraps=np.load) as load:
        out=A.read_arrays(path,h,A.INPUT_FIELDS,lambda:calls.append(True))
    assert set(out)==set(A.INPUT_FIELDS) and len(calls)==4 and load.call_count==1
    with pytest.raises(ValueError):A.read_arrays(path,h)


def test_comparison_plan_covers_both_source_kinds_without_equal_order_collision():
    specs=[dict(id='toy'+str(i)) for i in range(23)]
    oracles=[]
    for i,spec in enumerate(specs):
        for order in ((32,48) if i<20 else (None,)):
            kind='r29_saved';oracles.append(dict(id=spec['id'],order=order,reference_kind=kind,oracle_id=kind+'__order'+str(order)))
        for order in (32,48):
            kind='r32_log_conditional';oracles.append(dict(id=spec['id'],order=order,reference_kind=kind,oracle_id=kind+'__order'+str(order)))
        for order in (32,48):
            kind='r34_log_narrow';oracles.append(dict(id=spec['id'],order=order,reference_kind=kind,oracle_id=kind+'__order'+str(order)))
    plan=A.comparison_plan(specs,oracles)
    assert len(oracles)==135 and len(plan)==339
    assert sum(v['kind']=='new_orders' for v in plan)==23
    assert sum(v['kind']=='old_production_vs_new' for v in plan)==46
    matched=[v for v in plan if v['kind']=='new_vs_old_reference']
    assert len(matched)==270 and sum(v['old_reference_kind']=='r29_saved' for v in matched)==86
    assert sum(v['old_reference_kind']=='r32_log_conditional' for v in matched)==92
    assert sum(v['old_reference_kind']=='r34_log_narrow' for v in matched)==92
    assert len({(v['id'],v['new_order'],v['reference_key']) for v in matched})==270
    with pytest.raises(ValueError):A.comparison_plan(specs,oracles+[oracles[0]])


def test_all_scope_flags_are_exact_false_and_no_numerical_imports():
    value=dict(synthetic_inputs_read=False,independent_synthetic_reference=False,**A.SCOPE);A.scope_check(value)
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


def test_full_population_failure_retention_and_per_row_decode(tmp_path):
    run=tmp_path/'run';out=tmp_path/'audit';old=tmp_path/'old';run.mkdir();out.mkdir();old.mkdir()
    template,bank=toy();specs=[];prep=[];rows=[];banks={};oracles=[]
    for i in range(23):
        name='toy'+str(i);specs.append(dict(id=name,policy_overrides={},source_archive_path='old/'+name,
            source_archive_sha256='hash',production_array_sha256={k:'hash' for k in ('log_raw_mass','log_raw_fov_mass','probability')}))
        prep.append(dict(id=name,archive_path='prepared/'+name,array_sha256={k:'hash' for k in A.INPUT_FIELDS}))
        banks[str(run/'prepared'/name)]={k:bank[k] for k in A.INPUT_FIELDS}
        banks['old/'+name]={k:bank[k] for k in ('log_raw_mass','log_raw_fov_mass','probability')}
        for order in (32,48):
            row=copy.deepcopy(template);row.update(id=name,order=order,call_id=name+'__order'+str(order),archive_path='forward/'+name+str(order),array_sha256={k:'hash' for k in bank})
            value=copy.deepcopy(bank)
            if i==0 and order==48:
                row.update(attempted=False,passed=False,status='failed',failure_reason='mock deadline',synthetic_inputs_read=False,independent_synthetic_reference=False,array_sha256={});value={}
            banks[str(run/row['archive_path'])]=value;rows.append(row)
        for order in ((32,48) if i<20 else (None,)):
            kind='r29_saved';oracles.append(dict(id=name,order=order,reference_kind=kind,oracle_id=kind+'__order'+str(order)))
        for order in (32,48):
            kind='r32_log_conditional';oracles.append(dict(id=name,order=order,reference_kind=kind,oracle_id=kind+'__order'+str(order)))
        for order in (32,48):
            kind='r34_log_narrow';oracles.append(dict(id=name,order=order,reference_kind=kind,oracle_id=kind+'__order'+str(order)))
    A.C.write(run/'oracle_specs.json',oracles);A.C.write(old/'identities.json',[dict(passed=True) for _ in range(6)]);A.C.write(old/'near_constant_limit.json',dict(passed=True))
    lock=dict(policy=A.C.POLICY,reference_policy=A.C.REFERENCE_POLICY,gates=A.C.GATES)
    seal=dict(finished_unix=0,attempted_calls=45,reference_calls_completed=False,phase_wall_seconds=.1,total_cell_node_evaluations=0,phase_wall_passed=True,phase_node_budget_passed=True)
    def read(path,hashes,fields=None,on_decode=None):
        result=copy.deepcopy(banks[str(path)])
        if on_decode:
            for _ in result:on_decode()
        return result
    with patch.object(A,'verify_forward',return_value=(lock,seal,rows,prep,specs)),patch.object(A,'joint_locks'),\
            patch.object(A,'read_arrays',side_effect=read),patch.object(A,'read_oracle',return_value={k:bank[k] for k in ('log_raw_mass','log_raw_fov_mass','probability')}),\
            patch.object(A.C,'sha',return_value='hash'),patch.object(A.C,'OLD_AUDIT',old):
        result=A.audit(run,out)
    assert result['delivery_integrity_passed'] and not result['numerical_preflight_passed']
    checks=A.C.read(out/'checks.json');comparisons=A.C.read(out/'comparisons.json')
    assert len(checks)==46 and len(comparisons)==339 and checks[1]['synthetic_inputs_read'] is False
    assert checks[0]['synthetic_inputs_read'] is True
    assert all(r['synthetic_inputs_read'] is False and r['synthetic_arrays_used'] is True for r in comparisons)
    assert A.C.read(out/'completion.json')['verification_sha256']=='hash'
    with pytest.raises(ValueError):A.audit(run,out)


def test_joint_freeze_is_metadata_only_with_npz_decode_forbidden(tmp_path):
    run=tmp_path/'run';out=tmp_path/'audit';run.mkdir()
    specs=[dict(id='metadata'+str(i)) for i in range(23)];oracles=[];bindings={'prior.json':'priorhash'}
    A.C.write(run/'query_specs.json',specs);A.C.write(run/'oracle_specs.json',oracles)
    lock=dict(version=1,created_unix=0.,source_hashes={},runtime={},policy=A.C.POLICY,reference_policy=A.C.REFERENCE_POLICY,
        gates=A.C.GATES,bindings=bindings,case_ids=[s['id'] for s in specs],case_count=23,
        ordered_call_ids=[v[2] for v in A.ordered_calls(specs)],call_count=46,audit_root=str(out),
        query_specs_sha256=A.C.sha(run/'query_specs.json'),oracle_specs_sha256=A.C.sha(run/'oracle_specs.json'),
        synthetic_inputs_read=False,independent_synthetic_reference=False,**A.SCOPE)
    A.C.write(run/'lock.json',lock);(run/'lock.sha256').write_text(A.C.sha(run/'lock.json')+'\n')
    with patch.object(np,'load',side_effect=AssertionError('no decode before freeze')),\
            patch.object(A.C,'inherited_evidence',return_value=(bindings,specs,oracles)),\
            patch.object(A.C,'check_runtime',return_value={}),patch.object(A.C,'source_names',return_value=[]):
        A.freeze(run,out)
        assert A.joint_locks(run,out)[0]==lock
    assert A.C.read(out/'lock.json')['synthetic_inputs_read'] is False
    with pytest.raises(ValueError):A.freeze(run,out)


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


def test_gl8_bound_uses_independent_rational_coefficient_and_explicit_local_scope():
    row,bank=toy();result=A.narrow_gl8_check(row,A.C.REFERENCE_POLICY)
    assert result['passed'] and result['charged_narrow_normal_nodes']==0
    assert math.isclose(result['independently_computed_relative_remainder'],1.8779378947512937e-13,rel_tol=1e-15)
    assert result['exact_arithmetic_only'] and not result['includes_floating_point_error'] and not result['is_pipeline_error_bound']
    assert json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('field,value',[
    ('order',16),('branch_exponent_span',.5),('proof_envelope',.25),
    ('computed_relative_remainder',0.),('computed_relative_remainder',1e-15),
    ('conservative_relative_remainder_bound',1e-8),('exact_arithmetic_only',1),
    ('includes_floating_point_error',True),('is_pipeline_error_bound',True)])
def test_local_gl8_bound_tampering_or_pipeline_claim_cannot_pass(field,value):
    row,bank=toy();row['narrow_normal_quadrature_bound'][field]=value
    with pytest.raises(ValueError):A.narrow_gl8_check(row,A.C.REFERENCE_POLICY)


def test_success_narrow_charge_matches_eight_per_lse_interval_not_just_divisibility():
    row,bank=toy();row['lse_diagnostics']['by_scope']['narrow_normal_nodes']['sum_count']=3
    row['work']=work(narrow_y_pdf_nodes=24,elementary_inner_nodes=24,cell_node_evaluations=24)
    assert A.narrow_gl8_check(row,A.C.REFERENCE_POLICY)['narrow_interval_count']==3
    row['work']['narrow_y_pdf_nodes']=16
    with pytest.raises(ValueError):A.narrow_gl8_check(row,A.C.REFERENCE_POLICY)
    bad=work(narrow_y_pdf_nodes=7,elementary_inner_nodes=7,cell_node_evaluations=7)
    with pytest.raises(ValueError):A.work_check(bad,A.C.REFERENCE_POLICY,require_budget=False)


@pytest.mark.parametrize('suffix',['','__legacy'])
def test_old_oracle_field_mapping_accepts_unsuffixed_r32_and_legacy_without_decoding_inputs(tmp_path,suffix):
    fields={n:n+suffix for n in ('log_raw_mass','log_raw_fov_mass','probability')}
    values=dict(log_raw_mass=np.array([-1000.,-1001.]),log_raw_fov_mass=np.array([-999.6867383124818]),probability=np.array([.5,.5]))
    path=tmp_path/'oracle.npz';np.savez(path,**{fields[n]:v for n,v in values.items()},forbidden=np.array([123.]))
    spec=dict(source_archive_path=str(path),source_archive_sha256=A.C.sha(path),source_fields=fields,
        array_sha256={n:A.C.array_sha(v) for n,v in values.items()})
    result=A.read_oracle(spec)
    assert set(result)==set(values) and all(np.array_equal(result[n],v) for n,v in values.items())


def test_coalescing_native_boolean_and_full_saved_check_json_are_serializable(tmp_path):
    for edges in (np.array([0.,1.,18.]),np.array([1.,np.nextafter(1.,2.)])):
        result=A.expected_depth_partition(edges)
        assert type(result['strict_midpoints_exist']) is bool
        json.dumps(result['depth_edge_coalescing']|dict(strict_midpoints_exist=result['strict_midpoints_exist']),allow_nan=False)
    row,bank=toy();A.C.write(tmp_path/'saved_check.json',check(row,bank))
    assert A.C.read(tmp_path/'saved_check.json')['narrow_normal_quadrature']['passed']


def component_trace(kind='initial'):
    """Saved arrays only: artificial positive GK21 outputs, no rule execution."""
    row,bank=toy();row['order']=48;row['work']=work()
    bank['initial_depth_edges']=np.array([0.,1.,2.])
    bounds=[[0.,1.],[1.,2.]];values=[[1.,.2,.1],[.1,1.,.2]];errors=[0.,0.];active=[0,1]
    splits=0;pending=None;status='converged';complete=True
    if kind in ('split','orphan','both_tied'):
        errors[0]=1e-12
        bounds.extend([[0.,.5],[.5,1.]])
        values.extend([[.4,.08,.04],[.6,.12,.06]]);errors.extend([1e-15,1e-15])
        active=[2,3,1];splits=1
        if kind=='both_tied':
            errors[1]=1e-12;bounds.extend([[1.,1.5],[1.5,2.]])
            values.extend([[.05,.5,.1],[.05,.5,.1]]);errors.extend([1e-15,1e-15]);active=[2,3,4,5];splits=2
        if kind=='orphan':
            bounds=bounds[:-1];values=values[:-1];errors=errors[:-1];active=[0,1];splits=0
            pending=[.5,1.];status='exception';complete=False
    elif kind=='partial_initial':
        bounds=bounds[:1];values=values[:1];errors=errors[:1];active=[0]
        pending=[1.,2.];status='exception';complete=False
    elif kind=='empty':
        bounds=[];values=[];errors=[];active=[];pending=[0.,1.];status='exception';complete=False
    a=np.asarray(bounds,np.float64).reshape(-1,2);v=np.asarray(values,np.float64).reshape(-1,3)
    e=np.asarray(errors,np.float64);r=50*np.finfo(float).eps*v;active=np.asarray(active,np.int64)
    integral=np.asarray([math.fsum(float(v[i,j]) for i in active) for j in range(3)])
    trunc=math.fsum(float(e[i]) for i in active)
    history=np.asarray([math.fsum(map(float,r[:,j])) for j in range(3)])
    rounding=float(history.max());estimate=trunc+rounding;target=1e-12*float(abs(integral).max())/8
    callbacks=21*len(a)+(5 if not complete else 0)
    row['work'].update(outer_intervals=len(active),quadrature_depth_evaluations=callbacks,depth_evaluations=callbacks)
    row['scaled_outer_error_estimate']=estimate
    bank.update(outer_interval_bounds=a,outer_interval_integrals=v,outer_interval_truncation_error=e,
        outer_interval_roundoff=r,outer_active_interval_indices=active,outer_scaled_integral=integral)
    row['outer_component_roundoff']=dict(complete=complete,success=bool(complete and estimate<target),status=status,
        sum_active_truncation_error=trunc,maximum_historical_roundoff=rounding,estimated_error=estimate,target=target,
        initial_interval_count=2,completed_interval_evaluations=len(a),split_count=splits,callback_evaluations=callbacks,
        initial_checked_before_split=bool(len(a)>=2),roundoff_epsilon=float(np.finfo(float).eps),roundoff_factor=50.,
        pending_interval_bounds=pending,error_estimate_is_not_true_error_certificate=True)
    return row,bank,dict(structural_branch='stochastic_depth')


def component_check(row,bank,info,success=True):
    return A.component_roundoff_check(row,bank,info,A.C.REFERENCE_POLICY,2e-12,require_success=success)


@pytest.mark.parametrize('kind',['initial','split','both_tied'])
def test_saved_component_trace_replays_fsum_history_partition_and_strict_convergence(kind):
    row,bank,info=component_trace(kind);result=component_check(row,bank,info)
    assert result['passed'] and result['strict_convergence'] and result['history_includes_discarded_parents']
    assert result['callback_evaluations']==21*len(bank['outer_interval_bounds'])
    assert json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('kind',['orphan','partial_initial','empty'])
def test_failed_pending_rule_keeps_history_orphan_and_active_parent_without_fake_work(kind):
    row,bank,info=component_trace(kind)
    result=component_check(row,bank,info,False)
    assert result['passed'] and not result['strict_convergence']
    with pytest.raises(ValueError):component_check(row,bank,info,True)


def test_callback_that_raises_before_ledger_depth_increment_is_retained_only_on_failure():
    row,bank,info=component_trace('empty');row['work']['quadrature_depth_evaluations']-=1
    assert component_check(row,bank,info,False)['passed']
    row['work']['quadrature_depth_evaluations']-=1
    with pytest.raises(ValueError):component_check(row,bank,info,False)


def test_ignored_historical_parent_roundoff_changes_error_and_is_rejected():
    row,bank,info=component_trace('split');report=row['outer_component_roundoff']
    active=bank['outer_active_interval_indices'];r=bank['outer_interval_roundoff']
    wrong=float(np.max(np.sum(r[active],axis=0)))
    assert wrong<report['maximum_historical_roundoff']
    report['maximum_historical_roundoff']=wrong
    with pytest.raises(ValueError):component_check(row,bank,info)


def test_initial_partition_must_stop_if_already_converged():
    row,bank,info=component_trace('split');bank['outer_interval_truncation_error'][0]=0.
    with pytest.raises(ValueError):component_check(row,bank,info)


def test_exact_priority_tie_selects_leftmost_parent_first():
    row,bank,info=component_trace('both_tied')
    bank['outer_interval_bounds'][2:4]=np.array([[1.,1.5],[1.5,2.]])
    with pytest.raises(ValueError):component_check(row,bank,info)


@pytest.mark.parametrize('kind',['split','orphan'])
@pytest.mark.parametrize('stop',['roundoff','interval_limit'])
def test_completed_or_pending_split_cannot_bypass_a_prior_stopping_guard(kind,stop):
    row,bank,info=component_trace(kind);policy=dict(A.C.REFERENCE_POLICY)
    if stop=='roundoff':policy['outer_relative_tolerances']=[1e-11,1e-15]
    else:policy['maximum_depth_intervals']=2
    with pytest.raises(ValueError):A.component_roundoff_check(row,bank,info,policy,2e-12,require_success=kind=='split')


def test_active_leaf_order_is_deterministic_left_to_right():
    row,bank,info=component_trace('split')
    bank['outer_active_interval_indices']=bank['outer_active_interval_indices'][::-1].copy()
    with pytest.raises(ValueError):component_check(row,bank,info)


@pytest.mark.parametrize('change',['bounds','roundoff','active','counter','target','negative','nan','strict'])
def test_outer_trace_tampering_is_rejected(change):
    row,bank,info=component_trace('split')
    if change=='bounds':bank['outer_interval_bounds'][2,0]=.01
    elif change=='roundoff':bank['outer_interval_roundoff'][0,0]*=2
    elif change=='active':bank['outer_active_interval_indices'][0]=0
    elif change=='counter':row['outer_component_roundoff']['callback_evaluations']+=1
    elif change=='target':row['outer_component_roundoff']['target']*=2
    elif change=='negative':bank['outer_interval_integrals'][0,0]=-1.
    elif change=='nan':bank['outer_interval_truncation_error'][0]=np.nan
    elif change=='strict':row['outer_component_roundoff']['success']=False
    with pytest.raises(ValueError):component_check(row,bank,info)


def test_normal_variable_and_constant_trace_scope_remain_distinct():
    row,bank=toy();info=A.independent_depth(bank['input__mu'],bank['input__factor'])
    assert not component_check(row,bank,info)['applicable']
    bank['outer_active_interval_indices']=np.array([],np.int64)
    with pytest.raises(ValueError):component_check(row,bank,info)
