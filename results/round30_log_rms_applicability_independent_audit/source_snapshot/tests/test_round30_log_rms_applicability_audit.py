"""Distinct synthetic R30 identity/log-delivery toys; no model integration."""
import ast
import math
from pathlib import Path
import numpy as np
import pytest
from scipy.special import logsumexp

from experiments import round30_log_rms_applicability_audit as A


def inputs():
    bank=dict(input__mu=np.asarray([.3,-.2,2.]),input__factor=np.diag([.1,.2,.3]),
        input__bounds=A.literal_bounds(),input__content_bias=np.zeros(432))
    spec=dict(query_size=24,query_index=1,query_bounds=[24,0,48,24])
    return bank,spec


def test_four_input_bytes_are_copied_without_new_geometry_transform():
    old,spec=inputs();actual={n:v.copy() for n,v in old.items()}
    check=A.input_check(old,actual,spec)
    assert check['passed'] and check['canonical_input_bytes_exact']
    assert not check['new_geometric_transformation_performed']
    assert check['original_geometry_provenance_inherited']
    assert all(check[k] is False for k in A.SCOPE)


@pytest.mark.parametrize('change',('mean_ulp','factor_ulp','dtype','bounds','bias','extra','query_index','query_bounds'))
def test_identity_gate_rejects_any_changed_input_or_query(change):
    old,spec=inputs();actual={n:v.copy() for n,v in old.items()}
    if change=='mean_ulp':actual['input__mu'][0]=np.nextafter(actual['input__mu'][0],np.inf)
    elif change=='factor_ulp':actual['input__factor'][0,0]=np.nextafter(actual['input__factor'][0,0],np.inf)
    elif change=='dtype':actual['input__mu']=actual['input__mu'].astype(np.float32)
    elif change=='bounds':actual['input__bounds'][0,0]-=.5/8
    elif change=='bias':actual['input__content_bias'][0]=1e-100
    elif change=='extra':actual['mapping_matrix']=np.eye(3)
    elif change=='query_index':spec['query_index']=2
    else:spec['query_bounds'][0]+=1
    with pytest.raises(ValueError):A.input_check(old,actual,spec)


def test_reader_decodes_only_the_four_model_fields_and_tracks_actual_reads(tmp_path):
    old,_=inputs();path=tmp_path/'source.npz';np.savez(path,**old,forbidden=np.asarray([object()],object))
    hashes={n:A.C.array_sha(v) for n,v in old.items()};events=[]
    actual=A.read_arrays(path,hashes,fields=A.C.OLD_FIELDS,on_decode=lambda:events.append(True))
    assert len(events)==4 and set(actual)==set(old)
    with pytest.raises(ValueError,match='unexpected'):A.read_arrays(path,hashes)


def test_input_audit_retains72_rows_and_blocks_forward_on_one_failure(tmp_path,monkeypatch):
    run,out=tmp_path/'producer',tmp_path/'audit';run.mkdir();out.mkdir()
    bank,spec=inputs();source=tmp_path/'old.npz';archive=run/'copy.npz'
    np.savez(source,**bank);np.savez(archive,**bank)
    for path in (run/'lock.json',run/'prepared_complete.json',out/'lock.json'):A.C.write(path,{'synthetic':True})
    (run/'prepared').mkdir();A.C.write(run/'prepared/completion.json',{'synthetic':True})
    specs=[dict(spec,id='toy'+str(i),scene='toy',source_archive_path=str(source),source_archive_sha256=A.C.sha(source),
        array_sha256={n:A.C.array_sha(v) for n,v in bank.items()}) for i in range(72)]
    rows=[dict(passed=i!=18,failure_reason='synthetic failed input',archive_path='copy.npz',archive_sha256=A.C.sha(archive),
        array_sha256={n:A.C.array_sha(v) for n,v in bank.items()}) for i in range(72)]
    monkeypatch.setattr(A,'verify_prepared',lambda *a:({'query_specs_sha256':'toy'},dict(finished_unix=0.),rows,specs))
    monkeypatch.setattr(A,'joint_locks',lambda *a:None)
    with pytest.raises(RuntimeError,match='no forward authorized'):A.input_audit(run,out)
    checks=A.C.read(out/'input_audit/checks.json');receipt=A.C.read(out/'input_verification.json')
    assert len(checks)==72 and sum(r['passed'] for r in checks)==71
    assert checks[18]['real_inputs_read'] is False and checks[19]['real_inputs_read'] is True
    assert receipt['real_inputs_read'] is True and not receipt['passed'] and not receipt['all_input_bytes_exact']
    done=A.C.read(out/'input_audit/completion.json');A.C.verify_inventory(out/'input_audit',done['files'],exclude=('completion.json',))
    assert done['verification_sha256']==A.C.sha(out/'input_verification.json')


def failed_row(**changes):
    row=dict(passed=False,attempted=False,status='failed',failure_reason='phase deadline',work={},
        wall_seconds=0.,call_and_input_load_seconds=0.,peak_rss_kib_process=1,real_inputs_read=False,**A.SCOPE)
    row.update(changes);return row


def test_unattempted_failed_row_is_intact_delivery_without_fake_probability():
    bank,_=inputs();check=A.validate_forward_row(failed_row(),{},bank,A.C.POLICY)
    assert check['delivery_integrity_passed'] and not check['model_completed'] and not check['attempted']


def test_failed_partial_logs_masks_and_exact_tile_work_are_retained():
    bank,_=inputs();partial=dict(bank)
    partial.update(partial_log_raw_mass=np.r_[-1000.,np.full(431,-np.inf)],partial_log_raw_fov_mass=np.asarray([-np.inf]),
        partial_completed_cells=np.zeros(432,bool),partial_completed_fov=np.zeros(1,bool),
        partial_accumulated_tiles=np.r_[np.asarray([2],np.int64),np.zeros(431,np.int64)],partial_accumulated_fov_tiles=np.zeros(1,np.int64))
    row=failed_row(attempted=True,real_inputs_read=True,structural_zero=False,
        work=dict(cell_node_evaluations=1152,native_tiles_completed=2,fov_tiles_completed=0))
    check=A.validate_forward_row(row,partial,bank,A.C.POLICY)
    assert check['partial_delivery']['incomplete_finite_native_cells']==1 and not check['model_completed']
    partial['partial_completed_cells'][0]=True
    with pytest.raises(ValueError,match='completion'):A.validate_forward_row(row,partial,bank,A.C.POLICY)


@pytest.mark.parametrize('change',('missing_reason','negative_work','fake_work','fake_probability','read_flag','scope'))
def test_failed_row_cannot_hide_status_or_invent_outputs(change):
    bank,_=inputs();row=failed_row();output={}
    if change=='missing_reason':row['failure_reason']=' '
    elif change=='negative_work':row['work']={'cell_node_evaluations':-1}
    elif change=='fake_work':row['work']={'cell_node_evaluations':1}
    elif change=='fake_probability':output['probability']=np.zeros(432)
    elif change=='read_flag':output=dict(bank)
    else:row['independent_probability_reference']=True
    with pytest.raises(ValueError):A.validate_forward_row(row,output,bank,A.C.POLICY)


def test_subfloat_positive_mass_materialization_is_not_structural_zero():
    logs=np.asarray([-1200.,-710.,math.log(.1)]);ordinary=np.exp(logs)
    bank=dict(log_raw_mass=logs,materialized_raw_indices=np.asarray([1,2],np.int64),materialized_raw_mass=ordinary[1:])
    assert A.materialization_check(bank,False)==dict(ordinary_raw_unmaterialized_count=1,ordinary_raw_subnormal_count=1)
    with pytest.raises(ValueError):A.validate_logs(logs,True)


def test_no_model_or_probability_reference_import_and_cli_guard_last():
    tree=ast.parse(Path(A.__file__).read_text());imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    assert not any(n.startswith('models') or 'reference' in n or 'preflight' in n for n in imports)
    assert isinstance(tree.body[-1],ast.If) and '__name__' in ast.unparse(tree.body[-1].test)


def test_freeze_refuses_previous_preparation_before_metadata_or_decode(tmp_path,monkeypatch):
    run=tmp_path/'run';run.mkdir();(run/'prepared').mkdir()
    monkeypatch.setattr(A,'verify_producer_lock',lambda *a:pytest.fail('must stop before metadata'))
    with pytest.raises(ValueError):A.freeze(run,tmp_path/'audit')


def test_early_interrupt_seals_no_array_read_and_failed_delivery(tmp_path,monkeypatch):
    def stop(*a):raise KeyboardInterrupt('synthetic interruption')
    monkeypatch.setattr(A,'verify_forward',stop)
    with pytest.raises(KeyboardInterrupt):A.audit(tmp_path/'run',tmp_path)
    done=A.C.read(tmp_path/'completion.json');failure=A.C.read(tmp_path/'failure.json')
    assert not done['passed'] and failure['real_inputs_read'] is False
    assert all(failure[k] is False for k in A.SCOPE)
    A.C.verify_inventory(tmp_path,done['files'],exclude=('completion.json',))


def native():
    return np.asarray([[-.8,-.4,.2,.7],[.2,-.4,1.2,.7]],np.float64)


def test_complete_saved_structural_zero_path_checks_log_floor_and_metadata():
    bounds=native();logs=np.full(2,-np.inf);bias=np.zeros(2)
    normalized=A.probabilities(logs,bounds,bias,1e-30)
    bank=dict(input__mu=np.zeros(3),input__factor=np.zeros((3,3)),input__bounds=bounds,input__content_bias=bias,
        log_raw_mass=logs,log_raw_fov_mass=np.asarray([-np.inf]),
        log_floored_mass=normalized['log_floored_mass'],log_probability=normalized['log_probability'],probability=normalized['probability'],
        materialized_raw_indices=np.arange(2,dtype=np.int64),materialized_raw_mass=np.zeros(2))
    report=dict(by_scope={name:dict(sum_count=0,underflow_sum_count=0,underflow_term_count=0,
        maximum_log_scaled_omitted_bound=None,maximum_log_absolute_omitted_bound=None,maximum_log_relative_omitted_bound=None) for name in A.LSE_SCOPES},
        exact_arithmetic_local_omission_only=True,is_pipeline_error_bound=False,excludes_addition_absorption_density_quadrature_and_roundoff=True)
    logfloor=normalized['log_floor_mass']
    for name,terms in dict(native_total=logs,floor=np.column_stack((logs,logfloor)),floor_total=logfloor,
                          floored_total=logfloor,content_normalization=logfloor).items():
        report['by_scope'][name]=A.local_lse_diagnostic(terms)
    row=dict(passed=True,structural_branch='zero_depth_rms',structural_zero=True,rms_depth=0.,normalized_depth_mean=None,normalized_depth_std=None,
        front_mass=0.,quadrature_order=24,maximum_tile_side_in_h=2.,h=.25,floor_density=1e-30,projection_regularization='rms_depth_xy_gaussian',
        canonical_raw_representation='log_mass',ordinary_materialization_is_canonical=False,tail_truncation_used=False,
        quadrature_error_bound_available=False,exact_arithmetic_gl_bound_available=True,native_tile_count=0,fov_tile_count=0,
        work=dict(cell_node_evaluations=0,native_tiles_completed=0,fov_tiles_completed=0),
        analytic_truncation_bound=dict(exact_arithmetic_only=True,excludes_density_evaluation_and_roundoff=True,is_relative_error_certificate=False,
            raw_native_l1_absolute=0.,raw_direct_fov_absolute=0.),
        density_diagnostics=dict(constant_depth_nodes=0,positive_branch_nodes=0,negative_branch_nodes=0,negative_erfcx_nodes=0,negative_series_nodes=0,
            maximum_log_density=None,minimum_log_density=None,maximum_log_negative_tail_relative_remainder=None,
            minimum_scaled_negative_tail_series_lower_bound=None,maximum_scaled_negative_tail_series_upper_bound=None),
        lse_diagnostics=report,log_raw_mass_sum=None,log_total_area_floor=float(logsumexp(logfloor)),
        log_normalizer=float(logsumexp(logfloor)),log_content_normalizer=normalized['log_content_normalizer'],
        ordinary_raw_unmaterialized_count=0,ordinary_raw_subnormal_count=0,zero_raw_cells=2,all_raw_mass_zero=True,area_prior_used=True,
        raw_mass_below_total_floor=True,argmax_index=0,argmax_exact_ties=2,argmax_near_ties_at_1e_12=2)
    assert A.production_check(row,bank,A.C.POLICY,A.C.GATES)['passed']
    wrong=dict(bank,log_floored_mass=bank['log_floored_mass']+math.log(2.))
    assert not A.production_check(row,wrong,A.C.POLICY,A.C.GATES)['passed']
