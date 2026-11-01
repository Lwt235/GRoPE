"""Synthetic R28 mapping/provenance guards; never opens inherited arrays."""
import ast
import copy
import math
from pathlib import Path

import numpy as np
import pytest

from experiments import round28_rms_applicability_audit as A
from experiments import round28_rms_contract as C


def geometry():
    factor=np.asarray([[.1,.02,0.],[.02,.11,.01],[.01,-.02,.3]],np.float64)
    old=dict(input__mu=np.asarray([.12,-.2,2.]),input__factor=factor,
        input__bounds=A.literal_bounds(),input__H=np.asarray([[60.,.1],[0.,61.]]),
        input__offset=np.asarray([11.7,8.8]),target__covariance=factor@factor.T,
        query_bounds_px=np.asarray([24,0,48,24],np.int64),query_index=np.asarray(1,np.int64))
    h,c=old['input__H'],old['input__offset']
    transform=np.eye(3);transform[:2,:2]=h;transform[:2,2]=c
    prepared={'r23__'+n:v.copy() for n,v in old.items()}
    prepared.update(mapping_matrix=transform,input__mu=transform@old['input__mu'],
        input__factor=transform@factor,input__bounds=old['input__bounds'].copy(),
        input__content_bias=np.zeros(432,np.float64))
    spec=dict(query_index=1,query_bounds=[24,0,48,24],query_size=24)
    return old,prepared,spec


def test_fsum_homogeneous_map_and_covariance_match_independent_matrix_path():
    old,prepared,spec=geometry()
    result=A.geometry_check(old,prepared,spec)
    assert result['passed'] and result['maximum_input_normalized_error']<1e-12
    assert result['depth_rows_exact'] and result['principal_offset_unchanged']
    assert all(result[k] is False for k in A.SCOPE)
    expected,_=A.transformed_inputs(old)
    assert expected['mapping_matrix'][0,2]==old['input__offset'][0]
    assert expected['mapping_matrix'][0,2] != old['input__offset'][0]-.5/8


@pytest.mark.parametrize('field',['mapping','bounds','bias','depth_mean','depth_factor','source_copy'])
def test_exact_input_identities_reject_silent_convention_changes(field):
    old,prepared,spec=geometry()
    if field=='mapping':prepared['mapping_matrix'][0,2]-=.5/8
    elif field=='bounds':prepared['input__bounds'][0,0]-=.5/8
    elif field=='bias':prepared['input__content_bias'][0]=1e-100
    elif field=='depth_mean':prepared['input__mu'][2]+=1e-15
    elif field=='depth_factor':prepared['input__factor'][2,0]+=1e-15
    else:prepared['r23__input__mu'][0]+=1e-15
    with pytest.raises(ValueError):A.geometry_check(old,prepared,spec)


def test_target_covariance_is_independent_of_the_reused_factor():
    old,prepared,spec=geometry()
    old['target__covariance'][0,0]+=.01
    prepared['r23__target__covariance']=old['target__covariance'].copy()
    check=A.geometry_check(old,prepared,spec)
    assert not check['passed'] and check['errors']['factor_covariance']>1e-12


def test_homogeneous_error_is_checked_without_repair():
    old,prepared,spec=geometry()
    prepared['input__factor'][0,0]+=.01
    check=A.geometry_check(old,prepared,spec)
    assert not check['passed'] and check['errors']['input__factor']>1e-12


def test_query_identity_and_no_extra_fields():
    old,prepared,spec=geometry()
    with pytest.raises(ValueError,match='identity'):
        A.geometry_check(old,prepared,dict(spec,query_index=2))
    prepared['forbidden_label']=np.zeros(1)
    with pytest.raises(ValueError,match='exactly13'):A.geometry_check(old,prepared,spec)


def test_old_archive_reader_decodes_only_eight_whitelisted_fields(tmp_path):
    old,_,_=geometry();path=tmp_path/'old.npz'
    np.savez(path,**old,forbidden=np.asarray([object()],dtype=object))
    hashes={n:C.array_sha(v) for n,v in old.items()}
    result=A.read_arrays(path,hashes,fields=C.OLD_FIELDS)
    assert set(result)==set(C.OLD_FIELDS)
    assert all(np.array_equal(result[n],old[n]) for n in C.OLD_FIELDS)
    with pytest.raises(ValueError,match='unexpected'):
        A.read_arrays(path,hashes)


def test_single_area_floor_and_t0_normalization():
    raw=np.zeros(432,np.float64);bounds=A.literal_bounds();bias=np.zeros(432,np.float64)
    p,floor,floored,_=A.probabilities(raw,bounds,bias,1e-30)
    assert np.array_equal(floor,np.full(432,1e-30))
    assert np.array_equal(floor,floored)
    assert np.allclose(p,np.full(432,1/432),rtol=1e-14,atol=0)


def test_no_model_or_probability_reference_import():
    tree=ast.parse(Path(A.__file__).read_text())
    names=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    assert all(not (n or '').startswith('models') for n in names)
    assert all('reference' not in (n or '') for n in names)


def test_freeze_refuses_existing_preparation_before_metadata_call(tmp_path,monkeypatch):
    run,out=tmp_path/'producer',tmp_path/'audit';run.mkdir();(run/'prepared').mkdir()
    monkeypatch.setattr(A,'verify_producer_lock',lambda p:pytest.fail('must stop before checking/decoding prior attempt'))
    with pytest.raises(ValueError,match='precede'):A.freeze(run,out)


def test_cli_guard_follows_all_runtime_functions():
    """Import-only toys must not hide functions defined after main executes."""
    tree=ast.parse(Path(A.__file__).read_text())
    guards=[node for node in tree.body if isinstance(node,ast.If)
        and isinstance(node.test,ast.Compare)
        and isinstance(node.test.left,ast.Name) and node.test.left.id=='__name__']
    assert len(guards)==1 and tree.body[-1] is guards[0]
    functions={node.name:node for node in tree.body if isinstance(node,ast.FunctionDef)}
    assert functions['production_check'].lineno<guards[0].lineno
    assert functions['audit'].lineno<guards[0].lineno


def failed_row(**changes):
    row=dict(passed=False,attempted=False,status='failed',failure_reason='phase deadline',
        work={},wall_seconds=0.,call_and_input_load_seconds=0.,peak_rss_kib_process=1)
    row.update(changes)
    return row


def test_failed_and_unattempted_calls_are_delivery_outcomes_without_fake_probabilities():
    _,prepared,_=geometry()
    check=A.validate_forward_row(failed_row(),{},prepared,C.POLICY)
    assert check['delivery_integrity_passed'] and not check['model_completed'] and not check['attempted']
    partial={name:prepared[name] for name in ('input__mu','input__factor','input__bounds','input__content_bias')}
    partial.update(partial_raw_mass=np.zeros(432),partial_raw_fov_mass=np.zeros(1))
    check=A.validate_forward_row(failed_row(attempted=True,work={'cell_node_evaluations':576}),partial,prepared,C.POLICY)
    assert check['delivery_integrity_passed'] and check['attempted'] and not check['model_completed']


@pytest.mark.parametrize('change',('missing_reason','negative_work','over_budget','fake_unattempted_work','fake_probability','bad_partial_shape','input_tamper'))
def test_failed_row_schema_rejects_hidden_or_fabricated_outputs(change):
    _,prepared,_=geometry();row=failed_row();bank={}
    if change=='missing_reason':row['failure_reason']=' '
    elif change=='negative_work':row['work']={'cell_node_evaluations':-1}
    elif change=='over_budget':row.update(attempted=True,work={'cell_node_evaluations':C.POLICY['max_cell_node_evaluations']+1})
    elif change=='fake_unattempted_work':row['work']={'cell_node_evaluations':1}
    elif change=='fake_probability':bank['probability']=np.ones(432)
    elif change=='bad_partial_shape':
        row['attempted']=True;bank['partial_raw_mass']=np.zeros(1)
    else:
        bank={name:prepared[name].copy() for name in ('input__mu','input__factor','input__bounds','input__content_bias')}
        bank['input__mu'][0]+=1.
    with pytest.raises(ValueError):A.validate_forward_row(row,bank,prepared,C.POLICY)


def test_input_failure_seals_all72_checks_and_blocks_forward_authorization(tmp_path,monkeypatch):
    run,out=tmp_path/'run',tmp_path/'audit';run.mkdir();out.mkdir()
    old,prepared,spec=geometry();source=tmp_path/'old.npz';np.savez(source,**old)
    archive=run/'prepared.npz';np.savez(archive,**prepared)
    for path in (run/'lock.json',run/'prepared_complete.json',out/'lock.json'):
        C.write(path,{'synthetic':True})
    (run/'prepared').mkdir();C.write(run/'prepared/completion.json',{'synthetic':True})
    specs=[dict(spec,id='toy'+str(i),scene='toy',source_archive_path=str(source),
        archive_sha256=C.sha(source),array_sha256={n:C.array_sha(v) for n,v in old.items()}) for i in range(72)]
    rows=[dict(passed=i!=17,failure_reason='synthetic preparation failed',archive_path='prepared.npz',
        archive_sha256=C.sha(archive),array_sha256={n:C.array_sha(v) for n,v in prepared.items()}) for i in range(72)]
    lock=dict(gates=C.GATES,query_specs_sha256='synthetic')
    monkeypatch.setattr(A,'verify_prepared',lambda *args:(lock,{'finished_unix':0.},rows,specs))
    monkeypatch.setattr(A,'joint_locks',lambda *args:None)
    with pytest.raises(RuntimeError,match='no forward authorized'):A.input_audit(run,out)
    checks=C.read(out/'input_audit/checks.json');receipt=C.read(out/'input_verification.json')
    assert len(checks)==72 and sum(r['passed'] for r in checks)==71
    assert not receipt['passed'] and receipt['checked_queries']==72
    assert all(receipt[k] is False for k in A.SCOPE)
    done=C.read(out/'input_audit/completion.json')
    C.verify_inventory(out/'input_audit',done['files'],exclude=('completion.json',))
    assert not done['passed'] and done['verification_sha256']==C.sha(out/'input_verification.json')


def test_keyboard_interrupt_preserves_final_audit_failure_inventory(tmp_path,monkeypatch):
    def stop(*args):raise KeyboardInterrupt('synthetic interruption')
    monkeypatch.setattr(A,'verify_forward',stop)
    with pytest.raises(KeyboardInterrupt):A.audit(tmp_path/'run',tmp_path)
    done=C.read(tmp_path/'completion.json');failure=C.read(tmp_path/'failure.json')
    assert not done['passed'] and failure['error_type']=='KeyboardInterrupt'
    assert all(failure[k] is False for k in A.SCOPE)
    C.verify_inventory(tmp_path,done['files'],exclude=('completion.json',))


def test_completed_structural_zero_checks_saved_outputs_without_integrating():
    bounds=A.literal_bounds();raw=np.zeros(432,np.float64);bias=np.zeros(432,np.float64)
    p,floor,floored,logz=A.probabilities(raw,bounds,bias,C.POLICY['floor_density'])
    bank=dict(input__mu=np.zeros(3),input__factor=np.zeros((3,3)),input__bounds=bounds,
        input__content_bias=bias,raw_mass=raw,probability=p,floored_mass=floored,argmax=np.asarray(p.argmax(),np.int32))
    row=dict(passed=True,attempted=True,status='complete',wall_seconds=0.,call_and_input_load_seconds=0.,
        peak_rss_kib_process=1,work=dict(cell_node_evaluations=0,native_tiles_completed=0,fov_tiles_completed=0),
        raw_fov_mass=0.,structural_branch='zero_depth_rms',structural_zero=True,front_mass=0.,rms_depth=0.,
        normalized_depth_mean=None,normalized_depth_std=None,quadrature_order=24,maximum_tile_side_in_h=2.,
        projection_regularization='rms_depth_xy_gaussian',tail_truncation_used=False,
        quadrature_error_bound_available=False,exact_arithmetic_gl_bound_available=True,
        native_tile_count=0,fov_tile_count=0,h=C.POLICY['h'],floor_density=C.POLICY['floor_density'],
        raw_mass_sum=0.,total_area_floor=math.fsum(map(float,floor)),normalizer=math.fsum(map(float,floored)),
        log_content_normalizer=logz,zero_raw_cells=432,all_raw_mass_zero=True,area_prior_used=True,
        raw_mass_below_total_floor=True,argmax_index=int(p.argmax()),argmax_exact_ties=432,argmax_near_ties_at_1e_12=432,
        analytic_truncation_bound=dict(exact_arithmetic_only=True,excludes_density_evaluation_and_roundoff=True,
            is_relative_error_certificate=False,raw_native_l1_absolute=0.,raw_direct_fov_absolute=0.),
        density_diagnostics=dict(constant_depth_nodes=0,positive_branch_nodes=0,negative_branch_nodes=0,
            negative_erfcx_nodes=0,negative_series_nodes=0,maximum_log_negative_tail_relative_remainder=None,
            minimum_scaled_negative_tail_series_lower_bound=None,maximum_scaled_negative_tail_series_upper_bound=None))
    assert A.validate_forward_row(row,bank,bank,C.POLICY)['model_completed']
    assert A.production_check(row,bank,C.POLICY,C.GATES)['passed']
    wrong=dict(bank,probability=p.copy());wrong['probability'][0]+=.001
    assert not A.production_check(row,wrong,C.POLICY,C.GATES)['passed']
    wrong=dict(bank,floored_mass=floored*2)
    assert not A.production_check(row,wrong,C.POLICY,C.GATES)['passed']
