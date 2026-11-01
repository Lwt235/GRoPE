"""Mock-only R31 chronology/IO tests; no formal fixtures or integration."""
import ast
import json
from pathlib import Path
import time

import numpy as np
import pytest

from experiments import round31_log_conditional_preflight as P

C = P.C


def toy_old():
    return dict(input__mu=np.asarray([.375,-0.,1.625]),
        input__factor=np.asarray([[.2,.1,-0.], [.03,.4,0.], [.01,-.02,.3]], np.float64),
        input__bounds=np.asarray([[-.7,-.2,.1,.3],[.1,-.2,.8,.3]], np.float64),
        input__content_bias=np.asarray([.2,-.3], np.float64))


def toy_spec(path=None, name='toy'):
    row = dict(id=name, fixture=dict(id=name,policy_overrides={}))
    if path is not None:
        row.update(source_archive_path=str(path), source_archive_sha256=C.sha(path),
            array_sha256={n:C.array_sha(v) for n,v in toy_old().items()})
    return row


def save_old(tmp_path, name='old.npz'):
    path=tmp_path/name
    np.savez_compressed(path, **toy_old(), probability=np.asarray([.3,.7]), forbidden_oracle=np.asarray([999.]))
    return path


def save_prepared(tmp_path, name='toy'):
    out=tmp_path/'run';(out/'prepared/arrays').mkdir(parents=True);(out/'forward/arrays').mkdir(parents=True)
    for journal in ('call_starts.jsonl','reference_results.jsonl'):(out/'forward'/journal).touch()
    bank=P.copy_inputs(toy_old());path=out/'prepared/arrays'/(name+'.npz');np.savez_compressed(path,**bank)
    row=dict(id=name,policy_overrides={},archive_path=str(path.relative_to(out)),archive_sha256=C.sha(path),
        array_sha256={n:C.array_sha(v) for n,v in bank.items()},passed=True,status='prepared',synthetic_inputs_read=True)
    return out,row,bank


class ToyBudget:
    def __init__(self, maximum):self.maximum=maximum;self.used=0
    @property
    def remaining_nodes(self):return self.maximum-self.used
    def charge(self,count):
        if count>self.remaining_nodes:raise ValueError('toy global budget exhausted')
        self.used+=count


def success_reference(**kwargs):
    kwargs['budget'].charge(7)
    n=len(kwargs['bounds']);logs=np.full(n,-900.,np.float64);lp=np.full(n,-np.log(n),np.float64)
    return dict(status='complete',passed=True,order=kwargs['order'],wall_seconds=.00123,
        log_raw_mass=logs,log_raw_fov_mass=np.asarray([-900.+np.log(n)]),
        log_floored_mass=np.full(n,np.log(1e-30)),log_probability=lp,probability=np.exp(lp),
        work=dict(cell_node_evaluations=7,pilot_depth_evaluations=1,quadrature_depth_evaluations=2))


def invoke(out,row,**kwargs):
    options=dict(order=32,policy=C.POLICY,reference_policy=C.REFERENCE_POLICY,
        budget=ToyBudget(100),deadline=time.perf_counter()+10)
    options.update(kwargs)
    return P.run_case(out,row,**options)


def test_copy_preserves_four_fields_signed_zeros_and_no_aliasing():
    old=toy_old();new=P.copy_inputs(old)
    assert set(new)==P.PREPARED_FIELDS and len(new)==4
    for name,value in old.items():
        assert C.array_sha(new[name])==C.array_sha(value)
        assert new[name].tobytes()==value.tobytes()
        assert not np.shares_memory(new[name],value)
    assert np.signbit(new['input__mu'][1]) and np.signbit(new['input__factor'][0,2])


@pytest.mark.parametrize('change',('extra','missing','dtype','shape','nan','empty_bounds'))
def test_copy_rejects_changed_input_contract(change):
    old=toy_old()
    if change=='extra':old['probability']=np.asarray([.5,.5])
    if change=='missing':del old['input__factor']
    if change=='dtype':old['input__mu']=old['input__mu'].astype(np.float32)
    if change=='shape':old['input__factor']=old['input__factor'][:2]
    if change=='nan':old['input__mu'][0]=np.nan
    if change=='empty_bounds':old['input__bounds']=np.empty((0,4),np.float64)
    with pytest.raises(ValueError):P.copy_inputs(old)


def test_loader_decodes_exactly_four_allowlisted_fields(tmp_path,monkeypatch):
    path=save_old(tmp_path);spec=toy_spec(path);seen=[];real_load=np.load;state={}
    class Guard:
        def __enter__(self):self.archive=real_load(path,allow_pickle=False);return self
        def __exit__(self,*args):self.archive.close()
        def __getitem__(self,name):
            seen.append(name);assert name in C.OLD_FIELDS;return self.archive[name]
    monkeypatch.setattr(P.np,'load',lambda *a,**k:Guard())
    assert set(P.load_old(spec,state))==set(C.OLD_FIELDS)
    assert tuple(seen)==C.OLD_FIELDS and state['synthetic_inputs_read'] is True


def test_loader_hash_failure_precedes_decode(tmp_path,monkeypatch):
    path=save_old(tmp_path);spec=toy_spec(path);spec['source_archive_sha256']='wrong';state={}
    monkeypatch.setattr(P.np,'load',lambda *a,**k:pytest.fail('must not decode unauthenticated bytes'))
    with pytest.raises(ValueError,match='bytes changed'):P.load_old(spec,state)
    assert not state


def test_loader_canonical_failure_still_records_read(tmp_path):
    path=save_old(tmp_path);spec=toy_spec(path);spec['array_sha256']['input__factor']='wrong';state={}
    with pytest.raises(ValueError,match='canonical'):P.load_old(spec,state)
    assert state['synthetic_inputs_read']


def test_partial_decode_exception_records_read(tmp_path,monkeypatch):
    path=save_old(tmp_path);spec=toy_spec(path);state={}
    class Guard:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def __getitem__(self,name):
            if name==C.OLD_FIELDS[0]:return toy_old()[name]
            raise OSError('synthetic later field failure')
    monkeypatch.setattr(P.np,'load',lambda *a,**k:Guard())
    with pytest.raises(OSError,match='later field'):P.load_old(spec,state)
    assert state['synthetic_inputs_read']


def test_freeze_metadata_only_retains_oracle_bindings_and_existing_attempt(tmp_path,monkeypatch):
    (tmp_path/'source.py').write_text('# synthetic source\n')
    specs=[toy_spec(name='toy'+str(i)) for i in range(23)]
    oracles=[dict(id='toy',order=32,source_archive_path='metadata_only')]
    monkeypatch.setattr(C,'ROOT',tmp_path);monkeypatch.setattr(C,'source_names',lambda:['source.py'])
    monkeypatch.setattr(C,'inherited_evidence',lambda:({'toy_binding':'toy_digest'},specs,oracles))
    monkeypatch.setattr(C,'check_runtime',lambda:{'synthetic_runtime':True})
    monkeypatch.setattr(P.np,'load',lambda *a,**k:pytest.fail('freeze must not decode'))
    monkeypatch.setattr(P.REF,'integrate',lambda *a,**k:pytest.fail('freeze must not integrate'))
    out=tmp_path/'producer';P.freeze(out,tmp_path/'audit')
    lock=C.read(out/'lock.json')
    assert lock['case_ids']==[s['id'] for s in specs]
    assert lock['ordered_call_ids']==P.ordered_calls(specs)
    assert lock['oracle_specs_sha256']==C.sha(out/'oracle_specs.json')
    assert not lock['synthetic_inputs_read'] and not lock['independent_synthetic_reference']
    assert all(lock[k] is False for k in P.SCOPE)
    with pytest.raises(ValueError,match='existing'):P.freeze(out,tmp_path/'audit')


def test_prepare_preserves_failed_copy_and_complete_population(tmp_path,monkeypatch):
    path=save_old(tmp_path);out=tmp_path/'run';out.mkdir();C.write(out/'lock.json',{'toy':True})
    specs=[toy_spec(path,'toy'+str(i)) for i in range(3)];specs[1]['array_sha256']['input__mu']='wrong'
    C.write(out/'query_specs.json',specs)
    lock=dict(case_ids=[s['id'] for s in specs],query_specs_sha256=C.sha(out/'query_specs.json'))
    monkeypatch.setattr(P,'verify',lambda unused:lock);monkeypatch.setattr(P,'require_audit_freeze',lambda *a:'audit')
    monkeypatch.setattr(P.REF,'integrate',lambda *a,**k:pytest.fail('prepare must not integrate'))
    P.prepare(out);rows=C.read(out/'prepared/records.json')
    assert [r['passed'] for r in rows]==[True,False,True] and all(r['synthetic_inputs_read'] for r in rows)
    seal=C.read(out/'prepared_complete.json')
    assert seal['case_count']==3 and not seal['all_inputs_prepared']
    assert not seal['predictions_executed'] and not seal['reference_calls_executed']
    assert not seal['independent_synthetic_reference'] and seal['synthetic_inputs_read']
    done=C.read(out/'prepared/completion.json');C.verify_inventory(out/'prepared',done['files'],exclude=('completion.json',))
    with pytest.raises(ValueError,match='prior'):P.prepare(out)


def test_run_requires_input_audit_before_directory_or_call(tmp_path,monkeypatch):
    monkeypatch.setattr(P,'verify',lambda out:{})
    monkeypatch.setattr(P,'require_audit_freeze',lambda *a:'audit')
    def deny(*a):raise ValueError('input audit missing')
    monkeypatch.setattr(P,'require_input_audit',deny)
    monkeypatch.setattr(P.REF,'integrate',lambda *a,**k:pytest.fail('unverified reference'))
    with pytest.raises(ValueError,match='input audit'):P.run(tmp_path)
    assert not (tmp_path/'forward').exists()


def test_run_case_retains_canonical_subfloat_logs_and_exact_inputs(tmp_path,monkeypatch):
    out,row,bank=save_prepared(tmp_path);seen=[]
    def reference(**kwargs):seen.append(kwargs);return success_reference(**kwargs)
    monkeypatch.setattr(P.REF,'integrate',reference)
    result=invoke(out,row)
    assert result['attempted'] and result['passed'] and result['synthetic_inputs_read'] and len(seen)==1
    assert result['independent_synthetic_reference'] and all(result[k] is False for k in P.SCOPE)
    outputs={'log_raw_mass','log_raw_fov_mass','log_floored_mass','log_probability','probability'}
    assert set(result['array_sha256'])==P.PREPARED_FIELDS|outputs
    for name in P.INPUT_FIELDS:assert C.array_sha(seen[0][name])==C.array_sha(bank['input__'+name])
    assert result['budget_used_before']==0 and result['budget_used_after']==7 and result['reference_work_consistent']
    journal=[json.loads(s) for s in (out/'forward/reference_results.jsonl').read_text().splitlines()]
    assert journal[0]['available_array_sha256']==result['array_sha256']
    assert journal[0]['call_id']=='toy__order32'
    assert result['reference_reported_wall_seconds']==journal[0]['reference_reported_wall_seconds']==.00123
    assert all(result.get(key)==value for key,value in journal[0].items() if key!='available_array_sha256')
    with np.load(out/result['archive_path'],allow_pickle=False) as z:assert np.all(z['log_raw_mass']==-900.)


def test_returned_gate_failure_retains_all_outputs_and_variable_arrays(tmp_path,monkeypatch):
    out,row,_=save_prepared(tmp_path)
    def failed(**kwargs):
        result=success_reference(**kwargs)
        result.update(status='failed',passed=False,failure_reason='synthetic convergence failure',
            initial_depth_edges=np.asarray([-.5,.6]),pilot_zeta=np.asarray([.05]),
            pilot_log_integrand=np.asarray([[-900.,-901.,-899.]]),outer_log_scale=np.asarray([-900.,-901.,-899.]),
            outer_scaled_integral=np.ones(3),outer_underflow_counts=np.asarray([0,1,0],np.int64))
        return result
    monkeypatch.setattr(P.REF,'integrate',failed)
    saved=invoke(out,row)
    assert saved['attempted'] and not saved['passed'] and saved['reference_work_consistent']
    assert {'probability','pilot_log_integrand','outer_underflow_counts'} <= set(saved['array_sha256'])


def test_reference_exception_preserves_work_partial_logs_masks_and_metadata(tmp_path,monkeypatch):
    out,row,_=save_prepared(tmp_path)
    partial=dict(partial_log_raw_mass=np.asarray([-900.,-np.inf]),partial_log_raw_fov_mass=np.asarray([-np.inf]),
        partial_raw_computed=np.asarray([True,False,False]),last_conditional_log_mass=np.asarray([-910.,-920.,-909.]))
    def reference(**kwargs):
        kwargs['budget'].charge(9)
        error=ValueError('synthetic reference interruption')
        error.reference_work=dict(cell_node_evaluations=9,pilot_depth_evaluations=1,quadrature_depth_evaluations=0)
        error.reference_metadata=dict(method='mock conditional',diagnostic='before normalization')
        error.available_arrays=partial
        raise error
    monkeypatch.setattr(P.REF,'integrate',reference)
    saved=invoke(out,row)
    assert saved['attempted'] and not saved['passed'] and saved['charged_cell_node_evaluations']==9
    assert saved['method']=='mock conditional' and saved['reference_work_consistent']
    assert 'probability' not in saved['array_sha256']
    with np.load(out/saved['archive_path'],allow_pickle=False) as z:
        for key,value in partial.items():np.testing.assert_array_equal(z[key],value)


def test_deadline_does_not_decode_or_invent_inputs(tmp_path,monkeypatch):
    out,row,_=save_prepared(tmp_path)
    monkeypatch.setattr(P.np,'load',lambda *a,**k:pytest.fail('expired phase must not decode'))
    monkeypatch.setattr(P.REF,'integrate',lambda *a,**k:pytest.fail('expired phase must not call'))
    result=invoke(out,row,deadline=time.perf_counter()-1)
    assert not result['attempted'] and not result['passed'] and not result['synthetic_inputs_read']
    assert not result['independent_synthetic_reference'] and result['array_sha256']=={} and result['work']=={}
    assert result['charged_cell_node_evaluations']==0 and (out/'forward/call_starts.jsonl').read_text()==''


@pytest.mark.parametrize('tamper',('archive','canonical'))
def test_prepared_tamper_never_calls_reference(tmp_path,monkeypatch,tamper):
    out,row,_=save_prepared(tmp_path)
    if tamper=='archive':row['archive_sha256']='wrong'
    else:row['array_sha256']['input__mu']='wrong'
    monkeypatch.setattr(P.REF,'integrate',lambda *a,**k:pytest.fail('tampered input must not integrate'))
    result=invoke(out,row)
    assert not result['attempted'] and not result['passed'] and not result['array_sha256']
    assert result['synthetic_inputs_read'] is (tamper=='canonical')


def test_nonreference_exception_keeps_observed_charges_without_fabricating_work(tmp_path,monkeypatch):
    out,row,_=save_prepared(tmp_path)
    def reference(**kwargs):kwargs['budget'].charge(3);raise ValueError('synthetic unexpected failure')
    monkeypatch.setattr(P.REF,'integrate',reference)
    result=invoke(out,row)
    assert result['attempted'] and not result['passed'] and result['synthetic_inputs_read']
    assert set(result['array_sha256'])==P.PREPARED_FIELDS
    assert result['charged_cell_node_evaluations']==3 and result['work']=={} and not result['reference_work_consistent']


def test_zero_work_case_can_be_attempted_after_node_budget_exhausted(tmp_path,monkeypatch):
    out,row,_=save_prepared(tmp_path);budget=ToyBudget(0)
    def zero(**kwargs):
        assert kwargs['budget'].remaining_nodes==0
        return dict(status='complete',passed=True,work=dict(cell_node_evaluations=0),
            log_raw_mass=np.full(2,-np.inf),log_raw_fov_mass=np.asarray([-np.inf]))
    monkeypatch.setattr(P.REF,'integrate',zero)
    result=invoke(out,row,budget=budget)
    assert result['attempted'] and result['passed'] and result['charged_cell_node_evaluations']==0


def mock_run_state(tmp_path,monkeypatch):
    out=tmp_path/'run';(out/'prepared/arrays').mkdir(parents=True)
    C.write(out/'lock.json',dict(toy=True));C.write(out/'prepared_complete.json',dict(toy=True))
    rows=[]
    for name in ('alpha','beta'):
        bank=toy_old();path=out/'prepared/arrays'/(name+'.npz');np.savez_compressed(path,**bank)
        rows.append(dict(id=name,policy_overrides={'h':.37} if name=='beta' else {},
            archive_path=str(path.relative_to(out)),archive_sha256=C.sha(path),
            array_sha256={k:C.array_sha(v) for k,v in bank.items()}))
    rp=dict(C.REFERENCE_POLICY,max_total_cell_node_evaluations=100,max_phase_wall_seconds=10.)
    lock=dict(case_ids=['alpha','beta'],case_count=2,ordered_call_ids=P.ordered_calls(rows),
        query_specs_sha256='toy_specs',oracle_specs_sha256='toy_oracles',policy=C.POLICY,reference_policy=rp)
    monkeypatch.setattr(P,'verify',lambda unused:lock)
    monkeypatch.setattr(P,'require_audit_freeze',lambda *a:'audit')
    monkeypatch.setattr(P,'require_input_audit',lambda *a:('input_audit',rows))
    monkeypatch.setattr(P.REF,'NodeBudget',ToyBudget)
    return out,rows,lock


def test_run_seals_fixed_call_order_shared_budget_overrides_and_full_inventories(tmp_path,monkeypatch):
    out,rows,lock=mock_run_state(tmp_path,monkeypatch);calls=[]
    def reference(**kwargs):calls.append((kwargs['order'],kwargs['policy']['h']));return success_reference(**kwargs)
    monkeypatch.setattr(P.REF,'integrate',reference)
    P.run(out)
    records=C.read(out/'forward/records.json');seal=C.read(out/'forward_complete.json')
    assert [r['call_id'] for r in records]==lock['ordered_call_ids']
    assert calls==[(32,.25),(48,.25),(32,.37),(48,.37)]
    assert [r['budget_used_before'] for r in records]==[0,7,14,21]
    assert seal['total_cell_node_evaluations']==28 and seal['reference_calls_completed']
    assert seal['attempted_calls']==4 and seal['case_count']==2 and seal['call_count']==4
    assert seal['synthetic_inputs_read'] and seal['independent_synthetic_reference']
    assert all(seal[k] is False for k in P.SCOPE)
    for directory in (out/'forward',out):
        done=C.read(directory/'completion.json');C.verify_inventory(directory,done['files'],exclude=('completion.json',))
    with pytest.raises(ValueError,match='prior reference'):P.run(out)


def test_failed_reference_does_not_drop_later_population_or_alter_budget(tmp_path,monkeypatch):
    out,rows,lock=mock_run_state(tmp_path,monkeypatch);count=[]
    def reference(**kwargs):
        count.append(1)
        if len(count)==2:
            error=ValueError('toy failure');error.reference_work={'cell_node_evaluations':0};raise error
        return success_reference(**kwargs)
    monkeypatch.setattr(P.REF,'integrate',reference)
    P.run(out)
    records=C.read(out/'forward/records.json');seal=C.read(out/'forward_complete.json')
    assert len(records)==4 and [r['passed'] for r in records]==[True,False,True,True]
    assert seal['attempted_calls']==4 and seal['total_cell_node_evaluations']==21
    assert seal['all_case_attempts_retained'] and not seal['reference_calls_completed']


def test_prepare_engineering_failure_keeps_read_scope_inventory_and_exit(tmp_path,monkeypatch):
    path=save_old(tmp_path);out=tmp_path/'run';out.mkdir();C.write(out/'lock.json',{'toy':True})
    specs=[toy_spec(path)];C.write(out/'query_specs.json',specs)
    monkeypatch.setattr(P,'verify',lambda unused:dict(case_ids=['toy'],query_specs_sha256='x'))
    monkeypatch.setattr(P,'require_audit_freeze',lambda *a:'audit')
    def fail_save(path,**arrays):raise OSError('synthetic disk failure after decode')
    monkeypatch.setattr(P.np,'savez_compressed',fail_save)
    with pytest.raises(OSError,match='after decode'):P.prepare(out)
    done=C.read(out/'prepared/completion.json')
    assert not done['complete_case_inventory'] and done['synthetic_inputs_read']
    assert not done['independent_synthetic_reference']
    C.verify_inventory(out/'prepared',done['files'],exclude=('completion.json',))


def test_run_output_failure_keeps_attempt_scope_journal_inventory_and_exit(tmp_path,monkeypatch):
    out,rows,lock=mock_run_state(tmp_path,monkeypatch)
    monkeypatch.setattr(P.REF,'integrate',success_reference)
    def fail_save(path,**arrays):raise OSError('synthetic output failure after call')
    monkeypatch.setattr(P.np,'savez_compressed',fail_save)
    with pytest.raises(OSError,match='after call'):P.run(out)
    done=C.read(out/'completion.json')
    assert not done['complete_case_inventory'] and done['synthetic_inputs_read'] and done['independent_synthetic_reference']
    assert done['total_cell_node_evaluations']==7
    assert len((out/'forward/reference_results.jsonl').read_text().splitlines())==1
    C.verify_inventory(out,done['files'],exclude=('completion.json',))


def test_runner_imports_no_production_model_and_cli_guard_is_last():
    tree=ast.parse(Path(P.__file__).read_text())
    imports=[node for node in ast.walk(tree) if isinstance(node,(ast.Import,ast.ImportFrom))]
    assert not any(isinstance(node,ast.ImportFrom) and (node.module or '').startswith('models') for node in imports)
    assert isinstance(tree.body[-1],ast.If) and '__main__' in ast.unparse(tree.body[-1].test)
