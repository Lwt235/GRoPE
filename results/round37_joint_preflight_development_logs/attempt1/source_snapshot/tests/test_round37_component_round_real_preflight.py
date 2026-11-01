"""Mock-only R37 chronology/IO tests; no formal fixtures or integration."""
import ast
import json
from pathlib import Path
import time

import numpy as np
import pytest

from experiments import round37_component_round_real_preflight as P

C = P.C


@pytest.fixture(autouse=True)
def toy_key_population(monkeypatch):
    """Use distinct two-cell toy inputs, never the real 432-cell population."""
    monkeypatch.setattr(C,'KEY_COUNT',2)


def toy_old():
    return dict(input__mu=np.asarray([.375,-0.,1.625]),
        input__factor=np.asarray([[.2,.1,-0.], [.03,.4,0.], [.01,-.02,.3]], np.float64),
        input__bounds=np.asarray([[-.7,-.2,.1,.3],[.1,-.2,.8,.3]], np.float64),
        input__content_bias=np.asarray([.2,-.3], np.float64))


def toy_spec(path=None, name='toy'):
    row = dict(id=name,scene='toy_scene',pair=[3,7],pair_name='toy_pair',query_size=8,query_index=0,query_bounds=[0,0,8,8],policy_overrides={})
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
    row=dict(**toy_spec(name=name),archive_path=str(path.relative_to(out)),archive_sha256=C.sha(path),
        array_sha256={n:C.array_sha(v) for n,v in bank.items()},passed=True,status='prepared',real_inputs_read=True)
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


@pytest.mark.parametrize('change',('extra','missing','dtype','shape','nan','empty_bounds','wrong_key_count'))
def test_copy_rejects_changed_input_contract(change):
    old=toy_old()
    if change=='extra':old['probability']=np.asarray([.5,.5])
    if change=='missing':del old['input__factor']
    if change=='dtype':old['input__mu']=old['input__mu'].astype(np.float32)
    if change=='shape':old['input__factor']=old['input__factor'][:2]
    if change=='nan':old['input__mu'][0]=np.nan
    if change=='empty_bounds':old['input__bounds']=np.empty((0,4),np.float64)
    if change=='wrong_key_count':old['input__bounds']=old['input__bounds'][:1]
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
    assert tuple(seen)==C.OLD_FIELDS and state['real_inputs_read'] is True


def test_loader_hash_failure_precedes_decode(tmp_path,monkeypatch):
    path=save_old(tmp_path);spec=toy_spec(path);spec['source_archive_sha256']='wrong';state={}
    monkeypatch.setattr(P.np,'load',lambda *a,**k:pytest.fail('must not decode unauthenticated bytes'))
    with pytest.raises(ValueError,match='bytes changed'):P.load_old(spec,state)
    assert not state


def test_loader_canonical_failure_still_records_read(tmp_path):
    path=save_old(tmp_path);spec=toy_spec(path);spec['array_sha256']['input__factor']='wrong';state={}
    with pytest.raises(ValueError,match='canonical'):P.load_old(spec,state)
    assert state['real_inputs_read']


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
    assert state['real_inputs_read']


def toy_oracles(specs,tmp_path):
    rows=[]
    for kind,count in (('r33_log_conditional',19),('r35_log_real',21)):
        source=[dict(id=spec['id'],order=order,reference_kind=kind,
            oracle_id=kind+'__order'+str(order),
            source_archive_path=str(tmp_path/(kind+'_must_not_decode.npz')),source_archive_sha256='toy_digest',
            array_sha256={n:'toy_'+n for n in ('log_raw_mass','log_raw_fov_mass','probability')},
            source_fields={n:n for n in ('log_raw_mass','log_raw_fov_mass','probability')})
            for spec in specs for order in (32,48)]
        rows.extend(source[:count])
    return rows


def freeze_toy_metadata(tmp_path,monkeypatch):
    (tmp_path/'source.py').write_text('# synthetic source\n')
    specs=[toy_spec(name='toy'+str(i)) for i in range(C.CASE_COUNT)]
    oracles=toy_oracles(specs,tmp_path)
    assert all('fixture' not in spec for spec in specs)
    monkeypatch.setattr(C,'ROOT',tmp_path);monkeypatch.setattr(C,'source_names',lambda:['source.py'])
    monkeypatch.setattr(C,'inherited_evidence',lambda:({'toy_binding':'toy_digest'},specs,oracles))
    monkeypatch.setattr(C,'check_runtime',lambda:{'synthetic_runtime':True})
    monkeypatch.setattr(P.np,'load',lambda *a,**k:pytest.fail('freeze must not decode'))
    monkeypatch.setattr(P.REF,'integrate',lambda *a,**k:pytest.fail('freeze must not integrate'))
    out=tmp_path/'producer';P.freeze(out,tmp_path/'audit')
    return out,specs,oracles


def test_freeze_metadata_only_retains_40_source_keyed_oracles_and_existing_attempt(tmp_path,monkeypatch):
    out,specs,oracles=freeze_toy_metadata(tmp_path,monkeypatch)
    lock=C.read(out/'lock.json')
    assert lock['case_ids']==[s['id'] for s in specs]
    assert lock['ordered_call_ids']==P.ordered_calls(specs)
    assert len(oracles)==40 and C.read(out/'oracle_specs.json')==oracles
    assert lock['oracle_specs_sha256']==C.sha(out/'oracle_specs.json')
    assert lock['fail_fast']==C.FAIL_FAST
    assert not lock['real_inputs_read'] and not lock['independent_real_reference']
    assert all(lock[k] is False for k in P.SCOPE)
    with pytest.raises(ValueError,match='existing'):P.freeze(out,tmp_path/'audit')


@pytest.mark.parametrize('rebind_lock',(False,True))
def test_verify_rejects_changed_oracle_metadata_without_decode(tmp_path,monkeypatch,rebind_lock):
    out,_,_=freeze_toy_metadata(tmp_path,monkeypatch)
    changed=C.read(out/'oracle_specs.json');changed[0]['source_archive_sha256']='changed'
    C.write(out/'oracle_specs.json',changed)
    if rebind_lock:
        lock=C.read(out/'lock.json');lock['oracle_specs_sha256']=C.sha(out/'oracle_specs.json')
        C.write(out/'lock.json',lock);(out/'lock.sha256').write_text(C.sha(out/'lock.json')+'\n')
    with pytest.raises(ValueError,match='provenance'):P.verify(out)


def test_independent_freeze_must_bind_the_same_oracle_metadata(tmp_path,monkeypatch):
    out,_,_=freeze_toy_metadata(tmp_path,monkeypatch);lock=P.verify(out)
    audit=Path(lock['audit_root']);(audit/'source_snapshot').mkdir(parents=True)
    (audit/'source_snapshot/source.py').write_bytes((tmp_path/'source.py').read_bytes())
    design=dict(run=str(out.resolve()),producer_root=str(out.resolve()),
        producer_lock_sha256=C.sha(out/'lock.json'),query_specs_sha256=lock['query_specs_sha256'],
        oracle_specs_sha256=lock['oracle_specs_sha256'],source_hashes=lock['source_hashes'],
        runtime=lock['runtime'],created_unix=time.time(),real_inputs_read=False,
        independent_real_reference=False,**P.SCOPE)
    C.write(audit/'lock.json',design);(audit/'lock.sha256').write_text(C.sha(audit/'lock.json')+'\n')
    assert P.require_audit_freeze(out,lock)==C.sha(audit/'lock.json')
    design['oracle_specs_sha256']='different_oracle_population'
    C.write(audit/'lock.json',design);(audit/'lock.sha256').write_text(C.sha(audit/'lock.json')+'\n')
    with pytest.raises(ValueError,match='independent'):P.require_audit_freeze(out,lock)


def test_prepare_preserves_failed_copy_and_complete_population(tmp_path,monkeypatch):
    path=save_old(tmp_path);out=tmp_path/'run';out.mkdir();C.write(out/'lock.json',{'toy':True})
    specs=[toy_spec(path,'toy'+str(i)) for i in range(3)];specs[1]['array_sha256']['input__mu']='wrong'
    C.write(out/'query_specs.json',specs)
    lock=dict(case_ids=[s['id'] for s in specs],query_specs_sha256=C.sha(out/'query_specs.json'))
    monkeypatch.setattr(P,'verify',lambda unused:lock);monkeypatch.setattr(P,'require_audit_freeze',lambda *a:'audit')
    monkeypatch.setattr(P.REF,'integrate',lambda *a,**k:pytest.fail('prepare must not integrate'))
    P.prepare(out);rows=C.read(out/'prepared/records.json')
    assert [r['passed'] for r in rows]==[True,False,True] and all(r['real_inputs_read'] for r in rows)
    seal=C.read(out/'prepared_complete.json')
    assert seal['case_count']==3 and not seal['all_inputs_prepared']
    assert not seal['predictions_executed'] and not seal['reference_calls_executed']
    assert not seal['independent_real_reference'] and seal['real_inputs_read']
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
    assert result['attempted'] and result['passed'] and result['real_inputs_read'] and len(seen)==1
    assert result['independent_real_reference'] and all(result[k] is False for k in P.SCOPE)
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
    assert not result['attempted'] and not result['passed'] and not result['real_inputs_read']
    assert not result['independent_real_reference'] and result['array_sha256']=={} and result['work']=={}
    assert result['charged_cell_node_evaluations']==0 and (out/'forward/call_starts.jsonl').read_text()==''


@pytest.mark.parametrize('tamper',('archive','canonical'))
def test_prepared_tamper_never_calls_reference(tmp_path,monkeypatch,tamper):
    out,row,_=save_prepared(tmp_path)
    if tamper=='archive':row['archive_sha256']='wrong'
    else:row['array_sha256']['input__mu']='wrong'
    monkeypatch.setattr(P.REF,'integrate',lambda *a,**k:pytest.fail('tampered input must not integrate'))
    result=invoke(out,row)
    assert not result['attempted'] and not result['passed'] and not result['array_sha256']
    assert result['real_inputs_read'] is (tamper=='canonical')


def test_nonreference_exception_keeps_observed_charges_without_fabricating_work(tmp_path,monkeypatch):
    out,row,_=save_prepared(tmp_path)
    def reference(**kwargs):kwargs['budget'].charge(3);raise ValueError('synthetic unexpected failure')
    monkeypatch.setattr(P.REF,'integrate',reference)
    result=invoke(out,row)
    assert result['attempted'] and not result['passed'] and result['real_inputs_read']
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


def mock_run_state(tmp_path,monkeypatch,case_count=2):
    out=tmp_path/'run';(out/'prepared/arrays').mkdir(parents=True)
    C.write(out/'lock.json',dict(toy=True));C.write(out/'prepared_complete.json',dict(toy=True))
    C.write(out/'oracle_specs.json',[dict(id='alpha',order=32,oracle_id='toy_oracle')])
    rows=[]
    names=('alpha','beta') if case_count==2 else tuple('toy'+str(i) for i in range(case_count))
    for name in names:
        bank=toy_old();path=out/'prepared/arrays'/(name+'.npz');np.savez_compressed(path,**bank)
        rows.append(dict(**toy_spec(name=name),
            archive_path=str(path.relative_to(out)),archive_sha256=C.sha(path),
            array_sha256={k:C.array_sha(v) for k,v in bank.items()}))
    rp=dict(C.REFERENCE_POLICY,max_total_cell_node_evaluations=100,max_phase_wall_seconds=10.)
    lock=dict(case_ids=list(names),case_count=len(names),ordered_call_ids=P.ordered_calls(rows),
        query_specs_sha256='toy_specs',oracle_specs_sha256=C.sha(out/'oracle_specs.json'),policy=C.POLICY,reference_policy=rp)
    monkeypatch.setattr(P,'verify',lambda unused:lock)
    monkeypatch.setattr(P,'require_audit_freeze',lambda *a:'audit')
    monkeypatch.setattr(P,'require_input_audit',lambda *a:('input_audit',rows))
    monkeypatch.setattr(P.REF,'NodeBudget',ToyBudget)
    return out,rows,lock


def test_run_seals_fixed_call_order_shared_budget_no_overrides_and_full_inventories(tmp_path,monkeypatch):
    out,rows,lock=mock_run_state(tmp_path,monkeypatch);calls=[]
    def reference(**kwargs):calls.append((kwargs['order'],kwargs['policy']['h']));return success_reference(**kwargs)
    monkeypatch.setattr(P.REF,'integrate',reference)
    P.run(out)
    records=C.read(out/'forward/records.json');seal=C.read(out/'forward_complete.json')
    assert [r['call_id'] for r in records]==lock['ordered_call_ids']
    assert calls==[(32,.25),(48,.25),(32,.25),(48,.25)]
    assert [r['budget_used_before'] for r in records]==[0,7,14,21]
    assert seal['total_cell_node_evaluations']==28 and seal['reference_calls_completed']
    assert seal['attempted_calls']==4 and seal['case_count']==2 and seal['call_count']==4
    assert seal['oracle_specs_sha256']==C.sha(out/'oracle_specs.json')
    assert seal['real_inputs_read'] and seal['independent_real_reference']
    assert all(seal[k] is False for k in P.SCOPE)
    for directory in (out/'forward',out):
        done=C.read(directory/'completion.json');C.verify_inventory(directory,done['files'],exclude=('completion.json',))
    assert 'oracle_specs.json' in C.read(out/'completion.json')['files']
    with pytest.raises(ValueError,match='prior reference'):P.run(out)


def test_failed_reference_stops_later_calls_but_keeps_full_population(tmp_path,monkeypatch):
    out,rows,lock=mock_run_state(tmp_path,monkeypatch);count=[]
    def reference(**kwargs):
        count.append(1)
        if len(count)==2:
            error=ValueError('toy failure');error.reference_work={'cell_node_evaluations':0};raise error
        return success_reference(**kwargs)
    monkeypatch.setattr(P.REF,'integrate',reference)
    P.run(out)
    records=C.read(out/'forward/records.json');seal=C.read(out/'forward_complete.json')
    assert len(records)==4 and [r['passed'] for r in records]==[True,False,False,False]
    assert seal['attempted_calls']==2 and seal['total_cell_node_evaluations']==7
    assert seal['fail_fast_skipped_calls']==2 and seal['executed_failed_calls']==1
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
    assert not done['complete_case_inventory'] and done['real_inputs_read']
    assert not done['independent_real_reference']
    C.verify_inventory(out/'prepared',done['files'],exclude=('completion.json',))


def test_run_output_failure_keeps_attempt_scope_journal_inventory_and_exit(tmp_path,monkeypatch):
    out,rows,lock=mock_run_state(tmp_path,monkeypatch)
    monkeypatch.setattr(P.REF,'integrate',success_reference)
    def fail_save(path,**arrays):raise OSError('synthetic output failure after call')
    monkeypatch.setattr(P.np,'savez_compressed',fail_save)
    with pytest.raises(OSError,match='after call'):P.run(out)
    done=C.read(out/'completion.json')
    assert not done['complete_case_inventory'] and done['real_inputs_read'] and done['independent_real_reference']
    assert done['total_cell_node_evaluations']==7
    assert len((out/'forward/reference_results.jsonl').read_text().splitlines())==1
    C.verify_inventory(out,done['files'],exclude=('completion.json',))


def test_runner_imports_no_production_model_and_cli_guard_is_last():
    tree=ast.parse(Path(P.__file__).read_text())
    imports=[node for node in ast.walk(tree) if isinstance(node,(ast.Import,ast.ImportFrom))]
    assert not any(isinstance(node,ast.ImportFrom) and (node.module or '').startswith('models') for node in imports)
    assert isinstance(tree.body[-1],ast.If) and '__main__' in ast.unparse(tree.body[-1].test)


@pytest.mark.parametrize('failure_index',(0,1,3))
def test_fail_fast_first32_first48_or_last_retains_order_and_trigger(tmp_path,monkeypatch,failure_index):
    out,rows,lock=mock_run_state(tmp_path,monkeypatch);calls=[]
    def reference(**kwargs):
        index=len(calls);calls.append(kwargs['order']);result=success_reference(**kwargs)
        if index==failure_index:result.update(status='failed',passed=False,failure_reason='mock internal tail failure')
        return result
    monkeypatch.setattr(P.REF,'integrate',reference)
    P.run(out)
    records=C.read(out/'forward/records.json');seal=C.read(out/'forward_complete.json')
    failed=records[failure_index]
    expected=dict(call_id=failed['call_id'],index=failure_index,reason=failed['failure_reason'],
        archive_sha256=failed['archive_sha256'],stage='reference')
    assert seal['stop_trigger']==expected and failed['stop_trigger'] is None
    assert len(records)==4 and len(calls)==failure_index+1
    assert [r['call_index'] for r in records]==list(range(4))
    assert [r['call_id'] for r in records]==lock['ordered_call_ids']
    assert seal['executed_failed_calls']==1 and seal['fail_fast_skipped_calls']==3-failure_index
    assert seal['unattempted_failure_calls']==0
    starts=[json.loads(x) for x in (out/'forward/call_starts.jsonl').read_text().splitlines()]
    assert len(starts)==failure_index+1
    for row in records[failure_index+1:]:
        assert row['stop_trigger']==expected and row['not_attempted_reason']=='fail_fast'
        assert row['failure_stage']=='fail_fast' and not row['attempted'] and not row['passed']
        assert not row['real_inputs_read'] and not row['independent_real_reference']
        assert row['budget_used_before']==row['budget_used_after']==seal['total_cell_node_evaluations']
        assert row['work']=={} and row['array_sha256']=={}
        with np.load(out/row['archive_path'],allow_pickle=False) as archive:assert archive.files==[]
    assert all('independent_probability_reference' not in r for r in records)


def test_skip_never_loads_inputs_or_invokes_reference_and_serializes_trigger(tmp_path,monkeypatch):
    out,row,_=save_prepared(tmp_path);trigger=dict(call_id='earlier__order32',index=0,
        reason='ReferenceFailure: mock\nquoted "failure"',archive_sha256='mock_digest',stage='reference')
    monkeypatch.setattr(P.np,'load',lambda *a,**k:pytest.fail('skip must not decode'))
    monkeypatch.setattr(P.REF,'integrate',lambda **k:pytest.fail('skip must not call'))
    result=invoke(out,row,call_index=1,stop_trigger=trigger)
    journal=[json.loads(x) for x in (out/'forward/reference_results.jsonl').read_text().splitlines()]
    assert result['stop_trigger']==journal[0]['stop_trigger']==trigger
    assert result['not_attempted_reason']=='fail_fast' and result['work']=={}
    assert result['array_sha256']=={} and not result['real_inputs_read']
    assert (out/'forward/call_starts.jsonl').read_text()==''


@pytest.mark.parametrize('partial_decode',(False,True))
def test_input_decode_failure_triggers_once_then_skips_all_reads(tmp_path,monkeypatch,partial_decode):
    out,rows,lock=mock_run_state(tmp_path,monkeypatch);loads=[]
    class Broken:
        files=list(C.OLD_FIELDS)
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def __getitem__(self,name):
            if partial_decode and name==C.OLD_FIELDS[0]:return toy_old()[name]
            raise OSError('mock input decode failure')
    def load(*args,**kwargs):loads.append(args);return Broken()
    monkeypatch.setattr(P.np,'load',load)
    monkeypatch.setattr(P.REF,'integrate',lambda **k:pytest.fail('bad input must not call'))
    P.run(out)
    records=C.read(out/'forward/records.json');seal=C.read(out/'forward_complete.json')
    assert len(loads)==1 and len(records)==4
    assert records[0]['failure_stage']==records[0]['not_attempted_reason']=='input_decode'
    assert records[0]['real_inputs_read'] is partial_decode and records[0]['array_sha256']=={}
    assert seal['real_inputs_read'] is partial_decode and not seal['independent_real_reference']
    assert seal['stop_trigger']['stage']=='input_decode' and seal['attempted_calls']==0
    assert seal['unattempted_failure_calls']==1 and seal['fail_fast_skipped_calls']==3
    assert all(row['charged_cell_node_evaluations']==0 for row in records)


def test_phase_deadline_before_first_load_retains_144_style_schedule_without_calls(tmp_path,monkeypatch):
    out,rows,lock=mock_run_state(tmp_path,monkeypatch)
    lock['reference_policy']['max_phase_wall_seconds']=-1.
    monkeypatch.setattr(P.np,'load',lambda *a,**k:pytest.fail('deadline must precede decode'))
    monkeypatch.setattr(P.REF,'integrate',lambda **k:pytest.fail('deadline must precede reference'))
    P.run(out)
    records=C.read(out/'forward/records.json');seal=C.read(out/'forward_complete.json')
    assert len(records)==4 and seal['attempted_calls']==0
    assert records[0]['failure_stage']==records[0]['not_attempted_reason']=='phase_deadline'
    assert seal['stop_trigger']['stage']=='phase_deadline' and seal['fail_fast_skipped_calls']==3
    assert not seal['real_inputs_read'] and not seal['independent_real_reference']
    assert seal['total_cell_node_evaluations']==0 and not seal['phase_wall_passed']


def test_nonfinite_partial_arrays_remain_npz_evidence_not_invalid_json(tmp_path,monkeypatch):
    out,row,_=save_prepared(tmp_path)
    def reference(**kwargs):
        error=ValueError('mock nonfinite numerical failure')
        error.reference_work={'cell_node_evaluations':0}
        error.reference_metadata={'scaled_outer_error_estimate':None,'outer_convergence_success':False}
        error.available_arrays={'partial_log_raw_mass':np.asarray([np.nan,-np.inf]),
            'partial_log_raw_fov_mass':np.asarray([np.inf]),'partial_raw_computed':np.ones(3,bool)}
        raise error
    monkeypatch.setattr(P.REF,'integrate',reference)
    result=invoke(out,row)
    journal=json.loads((out/'forward/reference_results.jsonl').read_text())
    assert journal['scaled_outer_error_estimate'] is None and not result['passed']
    with np.load(out/result['archive_path'],allow_pickle=False) as archive:
        assert np.isnan(archive['partial_log_raw_mass'][0]) and np.isposinf(archive['partial_log_raw_fov_mass'][0])


@pytest.mark.parametrize('failure_index',(None,0,143))
def test_all_144_scheduled_mock_outcomes_and_oracle_inventory_are_retained(tmp_path,monkeypatch,failure_index):
    out,rows,lock=mock_run_state(tmp_path,monkeypatch,case_count=72)
    oracles=toy_oracles(rows,tmp_path)
    C.write(out/'oracle_specs.json',oracles);lock['oracle_specs_sha256']=C.sha(out/'oracle_specs.json')
    lock['reference_policy'].update(max_total_cell_node_evaluations=2000,max_phase_wall_seconds=30.)
    calls=[]
    def reference(**kwargs):
        index=len(calls);calls.append(kwargs['order']);result=success_reference(**kwargs)
        if index==failure_index:result.update(passed=False,status='failed',failure_reason='mock fixed-population failure')
        return result
    monkeypatch.setattr(P.REF,'integrate',reference)
    P.run(out)
    records=C.read(out/'forward/records.json');seal=C.read(out/'forward_complete.json')
    assert len(records)==144 and [r['call_id'] for r in records]==lock['ordered_call_ids']
    assert [r['call_index'] for r in records]==list(range(144))
    assert len(calls)==(144 if failure_index is None else failure_index+1)
    assert seal['reference_calls_completed'] is (failure_index is None)
    assert seal['oracle_specs_sha256']==lock['oracle_specs_sha256']
    assert C.read(out/'oracle_specs.json')==oracles and len(oracles)==40
    assert seal['total_cell_node_evaluations']==7*len(calls)
    assert seal['fail_fast_skipped_calls']==(0 if failure_index is None else 143-failure_index)
    if failure_index is not None:
        assert all(r['work']=={} and not r['array_sha256'] and not r['real_inputs_read'] for r in records[failure_index+1:])
    completion=C.read(out/'completion.json')
    assert completion['files']['oracle_specs.json']==lock['oracle_specs_sha256']
    C.verify_inventory(out,completion['files'],exclude=('completion.json',))


def component_trace(failed=False):
    """Saved-array mocks only; these are not integration or accuracy fixtures."""
    bounds=np.asarray([[0.,1.],[0.,.5]] if failed else [[0.,1.],[1.,2.]])
    arrays=dict(outer_interval_bounds=bounds,
        outer_interval_integrals=np.asarray([[1.,2.,3.],[.5,1.,1.5]]),
        outer_interval_truncation_error=np.asarray([.1,.02] if failed else [0.,0.]),
        outer_interval_roundoff=np.asarray([[1e-14,2e-14,3e-14],[.5e-14,1e-14,1.5e-14]]),
        outer_active_interval_indices=np.asarray([0] if failed else [0,1],np.int64))
    metadata=dict(complete=not failed,success=not failed,status='exception' if failed else 'converged',
        sum_active_truncation_error=.1 if failed else 0.,maximum_historical_roundoff=4.5e-14,
        estimated_error=.1+4.5e-14 if failed else 4.5e-14,target=1e-12,
        initial_interval_count=1 if failed else 2,completed_interval_evaluations=2,split_count=0,
        callback_evaluations=47 if failed else 42,initial_checked_before_split=True,
        roundoff_epsilon=float(np.finfo(np.float64).eps),roundoff_factor=50.,
        pending_interval_bounds=[.5,1.] if failed else None,error_estimate_is_not_true_error_certificate=True)
    return arrays,metadata


@pytest.mark.parametrize('failed',(False,True))
def test_complete_and_interrupted_component_trace_is_saved_without_allowlist_loss(tmp_path,monkeypatch,failed):
    out,row,_=save_prepared(tmp_path);trace,metadata=component_trace(failed)
    def reference(**kwargs):
        if not failed:
            result=success_reference(**kwargs)
            result.update(trace,outer_component_roundoff=metadata)
            return result
        kwargs['budget'].charge(9)
        error=RuntimeError('mock unfinished right child')
        error.reference_work=dict(cell_node_evaluations=9,quadrature_depth_evaluations=46)
        error.reference_metadata=dict(outer_component_roundoff=metadata)
        error.available_arrays=dict(trace,partial_log_raw_mass=np.full(2,-np.inf),
            partial_log_raw_fov_mass=np.full(1,-np.inf),partial_raw_computed=np.zeros(3,bool))
        raise error
    monkeypatch.setattr(P.REF,'integrate',reference)
    result=invoke(out,row)
    assert result['attempted'] and result['passed'] is (not failed)
    assert result['reference_work_consistent'] and result['outer_component_roundoff']==metadata
    journal=json.loads((out/'forward/reference_results.jsonl').read_text())
    assert journal['outer_component_roundoff']==metadata
    assert set(trace)<=set(result['array_sha256'])
    with np.load(out/result['archive_path'],allow_pickle=False) as archive:
        for name,value in trace.items():
            assert archive[name].dtype==value.dtype and archive[name].shape==value.shape
            assert archive[name].tobytes()==value.tobytes()
            assert C.array_sha(archive[name])==result['array_sha256'][name]
        if failed:
            np.testing.assert_array_equal(archive['outer_active_interval_indices'],[0])
            assert 'probability' not in archive.files


def test_mixed_oracle_source_ids_keep_duplicate_query_order_regressions(tmp_path,monkeypatch):
    out,_,oracles=freeze_toy_metadata(tmp_path,monkeypatch)
    from collections import Counter
    assert Counter(o['reference_kind'] for o in oracles)=={'r33_log_conditional':19,'r35_log_real':21}
    assert len({(o['id'],o['oracle_id']) for o in oracles})==40
    assert len({(o['id'],o['order']) for o in oracles})==21
    assert C.read(out/'oracle_specs.json')==oracles


def test_producer_is_unchanged_r35_io_and_failfast_except_round_and_reference_imports():
    source=Path(P.__file__).read_text()
    old=Path(P.__file__).with_name('round35_log_real_preflight.py').read_text()
    expected=old.replace('R35','R37').replace('round35_log_real','round37_component_round_real')
    expected=expected.replace('round34_log_narrow_reference','round36_component_round_reference')
    assert source==expected
