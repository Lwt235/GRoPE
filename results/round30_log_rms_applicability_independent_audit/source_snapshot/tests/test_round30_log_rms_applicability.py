"""Synthetic copy/IO/chronology toys; never decode formal R28 inputs."""
import json
from pathlib import Path
import time

import numpy as np
import pytest

from experiments import round30_log_rms_applicability as P

C = P.C


def toy_old():
    bounds = (np.asarray([(x,y,x+8,y+8) for y in range(0,144,8)
                         for x in range(0,192,8)], np.float64)-.5)/8.
    return dict(input__mu=np.asarray([.375,-0.,1.625]),
        input__factor=np.asarray([[.2,.1,-0.], [.03,.4,0.], [.01,-.02,.3]], np.float64),
        input__bounds=bounds, input__content_bias=np.zeros(432,np.float64))


def toy_spec(path=None, name='toy'):
    row = dict(id=name, scene='toy_scene', pair=[1,2], pair_name='toy_pair',
        query_size=24, query_index=0, query_bounds=[0,0,24,24])
    if path is not None:
        row.update(source_archive_path=str(path), archive_path='prepared/arrays/'+name+'.npz',
            archive_sha256=C.sha(path), array_sha256={n:C.array_sha(v) for n,v in toy_old().items()})
    return row


def save_old(tmp_path, name='old.npz'):
    path=tmp_path/name
    np.savez_compressed(path, **toy_old(), mapping_matrix=np.eye(3), r23__forbidden=np.asarray([999.]))
    return path


def save_prepared(tmp_path):
    out=tmp_path/'run';(out/'prepared/arrays').mkdir(parents=True);(out/'forward/arrays').mkdir(parents=True)
    for name in ('call_starts.jsonl','model_results.jsonl'):(out/'forward'/name).touch()
    bank=P.copy_inputs(toy_old());path=out/'prepared/arrays/toy.npz';np.savez_compressed(path,**bank)
    row=dict(toy_spec(),archive_path=str(path.relative_to(out)),archive_sha256=C.sha(path),
        array_sha256={n:C.array_sha(v) for n,v in bank.items()},passed=True,status='prepared',real_inputs_read=True)
    return out,row,bank


def success_model(**kwargs):
    n=len(kwargs['bounds']);logs=np.full(n,-900.,np.float64);lp=np.full(n,-np.log(n),np.float64)
    return dict(status='complete',passed=True,log_raw_mass=logs,log_raw_fov_mass=np.asarray([-900.+np.log(n)]),
        log_floored_mass=np.full(n,np.log(1e-30)),log_probability=lp,probability=np.exp(lp),
        materialized_raw_indices=np.empty(0,np.int64),materialized_raw_mass=np.empty(0,np.float64),
        work=dict(cell_node_evaluations=576,native_tiles_completed=1,fov_tiles_completed=0))


def test_copy_preserves_exact_four_fields_signed_zeros_and_no_aliasing():
    old=toy_old();new=P.copy_inputs(old)
    assert set(new)==P.PREPARED_FIELDS and len(new)==4
    for name,value in old.items():
        assert C.array_sha(new[name])==C.array_sha(value)
        assert new[name].tobytes()==value.tobytes()
        assert not np.shares_memory(new[name],value)
    assert np.signbit(new['input__mu'][1]) and np.signbit(new['input__factor'][0,2])
    new['input__mu'][0]=100.
    assert old['input__mu'][0]==.375


@pytest.mark.parametrize('change',('extra','missing','dtype','shape','nan'))
def test_copy_rejects_changed_input_contract(change):
    old=toy_old()
    if change=='extra':old['mapping_matrix']=np.eye(3)
    if change=='missing':del old['input__factor']
    if change=='dtype':old['input__mu']=old['input__mu'].astype(np.float32)
    if change=='shape':old['input__factor']=old['input__factor'][:2]
    if change=='nan':old['input__mu'][0]=np.nan
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
    path=save_old(tmp_path);spec=toy_spec(path);spec['archive_sha256']='wrong';state=dict(real_inputs_read=False)
    monkeypatch.setattr(P.np,'load',lambda *a,**k:pytest.fail('must not decode unauthenticated bytes'))
    with pytest.raises(ValueError,match='bytes changed'):P.load_old(spec,state)
    assert not state['real_inputs_read']


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


def test_freeze_metadata_only_preserves_existing_attempt(tmp_path,monkeypatch):
    (tmp_path/'source.py').write_text('# synthetic source\n')
    specs=[toy_spec(name='toy'+str(i)) for i in range(72)]
    monkeypatch.setattr(C,'ROOT',tmp_path);monkeypatch.setattr(C,'source_names',lambda:['source.py'])
    monkeypatch.setattr(C,'inherited_evidence',lambda:({'toy_binding':'toy_digest'},specs))
    monkeypatch.setattr(C,'check_runtime',lambda:{'synthetic_runtime':True})
    monkeypatch.setattr(P.np,'load',lambda *a,**k:pytest.fail('freeze must not decode'))
    monkeypatch.setattr(P.MODEL,'integrate',lambda *a,**k:pytest.fail('freeze must not predict'))
    out=tmp_path/'producer';P.freeze(out,tmp_path/'audit')
    assert C.read(out/'lock.json')['case_ids']==[s['id'] for s in specs]
    assert C.read(out/'lock.json')['real_inputs_read'] is False
    with pytest.raises(ValueError,match='existing'):P.freeze(out,tmp_path/'audit')


def test_prepare_retains_failed_copy_and_complete_population(tmp_path,monkeypatch):
    path=save_old(tmp_path);out=tmp_path/'run';out.mkdir();C.write(out/'lock.json',{'toy':True})
    specs=[toy_spec(path,'toy'+str(i)) for i in range(3)];specs[1]['array_sha256']['input__mu']='wrong'
    C.write(out/'query_specs.json',specs)
    lock=dict(case_ids=[s['id'] for s in specs],query_specs_sha256=C.sha(out/'query_specs.json'))
    monkeypatch.setattr(P,'verify',lambda unused:lock);monkeypatch.setattr(P,'require_audit_freeze',lambda *a:'audit')
    monkeypatch.setattr(P.MODEL,'integrate',lambda *a,**k:pytest.fail('prepare must not predict'))
    P.prepare(out);rows=C.read(out/'prepared/records.json')
    assert [r['passed'] for r in rows]==[True,False,True] and all(r['real_inputs_read'] for r in rows)
    seal=C.read(out/'prepared_complete.json')
    assert seal['case_count']==3 and not seal['all_inputs_prepared'] and not seal['predictions_executed']
    assert seal['real_inputs_read']
    done=C.read(out/'prepared/completion.json');C.verify_inventory(out/'prepared',done['files'],exclude=('completion.json',))
    with pytest.raises(ValueError,match='prior'):P.prepare(out)


def test_run_requires_input_audit_before_directory_or_model(tmp_path,monkeypatch):
    monkeypatch.setattr(P,'verify',lambda out:{})
    monkeypatch.setattr(P,'require_audit_freeze',lambda *a:'audit')
    def deny(*a):raise ValueError('input audit missing')
    monkeypatch.setattr(P,'require_input_audit',deny)
    monkeypatch.setattr(P.MODEL,'integrate',lambda *a,**k:pytest.fail('unverified forward'))
    with pytest.raises(ValueError,match='input audit'):P.run(tmp_path)
    assert not (tmp_path/'forward').exists()


def test_run_case_retains_canonical_subfloat_logs_and_exact_inputs(tmp_path,monkeypatch):
    out,row,bank=save_prepared(tmp_path);seen=[]
    def model(**kwargs):seen.append(kwargs);return success_model(**kwargs)
    monkeypatch.setattr(P.MODEL,'integrate',model)
    result=P.run_case(out,row,C.POLICY,time.perf_counter()+10)
    assert result['attempted'] and result['passed'] and result['real_inputs_read'] and len(seen)==1
    outputs={'log_raw_mass','log_raw_fov_mass','log_floored_mass','log_probability','probability',
        'materialized_raw_indices','materialized_raw_mass','argmax'}
    assert set(result['array_sha256'])==P.PREPARED_FIELDS|outputs
    for name in P.MODEL_FIELDS:assert C.array_sha(seen[0][name])==C.array_sha(bank['input__'+name])
    journal=[json.loads(s) for s in (out/'forward/model_results.jsonl').read_text().splitlines()]
    assert journal[0]['available_array_sha256']==result['array_sha256'] and journal[0]['real_inputs_read']
    with np.load(out/result['archive_path'],allow_pickle=False) as z:
        assert z['argmax'].dtype==np.int32 and z['argmax'].shape==()
        assert np.all(z['log_raw_mass']==-900.) and not len(z['materialized_raw_mass'])


def test_run_case_retains_partial_logs_masks_counts_and_work(tmp_path,monkeypatch):
    out,row,_=save_prepared(tmp_path);partial=np.r_[-900.,np.full(431,-np.inf)]
    result=dict(status='failed',passed=False,failure_reason='synthetic interruption',
        partial_log_raw_mass=partial,partial_log_raw_fov_mass=np.asarray([-np.inf]),
        partial_completed_cells=np.zeros(432,bool),partial_completed_fov=np.zeros(1,bool),
        partial_accumulated_tiles=np.r_[np.int64(1),np.zeros(431,np.int64)],partial_accumulated_fov_tiles=np.zeros(1,np.int64),
        work=dict(cell_node_evaluations=576,native_tiles_completed=1,fov_tiles_completed=0))
    monkeypatch.setattr(P.MODEL,'integrate',lambda **k:result)
    saved=P.run_case(out,row,C.POLICY,time.perf_counter()+10)
    assert saved['attempted'] and not saved['passed'] and saved['work']==result['work']
    assert 'probability' not in saved['array_sha256']
    with np.load(out/saved['archive_path'],allow_pickle=False) as z:
        for k,v in result.items():
            if isinstance(v,np.ndarray):np.testing.assert_array_equal(z[k],v)


def test_deadline_does_not_decode_or_invent_inputs(tmp_path,monkeypatch):
    out,row,_=save_prepared(tmp_path)
    monkeypatch.setattr(P.np,'load',lambda *a,**k:pytest.fail('expired phase must not decode'))
    monkeypatch.setattr(P.MODEL,'integrate',lambda *a,**k:pytest.fail('expired phase must not call'))
    result=P.run_case(out,row,C.POLICY,time.perf_counter()-1)
    assert not result['attempted'] and not result['passed'] and not result['real_inputs_read']
    assert result['array_sha256']=={} and result['work']=={}
    assert (out/'forward/call_starts.jsonl').read_text()==''


@pytest.mark.parametrize('tamper',('archive','canonical'))
def test_prepared_tamper_never_calls_model(tmp_path,monkeypatch,tamper):
    out,row,_=save_prepared(tmp_path)
    if tamper=='archive':row['archive_sha256']='wrong'
    else:row['array_sha256']['input__mu']='wrong'
    monkeypatch.setattr(P.MODEL,'integrate',lambda *a,**k:pytest.fail('tampered input must not predict'))
    result=P.run_case(out,row,C.POLICY,time.perf_counter()+10)
    assert not result['attempted'] and not result['passed'] and not result['array_sha256']
    assert result['real_inputs_read'] is (tamper=='canonical')


def test_model_exception_is_attempted_and_retains_inputs(tmp_path,monkeypatch):
    out,row,_=save_prepared(tmp_path)
    def model(**kwargs):raise ValueError('synthetic rejection')
    monkeypatch.setattr(P.MODEL,'integrate',model)
    result=P.run_case(out,row,C.POLICY,time.perf_counter()+10)
    assert result['attempted'] and not result['passed'] and result['real_inputs_read']
    assert set(result['array_sha256'])==P.PREPARED_FIELDS


def test_engineering_failure_keeps_read_scope_inventory_and_exit(tmp_path,monkeypatch):
    path=save_old(tmp_path);out=tmp_path/'run';out.mkdir();C.write(out/'lock.json',{'toy':True})
    specs=[toy_spec(path)];C.write(out/'query_specs.json',specs)
    monkeypatch.setattr(P,'verify',lambda unused:dict(case_ids=['toy'],query_specs_sha256='x'))
    monkeypatch.setattr(P,'require_audit_freeze',lambda *a:'audit')
    def fail_save(path,**arrays):raise OSError('synthetic disk failure after decode')
    monkeypatch.setattr(P.np,'savez_compressed',fail_save)
    with pytest.raises(OSError,match='after decode'):P.prepare(out)
    done=C.read(out/'prepared/completion.json')
    assert not done['complete_case_inventory'] and done['real_inputs_read']
    C.verify_inventory(out/'prepared',done['files'],exclude=('completion.json',))
