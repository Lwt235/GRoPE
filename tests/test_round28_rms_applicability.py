"""Independent synthetic IO/mapping/failure toys; no formal R23 archive access."""
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import pytest

from experiments import round28_rms_applicability as P

C = P.C


def toy_old():
    bounds = (np.asarray([(x,y,x+8,y+8) for y in range(0,144,8)
                         for x in range(0,192,8)], np.float64)-.5)/8.
    factor = np.asarray([[.2,.1,0.], [.03,.4,0.], [.01,-.02,.3]], np.float64)
    return dict(input__mu=np.asarray([.3,-.2,1.7]), input__factor=factor,
        input__bounds=bounds, input__H=np.asarray([[2.,.125],[-.25,3.]]),
        input__offset=np.asarray([11.25,8.125]), target__covariance=factor@factor.T,
        query_bounds_px=np.asarray([0,0,24,24],np.int64), query_index=np.asarray(0,np.int64))


def toy_spec(path=None, name='toy'):
    row = dict(id=name, scene='toy_scene', pair=[1,2], pair_name='toy_pair',
        query_size=24, query_index=0, query_bounds=[0,0,24,24])
    if path is not None:
        old = toy_old()
        row.update(source_archive_path=str(path), archive_path='prepared/arrays/'+name+'.npz',
            archive_sha256=C.sha(path), array_sha256={n:C.array_sha(v) for n,v in old.items()})
    return row


def save_old(tmp_path, name='old.npz'):
    path = tmp_path/name
    # Extra inherited fields must never be decoded by the new producer.
    np.savez_compressed(path, **toy_old(), source__forbidden=np.asarray([999.]))
    return path


def save_prepared(tmp_path):
    out = tmp_path/'run'
    (out/'prepared/arrays').mkdir(parents=True)
    (out/'forward/arrays').mkdir(parents=True)
    for name in ('call_starts.jsonl','model_results.jsonl'):
        (out/'forward'/name).touch()
    bank = P.map_geometry(toy_old(), toy_spec())
    path = out/'prepared/arrays/toy.npz'
    np.savez_compressed(path, **bank)
    row = dict(toy_spec(), archive_path=str(path.relative_to(out)), archive_sha256=C.sha(path),
        array_sha256={n:C.array_sha(v) for n,v in bank.items()}, passed=True, status='prepared')
    return out,row,bank


def test_mapping_preserves_full_intrinsics_offset_depth_and_thirteen_fields():
    old = toy_old(); original = {k:v.copy() for k,v in old.items()}
    new = P.map_geometry(old,toy_spec())
    assert set(new)==P.PREPARED_FIELDS and len(new)==13
    a = np.asarray([[2.,.125,11.25],[-.25,3.,8.125],[0.,0.,1.]])
    np.testing.assert_array_equal(new['mapping_matrix'],a)
    np.testing.assert_array_equal(new['input__mu'],a@old['input__mu'])
    np.testing.assert_array_equal(new['input__factor'],a@old['input__factor'])
    np.testing.assert_array_equal(new['input__mu'][2:],old['input__mu'][2:])
    np.testing.assert_array_equal(new['input__factor'][2],old['input__factor'][2])
    np.testing.assert_array_equal(new['input__bounds'],old['input__bounds'])
    assert np.all(new['input__content_bias']==0)
    for name,value in original.items():
        np.testing.assert_array_equal(old[name],value)
        np.testing.assert_array_equal(new['r23__'+name],value)
    np.testing.assert_allclose(new['input__factor']@new['input__factor'].T,
                               a@old['target__covariance']@a.T,rtol=1e-14,atol=1e-14)


@pytest.mark.parametrize('change',('extra','dtype','shape','nan','query_index','query_bounds'))
def test_mapping_rejects_changed_input_contract(change):
    old = toy_old()
    if change=='extra':old['source__mu']=np.zeros(3)
    if change=='dtype':old['input__H']=old['input__H'].astype(np.float32)
    if change=='shape':old['input__factor']=old['input__factor'][:2]
    if change=='nan':old['input__offset'][0]=np.nan
    if change=='query_index':old['query_index']=np.asarray(1,np.int64)
    if change=='query_bounds':old['query_bounds_px'][0]=1
    with pytest.raises(ValueError):P.map_geometry(old,toy_spec())


def test_mapping_overflow_is_failure_without_covariance_repair():
    old = toy_old();old['input__H'][0,0]=1e308;old['input__mu'][0]=1e308
    with pytest.raises((FloatingPointError,ValueError)):P.map_geometry(old,toy_spec())


def test_old_loader_decodes_exactly_eight_allowlisted_arrays(tmp_path,monkeypatch):
    path = save_old(tmp_path);spec=toy_spec(path)
    seen=[];real_load=np.load
    class Guard:
        def __enter__(self):
            self.archive=real_load(path,allow_pickle=False);return self
        def __exit__(self,*args):self.archive.close()
        def __getitem__(self,name):
            seen.append(name)
            assert name in C.OLD_FIELDS
            return self.archive[name]
    monkeypatch.setattr(P.np,'load',lambda *a,**k:Guard())
    assert set(P.load_old(spec))==set(C.OLD_FIELDS)
    assert tuple(seen)==C.OLD_FIELDS


def test_old_loader_rejects_archive_hash_before_decoding(tmp_path,monkeypatch):
    path=save_old(tmp_path);spec=toy_spec(path);spec['archive_sha256']='wrong'
    monkeypatch.setattr(P.np,'load',lambda *a,**k:pytest.fail('must not decode unauthenticated bytes'))
    with pytest.raises(ValueError,match='bytes changed'):P.load_old(spec)


def test_old_loader_rejects_canonical_hash_change(tmp_path):
    path=save_old(tmp_path);spec=toy_spec(path);spec['array_sha256']['input__offset']='wrong'
    with pytest.raises(ValueError,match='canonical'):P.load_old(spec)


def test_freeze_is_metadata_only_and_preserves_existing_attempt(tmp_path,monkeypatch):
    (tmp_path/'source.py').write_text('# synthetic source\n')
    specs=[dict(toy_spec(name='toy'+str(i))) for i in range(72)]
    monkeypatch.setattr(C,'ROOT',tmp_path)
    monkeypatch.setattr(C,'source_names',lambda:['source.py'])
    monkeypatch.setattr(C,'inherited_evidence',lambda:({'toy_binding':'toy_digest'},specs))
    monkeypatch.setattr(C,'check_runtime',lambda:{'synthetic_runtime':True})
    monkeypatch.setattr(P.np,'load',lambda *a,**k:pytest.fail('freeze must not decode arrays'))
    monkeypatch.setattr(P.MODEL,'integrate',lambda *a,**k:pytest.fail('freeze must not invoke model'))
    out=tmp_path/'producer'
    P.freeze(out,tmp_path/'audit')
    assert C.read(out/'lock.json')['case_ids']==[s['id'] for s in specs]
    assert C.read(out/'query_specs.json')==specs
    with pytest.raises(ValueError,match='existing'):P.freeze(out,tmp_path/'audit')


def test_prepare_retains_failed_mapping_and_complete_population(tmp_path,monkeypatch):
    path=save_old(tmp_path)
    out=tmp_path/'run';out.mkdir();C.write(out/'lock.json',{'toy':True})
    specs=[toy_spec(path,'toy'+str(i)) for i in range(3)]
    specs[1]['query_index']=9
    C.write(out/'query_specs.json',specs)
    lock=dict(case_ids=[s['id'] for s in specs],query_specs_sha256=C.sha(out/'query_specs.json'))
    monkeypatch.setattr(P,'verify',lambda unused:lock)
    monkeypatch.setattr(P,'require_audit_freeze',lambda *args:'audit_digest')
    monkeypatch.setattr(P.MODEL,'integrate',lambda *a,**k:pytest.fail('prepare must not predict'))
    P.prepare(out)
    rows=C.read(out/'prepared/records.json')
    assert [r['passed'] for r in rows]==[True,False,True]
    assert len(rows[1]['array_sha256'])==8
    assert rows[1]['failure_reason']
    seal=C.read(out/'prepared_complete.json')
    assert seal['case_count']==3 and not seal['all_inputs_prepared'] and not seal['predictions_executed']
    done=C.read(out/'prepared/completion.json')
    C.verify_inventory(out/'prepared',done['files'],exclude=('completion.json',))
    with pytest.raises(ValueError,match='prior'):P.prepare(out)


def test_run_is_blocked_before_any_forward_when_input_audit_missing(tmp_path,monkeypatch):
    monkeypatch.setattr(P,'verify',lambda out:{})
    monkeypatch.setattr(P,'require_audit_freeze',lambda *a:'audit')
    def deny(*a):raise ValueError('input audit missing')
    monkeypatch.setattr(P,'require_input_audit',deny)
    monkeypatch.setattr(P.MODEL,'integrate',lambda *a,**k:pytest.fail('unverified forward'))
    with pytest.raises(ValueError,match='input audit'):P.run(tmp_path)
    assert not (tmp_path/'forward').exists()


def test_run_case_keeps_complete_outputs_and_actual_call_journal(tmp_path,monkeypatch):
    out,row,bank=save_prepared(tmp_path);seen=[]
    def model(**kwargs):
        seen.append(kwargs)
        return dict(status='complete',passed=True,raw_mass=np.ones(432),
            floored_mass=np.ones(432),probability=np.full(432,1/432),work={'cell_node_evaluations':123})
    monkeypatch.setattr(P.MODEL,'integrate',model)
    result=P.run_case(out,row,C.POLICY,time.perf_counter()+10)
    assert result['attempted'] and result['passed'] and len(seen)==1
    assert set(result['array_sha256'])=={'input__'+n for n in P.MODEL_FIELDS}|{'raw_mass','probability','floored_mass','argmax'}
    np.testing.assert_array_equal(seen[0]['mu'],bank['input__mu'])
    journal=[json.loads(s) for s in (out/'forward/model_results.jsonl').read_text().splitlines()]
    assert journal[0]['available_array_sha256']==result['array_sha256']
    assert len((out/'forward/call_starts.jsonl').read_text().splitlines())==1
    with np.load(out/result['archive_path'],allow_pickle=False) as archive:
        assert archive['argmax'].dtype==np.int32 and archive['argmax'].shape==()


def test_run_case_retains_actual_partial_arrays_and_work(tmp_path,monkeypatch):
    out,row,bank=save_prepared(tmp_path)
    partial=np.r_[.125,np.zeros(431)]
    monkeypatch.setattr(P.MODEL,'integrate',lambda **kwargs:dict(status='failed',passed=False,
        failure_reason='synthetic numerical failure',partial_raw_mass=partial,
        partial_raw_fov_mass=np.asarray([0.]),work={'cell_node_evaluations':576}))
    result=P.run_case(out,row,C.POLICY,time.perf_counter()+10)
    assert result['attempted'] and not result['passed'] and result['work']['cell_node_evaluations']==576
    assert 'probability' not in result['array_sha256']
    with np.load(out/result['archive_path'],allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['partial_raw_mass'],partial)


def test_run_case_deadline_does_not_decode_or_invent_inputs(tmp_path,monkeypatch):
    out,row,bank=save_prepared(tmp_path)
    monkeypatch.setattr(P.np,'load',lambda *a,**k:pytest.fail('expired phase must not decode next archive'))
    monkeypatch.setattr(P.MODEL,'integrate',lambda *a,**k:pytest.fail('expired phase must not call model'))
    result=P.run_case(out,row,C.POLICY,time.perf_counter()-1)
    assert not result['attempted'] and not result['passed'] and result['array_sha256']=={}
    assert result['work']=={} and 'deadline' in result['failure_reason']
    assert (out/'forward/call_starts.jsonl').read_text()==''
    assert len((out/'forward/model_results.jsonl').read_text().splitlines())==1


def test_run_case_prepared_tamper_is_unattempted_failure(tmp_path,monkeypatch):
    out,row,bank=save_prepared(tmp_path);row['archive_sha256']='wrong'
    monkeypatch.setattr(P.MODEL,'integrate',lambda *a,**k:pytest.fail('tampered input must not predict'))
    result=P.run_case(out,row,C.POLICY,time.perf_counter()+10)
    assert not result['attempted'] and not result['passed'] and not result['array_sha256']


def test_run_case_thrown_model_exception_is_recorded_as_attempted(tmp_path,monkeypatch):
    out,row,bank=save_prepared(tmp_path)
    def model(**kwargs):raise ValueError('synthetic rejection')
    monkeypatch.setattr(P.MODEL,'integrate',model)
    result=P.run_case(out,row,C.POLICY,time.perf_counter()+10)
    assert result['attempted'] and not result['passed']
    assert set(result['array_sha256'])=={'input__'+n for n in P.MODEL_FIELDS}
    assert 'synthetic rejection' in result['failure_reason']


def test_stage_engineering_failure_preserves_inventory_and_raises(tmp_path,monkeypatch):
    out=tmp_path/'run';out.mkdir();C.write(out/'lock.json',{'toy':True})
    C.write(out/'query_specs.json',[])
    monkeypatch.setattr(P,'verify',lambda out:{'case_ids':[],'query_specs_sha256':'x'})
    monkeypatch.setattr(P,'require_audit_freeze',lambda *a:'audit')
    real_write=C.write
    def failing_write(path,value):
        if Path(path).name=='started.json':raise OSError('synthetic disk failure')
        return real_write(path,value)
    monkeypatch.setattr(C,'write',failing_write)
    with pytest.raises(OSError,match='synthetic disk failure'):P.prepare(out)
    failure=C.read(out/'prepared/completion.json')
    assert not failure['complete_case_inventory']
    C.verify_inventory(out/'prepared',failure['files'],exclude=('completion.json',))
