"""Saved-array arithmetic extension tests; no real numerical artifacts read."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments import round17_arithmetic_audit as A


def fixture_arrays(T=5.):
    raw=np.array([[0.,0.],[.2,.3],[1e-25,2e-25]],np.float64)
    cov=np.array([np.zeros((2,2)),[[.3,.1],[.1,.4]],np.eye(2)*2e38],np.float32)
    content=np.array([[.12345679,.9876543],[.3,-.4],[-.2,.1]],np.float32)
    bounds=np.array([[0.,0.,1.,1.],[1.,0.,4.,3.]],np.float64)
    area=np.prod(bounds[:,2:]-bounds[:,:2],axis=1)
    geometry=torch.from_numpy(np.log(raw+1e-30*area))
    saved=A.P.probabilities(geometry,torch.from_numpy(content),torch.from_numpy(cov),.5,T).numpy()
    return raw,content,cov,bounds,saved,.5,T


@pytest.mark.parametrize('T',[0.,5.,10.])
def test_all_arithmetic_and_tiny_population_preserved(T):
    values=fixture_arrays(T);result=A.audit_matrix(*values)
    assert result['passed'] and result['tau32_exact'] and result['bias32_exact']
    assert result['max_same_mass_probability_error']<1e-15
    assert result['raw_zero_fov_rows']==1 and result['raw_tiny_fov_rows']==2
    assert len(result['per_query_probability_error'])==len(result['actual_tau32'])==3
    assert len(result['array_hashes'])==10
    assert result['array_hashes']['saved_probability']==A.array_sha(values[4])


def test_strong_same_mass_check_detects_error_inside_old_integration_tolerance():
    values=list(fixture_arrays());values[4]=values[4].copy()
    values[4][1]+=[2e-8,-2e-8]
    result=A.audit_matrix(*values)
    assert not result['passed'] and result['tau32_exact'] and result['bias32_exact']
    assert 1e-12<result['max_same_mass_probability_error']<1e-4


def test_independent_formula_mismatch_fails_even_when_saved_probabilities_match(monkeypatch):
    values=fixture_arrays();original=A.P.trace_temperature
    monkeypatch.setattr(A.P,'trace_temperature',lambda *a,**kw:original(*a,**kw)+1e-4)
    result=A.audit_matrix(*values)
    assert not result['passed'] and not result['tau32_exact'] and not result['bias32_exact']
    assert result['max_same_mass_probability_error']<1e-15
    assert result['tau_mismatch_rows'] and result['bias_mismatch_rows']


def test_saved_dtype_and_native_area_floor_contract_are_not_silently_repaired():
    values=list(fixture_arrays());values[4]=values[4].astype(np.float32)
    with pytest.raises(ValueError,match='schema'):A.audit_matrix(*values)
    values=list(fixture_arrays());values[2]=values[2].astype(np.float64)
    with pytest.raises(ValueError,match='schema'):A.audit_matrix(*values)
    values=list(fixture_arrays());values[4]=values[4].copy();values[4][0]=[.5,.5]
    assert not A.audit_matrix(*values)['passed']


def test_freeze_is_required_before_any_train_directory(tmp_path,monkeypatch):
    run=tmp_path/'numeric';run.mkdir();(run/'train').mkdir()
    monkeypatch.setattr(A,'_base',lambda path:pytest.fail('train guard must precede base reads'))
    with pytest.raises(ValueError,match='before train'):A.freeze(tmp_path/'audit',run)
    assert not (tmp_path/'audit').exists()


def test_freeze_binds_new_sources_and_refuses_overwrite_or_snapshot_change(tmp_path,monkeypatch):
    run=tmp_path/'numeric';run.mkdir();base={'source_hashes':{'frozen.py':'b'*64}}
    monkeypatch.setattr(A,'_base',lambda path:(base,'a'*64))
    out=tmp_path/'audit';lock=A.freeze(out,run)
    assert lock['train_absent_at_freeze'] and lock['base_numeric_lock_sha256']=='a'*64
    assert lock['new_source_hashes']==A._source_hashes()
    assert lock['probability_matrices']==504 and lock['probability_rows']==4032
    assert A.verify_lock(out)[0]==lock
    with pytest.raises(FileExistsError):A.freeze(out,run)
    snapshot=out/'source_snapshot'/A.SOURCE_NAMES[0];snapshot.write_text('changed')
    with pytest.raises(ValueError,match='snapshot'):A.verify_lock(out)


def test_concurrent_train_start_cannot_be_claimed_as_prefrozen(tmp_path,monkeypatch):
    run=tmp_path/'numeric';run.mkdir();base={'source_hashes':{}}
    monkeypatch.setattr(A,'_base',lambda path:(base,'a'*64))
    original=A._source_hashes
    def race():
        (run/'train').mkdir()
        return original()
    monkeypatch.setattr(A,'_source_hashes',race)
    with pytest.raises(ValueError,match='during'):A.freeze(tmp_path/'audit',run)
    assert not (tmp_path/'audit'/'lock.json').exists()


def complete_stage(run):
    directory=run/'train';directory.mkdir(parents=True)
    A.write(directory/'summary.json',dict(passed=True,lock_sha256='a'*64))
    A.write(directory/'started.json',dict(started_unix=2.))
    value=dict(passed=True,lock_sha256='a'*64,files={p.name:A.sha(p) for p in directory.iterdir()})
    A.write(directory/'completion.json',value)
    return directory,value


def test_completed_stage_checks_all_bound_files_and_rejects_new_artifacts(tmp_path):
    directory,value=complete_stage(tmp_path)
    assert A._completed(tmp_path,'train','a'*64)[0]==value
    (directory/'unbound.npz').write_bytes(b'not an input')
    with pytest.raises(ValueError,match='population'):A._completed(tmp_path,'train','a'*64)


def test_completed_stage_does_not_accept_a_failed_numerical_stage(tmp_path):
    directory,value=complete_stage(tmp_path);value['passed']=False
    A.write(directory/'completion.json',value)
    with pytest.raises(ValueError,match='passing'):A._completed(tmp_path,'train','a'*64)


def test_unchanged_hash_binding_is_checked_before_archive_loading(tmp_path):
    directory,value=complete_stage(tmp_path)
    (directory/'summary.json').write_text('{}')
    with pytest.raises(ValueError,match='hash'):A._completed(tmp_path,'train','a'*64)


def test_fixed_populations_and_only_expected_saved_fields():
    assert len(A.CONDITIONS)==14 and len(set(A.CONDITIONS))==14
    assert len(A.SCENES)*len(A.CONDITIONS)==42
    assert 42*len(A.WIDTHS)==168
    assert 168*len(A.TEMPERATURES)==504 and 504*8==4032
    assert len(A.SUPPORT_FIELDS)==8 and len(A.GEOMETRY_FIELDS)==16
    assert all('target' not in x and 'gt_' not in x for x in A.SUPPORT_FIELDS|A.GEOMETRY_FIELDS)
