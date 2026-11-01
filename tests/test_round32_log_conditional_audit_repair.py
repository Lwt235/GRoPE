"""Pure serialization/path/metadata mocks; no saved formal arrays or integrals."""
from pathlib import Path
import json
from unittest.mock import patch
import numpy as np
import pytest
from experiments import round32_log_conditional_audit_repair as R


def test_nested_numpy_booleans_preserve_both_truth_values_and_json_style(tmp_path):
    source=dict(z=np.bool_(False),a=[np.bool_(True),dict(flag=np.bool_(False))],number=1.25)
    expected=dict(z=False,a=[True,dict(flag=False)],number=1.25)
    writer=R.C.write
    with R.serialized_boolean_writer(tmp_path):R.C.write(tmp_path/'value.json',source)
    assert R.C.write is writer
    assert (tmp_path/'value.json').read_text()==json.dumps(expected,indent=2,sort_keys=True,allow_nan=False)+'\n'
    assert type(source['z']) is np.bool_


@pytest.mark.parametrize('value',[np.int64(1),np.float32(1.),np.array([True]),object()])
def test_other_unsupported_types_still_fail_without_partial_json(tmp_path,value):
    writer=R.C.write
    with pytest.raises(TypeError),R.serialized_boolean_writer(tmp_path):R.C.write(tmp_path/'bad.json',{'value':value})
    assert R.C.write is writer and not (tmp_path/'bad.json').exists()


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf')])
def test_nonfinite_json_numbers_remain_rejected(tmp_path,value):
    with pytest.raises(ValueError),R.serialized_boolean_writer(tmp_path):R.C.write(tmp_path/'bad.json',{'value':value})
    assert not (tmp_path/'bad.json').exists()


def test_writer_confines_resolved_paths_and_restores_after_exception(tmp_path):
    out=tmp_path/'repair';out.mkdir();outside=tmp_path/'original';outside.mkdir();(out/'escape').symlink_to(outside,target_is_directory=True)
    writer=R.C.write
    with pytest.raises(ValueError),R.serialized_boolean_writer(out):R.C.write(out/'escape/value.json',{'value':np.bool_(True)})
    assert R.C.write is writer and not (outside/'value.json').exists()
    with pytest.raises(KeyboardInterrupt),R.serialized_boolean_writer(out):raise KeyboardInterrupt()
    assert R.C.write is writer


def test_frozen_math_output_remains_numpy_bool_and_only_serialization_changes(tmp_path):
    function=R.A.expected_depth_partition;before=R.C.sha(Path(R.A.__file__))
    result=function(np.array([0.,1.,18.]))
    assert type(result['strict_midpoints_exist']) is np.bool_
    record=dict(strict_midpoints_exist=result['strict_midpoints_exist'])
    with pytest.raises(TypeError):R.C.write(tmp_path/'old.json',record)
    with R.serialized_boolean_writer(tmp_path):R.C.write(tmp_path/'new.json',record)
    assert json.loads((tmp_path/'new.json').read_text())=={'strict_midpoints_exist':True}
    assert R.A.expected_depth_partition is function and R.C.sha(Path(R.A.__file__))==before


def test_disjoint_destination_and_prior_attempt_cannot_be_overwritten(tmp_path):
    run=tmp_path/'run';old=tmp_path/'old';run.mkdir();old.mkdir()
    for out in (run,old,run/'child',old/'child',tmp_path):
        with pytest.raises(ValueError):R.separate_destination(run,old,out)
    out=tmp_path/'repair';out.mkdir()
    with pytest.raises(ValueError):R.freeze(run,old,out)


def test_metadata_freeze_copies_only_original_input_evidence_without_decode(tmp_path):
    run=tmp_path/'run';old=tmp_path/'old';out=tmp_path/'repair';run.mkdir();old.mkdir()
    for name in R.COPY_FILES:
        (old/name).write_text('original-'+name)
    for name in ('source_snapshot/a.py','input_audit/started.json','input_audit/checks.json','input_audit/completion.json'):
        p=old/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('evidence-'+name)
    (old/'failure.json').write_text('retained original failure');(old/'checks.json').write_text('prior partial checks');(old/'completion.json').write_text('prior failed completion')
    sources=('wrapper.py','tests.py','protocol.md')
    for name in sources:(tmp_path/name).write_text(name)
    original_bytes={str(p.relative_to(old)):p.read_bytes() for p in old.rglob('*') if p.is_file()}
    bindings=dict(runtime={},reference_seal_finished_unix=0.,original_failure_sha256='failure',producer_files={},original_audit_files={})
    with patch.object(np,'load',side_effect=AssertionError('no decode at repair freeze')),\
            patch.object(R,'verify_original',return_value=bindings),patch.object(R,'extra_bindings',return_value={}),\
            patch.object(R.C,'check_runtime',return_value={}),patch.object(R.C,'ROOT',tmp_path),patch.object(R,'REPAIR_SOURCES',sources):
        lock=R.freeze(run,old,out)
    assert lock['post_seal_serialization_repair'] and not lock['prior_freeze_replaced']
    assert not (out/'failure.json').exists() and not (out/'checks.json').exists() and not (out/'completion.json').exists()
    assert {str(p.relative_to(old)):p.read_bytes() for p in old.rglob('*') if p.is_file()}==original_bytes
    for name in lock['copied_original_evidence']:assert (out/name).read_bytes()==(old/name).read_bytes()


def test_repaired_audit_preserves_numerical_failure_and_seals_once(tmp_path):
    out=tmp_path/'repair';out.mkdir();(out/'repair_lock.json').write_text('{}')
    lock=dict(producer_root=str(tmp_path/'run'),original_failure_sha256='failure')
    writer=R.C.write
    def fake_audit(run,destination):
        assert run==tmp_path/'run' and destination==out
        result=dict(numerical_preflight_passed=False,delivery_integrity_passed=True)
        R.C.write(out/'checks.json',{'strict_midpoints_exist':np.bool_(True),'passed':np.bool_(False)})
        R.C.write(out/'verification.json',result)
        R.C.write(out/'completion.json',dict(files=R.C.inventory(out)))
        return result
    with patch.object(R,'verify_repair',return_value=lock) as verify,patch.object(R.A,'audit',side_effect=fake_audit) as audit:
        result=R.run_repair(out)
        assert audit.call_count==1 and verify.call_count==2
        assert not result['verification']['numerical_preflight_passed']
        assert result['numerical_integrations_executed_by_repair']==0
        with pytest.raises(ValueError):R.run_repair(out)
    assert R.C.write is writer
    R.C.verify_inventory(out,R.C.read(out/'completion.json')['files'],exclude=('completion.json',))
