import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from experiments import round17_preflight as R


def test_row_check_preserves_declared_tolerances_and_reports_argmax():
    p = np.array([.50001,.49999],np.float64)
    ref = p[::-1].copy()
    row = R.row_check(p,ref,ref,p,13)
    assert row['passed'] and row['argmax_changed'] and row['query_index']==13
    assert row['reference_winner_advantage']==pytest.approx(.00002)
    far = np.array([.5002,.4998],np.float64)
    assert not R.row_check(far,ref,ref,far,13)['passed']


def test_reference_convergence_is_a_separate_gate():
    p = np.array([.4,.6],np.float64)
    reference64 = p+np.array([2e-8,-2e-8])
    assert not R.row_check(p,reference64,p,p,0)['passed']


@pytest.mark.parametrize('bad', [np.array([.4,.6],np.float32),
    np.array([.4,.5]), np.array([float('nan'),.6]), np.array([-.1,1.1])])
def test_invalid_probability_rows_rejected(bad):
    p=np.array([.4,.6],np.float64)
    with pytest.raises(ValueError): R.row_check(bad,p,p,p,0)


def test_no_label_fields_and_complete_fixed_population():
    assert R.FIELDS==('q_to_k_xy','q_to_k_pixel_cov','q_features','k_features')
    assert len(set(R.CONDITIONS))==14 and len(R.GRID)==12
    assert len(R.SCENES)*len(R.CONDITIONS)*len(R.GRID)*8==4032
    assert R.CONDITIONS[-1]==('g24','g24')
    assert R.absolute('relative')==R.ROOT/'relative'
    assert R.absolute('/tmp/absolute')==Path('/tmp/absolute')


def test_completed_stage_checks_exact_artifact_population(tmp_path):
    (tmp_path/'lock.json').write_text('{}')
    stage=tmp_path/'synthetic'; stage.mkdir()
    (stage/'summary.json').write_text('{"passed":true}')
    value=dict(passed=True,lock_sha256=R.sha(tmp_path/'lock.json'),
               files={'summary.json':R.sha(stage/'summary.json')})
    (stage/'completion.json').write_text(json.dumps(value))
    assert R.completed_stage(tmp_path,'synthetic')==value
    (stage/'unbound.json').write_text('{}')
    with pytest.raises(ValueError,match='population'): R.completed_stage(tmp_path,'synthetic')


def test_failed_preceding_stage_cannot_authorize_train(tmp_path):
    (tmp_path/'lock.json').write_text('{}')
    stage=tmp_path/'synthetic'; stage.mkdir()
    (stage/'completion.json').write_text(json.dumps(dict(passed=False,
        lock_sha256=R.sha(tmp_path/'lock.json'),files={})))
    with pytest.raises(ValueError,match='passing'): R.completed_stage(tmp_path,'synthetic')


def test_refuse_partial_stage_before_configuring_runtime(tmp_path):
    (tmp_path/'synthetic').mkdir()
    with mock.patch.object(R,'verify_lock',return_value={}),mock.patch.object(R.A,'configure_runtime') as config:
        with pytest.raises(FileExistsError): R.begin(tmp_path,'synthetic')
        config.assert_not_called()


def test_synthetic_adapter_uses_new_child_directory_and_merges_metadata(tmp_path):
    (tmp_path/'lock.json').write_text('{}')
    def independently_run(path):
        assert not path.exists()
        path.mkdir()
        (path/'verification.json').write_text('{"passed":true}')
        return dict(passed=True,labels_read=False,scientific_gate_passed=False)
    with mock.patch.object(R,'verify_lock',return_value={}), \
         mock.patch.object(R.A,'configure_runtime'), \
         mock.patch.object(R.N,'run_synthetic',side_effect=independently_run):
        R.synthetic(tmp_path)
    complete=R.completed_stage(tmp_path,'synthetic')
    assert 'independent/verification.json' in complete['files']
    assert json.loads((tmp_path/'synthetic/summary.json').read_text())['scientific_gate_passed'] is False
