"""R20 synthetic provenance, all-row anchor and chronological guard tests."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments import round20_anchor_audit as A
from experiments import round20_anchor_fixed as P


def cohort():
    return [dict(scene=s, pair=[i*2, i*2+1], pair_name=f'{i:03d}', input_sha256='synthetic-input')
            for s in A.SCENES for i in range(16)]


def records(entries):
    output = []
    for entry in entries:
        for c in A.CONDITIONS:
            for family in A.FAMILIES:
                h, temperature = A.N.CONFIGS[family]
                output.append(dict(entry, **c, family=family, config=dict(h=h, temperature=temperature),
                    shape=[A.COUNTS[c['query']], A.COUNTS[c['key']]], dtype='float64',
                    archive_path='forward/' + entry['scene'].replace('/', '__') + '/arrays/' + entry['pair_name'] + '/' + c['condition'] + '.npz'))
    return output


def test_exact_8064_matrix_role_and_condition_population():
    entries = cohort(); full = records(entries)
    assert len(full) == 8064
    A.validate_records(full, entries)
    P.validate_records(full, entries)
    for changed in (full[:-1], full[:-1] + [full[0]]):
        with pytest.raises(ValueError): A.validate_records(changed, entries)
    for field, value in (('dtype', 'float32'), ('scenario', 'scale'), ('pair_name', 'wrong'),
                         ('config', dict(h=.5, temperature=10.))):
        changed = deepcopy(full); changed[0][field] = value
        with pytest.raises(ValueError): A.validate_records(changed, entries)


def test_joint_static_closure_policy_and_runtime_interfaces_agree():
    assert A.source_closure() == P.source_names()
    assert A.role_configs() == P.ROLE_CONFIGS
    assert A.NUMERICAL_POLICY == P.NUMERICAL_POLICY
    assert A.N.REFERENCE_POLICY == P.NUMREF.REFERENCE_POLICY
    # A new interpreter avoids changing an already initialized test process's
    # inter-op setting. This executes no observation/label/data loader.
    code = '''
import numpy
import torch
from threadpoolctl import threadpool_limits
from experiments import round20_anchor_audit as A
from experiments import round20_anchor_fixed as P
torch.set_num_threads(4)
torch.set_num_interop_threads(4)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
with threadpool_limits(limits=4):
    assert A.runtime()==P.runtime()
'''
    subprocess.run([sys.executable, '-c', code], check=True, cwd=A.ROOT)


def synthetic_support(tmp_path):
    parts = A.M.REGION.literal_parts()
    y, x = np.mgrid[:144, :192]
    inputs = dict(q_to_k_xy=np.stack([x, y], -1).reshape(-1, 2).astype(np.float64),
        q_to_k_pixel_cov=np.tile(np.eye(2)[None], (27648, 1, 1)),
        q_features=np.ones((432, 4), np.float32), k_features=np.ones((432, 4), np.float32),
        q_predicted_depth=np.full((144, 192), 2., np.float64), q_K=np.eye(3), k_K=np.eye(3), q_to_k_transform=np.eye(4))
    old, previous, saved = {}, {}, {}
    for q, part in parts.items():
        b = part['bounds']; a = (b[:, :2] + b[:, 2:] - 1) / 2.
        mean = (a / 8. + [.125, -.125]).astype(np.float32)
        covariance = np.tile(np.eye(2, dtype=np.float32)[None] * .1, (len(b), 1, 1))
        delta = mean.astype(np.float64) - a / 8.
        ca = A.N.fixed_mean_second_moment(mean, covariance, a / 8.)
        values = dict(bounds_px=b, a_px=a, a_front=np.ones(len(b), bool), a_fov=np.ones(len(b), bool),
            mean=mean, cov=covariance, a=a/8., delta=delta, anchor_cov=ca,
            q_features=np.ones((len(b), 4), np.float32), k_features=np.ones((len(b), 4), np.float32),
            tau32=A.N.GAUSSIAN.independent_temperature(ca, .25, 10.).astype(np.float32))
        saved.update({q + '__' + k: v for k, v in values.items()})
        old.update({q + '__bounds': b.astype(np.int32), q + '__mean': mean, q + '__cov': covariance})
        if q != 'mixed':
            previous.update({q + '__' + k: values[k] for k in ('a_px', 'a_front', 'a_fov')})
    for name, values in (('inputs', inputs), ('old', old), ('r19', previous), ('supports', saved)):
        np.savez(tmp_path / (name + '.npz'), **values)
    entry = dict(input_path=str(tmp_path/'inputs.npz'), support_path=str(tmp_path/'old.npz'),
                 r19_archive_path=str(tmp_path/'r19.npz'))
    record = dict(archive_path='supports.npz', archive_sha256=A.sha(tmp_path/'supports.npz'),
                  array_sha256={k:A.array_sha(v) for k,v in saved.items()})
    return entry, record, saved


def test_all_four_partitions_anchor_reconstruction_and_old_r19_identity(tmp_path):
    entry, record, saved = synthetic_support(tmp_path)
    _, _, error, rows = A.independent_anchors(entry, record, tmp_path)
    assert rows == 831 and error == 0.
    # Update canonical hashes after a synthetic tamper: independent mathematics,
    # not merely a file digest, must reject it.
    saved['mixed__a_front'][0] = False
    np.savez(tmp_path/'supports.npz', **saved)
    record.update(archive_sha256=A.sha(tmp_path/'supports.npz'), array_sha256={k:A.array_sha(v) for k,v in saved.items()})
    with pytest.raises(ValueError, match='flag'):
        A.independent_anchors(entry, record, tmp_path)


def test_metric_tolerances_never_apply_to_counts_fraction_values_or_booleans():
    A.compare({'kl_sum':1.+1e-9}, {'kl_sum':1.}, 1e-8)
    with pytest.raises(ValueError): A.compare({'mass':25.}, {'mass':25}, 1e-8)
    with pytest.raises(ValueError): A.compare({'value':np.nextafter(.5, np.inf)}, {'value':.5}, 1e-8)
    with pytest.raises(ValueError): A.compare({'passed':1}, {'passed':True}, 1e-8)


def test_joint_freeze_cannot_follow_a_forward_attempt(tmp_path, monkeypatch):
    run, out = tmp_path/'producer', tmp_path/'auditor'
    (run/'forward').mkdir(parents=True)
    monkeypatch.setattr(A, 'check_producer_lock', lambda _: pytest.fail('must reject before metadata reads'))
    with pytest.raises(ValueError, match='precede'):
        A.freeze(run, out)
    assert not out.exists()


def test_final_audit_label_barrier_and_failed_attempt_preservation(tmp_path, monkeypatch):
    run, out = tmp_path/'producer', tmp_path/'auditor'
    run.mkdir(); out.mkdir()
    target = tmp_path/'target.npz'; target.write_bytes(b'never-read-before-accepted-audit')
    monkeypatch.setattr(A, 'check_audit_lock', lambda *_: ({'entries':[{'target_path':str(target)}]}, {}))
    monkeypatch.setattr(A, 'check_seal', lambda *_: ([], [], {}))
    original = A.sha
    def guarded_sha(path):
        assert Path(path) != target, 'target touched before the complete forward audit'
        return original(path)
    monkeypatch.setattr(A, 'sha', guarded_sha)
    with pytest.raises(FileNotFoundError): A.audit(run, out)
    assert A.read(out/'final_audit/completion.json')['passed'] is False
    with pytest.raises(ValueError, match='preserve'): A.audit(run, out)


def test_failed_numerical_audit_cannot_authorize_producer_labels(tmp_path, monkeypatch):
    run, out = tmp_path/'producer', tmp_path/'auditor'
    run.mkdir(); out.mkdir()
    for filename in ('lock.json', 'forward/completion.json', 'forward/records.json', 'forward/support_records.json'):
        path = run/filename; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('{}')
    seal = dict(producer_lock_sha256=P.sha(run/'lock.json'),
        forward_completion_sha256=P.sha(run/'forward/completion.json'),
        forward_records_sha256=P.sha(run/'forward/records.json'),
        support_records_sha256=P.sha(run/'forward/support_records.json'),
        actual_target_hashes_read=False, labels_decoded=False)
    (run/'forward_complete.json').write_text(json.dumps(seal))
    (out/'forward_verification.json').write_text(json.dumps(dict(passed=False)))
    monkeypatch.setattr(P, 'require_audit_freeze', lambda *_: out)
    monkeypatch.setattr(P, 'verify_inventory', lambda folder: dict(forward_complete_sha256=P.sha(run/'forward_complete.json')))
    with pytest.raises(ValueError, match='numerical'):
        P.require_forward_audit(run, {})
