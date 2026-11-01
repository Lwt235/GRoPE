import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments import round12_depth_train as trainer


def stress_bundle(tmp_path, passed=True):
    root = tmp_path/'stress'; root.mkdir()
    lock = dict(source_hashes={})
    (root/'lock.json').write_text(json.dumps(lock))
    lock_sha = trainer.sha(root/'lock.json')
    (root/'lock.sha256').write_text(lock_sha)
    summary = dict(lock_sha256=lock_sha, gates=dict(passed=passed))
    (root/'summary.json').write_text(json.dumps(summary))
    completion = dict(lock_sha256=lock_sha, fresh_test_read=False,
                      files={'summary.json': trainer.sha(root/'summary.json')})
    (root/'completion.json').write_text(json.dumps(completion))
    audit = tmp_path/'audit.json'
    audit.write_text(json.dumps(dict(passed=True, stress_lock_sha256=lock_sha,
        stress_summary_sha256=trainer.sha(root/'summary.json'),
        stress_completion_sha256=trainer.sha(root/'completion.json'))))
    return root, audit


def test_stress_pass_requires_bound_independent_audit(tmp_path):
    root, audit = stress_bundle(tmp_path)
    result = trainer.stress_authorization(root, audit)
    assert result['stress_passed'] and result['independent_audit_passed']
    a = json.loads(audit.read_text()); a['stress_summary_sha256'] = '0'*64
    audit.write_text(json.dumps(a))
    with pytest.raises(ValueError, match='audit'):
        trainer.stress_authorization(root, audit)


def test_failed_stress_prevents_freeze_before_geometry_or_old_labels_open(tmp_path, monkeypatch):
    root, audit = stress_bundle(tmp_path, passed=False)
    def should_not_read():
        pytest.fail('old input/label read before promotion check')
    monkeypatch.setattr(trainer, 'original_manifests', should_not_read)
    out = tmp_path/'depth'
    with pytest.raises(ValueError, match='stress gate failed'):
        trainer.freeze(out, tmp_path/'missing_geometry', root, audit)
    assert not out.exists()


def test_changed_completed_stress_artifact_is_rejected(tmp_path):
    root, audit = stress_bundle(tmp_path)
    completed = json.loads((root/'completion.json').read_text())
    completed['files']['extra.json'] = '0'*64
    (root/'extra.json').write_text('{}')
    (root/'completion.json').write_text(json.dumps(completed))
    a = json.loads(audit.read_text()); a['stress_completion_sha256'] = trainer.sha(root/'completion.json')
    audit.write_text(json.dumps(a))
    with pytest.raises(ValueError, match='completed artifact'):
        trainer.stress_authorization(root, audit)


def test_role_loader_checks_original_identity_before_geometry_read(monkeypatch):
    ids = [dict(scene='s', q_index=1, k_index=2, pair_id='1:2')]
    monkeypatch.setattr(trainer, 'load_data', lambda *args: ({}, ids, []))
    lock = dict(old_data={'train': [dict(scene='s', pairs=[[2, 1]])]}, geometry={})
    with pytest.raises(ValueError, match='ordering'):
        trainer.load_role_data('full4', 'train', lock)


def test_singleview_loader_never_reads_geometry(monkeypatch):
    ids = [dict(scene='s', q_index=1, k_index=2, pair_id='1:2')]
    data = {'q': torch.ones(1, 432, 128)}
    monkeypatch.setattr(trainer, 'load_data', lambda *args: (data, ids, []))
    lock = dict(old_data={'train': [dict(scene='s', pairs=[[1, 2]])]}, geometry=None)
    result, _, _ = trainer.load_role_data('singleview', 'train', lock)
    assert result is data and 'geometry_logits' not in result


def test_loader_reads_only_requested_role_and_joins_manifest(tmp_path, monkeypatch):
    ids = [dict(scene='s', q_index=1, k_index=2, pair_id='1:2')]
    data = {'q': torch.ones(1, 432, 128)}
    monkeypatch.setattr(trainer, 'load_data', lambda *args: (data, ids, []))
    path = tmp_path/'full4.npy'; np.save(path, np.full((1, 432, 432), -.7, dtype=np.float32))
    manifest = tmp_path/'manifest.json'; manifest.write_text('{}')
    row = dict(scene='s', path=str(tmp_path), manifest_sha256=trainer.sha(manifest),
               roles={'full4': dict(path=str(path), sha256=trainer.sha(path)),
                      'quad16': dict(path='/missing/never-open.npy', sha256='bad')})
    lock = dict(old_data={'train': [dict(scene='s', pairs=[[1, 2]])]}, geometry={'scenes': {'train': [row]}})
    result, _, _ = trainer.load_role_data('full4', 'train', lock)
    assert result['geometry_logits'].shape == (1, 432, 432)
    assert result['geometry_logits'].dtype == torch.float32
    np.testing.assert_allclose(result['geometry_logits'].numpy(), -.7)
