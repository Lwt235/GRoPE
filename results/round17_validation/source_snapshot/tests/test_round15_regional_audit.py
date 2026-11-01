from copy import deepcopy
from pathlib import Path

import pytest

from experiments import round15_regional_audit as A


def test_relative_and_absolute_provenance_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(A, 'ROOT', tmp_path)
    path = tmp_path/'results'/'inputs.npz'
    path.parent.mkdir(); path.write_bytes(b'not decoded')
    assert A.absolute('results/inputs.npz') == path
    assert A.absolute(path) == path
    A.verify_files({'results/inputs.npz':A.sha(path), str(path):A.sha(path)}, tmp_path)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        A.verify_files({'results/inputs.npz':'wrong'}, tmp_path)


def test_one_bit_replay_or_configuration_change_is_rejected():
    entry = dict(input_sha256='input', label_sha256='label')
    forward = dict(grid_index=6, config=dict(h=1.,temperature=0.), shape=[48,432],
                   dtype='float64', input_sha256='input', array_sha256='committed')
    measured = dict(**forward, replayed_array_sha256='committed', label_sha256='label', n_query=48, n_key=432)
    A.check_replayed_row(forward, measured, entry, (48,432))
    for changes in (dict(replayed_array_sha256='different'), dict(label_sha256='another-label'),
                    dict(config=dict(h=.5,temperature=0.)), dict(n_key=108)):
        bad = deepcopy(measured); bad.update(changes)
        with pytest.raises(ValueError):
            A.check_replayed_row(forward, bad, entry, (48,432))


def test_summary_tolerance_never_changes_selection_identity():
    expected = dict(grid_index=6, config=dict(h=1.,temperature=0.), J=.4, eligible=[6,9])
    close = deepcopy(expected); close['J'] += 1e-12
    assert A.compare_tree(close, expected) < 1e-10
    wrong_index = deepcopy(close); wrong_index['grid_index'] = 9
    with pytest.raises(ValueError, match='identity'):
        A.compare_tree(wrong_index, expected)
    missing = deepcopy(expected); del missing['config']
    with pytest.raises(ValueError, match='keys'):
        A.compare_tree(missing, expected)
    with pytest.raises(ValueError, match='numeric'):
        A.compare_tree(float('nan'), .4)


def test_audit_freeze_reads_only_prior_metadata_and_refuses_overwrite(tmp_path, monkeypatch):
    root = tmp_path/'repo'; root.mkdir(); run = root/'run'; run.mkdir()
    monkeypatch.setattr(A, 'ROOT', root)
    monkeypatch.setattr(A, 'OWN_SOURCES', ('audit.py',))
    monkeypatch.setattr(A.N, 'SOURCE_NAMES', ('reference.py',))
    for name in ('audit.py','reference.py','producer.py'):
        (root/name).write_text('source '+name)
    producer = dict(source_hashes={str(root/'producer.py'):A.sha(root/'producer.py')})
    A.write(run/'lock.json', producer)
    (run/'lock.sha256').write_text(A.sha(run/'lock.json')+'\n')
    out = root/'audit'; A.freeze(out, run)
    lock = A.read(out/'lock.json')
    assert lock['stage_b_results_opened'] is False
    assert lock['regional_labels_decoded'] is False
    assert lock['expected_rows'] == 64512 and lock['expected_saved_arrays'] == 4032
    assert lock['source_hashes'] == {n:A.sha(root/n) for n in ('audit.py','reference.py')}
    with pytest.raises(ValueError, match='overwrite'):
        A.freeze(out, run)
