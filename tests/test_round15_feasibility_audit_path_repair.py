import ast
from copy import deepcopy
import inspect
import json
from pathlib import Path

import pytest

from experiments import round15_feasibility_audit as original
from experiments import round15_feasibility_audit_path_repair as repair


def test_repository_relative_and_absolute_input_paths(tmp_path):
    relative = tmp_path/'results'/'inputs.npz'
    relative.parent.mkdir()
    relative.write_bytes(b'opaque forward-only input')
    absolute = tmp_path/'absolute.npz'
    absolute.write_bytes(b'another input')
    files = {'results/inputs.npz': repair.sha(relative), str(absolute): repair.sha(absolute)}
    repair.verify_files(tmp_path, files)
    with pytest.raises(FileNotFoundError):
        repair.verify_files(Path('/'), files)
    relative.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed artifact'):
        repair.verify_files(tmp_path, files)


def test_all_scientific_functions_remain_identical():
    for name in ('independent_center_selection', 'verify_constraint_table',
                 'sha', 'array_sha', 'read', 'write', 'verify_files'):
        assert inspect.getsource(getattr(original, name)) == inspect.getsource(getattr(repair, name))
    assert (original.SCENES, original.QUERIES, original.FAMILIES, original.GRID) == (
        repair.SCENES, repair.QUERIES, repair.FAMILIES, repair.GRID)


def test_audit_body_has_only_path_and_provenance_changes():
    expected = inspect.getsource(original.audit)
    expected = expected.replace("    verify_files(ROOT, lock['source_hashes'])\n",
        "    verify_repair_provenance(lock)\n    verify_files(ROOT, lock['source_hashes'])\n")
    expected = expected.replace("verify_files(Path('/'), producer['inputs']); verify_files(Path('/'), producer['manifests'])",
        "verify_files(ROOT, producer['inputs']); verify_files(ROOT, producer['manifests'])")
    expected = expected.replace("producer['created_unix'] <= lock['created_unix'] <= fc['started_unix']",
        "producer['created_unix'] <= lock['original_design_created_unix'] <= fc['started_unix']")
    expected = expected.replace("        independent_audit_lock_sha256=sha(out/'lock.json'), probability_arrays=len(rows),",
        "        independent_audit_lock_sha256=sha(out/'lock.json'), probability_arrays=len(rows),\n"
        "        original_audit_lock_sha256=lock['path_repair']['original_lock_sha256'],\n"
        "        original_design_created_unix=lock['original_design_created_unix'],\n"
        "        repair_created_unix=lock['created_unix'], path_repair=lock['path_repair'],")
    assert ast.dump(ast.parse(expected)) == ast.dump(ast.parse(inspect.getsource(repair.audit)))


@pytest.fixture
def metadata_chain(tmp_path, monkeypatch):
    root = tmp_path/'repo'; root.mkdir()
    prior = root/'old_audit'; run = root/'producer'
    prior.mkdir(); run.mkdir()
    old_sources = {'original.py': 'original source', 'original_test.py': 'original tests'}
    producer_sources = {'producer.py': 'producer source'}
    for name, content in {**old_sources, **producer_sources}.items():
        (root/name).write_text(content)
    for directory, names in ((prior, old_sources), (run, producer_sources)):
        (directory/'source_snapshot').mkdir()
        for name in names:
            (directory/'source_snapshot'/name).write_bytes((root/name).read_bytes())
    producer = dict(created_unix=10., source_hashes={p: repair.sha(root/p) for p in producer_sources})
    repair.write(run/'lock.json', producer)
    (run/'lock.sha256').write_text(repair.sha(run/'lock.json')+'\n')
    old = dict(created_unix=20., run=str(run), source_hashes={p: repair.sha(root/p) for p in old_sources},
        producer_sources=producer['source_hashes'], producer_lock_sha256=repair.sha(run/'lock.json'),
        before_new_arrays_metrics_summary=True, scientific_gate_passed=False,
        independent_reference_probability_tolerance=1e-4, coverage='all populations')
    repair.write(prior/'lock.json', old)
    (prior/'lock.sha256').write_text(repair.sha(prior/'lock.json')+'\n')
    failed = dict(recorded_unix=30., exception='FileNotFoundError', exit_code=1,
        new_probability_arrays_opened=False, source='experiments/round15_feasibility_audit.py',
        line=197, root_information_before_failure=dict(producer_reported_feasible_grid_indices=[6, 9], detailed_summary_opened=False))
    failure = prior/'path_resolution_failure.json'; repair.write(failure, failed)
    monkeypatch.setattr(repair, 'ROOT', root)
    monkeypatch.setattr(repair, 'ORIGINAL_LOCK_SHA256', repair.sha(prior/'lock.json'))
    monkeypatch.setattr(repair, 'FAILURE_SHA256', repair.sha(failure))
    monkeypatch.setattr(repair, 'REPAIR_SOURCES', tuple(old_sources))
    monkeypatch.setattr(repair.time, 'time', lambda: 40.)
    return root, prior, run, failure, old


def test_freeze_preserves_original_and_records_honest_new_time(metadata_chain):
    root, prior, run, failure, old = metadata_chain
    originals = {p: p.read_bytes() for p in prior.rglob('*') if p.is_file()}
    out = root/'repair'
    repair.freeze(out, run, prior, failure)
    lock = json.loads((out/'lock.json').read_text())
    assert lock['created_unix'] == 40.
    assert lock['original_design_created_unix'] == 20.
    assert lock['before_new_arrays_metrics_summary'] is False
    assert lock['before_repair_audit_array_reads'] is True
    assert lock['path_repair']['information_already_reported']['producer_reported_feasible_grid_indices'] == [6, 9]
    assert repair.verify_repair_provenance(lock) == old
    assert originals == {p: p.read_bytes() for p in prior.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='overwrite'):
        repair.freeze(out, run, prior, failure)


def test_repair_rejects_backdating_or_changed_scientific_policy(metadata_chain):
    root, prior, run, failure, _ = metadata_chain
    repair.freeze(root/'repair', run, prior, failure)
    lock = json.loads((root/'repair'/'lock.json').read_text())
    for change in ({'original_design_created_unix': 1.}, {'created_unix': 20.},
                   {'before_new_arrays_metrics_summary': True}):
        bad = deepcopy(lock); bad.update(change)
        with pytest.raises(ValueError, match='backdating'):
            repair.verify_repair_provenance(bad)
    bad = deepcopy(lock); bad['independent_reference_probability_tolerance'] = 1e-3
    with pytest.raises(ValueError, match='policy changed'):
        repair.verify_repair_provenance(bad)


def test_repair_rejects_original_source_snapshot_and_failure_tampering(metadata_chain):
    root, prior, run, failure, _ = metadata_chain
    for path in (root/'original.py', prior/'source_snapshot'/'original.py', failure):
        previous = path.read_bytes()
        path.write_bytes(previous+b' changed')
        with pytest.raises(ValueError):
            repair.original_contract(prior, run, failure)
        path.write_bytes(previous)
