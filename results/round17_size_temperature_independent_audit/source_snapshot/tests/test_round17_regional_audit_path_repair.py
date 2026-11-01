"""Synthetic checks for the narrow historical-snapshot policy repair."""
import ast
from copy import deepcopy
import inspect
import json

import pytest

from experiments import round17_regional_audit as original
from experiments import round17_regional_audit_path_repair as repair


def test_legacy_verifier_changes_only_the_historical_snapshot_block():
    before = inspect.getsource(original.verify_legacy)
    block = ("    # R12 used flat snapshot filenames; later rounds preserve relative paths.\n"
             "    for name, digest in prior['source_hashes'].items():\n"
             "        if sha(directory / 'source_snapshot' / Path(name).name) != digest:\n"
             "            raise ValueError('legacy source snapshot changed')\n")
    assert block in before
    expected = before.replace(block, '    verify_historical_snapshots(directory, prior)\n')
    assert ast.dump(ast.parse(expected)) == ast.dump(ast.parse(inspect.getsource(repair.verify_legacy)))
    # All metrics, numerical checks, selectors and original audit remain the
    # unchanged imported function objects rather than copied repair code.
    assert repair.original is original
    assert repair.GRID is original.GRID


@pytest.fixture
def legacy_files(tmp_path, monkeypatch):
    project, legacy = tmp_path / 'project', tmp_path / 'legacy'
    project.mkdir(); (legacy / 'source_snapshot').mkdir(parents=True)
    files = (*repair.HISTORICAL_SNAPSHOTS, 'experiments/inherited.py', 'ROUND10_PROTOCOL.md')
    for name in files:
        path = project / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name + '\n')
    sources = {name: repair.sha(project / name) for name in files}
    for name in repair.HISTORICAL_SNAPSHOTS:
        (legacy / 'source_snapshot' / name.split('/')[-1]).write_bytes((project / name).read_bytes())
    monkeypatch.setattr(repair, 'ROOT', project)
    monkeypatch.setattr(repair.H, 'ROOT', project)
    return project, legacy, dict(source_hashes=sources, inherited=dict(files={}))


def test_only_three_historical_snapshots_are_mandatory_including_r12_protocol(legacy_files):
    project, directory, prior = legacy_files
    result = repair.verify_historical_snapshots(directory, prior)
    assert set(result['existing']) == set(repair.HISTORICAL_SNAPSHOTS)
    assert result['historical_not_snapshotted'] == ['ROUND10_PROTOCOL.md', 'experiments/inherited.py']
    assert result['all_original_source_hashes_verified'] and not result['fabricated_snapshots']
    assert not (directory / 'source_snapshot' / 'ROUND10_PROTOCOL.md').exists()
    (directory / 'source_snapshot' / 'ROUND12_STRESS_PROTOCOL.md').unlink()
    with pytest.raises(ValueError, match='mandatory'):
        repair.verify_historical_snapshots(directory, prior)


def test_unsnapshotted_sources_still_require_their_original_hash(legacy_files):
    project, directory, prior = legacy_files
    for name in ('ROUND10_PROTOCOL.md', 'experiments/inherited.py'):
        path = project / name; contents = path.read_bytes(); path.write_bytes(contents + b'changed')
        with pytest.raises(ValueError, match='changed artifact'):
            repair.verify_historical_snapshots(directory, prior)
        path.write_bytes(contents)


def test_existing_optional_snapshot_is_verified_not_ignored(legacy_files):
    project, directory, prior = legacy_files
    path = directory / 'source_snapshot' / 'ROUND10_PROTOCOL.md'
    path.write_bytes((project / 'ROUND10_PROTOCOL.md').read_bytes())
    assert 'ROUND10_PROTOCOL.md' in repair.verify_historical_snapshots(directory, prior)['existing']
    path.write_bytes(b'wrong')
    with pytest.raises(ValueError, match='existing historical'):
        repair.verify_historical_snapshots(directory, prior)


def test_unbound_and_ambiguous_extra_snapshot_rejected(legacy_files):
    project, directory, prior = legacy_files
    extra = directory / 'source_snapshot' / 'unknown.py'; extra.write_text('extra')
    with pytest.raises(ValueError, match='unbound'):
        repair.verify_historical_snapshots(directory, prior)
    extra.unlink()
    (project / 'other').mkdir(); (project / 'other/inherited.py').write_text('other')
    prior['source_hashes']['other/inherited.py'] = repair.sha(project / 'other/inherited.py')
    (directory / 'source_snapshot' / 'inherited.py').write_bytes((project / 'experiments/inherited.py').read_bytes())
    with pytest.raises(ValueError, match='ambiguous'):
        repair.verify_historical_snapshots(directory, prior)


@pytest.fixture
def metadata_chain(legacy_files, monkeypatch):
    project, legacy, prior = legacy_files
    run = project / 'results' / 'round17_regional'; run.mkdir(parents=True)
    (legacy / 'lock.json').write_text(json.dumps(prior))
    (legacy / 'lock.sha256').write_text(repair.sha(legacy / 'lock.json') + '\n')
    producer = dict(created_unix=10., source_hashes=prior['source_hashes'],
                    legacy=dict(root=str(legacy), lock_sha256=repair.sha(legacy / 'lock.json')))
    (run / 'lock.json').write_text(json.dumps(producer))
    (run / 'lock.sha256').write_text(repair.sha(run / 'lock.json') + '\n')
    for name in producer['source_hashes']:
        path = run / 'source_snapshot' / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((project / name).read_bytes())
    # Invalid JSON proves freeze only hashes these score/result artifacts.
    for name in ('summary.json', 'completion.json', 'forward_complete.json'):
        (run / name).write_text('opaque result bytes, MUST NOT JSON-DECODE')
    logs = [project / 'audit.log', project / 'pipeline.log']
    for path in logs:
        path.write_text('opaque log including potential scores; hash only')
    failure = project / 'failure.json'
    failure.write_text(json.dumps(dict(recorded_unix=30., stage='metadata_precheck_before_labels',
        frozen_lock_sha256=repair.sha(run / 'lock.json'),
        affected_function='experiments.round17_regional_audit.verify_legacy',
        observed_error='FileNotFoundError: results/round12_stress/source_snapshot/ROUND10_PROTOCOL.md',
        effects_inspected_by_root=False, original_source_unchanged=True,
        producer_started_before_precheck_failure_was_inspected=True, evaluation_exists_at_recording=True)))
    for name in repair.REPAIR_SOURCES:
        path = project / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(name)
    monkeypatch.setattr(repair.time, 'time', lambda: 40.)
    return project, run, failure, logs


def test_freeze_hashes_results_without_decoding_and_does_not_fabricate_original_lock(metadata_chain):
    project, run, failure, logs = metadata_chain
    out = project / 'repair'
    frozen = repair.freeze(out, run, failure, logs, authors_have_not_read_regional_scores=True)
    assert frozen['created_unix'] == 40.
    assert frozen['original_contract']['original_design_created_unix'] == 10.
    assert frozen['original_contract']['evaluation_exists_at_failure_recording'] is True
    assert frozen['frozen_before_regional_forward'] is False
    assert frozen['author_attestation_not_a_machine_access_proof'] is True
    assert frozen['authors_have_read_new_regional_effects'] is False
    assert frozen['original_contract']['original_audit_directory_exists'] is False
    assert frozen['original_contract']['optional_original_failure_files'] == {}
    assert repair.verify_repair(out) == frozen
    assert not (run.with_name('round17_regional_independent_audit') / 'lock.json').exists()


def test_freeze_requires_honest_attestation_and_rejects_overwrite(metadata_chain):
    project, run, failure, logs = metadata_chain
    out = project / 'repair'
    with pytest.raises(ValueError, match='attestation'):
        repair.freeze(out, run, failure, logs)
    repair.freeze(out, run, failure, logs, authors_have_not_read_regional_scores=True)
    with pytest.raises(ValueError, match='existing'):
        repair.freeze(out, run, failure, logs, authors_have_not_read_regional_scores=True)


@pytest.mark.parametrize('change', ['summary', 'log', 'source', 'snapshot', 'repair_source', 'failure'])
def test_frozen_repair_rejects_artifact_changes_without_reading_scores(metadata_chain, change):
    project, run, failure, logs = metadata_chain
    out = project / 'repair'
    repair.freeze(out, run, failure, logs, authors_have_not_read_regional_scores=True)
    path = {'summary': run / 'summary.json', 'log': logs[0], 'source': project / 'ROUND10_PROTOCOL.md',
            'snapshot': run / 'source_snapshot' / 'ROUND10_PROTOCOL.md',
            'repair_source': project / repair.REPAIR_SOURCES[0], 'failure': failure}[change]
    if change == 'failure':
        value = json.loads(path.read_text()); value['effects_inspected_by_root'] = True
        path.write_text(json.dumps(value))
    else:
        path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError):
        repair.verify_repair(out)


def test_context_replaces_only_legacy_and_metadata_writer_and_restores_on_error(metadata_chain):
    project, run, failure, logs = metadata_chain
    out = project / 'repair'
    lock = repair.freeze(out, run, failure, logs, authors_have_not_read_regional_scores=True)
    before = dict(legacy=original.verify_legacy, write=original.write, audit=original._audit,
                  selection=original.M.select_metrics, metrics=original.C.measure, numeric=original.reference_check)
    nested = out / 'audit'; nested.mkdir()
    with pytest.raises(RuntimeError):
        with repair.repaired_auditor(out, lock):
            assert original.verify_legacy is repair.verify_legacy
            assert original._audit is before['audit']
            assert original.M.select_metrics is before['selection']
            assert original.C.measure is before['metrics']
            assert original.reference_check is before['numeric']
            original.write(nested / 'lock.json', dict(pre_forward_design=True))
            original.write(nested / 'verification.json', dict(passed=False, train_screen=dict(passed=False)))
            original.write(nested / 'other.json', dict(unchanged=True))
            raise RuntimeError('unrelated audit error')
    assert original.verify_legacy is before['legacy'] and original.write is before['write']
    recorded = json.loads((nested / 'verification.json').read_text())
    assert not recorded['passed'] and not recorded['train_screen']['passed']
    assert recorded['path_repair']['repair_frozen_unix'] == 40.
    assert recorded['path_repair']['path_repair_frozen_before_regional_forward'] is False
    assert json.loads((nested / 'other.json').read_text()) == dict(unchanged=True)


def test_unrelated_repaired_audit_failure_is_retained_and_rerun_refused(metadata_chain, monkeypatch):
    project, run, failure, logs = metadata_chain
    out = project / 'repair'
    repair.freeze(out, run, failure, logs, authors_have_not_read_regional_scores=True)
    original_check = original.verify_legacy
    def fail(*args):
        raise ValueError('independent numeric failure')
    monkeypatch.setattr(original, 'audit', fail)
    with pytest.raises(ValueError, match='numeric'):
        repair.audit(out)
    assert original.verify_legacy is original_check
    complete = json.loads((out / 'completion.json').read_text())
    assert complete['passed'] is False and 'failure.json' in complete['files']
    with pytest.raises(ValueError, match='overwrite'):
        repair.audit(out)
