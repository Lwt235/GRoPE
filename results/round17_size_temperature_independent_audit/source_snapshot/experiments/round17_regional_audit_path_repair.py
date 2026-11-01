"""Post-freeze repair of the R17 auditor's historical R12 snapshot policy.

All scientific code executes from the unchanged, pre-forward frozen auditor.
Only its verify_legacy snapshot check is replaced in this process. R12 copied
three explicitly named files, not its entire transitive source_hashes mapping.
Every current original source hash is still checked; the three declared old
snapshots are mandatory, and every additional existing snapshot is verified.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import time

from experiments import round17_regional_audit as original


ROOT, H, GRID = original.ROOT, original.H, original.GRID
sha, read, write, absolute = original.sha, original.read, original.write, original.absolute
HISTORICAL_SNAPSHOTS = ('ROUND12_STRESS_PROTOCOL.md', 'models/round12_stress_pe.py',
                        'experiments/round12_stress_probe.py')
REPAIR_SOURCES = ('experiments/round17_regional_audit_path_repair.py',
                  'tests/test_round17_regional_audit_path_repair.py',
                  'docs/round17_regional_audit_path_repair.md')
DEFAULT_FAILURE = ROOT / 'results/round17_regional_metadata_precheck_failure.json'
DEFAULT_LOGS = (ROOT / 'results/round17_regional_independent_audit.log',
                ROOT / 'results/round17_regional_pipeline.log')


def verify_historical_snapshots(directory, prior):
    """Apply the exact three-file R12 freeze contract, without fabricating files."""
    directory = Path(directory)
    H.verify_files(ROOT, prior['source_hashes'])
    sources = prior['source_hashes']
    if not set(HISTORICAL_SNAPSHOTS) <= set(sources):
        raise ValueError('original R12 three-file snapshot contract is incomplete')
    snapshot = directory / 'source_snapshot'
    for name in HISTORICAL_SNAPSHOTS:
        path = snapshot / Path(name).name
        if not path.is_file() or sha(path) != sources[name]:
            raise ValueError('mandatory historical snapshot missing or changed: ' + name)
    by_name = {}
    for name, digest in sources.items():
        by_name.setdefault(Path(name).name, []).append((name, digest))
    existing = {}
    for path in sorted(snapshot.rglob('*')):
        if not path.is_file():
            continue
        choices = by_name.get(path.name, [])
        if path.parent != snapshot or len(choices) != 1:
            raise ValueError('unbound or ambiguous historical flat snapshot: ' + str(path))
        name, digest = choices[0]
        if sha(path) != digest:
            raise ValueError('existing historical snapshot changed: ' + name)
        existing[name] = dict(path=str(path.resolve()), sha256=digest)
    return dict(required=list(HISTORICAL_SNAPSHOTS), existing=existing,
                historical_not_snapshotted=sorted(set(sources) - set(existing)),
                all_original_source_hashes_verified=True, fabricated_snapshots=False)


def verify_legacy(lock, old_summary):
    legacy = lock['legacy']; directory = absolute(legacy['root'])
    prior = read(directory / 'lock.json')
    if sha(directory / 'lock.json') != (directory / 'lock.sha256').read_text().strip():
        raise ValueError('legacy lock checksum differs')
    H.verify_files(ROOT, prior['source_hashes'])
    verify_historical_snapshots(directory, prior)
    if sha(directory / 'lock.json') != legacy['lock_sha256'] or legacy['upstream_files'] != prior['inherited']['files']:
        raise ValueError('legacy dependency chain changed')
    H.verify_files(ROOT, legacy['upstream_files'])
    family = dict(point='gaussian_point', fixed='gaussian_fixed', full='gaussian_full1',
                  full4='gaussian_full4', quad4='gaussian_quad4', quad16='gaussian_quad16')
    expected = {}
    for role in ('point', 'point_pck', 'fixed', 'full1', 'full4', 'quad4', 'quad16'):
        original = prior['configurations'][role]
        config = {n: original[n] for n in ('h', 'temperature')}
        expected[role] = dict(family=family[original['method']], grid_index=GRID.index(config), config=config)
    if (legacy['configurations'] != expected
            or legacy['uncalibrated_legacy_iso4'] != prior['configurations']['iso4']
            or expected['point']['config'] != dict(h=1., temperature=5.)
            or expected['quad16']['config'] != dict(h=.5, temperature=5.)):
        raise ValueError('legacy reference configurations changed')
    return {role: dict(**spec, summary=old_summary['metrics'][spec['family']][spec['grid_index']])
            for role, spec in expected.items()}


def original_contract(run, failure, logs):
    """Read only lock/failure metadata. Hash result files without decoding them."""
    run, failure = Path(run).resolve(), Path(failure).resolve()
    lock = H.checked_lock(run)
    digest = sha(run / 'lock.json')
    failed = read(failure)
    if (failed['frozen_lock_sha256'] != digest
            or failed['affected_function'] != 'experiments.round17_regional_audit.verify_legacy'
            or failed['observed_error'] != 'FileNotFoundError: results/round12_stress/source_snapshot/ROUND10_PROTOCOL.md'
            or failed['effects_inspected_by_root'] is not False or failed['original_source_unchanged'] is not True
            or failed['producer_started_before_precheck_failure_was_inspected'] is not True
            or failed['recorded_unix'] < lock['created_unix']):
        raise ValueError('unexpected original metadata failure/source chronology')
    required = [run / name for name in ('lock.json', 'lock.sha256', 'forward_complete.json',
                                        'summary.json', 'completion.json')]
    required += [failure, *(Path(log).resolve() for log in logs)]
    if len(logs) != 2 or len(set(map(str, required))) != len(required):
        raise ValueError('both original audit and pipeline logs must be bound')
    bound = {str(path): sha(path) for path in required}
    legacy_root = absolute(lock['legacy']['root'])
    legacy = read(legacy_root / 'lock.json')
    if (sha(legacy_root / 'lock.json') != lock['legacy']['lock_sha256']
            or (legacy_root / 'lock.sha256').read_text().strip() != lock['legacy']['lock_sha256']):
        raise ValueError('original legacy lock changed')
    historical = verify_historical_snapshots(legacy_root, legacy)
    # Some outer launchers may retain failure artifacts even though the original
    # auditor failed before creating its output directory. Bind only real files.
    original_out = run.with_name('round17_regional_independent_audit')
    optional = {str(original_out / name): sha(original_out / name)
                for name in ('failure.json', 'completion.json') if (original_out / name).is_file()}
    return dict(run=str(run), joint_lock_sha256=digest, original_design_created_unix=lock['created_unix'],
        original_source_hashes=lock['source_hashes'], bound_files=bound, optional_original_failure_files=optional,
        metadata_failure_path=str(failure), metadata_failure_recorded_unix=failed['recorded_unix'],
        metadata_failure_stage_label=failed['stage'], evaluation_exists_at_failure_recording=failed['evaluation_exists_at_recording'],
        historical_snapshot_contract=historical, original_audit_directory_exists=original_out.exists(),
        new_regional_summary_decoded=False, new_regional_arrays_decoded=False, new_regional_labels_decoded=False,
        original_audit_failure_before_new_regional_decoding=True)


def freeze(out, run, failure=DEFAULT_FAILURE, logs=DEFAULT_LOGS, *, authors_have_not_read_regional_scores=False):
    """Freeze this engineering repair honestly after the original execution."""
    out = Path(out).resolve()
    if out.exists():
        raise ValueError('refusing existing repair namespace')
    if authors_have_not_read_regional_scores is not True:
        raise ValueError('explicit repair-author no-regional-effect-access attestation required')
    contract = original_contract(run, failure, logs)
    sources = {name: sha(ROOT / name) for name in REPAIR_SOURCES}
    now = time.time()
    if now < contract['metadata_failure_recorded_unix']:
        raise ValueError('repair must not be backdated')
    out.mkdir(parents=True)
    for name in sources:
        path = out / 'source_snapshot' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    value = dict(created_unix=now, source_hashes=sources, original_contract=contract,
        scope='only R12 historical snapshot-presence policy inside verify_legacy',
        frozen_before_regional_forward=False, original_pre_forward_scientific_design_preserved=True,
        authors_have_read_new_regional_effects=False, author_attestation_not_a_machine_access_proof=True,
        new_regional_summary_decoded_at_freeze=False, new_regional_labels_decoded_at_freeze=False,
        scientific_checks_changed=False, scientific_gate_passed=False,
        audit_output=str(out / 'audit'), original_auditor_source='experiments/round17_regional_audit.py')
    write(out / 'lock.json', value)
    (out / 'lock.sha256').write_text(sha(out / 'lock.json') + '\n')
    print('R17_REGIONAL_SNAPSHOT_REPAIR_FROZEN', sha(out / 'lock.json'), flush=True)
    return value


def verify_repair(out):
    out = Path(out).resolve(); lock = read(out / 'lock.json')
    if sha(out / 'lock.json') != (out / 'lock.sha256').read_text().strip():
        raise ValueError('repair lock checksum changed')
    if set(lock['source_hashes']) != set(REPAIR_SOURCES):
        raise ValueError('repair source population changed')
    H.verify_files(ROOT, lock['source_hashes']); H.verify_files(out / 'source_snapshot', lock['source_hashes'])
    before = lock['original_contract']
    # The bound log list is explicit and cannot be replaced by a newer run.
    files = before['bound_files']; run = Path(before['run'])
    excluded = {str(run / name) for name in ('lock.json', 'lock.sha256', 'forward_complete.json', 'summary.json', 'completion.json')}
    excluded.add(before['metadata_failure_path'])
    logs = [path for path in files if path not in excluded]
    current = original_contract(run, before['metadata_failure_path'], logs)
    if current != before:
        raise ValueError('repair original metadata/artifact chain changed')
    if (lock['created_unix'] < before['metadata_failure_recorded_unix']
            or lock['frozen_before_regional_forward'] is not False
            or lock['authors_have_read_new_regional_effects'] is not False
            or lock['scientific_checks_changed'] is not False
            or lock['original_pre_forward_scientific_design_preserved'] is not True
            or lock['new_regional_summary_decoded_at_freeze'] is not False
            or lock['new_regional_labels_decoded_at_freeze'] is not False
            or lock['audit_output'] != str(out / 'audit')):
        raise ValueError('repair chronology/scope changed')
    return lock


@contextmanager
def repaired_auditor(out, lock):
    """Only replace the known check; add transparent output provenance fields."""
    old_check, old_write = original.verify_legacy, original.write
    nested = Path(lock['audit_output'])
    receipt = dict(repair_lock_sha256=sha(Path(out) / 'lock.json'), repair_frozen_unix=lock['created_unix'],
        repair_source_hashes=lock['source_hashes'], original_joint_lock_sha256=lock['original_contract']['joint_lock_sha256'],
        path_repair_scope=lock['scope'], path_repair_frozen_before_regional_forward=False,
        original_failure_bound_files=lock['original_contract']['bound_files'])
    def metadata_write(path, value):
        path = Path(path)
        if path.parent == nested and path.name in ('lock.json', 'verification.json'):
            value = dict(value, path_repair=receipt)
        return old_write(path, value)
    original.verify_legacy, original.write = verify_legacy, metadata_write
    try:
        yield
    finally:
        original.verify_legacy, original.write = old_check, old_write


def audit(out):
    out = Path(out).resolve(); lock = verify_repair(out)
    if (out / 'audit_started.json').exists() or (out / 'completion.json').exists() or Path(lock['audit_output']).exists():
        raise ValueError('refusing repair audit overwrite')
    started = time.time()
    if started < lock['created_unix']:
        raise ValueError('repair audit cannot precede its freeze')
    write(out / 'audit_started.json', dict(started_unix=started, repair_lock_sha256=sha(out / 'lock.json')))
    try:
        with repaired_auditor(out, lock):
            result = original.audit(Path(lock['original_contract']['run']), Path(lock['audit_output']))
    except Exception as error:
        write(out / 'failure.json', dict(passed=False, scientific_gate_passed=False, error_type=type(error).__name__,
            error=str(error), started_unix=started, failed_unix=time.time(), repair_lock_sha256=sha(out / 'lock.json')))
        write(out / 'completion.json', dict(passed=False, scientific_gate_passed=False,
            files={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}, finished_unix=time.time()))
        raise
    write(out / 'completion.json', dict(passed=result['passed'], scientific_gate_passed=False,
        repair_lock_sha256=sha(out / 'lock.json'), verification_path=str(Path(lock['audit_output']) / 'verification.json'),
        verification_sha256=sha(Path(lock['audit_output']) / 'verification.json'),
        files={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}, finished_unix=time.time()))
    return read(Path(lock['audit_output']) / 'verification.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('freeze', 'audit'))
    parser.add_argument('--run', type=Path, default=ROOT / 'results/round17_regional')
    parser.add_argument('--out', type=Path, default=ROOT / 'results/round17_regional_audit_path_repair')
    parser.add_argument('--failure', type=Path, default=DEFAULT_FAILURE)
    parser.add_argument('--logs', nargs=2, type=Path, default=DEFAULT_LOGS)
    parser.add_argument('--attest-no-regional-scores-read', action='store_true')
    args = parser.parse_args()
    if args.stage == 'freeze':
        freeze(args.out, args.run, args.failure, args.logs,
               authors_have_not_read_regional_scores=args.attest_no_regional_scores_read)
    else:
        result = audit(args.out)
        print('R17_REGIONAL_REPAIR_AUDIT', 'PASS' if result['passed'] else 'FAIL', flush=True)
        if not result['passed']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
