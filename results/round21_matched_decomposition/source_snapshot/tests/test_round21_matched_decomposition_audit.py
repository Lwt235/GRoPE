"""R21 provenance/chronology checks on generated metadata, never dataset files."""
from copy import deepcopy
import hashlib
from pathlib import Path
import numpy as np
import pytest
from experiments import round21_matched_decomposition_audit as A


def metadata():
    entries = [dict(scene=s, pair=[i, i + 1], pair_name=f'{i}_{i + 1}', input_sha256=f'input-{i}')
               for s in A.SCENES for i in range(16)]
    records, supports = [], []
    for entry in entries:
        stem = 'forward/' + entry['scene'].replace('/', '__') + '/arrays/' + entry['pair_name']
        supports.append(dict(entry, archive_path=stem + '/supports.npz'))
        for c in A.CONDITIONS:
            for family in A.FAMILIES:
                records.append(dict(entry, **c, family=family, config=A.role_configs()[family],
                    shape=[A.COUNTS[c['query']], 432], dtype='float64', archive_path=stem + '/' + c['condition'] + '.npz',
                    raw_mass_field=A.role_configs()[family]['geometry'] + '__raw_mass'))
    return entries, records, supports


def install_seal_fixture(monkeypatch, tmp_path):
    run = tmp_path / 'producer'; entries, records, supports = metadata()
    digest = lambda p: str(Path(p))
    seal = dict(producer_lock_sha256=digest(run / 'lock.json'), forward_completion_sha256=digest(run / 'forward/completion.json'),
        forward_records_sha256=digest(run / 'forward/records.json'), support_records_sha256=digest(run / 'forward/support_records.json'),
        pairs=48, matrices=1152, raw_matrices=576, raw_probability_bytes=A.RAW_BYTES, raw_mass_bytes=A.RAW_MASS_BYTES,
        all_probabilities_retained=True, all_raw_masses_retained=True, actual_target_hashes_read=False, labels_decoded=False,
        created_unix=5.)
    data = {run / 'forward_complete.json': seal, run / 'forward/records.json': records,
            run / 'forward/support_records.json': supports}
    completions = {run / 'seal': dict(forward_complete_sha256=digest(run / 'forward_complete.json'), all_raw_masses_retained=True)}
    expected = {'records.json', 'support_records.json'}
    for scene in A.SCENES:
        folder = run / 'forward' / scene.replace('/', '__')
        rr = [r for r in records if r['scene'] == scene]; ss = [r for r in supports if r['scene'] == scene]
        data[folder / 'records.json'] = rr; data[folder / 'support_records.json'] = ss
        data[folder / 'started.json'] = dict(created_unix=2., scene=scene, producer_lock_sha256=digest(run / 'lock.json'))
        names = {'started.json', 'records.json', 'support_records.json'}
        names.update(str((run / r['archive_path']).relative_to(folder)) for r in rr + ss)
        completions[folder] = dict(records=384, producer_lock_sha256=digest(run / 'lock.json'), scene=scene,
            labels_decoded=False, actual_target_hashes_read=False, started_unix=2., finished_unix=4., files=dict.fromkeys(names, 'synthetic'))
        expected.update(str((folder / n).relative_to(run / 'forward')) for n in names)
        expected.add(str((folder / 'completion.json').relative_to(run / 'forward')))
    completions[run / 'forward'] = dict(matrices=1152, labels_decoded=False, actual_target_hashes_read=False,
                                       all_raw_masses_retained=True, files=dict.fromkeys(expected, 'synthetic'))
    monkeypatch.setattr(A, 'read', lambda path: data[Path(path)])
    monkeypatch.setattr(A, 'sha', digest)
    monkeypatch.setattr(A, 'check_inventory', lambda path: completions[Path(path)])
    return run, dict(entries=entries), dict(created_unix=1.), data, completions


def test_complete_probability_raw_population_and_native_records():
    entries, records, _ = metadata(); A.validate_records(records, entries)
    assert len(records) == 1152 and len(A.role_configs()) == 8
    assert sum(A.COUNTS[c['query']] * 432 * 8 * 48 * 8 for c in A.CONDITIONS) == A.RAW_BYTES
    assert A.RAW_MASS_BYTES * 2 == A.RAW_BYTES


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'dtype', 'config', 'raw_field', 'shape'])
def test_changed_probability_population_rejected(change):
    entries, records, _ = metadata()
    if change == 'missing': records.pop()
    elif change == 'duplicate': records[-1] = deepcopy(records[0])
    elif change == 'dtype': records[0]['dtype'] = 'float32'
    elif change == 'config': records[0]['config']['temperature'] = 0.
    elif change == 'raw_field': records[0]['raw_mass_field'] = 'other__raw_mass'
    else: records[0]['shape'][0] -= 1
    with pytest.raises(ValueError): A.validate_records(records, entries)


def test_global_seal_requires_exact_inventory_and_full_raw_retention(monkeypatch, tmp_path):
    run, lock, design, _, _ = install_seal_fixture(monkeypatch, tmp_path)
    records, supports, seal = A.check_seal(run, lock, design)
    assert len(records) == 1152 and len(supports) == 48 and seal['raw_matrices'] == 576


@pytest.mark.parametrize('where', ['seal', 'forward', 'seal_completion'])
def test_missing_raw_retention_flag_rejected_at_every_global_commitment(monkeypatch, tmp_path, where):
    run, lock, design, data, completions = install_seal_fixture(monkeypatch, tmp_path)
    obj = data[run / 'forward_complete.json'] if where == 'seal' else completions[run / ('forward' if where == 'forward' else 'seal')]
    obj.pop('all_raw_masses_retained')
    with pytest.raises(ValueError): A.check_seal(run, lock, design)


@pytest.mark.parametrize('kind', ['extra_inventory', 'chronology', 'support_order'])
def test_unsealed_or_postlabel_order_changes_rejected(monkeypatch, tmp_path, kind):
    run, lock, design, data, completions = install_seal_fixture(monkeypatch, tmp_path)
    if kind == 'extra_inventory': completions[run / 'forward']['files']['extra.npz'] = 'unexpected'
    elif kind == 'chronology': completions[run / 'forward' / A.SCENES[0].replace('/', '__')]['started_unix'] = .5
    else: data[run / 'forward/support_records.json'].reverse()
    with pytest.raises(ValueError): A.check_seal(run, lock, design)


def test_both_historical_hash_algorithms_and_retained_array_contract(tmp_path):
    value = np.array([[.25, .75]], np.float64)
    manual = hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()).hexdigest()
    assert A.legacy_array_sha(value) == manual and manual != A.array_sha(value)
    for kind, digest in [('r15_commitment', manual), ('r20_saved', A.array_sha(value))]:
        source = dict(replay_kind=kind, array_sha256=digest, source_archive_path=None)
        assert A.inherited_probability(value, source)['commitment_exact']
    archive = tmp_path / 'synthetic.npz'; np.savez(archive, p=value)
    source = dict(replay_kind='r20_saved', array_sha256=A.array_sha(value), source_archive_path=str(archive),
                  source_archive_sha256=A.sha(archive), source_field='p')
    assert A.inherited_probability(value, source)['saved_array_checked']
    with pytest.raises(ValueError): A.inherited_probability(value[:, ::-1], source)


def test_joint_freeze_must_precede_any_forward(tmp_path):
    run = tmp_path / 'producer'; (run / 'forward').mkdir(parents=True)
    with pytest.raises(ValueError, match='precede every real forward'): A.freeze(run, tmp_path / 'audit')


def test_failed_final_guard_reads_no_arrays_and_preserves_attempt(monkeypatch, tmp_path):
    out = tmp_path / 'audit'; run = tmp_path / 'producer'
    def reject(*args): raise ValueError('numeric audit failed')
    monkeypatch.setattr(A, 'require_final_guards', reject)
    def forbidden(*args, **kwargs): raise AssertionError('target or prediction array opened')
    monkeypatch.setattr(A.np, 'load', forbidden)
    with pytest.raises(ValueError, match='numeric audit failed'): A.audit(run, out)
    failure = (out / 'final_audit/failure.json').read_bytes()
    assert A.read(out / 'final_audit/completion.json')['passed'] is False
    with pytest.raises(ValueError, match='preserve previous'): A.audit(run, out)
    assert failure == (out / 'final_audit/failure.json').read_bytes()


def test_source_closure_includes_ast_extracted_history_and_matches_producer():
    from experiments import round21_matched_decomposition as P
    assert 'experiments/round13_mixture_mechanism.py' in A.SOURCE_ROOTS
    assert A.source_closure() == P.source_names()
    assert A.NUMERICAL_POLICY == P.NUMERICAL_POLICY and A.N.REFERENCE_POLICY == P.NUMREF.REFERENCE_POLICY
