"""R29 descriptor and delivery toys; no formal fixture integration or real NPZ."""
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import pytest

from experiments import round29_log_rms_fixtures as F
from experiments import round29_log_rms_preflight as P
from experiments import round27_rms_fixtures as OLD


def test_fixed_descriptor_population_and_bias_repeats():
    rows, base, identity = P.population()
    assert len(rows) == 29 and base == list(F.BASE_IDS) and identity == list(F.IDENTITY_IDS)
    assert json.loads(json.dumps(rows, allow_nan=False)) == rows
    assert all(r['content_temperature'] is None for r in rows)
    for new, old in zip(rows[14:18], (rows[1], rows[5], rows[8], rows[11])):
        assert all(new[k] == old[k] for k in ('mu', 'factor', 'bounds'))
        assert old['bias_kind'] == 'zero' and new['bias_kind'] == 'fixed_native'
        assert max(abs(x) for x in new['content_bias']) <= .6


def test_all_old26_descriptors_preserve_exact_serialization_and_array_bytes():
    old = OLD.fixtures(); new = F.fixtures(); old_ids = {r['id'] for r in old}
    retained = [r for r in new if r['id'] in old_ids]
    assert json.dumps(retained, sort_keys=True, allow_nan=False).encode() == json.dumps(
        old, sort_keys=True, allow_nan=False).encode()
    for before, after in zip(old, retained):
        for name in ('mu', 'factor', 'bounds', 'content_bias'):
            a, b = np.asarray(before[name], np.float64), np.asarray(after[name], np.float64)
            assert a.shape == b.shape and a.tobytes() == b.tobytes()
    assert [r['id'] for r in new[20:23]] == list(F.ANALYTIC_BASE_IDS)
    assert [r['id'] for r in new[23:]] == list(OLD.IDENTITY_IDS)
    native, tiny, visible = new[20:23]
    assert native['mu'] == [12., 9., 1.] and native['bounds'] == new[12]['bounds']
    assert tiny['mu'] == visible['mu'] == [0., 0., 1.]
    assert tiny['bounds'] == F.rectangles([9.75, 10., 10.25], [-.25, 0., .25]).tolist()
    assert visible['bounds'] == [[0., -.25, 12., .25]]
    for row in (native, tiny, visible):
        assert row['factor'] == np.zeros((3, 3)).tolist()
        assert row['content_bias'] == [0.]*len(row['bounds'])
        assert row['policy_overrides'] == {} and row['content_temperature'] is None
    new[0]['mu'][0] = 123.
    assert OLD.fixtures() == old


def test_structural_zero_and_near_constant_descriptors_remain_distinct():
    rows = F.make_cases()
    assert rows[3]['mu'][2] == 0 and rows[3]['factor'][2] == [0., 0., 0.]
    assert rows[4]['mu'][2] == -1 and rows[4]['factor'][2] == [0., 0., 0.]
    assert rows[3]['bias_kind'] == 'fixed_native' and rows[4]['bias_kind'] == 'zero'
    assert rows[10]['factor'][2][2] == 2.**-36 > 0
    assert rows[10]['mu'] == rows[2]['mu'] and rows[10]['factor'][:2] == rows[2]['factor'][:2]
    assert rows[11]['mu'] == [0., 0., -12.] and rows[11]['factor'][2][2] == 1.
    b = np.asarray(rows[3]['bounds'])
    np.testing.assert_array_equal(np.prod(b[:, 2:]-b[:, :2], axis=1), [4.5, 7.5, 1.5, 2.5])


def test_native432_half_pixel_units_and_order():
    native = np.asarray(F.make_cases()[12]['bounds'])
    expected = np.asarray([[x, y, x+8, y+8] for y in range(0, 144, 8)
                           for x in range(0, 192, 8)], np.float64)
    np.testing.assert_array_equal(native, (expected-.5)/8.)
    np.testing.assert_array_equal(native[0], [-.0625, -.0625, .9375, .9375])
    assert native.shape == (432, 4) and np.prod(native[:, 2:]-native[:, :2], axis=1).sum() == 432.


def test_identity_transforms_carry_bias_and_do_not_mutate_bases():
    base = F.make_cases(); before = deepcopy(base); identities = F.make_identities(base)
    assert base == before
    parent = base[16]
    for row in identities[:4]:
        assert row['content_bias'] == parent['content_bias']
        assert row['identity'] == dict(parent=parent['id'], relation='equal')
    np.testing.assert_array_equal(identities[0]['factor'], np.asarray(parent['factor'])*7.)
    t = np.eye(3); t[:2, 2] = [.75, -.5]
    np.testing.assert_array_equal(identities[1]['factor'], t@np.asarray(parent['factor']))
    assert identities[1]['content_bias'] != F.native_bias(np.asarray(identities[1]['bounds'])).tolist()
    assert identities[2]['policy_overrides'] == dict(h=.25*3.7, floor_density=1e-30/3.7**2)
    b = np.asarray(parent['bounds']); rotated = np.asarray(identities[3]['bounds'])
    np.testing.assert_array_equal(rotated, np.column_stack((-b[:, 3], b[:, 0], -b[:, 1], b[:, 2])))


def test_permutation_and_unequal_merge_declarations():
    base = F.make_cases(); identity = F.make_identities(base)
    perm = identity[4]['identity']['permutation']
    assert perm == list(range(431, -1, -1))
    assert identity[4]['bounds'] == [base[12]['bounds'][i] for i in perm]
    groups = identity[5]['identity']['groups']
    assert sorted(i for group in groups for i in group) == list(range(16))
    b = np.asarray(identity[5]['bounds'])
    np.testing.assert_array_equal(np.prod(b[:, 2:]-b[:, :2], axis=1), [3., 1.]*4)
    assert identity[5]['content_bias'] == [0.]*8
    with pytest.raises(ValueError): F.make_identities(list(reversed(base)))


def toy():
    # Deliberately different from every declared R29 formal geometry/bounds.
    return dict(id='unit_delivery', kind='base', mu=[.17, -.11, 1.7],
        factor=[[.31, 0., 0.], [.08, .27, 0.], [.04, -.03, .2]],
        bounds=[[-.8, -.6, .1, .7], [.1, -.6, .9, .7]], content_bias=[.13, -.09],
        content_temperature=None, bias_kind='unit_only', policy_overrides={})


def directory(tmp_path):
    (tmp_path/'arrays').mkdir()
    return tmp_path


def test_successful_nonformal_toy_retains_inputs_probability_and_first_winner(tmp_path):
    out = directory(tmp_path)
    r = P.run_case(out, toy(), P.C.POLICY, time.perf_counter()+30.)
    assert r['passed'] and r['attempted'] and r['status'] == 'complete'
    with np.load(out/r['archive_path'], allow_pickle=False) as bank:
        assert set(bank.files) == {'input__mu', 'input__factor', 'input__bounds', 'input__content_bias',
                                   'log_raw_mass', 'log_raw_fov_mass', 'log_floored_mass', 'log_probability',
                                   'probability', 'materialized_raw_indices', 'materialized_raw_mass', 'argmax'}
        assert bank['argmax'].dtype == np.int32 and bank['argmax'].shape == ()
        assert int(bank['argmax']) == int(bank['probability'].argmax())
        assert abs(bank['probability'].sum()-1.) <= 2e-12
        assert np.isfinite(bank['log_raw_mass']).all()
        assert bank['materialized_raw_indices'].dtype == np.int64
        assert np.all(bank['materialized_raw_mass'] > 0)
    assert json.loads((out/'model_results.jsonl').read_text())['work'] == r['work']
    assert json.loads((out/'call_starts.jsonl').read_text())['id'] == 'unit_delivery'


def test_structural_zero_logs_archive_without_nonfinite_json(tmp_path):
    out = directory(tmp_path); case = toy()
    case.update(mu=[.17, -.11, -1.7], factor=np.zeros((3, 3)).tolist())
    row = P.run_case(out, case, P.C.POLICY, time.perf_counter()+30.)
    assert row['passed'] and row['structural_zero']
    assert row['log_raw_mass_sum'] is None
    json.dumps(row, allow_nan=False)
    with np.load(out/row['archive_path'], allow_pickle=False) as bank:
        assert np.isneginf(bank['log_raw_mass']).all()
        assert np.isneginf(bank['log_raw_fov_mass']).all()
        assert np.isfinite(bank['log_probability']).all()
        assert np.isfinite(bank['probability']).all()
        assert 'raw_mass' not in bank.files


def test_positive_unmaterializable_log_is_retained_without_fabricated_raw(tmp_path, monkeypatch):
    out = directory(tmp_path)
    monkeypatch.setattr(P.MODEL, 'integrate', lambda **kw: dict(status='complete', passed=True,
        structural_zero=False, work=dict(cell_node_evaluations=11),
        log_raw_mass=np.asarray([-1000., -1.]), log_raw_fov_mass=np.asarray([-1.]),
        log_floored_mass=np.asarray([-69., -1.]), log_probability=np.asarray([-68., 0.]),
        probability=np.asarray([np.exp(-68.), 1.]),
        materialized_raw_indices=np.asarray([1], np.int64), materialized_raw_mass=np.asarray([np.exp(-1.)])))
    row = P.run_case(out, toy(), P.C.POLICY, time.perf_counter()+30.)
    assert row['passed'] and not row['structural_zero']
    with np.load(out/row['archive_path'], allow_pickle=False) as bank:
        np.testing.assert_array_equal(bank['log_raw_mass'], [-1000., -1.])
        np.testing.assert_array_equal(bank['materialized_raw_indices'], [1])
        assert len(bank['materialized_raw_mass']) == 1 and 'raw_mass' not in bank.files


def test_validation_exception_retains_inputs_and_failure_without_P(tmp_path):
    out = directory(tmp_path); case = toy(); case['factor'] = [[1., 0.], [0., 1.]]
    r = P.run_case(out, case, P.C.POLICY, time.perf_counter()+30.)
    assert not r['passed'] and r['attempted'] and 'ValueError' in r['failure_reason']
    with np.load(out/r['archive_path'], allow_pickle=False) as bank:
        assert set(bank.files) == {'input__mu', 'input__factor', 'input__bounds', 'input__content_bias'}


def test_partial_numerical_failure_preserves_work_and_arrays(tmp_path, monkeypatch):
    out = directory(tmp_path)
    monkeypatch.setattr(P.MODEL, 'integrate', lambda **kwargs: dict(status='failed', passed=False,
        failure_reason='unit budget failure', work=dict(cell_node_evaluations=12),
        partial_log_raw_mass=np.asarray([-2., -np.inf]), partial_log_raw_fov_mass=np.asarray([-np.inf]),
        partial_completed_cells=np.asarray([True, False]), partial_completed_fov=np.asarray([False]),
        partial_accumulated_tiles=np.asarray([2, 0], np.int64),
        partial_accumulated_fov_tiles=np.asarray([0], np.int64)))
    r = P.run_case(out, toy(), P.C.POLICY, time.perf_counter()+30.)
    assert r['work'] == dict(cell_node_evaluations=12) and not r['passed']
    with np.load(out/r['archive_path'], allow_pickle=False) as bank:
        np.testing.assert_array_equal(bank['partial_log_raw_mass'], [-2., -np.inf])
        np.testing.assert_array_equal(bank['partial_completed_cells'], [True, False])
        assert bank['partial_completed_cells'].dtype == np.bool_
        np.testing.assert_array_equal(bank['partial_accumulated_tiles'], [2, 0])
        assert bank['partial_accumulated_tiles'].dtype == np.int64
        assert 'probability' not in bank.files


def test_expired_call_records_no_attempt_or_model_work(tmp_path, monkeypatch):
    out = directory(tmp_path)
    monkeypatch.setattr(P.MODEL, 'integrate', lambda **kw: pytest.fail('postdeadline integration'))
    r = P.run_case(out, toy(), P.C.POLICY, time.perf_counter()-1.)
    assert not r['attempted'] and not r['passed'] and r['work'] == {}
    assert not (out/'call_starts.jsonl').exists()
    with np.load(out/r['archive_path'], allow_pickle=False) as bank:
        assert len(bank.files) == 4 and all(k.startswith('input__') for k in bank.files)


def test_archive_failure_cannot_erase_returned_work_journal(tmp_path, monkeypatch):
    out = directory(tmp_path)
    monkeypatch.setattr(P.MODEL, 'integrate', lambda **kw: dict(status='failed', passed=False,
        failure_reason='unit budget failure', work=dict(cell_node_evaluations=13),
        partial_log_raw_mass=np.asarray([-2., -np.inf]),
        partial_completed_cells=np.asarray([True, False])))
    monkeypatch.setattr(P.np, 'savez_compressed', lambda *a, **kw: (_ for _ in ()).throw(OSError('unit save failure')))
    with pytest.raises(OSError): P.run_case(out, toy(), P.C.POLICY, time.perf_counter()+30.)
    journal = json.loads((out/'model_results.jsonl').read_text())
    assert journal['work']['cell_node_evaluations'] == 13
    assert {'partial_log_raw_mass', 'partial_completed_cells'} <= set(journal['available_array_sha256'])


def fake_lock(tmp_path, monkeypatch, seconds):
    cases = [toy(), dict(toy(), id='unit_delivery_second')]
    lock = dict(policy=P.C.POLICY, gates=dict(production_max_phase_wall_seconds=seconds),
        base_ids=[r['id'] for r in cases], identity_ids=[], fixtures_sha256='unit')
    P.write(tmp_path/'lock.json', lock); P.write(tmp_path/'fixtures.json', cases)
    monkeypatch.setattr(P, 'verify', lambda _: lock)
    monkeypatch.setattr(P, 'require_audit_freeze', lambda *_: 'unit_audit_lock')
    return lock


def test_all_expired_population_and_empty_start_journal_sealed(tmp_path, monkeypatch):
    fake_lock(tmp_path, monkeypatch, -1.)
    monkeypatch.setattr(P.MODEL, 'integrate', lambda **kw: pytest.fail('expired phase integration'))
    P.run(tmp_path)
    rows = P.read(tmp_path/'records.json'); seal = P.read(tmp_path/'forward_complete.json')
    assert len(rows) == 2 and all(not r['attempted'] for r in rows)
    assert seal['attempted_calls'] == 0 and not seal['phase_wall_passed']
    assert (tmp_path/'call_starts.jsonl').read_text() == ''
    assert len((tmp_path/'model_results.jsonl').read_text().splitlines()) == 2
    P.verify_inventory(tmp_path, P.read(tmp_path/'completion.json')['files'], exclude=('completion.json',))
    original = P.sha(tmp_path/'completion.json')
    with pytest.raises(ValueError, match='overwrite'): P.run(tmp_path)
    assert P.sha(tmp_path/'completion.json') == original


@pytest.mark.parametrize('error', [OSError('unit interruption'), KeyboardInterrupt()])
def test_outer_failures_seal_inventory_and_reraise(tmp_path, monkeypatch, error):
    fake_lock(tmp_path, monkeypatch, 30.)
    monkeypatch.setattr(P, 'run_case', lambda *a: (_ for _ in ()).throw(error))
    with pytest.raises(type(error)): P.run(tmp_path)
    completion = P.read(tmp_path/'completion.json')
    assert not completion['complete_case_inventory'] and not completion['phase_wall_passed']
    assert P.read(tmp_path/'failure.json')['error_type'] == type(error).__name__
    P.verify_inventory(tmp_path, completion['files'], exclude=('completion.json',))


def test_freeze_and_verify_are_metadata_only_and_detect_fixture_change(tmp_path, monkeypatch):
    source_root = tmp_path/'sources'; source_root.mkdir(); (source_root/'unit.py').write_text('x=1\n')
    monkeypatch.setattr(P, 'ROOT', source_root)
    monkeypatch.setattr(P, 'source_names', lambda: ['unit.py'])
    monkeypatch.setattr(P, 'inherited_bindings', lambda: {'unit_metadata': 'digest'})
    monkeypatch.setattr(P, 'check_runtime', lambda: {'unit_runtime': True})
    monkeypatch.setattr(P.np, 'load', lambda *a, **kw: pytest.fail('freeze decoded archive'))
    monkeypatch.setattr(P.MODEL, 'integrate', lambda **kw: pytest.fail('freeze integrated formal case'))
    out = tmp_path/'producer'; P.freeze(out, tmp_path/'audit'); lock = P.verify(out)
    assert len(lock['base_ids']) == 23 and len(lock['identity_ids']) == 6
    with pytest.raises(ValueError, match='existing'): P.freeze(out, tmp_path/'audit')
    (out/'fixtures.json').write_text('[]\n')
    with pytest.raises(ValueError, match='fixtures'): P.verify(out)


def test_shared_metadata_bindings_validate_receipts_snapshots_and_selection(tmp_path, monkeypatch):
    design = tmp_path/'design.md'; design.write_text('unit prospective design\n')
    model = tmp_path/'model.py'; model.write_text('unit_only = True\n')
    evidence = tmp_path/'evidence.json'; P.write(evidence, {'unit': True})
    previous, synthetic = tmp_path/'previous.json', tmp_path/'synthetic.json'
    common = dict(files={'evidence.json': P.sha(evidence)}, source_hashes={'model.py': P.sha(model)},
                  scientific_gate_passed=False)
    P.write(previous, dict(common, accepted_for_applicability_interpretation=True,
        applicability_calls_completed=False, fixed_query_count=72, model_completed_calls=40, labels_read=False))
    P.write(synthetic, dict(common, accepted_for_numerical_interpretation=True, numerical_preflight_passed=True))
    for name in ('round28_rms_applicability', 'round28_rms_applicability_independent_audit',
                 'round27_rms_preflight', 'round27_rms_independent_audit'):
        folder = tmp_path/'results'/name/'source_snapshot'; folder.mkdir(parents=True)
        (folder/'model.py').write_bytes(model.read_bytes())
    decision = tmp_path/'decision.json'
    P.write(decision, dict(root_acceptance_sha256=P.sha(previous), next_selected_work=dict(round='R29',
        design_path='design.md', design_sha256=P.sha(design), real_input_arrays_authorized=False,
        labels_authorized=False, efficacy_candidate_selected=False, relax_existing_gates=False)))
    monkeypatch.setattr(P.C, 'ROOT', tmp_path)
    monkeypatch.setattr(P.C, 'BINDINGS', ('previous.json', 'decision.json', 'design.md', 'synthetic.json'))
    monkeypatch.setattr(P.np, 'load', lambda *a, **kw: pytest.fail('metadata decoded archive'))
    monkeypatch.setattr(P.MODEL, 'integrate', lambda **kw: pytest.fail('metadata ran model'))
    bindings = P.inherited_bindings()
    assert bindings[str(design)] == P.sha(design) and bindings[str(model)] == P.sha(model)
    snapshot = tmp_path/'results/round27_rms_preflight/source_snapshot/model.py'
    snapshot.write_text('changed unit snapshot\n')
    with pytest.raises(ValueError, match='snapshot'): P.inherited_bindings()
    snapshot.write_bytes(model.read_bytes())
    design.write_text('changed prospective design\n')
    with pytest.raises(ValueError, match='selection'): P.inherited_bindings()


def test_audit_freeze_must_bind_producer_and_all_snapshots(tmp_path):
    producer, audit = tmp_path/'producer', tmp_path/'audit'; producer.mkdir(); audit.mkdir()
    P.write(producer/'lock.json', {'unit': True})
    (audit/'source_snapshot').mkdir(); source = audit/'source_snapshot/unit.py'; source.write_text('unit\n')
    lock = dict(created_unix=1., audit_root=str(audit), fixtures_sha256='unit_fixture',
                source_hashes={'unit.py': P.sha(source)}, runtime={'unit': True})
    a = dict(producer_lock_sha256=P.sha(producer/'lock.json'), fixtures_sha256='unit_fixture',
        source_hashes=lock['source_hashes'], runtime=lock['runtime'], created_unix=2., **P.SCOPE)
    P.write(audit/'lock.json', a); (audit/'lock.sha256').write_text(P.sha(audit/'lock.json')+'\n')
    assert P.require_audit_freeze(producer, lock) == P.sha(audit/'lock.json')
    for field in P.SCOPE:
        for mutation in ('missing', 'true'):
            changed = dict(a)
            if mutation == 'missing': changed.pop(field)
            else: changed[field] = True
            P.write(audit/'lock.json', changed)
            (audit/'lock.sha256').write_text(P.sha(audit/'lock.json')+'\n')
            with pytest.raises(ValueError, match='pre-forward freeze differs'):
                P.require_audit_freeze(producer, lock)
    P.write(audit/'lock.json', a); (audit/'lock.sha256').write_text(P.sha(audit/'lock.json')+'\n')
    source.write_text('changed\n')
    with pytest.raises(ValueError, match='snapshot'): P.require_audit_freeze(producer, lock)
