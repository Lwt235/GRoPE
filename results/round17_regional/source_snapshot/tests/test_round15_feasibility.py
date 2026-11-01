import json

import numpy as np
import pytest
from scipy.special import ndtr
import torch

from experiments import round15_feasibility as P


def synthetic_supports():
    mean = torch.tensor([[.2, .3], [1.4, .2]], dtype=torch.float32)
    covariance = torch.tensor([[[.12, .02], [.02, .2]], [[.25, -.04], [-.04, .15]]], dtype=torch.float32)
    features = torch.eye(2)
    full = dict(mean=mean, cov=covariance, mass=torch.full((2,), 64.), features=features)
    point = dict(full, cov=torch.zeros_like(covariance))
    native = np.array([[0, 0, 8, 8], [8, 0, 16, 8], [16, 0, 24, 8]], float)
    key = dict(cell_bounds=torch.tensor((native-.5)/8), cell_owner=torch.arange(3),
               mass=torch.full((3,), 64.), features=torch.tensor([[-1., 0.], [0., 1.], [1., 0.]]))
    return full, point, key


def test_complete_grid_native_probabilities_and_independent_gaussian_reference(monkeypatch):
    full, point, key = synthetic_supports()
    module = P.importlib.import_module(P.QUARTIC_MODULE)
    original, orders = module.quartic_cell_mass, []
    def observed(*args, **kwargs):
        orders.append(kwargs['quadrature_order'])
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'quartic_cell_mass', observed)
    with torch.no_grad():
        probabilities, diagnostics = P.condition_predictions(full, point, key)
    assert len(probabilities) == 36 and len(diagnostics) == 12
    assert orders == [24] * 8 and P.NUMERICAL_POLICY['quadrature_order'] == 24
    mu = point['mean'].double().numpy()
    boxes = key['cell_bounds'].numpy()
    for hi, h in enumerate((.25, .5, 1., 2.)):
        expected = np.prod(ndtr((boxes[None, :, 2:]-mu[:, None])/h)
                           - ndtr((boxes[None, :, :2]-mu[:, None])/h), axis=-1) + 1.e-30
        expected /= expected.sum(-1, keepdims=True)
        np.testing.assert_allclose(probabilities['gaussian_point', hi*3].numpy(), expected, atol=1.e-14, rtol=1.e-12)
    assert P.GRID[P.LEGACY_INDEX] == dict(h=.5, temperature=5.)
    assert P.RAW_PROBABILITY_BYTES == 3511517184


def test_point_limit_content_and_native_key_identity():
    full, point, key = synthetic_supports()
    with torch.no_grad():
        first, _ = P.condition_predictions(point, point, key)
        second, _ = P.condition_predictions(point, point, dict(key, features=-key['features']))
    for i, config in enumerate(P.GRID):
        torch.testing.assert_close(first['quartic_full', i], first['quartic_point', i], atol=0, rtol=0)
        if config['temperature'] == 0:
            torch.testing.assert_close(first['quartic_full', i], second['quartic_full', i], atol=0, rtol=0)
        else:
            assert not torch.allclose(first['quartic_full', i], second['quartic_full', i])
    with pytest.raises(ValueError, match='native g8'):
        P.condition_predictions(full, point, dict(key, cell_owner=torch.tensor([1, 0, 2])))


def test_point_preparation_does_not_compute_unused_covariance(monkeypatch):
    called = []
    original = P.prepare_support
    def observed(*args, **kwargs):
        called.append(kwargs['include_full_moments'])
        return original(*args, **kwargs)
    monkeypatch.setattr(P, 'prepare_support', observed)
    part = dict(owner=torch.tensor([0, 0, 1, 1]), bounds=torch.tensor([[0, 0, 2, 1], [2, 0, 4, 1]]),
                parent=torch.tensor([0, 0]), atoms=torch.tensor([0, 1]), mean=torch.zeros(2, 2))
    obs = dict(q_to_k_xy=torch.tensor([[0., 0.], [1., 0.], [2., 0.], [3., 0.]]),
               q_to_k_pixel_cov=torch.eye(2).expand(4, -1, -1), q_features=torch.eye(2))
    point = P.prepare_query(obs, part, False)
    assert called == [False] and torch.count_nonzero(point['cov']) == 0


def test_center_metric_first_argmax_boundary_and_invalid_centers():
    probability = np.array([[.5, .5], [.1, .9], [.8, .2]], np.float64)
    keys = np.array([[3.5, 3.5], [11.5, 3.5]])
    points = np.array([[11.5, 3.5], [3.49, 3.5], [np.nan, np.nan]])
    value = P.center_metric(probability, keys, points, np.array([True, True, False]))
    assert value['n_query'] == 3 and value['n_point'] == 2 and value['pck_count'] == 1
    assert value['argmax_sha256'] == P.array_sha(np.array([0, 1, 0]))
    with pytest.raises(ValueError, match='valid center'):
        P.center_metric(probability, keys, points, np.array([True, True, True]))


def fake_rows():
    # Only one fake pair per scene is enough to exercise selection/aggregation.
    entries = [dict(scene=scene, pair=[1, 2]) for scene in P.SCENES]
    rows = []
    for si, entry in enumerate(entries):
        for family in P.FAMILIES:
            for i, config in enumerate(P.GRID):
                for q in P.QUERIES:
                    # Different scene denominators expose accidental pooled selection.
                    denominator = 20 + si*10
                    rows.append(dict(**entry, family=family, grid_index=i, config=config, query=q,
                        pck_count=denominator, n_point=denominator, n_query=P.QUERY_COUNTS[q]))
    return entries, rows


def set_fraction(rows, family, index, query, counts):
    for row in rows:
        if (row['family'], row['grid_index'], row['query']) == (family, index, query):
            row['pck_count'] = counts[P.SCENES.index(row['scene'])]


def test_equal_scene_selection_and_first_grid_ties_are_global():
    entries, rows = fake_rows()
    for row in rows:
        if row['family'] == 'gaussian_point':
            row['pck_count'] = 0
    # cfg0 macro=(1+0+0)/3, cfg1=(0+0+1)/3: exact equal-scene tie -> cfg0.
    for q in P.QUERIES:
        set_fraction(rows, 'gaussian_point', 0, q, (20, 0, 0))
        set_fraction(rows, 'gaussian_point', 1, q, (0, 0, 40))
    summary = P.summarize_rows(rows, entries)
    assert summary['point_localization_selections']['gaussian_point']['grid_index'] == 0
    assert summary['metrics']['gaussian_point'][0]['equal_scene_equal_scale_pck8'] == pytest.approx(1/3)
    assert summary['feasible_grid_indices'] == list(range(12))
    assert not summary['final_primary_selected'] and not summary['scientific_gate_passed']


def test_any_stride_or_any_reference_can_make_candidate_infeasible():
    entries, rows = fake_rows()
    # Global Gaussian winner cfg0 has q24 poorer than legacy cfg4; check all refs.
    for row in rows:
        if row['family'] == 'gaussian_point':
            row['pck_count'] = 0
        elif row['family'] == 'quartic_point':
            row['pck_count'] = row['n_point']//2
    for q in ('g8', 'g16'):
        set_fraction(rows, 'gaussian_point', 0, q, (20, 30, 40))
    set_fraction(rows, 'gaussian_point', 4, 'g24', (20, 30, 40))
    for i in range(12):
        set_fraction(rows, 'quartic_full', i, 'g24', (19, 29, 39))
    summary = P.summarize_rows(rows, entries)
    assert summary['point_localization_selections']['gaussian_point']['grid_index'] == 0
    assert summary['feasible_grid_indices'] == []
    assert summary['decision_pending_independent_audit'] == 'stop_unchanged_candidate'
    for row in summary['feasibility']:
        failures = [c for c in row['checks'] if not c['passed']]
        assert len(failures) == 1 and failures[0]['reference'] == 'legacy_point_pck'


def test_exact_half_percentage_point_boundary_and_just_outside():
    entries, rows = [], []
    for scene in P.SCENES:
        # Five pairs give exactly 200 center-valid queries at each stride.
        for pair in range(5):
            entry = dict(scene=scene, pair=[pair, pair+10])
            entries.append(entry)
            for family in P.FAMILIES:
                for i, config in enumerate(P.GRID):
                    for q in P.QUERIES:
                        correct = 40
                        if family == 'quartic_full' and pair == 0 and q == 'g24':
                            correct -= 1  # 199/200: exactly .5pp below every reference.
                            if i == 1 and scene == P.SCENES[0]:
                                correct -= 1  # strictly outside, with identical denominators.
                        rows.append(dict(**entry, family=family, grid_index=i, config=config, query=q,
                            pck_count=correct, n_point=40, n_query=P.QUERY_COUNTS[q]))
    summary = P.summarize_rows(rows, entries)
    assert 0 in summary['feasible_grid_indices'] and 1 not in summary['feasible_grid_indices']
    assert summary['point_localization_selections']['gaussian_point']['grid_index'] == 0
    checks = [v for v in summary['feasibility'][0]['checks'] if v['query'] == 'g24']
    assert all(v['passed'] and v['difference_pp'] == -.5 for v in checks)


def test_population_rejects_missing_duplicate_and_changed_denominator():
    entries, rows = fake_rows()
    with pytest.raises(ValueError, match='population'):
        P.verify_population(rows[:-1], entries)
    with pytest.raises(ValueError, match='population'):
        P.verify_population(rows + rows[:1], entries)
    rows[0]['n_point'] -= 1
    rows[0]['pck_count'] -= 1
    with pytest.raises(ValueError, match='population differs'):
        P.verify_population(rows, entries)


def test_forward_loader_reads_only_allowed_observations(tmp_path):
    path = tmp_path/'input.npz'
    np.savez(path, **{k: np.zeros(1) for k in P.FIELDS}, sensor_label=np.ones(1), gt_xy=np.ones(1))
    assert set(P.load_forward_inputs(path)) == set(P.FIELDS)
    target = tmp_path/'target.npz'
    np.savez(target, **{q+'_point_xy': np.zeros((1, 2)) for q in P.QUERIES},
             **{q+'_point_visible': np.ones(1, bool) for q in P.QUERIES},
             **{q+'_counts': np.ones((1, 1)) for q in P.QUERIES})
    assert set(P.load_center_targets(target)) == {q+s for q in P.QUERIES for s in ('_point_xy', '_point_visible')}


def test_unsealed_or_incomplete_forward_blocks_every_target_decode(tmp_path, monkeypatch):
    monkeypatch.setattr(P, 'verify_lock', lambda out: {})
    monkeypatch.setattr(P, 'scene_entries', lambda lock, scene: [])
    def forbidden(*args):
        raise AssertionError('target reader ran before complete forward seal')
    monkeypatch.setattr(P, 'load_center_targets', forbidden)
    with pytest.raises(FileNotFoundError):
        P.evaluate(tmp_path, P.SCENES[0])
    P.write(tmp_path/'lock.json', {})
    for scene in P.SCENES[:2]:
        directory = tmp_path/'forward'/scene.replace('/', '__')
        directory.mkdir(parents=True)
        P.complete_stage(directory, tmp_path, 'forward', scene, 0., record_count=1728, labels_decoded=False)
    P.write(tmp_path/'forward_complete.json', dict(lock_sha256=P.sha(tmp_path/'lock.json'),
        forward_chain={}, array_count=5184, labels_decoded=False))
    with pytest.raises(FileNotFoundError):
        P.evaluate(tmp_path, P.SCENES[0])
    assert not (tmp_path/'evaluation').exists()


def test_stage_manifest_binds_saved_arrays_and_extra_files(tmp_path):
    P.write(tmp_path/'lock.json', {})
    directory = tmp_path/'shard'
    directory.mkdir()
    (directory/'p.bin').write_bytes(b'original probability')
    P.complete_stage(directory, tmp_path, 'forward', P.SCENES[0], 0.)
    P.verify_stage(directory, tmp_path, 'forward', P.SCENES[0])
    (directory/'p.bin').write_bytes(b'changed probability')
    with pytest.raises(ValueError, match='changed frozen file'):
        P.verify_stage(directory, tmp_path, 'forward', P.SCENES[0])
    (directory/'p.bin').write_bytes(b'original probability')
    (directory/'unexpected.bin').write_bytes(b'extra')
    with pytest.raises(ValueError, match='extra or missing'):
        P.verify_stage(directory, tmp_path, 'forward', P.SCENES[0])


def test_audit_must_bind_correct_result_and_never_select_primary(tmp_path, monkeypatch):
    monkeypatch.setattr(P, 'verify_lock', lambda out: {})
    for name, value in [('lock.json', {}), ('forward_complete.json', {}),
                        ('summary.json', dict(feasible=False, feasible_grid_indices=[]))]:
        P.write(tmp_path/name, value)
    P.write(tmp_path/'completion.json', dict(files={name: P.sha(tmp_path/name)
        for name in ('lock.json', 'forward_complete.json', 'summary.json')}))
    audit = dict(passed=True, feasible_grid_indices=[], scientific_gate_passed=False,
        **{f'feasibility_{name}_sha256': P.sha(tmp_path/(name+'.json'))
           for name in ('lock', 'forward_complete', 'summary', 'completion')})
    path = tmp_path/'audit.json'
    bad = dict(audit, feasibility_completion_sha256='wrong')
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='audit/hash'):
        P.audit_decision(tmp_path, path)
    path.write_text(json.dumps(audit))
    decision = P.audit_decision(tmp_path, path)
    assert decision['action'] == 'stop_unchanged_candidate'
    assert not decision['primary_selected'] and not decision['validation_authorized']
    assert decision['regional_design'] == dict(families=8, configurations_per_family=12, conditions=14, unchanged_J=True)


def test_canonical_hash_binds_shape_and_dtype():
    values = np.arange(4, dtype=np.float64)
    assert P.array_sha(values) != P.array_sha(values.reshape(2, 2))
    assert P.array_sha(values) != P.array_sha(values.astype(np.float32))
