import json

import numpy as np

import pytest

from scipy.special import ndtr

import torch

from experiments import round17_feasibility as P


def test_new_forward_full_grid_preserves_old_gaussian_t0_and_point_limit():
    mean=torch.tensor([[.2,.3],[1.4,.2]],dtype=torch.float32)
    cov=torch.tensor([[[.12,.02],[.02,.2]],[[.25,-.04],[-.04,.15]]],dtype=torch.float32)
    full=dict(mean=mean,cov=cov,mass=torch.full((2,),64.),features=torch.eye(2))
    key=dict(cell_bounds=torch.tensor([[-.0625,-.0625,.9375,.9375],[.9375,-.0625,1.9375,.9375]],dtype=torch.float64),
             cell_owner=torch.arange(2),mass=torch.full((2,),64.),features=torch.eye(2))
    with torch.no_grad():
        for chosen in (full,dict(full,cov=torch.zeros_like(cov))):
            actual,numeric=P.condition_predictions(chosen,key)
            assert len(actual)==12 and len(numeric)==4
            content=chosen['features']@key['features'].T
            for i,cfg in enumerate(P.GRID):
                raw=P.S.M.tiled_cell_mass(chosen['mean'],chosen['cov'].double(),key,cfg['h'])
                expected=(torch.log(raw+1e-30)+cfg['temperature']*content).softmax(-1)
                value=actual['trace_full',i]
                if cfg['temperature']==0 or not bool(chosen['cov'].any()):
                    assert torch.equal(value,expected)
                assert value.dtype==torch.float64 and torch.max(abs(value.sum(-1)-1))<1e-12
    assert P.RAW_PROBABILITY_BYTES==1170505728

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
        set_fraction(rows, 'trace_full', i, 'g24', (19, 29, 39))
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
            for family in P.ALL_FAMILIES:
                for i, config in enumerate(P.GRID):
                    for q in P.QUERIES:
                        correct = 40
                        if family == 'trace_full' and pair == 0 and q == 'g24':
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
        P.verify_population(rows[:-1], entries, P.ALL_FAMILIES)
    with pytest.raises(ValueError, match='population'):
        P.verify_population(rows + rows[:1], entries, P.ALL_FAMILIES)
    rows[0]['n_point'] -= 1
    rows[0]['pck_count'] -= 1
    with pytest.raises(ValueError, match='population differs'):
        P.verify_population(rows, entries, P.ALL_FAMILIES)

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
        P.complete_stage(directory, tmp_path, 'forward', scene, 0., record_count=576, labels_decoded=False)
    P.write(tmp_path/'forward_complete.json', dict(lock_sha256=P.sha(tmp_path/'lock.json'),
        forward_chain={}, array_count=1728, labels_decoded=False))
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
    assert decision['regional_design'] == dict(new_families=1, reused_families=8, configurations_per_family=12, conditions=14, unchanged_J=True)

def test_canonical_hash_binds_shape_and_dtype():
    values = np.arange(4, dtype=np.float64)
    assert P.array_sha(values) != P.array_sha(values.reshape(2, 2))
    assert P.array_sha(values) != P.array_sha(values.astype(np.float32))

def fake_rows():
    # Only one fake pair per scene is enough to exercise selection/aggregation.
    entries = [dict(scene=scene, pair=[1, 2]) for scene in P.SCENES]
    rows = []
    for si, entry in enumerate(entries):
        for family in P.ALL_FAMILIES:
            for i, config in enumerate(P.GRID):
                for q in P.QUERIES:
                    # Different scene denominators expose accidental pooled selection.
                    denominator = 20 + si*10
                    rows.append(dict(**entry, family=family, grid_index=i, config=config, query=q,
                        pck_count=denominator, n_point=denominator, n_query=P.QUERY_COUNTS[q]))
    return entries, rows
