from copy import deepcopy

import numpy as np
import pytest
import torch

from experiments import round15_regional as P


def synthetic_supports():
    mean = torch.tensor([[.2, .3], [1.4, .2]], dtype=torch.float32)
    cov = torch.tensor([[[.12, .02], [.02, .2]], [[.25, -.04], [-.04, .15]]], dtype=torch.float32)
    full = dict(mean=mean, cov=cov, features=torch.eye(2), mass=torch.full((2,), 64.))
    point = dict(full, cov=torch.zeros_like(cov))
    queries = dict(full=full, point=point)
    for name, count in (('full4', 4), ('quad4', 4), ('quad16', 16)):
        queries[name] = dict(full, nodes=mean[:, None].expand(-1, count, -1),
            node_cov=cov[:, None].expand(-1, count, -1, -1), node_weights=torch.full((2, count), 1/count))
    bounds = torch.tensor([[-.0625, -.0625, .9375, .9375], [.9375, -.0625, 1.9375, .9375]], dtype=torch.float64)
    key = dict(cell_bounds=bounds, native_cell_bounds=bounds, cell_owner=torch.arange(2),
               mass=torch.full((2,), 64.), features=torch.tensor([[-1., 0.], [0., 1.]]))
    return queries, key


def test_all_eight_families_complete_grid_and_collapsed_mixture_limit():
    queries, key = synthetic_supports()
    with torch.no_grad():
        values = {(f, i):p for f, i, p, _ in P.condition_predictions(queries, key)}
    assert len(values) == 96
    for i in range(12):
        for family in ('gaussian_full4', 'gaussian_quad4', 'gaussian_quad16'):
            np.testing.assert_allclose(values[family, i], values['gaussian_full1', i], atol=1.e-14, rtol=1.e-13)
    assert not np.array_equal(values['quartic_full', 4], values['gaussian_full1', 4])
    assert not np.array_equal(values['quartic_point', 4], values['gaussian_point', 4])


def test_quartic_gets_original24_bounds_gaussian_keeps12_children(monkeypatch):
    queries, key = synthetic_supports()
    native = torch.tensor([[-.0625, -.0625, 2.9375, 2.9375]], dtype=torch.float64)
    children, owners = P.S.M.split_cell_bounds(np.array([[0, 0, 24, 24]]))
    key = dict(key, native_cell_bounds=native, cell_bounds=torch.tensor(children),
               cell_owner=torch.tensor(owners), mass=torch.tensor([576.]), features=torch.tensor([[1., 0.]]))
    module = P.importlib.import_module(P.A.QUARTIC_MODULE)
    original = module.quartic_cell_mass
    def check_quartic(mean, cov, bounds, **kw):
        assert torch.equal(bounds, native) and bounds.shape == (1, 4)
        assert kw == dict(bandwidth=.5, quadrature_order=24, query_chunk_size=16, key_chunk_size=32)
        return original(mean, cov, bounds, **kw)
    monkeypatch.setattr(module, 'quartic_cell_mass', check_quartic)
    gaussian = P.S.M.tiled_cell_mass
    def check_gaussian(mean, cov, current_key, h):
        assert current_key['cell_bounds'].shape == (4, 4)
        assert torch.all(current_key['cell_bounds'][:, 2:]-current_key['cell_bounds'][:, :2] == 1.5)
        return gaussian(mean, cov, current_key, h)
    monkeypatch.setattr(P.S.M, 'tiled_cell_mass', check_gaussian)
    with torch.no_grad():
        q, _ = P.geometry(queries['full'], key, 'quartic_full', .5)
        g, _ = P.geometry(queries['full'], key, 'gaussian_full1', .5)
    assert q.shape == g.shape == (2, 1)


def test_same_native_floor_once_and_point_limit(monkeypatch):
    queries, key = synthetic_supports()
    key = dict(key, mass=torch.tensor([64., 256.]))
    module = P.importlib.import_module(P.A.QUARTIC_MODULE)
    monkeypatch.setattr(module, 'quartic_cell_mass', lambda *args, **kw: torch.zeros((2, 2), dtype=torch.float64))
    with torch.no_grad():
        logits, diagnostics = P.geometry(queries['point'], key, 'quartic_point', .25)
    np.testing.assert_allclose(logits.softmax(-1).numpy(), [[.2, .8], [.2, .8]], atol=1.e-14, rtol=0)
    assert diagnostics['raw_zero_fov_rows'] == 2 and diagnostics['raw_zero_cells'] == 4


def test_target_count_and_common_metric_keep_distinct_condition_semantics():
    parts = P.S.M.partitions()
    yy, xx = np.mgrid[:144, :192]
    labels = dict(gt_xy=np.stack((xx, yy), -1).reshape(-1, 2).astype(float),
                  gt_visible=np.ones(144*192, bool))
    direct, common = P.S.target_counts(labels, parts, 'g24', 'g8')
    assert direct.shape == (48, 432) and common.shape == (108, 432)
    assert np.all((direct == 0) | (direct == 64)) and np.all((direct > 0).sum(-1) == 9)
    assert direct.sum() == common.sum() == 27648
    probability = direct / 576
    condition = dict(query='g24', key='g8', scenario='scale')
    metric = P.measure(probability, parts, condition, direct, common)
    assert metric['kl_sum'] == pytest.approx(0.) and metric['direct_kl_sum'] == pytest.approx(0.)
    with pytest.raises(ValueError, match='no common16 parent'):
        P.measure(probability, parts, dict(condition, scenario='partition'), direct, common)
    # Coarse target g16 reconstructs uniform atomic density; common q16 truth
    # matches, while individual q8 native truths still select one g16 key.
    direct, common = P.S.target_counts(labels, parts, 'g8', 'g16')
    probability = direct / 64
    metric = P.measure(probability, parts, dict(query='g8', key='g16', scenario='partition'), direct, common)
    assert metric['kl_sum'] == pytest.approx(0., abs=1.e-10)


def test_raw_storage_population_matches_all_native_partitions():
    parts = P.S.M.partitions()
    cells = sum(len(parts[c['query']]['mass'])*len(parts[c['key']]['mass']) for c in P.CONDITIONS)
    assert cells * 96 * 8 == P.PROBABILITY_BYTES_PER_PAIR
    assert P.SAMPLE_BYTES == 1537304832 and P.TOTAL_RECORDS == 64512
    assert len(P.CONDITIONS) == 14 and len({c['condition'] for c in P.CONDITIONS}) == 14


def fake_rows():
    parts = P.S.M.partitions()
    entries = [dict(scene=scene, pair=[1, 2]) for scene in P.SCENES]
    rows = []
    for si, entry in enumerate(entries):
        mass = 100 * (si+1)
        for family in P.FAMILIES:
            for index, config in enumerate(P.GRID):
                for condition in P.CONDITIONS:
                    value = 1 + si + index/10 + (2 if condition['scenario'] == 'scale' else 0)
                    shape = [len(parts[condition['query']]['mass']), len(parts[condition['key']]['mass'])]
                    rows.append(dict(**entry, family=family, grid_index=index, config=config, **condition,
                        shape=shape, dtype='float64', array_sha256='same', replayed_array_sha256='same',
                        kl_sum=value*mass, js_sum=.1*mass, direct_kl_sum=(value+.5)*mass,
                        direct_js_sum=.2*mass, mass=mass, n_query=shape[0], n_key=shape[1]))
    return entries, rows


def test_equal_scene_and_equal_condition_family_objective_preserves_all_grid():
    entries, rows = fake_rows()
    result = P.summarize_rows(rows, entries, [6, 9])
    zero = result['metrics']['quartic_full'][0]
    assert zero['common9_kl'] == pytest.approx(2.)
    assert zero['direct5_kl'] == pytest.approx(4.)
    assert zero['J'] == pytest.approx(3.)  # not a pooled9:5 or mass-weighted scene mean
    assert all(len(values) == 12 for values in result['metrics'].values())
    selected = result['selection']
    assert selected['primary']['grid_index'] == 6
    assert selected['unconstrained_primary']['grid_index'] == 0
    assert all(v['grid_index'] == 0 for v in selected['controls'].values())
    assert selected['strong_point']['family'] == 'gaussian_point'
    assert selected['strong_quadrature']['family'] == 'gaussian_quad4'


def test_control_regional_optimum_is_not_filtered_by_candidate_eligibility():
    entries, rows = fake_rows()
    for row in rows:
        if row['family'] == 'gaussian_quad16' and row['grid_index'] == 2:
            row['kl_sum'] = .1 * row['mass']
    result = P.summarize_rows(rows, entries, [6, 9])
    assert result['selection']['strong_quadrature']['family'] == 'gaussian_quad16'
    assert result['selection']['strong_quadrature']['grid_index'] == 2
    with pytest.raises(ValueError, match='eligible'):
        P.summarize_rows(rows, entries, [])


def test_population_rejects_partial_grid_changed_mass_and_failed_replay():
    entries, rows = fake_rows()
    with pytest.raises(ValueError, match='population'):
        P.verify_population(rows[:-1], entries, metrics=True)
    changed = deepcopy(rows)
    changed[0]['mass'] -= 1
    with pytest.raises(ValueError, match='visible pixel mass'):
        P.verify_population(changed, entries, metrics=True)
    changed = deepcopy(rows)
    changed[0]['replayed_array_sha256'] = 'changed'
    with pytest.raises(ValueError, match='replay'):
        P.verify_population(changed, entries, metrics=True)


def test_one_bit_probability_replay_difference_is_fatal():
    probability = np.array([[.25, .75]], np.float64)
    record = dict(shape=[1, 2], array_sha256=P.array_sha(probability))
    assert P.require_replay(probability, record) == record['array_sha256']
    changed = probability.copy()
    changed[0, 0] = np.nextafter(changed[0, 0], 1.)
    with pytest.raises(ValueError, match='bitwise'):
        P.require_replay(changed, record)


def test_overlapping_operator_is_bound_to_stage_a_center_eligibility():
    record = dict(pair=[1, 2], family='quartic_full', grid_index=6, condition='g24_g8',
                  array_sha256='probability', input_sha256='input')
    prior = {((1, 2), 'quartic_full', 6, 'g24_g8'): dict(array_sha256='probability', input_sha256='input')}
    assert P.check_stage_a_overlap(record, prior) == 'probability'
    with pytest.raises(ValueError, match='Stage A probability/input'):
        P.check_stage_a_overlap(dict(record, array_sha256='changed'), prior)
    with pytest.raises(ValueError, match='Stage A probability/input'):
        P.check_stage_a_overlap(dict(record, input_sha256='changed'), prior)
    assert P.check_stage_a_overlap(dict(record, condition='g24_g24'), prior) is None


def test_unsealed_global_forward_blocks_label_reader(tmp_path, monkeypatch):
    monkeypatch.setattr(P, 'verify_lock', lambda out: {})
    monkeypatch.setattr(P, 'scene_entries', lambda lock, scene: [])
    def forbidden(entry):
        raise AssertionError('regional labels opened before global forward commitment')
    monkeypatch.setattr(P, 'load_regional_labels', forbidden)
    with pytest.raises(FileNotFoundError):
        P.evaluate(tmp_path, P.SCENES[0])
    assert not (tmp_path/'evaluation').exists()


def test_freeze_rejects_failed_stage_a_before_creating_output(tmp_path, monkeypatch):
    def rejected(directory):
        raise ValueError('Stage A independent audit/decision binding required')
    monkeypatch.setattr(P, 'stage_a_contract', rejected)
    out = tmp_path/'regional'
    with pytest.raises(ValueError, match='Stage A'):
        P.freeze(out, tmp_path/'stage_a')
    assert not out.exists()


def test_forward_inputs_and_regional_labels_are_separate_allowlists(tmp_path):
    arrays = {k:np.zeros(1) for k in P.A.FIELDS}
    arrays.update(gt_xy=np.zeros((27648, 2)), gt_visible=np.ones(27648, bool), g8_point_xy=np.ones((432, 2)))
    np.savez(tmp_path/'inputs.npz', **arrays)
    np.savez(tmp_path/'labels.npz', **arrays)
    assert set(P.A.load_forward_inputs(tmp_path/'inputs.npz')) == set(P.A.FIELDS)
    entry = dict(path=str(tmp_path), label_sha256=P.sha(tmp_path/'labels.npz'))
    assert set(P.load_regional_labels(entry)) == set(P.REGIONAL_FIELDS)


def test_audit_rejects_unbound_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(P, 'verify_lock', lambda out: dict(stage_a={}, legacy={}))
    for name in ('lock', 'forward_complete'):
        P.write(tmp_path/(name+'.json'), {})
    P.write(tmp_path/'summary.json', dict(selection={'primary': 'frozen'}))
    P.write(tmp_path/'completion.json', dict(files={name+'.json':P.sha(tmp_path/(name+'.json'))
        for name in ('lock','forward_complete','summary')}))
    audit = dict(passed=True, scientific_gate_passed=False, selection={'primary': 'different'},
        **{f'regional_{name}_sha256':P.sha(tmp_path/(name+'.json')) for name in ('lock','forward_complete','summary','completion')})
    P.write(tmp_path/'audit.json', audit)
    with pytest.raises(ValueError, match='selection/hash audit'):
        P.accept_audit(tmp_path, tmp_path/'audit.json')
    assert not (tmp_path/'selection.json').exists()
