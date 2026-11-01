from copy import deepcopy
import numpy as np
import pytest
import torch
from experiments import round17_regional as P


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
    record = dict(pair=[1, 2], family='trace_full', grid_index=6, condition='g24_g8',
                  array_sha256='probability', input_sha256='input')
    prior = {((1, 2), 'trace_full', 6, 'g24_g8'): dict(array_sha256='probability', input_sha256='input')}
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

def test_raw_storage_population_matches_all_native_partitions():
    parts=P.S.M.partitions()
    cells=sum(len(parts[c['query']]['mass'])*len(parts[c['key']]['mass']) for c in P.CONDITIONS)
    assert cells*12*8==P.PROBABILITY_BYTES_PER_PAIR
    assert P.SAMPLE_BYTES==192163104 and P.TOTAL_RECORDS==8064
    assert len(P.CONDITIONS)==14 and len({c['condition'] for c in P.CONDITIONS})==14


def fake_old():
    families=P.OLD.FAMILIES
    metrics={f:[dict(grid_index=i,config=P.GRID[i],J=4.,common9_kl=4.,direct5_kl=4.) for i in range(12)] for f in families}
    def choice(f,i=0):return dict(family=f,grid_index=i,config=P.GRID[i],J=4.)
    controls={f:choice(f) for f in families[1:]}
    return dict(metrics=metrics,selection=dict(controls=controls,primary=choice('quartic_full',6),
      strong_point=choice('gaussian_point'),strong_quadrature=choice('gaussian_quad16')))


def test_equal_scene_objective_and_fixed_eligible_selection(monkeypatch):
    entries,rows=fake_rows();old=fake_old()
    monkeypatch.setattr(P,'read',lambda path:old)
    result=P.summarize_rows(rows,entries,[6,9])
    zero=result['metrics']['trace_full'][0]
    assert zero['common9_kl']==pytest.approx(2.) and zero['direct5_kl']==pytest.approx(4.)
    assert zero['J']==pytest.approx(3.)
    assert result['selection']['primary']['grid_index']==6
    assert result['selection']['unconstrained_primary']['grid_index']==0
    assert result['selection']['controls']==old['selection']['controls']
    assert not result['train_screen']['passed']
    assert result['train_screen']['fixed_primary']==result['selection']['primary']
    with pytest.raises(ValueError,match='eligible'):P.summarize_rows(rows,entries,[])


def test_futility_individual_metrics_legacy_references_and_exact_boundaries():
    old=fake_old();selection=deepcopy(old['selection'])
    selection['primary']=dict(family='trace_full',grid_index=1,config=P.GRID[1],J=3.92)
    values=[dict(common9_kl=3.92,direct5_kl=3.92) for _ in range(12)]
    screen=P.train_screen(values,selection,old)
    assert screen['passed'] and len(screen['checks'])==8
    assert [c['reference_name'] for c in screen['checks']]==['strong_point','legacy_point']*2+['strong_quadrature','legacy_quad16']*2
    assert screen['checks'][1]['reference']['grid_index']==7
    assert screen['checks'][5]['reference']['grid_index']==4
    values[1]['direct5_kl']=np.nextafter(3.92,np.inf)
    assert not P.train_screen(values,selection,old)['passed']
    values[1]['direct5_kl']=3.92
    old['metrics']['gaussian_point'][7]['common9_kl']=0.
    screen=P.train_screen(values,selection,old)
    assert not screen['passed'] and screen['checks'][1]['relative_difference'] is None


def test_regional_forward_equals_stage_a_for_native_g8_on_synthetic_support():
    q=dict(mean=torch.tensor([[.2,.3],[1.4,.2]]),cov=torch.tensor([[[.1,.02],[.02,.2]],[[.3,-.01],[-.01,.2]]]),features=torch.eye(2))
    k=dict(cell_bounds=torch.tensor([[-.0625,-.0625,.9375,.9375],[.9375,-.0625,1.9375,.9375]],dtype=torch.float64),
      cell_owner=torch.arange(2),mass=torch.full((2,),64.),features=torch.eye(2))
    with torch.no_grad():
        a,_=P.A.condition_predictions(q,k)
        b=list(P.condition_predictions(q,k))
    assert len(b)==12
    for family,index,prob,_ in b:np.testing.assert_array_equal(prob,a[family,index].numpy())

