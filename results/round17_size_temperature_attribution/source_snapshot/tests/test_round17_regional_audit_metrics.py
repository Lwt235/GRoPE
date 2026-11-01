"""Synthetic population, weighting and stop-rule checks; no experiment reads."""
from copy import deepcopy
from math import nextafter

import pytest

from experiments import round17_regional_audit_metrics as M


def metric(index, common, direct=None):
    direct = common if direct is None else direct
    h, t = M.GRID[index]
    return dict(grid_index=index, config=dict(h=h, temperature=t),
                common9_kl=common, direct5_kl=direct, J=.5 * common + .5 * direct)


def spec(metrics, family, index):
    value = metrics[family][index]
    return dict(family=family, grid_index=index, config=deepcopy(value['config']), J=value['J'])


def historical_summary():
    metrics = {family: [metric(i, 10.) for i in range(12)] for family in M.OLD_FAMILIES}
    # Historical primary is intentionally NOT the unconstrained minimum.
    metrics['quartic_full'][4] = metric(4, 1.)
    metrics['quartic_full'][6] = metric(6, 8.)
    metrics['quartic_full'][9] = metric(9, 9.)
    controls = {family: spec(metrics, family, 0) for family in M.OLD_FAMILIES[1:]}
    selection = dict(primary=spec(metrics, 'quartic_full', 6),
        unconstrained_primary=spec(metrics, 'quartic_full', 4), controls=controls,
        strong_point=deepcopy(controls['gaussian_point']),
        strong_quadrature=deepcopy(controls['gaussian_quad4']), eligible_indices=[6, 9])
    return dict(metrics=metrics, selection=selection)


@pytest.fixture(scope='module')
def full_population():
    entries, rows = [], []
    for s, scene in enumerate(M.SCENES):
        for p in range(16):
            pair = [p * 15, p * 15 + 5]
            entries.append(dict(scene=scene, pair=pair, split='train'))
            mass = (1, 10, 100)[s] * (p + 1)
            for index, (h, t) in enumerate(M.GRID):
                for c, (q, k) in enumerate(M.CONDITIONS):
                    partition = c < 9
                    common = (s + 1) * (p + 1) / 16 + (c + 1) / 10 + .01 * (index + 1)
                    direct = (common + 5 if partition else
                              2 * (s + 1) * (p + 1) / 16 + (c - 8) / 10 + .01 * (index + 1))
                    rows.append(dict(scene=scene, pair=pair, family='trace_full', grid_index=index,
                        query=q, key=k, condition=q + '_' + k,
                        config=dict(h=h, temperature=t), scenario='partition' if partition else 'scale',
                        mass=mass, n_query=M.NATIVE_SIZES[q], n_key=M.NATIVE_SIZES[k],
                        kl_sum=mass * (common if partition else direct), js_sum=mass * .1,
                        direct_kl_sum=mass * direct, direct_js_sum=mass * (.2 if partition else .1)))
    return entries, rows


def test_full_8064_population_and_equal_scene_then_condition_weighting(full_population):
    entries, rows = full_population
    metrics, population = M.summarize_metrics(rows, entries)
    assert population == dict(rows=8064, pairs=48, mass_by_scene=dict(zip(M.SCENES, (136, 1360, 13600))),
                              unique_visible_pixel_mass=15096)
    first = metrics['trace_full'][0]
    assert first['common9_kl'] == pytest.approx(2 * .6875 + .5 + .01)
    assert first['direct5_kl'] == pytest.approx(4 * .6875 + .3 + .01)
    assert first['J'] == pytest.approx(.5 * (1.885 + 3.06))
    assert first['J'] != pytest.approx((9 * first['common9_kl'] + 5 * first['direct5_kl']) / 14)
    assert len(first['by_condition']) == 14
    assert first['by_scene_condition'][M.SCENES[0]]['g8_g8']['direct_kl'] > 5
    assert M.validate_population(rows) == population
    reversed_metrics, _ = M.summarize_metrics(list(reversed(rows)), list(reversed(entries)))
    assert reversed_metrics == metrics


def test_entry_contract_rejects_wrong_but_plausible_frame_cohort(full_population):
    entries, rows = full_population
    changed = deepcopy(entries)
    changed[0]['pair'] = [999, 1000]
    with pytest.raises(ValueError, match='unexpected cohort'):
        M.validate_population(rows, changed)
    changed = deepcopy(entries)
    changed[0]['split'] = 'val'
    with pytest.raises(ValueError, match='TRAIN'):
        M.validate_population(rows, changed)


@pytest.mark.parametrize('change,match', [
    ('missing', '8064'), ('duplicate', 'duplicate'), ('mass', 'mass changed'),
    ('family', 'family'), ('grid', 'fixed grid'), ('native', 'population changed'),
    ('nan', 'finite real'), ('js', 'JS exceeds'), ('scenario', 'scenario'),
    ('scale', 'direct native'), ('bool_mass', 'integer'), ('frame', 'frame index')])
def test_population_rejects_partial_or_corrupt_rows(full_population, change, match):
    entries, source = full_population
    rows = deepcopy(source)
    if change == 'missing':
        rows.pop()
    elif change == 'duplicate':
        rows[-1] = deepcopy(rows[0])
    elif change == 'mass':
        rows[0]['mass'] += 1
    elif change == 'family':
        rows[0]['family'] = 'quartic_full'
    elif change == 'grid':
        rows[0]['config']['h'] = .5
    elif change == 'native':
        rows[0]['n_key'] -= 1
    elif change == 'nan':
        rows[0]['kl_sum'] = float('nan')
    elif change == 'js':
        rows[0]['js_sum'] = rows[0]['mass']
    elif change == 'scenario':
        rows[0]['scenario'] = 'scale'
    elif change == 'scale':
        rows[9]['direct_kl_sum'] += 1
    elif change == 'bool_mass':
        rows[0]['mass'] = True
    elif change == 'frame':
        rows[0]['pair'] = [1., 2.]
    with pytest.raises(ValueError, match=match):
        M.validate_population(rows, entries)


def test_zero_mass_pair_kept_and_all_empty_scene_rejected(full_population):
    entries, source = full_population
    rows = deepcopy(source)
    for row in rows[:168]:
        row['mass'] = 0
        row.update({field: 0. for field in M.METRIC_FIELDS})
    assert M.validate_population(rows, entries)['pairs'] == 48
    rows[0]['kl_sum'] = 1e-12
    with pytest.raises(ValueError, match='zero visible mass'):
        M.validate_population(rows, entries)
    for row in rows[:16 * 168]:
        row['mass'] = 0
        row.update({field: 0. for field in M.METRIC_FIELDS})
    with pytest.raises(ValueError, match='no visible'):
        M.validate_population(rows, entries)


@pytest.mark.parametrize('eligible', [[], [6, 6], [-1], [12], [True], [6.]])
def test_eligible_set_never_falls_back_to_unconstrained(eligible):
    new = {'trace_full': [metric(i, 1. + i) for i in range(12)]}
    with pytest.raises(ValueError):
        M.select_metrics(new, historical_summary(), eligible)


def test_new_primary_constraint_stable_tie_and_old_primary_constraint_are_distinct():
    new = {'trace_full': [metric(i, 1.) for i in range(12)]}
    old = historical_summary()
    before = deepcopy(old)
    selection = M.select_metrics(new, old, [9, 6])
    assert selection['primary']['grid_index'] == 6
    assert selection['unconstrained_primary']['grid_index'] == 0
    assert selection['historical_primary']['grid_index'] == 6
    assert old['selection']['unconstrained_primary']['grid_index'] == 4
    assert selection['eligible_indices'] == [6, 9]
    assert selection['strong_point']['family'] == 'gaussian_point'
    assert selection['strong_quadrature']['family'] == 'gaussian_quad4'
    assert old == before
    selection['controls']['gaussian_point']['config']['h'] = 99
    assert old == before  # Output metadata cannot mutate historical bindings.


@pytest.mark.parametrize('change', ['family', 'grid', 'objective', 'control', 'strong', 'primary', 'tie'])
def test_historical_contract_rejects_changed_selection_or_missing_grid(change):
    old = historical_summary()
    if change == 'family':
        old['metrics'].pop('quartic_point')
    elif change == 'grid':
        old['metrics']['gaussian_full1'].pop()
    elif change == 'objective':
        old['metrics']['gaussian_full1'][0]['J'] += .1
    elif change == 'control':
        old['selection']['controls']['gaussian_full1'] = spec(old['metrics'], 'gaussian_full1', 1)
    elif change == 'strong':
        old['selection']['strong_point'] = spec(old['metrics'], 'quartic_point', 0)
    elif change == 'primary':
        old['selection']['primary'] = old['selection']['unconstrained_primary']
    elif change == 'tie':
        old['selection']['controls']['gaussian_fixed'] = spec(old['metrics'], 'gaussian_fixed', 11)
    with pytest.raises(ValueError):
        M.validate_historical_summary(old)


def test_all_eight_screen_references_use_old_metrics_including_distinct_legacy():
    old = historical_summary()
    old['metrics']['gaussian_point'][7] = metric(7, 11., 12.)
    old['metrics']['gaussian_quad16'][4] = metric(4, 13., 14.)
    new = {'trace_full': [metric(i, 9.) for i in range(12)]}
    selected = M.select_metrics(new, old, [6])
    result = M.train_screen(new, old, selected)
    assert result['passed'] and result['scientific_gate_passed'] is False
    assert result['next_action'] == 'prepare_old_validation_protocol'
    checks = result['checks']
    assert [c['name'] for c in checks] == [
        'common9_kl_vs_strong_point', 'common9_kl_vs_legacy_point',
        'direct5_kl_vs_strong_point', 'direct5_kl_vs_legacy_point',
        'common9_kl_vs_strong_quadrature', 'common9_kl_vs_legacy_quad16',
        'direct5_kl_vs_strong_quadrature', 'direct5_kl_vs_legacy_quad16']
    assert [c['reference_value'] for c in checks] == [10., 11., 10., 12., 10., 13., 10., 14.]
    assert checks[1]['reference']['grid_index'] == 7
    assert checks[1]['reference']['config'] == dict(h=1., temperature=5.)
    assert checks[5]['reference']['grid_index'] == 4
    assert checks[5]['reference']['config'] == dict(h=.5, temperature=5.)
    assert checks[1]['relative_difference'] == 9 / 11 - 1


@pytest.mark.parametrize('check_index,multiplier,metric_name', [
    (0, .98, 'common9_kl'), (2, .98, 'direct5_kl'),
    (4, 1.01, 'common9_kl'), (6, 1.02, 'direct5_kl')])
def test_screen_closed_boundary_has_no_hidden_tolerance(check_index, multiplier, metric_name):
    old = historical_summary()
    boundary = multiplier * 10.
    for actual, expected in ((boundary, True), (nextafter(boundary, float('inf')), False)):
        common, direct = (actual, 1.) if metric_name == 'common9_kl' else (1., actual)
        new = {'trace_full': [metric(i, common, direct) for i in range(12)]}
        selected = M.select_metrics(new, old, [6])
        check = M.train_screen(new, old, selected)['checks'][check_index]
        assert check['passed'] is expected
        assert check['bound'] == boundary


def test_failing_selected_configuration_does_not_trigger_eligible_reselection():
    old = historical_summary()
    new = {'trace_full': [metric(i, 20.) for i in range(12)]}
    new['trace_full'][6] = metric(6, 10., 1.)  # Best J, fails common point bound.
    new['trace_full'][9] = metric(9, 9., 9.)  # Would pass, forbidden rescue.
    selected = M.select_metrics(new, old, [6, 9])
    result = M.train_screen(new, old, selected)
    assert selected['primary']['grid_index'] == result['fixed_primary']['grid_index'] == 6
    assert not result['passed']
    assert result['next_action'] == 'stop_selected_candidate_before_old_validation'
    assert all(c['passed'] for c in M.train_screen(new, old,
        dict(selected, primary=spec(new, 'trace_full', 9)))['checks'])


def test_nonpositive_reference_never_yields_percentage_success():
    old = historical_summary()
    old['metrics']['gaussian_point'][0] = metric(0, 0.)
    old['selection']['controls']['gaussian_point'] = spec(old['metrics'], 'gaussian_point', 0)
    old['selection']['strong_point'] = spec(old['metrics'], 'gaussian_point', 0)
    new = {'trace_full': [metric(i, 0.) for i in range(12)]}
    selected = M.select_metrics(new, old, [6])
    screen = M.train_screen(new, old, selected)
    assert screen['checks'][0]['relative_difference'] is None
    assert screen['checks'][0]['passed'] is False
    assert screen['passed'] is False


def test_complete_pure_api_has_candidate_only_metrics_and_no_scene_gate(full_population):
    entries, rows = full_population
    old = historical_summary()
    result = M.summarize(rows, old, [9, 6], entries=entries)
    assert set(result) == {'metrics', 'selection', 'population', 'train_screen'}
    assert set(result['metrics']) == {'trace_full'}
    assert result['population']['rows'] == 8064
    assert result['selection']['primary']['grid_index'] == 6
    assert len(result['selection']['controls']) == 7
    assert len(result['train_screen']['checks']) == 8
    assert result['train_screen']['passed']  # No added per-scene rule.
