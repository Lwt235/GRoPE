"""Independent synthetic label/count toys; no archive or label file access."""
from copy import deepcopy
from fractions import Fraction
import math

import numpy as np
import pytest

from experiments import round25_student_metrics as A


def rows_fixture():
    entries, rows = [], []
    for si, scene in enumerate(A.SCENES):
        valid_total = A.VALID_BY_SCENE[scene]
        for index in range(16):
            entry = dict(scene=scene, pair=[2 * index, 2 * index + 1])
            entries.append(entry)
            valid = valid_total // 16 + int(index < valid_total % 16)
            mass = (si + 1) * (index + 1)
            kl, js = si + 1., (si + 1.) / 10
            region = dict(mass=mass, n_query=48, n_key=432, kl_sum=mass * kl,
                direct_kl_sum=mass * kl, js_sum=mass * js, direct_js_sum=mass * js)
            center = dict(n_query=48, n_point=valid, pck_count=valid - index % 3,
                distance_sum=valid * (si + 1.), distance_squared_sum=valid * (si + 2.) ** 2,
                mean=si + 1., rmse=si + 2.)
            rows.append(dict(entry, region=region, center=center))
    return entries, rows


def baselines(summary):
    pck = A.BASE.exact_fraction(summary['center']['macro']['pck'])
    old = {name: dict(by_condition={'g24_g8': dict(kl=2.)}) for name in A.REGIONAL_ROLES}
    same = {}
    for names, source in ((A.OLD_CENTER_ROLES, old), (A.SAME_A_CENTER_ROLES, same)):
        for name in names:
            source[name] = dict(center=dict(by_condition={'g24_g8': dict(pck=A.BASE.fraction_dict(pck))}))
    return dict(old_by_role=old, by_role=same)


def test_single_condition_equal_scene_reduction_and_exact_pck():
    entries, rows = rows_fixture(); value = A.summarize(rows, entries)
    assert value['population'] == dict(pairs=48, queries=2304, valid_centers=1096)
    assert value['region']['macro']['kl'] == 2.
    pooled = sum(r['region']['kl_sum'] for r in rows) / sum(r['region']['mass'] for r in rows)
    assert pooled != value['region']['macro']['kl']
    expected = sum((Fraction(sum(r['center']['pck_count'] for r in rows if r['scene'] == s), A.VALID_BY_SCENE[s]) for s in A.SCENES), Fraction()) / 3
    assert A.BASE.exact_fraction(value['center']['macro']['pck']) == expected
    assert value['center']['macro']['mean'] == 2.
    assert value['center']['macro']['rmse'] == 3.
    assert A.summarize(list(reversed(rows)), entries) == value
    assert 'common9_kl' not in str(value) and 'direct5_kl' not in str(value)


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'foreign_pair', 'bool_pair', 'wrong_query',
    'wrong_key', 'wrong_condition', 'bad_scale_metric', 'nan_region', 'negative_mass',
    'center_denominator', 'center_boolean_count', 'bad_pck', 'missing_distance', 'negative_distance', 'wrong_mean'])
def test_fixed_population_and_metric_guards(kind):
    entries, rows = rows_fixture()
    if kind == 'missing': rows.pop()
    elif kind == 'duplicate': rows[-1] = deepcopy(rows[0])
    elif kind == 'foreign_pair': rows[0]['pair'] = [1000, 1001]
    elif kind == 'bool_pair': rows[0]['pair'][0] = False
    elif kind == 'wrong_query': rows[0]['region']['n_query'] = 47
    elif kind == 'wrong_key': rows[0]['region']['n_key'] = 431
    elif kind == 'wrong_condition': rows[0]['condition'] = 'g8_g8'
    elif kind == 'bad_scale_metric': rows[0]['region']['direct_kl_sum'] += .01
    elif kind == 'nan_region': rows[0]['region']['kl_sum'] = float('nan')
    elif kind == 'negative_mass': rows[0]['region']['mass'] = -1
    elif kind == 'center_denominator': rows[0]['center']['n_point'] -= 1
    elif kind == 'center_boolean_count': rows[0]['center']['n_point'] = True
    elif kind == 'bad_pck': rows[0]['center']['pck_count'] = 49
    elif kind == 'missing_distance': del rows[0]['center']['distance_sum']
    elif kind == 'negative_distance': rows[0]['center']['distance_sum'] = -1.
    elif kind == 'wrong_mean': rows[0]['center']['mean'] += .01
    with pytest.raises((ValueError, KeyError)):
        A.summarize(rows, entries)


def test_scene_denominators_cannot_be_swapped_despite_same_global_total():
    entries, rows = rows_fixture()
    for index, delta in ((0, -1), (16, 1)):
        c = rows[index]['center']; c['n_point'] += delta; c['pck_count'] += delta
        c['distance_sum'] = c['mean'] * c['n_point']; c['distance_squared_sum'] = c['rmse'] ** 2 * c['n_point']
    with pytest.raises(ValueError, match='scene center population'):
        A.summarize(rows, entries)


def test_duplicate_entries_and_missing_scene_are_rejected():
    entries, rows = rows_fixture(); entries[-1] = deepcopy(entries[0])
    with pytest.raises(ValueError): A.summarize(rows, entries)


def test_eight_checks_preserve_aliases_and_use_exact_fraction_boundary():
    entries, rows = rows_fixture(); summary = A.summarize(rows, entries); old = baselines(summary)
    candidate = A.BASE.exact_fraction(summary['center']['macro']['pck'])
    role = A.SAME_A_CENTER_ROLES[-1]
    old['by_role'][role]['center']['by_condition']['g24_g8']['pck'] = A.BASE.fraction_dict(candidate + Fraction(1, 200))
    result = A.screen(summary, old)
    assert result['passed'] and len(result['checks']) == 8
    assert result['checks'][1]['alias_of'] == 'strong_quad'
    assert result['regional_alias_groups'][0]['distinct_operators'] == 1
    assert result['matched_gaussian_is_diagnostic_only'] and not result['scientific_gate_passed']
    # An exact epsilon below the inherited half-percentage-point limit fails,
    # even though its float display can round to the same number.
    old['by_role'][role]['center']['by_condition']['g24_g8']['pck'] = A.BASE.fraction_dict(candidate + Fraction(1, 200) + Fraction(1, 10**25))
    result = A.screen(summary, old)
    assert not result['passed'] and sum(not r['passed'] for r in result['checks']) == 1


def test_regional_bound_is_inclusive_without_epsilon_or_alias_deduplication():
    entries, rows = rows_fixture(); summary = A.summarize(rows, entries); old = baselines(summary)
    summary['region']['macro']['kl'] = 2. * 1.02
    assert A.screen(summary, old)['passed']
    summary['region']['macro']['kl'] = np.nextafter(2. * 1.02, np.inf)
    check = A.screen(summary, old)
    assert not check['passed'] and sum(not r['passed'] for r in check['checks']) == 2
    old['old_by_role']['legacy_quad16']['by_condition']['g24_g8']['kl'] += .01
    with pytest.raises(ValueError, match='aliases'): A.screen(summary, old)


def test_rounded_fraction_and_nonpositive_reference_are_not_passes():
    entries, rows = rows_fixture(); summary = A.summarize(rows, entries); old = baselines(summary)
    p = old['old_by_role']['point_localizer']['center']['by_condition']['g24_g8']['pck']
    p['value'] = round(p['value'], 4)
    with pytest.raises(ValueError): A.screen(summary, old)
    old = baselines(summary)
    for name in A.REGIONAL_ROLES: old['old_by_role'][name]['by_condition']['g24_g8']['kl'] = 0.
    assert not A.screen(summary, old)['passed']


def test_independent_label_primitives_use_native_population_and_original_visibility():
    # Synthetic all-pixel identity map; only seven pixel targets are visible.
    y, x = np.mgrid[:144, :192]
    xy = np.stack([x, y], axis=-1).reshape(-1, 2).astype(np.float64)
    visible = np.zeros(27648, bool); visible[[0, 7, 8, 23, 24, 191, 192]] = True
    xy[~visible] = np.nan
    parts = {q: dict(bounds=A.BASE.REGION.partition_bounds(q).astype(np.int32)) for q in ('g24', 'g8')}
    direct, common = A.target_counts(dict(gt_xy=xy, gt_visible=visible), parts)
    assert direct.shape == (48, 432) and common.shape == (108, 432)
    assert direct.sum() == common.sum() == 7
    assert direct[0, 0] == 3 and direct[0, 1] == 1 and direct[0, 2] == 1
    p = np.full((48, 432), 1 / 432, np.float64)
    result = A.region_metrics(p, parts, A.CONDITION, direct, common)
    assert result['mass'] == 7 and result['n_query'] == 48 and result['n_key'] == 432
    assert result['kl_sum'] == result['direct_kl_sum']
    assert result['js_sum'] == result['direct_js_sum']
    with pytest.raises(ValueError): A.target_counts(dict(gt_xy=xy, gt_visible=visible), parts, 'g8', 'g8')
    with pytest.raises(ValueError): A.region_metrics(p.astype(np.float32), parts, A.CONDITION, direct, common)


def test_center_pck_eight_pixel_boundary_and_empty_rows():
    bounds = A.BASE.REGION.partition_bounds('g8').astype(np.int32)
    winner = np.zeros(48, np.int32); point = np.full((48, 2), np.nan); valid = np.arange(48) < 3
    point[:3] = [11.5, 3.5]; point[1, 0] = np.nextafter(11.5, np.inf)
    value = A.center_metrics(winner, bounds, point, valid)
    assert value['pck_count'] == 2 and value['n_point'] == 3 and value['n_query'] == 48
    assert value['distance_sum'] >= 24.
    empty = A.center_metrics(winner, bounds, point, np.zeros(48, bool))
    assert empty['n_point'] == 0 and empty['distance_sum'] == 0 and empty['mean'] is None and empty['rmse'] is None


def test_primary_and_independent_summary_and_all_eight_checks_agree(monkeypatch):
    from experiments import round25_student_evaluate as producer
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('synthetic cross-check must not load archives'))
    entries, rows = rows_fixture()
    independent = A.summarize(rows, entries)
    assert producer.primary_summary(rows, entries) == independent
    baseline = baselines(independent)
    candidate = A.BASE.exact_fraction(independent['center']['macro']['pck'])
    # Include one exact PCK failure too small to survive a float-only decision.
    role = A.SAME_A_CENTER_ROLES[-1]
    baseline['by_role'][role]['center']['by_condition']['g24_g8']['pck'] = A.BASE.fraction_dict(
        candidate + Fraction(1, 200) + Fraction(1, 10**25))
    audited = A.screen(independent, baseline)
    fields = ('reference', 'metric', 'candidate_value', 'reference_value', 'bound', 'passed')
    expected = dict(passed=audited['passed'], checks=[{k: c[k] for k in fields} for c in audited['checks']])
    assert producer.primary_checks(independent, baseline) == expected
    assert not expected['passed'] and sum(not c['passed'] for c in expected['checks']) == 1


def test_primary_and_independent_target_and_metric_routes_agree_on_synthetic_pixels(monkeypatch):
    from experiments import round25_student_evaluate as producer
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('synthetic cross-check must not load archives'))
    parts = A.BASE.REGION.literal_parts()
    y, x = np.mgrid[:144, :192]
    xy = np.stack([x, y], axis=-1).reshape(-1, 2).astype(np.float64)
    visible = np.arange(27648) % 7 == 0
    xy[~visible] = np.nan
    labels = dict(gt_xy=xy, gt_visible=visible)
    primary = producer.M.target_counts(labels, parts, 'g24', 'g8')
    independent = A.target_counts(labels, parts)
    for a, b in zip(primary, independent):
        np.testing.assert_array_equal(a, b)
    weights = np.arange(1, 433, dtype=np.float64)
    p = np.tile(weights / weights.sum(), (48, 1))
    first = producer.M.region_metrics(p, parts, A.CONDITION, *primary)
    second = A.region_metrics(p, parts, A.CONDITION, *independent)
    assert producer.discrepancy(first, second) < 1e-9
    winner = np.full(48, 431, np.int32)
    point = np.tile([188., 140.], (48, 1))
    valid = np.arange(48) % 3 != 0
    first = producer.M.center_metrics(winner, parts['g8']['bounds'], point, valid)
    second = A.center_metrics(winner, parts['g8']['bounds'], point, valid)
    assert producer.discrepancy(first, second) < 1e-9
