"""Pure synthetic R20 metric populations and exact frozen-screen boundaries."""
from copy import deepcopy
from fractions import Fraction

import numpy as np
import pytest

from experiments import round20_anchor_audit_metrics as A
from experiments import round20_anchor_fixed_metrics as P


def baseline(kl=1., pck=Fraction(4, 5)):
    return dict(macro=dict(common9_kl=kl, direct5_kl=kl, J=kl),
        by_condition={c['condition']: dict(kl=kl) for c in A.CONDITIONS},
        center=dict(by_condition={q + '_g8': dict(pck=A.fraction_dict(pck)) for q in ('g8', 'g16', 'g24')}))


def screen_case():
    by_role = {name: baseline() for name in ('anchor_trace', 'anchor_point_regional',
        'anchor_quartic_regional', 'anchor_fixed_regional', 'anchor_point_localizer',
        'anchor_quartic_localizer', 'anchor_point_legacy')}
    by_role['anchor_trace'] = baseline(.98, Fraction(159, 200))
    old = {name: baseline() for name in ('strong_point', 'legacy_point', 'strong_quad',
        'legacy_quad16', 'point_localizer', 'quartic_localizer', 'legacy_point_localizer')}
    return by_role, old


def test_all_sixty_checks_and_exact_half_percentage_point_boundary():
    new, old = screen_case()
    expected = A.screen(new, old)
    assert expected == P.screen(new, old)
    assert len(expected['checks']) == 60 and expected['passed']
    for group, counts in (('old_macro', 8), ('old_condition', 28), ('same_a_macro', 6),
                          ('old_center', 9), ('same_a_center', 9)):
        assert sum(c['category'] == group for c in expected['checks']) == counts
    changed = deepcopy(new)
    changed['anchor_trace']['center']['by_condition']['g24_g8']['pck'] = A.fraction_dict(Fraction(159, 200) - Fraction(1, 10**12))
    result = A.screen(changed, old)
    assert not result['passed']
    assert sum(not c['passed'] for c in result['checks']) == 6


@pytest.mark.parametrize('category,role,metric,bound', [
    ('old_macro', 'strong_point', 'common9_kl', .98),
    ('old_macro', 'strong_quad', 'common9_kl', 1.01),
    ('old_macro', 'legacy_quad16', 'direct5_kl', 1.02),
    ('same_a_macro', 'anchor_quartic_regional', 'common9_kl', .98)])
def test_region_multiplier_boundaries_have_no_hidden_epsilon(category, role, metric, bound):
    new, old = screen_case()
    new['anchor_trace']['macro'][metric] = bound
    find = lambda result: next(c for c in result['checks'] if c['category'] == category and c['reference'] == role and c['metric'] == metric)
    assert find(A.screen(new, old))['passed']
    new['anchor_trace']['macro'][metric] = np.nextafter(bound, np.inf)
    assert not find(A.screen(new, old))['passed']


def test_every_condition_is_checked_and_nonpositive_reference_cannot_pass():
    new, old = screen_case()
    condition = 'g24_g24'
    new['anchor_trace']['by_condition'][condition]['kl'] = np.nextafter(1.02, np.inf)
    failed = [c for c in A.screen(new, old)['checks'] if not c['passed']]
    assert len(failed) == 2 and all(c['category'] == 'old_condition' for c in failed)
    old['legacy_point']['macro']['common9_kl'] = 0.
    assert not next(c for c in A.screen(new, old)['checks'] if c['name'] == 'common9_kl_vs_legacy_point')['passed']


def synthetic_rows():
    entries = [dict(scene=s, pair=[2*i, 2*i+1]) for s in A.SCENES for i in range(16)]
    rows, centers = [], []
    for index, entry in enumerate(entries):
        scene_index = A.SCENES.index(entry['scene'])
        mass = 100 * (scene_index + 1)
        for ci, condition in enumerate(A.CONDITIONS):
            row = dict(entry, **condition, mass=mass, n_query=A.COUNTS[condition['query']],
                       n_key=A.COUNTS[condition['key']])
            for mi, metric in enumerate(A.METRICS):
                row[metric + '_sum'] = mass * (1 + scene_index + ci / 100 + mi / 10)
            rows.append(row)
            query = condition['query']
            if query != 'mixed':
                total = A.VALID_COUNTS[query]
                valid = total // 48 + int(index < total % 48)
                centers.append(dict(entry, condition=condition['condition'], n_query=A.COUNTS[query],
                    pck_count=valid - 1, n_point=valid, distance_sum=valid * 2.,
                    distance_squared_sum=valid * 5., mean=2., rmse=np.sqrt(5.)))
    conditions = tuple(c['condition'] for c in A.CONDITIONS if c['query'] != 'mixed')
    return entries, rows, centers, conditions


def test_complete_population_equal_scene_ratios_and_original_center_totals():
    entries, rows, centers, conditions = synthetic_rows()
    independent = A.summarize_role(rows, centers, entries, center_conditions=conditions)
    assert independent == P.summarize_role(rows, centers)
    assert independent['by_condition']['g8_g8']['kl'] == 2.
    assert independent['center']['by_condition']['g8_g8']['rmse'] == pytest.approx(np.sqrt(5.))
    for bad in (rows[:-1], rows[:-1] + [rows[0]]):
        with pytest.raises(ValueError):
            A.summarize_role(bad, centers, entries, center_conditions=conditions)
    changed = deepcopy(centers)
    changed[0]['n_point'] -= 1
    with pytest.raises(ValueError):
        A.summarize_role(rows, changed, entries, center_conditions=conditions)


def test_original_dense_counts_common_projection_and_zero_mass_rows():
    parts = A.REGION.literal_parts()
    y, x = np.mgrid[:144, :192]
    xy = np.stack([x, y], axis=-1).reshape(-1, 2).astype(np.float64)
    visible = np.zeros(27648, dtype=bool)
    visible[:192] = True
    labels = dict(gt_xy=xy, gt_visible=visible)
    condition = next(c for c in A.CONDITIONS if c['condition'] == 'mixed_g16')
    direct, common = A.REGION.independent_counts(labels, parts, 'mixed', 'g16')
    d2, c2 = P.target_counts(labels, parts, 'mixed', 'g16')
    np.testing.assert_array_equal(direct, d2); np.testing.assert_array_equal(common, c2)
    p = np.full((243, 108), 1 / 108, dtype=np.float64)
    independent = A.region_metrics(p, parts, condition, direct, common)
    production = P.region_metrics(p, parts, condition, direct, common)
    assert independent['mass'] == 192
    for key in independent:
        assert independent[key] == pytest.approx(production[key], abs=1e-8)


def test_native_argmax_center_distance_boundary_and_valid_population():
    bounds = A.REGION.partition_bounds('g8')
    winners = np.zeros(48, np.int32)
    point = np.tile([11.5, 3.5], (48, 1)).astype(np.float64)
    valid = np.arange(48) < 3
    point[1, 0] = np.nextafter(11.5, np.inf)
    point[~valid] = np.nan
    result = A.center_metrics(winners, bounds, point, valid)
    assert result['pck_count'] == 2 and result['n_point'] == 3
    assert result == P.center_metrics(winners, bounds, point, valid)


def test_fraction_encoding_cannot_be_rounded_or_noncanonical():
    with pytest.raises(ValueError):
        A.exact_fraction(dict(numerator=2, denominator=4, value=.5))
    with pytest.raises(ValueError):
        A.exact_fraction(dict(numerator=1, denominator=3, value=.3333333))
