"""Synthetic exact populations, paired transitions and equal-scene reductions."""
from copy import deepcopy
from fractions import Fraction
import math
import numpy as np
import pytest
from experiments import round21_matched_decomposition_audit_metrics as A


def population():
    entries = [dict(scene=s, pair=[i, i + 1]) for s in A.SCENES for i in range(16)]
    rows, centers, paired = [], [], []
    declaration = A.CONTRASTS[4]
    for ordinal, entry in enumerate(entries):
        level = ordinal // 16 + 1
        for c in A.CONDITIONS:
            q = c['query']; n = A.COUNTS[q]
            valid = A.VALID_COUNTS[q] // 48 + (ordinal < A.VALID_COUNTS[q] % 48)
            cm = dict(n_query=n, n_point=valid, pck_count=valid // 2,
                      distance_sum=float(valid * level), distance_squared_sum=float(valid * level ** 2),
                      mean=float(level), rmse=float(level))
            mass = ordinal % 16 + 1
            base = dict(entry, **c, n_query=n, n_key=432, mass=mass)
            rows.append(dict(base, **{m + '_sum': float(mass * level) for m in A.METRICS}))
            centers.append(dict(entry, condition=c['condition'], **cm))
            pc = dict(n_query=n, n_point=valid, from_correct=0, to_correct=1, both_correct=0,
                      both_wrong=valid - 1, benefit=1, harm=0, flip_valid=1, flip_all=1,
                      distance_delta_sum=-1., squared_distance_delta_sum=-3.,
                      from_distance_squared_sum=float(4 * valid), to_distance_squared_sum=float(4 * valid - 3),
                      rmse_delta=math.sqrt((4 * valid - 3) / valid) - 2,
                      distance_benefit=1, distance_harm=0, distance_equal=valid - 1)
            paired.append(dict(base, contrast=declaration['name'], from_role=declaration['from_role'],
                to_role=declaration['to_role'], **{m + '_delta_sum': float(-mass * level) for m in A.METRICS}, center=pc))
    return entries, rows, centers, paired


def test_three_native_conditions_ten_contrasts_without_scientific_selection():
    assert len(A.CONDITIONS) == 3 and len(A.CONTRASTS) == 10 and len(A.FAMILIES) == 8
    assert [c['condition'] for c in A.CONDITIONS] == ['g8_g8', 'g16_g8', 'g24_g8']
    assert not hasattr(A, 'screen')


def test_role_macro_is_equal_scene_and_scale_with_exact_fraction():
    entries, rows, centers, _ = population()
    result = A.summarize_role(rows, centers, entries)
    assert result['macro'] == {m: 2. for m in A.METRICS}
    assert result['center']['macro']['mean'] == 2. and result['center']['macro']['rmse'] == 2.
    expected = sum((Fraction(sum(r['pck_count'] for r in centers if r['scene'] == s and r['condition'] == c['condition']),
                             sum(r['n_point'] for r in centers if r['scene'] == s and r['condition'] == c['condition']))
                    for s in A.SCENES for c in A.CONDITIONS), Fraction()) / 9
    assert A.exact_fraction(result['center']['macro']['pck']) == expected
    assert expected != Fraction(sum(r['pck_count'] for r in centers), sum(r['n_point'] for r in centers))


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'scene', 'count', 'valid', 'negative'])
def test_role_rejects_changed_population_or_divergence(change):
    entries, rows, centers, _ = population()
    if change == 'missing': rows.pop()
    elif change == 'duplicate': rows[-1] = deepcopy(rows[0])
    elif change == 'scene': rows[-1]['scene'] = 'unselected_scene'
    elif change == 'count': rows[0]['n_query'] -= 1
    elif change == 'valid': centers[0]['n_point'] -= 1
    else: rows[0]['js_sum'] = -1e-3
    with pytest.raises(ValueError): A.summarize_role(rows, centers, entries)


def test_paired_boundary_transition_and_invalid_rows_are_retained():
    bounds = np.array([[0, 0, 1, 1], [16, 0, 17, 1]], np.int64)
    before, after = np.array([0, 0, 1, 0], np.int32), np.array([1, 1, 0, 1], np.int32)
    point = np.array([[16., 0.], [8., 0.], [16., 0.], [np.nan, np.nan]])
    valid = np.array([True, True, True, False])
    result = A.paired_center(before, after, bounds, point, valid)
    assert (result['benefit'], result['harm'], result['both_correct'], result['both_wrong']) == (1, 1, 1, 0)
    assert result['flip_valid'] == 3 and result['flip_all'] == 4
    assert result['distance_delta_sum'] == result['squared_distance_delta_sum'] == result['rmse_delta'] == 0
    empty = A.paired_center(before, after, bounds, point, np.zeros(4, bool))
    assert empty['n_point'] == 0 and empty['rmse_delta'] is None and empty['flip_all'] == 4


def test_continuous_order_uses_squared_distance_before_sqrt_rounding():
    bounds = np.array([[0, 0, 1, 1], [1, 0, 2, 1]], np.int64)
    result = A.paired_center(np.array([0], np.int32), np.array([1], np.int32), bounds,
                             np.array([[.125, 7e7]]), np.array([True]))
    assert result['distance_delta_sum'] == 0. and result['squared_distance_delta_sum'] == 1.
    assert (result['distance_benefit'], result['distance_harm'], result['distance_equal']) == (0, 1, 0)


def test_rmse_difference_after_aggregate_and_exact_paired_pck():
    entries, _, _, rows = population(); result = A.summarize_contrast(rows, entries, A.CONTRASTS[4])
    assert result['macro'] == {m + '_delta': -2. for m in A.METRICS}
    scene, cond = A.SCENES[2], 'g24_g8'  # This scene includes both 22- and 23-valid-row pairs.
    samples = [r['center'] for r in rows if r['scene'] == scene and r['condition'] == cond]
    cell = result['center']['by_scene_condition'][scene][cond]
    n = sum(r['n_point'] for r in samples)
    assert A.exact_fraction(cell['pck_delta']) == Fraction(16, n)
    assert cell['rmse_delta'] == math.sqrt(sum(r['to_distance_squared_sum'] for r in samples) / n) - 2.
    assert cell['rmse_delta'] != sum(r['rmse_delta'] for r in samples) / 16


def test_difference_of_differences_has_no_invented_paired_transitions():
    entries, _, _, rows = population(); template = A.summarize_contrast(rows, entries, A.CONTRASTS[4])
    contrasts = {c['name']: dict(deepcopy(template), from_role=c['from_role'], to_role=c['to_role']) for c in A.CONTRASTS}
    result = A.interactions(contrasts)
    assert set(result) == {'anchor', 'widening', 'shape'}
    for item in result.values():
        assert item['macro'] == {m + '_delta': 0. for m in A.METRICS}
        assert A.exact_fraction(item['center']['macro']['pck_delta']) == 0
        cell = item['center']['by_scene_condition'][A.SCENES[0]]['g8_g8']
        assert cell['pck_delta_count'] == 0 and cell['rmse_delta'] == 0.
        assert 'benefit' not in cell and 'harm' not in cell and 'from_distance_squared_sum' not in cell
    bad = deepcopy(contrasts); bad['anchor_content']['by_scene_condition'][A.SCENES[0]]['g8_g8']['mass'] += 1
    with pytest.raises(ValueError, match='mass'): A.interactions(bad)


def test_synthetic_schema_matches_separately_implemented_producer_reductions():
    from experiments import round21_matched_decomposition_metrics as P
    entries, rows, centers, paired = population()
    combined = [dict(r, center={k: v for k, v in c.items() if k not in ('scene', 'pair', 'condition')}) for r, c in zip(rows, centers)]
    from experiments.round20_anchor_audit import compare
    compare(A.summarize_role(rows, centers, entries), P.summarize(combined), 1e-12)
    independent = A.summarize_contrast(paired, entries, A.CONTRASTS[4])
    expected = dict(from_role=A.CONTRASTS[4]['from_role'], to_role=A.CONTRASTS[4]['to_role'], **P.summarize(paired, paired=True))
    compare(independent, expected, 1e-12)
