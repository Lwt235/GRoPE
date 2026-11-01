from copy import deepcopy

import pytest

from experiments.round15_feasibility_audit import (
    SCENES, QUERIES, FAMILIES, GRID, independent_center_selection, verify_constraint_table,
)


def table():
    return [dict(family=f, h=h, temperature=t, scene=s, query=q, pair=[1, 2],
                 correct=900, valid=1000)
            for f in FAMILIES for h, t in GRID for s in SCENES for q in QUERIES]


def test_global_point_choice_and_ordered_ties():
    rows = table()
    for r in rows:
        if r['family'] == 'quartic_point' and (r['h'], r['temperature']) == (1., 5.):
            r['correct'] = 910 if r['query'] == 'g24' else 900
    result = independent_center_selection(rows, expected_pairs=1)
    assert result['point_configurations'] == {'gaussian_point': (.25, 0.), 'quartic_point': (1., 5.)}
    assert result['stop_candidate'] and not result['regional_primary_selected']


def test_legacy_scale_protection_survives_a_new_point_selection():
    rows = table()
    for r in rows:
        if r['family'] == 'gaussian_point' and (r['h'], r['temperature']) == (.5, 5.):
            r['correct'] = 920 if r['query'] == 'g8' else 880
    result = independent_center_selection(rows, expected_pairs=1)
    assert result['point_configurations']['gaussian_point'] == (.25, 0.)
    assert result['stop_candidate']
    assert any(c['reference'] == 'legacy_point_pck' and not c['passed'] for c in result['checks'])


def test_equal_scene_not_pooled_and_strict_population_checks():
    rows = table()
    for r in rows:
        if r['scene'] == SCENES[0]:
            r['valid'], r['correct'] = 100, 50
    result = independent_center_selection(rows, expected_pairs=1)
    assert result['scores']['quartic_full', (.25, 0.)]['by_query']['g8'] == pytest.approx((.5+.9+.9)/3)
    assert len(result['eligible']) == 12
    with pytest.raises(ValueError, match='duplicate'):
        independent_center_selection(rows+[rows[0]], expected_pairs=1)
    rows[0]['valid'] = 101
    with pytest.raises(ValueError, match='population'):
        independent_center_selection(rows, expected_pairs=1)


def test_no_tolerance_beyond_the_declared_half_point_guard():
    rows = table()
    for r in rows:
        if r['family'] == 'quartic_full':
            r['correct'] = 894
    result = independent_center_selection(rows, expected_pairs=1)
    assert result['eligible'] == [] and result['stop_candidate']
    rows[0]['correct'] = 1001
    with pytest.raises(ValueError, match='counts'):
        independent_center_selection(rows, expected_pairs=1)


def test_exact_half_point_boundary_uses_integer_ratios():
    rows = table()
    for row in rows:
        if row['family'] == 'quartic_full':
            row['correct'] = 895
    result = independent_center_selection(rows, expected_pairs=1)
    assert len(result['eligible']) == 12
    assert all(c['passed'] and c['difference_pp'] == -.5 for c in result['checks'])


def test_constraint_audit_rejects_missing_and_duplicated_failed_guards():
    rows = table()
    for row in rows:
        if row['family'] == 'quartic_full':
            row['correct'] = 890
    result = independent_center_selection(rows, expected_pairs=1)
    references = {'selected_gaussian_point': 'gaussian_point',
                  'selected_quartic_point': 'quartic_point', 'legacy_point_pck': 'legacy_point_pck'}
    saved = []
    for i, (h, temperature) in enumerate(GRID):
        guards = [dict(query=c['query'], reference=references[c['reference']],
                       passed=c['passed'], difference_pp=c['difference_pp'])
                  for c in result['checks'] if (c['h'], c['temperature']) == (h, temperature)]
        saved.append(dict(grid_index=i, config=dict(h=h, temperature=temperature),
                          feasible=False, checks=guards))
    verify_constraint_table(saved, result['checks'], [])
    with pytest.raises(ValueError, match='configuration guards'):
        verify_constraint_table(saved[:-1], result['checks'], [])
    duplicate = deepcopy(saved)
    duplicate[0]['checks'][1] = duplicate[0]['checks'][0]
    with pytest.raises(ValueError, match='duplicate'):
        verify_constraint_table(duplicate, result['checks'], [])
    omitted = deepcopy(saved)
    omitted[0]['checks'].pop()
    with pytest.raises(ValueError, match='configuration guards'):
        verify_constraint_table(omitted, result['checks'], [])
