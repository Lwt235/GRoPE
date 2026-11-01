"""Synthetic only: no source archives, labels, effects or selected configs."""
from copy import deepcopy
import pytest

from experiments import round17_validation_metrics as M


def role_manifest():
    specs = {alias: dict(family=family, h=.5, temperature=5.)
             for alias, family in M.FAMILY_BY_ALIAS.items()}
    specs['primary'].update(h=1., temperature=0.)
    specs['matched_isotropic'].update(h=1., temperature=0.)
    specs['matched_constant_temperature'].update(h=1., temperature=0.)
    specs['localizer_gaussian_point'].update(h=1.)
    for role, (family, h, temperature) in M.LEGACY.items():
        specs['legacy_'+role] = dict(family=family, h=h, temperature=temperature)
    return M.build_roles(specs, 'regional_gaussian_point', 'regional_gaussian_quad4')


def test_matched_constant_temperature_cannot_recalibrate():
    roles=role_manifest()
    specs={a:dict(roles['operators'][o]) for a,o in roles['aliases'].items()}
    specs['matched_constant_temperature']['temperature']=5.
    with pytest.raises(ValueError,match='matched_constant_temperature'):
        M.build_roles(specs,roles['strong_point'],roles['strong_quad'])


def cohort():
    return [dict(scene=scene, pair=[2*i, 2*i+1]) for scene in M.VAL_SCENES for i in range(16)]


def rows_for(roles, entries):
    rows = []
    primary = roles['aliases']['primary']
    for operator, config in roles['operators'].items():
        kl = .8 if operator == primary else 1.
        for entry in entries:
            for scenario, conditions in M.CONDITIONS.items():
                for q, k in conditions:
                    rows.append(dict(operator=operator, **entry, scenario=scenario,
                        condition=q+'_'+k, query=q, key=k, config=dict(config),
                        kl_sum=1000*kl, js_sum=100., direct_kl_sum=1000*kl,
                        direct_js_sum=100., mass=1000, pck_count=20, n_point=25,
                        n_query=M.COUNTS[q], n_key=M.COUNTS[k],
                        numerical=dict(all_finite=True, max_normalization_error=0.)))
    return rows


def timing_for(roles):
    entries = [dict(scene=scene, pair=[0, 1]) for scene in M.TIMING_SCENES]
    rows = []
    for operator in roles['operators']:
        seconds = .8 if operator == roles['aliases']['primary'] else 1.
        for entry in entries:
            for q, k in M.ORIGINAL:
                rows.append(dict(**entry, operator=operator, query=q, key=k,
                    condition=q+'_'+k, samples_seconds=[seconds]*15, warmup_seconds=[seconds]*3))
    return entries, rows


@pytest.fixture(scope='module')
def fixture():
    roles = role_manifest()
    entries = cohort()
    rows = rows_for(roles, entries)
    timing_entries, timing_rows = timing_for(roles)
    metrics = M.summarize(rows, entries, roles)
    timing = M.aggregate_timing(timing_rows, timing_entries, roles)
    return roles, entries, rows, metrics, timing_entries, timing_rows, timing


def set_metric(rows, predicate, value):
    for row in rows:
        if predicate(row):
            row['kl_sum'] = value*row['mass']
            if row['scenario'] == 'scale':
                row['direct_kl_sum'] = row['kl_sum']


def test_roles_exact19_aliases_first_seen_dedup_and_isotropic_separate():
    roles = role_manifest()
    assert list(roles['aliases']) == list(M.ALIAS_ORDER) and len(M.ALIAS_ORDER) == 19
    assert list(roles['operators']) == [f'o{i:02d}' for i in range(len(roles['operators']))]
    assert roles['aliases']['legacy_point_pck'] == roles['aliases']['regional_gaussian_point']
    assert roles['aliases']['localizer_gaussian_point'] == roles['aliases']['legacy_point']
    assert roles['aliases']['matched_isotropic'] != roles['aliases']['primary']
    assert 'strong_point' not in roles['aliases'] and 'strong_quad' not in roles['aliases']
    assert M.validate_roles(roles) == roles


def test_roles_reject_legacy_changes_isotropic_retuning_or_noncanonical_ids():
    roles = role_manifest()
    specs = {a: dict(roles['operators'][o]) for a, o in roles['aliases'].items()}
    changed = deepcopy(specs)
    changed['legacy_point']['h'] = .5
    with pytest.raises(ValueError, match='legacy'):
        M.build_roles(changed, roles['strong_point'], roles['strong_quad'])
    changed = deepcopy(specs)
    changed['matched_isotropic']['temperature'] = 5.
    with pytest.raises(ValueError, match='isotropic'):
        M.build_roles(changed, roles['strong_point'], roles['strong_quad'])
    changed = deepcopy(roles)
    changed['operators']['o99'] = changed['operators'].pop('o00')
    changed['aliases']['primary'] = 'o99'
    with pytest.raises(ValueError, match='canonical'):
        M.validate_roles(changed)
    with pytest.raises(ValueError, match='regional point'):
        M.build_roles(specs, 'legacy_point', roles['strong_quad'])


def test_population_complete68_native_conditions_and_shared_counts(fixture):
    roles, entries, rows, *_ = fixture
    population = M.validate_population(rows, entries, roles)
    assert population['records'] == 2176*len(roles['operators'])
    assert sum(len(c) for c in M.CONDITIONS.values()) == 68
    with pytest.raises(ValueError, match='population'):
        M.validate_population(rows[:-1], entries, roles)
    duplicate = list(rows)
    duplicate[-1] = rows[0]
    with pytest.raises(ValueError, match='duplicate'):
        M.validate_population(duplicate, entries, roles)


@pytest.mark.parametrize('mutation,match', [
    (lambda row: row.update(mass=999), 'pixel mass'),
    (lambda row: row.update(n_point=24), 'center population'),
    (lambda row: row.update(n_query=108), 'query/key'),
    (lambda row: row['config'].update(h=2.), 'operator/config'),
    (lambda row: row.update(pck_count=26), 'pck_count'),
    (lambda row: row.update(kl_sum=float('nan')), 'finite numeric'),
    (lambda row: row['numerical'].update(all_finite=False), 'finite check'),
    (lambda row: row['numerical'].update(max_normalization_error=1.1e-12), 'normalization'),
])
def test_population_rejects_changed_supervision_config_and_bad_probabilities(fixture, mutation, match):
    roles, entries, rows, *_ = fixture
    changed = list(rows)
    changed[0] = deepcopy(changed[0])
    mutation(changed[0])
    with pytest.raises(ValueError, match=match):
        M.validate_population(changed, entries, roles)


def test_scale_uses_direct_metrics_and_cohort_is_fixed(fixture):
    roles, entries, rows, *_ = fixture
    changed = list(rows)
    index = next(i for i, row in enumerate(rows) if row['scenario'] == 'scale')
    changed[index] = dict(changed[index], direct_kl_sum=5.)
    with pytest.raises(ValueError, match='direct native'):
        M.validate_population(changed, entries, roles)
    wrong_entries = deepcopy(entries)
    wrong_entries[0]['scene'] = M.TIMING_SCENES[0]
    with pytest.raises(ValueError, match='scenes'):
        M.validate_population(rows, wrong_entries, roles)


def test_scene_and_condition_macros_do_not_pool_unequal_scene_masses(fixture):
    roles, entries, rows, *_ = fixture
    changed = []
    for row in rows:
        row = deepcopy(row)
        si = M.VAL_SCENES.index(row['scene'])
        ci = M.CONDITIONS[row['scenario']].index((row['query'], row['key']))
        value = 1+2*si+ci/10
        row['mass'] = (100, 1000)[si]
        row['kl_sum'] = value*row['mass']
        row['direct_kl_sum'] = (value if row['scenario'] == 'scale' else value+5)*row['mass']
        changed.append(row)
    metrics = M.summarize(changed, entries, roles)
    assert metrics['zero']['primary']['kl'] == pytest.approx(2.4)
    assert metrics['zero']['primary']['direct_kl'] == pytest.approx(7.4)
    assert metrics['scale']['primary']['kl'] == pytest.approx(2.2)
    assert metrics['zero']['primary']['by_scene'][M.VAL_SCENES[0]] == pytest.approx(1.4)


def test_all_gates_pass_but_do_not_authorize_general_or_fresh_claims(fixture):
    roles, _, _, metrics, _, _, timing = fixture
    result = M.gates(metrics, roles, timing)
    assert result['passed']
    assert result['new_q16_protection']['new_predeclared_guard']
    assert not result['new_q16_protection']['historical_R12_gate']
    assert not result['general_research_goal_achieved']
    assert not result['fresh_confirmation_authorized']
    assert result['isotropic_is_attribution_only']


def test_pck_equal_scene_fraction_does_not_pool_unequal_visible_counts(fixture):
    roles, entries, rows, *_ = fixture
    changed = []
    for row in rows:
        n_point, correct = ((20, 10) if row['scene'] == M.VAL_SCENES[0] else (40, 36))
        changed.append(dict(row, n_point=n_point, pck_count=correct))
    metrics = M.summarize(changed, entries, roles)
    pck = metrics['zero']['primary']['pck_by_condition']['g16_g8']
    assert pck['exact'] == dict(numerator=7, denominator=10)
    assert pck['macro'] == .7
    assert pck['macro'] != (10+36)/(20+40)


def test_q16_exact_half_pp_boundary_and_one_count_outside(fixture):
    roles, entries, rows, _, _, _, timing = fixture
    changed = deepcopy(rows)
    op = roles['aliases']['primary']
    for row in changed:
        if (row['operator'] == op and row['scenario'] == 'zero'
                and row['condition'] == 'g16_g8' and row['pair'] == [0, 1]):
            row['pck_count'] = 18  # Each scene: 318/400 versus 320/400.
    result = M.gates(M.summarize(changed, entries, roles), roles, timing)
    assert result['passed']
    comparisons = result['new_q16_protection']['comparisons']
    assert set(comparisons) == set(M.LOCALIZERS)
    assert all(v['exact_difference'] == {'numerator': -1, 'denominator': 200} for v in comparisons.values())
    next(row for row in changed if row['operator'] == op and row['scenario'] == 'zero'
         and row['condition'] == 'g16_g8' and row['pair'] == [0, 1])['pck_count'] = 17
    result = M.gates(M.summarize(changed, entries, roles), roles, timing)
    assert not result['passed'] and not result['new_q16_protection']['passed']
    assert result['original']['passed'] and result['scale']['passed']


def test_pck_rejects_float_summary_or_fraction_tampering(fixture):
    roles, _, _, metrics, _, _, timing = fixture
    changed = deepcopy(metrics)
    changed['zero']['primary']['pck_by_condition']['g8_g8']['macro'] += 1.e-12
    with pytest.raises(ValueError, match='integer counts'):
        M.gates(changed, roles, timing)


@pytest.mark.parametrize('alias,scenario,condition', [
    ('localizer_gaussian_point', 'zero', 'g8_g8'),
    ('localizer_quartic_point', 'scale', 'g24_g8'),
    ('legacy_point_pck', 'depth11', 'g8_g8'),
])
def test_each_localizer_and_each_scope_can_independently_fail(fixture, alias, scenario, condition):
    roles, entries, rows, _, _, _, timing = fixture
    changed = deepcopy(rows)
    op = roles['aliases'][alias]
    for row in changed:
        if row['operator'] == op and row['scenario'] == scenario and row['condition'] == condition:
            row['pck_count'] = 21
    result = M.gates(M.summarize(changed, entries, roles), roles, timing)
    assert not result['passed']
    section = (result['original']['pck_g8'] if scenario == 'zero' else
               result['scale']['pck_g24'] if scenario == 'scale' else result['stress'][scenario]['pck_g8'])
    assert not section['comparisons'][alias]['passed']


def test_legacy_point_and_quad_remain_mandatory_when_new_references_pass(fixture):
    roles, entries, rows, _, _, _, timing = fixture
    changed = deepcopy(rows)
    point = roles['aliases']['legacy_point']
    quad = roles['aliases']['legacy_quad16']
    set_metric(changed, lambda r: r['operator'] == point, .81)
    set_metric(changed, lambda r: r['operator'] == quad, .79)
    result = M.gates(M.summarize(changed, entries, roles), roles, timing)
    assert not result['passed']
    assert result['original']['point']['comparisons'][roles['strong_point']]['passed']
    assert not result['original']['point']['comparisons']['legacy_point']['passed']
    assert result['original']['accuracy_cost']['comparisons'][roles['strong_quad']]['passed']
    assert not result['original']['accuracy_cost']['comparisons']['legacy_quad16']['passed']


def test_both_scenes_point_gain_required_even_when_macro_improves(fixture):
    roles, entries, rows, _, _, _, timing = fixture
    changed = deepcopy(rows)
    op = roles['aliases']['primary']
    set_metric(changed, lambda r: r['operator'] == op and r['scene'] == M.VAL_SCENES[0], .5)
    set_metric(changed, lambda r: r['operator'] == op and r['scene'] == M.VAL_SCENES[1], 1.)
    result = M.gates(M.summarize(changed, entries, roles), roles, timing)
    comparison = result['original']['point']['comparisons'][roles['strong_point']]
    assert comparison['checks']['relative_gain_at_least_2pct']
    assert not comparison['checks']['both_scenes_improve']
    assert not result['passed']


def test_stress_guard_is_nine_condition_macro_not_new_per_condition_gate(fixture):
    roles, entries, rows, _, _, _, timing = fixture
    changed = deepcopy(rows)
    set_metric(changed, lambda r: r['operator'] == roles['aliases']['primary']
               and r['scenario'] == 'ry_plus1' and r['condition'] == 'g16_g16', 1.10)
    metrics = M.summarize(changed, entries, roles)
    assert metrics['ry_plus1']['primary']['by_condition']['g16_g16'] > 1.02
    assert M.gates(metrics, roles, timing)['passed']
    set_metric(changed, lambda r: r['operator'] == roles['aliases']['primary']
               and r['scenario'] == 'ry_plus1', 1.03)
    result = M.gates(M.summarize(changed, entries, roles), roles, timing)
    assert not result['passed'] and not result['stress']['ry_plus1']['passed']


def test_each_original_and_scale_condition_has_quad_guard(fixture):
    roles, entries, rows, _, _, _, timing = fixture
    for scenario, condition in (('zero', 'mixed_mixed'), ('scale', 'g24_g24')):
        changed = deepcopy(rows)
        set_metric(changed, lambda r: r['operator'] == roles['aliases']['primary']
                   and r['scenario'] == scenario and r['condition'] == condition, 1.03)
        result = M.gates(M.summarize(changed, entries, roles), roles, timing)
        section = result['original'] if scenario == 'zero' else result['scale']
        assert not section['partition']['passed'] and not result['passed']


def test_efficiency_route_has_exact_declared_error_and_speed_requirements(fixture):
    roles, entries, rows, _, timing_entries, timing_rows, _ = fixture
    changed = deepcopy(rows)
    for alias in ('regional_gaussian_quad4', 'legacy_quad16'):
        op = roles['aliases'][alias]
        set_metric(changed, lambda r: r['operator'] == op, .8/1.005)
    changed_times = deepcopy(timing_rows)
    for row in changed_times:
        row['samples_seconds'] = ([.5]*15 if row['operator'] == roles['aliases']['primary'] else [1.]*15)
    metrics = M.summarize(changed, entries, roles)
    timing = M.aggregate_timing(changed_times, timing_entries, roles)
    result = M.gates(metrics, roles, timing)
    assert result['passed']
    assert all(v['efficiency_route'] and not v['accuracy_route']
               for v in result['original']['accuracy_cost']['comparisons'].values())
    for row in changed_times:
        if row['operator'] == roles['aliases']['primary']:
            row['samples_seconds'] = [.5001]*15
    timing = M.aggregate_timing(changed_times, timing_entries, roles)
    assert not M.gates(metrics, roles, timing)['original']['accuracy_cost']['passed']


def test_timing_uses_per_cell_median_then_equal45_cell_mean(fixture):
    roles, _, _, _, entries, rows, _ = fixture
    changed = deepcopy(rows)
    operator = roles['aliases']['primary']
    selected = [r for r in changed if r['operator'] == operator]
    for i, row in enumerate(selected):
        row['samples_seconds'] = [.01]*7+[float(i+1)]*8
        row['warmup_seconds'] = [1000.]*3
    timing = M.aggregate_timing(changed, entries, roles)
    assert timing[operator]['mean_seconds'] == 23.
    assert len(timing[operator]['cells']) == 45


@pytest.mark.parametrize('mutation,match', [
    (lambda row: row.update(pair=[7, 9]), 'wrong pair'),
    (lambda row: row.update(samples_seconds=[1.]*14), 'repeat/warmup'),
    (lambda row: row.update(warmup_seconds=[1.]*2), 'repeat/warmup'),
    (lambda row: row['samples_seconds'].__setitem__(0, 0.), 'positive'),
    (lambda row: row['warmup_seconds'].__setitem__(0, -1.), 'positive'),
    (lambda row: row['samples_seconds'].__setitem__(0, float('inf')), 'finite numeric'),
])
def test_timing_rejects_wrong_pair_counts_and_nonfinite_samples(fixture, mutation, match):
    roles, _, _, _, entries, rows, _ = fixture
    changed = deepcopy(rows)
    mutation(changed[0])
    with pytest.raises(ValueError, match=match):
        M.aggregate_timing(changed, entries, roles)


def test_timing_missing_operator_and_probability_check_coverage_rejected(fixture):
    roles, _, _, metrics, entries, rows, timing = fixture
    with pytest.raises(ValueError, match='population'):
        M.aggregate_timing(rows[:-1], entries, roles)
    changed = deepcopy(metrics)
    changed['zero']['primary']['probability_checks']['matrices'] -= 1
    with pytest.raises(ValueError, match='cover all'):
        M.gates(changed, roles, timing)


def test_isotropic_is_reported_but_never_reselected_or_a_new_quality_gate(fixture):
    roles, entries, rows, _, _, _, timing = fixture
    changed = deepcopy(rows)
    set_metric(changed, lambda r: r['operator'] == roles['aliases']['matched_isotropic'], .1)
    metrics = M.summarize(changed, entries, roles)
    result = M.gates(metrics, roles, timing)
    assert metrics['zero']['matched_isotropic']['kl'] < metrics['zero']['primary']['kl']
    assert result['passed'] and roles['primary'] == 'primary'
