"""Pure old-validation aggregation and inherited gates; no I/O or selection.

This draft consumes already frozen role specifications and metric records.
It never loads data, imports a forward operator, chooses a configuration, or
authorizes an experiment. Source/audit provenance belongs to the caller.
"""
from __future__ import annotations

from fractions import Fraction
import math
from numbers import Integral, Real
from statistics import median


VAL_SCENES = ('harvard_c8/hv_c8_3', 'brown_bm_4/brown_bm_4')
TIMING_SCENES = ('harvard_c3/hv_c3_1', 'harvard_c5/hv_c5_1',
                 'harvard_c6/hv_c6_1', *VAL_SCENES)
COUNTS = dict(g8=432, g16=108, mixed=243, g24=48)
STATES = ('zero', 'depth09', 'depth11', 'ry_plus1', 'ry_minus1',
          'tx_plus01', 'tx_minus01')
ORIGINAL = tuple((q, k) for q in ('g8', 'g16', 'mixed')
                 for k in ('g8', 'g16', 'mixed'))
SCALE = (('g8', 'g24'), ('g24', 'g8'), ('g16', 'g24'),
         ('g24', 'g16'), ('g24', 'g24'))
SCENARIOS = (*STATES, 'scale')
CONDITIONS = {s: (SCALE if s == 'scale' else ORIGINAL) for s in SCENARIOS}
REGIONAL_FAMILIES = ('quartic_point', 'gaussian_point', 'gaussian_fixed',
                    'gaussian_full1', 'gaussian_full4', 'gaussian_quad4',
                    'gaussian_quad16')
LEGACY = dict(point=('gaussian_point', 1., 5.),
              point_pck=('gaussian_point', .5, 5.),
              fixed=('gaussian_fixed', .5, 5.),
              full1=('gaussian_full1', .25, 0.),
              full4=('gaussian_full4', .5, 5.),
              quad4=('gaussian_quad4', .5, 5.),
              quad16=('gaussian_quad16', .5, 5.))
ALIAS_ORDER = ('primary', *(f'regional_{f}' for f in REGIONAL_FAMILIES),
               'localizer_gaussian_point', 'localizer_quartic_point',
               *(f'legacy_{r}' for r in LEGACY), 'matched_constant_temperature', 'matched_isotropic')
FAMILY_BY_ALIAS = dict(primary='trace_full',
    **{f'regional_{f}': f for f in REGIONAL_FAMILIES},
    localizer_gaussian_point='gaussian_point', localizer_quartic_point='quartic_point',
    **{f'legacy_{r}': values[0] for r, values in LEGACY.items()},
    matched_constant_temperature='gaussian_full1', matched_isotropic='trace_isotropic_matched')
LOCALIZERS = ('localizer_gaussian_point', 'localizer_quartic_point', 'legacy_point_pck')
PCK_TOLERANCE = Fraction(1, 200)
METRIC_FIELDS = ('kl_sum', 'js_sum', 'direct_kl_sum', 'direct_js_sum')


def _integer(value, name, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(name + ' must be an integer')
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(name + ' outside permitted range')
    return value


def _finite(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(name + ' must be finite numeric')
    value = float(value)
    if minimum is not None and value < minimum:
        raise ValueError(name + ' below permitted minimum')
    return value


def _configuration(spec):
    if set(spec) != {'family', 'h', 'temperature'}:
        raise ValueError('operator configuration requires family/h/temperature only')
    h = _finite(spec['h'], 'h')
    temperature = _finite(spec['temperature'], 'temperature')
    if h not in (.25, .5, 1., 2.) or temperature not in (0., 5., 10.):
        raise ValueError('operator configuration outside frozen grid')
    return dict(family=spec['family'], h=h, temperature=temperature)


def build_roles(alias_specs, strong_point, strong_quad):
    """Canonical first-seen deduplication of declared specs, never selection.

    The caller must bind one shared frozen numerical/content contract. Family
    identity remains part of deduplication even if particular arrays coincide.
    """
    if set(alias_specs) != set(ALIAS_ORDER):
        raise ValueError('exactly the 19 declared aliases are required')
    if strong_point not in ('regional_gaussian_point', 'regional_quartic_point'):
        raise ValueError('strong_point must reference a regional point alias')
    if strong_quad not in ('regional_gaussian_quad4', 'regional_gaussian_quad16'):
        raise ValueError('strong_quad must reference a regional quadrature alias')
    aliases, operators, lookup = {}, {}, {}
    for alias in ALIAS_ORDER:
        spec = _configuration(alias_specs[alias])
        if spec['family'] != FAMILY_BY_ALIAS[alias]:
            raise ValueError('wrong family for alias ' + alias)
        if alias.startswith('legacy_'):
            family, h, temperature = LEGACY[alias.removeprefix('legacy_')]
            if spec != dict(family=family, h=h, temperature=temperature):
                raise ValueError('legacy configuration changed')
        if alias in ('matched_isotropic','matched_constant_temperature'):
            primary = alias_specs['primary']
            if (spec['h'], spec['temperature']) != (primary['h'], primary['temperature']):
                raise ValueError(alias+' must use primary h/temperature')
        identity = spec['family'], spec['h'], spec['temperature']
        if identity not in lookup:
            operator = f'o{len(operators):02d}'
            lookup[identity] = operator
            operators[operator] = spec
        aliases[alias] = lookup[identity]
    if not 9 <= len(operators) <= 19:
        raise ValueError('unexpected number of unique operators')
    return dict(aliases=aliases, operators=operators, strong_point=strong_point,
                strong_quad=strong_quad, primary='primary')


def validate_roles(roles):
    if set(roles) != {'aliases', 'operators', 'strong_point', 'strong_quad', 'primary'}:
        raise ValueError('invalid role manifest fields')
    if list(roles['aliases']) != list(ALIAS_ORDER):
        raise ValueError('alias declaration order changed')
    try:
        specs = {alias: roles['operators'][operator]
                 for alias, operator in roles['aliases'].items()}
        expected = build_roles(specs, roles['strong_point'], roles['strong_quad'])
    except KeyError as exc:
        raise ValueError('alias references missing operator') from exc
    if roles != expected or list(roles['operators']) != list(expected['operators']):
        raise ValueError('noncanonical operator deduplication or role manifest')
    return roles


def _pair(value):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError('pair must contain exactly two frame indices')
    return tuple(_integer(x, 'frame index') for x in value)


def _cohort(entries, *, timing=False):
    scenes = TIMING_SCENES if timing else VAL_SCENES
    per_scene = 1 if timing else 16
    if len(entries) != len(scenes)*per_scene:
        raise ValueError('cohort size changed')
    identities = [(entry['scene'], _pair(entry['pair'])) for entry in entries]
    if len(set(identities)) != len(identities):
        raise ValueError('duplicate cohort pair')
    if set(s for s, _ in identities) != set(scenes):
        raise ValueError('unexpected cohort scenes')
    if any(sum(s == scene for s, _ in identities) != per_scene for scene in scenes):
        raise ValueError('wrong pair count per scene')
    return identities


def validate_population(rows, entries, roles):
    """Require every unique operator/pair/condition and shared supervision."""
    validate_roles(roles)
    cohort = _cohort(entries)
    expected = {(operator, scene, pair, scenario, q+'_'+k)
                for operator in roles['operators'] for scene, pair in cohort
                for scenario, conditions in CONDITIONS.items() for q, k in conditions}
    if len(rows) != len(expected):
        raise ValueError('incomplete validation record population')
    seen, pixel_populations, center_populations = set(), {}, {}
    for row in rows:
        pair = _pair(row['pair'])
        identity = row['operator'], row['scene'], pair, row['scenario'], row['condition']
        if identity in seen or identity not in expected:
            raise ValueError('duplicate or unauthorized validation identity')
        seen.add(identity)
        q, k = row['query'], row['key']
        if (q, k) not in CONDITIONS[row['scenario']] or row['condition'] != q+'_'+k:
            raise ValueError('condition/query/key mismatch')
        if _configuration(row['config']) != roles['operators'][row['operator']]:
            raise ValueError('row operator/config mismatch')
        nq = _integer(row['n_query'], 'n_query')
        nk = _integer(row['n_key'], 'n_key')
        if (nq, nk) != (COUNTS[q], COUNTS[k]):
            raise ValueError('native query/key population changed')
        mass = _integer(row['mass'], 'mass', maximum=144*192)
        npoint = _integer(row['n_point'], 'n_point', maximum=nq)
        _integer(row['pck_count'], 'pck_count', maximum=npoint)
        for field in METRIC_FIELDS:
            value = _finite(row[field], field)
            if mass == 0 and value != 0:
                raise ValueError('empty regional population has nonzero metric sum')
        if row['scenario'] == 'scale' and any(row[a] != row[b] for a, b in
                (('kl_sum', 'direct_kl_sum'), ('js_sum', 'direct_js_sum'))):
            raise ValueError('scale metric must equal direct native metric')
        pixel_key, center_key = (row['scene'], pair), (row['scene'], pair, q)
        if pixel_populations.setdefault(pixel_key, mass) != mass:
            raise ValueError('roles/states/partitions changed visible pixel mass')
        if center_populations.setdefault(center_key, npoint) != npoint:
            raise ValueError('roles/states/keys changed center population')
        numerical = row['numerical']
        if numerical.get('all_finite') is not True:
            raise ValueError('prediction finite check did not pass')
        error = _finite(numerical['max_normalization_error'], 'normalization error', 0)
        if error > 1.e-12:
            raise ValueError('prediction normalization check did not pass')
    if seen != expected:
        raise ValueError('missing validation identity')
    return dict(pairs=len(cohort), unique_operators=len(roles['operators']),
                records=len(rows), conditions_per_pair=68)


def _mean(values):
    values = list(values)
    if not values:
        raise ValueError('empty macro population')
    return math.fsum(values)/len(values)


def _fraction_dict(value):
    return dict(numerator=value.numerator, denominator=value.denominator)


def _pck_counts(by_scene):
    if set(by_scene) != set(VAL_SCENES):
        raise ValueError('PCK scene population changed')
    fractions = {}
    for scene in VAL_SCENES:
        counts = by_scene[scene]
        denominator = _integer(counts['n_point'], 'PCK denominator', minimum=1)
        numerator = _integer(counts['pck_count'], 'PCK numerator', maximum=denominator)
        fractions[scene] = Fraction(numerator, denominator)
    value = sum(fractions.values(), Fraction(0))/len(VAL_SCENES)
    return value, fractions


def summarize(rows, entries, roles):
    validate_population(rows, entries, roles)
    grouped = {}
    for row in rows:
        identity = row['operator'], row['scenario'], row['scene'], row['condition']
        grouped.setdefault(identity, []).append(row)
    output = {}
    for scenario, conditions in CONDITIONS.items():
        by_operator = {}
        for operator in roles['operators']:
            detail, pck, numerical = {}, {}, []
            for scene in VAL_SCENES:
                detail[scene] = {}
                for q, k in conditions:
                    condition = q+'_'+k
                    selected = grouped[operator, scenario, scene, condition]
                    mass = sum(row['mass'] for row in selected)
                    if mass <= 0:
                        raise ValueError('empty scene/condition regional denominator')
                    detail[scene][condition] = dict(
                        **{field.removesuffix('_sum'): math.fsum(r[field] for r in selected)/mass
                           for field in METRIC_FIELDS}, mass=mass)
                    pck.setdefault(condition, {})[scene] = dict(
                        pck_count=sum(row['pck_count'] for row in selected),
                        n_point=sum(row['n_point'] for row in selected))
                    numerical.extend(row['numerical'] for row in selected)
            pck_by_condition = {}
            for condition, counts in pck.items():
                exact, fractions = _pck_counts(counts)
                pck_by_condition[condition] = dict(macro=float(exact),
                    by_scene={s: float(v) for s, v in fractions.items()},
                    counts_by_scene=counts, exact=_fraction_dict(exact))
            by_scene_condition = {s: {c: v['kl'] for c, v in values.items()} for s, values in detail.items()}
            by_scene = {s: _mean(values.values()) for s, values in by_scene_condition.items()}
            by_condition = {q+'_'+k: _mean(detail[s][q+'_'+k]['kl'] for s in VAL_SCENES) for q, k in conditions}
            by_operator[operator] = dict(
                **{field: _mean(_mean(v[field] for v in detail[s].values()) for s in VAL_SCENES)
                   for field in ('kl', 'js', 'direct_kl', 'direct_js')},
                by_scene=by_scene, by_condition=by_condition,
                by_scene_condition=by_scene_condition, detail=detail,
                pck_by_condition=pck_by_condition,
                probability_checks=dict(all_finite=True, matrices=len(numerical),
                    max_normalization_error=max(v['max_normalization_error'] for v in numerical)))
        output[scenario] = {alias: by_operator[operator] for alias, operator in roles['aliases'].items()}
    return output


def aggregate_timing(timing_rows, timing_entries, roles):
    """Validate 5x9 paired cells per operator; mean of complete-call medians."""
    validate_roles(roles)
    cohort = _cohort(timing_entries, timing=True)
    expected = {(operator, scene, pair, q+'_'+k) for operator in roles['operators']
                for scene, pair in cohort for q, k in ORIGINAL}
    if len(timing_rows) != len(expected):
        raise ValueError('incomplete paired timing population')
    seen, by_operator = set(), {operator: [] for operator in roles['operators']}
    for row in timing_rows:
        identity = row['operator'], row['scene'], _pair(row['pair']), row['condition']
        if identity not in expected or identity in seen:
            raise ValueError('wrong pair or duplicate timing identity')
        seen.add(identity)
        if ((row['query'], row['key']) not in ORIGINAL
                or row['condition'] != row['query']+'_'+row['key']):
            raise ValueError('timing condition mismatch')
        values = []
        for name, count in (('samples_seconds', 15), ('warmup_seconds', 3)):
            if len(row[name]) != count:
                raise ValueError('timing repeat/warmup count changed')
            samples = [_finite(value, name) for value in row[name]]
            if any(value <= 0 for value in samples):
                raise ValueError('timing samples must be positive')
            if name == 'samples_seconds':
                values = samples
        by_operator[row['operator']].append(dict(row, median_seconds=float(median(values))))
    if seen != expected:
        raise ValueError('missing timing identity')
    return {operator: dict(mean_seconds=_mean(cell['median_seconds'] for cell in cells),
                           cells=cells) for operator, cells in by_operator.items()}


def _pck_value(metric, condition):
    saved = metric['pck_by_condition'][condition]
    exact, fractions = _pck_counts(saved['counts_by_scene'])
    if (saved['exact'] != _fraction_dict(exact) or saved['macro'] != float(exact)
            or saved['by_scene'] != {s: float(v) for s, v in fractions.items()}):
        raise ValueError('PCK summary does not match integer counts')
    return exact


def _pck_gate(metrics, condition):
    candidate = _pck_value(metrics['primary'], condition)
    comparisons = {}
    for reference in LOCALIZERS:
        difference = candidate-_pck_value(metrics[reference], condition)
        comparisons[reference] = dict(passed=difference >= -PCK_TOLERANCE,
            difference_pp=float(100*difference), exact_difference=_fraction_dict(difference))
    return dict(passed=all(v['passed'] for v in comparisons.values()),
                condition=condition, comparisons=comparisons)


def _gain(candidate, reference):
    candidate = _finite(candidate, 'candidate KL')
    reference = _finite(reference, 'reference KL')
    if reference <= 0:
        raise ValueError('relative KL requires positive reference denominator')
    return 1-candidate/reference


def _point_gate(metrics, references):
    full = metrics['primary']
    comparisons = {}
    for reference in references:
        control = metrics[reference]
        gain = _gain(full['kl'], control['kl'])
        checks = dict(relative_gain_at_least_2pct=full['kl'] <= .98*control['kl'],
            both_scenes_improve=all(full['by_scene'][s] < control['by_scene'][s] for s in VAL_SCENES))
        comparisons[reference] = dict(passed=all(checks.values()), checks=checks, relative_gain=gain)
    return dict(passed=all(v['passed'] for v in comparisons.values()), comparisons=comparisons)


def _partition_gate(metrics, references):
    comparisons = {}
    full = metrics['primary']['by_condition']
    for reference in references:
        control = metrics[reference]['by_condition']
        conditions = {condition: dict(passed=value <= 1.02*control[condition],
            relative_gain=_gain(value, control[condition])) for condition, value in full.items()}
        comparisons[reference] = dict(passed=all(v['passed'] for v in conditions.values()), conditions=conditions)
    return dict(passed=all(v['passed'] for v in comparisons.values()), comparisons=comparisons)


def gates(metrics, roles, timing):
    """Conjunction of all old comparisons plus the predeclared new q16 guard."""
    validate_roles(roles)
    if set(metrics) != set(SCENARIOS) or set(timing) != set(roles['operators']):
        raise ValueError('gate metric/timing population changed')
    numerical = {}
    for scenario, conditions in CONDITIONS.items():
        if set(metrics[scenario]) != set(ALIAS_ORDER):
            raise ValueError('missing gate aliases')
        expected_conditions = {q+'_'+k for q, k in conditions}
        for alias, values in metrics[scenario].items():
            if (set(values['by_scene']) != set(VAL_SCENES)
                    or set(values['by_condition']) != expected_conditions
                    or set(values['pck_by_condition']) != expected_conditions):
                raise ValueError('gate scene/condition population changed')
            check = values['probability_checks']
            error = _finite(check['max_normalization_error'], 'normalization error', 0)
            if check['matrices'] != 32*len(conditions):
                raise ValueError('finite checks do not cover all predictions')
            numerical[scenario+'__'+alias] = check['all_finite'] is True and error <= 1.e-12
    for operator, value in timing.items():
        seconds = _finite(value['mean_seconds'], 'mean timing')
        if seconds <= 0 or len(value['cells']) != 45:
            raise ValueError('invalid full paired timing aggregate')
        actual = _mean(_finite(cell['median_seconds'], 'median timing') for cell in value['cells'])
        if seconds != actual:
            raise ValueError('timing mean inconsistent with cell medians')
    point_refs = (roles['strong_point'], 'legacy_point')
    quad_refs = (roles['strong_quad'], 'legacy_quad16')
    zero = metrics['zero']
    cost = {}
    primary_seconds = timing[roles['aliases']['primary']]['mean_seconds']
    for reference in quad_refs:
        control_seconds = timing[roles['aliases'][reference]]['mean_seconds']
        full_kl, control_kl = zero['primary']['kl'], zero[reference]['kl']
        gain = _gain(full_kl, control_kl)
        accuracy = full_kl <= .98*control_kl and primary_seconds <= 1.25*control_seconds
        efficiency = full_kl <= 1.01*control_kl and primary_seconds <= .5*control_seconds
        cost[reference] = dict(passed=accuracy or efficiency, accuracy_route=accuracy,
            efficiency_route=efficiency, relative_gain=gain, time_ratio=primary_seconds/control_seconds)
    original = dict(point=_point_gate(zero, point_refs),
        accuracy_cost=dict(passed=all(v['passed'] for v in cost.values()), comparisons=cost),
        partition=_partition_gate(zero, quad_refs), pck_g8=_pck_gate(zero, 'g8_g8'))
    original['passed'] = all(value['passed'] for value in original.values())
    q16 = _pck_gate(zero, 'g16_g8')
    q16['new_predeclared_guard'] = True
    q16['historical_R12_gate'] = False
    scale = dict(point=_point_gate(metrics['scale'], point_refs),
        partition=_partition_gate(metrics['scale'], quad_refs),
        pck_g24=_pck_gate(metrics['scale'], 'g24_g8'))
    scale['passed'] = all(value['passed'] for value in scale.values())
    stress = {}
    for state in STATES:
        values = metrics[state]
        comparisons = {reference: dict(
            passed=values['primary']['kl'] <= 1.02*values[reference]['kl'],
            relative_gain=_gain(values['primary']['kl'], values[reference]['kl'])) for reference in quad_refs}
        pck = _pck_gate(values, 'g8_g8')
        stress[state] = dict(passed=all(v['passed'] for v in comparisons.values()) and pck['passed'],
            quad_comparisons=comparisons, pck_g8=pck)
    finite = dict(passed=all(numerical.values()), checks=numerical)
    passed = original['passed'] and q16['passed'] and scale['passed'] and finite['passed'] and all(v['passed'] for v in stress.values())
    return dict(passed=passed, original=original, new_q16_protection=q16, scale=scale,
                stress=stress, predictions=finite, general_research_goal_achieved=False,
                fresh_confirmation_authorized=False, isotropic_is_attribution_only=True,
                constant_temperature_is_attribution_only=True)
