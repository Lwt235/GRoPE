"""Independent R20 region/center reductions and sixty fixed screen checks.

Only caller-supplied authorized arrays/rows are accepted. No file loader,
production metric, model, calibration or alternative selection is imported.
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import math

import numpy as np
from experiments import round15_regional_audit_metrics as REGION


SCENES = REGION.SCENES
COUNTS = REGION.NATIVE_SIZES
CONDITIONS = tuple(dict(scenario='partition' if (q, k) in REGION.PARTITION_CONDITIONS else 'scale',
    query=q, key=k, condition=q + '_' + k) for q, k in REGION.CONDITIONS)
METRICS = ('kl', 'js', 'direct_kl', 'direct_js')
VALID_COUNTS = dict(g8=9521, g16=2463, g24=1096)


def integer(value, minimum=0, maximum=None):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or value < minimum or (maximum is not None and value > maximum)):
        raise ValueError('invalid integer population/count')
    return int(value)


def number(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) or not math.isfinite(value):
        raise ValueError('invalid finite metric')
    return float(value)


def fraction_dict(value):
    return dict(numerator=value.numerator, denominator=value.denominator, value=float(value))


def exact_fraction(record):
    value = Fraction(integer(record['numerator'], minimum=-10**30), integer(record['denominator'], minimum=1))
    if record != fraction_dict(value):
        raise ValueError('noncanonical or rounded Fraction')
    return value


def average(values):
    values = list(values)
    if not values:
        raise ValueError('empty averaging population')
    return math.fsum(values) / len(values)


def region_metrics(probability, parts, condition, direct, common):
    p = np.asarray(probability)
    if p.dtype != np.float64:
        raise ValueError('saved native float64 probabilities required')
    return REGION.metric_reference(p, direct, common, parts[condition['query']],
                                   parts[condition['key']], condition['scenario'])


def center_metrics(argmax, bounds, point, valid):
    argmax, bounds, point, valid = map(np.asarray, (argmax, bounds, point, valid))
    n = len(argmax)
    if (argmax.dtype != np.int32 or argmax.shape != (n,) or bounds.dtype != np.int64
            or bounds.ndim != 2 or bounds.shape[1] != 4 or point.shape != (n, 2)
            or valid.dtype != np.bool_ or valid.shape != (n,)
            or not np.isfinite(point[valid]).all() or np.any(argmax < 0) or np.any(argmax >= len(bounds))):
        raise ValueError('exact center targets and native first-index winners required')
    center = (bounds[:, :2].astype(np.float64) + bounds[:, 2:] - 1.) * .5
    errors = center[argmax[valid]] - point[valid]
    distance2 = errors[:, 0] ** 2 + errors[:, 1] ** 2
    n_valid = int(valid.sum())
    total = math.fsum(math.sqrt(float(v)) for v in distance2)
    square = math.fsum(map(float, distance2))
    return dict(pck_count=int(np.count_nonzero(distance2 <= 64.)), n_point=n_valid, n_query=n,
        distance_sum=total, distance_squared_sum=square,
        mean=total / n_valid if n_valid else None,
        rmse=math.sqrt(square / n_valid) if n_valid else None)


def _cohort(entries):
    identities = [(e['scene'], tuple(integer(v) for v in e['pair'])) for e in entries]
    if (len(identities) != 48 or len(set(identities)) != 48
            or any(sum(s == scene for s, _ in identities) != 16 for scene in SCENES)):
        raise ValueError('complete original three-scene cohort required')
    return identities


def _validate_rows(regional, centers, entries, center_conditions):
    cohort = _cohort(entries)
    lookup = {c['condition']: c for c in CONDITIONS}
    expected_region = {(s, pair, c) for s, pair in cohort for c in lookup} if regional else set()
    expected_center = {(s, pair, c) for s, pair in cohort for c in center_conditions} if centers else set()
    seen, populations = set(), {}
    for r in regional:
        key = (r['scene'], tuple(r['pair']), r['condition'])
        if key not in expected_region or key in seen:
            raise ValueError('unexpected/missing/duplicate regional identity')
        seen.add(key); c = lookup[r['condition']]
        if any(r.get(field) != c[field] for field in ('query', 'key', 'scenario')):
            raise ValueError('condition metadata mismatch')
        if integer(r['n_query']) != COUNTS[c['query']] or integer(r['n_key']) != COUNTS[c['key']]:
            raise ValueError('native population changed')
        mass = integer(r['mass'])
        if key[:2] in populations and populations[key[:2]] != mass:
            raise ValueError('regional visible-pixel population changed across conditions')
        populations[key[:2]] = mass
        for field in METRICS:
            if number(r[field + '_sum']) < -1e-8:
                raise ValueError('negative divergence sum')
    if seen != expected_region:
        raise ValueError('incomplete regional population')
    seen, valid_populations = set(), {}
    for r in centers:
        key = (r['scene'], tuple(r['pair']), r['condition'])
        if key not in expected_center or key in seen:
            raise ValueError('unexpected/missing/duplicate center identity')
        seen.add(key); c = lookup[r['condition']]
        n = integer(r['n_point'], maximum=COUNTS[c['query']])
        integer(r['pck_count'], maximum=n)
        if integer(r['n_query']) != COUNTS[c['query']]:
            raise ValueError('center query population changed')
        point_key = (*key[:2], c['query'])
        if point_key in valid_populations and valid_populations[point_key] != n:
            raise ValueError('center-valid mask changed across native keys')
        valid_populations[point_key] = n
    if seen != expected_center:
        raise ValueError('incomplete center population')
    for q in {lookup[c]['query'] for c in center_conditions}:
        if sum(n for (s, pair, query), n in valid_populations.items() if query == q) != VALID_COUNTS[q]:
            raise ValueError('original exact valid-center total changed')


def _center_reduction(rows):
    counts = {name: sum(integer(r[name]) for r in rows) for name in ('pck_count', 'n_point', 'n_query')}
    if counts['n_point'] <= 0:
        raise ValueError('empty scene center population')
    counts['pck'] = fraction_dict(Fraction(counts['pck_count'], counts['n_point']))
    has_continuous = ['distance_sum' in r for r in rows]
    if any(has_continuous) != all(has_continuous):
        raise ValueError('continuous provenance differs within one role')
    if all(has_continuous):
        total = math.fsum(number(r['distance_sum']) for r in rows)
        square = math.fsum(number(r['distance_squared_sum']) for r in rows)
        counts.update(distance_sum=total, distance_squared_sum=square,
                      mean=total / counts['n_point'], rmse=math.sqrt(square / counts['n_point']))
    return counts


def summarize_role(regional_rows, center_rows, entries, *, center_conditions=()):
    _validate_rows(regional_rows, center_rows, entries, center_conditions)
    region_groups, center_groups = defaultdict(list), defaultdict(list)
    for row in regional_rows:
        region_groups[row['scene'], row['condition']].append(row)
    for row in center_rows:
        center_groups[row['scene'], row['condition']].append(row)
    by_scene_condition, by_condition, by_scene, macro = {}, {}, {}, {}
    if regional_rows:
        for scene in SCENES:
            by_scene_condition[scene] = {}
            for condition in CONDITIONS:
                selected = region_groups[scene, condition['condition']]
                mass = sum(r['mass'] for r in selected)
                if mass <= 0:
                    raise ValueError('empty regional scene/condition')
                by_scene_condition[scene][condition['condition']] = dict(mass=mass,
                    **{metric: math.fsum(r[metric + '_sum'] for r in selected) / mass for metric in METRICS})
            common = average(by_scene_condition[scene][c['condition']]['kl'] for c in CONDITIONS if c['scenario'] == 'partition')
            direct = average(by_scene_condition[scene][c['condition']]['kl'] for c in CONDITIONS if c['scenario'] == 'scale')
            by_scene[scene] = dict(common9_kl=common, direct5_kl=direct, J=.5 * (common + direct))
        for condition in CONDITIONS:
            c = condition['condition']
            by_condition[c] = {m: average(by_scene_condition[s][c][m] for s in SCENES) for m in METRICS}
        macro = {m: average(by_scene[s][m] for s in SCENES) for m in ('common9_kl', 'direct5_kl')}
        macro['J'] = .5 * (macro['common9_kl'] + macro['direct5_kl'])
    center_scene, center_condition = {}, {}
    if center_rows:
        ordered = [c['condition'] for c in CONDITIONS if c['condition'] in center_conditions]
        for scene in SCENES:
            center_scene[scene] = {c: _center_reduction(center_groups[scene, c]) for c in ordered}
        for c in ordered:
            result = dict(pck=fraction_dict(sum((exact_fraction(center_scene[s][c]['pck']) for s in SCENES), Fraction()) / 3))
            if all('mean' in center_scene[s][c] for s in SCENES):
                result.update(mean=average(center_scene[s][c]['mean'] for s in SCENES),
                              rmse=average(center_scene[s][c]['rmse'] for s in SCENES))
            center_condition[c] = result
    return dict(by_scene_condition=by_scene_condition, by_condition=by_condition, by_scene=by_scene,
                macro=macro, center=dict(by_scene_condition=center_scene, by_condition=center_condition))


def screen(by_role, old_by_role):
    """The one frozen primary; no alternative attribution winner or epsilon."""
    primary = by_role['anchor_trace']
    checks = []
    old_macro_rules = [('strong_point', .98, .98), ('legacy_point', .98, .98),
                       ('strong_quad', 1.01, 1.02), ('legacy_quad16', 1.01, 1.02)]
    specifications = []
    for role, common_bound, direct_bound in old_macro_rules:
        for metric, multiplier in zip(('common9_kl', 'direct5_kl'), (common_bound, direct_bound)):
            specifications.append(('old_macro', metric + '_vs_' + role, role, metric, None, multiplier, old_by_role))
    for c in CONDITIONS:
        for role in ('strong_quad', 'legacy_quad16'):
            specifications.append(('old_condition', c['condition'] + '_vs_' + role, role, 'kl', c['condition'], 1.02, old_by_role))
    for role in ('anchor_point_regional', 'anchor_quartic_regional', 'anchor_fixed_regional'):
        for metric in ('common9_kl', 'direct5_kl'):
            specifications.append(('same_a_macro', metric + '_vs_' + role, role, metric, None, .98, by_role))
    for category, name, role, metric, condition, multiplier, source in specifications:
        candidate = primary['by_condition'][condition]['kl'] if condition else primary['macro'][metric]
        reference = source[role]['by_condition'][condition]['kl'] if condition else source[role]['macro'][metric]
        candidate, reference = number(candidate), number(reference)
        bound = reference * multiplier
        checks.append(dict(category=category, name=name, metric=metric, reference=role,
            candidate_value=candidate, reference_value=reference, bound=bound,
            passed=bool(reference > 0 and candidate <= bound)))
    for category, names, source in (
        ('old_center', ('point_localizer', 'quartic_localizer', 'legacy_point_localizer'), old_by_role),
        ('same_a_center', ('anchor_point_localizer', 'anchor_quartic_localizer', 'anchor_point_legacy'), by_role)):
        for query in ('g8', 'g16', 'g24'):
            condition = query + '_g8'
            candidate = exact_fraction(primary['center']['by_condition'][condition]['pck'])
            for role in names:
                reference = exact_fraction(source[role]['center']['by_condition'][condition]['pck'])
                bound = reference - Fraction(1, 200)
                checks.append(dict(category=category, name=condition + '_vs_' + role, metric='pck', reference=role,
                    candidate_value=fraction_dict(candidate), reference_value=fraction_dict(reference),
                    bound=fraction_dict(bound), passed=candidate >= bound))
    if len(checks) != 60:
        raise ValueError('incomplete fixed development screen')
    passed = all(c['passed'] for c in checks)
    return dict(passed=passed, checks=checks, primary='anchor_trace', scientific_gate_passed=False,
        next_action='consider_separately_frozen_same_information_calibration' if passed else 'stop_fixed_anchor_candidate')
