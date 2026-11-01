"""Independent fixed R25 q24-to-g8 metrics; no files or model calls.

Only already-authorized caller arrays/rows are accepted. R20 independent
primitives supply the inherited label mathematics; this module owns its
single-condition population reduction and eight fixed comparisons.
"""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np

from experiments import round20_anchor_audit_metrics as BASE

SCENES = BASE.SCENES
CONDITION = dict(condition='g24_g8', scenario='scale', query='g24', key='g8')
METRICS = ('kl', 'js', 'direct_kl', 'direct_js')
VALID_BY_SCENE = dict(zip(SCENES, (382, 402, 312)))
POPULATION = dict(pairs=48, queries=2304, valid_centers=1096)
REGIONAL_ROLES = ('strong_quad', 'legacy_quad16')
OLD_CENTER_ROLES = ('point_localizer', 'quartic_localizer', 'legacy_point_localizer')
SAME_A_CENTER_ROLES = ('anchor_point_localizer', 'anchor_quartic_localizer', 'anchor_point_legacy')


def _identity(record):
    if record.get('scene') not in SCENES:
        raise ValueError('unexpected fixed TRAIN scene')
    pair = record.get('pair')
    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
        raise ValueError('pair requires two integer frame identities')
    return record['scene'], tuple(BASE.integer(v) for v in pair)


def _condition(query, key):
    if (query, key) != ('g24', 'g8'):
        raise ValueError('R25 only permits the fixed g24_g8 condition')


def _bounds(value, name):
    bounds = np.asarray(value)
    literal = BASE.REGION.partition_bounds(name)
    if (bounds.dtype.kind not in 'iu' or bounds.shape != literal.shape
            or not np.array_equal(bounds, literal)):
        raise ValueError('literal original native bounds required')
    return bounds.astype(np.int64, copy=False)


def target_counts(labels, parts, query='g24', key='g8'):
    """Independent direct[48,432] and common[108,432] visible-pixel counts.

    Reconstruct private pixel lookups from literal bounds; archived producer
    partition dictionaries need not contain the auditor's lookup fields.
    """
    _condition(query, key)
    return BASE.REGION.reconstruct_targets(labels['gt_xy'], labels['gt_visible'],
        _bounds(parts[query]['bounds'], query), _bounds(parts[key]['bounds'], key))


def region_metrics(probability, parts, condition, direct, common):
    if condition != CONDITION:
        raise ValueError('exact fixed scale-condition descriptor required')
    p = np.asarray(probability)
    if p.dtype != np.float64 or p.shape != (48, 432):
        raise ValueError('saved float64 [48,432] probabilities required')
    return BASE.REGION.metric_reference(p, direct, common,
        _bounds(parts['g24']['bounds'], 'g24'), _bounds(parts['g8']['bounds'], 'g8'), 'scale')


def center_metrics(argmax, bounds, point, valid):
    if np.asarray(argmax).shape != (48,):
        raise ValueError('all 48 native q24 winners are required')
    return BASE.center_metrics(argmax, _bounds(bounds, 'g8'), point, valid)


def _center_row(row):
    n = BASE.integer(row['n_point'], maximum=48)
    BASE.integer(row['pck_count'], maximum=n)
    if BASE.integer(row['n_query']) != 48:
        raise ValueError('center query population changed')
    total, square = (BASE.number(row[k]) for k in ('distance_sum', 'distance_squared_sum'))
    if total < 0 or square < 0:
        raise ValueError('negative continuous center error sum')
    if n == 0:
        if total != 0 or square != 0 or row['mean'] is not None or row['rmse'] is not None:
            raise ValueError('empty center row must retain exact zero sums and null averages')
    else:
        for key, expected in (('mean', total / n), ('rmse', math.sqrt(square / n))):
            if not math.isclose(BASE.number(row[key]), expected, rel_tol=1e-12, abs_tol=1e-10):
                raise ValueError('continuous center average disagrees with full sums')


def summarize(rows, entries):
    """Full 48-pair reduction: pooled within scene, equal weight across scenes.

    The macro RMSE is an average of scene RMSEs, never a pooled-population
    RMSE. No common9/direct5 or multi-condition objective is constructed.
    """
    expected = [_identity(e) for e in entries]
    if (len(expected) != 48 or len(set(expected)) != 48
            or any(sum(s == scene for s, _ in expected) != 16 for scene in SCENES)):
        raise ValueError('complete unique 3-by-16 old TRAIN cohort required')
    lookup = {}
    for row in rows:
        identity = _identity(row)
        if identity not in expected or identity in lookup:
            raise ValueError('unexpected or duplicate pair row')
        for name, value in CONDITION.items():
            if name in row and row[name] != value:
                raise ValueError('row condition metadata changed')
        regional, center = row['region'], row['center']
        if (BASE.integer(regional['n_query']) != 48 or BASE.integer(regional['n_key']) != 432):
            raise ValueError('native regional query/key population changed')
        BASE.integer(regional['mass'], maximum=144 * 192)
        for metric in METRICS:
            if BASE.number(regional[metric + '_sum']) < -1e-8:
                raise ValueError('invalid negative regional divergence')
        if any(regional[m + '_sum'] != regional['direct_' + m + '_sum'] for m in ('kl', 'js')):
            raise ValueError('g24_g8 scale metrics must equal their direct counterparts')
        _center_row(center)
        lookup[identity] = row
    if set(lookup) != set(expected):
        raise ValueError('incomplete fixed pair population')
    regional_scene, center_scene = {}, {}
    for scene in SCENES:
        selected = [lookup[identity] for identity in expected if identity[0] == scene]
        regional = [r['region'] for r in selected]
        centers = [r['center'] for r in selected]
        mass = sum(BASE.integer(r['mass']) for r in regional)
        if mass <= 0:
            raise ValueError('empty scene regional population')
        sums = {m + '_sum': math.fsum(BASE.number(r[m + '_sum']) for r in regional) for m in METRICS}
        regional_scene[scene] = dict(mass=mass, **sums, **{m: sums[m + '_sum'] / mass for m in METRICS})
        count = {k: sum(BASE.integer(r[k]) for r in centers) for k in ('pck_count', 'n_point', 'n_query')}
        if count['n_point'] != VALID_BY_SCENE[scene] or count['n_query'] != 768:
            raise ValueError('original exact scene center population changed')
        total = math.fsum(BASE.number(r['distance_sum']) for r in centers)
        square = math.fsum(BASE.number(r['distance_squared_sum']) for r in centers)
        center_scene[scene] = dict(**count, pck=BASE.fraction_dict(Fraction(count['pck_count'], count['n_point'])),
            distance_sum=total, distance_squared_sum=square, mean=total / count['n_point'],
            rmse=math.sqrt(square / count['n_point']))
    if (sum(r['n_query'] for r in center_scene.values()) != 2304
            or sum(r['n_point'] for r in center_scene.values()) != 1096):
        raise ValueError('original total population changed')
    macro_pck = sum((BASE.exact_fraction(center_scene[s]['pck']) for s in SCENES), Fraction()) / 3
    return dict(condition='g24_g8', population=dict(POPULATION),
        region=dict(by_scene=regional_scene, macro={m: math.fsum(regional_scene[s][m] for s in SCENES) / 3 for m in METRICS}),
        center=dict(by_scene=center_scene, macro=dict(pck=BASE.fraction_dict(macro_pck),
            mean=math.fsum(center_scene[s]['mean'] for s in SCENES) / 3,
            rmse=math.fsum(center_scene[s]['rmse'] for s in SCENES) / 3)))


def screen(summary, r20summary):
    """Eight inherited fixed checks; matched Gaussian is not an extra gate."""
    if summary.get('condition') != 'g24_g8' or summary.get('population') != POPULATION:
        raise ValueError('complete fixed-condition summary required')
    candidate_kl = BASE.number(summary['region']['macro']['kl'])
    candidate_pck = BASE.exact_fraction(summary['center']['macro']['pck'])
    if not 0 <= candidate_pck <= 1:
        raise ValueError('PCK must be a canonical probability Fraction')
    old, same_a = r20summary['old_by_role'], r20summary['by_role']
    regional_values = [BASE.number(old[name]['by_condition']['g24_g8']['kl']) for name in REGIONAL_ROLES]
    if regional_values[0] != regional_values[1]:
        raise ValueError('the two frozen quad16 aliases must carry the same regional reference')
    checks = []
    for name, reference in zip(REGIONAL_ROLES, regional_values):
        bound = reference * 1.02
        checks.append(dict(category='old_region', name='g24_g8_kl_vs_' + name, metric='kl', reference=name,
            candidate_value=candidate_kl, reference_value=reference, bound=bound,
            alias_of='strong_quad' if name == 'legacy_quad16' else None,
            passed=bool(reference > 0 and candidate_kl <= bound)))
    for category, roles, source in (('old_center', OLD_CENTER_ROLES, old), ('same_a_center', SAME_A_CENTER_ROLES, same_a)):
        for name in roles:
            reference = BASE.exact_fraction(source[name]['center']['by_condition']['g24_g8']['pck'])
            if not 0 <= reference <= 1:
                raise ValueError('reference PCK outside probability range')
            bound = reference - Fraction(1, 200)
            checks.append(dict(category=category, name='g24_g8_pck_vs_' + name, metric='pck', reference=name,
                candidate_value=BASE.fraction_dict(candidate_pck), reference_value=BASE.fraction_dict(reference),
                bound=BASE.fraction_dict(bound), passed=candidate_pck >= bound))
    if len(checks) != 8:
        raise ValueError('complete eight-check screen required')
    passed = all(c['passed'] for c in checks)
    return dict(passed=passed, checks=checks, regional_alias_groups=[dict(roles=list(REGIONAL_ROLES),
        family='gaussian_quad16', grid_index=4, h=.5, temperature=5., distinct_operators=1)],
        matched_gaussian_role='anchor_trace', matched_gaussian_is_diagnostic_only=True,
        scientific_gate_passed=False, candidate_selected=False,
        next_action='consider_separately_frozen_full_validation' if passed else 'stop_fixed_student_candidate')
