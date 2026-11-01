"""Independent pure aggregation/selection for the Round17 TRAIN regional audit.

There are no producer/model imports, file reads, or new predictions here. The
outer auditor binds the supplied rows and historical Round15 summary to their
frozen sources. In particular, this module does not independently rebuild the
historical eight-family metrics; it checks their grid and selection contracts.
"""
from __future__ import annotations

from copy import deepcopy
from math import fsum, isfinite, log
from numbers import Integral, Real


SCENES = ('harvard_c3/hv_c3_1', 'harvard_c5/hv_c5_1', 'harvard_c6/hv_c6_1')
FAMILY = 'trace_full'
OLD_FAMILIES = ('quartic_full', 'quartic_point', 'gaussian_point', 'gaussian_fixed',
                'gaussian_full1', 'gaussian_full4', 'gaussian_quad4', 'gaussian_quad16')
GRID = tuple((h, t) for h in (.25, .5, 1., 2.) for t in (0., 5., 10.))
PARTITION_CONDITIONS = tuple((q, k) for q in ('g8', 'g16', 'mixed')
                             for k in ('g8', 'g16', 'mixed'))
SCALE_CONDITIONS = (('g8', 'g24'), ('g24', 'g8'), ('g16', 'g24'),
                    ('g24', 'g16'), ('g24', 'g24'))
CONDITIONS = PARTITION_CONDITIONS + SCALE_CONDITIONS
NATIVE_SIZES = {'g8': 432, 'g16': 108, 'mixed': 243, 'g24': 48}
METRIC_FIELDS = ('kl_sum', 'js_sum', 'direct_kl_sum', 'direct_js_sum')
METRIC_NAMES = ('kl', 'js', 'direct_kl', 'direct_js')


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(name + ' must be an integer')
    return int(value)


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(name + ' must be a finite real number')
    return float(value)


def _identity(row):
    pair = row['pair']
    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
        raise ValueError('pair must contain two integer frame indices')
    pair = tuple(_integer(x, 'frame index') for x in pair)
    if min(pair) < 0:
        raise ValueError('negative frame index')
    if 'split' in row and row['split'] != 'train':
        raise ValueError('only old TRAIN entries are permitted')
    return row['scene'], pair


def _cohort(entries):
    identities = [_identity(entry) for entry in entries]
    if (len(identities) != 48 or len(set(identities)) != 48
            or {s for s, _ in identities} != set(SCENES)
            or any(sum(s == scene for s, _ in identities) != 16 for scene in SCENES)):
        raise ValueError('expected 48 unique TRAIN pairs, 16 per fixed scene')
    return set(identities)


def validate_population(rows, entries=None):
    """Require all 8,064 rows; optionally bind the exact frozen frame cohort.

    Without entries, the strict fixed scene/pair cardinalities still apply, but
    the outer auditor must bind actual frame IDs. One pair has the same visible
    sensor mass in every grid/condition; empty queries remain in native sizes.
    """
    if len(rows) != 48 * 12 * 14:
        raise ValueError('expected exactly 8064 regional rows')
    if entries is None:
        identities = {_identity(row) for row in rows}
        entries = [dict(scene=s, pair=p) for s, p in identities]
    cohort = _cohort(entries)
    seen, masses = set(), {}
    scenarios = {c: 'partition' for c in PARTITION_CONDITIONS}
    scenarios.update({c: 'scale' for c in SCALE_CONDITIONS})
    for row in rows:
        identity = _identity(row)
        index = _integer(row['grid_index'], 'grid_index')
        condition = row['query'], row['key']
        if (identity not in cohort or row['family'] != FAMILY or not 0 <= index < 12
                or condition not in scenarios or row['scenario'] != scenarios[condition]):
            raise ValueError('unexpected cohort/family/grid/condition/scenario')
        if 'condition' in row and row['condition'] != '_'.join(condition):
            raise ValueError('condition name disagrees with query/key')
        if 'config' in row and row['config'] != dict(h=GRID[index][0], temperature=GRID[index][1]):
            raise ValueError('configuration disagrees with fixed grid')
        unique = identity + (index,) + condition
        if unique in seen:
            raise ValueError('duplicate regional row')
        seen.add(unique)
        mass = _integer(row['mass'], 'mass')
        if not 0 <= mass <= 144 * 192:
            raise ValueError('visible mass outside image population')
        if identity in masses and masses[identity] != mass:
            raise ValueError('visible mass changed between configurations/conditions')
        masses[identity] = mass
        if (_integer(row['n_query'], 'n_query') != NATIVE_SIZES[condition[0]]
                or _integer(row['n_key'], 'n_key') != NATIVE_SIZES[condition[1]]):
            raise ValueError('native query/key population changed')
        values = [_number(row[f], f) for f in METRIC_FIELDS]
        if any(value < -1e-8 for value in values):
            raise ValueError('invalid negative summed divergence')
        if any(values[i] > mass * log(2.) + 1e-8 for i in (1, 3)):
            raise ValueError('JS exceeds visible-mass bound')
        if mass == 0 and any(value != 0 for value in values):
            raise ValueError('zero visible mass has nonzero contribution')
        if row['scenario'] == 'scale' and (values[0] != values[2] or values[1] != values[3]):
            raise ValueError('scale metrics must equal direct native metrics')
    if len(seen) != len(cohort) * len(GRID) * len(CONDITIONS):
        raise ValueError('incomplete Cartesian regional population')
    mass_by_scene = {s: sum(m for (scene, _), m in masses.items() if scene == s) for s in SCENES}
    if any(m <= 0 for m in mass_by_scene.values()):
        raise ValueError('scene has no visible sensor mass')
    return dict(rows=len(rows), pairs=len(cohort), mass_by_scene=mass_by_scene,
                unique_visible_pixel_mass=sum(masses.values()))


def summarize_metrics(rows, entries=None):
    """Mass-weight pairs, then equally weight conditions and fixed scenes."""
    population = validate_population(rows, entries)
    buckets = {}
    for row in rows:
        key = row['grid_index'], row['scene'], row['query'] + '_' + row['key']
        buckets.setdefault(key, []).append(row)
    names = tuple('_'.join(c) for c in CONDITIONS)
    values = []
    for index, (h, temperature) in enumerate(GRID):
        scene_conditions, scene_macros = {}, {}
        for scene in SCENES:
            conditions = {}
            for name in names:
                group = sorted(buckets[index, scene, name], key=lambda row: tuple(row['pair']))
                mass = sum(row['mass'] for row in group)
                conditions[name] = {name_: fsum(float(row[field]) for row in group) / mass
                                     for name_, field in zip(METRIC_NAMES, METRIC_FIELDS)}
                conditions[name]['mass'] = mass
            common = fsum(conditions[n]['kl'] for n in names[:9]) / 9
            direct = fsum(conditions[n]['direct_kl'] for n in names[9:]) / 5
            scene_conditions[scene] = conditions
            scene_macros[scene] = dict(common9_kl=common, direct5_kl=direct, J=.5 * common + .5 * direct)
        common = fsum(scene_macros[s]['common9_kl'] for s in SCENES) / 3
        direct = fsum(scene_macros[s]['direct5_kl'] for s in SCENES) / 3
        values.append(dict(grid_index=index, config=dict(h=h, temperature=temperature),
            J=.5 * common + .5 * direct, common9_kl=common, direct5_kl=direct,
            by_scene=scene_macros, by_scene_condition=scene_conditions,
            by_condition={name: {field: fsum(scene_conditions[s][name][field] for s in SCENES) / 3
                                  for field in METRIC_NAMES} for name in names}))
    return {FAMILY: values}, population


def _eligible(indices):
    indices = [_integer(index, 'eligible index') for index in indices]
    if not indices:
        raise ValueError('empty Stage A eligible set: stop before Stage B')
    if len(set(indices)) != len(indices) or any(not 0 <= i < 12 for i in indices):
        raise ValueError('invalid eligible grid set')
    return sorted(indices)


def _check_grid(values):
    if len(values) != 12:
        raise ValueError('each family requires all twelve configurations')
    for index, value in enumerate(values):
        if (_integer(value['grid_index'], 'grid_index') != index
                or value['config'] != dict(h=GRID[index][0], temperature=GRID[index][1])):
            raise ValueError('invalid ordered regional grid')
        j, common, direct = (_number(value[f], f) for f in ('J', 'common9_kl', 'direct5_kl'))
        if min(j, common, direct) < -1e-8 or j != .5 * common + .5 * direct:
            raise ValueError('J disagrees with the fixed half/half objective')


def _spec(metrics, family, index):
    value = metrics[family][index]
    return dict(family=family, grid_index=index, config=deepcopy(value['config']), J=float(value['J']))


def _choose(metrics, family, indices=range(12)):
    # Sorted indices implement fixed grid order even if caller order differs.
    index = min(sorted(indices), key=lambda i: metrics[family][i]['J'])
    return _spec(metrics, family, index)


def validate_historical_summary(old_summary):
    """Check historical selections without retuning them with the new rows.

    Source/provenance and independent R15 row replay remain the outer auditor's
    responsibility. The old quartic primary is constrained; its unconstrained
    minimum is separate and must not overwrite it.
    """
    metrics, selection = old_summary['metrics'], old_summary['selection']
    if set(metrics) != set(OLD_FAMILIES):
        raise ValueError('historical summary requires all eight families')
    for values in metrics.values():
        _check_grid(values)
    old_eligible = _eligible(selection['eligible_indices'])
    controls = {f: _choose(metrics, f) for f in OLD_FAMILIES[1:]}
    expected = dict(
        primary=_choose(metrics, 'quartic_full', old_eligible),
        unconstrained_primary=_choose(metrics, 'quartic_full'),
        controls=controls,
        strong_point=min((controls[f] for f in ('gaussian_point', 'quartic_point')), key=lambda v: v['J']),
        strong_quadrature=min((controls[f] for f in ('gaussian_quad4', 'gaussian_quad16')), key=lambda v: v['J']))
    for key, value in expected.items():
        if selection[key] != value:
            raise ValueError('historical ' + key + ' violates its frozen minimum/tie contract')
    return deepcopy(expected)


def select_metrics(metrics, old_summary, eligible):
    eligible = _eligible(eligible)
    if set(metrics) != {FAMILY}:
        raise ValueError('new metrics must contain only trace_full')
    _check_grid(metrics[FAMILY])
    historical = validate_historical_summary(old_summary)
    return dict(primary=_choose(metrics, FAMILY, eligible),
                unconstrained_primary=_choose(metrics, FAMILY), eligible_indices=eligible,
                controls=historical['controls'], strong_point=historical['strong_point'],
                strong_quadrature=historical['strong_quadrature'],
                historical_primary=historical['primary'])


def train_screen(metrics, old_summary, selection):
    """Evaluate the fixed selected primary once; never search for a fallback.

    Percent differences are descriptive. The comparisons use the declared
    direct multiplier bounds without a hidden numerical tolerance. A nonpositive
    reference cannot support a percentage improvement and therefore cannot pass.
    """
    primary = selection['primary']
    candidate = metrics[FAMILY][primary['grid_index']]
    old = old_summary['metrics']
    references = dict(strong_point=selection['strong_point'],
                      legacy_point=_spec(old, 'gaussian_point', 7),
                      strong_quadrature=selection['strong_quadrature'],
                      legacy_quad16=_spec(old, 'gaussian_quad16', 4))
    checks = []
    for metric, names, multiplier in (
            ('common9_kl', ('strong_point', 'legacy_point'), .98),
            ('direct5_kl', ('strong_point', 'legacy_point'), .98),
            ('common9_kl', ('strong_quadrature', 'legacy_quad16'), 1.01),
            ('direct5_kl', ('strong_quadrature', 'legacy_quad16'), 1.02)):
        for name in names:
            ref = references[name]
            actual = float(candidate[metric])
            expected = float(old[ref['family']][ref['grid_index']][metric])
            checks.append(dict(name=metric + '_vs_' + name, metric=metric,
                reference_name=name, reference=deepcopy(ref), candidate_value=actual,
                reference_value=expected, relative_difference=actual / expected - 1 if expected > 0 else None,
                maximum_relative_difference=multiplier - 1, bound=multiplier * expected,
                passed=bool(expected > 0 and actual <= multiplier * expected)))
    passed = all(check['passed'] for check in checks)
    return dict(passed=passed, checks=checks, fixed_primary=deepcopy(primary),
                scientific_gate_passed=False,
                next_action=('prepare_old_validation_protocol' if passed else 'stop_selected_candidate_before_old_validation'))


def summarize(new_rows, old_summary, eligible, *, entries=None):
    """Complete pure audit API; outer code owns provenance and execution order."""
    eligible = _eligible(eligible)
    metrics, population = summarize_metrics(new_rows, entries)
    selection = select_metrics(metrics, old_summary, eligible)
    return dict(metrics=metrics, selection=selection, population=population,
                train_screen=train_screen(metrics, old_summary, selection))
