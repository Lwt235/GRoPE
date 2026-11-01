"""Independent NumPy mathematics for Round15 TRAIN regional auditing.

No producer/model imports, files, hashes, CLI, or access to observations occur
here. Callers supply already-authorized arrays/rows and the frozen cohort.
Pixel rectangles use integer half-open bounds on the fixed 144 x 192 image;
their physical edges are the integer bounds minus one half pixel.
"""
from __future__ import annotations

from math import fsum

import numpy as np


SCENES = ('harvard_c3/hv_c3_1', 'harvard_c5/hv_c5_1', 'harvard_c6/hv_c6_1')
FAMILIES = ('quartic_full', 'quartic_point', 'gaussian_point', 'gaussian_fixed',
            'gaussian_full1', 'gaussian_full4', 'gaussian_quad4', 'gaussian_quad16')
GRID = tuple((h, t) for h in (.25, .5, 1., 2.) for t in (0., 5., 10.))
PARTITION_CONDITIONS = tuple((q, k) for q in ('g8', 'g16', 'mixed')
                             for k in ('g8', 'g16', 'mixed'))
SCALE_CONDITIONS = (('g8', 'g24'), ('g24', 'g8'), ('g16', 'g24'),
                    ('g24', 'g16'), ('g24', 'g24'))
CONDITIONS = PARTITION_CONDITIONS + SCALE_CONDITIONS
POINT_FAMILIES = ('gaussian_point', 'quartic_point')
QUADRATURE_FAMILIES = ('gaussian_quad4', 'gaussian_quad16')
IMAGE_HW = (144, 192)
NATIVE_SIZES = {'g8': 432, 'g16': 108, 'mixed': 243, 'g24': 48}
METRIC_FIELDS = ('kl_sum', 'js_sum', 'direct_kl_sum', 'direct_js_sum')


def partition_bounds(name):
    """Independently construct the four literal, observation-free partitions."""
    if name in ('g8', 'g16', 'g24'):
        side = int(name[1:])
        return np.asarray([(x, y, x + side, y + side)
                           for y in range(0, 144, side)
                           for x in range(0, 192, side)], dtype=np.int64)
    if name != 'mixed':
        raise ValueError('unknown native partition')
    bounds = []
    # Macroblocks cycle intact, 2x2, horizontal halves, vertical halves.
    for by in range(9):
        for bx in range(12):
            width, height = ((16, 16), (8, 8), (16, 8), (8, 16))[(bx + by) % 4]
            for dy in range(0, 16, height):
                for dx in range(0, 16, width):
                    x, y = bx * 16 + dx, by * 16 + dy
                    bounds.append((x, y, x + width, y + height))
    return np.asarray(bounds, dtype=np.int64)


def prepare_partition(bounds):
    """Validate literal coverage; return reusable native lookup/area metadata."""
    bounds = np.asarray(bounds)
    if (bounds.ndim != 2 or bounds.shape[1] != 4 or len(bounds) == 0
            or not np.isfinite(bounds).all()
            or not np.array_equal(bounds, bounds.astype(np.int64))):
        raise ValueError('bounds must be nonempty finite integer rectangles')
    bounds = bounds.astype(np.int64, copy=True)
    lookup = np.full(IMAGE_HW, -1, dtype=np.int64)
    for i, (x0, y0, x1, y1) in enumerate(bounds):
        if not (0 <= x0 < x1 <= 192 and 0 <= y0 < y1 <= 144):
            raise ValueError('rectangle outside fixed image')
        if np.any(lookup[y0:y1, x0:x1] != -1):
            raise ValueError('overlapping native rectangles')
        lookup[y0:y1, x0:x1] = i
    if np.any(lookup == -1):
        raise ValueError('incomplete native partition')
    area = (bounds[:, 2] - bounds[:, 0]) * (bounds[:, 3] - bounds[:, 1])
    parent = (bounds[:, 1] // 16) * 12 + bounds[:, 0] // 16
    contained = ((bounds[:, :2] // 16) == ((bounds[:, 2:] - 1) // 16)).all(-1)
    parent = np.where(contained, parent, -1)
    return dict(bounds=bounds, lookup=lookup, area=area, parent=parent,
                atoms=lookup[4::8, 4::8].ravel())


def _partition(value):
    # Prepared metadata is owned by this module/caller, never a producer object.
    return value if isinstance(value, dict) else prepare_partition(value)


def literal_parts():
    """Native saved-partition fields plus private reusable lookup metadata.

    Means/covariances are in reference-eight-pixel units, as required by the
    numerical audit: uniform rectangle covariance is diag(width**2)/12/64.
    These moments describe continuous pixel support, not just pixel centers.
    """
    parts = {}
    for name in ('g8', 'g16', 'mixed', 'g24'):
        part = prepare_partition(partition_bounds(name))
        bounds = part['bounds']
        width = bounds[:, 2:] - bounds[:, :2]
        cov = np.zeros((len(bounds), 2, 2), dtype=np.float64)
        cov[:, 0, 0] = width[:, 0] ** 2 / (12. * 64)
        cov[:, 1, 1] = width[:, 1] ** 2 / (12. * 64)
        part.update(owner=part['lookup'].ravel(), mass=part['area'],
                    mean=(bounds[:, :2] + bounds[:, 2:] - 1) / 16., cov=cov)
        parts[name] = part
    return parts


def reconstruct_targets(gt_xy, visible, query_bounds, key_bounds):
    """Return (native counts, common16-to8 counts) from every source pixel.

    Invisible points may be nonfinite. Visible points must already satisfy the
    independent sensor visibility decision; none are silently discarded here.
    The inherited nearest-pixel tie rule is floor(x + .5 + 1e-9), clipped only
    after checking the physical image interval [-.5, extent-.5).
    """
    xy, visible = np.asarray(gt_xy, np.float64), np.asarray(visible)
    if xy.shape != (144 * 192, 2) or visible.shape != (144 * 192,) or visible.dtype != np.bool_:
        raise ValueError('expected all dense source pixels and boolean visibility')
    good_xy = xy[visible]
    if (not np.isfinite(good_xy).all() or np.any(good_xy < -.5)
            or np.any(good_xy >= np.array([191.5, 143.5]))):
        raise ValueError('visible sensor projection outside image')
    q, k = _partition(query_bounds), _partition(key_bounds)
    target = np.floor(good_xy + .5 + 1e-9).astype(np.int64)
    target = np.clip(target, [0, 0], [191, 143])
    qi = q['lookup'].ravel()[visible]
    ki = k['lookup'][target[:, 1], target[:, 0]]
    direct = np.zeros((len(q['area']), len(k['area'])), dtype=np.int64)
    np.add.at(direct, (qi, ki), 1)
    source = np.flatnonzero(visible)
    common_q = ((source // 192) // 16) * 12 + (source % 192) // 16
    common_k = (target[:, 1] // 8) * 24 + target[:, 0] // 8
    common = np.zeros((108, 432), dtype=np.int64)
    np.add.at(common, (common_q, common_k), 1)
    if direct.sum() != visible.sum() or common.sum() != visible.sum():
        raise ValueError('lost visible population')
    return direct, common


def _probability(value, shape=None):
    probability = np.asarray(value, dtype=np.float64)
    if (probability.ndim != 2 or (shape is not None and probability.shape != shape)
            or not np.isfinite(probability).all() or np.any(probability < 0)
            or np.any(np.abs(probability.sum(-1) - 1) > 1e-12)):
        raise ValueError('invalid or unnormalized probability matrix')
    return probability


def common_prediction(probability, query_bounds, key_bounds):
    """Lift native keys by area and average source queries by area into g16.

    This is an observation-free area map, not a visible-mass-weighted average.
    g24 source queries have no exact g16 parent and are rejected.
    """
    q, k = _partition(query_bounds), _partition(key_bounds)
    p = _probability(probability, (len(q['area']), len(k['area'])))
    if np.any(q['parent'] < 0) or np.any(q['bounds'] % 8) or np.any(k['bounds'] % 8):
        raise ValueError('native partition has no exact common16-to8 map')
    if np.any(np.bincount(q['parent'], weights=q['area'], minlength=108) != 256):
        raise ValueError('common query parents do not cover 256 pixels')
    lifted = p[:, k['atoms']] * (64. / k['area'][k['atoms']])[None, :]
    output = np.zeros((108, 432), dtype=np.float64)
    np.add.at(output, q['parent'], lifted * (q['area'] / 256.)[:, None])
    return _probability(output, (108, 432))


def divergence_sums(probability, counts):
    """Visible-pixel weighted KL/JS with the frozen 1e-15 log clamp.

    Zero-mass query rows stay in the matrix and contribute exactly zero. The
    probability is not clipped or renormalized: only logarithms are clamped.
    """
    counts = np.asarray(counts)
    if (counts.ndim != 2 or not np.isfinite(counts).all() or np.any(counts < 0)
            or not np.array_equal(counts, counts.astype(np.int64))):
        raise ValueError('counts must be finite nonnegative integers')
    p = _probability(probability, counts.shape)
    mass = counts.sum(-1, dtype=np.int64)
    truth = counts / np.maximum(mass, 1)[:, None]
    log_p = np.log(np.maximum(p, 1e-15))
    log_g = np.log(np.maximum(truth, 1e-15))
    log_m = np.log(np.maximum((truth + p) / 2, 1e-15))
    kl = np.sum(counts * (log_g - log_p), dtype=np.float64)
    js_by_query = .5 * np.sum(truth * (log_g - log_m) + p * (log_p - log_m), axis=-1)
    return float(kl), float(np.dot(js_by_query, mass)), int(mass.sum())


def metric_reference(probability, direct_counts, common_counts,
                     query_bounds, key_bounds, scenario):
    """Independent canonical metric fields for one saved native probability."""
    direct_kl, direct_js, mass = divergence_sums(probability, direct_counts)
    if scenario == 'partition':
        kl, js, common_mass = divergence_sums(
            common_prediction(probability, query_bounds, key_bounds), common_counts)
        if common_mass != mass:
            raise ValueError('direct/common target populations differ')
    elif scenario == 'scale':
        kl, js = direct_kl, direct_js
    else:
        raise ValueError('unknown metric scenario')
    return dict(kl_sum=kl, js_sum=js, direct_kl_sum=direct_kl,
                direct_js_sum=direct_js, mass=mass,
                n_query=int(np.shape(probability)[0]), n_key=int(np.shape(probability)[1]))


def independent_counts(labels, parts, query, key):
    """Convenience interface for the source/hash-owning independent auditor."""
    return reconstruct_targets(labels['gt_xy'], labels['gt_visible'], parts[query], parts[key])


def measure(probability, parts, query, key, scenario, direct, common):
    return metric_reference(probability, direct, common, parts[query], parts[key], scenario)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    return int(value)


def _identity(row):
    pair = row['pair']
    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
        raise ValueError('pair must contain two integer frame indices')
    pair = tuple(_integer(x, 'frame index') for x in pair)
    if min(pair) < 0:
        raise ValueError('negative frame index')
    return row['scene'], pair


def _cohort(entries):
    identities = []
    for entry in entries:
        if 'split' in entry and entry['split'] != 'train':
            raise ValueError('cohort must be old TRAIN only')
        identities.append(_identity(entry))
    if (len(identities) != 48 or len(set(identities)) != 48
            or {s for s, _ in identities} != set(SCENES)
            or any(sum(s == scene for s, _ in identities) != 16 for scene in SCENES)):
        raise ValueError('expected 48 unique TRAIN pairs, 16 in each fixed scene')
    return set(identities)


def validate_population(rows, entries):
    """Verify all 64,512 identities and shared native/common pixel populations.

    A pair has one visible sensor population across every family, configuration
    and condition. n_query/n_key count all native tokens, including empty rows.
    Tiny signed roundoff in summed divergences is retained, never clamped.
    """
    cohort = _cohort(entries)
    if len(rows) != 48 * 8 * 12 * 14:
        raise ValueError('expected exactly 64512 metric rows')
    seen, masses = set(), {}
    condition_scenarios = {c: 'partition' for c in PARTITION_CONDITIONS}
    condition_scenarios.update({c: 'scale' for c in SCALE_CONDITIONS})
    for row in rows:
        identity = _identity(row)
        family = row['family']
        index = _integer(row['grid_index'], 'grid_index')
        condition = (row['query'], row['key'])
        if (identity not in cohort or family not in FAMILIES or not 0 <= index < 12
                or condition not in condition_scenarios
                or row['scenario'] != condition_scenarios[condition]):
            raise ValueError('unexpected cohort/family/grid/condition/scenario')
        if 'condition' in row and row['condition'] != '_'.join(condition):
            raise ValueError('condition name disagrees with query/key')
        if 'config' in row and row['config'] != dict(h=GRID[index][0], temperature=GRID[index][1]):
            raise ValueError('configuration disagrees with fixed grid index')
        unique = identity + (family, index) + condition
        if unique in seen:
            raise ValueError('duplicate metric row')
        seen.add(unique)
        mass = _integer(row['mass'], 'mass')
        if not 0 <= mass <= 144 * 192:
            raise ValueError('visible pixel mass outside image population')
        if identity in masses and masses[identity] != mass:
            raise ValueError('visible pixel mass differs across conditions/configurations')
        masses[identity] = mass
        if (_integer(row['n_query'], 'n_query') != NATIVE_SIZES[condition[0]]
                or _integer(row['n_key'], 'n_key') != NATIVE_SIZES[condition[1]]):
            raise ValueError('native query/key population changed')
        values = np.asarray([row[field] for field in METRIC_FIELDS], dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values < -1e-8):
            raise ValueError('invalid summed divergence')
        if np.any(values[[1, 3]] > mass * np.log(2.) + 1e-8):
            raise ValueError('JS exceeds its visible-mass bound')
        if mass == 0 and np.any(values != 0):
            raise ValueError('zero visible mass has nonzero metric contribution')
        if row['scenario'] == 'scale' and (values[0] != values[2] or values[1] != values[3]):
            raise ValueError('scale metrics must be direct native metrics')
    # Unique membership plus the exact Cartesian-product cardinality establishes
    # completeness; no selective subset can pass by duplicating a different row.
    if len(seen) != len(cohort) * len(FAMILIES) * len(GRID) * len(CONDITIONS):
        raise ValueError('incomplete Cartesian metric population')
    mass_by_scene = {s: sum(m for (scene, _), m in masses.items() if scene == s) for s in SCENES}
    if any(m <= 0 for m in mass_by_scene.values()):
        raise ValueError('scene has no visible sensor mass')
    return dict(rows=len(rows), pairs=len(cohort), mass_by_scene=mass_by_scene,
                unique_visible_pixel_mass=sum(masses.values()))


def summarize_metrics(rows, entries):
    """Compute fixed scene/condition weighted metrics without any selection."""
    population = validate_population(rows, entries)
    buckets = {}
    for row in rows:
        key = row['family'], row['grid_index'], row['scene'], row['query'] + '_' + row['key']
        buckets.setdefault(key, []).append(row)
    output = {}
    names = tuple(q + '_' + k for q, k in CONDITIONS)
    for family in FAMILIES:
        family_rows = []
        for index, (h, temperature) in enumerate(GRID):
            by_scene_condition, by_scene = {}, {}
            for scene in SCENES:
                conditions = {}
                for name in names:
                    group = sorted(buckets[family, index, scene, name], key=lambda r: tuple(r['pair']))
                    mass = sum(r['mass'] for r in group)
                    conditions[name] = {
                        label: fsum(float(r[field]) for r in group) / mass
                        for label, field in zip(('kl', 'js', 'direct_kl', 'direct_js'), METRIC_FIELDS)}
                    conditions[name]['mass'] = mass
                common9 = fsum(conditions[n]['kl'] for n in names[:9]) / 9
                direct5 = fsum(conditions[n]['direct_kl'] for n in names[9:]) / 5
                by_scene_condition[scene] = conditions
                by_scene[scene] = dict(J=.5 * common9 + .5 * direct5,
                                       common9_kl=common9, direct5_kl=direct5)
            common9 = fsum(by_scene[s]['common9_kl'] for s in SCENES) / 3
            direct5 = fsum(by_scene[s]['direct5_kl'] for s in SCENES) / 3
            by_condition = {name: {
                field: fsum(by_scene_condition[s][name][field] for s in SCENES) / 3
                for field in ('kl', 'js', 'direct_kl', 'direct_js')} for name in names}
            family_rows.append(dict(grid_index=index, config=dict(h=h, temperature=temperature),
                J=.5 * common9 + .5 * direct5, common9_kl=common9, direct5_kl=direct5,
                by_scene=by_scene, by_condition=by_condition, by_scene_condition=by_scene_condition))
        output[family] = family_rows
    return output, population


def select_metrics(metrics, feasible_indices):
    """Select by TRAIN J only; localization constraints apply to primary only.

    Exact ties take the first fixed grid index. Strong point ties prefer the
    Gaussian point; quadrature ties prefer four points. PCK is intentionally not
    an input and cannot weaken a regional control.
    """
    eligible = [_integer(i, 'eligible index') for i in feasible_indices]
    if not eligible:
        raise ValueError('no Stage A feasible candidate; stop before Stage B selection')
    if len(set(eligible)) != len(eligible) or any(not 0 <= i < 12 for i in eligible):
        raise ValueError('invalid feasible grid set')
    eligible.sort()
    if set(metrics) != set(FAMILIES):
        raise ValueError('all eight regional families are required')
    for family in FAMILIES:
        values = metrics[family]
        if len(values) != 12:
            raise ValueError('each regional family requires all twelve configurations')
        for i, value in enumerate(values):
            if (_integer(value['grid_index'], 'grid_index') != i
                    or value['config'] != dict(h=GRID[i][0], temperature=GRID[i][1])
                    or not np.isfinite(value['J'])):
                raise ValueError('invalid ordered regional grid')

    def choose(family, indices):
        index = min(indices, key=lambda i: metrics[family][i]['J'])
        value = metrics[family][index]
        return dict(family=family, grid_index=index, config=dict(value['config']), J=float(value['J']))

    controls = {f: choose(f, range(12)) for f in FAMILIES[1:]}
    return dict(primary=choose('quartic_full', eligible), controls=controls,
                unconstrained_primary=choose('quartic_full', range(12)),
                strong_point=dict(min((controls[f] for f in POINT_FAMILIES), key=lambda v: v['J'])),
                strong_quadrature=dict(min((controls[f] for f in QUADRATURE_FAMILIES), key=lambda v: v['J'])),
                eligible_indices=eligible)


def summarize_and_select(rows, entries, feasible_indices):
    """Root-auditor interface: full independent metrics, population and selection."""
    metrics, population = summarize_metrics(rows, entries)
    return dict(metrics=metrics, population=population,
                selection=select_metrics(metrics, feasible_indices))


def aggregate(rows, eligible_indices, entries):
    """Alias with argument order convenient for the independent root auditor."""
    return summarize_and_select(rows, entries, eligible_indices)
