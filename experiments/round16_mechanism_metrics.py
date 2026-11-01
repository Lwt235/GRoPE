"""Pure analysis of already saved Round15 probabilities; no inference or I/O.

All distances are physical pixels. Query covariance is in reference-8 units.
Correctness is ternary: -1 means an unavailable target, never an error. The
``all`` population includes every query for geometry, but its PCK denominator
still counts only valid targets. Phase and axial angle are retained per query;
we deliberately do not average those circular quantities as linear scalars.
"""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np


NUMERIC_FIELDS = (
    'dist_mu_px', 'zero_dist_mu_px', 'nearest_dist_mu_px',
    'mu_target_dist_px', 'winner_to_zero_dx_px', 'winner_to_zero_dy_px',
    'winner_to_zero_dist_px', 'top_geometry_gap', 'geom_penalty',
    'content_gain', 'advantage', 'identity_residual', 'identity_error',
    'fov_distance_px', 'S_condition',
)
PAIRED_NUMERIC_FIELDS = (
    'dist_mu_px_difference', 'zero_dist_mu_px_difference',
    'top_geometry_gap_difference',
)


def _array(value, shape, name, dtype=None):
    a = np.asarray(value)
    if a.shape != shape or (dtype is not None and a.dtype != dtype):
        raise ValueError(f'{name}: expected shape {shape} and dtype {dtype}')
    return a


def _probability(value, name, shape=None):
    a = np.asarray(value)
    if a.ndim != 2 or a.dtype != np.float64 or a.shape[1] < 2:
        raise ValueError(f'{name}: expected float64 [Q,K], K >= 2')
    if shape is not None and a.shape != shape:
        raise ValueError(f'{name}: probability shapes differ')
    if (not np.isfinite(a).all() or np.any(a <= 0)
            or np.max(np.abs(a.sum(1) - 1), initial=0) > 1.e-12):
        raise ValueError(f'{name}: expected strictly positive normalized probabilities')
    return a


def _grid(key):
    """Native g8 Cartesian centers, in their original (possibly permuted) order."""
    axes = [np.unique(key[:, d]) for d in range(2)]
    if (any(len(a) > 1 and not np.all(np.diff(a) == 8.) for a in axes)
            or len(axes[0]) * len(axes[1]) != len(key)
            or len(np.unique(key, axis=0)) != len(key)):
        raise ValueError('key_px must be a complete native 8-pixel Cartesian grid')
    return np.array([a[0] - 4. for a in axes]), np.array([a[-1] + 4. for a in axes])


def _correct(winner, key, target, valid):
    answer = np.full(len(winner), -1, dtype=np.int8)
    answer[valid] = (np.square(key[winner[valid]] - target[valid]).sum(1) <= 64.)
    return answer


def analyze(p0, pT, content32, mu_px, cov_reference8, key_px, target_px,
            valid, *, h, temperature):
    """Return one-dimensional arrays without filtering any query.

    ``p0`` and ``pT`` must use the same family, bandwidth, and native query/key
    partitions. The caller binds this identity to saved artifacts. T*content is
    rounded in float32 *before* conversion and subtraction, as in production.
    Winners and the second geometry winner use exact, first-index ties.
    """
    p0 = _probability(p0, 'p0')
    pT = _probability(pT, 'pT', p0.shape)
    n, k = p0.shape
    content = _array(content32, (n, k), 'content32', np.dtype(np.float32))
    mu = _array(mu_px, (n, 2), 'mu_px').astype(np.float64)
    cov = _array(cov_reference8, (n, 2, 2), 'cov_reference8').astype(np.float64)
    key = _array(key_px, (k, 2), 'key_px').astype(np.float64)
    target = _array(target_px, (n, 2), 'target_px').astype(np.float64)
    valid = _array(valid, (n,), 'valid', np.dtype(np.bool_))
    if (not all(np.isfinite(a).all() for a in (content, mu, cov, key))
            or not np.isfinite(target[valid]).all()
            or not math.isfinite(float(h)) or h <= 0
            or not math.isfinite(float(temperature)) or temperature < 0):
        raise ValueError('nonfinite geometry/content/valid target or invalid h/T')
    covariance_scale = np.maximum(1., np.max(np.abs(cov), axis=(1, 2)))
    if not np.all(np.abs(cov - cov.swapaxes(1, 2)) <= 1.e-12 * covariance_scale[:, None, None]):
        raise ValueError('cov_reference8 must be symmetric')
    lower, upper = _grid(key)
    regularized = cov + float(h) ** 2 * np.eye(2)[None]
    eigenvalues, eigenvectors = np.linalg.eigh(regularized)
    if np.any(eigenvalues[:, 0] <= 0):
        raise ValueError('C+h^2 I must be positive definite')
    isotropic = eigenvalues[:, 0] == eigenvalues[:, 1]
    major = eigenvectors[:, :, 1]
    angle = np.remainder(np.arctan2(major[:, 1], major[:, 0]), np.pi)
    angle[angle == np.pi] = 0.
    angle[isotropic] = 0.

    zero_winner = np.argmax(p0, axis=1).astype(np.int64)
    winner = np.argmax(pT, axis=1).astype(np.int64)
    rows = np.arange(n)
    runners_up = p0.copy()
    runners_up[rows, zero_winner] = -np.inf
    j2 = np.argmax(runners_up, axis=1).astype(np.int64)
    distance2 = np.square(mu[:, None] - key[None]).sum(2)
    nearest = np.argmin(distance2, axis=1).astype(np.int64)
    lp0, lpT = np.log(p0), np.log(pT)
    scaled = np.multiply(np.float32(temperature), content, dtype=np.float32).astype(np.float64)
    if not np.isfinite(scaled).all():
        raise ValueError('float32 temperature multiplication overflowed')
    penalty = lp0[rows, zero_winner] - lp0[rows, winner]
    gain = scaled[rows, winner] - scaled[rows, zero_winner]
    advantage = lpT[rows, winner] - lpT[rows, zero_winner]
    residual = advantage - (gain - penalty)
    displacement = key[winner] - key[zero_winner]
    mu_target = np.full(n, np.nan, dtype=np.float64)
    mu_target[valid] = np.linalg.norm(mu[valid] - target[valid], axis=1)
    phase = np.remainder((mu - lower) / 8., 1.)
    outside = np.maximum(np.maximum(lower - mu, mu - upper), 0.)
    return {
        'winner': winner, 'zero_winner': zero_winner, 'j2': j2,
        'nearest_mu': nearest,
        'tie_count': np.sum(p0 == p0[rows, zero_winner, None], axis=1, dtype=np.int64),
        'final_tie_count': np.sum(pT == pT[rows, winner, None], axis=1, dtype=np.int64),
        'flip': winner != zero_winner, 'valid': valid.copy(),
        'correct': _correct(winner, key, target, valid),
        'zero_correct': _correct(zero_winner, key, target, valid),
        'nearest_correct': _correct(nearest, key, target, valid),
        'dist_mu_px': np.sqrt(distance2[rows, winner]),
        'zero_dist_mu_px': np.sqrt(distance2[rows, zero_winner]),
        'nearest_dist_mu_px': np.sqrt(distance2[rows, nearest]),
        'mu_target_dist_px': mu_target,
        'winner_to_zero_dx_px': displacement[:, 0],
        'winner_to_zero_dy_px': displacement[:, 1],
        'winner_to_zero_dist_px': np.linalg.norm(displacement, axis=1),
        'top_geometry_gap': lp0[rows, zero_winner] - lp0[rows, j2],
        'geom_penalty': penalty, 'content_gain': gain, 'advantage': advantage,
        'identity_residual': residual, 'identity_error': np.abs(residual),
        'mu_phase_x': phase[:, 0], 'mu_phase_y': phase[:, 1],
        'mu_in_fov': np.all((mu >= lower) & (mu < upper), axis=1),
        'fov_distance_px': np.linalg.norm(outside, axis=1),
        'S_condition': eigenvalues[:, 1] / eigenvalues[:, 0],
        'S_major_angle_rad': angle, 'isotropic': isotropic,
    }


def _validate_arrays(arrays):
    valid = np.asarray(arrays['valid'])
    if valid.ndim != 1 or valid.dtype != np.bool_:
        raise ValueError('valid must be a one-dimensional bool array')
    n = len(valid)
    for name, value in arrays.items():
        if np.asarray(value).shape != (n,):
            raise ValueError(f'{name}: expected a one-dimensional query array')
    for name in ('correct', 'zero_correct', 'nearest_correct'):
        a = np.asarray(arrays[name])
        if (not np.issubdtype(a.dtype, np.integer)
                or not np.isin(a[valid], [0, 1]).all()
                or not np.all(a[~valid] == -1)):
            raise ValueError('correctness must be 0/1 on valid and -1 on invalid targets')
    return valid


def _moments(out, name, values, mask):
    values = np.asarray(values, dtype=np.float64)[mask]
    if not np.isfinite(values).all():
        raise ValueError(f'{name}: nonfinite value in its reporting population')
    out[name + '_count'] = int(len(values))
    out[name + '_sum'] = math.fsum(float(x) for x in values)
    out[name + '_sum_sq'] = math.fsum(float(x) * float(x) for x in values)


def stats(arrays):
    """Additive geometry and valid-target transition counts for two populations."""
    valid = _validate_arrays(arrays)
    out = {}
    for population, mask in (('all', np.ones(len(valid), dtype=bool)), ('valid', valid)):
        labeled = mask & valid
        correct, zero, nearest = (np.asarray(arrays[k]) == 1 for k in
                                  ('correct', 'zero_correct', 'nearest_correct'))
        count = lambda condition: int(np.count_nonzero(mask & condition))
        labeled_count = lambda condition: int(np.count_nonzero(labeled & condition))
        result = {
            'n': int(mask.sum()), 'n_valid': int(labeled.sum()),
            'n_flip': count(arrays['flip']),
            'n_geometry_tie': count(arrays['tie_count'] > 1),
            'n_final_tie': count(arrays['final_tie_count'] > 1),
            'n_isotropic': count(arrays['isotropic']),
            'n_mu_in_fov': count(arrays['mu_in_fov']),
            'n_correct': labeled_count(correct),
            'n_zero_correct': labeled_count(zero),
            'n_nearest_correct': labeled_count(nearest),
            'n_both_correct': labeled_count(zero & correct),
            'n_both_wrong': labeled_count(~zero & ~correct),
            'n_harm': labeled_count(zero & ~correct),
            'n_benefit': labeled_count(~zero & correct),
            'n_nearest_harm': labeled_count(nearest & ~correct),
            'n_nearest_benefit': labeled_count(~nearest & correct),
        }
        for name in NUMERIC_FIELDS:
            _moments(result, name, arrays[name], labeled if name == 'mu_target_dist_px' else mask)
        out[population] = result
    return out


def paired_stats(full, point):
    """Matched full-minus-point statistics; caller fixes family/h/T identity."""
    valid = _validate_arrays(full)
    other_valid = _validate_arrays(point)
    if not np.array_equal(valid, other_valid):
        raise ValueError('paired targets/valid population differ')
    for field in ('nearest_mu', 'nearest_correct', 'nearest_dist_mu_px',
                  'mu_target_dist_px', 'mu_phase_x', 'mu_phase_y',
                  'S_condition', 'S_major_angle_rad', 'isotropic'):
        if not np.array_equal(full[field], point[field], equal_nan=True):
            raise ValueError(f'paired shared geometry/target differs: {field}')
    differences = {name + '_difference': full[name] - point[name] for name in
                   ('dist_mu_px', 'zero_dist_mu_px', 'top_geometry_gap')}
    out = {}
    for population, mask in (('all', np.ones(len(valid), dtype=bool)), ('valid', valid)):
        labeled = mask & valid
        fc, pc = full['correct'] == 1, point['correct'] == 1
        count = lambda condition: int(np.count_nonzero(mask & condition))
        labeled_count = lambda condition: int(np.count_nonzero(labeled & condition))
        result = {
            'n': int(mask.sum()), 'n_valid': int(labeled.sum()),
            'n_same_winner': count(full['winner'] == point['winner']),
            'n_same_zero_winner': count(full['zero_winner'] == point['zero_winner']),
            'n_full_flip': count(full['flip']), 'n_point_flip': count(point['flip']),
            'n_full_flip_point_not': count(full['flip'] & ~point['flip']),
            'n_point_flip_full_not': count(point['flip'] & ~full['flip']),
            'n_full_correct': labeled_count(fc), 'n_point_correct': labeled_count(pc),
            'n_both_correct': labeled_count(fc & pc),
            'n_both_wrong': labeled_count(~fc & ~pc),
            'n_full_correct_point_wrong': labeled_count(fc & ~pc),
            'n_full_wrong_point_correct': labeled_count(~fc & pc),
        }
        for name, values in differences.items():
            _moments(result, name, values, mask)
        out[population] = result
    return out


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _derive(additive):
    """Ratios only: integer populations remain in the separate additive block."""
    a = additive
    ordinary = {
        'n_flip': 'flip_rate', 'n_geometry_tie': 'geometry_tie_rate',
        'n_final_tie': 'final_tie_rate', 'n_isotropic': 'isotropic_rate',
        'n_mu_in_fov': 'mu_in_fov_rate', 'n_same_winner': 'same_winner_rate',
        'n_same_zero_winner': 'same_zero_winner_rate',
        'n_full_flip': 'full_flip_rate', 'n_point_flip': 'point_flip_rate',
        'n_full_flip_point_not': 'full_only_flip_rate',
        'n_point_flip_full_not': 'point_only_flip_rate',
    }
    labeled = {
        'n_correct': 'pck8', 'n_zero_correct': 'zero_pck8',
        'n_nearest_correct': 'nearest_pck8', 'n_both_correct': 'both_correct_rate',
        'n_both_wrong': 'both_wrong_rate', 'n_harm': 'harm_rate',
        'n_benefit': 'benefit_rate', 'n_nearest_harm': 'nearest_harm_rate',
        'n_nearest_benefit': 'nearest_benefit_rate',
        'n_full_correct': 'full_pck8', 'n_point_correct': 'point_pck8',
        'n_full_correct_point_wrong': 'full_benefit_rate',
        'n_full_wrong_point_correct': 'full_harm_rate',
    }
    derived = {name: _ratio(a[key], a['n']) for key, name in ordinary.items() if key in a}
    derived.update({name: _ratio(a[key], a['n_valid']) for key, name in labeled.items() if key in a})
    if 'n_correct' in a:
        derived['net_pck_change'] = _ratio(a['n_correct'] - a['n_zero_correct'], a['n_valid'])
    if 'n_full_correct' in a:
        derived['full_minus_point_pck'] = _ratio(a['n_full_correct'] - a['n_point_correct'], a['n_valid'])
    for key in a:
        if key.endswith('_count'):
            name = key[:-6]
            count = a[key]
            mean = _ratio(a[name + '_sum'], count)
            derived[name + '_mean'] = mean
            derived[name + '_std'] = (
                math.sqrt(max(0., a[name + '_sum_sq'] / count - mean ** 2)) if count else None)
    return derived


def _merge(items):
    keys = set(items[0])
    if any(set(item) != keys for item in items):
        raise ValueError('inconsistent additive schema')
    result = {}
    for name in sorted(keys):
        values = [item[name] for item in items]
        integer = name == 'n' or name.startswith('n_') or name.endswith('_count')
        if integer:
            if any(not isinstance(v, (int, np.integer)) or isinstance(v, (bool, np.bool_)) or v < 0 for v in values):
                raise ValueError(f'{name}: invalid integer count')
            result[name] = sum(int(v) for v in values)
        else:
            if not all(np.isfinite(v) for v in values):
                raise ValueError(f'{name}: nonfinite additive scalar')
            result[name] = math.fsum(float(v) for v in values)
    return result


def summarize(rows):
    """Pool pairs within scene; average scene-derived statistics with equal weight.

    A missing denominator in any included scene produces a ``None`` macro for
    that statistic, rather than silently changing its scene population. No
    cross-scale or cross-configuration score is created. Paired rows may use
    the comparison label in ``family`` and the same function unchanged.
    """
    groups = defaultdict(lambda: defaultdict(list))
    identities = set()
    for row in rows:
        pair = tuple(row['pair']) if isinstance(row['pair'], (list, tuple)) else row['pair']
        identity = (row['scene'], pair, row['family'], row['grid_index'], row['query'])
        if identity in identities:
            raise ValueError(f'duplicate row identity: {identity}')
        identities.add(identity)
        if set(row['stats']) != {'all', 'valid'}:
            raise ValueError('expected all and valid populations')
        groups[(row['family'], row['grid_index'], row['query'])][row['scene']].append(row)
    result = {}
    for (family, grid, query), scenes in sorted(groups.items()):
        by_scene = {}
        for scene, items in sorted(scenes.items()):
            items = sorted(items, key=lambda row: str(row['pair']))
            by_scene[scene] = {}
            for population in ('all', 'valid'):
                additive = _merge([row['stats'][population] for row in items])
                by_scene[scene][population] = {'additive': additive, 'derived': _derive(additive)}
        macro = {}
        for population in ('all', 'valid'):
            scene_derived = [value[population]['derived'] for value in by_scene.values()]
            if any(set(d) != set(scene_derived[0]) for d in scene_derived):
                raise ValueError('inconsistent scene derived schema')
            macro[population] = {}
            for name in sorted(scene_derived[0]):
                values = [d[name] for d in scene_derived]
                macro[population][name] = (None if any(v is None for v in values)
                                           else math.fsum(values) / len(values))
        result.setdefault(family, {}).setdefault(str(grid), {})[query] = {
            'by_scene': by_scene, 'macro': macro,
        }
    return result
