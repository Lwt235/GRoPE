"""Fixed covariance-matched t4 native-cell mass, with positive tiled GL24.

This module reads no files. It is neither Gaussian convolution nor the ray
operator. The supplied covariance is an actual covariance, not Student shape.
"""
from __future__ import annotations

from functools import lru_cache
import math
import time

import numpy as np


class NumericalFailure(RuntimeError):
    pass


@lru_cache(None)
def _legendre(order):
    return np.polynomial.legendre.leggauss(order)


def _float64(value, shape, name):
    array = np.asarray(value)
    if array.dtype != np.float64 or array.shape != shape or not np.isfinite(array).all():
        raise ValueError('finite float64 '+name+' with prescribed shape required')
    return array


def prepare(mu, covariance, bounds, content_bias, policy):
    mu = _float64(mu, (2,), 'mu')
    covariance = _float64(covariance, (2, 2), 'covariance')
    bounds = np.asarray(bounds)
    if bounds.ndim != 2 or bounds.shape[1:] != (4,) or not len(bounds):
        raise ValueError('nonempty native (N,4) bounds required')
    bounds = _float64(bounds, bounds.shape, 'bounds')
    bias = np.zeros(len(bounds), np.float64) if content_bias is None else _float64(content_bias, (len(bounds),), 'content_bias')
    if (policy['degrees_of_freedom'] != 4 or policy['quadrature_order'] != 24
            or policy['maximum_tile_side_in_h'] != 4.):
        raise ValueError('fixed t4/GL24/4h production policy required')
    h, floor_density = float(policy['h']), float(policy['floor_density'])
    if not (math.isfinite(h) and h > 0 and math.isfinite(floor_density) and floor_density > 0):
        raise ValueError('positive finite h and floor required')
    if int(policy['max_cell_node_evaluations']) != policy['max_cell_node_evaluations'] or policy['max_cell_node_evaluations'] <= 0:
        raise ValueError('positive integer work budget required')
    if not np.array_equal(covariance, covariance.T):
        raise ValueError('explicitly symmetric covariance required; no symmetrization')
    scale = max(1., float(np.max(abs(covariance))))
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues[0] < -float(policy['covariance_admission_relative_tolerance'])*scale:
        raise ValueError('covariance is not PSD within declared admission tolerance')
    with np.errstate(over='ignore', invalid='ignore'):
        total = covariance+h*h*np.eye(2)
    if not np.isfinite(total).all():
        raise ValueError('finite total covariance required; no overflow repair')
    factor = np.linalg.cholesky(total)
    if not np.isfinite(factor).all() or not (np.diag(factor) > 0).all():
        raise ValueError('finite Cholesky factor with positive diagonal required')
    size = bounds[:, 2:]-bounds[:, :2]
    if not (size > 0).all():
        raise ValueError('strictly positive native rectangle sides required')
    areas = np.prod(size, axis=1)
    enclosing = np.r_[bounds[:, :2].min(0), bounds[:, 2:].max(0)]
    overlap = np.minimum(bounds[:, None, 2:], bounds[None, :, 2:])-np.maximum(bounds[:, None, :2], bounds[None, :, :2])
    duplicated = (overlap > 0).all(-1)
    np.fill_diagonal(duplicated, False)
    if (duplicated.any() or abs(float(areas.sum())-float(np.prod(enclosing[2:]-enclosing[:2])))
            > 1e-12*max(1., float(areas.sum()))):
        raise ValueError('disjoint cells must tile one rectangular FOV')
    return dict(mu=mu, covariance=covariance, total_covariance=total, factor=factor,
                bounds=bounds, bias=bias, areas=areas, fov=enclosing,
                admitted_covariance_min_eigenvalue=float(eigenvalues[0]), h=h, floor_density=floor_density)


def integration_tiles(bounds, maximum_side, maximum_tiles=None):
    """Exact equal subdivisions; counts depend only on declared cell geometry."""
    tiles, owners = [], []
    for index, box in enumerate(bounds):
        counts = np.ceil((box[2:]-box[:2])/maximum_side).astype(np.int64)
        if (counts <= 0).any():
            raise NumericalFailure('invalid tile subdivision count')
        nx, ny = map(int, counts)
        if maximum_tiles is not None and len(tiles)+nx*ny > maximum_tiles:
            raise NumericalFailure('tile allocation exceeds fixed cell-node budget')
        width = (box[2:]-box[:2])/counts
        for iy in range(ny):
            for ix in range(nx):
                low = box[:2]+np.asarray([ix, iy])*width
                high = box[:2]+np.asarray([ix+1, iy+1])*width
                tiles.append(np.r_[low, high]); owners.append(index)
    return np.asarray(tiles, np.float64), np.asarray(owners, np.int64)


def _tile_integrals(tiles, prepared, work, policy, deadline, result, owner, scope):
    nodes, weights = _legendre(24)
    product = weights[:, None]*weights[None, :]
    factor, mean = prepared['factor'], prepared['mu']
    log_normalizer = -math.log(math.pi)-math.log(factor[0, 0])-math.log(factor[1, 1])
    for start in range(0, len(tiles), 32):
        if deadline is not None and time.perf_counter() >= deadline:
            raise NumericalFailure('production phase wall budget exceeded')
        batch = tiles[start:start+32]
        charged = len(batch)*24*24
        if work['cell_node_evaluations']+charged > policy['max_cell_node_evaluations']:
            raise NumericalFailure('cell-node evaluation budget exceeded')
        work['cell_node_evaluations'] += charged
        middle, half = (batch[:, :2]+batch[:, 2:])/2, (batch[:, 2:]-batch[:, :2])/2
        x = middle[:, 0, None]+half[:, 0, None]*nodes
        y = middle[:, 1, None]+half[:, 1, None]*nodes
        u = (x-mean[0])/factor[0, 0]
        v = (y[:, None, :]-mean[1]-factor[1, 0]*u[:, :, None])/factor[1, 1]
        radial = u[:, :, None]**2+v*v
        if not np.isfinite(radial).all():
            raise NumericalFailure('nonfinite Mahalanobis radius; no tail repair')
        density = np.exp(log_normalizer-3.*np.log1p(radial/2.))
        values = np.sum(density*product, axis=(1, 2))*np.prod(half, axis=1)
        if not np.isfinite(values).all() or (values < 0).any():
            raise NumericalFailure('invalid positive GL mass')
        np.add.at(result, owner[start:start+len(batch)], values)
        work[scope+'_tiles_completed'] += len(batch)


def integrate(mu, covariance, bounds, content_bias=None, *, policy, deadline=None):
    """Compute raw density mass before one floor and native constant content."""
    p = prepare(mu, covariance, bounds, content_bias, policy)
    work = dict(cell_node_evaluations=0, native_tiles_completed=0, fov_tiles_completed=0)
    common = dict(status='failed', passed=False, work=work, degrees_of_freedom=4,
                  quadrature_order=24, h=p['h'], floor_density=p['floor_density'],
                  admitted_covariance_min_eigenvalue=p['admitted_covariance_min_eigenvalue'],
                  quadrature_error_bound_available=False, tail_truncation_used=False)
    raw, direct = np.zeros(len(p['bounds']), np.float64), np.zeros(1, np.float64)
    try:
        if deadline is not None and time.perf_counter() >= deadline:
            raise NumericalFailure('production phase wall budget exceeded')
        maximum_side = policy['maximum_tile_side_in_h']*p['h']
        maximum_tiles = int(policy['max_cell_node_evaluations'])//(24*24)
        tiles, owner = integration_tiles(p['bounds'], maximum_side, maximum_tiles)
        fov_tiles, fov_owner = integration_tiles(p['fov'][None], maximum_side, maximum_tiles)
        common.update(native_tile_count=len(tiles), fov_tile_count=len(fov_tiles))
        _tile_integrals(tiles, p, work, policy, deadline, raw, owner, 'native')
        _tile_integrals(fov_tiles, p, work, policy, deadline, direct, fov_owner, 'fov')
        floor = p['floor_density']*p['areas']
        floored = raw+floor
        normalizer = float(np.sum(floored))
        if not math.isfinite(normalizer) or normalizer <= 0:
            raise NumericalFailure('invalid raw-plus-floor denominator')
        logits = np.log(floored)+p['bias']
        shifted = np.exp(logits-float(np.max(logits)))
        probability = shifted/float(np.sum(shifted))
        if not np.isfinite(probability).all() or (probability < 0).any():
            raise NumericalFailure('invalid normalized probability')
        common.update(status='complete', passed=True, raw_mass=raw, probability=probability,
            floored_mass=floored, raw_mass_sum=float(np.sum(raw)), raw_fov_mass=float(direct[0]),
            normalizer=normalizer, log_content_normalizer=float(np.max(logits)+math.log(float(np.sum(shifted)))),
            total_area_floor=float(np.sum(floor)), zero_raw_cells=int(np.sum(raw == 0)),
            area_prior_used=bool(np.all(raw == 0) and np.all(p['bias'] == p['bias'][0])),
            all_raw_mass_zero=bool(np.all(raw == 0)), raw_mass_below_total_floor=bool(np.sum(raw) < np.sum(floor)),
            raw_mass_above_one_roundoff=max(0., float(np.max(np.r_[raw, direct]))-1.),
            max_normalization_error=abs(float(np.sum(probability))-1.), argmax_index=int(np.argmax(probability)),
            argmax_exact_ties=int(np.sum(probability == np.max(probability))),
            argmax_near_ties_at_1e_12=int(np.sum(np.max(probability)-probability <= 1e-12)))
    except (NumericalFailure, FloatingPointError, OverflowError) as error:
        common.update(failure_reason=str(error), partial_raw_mass=raw,
                      partial_raw_fov_mass=direct.copy())
    return common
