"""Role-specific forward preparation for the R15 development-validation draft.

Only four predicted-observation tensors are accepted. This module does not read
files, labels, cameras, selection results, or set process/thread configuration.
The caller owns the fixed CPU-four-thread runtime and all promotion guards.

Physical input coordinates/covariances are divided by 8/64. Quartic integrates
native half-pixel rectangles with the frozen tiled GL24 operator; Gaussian
retains the R12 GL32 children (a 24px key has four 12px integration children).
Children are numerical integration subdivisions, never additional tokens.
"""
from __future__ import annotations

import torch

from models.round10_partition_pe import group_mean, pool_content, prepare_support
from models.round11_stratum_pe import prepare_gaussian_strata
from models.round10_cell_integral_fast import gaussian_cell_mass_fast
from models.round15_quartic_tiled import quartic_cell_mass


FAMILIES = ('quartic_full', 'quartic_point', 'gaussian_point', 'gaussian_fixed',
            'gaussian_full1', 'gaussian_full4', 'gaussian_quad4', 'gaussian_quad16',
            'quartic_isotropic_matched')
OBSERVATION_FIELDS = frozenset(('q_to_k_xy', 'q_to_k_pixel_cov', 'q_features', 'k_features'))


def _observations(observations):
    if set(observations) != OBSERVATION_FIELDS:
        raise ValueError('exact four predicted-observation fields required')
    if any(not isinstance(v, torch.Tensor) or v.dtype != torch.float32
           for v in observations.values()):
        raise ValueError('production observations must be float32 tensors')


def _family(family):
    if family not in FAMILIES:
        raise ValueError('undeclared positional family')


def prepare_query(observations, partition, family):
    """Compute only this family's statistics, using the original float32 order.

    Point/fixed have no full scatter; quadrature has no stratum scatter. The
    matched isotropic role accumulates only the two covariance diagonals, then
    averages them in double precision. It never borrows the primary's C or
    constructs a per-pixel off-diagonal outer product.
    """
    _observations(observations)
    _family(family)
    xy = observations['q_to_k_xy']/8
    uses_pixel_matrix = family in ('quartic_full', 'gaussian_full1', 'gaussian_full4',
                                  'gaussian_quad4', 'gaussian_quad16')
    pixel_cov = observations['q_to_k_pixel_cov']/64 if uses_pixel_matrix else None
    owner, bounds, parent = (partition[k] for k in ('owner', 'bounds', 'parent'))
    args = (xy, pixel_cov, owner, bounds, parent)
    count = len(bounds)
    if family == 'gaussian_full4':
        query = prepare_gaussian_strata(*args)
    elif family in ('gaussian_quad4', 'gaussian_quad16'):
        samples = 4 if family == 'gaussian_quad4' else 16
        query = prepare_support(*args, samples=samples, include_full_moments=False)
    elif family in ('quartic_full', 'gaussian_full1'):
        query = prepare_support(*args, include_full_moments=True)
    else:
        query = prepare_support(*args, include_full_moments=False)
        if family == 'quartic_isotropic_matched':
            centered = xy-query['mean'][owner]
            within_diagonal = observations['q_to_k_pixel_cov'].diagonal(dim1=-2, dim2=-1)/64
            diagonal = group_mean(centered.square()+within_diagonal,
                                  owner, count)
            variance = diagonal.double().mean(-1)
            query['cov'] = torch.diag_embed(variance[:, None].expand(-1, 2))
        elif family != 'gaussian_fixed':
            query['cov'] = xy.new_zeros((count, 2, 2))
    query['features'] = pool_content(observations['q_features'], partition['atoms'], count)
    query['family'] = family
    return query


def _key_cells(bounds, family=None):
    """Rebuild only required integration bounds, preserving R12 child order."""
    if bounds.ndim != 2 or bounds.shape[1] != 4:
        raise ValueError('integer half-open rectangle bounds required')
    if bounds.dtype != torch.int64:
        raise ValueError('partition bounds must be int64')
    size = bounds[:, 2:]-bounds[:, :2]
    small = (size <= 16).all(-1) & (size > 0).all(-1)
    large = (size == 24).all(-1)
    if not bool((small | large).all()):
        raise ValueError('only positive <=16px cells or native 24px squares supported')
    mean = ((bounds[:, :2]+bounds[:, 2:]-1).float()/2)/8
    mass = size.prod(-1).float()
    key = dict(mean=mean, mass=mass)
    if family is None or family.startswith('quartic_'):
        key['native_cell_bounds'] = (bounds.double()-.5)/8
    if family is None or family.startswith('gaussian_'):
        # repeat_interleave preserves each native key followed by y-major children.
        repeats = torch.where(large, 4, 1)
        owner = torch.repeat_interleave(torch.arange(len(bounds), device=bounds.device), repeats)
        starts = repeats.cumsum(0)-repeats
        child_index = torch.arange(len(owner), device=bounds.device)-starts[owner]
        offsets = torch.stack((child_index % 2, child_index//2), -1)*12
        child_low = bounds[owner, :2]+offsets
        child_size = torch.where(large[owner, None], 12, size[owner])
        children = torch.cat((child_low, child_low+child_size), -1)
        key.update(cell_bounds=(children.double()-.5)/8, cell_owner=owner)
    return key


def prepare_key(observations, partition, family=None):
    """Prepare the role's native area/integration bounds and pooled content.

    ``None`` retains the compatibility superset for synthetic comparisons.
    Production prediction and timing both pass the actual family explicitly.
    """
    _observations(observations)
    if family is not None:
        _family(family)
    key = _key_cells(partition['bounds'], family)
    key['features'] = pool_content(observations['k_features'], partition['atoms'], len(key['mass']))
    return key


def _gaussian_mass(mean, covariance, key, h):
    child = gaussian_cell_mass_fast(mean, covariance, key['cell_bounds'], bandwidth=h,
        quadrature_order=32, query_chunk_size=256, key_chunk_size=432)
    mass = child.new_zeros((*child.shape[:-1], len(key['mass'])))
    mass.index_add_(-1, key['cell_owner'], child)
    return mass


def geometry(query, key, family, h):
    """Return float64 native-key log masses and unfiltered raw-FOV diagnostics."""
    _family(family)
    if query.get('family', family) != family:
        raise ValueError('query was prepared for a different family')
    if family.startswith('quartic_'):
        mass = quartic_cell_mass(query['mean'], query['cov'], key['native_cell_bounds'],
            bandwidth=h, quadrature_order=24, query_chunk_size=16, key_chunk_size=32)
    elif family in ('gaussian_full4', 'gaussian_quad4', 'gaussian_quad16'):
        child = _gaussian_mass(query['nodes'], query['node_cov'].double(), key, h)
        mass = (child*query['node_weights'].double()[..., None]).sum(1)
    else:
        if family == 'gaussian_fixed':
            covariance = query['mass'].double()[:, None, None]/(64*12)*torch.eye(
                2, device=query['mean'].device, dtype=torch.float64)
        else:
            covariance = query['cov'].double()
        mass = _gaussian_mass(query['mean'], covariance, key, h)
    if mass.dtype != torch.float64 or not bool(torch.isfinite(mass).all()) or bool((mass < 0).any()):
        raise ValueError('invalid raw native mass')
    fov = mass.sum(-1)
    diagnostics = dict(raw_zero_cells=int((mass==0).sum()), raw_zero_fov_rows=int((fov==0).sum()),
        raw_tiny_fov_rows=int((fov<1.e-20).sum()), raw_fov_min=float(fov.min()), raw_fov_max=float(fov.max()),
        raw_floor_fov_mass=float(1.e-30*(key['mass'].double()/64).sum()))
    return (mass+1.e-30*(key['mass'].double()/64)[None]).log(), diagnostics


def predict(query, key, family, h, temperature):
    """The full content path also executes at T=0, matching the timed forward."""
    logits, diagnostics = geometry(query, key, family, h)
    content = query['features']@key['features'].T
    probability = (logits+temperature*content).softmax(-1)
    diagnostics['all_finite'] = bool(torch.isfinite(probability).all())
    diagnostics['max_normalization_error'] = float((probability.sum(-1)-1).abs().max())
    if not diagnostics['all_finite'] or bool((probability < 0).any()):
        raise ValueError('invalid normalized probability')
    return probability, diagnostics
