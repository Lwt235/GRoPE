"""R20 fixed source-centered Gaussian and same-information controls.

This adapter does not read files, cameras, labels or selection results. The
caller supplies the frozen R13/R19 source-center projection as float64
reference-8 coordinates. Original moments/content keep their old float32
preparation; a and C_a remain float64. Integration delegates to the unchanged
R15 Gaussian/quartic primitives, and trace arithmetic delegates to R17.

The declared configurations describe twelve fixed complete operators, not a
search space. Optional h/T arguments support synthetic identities; the real
runner must bind the exact ROLE_CONFIGS in its immutable source contract.
"""
from __future__ import annotations

import torch

from models import round15_validation_pe as OLD
from models import round17_trace_pe as TRACE


ROLE_CONFIGS = {
    'anchor_trace': dict(h=.25, temperature=10., old_family='gaussian_full1',
                         content_policy='trace'),
    'mean_only_matched_tau': dict(h=.25, temperature=10., old_family='gaussian_full1',
                                  content_policy='trace'),
    'anchor_constant': dict(h=.25, temperature=10., old_family='gaussian_full1',
                            content_policy='constant'),
    'anchor_isotropic_matched': dict(h=.25, temperature=10., old_family='gaussian_full1',
                                     content_policy='trace'),
    'anchor_point_regional': dict(h=1., temperature=5., old_family='gaussian_point',
                                  content_policy='constant'),
    'anchor_point_localizer': dict(h=.25, temperature=5., old_family='gaussian_point',
                                   content_policy='constant'),
    'anchor_point_legacy': dict(h=.5, temperature=5., old_family='gaussian_point',
                                content_policy='constant'),
    'anchor_point_matched': dict(h=.25, temperature=10., old_family='gaussian_point',
                                 content_policy='constant'),
    'anchor_fixed_regional': dict(h=.5, temperature=5., old_family='gaussian_fixed',
                                  content_policy='constant'),
    'anchor_fixed_matched': dict(h=.25, temperature=10., old_family='gaussian_fixed',
                                 content_policy='constant'),
    'anchor_quartic_regional': dict(h=1., temperature=5., old_family='quartic_point',
                                    content_policy='constant'),
    'anchor_quartic_localizer': dict(h=.25, temperature=5., old_family='quartic_point',
                                     content_policy='constant'),
}
FAMILIES = tuple(ROLE_CONFIGS)
FULL_FAMILIES = FAMILIES[:4]
TRACE_FAMILIES = tuple(n for n in FAMILIES if ROLE_CONFIGS[n]['content_policy'] == 'trace')
OBSERVATION_FIELDS = OLD.OBSERVATION_FIELDS


def _family(family):
    if family not in ROLE_CONFIGS:
        raise ValueError('undeclared R20 positional family')
    return ROLE_CONFIGS[family]


def _anchor(anchor, mean):
    if (not isinstance(anchor, torch.Tensor) or anchor.dtype != torch.float64
            or anchor.shape != mean.shape or anchor.device != mean.device
            or anchor.ndim != 2 or anchor.shape[-1] != 2 or not anchor.numel()
            or not bool(torch.isfinite(anchor).all())):
        raise ValueError('anchor must be finite float64 [Q,2] in reference-8 units on the observation device')
    return anchor


def anchor_second_moment(mean, covariance, anchor):
    """Return d=old_mu-a and C_a from original float32 moments, without repair."""
    if (not isinstance(mean, torch.Tensor) or mean.dtype != torch.float32
            or mean.ndim != 2 or mean.shape[-1] != 2 or not mean.numel()
            or not bool(torch.isfinite(mean).all())):
        raise ValueError('original mean must be finite float32 [Q,2]')
    _anchor(anchor, mean)
    if (not isinstance(covariance, torch.Tensor) or covariance.dtype != torch.float32
            or covariance.shape != (*mean.shape[:-1], 2, 2)
            or covariance.device != mean.device):
        raise ValueError('original covariance must be float32 [Q,2,2] matching mean')
    original = TRACE._covariance(covariance)
    delta = mean.double() - anchor
    centered = original + delta[..., :, None] * delta[..., None, :]
    # Validate the result as executed; neither an invalid original C nor an
    # overflowed rank-one product is silently repaired or hidden by widening.
    TRACE._covariance(centered)
    return delta, centered


def prepare_query(observations, partition, family, anchor):
    """Prepare this role's own statistics and replace only its declared geometry.

    Point/fixed delegate to their original lightweight preparation. They do
    not compute full scatter, C_a or the primary's content temperature.
    """
    config = _family(family)
    query = OLD.prepare_query(observations, partition, config['old_family'])
    old_mean = query['mean']
    _anchor(anchor, old_mean)
    query['old_mean'] = old_mean
    query['mean'] = anchor
    query['anchor_delta'] = old_mean.double() - anchor
    if family in FULL_FAMILIES:
        original = query['cov']
        delta, centered = anchor_second_moment(old_mean, original, anchor)
        query['original_cov'] = original
        query['anchor_delta'] = delta
        query['tau_cov'] = centered
        if family == 'mean_only_matched_tau':
            query['cov'] = original.double()
        elif family == 'anchor_isotropic_matched':
            variance = centered.diagonal(dim1=-2, dim2=-1).sum(-1) / 2
            query['cov'] = torch.diag_embed(variance[:, None].expand(-1, 2))
        else:
            query['cov'] = centered
    elif config['old_family'] == 'gaussian_fixed':
        query['cov'] = query['mass'].double()[:, None, None] / (64 * 12) * torch.eye(
            2, device=anchor.device, dtype=torch.float64)
    else:
        query['cov'] = anchor.new_zeros((len(anchor), 2, 2))
    query['family'] = family
    return query


def prepare_key(observations, partition, family=None):
    """Retain exactly the old role-specific native integration bounds/content."""
    old_family = None if family is None else _family(family)['old_family']
    return OLD.prepare_key(observations, partition, old_family)


def raw_mass(query, key, family, h=None):
    """Unfloored native key masses, with Gaussian children already summed."""
    config = _family(family)
    if query.get('family') != family:
        raise ValueError('query was prepared for a different family')
    bandwidth = config['h'] if h is None else h
    if config['old_family'] == 'quartic_point':
        return OLD.quartic_cell_mass(query['mean'], query['cov'], key['native_cell_bounds'],
            bandwidth=bandwidth, quadrature_order=24, query_chunk_size=16, key_chunk_size=32)
    return OLD._gaussian_mass(query['mean'], query['cov'], key, bandwidth)


def _geometry(raw, query, key):
    if (raw.dtype != torch.float64 or raw.shape != (len(query['mean']), len(key['mass']))
            or not bool(torch.isfinite(raw).all()) or bool((raw < 0).any())):
        raise ValueError('invalid raw native mass')
    fov = raw.sum(-1)
    floor = 1.e-30 * (key['mass'].double() / 64)
    diagnostics = dict(raw_zero_cells=int((raw == 0).sum()),
        raw_zero_fov_rows=int((fov == 0).sum()), raw_tiny_fov_rows=int((fov < 1.e-20).sum()),
        raw_fov_min=float(fov.min()), raw_fov_max=float(fov.max()),
        raw_floor_fov_mass=float(1.e-30 * (key['mass'].double() / 64).sum()))
    return (raw + floor[None]).log(), diagnostics


def geometry(query, key, family, h=None):
    """Float64 log native masses with exactly one old native-area floor."""
    return _geometry(raw_mass(query, key, family, h), query, key)


def prepared_terms(query, key, family, h=None, temperature=None):
    """Expose executed native mass, geometry and exact content arithmetic.

    This is production arithmetic for reproducibility/hybrid audits, not an
    independent numerical reference. No probability or row is filtered.
    """
    config = _family(family)
    bandwidth = config['h'] if h is None else h
    value = config['temperature'] if temperature is None else temperature
    raw = raw_mass(query, key, family, bandwidth)
    logits, diagnostics = _geometry(raw, query, key)
    content = query['features'] @ key['features'].T
    if content.dtype != torch.float32 or not bool(torch.isfinite(content).all()):
        raise ValueError('content must be finite float32')
    if config['content_policy'] == 'trace':
        covariance = query['tau_cov']
        # Keep TRACE.content_bias's complete content/covariance contract, but
        # compute tau only once because the audit also needs the actual tau32.
        if (content.ndim < 1 or not content.numel()
                or not isinstance(covariance, torch.Tensor)
                or content.shape[:-1] != covariance.shape[:-2]
                or content.device != covariance.device):
            raise ValueError('content must be float32 [...,K] matching covariance shape/device')
        tau32 = TRACE.trace_temperature(covariance, bandwidth, value).to(torch.float32)
        bias32 = tau32[..., None] * content
        if not bool(torch.isfinite(bias32).all()):
            raise ValueError('float32 content bias overflowed')
    else:
        scalar = TRACE._scalar(value, 'temperature', content.device, positive=False).to(torch.float32)
        if not bool(torch.isfinite(scalar)):
            raise ValueError('temperature is not representable in float32')
        tau32 = scalar.expand(len(query['mean']))
        bias32 = tau32[:, None] * content
        if not bool(torch.isfinite(bias32).all()):
            raise ValueError('float32 content bias overflowed')
    return dict(raw_mass=raw, geometry=logits, content32=content, tau32=tau32,
                bias32=bias32, diagnostics=diagnostics)


def predict(query, key, family, h=None, temperature=None):
    """Return all float64 native probabilities and the unfiltered diagnostics."""
    terms = prepared_terms(query, key, family, h, temperature)
    probability = (terms['geometry'] + terms['bias32'].double()).softmax(-1)
    diagnostics = dict(terms['diagnostics'], all_finite=bool(torch.isfinite(probability).all()),
        max_normalization_error=float((probability.sum(-1) - 1).abs().max()))
    if not diagnostics['all_finite'] or bool((probability < 0).any()):
        raise ValueError('invalid normalized probability')
    return probability, diagnostics
