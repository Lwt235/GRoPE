"""Unrun R17 validation draft: role-owned Gaussian geometry and trace bias.

Every trace role prepares its own complete float32 covariance. The matched
isotropic role changes only the covariance passed to Gaussian integration;
its temperature always uses the original float32 full covariance, including
the original off-diagonal entries and their PSD/symmetry validation.

The seven historical controls delegate unchanged to R15. A matched constant
temperature control is the existing ``gaussian_full1`` family at the primary
h/T, not a new family. This module reads no files or labels and does not set
threads. Prediction and timing must both include each role's own preparation,
geometry, content product, R17 checks/arithmetic, and normalization.
"""
from __future__ import annotations

import torch

from models import round15_validation_pe as OLD
from models import round17_trace_pe as TRACE


TRACE_FAMILIES = ('trace_full', 'trace_isotropic_matched')
CONTROL_FAMILIES = ('quartic_point', 'gaussian_point', 'gaussian_fixed',
                    'gaussian_full1', 'gaussian_full4', 'gaussian_quad4', 'gaussian_quad16')
FAMILIES = TRACE_FAMILIES + CONTROL_FAMILIES
OBSERVATION_FIELDS = OLD.OBSERVATION_FIELDS


def _family(family):
    if family not in FAMILIES:
        raise ValueError('undeclared R17 positional family')


def prepare_query(observations, partition, family):
    """Prepare one role, never borrowing the primary's covariance or tau."""
    _family(family)
    if family in CONTROL_FAMILIES:
        return OLD.prepare_query(observations, partition, family)
    query = OLD.prepare_query(observations, partition, 'gaussian_full1')
    original_covariance = query['cov']
    query['tau_cov'] = original_covariance
    if family == 'trace_isotropic_matched':
        variance = original_covariance.double().diagonal(dim1=-2, dim2=-1).mean(-1)
        query['cov'] = torch.diag_embed(variance[..., None].expand(-1, 2))
    query['family'] = family
    return query


def prepare_key(observations, partition, family=None):
    """Use only the role's required bounds; 24px Gaussian keys keep four children.

    None preserves the old synthetic compatibility superset. Actual prediction
    and cost callers must pass the family, as in the R15 role-specific contract.
    """
    if family is not None:
        _family(family)
    old_family = 'gaussian_full1' if family in TRACE_FAMILIES else family
    return OLD.prepare_key(observations, partition, old_family)


def geometry(query, key, family, h):
    """Unchanged Gaussian GL32 native mass, one area floor, and raw-FOV checks."""
    _family(family)
    if query.get('family') != family:
        raise ValueError('query was prepared for a different family')
    if family in CONTROL_FAMILIES:
        return OLD.geometry(query, key, family, h)
    # Only the family tag is adapted. No tensor preparation, tau calculation,
    # covariance conversion beyond OLD.geometry, or second floor is hidden here.
    return OLD.geometry(dict(query, family='gaussian_full1'), key, 'gaussian_full1', h)


def predict(query, key, family, h, temperature):
    """Execute content and all R17 validation even at T=0; return double p."""
    _family(family)
    if family in CONTROL_FAMILIES:
        return OLD.predict(query, key, family, h, temperature)
    logits, diagnostics = geometry(query, key, family, h)
    content = query['features'] @ key['features'].T
    probability = TRACE.probabilities(logits, content, query['tau_cov'], h, temperature)
    diagnostics['all_finite'] = bool(torch.isfinite(probability).all())
    diagnostics['max_normalization_error'] = float((probability.sum(-1)-1).abs().max())
    if not diagnostics['all_finite'] or bool((probability < 0).any()):
        raise ValueError('invalid normalized probability')
    return probability, diagnostics
