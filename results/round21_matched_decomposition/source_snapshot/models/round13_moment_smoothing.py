"""Fixed moment-preserving Gaussian-mixture contraction; no learned parameters.

This is a known Gaussian Ornstein--Uhlenbeck/Mehler transformation applied to
the four existing query components. It makes no general unimodality claim.
"""
from __future__ import annotations

import math
import torch

RHO_SQUARED = 2.0 / 3.0


def smooth_gaussian_strata(query):
    """Return four contracted components, computing moments from their inputs.

    Expected keys: nodes [...,4,2], node_cov [...,4,2,2], node_weights [...,4].
    Other fields are preserved. Weights are validated, never silently rescaled.
    Input tensors are not mutated, and gradients remain attached to all moments.
    """
    nodes, covariance, weights = (query[k] for k in ('nodes', 'node_cov', 'node_weights'))
    if nodes.ndim < 2 or nodes.shape[-2:] != (4, 2):
        raise ValueError('exactly four 2D components required')
    if covariance.shape != (*nodes.shape, 2) or weights.shape != nodes.shape[:-1]:
        raise ValueError('inconsistent Gaussian component shapes')
    if nodes.dtype not in (torch.float32, torch.float64) or any(
        x.dtype != nodes.dtype or x.device != nodes.device for x in (covariance, weights)
    ):
        raise ValueError('same float32/float64 dtype and device required')
    if nodes.numel() == 0 or not all(bool(torch.isfinite(x).all()) for x in (nodes, covariance, weights)):
        raise ValueError('finite nonempty components required')
    tolerance = 1.e-6 if nodes.dtype == torch.float32 else 1.e-12
    if bool((weights < 0).any()) or not bool(((weights.sum(-1)-1).abs() <= tolerance).all()):
        raise ValueError('nonnegative normalized component weights required')
    scale = covariance.abs().amax((-1, -2)).clamp_min(1.)
    if not bool(((covariance-covariance.transpose(-1, -2)).abs() <= tolerance*scale[..., None, None]).all()):
        raise ValueError('symmetric component covariance required')
    if bool((torch.linalg.eigvalsh(covariance) < -tolerance*scale[..., None]).any()):
        raise ValueError('positive semidefinite component covariance required')
    mean = (nodes*weights[..., None]).sum(-2)
    delta = nodes-mean[..., None, :]
    total_covariance = ((covariance+delta[..., :, None]*delta[..., None, :])
                        * weights[..., None, None]).sum(-3)
    result = dict(query)
    result.update(nodes=mean[..., None, :]+math.sqrt(RHO_SQUARED)*delta,
                  node_cov=RHO_SQUARED*covariance+(1-RHO_SQUARED)*total_covariance[..., None, :, :],
                  node_weights=weights, mean=mean, cov=total_covariance)
    return result
