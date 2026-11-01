"""Support PE and same-input stratified quadrature for partition experiments.

All geometry is supplied from RGB predictions. Coordinates use eight pixels as
one unit; key mass is actual image area. This is a relative score, not a claim
that unequal key covariances have an exact five-dimensional feature map.
"""
from __future__ import annotations

import math
import torch


def group_mean(values, owners, count):
    flat = values.flatten(1)
    sums = flat.new_zeros((count, flat.shape[-1]))
    sums.index_add_(0, owners, flat)
    mass = torch.bincount(owners, minlength=count).to(values.dtype)
    if bool((mass == 0).any()):
        raise ValueError("empty support group")
    return (sums / mass[:, None]).reshape(count, *values.shape[1:])


def prepare_support(xy, pixel_cov, pixel_owner, bounds, parent, *, samples=1, include_full_moments=True):
    """Full moments or equal-area stratum means; both consume every input pixel.

    For quadrature each node is its stratum's mean projected position, with
    only mean within-pixel covariance. Its omitted within-stratum scatter is
    precisely the integration approximation being tested.
    """
    count = len(bounds)
    mean = group_mean(xy, pixel_owner, count)
    mass = torch.bincount(pixel_owner, minlength=count).to(xy.dtype)
    result = dict(mean=mean, mass=mass, parent=parent)
    if include_full_moments:
        centered = xy - mean[pixel_owner]
        result['cov'] = group_mean(centered[:, :, None] * centered[:, None, :] + pixel_cov,
                                  pixel_owner, count)
    if samples != 1:
        side = math.isqrt(samples)
        if side * side != samples:
            raise ValueError("square quadrature budget required")
        # Bounds and raster positions use integer indices, before conversion to units.
        width = int(bounds[:, 2].max())
        index = torch.arange(len(xy), device=xy.device)
        x, y = index % width, index // width
        cell = bounds[pixel_owner]
        sx = (x-cell[:, 0]) * side // (cell[:, 2]-cell[:, 0])
        sy = (y-cell[:, 1]) * side // (cell[:, 3]-cell[:, 1])
        strata = pixel_owner * samples + sy * side + sx
        result['nodes'] = group_mean(xy, strata, count*samples).reshape(count, samples, 2)
        result['node_cov'] = group_mean(pixel_cov, strata, count*samples).reshape(count, samples, 2, 2)
        weights = torch.bincount(strata, minlength=count*samples).to(xy.dtype).reshape(count, samples)
        result['node_weights'] = weights / mass[:, None]
    return result


def _log_density(delta, covariance):
    a, b, c = covariance[..., 0, 0], covariance[..., 0, 1], covariance[..., 1, 1]
    determinant = a*c-b.square()
    if not bool(torch.isfinite(determinant).all()) or bool((determinant <= 0).any()):
        raise ValueError("nonpositive pair covariance")
    distance = (c*delta[..., 0].square() - 2*b*delta[..., 0]*delta[..., 1]
                + a*delta[..., 1].square()) / determinant
    return -.5 * (distance + determinant.log()) - math.log(2*math.pi)


def geometry_logits(query, key, method, bandwidth):
    """Known log density overlap or quadrature under exactly the same key model."""
    eye = torch.eye(2, device=query['mean'].device, dtype=query['mean'].dtype)
    kcov = key['cov']
    if method in ('quad4', 'quad16'):
        delta = query['nodes'][:, :, None, :] - key['mean'][None, None, :, :]
        covariance = query['node_cov'][:, :, None] + kcov[None, None] + bandwidth**2*eye
        node = _log_density(delta, covariance) + query['node_weights'].log()[:, :, None]
        return torch.logsumexp(node, dim=1) + key['mass'].log()[None]
    qcov = query['cov']
    if method == 'point':
        qcov, kcov = torch.zeros_like(qcov), torch.zeros_like(kcov)
    elif method == 'center':
        qcov = torch.zeros_like(qcov)
    elif method == 'isotropic':
        qcov = qcov.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None, None]*eye
        kcov = kcov.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None, None]*eye
    elif method == 'fixed':
        qcov = query['mass'][:, None, None]/(64*12)*eye
        kcov = key['mass'][:, None, None]/(64*12)*eye
    elif method == 'rotated':
        rotation = eye.new_tensor([[0., -1.], [1., 0.]])
        qcov = rotation @ qcov @ rotation.T
    elif method != 'full':
        raise ValueError(method)
    covariance = qcov[:, None] + kcov[None] + bandwidth**2*eye
    return (_log_density(query['mean'][:, None]-key['mean'][None], covariance)
            + key['mass'].log()[None])


def pool_content(base_features, base_owner, count):
    pooled = group_mean(base_features, base_owner, count)
    return torch.nn.functional.normalize(pooled, dim=-1)


def common_prediction(probabilities, query, key, atomic_key_owner, parents=108):
    """Lift to 8px target atoms, then source-area average to 16px queries.

    This function sees no sensor validity or visible-pixel counts. Coarse cells
    reconstruct a constant density on each target atom they contain.
    """
    lifted = probabilities[:, atomic_key_owner] * (64/key['mass'][atomic_key_owner])[None]
    output = lifted.new_zeros((parents, len(atomic_key_owner)))
    output.index_add_(0, query['parent'], lifted*query['mass'][:, None]/256)
    return output
