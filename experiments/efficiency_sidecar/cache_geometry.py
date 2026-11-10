"""Explicit per-token geometry cache API; no changes to live model classes."""
import torch

from experiments.round65_data import quaternion_rotation
from experiments.efficiency_sidecar.adapters import quadratic_descriptors
from models.round65_actual_gs import common_features, vectorize


def prepare(covariance, *, precision):
    if covariance.requires_grad:
        raise ValueError('Fixed input geometry only.')
    s = covariance.double()
    values, vectors = torch.linalg.eigh(s)
    logs = .5*values.clamp_min(1e-15).div(.04).log()
    result = {'log_matrix': (vectors*logs.unsqueeze(-2)) @ vectors.transpose(-1, -2)}
    if precision:
        # Do not reuse floored eigenvalues: the original inverse has no extra floor.
        result['precision'] = torch.linalg.inv(s)
    return result


def rotate(cache, quaternion):
    exemplar = next(iter(cache.values()))
    r = torch.as_tensor(quaternion_rotation(quaternion), device=exemplar.device, dtype=torch.float64)[:, None]
    return {k: r @ value @ r.transpose(-1, -2) for k, value in cache.items()}


def cached_descriptor(centers, covariance, precision):
    """Same polynomial relation as the sidecar; use the caller's aligned P."""
    c, s, p = centers.double(), covariance.double(), precision
    d2 = (c.unsqueeze(-2)-c.unsqueeze(-3)).square().sum(-1)
    x = c-c.mean(-2, keepdim=True)
    alpha = (x*(p@x.unsqueeze(-1)).squeeze(-1)).sum(-1, keepdim=True)
    beta = -((p+p.transpose(-1, -2))@x.unsqueeze(-1)).squeeze(-1)
    off = torch.stack((p[..., 0, 1]+p[..., 1, 0], p[..., 0, 2]+p[..., 2, 0],
                       p[..., 1, 2]+p[..., 2, 1]), -1)
    left = torch.cat((alpha, beta, p.diagonal(dim1=-2, dim2=-1), off), -1)
    cross = torch.stack((x[..., 0]*x[..., 1], x[..., 0]*x[..., 2], x[..., 1]*x[..., 2]), -1)
    right = torch.cat((torch.ones_like(alpha), x, x.square(), cross), -1)
    metric = (left@right.transpose(-1, -2)).masked_fill(d2 == 0, 0)
    trace = p.flatten(-2)@s.transpose(-1, -2).flatten(-2).transpose(-1, -2)
    terms = torch.stack((metric, metric.transpose(-1, -2), trace, trace.transpose(-1, -2)), -1)
    return torch.cat(((d2.sqrt()/.2).unsqueeze(-1), terms.clamp_min(0).log1p()), -1).to(centers)


def forward(model, centers, covariance, appearance, *, log_matrix, precision=None):
    """Cache must follow the exact same token gather and rotation as covariance.

    Only rigid rotations are supported. Do not use stale cache after scale,
    covariance noise, edits or parameter updates. This is not an RGB adapter.
    """
    if model.role not in ('relative5', 'distance_mlp'):
        raise ValueError(model.role)
    values = (centers, covariance, appearance, log_matrix) + (() if precision is None else (precision,))
    if any(v.requires_grad for v in values):
        raise ValueError('Fixed inputs only.')
    if log_matrix.shape != covariance.shape or (precision is not None and precision.shape != covariance.shape):
        raise ValueError('Cache and covariance must be aligned.')
    if log_matrix.dtype != torch.float64 or (precision is not None and precision.dtype != torch.float64):
        raise ValueError('Cache must retain FP64 geometry.')
    common = torch.cat((centers, vectorize(log_matrix).to(centers), appearance.to(centers)), -1)
    x = model.stem(common)
    if model.role == 'distance_mlp':
        desc = model.position.descriptors(centers, covariance)
    else:
        desc = (quadratic_descriptors(centers, covariance) if precision is None else
                cached_descriptor(centers, covariance, precision))
    relation = model.position.mlp(desc)
    for layer in model.layers:
        x = layer(x, relation)
    x = model.final_norm(x)
    return model.classifier(torch.cat((x.mean(1), x.max(1).values), -1))
