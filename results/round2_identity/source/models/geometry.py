"""Small, explicit geometry operators; camera coordinates use +z forward.

All covariance arguments are SPD. Shape/support covariance and position-error
covariance are distinct inputs: they must not be silently interchanged.
"""
import torch


def covariance(rotation, scales):
    return (rotation * scales.square().unsqueeze(-2)) @ rotation.transpose(-1, -2)


def project(mean, rotation, translation):
    """Normalized pinhole coordinates and world-coordinate Jacobian.

    Callers must reject nonpositive depths; no silent clamping behind camera.
    """
    p = torch.einsum('...ij,...nj->...ni', rotation, mean) + translation.unsqueeze(-2)
    x, y, z = p.unbind(-1)
    zeros = torch.zeros_like(z)
    jac = torch.stack((1 / z, zeros, -x / z.square(),
                       zeros, 1 / z, -y / z.square()), -1).reshape(*z.shape, 2, 3)
    return p[..., :2] / z.unsqueeze(-1), jac @ rotation.unsqueeze(-3), z


def project_covariance(jacobian, cov):
    return jacobian @ cov @ jacobian.transpose(-1, -2)


def spd_log(cov, floor=1e-10):
    eigenvalues, vectors = torch.linalg.eigh(cov)
    return (vectors * eigenvalues.clamp_min(floor).log().unsqueeze(-2)) @ vectors.transpose(-1, -2)


def logcov_descriptor(cov, remove_scale=False):
    """Vectorization preserves Frobenius distances (off-diagonals get sqrt(2))."""
    log = spd_log(cov)
    d = cov.shape[-1]
    if remove_scale:
        log = log - torch.diag_embed(log.diagonal(dim1=-2, dim2=-1).mean(-1).unsqueeze(-1).expand(*log.shape[:-2], d))
    ij = torch.triu_indices(d, d, device=cov.device)
    weights = torch.where(ij[0] == ij[1], 1., 2. ** .5).to(cov)
    return log[..., ij[0], ij[1]] * weights


def pairwise_squared(x, y):
    return (x.unsqueeze(-2) - y.unsqueeze(-3)).square().sum(-1)


def covariance_moments(cov):
    """Gauge-safe smooth alternative, including at repeated eigenvalues.

    Trace-normalized second moment plus log trace. Pairwise Euclidean distance
    is invariant to a common rotation; a common change of length units cancels.
    """
    d = cov.shape[-1]
    trace = cov.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-12)
    normalized = cov * (d / trace)[..., None, None]
    ij = torch.triu_indices(d, d, device=cov.device)
    weights = torch.where(ij[0] == ij[1], 1., 2. ** .5).to(cov)
    vector = normalized[..., ij[0], ij[1]] * weights
    return torch.cat((vector, trace.log()[..., None]), -1)


def expected_phase(mean, cov, frequencies):
    """Characteristic function E exp(i omega^T X); returns real, imaginary."""
    phase = mean @ frequencies.T
    variance = torch.einsum('fd,...de,fe->...f', frequencies, cov, frequencies)
    amplitude = torch.exp(-.5 * variance)
    return amplitude * phase.cos(), amplitude * phase.sin()


def rotary(features, mean, frequencies, cov=None):
    phase = mean @ frequencies.T
    if cov is None:
        c, s = phase.cos(), phase.sin()
    else:
        c, s = expected_phase(mean, cov, frequencies)
    pairs = features.reshape(*features.shape[:-1], -1, 2)
    a, b = pairs.unbind(-1)
    return torch.stack((a * c - b * s, a * s + b * c), -1).flatten(-2)


def affine_frame(mean, rotation, scales):
    frame = torch.zeros(*mean.shape[:-1], 4, 4, dtype=mean.dtype, device=mean.device)
    frame[..., :3, :3] = rotation * scales.unsqueeze(-2)
    frame[..., :3, 3] = mean
    frame[..., 3, 3] = 1
    return frame


def tangent_transfer(jac_a, jac_b, tangent):
    """Exact local solve, intentionally exposes degeneracy to the caller."""
    aa, ab = jac_a @ tangent, jac_b @ tangent
    return torch.linalg.solve(aa.transpose(-1, -2), ab.transpose(-1, -2)).transpose(-1, -2), torch.linalg.cond(aa)
