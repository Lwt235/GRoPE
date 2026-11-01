"""World-space Gaussian geometry as compact Q/K position transformations.

The density variants use the classical Bhattacharyya kernel of square-root
Gaussian densities. They evaluate the infinite-dimensional PE implicitly;
they are not a new kernel or a small-dimensional standard RoPE implementation.
No labels, tracks, primitive identities, pair gates, or additive geometry biases
are consumed. Covariance means spatial support, not localization uncertainty.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F

from models.geometry import rotary


METHODS = [
    'world_rope', 'world_expected', 'world_isometric',
    'world_isometric_plane', 'world_isometric_isovolume',
    'world_center_half', 'world_moment_rope', 'world_normal_rope',
    'world_density_softmax', 'world_density_linear',
    'density_linear_plane', 'density_linear_isovolume',
    'world_absolute',
]


def dense_frequencies(count, dimension, seed=20261029):
    """Fixed directions and common 0.2--6.3 frequency norms, without RNG drift."""
    generator = torch.Generator().manual_seed(seed)
    direction = F.normalize(torch.randn(count, dimension, generator=generator), dim=-1)
    norms = torch.logspace(math.log10(.2), math.log10(6.3), count)
    return direction * norms[:, None]


def oblique_world_frequencies():
    """24 fixed spherical-Fibonacci directions, all genuinely three-dimensional."""
    index = torch.arange(24, dtype=torch.float32)
    z = 1 - 2*(index+.5)/24
    angle = index * math.pi * (3-math.sqrt(5))
    radius = torch.sqrt(1-z.square())
    direction = torch.stack([radius*angle.cos(), radius*angle.sin(), z], -1)
    norms = torch.logspace(math.log10(.2), math.log10(6.3), 24)
    return direction * norms[:, None]


def world_unit(cov_q, cov_k, q_valid=None, key_valid=None):
    """sqrt(median(trace covariance)) over both views' valid observations.

    Per-pair geometry, including at inference, determines this unit. No dataset
    or target statistics are fitted. Invalid tokens do not determine the unit;
    an entirely invalid pair uses 1 and is left for evaluator coverage masking.
    """
    tq = cov_q.diagonal(dim1=-2, dim2=-1).sum(-1)
    tk = cov_k.diagonal(dim1=-2, dim2=-1).sum(-1)
    q_valid = torch.ones_like(tq, dtype=torch.bool) if q_valid is None else q_valid.bool()
    key_valid = torch.ones_like(tk, dtype=torch.bool) if key_valid is None else key_valid.bool()
    traces = torch.cat([tq, tk], -1)
    valid = torch.cat([q_valid, key_valid], -1) & torch.isfinite(traces) & (traces > 0)
    median = torch.nanmedian(torch.where(valid, traces, torch.full_like(traces, float('nan'))), dim=-1).values
    return torch.where(torch.isfinite(median) & (median > 0), median.sqrt(), torch.ones_like(median))


def normalized_world_geometry(data):
    """Normalize means and covariances together, without changing their shape."""
    mq, mk, cq, ck = data['mq'], data['mk'], data['cq'], data['ck']
    q_valid = data.get('q_valid')
    key_valid = data.get('key_valid', data.get('k_valid'))
    unit = world_unit(cq, ck, q_valid, key_valid)
    mean_scale = unit[..., None, None]
    cov_scale = unit[..., None, None, None].square()
    mq, mk, cq, ck = mq/mean_scale, mk/mean_scale, cq/cov_scale, ck/cov_scale
    # Invalid entries can be arbitrary exporter placeholders. Make geometry
    # operators well-defined; query rows are still excluded by the evaluator.
    identity = torch.eye(3, dtype=cq.dtype, device=cq.device)
    if q_valid is not None:
        mq = torch.where(q_valid.bool()[..., None], mq, torch.zeros_like(mq))
        cq = torch.where(q_valid.bool()[..., None, None], cq, identity)
    if key_valid is not None:
        mk = torch.where(key_valid.bool()[..., None], mk, torch.zeros_like(mk))
        ck = torch.where(key_valid.bool()[..., None, None], ck, identity)
    return mq, cq, mk, ck, unit


def symmetric_vector(matrix):
    """Six coordinates preserving the Frobenius inner product of symmetric 3x3s."""
    indices = torch.triu_indices(3, 3, device=matrix.device)
    weight = torch.where(indices[0] == indices[1], 1., math.sqrt(2)).to(matrix)
    return matrix[..., indices[0], indices[1]] * weight


def moment_descriptor(covariance):
    trace = covariance.diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.cat([symmetric_vector(covariance/trace[..., None, None]), trace.log()[..., None]], -1)


def normal_projector(covariance):
    """Sign-free smallest-axis projector; a unique normal needs an eigen-gap.

    This is a fixed-observation control, not an end-to-end covariance predictor.
    At a spherical covariance no physically unique normal exists; the eigensolver
    convention is exposed rather than interpreted as geometry.
    """
    _, vectors = torch.linalg.eigh(covariance)
    normal = vectors[..., :, 0]
    return normal[..., :, None] * normal[..., None, :]


def plane_isotropic(covariance):
    """Preserve normal, normal variance and tangential eigenvalue product."""
    values, vectors = torch.linalg.eigh(covariance)
    normal = vectors[..., :, 0]
    projector = normal[..., :, None]*normal[..., None, :]
    tangent_variance = (values[..., 1]*values[..., 2]).sqrt()
    identity = torch.eye(3, dtype=covariance.dtype, device=covariance.device)
    return values[..., 0, None, None]*projector + tangent_variance[..., None, None]*(identity-projector)


def isovolume_isotropic(covariance):
    """Discard orientation and anisotropy while preserving determinant."""
    variance = torch.exp(torch.linalg.slogdet(covariance)[1]/3)
    return torch.eye(3, dtype=covariance.dtype, device=covariance.device)*variance[..., None, None]


def log_bhattacharyya(mean_a, cov_a, mean_b, cov_b):
    """Stable log of the exact Gaussian probability-product kernel.

    Inputs must be SPD. No covariance floor silently changes the target kernel.
    Log form avoids underflow in linear density-attention logits.
    """
    average = (cov_a[..., :, None, :, :] + cov_b[..., None, :, :, :])/2
    delta = mean_a[..., :, None, :] - mean_b[..., None, :, :]
    mahal = (delta[..., None, :] @ torch.linalg.solve(average, delta[..., None]))[..., 0, 0]/8
    determinant = torch.linalg.slogdet(average)[1]/2
    determinant = determinant-(torch.linalg.slogdet(cov_a)[1][..., :, None]+torch.linalg.slogdet(cov_b)[1][..., None, :])/4
    # Exact log BC <= 0; suppress only tiny roundoff overshoot, not covariances.
    return (-mahal-determinant).clamp(max=0)


def isometric_rotary(content, mean, covariance, frequency):
    """Expected RoPE's 2D contraction plus its complementary two dimensions."""
    variance = torch.einsum('fd,...de,fe->...f', frequency, covariance, frequency)
    rotated = rotary(content, mean, frequency, covariance)
    complement = (-torch.expm1(-variance)).clamp_min(0).sqrt()
    residual = (content.reshape(*content.shape[:-1], -1, 2)*complement[..., None]).flatten(-2)
    return torch.cat([rotated, residual], -1)


class GaussianWorldMatcher(nn.Module):
    def __init__(self, method):
        super().__init__()
        if method not in METHODS:
            raise ValueError(method)
        self.method = method
        self.encoder = nn.Sequential(nn.Linear(108, 96), nn.GELU(), nn.Linear(96, 48))
        self.logtemp = nn.Parameter(torch.tensor(2.2))
        self.dustbin = nn.Parameter(torch.tensor(2.))
        self.dustbin_head = nn.Linear(48, 1)
        if method == 'world_absolute':
            # Same geometric information (3 mean + 7 invertible moment coords)
            # and deliberately more learnable capacity as a strong control.
            self.geometry_encoder = nn.Sequential(nn.Linear(10, 48), nn.GELU(), nn.Linear(48, 48))
        frequency = oblique_world_frequencies()
        self.register_buffer('freq3', frequency)
        # A shared subset spans the complete world frequency range. Both
        # descriptor-RoPE methods devote 24 channels to these center phases.
        self.register_buffer('center_indices', torch.linspace(0, 23, 12).round().long())
        self.register_buffer('moment_freq', dense_frequencies(12, 7))
        self.register_buffer('normal_freq', dense_frequencies(12, 6))

    def forward(self, data):
        raw_q, raw_k = self.encoder(data['q']), self.encoder(data['k'])
        original_q = F.normalize(raw_q, dim=-1)
        q, k = original_q, F.normalize(raw_k, dim=-1)
        mq, cq, mk, ck, _ = normalized_world_geometry(data)
        name = self.method
        linear = name in ['world_density_linear', 'density_linear_plane', 'density_linear_isovolume']
        if name in ['world_rope', 'world_expected', 'world_isometric', 'world_isometric_plane', 'world_isometric_isovolume']:
            if name.startswith('world_isometric'):
                # Normalize with the original observations first. These are
                # operator-only shape ablations, not a change of length units.
                if name == 'world_isometric_plane':
                    cq, ck = plane_isotropic(cq), plane_isotropic(ck)
                elif name == 'world_isometric_isovolume':
                    cq, ck = isovolume_isotropic(cq), isovolume_isotropic(ck)
                q = isometric_rotary(q, mq, cq, self.freq3)
                k = isometric_rotary(k, mk, ck, self.freq3)
            else:
                expected = name == 'world_expected'
                q = rotary(q, mq, self.freq3, cq if expected else None)
                k = rotary(k, mk, self.freq3, ck if expected else None)
            score = q @ k.transpose(-1, -2)
        elif name in ['world_center_half', 'world_moment_rope', 'world_normal_rope']:
            if name == 'world_moment_rope':
                dq, dk = moment_descriptor(cq), moment_descriptor(ck)
                descriptor_frequency = self.moment_freq
            elif name == 'world_normal_rope':
                dq, dk = symmetric_vector(normal_projector(cq)), symmetric_vector(normal_projector(ck))
                descriptor_frequency = self.normal_freq
            center_frequency = self.freq3[self.center_indices]
            descriptor_q = q[..., 24:] if name == 'world_center_half' else rotary(q[..., 24:], dq, descriptor_frequency)
            descriptor_k = k[..., 24:] if name == 'world_center_half' else rotary(k[..., 24:], dk, descriptor_frequency)
            q = torch.cat([rotary(q[..., :24], mq, center_frequency), descriptor_q], -1)
            k = torch.cat([rotary(k[..., :24], mk, center_frequency), descriptor_k], -1)
            score = q @ k.transpose(-1, -2)
        elif name == 'world_absolute':
            q = F.normalize(q+self.geometry_encoder(torch.cat([mq, moment_descriptor(cq)], -1)), dim=-1)
            k = F.normalize(k+self.geometry_encoder(torch.cat([mk, moment_descriptor(ck)], -1)), dim=-1)
            score = q @ k.transpose(-1, -2)
        else:
            if name == 'density_linear_plane':
                cq, ck = plane_isotropic(cq), plane_isotropic(ck)
            elif name == 'density_linear_isovolume':
                cq, ck = isovolume_isotropic(cq), isovolume_isotropic(ck)
            log_kernel = log_bhattacharyya(mq, cq, mk, ck)
            if linear:
                q, k = F.normalize(F.softplus(raw_q), dim=-1), F.normalize(F.softplus(raw_k), dim=-1)
                # softmax(log dot + log BC) is exactly row-normalized dot*BC.
                # logtemp scales real-candidate mass relative to the dustbin;
                # multiplying log-score by exp(logtemp) would exponentiate the
                # kernel and silently change this linear-attention definition.
                score = (q @ k.transpose(-1, -2)).clamp_min(torch.finfo(q.dtype).tiny).log()+log_kernel
            else:
                score = (q @ k.transpose(-1, -2))*log_kernel.exp()
        if linear:
            score = score+self.logtemp.clamp(max=math.log(60))
        else:
            score = self.logtemp.exp().clamp(max=60)*score
        key_valid = data.get('key_valid', data.get('k_valid'))
        if key_valid is not None:
            score = score.masked_fill(~key_valid.bool()[..., None, :], -1e4)
        unmatched = self.dustbin+self.dustbin_head(original_q).squeeze(-1)
        return torch.cat([score, unmatched[..., None]], -1)
