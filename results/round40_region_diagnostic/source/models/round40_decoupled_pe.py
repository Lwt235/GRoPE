"""R40: center RoPE and a separate seven-dimensional support descriptor.

Known RoPE and Euclidean descriptor kernels, not a new kernel theorem.  The
shape penalty is an additive positional bias; it does not certify area mass.
"""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from models.round7_depth_attention import DepthAttention, FIELDS

NEW_ROLES = ('center_rope', 'expected_rope', 'normalized_rope', 'decoupled',
             'shape_isotropic', 'shape_shuffle', 'center_rbf')
LEGACY_ROLES = ('point', 'point_pck', 'fixed', 'full', 'isotropic', 'singleview')
ROLES = NEW_ROLES + LEGACY_ROLES
SHAPE_ROLES = ('decoupled', 'shape_isotropic', 'shape_shuffle')
EXTRA_FIELDS = ('q_anchor3', 'k_anchor3', 'q_cov3', 'k_cov3')


def shape_descriptor(cov):
    # Trace-normalized second moment and log trace, exactly the early smooth
    # descriptor. Input units are metres. The floor only defines zero support.
    trace = cov.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-12)
    c = 3. * cov / trace[..., None, None]
    return torch.stack((c[..., 0, 0], c[..., 1, 1], c[..., 2, 2],
                        math.sqrt(2.) * c[..., 0, 1],
                        math.sqrt(2.) * c[..., 0, 2],
                        math.sqrt(2.) * c[..., 1, 2], trace.log()), -1)


def squared_distance(q, k):
    # Rounding can make an exactly zero distance slightly negative.
    return (q.square().sum(-1)[..., :, None] + k.square().sum(-1)[..., None, :]
            - 2. * (q @ k.transpose(-1, -2))).clamp_min(0.)


def rotary(x, anchor, covariance, frequencies):
    phase = anchor @ frequencies.T
    pair = x.reshape(*x.shape[:-1], -1, 2)
    a, b = pair.unbind(-1)
    cs, sn = phase.cos(), phase.sin()
    result = torch.stack((a * cs - b * sn, a * sn + b * cs), -1)
    if covariance is not None:
        variance = torch.einsum('fd,...de,fe->...f', frequencies, covariance, frequencies)
        result = result * torch.exp(-.5 * variance.clamp_min(0.))[..., None]
    return result.flatten(-2)


class DecoupledDepthAttention(DepthAttention):
    def __init__(self, role):
        if role not in ROLES:
            raise ValueError(role)
        super().__init__(role if role in LEGACY_ROLES else 'point')
        self.role = role
        if role in NEW_ROLES:
            # Same axis schedule / frequency range as the original mechanism
            # probe, extended to the existing 128-channel Q/K head. rad/metre.
            axes = torch.eye(3)[torch.arange(64) % 3]
            self.register_buffer('frequencies', axes * torch.logspace(0., 1.3, 64)[:, None])
        if role in SHAPE_ROLES or role == 'center_rbf':
            self.strength_raw = nn.Parameter(torch.tensor(math.log(math.expm1(1.))))
        rng = np.random.default_rng(20261101)
        if role == 'shape_shuffle':
            self.register_buffer('q_permutation', torch.from_numpy(rng.permutation(432)))
            self.register_buffer('k_permutation', torch.from_numpy(rng.permutation(432)))

    def attention_logits(self, data, zero_shape=False):
        q = F.normalize(self.qk(data['q']), dim=-1)
        k = F.normalize(self.qk(data['k']), dim=-1)
        qa, ka = data['q_anchor3'], data['k_anchor3']
        qc, kc = data['q_cov3'], data['k_cov3']
        use_cov = self.role in ('expected_rope', 'normalized_rope')
        q = rotary(q, qa, qc if use_cov else None, self.frequencies)
        k = rotary(k, ka, kc if use_cov else None, self.frequencies)
        if self.role == 'normalized_rope':
            q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        logits = 10. * (q @ k.transpose(-1, -2))
        if self.role in SHAPE_ROLES:
            if self.role == 'shape_isotropic':
                eye = torch.eye(3, dtype=qc.dtype, device=qc.device)
                qc = qc.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] * eye
                kc = kc.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] * eye
            elif self.role == 'shape_shuffle':
                qc, kc = qc[..., self.q_permutation, :, :], kc[..., self.k_permutation, :, :]
            penalty = squared_distance(shape_descriptor(qc), shape_descriptor(kc)) / 7.
        elif self.role == 'center_rbf':
            penalty = squared_distance(qa, ka)  # metres squared / (one metre)^2
        else:
            return logits
        strength = F.softplus(self.strength_raw)
        return logits - (0. if zero_shape else strength) * penalty

    def forward(self, data, intervention=None, return_attention=False):
        if self.role in LEGACY_ROLES:
            return super().forward(data, intervention, return_attention)
        d = {key: data[key] for key in FIELDS + EXTRA_FIELDS}
        attn = self.attention_logits(d, zero_shape=intervention == 'zero_shape').softmax(-1)
        qbase = d['q_depth'].clamp(.1, 20).log()
        qhidden = F.gelu(self.query(torch.cat((d['q'], qbase[..., None]), -1)))
        values = F.gelu(self.value(torch.cat((d['k'],
            d['k_value_depth'].clamp(.1, 20).log()[..., None]), -1)))
        if intervention == 'zero_v':
            values = torch.zeros_like(values)
        elif intervention == 'reverse_v':
            values = values.flip(-2)
        elif intervention not in (None, 'zero_shape'):
            raise ValueError(intervention)
        residual = self.head(torch.cat((qhidden, attn @ values), -1)).squeeze(-1).clamp(-3, 3)
        depth = (qbase + residual).exp().clamp(.1, 20)
        return (depth, attn, values) if return_attention else depth
