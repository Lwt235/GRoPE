"""Two-block conditional matcher for matched positional-operator ablations.

This is an actual bidirectional cross-attention network with values, residuals
and FFNs. It is not a reproduction of WorldRoPE's geometry prediction or V/O
rotations. Geometry is supplied by the declared input-view estimator only.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F

from models.gaussian_pe import normalized_rotary
from models.gaussian_spectral_pe import resolvent_rotary
from models.geometry import rotary


METHODS = ('point', 'expected', 'normalized', 'isotropic', 'fixed', 'resolvent')
INPUT_DIM = 128
MODEL_DIM = 96
HEADS = 2
HEAD_DIM = 48


def _temperature(log_temperature):
    # Clamp before exp to avoid an overflowing intermediate at large logits.
    return log_temperature.clamp(max=math.log(60.)).exp()


class _CrossAttentionBlock(nn.Module):
    """One pre-LN block, shared across directions with simultaneous updates."""
    def __init__(self):
        super().__init__()
        self.attention_norm = nn.LayerNorm(MODEL_DIM)
        self.qkv = nn.Linear(MODEL_DIM, 3*MODEL_DIM)
        # The default nonzero weight initialization retains the V path from
        # the first step. No bias means zero_values removes the whole update.
        self.output = nn.Linear(MODEL_DIM, MODEL_DIM, bias=False)
        self.log_temperature = nn.Parameter(torch.tensor(2.2))
        self.ffn_norm = nn.LayerNorm(MODEL_DIM)
        self.ffn = nn.Sequential(nn.Linear(MODEL_DIM, 2*MODEL_DIM), nn.GELU(),
                                 nn.Linear(2*MODEL_DIM, MODEL_DIM))

    def _project(self, content):
        batch, tokens, _ = content.shape
        return self.qkv(self.attention_norm(content)).reshape(
            batch, tokens, 3, HEADS, HEAD_DIM).permute(2, 0, 3, 1, 4).unbind(0)

    def _attend(self, query, key, value, zero_values):
        probability = (_temperature(self.log_temperature)*(
            query @ key.transpose(-1, -2))).softmax(dim=-1)
        if zero_values:
            value = value*0
        aggregate = probability @ value
        aggregate = aggregate.transpose(1, 2).reshape(
            aggregate.shape[0], aggregate.shape[2], MODEL_DIM)
        return self.output(aggregate)

    def forward(self, q, k, data, position_encoder, zero_values=False):
        # Project both incoming states before either residual update.
        qq, qk, qv = self._project(q)
        kq, kk, kv = self._project(k)
        qq, qk = F.normalize(qq, dim=-1), F.normalize(qk, dim=-1)
        kq, kk = F.normalize(kq, dim=-1), F.normalize(kk, dim=-1)

        q_query = position_encoder(qq, data['qmean'], data['qcov'])
        k_in_q = position_encoder(kk, data['k_to_q'], data['k_to_qcov'])
        k_query = position_encoder(kq, data['kmean'], data['kcov'])
        q_in_k = position_encoder(qk, data['q_to_k'], data['q_to_kcov'])
        q_update = self._attend(q_query, k_in_q, kv, zero_values)
        k_update = self._attend(k_query, q_in_k, qv, zero_values)
        q, k = q+q_update, k+k_update
        return q+self.ffn(self.ffn_norm(q)), k+self.ffn(self.ffn_norm(k))


class Round5CrossAttentionMatcher(nn.Module):
    """Frozen-feature128 -> two shared-direction96D blocks -> matching48D.

    Data fields q/k have shapes [B,Nq/Nk,128]. Native qmean/qcov and
    kmean/kcov use each view's token-stride coordinates. k_to_q/cov describes
    key support in the query frame, and q_to_k/cov the reverse direction.
    All supplied candidate tokens participate: no labels, visibility masks,
    target depth, dustbin, or label-derived attention mask is read here.

    `zero_values` is a forward-only path audit, not another trained method.
    """
    def __init__(self, method, resolution=.25):
        super().__init__()
        if method not in METHODS:
            raise ValueError(f'Unknown positional method: {method}')
        if not math.isfinite(resolution) or resolution < 0:
            raise ValueError('resolution must be finite and nonnegative')
        self.method = method
        self.input_projection = nn.Linear(INPUT_DIM, MODEL_DIM)
        self.layers = nn.ModuleList([_CrossAttentionBlock(), _CrossAttentionBlock()])
        self.final_norm = nn.LayerNorm(MODEL_DIM)
        self.head = nn.Linear(MODEL_DIM, HEAD_DIM)
        self.log_temperature = nn.Parameter(torch.tensor(2.2))
        angle = torch.arange(24)*math.pi*(3-math.sqrt(5))
        frequency = torch.stack([angle.cos(), angle.sin()], dim=-1)*torch.logspace(-1, .6, 24)[:, None]
        self.register_buffer('frequency', frequency)
        self.register_buffer('resolution', torch.tensor(float(resolution)))

    def _position_encode(self, content, mean, covariance):
        mean = mean.unsqueeze(1)
        if self.method == 'point':
            return rotary(content.double(), mean.double(), self.frequency.double()).to(content.dtype)
        eye = torch.eye(2, dtype=covariance.dtype, device=covariance.device)
        if self.method == 'fixed':
            covariance = torch.ones_like(covariance)*eye/12
        elif self.method == 'isotropic':
            covariance = eye*(covariance.diagonal(dim1=-2, dim2=-1).sum(-1)/2)[..., None, None]
        covariance = (covariance+eye*self.resolution.square()).unsqueeze(1)
        if self.method == 'expected':
            return rotary(content.double(), mean.double(), self.frequency.double(),
                          covariance.double()).to(content.dtype)
        encode = resolvent_rotary if self.method == 'resolvent' else normalized_rotary
        return encode(content, mean, covariance, self.frequency)

    def encode(self, data, zero_values=False):
        """Return actual contextual states for gradient/path audits."""
        q = self.input_projection(data['q'])
        k = self.input_projection(data['k'])
        for layer in self.layers:
            q, k = layer(q, k, data, self._position_encode, zero_values)
        return q, k

    def forward(self, data, zero_values=False):
        q, k = self.encode(data, zero_values=zero_values)
        q = F.normalize(self.head(self.final_norm(q)), dim=-1)
        k = F.normalize(self.head(self.final_norm(k)), dim=-1)
        # Final scores contain no additional PE or pairwise geometric module.
        return _temperature(self.log_temperature)*(q @ k.transpose(-1, -2))
