"""R7's unchanged learned head with an explicit R17 numerical interface.

This module defines interface modes, not scientific comparison roles or a
selected bandwidth/temperature. Fusion requires externally supplied pure
geometry logits in float64; Q/K, depths, the learned V and the head remain
float32. Trace temperature uses the original full support covariance even
when a future caller supplies geometry from an isotropic control. The
geometry cache's provenance is the caller's responsibility.

Only attention is cast to float32 immediately before attention @ V. The
singleview mode retains R7's original float32 geometry/softmax and own-query
V path, with h=.5 and T=5; it never reads cross-source/cache/label fields.
No network, learned parameter, source filtering, or additional floor is
introduced here. Actual task efficacy requires a separately frozen study.
"""
from __future__ import annotations

import math
from numbers import Real

import torch
import torch.nn.functional as F

from models.gaussian_quadratic_pe import controlled_features
from models.round7_depth_attention import DepthAttention as OriginalDepthAttention
from models.round7_depth_attention import depth_loss
from models import round17_trace_pe as TRACE


MODES = ('trace', 'constant', 'singleview')
FUSION_FIELDS = ('q', 'k', 'q_depth', 'k_value_depth', 'geometry_logits')
TRACE_FIELDS = FUSION_FIELDS + ('support_covariance',)
SINGLEVIEW_FIELDS = ('q', 'qmean', 'qcov', 'q_depth', 'q_value_depth')
INTERVENTIONS = (None, 'zero_v', 'reverse_v')


def _configuration_scalar(value, name, *, positive):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a real scalar')
    result = float(value)
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        raise ValueError(f'{name} must be finite and {"positive" if positive else "nonnegative"}')
    return result


def _tensor(data, name, shape, device, dtype=torch.float32):
    value = data[name]
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
        raise ValueError(f'{name} must have shape {tuple(shape)}')
    if value.dtype != dtype or value.device != device:
        raise ValueError(f'{name} must have dtype {dtype} and device {device}')
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f'{name} must be finite')
    return value


class DepthAttention(OriginalDepthAttention):
    def __init__(self, mode, *, bandwidth=None, temperature=None):
        if mode not in MODES:
            raise ValueError('unknown Round17 interface mode')
        if mode == 'singleview':
            bandwidth = .5 if bandwidth is None else bandwidth
            temperature = 5. if temperature is None else temperature
        elif bandwidth is None or temperature is None:
            raise ValueError('fusion modes require explicit bandwidth and temperature')
        bandwidth = _configuration_scalar(bandwidth, 'bandwidth', positive=True)
        temperature = _configuration_scalar(temperature, 'temperature', positive=False)
        if not math.isfinite(bandwidth * bandwidth) or bandwidth * bandwidth == 0:
            raise ValueError('bandwidth square must be finite and positive in float64')
        if mode == 'singleview' and (bandwidth != .5 or temperature != 5.):
            raise ValueError('singleview retains original bandwidth .5 and temperature 5')
        # This call, including its RNG consumption, is exactly the R7
        # constructor. Do not create a parameter or random tensor before it.
        super().__init__('full')
        self.mode = self.role = mode
        self.bandwidth, self.temperature = bandwidth, temperature
        self.method = 'full' if mode == 'singleview' else 'precomputed_geometry'

    def _inputs(self, data):
        # Read only the allowlist. In particular, do not enumerate arbitrary
        # input fields, inspect label tensors, or consult visibility masks.
        names = (SINGLEVIEW_FIELDS if self.mode == 'singleview' else
                 TRACE_FIELDS if self.mode == 'trace' else FUSION_FIELDS)
        d = {name: data[name] for name in names}
        q = d['q']
        if (not isinstance(q, torch.Tensor) or q.ndim != 3
                or q.shape[-1] != 128 or not q.shape[0] or not q.shape[1]):
            raise ValueError('q must be nonempty [B,Q,128]')
        batch, queries, _ = q.shape
        device = self.qk.weight.device
        for name, parameter in self.named_parameters():
            if parameter.dtype != torch.float32 or parameter.device != device:
                raise ValueError(f'model parameter {name} must be float32 on {device}')
            if not bool(torch.isfinite(parameter).all()):
                raise ValueError(f'model parameter {name} must be finite')
        _tensor(d, 'q', (batch, queries, 128), device)
        _tensor(d, 'q_depth', (batch, queries), device)
        if self.mode == 'singleview':
            _tensor(d, 'qmean', (batch, queries, 2), device)
            _tensor(d, 'qcov', (batch, queries, 2, 2), device)
            _tensor(d, 'q_value_depth', (batch, queries), device)
            d.update(k=d['q'], kmean=d['qmean'], kcov=d['qcov'],
                     q_to_k=d['qmean'], q_to_kcov=d['qcov'],
                     k_value_depth=d['q_value_depth'])
        else:
            k = d['k']
            if (not isinstance(k, torch.Tensor) or k.ndim != 3
                    or k.shape[0] != batch or k.shape[-1] != 128 or not k.shape[1]):
                raise ValueError('k must be nonempty [B,K,128] matching q batch')
            keys = k.shape[1]
            _tensor(d, 'k', (batch, keys, 128), device)
            _tensor(d, 'k_value_depth', (batch, keys), device)
            _tensor(d, 'geometry_logits', (batch, queries, keys), device, torch.float64)
            if self.mode == 'trace':
                _tensor(d, 'support_covariance', (batch, queries, 2, 2), device)
                # Symmetry/PSD/trace checks, without repair, are performed by
                # TRACE.probabilities as part of the executed operator below.
        return d

    def forward(self, data, intervention=None, return_attention=False):
        if intervention not in INTERVENTIONS:
            raise ValueError('unknown value intervention')
        d = self._inputs(data)
        q = F.normalize(self.qk(d['q']), dim=-1)
        k = F.normalize(self.qk(d['k']), dim=-1)
        content = q @ k.transpose(-1, -2)
        if content.dtype != torch.float32 or not bool(torch.isfinite(content).all()):
            raise ValueError('online Q/K content must be finite float32')
        if self.mode == 'singleview':
            qg, kg = controlled_features(d, 'full', .5)
            geometry = qg @ kg.transpose(-1, -2)
            # Preserve the original float32 operation order exactly.
            attention = (5. * content + geometry).softmax(-1)
        elif self.mode == 'trace':
            attention = TRACE.probabilities(
                d['geometry_logits'], content, d['support_covariance'],
                self.bandwidth, self.temperature)
        else:
            temperature32 = torch.as_tensor(
                self.temperature, dtype=torch.float32, device=content.device)
            bias = temperature32 * content
            if not bool(torch.isfinite(temperature32)) or not bool(torch.isfinite(bias).all()):
                raise ValueError('constant float32 temperature/content bias overflowed')
            attention = (d['geometry_logits'] + bias.to(dtype=torch.float64)).softmax(-1)
        if not bool(torch.isfinite(attention).all()):
            raise ValueError('attention must be finite')
        qbase = d['q_depth'].clamp(.1, 20).log()
        qhidden = F.gelu(self.query(torch.cat([d['q'], qbase[..., None]], -1)))
        values = F.gelu(self.value(torch.cat([
            d['k'], d['k_value_depth'].clamp(.1, 20).log()[..., None]], -1)))
        if intervention == 'zero_v':
            values = torch.zeros_like(values)
        elif intervention == 'reverse_v':
            values = values.flip(-2)
        # Fusion softmax is float64; this one explicit cast defines the new
        # actual-V interface. For singleview it is a float32 no-op.
        aggregate = attention.to(dtype=torch.float32) @ values
        residual = self.head(torch.cat([qhidden, aggregate], -1)).squeeze(-1).clamp(-3, 3)
        depth = (qbase + residual).exp().clamp(.1, 20)
        if (values.dtype != torch.float32 or depth.dtype != torch.float32
                or not bool(torch.isfinite(values).all()) or not bool(torch.isfinite(depth).all())):
            raise ValueError('learned V and depth must be finite float32')
        return (depth, attention, values) if return_attention else depth
