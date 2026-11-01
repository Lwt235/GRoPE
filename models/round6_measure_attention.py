"""One Gaussian relative-position mechanism, shared by attention and matching.

All controls have identical learned parameters and receive identical features,
projected means and known native cell sizes. Geometry is evaluated in the key
image's coordinate frame. No labels or visibility masks enter this module.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F

from models.gaussian_measure_pe import gaussian_measure_log_bias
from models.round5_cross_attention import INPUT_DIM, MODEL_DIM, HEADS, HEAD_DIM, _temperature


METHODS = ('point', 'fixed', 'isotropic', 'full')
FIELDS = ('q', 'k', 'qmean', 'qcov', 'kmean', 'kcov',
          'k_to_q', 'k_to_qcov', 'q_to_k', 'q_to_kcov')


def position_bias(data, method, bandwidth, reverse=False):
    """Source pushforward Gaussian against native target Gaussian cell support.

    Fixed keeps each side's KNOWN native size, including 8/16 differences.
    All variants keep the same target-cell area mass. Its coefficient is one.
    """
    source, target = ('k', 'q') if reverse else ('q', 'k')
    mean = data[source+'_to_'+target]
    cov = data[source+'_to_'+target+'cov']
    target_cov = data[target+'cov']
    if method == 'fixed':
        cov = data[source+'cov']
    elif method == 'point':
        cov, target_cov = torch.zeros_like(cov), torch.zeros_like(target_cov)
    elif method == 'isotropic':
        eye = torch.eye(2, device=cov.device, dtype=cov.dtype)
        cov = cov.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None]*eye
        target_cov = target_cov.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None]*eye
    elif method != 'full':
        raise ValueError(method)
    native = data[target+'cov']
    mass = 12*torch.linalg.det(native).clamp_min(1e-12).sqrt()
    return gaussian_measure_log_bias(mean, cov, data[target+'mean'], target_cov,
        key_mass=mass, method='full', bandwidth=bandwidth)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_norm = nn.LayerNorm(MODEL_DIM)
        self.qkv = nn.Linear(MODEL_DIM, 3*MODEL_DIM)
        self.output = nn.Linear(MODEL_DIM, MODEL_DIM, bias=False)
        self.log_temperature = nn.Parameter(torch.tensor(2.2))
        self.ffn_norm = nn.LayerNorm(MODEL_DIM)
        self.ffn = nn.Sequential(nn.Linear(MODEL_DIM, 2*MODEL_DIM), nn.GELU(),
                                 nn.Linear(2*MODEL_DIM, MODEL_DIM))

    def project(self, x):
        b, n, _ = x.shape
        q, k, v = self.qkv(self.attention_norm(x)).reshape(
            b, n, 3, HEADS, HEAD_DIM).permute(2, 0, 3, 1, 4).unbind(0)
        return F.normalize(q, dim=-1), F.normalize(k, dim=-1), v

    def attend(self, q, k, v, bias, zero_values):
        score = _temperature(self.log_temperature)*(q@k.transpose(-1, -2))
        if bias is not None:
            score = score+bias[:, None]
        if zero_values:
            v = v*0
        a = score.softmax(-1)@v
        a = a.transpose(1, 2).reshape(a.shape[0], a.shape[2], MODEL_DIM)
        return self.output(a)

    def forward(self, q, k, bias, reverse_bias, zero_values=False):
        qq, qk, qv = self.project(q)
        kq, kk, kv = self.project(k)
        q2 = q+self.attend(qq, kk, kv, bias, zero_values)
        k2 = k+self.attend(kq, qk, qv, reverse_bias, zero_values)
        return q2+self.ffn(self.ffn_norm(q2)), k2+self.ffn(self.ffn_norm(k2))


class Round6MeasureMatcher(nn.Module):
    def __init__(self, method='full', bandwidth=1., placement='all'):
        super().__init__()
        if method not in METHODS or placement not in ('all', 'head') or bandwidth <= 0:
            raise ValueError('invalid method, placement or bandwidth')
        self.method, self.placement = method, placement
        self.input_projection = nn.Linear(INPUT_DIM, MODEL_DIM)
        self.layers = nn.ModuleList([_Block(), _Block()])
        self.final_norm = nn.LayerNorm(MODEL_DIM)
        self.head = nn.Linear(MODEL_DIM, HEAD_DIM)
        self.log_temperature = nn.Parameter(torch.tensor(2.2))
        # ONE shared learnable scale; no learned covariance encoder or gate.
        self.log_bandwidth = nn.Parameter(torch.tensor(math.log(bandwidth)))

    @property
    def bandwidth(self):
        return self.log_bandwidth.clamp(math.log(.05), math.log(32.)).exp()

    def forward(self, data, zero_values=False):
        bias = position_bias(data, self.method, self.bandwidth)
        reverse = position_bias(data, self.method, self.bandwidth, True) if self.placement == 'all' else None
        q, k = self.input_projection(data['q']), self.input_projection(data['k'])
        for layer in self.layers:
            q, k = layer(q, k, bias if self.placement == 'all' else None, reverse, zero_values)
        q = F.normalize(self.head(self.final_norm(q)), dim=-1)
        k = F.normalize(self.head(self.final_norm(k)), dim=-1)
        return _temperature(self.log_temperature)*(q@k.transpose(-1, -2))+bias
