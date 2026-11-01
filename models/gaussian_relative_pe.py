"""Exact real-arithmetic reparameterization of Gaussian PE with fixed query C.

This is an optional conversion of gaussian_konly checkpoints, not a replacement
for the full variable-query model. Relative C may be indefinite: its response is
a ratio of positive Gaussian filters, not a Gaussian characteristic function.
Floating equivalence is audited on ordinary encoder outputs whose original and
converted norms exceed F.normalize's epsilon. Extremely tiny encoder outputs
can encounter different epsilon branches; this is not a universal bitwise map.
"""
import copy
import torch
from torch import nn
import torch.nn.functional as F

from models.gaussian_pe import normalized_rotary
from models.geometry import rotary


class RelativeSupportMatcher(nn.Module):
    def __init__(self, fixed_query_model):
        super().__init__()
        if fixed_query_model.method != 'gaussian_konly':
            raise ValueError('Only fixed-query Gaussian checkpoints can be converted')
        self.encoder = copy.deepcopy(fixed_query_model.encoder)
        self.logtemp = nn.Parameter(fixed_query_model.logtemp.detach().clone())
        self.dustbin = nn.Parameter(fixed_query_model.dustbin.detach().clone())
        self.dustbin_head = copy.deepcopy(fixed_query_model.dustbin_head)
        frequency = fixed_query_model.freq2.detach().clone()
        self.register_buffer('frequency', frequency)
        # The query uses C0 = I/12 + resolution^2 I. D0 acts equally on each
        # sin/cos channel pair, and hence commutes with all phase rotations.
        variance = frequency.double().square().sum(-1) * (
            1/12 + float(fixed_query_model.resolution.square()))
        amplitude = (-variance/2).exp().repeat_interleave(2)
        self.register_buffer('inverse_reference', amplitude.reciprocal().to(frequency.dtype))
        with torch.no_grad():
            last = self.encoder[-1]
            last.weight.mul_(amplitude.to(last.weight.dtype)[:, None])
            last.bias.mul_(amplitude.to(last.bias.dtype))

    def forward(self, data):
        raw_q, raw_k = self.encoder(data['q']), self.encoder(data['k'])
        q, k = F.normalize(raw_q, dim=-1), F.normalize(raw_k, dim=-1)
        # Recover the original normalized content for the rejection head;
        # using q here would silently change unmatched probabilities.
        original_q = F.normalize(raw_q*self.inverse_reference, dim=-1)
        q = rotary(q.double(), data['pixq'].double(), self.frequency.double()).to(q.dtype)
        eye = torch.eye(2, device=k.device, dtype=k.dtype)
        relative_cov = data['projected_kcov'] - eye/12
        k = normalized_rotary(k, data['k_to_q'], relative_cov, self.frequency)
        score = self.logtemp.exp().clamp(max=60)*(q@k.transpose(-1, -2))
        unmatched = self.dustbin+self.dustbin_head(original_q).squeeze(-1)
        return torch.cat([score, unmatched[..., None]], -1)
