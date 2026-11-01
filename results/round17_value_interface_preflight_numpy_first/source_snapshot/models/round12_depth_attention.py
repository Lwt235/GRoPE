"""Round7's 41,345-parameter head with frozen regional geometry logits.

Only the cross-view positional bias changes.  Content Q/K remains learned.
The single-view control retains the original native quadratic geometry.
"""
import math

import torch
import torch.nn.functional as F

from models.gaussian_quadratic_pe import controlled_features
from models.round7_depth_attention import DepthAttention as OriginalDepthAttention
from models.round7_depth_attention import depth_loss


FUSION_ROLES = ('full4', 'iso4_matched', 'full1', 'quad4', 'quad16',
                'point', 'point_pck', 'fixed')
ROLES = FUSION_ROLES + ('singleview',)
FUSION_FIELDS = ('q', 'k', 'q_depth', 'k_value_depth', 'geometry_logits')
SINGLEVIEW_FIELDS = ('q', 'qmean', 'qcov', 'q_depth', 'q_value_depth')


class DepthAttention(OriginalDepthAttention):
    def __init__(self, role='full4', temperature=5.):
        if role not in ROLES:
            raise ValueError('unknown Round12 role')
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError('finite nonnegative content temperature required')
        if role == 'singleview' and temperature != 5.:
            raise ValueError('singleview retains original temperature 5')
        # Calling the original constructor preserves parameter order, random
        # initialization and dimensions exactly, including the identity Q/K.
        super().__init__('full')
        self.role, self.temperature = role, float(temperature)
        self.bandwidth = .5 if role == 'singleview' else None
        self.method = 'full' if role == 'singleview' else 'precomputed_geometry'

    def forward(self, data, intervention=None, return_attention=False):
        if self.role == 'singleview':
            d = {name: data[name] for name in SINGLEVIEW_FIELDS}
            d.update(k=d['q'], kmean=d['qmean'], kcov=d['qcov'],
                     q_to_k=d['qmean'], q_to_kcov=d['qcov'],
                     k_value_depth=d['q_value_depth'])
            qg, kg = controlled_features(d, 'full', .5)
            geometry = qg @ kg.transpose(-1, -2)
        else:
            d = {name: data[name] for name in FUSION_FIELDS}
            geometry = d['geometry_logits']
            if geometry.shape != d['q'].shape[:-1] + (d['k'].shape[-2],):
                raise ValueError('geometry and Q/K shape mismatch')
            if geometry.dtype != d['q'].dtype or geometry.device != d['q'].device:
                raise ValueError('geometry dtype/device differs from Q/K')
        q = F.normalize(self.qk(d['q']), dim=-1)
        k = F.normalize(self.qk(d['k']), dim=-1)
        logits = self.temperature * (q @ k.transpose(-1, -2)) + geometry
        attention = logits.softmax(-1)
        qbase = d['q_depth'].clamp(.1, 20).log()
        qhidden = F.gelu(self.query(torch.cat([d['q'], qbase[..., None]], -1)))
        values = F.gelu(self.value(torch.cat([
            d['k'], d['k_value_depth'].clamp(.1, 20).log()[..., None]], -1)))
        if intervention == 'zero_v':
            values = torch.zeros_like(values)
        elif intervention == 'reverse_v':
            values = values.flip(-2)
        elif intervention is not None:
            raise ValueError(intervention)
        aggregate = attention @ values
        residual = self.head(torch.cat([qhidden, aggregate], -1)).squeeze(-1).clamp(-3, 3)
        depth = (qbase + residual).exp().clamp(.1, 20)
        return (depth, attention, values) if return_attention else depth
