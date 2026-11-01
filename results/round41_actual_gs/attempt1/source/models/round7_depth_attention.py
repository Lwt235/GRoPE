"""One learned attention/V block, matched positional controls and self-context."""
import torch
from torch import nn
import torch.nn.functional as F
from models.gaussian_quadratic_pe import controlled_features

CONFIGS = {'full': ('full', .5, 5.), 'fixed': ('fixed', .5, 5.),
           'isotropic': ('isotropic', .5, 5.), 'point': ('point', 1., 10.),
           'point_pck': ('point', .5, 10.), 'singleview': ('full', .5, 5.)}
FIELDS = ('q', 'k', 'qmean', 'qcov', 'kmean', 'kcov', 'q_to_k', 'q_to_kcov',
          'q_depth', 'q_value_depth', 'k_value_depth')


class DepthAttention(nn.Module):
    def __init__(self, role='full'):
        super().__init__()
        self.role = role
        self.method, self.bandwidth, self.temperature = CONFIGS[role]
        self.qk = nn.Linear(128, 128, bias=False)
        self.query = nn.Linear(129, 64)
        self.value = nn.Linear(129, 64)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
        with torch.no_grad():
            self.qk.weight.copy_(torch.eye(128))
            self.head[-1].weight.normal_(0, .001)
            self.head[-1].bias.zero_()

    def forward(self, data, intervention=None, return_attention=False):
        # An explicit allowlist makes label fields unavailable to the model.
        d = {key: data[key] for key in FIELDS}
        if self.role == 'singleview':
            d.update(k=d['q'], kmean=d['qmean'], kcov=d['qcov'],
                     q_to_k=d['qmean'], q_to_kcov=d['qcov'],
                     k_value_depth=d['q_value_depth'])
        qg, kg = controlled_features(d, self.method, self.bandwidth)
        q = F.normalize(self.qk(d['q']), dim=-1)
        k = F.normalize(self.qk(d['k']), dim=-1)
        logits = self.temperature * (q @ k.transpose(-1, -2)) + qg @ kg.transpose(-1, -2)
        attn = logits.softmax(-1)
        qbase = d['q_depth'].clamp(.1, 20).log()
        qhidden = F.gelu(self.query(torch.cat([d['q'], qbase[..., None]], -1)))
        values = F.gelu(self.value(torch.cat([d['k'], d['k_value_depth'].clamp(.1,20).log()[..., None]], -1)))
        if intervention == 'zero_v':
            values = torch.zeros_like(values)
        elif intervention == 'reverse_v':
            values = values.flip(-2)
        elif intervention is not None:
            raise ValueError(intervention)
        aggregate = attn @ values
        residual = self.head(torch.cat([qhidden, aggregate], -1)).squeeze(-1).clamp(-3,3)
        depth = (qbase + residual).exp().clamp(.1,20)
        return (depth, attn, values) if return_attention else depth


def depth_loss(prediction, target, valid):
    safe_target = torch.where(valid, target, torch.ones_like(target))
    error = (prediction.log() - safe_target.log()).abs()
    return torch.where(valid, error, torch.zeros_like(error)).sum()/valid.sum().clamp_min(1)
