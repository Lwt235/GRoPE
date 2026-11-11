"""One internal memory read with Gaussian support bound to the same content slot."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from models.gaussian_memory_vggt_adapter import START, native_tail, project_memory


def projected_slot_weights(memory, cameras, intrinsic, center_variance=None):
    """Use the existing projection and preserve original slot identities.

    Geometry is frozen in this experiment. The returned null mass represents
    transmittance, not another learned scene feature.
    """
    projection = project_memory(memory, cameras, intrinsic, center_variance)
    y, x = torch.meshgrid(torch.arange(15, device=cameras.device) + .5,
                         torch.arange(20, device=cameras.device) + .5, indexing='ij')
    pixels = torch.stack((x, y), -1).reshape(-1, 2)
    delta = pixels[None, :, None] - projection['projected'][:, None]
    distance = torch.einsum('vpni,vnij,vpnj->vpn', delta,
                           torch.linalg.inv(projection['footprint']), delta).clamp_min(0)
    alpha = memory['opacity'].float().reshape(1, 1, -1) * torch.exp(-.5 * distance)
    alpha = torch.where(projection['valid'][:, None] & (distance <= 9), alpha, 0.).clamp(0, .99)
    order = projection['center'][..., 2].argsort(dim=-1, stable=True)
    sorted_alpha = alpha.gather(2, order[:, None].expand(-1, 300, -1))
    survival = (1 - sorted_alpha).cumprod(-1)
    preceding = torch.cat((torch.ones_like(survival[..., :1]), survival[..., :-1]), -1)
    sorted_weights = sorted_alpha * preceding
    # Each view has its own depth order. Scatter back before attending to K/V.
    weights = torch.zeros_like(sorted_weights).scatter(2, order[:, None].expand(-1, 300, -1), sorted_weights)
    return {'weights': weights, 'coverage': weights.sum(-1), 'transmittance': survival[..., -1]}


def attention_prior(projected, kind, slots=256):
    """Fixed geometry prior for the matched local/content/coverage/bound cases."""
    weights, coverage, transmittance = (projected[k] for k in ('weights', 'coverage', 'transmittance'))
    assert weights.shape[-1] == slots
    if kind in ('local', 'content'):
        return torch.cat((weights.new_full(weights.shape, -math.log(slots)),
                          weights.new_full((*coverage.shape, 1), -torch.inf)), -1)
    if kind == 'coverage':
        weights = (coverage / slots)[..., None].expand_as(weights)
    elif kind not in ('bound', 'center'):
        raise ValueError(kind)
    # The prior is frozen; log(0) intentionally masks a slot. Empty support has
    # exactly one finite option: the zero-valued null slot with log mass zero.
    return torch.cat((weights.log(), transmittance[..., None].log()), -1)


class BoundMemoryAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(1024)
        self.content_norm = nn.LayerNorm(32)
        self.q = nn.Linear(1024, 64)
        self.k = nn.Linear(32, 64)
        self.v = nn.Linear(32, 64)
        self.out = nn.Linear(64, 1024, bias=False)
        nn.init.zeros_(self.out.weight)

    def attention(self, attention, x, pos, legacy_pe):
        # The shared native-tail interface supplies legacy_pe. This adapter
        # leaves native self-attention unchanged and has no learned A/B maps.
        return attention(x, pos=pos)

    def content_residual(self, tokens, payload):
        patches = tokens[:, :, START:]
        q = self.q(self.norm(patches)).reshape(1, -1, 4, 16).transpose(1, 2)
        content = self.content_norm(payload['content'])
        k = self.k(content).reshape(1, -1, 4, 16).transpose(1, 2)
        v = self.v(content).reshape(1, -1, 4, 16).transpose(1, 2)
        k = torch.cat((k, k.new_zeros(1, 4, 1, 16)), 2)
        v = torch.cat((v, v.new_zeros(1, 4, 1, 16)), 2)
        prior = payload['log_prior'].reshape(1, 1, q.shape[2], k.shape[2])
        value = F.scaled_dot_product_attention(q, k, v, attn_mask=prior, dropout_p=0.)
        value = value.transpose(1, 2).reshape(1, 4, 300, 64)
        return tokens + F.pad(self.out(value), (0, 0, START, 0))


def bound_native_tail(backbone, sample, adapter, content, log_prior, checkpoint_blocks=False):
    placeholder = sample['prefix'].new_zeros(4, 305, 10)
    return native_tail(backbone, sample, adapter, placeholder,
                       {'content': content, 'log_prior': log_prior}, checkpoint_blocks)
