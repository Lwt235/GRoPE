"""One residual read into actual VGGT patch tokens; no teacher at inference."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class ViewTokenAdapter(nn.Module):
    def __init__(self, variant, center_variance=.7146816253662109, token_dim=2048):
        super().__init__()
        assert variant in ('local', 'center', 'gaussian')
        self.variant, self.heads = variant, 4
        self.width = 66 if variant == 'local' else 64
        self.query = nn.Linear(token_dim + 5, self.width)
        self.local = nn.Linear(self.width, self.width)
        self.output = nn.Linear(self.width, token_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        if variant != 'local':
            self.key = nn.Linear(42, self.width)
            self.value = nn.Linear(32, self.width)
            self.position_scale = nn.Parameter(torch.full((4,), math.log(math.expm1(1.))))
        self.register_buffer('center_variance', torch.tensor(float(center_variance)))

    def forward(self, tokens, geometry, condition='normal'):
        # Raw residual preserves the existing token mean and scale. The frozen
        # DPT head applies LayerNorm; training compares those normalized tokens.
        tokens = tokens.float()
        normalized = F.layer_norm(tokens, (tokens.shape[-1],))
        q = F.gelu(self.query(torch.cat([normalized, geometry['query']], -1)))
        hidden = self.local(q)
        if self.variant != 'local' and condition != 'removed':
            B, P = q.shape[:2]
            N = geometry['key'].shape[1]
            Q = q.view(B, P, 4, 16).transpose(1, 2)
            K = self.key(geometry['key']).view(B, N, 4, 16).transpose(1, 2)
            V = self.value(geometry['value']).view(B, N, 4, 16).transpose(1, 2)
            distance = (geometry['distance2'] / self.center_variance if self.variant == 'center'
                        else geometry['mahalanobis'])
            score = Q @ K.transpose(-1, -2) / 4.
            score = score - .5 * F.softplus(self.position_scale)[None, :, None, None] * distance[:, None]
            score = score + geometry['log_weight'][:, None, None]
            score = torch.cat([score, score.new_zeros(B, 4, P, 1)], -1)
            read = (score.softmax(-1)[..., :-1] @ V).transpose(1, 2).reshape(B, P, 64)
            hidden = hidden + read
        scale = (tokens.var(-1, unbiased=False, keepdim=True) + 1e-5).sqrt()
        return tokens + scale * self.output(F.gelu(hidden))


def normalized_error(prediction, teacher):
    return (F.layer_norm(prediction.float(), (prediction.shape[-1],)) -
            F.layer_norm(teacher.float(), (teacher.shape[-1],))).square().mean(-1)


def apply_rgb_mask(images, patch_masks):
    """Only observed pixels enter the backbone; masking after DINO is invalid."""
    pixel_mask = patch_masks.reshape(-1, 15, 20).repeat_interleave(14, 1).repeat_interleave(14, 2)
    fill = images.new_tensor([.485, .456, .406])[None, :, None, None]
    return torch.where(pixel_mask[:, None], fill, images)


def inject_last_layer(token_list, patch_start, adapter, geometry):
    """Return a new VGGT token list consumed by its existing camera/depth heads."""
    result = list(token_list)
    last = token_list[-1].float().clone()
    B, S = last.shape[:2]
    patch = last[:, :, patch_start:].reshape(B * S, -1, last.shape[-1])
    last[:, :, patch_start:] = adapter(patch, geometry).reshape(B, S, -1, last.shape[-1])
    result[-1] = last
    return result
