"""One input-level residual fusion; no added attention or replacement depth head."""
import torch
from torch import nn
import torch.nn.functional as F

from models.gaussian_scene_native import forward_read as native_forward
from models.gaussian_scene_memory import rasterize_field
from models.gaussian_memory_coarse_pe import patch_tokens


class RenderedFeatureFusion(nn.Module):
    def __init__(self, center_pe=False):
        super().__init__()
        self.input = nn.Linear(2048, 256)
        self.output = nn.Linear(256, 1024)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, patches, extra):
        return self.output(F.gelu(self.input(torch.cat((patches.float(), extra.float()), -1))))


def forward_read(model, sample, reader=None, kind='Local', checkpoint_blocks=True, trace=None):
    changed = dict(sample)
    if reader is not None:
        patches = sample['patches']
        if kind == 'Local':
            extra = patches.float().mean(1, keepdim=True).expand_as(patches)
        else:
            extra = sample['extra'][None] if sample['extra'].ndim == 2 else sample['extra']
        residual = reader(patches, extra)
        if trace is not None and residual.requires_grad:
            residual.register_hook(lambda grad: trace.__setitem__('input', float(grad.float().norm())))
        changed['patches'] = patches + residual
    return native_forward(model, changed, reader=None, checkpoint_blocks=checkpoint_blocks)


@torch.no_grad()
def fresh_sample(model, image, query, field, kind):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        patches = patch_tokens(model, image)
    extra = None
    if kind != 'Local':
        K = query['K'].clone()
        K[:2] /= 14
        valid = field['has_content']
        if kind == 'Render':
            colors = field['features'].float() * valid[:, None]
            rendered, _, _ = rasterize_field(field, query['c2w'][None], K[None], 32, 32, colors)
            extra = rendered[0].reshape(1024, 1024)
        elif kind == 'Global':
            covered, _, _ = rasterize_field(field, query['c2w'][None], K[None], 32, 32, valid.float()[:, None])
            extra = covered[0].reshape(1024, 1) * field['features'][valid].float().mean(0)
        else:
            raise ValueError(kind)
    return dict(patches=patches, extra=extra, hw=(448, 448))
