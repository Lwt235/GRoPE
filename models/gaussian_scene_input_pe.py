"""One input-level scene attention residual; native Transformer layers unchanged."""
import torch
from models.gaussian_scene_learned_projection import LearnedProjectionReader
from models.gaussian_scene_native import forward_read as native_forward


class InputPEReader(LearnedProjectionReader):
    def __init__(self):
        super().__init__()

    def input_residual(self, sample, kind):
        patches = sample['patches']
        if kind == 'L':
            return self(patches, patches[0], None, None)
        bias = self.position_bias(kind, sample['geometry'], special_tokens=0)
        return self(patches, sample['features'], bias, sample['valid'])


def forward_read(model, sample, reader=None, kind='L', checkpoint_blocks=True, trace=None):
    with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
        changed = dict(sample)
        if reader is not None:
            residual = reader.input_residual(sample, kind)
            if trace is not None and residual.requires_grad:
                residual.register_hook(lambda grad: trace.__setitem__('input', float(grad.float().norm())))
            changed['patches'] = sample['patches'] + residual
        return native_forward(model, changed, reader=None, checkpoint_blocks=checkpoint_blocks)
