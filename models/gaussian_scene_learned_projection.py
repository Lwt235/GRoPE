"""A single learned PE mapping retaining the existing center relation."""
import torch
from torch import nn
import torch.nn.functional as F

from models.gaussian_scene_memory import SceneReader


class LearnedProjectionReader(SceneReader):
    def __init__(self, center_pe=True):
        # Preserve the original seeded common initialization, then append a zero column.
        super().__init__(center_pe=True)
        original = self.center_pe[0]
        extended = nn.Linear(8, 16)
        with torch.no_grad():
            extended.weight[:, :7].copy_(original.weight)
            extended.weight[:, 7].zero_()
            extended.bias.copy_(original.bias)
        self.center_pe[0] = extended

    def position_bias(self, kind, geometry, special_tokens=5):
        assert kind in ('P', 'S', 'G') and geometry.shape[-1] == 8
        bias = self.center_pe(geometry).permute(2, 0, 1)
        return F.pad(bias, (0, 0, special_tokens, 0))


def spherical_field(field):
    """Alter covariance only for PE rasterization; never rewrite saved content."""
    result = dict(field)
    radius = field['scales'].float().square().mean(-1).sqrt()
    result['scales'] = radius[:, None].expand(-1, 3).contiguous()
    result['rotations'] = torch.zeros_like(field['rotations'])
    result['rotations'][:, 0] = 1
    return result
