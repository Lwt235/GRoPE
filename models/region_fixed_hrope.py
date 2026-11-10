"""HRoPE's existing operator with its per-token transform fixed to H=K."""
import torch
from torch import nn

from models.region_hrope_control import CameraRegionMatcher


class ConstantHomographyParameters(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('raw', torch.tensor(
            [1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., .25]))

    def forward(self, features):
        return self.raw.to(features).expand(*features.shape[:-1], 13)


class FixedHRegionMatcher(CameraRegionMatcher):
    def __init__(self, role='fixed_hrope'):
        if role != 'fixed_hrope':
            raise ValueError(role)
        # Consume exactly the original initialization draws, including its head.
        super().__init__('predicted_hrope')
        self.position = ConstantHomographyParameters()
        # Inherit the entire HRoPE forward without changing its operator/placement.
