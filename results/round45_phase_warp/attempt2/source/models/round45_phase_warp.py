"""Gaussian coordinate maps inside RoPE, initialized exactly at point RoPE.

The learned field is a positional chart, not a physical deformation estimate.
It does not consume input-token covariance or preserve raw translation invariance.
"""
import torch
from torch import nn
from models.geometry import rotary
from models.round5_cross_attention import Round5CrossAttentionMatcher
from models.round44_spatial_pe import SpatialField

ROLES = ('warp_full', 'warp_iso', 'warp_fixed', 'warp_fixed_centers',
         'fourier_warp', 'mlp_warp', 'affine_warp', 'free_frequency')
BASIS_ROLES = dict(warp_full='spatial_full', warp_iso='spatial_iso',
                   warp_fixed='spatial_fixed', warp_fixed_centers='spatial_fixed_centers',
                   fourier_warp='fourier_learned', mlp_warp='coordinate_mlp')


class CoordinateWarp(SpatialField):
    def __init__(self, role):
        # Retain the exact R44 basis initialization, including its RNG order.
        super().__init__(BASIS_ROLES[role])
        # Replaces the old parameter; no inactive 36x48 projection is retained.
        self.projection = nn.Parameter(torch.zeros(36, 2))

    def forward(self, mean):
        return super().forward(mean) * self.chart_extent.double()


class AffineWarp(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('chart_center', torch.tensor([11.9375, 8.9375]))
        self.register_buffer('chart_extent', torch.tensor([12., 9.]))
        self.affine_offset = nn.Parameter(torch.zeros(2, 2))

    def forward(self, mean):
        x = (mean.double() - self.chart_center.double()) / self.chart_extent.double()
        return (x @ self.affine_offset.double().T) * self.chart_extent.double()


class FrequencyAdjustment(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.register_buffer('frequency_scale', base.double().norm(dim=-1, keepdim=True))
        self.frequency_offset = nn.Parameter(torch.zeros_like(base))


class PhaseWarpMatcher(Round5CrossAttentionMatcher):
    def __init__(self, role):
        if role not in ROLES:
            raise ValueError(role)
        super().__init__('point')
        self.method = role
        if role in BASIS_ROLES:
            self.position_field = CoordinateWarp(role)
        elif role == 'affine_warp':
            self.position_field = AffineWarp()
        else:
            self.position_field = FrequencyAdjustment(self.frequency)

    def position_arguments(self, mean):
        frequency = self.frequency.double()
        if self.method == 'free_frequency':
            field = self.position_field
            frequency = frequency + field.frequency_scale * field.frequency_offset.double()
            return mean.double(), frequency
        # Addition in native coordinates avoids a normalize/denormalize roundtrip.
        return mean.double() + self.position_field(mean), frequency

    def _position_encode(self, content, mean, covariance):
        coordinates, frequency = self.position_arguments(mean)
        return rotary(content.double(), coordinates.unsqueeze(1), frequency).to(content.dtype)


def make_model(role):
    return PhaseWarpMatcher(role)
