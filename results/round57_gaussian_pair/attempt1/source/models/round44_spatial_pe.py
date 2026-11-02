"""Explicit Gaussian spatial fields as Transformer PE; an RBF encoder probe.

This is an application of known Gaussian basis encoders, not a new basis theorem.
Learned field covariance is separate from physical input-token covariance.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from models.round5_cross_attention import Round5CrossAttentionMatcher

ROLES = ('spatial_full', 'spatial_diag', 'spatial_iso', 'spatial_fixed',
         'spatial_fixed_centers', 'fourier_fixed', 'fourier_learned', 'coordinate_mlp')


def gaussian_response(points, centers, precision_factor):
    """P = L L^T; valid for arbitrary input dimension, with no density normalizer."""
    delta = points.unsqueeze(-2) - centers
    z = torch.einsum('...gi,gij->...gj', delta, precision_factor)
    return torch.exp(-0.5 * z.square().sum(-1))


class SpatialField(nn.Module):
    def __init__(self, role):
        super().__init__()
        if role not in ROLES:
            raise ValueError(role)
        self.role = role
        # Common projection is drawn BEFORE role-specific random initialization.
        self.projection = nn.Parameter(torch.randn(36, 48) * .02)
        self.register_buffer('chart_center', torch.tensor([11.9375, 8.9375]))
        self.register_buffer('chart_extent', torch.tensor([12., 9.]))
        if role.startswith('spatial_'):
            axis = (torch.arange(6) + .5) / 3 - 1
            yy, xx = torch.meshgrid(axis, axis, indexing='ij')
            self.register_buffer('centers', torch.stack((xx.flatten(), yy.flatten()), -1))
            if role not in ('spatial_fixed', 'spatial_fixed_centers'):
                self.mean_offset = nn.Parameter(torch.zeros(36, 2))
            if role != 'spatial_fixed':
                self.log_precision = nn.Parameter(torch.zeros(36, 1 if role == 'spatial_iso' else 2))
                if role in ('spatial_full', 'spatial_fixed_centers'):
                    self.lower = nn.Parameter(torch.zeros(36, 1))
        elif role.startswith('fourier_'):
            angle = torch.arange(18) * math.pi * (3 - math.sqrt(5))
            frequency = torch.stack((angle.cos(), angle.sin()), -1) * torch.logspace(-1, .6, 18)[:, None]
            self.register_buffer('base_frequency', frequency * self.chart_extent)
            self.register_buffer('base_phase', frequency @ self.chart_center)
            if role == 'fourier_learned':
                self.frequency_offset = nn.Parameter(torch.zeros(18, 2))
                self.phase_offset = nn.Parameter(torch.zeros(18))
        else:
            self.register_buffer('base_mlp', torch.randn(2, 36) / math.sqrt(2))
            self.mlp_offset = nn.Parameter(torch.zeros(2, 36))
            self.bias = nn.Parameter(torch.zeros(36))
        # Label-free reference grid fixes initial feature energy across families.
        yy, xx = torch.meshgrid(torch.arange(18), torch.arange(24), indexing='ij')
        raw = torch.stack((xx.flatten() + .4375, yy.flatten() + .4375), -1).double()
        with torch.no_grad():
            basis = self.basis((raw - self.chart_center.double()) / self.chart_extent.double())
            rms = basis.square().sum(-1).mean().sqrt()
        self.register_buffer('initial_rms', rms)

    def precision_factor(self):
        base = self.centers.double()
        diag = self.log_precision.double().exp().expand(-1, 2) if hasattr(self, 'log_precision') else torch.ones_like(base)
        factor = torch.diag_embed(diag)
        if hasattr(self, 'lower'):
            extra = torch.zeros_like(factor)
            extra[:, 1, 0] = self.lower[:, 0].double()
            factor = factor + extra
        return 3 * factor  # initial spatial standard deviation is 1/3.

    def basis(self, coordinates):
        x = coordinates.double()
        if self.role.startswith('spatial_'):
            centers = self.centers.double()
            if hasattr(self, 'mean_offset'):
                centers = centers + self.mean_offset.double()
            return gaussian_response(x, centers, self.precision_factor())
        if self.role.startswith('fourier_'):
            omega = self.base_frequency.double()
            phase = self.base_phase.double()
            if hasattr(self, 'frequency_offset'):
                omega = omega + self.frequency_offset.double() * self.chart_extent.double()
                phase = phase + self.phase_offset.double()
            angles = x @ omega.T + phase
            return torch.cat((angles.cos(), angles.sin()), -1)
        return F.gelu(x @ (self.base_mlp.double() + self.mlp_offset.double()) + self.bias.double())

    def forward(self, mean):
        x = (mean.double() - self.chart_center.double()) / self.chart_extent.double()
        return (self.basis(x) / self.initial_rms) @ self.projection.double()


class SpatialMatcher(Round5CrossAttentionMatcher):
    def __init__(self, role):
        super().__init__('point')
        self.method = role
        self.position_field = SpatialField(role)

    def _position_encode(self, content, mean, covariance):
        # The new hypotheses do not consume input-token physical covariance.
        position = self.position_field(mean).unsqueeze(1)
        return F.normalize(content.double() + position, dim=-1).to(content.dtype)


def make_model(role):
    return SpatialMatcher(role)
