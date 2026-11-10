"""Camera-information-matched regional PE controls, including predicted HRoPE."""
import math
import torch
from torch import nn
from torch.nn import functional as F

from models.round58_support_pe import DIM, HEADS, HD, AttentionBlock
from models.round65_actual_gs import DirectPosition, shape_features
from models.round69_compact_relation import CompactPosition
from models.region_geometry_controls import GeometricPosition
from models.region_shape_attribution import fixed_hook
from models.round50_local_frame import rotation6

ROLES = ('relative5', 'distance_mlp', 'geotransformer_rpe', 'fixed_isotropic5', 'predicted_hrope')


def homographies(raw, intrinsics):
    rotation = rotation6(raw[..., :6])
    rt = rotation.transpose(-1, -2)
    translation = raw[..., 6:9].clamp(-.5, .5)
    normal = F.normalize(raw[..., 9:12], dim=-1)
    kappa = raw[..., 12].clamp(-.5, .5)
    matrix = rotation+kappa[..., None, None]*translation[..., :, None]*normal[..., None, :]
    u = (rt@translation[..., None])[..., 0]
    v = (normal[..., None, :]@rt)[..., 0, :]
    denominator = (1+kappa*(normal*u).sum(-1)).clamp_min(.1)
    inverse = rt-kappa[..., None, None]*u[..., :, None]*v[..., None, :]/denominator[..., None, None]
    return intrinsics@matrix.clamp(-5, 5), inverse.clamp(-5, 5)@torch.linalg.inv(intrinsics)


def transport(value, matrix, image_positions, inverse_rope):
    # Values [B,V,H,N,48], matrices [B,V,N,3,3], positions [B,V,N,2].
    geom = value[..., :24].reshape(*value.shape[:-1], 8, 3)
    geom = torch.einsum('bvnij,bvhnkj->bvhnki', matrix, geom).flatten(-2)
    frequency = 10000. ** (-2*torch.arange(1, 7, device=value.device, dtype=value.dtype)/24.)
    phase = (8*image_positions[..., :, None]*frequency).flatten(-2).unsqueeze(2)
    if inverse_rope:
        phase = -phase
    a, b = value[..., 24:].reshape(*value.shape[:-1], 12, 2).unbind(-1)
    rotary = torch.stack((a*phase.cos()-b*phase.sin(), a*phase.sin()+b*phase.cos()), -1).flatten(-2)
    return torch.cat((geom, rotary), -1)


def hrope_layer(layer, x, matrix, inverse, image_positions, cross):
    batch, views, tokens, _ = x.shape
    q, k, v = layer.qkv(layer.norm(x)).reshape(batch, views, tokens, 3, HEADS, HD).permute(3, 0, 1, 4, 2, 5).unbind(0)
    q = transport(q, matrix.transpose(-1, -2), image_positions, True)
    k = transport(k, inverse, image_positions, True)
    v = transport(v, inverse, image_positions, True)
    if cross:
        k, v = k.flip(1), v.flip(1)
    probability = ((q@k.transpose(-1, -2))/math.sqrt(HD)).softmax(-1)
    aggregate = transport(probability@v, matrix, image_positions, False)
    aggregate = aggregate.transpose(-3, -2).reshape(batch, views, tokens, DIM)
    x = x+layer.output(aggregate)
    return x+layer.ffn(layer.ffn_norm(x))


class CameraRegionMatcher(nn.Module):
    def __init__(self, role):
        super().__init__()
        if role not in ROLES:
            raise ValueError(role)
        self.role = role
        # Same draws as the existing regional backbone and PE controls.
        self.stem = nn.Sequential(nn.Linear(20, DIM), nn.GELU(), nn.Linear(DIM, DIM))
        self.layers = nn.ModuleList([AttentionBlock() for _ in range(4)])
        self.final_norm = nn.LayerNorm(DIM)
        self.head = nn.Linear(DIM, HD)
        self.log_temperature = nn.Parameter(torch.tensor(2.2))
        with torch.random.fork_rng(devices=[]):
            wider = nn.Linear(28, DIM)
            with torch.no_grad():
                wider.weight.zero_()
                wider.weight[:, :20].copy_(self.stem[0].weight)
                wider.bias.copy_(self.stem[0].bias)
            self.stem[0] = wider
        if role in ('relative5', 'fixed_isotropic5'):
            self.position = CompactPosition('relative5')
            if role == 'fixed_isotropic5':
                self.position.mlp.register_forward_pre_hook(fixed_hook)
        elif role == 'distance_mlp':
            self.position = DirectPosition(role)
        elif role == 'geotransformer_rpe':
            self.position = GeometricPosition()
        else:
            self.position = nn.Linear(DIM, 13)
            nn.init.zeros_(self.position.weight)
            nn.init.zeros_(self.position.bias)
            with torch.no_grad():
                self.position.bias[:6].copy_(torch.tensor([1., 0, 0, 0, 1., 0]))
                self.position.bias[11] = 1.
                self.position.bias[12] = .25

    def forward(self, features, centers, covariance, permutation, intrinsics, image_positions):
        camera = intrinsics[..., :2, :].flatten(-2).unsqueeze(-2).expand(*features.shape[:-1], 6)
        content = torch.cat((features, centers, shape_features(covariance).to(features),
                             camera.to(features), image_positions.to(features)), -1)
        x = self.stem(content)
        if self.role == 'predicted_hrope':
            matrix, inverse = homographies(self.position(x), intrinsics.unsqueeze(-3))
            for index, layer in enumerate(self.layers):
                x = hrope_layer(layer, x, matrix, inverse, image_positions, cross=index % 2 == 1)
        else:
            relative = self.position(centers) if self.role == 'geotransformer_rpe' else self.position(centers, covariance)
            for index, layer in enumerate(self.layers):
                r = relative if index % 2 == 0 else None
                if r is not None and self.role == 'geotransformer_rpe':
                    r = self.position.projections[index//2](r)
                x = layer(x, r, cross=index % 2 == 1)
        descriptor = F.normalize(self.head(self.final_norm(x)), dim=-1)
        temperature = self.log_temperature.clamp(max=math.log(60.)).exp()
        return temperature*(descriptor[:, 0]@descriptor[:, 1].transpose(-1, -2))
