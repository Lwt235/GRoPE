"""A saved Gaussian feature field and one shared view-conditioned reader.

The writer/rendering entry points accept source data or a saved field and a
camera. They never accept query RGB or supervision. Covariance denotes support.
"""
import torch
from torch import nn
import torch.nn.functional as F


def field_bytes(field):
    """Count actual persistent tensor payload, including camera/scale metadata."""
    def count(value):
        if torch.is_tensor(value):
            return value.numel() * value.element_size()
        if isinstance(value, dict):
            return sum(count(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return sum(count(v) for v in value)
        return 0
    return count(field)


def rasterize_field(field, c2w, intrinsic, height, width, colors=None, mode='RGB'):
    """Use standard gsplat compositing; camera/intrinsics are in field coordinates."""
    from gsplat import rasterization
    return rasterization(
        means=field['means'].float().contiguous(),
        quats=F.normalize(field['rotations'].float(), dim=-1).contiguous(),
        scales=field['scales'].float().contiguous(),
        opacities=field['opacities'].float().contiguous(),
        colors=(field['colors'] if colors is None else colors).float().contiguous(),
        viewmats=torch.linalg.inv(c2w.float()).contiguous(),
        Ks=intrinsic.float().contiguous(), width=width, height=height,
        near_plane=.01, far_plane=100., packed=False, eps2d=.3,
        channel_chunk=32, render_mode=mode)


@torch.no_grad()
def contribution_weights(field, c2w, intrinsic, height, width):
    """Per-basis alpha contributions at the declared feature raster resolution.

    Rendering basis colors gives the actual renderer's weights, including
    front-to-back transmittance. This is not an analytic overlap surrogate or
    a full-patch probability integral.
    """
    identity = torch.eye(len(field['means']), device=field['means'].device)
    weights, alpha, _ = rasterize_field(field, c2w, intrinsic, height, width, identity)
    return weights, alpha


@torch.no_grad()
def lift_source_features(field, source_features, source_c2w, source_intrinsic):
    """Bind frozen source feature pixels using their standard alpha contributions."""
    views, height, width, channels = source_features.shape
    assert views == len(source_c2w) == len(source_intrinsic)
    weights, _ = contribution_weights(field, source_c2w, source_intrinsic, height, width)
    weights = weights.reshape(-1, len(field['means']))
    mass = weights.sum(0)
    features = weights.T @ source_features.reshape(-1, channels).float()
    features = features / mass[:, None].clamp_min(1e-8)
    return features.half().contiguous(), mass.float().contiguous(), mass > 1e-8


def center_ray_descriptor(means, c2w, intrinsic, height, width, unit):
    """Seven center-only quantities; no depth, shape or alpha input is accepted."""
    yy, xx = torch.meshgrid(torch.arange(height, device=means.device) + .5,
                            torch.arange(width, device=means.device) + .5, indexing='ij')
    pixels = torch.stack((xx, yy, torch.ones_like(xx)), -1).reshape(-1, 3)
    rays = F.normalize(pixels @ torch.linalg.inv(intrinsic.float()).T @ c2w[:3, :3].T, dim=-1)
    delta = (means.float() - c2w[:3, 3]) / torch.as_tensor(unit, device=means.device)
    along = rays @ delta.T
    radius2 = (delta.square().sum(-1)[None] - along.square()).clamp_min(0)
    scales = means.new_tensor([.02, .05, .1, .2, .5])
    radial = torch.log1p(radius2[..., None] / scales.square())
    distance = torch.log1p(delta.norm(dim=-1))[None].expand_as(along)
    return torch.cat((radial, torch.asinh(along)[..., None], distance[..., None]), -1)


class SceneReader(nn.Module):
    """The same width256/four-head reader is reused at four global layers."""
    def __init__(self, center_pe=False):
        super().__init__()
        self.query_norm = nn.LayerNorm(1024)
        self.memory_norm = nn.LayerNorm(1024)
        self.query = nn.Linear(1024, 256)
        self.key = nn.Linear(1024, 256)
        self.value = nn.Linear(1024, 256)
        self.output = nn.Linear(256, 1024)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.center_pe = nn.Sequential(nn.Linear(7, 16), nn.GELU(),
                                       nn.Linear(16, 4, bias=False)) if center_pe else None
        if self.center_pe is not None:
            nn.init.zeros_(self.center_pe[-1].weight)

    def position_bias(self, kind, geometry, special_tokens=5):
        if kind in ('L', 'C'):
            return None
        if kind in ('T', 'P'):
            bias = self.center_pe(geometry).permute(2, 0, 1)
        elif kind == 'G':
            bias = torch.log1p(geometry.shape[-1] * geometry.float())[None].expand(4, -1, -1)
        else:
            raise ValueError(kind)
        return F.pad(bias, (0, 0, special_tokens, 0))

    def forward(self, query, features, position_bias=None, valid=None):
        if features.shape[-2] == 0 or (valid is not None and not bool(valid.any())):
            return torch.zeros_like(query)
        q = self.query(self.query_norm(query)).reshape(1, -1, 4, 64).transpose(1, 2)
        normalized = self.memory_norm(features)
        k = self.key(normalized).reshape(1, -1, 4, 64).transpose(1, 2)
        v = self.value(normalized).reshape(1, -1, 4, 64).transpose(1, 2)
        bias = position_bias
        if valid is not None:
            mask = torch.where(valid, 0., -torch.inf)[None, None, :]
            bias = mask if bias is None else bias + mask
        if bias is not None:
            bias = bias.to(q.dtype)
        result = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.)
        return self.output(result.transpose(1, 2).reshape(1, -1, 256))


def field_to_device(field, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in field.items()}
