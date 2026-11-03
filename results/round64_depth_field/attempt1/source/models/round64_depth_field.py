"""Single RGB depth field, coupled center/support, original R62 relation PE."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from models.round53_ray_frame import center_rays
from models.round62_rgb_relations import (RGBRelationMatcher, common_frame, FLOOR,
    select_inputs, make_model as reference_model)

NEW_ROLES = ('depth_surface', 'depth_footprint', 'depth_local_mlp')
REFERENCE_ROLES = ('camera_only', 'camera_plane', 'ray_point', 'ray_affine',
                   'camera_mlp', 'free_relation', 'predicted_full', 'predicted_point')
ROLES = REFERENCE_ROLES + NEW_ROLES
HEIGHT, WIDTH, STRIDE, PIXEL_VARIANCE = 18, 24, 8., 5.25


def grid_indices(position, geometry):
    if position.shape[1:] != (HEIGHT*WIDTH, 2):
        raise ValueError('R64 requires the complete 18x24 image grid')
    center = geometry[:, :1, 14:16]*position.new_tensor([192., 144.])
    ij = ((position*STRIDE+center-3.5)/STRIDE).round().long()
    return ij[..., 1]*WIDTH+ij[..., 0]


def validate_grid(data):
    """Once per loaded input split; no device synchronization inside attention."""
    for prefix in ('q', 'k'):
        pos, geom = data[prefix+'pos'], data[prefix+'geom']
        index = grid_indices(pos, geom)
        expected = torch.arange(HEIGHT*WIDTH, device=pos.device).expand_as(index)
        if not torch.equal(index.sort(-1).values, expected):
            raise ValueError('missing, duplicate or out-of-range grid location')
        ij = torch.stack((index % WIDTH, index // WIDTH), -1).to(pos.dtype)
        center = geom[:, :1, 14:16]*pos.new_tensor([192., 144.])
        if not torch.allclose(pos*STRIDE+center, ij*STRIDE+3.5, atol=5e-5, rtol=0):
            raise ValueError('not the declared pixel grid')


def depth_derivatives(depth, position, geometry):
    index = grid_indices(position, geometry)
    grid = torch.zeros_like(depth).scatter(1, index, depth).reshape(-1, HEIGHT, WIDTH)
    du = torch.cat(((grid[..., 1:2]-grid[..., :1])/STRIDE,
                    (grid[..., 2:]-grid[..., :-2])/(2*STRIDE),
                    (grid[..., -1:]-grid[..., -2:-1])/STRIDE), -1)
    dv = torch.cat(((grid[:, 1:2]-grid[:, :1])/STRIDE,
                    (grid[:, 2:]-grid[:, :-2])/(2*STRIDE),
                    (grid[:, -1:]-grid[:, -2:-1])/STRIDE), -2)
    return du.flatten(1).gather(1, index), dv.flatten(1).gather(1, index)


def depth_fields(raw, data, prefix):
    z = 1/30+F.softplus(raw[..., 0]+math.log(math.expm1(29/30)))
    rays = center_rays(data[prefix+'pos'], data[prefix+'K'])
    du, dv = depth_derivatives(z, data[prefix+'pos'], data[prefix+'geom'])
    fx = data[prefix+'K'][:, 0, 0, None]*192
    fy = data[prefix+'K'][:, 1, 1, None]*144
    return z, rays, du, dv, fx, fy


def local_gaussian(z, rays, du, dv, fx, fy, surface=True):
    zero = torch.zeros_like(z)
    ju = torch.stack((z/fx, zero, zero), -1)
    jv = torch.stack((zero, z/fy, zero), -1)
    if surface:
        ju = ju+rays*du[..., None]
        jv = jv+rays*dv[..., None]
    cov = PIXEL_VARIANCE*(ju[..., :, None]*ju[..., None, :]+jv[..., :, None]*jv[..., None, :])
    cov = cov+FLOOR**2*torch.eye(3, device=z.device, dtype=z.dtype)
    return z[..., None]*rays, cov


class DepthFieldMatcher(RGBRelationMatcher):
    def __init__(self, variant):
        if variant not in NEW_ROLES:
            raise ValueError(variant)
        super().__init__('predicted_point')
        self.variant = variant
        if variant == 'depth_local_mlp':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(torch.initial_seed()+62000)
                self.field['relation'] = nn.Sequential(nn.Linear(10, 96), nn.GELU(),
                                                        nn.Linear(96, 96, bias=False))

    def fields(self, features, data, prefix):
        return depth_fields(self.field['predictor'](features), data, prefix)

    def geometry(self, features, data, prefix):
        fields = self.fields(features, data, prefix)
        mean, covariance = local_gaussian(*fields, surface=self.variant == 'depth_surface')
        return common_frame(mean, covariance, data[prefix+'geom'])

    def relative(self, fq, fk, data):
        if self.variant != 'depth_local_mlp':
            return super().relative(fq, fk, data)
        endpoints = []
        for features, prefix in ((fq, 'q'), (fk, 'k')):
            z, rays, du, dv, fx, fy = self.fields(features, data, prefix)
            endpoints.append(torch.stack((z, fx*du, fy*dv, rays[..., 0], rays[..., 1]), -1))
        a, b = endpoints
        batch, nq, _ = a.shape
        nk = b.shape[1]
        pair = torch.cat((a[:, :, None].expand(batch, nq, nk, 5),
                          b[:, None].expand(batch, nq, nk, 5)), -1)
        reverse = torch.cat((pair[..., 5:], pair[..., :5]), -1).transpose(1, 2)
        return self.field['relation'](pair), self.field['relation'](reverse)


def make_model(role):
    return DepthFieldMatcher(role) if role in NEW_ROLES else reference_model(role)
