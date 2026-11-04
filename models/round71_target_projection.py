"""Target-to-reference Gaussian support, with matched projection controls."""
import math

import torch
from torch import nn
import torch.nn.functional as F

from models.round70_nvs import (CompactNVS, BASE_FIELDS, OBSERVED_FIELDS,
    HEIGHT, WIDTH, PATCH, DIM, HEADS, HD, TOKENS, Camera, patchify, unpatchify,
    shape_features, _SymmetricLog)
from third_party.round71_urope.urope_compatible import _prepare_apply_fns

ROLES = ('observed_target_distance', 'observed_target_direct',
         'observed_target_relative5', 'observed_urope')


def project_reference(mean, covariance, K, c2w, coverage):
    """World Gaussian -> target normalized-image local moments, OpenCV axes."""
    w2c = torch.linalg.inv(c2w[:, 2].double())
    r, t = w2c[:, :3, :3], w2c[:, :3, 3]
    m = torch.einsum('bij,bnj->bni', r, mean.double())+t[:, None]
    s = r[:, None]@covariance.double()@r[:, None].transpose(-1, -2)
    k = K[:, 2].double().clone()
    k[:, 0] /= WIDTH
    k[:, 1] /= HEIGHT
    h = torch.einsum('bij,bnj->bni', k, m)
    valid = (m[..., 2] > .1) & (coverage > 0)
    # Invalid rows get finite dummy arithmetic; their extra attention term is zero.
    z = torch.where(valid, h[..., 2], torch.ones_like(h[..., 2]))
    u = h[..., :2]/z[..., None]
    jac = (k[:, None, :2, :]-u[..., :, None]*k[:, None, 2:3, :])/z[..., None, None]
    cov = jac@s@jac.transpose(-1, -2)
    cov = (cov+cov.transpose(-1, -2))*.5+torch.eye(2, device=mean.device)*1e-8
    return u.to(mean), cov.to(mean), valid


def target_centers(device, dtype):
    y, x = torch.meshgrid((torch.arange(HEIGHT//PATCH, device=device)+.5)*PATCH/HEIGHT,
                         (torch.arange(WIDTH//PATCH, device=device)+.5)*PATCH/WIDTH,
                         indexing='ij')
    return torch.stack((x, y), -1).reshape(-1, 2).to(dtype)


def projection_descriptors(u, cov, kind):
    p = target_centers(u.device, u.dtype)
    delta = p[None, :, None]-u[:, None]
    distance = delta.norm(dim=-1, keepdim=True)/.2
    if kind == 'distance':
        return distance
    if kind == 'direct':
        logc = _SymmetricLog.apply(cov)
        content = torch.stack((logc[..., 0, 0], logc[..., 0, 1]*math.sqrt(2), logc[..., 1, 1]), -1)
        return torch.cat((delta/.2, distance, content[:, None].expand(-1, TOKENS, -1, -1)), -1)
    if kind != 'relative5':
        raise ValueError(kind)
    qdiag = cov.new_tensor([(PATCH/WIDTH)**2/12, (PATCH/HEIGHT)**2/12]).double()
    d = delta.double(); s = cov.double(); inv = torch.linalg.inv(s)
    mi = (d.square()/qdiag).sum(-1)
    mj = torch.einsum('bijd,bjde,bije->bij', d, inv, d)
    ti = (s.diagonal(dim1=-2, dim2=-1)/qdiag).sum(-1)[:, None].expand_as(mi)
    tj = (inv.diagonal(dim1=-2, dim2=-1)*qdiag).sum(-1)[:, None].expand_as(mi)
    terms = torch.stack((mi, mj, ti, tj), -1).clamp_min(0).log1p().to(u)
    return torch.cat((distance, terms), -1)


def prepare_urope(K, c2w):
    """Unchanged upstream transforms; input OpenCV -> upstream OpenGL API."""
    flip = torch.diag(c2w.new_tensor([1., -1., -1., 1.]))
    return _prepare_apply_fns(head_dim=HD, viewmats=flip@torch.linalg.inv(c2w), Ks=K,
        patches_x=WIDTH//PATCH, patches_y=HEIGHT//PATCH, image_width=WIDTH,
        image_height=HEIGHT, group_size=1, depth_num=4, min_depth=2/3, max_depth=20/3)


def apply_urope(q, k, v, funcs, **kwargs):
    b, h, n, d = q.shape
    fq, fk, fo = funcs
    qc = q.reshape(b, h, 3, TOKENS, d).permute(0, 2, 1, 3, 4).reshape(b*3, h, TOKENS, d)
    kc = k[:, None].expand(-1, 3, -1, -1, -1).reshape(b*3, h, n, d)
    vc = v[:, None].expand(-1, 3, -1, -1, -1).reshape(b*3, h, n, d)
    # Upstream coefficients are float32; keep the common attention precision.
    out = F.scaled_dot_product_attention(fq(qc).to(q), fk(kc).to(k), fk(vc).to(v), **kwargs)
    return fo(out).to(q).reshape(b, 3, h, TOKENS, d).permute(0, 2, 1, 3, 4).reshape_as(q)


class TargetProjectionNVS(CompactNVS):
    def __init__(self, role):
        if role not in ROLES:
            raise ValueError(role)
        super().__init__('observed_prope')
        self.role = role
        self.kind = role.removeprefix('observed_target_') if '_target_' in role else 'urope'
        if self.kind != 'urope':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(torch.initial_seed()+70000)
                self.relative = nn.Sequential(nn.Linear({'distance':1, 'direct':6, 'relative5':5}[self.kind], 96),
                                              nn.GELU(), nn.Linear(96, DIM, bias=False))

    def forward(self, data, disable_relation=False):
        if set(data) != set(BASE_FIELDS+OBSERVED_FIELDS):
            raise ValueError('only reference RGB/geometry and public cameras are accepted')
        rgb, K, c2w = data['ref_rgb'], data['K'], data['c2w']
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            rays = patchify(self.create_rays(Camera(K, c2w, WIDTH, HEIGHT)), PATCH)
        x = self.input_tokenizer(torch.cat((patchify(rgb, PATCH), rays[:, :2]), -1))
        mean, cov, coverage = self.geometry(x, data)
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            content = torch.cat((mean, shape_features(cov), coverage[..., None]), -1)
        x = x.flatten(1, 2)+self.geometry_embed(content)
        tokens = torch.cat((x, self.query_tokenizer(rays[:, 2])), 1)
        relative = None
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            if self.kind == 'urope':
                funcs = prepare_urope(K, c2w)
            else:
                self.attention._precompute_and_cache_apply_fns(torch.linalg.inv(c2w), K)
                if not disable_relation:
                    u, scov, valid = project_reference(mean, cov, K, c2w, coverage)
                    desc = projection_descriptors(u, scov, self.kind)
        if self.kind != 'urope' and not disable_relation:
            relative = self.relative(desc).reshape(len(rgb), TOKENS, 2*TOKENS, HEADS, HD)
            relative = relative*valid[:, None, :, None, None]

        def attention(q, k, v, **kwargs):
            if self.kind == 'urope':
                return apply_urope(q, k, v, funcs, **kwargs)
            mask = None
            if relative is not None:
                bias = torch.einsum('bhid,bijhd->bhij', q[:, :, 2*TOKENS:], relative)/math.sqrt(HD)
                mask = F.pad(bias, (0, TOKENS, 2*TOKENS, 0))
            out = F.scaled_dot_product_attention(self.attention._apply_to_q(q),
                self.attention._apply_to_kv(k), self.attention._apply_to_kv(v), attn_mask=mask, **kwargs)
            return self.attention._apply_to_o(out)

        output = self.encoder(tokens, sdpa_fn=attention)
        return unpatchify(self.output_layer(output[:, -TOKENS:]), HEIGHT, WIDTH, PATCH)
