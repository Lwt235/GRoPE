"""R71 target-reference PE with the unchanged R70 RGB geometry predictor."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from models.round70_nvs import (CompactNVS, BASE_FIELDS, HEIGHT, WIDTH, PATCH,
    DIM, HEADS, HD, TOKENS, Camera, patchify, unpatchify, shape_features)
from models.round71_target_projection import (project_reference,
    projection_descriptors, prepare_urope, apply_urope)

ROLES = ('rgb_target_distance', 'rgb_target_direct', 'rgb_target_relative5', 'rgb_urope')


class RGBTargetProjectionNVS(CompactNVS):
    def __init__(self, role):
        if role not in ROLES:
            raise ValueError(role)
        super().__init__('rgb_prope')
        self.role = role
        self.kind = role.removeprefix('rgb_target_') if '_target_' in role else 'urope'
        if self.kind != 'urope':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(torch.initial_seed()+70000)
                self.relative = nn.Sequential(nn.Linear({'distance':1, 'direct':6, 'relative5':5}[self.kind], 96),
                                              nn.GELU(), nn.Linear(96, DIM, bias=False))

    def forward(self, data, disable_relation=False):
        if set(data) != set(BASE_FIELDS):
            raise ValueError('only reference RGB and public cameras are accepted')
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
