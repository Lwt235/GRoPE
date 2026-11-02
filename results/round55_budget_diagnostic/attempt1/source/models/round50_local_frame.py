"""Learned local-frame PE and information-matched operator controls."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from models.round5_cross_attention import _CrossAttentionBlock, _temperature
from models.round47_conditional_gaussian import GaussianPosition

ROLES = ('rope', 'mlp', 'hrope', 'gs_frame', 'gs_iso', 'affine', 'frame_identity', 'gaussian_bias')


def rotation6(raw):
    a = F.normalize(raw[..., :3], dim=-1)
    b = raw[..., 3:6]-(raw[..., 3:6]*a).sum(-1, keepdim=True)*a
    b = F.normalize(b, dim=-1)
    return torch.stack((a, b, torch.cross(a, b, dim=-1)), -1)


def rotate(value, xy, inverse=False):
    width = value.shape[-1]; pairs = width//4
    omega = torch.logspace(-1, .6, pairs, dtype=value.dtype, device=value.device)
    phase = (xy[..., :, None]*omega).flatten(-2).unsqueeze(1)
    if inverse: phase = -phase
    z = value.reshape(*value.shape[:-1], -1, 2)
    a, b = z.unbind(-1)
    return torch.stack((a*phase.cos()-b*phase.sin(), a*phase.sin()+b*phase.cos()), -1).flatten(-2)


def block_apply(value, matrix):
    d = matrix.shape[-1]
    z = value.reshape(*value.shape[:-1], -1, d)
    return torch.einsum('bnij,bhnrj->bhnri', matrix, z).flatten(-2)


class LocalFrameMatcher(nn.Module):
    def __init__(self, role):
        super().__init__()
        if role not in ROLES: raise ValueError(role)
        self.role = role
        # All roles see identical RGB AND geometry, including ordinary RoPE.
        self.input_projection = nn.Sequential(nn.Linear(220, 96), nn.GELU(), nn.Linear(96, 96))
        self.layers = nn.ModuleList([_CrossAttentionBlock(), _CrossAttentionBlock()])
        self.final_norm = nn.LayerNorm(96)
        self.head = nn.Linear(96, 48)
        self.log_temperature = nn.Parameter(torch.tensor(2.2))
        if role == 'mlp':
            self.field = nn.Sequential(nn.Linear(96, 16), nn.GELU(), nn.Linear(16, 2))
            nn.init.zeros_(self.field[-1].weight); nn.init.zeros_(self.field[-1].bias)
        elif role == 'gaussian_bias':
            self.field = GaussianPosition('conditional2d')
        elif role in ('hrope', 'gs_frame', 'gs_iso', 'affine'):
            size = {'hrope':13, 'gs_frame':12, 'gs_iso':4, 'affine':12}[role]
            self.field = nn.Linear(96, size)
            nn.init.zeros_(self.field.weight); nn.init.zeros_(self.field.bias)
            if role in ('hrope', 'gs_frame'):
                with torch.no_grad(): self.field.bias[:6].copy_(torch.tensor([1., 0, 0, 0, 1, 0]))
            if role == 'hrope':
                with torch.no_grad():
                    self.field.bias[11] = 1.
                    self.field.bias[12] = math.atanh(.5)

    def matrices(self, features, intrinsics):
        b, n, _ = features.shape
        eye = torch.eye(3, device=features.device, dtype=features.dtype).expand(b, n, 3, 3)
        if self.role == 'frame_identity':
            a = torch.eye(4, device=features.device, dtype=features.dtype).expand(b, n, 4, 4)
            return a, a
        raw = self.field(features)
        if self.role == 'hrope':
            r = rotation6(raw[..., :6]); rt = r.transpose(-1, -2)
            t = .5*torch.tanh(raw[..., 6:9]); normal = F.normalize(raw[..., 9:12], dim=-1)
            kappa = .5*torch.tanh(raw[..., 12])
            a = r+kappa[..., None, None]*t[..., :, None]*normal[..., None, :]
            u = (rt@t[..., None])[..., 0]; v = (normal[..., None, :]@rt)[..., 0, :]
            denom = 1+kappa*(normal*u).sum(-1)
            ai = rt-kappa[..., None, None]*u[..., :, None]*v[..., None, :]/denom[..., None, None]
            # Normalized intrinsics from the common calibrated inputs.
            ki = torch.diag_embed(1/intrinsics.diagonal(dim1=-2, dim2=-1))
            return intrinsics[:, None]@a, ai@ki[:, None]
        if self.role == 'affine':
            log_l = .5*torch.tanh(raw[..., :9]).reshape(b, n, 3, 3)
            linear = torch.matrix_exp(log_l); inverse = torch.matrix_exp(-log_l)
            mu = torch.tanh(raw[..., 9:12])
        else:
            if self.role == 'gs_iso':
                r = eye; logs = .5*torch.tanh(raw[..., :1]).expand(b, n, 3); mu = torch.tanh(raw[..., 1:4])
            else:
                r = rotation6(raw[..., :6])
                logs = .5*torch.tanh(raw[..., 6:9]); mu = torch.tanh(raw[..., 9:12])
            # Canonical SPD square root: equivalent Gaussian axis sign/permutation
            # representations produce the same transform, unlike a raw R*S frame.
            linear = r@torch.diag_embed(logs.exp())@r.transpose(-1, -2)
            inverse = r@torch.diag_embed((-logs).exp())@r.transpose(-1, -2)
        # G takes local Gaussian coordinates to a latent common frame; A=G^-1.
        g = torch.zeros(b, n, 4, 4, dtype=features.dtype, device=features.device)
        a = torch.zeros_like(g)
        g[..., :3, :3] = linear; g[..., :3, 3] = mu; g[..., 3, 3] = 1
        a[..., :3, :3] = inverse; a[..., :3, 3] = -(inverse@mu[..., None])[..., 0]; a[..., 3, 3] = 1
        return a, g

    def transport(self, value, matrix, pos, operation):
        # Choose rotary convention so Q/K use the conventional positive phase.
        first = block_apply(value[..., :24], matrix)
        last = rotate(value[..., 24:], pos, inverse=(operation == 'output'))
        return torch.cat((first, last), -1)

    def encode(self, data, regime='oracle', zero_values=False):
        suffix = 'geom' if regime == 'oracle' else 'noisy'
        q = self.input_projection(torch.cat((data['qrgb']-.5, data['q'+suffix]), -1))
        k = self.input_projection(torch.cat((data['krgb']-.5, data['k'+suffix]), -1))
        qp, kp = data['qpos'], data['kpos']
        for layer in self.layers:
            fq, fk = layer.attention_norm(q), layer.attention_norm(k)
            qq, qk, qv = layer._project(q); kq, kk, kv = layer._project(k)
            qq, qk, kq, kk = [F.normalize(t, dim=-1) for t in (qq, qk, kq, kk)]
            if self.role in ('rope', 'mlp'):
                qxy, kxy = qp, kp
                if self.role == 'mlp':
                    qxy = qp+4*torch.tanh(self.field(fq)/4); kxy = kp+4*torch.tanh(self.field(fk)/4)
                qq, qk = rotate(qq, qxy), rotate(qk, qxy)
                kq, kk = rotate(kq, kxy), rotate(kk, kxy)
                aq = ak = None
            elif self.role == 'gaussian_bias':
                aq = ak = None
            else:
                aq, iq = self.matrices(fq, data['qK']); ak, ik = self.matrices(fk, data['kK'])
                qq = self.transport(qq, aq.transpose(-1, -2), qp, 'query')
                qk = self.transport(qk, iq, qp, 'key'); qv = self.transport(qv, iq, qp, 'value')
                kq = self.transport(kq, ak.transpose(-1, -2), kp, 'query')
                kk = self.transport(kk, ik, kp, 'key'); kv = self.transport(kv, ik, kp, 'value')
            sq = _temperature(layer.log_temperature)*(qq@kk.transpose(-1, -2))
            sk = _temperature(layer.log_temperature)*(kq@qk.transpose(-1, -2))
            if self.role == 'gaussian_bias':
                sq = sq+self.field.bias(fq, qp, kp).to(sq.dtype)
                sk = sk+self.field.bias(fk, kp, qp).to(sk.dtype)
            if zero_values: qv, kv = qv*0, kv*0
            dq, dk = sq.softmax(-1)@kv, sk.softmax(-1)@qv
            if aq is not None:
                dq = self.transport(dq, aq, qp, 'output'); dk = self.transport(dk, ak, kp, 'output')
            def update(x, dx):
                x = x+layer.output(dx.transpose(1, 2).reshape(x.shape))
                return x+layer.ffn(layer.ffn_norm(x))
            q, k = update(q, dq), update(k, dk)
        return q, k

    def forward(self, data, regime='oracle', zero_values=False):
        q, k = self.encode(data, regime, zero_values)
        q, k = [F.normalize(self.head(self.final_norm(t)), dim=-1) for t in (q, k)]
        return _temperature(self.log_temperature)*(q@k.transpose(-1, -2))
