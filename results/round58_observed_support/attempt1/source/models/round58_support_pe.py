"""One relative operator over observed Gaussian supports, matched to controls.

No pose, labels, world coordinates or masks are accepted by the model API.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

DIM, HEADS, HD = 96, 2, 48
ROLES = ('gaussian', 'gaussian_distance', 'distance', 'distance_angle', 'invariant_mlp', 'nope')


def sine_embedding(value):
    frequency = torch.exp(torch.arange(0, DIM, 2, device=value.device, dtype=value.dtype)
                          * (-math.log(10000.)/DIM))
    phase = value.unsqueeze(-1)*frequency
    return torch.stack((phase.sin(), phase.cos()), -1).flatten(-2)


def gaussian_terms(centers, covariance):
    """FP64 contractions; return squared Euclidean, two directed metric/trace terms."""
    c, s = centers.double(), covariance.double()
    p = torch.linalg.inv(s)
    delta = c.unsqueeze(-2)-c.unsqueeze(-3)
    d2 = delta.square().sum(-1)
    mi = torch.einsum('...ijd,...ide,...ije->...ij', delta, p, delta)
    mj = mi.transpose(-1, -2)
    tij = torch.einsum('...ide,...jed->...ij', p, s)
    return d2, mi, mj, tij, tij.transpose(-1, -2)


def scalar_geometry(centers, covariance):
    d2, mi, mj, ti, tj = gaussian_terms(centers, covariance)
    # Exact self distance is zero; avoid a tiny positive inverse-product residual.
    diagonal = torch.eye(centers.shape[-2], device=centers.device, dtype=torch.bool)
    compatibility = (ti+tj-6).clamp_min(0).masked_fill(diagonal, 0)
    metric = (mi+mj).clamp_min(0)
    return d2, metric, compatibility


def triplet_angles(centers, neighbors=3):
    delta = centers.unsqueeze(-3)-centers.unsqueeze(-2)  # j-i
    d2 = delta.square().sum(-1)
    diagonal = torch.eye(centers.shape[-2], device=centers.device, dtype=torch.bool)
    ids = d2.masked_fill(diagonal, float('inf')).topk(neighbors, largest=False).indices
    near = torch.gather(delta, -2, ids.unsqueeze(-1).expand(*ids.shape, 3))
    dot = torch.einsum('...ikd,...ijd->...ijk', near, delta)
    cross = torch.linalg.cross(near.unsqueeze(-3), delta.unsqueeze(-2), dim=-1).norm(dim=-1)
    return torch.atan2(cross, dot)


class SupportPosition(nn.Module):
    def __init__(self, role):
        super().__init__()
        self.role = role
        if role in ('gaussian', 'gaussian_distance', 'distance', 'distance_angle'):
            self.projection = nn.Linear(DIM, DIM, bias=False)
            if role == 'distance_angle':
                self.angle_projection = nn.Linear(DIM, DIM, bias=False)
        elif role == 'invariant_mlp':
            self.mlp = nn.Sequential(nn.Linear(11, DIM), nn.GELU(), nn.Linear(DIM, DIM, bias=False))
        elif role != 'nope':
            raise ValueError(role)

    def forward(self, centers, covariance):
        if self.role == 'nope':
            return None
        dtype = centers.dtype
        if self.role in ('gaussian', 'gaussian_distance'):
            _, metric, compatibility = scalar_geometry(centers, covariance)
            if self.role == 'gaussian':
                metric = metric+compatibility
            return self.projection(sine_embedding((.5*metric).sqrt()).to(dtype))
        if self.role in ('distance', 'distance_angle'):
            distance = (centers.unsqueeze(-2)-centers.unsqueeze(-3)).norm(dim=-1)/.2
            result = self.projection(sine_embedding(distance))
            if self.role == 'distance_angle':
                angles = triplet_angles(centers)/(math.pi/12)
                result = result+self.angle_projection(sine_embedding(angles)).max(-2).values
            return result
        d2, mi, mj, ti, tj = gaussian_terms(centers, covariance)
        spectrum = .5*torch.linalg.eigvalsh(covariance.double()).clamp_min(1e-15).div(.04).log()
        n = centers.shape[-2]
        a = spectrum.unsqueeze(-2).expand(*spectrum.shape[:-1], n, 3)
        b = spectrum.unsqueeze(-3).expand(*spectrum.shape[:-2], n, n, 3)
        scalars = torch.stack((d2.sqrt()/.2, mi.clamp_min(0).log1p(), mj.clamp_min(0).log1p(),
                               ti.clamp_min(0).log1p(), tj.clamp_min(0).log1p()), -1)
        inputs = torch.cat((scalars[..., :1], a, b, scalars[..., 1:]), -1).to(dtype)
        return self.mlp(inputs)


class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(DIM)
        self.qkv = nn.Linear(DIM, 3*DIM)
        self.output = nn.Linear(DIM, DIM)
        self.ffn_norm = nn.LayerNorm(DIM)
        self.ffn = nn.Sequential(nn.Linear(DIM, 2*DIM), nn.GELU(), nn.Linear(2*DIM, DIM))

    def forward(self, x, relative=None, cross=False):
        # x [B,2,N,D]; both views update from the same pre-update states.
        batch, views, tokens, _ = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(batch, views, tokens, 3, HEADS, HD).permute(3, 0, 1, 4, 2, 5).unbind(0)
        if cross:
            k, v = k.flip(1), v.flip(1)
        score = q@k.transpose(-1, -2)
        if relative is not None:
            r = relative.reshape(batch, views, tokens, tokens, HEADS, HD)
            score = score+torch.einsum('bvhid,bvijhd->bvhij', q, r)
        probability = (score/math.sqrt(HD)).softmax(-1)
        aggregate = (probability@v).transpose(-3, -2).reshape(batch, views, tokens, DIM)
        x = x+self.output(aggregate)
        return x+self.ffn(self.ffn_norm(x))


class SupportMatcher(nn.Module):
    def __init__(self, role):
        super().__init__()
        if role not in ROLES:
            raise ValueError(role)
        self.role = role
        self.stem = nn.Sequential(nn.Linear(11, DIM), nn.GELU(), nn.Linear(DIM, DIM))
        self.layers = nn.ModuleList([AttentionBlock() for _ in range(4)])
        self.final_norm = nn.LayerNorm(DIM)
        self.head = nn.Linear(DIM, HD)
        self.log_temperature = nn.Parameter(torch.tensor(2.2))
        self.position = SupportPosition(role)  # common initialization precedes role-specific draws

    def forward(self, features, centers, covariance):
        x = self.stem(features)
        r = self.position(centers, covariance)
        for index, layer in enumerate(self.layers):
            x = layer(x, r if index % 2 == 0 else None, cross=index % 2 == 1)
        descriptor = F.normalize(self.head(self.final_norm(x)), dim=-1)
        temperature = self.log_temperature.clamp(max=math.log(60.)).exp()
        return temperature*(descriptor[:, 0]@descriptor[:, 1].transpose(-1, -2))
