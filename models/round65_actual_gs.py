"""Matched Transformers receiving native Gaussian geometry and appearance.

All roles see identical content attributes. Only explicit PE construction
changes. No RGB-to-geometry prediction, camera inputs or rendering module.
"""
import math
import torch
from torch import nn
from models.geometry import rotary
from models.round58_support_pe import DIM, HEADS, HD, SupportPosition

ROLES = ('gaussian_relation', 'direct_mlp', 'distance_mlp', 'center_rope', 'nope')


def vectorize(matrix):
    indices = torch.triu_indices(3, 3, device=matrix.device)
    weights = torch.where(indices[0] == indices[1], 1., math.sqrt(2.)).to(matrix)
    return matrix[..., indices[0], indices[1]]*weights


def shape_features(covariance):
    values, vectors = torch.linalg.eigh(covariance.double())
    # The data pipeline already adds the common 1e-8 I. This threshold is the
    # same last-resort eigenvalue floor as the retained R58 operator.
    logs = .5*values.clamp_min(1e-15).div(.04).log()
    matrix = (vectors*logs.unsqueeze(-2))@vectors.transpose(-1, -2)
    return vectorize(matrix)


def common_features(centers, covariance, appearance):
    if centers.shape[-1] != 3 or appearance.shape != (*centers.shape[:-1], 4):
        raise ValueError('expected center3 and opacity/DC appearance4')
    if covariance.shape != (*centers.shape[:-1], 3, 3):
        raise ValueError('expected native covariance per token')
    return torch.cat((centers, shape_features(covariance).to(centers), appearance.to(centers)), -1)


class DirectPosition(nn.Module):
    def __init__(self, role):
        super().__init__()
        self.role = role
        width = 16 if role == 'direct_mlp' else 1
        self.mlp = nn.Sequential(nn.Linear(width, DIM), nn.GELU(), nn.Linear(DIM, DIM, bias=False))

    def descriptors(self, centers, covariance):
        delta = (centers.unsqueeze(-2)-centers.unsqueeze(-3))/.2
        distance = delta.norm(dim=-1, keepdim=True)
        if self.role == 'distance_mlp':
            return distance
        shape = shape_features(covariance).to(centers)
        n = centers.shape[-2]
        left = shape.unsqueeze(-2).expand(*shape.shape[:-1], n, 6)
        right = shape.unsqueeze(-3).expand(*shape.shape[:-2], n, n, 6)
        return torch.cat((delta, distance, left, right), -1)

    def forward(self, centers, covariance):
        return self.mlp(self.descriptors(centers, covariance))


def frequencies():
    # Fixed spherical schedule used in the prior native classification branch.
    i = torch.arange(24, dtype=torch.float64)
    z = 1-2*(i+.5)/24
    angle = i*math.pi*(3-math.sqrt(5))
    radius = (1-z.square()).sqrt()
    directions = torch.stack((radius*angle.cos(), radius*angle.sin(), z), -1)
    return (directions*torch.logspace(-1, .6, 24, dtype=torch.float64)[:, None]).float()


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(DIM)
        self.qkv = nn.Linear(DIM, 3*DIM)
        self.output = nn.Linear(DIM, DIM)
        self.ffn_norm = nn.LayerNorm(DIM)
        self.ffn = nn.Sequential(nn.Linear(DIM, 2*DIM), nn.GELU(), nn.Linear(2*DIM, DIM))

    def forward(self, x, relative=None, centers=None, frequency=None):
        b, n, _ = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, n, 3, HEADS, HD).permute(2, 0, 3, 1, 4).unbind(0)
        if frequency is not None:
            position = 12*centers.double().unsqueeze(1)
            q = rotary(q.double(), position, frequency.double()).to(x.dtype)
            k = rotary(k.double(), position, frequency.double()).to(x.dtype)
        score = q@k.transpose(-1, -2)
        if relative is not None:
            r = relative.reshape(b, n, n, HEADS, HD)
            score = score+torch.einsum('bhid,bijhd->bhij', q, r)
        probability = (score/math.sqrt(HD)).softmax(-1)
        aggregate = (probability@v).transpose(1, 2).reshape(b, n, DIM)
        x = x+self.output(aggregate)
        return x+self.ffn(self.ffn_norm(x))


class NativeGSClassifier(nn.Module):
    def __init__(self, role):
        super().__init__()
        if role not in ROLES:
            raise ValueError(role)
        self.role = role
        self.stem = nn.Sequential(nn.Linear(13, DIM), nn.GELU(), nn.Linear(DIM, DIM))
        self.layers = nn.ModuleList([Block() for _ in range(4)])
        self.final_norm = nn.LayerNorm(DIM)
        self.classifier = nn.Sequential(nn.Linear(2*DIM, 128), nn.GELU(),
                                        nn.Dropout(.3), nn.Linear(128, 60))
        self.register_buffer('frequency', frequencies())
        # Complete shared construction before any role-specific random draws.
        if role == 'gaussian_relation':
            self.position = SupportPosition('invariant_mlp')
        elif role in ('direct_mlp', 'distance_mlp'):
            self.position = DirectPosition(role)
        else:
            self.position = nn.Module()

    def forward(self, centers, covariance, appearance):
        x = self.stem(common_features(centers, covariance, appearance))
        relative = self.position(centers, covariance) if self.role in ROLES[:3] else None
        for layer in self.layers:
            x = layer(x, relative, centers, self.frequency if self.role == 'center_rope' else None)
        x = self.final_norm(x)
        return self.classifier(torch.cat((x.mean(1), x.max(1).values), -1))
