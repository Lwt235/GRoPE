"""Existing geometric PE controls on the unchanged R68 common backbone."""
import math
import torch
from torch import nn
from torch.nn import functional as F

from models.round58_support_pe import DIM, HD, AttentionBlock
from models.round59_relation_controls import RelativeMLP
from models.round65_actual_gs import shape_features
from third_party.region_geotransformer_reference.embedding import GeometricStructureEmbedding

ROLES = ('geotransformer_rpe', 'angle_descriptor_mlp')


class GeometricPosition(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = GeometricStructureEmbedding(DIM, .2, 15., 3, reduction_a='max')
        # Upstream RPEMultiHeadAttention has a distinct proj_p in each self layer.
        self.projections = nn.ModuleList([nn.Linear(DIM, DIM) for _ in range(2)])

    def forward(self, centers):
        batch, views, count, _ = centers.shape
        return self.embedding(centers.reshape(batch*views, count, 3)).reshape(batch, views, count, count, DIM)


class GeometryControlMatcher(nn.Module):
    def __init__(self, role):
        super().__init__()
        if role not in ROLES:
            raise ValueError(role)
        self.role = role
        # Keep all common draws in exactly the R68/R69 construction order.
        self.stem = nn.Sequential(nn.Linear(20, DIM), nn.GELU(), nn.Linear(DIM, DIM))
        self.layers = nn.ModuleList([AttentionBlock() for _ in range(4)])
        self.final_norm = nn.LayerNorm(DIM)
        self.head = nn.Linear(DIM, HD)
        self.log_temperature = nn.Parameter(torch.tensor(2.2))
        self.position = GeometricPosition() if role == 'geotransformer_rpe' else RelativeMLP(role)

    def forward(self, features, centers, covariance, permutation, disable_relation=False):
        content = torch.cat((features, centers, shape_features(covariance).to(features)), -1)
        x = self.stem(content)
        relative = None
        if not disable_relation:
            relative = (self.position(centers) if self.role == 'geotransformer_rpe'
                        else self.position(features, centers, covariance))
        for index, layer in enumerate(self.layers):
            r = relative if index % 2 == 0 else None
            if r is not None and self.role == 'geotransformer_rpe':
                r = self.position.projections[index//2](r)
            x = layer(x, r, cross=index % 2 == 1)
        descriptor = F.normalize(self.head(self.final_norm(x)), dim=-1)
        temperature = self.log_temperature.clamp(max=math.log(60.)).exp()
        return temperature*(descriptor[:, 0]@descriptor[:, 1].transpose(-1, -2))
