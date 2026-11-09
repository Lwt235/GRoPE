"""Fixed local Gaussian moments, shared native content, four positional roles."""
import torch
from torch import nn

from experiments.round58_data import fps_and_neighbors
from experiments.native_region_moments_diagnostic import mixture_moments
from models.round58_support_pe import DIM
from models.round65_actual_gs import NativeGSClassifier, DirectPosition, common_features
from models.round69_compact_relation import CompactPosition
from third_party.region_geotransformer_reference.embedding import GeometricStructureEmbedding

ROLES = ('relative5', 'centers_only', 'distance_mlp', 'geotransformer_rpe')
GROUPS, NEIGHBORS, FLOOR = 64, 32, 1e-8


@torch.no_grad()
def pool_regions(centers, covariance, appearance):
    if centers.ndim != 3 or centers.shape[1] < max(GROUPS, NEIGHBORS):
        raise ValueError('expected a batch of at least 64 primitive centers')
    if covariance.shape != (*centers.shape, 3) or appearance.shape != (*centers.shape[:-1], 4):
        raise ValueError('primitive attributes must stay paired')
    points = centers.double()
    anchors, indices = fps_and_neighbors(points, GROUPS, NEIGHBORS)
    rows = torch.arange(len(points), device=points.device)[:, None, None]
    mean, within, between, full = mixture_moments(points[rows, indices], covariance.double()[rows, indices])
    point_covariance = between + FLOOR * torch.eye(3, dtype=between.dtype, device=between.device)
    return dict(centers=mean.to(centers), covariance=full,
                centers_covariance=point_covariance,
                appearance=appearance[rows, indices].mean(-2),
                anchors=anchors, indices=indices, within=within, between=between)


class NativeGeometricPosition(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = GeometricStructureEmbedding(DIM, .2, 15., 3, reduction_a='max')
        self.projections = nn.ModuleList([nn.Linear(DIM, DIM) for _ in range(4)])

    def forward(self, centers):
        return self.embedding(centers)


class NativeRegionClassifier(NativeGSClassifier):
    def __init__(self, role):
        if role not in ROLES:
            raise ValueError(role)
        super().__init__('nope')
        self.role = role
        if role in ('relative5', 'centers_only'):
            self.position = CompactPosition('relative5')
        elif role == 'distance_mlp':
            self.position = DirectPosition(role)
        else:
            self.position = NativeGeometricPosition()

    def forward_pooled(self, region, disable_relation=False):
        centers, full, appearance = (region[k] for k in ('centers', 'covariance', 'appearance'))
        x = self.stem(common_features(centers, full, appearance))
        relative = None
        if not disable_relation:
            if self.role == 'geotransformer_rpe':
                relative = self.position(centers)
            else:
                covariance = region['centers_covariance'] if self.role == 'centers_only' else full
                relative = self.position(centers, covariance)
        for i, layer in enumerate(self.layers):
            r = relative
            if r is not None and self.role == 'geotransformer_rpe':
                r = self.position.projections[i](r)
            x = layer(x, r, centers, None)
        x = self.final_norm(x)
        return self.classifier(torch.cat((x.mean(1), x.max(1).values), -1))

    def forward(self, centers, covariance, appearance):
        return self.forward_pooled(pool_regions(centers, covariance, appearance))
