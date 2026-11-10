"""Keep each existing relation unchanged inside the same fixed rotary backbone."""
import math
import torch
from torch.nn import functional as F

from models.round58_support_pe import DIM, HEADS, HD
from models.round65_actual_gs import shape_features
from models.region_hrope_control import CameraRegionMatcher, homographies, transport
from models.region_fixed_hrope import ConstantHomographyParameters

ROLES = ('relative5', 'distance_mlp', 'geotransformer_rpe', 'fixed_isotropic5')


def rotary_relation_layer(layer, x, matrix, inverse, image_positions, relative=None, cross=False):
    batch, views, tokens, _ = x.shape
    q, k, v = layer.qkv(layer.norm(x)).reshape(batch, views, tokens, 3, HEADS, HD).permute(3, 0, 1, 4, 2, 5).unbind(0)
    raw_q = q
    q = transport(q, matrix.transpose(-1, -2), image_positions, True)
    k = transport(k, inverse, image_positions, True)
    v = transport(v, inverse, image_positions, True)
    if cross:
        assert relative is None, 'No unaligned cross-frame 3D relations.'
        k, v = k.flip(1), v.flip(1)
    score = q@k.transpose(-1, -2)
    if relative is not None:
        r = relative.reshape(batch, views, tokens, tokens, HEADS, HD)
        score = score+torch.einsum('bvhid,bvijhd->bvhij', raw_q, r)
    probability = (score/math.sqrt(HD)).softmax(-1)
    aggregate = transport(probability@v, matrix, image_positions, False)
    aggregate = aggregate.transpose(-3, -2).reshape(batch, views, tokens, DIM)
    x = x+layer.output(aggregate)
    return x+layer.ffn(layer.ffn_norm(x))


class RotaryRelationMatcher(CameraRegionMatcher):
    def __init__(self, role):
        if role not in ROLES:
            raise ValueError(role)
        super().__init__(role)
        self.fixed_transform = ConstantHomographyParameters()

    def forward(self, features, centers, covariance, permutation, intrinsics, image_positions):
        camera = intrinsics[..., :2, :].flatten(-2).unsqueeze(-2).expand(*features.shape[:-1], 6)
        content = torch.cat((features, centers, shape_features(covariance).to(features),
                             camera.to(features), image_positions.to(features)), -1)
        x = self.stem(content)
        matrix, inverse = homographies(self.fixed_transform(x), intrinsics.unsqueeze(-3))
        relative = self.position(centers) if self.role == 'geotransformer_rpe' else self.position(centers, covariance)
        for index, layer in enumerate(self.layers):
            r = relative if index % 2 == 0 else None
            if r is not None and self.role == 'geotransformer_rpe':
                r = self.position.projections[index//2](r)
            x = rotary_relation_layer(layer, x, matrix, inverse, image_positions, r, cross=index % 2 == 1)
        descriptor = F.normalize(self.head(self.final_norm(x)), dim=-1)
        temperature = self.log_temperature.clamp(max=math.log(60.)).exp()
        return temperature*(descriptor[:, 0]@descriptor[:, 1].transpose(-1, -2))
