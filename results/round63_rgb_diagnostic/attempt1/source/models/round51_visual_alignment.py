"""R51 input adapter; frozen R50 operators remain unchanged."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from models.round50_local_frame import LocalFrameMatcher, rotation6, block_apply, rotate

ROLES = ('rope', 'mlp', 'hrope', 'hrope_paper', 'gs_frame', 'gs_iso',
         'affine', 'frame_identity', 'gaussian_bias', 'nope')
INPUT_FIELDS = ('q', 'k', 'qgeom', 'kgeom', 'qpos', 'kpos', 'qK', 'kK')


class VisualAlignmentMatcher(LocalFrameMatcher):
    def __init__(self, role):
        if role not in ROLES: raise ValueError(role)
        super().__init__('rope' if role == 'nope' else 'hrope' if role == 'hrope_paper' else role)
        self.experiment_role = role
        if role == 'hrope_paper':
            with torch.no_grad(): self.field.bias[12] = .25
        # Reset this shared layer independently of role-specific RNG consumption.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(torch.initial_seed()+51000)
            self.input_projection[0] = nn.Linear(128+18, 96)

    def matrices(self, features, intrinsics):
        if self.experiment_role != 'hrope_paper':
            return super().matrices(features, intrinsics)
        raw = self.field(features)
        r = rotation6(raw[..., :6]); rt = r.transpose(-1, -2)
        t = raw[..., 6:9].clamp(-.5, .5)
        normal = F.normalize(raw[..., 9:12], dim=-1)
        # Same nonzero initial kappa (.25) as the R50 adaptation.
        kappa = raw[..., 12].clamp(-.5, .5)
        a = r+kappa[..., None, None]*t[..., :, None]*normal[..., None, :]
        u = (rt@t[..., None])[..., 0]
        v = (normal[..., None, :]@rt)[..., 0, :]
        denominator = (1+kappa*(normal*u).sum(-1)).clamp_min(.1)
        ai = rt-kappa[..., None, None]*u[..., :, None]*v[..., None, :]/denominator[..., None, None]
        # Appendix A.5 safety bounds; the determinant bound makes these inactive
        # for nondegenerate rotations in the exact stated parameter range.
        a, ai = a.clamp(-5, 5), ai.clamp(-5, 5)
        ki = torch.diag_embed(1/intrinsics.diagonal(dim1=-2, dim2=-1))
        return intrinsics[:, None]@a, ai@ki[:, None]

    def transport(self, value, matrix, pos, operation):
        if self.experiment_role != 'hrope_paper':
            return super().transport(value, matrix, pos, operation)
        # Paper Hcal=diag(H,Phi), hence transpose/inverse use negative phase,
        # while output uses Phi. Frequency band remains matched to other roles.
        return torch.cat((block_apply(value[..., :24], matrix),
            rotate(value[..., 24:], pos, inverse=(operation != 'output'))), -1)

    def encode(self, data, regime='oracle', zero_values=False):
        if regime != 'oracle': raise ValueError('R51 has one common input regime')
        selected = {k:data[k] for k in INPUT_FIELDS}
        # Frozen visual features are L2-normalized; fixed RMS scaling is common.
        selected['qrgb'] = selected.pop('q')*math.sqrt(128)+.5
        selected['krgb'] = selected.pop('k')*math.sqrt(128)+.5
        if self.experiment_role == 'nope':
            selected['qpos'] = torch.zeros_like(selected['qpos'])
            selected['kpos'] = torch.zeros_like(selected['kpos'])
        return super().encode(selected, 'oracle', zero_values)
