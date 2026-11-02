"""Unfrozen R53 reserve: ray-anchored means and inverse Gaussian frames.

No task result is implied. Keep R52 frozen and finish it before selecting a
successor experiment. All geometry is inferred from the same common inputs.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from models.round51_visual_alignment import VisualAlignmentMatcher, INPUT_FIELDS
from models.round5_cross_attention import _temperature
from models.round50_local_frame import block_apply
from models.round52_camera_frame import CameraFrameMatcher

ROLES = ('nope', 'mlp', 'hrope', 'projection_only', 'pose_only', 'ray_point', 'ray_iso',
         'ray_gaussian', 'ray_affine', 'ray_gaussian_hybrid')
FIELDS = dict(ray_point=1, ray_iso=2, ray_gaussian=7, ray_affine=10, ray_gaussian_hybrid=7)


def camera_pose(geometry):
    b = len(geometry)
    t = torch.eye(4, dtype=geometry.dtype, device=geometry.device).expand(b,4,4).clone()
    t[:, :3, :3] = geometry[:,0,:9].reshape(b,3,3)
    t[:, :3, 3] = geometry[:,0,9:12]
    return t


def center_rays(positions, intrinsics):
    # positions are centered native pixels divided by the 8px token stride.
    normalized_image = positions*positions.new_tensor([8/192, 8/144])
    homogeneous = torch.cat((normalized_image, torch.ones_like(positions[...,:1])), -1)
    return torch.einsum('bij,bnj->bni', torch.linalg.inv(intrinsics), homogeneous)


class RayFrameMatcher(VisualAlignmentMatcher):
    def __init__(self, role):
        if role not in ROLES or role == 'projection_only':
            raise ValueError(role)
        super().__init__(role if role in ('nope','mlp','hrope') else 'frame_identity')
        self.experiment_role = role
        if role in FIELDS:
            self.field = nn.Linear(96, FIELDS[role])
            nn.init.zeros_(self.field.weight); nn.init.zeros_(self.field.bias)

    def gaussian_frame(self, features, rays):
        b,n,_ = features.shape
        identity = torch.eye(3, dtype=features.dtype, device=features.device).expand(b,n,3,3)
        if self.experiment_role == 'pose_only':
            g = torch.eye(4, dtype=features.dtype, device=features.device).expand(b,n,4,4)
            return g, g
        raw = self.field(features)
        depth = .1+F.softplus(raw[...,0]+math.log(math.expm1(.9)))
        mu = depth[...,None]*rays
        if self.experiment_role == 'ray_point':
            linear = inverse = identity
        elif self.experiment_role == 'ray_iso':
            logscale = .5*torch.tanh(raw[...,1])
            linear = identity*logscale.exp()[...,None,None]
            inverse = identity*(-logscale).exp()[...,None,None]
        elif self.experiment_role in ('ray_gaussian', 'ray_gaussian_hybrid'):
            # A unique positive-diagonal Cholesky factor of Sigma, equivalent
            # to a GS covariance but with no redundant 6D rotation at a sphere.
            linear = torch.diag_embed((.5*torch.tanh(raw[...,1:4])).exp())
            ij = torch.tril_indices(3,3,offset=-1,device=features.device)
            lower = torch.zeros_like(linear)
            lower[...,ij[0],ij[1]] = .5*torch.tanh(raw[...,4:7])
            linear = linear+lower
            inverse = torch.linalg.solve_triangular(linear, identity, upper=False)
        else:
            generator = .5*torch.tanh(raw[...,1:10]).reshape(b,n,3,3)
            linear, inverse = torch.matrix_exp(generator), torch.matrix_exp(-generator)
        g = torch.zeros(b,n,4,4,dtype=features.dtype,device=features.device)
        gi = torch.zeros_like(g)
        g[...,:3,:3] = linear; g[...,:3,3] = mu; g[...,3,3] = 1
        gi[...,:3,:3] = inverse; gi[...,:3,3] = -(inverse@mu[...,None])[...,0]; gi[...,3,3] = 1
        return g, gi

    def ray_matrices(self, features, rays, camera):
        g, gi = self.gaussian_frame(features, rays)
        c, ci = camera
        return gi@ci[:,None], c[:,None]@g

    def transport(self, value, matrix, pos, operation):
        if self.experiment_role in ('nope', 'mlp', 'hrope', 'ray_gaussian_hybrid'):
            return super().transport(value, matrix, pos, operation)
        # One 4D frame repeated over all 48 channels. Native pixel phases are
        # absent; their inclusion is isolated by the explicit hybrid control.
        return block_apply(value, matrix)

    def encode(self, data, regime='oracle', zero_values=False):
        if self.experiment_role in ('nope','mlp','hrope'):
            return super().encode(data, regime, zero_values)
        if regime != 'oracle':
            raise ValueError('one identical-input regime')
        q = self.input_projection(torch.cat(((data['q']*math.sqrt(128)+.5)-.5,data['qgeom']),-1))
        k = self.input_projection(torch.cat(((data['k']*math.sqrt(128)+.5)-.5,data['kgeom']),-1))
        qp,kp = data['qpos'],data['kpos']
        qr,kr = center_rays(qp,data['qK']),center_rays(kp,data['kK'])
        qc,kc = camera_pose(data['qgeom']),camera_pose(data['kgeom'])
        qc,kc = (qc,torch.linalg.inv(qc)),(kc,torch.linalg.inv(kc))
        for layer in self.layers:
            fq,fk = layer.attention_norm(q),layer.attention_norm(k)
            qq,qk,qv = layer._project(q); kq,kk,kv = layer._project(k)
            qq,qk,kq,kk = [F.normalize(t,dim=-1) for t in (qq,qk,kq,kk)]
            aq,iq = self.ray_matrices(fq,qr,qc); ak,ik = self.ray_matrices(fk,kr,kc)
            qq = self.transport(qq,aq.transpose(-1,-2),qp,'query')
            qk = self.transport(qk,iq,qp,'key'); qv = self.transport(qv,iq,qp,'value')
            kq = self.transport(kq,ak.transpose(-1,-2),kp,'query')
            kk = self.transport(kk,ik,kp,'key'); kv = self.transport(kv,ik,kp,'value')
            sq = _temperature(layer.log_temperature)*(qq@kk.transpose(-1,-2))
            sk = _temperature(layer.log_temperature)*(kq@qk.transpose(-1,-2))
            if zero_values:
                qv,kv = qv*0,kv*0
            dq = self.transport(sq.softmax(-1)@kv,aq,qp,'output')
            dk = self.transport(sk.softmax(-1)@qv,ak,kp,'output')
            def update(x,dx):
                x = x+layer.output(dx.transpose(1,2).reshape(x.shape))
                return x+layer.ffn(layer.ffn_norm(x))
            q,k = update(q,dq),update(k,dk)
        return q,k


def make_model(role):
    # R52 established this zero-parameter PE as a mandatory strong control.
    # Return its frozen implementation without changing state names/arithmetic.
    if role == 'projection_only':
        return CameraFrameMatcher('camera_only')
    return RayFrameMatcher(role)
