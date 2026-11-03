"""One relative camera x Gaussian-frame operator, on R51's identical inputs."""
import math
import torch
import torch.nn.functional as F
from models.round51_visual_alignment import VisualAlignmentMatcher, INPUT_FIELDS
from models.round50_local_frame import LocalFrameMatcher
from models.round5_cross_attention import _temperature

ROLES = ('nope', 'mlp', 'hrope', 'local_gs', 'camera_only', 'camera_iso',
         'camera_gs', 'camera_affine')
BASE = dict(local_gs='gs_frame', camera_only='frame_identity',
            camera_iso='gs_iso', camera_gs='gs_frame', camera_affine='affine')


def camera_lift(geometry, intrinsics):
    """Reference -> centered image homogeneous chart, retaining camera depth.

    geometry stores camera-to-reference R and origin/3m. All 4D points use
    the same fixed 3m unit. This is invertible because depth is retained;
    it is NOT an inverse of a depth-free 3D-to-2D projection.
    """
    b = len(geometry)
    t = torch.eye(4, dtype=geometry.dtype, device=geometry.device).expand(b, 4, 4).clone()
    t[:, :3, :3] = geometry[:, 0, :9].reshape(b, 3, 3)
    t[:, :3, 3] = geometry[:, 0, 9:12]
    k = torch.eye(4, dtype=geometry.dtype, device=geometry.device).expand(b, 4, 4).clone()
    k[:, :3, :3] = intrinsics
    # Measured rotations may contain rounding error; no unrecorded SO(3) repair.
    return k @ torch.linalg.inv(t), t @ torch.linalg.inv(k)


class CameraFrameMatcher(VisualAlignmentMatcher):
    def __init__(self, role):
        if role not in ROLES:
            raise ValueError(role)
        super().__init__(BASE.get(role, role))
        self.experiment_role = role

    def matrices(self, features, intrinsics, camera=None):
        if self.experiment_role not in BASE:
            return super().matrices(features, intrinsics)
        inverse, gaussian = LocalFrameMatcher.matrices(self, features, intrinsics)
        if self.experiment_role == 'local_gs':
            return gaussian, inverse
        if camera is None:
            raise ValueError('camera matrices required')
        p, pi = camera
        return p[:, None] @ gaussian, inverse @ pi[:, None]

    def encode(self, data, regime='oracle', zero_values=False):
        if self.experiment_role not in BASE:
            return super().encode(data, regime, zero_values)
        if regime != 'oracle':
            raise ValueError('one identical-input R52 regime')
        # Preserve R51's exact common input arithmetic and initialization.
        q = self.input_projection(torch.cat(((data['q']*math.sqrt(128)+.5)-.5, data['qgeom']), -1))
        k = self.input_projection(torch.cat(((data['k']*math.sqrt(128)+.5)-.5, data['kgeom']), -1))
        qp, kp = data['qpos'], data['kpos']
        qc = kc = None
        if self.experiment_role != 'local_gs':
            qc = camera_lift(data['qgeom'], data['qK'])
            kc = camera_lift(data['kgeom'], data['kK'])
        for layer in self.layers:
            fq, fk = layer.attention_norm(q), layer.attention_norm(k)
            qq, qk, qv = layer._project(q)
            kq, kk, kv = layer._project(k)
            qq, qk, kq, kk = [F.normalize(t, dim=-1) for t in (qq, qk, kq, kk)]
            aq, iq = self.matrices(fq, data['qK'], qc)
            ak, ik = self.matrices(fk, data['kK'], kc)
            qq = self.transport(qq, aq.transpose(-1, -2), qp, 'query')
            qk = self.transport(qk, iq, qp, 'key')
            qv = self.transport(qv, iq, qp, 'value')
            kq = self.transport(kq, ak.transpose(-1, -2), kp, 'query')
            kk = self.transport(kk, ik, kp, 'key')
            kv = self.transport(kv, ik, kp, 'value')
            sq = _temperature(layer.log_temperature)*(qq @ kk.transpose(-1, -2))
            sk = _temperature(layer.log_temperature)*(kq @ qk.transpose(-1, -2))
            if zero_values:
                qv, kv = qv*0, kv*0
            dq = self.transport(sq.softmax(-1) @ kv, aq, qp, 'output')
            dk = self.transport(sk.softmax(-1) @ qv, ak, kp, 'output')
            def update(x, dx):
                x = x+layer.output(dx.transpose(1, 2).reshape(x.shape))
                return x+layer.ffn(layer.ffn_norm(x))
            q, k = update(q, dq), update(k, dk)
        return q, k
