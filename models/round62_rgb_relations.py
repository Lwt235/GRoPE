"""R62: learned Gaussian relations in a matched calibrated RGB matcher."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from models.round5_cross_attention import _temperature
from models.round51_visual_alignment import INPUT_FIELDS
from models.round52_camera_frame import CameraFrameMatcher, camera_lift
from models.round53_ray_frame import center_rays
from models.round54_calibrated_plane import CalibratedPlaneMatcher

REFERENCE_ROLES = ('camera_only', 'camera_plane')
NEW_ROLES = ('camera_mlp', 'free_relation', 'predicted_full', 'predicted_point',
             'observed_full', 'observed_point')
ROLES = REFERENCE_ROLES + NEW_ROLES
OBSERVED_FIELDS = ('qobserved_mean', 'qobserved_cov', 'kobserved_mean', 'kobserved_cov')
FLOOR = 1/300
POINT_VARIANCE = .025**2 + FLOOR**2
REVERSE = (0, 4, 5, 6, 1, 2, 3, 8, 7, 10, 9)


def common_frame(mean, covariance, geometry):
    rotation = geometry[:, 0, :9].reshape(-1, 3, 3)
    translation = geometry[:, 0, 9:12]
    return (torch.einsum('bij,bnj->bni', rotation, mean)+translation[:, None],
            rotation[:, None]@covariance@rotation[:, None].transpose(-1, -2))


def gaussian_from_raw(raw, rays, full=True):
    depth = 1/30+F.softplus(raw[..., 0]+math.log(math.expm1(29/30)))
    mean = depth[..., None]*rays
    eye = torch.eye(3, device=raw.device, dtype=raw.dtype)
    if not full:
        covariance = eye.expand(*mean.shape[:-1], 3, 3)*POINT_VARIANCE
    else:
        diagonal = raw.new_tensor([.025, .030, .010])*(2*torch.tanh(raw[..., 1:4]/2)).exp()
        linear = torch.diag_embed(diagonal)
        ij = torch.tril_indices(3, 3, offset=-1, device=raw.device)
        lower = torch.zeros_like(linear)
        lower[..., ij[0], ij[1]] = .03*torch.tanh(raw[..., 4:7])
        linear = linear+lower
        covariance = linear@linear.transpose(-1, -2)+FLOOR**2*eye
    return mean, covariance


def relation_inputs(qmean, qcov, kmean, kcov):
    """Rectangular version of R58's exact11 relations; native frames already aligned."""
    qm, qs, km, ks = [x.double() for x in (qmean, qcov, kmean, kcov)]
    qp, kp = torch.linalg.inv(qs), torch.linalg.inv(ks)
    delta = qm.unsqueeze(-2)-km.unsqueeze(-3)
    distance = torch.linalg.vector_norm(delta, dim=-1)/.2
    mi = torch.einsum('bijd,bide,bije->bij', delta, qp, delta)
    mj = torch.einsum('bijd,bjde,bije->bij', delta, kp, delta)
    ti = torch.einsum('bide,bjed->bij', qp, ks)
    tj = torch.einsum('bide,bjed->bij', qs, kp)
    qspec = .5*torch.linalg.eigvalsh(qs).clamp_min(1e-15).div(.04).log()
    kspec = .5*torch.linalg.eigvalsh(ks).clamp_min(1e-15).div(.04).log()
    batch, nq, nk = distance.shape
    result = torch.cat((distance[..., None], qspec[:, :, None].expand(batch, nq, nk, 3),
                        kspec[:, None].expand(batch, nq, nk, 3),
                        torch.stack((mi, mj, ti, tj), -1).clamp_min(0).log1p()), -1)
    return result.to(qmean.dtype)


def select_inputs(data, role):
    fields = INPUT_FIELDS + (OBSERVED_FIELDS if role.startswith('observed_') else ())
    return {k:data[k] for k in fields}


class RGBRelationMatcher(CameraFrameMatcher):
    def __init__(self, variant):
        if variant not in NEW_ROLES:
            raise ValueError(variant)
        # Identical R55 common draws, state names, camera operators and V path.
        super().__init__('camera_only')
        self.variant = variant
        if variant == 'camera_mlp':
            self.field = nn.Sequential(nn.Linear(96, 16), nn.GELU(), nn.Linear(16, 2))
            nn.init.zeros_(self.field[-1].weight)
            nn.init.zeros_(self.field[-1].bias)
        else:
            self.field = nn.ModuleDict()
            if not variant.startswith('observed_'):
                width = 1 if variant == 'predicted_point' else 7
                self.field['predictor'] = nn.Linear(96, width)
                nn.init.zeros_(self.field['predictor'].weight)
                nn.init.zeros_(self.field['predictor'].bias)
            # Same relation-map initial state for all11-input Gaussian roles,
            # independent of parameter-head presence or width.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(torch.initial_seed()+62000)
                self.field['relation'] = nn.Sequential(nn.Linear(18 if variant == 'free_relation' else 11, 96),
                                                       nn.GELU(), nn.Linear(96, 96, bias=False))

    def geometry(self, features, data, prefix):
        if self.variant.startswith('observed_'):
            mean, covariance = data[prefix+'observed_mean'], data[prefix+'observed_cov']
            if self.variant == 'observed_point':
                covariance = torch.eye(3, device=mean.device, dtype=mean.dtype).expand_as(covariance)*POINT_VARIANCE
        else:
            rays = center_rays(data[prefix+'pos'], data[prefix+'K'])
            mean, covariance = gaussian_from_raw(self.field['predictor'](features), rays,
                                                 self.variant == 'predicted_full')
        return common_frame(mean, covariance, data[prefix+'geom'])

    def relative(self, fq, fk, data):
        if self.variant == 'free_relation':
            a, b = self.field['predictor'](fq), self.field['predictor'](fk)
            a = torch.cat((a, data['qpos']/12), -1)
            b = torch.cat((b, data['kpos']/12), -1)
            batch, nq, _ = a.shape
            nk = b.shape[1]
            pair = torch.cat((a[:, :, None].expand(batch, nq, nk, 9),
                              b[:, None].expand(batch, nq, nk, 9)), -1)
            reverse = torch.cat((pair[..., 9:], pair[..., :9]), -1).transpose(1, 2)
        else:
            qm, qs = self.geometry(fq, data, 'q')
            km, ks = self.geometry(fk, data, 'k')
            pair = relation_inputs(qm, qs, km, ks)
            reverse = pair[..., list(REVERSE)].transpose(1, 2)
        return self.field['relation'](pair), self.field['relation'](reverse)

    def encode(self, data, regime='oracle', zero_values=False, zero_relation=False):
        if regime != 'oracle':
            raise ValueError('one declared input regime')
        # Whitelist excludes labels and privileged geometry from RGB variants.
        data = select_inputs(data, self.variant)
        q = self.input_projection(torch.cat(((data['q']*math.sqrt(128)+.5)-.5, data['qgeom']), -1))
        k = self.input_projection(torch.cat(((data['k']*math.sqrt(128)+.5)-.5, data['kgeom']), -1))
        qc = camera_lift(data['qgeom'], data['qK'])
        kc = camera_lift(data['kgeom'], data['kK'])
        for layer in self.layers:
            fq, fk = layer.attention_norm(q), layer.attention_norm(k)
            qq, qk, qv = layer._project(q)
            kq, kk, kv = layer._project(k)
            qraw, kraw = qq, kq
            qq, qk, kq, kk = [F.normalize(t, dim=-1) for t in (qq, qk, kq, kk)]
            aq, iq = self.matrices(fq, data['qK'], qc)
            ak, ik = self.matrices(fk, data['kK'], kc)
            qp, kp = data['qpos'], data['kpos']
            if self.variant == 'camera_mlp' and not zero_relation:
                qp = qp+4*torch.tanh(self.field(fq)/4)
                kp = kp+4*torch.tanh(self.field(fk)/4)
            qq = self.transport(qq, aq.transpose(-1, -2), qp, 'query')
            qk = self.transport(qk, iq, qp, 'key')
            qv = self.transport(qv, iq, qp, 'value')
            kq = self.transport(kq, ak.transpose(-1, -2), kp, 'query')
            kk = self.transport(kk, ik, kp, 'key')
            kv = self.transport(kv, ik, kp, 'value')
            sq = _temperature(layer.log_temperature)*(qq@kk.transpose(-1, -2))
            sk = _temperature(layer.log_temperature)*(kq@qk.transpose(-1, -2))
            if self.variant != 'camera_mlp' and not zero_relation:
                rq, rk = self.relative(fq, fk, data)
                rq = rq.reshape(*rq.shape[:-1], 2, 48)
                rk = rk.reshape(*rk.shape[:-1], 2, 48)
                sq = sq+torch.einsum('bhid,bijhd->bhij', qraw, rq)/math.sqrt(48)
                sk = sk+torch.einsum('bhid,bijhd->bhij', kraw, rk)/math.sqrt(48)
            if zero_values:
                qv, kv = qv*0, kv*0
            dq = self.transport(sq.softmax(-1)@kv, aq, qp, 'output')
            dk = self.transport(sk.softmax(-1)@qv, ak, kp, 'output')
            def update(x, dx):
                x = x+layer.output(dx.transpose(1, 2).reshape(x.shape))
                return x+layer.ffn(layer.ffn_norm(x))
            q, k = update(q, dq), update(k, dk)
        return q, k


def make_model(role):
    if role == 'camera_only':
        return CameraFrameMatcher(role)
    if role == 'camera_plane':
        return CalibratedPlaneMatcher(role)
    return RGBRelationMatcher(role)
