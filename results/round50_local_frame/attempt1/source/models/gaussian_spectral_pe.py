"""Round 4: Gaussian scale mixtures and symmetric support controls.

The resolvent is E_T[exp(-T v/2)], T~Exponential(mean=1), so it has
the same small-v slope and covariance as the Gaussian characteristic function.
The final content normalization is nonlinear, exactly as in normalized PE.
No trained parameters or feature dimensions are added by these alternatives.
"""
import math

import torch
import torch.nn.functional as F

from models.gaussian_pe import GaussianPEMatcher, normalized_rotary


CONTROLS = ('full', 'qonly', 'konly', 'none', 'trace', 'area', 'shuffled')


def resolvent_rotary(features, mean, covariance, frequency):
    """Norm-preserving rotary with amplitude 1 / (1 + w^T C w / 2).

    Log/content scaling handles tiny content and inactive frequency blocks.
    Double intermediates deliberately match the conservative Gaussian reference.
    A faster implementation is a separate numerical question.
    """
    work = features.double()
    scale = work.abs().amax(-1, keepdim=True)
    unit = work / torch.where(scale > 0, scale, torch.ones_like(scale))
    original_norm = unit.norm(dim=-1, keepdim=True) * scale
    pair = unit.reshape(*unit.shape[:-1], -1, 2)
    active = pair.abs().amax(-1) > 0
    radius = torch.hypot(torch.where(active, pair[..., 0], torch.ones_like(pair[..., 0])),
                         torch.where(active, pair[..., 1], torch.zeros_like(pair[..., 1])))
    frequency = frequency.double()
    variance = torch.einsum('fd,...de,fe->...f', frequency, covariance.double(), frequency)
    logamp = -torch.log1p(variance.clamp_min(0) / 2)
    reference = torch.where(active, logamp, -torch.inf).amax(-1, keepdim=True)
    reference = torch.where(active.any(-1, keepdim=True), reference, 0.)
    logamp = logamp - reference
    logweight = logamp + radius.log()
    shift = torch.where(active, logweight, -torch.inf).amax(-1, keepdim=True)
    shift = torch.where(active.any(-1, keepdim=True), shift, 0.)
    exponent = torch.where(active, logweight - shift,
        (logamp - shift).clamp(max=math.log(torch.finfo(torch.float64).max)))
    base = torch.where(active[..., None], pair / radius[..., None], pair)
    filtered = (base * exponent.exp()[..., None]).flatten(-2)
    norm = filtered.norm(dim=-1, keepdim=True)
    filtered = filtered / torch.where(norm > 0, norm, torch.ones_like(norm))
    phase = mean.double() @ frequency.T
    x, y = filtered.reshape(*filtered.shape[:-1], -1, 2).unbind(-1)
    rotated = torch.stack([x * phase.cos() - y * phase.sin(),
                           x * phase.sin() + y * phase.cos()], -1).flatten(-2)
    return (rotated * original_norm).to(features.dtype)


class GaussianSpectralMatcher(GaussianPEMatcher):
    """Methods 'gaussian_CONTROL', 'resolvent_CONTROL', gaussian{25,50,200}_full.

    Controls apply symmetrically to q and k unless qonly/konly is explicit.
    Means, content, feature count, initialization and trainable parameters match.
    Covariance scale controls multiply physical C before the common blur floor.
    """
    def __init__(self, method, count=256, resolution=.25):
        super().__init__('normalized_shape', count, resolution)
        self.method = method
        self.kernel, self.control = method.split('_')
        if self.control not in CONTROLS:
            raise ValueError(method)
        if self.kernel not in ('gaussian', 'resolvent', 'gaussian25', 'gaussian50', 'gaussian200'):
            raise ValueError(method)

    def positional_inputs(self, data):
        mq, mk = data['pixq'], data['k_to_q']
        eye = torch.eye(2, device=mq.device, dtype=mq.dtype)
        cq = data.get('projected_qcov', eye.expand(*mq.shape[:-1], 2, 2) / 12)
        ck = data['projected_kcov']
        fixed_q, fixed_k = torch.ones_like(cq) * eye / 12, torch.ones_like(ck) * eye / 12
        if self.control in ('konly', 'none'):
            cq = fixed_q
        if self.control in ('qonly', 'none'):
            ck = fixed_k
        if self.control == 'trace':
            cq = eye * (cq.diagonal(dim1=-2, dim2=-1).sum(-1)/2)[..., None, None]
            ck = eye * (ck.diagonal(dim1=-2, dim2=-1).sum(-1)/2)[..., None, None]
        if self.control == 'area':
            cq = eye * torch.linalg.det(cq).clamp_min(1e-12).sqrt()[..., None, None]
            ck = eye * torch.linalg.det(ck).clamp_min(1e-12).sqrt()[..., None, None]
        if self.control == 'shuffled':
            cq, ck = cq.roll(13, dims=-3), ck.roll(13, dims=-3)
        factor = {'gaussian25': .25, 'gaussian50': .5, 'gaussian200': 2.}.get(self.kernel, 1.)
        return mq, cq * factor + eye * self.resolution.square(), mk, ck * factor + eye * self.resolution.square()

    def forward(self, data):
        q = F.normalize(self.encoder(data['q']), dim=-1)
        k = F.normalize(self.encoder(data['k']), dim=-1)
        original_q = q
        mq, cq, mk, ck = self.positional_inputs(data)
        encode = resolvent_rotary if self.kernel == 'resolvent' else normalized_rotary
        q, k = encode(q, mq, cq, self.freq2), encode(k, mk, ck, self.freq2)
        score = self.logtemp.exp().clamp(max=60) * (q @ k.transpose(-1, -2))
        unmatched = self.dustbin + self.dustbin_head(original_q).squeeze(-1)
        return torch.cat([score, unmatched[..., None]], -1)
