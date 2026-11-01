"""A fixed-width orthogonal completion of Gaussian expected rotary phases.

For each four-dimensional content block (u, v), u and v in R^2, the map is
diag(R(theta), I) @ [[a I, -b I], [b I, a I]], with
a=exp(-omega.T C omega / 2), b=sqrt(1-a^2). This is a classical orthogonal
completion, not a new Gaussian kernel or a density-overlap embedding.

Covariance describes spatial support. At broad support the positional block
switches from u to v rather than discarding information, so the map is not an
anti-aliasing expectation. Geometry is fixed in the current matching probes.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F


METHODS = [
    'unitary_point', 'unitary_fixed', 'unitary_shape',
    'unitary_trace', 'unitary_area', 'unitary_shuffled',
]


def frequencies(count=12):
    """Oblique golden-angle directions spanning the common .1--4 norm range."""
    angle = torch.arange(count, dtype=torch.float32)*math.pi*(3-math.sqrt(5))
    direction = torch.stack([angle.cos(), angle.sin()], -1)
    magnitude = torch.logspace(math.log10(.1), math.log10(4.), count)
    return direction*magnitude[:, None]


def encode(content, mean, covariance, frequency):
    """Apply independent orthogonal 4x4 blocks, retaining the content width.

    ``content`` has 4*len(frequency) channels, ``mean`` has two coordinates,
    and ``covariance`` must be positive semidefinite 2x2 spatial support.
    Negative directional variance due to roundoff is clipped to zero.

    Forward evaluation is defined at zero covariance, but the square-root
    complementary amplitude has a covariance derivative singularity there.
    Strictly positive covariance is needed for finite geometry gradients.
    """
    if content.shape[-1] != 4*frequency.shape[0]:
        raise ValueError('content width must be four times the frequency count')
    blocks = content.reshape(*content.shape[:-1], frequency.shape[0], 4)
    u, v = blocks[..., :2], blocks[..., 2:]
    variance = torch.einsum('fd,...de,fe->...f', frequency, covariance, frequency).clamp_min(0)
    a = (-variance/2).exp()[..., None]
    b = (-torch.expm1(-variance)).sqrt()[..., None]
    first, second = a*u-b*v, b*u+a*v
    angle = mean @ frequency.T
    x, y = first.unbind(-1)
    rotated = torch.stack([x*angle.cos()-y*angle.sin(),
                           x*angle.sin()+y*angle.cos()], -1)
    return torch.cat([rotated, second], -1).flatten(-2)


class GaussianUnitaryMatcher(nn.Module):
    def __init__(self, method, resolution=.25):
        super().__init__()
        if method not in METHODS:
            raise ValueError(method)
        if resolution < 0:
            raise ValueError('resolution must be nonnegative')
        self.method = method
        self.encoder = nn.Sequential(nn.Linear(108, 96), nn.GELU(), nn.Linear(96, 48))
        self.logtemp = nn.Parameter(torch.tensor(2.2))
        self.dustbin = nn.Parameter(torch.tensor(2.))
        self.dustbin_head = nn.Linear(48, 1)
        self.register_buffer('freq2', frequencies())
        self.register_buffer('resolution', torch.tensor(float(resolution)))

    def positional_inputs(self, data):
        mq, mk = data['pixq'], data['k_to_q']
        identity = torch.eye(2, device=mq.device, dtype=mq.dtype)
        if self.method == 'unitary_point':
            # This control has t=0 even when the support-family resolution
            # buffer is positive; otherwise it would not be a point control.
            cq = mq.new_zeros(*mq.shape[:-1], 2, 2)
            ck = mk.new_zeros(*mk.shape[:-1], 2, 2)
            return mq, cq, mk, ck
        cq = data.get('projected_qcov', identity.expand(*mq.shape[:-1], 2, 2)/12)
        if self.method == 'unitary_fixed':
            ck = identity.expand(*mk.shape[:-1], 2, 2)/12
        else:
            ck = data['projected_kcov']
            if self.method == 'unitary_trace':
                variance = ck.diagonal(dim1=-2, dim2=-1).sum(-1)/2
                ck = identity*variance[..., None, None]
            elif self.method == 'unitary_area':
                variance = torch.linalg.det(ck).clamp_min(0).sqrt()
                ck = identity*variance[..., None, None]
            elif self.method == 'unitary_shuffled':
                # Fixed index derangement at the experiment's 48-token grids;
                # does not inspect labels or a cross-view matching identity.
                ck = ck.roll(13, dims=-3)
        # Apply the SAME known query rectangle and physical blur to every
        # support-family method. Only the key-support control changes.
        blur = identity*self.resolution.square()
        return mq, cq+blur, mk, ck+blur

    def forward(self, data):
        q = F.normalize(self.encoder(data['q']), dim=-1)
        k = F.normalize(self.encoder(data['k']), dim=-1)
        original_q = q
        mq, cq, mk, ck = self.positional_inputs(data)
        q = encode(q, mq, cq, self.freq2)
        k = encode(k, mk, ck, self.freq2)
        score = self.logtemp.exp().clamp(max=60)*(q @ k.transpose(-1, -2))
        unmatched = self.dustbin+self.dustbin_head(original_q).squeeze(-1)
        return torch.cat([score, unmatched[..., None]], -1)
