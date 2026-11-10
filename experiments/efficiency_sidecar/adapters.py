"""Private-instance native adapters. No global patches or new learned weights.

Fixed geometry only: these adapters are not certified for learned geometry.
FP64 geometry and the existing FP32 network are retained.
"""
from types import MethodType

import torch
from torch.utils.checkpoint import checkpoint

from models.factored_relation_runtime import carry_projection, native_block
from models.round69_compact_relation import descriptors
from models.region_shape_attribution import fixed_inputs, fixed_hook


def quadratic_descriptors(centers, covariance):
    """Evaluate directed quadratic forms with a rank-10 polynomial product.

    Delta-based Euclidean distances are intentionally retained: the square root
    is sensitive to cancellation near coincident centers. Recenter only the
    polynomial expansion; retain the original covariance, inverse and floors.
    """
    c, s = centers.double(), covariance.double()
    p = torch.linalg.inv(s)
    delta = c.unsqueeze(-2) - c.unsqueeze(-3)
    d2 = delta.square().sum(-1)
    del delta
    x = c - c.mean(-2, keepdim=True)
    px = (p @ x.unsqueeze(-1)).squeeze(-1)
    alpha = (x * px).sum(-1, keepdim=True)
    beta = -((p + p.transpose(-1, -2)) @ x.unsqueeze(-1)).squeeze(-1)
    diagonal = p.diagonal(dim1=-2, dim2=-1)
    off = torch.stack((p[..., 0, 1] + p[..., 1, 0],
                       p[..., 0, 2] + p[..., 2, 0],
                       p[..., 1, 2] + p[..., 2, 1]), -1)
    left = torch.cat((alpha, beta, diagonal, off), -1)
    cross = torch.stack((x[..., 0] * x[..., 1], x[..., 0] * x[..., 2],
                         x[..., 1] * x[..., 2]), -1)
    right = torch.cat((torch.ones_like(alpha), x, x.square(), cross), -1)
    metric = left @ right.transpose(-1, -2)
    # The original quadratic form is exactly zero for identical centers.
    metric = metric.masked_fill(d2 == 0, 0)
    trace = p.flatten(-2) @ s.transpose(-1, -2).flatten(-2).transpose(-1, -2)
    terms = torch.stack((metric, metric.transpose(-1, -2),
                         trace, trace.transpose(-1, -2)), -1).clamp_min(0).log1p()
    return torch.cat(((d2.sqrt() / .2).unsqueeze(-1), terms), -1).to(centers)


def radial_descriptors(centers):
    """Same FP64 distance -> input dtype -> radial operations as the old hook."""
    c = centers.double()
    distance = ((c.unsqueeze(-2) - c.unsqueeze(-3)).square().sum(-1).sqrt() / .2).to(centers)
    # fixed_inputs reads only column zero; expand is a view, not a large copy.
    return fixed_inputs(distance.unsqueeze(-1).expand(*distance.shape, 5))


def position_forward(self, centers, covariance):
    if centers.requires_grad or covariance.requires_grad:
        raise ValueError('Sidecar adapters are validated only for fixed geometry.')
    role = self._efficiency_role
    if role == 'fixed_isotropic5' and self._radial_shortcut:
        value = radial_descriptors(centers)
    elif role == 'distance_mlp':
        value = self.descriptors(centers, covariance)
    else:
        value = (quadratic_descriptors(centers, covariance) if self._quadratic else
                 descriptors('relative5', centers, covariance))
        if role == 'fixed_isotropic5':
            value = fixed_inputs(value)

    def hidden(v):
        return self.mlp[1](self.mlp[0](v))

    # Only recompute the cheap first projection/GELU, not all four blocks.
    z = (checkpoint(hidden, value, use_reentrant=False, preserve_rng_state=False)
         if self._recompute and torch.is_grad_enabled() else hidden(value))
    return self.mlp[2](z)


def adapt(model, *, factored=True, quadratic=False, recompute=False, radial_shortcut=False):
    """Call on a copy. State-dict names/values and the backbone stay unchanged."""
    if getattr(model, '_efficiency_sidecar', False) or getattr(model, '_factored_relation_runtime', False):
        raise ValueError('Use a fresh private instance, not an already adapted model.')
    if model.role not in ('relative5', 'distance_mlp', 'fixed_isotropic5'):
        raise ValueError(model.role)
    position = model.position
    hooks = list(position.mlp._forward_pre_hooks.values())
    expected = [fixed_hook] if model.role == 'fixed_isotropic5' else []
    if hooks != expected or position.mlp._forward_hooks:
        raise ValueError('Unexpected MLP hooks: refusing to bypass them.')
    if position.mlp[-1].bias is not None or len(position.mlp) != 3:
        raise ValueError('Expected Linear/GELU/bias-free Linear.')
    position._efficiency_role = model.role
    position._quadratic = quadratic
    position._recompute = recompute
    position._radial_shortcut = radial_shortcut
    position.forward = MethodType(position_forward, position)
    if factored:
        position.mlp[-1].forward = MethodType(carry_projection, position.mlp[-1])
        for layer in model.layers:
            layer.forward = MethodType(native_block, layer)
    model._efficiency_sidecar = True
    return model
