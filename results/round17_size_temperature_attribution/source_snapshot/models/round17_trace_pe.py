"""Trace-calibrated content temperature for the unchanged Gaussian cell bias.

Only tau = T*h^2/(h^2+tr(C)) is new. C is the *executed* covariance in
reference-8 coordinates (zero for a point control), not shared diagnostic
context. The caller supplies the frozen GL32 Gaussian log masses including
the original native-area floor; this module neither integrates nor adds a
second floor. It introduces no learned parameter or alternative kernel.

The scalar formula is evaluated in float64 and cast to the covariance dtype.
The production content multiply is tau32*content32, followed by float64
addition/softmax. Shape, finite-value, symmetry and 2x2 PSD checks are part of
the runtime and must be included in cost measurements. The PSD check uses no
eigendecomposition. This implementation supports float32/float64 covariance;
positive h must have a finite, nonzero float64 square.

The continuous point-density response bound does not establish a native-cell
argmax, PCK, or regional-quality guarantee.
"""
from __future__ import annotations

import torch


PSD_TOLERANCE = 1.e-12


def _scalar(value, name, device, *, positive):
    value = torch.as_tensor(value, device=device, dtype=torch.float64)
    if value.numel() != 1:
        raise ValueError(f'{name} must be scalar')
    value = value.reshape(())
    if not bool(torch.isfinite(value)) or not bool(value > 0 if positive else value >= 0):
        raise ValueError(f'{name} must be finite and {"positive" if positive else "nonnegative"}')
    return value


def _covariance(covariance):
    if (not isinstance(covariance, torch.Tensor) or covariance.ndim < 2
            or covariance.shape[-2:] != (2, 2) or not covariance.numel()
            or covariance.dtype not in (torch.float32, torch.float64)):
        raise ValueError('covariance must be a nonempty float32/float64 tensor [...,2,2]')
    cov = covariance.to(dtype=torch.float64)
    check = cov.detach()
    if not bool(torch.isfinite(check).all()):
        raise ValueError('covariance must be finite')
    scale = check.abs().amax(dim=(-2, -1)).clamp_min(1.)
    normalized = check / scale[..., None, None]
    if not bool(((normalized - normalized.transpose(-1, -2)).abs() <= PSD_TOLERANCE).all()):
        raise ValueError('covariance must be symmetric')
    # For a symmetric 2x2 matrix, C+eps*scale*I is PSD iff its two diagonal
    # entries and determinant are nonnegative. Scaling prevents overflow.
    a = normalized[..., 0, 0] + PSD_TOLERANCE
    d = normalized[..., 1, 1] + PSD_TOLERANCE
    b = .5 * normalized[..., 0, 1] + .5 * normalized[..., 1, 0]
    if not bool(((a >= 0) & (d >= 0) & (a * d >= b.square())).all()):
        raise ValueError('covariance must be positive semidefinite within tolerance')
    # This scalar sign check also avoids interpreting numerical negative
    # trace as a temperature greater than T. No covariance is repaired.
    if not bool((normalized[..., 0, 0] + normalized[..., 1, 1] >= 0).all()):
        raise ValueError('covariance trace must be nonnegative')
    return cov


def trace_temperature(covariance, bandwidth, temperature):
    """Return tau with shape C.shape[:-2], preserving C's dtype/device/gradients.

    A common positive scale cancels algebraically in the ratio. This avoids
    overflow in tr(C)+h^2 even for finite float64 diagonal entries near the
    dtype limit. There is no clipping, tuned epsilon, or special-case detach
    of the mathematical temperature. At C=0 the ratio is exactly one.
    """
    cov = _covariance(covariance)
    h = _scalar(bandwidth, 'bandwidth', cov.device, positive=True)
    t = _scalar(temperature, 'temperature', cov.device, positive=False)
    h2 = h.square()
    if not bool(torch.isfinite(h2)) or not bool(h2 > 0):
        raise ValueError('bandwidth square must be finite and positive in float64')
    diagonal = cov.diagonal(dim1=-2, dim2=-1)
    scale = torch.maximum(diagonal.amax(-1), h2)
    relative_h2 = h2 / scale
    denominator = relative_h2 + (diagonal / scale[..., None]).sum(-1)
    tau = t * (relative_h2 / denominator)
    result = tau.to(dtype=covariance.dtype)
    if not bool(torch.isfinite(result).all()):
        raise ValueError('temperature is not representable in covariance dtype')
    return result


def content_bias(covariance, bandwidth, temperature, content):
    """Return the exact float32 content bias; no content dot product is added.

    ``content`` has shape C.shape[:-2]+(K,), is float32, and is the original
    shared normalized-feature dot product. The covariance may be float32 or
    float64; tau is rounded to float32 before multiplication in both cases.
    """
    if (not isinstance(content, torch.Tensor) or content.dtype != torch.float32
            or content.ndim < 1 or not content.numel()
            or not isinstance(covariance, torch.Tensor)
            or content.shape[:-1] != covariance.shape[:-2]
            or content.device != covariance.device):
        raise ValueError('content must be float32 [...,K] matching covariance shape/device')
    if not bool(torch.isfinite(content).all()):
        raise ValueError('content must be finite')
    tau32 = trace_temperature(covariance, bandwidth, temperature).to(dtype=torch.float32)
    bias = tau32[..., None] * content
    if not bool(torch.isfinite(bias).all()):
        raise ValueError('float32 content bias overflowed')
    return bias


def probabilities(geometry, content, covariance, bandwidth, temperature):
    """Float64 softmax of unchanged floored Gaussian log mass plus trace bias."""
    if (not isinstance(geometry, torch.Tensor) or geometry.dtype != torch.float64
            or not isinstance(content, torch.Tensor) or geometry.shape != content.shape
            or geometry.device != content.device or not bool(torch.isfinite(geometry).all())):
        raise ValueError('geometry must be finite float64 logits matching content shape/device')
    bias = content_bias(covariance, bandwidth, temperature, content)
    return (geometry + bias.to(dtype=torch.float64)).softmax(dim=-1)
