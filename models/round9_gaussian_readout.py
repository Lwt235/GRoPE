"""Read observed RGB using a projected Gaussian and five positional channels.

Pixel centres are (x, y) = (column, row), including centres 0 and W-1/H-1.
All pixels participate: chunk sizes only bound memory, not Gaussian support.
The common pixel-cell covariance is I/12 and bandwidth is in pixel units.
This module has no learned parameters, renderer, or target-image input.
"""
from typing import NamedTuple

import torch

from models.gaussian_quadratic_pe import quadratic_features


METHODS = ("full", "isotropic", "fixed", "point")


class ReadoutResult(NamedTuple):
    rgb: torch.Tensor
    valid: torch.Tensor


def _image_geometry(image, means, key_mask, query_valid, compute_dtype):
    if image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2]) < 1:
        raise ValueError("image must have shape [H,W,3] with nonempty spatial axes")
    if means.ndim != 2 or means.shape[-1] != 2:
        raise ValueError("means must have shape [N,2]")
    if not image.is_floating_point() or not means.is_floating_point():
        raise ValueError("image and means must be floating-point tensors")
    if image.device != means.device:
        raise ValueError("image and means must be on the same device")
    if not torch.isfinite(image).all() or not torch.isfinite(means).all():
        raise ValueError("image and means must be finite, including masked positions")
    dtype = compute_dtype or torch.promote_types(image.dtype, means.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("compute_dtype must be torch.float32 or torch.float64")
    height, width, _ = image.shape
    count = len(means)
    if key_mask is not None:
        if key_mask.dtype != torch.bool or key_mask.device != image.device:
            raise ValueError("key_mask must be Boolean and on the image device")
        if tuple(key_mask.shape) == (height, width):
            key_mask = key_mask.reshape(1, height * width)
        elif tuple(key_mask.shape) == (count, height, width):
            key_mask = key_mask.reshape(count, height * width)
        else:
            raise ValueError("key_mask must have shape [H,W] or [N,H,W]")
    if query_valid is None:
        query_valid = torch.ones(count, dtype=torch.bool, device=image.device)
    elif query_valid.shape != (count,) or query_valid.dtype != torch.bool or query_valid.device != image.device:
        raise ValueError("query_valid must be Boolean [N] on the image device")
    return image.to(dtype), means.to(dtype), key_mask, query_valid


def _controlled_covariance(covariances, means, method, fixed_variance):
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if covariances.shape != (len(means), 2, 2) or covariances.device != means.device:
        raise ValueError("covariances must have shape [N,2,2] on the means device")
    if not covariances.is_floating_point() or not torch.isfinite(covariances).all():
        raise ValueError("covariances must be finite floating-point tensors")
    cov = covariances.to(means.dtype)
    if not torch.allclose(cov, cov.transpose(-1, -2), rtol=1e-5, atol=1e-7):
        raise ValueError("covariances must be symmetric")
    cov = (cov + cov.transpose(-1, -2)) / 2
    # A projected, flattened Gaussian may be singular before the shared pixel
    # and bandwidth terms. PSD is therefore allowed; indefinite inputs are not.
    if len(cov) and (torch.linalg.eigvalsh(cov) < 0).any():
        raise ValueError("covariances must be positive semidefinite")
    eye = torch.eye(2, dtype=means.dtype, device=means.device)
    if method == "isotropic":
        return cov.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None, None] * eye
    if method == "fixed":
        variance = torch.as_tensor(fixed_variance, dtype=means.dtype, device=means.device)
        if variance.ndim or not torch.isfinite(variance) or variance <= 0:
            raise ValueError("fixed_variance must be a positive finite scalar")
        return variance * eye.expand(len(means), 2, 2)
    if method == "point":
        return torch.zeros_like(cov)
    return cov


def gaussian_readout(image, means, covariances, *, method="full", bandwidth=0.5,
                     fixed_variance=1.0, key_mask=None, query_valid=None,
                     query_chunk_size=256, key_chunk_size=16384,
                     compute_dtype=None):
    """Return normalized all-pixel Gaussian RGB averages and row validity.

    ``image`` is source RGB [H,W,3], ``means`` [N,2], and ``covariances``
    [N,2,2], expressed in source-image pixel coordinates. Masks are optional
    Boolean [H,W] or [N,H,W] source-key masks and Boolean [N] query visibility.
    All methods use covariance C_query + I/12 + bandwidth**2 I. Point sets
    C_query=0; isotropic preserves its trace; fixed uses fixed_variance I.

    Invisible queries and rows with no allowed pixels yield RGB=0, valid=False.
    The caller owns any fallback and must not interpret that zero as evidence.
    Gaussian means may lie outside the image (the finite image truncates and
    renormalizes the kernel); callers can mark them invisible if appropriate.
    Gradients propagate through source RGB, means, covariance and bandwidth.
    Float64 is recommended for numerical equivalence audits of narrow kernels.
    """
    if not isinstance(query_chunk_size, int) or query_chunk_size < 1 or not isinstance(key_chunk_size, int) or key_chunk_size < 1:
        raise ValueError("chunk sizes must be positive integers")
    if compute_dtype is None and covariances.is_floating_point():
        compute_dtype = torch.promote_types(torch.promote_types(image.dtype, means.dtype), covariances.dtype)
    image, means, mask, visible = _image_geometry(image, means, key_mask, query_valid, compute_dtype)
    cov = _controlled_covariance(covariances, means, method, fixed_variance)
    h = torch.as_tensor(bandwidth, dtype=means.dtype, device=means.device)
    if h.ndim or not torch.isfinite(h) or h <= 0:
        raise ValueError("bandwidth must be a positive finite scalar")
    height, width, _ = image.shape
    y, x = torch.meshgrid(torch.arange(height, dtype=means.dtype, device=means.device),
                          torch.arange(width, dtype=means.dtype, device=means.device), indexing="ij")
    pixels = torch.stack((x, y), -1).reshape(-1, 2)
    common_cov = torch.eye(2, dtype=means.dtype, device=means.device) / 12
    q_features, k_features = quadratic_features(means, cov, pixels, common_cov, h)
    values = image.reshape(-1, 3)
    chunks, validity = [], []
    for qs in range(0, len(means), query_chunk_size):
        qe = min(qs + query_chunk_size, len(means))
        q = q_features[qs:qe]
        numerator = means.new_zeros((qe - qs, 3))
        denominator = means.new_zeros(qe - qs)
        maximum = means.new_full((qe - qs,), -torch.inf)
        qmask = None if mask is None else (mask if len(mask) == 1 else mask[qs:qe])
        for ks in range(0, len(pixels), key_chunk_size):
            ke = min(ks + key_chunk_size, len(pixels))
            logits = q @ k_features[ks:ke].T
            allowed = visible[qs:qe, None]
            if qmask is not None:
                allowed = allowed & qmask[:, ks:ke]
            logits = logits.masked_fill(~allowed, -torch.inf)
            new_maximum = torch.maximum(maximum, logits.max(-1).values)
            safe_maximum = torch.where(torch.isfinite(new_maximum), new_maximum, 0)
            prior_scale = torch.exp(maximum - safe_maximum)
            weights = torch.exp(logits - safe_maximum[:, None])
            numerator = numerator * prior_scale[:, None] + weights @ values[ks:ke]
            denominator = denominator * prior_scale + weights.sum(-1)
            maximum = new_maximum
        valid = denominator > 0
        # An empty row divides by one and retains its exact zero numerator.
        rgb = numerator / torch.where(valid, denominator, 1)[:, None]
        chunks.append(rgb)
        validity.append(valid)
    if not chunks:
        return ReadoutResult(image.new_empty((0, 3)), visible)
    return ReadoutResult(torch.cat(chunks), torch.cat(validity))


def bilinear_readout(image, means, *, key_mask=None, query_valid=None,
                     compute_dtype=None):
    """Read four neighboring pixel centres with identical visibility/mask API.

    Masked neighbors are omitted and remaining bilinear weights renormalized.
    Centers outside [0,W-1] x [0,H-1] have valid=False (no border extrapolation).
    At image borders duplicate clipped neighbors are harmless: their fractional
    weights still sum to one. An unsupported row yields a safe exact zero RGB.
    """
    image, means, mask, visible = _image_geometry(image, means, key_mask, query_valid, compute_dtype)
    height, width, _ = image.shape
    x, y = means.unbind(-1)
    inside = (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    x0, y0 = x.floor(), y.floor()
    dx, dy = x - x0, y - y0
    weights = torch.stack(((1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy), -1)
    xx = torch.stack((x0, x0+1, x0, x0+1), -1).clamp(0, width-1).long()
    yy = torch.stack((y0, y0, y0+1, y0+1), -1).clamp(0, height-1).long()
    indices = yy * width + xx
    allowed = (inside & visible)[:, None].expand(-1, 4)
    if mask is not None:
        if len(mask) == 1:
            allowed = allowed & mask[0, indices]
        else:
            allowed = allowed & mask.gather(1, indices)
    weights = weights * allowed
    denominator = weights.sum(-1)
    valid = denominator > 0
    rgb = (image.reshape(-1, 3)[indices] * weights[..., None]).sum(1)
    rgb = rgb / torch.where(valid, denominator, 1)[:, None]
    return ReadoutResult(rgb, valid)
