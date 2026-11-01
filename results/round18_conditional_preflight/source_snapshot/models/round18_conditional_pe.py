"""Synthetic prototype of Gaussian geometry-conditioned local attention.

This is an integral operator, not a claimed finite-dimensional Q/K embedding.
Gauss-Hermite order is an explicit numerical approximation. No real-data
efficacy, runtime advantage, or convergence is implied by this implementation.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr, roots_hermitenorm


def validate(mean, covariance, bounds, content, h, temperature, floor_density):
    mean = np.asarray(mean, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    bounds = np.asarray(bounds, dtype=np.float64)
    content = np.asarray(content, dtype=np.float64)
    if mean.shape != (2,) or covariance.shape != (2, 2):
        raise ValueError('expected one 2D Gaussian')
    if bounds.ndim != 2 or bounds.shape[1] != 4 or not len(bounds):
        raise ValueError('expected nonempty [xmin,ymin,xmax,ymax] rectangles')
    if content.shape != (len(bounds),):
        raise ValueError('one content score per key is required')
    if not all(np.isfinite(v).all() for v in (mean, covariance, bounds, content)):
        raise ValueError('all inputs must be finite')
    if not np.array_equal(covariance, covariance.T):
        raise ValueError('covariance must be symmetric; no repair is applied')
    if np.linalg.eigvalsh(covariance).min() < 0:
        raise ValueError('covariance must be positive semidefinite')
    if np.any(bounds[:, 2:] <= bounds[:, :2]):
        raise ValueError('rectangles must have positive area')
    overlap = (np.minimum(bounds[:, None, 2:], bounds[None, :, 2:])
               > np.maximum(bounds[:, None, :2], bounds[None, :, :2])).all(axis=-1)
    np.fill_diagonal(overlap, False)
    if overlap.any():
        raise ValueError('key rectangles must have disjoint interiors')
    if np.max(np.abs(content)) > 1:
        raise ValueError('declared cosine-score domain is [-1,1]')
    if not np.isfinite(h) or h <= 0 or not np.isfinite(temperature) or temperature < 0:
        raise ValueError('finite positive h and finite nonnegative temperature required')
    if not np.isfinite(floor_density) or floor_density <= 0:
        raise ValueError('positive finite area-floor density required')
    return mean, covariance, bounds, content


def interval_mass(low, high):
    """Mirror a positive interval into the left tail to avoid 1 minus 1."""
    right = low > 0
    return np.where(right, ndtr(-low) - ndtr(-high), ndtr(high) - ndtr(low))


def rectangle_masses(centers, bounds, h, floor_density=1e-30):
    low = (bounds[None, :, :2] - centers[:, None, :]) / h
    high = (bounds[None, :, 2:] - centers[:, None, :]) / h
    mass = interval_mass(low, high).prod(axis=-1)
    area = (bounds[:, 2:] - bounds[:, :2]).prod(axis=-1)
    return mass + floor_density * area[None, :]


def integrate(mean, covariance, bounds, content, h, temperature, *, order=64,
              floor_density=1e-30, chunk=2048):
    mean, covariance, bounds, content = validate(
        mean, covariance, bounds, content, h, temperature, floor_density)
    if not isinstance(order, int) or order < 2 or not isinstance(chunk, int) or chunk < 1:
        raise ValueError('positive integer integration order/chunk required')
    z, w = roots_hermitenorm(order)
    w = w / np.sqrt(2 * np.pi)
    standard = np.stack(np.meshgrid(z, z, indexing='ij'), axis=-1).reshape(-1, 2)
    weights = np.multiply.outer(w, w).reshape(-1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    factor = eigenvectors * np.sqrt(eigenvalues)[None, :]
    centers = mean + standard @ factor.T
    exponent = np.exp(temperature * (content - content.max()))
    numerator = np.zeros(len(bounds), dtype=np.float64)
    geometry_mass = np.zeros_like(numerator)
    bare = np.zeros_like(numerator)
    for start in range(0, len(weights), chunk):
        mass = rectangle_masses(centers[start:start + chunk], bounds, h, floor_density)
        total = mass.sum(axis=-1)
        unnormalized = mass * exponent[None, :]
        local = unnormalized / unnormalized.sum(axis=-1, keepdims=True)
        weight = weights[start:start + chunk]
        numerator += np.einsum('n,n,nk->k', weight, total, local)
        geometry_mass += np.einsum('n,nk->k', weight, mass)
        bare += np.einsum('n,nk->k', weight, local)
    z_geometry = geometry_mass.sum()
    old = geometry_mass * exponent
    return dict(probability=numerator / z_geometry, z_geometry=z_geometry,
        numerator=numerator, geometry_mass=geometry_mass,
        old_mix_then_normalize=old / old.sum(),
        bare_expectation=bare / weights.sum(), quadrature_mass=weights.sum(),
        quadrature_nodes=len(weights), order=order)
