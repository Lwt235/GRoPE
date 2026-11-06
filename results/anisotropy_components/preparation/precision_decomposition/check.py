"""Independent arithmetic checks; no training/data/model imports."""
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def terms(cov):
    precision = np.linalg.inv(cov)
    r, h = np.trace(cov) / 3, np.trace(precision) / 3
    return precision, r, h, precision - h * np.eye(3), cov - r * np.eye(3)


def main():
    rng = np.random.default_rng(20261107)
    rotations = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            rotation = np.eye(3)[:, perm] @ np.diag(signs)
            if np.linalg.det(rotation) > 0:
                rotations.append(rotation)
    assert len(rotations) == 24
    errors = {name: [] for name in (
        'metric_identity', 'shape_identity', 'metric_trace_difference',
        'shape_trace_difference', 'orthogonal_direction_mean',
        'proper_rotation_mean', 'spherical_residual')}
    factors = []
    for _ in range(96):
        covariances = []
        for _side in range(2):
            axes, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            eigenvalues = np.exp(rng.uniform(np.log(.1), np.log(10), size=3))
            covariances.append(axes @ np.diag(eigenvalues) @ axes.T)
        delta = rng.normal(size=3)
        d2 = delta @ delta
        for ci, cj in (covariances, covariances[::-1]):
            pi, ri, hi, bi, _ = terms(ci)
            _, rj, _, _, cj0 = terms(cj)
            metric, shape = delta @ pi @ delta, np.trace(pi @ cj)
            errors['metric_identity'].append(abs(metric - (hi * d2 + delta @ bi @ delta)))
            errors['shape_identity'].append(abs(shape - (3 * hi * rj + np.trace(bi @ cj0))))
            errors['metric_trace_difference'].append(abs(
                metric - d2 / ri - ((hi - 1 / ri) * d2 + delta @ bi @ delta)))
            errors['shape_trace_difference'].append(abs(
                shape - 3 * rj / ri - (3 * rj * (hi - 1 / ri) + np.trace(bi @ cj0))))
            basis, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            directional_mean = np.mean([d2 * u @ pi @ u for u in basis.T])
            shape_mean = np.mean([np.trace(pi @ rot @ cj @ rot.T) for rot in rotations])
            errors['orthogonal_direction_mean'].append(abs(directional_mean - hi * d2))
            errors['proper_rotation_mean'].append(abs(shape_mean - 3 * hi * rj))
            factors.append(float(ri * hi))
        radius = float(np.exp(rng.uniform(-2, 2)))
        _, r, h, b, c = terms(radius * np.eye(3))
        errors['spherical_residual'].append(float(max(abs(r * h - 1), np.max(abs(b)), np.max(abs(c)))))
    maxima = {name: float(max(values)) for name, values in errors.items()}
    assert max(maxima.values()) < 1e-10, maxima
    assert min(factors) >= 1 - 1e-12
    ci, cj = np.diag([.25, 1., 4.]), np.eye(3)
    pi, ri, hi, _, _ = terms(ci)
    counterexample = {
        'r_i': float(ri), 'h_i': float(hi), 'p_i': float(ri * hi),
        'shape_full': float(np.trace(pi @ cj)), 'shape_trace': float(3 / ri),
        'metric_axis_values': np.diag(pi).tolist(),
        'metric_trace_each_axis': float(1 / ri),
    }
    assert counterexample['shape_full'] == 5.25
    result = {
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'algebra_only_no_model_or_dataset_access',
        'seed': 20261107, 'dtype': 'float64', 'spd_pairs': 96,
        'ordered_pairs': 192, 'eigenvalue_range': [.1, 10.],
        'proper_rotations': len(rotations), 'max_absolute_errors': maxima,
        'minimum_precision_factor': min(factors), 'counterexample': counterexample,
        'new_trained_models': 0, 'new_data_scores': 0, 'passed': True,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    Path(__file__).with_name('check.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
