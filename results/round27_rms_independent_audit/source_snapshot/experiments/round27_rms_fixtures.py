"""R27 descriptors: preserve all frozen R26 inputs, add two wide-FOV bases."""
from copy import deepcopy

import numpy as np

from experiments import round26_rms_fixtures as OLD


BASE_IDS = OLD.BASE_IDS + ('25_wide_constant_axis', '26_wide_constant_correlated')
IDENTITY_IDS = OLD.IDENTITY_IDS
rectangles = OLD.rectangles
native_bias = OLD.native_bias


def make_cases():
    """Twenty base descriptors; no integration or empirical-array loading."""
    rows = deepcopy(OLD.make_cases())
    axis = rectangles([-4., 0., 4.], [-4., 0., 4.])
    correlated = rectangles([-4., -.3, 4.], [-4., 1.2, 4.])
    for name, mean, factor, bounds, biased in (
        (BASE_IDS[-2], [0., 0., 1.], np.zeros((3, 3)), axis, False),
        (BASE_IDS[-1], [.21, -.13, 1.7],
         [[.15, 0., 0.], [.36, .05, 0.], [0., 0., 0.]], correlated, True),
    ):
        rows.append(dict(id=name, kind='base', mu=np.asarray(mean, np.float64).tolist(),
            factor=np.asarray(factor, np.float64).tolist(), bounds=bounds.tolist(),
            content_bias=(native_bias(bounds) if biased else np.zeros(len(bounds))).tolist(),
            content_temperature=None, bias_kind='fixed_native' if biased else 'zero',
            policy_overrides={}))
    return rows


def make_identities(base_inputs):
    if [r['id'] for r in base_inputs] != list(BASE_IDS):
        raise ValueError('the exact twenty ordered base inputs are required')
    return deepcopy(OLD.make_identities(deepcopy(base_inputs[:len(OLD.BASE_IDS)])))


def fixtures():
    base = make_cases()
    return base + make_identities(base)
