"""R29 descriptors only: unchanged R27 inputs plus three analytic-tail cases."""
from copy import deepcopy

import numpy as np

from experiments import round27_rms_fixtures as OLD


OLD_BASE_IDS = OLD.BASE_IDS
ANALYTIC_BASE_IDS = ('27_native432_far_cells', '28_subfloat_fov', '29_visible_far_tiles')
BASE_IDS = OLD_BASE_IDS + ANALYTIC_BASE_IDS
IDENTITY_IDS = OLD.IDENTITY_IDS
rectangles = OLD.rectangles
native_bias = OLD.native_bias


def make_cases():
    """Twenty inherited bases and three prescribed constant-depth descriptors."""
    rows = deepcopy(OLD.make_cases())
    for name, mean, bounds in (
        (ANALYTIC_BASE_IDS[0], [12., 9., 1.], np.asarray(rows[12]['bounds'], np.float64)),
        (ANALYTIC_BASE_IDS[1], [0., 0., 1.], rectangles([9.75, 10., 10.25], [-.25, 0., .25])),
        (ANALYTIC_BASE_IDS[2], [0., 0., 1.], np.asarray([[0., -.25, 12., .25]], np.float64)),
    ):
        rows.append(dict(id=name, kind='base', mu=mean, factor=np.zeros((3, 3), np.float64).tolist(),
            bounds=bounds.tolist(), content_bias=np.zeros(len(bounds), np.float64).tolist(),
            content_temperature=None, bias_kind='zero', policy_overrides={}))
    return rows


def make_identities(base_inputs):
    if [r['id'] for r in base_inputs] != list(BASE_IDS):
        raise ValueError('the exact twenty-three ordered base inputs are required')
    return deepcopy(OLD.make_identities(deepcopy(base_inputs[:len(OLD_BASE_IDS)])))


def fixtures():
    base = make_cases()
    return base + make_identities(base)
