"""Fixed R26 synthetic inputs only; no integration, files or empirical arrays."""
from copy import deepcopy

import numpy as np


BASE_IDS = (
    '01_point', '02_constant_axis', '03_constant_correlated', '04_zero_depth',
    '05_negative_constant', '06_radial_central', '07_radial_noncentral',
    '08_full_central', '09_full_noncentral', '10_negative_mean',
    '11_near_constant', '12_tiny_front', '13_native432', '14_rank2',
    '15_constant_axis_bias', '16_radial_central_bias', '17_full_noncentral_bias',
    '18_tiny_front_bias')
IDENTITY_IDS = ('19_world_scale', '20_translation', '21_screen_units',
                '22_axis_rotation', '23_key_permutation', '24_unequal_merge')


def rectangles(xedges, yedges):
    return np.asarray([[x0, y0, x1, y1]
        for y0, y1 in zip(yedges[:-1], yedges[1:])
        for x0, x1 in zip(xedges[:-1], xedges[1:])], np.float64)


def native_bias(bounds):
    centers = (bounds[:, :2] + bounds[:, 2:]) / 2
    return .6 * np.sin(.7 * centers[:, 0] - .2 * centers[:, 1])


def make_cases():
    """Eighteen base calls: fourteen geometries plus four fixed-bias repeats."""
    default = rectangles([-2., -1., 0., 1., 2.], [-2., -1., 0., 1., 2.])
    unequal = rectangles([-2., -.5, 2.], [-2., 1., 2.])
    native = np.asarray([[x, y, x + 8, y + 8]
        for y in range(0, 144, 8) for x in range(0, 192, 8)], np.float64)
    native = (native - .5) / 8.
    radial = [[.6, 0., 0.], [-.4, 0., 0.], [1., 0., 0.]]
    correlated = [[.8, 0., 0.], [.3, .5, 0.], [0., 0., 0.]]
    data = [
        ([0., 0., 2.], np.zeros((3, 3)), default),
        ([.3, -.4, 2.], np.diag([.8, .5, 0.]), default),
        ([.3, -.4, 2.], correlated, default),
        ([.4, -.2, 0.], np.diag([.8, .5, 0.]), unequal),
        ([.4, -.2, -1.], np.diag([.8, .5, 0.]), unequal),
        ([0., 0., 0.], radial, default),
        ([1.2, -.8, 2.], radial, default),
        ([0., 0., 0.], [[1., 0., 0.], [.4, .7, 0.], [.3, -.2, 1.]], default),
        ([.5, -.7, 1.2], [[.7, 0., 0.], [.3, .5, 0.], [.2, -.25, .6]], default),
        ([.5, -.4, -1.], [[.8, 0., 0.], [-.2, .6, 0.], [.3, -.2, 1.]], default),
        ([.3, -.4, 2.], [[.8, 0., 0.], [.3, .5, 0.], [0., 0., 2.**-36]], default),
        ([0., 0., -12.], np.diag([0., 0., 1.]), default),
        ([12., 9., 1.], [[.8, 0., 0.], [.2, .7, 0.], [.15, -.1, .3]], native),
        ([.5, -.4, 1.], [[1., 0., 0.], [.2, .6, 0.], [.3, -.2, 0.]], default),
    ]
    rows = []
    for index, (mean, factor, bounds) in enumerate(data):
        use_bias = index == 3
        rows.append(dict(id=BASE_IDS[index], kind='base',
            mu=np.asarray(mean, np.float64).tolist(), factor=np.asarray(factor, np.float64).tolist(),
            bounds=bounds.tolist(), content_bias=(native_bias(bounds) if use_bias else np.zeros(len(bounds))).tolist(),
            content_temperature=None, bias_kind='fixed_native' if use_bias else 'zero', policy_overrides={}))
    for name, source in zip(BASE_IDS[14:], (1, 5, 8, 11)):
        row = deepcopy(rows[source])
        row.update(id=name, bias_kind='fixed_native',
            content_bias=native_bias(np.asarray(row['bounds'], np.float64)).tolist())
        rows.append(row)
    return rows


def make_identities(base_inputs):
    if [r['id'] for r in base_inputs] != list(BASE_IDS):
        raise ValueError('the exact eighteen ordered base inputs are required')
    parents = {r['id']: r for r in base_inputs}
    parent = parents['17_full_noncentral_bias']
    rows = []

    def identity(name, base, relation='equal', **extra):
        row = deepcopy(base)
        row.update(id=name, kind='identity', identity=dict(parent=base['id'], relation=relation, **extra))
        return row

    row = identity(IDENTITY_IDS[0], parent)
    row['mu'] = (np.asarray(row['mu']) * 7.).tolist()
    row['factor'] = (np.asarray(row['factor']) * 7.).tolist()
    rows.append(row)

    row = identity(IDENTITY_IDS[1], parent)
    shift = np.asarray([.75, -.5])
    transform = np.eye(3); transform[:2, 2] = shift
    row['mu'] = (transform @ np.asarray(row['mu'])).tolist()
    row['factor'] = (transform @ np.asarray(row['factor'])).tolist()
    row['bounds'] = (np.asarray(row['bounds']) + np.tile(shift, 2)).tolist()
    rows.append(row)

    row = identity(IDENTITY_IDS[2], parent)
    scale = 3.7; transform = np.diag([scale, scale, 1.])
    row['mu'] = (transform @ np.asarray(row['mu'])).tolist()
    row['factor'] = (transform @ np.asarray(row['factor'])).tolist()
    row['bounds'] = (np.asarray(row['bounds']) * scale).tolist()
    row['policy_overrides'] = dict(h=.25 * scale, floor_density=1e-30 / scale**2)
    rows.append(row)

    row = identity(IDENTITY_IDS[3], parent)
    transform = np.asarray([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    row['mu'] = (transform @ np.asarray(row['mu'])).tolist()
    row['factor'] = (transform @ np.asarray(row['factor'])).tolist()
    b = np.asarray(row['bounds'])
    row['bounds'] = np.column_stack((-b[:, 3], b[:, 0], -b[:, 1], b[:, 2])).tolist()
    rows.append(row)

    native = parents['13_native432']; permutation = list(range(431, -1, -1))
    row = identity(IDENTITY_IDS[4], native, 'permutation', permutation=permutation)
    row['bounds'] = [row['bounds'][i] for i in permutation]
    row['content_bias'] = [row['content_bias'][i] for i in permutation]
    rows.append(row)

    base = parents['09_full_noncentral']
    groups = [g for y in range(4) for g in ([4*y, 4*y+1, 4*y+2], [4*y+3])]
    row = identity(IDENTITY_IDS[5], base, 'merge', groups=groups)
    b = np.asarray(base['bounds'])
    row['bounds'] = [np.r_[b[g, :2].min(0), b[g, 2:].max(0)].tolist() for g in groups]
    row['content_bias'] = [0.] * len(groups)
    rows.append(row)
    return rows


def fixtures():
    base = make_cases()
    return base + make_identities(base)
