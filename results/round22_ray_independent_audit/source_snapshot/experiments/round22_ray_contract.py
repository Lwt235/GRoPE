"""Prospective, synthetic-only R22 inputs and numerical policy; no integrator."""
from __future__ import annotations

from copy import deepcopy

import numpy as np


POLICY = dict(
    h=0.25, floor_density=1e-30, depth_extent=14.0,
    outer_epsabs=1e-12, outer_epsrel=1e-10, outer_limit=512,
    max_depth_evaluations=32768, max_breakpoints=256,
    inner_epsabs=1e-13, inner_epsrel=1e-11, inner_limit=128,
    max_inner_evaluations=4000000,
    geometry_sigma_offsets=[-8.0, 0.0, 8.0],
    density_breakpoints=[-8.0, -4.0, 0.0, 4.0, 8.0],
)
REFERENCE_POLICY = dict(
    orders=[64, 96], blur_extent=14.0, panel_width_h_fraction=0.5,
    max_panels=128, max_total_cell_node_evaluations=2000000,
    max_phase_wall_seconds=120.0,
)
GATES = dict(
    production_probability_absolute=1e-4,
    production_raw_fov_relative=1e-6,
    production_raw_l1_over_reference_fov=1e-6,
    reference_probability_absolute=1e-8,
    reference_raw_fov_relative=1e-8,
    reference_raw_l1_over_reference_fov=1e-8,
    normalization_absolute=2e-12,
    amplified_analytic_tail=1e-8,
    integrated_identity_absolute=1e-8,
    direct_fov_additivity_absolute=1e-8,
    fixed_point_limit_absolute=1e-8,
    exact_zero_absolute=0.0,
    production_max_phase_wall_seconds=120.0,
)
SOURCE_FILES = (
    "ROUND22_RAY_PREFLIGHT_PROTOCOL.md",
    "experiments/round22_ray_contract.py",
    "models/round22_ray_pushforward.py",
    "experiments/round22_ray_preflight.py",
    "experiments/round22_ray_reference.py",
    "experiments/round22_ray_audit.py",
    "tests/test_round22_ray_pushforward.py",
    "tests/test_round22_ray_reference.py",
    "docs/ray_pushforward_pe_feasibility_draft.md",
    "docs/ray_pushforward_pe_adversarial_review_draft.md",
)


def unit_bounds():
    return [[float(x), float(y), float(x+1), float(y+1)]
            for y in range(-2, 2) for x in range(-2, 2)]


def fixtures():
    """Fifteen fully specified base inputs, with independent-reference tags."""
    rows = []

    def add(name, kind, mu, factor, **extra):
        rows.append(dict(id=name, kind=kind, mu=mu, factor=factor,
                         bounds=unit_bounds(), H=[[1.0, 0.0], [0.0, 1.0]],
                         offset=[0.0, 0.0], **extra))

    empty = [[], [], []]
    add("point_front", "point", [0.2, -0.3, 1.0], deepcopy(empty))
    add("point_behind", "point", [0.2, -0.3, -1.0], deepcopy(empty))
    add("point_camera_origin", "point", [0.0, 0.0, 0.0], deepcopy(empty))
    add("point_outside_fov", "point", [3.0, 0.0, 1.0], deepcopy(empty))
    add("plane_constant_diagonal", "constant_depth", [0.2, -0.3, 2.0],
        [[1.0, 0.0], [0.0, 0.6], [0.0, 0.0]])
    add("plane_constant_correlated", "constant_depth", [0.2, -0.3, 2.0],
        [[1.0, 0.3], [0.2, 0.7], [0.0, 0.0]])
    k = np.asarray([0.2, -0.3, 1.0], dtype=np.float64)
    for name, m in (("radial_front", 2.0), ("radial_behind", -2.0),
                    ("radial_below_floor", -12.0)):
        add(name, "radial", (k*m).tolist(), k[:, None].tolist(),
            reference_parameters=dict(k=[0.2, -0.3], depth_mean=m, depth_sd=1.0))
    for beta in (32.0, 512.0):
        m, z_star, u_star = 1.0, 1.3, 0.2
        add("offset_rank1_beta"+str(int(beta)), "offset_rank1",
            [beta*(m-z_star)+u_star*z_star, 0.0, m],
            [[beta], [0.0], [1.0]], reference_parameters=dict(
                beta=beta, z_star=z_star, u_star=u_star,
                depth_mean=m, depth_sd=1.0))
    add("plane_through_origin", "origin_plane", [0.0, 0.0, 0.0],
        [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    add("isotropic_origin", "origin_isotropic", [0.0, 0.0, 0.0],
        np.eye(3).tolist())
    add("correlated_origin", "origin_correlated", [0.0, 0.0, 0.0],
        [[1.0, 0.0, 0.0], [0.4, 0.7, 0.0], [0.3, -0.2, 1.0]])
    add("plane_tilted", "tilted_plane", [0.0, 0.0, 1.0],
        [[1.0, 0.0], [0.0, 0.5], [0.5, 0.0]],
        reference_parameters=dict(z0=1.0, slope=0.5,
                                  s_mean=0.0, s_sd=1.0, v_mean=0.0, v_sd=0.5))
    assert len(rows) == 15
    return rows


def identity_fixtures():
    """Eight prescribed transforms/limits, not extra efficacy candidates."""
    base = {r["id"]: r for r in fixtures()}
    rows = []

    def copy(name, parent, relation):
        row = deepcopy(base[parent])
        row.update(id=name, identity=dict(parent=parent, relation=relation))
        rows.append(row)
        return row

    row = copy("identity_world_scale", "offset_rank1_beta32", "equal")
    row["mu"] = (np.asarray(row["mu"])*3.7).tolist()
    row["factor"] = (np.asarray(row["factor"])*3.7).tolist()
    row = copy("identity_origin_scale", "isotropic_origin", "equal")
    row["factor"] = (np.asarray(row["factor"])*1e-6).tolist()
    row = copy("identity_screen_units", "isotropic_origin", "equal")
    row["H"] = (np.eye(2)*3.7).tolist()
    row["bounds"] = (np.asarray(row["bounds"])*3.7).tolist()
    row["policy_overrides"] = dict(h=0.25*3.7, floor_density=1e-30/(3.7**2))
    row = copy("identity_screen_translation", "isotropic_origin", "equal")
    row["offset"] = [1.25, -0.75]
    row["bounds"] = (np.asarray(row["bounds"])+[1.25, -0.75, 1.25, -0.75]).tolist()
    row = copy("identity_key_permutation", "isotropic_origin", "permutation")
    permutation = list(range(15, -1, -1))
    row["bounds"] = [row["bounds"][i] for i in permutation]
    row["identity"]["permutation"] = permutation
    row = copy("identity_unequal_merge", "isotropic_origin", "merge")
    row["bounds"] = [[-2.0, -2.0, -1.0, 0.0], [-1.0, -2.0, 2.0, 0.0],
                     [-2.0, 0.0, -1.0, 2.0], [-1.0, 0.0, 2.0, 2.0]]
    row["identity"]["groups"] = [[0, 4], [1, 2, 3, 5, 6, 7],
                                 [8, 12], [9, 10, 11, 13, 14, 15]]
    row = copy("identity_factor_rotation", "plane_constant_correlated", "equal")
    row["factor"] = (np.asarray(row["factor"]) @ np.asarray([[0.0, -1.0], [1.0, 0.0]])).tolist()
    row = copy("identity_fixed_point_limit", "point_front", "point_limit")
    row["factor"] = (np.eye(3)*1e-6).tolist()
    assert len(rows) == 8
    return rows
