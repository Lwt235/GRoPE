"""Metadata-only R23 applicability population; never loads archive arrays."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from experiments.round22_ray_contract import POLICY as R22_POLICY, SOURCE_FILES as R22_SOURCES

ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = ROOT / "results/round20_anchor_fixed/lock.json"
SCENES = ("harvard_c3/hv_c3_1", "harvard_c5/hv_c5_1", "harvard_c6/hv_c6_1")
INPUT_KEYS = ("q_points3", "q_pixel_cov3", "q_K", "k_K", "q_to_k_transform", "q_predicted_depth")
IMAGE_HW = (144, 192)
QUERY_SIZES = (8, 16, 24)
POLICY = deepcopy(R22_POLICY)
GATES = dict(input_normalized_absolute=1e-10, normalization_absolute=2e-12,
             direct_fov_additivity_absolute=1e-8, amplified_analytic_tail=1e-8,
             production_max_phase_wall_seconds=600.0)
BINDINGS = (
    "results/round22_ray_root_acceptance_20261031.json",
    "results/round22_ray_preflight/lock.json",
    "results/round22_ray_independent_audit/verification.json",
    "results/round22_ray_post_result_decision_20261031.json",
    "results/round20_anchor_fixed/lock.json",
    "results/round20_anchor_root_acceptance_20261031.json",
)
SOURCE_FILES = tuple(R22_SOURCES) + (
    "ROUND23_RAY_APPLICABILITY_PROTOCOL.md",
    "experiments/round23_ray_contract.py",
    "experiments/round23_ray_applicability.py",
    "experiments/round23_ray_applicability_audit.py",
    "tests/test_round23_ray_applicability.py",
    "tests/test_round23_ray_applicability_audit.py",
)


def entries():
    old = json.loads(METADATA_PATH.read_text())["entries"]
    result = []
    fields = ("scene", "pair", "pair_name", "input_path", "input_sha256", "manifest_path", "manifest_sha256")
    for scene in SCENES:
        choices = [r for r in old if r["scene"] == scene]
        row = choices[0]
        if row["pair"] != [30, 45] or row["pair_name"] != "000_0000030_0000045":
            raise ValueError("unexpected first metadata pair")
        expected = ROOT / "results/round10_partition_data/train" / scene.replace("/", "__") / row["pair_name"] / "inputs.npz"
        if Path(row["input_path"]).resolve() != expected.resolve():
            raise ValueError("unexpected source archive location")
        result.append({k: row[k] for k in fields})
    return result


def pixel_bounds(size):
    height, width = IMAGE_HW
    if type(size) is not int or size <= 0 or height % size or width % size:
        raise ValueError("size must divide the fixed image")
    return np.asarray([[x, y, x+size, y+size]
                       for y in range(0, height, size)
                       for x in range(0, width, size)], dtype=np.int64)


def key_bounds():
    return (pixel_bounds(8).astype(np.float64)-0.5)/8.0


def query_specs():
    rows = []
    for entry in entries():
        slug = entry["scene"].replace("/", "__")
        for size in QUERY_SIZES:
            bounds = pixel_bounds(size)
            indices = np.linspace(0, len(bounds)-1, 8).astype(np.int64)
            for index in indices:
                rows.append(dict(id=f"{slug}__g{size}__q{int(index):04d}",
                                 scene=entry["scene"], pair=entry["pair"],
                                 pair_name=entry["pair_name"], query_size=size,
                                 query_index=int(index), query_bounds=bounds[index].tolist()))
    if len(rows) != 72 or len({r["id"] for r in rows}) != 72:
        raise ValueError("fixed unique 72-query population required")
    return rows
