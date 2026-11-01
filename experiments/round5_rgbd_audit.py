"""Fixed first-10-pair, training-room-only SUN3D input sanity audit (no fitting)."""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from experiments import round5_rgbd_data as data


def rgb_array(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float64) / 255


def bilinear_rgb(image, pixels):
    """Diagnostic RGB sampling only; sensor depth is never interpolated."""
    h, w = image.shape[:2]
    xy = np.clip(pixels, [0, 0], [w-1, h-1])
    lower = np.floor(xy).astype(int)
    upper = np.minimum(lower + 1, [w-1, h-1])
    fraction = xy-lower
    x0, y0 = lower.T
    x1, y1 = upper.T
    fx, fy = fraction.T
    return ((1-fx)[:, None]*(1-fy)[:, None]*image[y0, x0]
            + fx[:, None]*(1-fy)[:, None]*image[y0, x1]
            + (1-fx)[:, None]*fy[:, None]*image[y1, x0]
            + fx[:, None]*fy[:, None]*image[y1, x1])


def summary(values):
    values = np.asarray(values)
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "minimum": float(values.min()), "maximum": float(values.max())}


def audit(data_root, output):
    scene = "harvard_c3/hv_c3_1"
    frames, selection = data.load_sun3d_scene(data_root, scene)
    pairs = selection["pairs"][:10]  # Fixed order; no quality or method-score selection.
    original_indices = sorted(set(i for pair in pairs for i in pair))
    rgb, depths, frame_rows = {}, {}, []
    for index in original_indices:
        frame = frames[index]
        rgb[index] = rgb_array(frame.rgb_path)
        depths[index] = data.load_sensor_depth(frame.depth_path)
        if rgb[index].shape[:2] != depths[index].shape:
            raise ValueError("aligned RGB/depth dimensions differ")
        depth = depths[index]
        known = np.isfinite(depth) & (depth > 0)
        frame_rows.append({"original_index": index, "frame": frame.frame_id,
            "rgb_shape": list(rgb[index].shape), "depth_shape": list(depth.shape),
            "depth_positive_fraction": float(known.mean()),
            "depth_quantiles_m": dict(zip(["min", "p01", "p05", "p50", "p95", "p99", "max"],
                map(float, np.percentile(depth[known], [0, 1, 5, 50, 95, 99, 100])))),
            "rgb_sha256": data.sha256(frame.rgb_path),
            "depth_sha256": data.sha256(frame.depth_path)})

    pair_rows = []
    projected_photo, unwarped_photo = [], []
    same_frame_rows = []
    grid = data.token_grid()
    for qid in sorted(set(q for q, _ in pairs)):
        q = frames[qid]
        labels = data.pair_supervision(depths[qid], depths[qid], q.K_native,
                                       q.K_native, q.c2w, q.c2w)
        qnative = data.resize_pixel_coordinates(grid, data.DEFAULT_IMAGE_HW, depths[qid].shape)
        _, known = data.sample_nearest_depth(depths[qid], qnative)
        same_frame_rows.append({"query": qid, "query_known_count": int(known.sum()),
            "same_frame_visible_count": int(labels.gt_valid.sum()),
            "validity_identical_to_query_sensor_known": bool(np.array_equal(known, labels.gt_valid)),
            "max_resize_pixel_error": float(np.max(np.abs(labels.pixcontinuous[known]-grid[known]))),
            "status_counts": labels.metadata["status_counts"]})

    for qid, kid in pairs:
        q, k = frames[qid], frames[kid]
        labels = data.pair_supervision(depths[qid], depths[kid], q.K_native,
                                       k.K_native, q.c2w, k.c2w)
        visible = labels.gt_valid
        qnative = data.resize_pixel_coordinates(grid, data.DEFAULT_IMAGE_HW, depths[qid].shape)
        knative = data.resize_pixel_coordinates(labels.pixcontinuous[visible],
                                               data.DEFAULT_IMAGE_HW, depths[kid].shape)
        colorq = bilinear_rgb(rgb[qid], qnative[visible])
        colork = bilinear_rgb(rgb[kid], knative)
        unwarped_native = data.resize_pixel_coordinates(grid[visible], data.DEFAULT_IMAGE_HW,
                                                        depths[kid].shape)
        color_unwarped = bilinear_rgb(rgb[kid], unwarped_native)
        photo = np.abs(colorq-colork).mean(axis=1)
        unwarped = np.abs(colorq-color_unwarped).mean(axis=1)
        projected_photo.extend(photo.tolist())
        unwarped_photo.extend(unwarped.tolist())

        # Algebraic roundtrip through both supplied C2W poses, using original rays/depth.
        qz, known = data.sample_nearest_depth(depths[qid], qnative)
        rays = np.column_stack([qnative[known], np.ones(known.sum())]) @ np.linalg.inv(q.K_native).T
        qpoints = rays*qz[known, None]
        world = qpoints@q.c2w[:3, :3].T + q.c2w[:3, 3]
        keycamera = (np.column_stack([world, np.ones(len(world))]) @ np.linalg.inv(k.c2w).T)[:, :3]
        world_again = keycamera@k.c2w[:3, :3].T + k.c2w[:3, 3]
        query_again = (np.column_stack([world_again, np.ones(len(world_again))]) @ np.linalg.inv(q.c2w).T)[:, :3]
        native_again = data.project_points_and_jacobian(query_again, q.K_native)[0]
        relative = np.linalg.inv(k.c2w)@q.c2w
        angle = np.degrees(np.arccos(np.clip((np.trace(relative[:3, :3])-1)/2, -1, 1)))
        pair_rows.append({"query": qid, "key": kid,
            "relative_translation_norm_m": float(np.linalg.norm(relative[:3, 3])),
            "relative_rotation_degrees": float(angle),
            "roundtrip_max_native_pixel_error": float(np.abs(native_again-qnative[known]).max()),
            "roundtrip_max_world_error_m": float(np.abs(world_again-world).max()),
            "photometric_reprojected_rgb_l1": float(photo.mean()) if len(photo) else None,
            "photometric_unwarped_rgb_l1": float(unwarped.mean()) if len(unwarped) else None,
            **labels.metadata})

    all_poses = data.read_sun3d_extrinsics(Path(data_root)/selection["pose_file"])
    rotations = all_poses[:, :3, :3]
    determinants = np.linalg.det(rotations)
    orthogonal_error = np.linalg.norm(rotations.transpose(0, 2, 1)@rotations-np.eye(3), axis=(1, 2))
    result = {"scope": "training-only; first 10 pair entries in frozen selection; no fitting/selection",
        "scene": scene, "selected_pairs": pairs, "test_sensor_depth_or_RGB_read": False,
        "method_scores_read": False, "cpu_threads": 1,
        "selection_sha256": data.sha256(Path(data_root)/"selection.json"),
        "loader_sha256": data.sha256(data.__file__), "audit_source_sha256": data.sha256(__file__),
        "official_pre_alignment_source": "https://github.com/PrincetonVision/SUN3Dsfm#our-depth-file-format",
        "camera_information": "provided RGB-D reconstruction C2W shared by all methods; not strict whole-system RGB-only",
        "pose_summary": {"count": len(all_poses), "rotation_determinant": summary(determinants),
            "rotation_orthogonality_frobenius": summary(orthogonal_error),
            "translation_norm_m": summary(np.linalg.norm(all_poses[:, :3, 3], axis=1)),
            "nonidentity_pose_count": int(np.any(np.abs(all_poses-np.eye(4))>1e-8, axis=(1, 2)).sum())},
        "frames": frame_rows, "same_frame": same_frame_rows, "pairs": pair_rows,
        "photometric_diagnostic": {
            "sampling": "native RGB bilinear, visible GT rows only, RGB [0,1], no brightness compensation",
            "reprojected_rgb_l1": float(np.mean(projected_photo)),
            "unwarped_rgb_l1": float(np.mean(unwarped_photo)),
            "pair_count_reprojection_improves": int(sum(r["photometric_reprojected_rgb_l1"] < r["photometric_unwarped_rgb_l1"] for r in pair_rows)),
            "use": "sanity diagnostic only; not a model-selection or pair-exclusion rule",
        },
        "limitations": ["roundtrip is algebraic and cannot establish pose accuracy or depth registration alone",
            "photometric residual includes lighting, occlusion edges, RGB/depth asynchrony and pose/measurement errors",
            "dataset pose is a reconstruction estimate, sensor-supervised correspondence is approximate",
            "no manual contact-sheet inspection was performed by this numerical audit"],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"output": str(output), "same_frame": same_frame_rows,
        "visible_fractions": [row["gt_valid_fraction"] for row in pair_rows],
        "photometric_diagnostic": result["photometric_diagnostic"]}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/round5_sun3d"))
    parser.add_argument("--output", type=Path, default=Path("results/round5_inputs/training_geometry_audit.json"))
    options = parser.parse_args()
    audit(options.data_root, options.output)
