"""SUN3D sparse reader and supervision-only aligned RGB-D correspondence math.

Public model observations contain RGB, K, C2W and a complete image grid; they
never contain sensor depth or a GT validity mask.  ``pair_supervision`` is a
separate interface.  No RGB geometry predictor is imported here.

Coordinates are zero-based pixel centers.  Resize uses
``x' = sx * (x + .5) - .5``; an 8-pixel cell is centered at 3.5, 11.5, ... .
Native sensor depth is sampled by nearest pixel (ties toward +infinity, with a
1e-9 native-pixel numerical guard), never interpolated across a depth edge.
Labels are conditional on GT-visible matches,
not an occlusion/rejection task.  Missing measurements are unknown.

Official format references (not assumptions inferred from RGB/depth filenames):
https://github.com/PrincetonVision/SUN3Dsfm#our-depth-file-format
https://github.com/PrincetonVision/SUN3Dsfm/blob/master/depthRead.m
https://github.com/PrincetonVision/SUN3Dsfm/blob/master/depth2XYZcamera.m
https://github.com/PrincetonVision/SUN3Dsfm/blob/master/loadSUN3D.m
https://github.com/PrincetonVision/SUN3Dsfm/blob/master/outputCameraExtrinsics.m
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image


VALID = 0
QUERY_DEPTH_MISSING = 1
BEHIND_CAMERA = 2
OUTSIDE_IMAGE = 3
KEY_DEPTH_MISSING = 4
DEPTH_INCONSISTENT = 5
NONFINITE_PROJECTION = 6
STATUS_NAMES = {
    VALID: "visible", QUERY_DEPTH_MISSING: "query_depth_missing",
    BEHIND_CAMERA: "behind_camera", OUTSIDE_IMAGE: "outside_image",
    KEY_DEPTH_MISSING: "key_depth_missing",
    DEPTH_INCONSISTENT: "depth_inconsistent",
    NONFINITE_PROJECTION: "nonfinite_projection",
}
DEFAULT_IMAGE_HW = (144, 192)
DEFAULT_STRIDE = 8
NEAREST_TIE_EPSILON = 1e-9


def _hw(shape: tuple[int, int]) -> tuple[int, int]:
    h, w = map(int, shape)
    if h <= 0 or w <= 0:
        raise ValueError("image dimensions must be positive")
    return h, w


def _K(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("K must be a finite 3x3 matrix")
    if abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("K must be invertible")
    if not np.allclose(matrix[2], [0, 0, 1], atol=1e-10, rtol=0):
        raise ValueError("K must have last row [0,0,1]")
    return matrix


def _pose(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("C2W must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-10, rtol=0):
        raise ValueError("C2W must be affine homogeneous")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3, rtol=0):
        raise ValueError("C2W rotation is not approximately orthogonal")
    if np.linalg.det(rotation) <= 0:
        raise ValueError("C2W rotation must preserve orientation")
    return matrix


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_sun3d_depth(encoded: np.ndarray) -> np.ndarray:
    """Undo SUN3D's circular 3-bit shift, then convert integer mm to meters."""
    encoded = np.asarray(encoded)
    if encoded.ndim != 2 or encoded.dtype.kind not in "ui":
        raise ValueError("encoded SUN3D depth must be a 2D integer array")
    if np.any(encoded < 0) or np.any(encoded > 65535):
        raise ValueError("SUN3D depth values must fit uint16")
    bits = encoded.astype(np.uint32)
    millimeters = ((bits >> 3) | (bits << 13)) & 65535
    return millimeters.astype(np.float32) / np.float32(1000)


def load_sensor_depth(path: str | Path) -> np.ndarray:
    """Read a native SUN3D depth image. This is a supervision-only operation."""
    with Image.open(path) as image:
        encoded = np.asarray(image)
    return decode_sun3d_depth(encoded)


def read_sun3d_intrinsics(path: str | Path) -> np.ndarray:
    """Read SUN3D K and convert MATLAB's one-based centers to zero-based."""
    matrix = _K(np.loadtxt(path).reshape(3, 3)).copy()
    matrix[:2, 2] -= 1
    return matrix


def read_sun3d_extrinsics(path: str | Path) -> np.ndarray:
    """Read consecutive 3x4 C2W matrices in the FULL RGB listing order."""
    rows = np.loadtxt(path, ndmin=2)
    if rows.shape[1] != 4 or rows.shape[0] % 3:
        raise ValueError("SUN3D extrinsics must contain three 4-column rows per RGB")
    poses = np.broadcast_to(np.eye(4), (len(rows) // 3, 4, 4)).copy()
    poses[:, :3] = rows.reshape(-1, 3, 4)
    for pose in poses:
        _pose(pose)
    return poses


def resize_intrinsics(K_native: np.ndarray, source_hw: tuple[int, int],
                      target_hw: tuple[int, int] = DEFAULT_IMAGE_HW) -> np.ndarray:
    """Return K for a half-pixel resize of zero-based native pixel centers."""
    sh, sw = _hw(source_hw)
    th, tw = _hw(target_hw)
    sx, sy = tw / sw, th / sh
    transform = np.array([[sx, 0, (sx - 1) / 2],
                          [0, sy, (sy - 1) / 2], [0, 0, 1]])
    return transform @ _K(K_native)


def resize_pixel_coordinates(pixels: np.ndarray, source_hw: tuple[int, int],
                             target_hw: tuple[int, int]) -> np.ndarray:
    sh, sw = _hw(source_hw)
    th, tw = _hw(target_hw)
    return (np.asarray(pixels, dtype=np.float64) + .5) * [tw / sw, th / sh] - .5


def token_grid(image_hw: tuple[int, int] = DEFAULT_IMAGE_HW,
               stride: int = DEFAULT_STRIDE) -> np.ndarray:
    h, w = _hw(image_hw)
    if stride <= 0 or h % stride or w % stride:
        raise ValueError("image dimensions must be divisible by positive stride")
    x = (np.arange(w // stride) + .5) * stride - .5
    y = (np.arange(h // stride) + .5) * stride - .5
    xx, yy = np.meshgrid(x, y)
    return np.stack([xx, yy], axis=-1).reshape(-1, 2)


def sample_nearest_depth(depth: np.ndarray, pixels_native: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Nearest native sensor samples; outside/holes/nonpositive values are unknown.

    A 1e-9 native-pixel guard makes half-integer ties stable to double precision
    camera roundtrips. Without it, identity reprojection can pick an adjacent
    pixel across a depth edge and incorrectly reject the original observation.
    """
    depth = np.asarray(depth)
    if depth.ndim != 2:
        raise ValueError("sensor depth must be a 2D meters array")
    pixels = np.asarray(pixels_native, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("pixels must have shape [N,2]")
    h, w = depth.shape
    finite = np.isfinite(pixels).all(axis=1)
    safe = np.where(np.isfinite(pixels), pixels, 0)
    inside = finite & (safe[:, 0] >= -.5) & (safe[:, 0] < w - .5)
    inside &= (safe[:, 1] >= -.5) & (safe[:, 1] < h - .5)
    # Clip only the read address; clipped out-of-image samples are still invalid.
    ix = np.floor(np.clip(safe[:, 0], -.5, w - .5) + .5 + NEAREST_TIE_EPSILON).astype(np.int64)
    iy = np.floor(np.clip(safe[:, 1], -.5, h - .5) + .5 + NEAREST_TIE_EPSILON).astype(np.int64)
    values = depth[np.clip(iy, 0, h - 1), np.clip(ix, 0, w - 1)].astype(np.float64)
    known = inside & np.isfinite(values) & (values > 0)
    return np.where(known, values, np.nan), known


def project_points_and_jacobian(points_camera: np.ndarray, K: np.ndarray
                                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pinhole projection and analytic J=d(pixel)/d(X,Y,Z), in pixel units.

    Invalid/behind-camera rows have NaN pixels and zero Jacobian.  This utility
    does not add a covariance floor or divide by the token stride.
    """
    points = np.asarray(points_camera, dtype=np.float64)
    if points.shape[-1] != 3:
        raise ValueError("camera points must end in dimension 3")
    matrix = _K(K)
    homogeneous = points @ matrix.T
    valid = np.isfinite(points).all(axis=-1) & (points[..., 2] > 0)
    denominator = np.where(valid, homogeneous[..., 2], 1)
    pixels = homogeneous[..., :2] / denominator[..., None]
    jacobian = (matrix[:2] * denominator[..., None, None]
                - homogeneous[..., :2, None] * matrix[2]) / denominator[..., None, None]**2
    return (np.where(valid[..., None], pixels, np.nan),
            np.where(valid[..., None, None], jacobian, 0), valid)


def project_covariance(points_camera: np.ndarray, covariance_camera: np.ndarray,
                       K: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linearize C in camera coordinates: C_pixel=J C_camera J^T, no floor."""
    pixels, jacobian, valid = project_points_and_jacobian(points_camera, K)
    covariance = np.asarray(covariance_camera, dtype=np.float64)
    if covariance.shape[-2:] != (3, 3) or not np.isfinite(covariance).all():
        raise ValueError("camera covariance must be finite [...,3,3]")
    projected = jacobian @ covariance @ np.swapaxes(jacobian, -1, -2)
    return pixels, projected, valid


def bilinear_token_labels(pixels: np.ndarray, valid: np.ndarray,
                          image_hw: tuple[int, int] = DEFAULT_IMAGE_HW,
                          stride: int = DEFAULT_STRIDE) -> tuple[np.ndarray, np.ndarray]:
    """At most 4 grid neighbors per row; outside the center hull clamp to edge.

    Pixels are required to be inside image support.  At the outer half-cell,
    clamping prevents dropping otherwise visible edge correspondences. Invalid
    rows have indices -1 and weights 0; these fields are supervision only.
    """
    token_grid(image_hw, stride)  # Validate the image/stride contract.
    h, w = _hw(image_hw)
    pixels = np.asarray(pixels, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if pixels.shape != (len(valid), 2):
        raise ValueError("pixels and valid rows disagree")
    if np.any(valid & (~np.isfinite(pixels).all(axis=1))):
        raise ValueError("valid target coordinates must be finite")
    if np.any(valid & ((pixels[:, 0] < -.5) | (pixels[:, 0] >= w - .5)
                       | (pixels[:, 1] < -.5) | (pixels[:, 1] >= h - .5))):
        raise ValueError("valid targets must be within image support")
    gh, gw = h // stride, w // stride
    grid = (np.where(np.isfinite(pixels), pixels, 0) + .5) / stride - .5
    grid = np.clip(grid, [0, 0], [gw - 1, gh - 1])
    lower = np.floor(grid).astype(np.int64)
    upper = np.minimum(lower + 1, [gw - 1, gh - 1])
    fraction = grid - lower
    x0, y0 = lower.T
    x1, y1 = upper.T
    fx, fy = fraction.T
    indices = np.stack([y0 * gw + x0, y0 * gw + x1,
                        y1 * gw + x0, y1 * gw + x1], axis=1).astype(np.int32)
    weights = np.stack([(1-fx)*(1-fy), fx*(1-fy), (1-fx)*fy, fx*fy], axis=1)
    return np.where(valid[:, None], indices, -1), np.where(valid[:, None], weights, 0).astype(np.float32)


@dataclass(frozen=True)
class PairSupervision:
    gt_valid: np.ndarray
    pixcontinuous: np.ndarray
    key_indices: np.ndarray
    key_weights: np.ndarray
    status: np.ndarray
    metadata: dict[str, Any]

    def dense_labels(self) -> np.ndarray:
        """Transient convenience for training/tests; exported labels stay sparse."""
        size = int(self.metadata["token_count"])
        dense = np.zeros((len(self.gt_valid), size), dtype=np.float32)
        rows = np.repeat(np.arange(len(self.gt_valid)), 4)
        keys = self.key_indices.ravel()
        keep = keys >= 0
        np.add.at(dense, (rows[keep], keys[keep]), self.key_weights.ravel()[keep])
        return dense


def pair_supervision(query_depth_native: np.ndarray, key_depth_native: np.ndarray,
                     K_query_native: np.ndarray, K_key_native: np.ndarray,
                     c2w_query: np.ndarray, c2w_key: np.ndarray,
                     image_hw: tuple[int, int] = DEFAULT_IMAGE_HW,
                     stride: int = DEFAULT_STRIDE,
                     absolute_tolerance: float = .05,
                     relative_tolerance: float = .05) -> PairSupervision:
    """Project query cell centers and require |z_pred-z_sensor| <= .05+.05*z_sensor.

    Both K arguments are native-resolution ZERO-BASED. C2W translations and
    sensor depths must be meters.  All rows survive in the returned arrays;
    ``gt_valid`` is a loss/evaluation selector, never a forward/key mask.
    Depth mismatch is not claimed to be a proven occlusion (pose errors also
    cause it).  No RGB-estimated depth or its quality participates in validity.
    """
    if absolute_tolerance < 0 or relative_tolerance < 0:
        raise ValueError("depth tolerances must be nonnegative")
    if not np.isfinite([absolute_tolerance, relative_tolerance]).all():
        raise ValueError("depth tolerances must be finite")
    qdepth, kdepth = np.asarray(query_depth_native), np.asarray(key_depth_native)
    if qdepth.ndim != 2 or kdepth.ndim != 2:
        raise ValueError("sensor depths must be 2D meters arrays")
    Kq, Kk = _K(K_query_native), _K(K_key_native)
    relative = np.linalg.inv(_pose(c2w_key)) @ _pose(c2w_query)
    centers = token_grid(image_hw, stride)
    qnative = resize_pixel_coordinates(centers, image_hw, qdepth.shape)
    qz, qknown = sample_nearest_depth(qdepth, qnative)
    rays = np.column_stack([qnative, np.ones(len(centers))]) @ np.linalg.inv(Kq).T
    qpoints = rays * np.where(qknown, qz, 0)[:, None]
    kpoints = qpoints @ relative[:3, :3].T + relative[:3, 3]
    knative, _, project_valid = project_points_and_jacobian(kpoints, Kk)
    pix = resize_pixel_coordinates(knative, kdepth.shape, image_hw)
    h, w = _hw(image_hw)
    finite = np.isfinite(pix).all(axis=1)
    inside = finite & (pix[:, 0] >= -.5) & (pix[:, 0] < w - .5)
    inside &= (pix[:, 1] >= -.5) & (pix[:, 1] < h - .5)
    kz, kknown = sample_nearest_depth(kdepth, knative)
    consistent = np.abs(kpoints[:, 2] - kz) <= absolute_tolerance + relative_tolerance * kz
    # Ordered, mutually exclusive reasons. Query/key sensor holes are unknown.
    status = np.full(len(centers), VALID, dtype=np.uint8)
    active = qknown.copy()
    status[~active] = QUERY_DEPTH_MISSING
    bad = active & ~np.isfinite(kpoints).all(axis=1)
    status[bad] = NONFINITE_PROJECTION
    active &= ~bad
    bad = active & ~project_valid
    status[bad] = BEHIND_CAMERA
    active &= ~bad
    bad = active & ~inside
    status[bad] = OUTSIDE_IMAGE
    active &= ~bad
    bad = active & ~kknown
    status[bad] = KEY_DEPTH_MISSING
    active &= ~bad
    bad = active & ~consistent
    status[bad] = DEPTH_INCONSISTENT
    active &= ~bad
    indices, weights = bilinear_token_labels(pix, active, image_hw, stride)
    pix = np.where(qknown[:, None] & project_valid[:, None], pix, np.nan)
    if active.any():
        # True nearest grid, independent of bilinear lower-corner ordering.
        coordinates = np.clip(np.floor((pix[active] + .5) / stride),
                              [0, 0], [w//stride-1, h//stride-1]).astype(int)
        nearest = (coordinates + .5) * stride - .5
        oracle_error = np.linalg.norm(nearest - pix[active], axis=1)
    else:
        oracle_error = np.array([], dtype=np.float64)
    metadata = {
        "image_hw": list(image_hw), "stride": int(stride), "token_count": len(centers),
        "absolute_tolerance_m": absolute_tolerance, "relative_tolerance": relative_tolerance,
        "depth_consistency": "abs(projected_z-sensor_z)<=absolute+relative*sensor_z",
        "pixel_convention": "zero-based centers; half-pixel resize; cells at (i+.5)*stride-.5",
        "sensor_sampling": "native nearest pixel, floor(x+.5+1e-9); missing=unknown",
        "task": "matching conditional on GT-visible correspondences; no rejection labels",
        "status_counts": {name: int((status == code).sum()) for code, name in STATUS_NAMES.items()},
        "gt_valid_fraction": float(active.mean()),
        "nearest_grid_oracle_max_epe": float(oracle_error.max()) if len(oracle_error) else None,
        "nearest_grid_oracle_mean_epe": float(oracle_error.mean()) if len(oracle_error) else None,
        "nearest_grid_oracle_pck8": float((oracle_error <= 8).mean()) if len(oracle_error) else None,
    }
    return PairSupervision(active, pix.astype(np.float32), indices, weights, status, metadata)


@dataclass(frozen=True)
class FrameObservation:
    """Only inference-available RGB and shared camera metadata, with full grid."""
    frame_id: str
    rgb: np.ndarray
    K: np.ndarray
    c2w: np.ndarray
    pixels: np.ndarray


@dataclass(frozen=True)
class Sun3DFrame:
    frame_id: str
    original_index: int
    rgb_path: Path
    depth_path: Path
    K_native: np.ndarray
    c2w: np.ndarray
    time_gap_raw: int

    def observation(self, image_hw: tuple[int, int] = DEFAULT_IMAGE_HW,
                    stride: int = DEFAULT_STRIDE) -> FrameObservation:
        """Reads RGB only: deleting/changing depth files cannot alter this output."""
        with Image.open(self.rgb_path) as image:
            image = image.convert("RGB")
            native_hw = (image.height, image.width)
            th, tw = _hw(image_hw)
            rgb = np.asarray(image.resize((tw, th), Image.Resampling.BILINEAR)).copy()
        return FrameObservation(self.frame_id, rgb,
                                resize_intrinsics(self.K_native, native_hw, image_hw),
                                self.c2w.copy(), token_grid(image_hw, stride))


def _listing(path: Path, extension: str) -> list[str]:
    return sorted(set(re.findall(r'href="([^"/]+\.' + extension + r')"', path.read_text())))


def sun3d_timestamp(filename: str) -> int:
    match = re.fullmatch(r"\d+-(\d+)\.(?:jpg|png)", filename)
    if not match:
        raise ValueError(f"invalid SUN3D frame filename: {filename}")
    return int(match.group(1))


def load_sun3d_scene(data_root: str | Path, scene: str,
                     verify_hashes: bool = True) -> tuple[dict[int, Sun3DFrame], dict[str, Any]]:
    """Load root's sparse ``selection.json`` without confusing sparse/full indices.

    All native metadata is verified; no RGB or sensor image is read here.
    Frames remain indexed by their original ZERO-BASED full RGB ordinal.
    """
    data_root = Path(data_root)
    selection = json.loads((data_root / "selection.json").read_text())
    rows = [row for row in selection["scenes"] if row["scene"] == scene]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one selected scene {scene}")
    row = rows[0]
    directory = data_root / scene
    K_path, pose_path = directory / "intrinsics.txt", data_root / row["pose_file"]
    if verify_hashes:
        for path, key in [(K_path, "K_sha256"), (pose_path, "pose_sha256")]:
            if sha256(path) != row[key]:
                raise ValueError(f"metadata hash mismatch: {path}")
    rgb_names = _listing(directory / "image_index.html", "jpg")
    depth_names = _listing(directory / "depth_index.html", "png")
    poses, K = read_sun3d_extrinsics(pose_path), read_sun3d_intrinsics(K_path)
    if len(rgb_names) != row["rgb_count"] or len(poses) != len(rgb_names):
        raise ValueError("full RGB listing/pose count mismatch")
    if len(depth_names) != row["depth_count"]:
        raise ValueError("full depth listing count mismatch")
    depth_times = np.array([sun3d_timestamp(name) for name in depth_names], dtype=np.int64)
    frames = {}
    for item in row["frames"]:
        index = int(item["index"])
        if index < 0 or index >= len(rgb_names) or rgb_names[index] != item["rgb"]:
            raise ValueError("selected RGB does not match original full-list index")
        gaps = np.abs(depth_times - sun3d_timestamp(item["rgb"]))
        nearest = int(gaps.argmin())
        if depth_names[nearest] != item["depth"] or int(gaps[nearest]) != item["time_gap_us"]:
            raise ValueError("selected depth is not official nearest timestamp association")
        if index in frames:
            raise ValueError("duplicate selected RGB index")
        frames[index] = Sun3DFrame(f"{scene}/{item['rgb']}", index,
                                  directory / "image" / item["rgb"],
                                  directory / "depth" / item["depth"],
                                  K.copy(), poses[index].copy(), int(gaps[nearest]))
    for q, k in row["pairs"]:
        if q not in frames or k not in frames:
            raise ValueError("selected pair references an absent sparse frame")
    return frames, row


def export_pair_supervision(q: Sun3DFrame, k: Sun3DFrame, output: str | Path,
                            **kwargs: Any) -> PairSupervision:
    """Save sparse labels separately from any forward RGB/geometry cache."""
    qdepth, kdepth = load_sensor_depth(q.depth_path), load_sensor_depth(k.depth_path)
    # Alignment is official; additionally reject size mismatches instead of rescaling depth.
    for frame, depth in [(q, qdepth), (k, kdepth)]:
        with Image.open(frame.rgb_path) as image:
            if (image.height, image.width) != depth.shape:
                raise ValueError("pre-aligned SUN3D RGB/depth sizes disagree")
    labels = pair_supervision(qdepth, kdepth, q.K_native, k.K_native, q.c2w, k.c2w, **kwargs)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(labels.metadata, query_frame=q.frame_id, key_frame=k.frame_id,
                    query_original_index=q.original_index, key_original_index=k.original_index,
                    query_depth_sha256=sha256(q.depth_path), key_depth_sha256=sha256(k.depth_path))
    np.savez_compressed(output, gt_valid=labels.gt_valid, pixcontinuous=labels.pixcontinuous,
                        key_indices=labels.key_indices, key_weights=labels.key_weights,
                        status=labels.status, metadata_json=np.array(json.dumps(metadata)))
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/round5_sun3d"))
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frames, scene = load_sun3d_scene(args.data_root, args.scene)
    records = []
    for qid, kid in scene["pairs"]:
        path = args.output / f"{qid:07d}_{kid:07d}.npz"
        labels = export_pair_supervision(frames[qid], frames[kid], path)
        records.append(dict(query=qid, key=kid, file=str(path), **labels.metadata))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps({"scene": args.scene,
        "source_sha256": sha256(__file__), "pairs": records}, indent=2))


if __name__ == "__main__":
    main()
