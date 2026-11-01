"""CPU-only regional token data from real photographs, SfM tracks, and 3DGS.

Labels are sparse observed-track transfer distributions, NOT dense surface area
or occlusion truth. Geometry comes from a separate trained GS reconstruction.
Query/key regional fits use disjoint primitive subsets; primitive IDs and track
IDs are never supplied as model features. The renderer is explicitly an
approximate opaque-center z-buffer, not alpha-composited 3DGS rendering.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw

DEFAULT_SOURCES = {
    "teatime": "/home/liwt/miniColmap/outputs/ui_3dgs_20261024_112609_a1d94a",
    "waldo_kitchen": "/home/liwt/miniColmap/outputs/ui_3dgs_20261019_230921_797f2a",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def scaled_intrinsics(K, original_size, new_size):
    sx, sy = np.array(new_size) / np.array(original_size)
    return np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1.]]) @ K


def fixed_regions(width=192, height=144, grid=(8, 6)):
    gw, gh = grid
    x, y = np.meshgrid(np.arange(gw), np.arange(gh))
    mean = np.stack([(x + .5) * width / gw - .5,
                     (y + .5) * height / gh - .5], -1).reshape(-1, 2)
    cov = np.broadcast_to(np.diag([(width / gw) ** 2 / 12,
                                  (height / gh) ** 2 / 12]), (gw * gh, 2, 2)).copy()
    return mean, cov


def region_indices(pixels, width, height, grid):
    """Integer pixel centers with half-pixel image bounds and disjoint regions."""
    pixels = np.asarray(pixels)
    finite = np.isfinite(pixels).all(-1)
    inside = finite & (pixels[:, 0] >= -.5) & (pixels[:, 0] < width - .5)
    inside &= (pixels[:, 1] >= -.5) & (pixels[:, 1] < height - .5)
    safe = np.where(finite[:, None], pixels, 0)
    gx = np.floor((safe[:, 0] + .5) * grid[0] / width).astype(np.int64)
    gy = np.floor((safe[:, 1] + .5) * grid[1] / height).astype(np.int64)
    idx = gy * grid[0] + gx
    return idx, inside


def project_gaussian(mean, cov, K, R, t):
    camera = mean @ R.T + t
    z = np.maximum(camera[:, 2], 1e-8)
    pixh = camera @ K.T
    pixels = pixh[:, :2] / z[:, None]
    Jcam = np.zeros((len(mean), 2, 3))
    Jcam[:, 0, 0] = K[0, 0] / z
    Jcam[:, 0, 1] = K[0, 1] / z
    Jcam[:, 0, 2] = -(K[0, 0] * camera[:, 0] + K[0, 1] * camera[:, 1]) / z ** 2
    Jcam[:, 1, 1] = K[1, 1] / z
    Jcam[:, 1, 2] = -K[1, 1] * camera[:, 1] / z ** 2
    J = Jcam @ R
    projected_cov = J @ cov @ np.swapaxes(J, -1, -2)
    return pixels, projected_cov, camera[:, 2] > 1e-8


def fit_regions(means, covariances, opacities, K, R, t, *, width=192,
                height=144, grid=(8, 6), opacity_threshold=.3,
                near_surface_fraction=.015, min_points=12, eigen_floor=1e-7):
    """Opaque center-z-buffer followed by opacity-weighted Gaussian moment fit.

    All arguments are geometry/camera observations; labels are unavailable here.
    Each valid region approximates the mixture's first two moments, including
    the individual primitives' own covariances (law of total covariance).
    """
    camera = means @ R.T + t
    z = camera[:, 2]
    positive = np.isfinite(camera).all(-1) & (z > 1e-6) & (opacities >= opacity_threshold)
    safez = np.maximum(z, 1e-8)
    pixels = (camera @ K.T)[:, :2] / safez[:, None]
    regions, inside = region_indices(pixels, width, height, grid)
    selected = positive & inside
    ids = np.flatnonzero(selected)
    xy = np.floor(pixels[ids] + .5).astype(np.int64)
    linear = xy[:, 1] * width + xy[:, 0]
    zbuffer = np.full(width * height, np.inf)
    np.minimum.at(zbuffer, linear, z[ids])
    visible = z[ids] <= zbuffer[linear] * (1 + near_surface_fraction)
    ids = ids[visible]
    token_count = grid[0] * grid[1]
    out_mean = np.zeros((token_count, 3))
    out_cov = np.broadcast_to(np.eye(3) * eigen_floor, (token_count, 3, 3)).copy()
    count = np.zeros(token_count, np.int32)
    for i in range(token_count):
        point_ids = ids[regions[ids] == i]
        count[i] = len(point_ids)
        if len(point_ids) < min_points:
            continue
        weights = opacities[point_ids].astype(np.float64)
        weights /= weights.sum()
        xyz = means[point_ids]
        mu = weights @ xyz
        centered = xyz - mu
        C = np.einsum("n,ni,nj->ij", weights, centered, centered)
        C += np.einsum("n,nij->ij", weights, covariances[point_ids])
        values, vectors = np.linalg.eigh((C + C.T) * .5)
        out_mean[i] = mu
        out_cov[i] = (vectors * np.maximum(values, eigen_floor)) @ vectors.T
    valid = count >= min_points
    return {"mean": out_mean, "cov": out_cov, "valid": valid, "point_count": count,
            "zbuffer": zbuffer.reshape(height, width),
            "stats": {"front_and_inside": int(selected.sum()), "visible_centers": len(ids),
                      "valid_regions": int(valid.sum()),
                      "zbuffer_pixel_fraction": float(np.isfinite(zbuffer).mean())}}


def track_regions(model, frame, width=192, height=144, grid=(8, 6)):
    """Build evaluator-only map from real observed track IDs to image regions."""
    image_id = frame["id"]
    observed = {}
    # Apply exactly the same half-pixel resize mapping to observed keypoints.
    sx, sy = width / frame["width"], height / frame["height"]
    keys = np.asarray(frame["keypoints"], dtype=np.float64).reshape(-1, 2)
    pix = (keys + .5) * np.array([sx, sy]) - .5
    regions, inside = region_indices(pix, width, height, grid)
    for point in model["points"]:
        for iid, key_idx in point["observations"]:
            if iid == image_id and 0 <= key_idx < len(keys) and inside[key_idx]:
                observed[int(point["id"])] = int(regions[key_idx])
                break
    return observed


def soft_track_labels(query_tracks, key_tracks, token_count=48, min_tracks=4):
    counts = np.zeros((token_count, token_count), np.int32)
    query_total = np.bincount(list(query_tracks.values()), minlength=token_count)
    for track_id, qi in query_tracks.items():
        if track_id in key_tracks:
            counts[qi, key_tracks[track_id]] += 1
    shared = counts.sum(1)
    soft = counts / np.maximum(shared[:, None], 1)
    return {"soft_labels": soft.astype(np.float32), "label_valid": shared >= min_tracks,
            "track_counts": counts, "query_track_count": query_total.astype(np.int32),
            "shared_track_count": shared,
            "track_observation_fraction": shared / np.maximum(query_total, 1)}


def rgb_features(path, grid=(8, 6)):
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        small = np.asarray(rgb.resize((grid[0] * 6, grid[1] * 6), Image.Resampling.BOX), np.float32) / 255
        preview = rgb.resize((192, 144), Image.Resampling.BOX)
    features = small.reshape(grid[1], 6, grid[0], 6, 3).transpose(0, 2, 1, 3, 4).reshape(-1, 108)
    return features, preview


def make_pairs(ordered_ids, groups=8):
    """No image belongs to more than one train/validation/test camera block."""
    pairs = []
    blocks = np.array_split(ordered_ids, 3)
    for split, ids in zip(("train", "validation", "test"), blocks):
        for group_id, segment in enumerate(np.array_split(ids, min(groups, len(ids) // 3))):
            q = int(segment[0])
            for distance, index in (("near", max(1, len(segment) // 3)),
                                    ("medium", len(segment) - 1)):
                k = int(segment[index])
                if any(p["q_id"] == q and p["k_id"] == k for p in pairs):
                    continue
                pairs.append(dict(split=split, baseline_group=distance, group=group_id, q_id=q, k_id=k))
    return pairs


def load_scene(output):
    model_path = output / "source_model.json"
    model = json.loads(model_path.read_text())
    status = json.loads((output / "run_status.json").read_text())
    dataset_path = Path(status["dataset"])
    dataset = json.loads(dataset_path.read_text())
    cameras = {x["id"]: x for x in model["cameras"]}
    poses = {x["image_id"]: x for x in model["poses"]}
    data_images = {x["id"]: x for x in dataset["images"]}
    frames = {}
    for image in model["images"]:
        iid = image["id"]
        if iid not in poses or iid not in data_images:
            continue
        camera = cameras[image["camera_id"]]
        original_size = (camera["width"], camera["height"])
        K = np.asarray(camera["K"], np.float64)
        image_path = (dataset_path.parent / data_images[iid]["path"]).resolve()
        if not image_path.exists():
            continue
        frames[iid] = dict(image, **{k: poses[iid][k] for k in ("R", "t")},
                           K_original=K, K=scaled_intrinsics(K, original_size, (192, 144)),
                           width=original_size[0], height=original_size[1], image_path=image_path)
    ids = sorted(frames, key=lambda i: frames[i]["name"])
    normalization = json.loads((output / "training_scene.json").read_text())["normalization"]
    # Exported gaussians.npz is already in original world coordinates, unlike
    # normalized checkpoint.pt and training_scene.json frame translations.
    with np.load(output / "gaussians.npz", allow_pickle=False) as f:
        geometry = {key: np.asarray(f[key]) for key in ("means", "covariances", "opacities")}
    return model, frames, ids, geometry, dataset_path, normalization


def export_scene(name, source, destination, groups=8, opacity_threshold=.3):
    start = time.monotonic()
    model, frames, ids, geometry, dataset_path, normalization = load_scene(source)
    pairs = make_pairs(ids, groups)
    destination.mkdir(parents=True, exist_ok=True)
    cache = {}
    previews = {}
    frame_records = []
    source_images = {}
    floor = (1e-4 * normalization["scale"]) ** 2
    used_ids = sorted({p[key] for p in pairs for key in ("q_id", "k_id")})
    for number, iid in enumerate(used_ids):
        frame = frames[iid]
        R, t = np.asarray(frame["R"]), np.asarray(frame["t"])
        features, preview = rgb_features(frame["image_path"])
        fits = []
        for parity in (0, 1):
            fit = fit_regions(geometry["means"][parity::2], geometry["covariances"][parity::2],
                              geometry["opacities"][parity::2], frame["K"], R, t,
                              opacity_threshold=opacity_threshold, eigen_floor=floor)
            fits.append(fit)
        tracks = track_regions(model, frame)
        cache[iid] = dict(features=features, fits=fits, tracks=tracks)
        previews[iid] = preview
        source_images[str(frame["image_path"])] = sha256(frame["image_path"])
        frame_records.append(dict(image_id=iid, name=frame["name"], K=frame["K"].tolist(),
                                  R=frame["R"], t=frame["t"],
                                  geometry_stats=[f["stats"] for f in fits], observed_tracks=len(tracks)))
        if (number + 1) % 12 == 0:
            print(f"{name}: prepared {number + 1}/{len(used_ids)} views", flush=True)
    pixel_mean, pixel_cov = fixed_regions()
    records = []
    preview_rows = []
    for index, pair in enumerate(pairs):
        qi, ki = pair["q_id"], pair["k_id"]
        qdata, kdata = cache[qi], cache[ki]
        qframe, kframe = frames[qi], frames[ki]
        R, t, K = np.asarray(qframe["R"]), np.asarray(qframe["t"]), qframe["K"]
        arrays = {"q_features": qdata["features"], "k_features": kdata["features"],
                  "q_pixel_mean": pixel_mean.astype(np.float32), "q_pixel_cov": pixel_cov.astype(np.float32),
                  "K_query": K.astype(np.float64), "R_query": R, "t_query": t,
                  "K_key": kframe["K"], "R_key": np.asarray(kframe["R"]), "t_key": np.asarray(kframe["t"])}
        for alternate in (False, True):
            suffix = "_alt" if alternate else ""
            # Swapped parity is a sensitivity check; in both settings the two
            # views use disjoint primitive IDs, so no matched primitive survives.
            qfit = qdata["fits"][int(alternate)]
            kfit = kdata["fits"][1 - int(alternate)]
            for side, fit in (("q", qfit), ("k", kfit)):
                pixels, C2, front = project_gaussian(fit["mean"], fit["cov"], K, R, t)
                valid = fit["valid"] & front
                pixels[~valid] = 0
                C2[~valid] = np.eye(2)
                for field, value in (("mean", fit["mean"]), ("cov", fit["cov"]),
                                     ("projected_mean", pixels), ("projected_cov", C2)):
                    arrays[f"{side}_{field}{suffix}"] = value.astype(np.float32)
                arrays[f"{side}_valid{suffix}"] = valid
                arrays[f"{side}_fit_point_count{suffix}"] = fit["point_count"]
        arrays.update(soft_track_labels(qdata["tracks"], kdata["tracks"]))
        filename = f"pair_{index:03d}_{pair['split']}.npz"
        np.savez_compressed(destination / filename, **arrays)
        qcenter = -np.asarray(qframe["R"]).T @ np.asarray(qframe["t"])
        kcenter = -np.asarray(kframe["R"]).T @ np.asarray(kframe["t"])
        common = int(arrays["track_counts"].sum())
        effective = arrays["label_valid"] & arrays["q_valid"]
        records.append(dict(pair, file=filename, common_tracks=common,
                            label_valid_rows=int(arrays["label_valid"].sum()),
                            geometry_and_label_valid_rows=int(effective.sum()),
                            camera_distance=float(np.linalg.norm(qcenter - kcenter)),
                            q_name=qframe["name"], k_name=kframe["name"],
                            npz_sha256=sha256(destination / filename)))
        if pair["group"] == 0:
            row = Image.new("RGB", (384, 162), "white")
            row.paste(previews[qi], (0, 18)); row.paste(previews[ki], (192, 18))
            draw = ImageDraw.Draw(row)
            draw.text((2, 2), f"{name} {pair['split']} {pair['baseline_group']} tracks={common}", fill="black")
            for x in range(24, 192, 24):
                draw.line((x, 18, x, 161), fill="white");draw.line((192+x,18,192+x,161),fill="white")
            for y in range(24, 144, 24):
                draw.line((0,18+y,383,18+y),fill="white")
            preview_rows.append(row)
    if preview_rows:
        preview = Image.new("RGB", (384, 162 * len(preview_rows)), "white")
        for i, row in enumerate(preview_rows):
            preview.paste(row, (0, i * 162))
        preview.save(destination / "preview.png")
    source_paths = [source / "gaussians.npz", source / "source_model.json", source / "training_scene.json",
                    source / "run_status.json", dataset_path]
    manifest = dict(scene=name, source_output=str(source), dataset=str(dataset_path),
                    source_sha256={str(p): sha256(p) for p in source_paths}, source_images_sha256=source_images,
                    normalization=normalization, coordinate_system="original unnormalized world coordinates",
                    pair_count=len(records), available_frames=len(frames), used_views=len(used_ids),
                    primitive_count=len(geometry["means"]), camera_split="contiguous thirds by original image filename",
                    seed=None, groups_per_split=groups, token_grid=[8,6], image_size=[192,144], feature_shape=[48,108],
                    opacity_threshold=opacity_threshold, near_surface_fraction=.015, minimum_fit_points=12,
                    minimum_shared_tracks=4, primitive_partitions="even q / odd k; alternate swaps both; disjoint within each pair",
                    label_definition="row-normalized counts of independently observed common SfM tracks in fixed image regions",
                    limitations=["Sparse salient-track transfer is not dense area overlap or visibility truth.",
                                 "Trained GS geometry may incorporate all source photos; camera split tests matching generalization, not unseen-geometry reconstruction.",
                                 "Opaque center z-buffer is not alpha-composited Gaussian rasterization.",
                                 "One Gaussian per region compresses potentially disconnected surfaces.",
                                 "Track labels come from the SfM source used to initialize GS, though disjoint primitives are used at the two token ends.",
                                 "Invalid geometry or insufficient observed tracks require evaluator masking; counts/coverage are never model inputs."],
                    frames=frame_records, pairs=records, seconds=round(time.monotonic()-start,3))
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    print(f"{name}: {len(records)} pairs, {len(used_ids)} views, {manifest['seconds']:.1f}s", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", nargs="+", choices=list(DEFAULT_SOURCES), default=list(DEFAULT_SOURCES))
    parser.add_argument("--out", type=Path, default=Path("results/round3_real_support_data"))
    parser.add_argument("--groups-per-split", type=int, default=8)
    parser.add_argument("--opacity-threshold", type=float, default=.3)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name in args.scenes:
        manifest = export_scene(name, Path(DEFAULT_SOURCES[name]), args.out / name,
                                args.groups_per_split, args.opacity_threshold)
        summaries.append({k: manifest[k] for k in ("scene", "pair_count", "used_views", "seconds")})
    source = Path(__file__).resolve()
    shutil.copy2(source, args.out / "source_snapshot.py")
    (args.out / "summary.json").write_text(json.dumps(dict(scenes=summaries, source_sha256=sha256(source)), indent=2))
    (args.out / "README.md").write_text("""# Real photographic regional support data

This dataset exports fixed 8 x 6 image regions from LERF-OVS teatime and waldo_kitchen photographs. Each token contains a 6 x 6 RGB patch and the first two moments of independently visible reconstructed Gaussians. Query and key use mutually exclusive even/odd primitive subsets; `_alt` arrays swap these subsets. No primitive ID or SfM track ID is exported as a token input.

`soft_labels` count the common SfM tracks observed in each pair of regions and normalize each query row. These are sparse salient-track transfer proxies, not dense surface-area overlap and not occlusion labels. Rows with fewer than four common tracks have `label_valid=False`; evaluators must also handle `q_valid`/`k_valid`. `track_counts`, all count/fraction arrays, and label arrays are evaluator/training targets only and must never enter the matcher.

The CPU geometry exporter projects high-opacity (>=0.3) GS centers into a 192 x 144 z-buffer, retains centers within 1.5% of the first depth, and moment-fits each region, including internal primitive covariance. It does not implement alpha compositing. The Gaussian reconstruction can include all scene photos, so split cameras isolate matcher fitting, not reconstruction.

Every scene manifest records original source paths, SHA256 hashes of geometry/SfM/cameras/images, fit statistics, pair splits, and limitations. `q_pixel_mean/cov` are fixed image-region support; `q_projected_mean/cov` and `k_projected_mean/cov` project fitted world support into the query camera. Coordinates use unnormalized source-model world units. `source_snapshot.py` and `summary.json` identify exporter code.

Source outputs:
- teatime: `/home/liwt/miniColmap/outputs/ui_3dgs_20261024_112609_a1d94a`
- waldo_kitchen: `/home/liwt/miniColmap/outputs/ui_3dgs_20261019_230921_797f2a`

Reproduce: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.real_support_data`.
""")


if __name__ == "__main__":
    main()
