"""Independent metadata, camera and context-RGB replay; no target RGB opened."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit(data, out):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    records = []
    for scene in ("teatime", "waldo_kitchen"):
        directory = Path(data) / scene
        manifest = json.loads((directory / "manifest.json").read_text())
        targets = json.loads((directory / "target_cameras.json").read_text())["frames"]
        dataset = json.loads(Path(manifest["source_dataset"]).read_text())
        assert set(dataset) == {"cameras", "images", "pose_convention", "poses"}
        assert dataset["pose_convention"] == "world_to_camera"
        assert digest(manifest["source_dataset"]) == manifest["source_dataset_sha256"]
        for name, key in (("context.npz", "context_archive_sha256"), ("target_cameras.json", "target_cameras_sha256")):
            assert digest(directory / name) == manifest[key]
        assert manifest["target_rgb_reads"] == 0
        assert manifest["old_geometry_keypoints_tracks_read"] is False
        images = {int(row["id"]): row for row in dataset["images"]}
        poses = {int(row["image_id"]): row for row in dataset["poses"]}
        cameras = {int(row["id"]): row for row in dataset["cameras"]}
        ordered = sorted(poses, key=lambda identity: images[identity]["name"])
        # Independently reproduce array_split's first two block sizes.
        quotient, remainder = divmod(len(ordered), 3)
        development_count = 2 * quotient + min(remainder, 2)
        development = ordered[:development_count]
        picked = [development[(index * (development_count - 1)) // 31] for index in range(32)]
        context_ids = [identity for index, identity in enumerate(picked) if index % 4 != 3]
        target_ids = [identity for index, identity in enumerate(picked) if index % 4 == 3]
        assert manifest["split"]["context_ids"] == context_ids
        assert manifest["split"]["target_ids"] == target_ids
        assert [row["id"] for row in targets] == target_ids
        assert [row["split"] for row in targets] == ["calibration", "gate"] * 4
        assert set(picked).isdisjoint(ordered[development_count:])
        source_frames = manifest["context_frames"]
        assert [row["id"] for row in source_frames] == context_ids
        assert {pair["first_id"] for pair in manifest["pairs"]}.issubset(context_ids)
        assert {pair["second_id"] for pair in manifest["pairs"]}.issubset(context_ids)
        world_rotation = np.asarray([poses[identity]["R"] for identity in context_ids])
        world_translation = np.asarray([poses[identity]["t"] for identity in context_ids])
        world_centers = np.stack([-rotation.T @ translation for rotation, translation in zip(world_rotation, world_translation)])
        center = world_centers.mean(0)
        scale = 1.1 * np.linalg.norm(world_centers - center, axis=1).max()
        np.testing.assert_allclose(center, manifest["normalization"]["center"], rtol=0, atol=1e-14)
        np.testing.assert_allclose(scale, manifest["normalization"]["scale"], rtol=0, atol=1e-14)
        with np.load(directory / "context.npz", allow_pickle=False) as archive:
            arrays = {key: np.array(archive[key]) for key in archive.files}
        assert set(arrays) == {"points", "colors", "rgb", "K", "R", "t", "ids"}
        assert all(np.isfinite(array).all() for array in arrays.values())
        assert arrays["ids"].tolist() == context_ids
        assert arrays["rgb"].shape == (24, 144, 192, 3)
        assert arrays["points"].shape == arrays["colors"].shape
        assert arrays["points"].shape[1] == 3
        assert arrays["rgb"].min() >= 0 and arrays["rgb"].max() <= 1
        assert arrays["colors"].min() >= 0 and arrays["colors"].max() <= 1
        max_rgb_error = 0.
        max_camera_error = 0.
        context_paths = set()
        dataset_root = Path(manifest["source_dataset"]).parent
        for index, frame in enumerate(source_frames):
            identity = frame["id"]
            path = (dataset_root / images[identity]["path"]).resolve()
            assert path == Path(frame["image_path"]).resolve()
            assert digest(path) == manifest["source_image_sha256"][str(path)]
            context_paths.add(str(path))
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"))
            replay = cv2.resize(rgb, (192, 144), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
            max_rgb_error = max(max_rgb_error, float(np.abs(replay - arrays["rgb"][index]).max()))
            original_camera = cameras[images[identity]["camera_id"]]
            sx, sy = 192 / original_camera["width"], 144 / original_camera["height"]
            resize = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
            intrinsic = resize @ np.asarray(original_camera["K"])
            rotation = np.asarray(poses[identity]["R"])
            translation = (rotation @ center + np.asarray(poses[identity]["t"])) / scale
            for expected, actual in ((intrinsic, arrays["K"][index]), (rotation, arrays["R"][index]), (translation, arrays["t"][index])):
                max_camera_error = max(max_camera_error, float(np.abs(expected - actual).max()))
        assert context_paths == set(manifest["source_image_sha256"])
        for frame in targets:
            # Inspect only camera metadata and paths; target bytes are never read.
            identity = frame["id"]
            assert identity in target_ids
            assert str(Path(frame["image_path"]).resolve()) not in context_paths
            path = (dataset_root / images[identity]["path"]).resolve()
            assert path == Path(frame["image_path"]).resolve()
            camera = cameras[images[identity]["camera_id"]]
            sx, sy = 192 / camera["width"], 144 / camera["height"]
            resize = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
            rotation = np.asarray(poses[identity]["R"])
            translation = (rotation @ center + np.asarray(poses[identity]["t"])) / scale
            for expected, actual in ((resize @ np.asarray(camera["K"]), frame["K"]), (rotation, frame["R"]), (translation, frame["t"])):
                max_camera_error = max(max_camera_error, float(np.abs(expected - actual).max()))
        assert max_rgb_error == 0
        assert max_camera_error < 1e-12
        records.append(dict(scene=scene, context_images=24, calibration_images=4, gate_images=4,
                            initialization_points=len(arrays["points"]), context_rgb_max_error=max_rgb_error,
                            camera_max_error=max_camera_error, context_hashes_checked=len(context_paths),
                            target_rgb_reads=0, sealed_third_block_reads=0,
                            manifest_sha256=digest(directory / "manifest.json")))
    summary = dict(passed=True, independent_numpy_replay=True, no_exporter_or_trainer_import=True,
                   scenes=records, audit_source_sha256=digest(__file__),
                   limit="Metadata and context-RGB replay verifies implemented contract; given cameras still originate from full-image SfM")
    out.mkdir(parents=True)
    (out / "verification.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("results/round9_context_data"))
    parser.add_argument("--out", type=Path, default=Path("results/round9_data_audit"))
    args = parser.parse_args()
    print(json.dumps(audit(args.data, args.out)))
