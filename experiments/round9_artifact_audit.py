"""Pre-evaluation Round-9 artifact-chain and shared-fallback audit.

Reads no target RGB. Uses saved arrays and weights only; no model/trainer import.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check(path, expected):
    if sha(path) != expected:
        raise ValueError(f"Artifact hash mismatch: {path}")


def npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.array(archive[key]) for key in archive.files}


def reconstruct_covariance(scales, quaternions):
    quaternions = quaternions.astype(np.float64)
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    w, x, y, z = quaternions.T
    matrix = np.empty((len(w), 3, 3))
    matrix[:, 0, 0] = 1 - 2 * (y*y + z*z)
    matrix[:, 0, 1] = 2 * (x*y - z*w)
    matrix[:, 0, 2] = 2 * (x*z + y*w)
    matrix[:, 1, 0] = 2 * (x*y + z*w)
    matrix[:, 1, 1] = 1 - 2 * (x*x + z*z)
    matrix[:, 1, 2] = 2 * (y*z - x*w)
    matrix[:, 2, 0] = 2 * (x*z - y*w)
    matrix[:, 2, 1] = 2 * (y*z + x*w)
    matrix[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return np.einsum("nik,nk,njk->nij", matrix, scales.astype(np.float64)**2, matrix)


def audit(runs, data, out):
    runs, data, out = Path(runs), Path(data), Path(out)
    if out.exists():
        raise FileExistsError(out)
    lock_path = runs / "lock.json"
    lock = json.loads(lock_path.read_text())
    for path, expected in {**lock["source_hashes"], **lock["data_hashes"]}.items():
        check(path, expected)
    records = []
    for scene in lock["scenes"]:
        directory = runs / scene
        run = json.loads((directory / "run.json").read_text())
        check(lock_path, run["lock_sha256"])
        check(directory / "training.json", run["training_sha256"])
        for filename, key in (("predictions.npz", "prediction_sha256"), ("colors.npz", "colors_sha256"), ("operator_samples.json", "operator_samples_sha256")):
            check(directory / filename, run[key])
        training = json.loads((directory / "training.json").read_text())
        check(directory / "state.pt", training["state_sha256"])
        check(directory / "gaussians.npz", training["gaussians_sha256"])
        check(lock_path, training["lock_sha256"])
        if run["source_hashes"] != lock["source_hashes"] or training["source_hashes"] != lock["source_hashes"]:
            raise ValueError("Source hash dictionaries disagree")
        if run["data_hashes"] != lock["data_hashes"] or training["data_hashes"] != lock["data_hashes"]:
            raise ValueError("Data hash dictionaries disagree")
        assert run["target_rgb_read"] is False and training["target_rgb_read"] is False
        assert training["steps"] == 3000 and training["seed"] == 11
        assert training["checkpoint_selection"] == "fixed final step, no validation"
        manifest = json.loads((data / scene / "manifest.json").read_text())
        target_frames = json.loads((data / scene / "target_cameras.json").read_text())["frames"]
        context_ids = manifest["split"]["context_ids"]
        assert all(row["context_image_id"] in context_ids for row in training["history"])
        initial = training["initial_point_count"]
        for step, record in zip((500, 1000, 1500, 2000), training["topology"], strict=True):
            splits = min(initial, 12000 - initial)
            assert record["step"] == step and record["split_parents"] == splits
            initial += splits
            assert record["primitives"] == initial
        assert initial == training["point_count"] == run["point_count"]
        gaussians = npz(directory / "gaussians.npz")
        state = {key: value.cpu().numpy() for key, value in torch.load(directory / "state.pt", map_location="cpu", weights_only=True).items()}
        assert all(np.isfinite(value).all() for value in gaussians.values())
        np.testing.assert_array_equal(gaussians["means"], state["means"])
        np.testing.assert_array_equal(gaussians["quaternions"], state["quaternions"])
        np.testing.assert_allclose(gaussians["scales"], np.exp(state["log_scales"].astype(np.float64)), rtol=2e-7)
        np.testing.assert_allclose(gaussians["colors"], 1 / (1 + np.exp(-state["color_logits"].astype(np.float64))), rtol=2e-7, atol=1e-8)
        np.testing.assert_allclose(gaussians["opacities"], 1 / (1 + np.exp(-state["opacity_logits"].astype(np.float64))), rtol=2e-7, atol=1e-8)
        reconstructed = reconstruct_covariance(gaussians["scales"], gaussians["quaternions"])
        error = np.linalg.norm(reconstructed - gaussians["covariances"], axis=(1, 2)) / np.maximum(np.linalg.norm(reconstructed, axis=(1, 2)), 1e-30)
        assert error.max() < 2e-6
        color_archive, prediction_archive = npz(directory / "colors.npz"), npz(directory / "predictions.npz")
        colors, counts = color_archive["colors"], color_archive["source_view_counts"]
        assert color_archive["methods"].tolist() == prediction_archive["methods"].tolist() == lock["methods"]
        assert prediction_archive["target_ids"].tolist() == [row["id"] for row in target_frames]
        assert prediction_archive["target_splits"].tolist() == [row["split"] for row in target_frames]
        assert colors.shape == (18, initial, 3) and counts.shape == (initial,)
        assert np.issubdtype(counts.dtype, np.integer) and counts.min() >= 0 and counts.max() <= 24
        assert sum(row["visible_primitive_count"] for row in run["views"]) == counts.sum()
        readable = counts > 0
        assert readable.sum() == run["readable_primitive_count"]
        assert (~readable).sum() == run["fallback_primitive_count"]
        assert [row["id"] for row in run["views"]] == context_ids
        for method_index in range(18):
            np.testing.assert_array_equal(colors[method_index, ~readable], colors[-1, ~readable])
        np.testing.assert_array_equal(colors[-1], gaussians["colors"])
        assert np.isfinite(colors).all() and colors.min() >= 0 and colors.max() <= 1
        predictions = prediction_archive["predictions"]
        assert predictions.shape == (18, 8, 144, 192, 3)
        assert np.isfinite(predictions).all() and predictions.min() >= 0 and predictions.max() <= 1 + 1e-6
        alpha, fallback = prediction_archive["alpha"], prediction_archive["fallback_contribution"]
        assert alpha.shape == fallback.shape == (8, 144, 192)
        assert np.isfinite(alpha).all() and np.isfinite(fallback).all()
        assert alpha.min() >= 0 and alpha.max() <= 1 + 1e-6
        assert fallback.min() >= 0 and np.max(fallback - alpha) <= 1e-6
        frame_geometry = []
        for index, frame in enumerate(target_frames):
            frame_geometry.append(dict(id=frame["id"], split=frame["split"],
                mean_alpha=float(alpha[index].astype(np.float64).mean()),
                mean_fallback_contribution=float(fallback[index].astype(np.float64).mean()),
                fallback_fraction_of_alpha=float(fallback[index].astype(np.float64).sum() / max(alpha[index].astype(np.float64).sum(), 1e-30))))
        records.append(dict(scene=scene, primitives=initial, readable_primitives=int(readable.sum()),
                            fallback_primitives=int((~readable).sum()), covariance_rs_max_relative_error=float(error.max()),
                            unchanged_fallback_colors=True, target_geometry=frame_geometry,
                            context_mean_frame_psnr=float(np.mean([row["psnr"] for row in training["context_metrics"]])),
                            run_sha256=sha(directory / "run.json"), target_rgb_reads=0))
    result = dict(passed=True, lock_sha256=sha(lock_path), source_sha256=sha(__file__), scenes=records,
                  verification="Full lock/run/training/state/gaussians/colors/predictions/operator artifact chain; independent NumPy covariance and fallback checks",
                  target_rgb_reads=0, model_or_trainer_imports=False)
    out.mkdir(parents=True)
    (out / "verification.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("results/round9_gs_readout"))
    parser.add_argument("--data", type=Path, default=Path("results/round9_context_data"))
    parser.add_argument("--out", type=Path, default=Path("results/round9_artifact_audit"))
    args = parser.parse_args()
    print(json.dumps(audit(args.runs, args.data, args.out)))
