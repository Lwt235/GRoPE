"""Independent, staged NumPy analysis of frozen Round-9 RGB predictions.

No model or training module is imported. Calibration opens only calibration
RGB, writes a hashed selection, and evaluation verifies that lock before any
gate RGB bytes are read. RGB uses the context exporter's INTER_AREA resize.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCENES = ("teatime", "waldo_kitchen")
ROLES = ("full", "isotropic", "fixed", "point")
SIMPLE = ("isotropic", "fixed", "point", "bilinear")
WIDTHS = ("0.25", "0.5", "1", "2")
METHODS = tuple(f"{role}_h{width}" for role in ROLES for width in WIDTHS) + ("bilinear", "original_gs")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def checked_hash(path, expected, description):
    actual = sha(path)
    if actual != expected:
        raise ValueError(f"{description} hash changed: {path}")
    return actual


def source_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_inputs(runs, data):
    """Read predictions, metadata and source/context hashes, never target RGB."""
    result, provenance = {}, {}
    for scene in SCENES:
        run_root, data_root = Path(runs) / scene, Path(data) / scene
        prediction_path = run_root / "predictions.npz"
        run_path, manifest_path = run_root / "run.json", data_root / "manifest.json"
        camera_path, context_path = data_root / "target_cameras.json", data_root / "context.npz"
        run = json.loads(run_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        cameras = json.loads(camera_path.read_text())
        expected_prediction = run.get("predictions_sha256", run.get("prediction_sha256"))
        if expected_prediction is None:
            raise ValueError("run.json must declare predictions_sha256")
        artifacts = {
            str(prediction_path.resolve()): checked_hash(prediction_path, expected_prediction, "prediction"),
            str(run_path.resolve()): sha(run_path),
            str(manifest_path.resolve()): checked_hash(manifest_path, run["data_manifest_sha256"], "data manifest"),
            str(camera_path.resolve()): checked_hash(camera_path, run["target_cameras_sha256"], "target cameras"),
            str(context_path.resolve()): checked_hash(context_path, run["context_archive_sha256"], "context archive"),
        }
        checked_hash(camera_path, manifest["target_cameras_sha256"], "manifest target cameras")
        checked_hash(context_path, manifest["context_archive_sha256"], "manifest context archive")
        if manifest.get("target_rgb_reads") != 0 or manifest.get("old_geometry_keypoints_tracks_read") is not False:
            raise ValueError("context export did not preserve pixel/geometry access contract")
        context_ids = manifest["split"]["context_ids"]
        declared_targets = manifest["split"]["target_ids"]
        frames = cameras["frames"]
        frame_ids = [int(row["id"]) for row in frames]
        frame_splits = [row["split"] for row in frames]
        if len(context_ids) != 24 or len(set(context_ids)) != 24:
            raise ValueError("expected 24 distinct context IDs")
        if len(frames) != 8 or len(set(frame_ids)) != 8 or frame_ids != declared_targets:
            raise ValueError("target identity/order mismatch")
        if set(context_ids) & set(frame_ids):
            raise ValueError("context and target IDs overlap")
        if frame_splits.count("calibration") != 4 or frame_splits.count("gate") != 4:
            raise ValueError("expected 4 calibration and 4 gate images")
        context_paths = {str(Path(path).resolve()) for path in manifest["source_image_sha256"]}
        target_paths = [str(Path(row["image_path"]).resolve()) for row in frames]
        if len(set(target_paths)) != 8 or context_paths & set(target_paths):
            raise ValueError("context and target image paths overlap")
        sources = run.get("source_hashes")
        if not isinstance(sources, dict) or not sources:
            raise ValueError("run.json must declare nonempty source_hashes")
        for path, expected in sources.items():
            resolved = source_path(path)
            artifacts[str(resolved.resolve())] = checked_hash(resolved, expected, "run source")
        for path, expected in manifest["source_image_sha256"].items():
            artifacts[str(Path(path).resolve())] = checked_hash(path, expected, "context RGB")
        dataset = manifest["source_dataset"]
        artifacts[str(Path(dataset).resolve())] = checked_hash(dataset, manifest["source_dataset_sha256"], "camera dataset")
        with np.load(prediction_path, allow_pickle=False) as archive:
            required = {"predictions", "methods", "target_ids", "target_splits"}
            if not required.issubset(archive.files):
                raise ValueError("prediction archive missing required fields")
            predictions = np.array(archive["predictions"])
            methods = archive["methods"].tolist()
            ids, splits = archive["target_ids"].tolist(), archive["target_splits"].tolist()
        if len(methods) != len(METHODS) or set(methods) != set(METHODS):
            raise ValueError("unexpected or duplicate methods")
        if ids != frame_ids or splits != frame_splits:
            raise ValueError("prediction target identity/order/split mismatch")
        height, width = int(frames[0]["height"]), int(frames[0]["width"])
        if any(int(f["height"]) != height or int(f["width"]) != width for f in frames):
            raise ValueError("target resolution differs within scene")
        if predictions.shape != (len(METHODS), 8, height, width, 3):
            raise ValueError("incorrect prediction shape")
        if predictions.dtype != np.float32 or not np.isfinite(predictions).all():
            raise ValueError("predictions must be finite float32")
        if predictions.min() < -1e-6 or predictions.max() > 1 + 1e-6:
            raise ValueError("prediction outside declared [0,1] RGB contract")
        result[scene] = dict(predictions=predictions, methods=methods, frames=frames)
        provenance[scene] = dict(artifact_hashes=artifacts, target_paths=target_paths,
                                 target_ids=frame_ids, target_splits=frame_splits)
    return result, provenance


def read_target(frame):
    """Hash exactly the bytes decoded, then match context INTER_AREA resize."""
    import cv2
    import io
    raw = Path(frame["image_path"]).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    with Image.open(io.BytesIO(raw)) as image:
        if image.size != (int(frame["original_width"]), int(frame["original_height"])):
            raise ValueError("target original image size differs from calibration")
        rgb = np.asarray(image.convert("RGB"))
    rgb = cv2.resize(rgb, (int(frame["width"]), int(frame["height"])), interpolation=cv2.INTER_AREA)
    return rgb.astype(np.float64) / 255, digest


def psnr(mse):
    # Exact zero has infinite PSNR; represent it as null in strict JSON.
    return None if mse == 0 else float(-10 * np.log10(mse))


def frame_metrics(prediction, truth):
    if prediction.shape != truth.shape or prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError("RGB metric requires matching HxWx3 arrays")
    delta = prediction.astype(np.float64) - truth.astype(np.float64)
    mse = float(np.mean(np.square(delta), dtype=np.float64))
    return {"mse": mse, "psnr": psnr(mse), "rgb_elements": int(delta.size),
            "pixels": int(delta.shape[0] * delta.shape[1])}


def score_split(inputs, split, *, selected=None):
    """Only this function opens target RGB; split filter precedes every read."""
    rows, image_hashes = [], {}
    for scene in SCENES:
        item = inputs[scene]
        for index, frame in enumerate(item["frames"]):
            if frame["split"] != split:
                continue
            truth, digest = read_target(frame)
            image_hashes[str(Path(frame["image_path"]).resolve())] = digest
            for method_index, method in enumerate(item["methods"]):
                if selected is not None and method not in selected:
                    continue
                rows.append(dict(scene=scene, split=split, frame_id=int(frame["id"]),
                                 name=frame["name"], method=method,
                                 **frame_metrics(item["predictions"][method_index, index], truth)))
    return rows, image_hashes


def aggregate(rows, method):
    by_scene = {}
    for scene in SCENES:
        subset = [row for row in rows if row["scene"] == scene and row["method"] == method]
        if len(subset) != 4 or len({row["frame_id"] for row in subset}) != 4:
            raise ValueError("expected exactly four scored frames per scene and method")
        mse = float(np.mean([row["mse"] for row in subset]))
        finite_psnr = [row["psnr"] for row in subset]
        by_scene[scene] = dict(mse=mse, psnr_from_mse=psnr(mse),
                               mean_frame_psnr=None if None in finite_psnr else float(np.mean(finite_psnr)),
                               frames=4, pixels=sum(row["pixels"] for row in subset))
    mse = float(np.mean([value["mse"] for value in by_scene.values()]))
    frame_psnrs = [row["psnr"] for row in rows if row["method"] == method]
    return dict(mse=mse, psnr_from_mse=psnr(mse),
                mean_frame_psnr=None if None in frame_psnrs else float(np.mean(frame_psnrs)), scenes=by_scene)


def select_methods(calibration_rows):
    table = {method: aggregate(calibration_rows, method) for method in METHODS}
    selected = {}
    for role in ROLES:
        candidates = [f"{role}_h{width}" for width in WIDTHS]
        selected[role] = min(candidates, key=lambda method: table[method]["mse"])
    selected.update(bilinear="bilinear", original_gs="original_gs")
    best_simple = min(SIMPLE, key=lambda role: table[selected[role]]["mse"])
    return selected, best_simple, table


def calibrate(runs, data, out):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite calibration output: {out}")
    inputs, provenance = load_inputs(runs, data)
    rows, image_hashes = score_split(inputs, "calibration")
    selected, best_simple, table = select_methods(rows)
    import cv2
    selection = dict(
        stage="calibration_locked_before_gate_rgb", runs=str(Path(runs).resolve()),
        data=str(Path(data).resolve()), scenes=list(SCENES), widths=list(WIDTHS),
        selected=selected, calibration_best_simple_role=best_simple,
        calibration=table, calibration_image_sha256=image_hashes,
        provenance=provenance, analysis_source_sha256=sha(__file__),
        image_transform=dict(decode="PIL convert RGB", resize="cv2.INTER_AREA", opencv=cv2.__version__,
                             value_range="sRGB bytes / 255; no linear-light conversion"),
        selection_rule="Per-role minimum two-scene calibration macro RGB MSE; width order breaks ties",
        primary_rule="Gate best simple role after independent calibration per role; relative MSE reduction >=2% and positive in both scenes",
        target_rgb_reads=dict(calibration=8, gate=0), random_training_replicates=1,
        scope="Two reused development scenes; given shared SfM cameras; no generalization confidence interval")
    out.mkdir(parents=True)
    write_json(out / "calibration_frame_metrics.json", rows)
    write_json(out / "selection.json", selection)
    (out / "selection.sha256").write_text(sha(out / "selection.json") + "\n")
    return selection


def compare(full, control):
    delta = control["mse"] - full["mse"]
    relative = None if control["mse"] == 0 else float(delta / control["mse"])
    scenes = {scene: control["scenes"][scene]["mse"] - full["scenes"][scene]["mse"] for scene in SCENES}
    return dict(absolute_mse_reduction=delta, relative_mse_reduction=relative,
                scene_absolute_reductions=scenes, scenes_improved=sum(value > 0 for value in scenes.values()),
                threshold_relative_reduction=0.02, passed=relative is not None and relative >= .02 and all(value > 0 for value in scenes.values()))


def evaluate(out):
    out = Path(out)
    if (out / "summary.json").exists() or (out / "gate_frame_metrics.json").exists():
        raise FileExistsError("Refusing to overwrite gate evaluation")
    selection_path = out / "selection.json"
    checked_hash(selection_path, (out / "selection.sha256").read_text().strip(), "selection")
    selection = json.loads(selection_path.read_text())
    checked_hash(__file__, selection["analysis_source_sha256"], "analysis source")
    inputs, provenance = load_inputs(selection["runs"], selection["data"])
    if provenance != selection["provenance"]:
        raise ValueError("frozen prediction/data/source provenance changed")
    for path, expected in selection["calibration_image_sha256"].items():
        checked_hash(path, expected, "calibration RGB")
    rows, image_hashes = score_split(inputs, "gate", selected=set(selection["selected"].values()))
    models = {role: dict(method=method, calibration=selection["calibration"][method], gate=aggregate(rows, method))
              for role, method in selection["selected"].items()}
    best_simple = min(SIMPLE, key=lambda role: models[role]["gate"]["mse"])
    calibration_simple = selection["calibration_best_simple_role"]
    primary = dict(control_role=best_simple, control_method=models[best_simple]["method"],
                   **compare(models["full"]["gate"], models[best_simple]["gate"]))
    summary = dict(
        scope=selection["scope"], scenes=list(SCENES), selected=selection["selected"], models=models,
        primary_gate=primary,
        calibration_locked_control=dict(control_role=calibration_simple,
                                        **compare(models["full"]["gate"], models[calibration_simple]["gate"])),
        original_gs_reference=compare(models["full"]["gate"], models["original_gs"]["gate"]),
        gate_image_sha256=image_hashes, selection_sha256=sha(selection_path),
        analysis_source_sha256=sha(__file__), primary_population="All RGB pixels in every predeclared gate frame",
        aggregation="Mean frame MSE within scene, then equal scene mean; PSNR-from-MSE and mean-frame PSNR kept distinct",
        limits=["One independently fitted GS per scene; deterministic readouts are not seed replicates",
                "Shared full-image SfM cameras; GS initialization and fitting use context RGB only",
                "Four calibration and four gate views in each of two reused development scenes",
                "Gate best simple control is conservative comparison, not extra method tuning",
                "Failure stops this branch; passing sampling gate does not establish a novel or general GS method"],
        verification=dict(passed=True, source_model_or_trainer_imports=False, hash_checked_inputs=True,
                          calibration_locked_before_gate_rgb=True, gate_rgb_images_read=8,
                          scored_prediction_frames=len(rows), any_historic_test_data_read=False))
    write_json(out / "gate_frame_metrics.json", rows)
    write_json(out / "summary.json", summary)
    lines = ["# Round 9 independent staged RGB analysis", "", summary["scope"], "",
             "Calibration selected each role's bandwidth before gate RGB was opened. All gate pixels are scored.", "",
             "| Role | Selected method | Calibration MSE | Gate MSE | Gate PSNR from macro MSE |", "|---|---|---:|---:|---:|"]
    for role, record in models.items():
        db = record["gate"]["psnr_from_mse"]
        lines.append(f"| {role} | {record['method']} | {record['calibration']['mse']:.9f} | {record['gate']['mse']:.9f} | {'inf' if db is None else f'{db:.4f}'} |")
    relative = primary["relative_mse_reduction"]
    lines += ["", f"Primary control: `{best_simple}`. Relative full MSE reduction: {'undefined' if relative is None else f'{100*relative:.4f}%'}; scenes improved: {primary['scenes_improved']}/2; gate passed: **{primary['passed']}**.", "", "Limits:", ""]
    lines += [f"- {item}" for item in summary["limits"]]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", choices=("all", "calibrate", "evaluate"), default="all")
    parser.add_argument("--runs", type=Path, default=Path("results/round9_gs_readout"))
    parser.add_argument("--data", type=Path, default=Path("results/round9_context_data"))
    parser.add_argument("--out", type=Path, default=Path("results/round9_analysis"))
    args = parser.parse_args()
    if args.stage in ("all", "calibrate"):
        value = calibrate(args.runs, args.data, args.out)
        print(json.dumps({"stage": "calibrated", "selected": value["selected"], "gate_rgb_reads": 0}), flush=True)
    if args.stage in ("all", "evaluate"):
        value = evaluate(args.out)
        print(json.dumps({"stage": "evaluated", "primary_gate": value["primary_gate"]}), flush=True)


if __name__ == "__main__":
    main()
