"""Second independent full-pixel recomputation after the locked Round-9 gate.

Does not import the analyzer, model, renderer or training code. Scores all
calibration configurations and only the six already-selected gate methods.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(runs, data, analysis, out):
    runs, data, analysis, out = map(Path, (runs, data, analysis, out))
    if out.exists():
        raise FileExistsError(out)
    selection = json.loads((analysis / "selection.json").read_text())
    summary = json.loads((analysis / "summary.json").read_text())
    assert sha(analysis / "selection.json") == summary["selection_sha256"]
    assert sha(analysis / "selection.json") == (analysis / "selection.sha256").read_text().strip()
    scenes = ["teatime", "waldo_kitchen"]
    names = [f"{role}_h{h}" for role in ("full", "isotropic", "fixed", "point") for h in ("0.25", "0.5", "1", "2")] + ["bilinear", "original_gs"]
    selected = selection["selected"]
    selected_gate_names = set(selected.values())
    image_hashes = dict(selection["calibration_image_sha256"], **summary["gate_image_sha256"])
    rows, coverage = [], {}
    for scene in scenes:
        frames = json.loads((data / scene / "target_cameras.json").read_text())["frames"]
        with np.load(runs / scene / "predictions.npz", allow_pickle=False) as archive:
            predictions = np.array(archive["predictions"])
            method_names = archive["methods"].tolist()
            alpha = np.array(archive["alpha"], np.float64)
            fallback = np.array(archive["fallback_contribution"], np.float64)
        assert method_names == names
        gate_indices = [i for i, frame in enumerate(frames) if frame["split"] == "gate"]
        coverage[scene] = dict(gate_mean_alpha=float(np.mean(alpha[gate_indices])),
            gate_mean_fallback_contribution=float(np.mean(fallback[gate_indices])),
            gate_fallback_fraction_of_alpha=float(np.sum(fallback[gate_indices]) / np.sum(alpha[gate_indices])))
        for frame_index, frame in enumerate(frames):
            path = Path(frame["image_path"])
            assert sha(path) == image_hashes[str(path.resolve())]
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"))
            target = cv2.resize(rgb, (frame["width"], frame["height"]), interpolation=cv2.INTER_AREA)
            target = target.astype(np.float64) / 255
            for index, name in enumerate(names):
                if frame["split"] == "gate" and name not in selected_gate_names:
                    continue
                residual = predictions[index, frame_index].astype(np.float64) - target
                mse = float(np.sum(residual * residual, dtype=np.float64) / residual.size)
                rows.append(dict(scene=scene, split=frame["split"], id=frame["id"], method=name,
                                 mse=mse, psnr=float(-10 * np.log10(mse))))

    def metric(name, split, scene=None):
        local = [row["mse"] for row in rows if row["method"] == name and row["split"] == split and (scene is None or row["scene"] == scene)]
        assert len(local) == (8 if scene is None else 4)
        return float(np.mean(local))

    # Equal four-frame counts mean the overall frame mean equals scene macro.
    choices = {}
    for role in ("full", "isotropic", "fixed", "point"):
        candidates = [f"{role}_h{h}" for h in ("0.25", "0.5", "1", "2")]
        choices[role] = min(candidates, key=lambda name: metric(name, "calibration"))
    choices.update(bilinear="bilinear", original_gs="original_gs")
    assert choices == selected
    controls = ("isotropic", "fixed", "point", "bilinear")
    best_calibration = min(controls, key=lambda role: metric(choices[role], "calibration"))
    best_gate = min(controls, key=lambda role: metric(choices[role], "gate"))
    assert best_calibration == selection["calibration_best_simple_role"]
    assert best_gate == summary["primary_gate"]["control_role"]
    full_mse = metric(choices["full"], "gate")
    control_mse = metric(choices[best_gate], "gate")
    gains = {scene: metric(choices[best_gate], "gate", scene) - metric(choices["full"], "gate", scene) for scene in scenes}
    reduction = (control_mse - full_mse) / control_mse
    gate = reduction >= .02 and all(value > 0 for value in gains.values())
    assert bool(gate) == summary["primary_gate"]["passed"]
    all_errors = []
    for name in names:
        reference = selection["calibration"][name]
        all_errors.append(abs(metric(name, "calibration") - reference["mse"]))
        for scene in scenes:
            all_errors.append(abs(metric(name, "calibration", scene) - reference["scenes"][scene]["mse"]))
    for role, name in choices.items():
        reference = summary["models"][role]["gate"]
        all_errors.append(abs(metric(name, "gate") - reference["mse"]))
        for scene in scenes:
            all_errors.append(abs(metric(name, "gate", scene) - reference["scenes"][scene]["mse"]))
    frame_references = json.loads((analysis / "calibration_frame_metrics.json").read_text()) + json.loads((analysis / "gate_frame_metrics.json").read_text())
    reference_lookup = {(row["scene"], row["split"], row["frame_id"], row["method"]): row for row in frame_references}
    for row in rows:
        reference = reference_lookup[(row["scene"], row["split"], row["id"], row["method"])]
        all_errors += [abs(row["mse"] - reference["mse"]), abs(row["psnr"] - reference["psnr"])]
    assert max(all_errors) < 1e-12
    table = {}
    for role, name in choices.items():
        table[role] = dict(calibration_mse=metric(name, "calibration"), gate_mse=metric(name, "gate"),
                           gate_scene_mse={scene: metric(name, "gate", scene) for scene in scenes})
    comparisons = {}
    for role in (*controls, "original_gs"):
        baseline = metric(choices[role], "gate")
        comparisons[role] = dict(relative_macro_mse_reduction=(baseline - full_mse) / baseline,
            per_scene_relative_mse_reduction={scene: 1 - metric(choices["full"], "gate", scene) / metric(choices[role], "gate", scene) for scene in scenes})
    result = dict(passed=True, independent_metric_and_selection_recomputation=True,
                  imports_model_trainer_or_analyzer=False, scored_frames=len(rows), maximum_absolute_metric_error=max(all_errors),
                  selected=choices, calibration_best_control=best_calibration, gate_best_control=best_gate,
                  primary_gate=dict(relative_macro_mse_reduction=reduction, scene_absolute_reductions=gains, passed=bool(gate)),
                  metrics=table, comparisons=comparisons, coverage=coverage,
                  source_sha256=sha(__file__), selection_sha256=sha(analysis / "selection.json"),
                  summary_sha256=sha(analysis / "summary.json"), target_rgb_images_replayed=16,
                  any_historic_test_read=False)
    out.mkdir(parents=True)
    (out / "verification.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("results/round9_gs_readout"))
    parser.add_argument("--data", type=Path, default=Path("results/round9_context_data"))
    parser.add_argument("--analysis", type=Path, default=Path("results/round9_analysis"))
    parser.add_argument("--out", type=Path, default=Path("results/round9_metric_audit"))
    args = parser.parse_args()
    result = audit(args.runs, args.data, args.analysis, args.out)
    print(json.dumps({key: result[key] for key in ("passed", "maximum_absolute_metric_error", "primary_gate", "comparisons", "coverage")}))
