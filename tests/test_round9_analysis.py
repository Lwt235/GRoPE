import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from experiments import round9_analysis as analysis


def dump(path, value):
    Path(path).write_text(json.dumps(value))


@pytest.fixture
def experiment(tmp_path):
    pytest.importorskip("cv2", reason="Round-9 image protocol requires the MiniColmap OpenCV environment")
    runs, data = tmp_path / "runs", tmp_path / "data"
    source = tmp_path / "source.py"
    source.write_text("# frozen source\n")
    truth = 128 / 255
    for scene in analysis.SCENES:
        rd, dd = runs / scene, data / scene
        rd.mkdir(parents=True)
        dd.mkdir(parents=True)
        dataset = dd / "dataset.json"
        dataset.write_text("{}")
        context_hashes = {}
        for index in range(24):
            path = dd / f"context_{index}.png"
            Image.fromarray(np.full((6, 8, 3), 128, np.uint8)).save(path)
            context_hashes[str(path)] = analysis.sha(path)
        frames = []
        for index in range(8):
            path = dd / f"target_{index}.png"
            Image.fromarray(np.full((6, 8, 3), 128, np.uint8)).save(path)
            frames.append(dict(id=100 + index, name=path.name, image_path=str(path),
                               width=8, height=6, original_width=8, original_height=6,
                               split="calibration" if index % 2 == 0 else "gate"))
        dump(dd / "target_cameras.json", dict(frames=frames))
        np.savez(dd / "context.npz", ids=np.arange(24))
        manifest = dict(scene=scene, split=dict(context_ids=list(range(24)), target_ids=list(range(100, 108))),
                        source_image_sha256=context_hashes, source_dataset=str(dataset),
                        source_dataset_sha256=analysis.sha(dataset), target_rgb_reads=0,
                        old_geometry_keypoints_tracks_read=False,
                        target_cameras_sha256=analysis.sha(dd / "target_cameras.json"),
                        context_archive_sha256=analysis.sha(dd / "context.npz"))
        dump(dd / "manifest.json", manifest)
        predictions = np.empty((18, 8, 6, 8, 3), np.float32)
        for m, method in enumerate(analysis.METHODS):
            role = method.split("_h")[0]
            for index in range(8):
                if index % 2 == 0:
                    error = .02 + (.01 * analysis.WIDTHS.index(method.split("_h")[1]) if "_h" in method else .04)
                else:
                    error = dict(full=.03, isotropic=.04, fixed=.05, point=.06, bilinear=.07, original_gs=.02)[role]
                predictions[m, index] = truth + error
        np.savez_compressed(rd / "predictions.npz", predictions=predictions,
                            methods=np.asarray(analysis.METHODS), target_ids=np.arange(100, 108),
                            target_splits=np.asarray([f["split"] for f in frames]))
        dump(rd / "run.json", dict(predictions_sha256=analysis.sha(rd / "predictions.npz"),
                                   source_hashes={str(source): analysis.sha(source)},
                                   data_manifest_sha256=analysis.sha(dd / "manifest.json"),
                                   target_cameras_sha256=analysis.sha(dd / "target_cameras.json"),
                                   context_archive_sha256=analysis.sha(dd / "context.npz")))
    return dict(runs=runs, data=data, out=tmp_path / "analysis", source=source)


def test_staged_analysis_never_reads_gate_before_selection(experiment, monkeypatch):
    reads = []
    original = analysis.read_target

    def observe(frame):
        if frame["split"] == "gate":
            assert (experiment["out"] / "selection.json").exists()
            assert (experiment["out"] / "selection.sha256").exists()
        reads.append(frame["split"])
        return original(frame)

    monkeypatch.setattr(analysis, "read_target", observe)
    selection = analysis.calibrate(experiment["runs"], experiment["data"], experiment["out"])
    assert reads == ["calibration"] * 8
    assert selection["selected"]["full"] == "full_h0.25"
    summary = analysis.evaluate(experiment["out"])
    assert reads == ["calibration"] * 8 + ["gate"] * 8
    assert summary["primary_gate"]["control_role"] == "isotropic"
    assert summary["primary_gate"]["passed"]
    assert summary["primary_gate"]["relative_mse_reduction"] == pytest.approx(1 - .03**2 / .04**2, abs=1e-6)
    assert not summary["original_gs_reference"]["passed"]
    assert summary["verification"]["scored_prediction_frames"] == 48


@pytest.mark.parametrize("mutation", ["source", "prediction", "calibration_rgb", "selection"])
def test_lock_mutations_fail_before_any_gate_read(experiment, monkeypatch, mutation):
    analysis.calibrate(experiment["runs"], experiment["data"], experiment["out"])
    paths = dict(source=experiment["source"], prediction=experiment["runs"] / "teatime/predictions.npz",
                 calibration_rgb=experiment["data"] / "teatime/target_0.png",
                 selection=experiment["out"] / "selection.json")
    with paths[mutation].open("ab") as handle:
        handle.write(b"changed")
    monkeypatch.setattr(analysis, "read_target", lambda frame: pytest.fail("must fail before reading gate RGB"))
    with pytest.raises(ValueError, match="hash changed"):
        analysis.evaluate(experiment["out"])


def test_gate_rgb_does_not_affect_calibration_choice(experiment):
    for scene in analysis.SCENES:
        for index in (1, 3, 5, 7):
            (experiment["data"] / scene / f"target_{index}.png").unlink()
    selection = analysis.calibrate(experiment["runs"], experiment["data"], experiment["out"])
    assert selection["target_rgb_reads"] == dict(calibration=8, gate=0)


def test_full_pixel_metric_not_selected_region():
    truth = np.zeros((2, 2, 3))
    prediction = truth.copy()
    prediction[0, 0] = 1
    result = analysis.frame_metrics(prediction, truth)
    assert result["mse"] == .25
    assert result["pixels"] == 4
    assert result["rgb_elements"] == 12


def test_two_scene_gate_rejects_one_scene_regression():
    control = dict(mse=.1, scenes={"teatime": dict(mse=.1), "waldo_kitchen": dict(mse=.1)})
    full = dict(mse=.08, scenes={"teatime": dict(mse=.05), "waldo_kitchen": dict(mse=.11)})
    result = analysis.compare(full, control)
    assert result["relative_mse_reduction"] == pytest.approx(.2)
    assert result["scenes_improved"] == 1
    assert not result["passed"]


def test_zero_error_serializable_and_does_not_count_as_improvement():
    perfect = dict(mse=0, scenes={scene: dict(mse=0) for scene in analysis.SCENES})
    assert analysis.psnr(0) is None
    result = analysis.compare(perfect, perfect)
    assert result["relative_mse_reduction"] is None
    assert not result["passed"]
    json.dumps(result, allow_nan=False)


def test_repeated_output_is_refused(experiment):
    analysis.calibrate(experiment["runs"], experiment["data"], experiment["out"])
    with pytest.raises(FileExistsError):
        analysis.calibrate(experiment["runs"], experiment["data"], experiment["out"])
    analysis.evaluate(experiment["out"])
    with pytest.raises(FileExistsError):
        analysis.evaluate(experiment["out"])


@pytest.mark.parametrize("field", ["methods", "target_ids", "target_splits", "predictions"])
def test_prediction_contract_is_enforced(experiment, field):
    path = experiment["runs"] / "teatime/predictions.npz"
    with np.load(path) as archive:
        arrays = {key: np.array(archive[key]) for key in archive.files}
    if field == "methods":
        arrays[field][0] = arrays[field][1]
    elif field == "target_ids":
        arrays[field][0] = 999
    elif field == "target_splits":
        arrays[field][0] = "gate"
    else:
        arrays[field][0, 0, 0, 0, 0] = np.nan
    np.savez_compressed(path, **arrays)
    run_path = path.parent / "run.json"
    run = json.loads(run_path.read_text())
    run["predictions_sha256"] = analysis.sha(path)
    dump(run_path, run)
    with pytest.raises(ValueError):
        analysis.calibrate(experiment["runs"], experiment["data"], experiment["out"])
