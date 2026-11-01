"""Replay completed five-seed isometric PE confirmations without test tuning.

Real-data comparisons use the runner's >=95% retained-label-mass mask. The
strong geometric baseline is frozen from the OLD geometry-development sweep,
including its validation-preferred score mode. Camera and training-seed
uncertainty are reported separately and jointly; rows are not independent
experimental replicates. Probability archives are float32, metrics float64.

The validation_baseline stage never opens test NPZ files. Confirmation stages
refuse manifests without elapsed_seconds or an entire declared seed/method
Cartesian product. The default all stage checks BOTH runs before test access.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from torch import nn

from experiments.matching_probe import digest
from experiments import rendered_data
from experiments.rendered_metrics import metrics as point_metrics
from experiments.region_pe_probe import real_dataset, evaluate as region_evaluate
from experiments.real_support_geometry_probe import normalised_log_probability
from models.gaussian_pe import GaussianPEMatcher


SEEDS = [11, 22, 33, 44, 55]
ROW_METRICS = ["cross_entropy", "kl", "js", "top1_target_mass",
               "dominant_region_accuracy", "top1_max_bin_accuracy"]
BASELINE_METHODS = ["projected_center_rbf", "pixel_center_rbf"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def function_ast_hashes(source, names):
    definitions = {node.name: node for node in ast.parse(source).body
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    return {name: hashlib.sha256(ast.dump(definitions[name], include_attributes=False).encode()).hexdigest()
            for name in names}


def load_completed_manifest(root, domain):
    """No dataset files are opened before this completeness check."""
    path = Path(root) / "manifest.json"
    contents = path.read_bytes()
    manifest = json.loads(contents)
    manifest["_analysis_input_sha256"] = hashlib.sha256(contents).hexdigest()
    if "elapsed_seconds" not in manifest:
        raise ValueError(f"{root}: run is not complete (elapsed_seconds absent)")
    config = manifest["config"]
    seeds = [int(s) for s in config["seeds"].split(",")]
    if seeds != SEEDS or config["phase"] != "confirmation":
        raise ValueError(f"{root}: expected all five confirmation seeds {SEEDS}")
    methods = config["methods"].split(",")
    if domain == "real":
        if config["domain"] != "real":
            raise ValueError("real analysis requires the real-domain runner")
        expected = {(s, m) for s in seeds for m in methods}
        actual = [(r["seed"], r["method"]) for r in manifest["records"]]
    else:
        regimes = config["regimes"].split(",")
        expected = {(s, m, r) for s in seeds for m in methods for r in regimes}
        actual = [(r["seed"], r["method"], r["regime"]) for r in manifest["records"]]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError(f"{root}: missing or duplicate seed/method/regime records")
    return manifest


def verify_source_snapshot(root, manifest, critical):
    archived, live = {}, {}
    for name, expected in manifest["source_hashes"].items():
        path = Path(root) / "source" / name
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"archived source hash mismatch: {path}")
        archived[name] = actual
    for dependency in critical:
        name, functions = dependency if isinstance(dependency, tuple) else (dependency, None)
        actual = sha256(name)
        expected = manifest["source_hashes"].get(name)
        if functions is not None:
            archived_functions = function_ast_hashes((Path(root) / "source" / name).read_text(), functions)
            live_functions = function_ast_hashes(Path(name).read_text(), functions)
            if archived_functions != live_functions:
                raise ValueError(f"live replay helper differs from training: {name}")
            live[name] = dict(archived_file_sha256=expected, live_file_sha256=actual,
                              used_function_ast_sha256=live_functions,
                              check="used helper bodies match; unused training factory may differ")
            continue
        if actual != expected:
            raise ValueError(f"live replay dependency differs from training: {name}")
        live[name] = actual
    return dict(archived_files_verified=len(archived), live_dependencies=live)


def frozen_baselines(out):
    development_path = Path("results/round3_real_geometry_dev/summary.json")
    development = json.loads(development_path.read_text())
    original_freeze = json.loads(Path(
        "results/round3_real_geometry_confirmation/frozen_settings.json").read_text())
    if sha256(development_path) != original_freeze["development_summary_sha256"]:
        raise ValueError("old geometry-development summary changed")
    selected = {}
    for method in BASELINE_METHODS:
        mode = min(("normalise_kernel", "softmax_kernel"), key=lambda mode:
                   development["summary"][f"{method}/{mode}/validation/all"]["soft_ce"])
        key = f"{method}/{mode}"
        selected[method] = dict(mode=mode, **original_freeze["selected"][key])
    result = dict(development_summary=str(development_path),
                  development_summary_sha256=sha256(development_path),
                  mode_selection="minimum OLD aggregate development validation CE",
                  no_new_validation_or_test_tuning=True, selected=selected)
    path = Path(out) / "baseline_settings.json"
    if path.exists() and json.loads(path.read_text()) != result:
        raise ValueError("existing frozen baseline settings differ")
    write_json(path, result)
    return result


def aligned_real_inputs(root, scene, split, alternate=False):
    data, metadata = real_dataset(root, scene, split, alternate)
    folder = Path(root) / scene
    manifest = json.loads((folder / "manifest.json").read_text())
    pairs = [r for r in manifest["pairs"] if r["split"] == split]
    q_fit, q_pixel, k_fit, hashes = [], [], [], {}
    suffix = "_alt" if alternate else ""
    for record in pairs:
        path = folder / record["file"]
        actual = sha256(path)
        if actual != record["npz_sha256"]:
            raise ValueError(f"real data hash mismatch: {path}")
        hashes[str(path)] = actual
        with np.load(path, allow_pickle=False) as raw:
            q_fit.append(raw["q_projected_mean" + suffix].astype(np.float64) / 24.)
            q_pixel.append(raw["q_pixel_mean"].astype(np.float64) / 24.)
            k_fit.append(raw["k_projected_mean" + suffix].astype(np.float64) / 24.)
    metadata["source_npz_sha256_verified"] = hashes
    means = dict(projected_center_rbf=np.stack(q_fit), pixel_center_rbf=np.stack(q_pixel),
                 key=np.stack(k_fit))
    return data, metadata, pairs, means


def real_probability_metrics(probability, data):
    p = np.asarray(probability, np.float64)
    y = data["soft_labels"].cpu().numpy().astype(np.float64)
    valid = data["label_valid"].cpu().numpy()
    ce = -(y * np.log(np.maximum(p, 1e-30))).sum(-1)
    entropy = -(y * np.log(np.maximum(y, 1e-30))).sum(-1)
    midpoint = (p + y) / 2
    js = .5 * (y * np.log(np.maximum(y, 1e-30) / np.maximum(midpoint, 1e-30)) +
               p * np.log(np.maximum(p, 1e-30) / np.maximum(midpoint, 1e-30))).sum(-1)
    prediction = p.argmax(-1)
    mass = np.take_along_axis(y, prediction[..., None], -1)[..., 0]
    rows = dict(cross_entropy=ce, kl=ce-entropy, js=js, top1_target_mass=mass,
                dominant_region_accuracy=(prediction == y.argmax(-1)).astype(float),
                top1_max_bin_accuracy=(mass >= y.max(-1) - 1e-8).astype(float),
                target_entropy=entropy)
    summary = {key: float(value[valid].mean()) for key, value in rows.items()}
    summary["valid_query_rows"] = int(valid.sum())
    return summary, rows


def baseline_probability(data, means, method, configuration):
    delta = means[method][:, :, None] - means["key"][:, None]
    kernel = -.5 * np.sum(delta ** 2, axis=-1) / float(configuration["bandwidth"]) ** 2
    key_valid = data["key_valid"].cpu().numpy()
    logp = np.stack([normalised_log_probability(k, valid, configuration["mode"],
                                               configuration["temperature"])
                     for k, valid in zip(kernel, key_valid)])
    return np.exp(logp), logp


class SuppliedScores(nn.Module):
    def forward(self, data):
        scores = data["_audit_score"]
        return torch.cat([scores, torch.zeros_like(scores[..., :1])], -1)


def compare_summary(replayed, expected, tolerance=3e-6):
    errors = {}
    for metric, value in expected.items():
        if metric not in replayed or value is None:
            continue
        error = abs(float(replayed[metric]) - float(value))
        if error > tolerance:
            raise ValueError(f"checkpoint metric replay mismatch: {metric}: {error}")
        errors[metric] = error
    return errors


def validation_baseline(args, settings):
    outputs = {}
    for scene in ("teatime", "waldo_kitchen"):
        data, metadata, _, means = aligned_real_inputs(args.real_root, scene, "validation")
        for method, parameters in settings["selected"].items():
            probability, logp = baseline_probability(data, means, method, parameters)
            summary, _ = real_probability_metrics(probability, data)
            supplied = dict(data, _audit_score=torch.from_numpy(logp))
            expected, _ = region_evaluate(SuppliedScores(), supplied)
            errors = compare_summary(summary, expected, tolerance=2e-12)
            outputs[f"{scene}/{method}"] = dict(summary=summary,
                metric_error_vs_training_evaluator=errors, metadata=metadata)
    write_json(args.out / "validation_baseline.json", dict(
        stage="validation only; no checkpoint or test NPZ opened", results=outputs))
    print("Validation-baseline metrics agree with the training evaluator.", flush=True)


def real_row_identity(pairs, valid, scene):
    pair_index, query_index = np.nonzero(valid)
    identities = [dict(scene=scene, q_id=int(pairs[p]["q_id"]),
                       k_id=int(pairs[p]["k_id"]), query_region=int(q),
                       pair_file=pairs[p]["file"]) for p, q in zip(pair_index, query_index)]
    return identities


def paired_camera_seed_statistics(a, b, identities, bootstrap=4000):
    """Crossed seed/camera bootstrap on identical real evaluation rows.

    Same camera draws are shared across training seeds, since they evaluate the
    same photographs. Scenes are fixed strata, not population-level draws.
    """
    differences = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    if differences.ndim != 2 or differences.shape != (len(SEEDS), len(identities)):
        raise ValueError("expected five paired seed-by-row arrays")
    clusters = sorted({(r["scene"], int(r["q_id"])) for r in identities})
    index = {key: i for i, key in enumerate(clusters)}
    membership = np.array([index[(r["scene"], int(r["q_id"]))] for r in identities])
    counts = np.bincount(membership, minlength=len(clusters))
    sums = np.stack([np.bincount(membership, weights=d, minlength=len(clusters))
                     for d in differences])
    generator = np.random.default_rng(219782)
    seed_draws = generator.integers(0, len(SEEDS), (bootstrap, len(SEEDS)))
    camera_draws = []
    for scene in sorted({c[0] for c in clusters}):
        candidates = np.array([i for i, cluster in enumerate(clusters) if cluster[0] == scene])
        camera_draws.append(candidates[generator.integers(0, len(candidates),
                                                        (bootstrap, len(candidates)))])
    camera_draws = np.concatenate(camera_draws, axis=1)
    denominator = counts[camera_draws].sum(-1)
    crossed = sums[seed_draws[:, :, None], camera_draws[:, None, :]].sum((1, 2))
    crossed /= len(SEEDS) * denominator
    seed_means = differences.mean(-1)
    seed_only = seed_means[seed_draws].mean(-1)
    camera_only = sums.mean(0)[camera_draws].sum(-1) / denominator
    return dict(mean=float(seed_means.mean()),
                seed_differences={str(s): float(v) for s, v in zip(SEEDS, seed_means)},
                ci95_seed=np.quantile(seed_only, [.025, .975]).tolist(),
                ci95_camera=np.quantile(camera_only, [.025, .975]).tolist(),
                ci95_seed_camera=np.quantile(crossed, [.025, .975]).tolist(),
                positive_seed_fraction=float(np.mean(seed_means > 0)),
                query_cameras=len(clusters), rows=len(identities), bootstrap=bootstrap,
                estimand="mean across seeds of row-weighted paired metric difference",
                scope="fixed observed scenes; camera bootstrap stratified within scene")


@torch.no_grad()
def model_probability(model, data, batch=32):
    probabilities = []
    for start in range(0, len(data["q"]), batch):
        part = {key: value[start:start+batch] for key, value in data.items()}
        score = model(part)[..., :-1]
        score = score.masked_fill(~part["key_valid"][..., None, :], -10000.)
        probabilities.append(score.softmax(-1).cpu().numpy())
    return np.concatenate(probabilities)


def load_checkpoint(root, record, config, hashes, point=False):
    name = (f"{record['regime']}-{record['seed']}-{record['method']}.pt" if point else
            f"{record['seed']}-{record['method']}.pt")
    path = Path(root) / "checkpoints" / name
    checksum = sha256(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["method"] != record["method"] or state["features"] != config["features"]:
        raise ValueError(f"checkpoint metadata mismatch: {path}")
    if "resolution" in state and abs(state["resolution"] - config["resolution"]) > 1e-12:
        raise ValueError(f"checkpoint resolution mismatch: {path}")
    model = GaussianPEMatcher(record["method"], config["features"], config["resolution"])
    model.load_state_dict(state["state_dict"], strict=True)
    if not torch.isfinite(model.resolution) or abs(float(model.resolution) - config["resolution"]) > 1e-6:
        raise ValueError("checkpoint resolution buffer differs from run config")
    if sha256(path) != checksum:
        raise ValueError(f"checkpoint changed while loading: {path}")
    hashes[str(path)] = checksum
    return model.eval()


def checkpoint_hash_final_check(hashes):
    for path, expected in hashes.items():
        if sha256(path) != expected:
            raise ValueError(f"checkpoint changed during replay: {path}")


def manifest_hash_final_check(root, manifest):
    expected = manifest["_analysis_input_sha256"]
    if sha256(Path(root) / "manifest.json") != expected:
        raise ValueError(f"training manifest changed during replay: {root}")
    return expected


def analyse_real(args, manifest, settings):
    root, out = args.real_run, args.out
    config = manifest["config"]
    source = verify_source_snapshot(root, manifest, [
        "models/gaussian_pe.py", "models/geometry.py",
        ("experiments/region_pe_probe.py", ("real_dataset", "evaluate")),
        "experiments/matching_probe.py"])
    methods = config["methods"].split(",")
    if "isometric_shape" not in methods:
        raise ValueError("confirmation does not contain isometric_shape")
    datasets, metadata, pair_records, position = {}, {}, {}, {}
    for scene in ("teatime", "waldo_kitchen"):
        for alternate in (False, True):
            condition = scene + ("_alt" if alternate else "")
            data, meta, pairs, means = aligned_real_inputs(config["real_root"], scene, "test", alternate)
            datasets[condition], metadata[condition] = data, meta
            pair_records[condition], position[condition] = pairs, means
            expected = {manifest["datasets"][str(seed)]["test"][condition] for seed in SEEDS}
            if expected != {digest(data)}:
                raise ValueError(f"replayed real dataset digest mismatch: {condition}")
            for seed in SEEDS:
                saved_meta = manifest["datasets"][str(seed)]["meta"]["test"][condition]
                if (saved_meta["scene_manifest_sha256"] != meta["scene_manifest_sha256"]
                        or saved_meta["pairs"] != meta["pairs"]):
                    raise ValueError(f"real camera/row metadata changed: {condition}")
    measurement, identities = {}, {}
    for condition, data in datasets.items():
        scene = metadata[condition]["scene"]
        valid = data["label_valid"].numpy()
        identities[condition] = real_row_identity(pair_records[condition], valid, scene)
        measurement[condition] = {m: {metric: [] for metric in ROW_METRICS}
                                  for m in methods + BASELINE_METHODS}
    archive = out / "predictions" / "real"
    archive.mkdir(parents=True, exist_ok=True)
    summaries, hashes, replay_errors, all_csv_rows = [], {}, [], []
    for method, parameters in settings["selected"].items():
        for condition, data in datasets.items():
            probability, _ = baseline_probability(data, position[condition], method, parameters)
            summary, rows = real_probability_metrics(probability, data)
            summaries.append(dict(method=method, seed="fixed_geometry", condition=condition, **summary))
            valid = data["label_valid"].numpy()
            for metric in ROW_METRICS:
                measurement[condition][method][metric] = [rows[metric][valid]] * len(SEEDS)
            np.savez_compressed(archive / f"fixed-{method}-{condition}.npz",
                                probability=probability[valid].astype(np.float32),
                                targets=data["soft_labels"].numpy()[valid],
                                **{metric: rows[metric][valid] for metric in ROW_METRICS})
            for i, identity in enumerate(identities[condition]):
                all_csv_rows.append(dict(method=method, seed="fixed_geometry", condition=condition,
                    **identity, **{metric: float(rows[metric][valid][i]) for metric in ROW_METRICS}))
    records = sorted(manifest["records"], key=lambda r: (r["seed"], methods.index(r["method"])))
    for record in records:
        model = load_checkpoint(root, record, config, hashes)
        for condition, data in datasets.items():
            probability = model_probability(model, data)
            summary, rows = real_probability_metrics(probability, data)
            errors = compare_summary(summary, record["test"][condition])
            replay_errors.append(dict(seed=record["seed"], method=record["method"],
                                      condition=condition, errors=errors))
            summaries.append(dict(seed=record["seed"], method=record["method"],
                                  condition=condition, selected_step=record["step"], **summary))
            valid = data["label_valid"].numpy()
            for metric in ROW_METRICS:
                measurement[condition][record["method"]][metric].append(rows[metric][valid])
            np.savez_compressed(archive / f"{record['seed']}-{record['method']}-{condition}.npz",
                                probability=probability[valid], targets=data["soft_labels"].numpy()[valid],
                                **{metric: rows[metric][valid] for metric in ROW_METRICS})
            for i, identity in enumerate(identities[condition]):
                all_csv_rows.append(dict(seed=record["seed"], method=record["method"],
                    condition=condition, **identity,
                    **{metric: float(rows[metric][valid][i]) for metric in ROW_METRICS}))
        print(f"real replay seed={record['seed']} method={record['method']}", flush=True)
    for condition in measurement:
        for method in measurement[condition]:
            for metric in ROW_METRICS:
                measurement[condition][method][metric] = np.stack(measurement[condition][method][metric])
    for suffix in ("", "_alt"):
        condition = "pooled" + suffix
        sources = [scene + suffix for scene in ("teatime", "waldo_kitchen")]
        identities[condition] = sum((identities[c] for c in sources), [])
        measurement[condition] = {method: {metric: np.concatenate(
            [measurement[c][method][metric] for c in sources], axis=1)
            for metric in ROW_METRICS} for method in methods + BASELINE_METHODS}
    differences, aggregate = {}, {}
    for condition, values in measurement.items():
        aggregate[condition] = {}
        for method, metrics in values.items():
            aggregate[condition][method] = {metric: dict(
                mean=float(array.mean()), per_seed={str(seed): float(v) for seed, v in
                                                  zip(SEEDS, array.mean(1))})
                for metric, array in metrics.items()}
        for control in methods + BASELINE_METHODS:
            if control == "isometric_shape":
                continue
            for metric in ROW_METRICS:
                key = f"{condition}/isometric_shape_minus_{control}/{metric}"
                differences[key] = paired_camera_seed_statistics(
                    values["isometric_shape"][metric], values[control][metric],
                    identities[condition], args.bootstrap)
    checkpoint_hash_final_check(hashes)
    with (out / "real_per_row.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_csv_rows[0]))
        writer.writeheader()
        writer.writerows(all_csv_rows)
    output = dict(protocol=dict(
        stage="completed five-seed checkpoint replay", seeds=SEEDS,
        mask=">=95% retained target mass; same as training runner", source=source,
        checkpoint_hash_scope="first replay hash, verified unchanged after replay; training did not prestore hashes",
        scene_scope="teatime fitting; teatime and waldo_kitchen test cameras; conditional supplied reconstruction",
        prediction_archive="float32 probabilities for valid rows, float64 metrics; CSV row order indexes each archive",
        lower_is_better=["cross_entropy", "kl", "js"],
        fixed_baseline="duplicated across seeds for paired comparisons; not five independently trained baselines"),
        training_manifest_sha256=manifest_hash_final_check(root, manifest), metadata=metadata,
        baseline_settings=settings, summaries=summaries, aggregate=aggregate,
        differences=differences, checkpoint_sha256=hashes,
        replay_errors=replay_errors)
    write_json(out / "real_summary.json", output)
    return output


def paired_point_statistics(numerator, denominator, bootstrap=4000):
    """Nested seed -> independent synthetic scenes bootstrap for a difference."""
    numerator, denominator = np.asarray(numerator), np.asarray(denominator)
    if numerator.shape != denominator.shape or numerator.shape[0] != len(SEEDS):
        raise ValueError("expected paired seed-by-independent-scene arrays")
    generator = np.random.default_rng(319782)
    seed_draws = generator.integers(0, len(SEEDS), (bootstrap, len(SEEDS)))
    scene_draws = generator.integers(0, numerator.shape[1],
                                    (bootstrap, len(SEEDS), numerator.shape[1]))
    selected_n = numerator[seed_draws[..., None], scene_draws].sum(-1)
    selected_d = denominator[seed_draws[..., None], scene_draws].sum(-1)
    statistics = np.divide(selected_n, selected_d, out=np.zeros_like(selected_n),
                           where=selected_d > 0).mean(-1)
    means = numerator.sum(-1) / denominator.sum(-1)
    return dict(mean=float(means.mean()),
                seed_differences={str(s): float(v) for s, v in zip(SEEDS, means)},
                ci95_seed=np.quantile(means[seed_draws].mean(-1), [.025, .975]).tolist(),
                ci95_seed_scene=np.quantile(statistics, [.025, .975]).tolist(),
                positive_seed_fraction=float(np.mean(means > 0)),
                scenes_per_seed=numerator.shape[1], seeds=len(SEEDS), bootstrap=bootstrap,
                scope="independent synthetic scenes within independently sampled training-seed datasets")


def make_point_test(seed, config, predictor):
    namespace = 2100
    cache = {}
    datasets = {condition: rendered_data.make_dataset(namespace+200+seed,
                 config["test_scenes"], condition, cache)[0] for condition in rendered_data.CONDITIONS}
    from experiments.rgb_geometry_probe import observation_from_rgb
    original = rendered_data.patch_observation
    try:
        rendered_data.patch_observation = lambda view, cx, cy, rng, condition: observation_from_rgb(
            predictor, view, cx, cy, rng, condition)
        datasets["rgb_pred"] = rendered_data.make_dataset(namespace+200+seed,
            config["test_scenes"], "rgb_pred", cache)[0]
    finally:
        rendered_data.patch_observation = original
    return datasets


@torch.no_grad()
def point_probability(model, data, batch=32):
    outputs, loss = [], []
    for start in range(0, len(data["q"]), batch):
        part = {key: value[start:start+batch] for key, value in data.items()}
        logits = model(part)
        outputs.append(logits.softmax(-1).cpu().numpy())
        loss.append(torch.nn.functional.cross_entropy(logits.flatten(0, 1),
                    part["labels"].flatten(), reduction="none").reshape_as(part["labels"]).cpu().numpy())
    return np.concatenate(outputs), np.concatenate(loss)


def analyse_point(args, manifest):
    root, out = args.point_run, args.out
    config = manifest["config"]
    source = verify_source_snapshot(root, manifest, ["models/gaussian_pe.py", "models/geometry.py",
        "experiments/rendered_data.py", "experiments/rendered_scene.py",
        "experiments/rgb_geometry_probe.py", "experiments/matching_probe.py"])
    from experiments.rgb_geometry_probe import RGBGeometry
    predictor_path = Path("results/round2_rgb_geometry/checkpoints/geometry-11.pt")
    if sha256(predictor_path) != manifest["predictor_sha256"]:
        raise ValueError("frozen RGB geometry predictor changed")
    predictor = RGBGeometry()
    predictor.load_state_dict(torch.load(predictor_path, map_location="cpu", weights_only=True)["state_dict"])
    predictor.eval()
    methods = config["methods"].split(",")
    regimes = config["regimes"].split(",")
    archive = out / "predictions" / "point"
    archive.mkdir(parents=True, exist_ok=True)
    required = ["q", "k", "mq", "mk", "pixq", "k_to_q", "projected_kcov", "labels", "pixel_error"]
    scene_statistics, summaries, replay_errors, hashes = {}, [], [], {}
    for seed in SEEDS:
        tests = make_point_test(seed, config, predictor)
        for regime in regimes:
            for condition, data in tests.items():
                if digest(data) != manifest["datasets"][f"{regime}/{seed}"]["test"][condition]:
                    raise ValueError(f"point dataset digest mismatch {seed}/{condition}")
            for method in methods:
                record = next(r for r in manifest["records"] if
                    (r["seed"], r["method"], r["regime"]) == (seed, method, regime))
                model = load_checkpoint(root, record, config, hashes, point=True)
                for condition, dataset in tests.items():
                    data = {key: dataset[key] for key in required}
                    probability, loss = point_probability(model, data)
                    labels = data["labels"].numpy()
                    summary, _ = point_metrics(probability, labels, data["pixel_error"].numpy())
                    summary["loss"] = float(loss.mean())
                    errors = compare_summary(summary, record["test"][condition])
                    replay_errors.append(dict(seed=seed, regime=regime, method=method,
                                              condition=condition, errors=errors))
                    summaries.append(dict(seed=seed, regime=regime, method=method,
                                          condition=condition, selected_step=record["step"], **summary))
                    predicted = probability.argmax(-1)
                    key_count = probability.shape[-1]-1
                    visible = labels < key_count
                    correct = predicted == labels
                    closed_correct = probability[..., :-1].argmax(-1) == labels
                    unmatched = ~visible
                    brier = (probability[..., -1] - unmatched) ** 2
                    components = dict(
                        visible_r1=((correct & visible).sum(-1), visible.sum(-1)),
                        closed_visible_r1=((closed_correct & visible).sum(-1), visible.sum(-1)),
                        false_match_rate=((unmatched & (predicted < key_count)).sum(-1), unmatched.sum(-1)),
                        cross_entropy=(loss.sum(-1), np.full(len(loss), loss.shape[-1])),
                        overall_accuracy=(correct.sum(-1), np.full(len(loss), loss.shape[-1])),
                        brier_unmatched=(brier.sum(-1), np.full(len(loss), loss.shape[-1])))
                    scene_statistics[(seed, regime, method, condition)] = components
                    np.savez_compressed(archive / f"{regime}-{seed}-{method}-{condition}.npz",
                        probability=probability, labels=labels, cross_entropy=loss,
                        kl=loss, top1_target_mass=correct.astype(np.uint8), visible=visible,
                        closed_correct=closed_correct,
                        **{f"{metric}_numerator": values[0] for metric, values in components.items()},
                        **{f"{metric}_denominator": values[1] for metric, values in components.items()})
                print(f"point replay seed={seed} regime={regime} method={method}", flush=True)
        del tests
    comparisons = {}
    conditions = list(manifest["records"][0]["test"])
    for regime in regimes:
        for condition in conditions:
            for control in methods:
                if control == "isometric_shape":
                    continue
                metric_names = scene_statistics[(SEEDS[0], regime, control, condition)]
                for metric in metric_names:
                    numerators, denominators = [], []
                    for seed in SEEDS:
                        a = scene_statistics[(seed, regime, "isometric_shape", condition)][metric]
                        b = scene_statistics[(seed, regime, control, condition)][metric]
                        np.testing.assert_array_equal(a[1], b[1])
                        numerators.append(np.asarray(a[0], np.float64)-b[0])
                        denominators.append(np.asarray(a[1], np.float64))
                    comparisons[f"{regime}/{condition}/isometric_shape_minus_{control}/{metric}"] = \
                        paired_point_statistics(np.stack(numerators), np.stack(denominators), args.bootstrap)
    checkpoint_hash_final_check(hashes)
    output = dict(protocol=dict(seeds=SEEDS, source=source,
        domain="old point-to-patch task; occlusion/unmatched present",
        checkpoint_hash_scope="first replay, checked unchanged after replay",
        uncertainty="nested seed/scene paired bootstrap for decomposable metrics; AP/ECE summaries are seed values only",
        probability_archive="float32 probabilities, per-row hard-label CE/KL and correctness; scene order follows deterministic dataset",
        lower_is_better=["cross_entropy", "false_match_rate", "brier_unmatched"]),
        training_manifest_sha256=manifest_hash_final_check(root, manifest),
        checkpoint_sha256=hashes, summaries=summaries, differences=comparisons,
        replay_errors=replay_errors)
    write_json(out / "point_summary.json", output)
    return output


def write_report(out, real=None, point=None):
    # Separate real/point replay invocations must preserve the other completed
    # section; neither a partial training run nor an absent section is inferred.
    if real is None and (Path(out) / "real_summary.json").exists():
        real = json.loads((Path(out) / "real_summary.json").read_text())
    if point is None and (Path(out) / "point_summary.json").exists():
        point = json.loads((Path(out) / "point_summary.json").read_text())
    lines = ["# Isometric PE frozen confirmation replay", "",
             "Only completed five-seed runs enter this report. All differences are candidate minus control; lower CE/KL is better. Camera intervals concern the observed reconstructed scenes, not a scene population."]
    if real is not None:
        lines += ["", "## Real sparse-track transfer", "",
                  "All methods use the runner's >=95% retained-mass mask. The strong projected RBF uses its OLD validation-preferred score mode and frozen width/temperature. Fixed baselines are not independent seed replicates.", "",
                  "| Condition | Control | CE difference | seed-camera 95% CI | Five seed differences |",
                  "|---|---|---:|---|---|"]
        controls = ["isometric_fixed", "isometric_isotropic", "isometric_area", "isometric_shuffled",
                    "expected_shape", "projected_rope", "absolute_geometry", "projected_center_rbf"]
        for condition in ("teatime", "waldo_kitchen", "pooled", "teatime_alt", "waldo_kitchen_alt", "pooled_alt"):
            for control in controls:
                key = f"{condition}/isometric_shape_minus_{control}/cross_entropy"
                if key not in real["differences"]:
                    continue
                value = real["differences"][key]
                lo, hi = value["ci95_seed_camera"]
                seed_values = ", ".join(f"{x:+.4f}" for x in value["seed_differences"].values())
                lines.append(f"| {condition} | {control} | {value['mean']:+.5f} | [{lo:+.5f}, {hi:+.5f}] | {seed_values} |")
        lines += ["", "Sparse-track transfer is a proxy; the supplied reconstruction may have used the evaluation photographs. GS-subset swaps can change eligible rows, so primary/alt are separate robustness evaluations rather than identical-mask paired geometry effects."]
    if point is not None:
        lines += ["", "## Original point task", "",
                  "Visible Recall@1 includes rejecting a visible match as an error. Closed-candidate ranking, unmatched AP, false acceptance and calibration remain separate in JSON and probability archives.", "",
                  "| Regime | Condition | Control | Visible R1 difference | seed-scene 95% CI |",
                  "|---|---|---|---:|---|"]
        for key, value in point["differences"].items():
            regime, condition, pair, metric = key.split("/")
            control = pair.removeprefix("isometric_shape_minus_")
            if metric != "visible_r1" or control not in ("isometric_fixed", "projected_rope", "expected_shape"):
                continue
            lo, hi = value["ci95_seed_scene"]
            lines.append(f"| {regime} | {condition} | {control} | {value['mean']:+.5f} | [{lo:+.5f}, {hi:+.5f}] |")
    lines += ["", "This report records evidence and failures; a positive isolated comparison does not establish the complete research objective. Probability files are local replay artifacts; source, input and checkpoint hashes are in the JSON summaries."]
    (Path(out) / "README.md").write_text("\n".join(lines) + "\n")


def main(args):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.out.mkdir(parents=True, exist_ok=True)
    source_path = Path(__file__).resolve()
    execution_source_sha256 = sha256(source_path)
    settings = frozen_baselines(args.out)
    if args.stage == "validation_baseline":
        validation_baseline(args, settings)
        return
    # Verify all requested manifests BEFORE any confirmation data access.
    real_manifest = load_completed_manifest(args.real_run, "real") if args.stage in ("all", "real") else None
    point_manifest = load_completed_manifest(args.point_run, "point") if args.stage in ("all", "point") else None
    real = analyse_real(args, real_manifest, settings) if real_manifest is not None else None
    point = analyse_point(args, point_manifest) if point_manifest is not None else None
    if sha256(source_path) != execution_source_sha256:
        raise ValueError("analysis source changed during replay; rerun the frozen source")
    shutil.copy2(source_path, args.out / "source_snapshot.py")
    write_json(args.out / "source_sha256.json", {str(source_path): sha256(source_path)})
    write_report(args.out, real, point)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["validation_baseline", "all", "real", "point"], default="all")
    parser.add_argument("--real-run", type=Path, default=Path("results/round3_isometric_real_confirmation"))
    parser.add_argument("--point-run", type=Path, default=Path("results/round3_isometric_point_confirmation"))
    parser.add_argument("--real-root", type=Path, default=Path("results/round3_real_support_data_adjacent"))
    parser.add_argument("--out", type=Path, default=Path("results/round3_isometric_analysis"))
    parser.add_argument("--bootstrap", type=int, default=4000)
    main(parser.parse_args())
