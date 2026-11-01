"""Audit and replay the frozen five-seed dense-region isometric confirmation.

No model fitting, checkpoint selection, hyperparameter search, or label change
is performed. Tests use namespace 4300+seed, independent of 4100/4200 training
and validation namespaces. Probabilities come from the archived model class.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from experiments.matching_probe import digest
from experiments.region_support_data import make_dataset, CONDITIONS
from experiments.isometric_confirmation_analysis import (
    SEEDS, ROW_METRICS, archived_matcher, checkpoint_hash_final_check,
    compare_summary, load_checkpoint, manifest_hash_final_check,
    paired_point_statistics, real_probability_metrics, sha256,
    verify_source_snapshot, write_json,
)


METHODS = ["projected_rope", "expected_fixed", "expected_shape", "isometric_fixed",
           "isometric_shape", "isometric_isotropic", "isometric_area",
           "isometric_shuffled", "absolute_geometry"]
CONTROLS = ["isometric_fixed", "isometric_isotropic", "isometric_area",
            "isometric_shuffled", "projected_rope", "expected_shape", "absolute_geometry"]


def audit_run(root):
    contents = (root / "manifest.json").read_bytes()
    manifest = json.loads(contents)
    manifest["_analysis_input_sha256"] = hashlib.sha256(contents).hexdigest()
    if "elapsed_seconds" not in manifest:
        raise ValueError("all training seeds must be terminal before test replay")
    required = dict(domain="synthetic", phase="confirmation", seeds="11,22,33,44,55",
                    methods=",".join(METHODS), features=256, resolution=.25, steps=600,
                    val_every=50, batch=16, train_scenes=192, val_scenes=48, test_scenes=96)
    for field, value in required.items():
        if manifest["config"][field] != value:
            raise ValueError(f"frozen confirmation configuration changed: {field}")
    actual = [(r["seed"], r["method"]) for r in manifest["records"]]
    expected = {(seed, method) for seed in SEEDS for method in METHODS}
    if len(actual) != 45 or len(set(actual)) != 45 or set(actual) != expected:
        raise ValueError("incomplete or duplicate seed/method records")
    history_path = root / "training_history.json"
    history = json.loads(history_path.read_text())
    selected = []
    for record in manifest["records"]:
        rows = [r for r in history if (r["seed"], r["method"]) ==
                (record["seed"], record["method"])]
        if [r["step"] for r in rows] != [0] + list(range(50, 601, 50)):
            raise ValueError("validation history incomplete or wrong evaluation schedule")
        best = min(rows, key=lambda r: r["validation"]["cross_entropy"])
        if best["step"] != record["step"]:
            raise ValueError("selected checkpoint is not the earliest validation CE minimum")
        compare_summary(record["best_validation"], best["validation"], tolerance=1e-14)
        if set(record["test"]) != set(CONDITIONS):
            raise ValueError("test-condition records incomplete")
        selected.append(dict(seed=record["seed"], method=record["method"],
                             selected_step=record["step"],
                             minimum_validation_ce=best["validation"]["cross_entropy"]))
    # The entropy depends only on the common labels, not on a model. This also
    # catches accidentally comparing records from different datasets/masks.
    entropy_errors = []
    for seed in SEEDS:
        for condition in CONDITIONS:
            entropies = [r["test"][condition]["cross_entropy"] - r["test"][condition]["kl"]
                         for r in manifest["records"] if r["seed"] == seed]
            spread = float(np.ptp(entropies))
            if spread > 1e-12:
                raise ValueError("CE-KL entropy differs across methods")
            entropy_errors.append(spread)
    return manifest, dict(
        terminal_marker=manifest["elapsed_seconds"], complete_records=45,
        all_five_seeds=True, all_predeclared_methods_present=True,
        frozen_config=required, selection_rule="earliest minimum validation CE; includes step zero",
        training_history_sha256=sha256(history_path),
        every_checkpoint_selection_reproduced=True, selections=selected,
        maximum_cross_method_target_entropy_spread=max(entropy_errors))


@torch.no_grad()
def probabilities(model, data, batch=32):
    return np.concatenate([model({key: value[start:start+batch] for key, value in data.items()})
                           [..., :-1].softmax(-1).cpu().numpy()
                           for start in range(0, len(data["q"]), batch)])


def cached_scene_audit(root, cache, seed, method, condition, rows, valid):
    count = valid.sum(-1)
    prefix = f"{seed}/{method}/{condition}"
    np.testing.assert_array_equal(cache[f"{prefix}/valid_rows"], count)
    errors = {}
    for metric in ("kl", "js", "top1_target_mass", "dominant_region_accuracy"):
        replayed = np.divide((rows[metric] * valid).sum(-1), count,
            out=np.full(len(count), np.nan), where=count > 0)
        expected = cache[f"{prefix}/{metric}"]
        np.testing.assert_array_equal(np.isnan(expected), np.isnan(replayed))
        error = float(np.max(np.abs(expected[count > 0]-replayed[count > 0])))
        if error > 2e-12:
            raise ValueError(f"cached scene metrics disagree: {prefix}/{metric}: {error}")
        errors[metric] = error
    return errors


def report(out, results):
    lines = ["# Dense synthetic support: frozen isometric confirmation", "",
        "Five completed training seeds; 45 archived checkpoints replayed on independently sampled test scenes. The task predicts dense pixel-area transfer conditional on visibility. It has no dustbin/occlusion-rejection supervision and is not real-photograph matching or NVS.", "",
        "Training/validation/test namespaces are 4100/4200/4300 plus the training seed. Fixed settings: 192/48/96 scenes per seed, 600 optimizer steps, validation every 50 steps including the untrained state, resolution sigma=.25 patch units, nine predeclared methods. No new fitting, selection or tuning was performed.", "",
        "| Condition | Control | Full-minus-control KL (= CE difference) | 95% seed-scene CI | Five seed differences | JS difference | JS 95% CI |",
        "|---|---|---:|---|---|---:|---|"]
    for condition in CONDITIONS:
        for control in CONTROLS:
            kl = results["differences"][f"{condition}/isometric_shape_minus_{control}/kl"]
            js = results["differences"][f"{condition}/isometric_shape_minus_{control}/js"]
            lo, hi = kl["ci95_seed_scene"]
            jlo, jhi = js["ci95_seed_scene"]
            seeds = ", ".join(f"{value:+.5f}" for value in kl["seed_differences"].values())
            lines.append(f"| {condition} | {control} | {kl['mean']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {seeds} | {js['mean']:+.6f} | [{jlo:+.6f}, {jhi:+.6f}] |")
    lines += ["", "Negative differences favor full covariance. The main controls have the same 96D output: fixed covariance; trace-preserving isotropic covariance; determinant/area-preserving isotropic covariance; and cyclically shuffled covariance. The latter is an incorrect-support intervention, not a same-information competitor.", "",
              "## Absolute means", "",
              "| Condition | Method | CE | KL | JS | Top1 target mass |",
              "|---|---|---:|---:|---:|---:|"]
    for condition in CONDITIONS:
        for method in METHODS:
            r = results["aggregate"][condition][method]
            lines.append(f"| {condition} | {method} | {r['cross_entropy']['mean']:.6f} | {r['kl']['mean']:.6f} | {r['js']['mean']:.6f} | {r['top1_target_mass']['mean']:.6f} |")
    lines += ["", "## Scope and verification", "",
        "Query/key rectangles are generated independently; covariance is fitted from each view's own surface samples. Labels count dense reprojected surface pixels rather than a Gaussian formula or shared random support identity. Targets measure image-pixel mass, not physical surface-area IoU or Gaussian overlap.", "",
        "Each paired bootstrap first resamples training seeds, then independent synthetic scenes within the selected seed. It preserves all valid query rows in a scene and computes a ratio of sums within each seed before averaging seeds. Rows are not independent replicates. CE and KL differences coincide because every method uses identical labels and masks; target entropy is checked explicitly.", "",
        "Every source snapshot, checkpoint and test-dataset digest is verified. Archived model code is executed directly; subsequent current-model candidates are excluded. Replay metrics are compared with all saved test records and per-scene caches, and checkpoint selection is checked against the complete validation history. See summary.json for hashes, raw five-seed differences and checks; predictions contains float32 probability archives.", "",
        "These results can support or reject a dense synthetic support mechanism. Even consistent synthetic gains cannot replace real-data replication on new cameras, robust predicted geometry, strong geometric/learned competitors, and rejection/calibration checks. No overall research-success claim follows from this report alone."]
    (out / "README.md").write_text("\n".join(lines) + "\n")


def main(args):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.out.mkdir(parents=True, exist_ok=True)
    executing_source = sha256(__file__)
    manifest, run_audit = audit_run(args.run)
    sources = verify_source_snapshot(args.run, manifest, ["models/geometry.py",
        "experiments/region_support_data.py", "experiments/rendered_scene.py",
        "experiments/matching_probe.py"])
    matcher_class, sources["executed_archived_model"] = archived_matcher(args.run, manifest)
    checkpoints, summaries, replay_audits, data_hashes = {}, [], [], {}
    per_scene, rows_csv, cached_sha = {}, [], sha256(args.run / "per_scene.npz")
    archive = args.out / "predictions"
    archive.mkdir(exist_ok=True)
    with np.load(args.run / "per_scene.npz", allow_pickle=False) as cached:
        for seed in SEEDS:
            cache = {}
            tests = {condition: make_dataset(4300+seed, manifest["config"]["test_scenes"],
                                             condition, cache)[0] for condition in CONDITIONS}
            del cache
            for condition, data in tests.items():
                actual = digest(data)
                if actual != manifest["datasets"][str(seed)]["test"][condition]:
                    raise ValueError(f"test data digest mismatch {seed}/{condition}")
                data_hashes[f"{seed}/{condition}"] = actual
            for method in METHODS:
                record = next(row for row in manifest["records"] if
                              (row["seed"], row["method"]) == (seed, method))
                model = load_checkpoint(args.run, record, manifest["config"], checkpoints, matcher_class)
                for condition, data in tests.items():
                    probability = probabilities(model, data)
                    summary, rows = real_probability_metrics(probability, data)
                    errors = compare_summary(summary, record["test"][condition], tolerance=2e-12)
                    valid = data["label_valid"].numpy()
                    scene_errors = cached_scene_audit(args.run, cached, seed, method,
                                                      condition, rows, valid)
                    replay_audits.append(dict(seed=seed, method=method, condition=condition,
                                             record_errors=errors, scene_cache_errors=scene_errors))
                    summaries.append(dict(seed=seed, method=method, condition=condition, **summary))
                    counts = valid.sum(-1)
                    per_scene[(seed, method, condition)] = dict(
                        count=counts, **{metric: (rows[metric]*valid).sum(-1) for metric in ROW_METRICS})
                    np.savez_compressed(archive / f"{seed}-{method}-{condition}.npz",
                        probability=probability, targets=data["soft_labels"].numpy(),
                        label_valid=valid, scene_seed=(4300+seed)*1000+np.arange(len(valid)),
                        **rows)
                    for scene_index, query_index in zip(*np.nonzero(valid)):
                        rows_csv.append(dict(seed=seed, method=method, condition=condition,
                            scene_index=int(scene_index), scene_seed=(4300+seed)*1000+int(scene_index),
                            query_region=int(query_index),
                            **{metric: float(rows[metric][scene_index, query_index]) for metric in ROW_METRICS}))
                print(f"dense-region replay seed={seed} method={method}", flush=True)
            del tests
    aggregate, differences, ce_kl_errors = {}, {}, []
    for condition in CONDITIONS:
        aggregate[condition] = {}
        for method in METHODS:
            rows = [r for r in summaries if r["condition"] == condition and r["method"] == method]
            aggregate[condition][method] = {metric: dict(
                mean=float(np.mean([r[metric] for r in rows])),
                per_seed={str(r["seed"]): r[metric] for r in rows}) for metric in ROW_METRICS}
        for control in CONTROLS:
            counts = np.stack([per_scene[(seed, "isometric_shape", condition)]["count"] for seed in SEEDS])
            for seed in SEEDS:
                np.testing.assert_array_equal(per_scene[(seed, "isometric_shape", condition)]["count"],
                                              per_scene[(seed, control, condition)]["count"])
            for metric in ROW_METRICS:
                delta = np.stack([per_scene[(seed, "isometric_shape", condition)][metric] -
                                  per_scene[(seed, control, condition)][metric] for seed in SEEDS])
                key = f"{condition}/isometric_shape_minus_{control}/{metric}"
                differences[key] = paired_point_statistics(delta, counts, args.bootstrap)
            prefix = f"{condition}/isometric_shape_minus_{control}"
            error = abs(differences[f"{prefix}/cross_entropy"]["mean"] -
                        differences[f"{prefix}/kl"]["mean"])
            if error > 1e-12:
                raise ValueError("paired CE and KL differences disagree")
            ce_kl_errors.append(error)
    checkpoint_hash_final_check(checkpoints)
    if sha256(args.run / "per_scene.npz") != cached_sha:
        raise ValueError("cached scene archive changed during replay")
    with (args.out / "per_row.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_csv[0]))
        writer.writeheader()
        writer.writerows(rows_csv)
    result = dict(protocol=dict(
        task="dense visible image-region transfer conditional on visibility",
        no_unmatched_or_occlusion_rejection_supervision=True,
        seed_namespaces=dict(training=4100, validation=4200, test=4300),
        bootstrap="paired seed -> independent scenes; valid-row ratio within each seed; mean across seeds",
        no_retraining_or_tuning=True, lower_is_better=["cross_entropy", "kl", "js"],
        scope="synthetic support mechanism; not real-photo matching, NVS, or scene-population validation"),
        run_audit=run_audit, sources=sources,
        training_manifest_sha256=manifest_hash_final_check(args.run, manifest),
        per_scene_input_sha256=cached_sha, checkpoint_sha256=checkpoints,
        test_data_sha256=data_hashes, replay_audits=replay_audits,
        maximum_paired_ce_kl_difference_error=max(ce_kl_errors),
        summaries=summaries, aggregate=aggregate, differences=differences)
    write_json(args.out / "summary.json", result)
    if sha256(__file__) != executing_source:
        raise ValueError("analysis source changed during replay")
    shutil.copy2(__file__, args.out / "source_snapshot.py")
    write_json(args.out / "source_sha256.json", {str(Path(__file__).resolve()): executing_source})
    report(args.out, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("results/round3_isometric_region_confirmation"))
    parser.add_argument("--out", type=Path, default=Path("results/round3_isometric_region_analysis"))
    parser.add_argument("--bootstrap", type=int, default=4000)
    main(parser.parse_args())
