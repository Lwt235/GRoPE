"""Retrain both-full versus both-center GS covariance, using the old raw protocol.

This is a mechanism audit on already inspected data, not a prospective holdout.
The full arm must reproduce old selected Gaussian-scale checkpoints bitwise.
No old source/results are modified; no GPU is used.
"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from experiments.isometric_confirmation_analysis import (
    SEEDS, model_probability, paired_camera_seed_statistics,
    real_probability_metrics, real_row_identity, sha256,
)
from experiments.matching_probe import digest
from experiments.query_support import use_fitted_query
from experiments.real_support_data import project_gaussian
from experiments.region_pe_probe import real_dataset, evaluate, soft_loss
from models.gaussian_spectral_pe import GaussianSpectralMatcher


METHOD = "gaussian200_full"
CONDITIONS = ["full", "both_centeronly"]
SCENES = ["teatime", "waldo_kitchen", "figurines", "temple"]
METRICS = ["cross_entropy", "js", "top1_target_mass", "dominant_region_accuracy"]
COV_KEYS = ["cq", "ck", "projected_qcov", "projected_kcov"]
SOURCES = [__file__, "models/gaussian_pe.py", "models/gaussian_spectral_pe.py",
           "models/geometry.py", "experiments/real_support_data.py",
           "experiments/region_pe_probe.py", "experiments/query_support.py",
           "experiments/matching_probe.py", "experiments/isometric_confirmation_analysis.py"]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def component_dataset(root, decomposition, scene, splits, alternate, hashes):
    """Project separately floored center dispersion at unchanged original means."""
    blocks, pairs = [], []
    folder = Path(root) / scene
    hashes[str(folder / "manifest.json")] = sha256(folder / "manifest.json")
    for split in splits:
        data, metadata = real_dataset(root, scene, split, alternate)
        data, metadata = use_fitted_query(data, metadata, folder, alternate)
        blocks.append(data)
        pairs.extend(metadata["pairs"])
    full = {key: torch.cat([block[key] for block in blocks]) for key in blocks[0]}
    center = dict(full)
    for key in COV_KEYS:
        center[key] = full[key].clone()
    suffix = "_alt" if alternate else ""
    diagnostics = []
    for index, pair in enumerate(pairs):
        source = folder / pair["file"]
        stage = "development" if scene in SCENES[:2] and "_test" not in pair["file"] else "confirmation"
        component = Path(decomposition) / stage / "pairs" / scene / pair["file"]
        hashes[str(source)], hashes[str(component)] = sha256(source), sha256(component)
        if hashes[str(source)] != pair["sha256"]:
            raise ValueError("original NPZ hash mismatch")
        with np.load(source, allow_pickle=False) as original, np.load(component, allow_pickle=False) as part:
            K, R, t = original["K_query"], original["R_query"], original["t_query"]
            for side in ["q", "k"]:
                valid = original[f"{side}_valid{suffix}"].astype(bool)
                mean = original[f"{side}_mean{suffix}"].astype(np.float64)
                np.testing.assert_array_equal(part[f"{side}_valid{suffix}"], valid)
                np.testing.assert_allclose(part[f"{side}_mean{suffix}"], mean, rtol=1e-6, atol=1e-7)
                Ccenter = part[f"{side}_centeronly{suffix}"].copy()
                Cfull64 = part[f"{side}_full{suffix}"].copy()
                _, original_projection, _ = project_gaussian(mean, Cfull64, K, R, t)
                _, center_projection, _ = project_gaussian(mean, Ccenter, K, R, t)
                serialized_projection = original[f"{side}_projected_cov{suffix}"].astype(np.float64)
                error = np.linalg.norm(original_projection[valid]-serialized_projection[valid], axis=(-2, -1))
                error /= np.maximum(np.linalg.norm(serialized_projection[valid], axis=(-2, -1)), 1e-20)
                if len(error) and error.max() > 1e-4:
                    raise ValueError("float64 full component does not reproduce original projection")
                projected_key = "projected_qcov" if side == "q" else "projected_kcov"
                world_key = "cq" if side == "q" else "ck"
                center[projected_key][index, valid] = torch.as_tensor(
                    center_projection[valid]/24**2 + np.eye(2)*1e-6, dtype=full[projected_key].dtype)
                center[world_key][index, valid] = torch.as_tensor(Ccenter[valid], dtype=full[world_key].dtype)
                if torch.linalg.eigvalsh(center[projected_key][index]).min() <= 0:
                    raise ValueError("projected center-only covariance not positive definite")
                diagnostics.append({"file": pair["file"], "side": side,
                    "full_projection_max_relative_error": float(error.max()) if len(error) else None,
                    "valid_regions": int(valid.sum())})
    for key in full:
        if key not in COV_KEYS and not torch.equal(full[key], center[key]):
            raise ValueError("noncovariance input changed: " + key)
    valid = full["label_valid"].numpy()
    metadata = {"scene": scene, "splits": splits, "alternate": alternate, "pairs": pairs,
        "valid_rows": int(valid.sum()), "pair_count": len(pairs),
        "full_digest": digest(full), "center_digest": digest(center),
        "noncovariance_digest": digest({key: value for key, value in full.items() if key not in COV_KEYS}),
        "projection_audit": diagnostics}
    return {"full": full, "both_centeronly": center}, metadata, real_row_identity(pairs, valid, scene)


def state_error(actual, expected):
    if actual.keys() != expected.keys():
        raise ValueError("state keys differ")
    return {key: float((value-expected[key]).abs().max())
            for key, value in actual.items() if not torch.equal(value, expected[key])}


def train_arm(seed, condition, train, validation, batches, out, old_manifest):
    torch.manual_seed(seed)
    model = GaussianSpectralMatcher(METHOD, 256, .25)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.0001)
    metrics, _ = evaluate(model, validation)
    best, best_step, best_metrics = metrics["cross_entropy"], 0, metrics
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    history = [{"step": 0, "validation": metrics}]
    started = time.time()
    max_gradient = 0.
    for step in range(500):
        part = {key: value[batches[step]] for key, value in train.items()}
        loss = soft_loss(model(part), part)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training loss; no retuning permitted")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        max_gradient = max(max_gradient, float(norm))
        optimizer.step()
        if (step+1) % 20 == 0:
            metrics, _ = evaluate(model, validation)
            history.append({"step": step+1, "train_loss": float(loss), "validation": metrics})
            if metrics["cross_entropy"] < best:
                best, best_step, best_metrics = metrics["cross_entropy"], step+1, metrics
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    verification = None
    if condition == "full":
        path = Path("results/round4_spectral_real_raw/checkpoints") / f"{seed}-{METHOD}.pt"
        old = torch.load(path, map_location="cpu", weights_only=True)
        mismatch = state_error(best_state, old["state_dict"])
        record = next(row for row in old_manifest["records"] if row["seed"] == seed and row["method"] == METHOD)
        if mismatch or best_step != record["step"] or best_metrics != record["best_validation"]:
            failure = {"seed": seed, "condition": condition, "state_mismatch": mismatch,
                "selected_step": best_step, "old_selected_step": record["step"],
                "best_validation": best_metrics, "old_best_validation": record["best_validation"]}
            write_json(out / "replication_failure.json", failure)
            raise ValueError("full retrain is not a bitwise replication; diagnose before evaluation")
        verification = {"old_checkpoint": str(path), "old_checkpoint_sha256": sha256(path),
                        "bitwise_state_equal": True, "validation_metrics_exactly_equal": True}
    path = out / "checkpoints" / f"{seed}-{condition}.pt"
    path.parent.mkdir(exist_ok=True)
    torch.save({"state_dict": best_state, "method": METHOD, "condition": condition,
                "features": 256, "resolution": .25}, path)
    record = {"seed": seed, "condition": condition, "method": METHOD,
        "selected_step": best_step, "best_validation": best_metrics,
        "seconds": time.time()-started, "maximum_unclipped_gradient_norm": max_gradient,
        "all_training_gradients_finite": True, "checkpoint": str(path),
        "checkpoint_sha256": sha256(path), "full_replication": verification}
    return model.eval(), record, history


def scene_macro_statistics(a, b, identities, bootstrap=4000):
    """Exploratory crossed seed/place/camera bootstrap; only 2 or 4 old places."""
    delta = np.asarray(a, np.float64)-np.asarray(b, np.float64)
    places = sorted({row["scene"] for row in identities})
    rng = np.random.default_rng(591201)
    seed_draws = rng.integers(0, len(SEEDS), (bootstrap, len(SEEDS)))
    place_draws = rng.integers(0, len(places), (bootstrap, len(places)))
    place_means, crossed_places = [], []
    per_place = {}
    for place in places:
        row_index = [i for i, row in enumerate(identities) if row["scene"] == place]
        cameras = sorted({identities[i]["q_id"] for i in row_index})
        mapping = {camera: i for i, camera in enumerate(cameras)}
        member = np.array([mapping[identities[i]["q_id"]] for i in row_index])
        counts = np.bincount(member)
        sums = np.stack([np.bincount(member, weights=values[row_index], minlength=len(cameras)) for values in delta])
        draws = rng.integers(0, len(cameras), (bootstrap, len(cameras)))
        sampled = sums[:, draws].sum(-1) / counts[draws].sum(-1)[None]
        crossed_places.append(sampled[seed_draws, np.arange(bootstrap)[:, None]].mean(-1))
        means = delta[:, row_index].mean(-1)
        place_means.append(means)
        per_place[place] = float(means.mean())
    means = np.stack(place_means, axis=1)
    cross = np.stack(crossed_places, axis=1)
    seed_place_camera = cross[np.arange(bootstrap)[:, None], place_draws].mean(-1)
    seed_place = means[seed_draws[:, :, None], place_draws[:, None, :]].mean((1, 2))
    return {"mean": float(means.mean()), "scene_differences": per_place,
        "seed_differences": {str(seed): float(value) for seed, value in zip(SEEDS, means.mean(-1))},
        "ci95_seed_place": np.quantile(seed_place, [.025, .975]).tolist(),
        "ci95_seed_place_camera": np.quantile(seed_place_camera, [.025, .975]).tolist(),
        "leave_one_place_out": {place: float(np.delete(means, i, axis=1).mean())
                                for i, place in enumerate(places)} if len(places)>1 else {},
        "places": len(places), "bootstrap": bootstrap,
        "estimand": "equal-place macro of within-place row means, averaged across paired training seeds",
        "limitation": "descriptive bootstrap on tiny reused reconstructed-scene set, not a reliable broad place-population CI"}


def main(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    out = args.out
    if out.exists() and any(out.iterdir()):
        raise ValueError("use a new empty experiment directory")
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    old_path = Path("results/round4_spectral_real_raw/manifest.json")
    old = json.loads(old_path.read_text())
    selected = json.loads(Path("results/round4_spectral_real_analysis/summary.json").read_text())["selected_gaussian_scale"]["raw"]
    if selected["selected"] != METHOD:
        raise ValueError("old validation selection differs from frozen requested method")
    for source in ["models/gaussian_pe.py", "models/gaussian_spectral_pe.py", "models/geometry.py",
                   "experiments/region_pe_probe.py", "experiments/query_support.py"]:
        if sha256(source) != old["source_hashes"][source]:
            raise ValueError("critical old training source changed: " + source)
    hashes = {str(old_path): sha256(old_path)}
    train, tm, _ = component_dataset(args.real_root, args.decomposition, "teatime", ["train"], False, hashes)
    validation, vm, _ = component_dataset(args.real_root, args.decomposition, "teatime", ["validation"], False, hashes)
    for seed in SEEDS:
        if tm["full_digest"] != old["datasets"][str(seed)]["train"] or vm["full_digest"] != old["datasets"][str(seed)]["validation"]:
            raise ValueError("full train/validation digest differs from old data")
    source_hashes = {str(Path(path).relative_to(Path.cwd())) if Path(path).is_absolute() else path: sha256(path) for path in SOURCES}
    for source in source_hashes:
        destination = out / "source" / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    protocol = {"purpose": "retrained old GS internal covariance mechanism audit, not new independent confirmation",
        "method": METHOD, "old_validation_scale_selection": selected, "conditions": CONDITIONS,
        "seeds": SEEDS, "device": "cpu", "threads": 1,
        "config": {"steps": 500, "batch": 16, "val_every": 20, "features": 256, "resolution": .25,
            "optimizer": "AdamW", "learning_rate": .002, "weight_decay": .0001, "gradient_clip": 5.},
        "initialization_and_batches": "torch.manual_seed(seed) per arm; identical 380000+seed sampled batches per seed",
        "selection": "each arm independently uses earliest minimum validation CE among step0,20,...,500",
        "covariance": "both sides replace full world C with separately original-floor-clamped center dispersion C; project at unchanged serialized means; /24^2+loader1e-6I; old gaussian200 scales C by2 then adds .25^2I once",
        "fixed": "same full means, RGB features, cameras, q/key validity, >=95% retained-label-mass rows, labels, optimization budget",
        "full_replication": "must exactly match old selected checkpoint tensors, step and validation metrics before evaluating legacy scenes",
        "training": tm, "validation": vm, "source_hashes": source_hashes,
        "evaluation": {"scenes": SCENES, "partitions": ["primary", "alt"],
            "teatime_waldo": "old test only", "figurines_temple": "old train+validation+test decomposition pairs",
            "primary": "external figurines+temple, both parities reported; no parity selection",
            "statistics": "4000 paired seed/query-camera draws within scene; additionally exploratory equal-scene macro seed/place/camera bootstrap and leave-one-scene sensitivity"},
        "limits": ["all evaluation data and conditions already inspected in earlier rounds",
            "center means still originate in optimized GS; center-only is not raw SfM",
            "fixed opaque-center visibility and sparse observed-track conditional labels",
            "only four reconstructed scenes; no universal claim that optimized primitive R/S cannot help",
            "no separate covariance-scale retuning for center-only; this estimates removal at the old selected common physical scale"],
        "metadata_source_sha256": dict(hashes)}
    write_json(out / "frozen_protocol.json", protocol)
    manifest = {"protocol_sha256": sha256(out / "frozen_protocol.json"), "source_hashes": source_hashes, "records": [], "history": []}
    models = {}
    for seed in SEEDS:
        eligible = train["full"]["label_valid"].any(-1).nonzero()[:, 0].cpu()
        generator = torch.Generator().manual_seed(380000+seed)
        batches = eligible[torch.randint(0, len(eligible), (500, 16), generator=generator)]
        for condition in CONDITIONS:
            model, record, history = train_arm(seed, condition, train[condition], validation[condition], batches, out, old)
            models[seed, condition] = model
            manifest["records"].append(record)
            manifest["history"].append({"seed": seed, "condition": condition, "rows": history})
            write_json(out / "manifest.json", manifest)
            print(json.dumps(record), flush=True)
    if len(manifest["records"]) != 10:
        raise ValueError("incomplete fixed experiment matrix")
    manifest["training_elapsed_seconds"] = time.time()-started
    write_json(out / "manifest.json", manifest)

    # Legacy evaluation labels are opened only after all ten training records.
    metrics, identities, metadata, records = {}, {}, {}, []
    probability_dir = out / "probabilities"
    probability_dir.mkdir()
    probability_hashes, rows_archive = {}, {}
    for scene in SCENES:
        root = args.real_root if scene in SCENES[:2] else args.external_root
        splits = ["test"] if scene in SCENES[:2] else ["train", "validation", "test"]
        for alternate in [False, True]:
            datasets, meta, ids = component_dataset(root, args.decomposition, scene, splits, alternate, hashes)
            identities[scene, alternate] = ids
            metadata[f"{scene}/{int(alternate)}"] = meta
            valid = datasets["full"]["label_valid"].numpy()
            for condition in CONDITIONS:
                metrics[scene, alternate, condition] = {metric: [] for metric in METRICS}
            for seed in SEEDS:
                archive = {"valid": valid, "soft_labels": datasets["full"]["soft_labels"].numpy(),
                           "key_valid": datasets["full"]["key_valid"].numpy()}
                for condition in CONDITIONS:
                    probability = model_probability(models[seed, condition], datasets[condition])
                    summary, values = real_probability_metrics(probability, datasets[condition])
                    archive[condition] = probability
                    for metric in METRICS:
                        metrics[scene, alternate, condition][metric].append(values[metric][valid])
                        rows_archive[f"{seed}/{scene}/{int(alternate)}/{condition}/{metric}"] = values[metric][valid]
                    records.append({"seed": seed, "scene": scene, "alternate": alternate,
                                    "condition": condition, "metrics": summary})
                path = probability_dir / f"{seed}-{scene}-{int(alternate)}.npz"
                np.savez_compressed(path, **archive)
                probability_hashes[str(path)] = sha256(path)
    metrics = {key: {metric: np.stack(values) for metric, values in values.items()} for key, values in metrics.items()}
    differences, means = {}, {}
    groups = {scene: [scene] for scene in SCENES}
    groups.update(external=SCENES[2:], all=SCENES)
    for alternate in [False, True]:
        for group, scenes in groups.items():
            ids = [row for scene in scenes for row in identities[scene, alternate]]
            key = f"{int(alternate)}/{group}"
            differences[key], means[key] = {}, {}
            for metric in METRICS:
                full = np.concatenate([metrics[scene, alternate, "full"][metric] for scene in scenes], axis=1)
                center = np.concatenate([metrics[scene, alternate, "both_centeronly"][metric] for scene in scenes], axis=1)
                means[key][metric] = {"full": float(full.mean()), "both_centeronly": float(center.mean())}
                differences[key][metric] = {"row_weighted_seed_camera": paired_camera_seed_statistics(full, center, ids),
                    "scene_macro_seed_place_camera": scene_macro_statistics(full, center, ids)}
    for path, checksum in hashes.items():
        if sha256(path) != checksum:
            raise ValueError("input changed during experiment: " + path)
    for source, checksum in source_hashes.items():
        if sha256(source) != checksum:
            raise ValueError("source changed during experiment: " + source)
    np.savez_compressed(out / "row_metrics.npz", **rows_archive)
    write_json(out / "row_identities.json", {f"{scene}/{int(alt)}": ids for (scene, alt), ids in identities.items()})
    summary = {"scope": protocol["purpose"], "difference": "full minus retrained both-center; negative CE favors internal primitive covariance",
        "means": means, "differences": differences, "records": records, "metadata": metadata,
        "source_data_sha256": hashes, "probability_sha256": probability_hashes,
        "protocol_sha256": manifest["protocol_sha256"], "elapsed_seconds": time.time()-started}
    write_json(out / "summary.json", summary)
    manifest["elapsed_seconds"] = time.time()-started
    write_json(out / "manifest.json", manifest)
    lines = ["# Legacy GS covariance: paired full versus center-only retraining", "",
        "Both arms train on identical old Teatime RGB/features/means/cameras/labels and batches, five seeds, 500 steps, validation every20 including0. The old validation-selected Gaussian scale2 and common resolution .25 are fixed for both arms. The complete full arm reproduces every old checkpoint tensor, selected step and validation metric exactly. Center-only removes internal primitive covariance on both query and key before the common scale2; each arm selects its own checkpoint by its corresponding validation CE.", "",
        "This is a descriptive mechanism audit on previously inspected scenes/conditions. It addresses the frozen-input-swap distribution-mismatch limitation. It is not a new independent confirmation or a globally scale-retuned center-only comparison.", "",
        "| parity | group | full CE | center CE | full-center | paired seed/camera 95% CI |", "|---|---|---:|---:|---:|---|"]
    for alternate in [False, True]:
        for group in groups:
            key = f"{int(alternate)}/{group}"
            value = means[key]["cross_entropy"]
            diff = differences[key]["cross_entropy"]["row_weighted_seed_camera"]
            lo, hi = diff["ci95_seed_camera"]
            lines.append(f"| {int(alternate)} | {group} | {value['full']:.6f} | {value['both_centeronly']:.6f} | {diff['mean']:+.6f} | [{lo:+.6f},{hi:+.6f}] |")
    lines += ["", "Positive CE differences favor retrained center-only. Both parity results, per-seed effects, per-scene effects and leave-one-place sensitivity are retained. The additional scene-macro seed/place/camera intervals use only four reused reconstructed scenes (two for external); they are descriptive and cannot establish broad scene-population generalization. Sparse-track targets are not dense visibility truth. Center-only preserves optimized GS means, fixed visibility and center dispersion: a result here cannot establish that all optimized primitive R/S are universally useless.", "",
        "Source/data hashes, all10 checkpoints, all40 probability archives, sparse-track labels/masks, row metrics/identities, complete validation histories and finite-gradient diagnostics are saved. No GPU or SUN3D evaluation data were used.", ""]
    (out / "README.md").write_text("\n".join(lines))
    print("COMPLETE", summary["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-root", type=Path, default=Path("results/round3_real_support_data_adjacent"))
    parser.add_argument("--external-root", type=Path, default=Path("results/round3_external_scenes"))
    parser.add_argument("--decomposition", type=Path, default=Path("results/round3_cov_decomposition"))
    parser.add_argument("--out", type=Path, default=Path("results/round5_primitive_retrain_raw"))
    main(parser.parse_args())
