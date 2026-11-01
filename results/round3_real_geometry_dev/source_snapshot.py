"""Parameter-free development diagnostics for real regional Gaussian support.

Only train/validation pair files are opened. Sparse SfM-track transfer targets
are independent of the Gaussian kernel being evaluated. The probe distinguishes
row-normalizing BC from using BC as a softmax logit; each method has its own
validation-only temperature choice. No RGB fitting or GPU is involved.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

TEMPERATURES = [1., 2., 4., 8., 16., 32., 64., 128.]
PIXEL_WIDTHS = [.125, .25, .5, 1., 2., 4., 8.]
WORLD_WIDTHS = [.001, .003, .01, .03, .1, .3, 1.]
SHUFFLE_SEEDS = [110, 111, 112, 113, 114]


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def log_bc(mu_a, cov_a, mu_b, cov_b):
    """Log Bhattacharyya coefficient for normalized nondegenerate Gaussians."""
    avg = .5 * (cov_a[:, None] + cov_b[None])
    delta = mu_a[:, None] - mu_b[None]
    sign, ld = np.linalg.slogdet(avg)
    sa, la = np.linalg.slogdet(cov_a)
    sb, lb = np.linalg.slogdet(cov_b)
    if np.any(sign <= 0) or np.any(sa <= 0) or np.any(sb <= 0):
        raise ValueError("Gaussian covariance must be SPD")
    mahalanobis = np.einsum("...i,...i->...", delta, np.linalg.solve(avg, delta[..., None])[..., 0])
    result = -.125 * mahalanobis - .5 * ld + .25 * la[:, None] + .25 * lb[None]
    return np.minimum(result, 0.)


def normalised_log_probability(log_kernel, valid_key, mode, temperature=1.):
    if mode == "normalise_kernel":
        logits = log_kernel
    elif mode == "softmax_kernel":
        logits = temperature * np.exp(log_kernel)
    else:
        raise ValueError(mode)
    logits = np.where(valid_key[None], logits, -np.inf)
    maximum = np.max(logits, axis=1, keepdims=True)
    if not np.isfinite(maximum).all():
        raise ValueError("at least one valid candidate is required")
    logsum = maximum + np.log(np.exp(logits - maximum).sum(axis=1, keepdims=True))
    return logits - logsum


def row_metrics(log_probability, targets):
    ce = -np.sum(targets * np.where(targets > 0, log_probability, 0), axis=1)
    entropy = -np.sum(targets * np.log(np.maximum(targets, 1e-300)), axis=1)
    predictions = np.argmax(log_probability, axis=1)
    hit_mass = targets[np.arange(len(targets)), predictions]
    maximum_mass = targets.max(axis=1)
    return dict(soft_ce=ce, kl=ce-entropy, top1_target_mass=hit_mass,
                top1_max_bin_accuracy=(hit_mass >= maximum_mass - 1e-8).astype(float),
                oracle_max_target_mass=maximum_mass)


def build_examples(root):
    examples, source_hashes, loading_audit = [], {}, []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        scene = manifest["scene"]
        scene_scale = manifest["normalization"]["scale"]
        source_hashes[str(manifest_path)] = file_sha256(manifest_path)
        for pair in manifest["pairs"]:
            # Deliberately do not open test NPZ files, including for statistics.
            if pair["split"] not in ("train", "validation"):
                continue
            path = manifest_path.parent / pair["file"]
            source_hashes[str(path)] = file_sha256(path)
            with np.load(path, allow_pickle=False) as f:
                data = {k: np.asarray(f[k], np.float64) for k in f.files if k not in ("q_features", "k_features")}
            q_valid = data["q_valid"].astype(bool)
            k_valid = data["k_valid"].astype(bool)
            base_rows = data["label_valid"].astype(bool) & q_valid
            target = data["soft_labels"].copy()
            retained = target[:, k_valid].sum(1)
            row_valid = base_rows & (retained > 0)
            loading_audit.append(dict(scene=scene, split=pair["split"], file=str(path), q_id=pair["q_id"],
                                      common_tracks=int(data["track_counts"].sum()),
                                      valid_before_key_mask=int(base_rows.sum()),
                                      valid_after_key_mask=int(row_valid.sum()),
                                      label_mass_lost=float(np.mean(1-retained[base_rows])) if base_rows.any() else None,
                                      valid_key_count=int(k_valid.sum())))
            if not row_valid.any() or not k_valid.any():
                continue
            target[:, ~k_valid] = 0
            target /= np.maximum(retained[:, None], 1e-300)
            row_indices = np.flatnonzero(row_valid)
            q_mu = data["q_pixel_mean"] / 24.
            q_cov = data["q_pixel_cov"] / 24.**2
            k_mu = data["k_projected_mean"] / 24.
            k_cov = data["k_projected_cov"] / 24.**2
            q_fit_mu = data["q_projected_mean"] / 24.
            q_fit_cov = data["q_projected_cov"] / 24.**2
            fixed_cov = np.broadcast_to(np.eye(2)/12., k_cov.shape).copy()
            same_area_variance = np.sqrt(np.linalg.det(k_cov))
            same_area_cov = np.eye(2)[None] * same_area_variance[:, None, None]
            same_trace_cov = np.eye(2)[None] * (np.trace(k_cov, axis1=-2, axis2=-1)/2.)[:, None, None]
            q_same_area_cov = np.eye(2)[None] * np.sqrt(np.linalg.det(q_fit_cov))[:, None, None]
            q_world_cov = data["q_cov"]/scene_scale**2
            k_world_cov = data["k_cov"]/scene_scale**2
            q_world_isovolume = np.eye(3)[None] * np.cbrt(np.linalg.det(q_world_cov))[:, None, None]
            k_world_isovolume = np.eye(3)[None] * np.cbrt(np.linalg.det(k_world_cov))[:, None, None]
            q_world_isotrace = np.eye(3)[None] * (np.trace(q_world_cov,axis1=-2,axis2=-1)/3)[:, None, None]
            k_world_isotrace = np.eye(3)[None] * (np.trace(k_world_cov,axis1=-2,axis2=-1)/3)[:, None, None]
            def plane_isotropic(C):
                values, vectors = np.linalg.eigh(C)
                normal = vectors[:, :, 0]
                projector = normal[:, :, None] * normal[:, None, :]
                tangent_variance = np.sqrt(values[:, 1] * values[:, 2])
                return values[:, 0, None, None]*projector + tangent_variance[:, None, None]*(np.eye(3)[None]-projector)
            q_world_plane = plane_isotropic(q_world_cov)
            k_world_plane = plane_isotropic(k_world_cov)
            kernels = {
                "full_bc": {"default": log_bc(q_mu, q_cov, k_mu, k_cov)},
                "fixed_key_cov": {"default": log_bc(q_mu, q_cov, k_mu, fixed_cov)},
                "isotropic_same_area": {"default": log_bc(q_mu, q_cov, k_mu, same_area_cov)},
                "isotropic_same_trace": {"default": log_bc(q_mu, q_cov, k_mu, same_trace_cov)},
                "projected_query_bc": {"default": log_bc(q_fit_mu, q_fit_cov, k_mu, k_cov)},
                "world_full_bc": {"default": log_bc(data["q_mean"]/scene_scale, q_world_cov,
                                                      data["k_mean"]/scene_scale, k_world_cov)},
                "projected_query_isotropic_same_area": {"default": log_bc(q_fit_mu, q_same_area_cov, k_mu, same_area_cov)},
                "world_isotropic_same_volume": {"default": log_bc(data["q_mean"]/scene_scale, q_world_isovolume,
                                                                    data["k_mean"]/scene_scale, k_world_isovolume)},
                "world_isotropic_same_trace": {"default": log_bc(data["q_mean"]/scene_scale, q_world_isotrace,
                                                                   data["k_mean"]/scene_scale, k_world_isotrace)},
                "world_plane_isotropic": {"default": log_bc(data["q_mean"]/scene_scale, q_world_plane,
                                                               data["k_mean"]/scene_scale, k_world_plane)},
            }
            for method, delta, widths in [
                ("pixel_center_rbf", q_mu[:, None]-k_mu[None], PIXEL_WIDTHS),
                ("projected_center_rbf", q_fit_mu[:, None]-k_mu[None], PIXEL_WIDTHS),
                ("world_center_rbf", (data["q_mean"][:, None]-data["k_mean"][None])/scene_scale, WORLD_WIDTHS),
            ]:
                squared = np.sum(delta**2, axis=-1)
                kernels[method] = {str(width): -.5*squared/width**2 for width in widths}
            valid_ids = np.flatnonzero(k_valid)
            for seed in SHUFFLE_SEEDS:
                # A per-file RNG keeps shuffles fixed under data iteration order.
                digest = hashlib.sha256((str(path)+f":{seed}").encode()).digest()
                rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
                shuffled = k_cov.copy()
                shuffled[valid_ids] = k_cov[rng.permutation(valid_ids)]
                kernels[f"covshuffle_{seed}"] = {"default": log_bc(q_mu, q_cov, k_mu, shuffled)}
            examples.append(dict(scene=scene, split=pair["split"], q_id=pair["q_id"],
                                 k_id=pair["k_id"], baseline_group=pair["baseline_group"],
                                 row_indices=row_indices, targets=target[row_valid], k_valid=k_valid,
                                 retained=retained[row_valid],
                                 kernels={method: {h: K[row_valid] for h,K in variants.items()}
                                          for method,variants in kernels.items()}))
    return examples, source_hashes, loading_audit


def evaluate_configuration(examples, method, bandwidth, mode, temperature):
    rows = []
    for example in examples:
        logp = normalised_log_probability(example["kernels"][method][bandwidth], example["k_valid"], mode, temperature)
        metrics = row_metrics(logp, example["targets"])
        for i, original_row in enumerate(example["row_indices"]):
            rows.append(dict(scene=example["scene"], split=example["split"], q_id=example["q_id"],
                             k_id=example["k_id"], query_region=int(original_row),
                             baseline_group=example["baseline_group"], retained_target_mass=float(example["retained"][i]),
                             **{key:float(values[i]) for key,values in metrics.items()}))
    return rows


def summarise(rows):
    metrics = ["soft_ce", "kl", "top1_target_mass", "top1_max_bin_accuracy", "oracle_max_target_mass", "retained_target_mass"]
    return dict(rows=len(rows), cameras=len({(r["scene"],r["q_id"]) for r in rows}),
                **{metric:float(np.mean([r[metric] for r in rows])) for metric in metrics})


def cluster_difference(rows_a, rows_b, metric, bootstrap=4000):
    def key(row):
        return row["scene"],row["q_id"],row["k_id"],row["query_region"]
    b_lookup = {key(r):r for r in rows_b}
    clusters = {}
    for a in rows_a:
        b = b_lookup[key(a)]
        cluster = (a["scene"],a["q_id"])
        clusters.setdefault(cluster, []).append(a[metric]-b[metric])
    if not clusters:
        return None
    names = sorted(clusters)
    sums = np.array([np.sum(clusters[k]) for k in names])
    counts = np.array([len(clusters[k]) for k in names])
    rng = np.random.default_rng(219782)
    draws = rng.integers(0,len(names),size=(bootstrap,len(names)))
    statistics = sums[draws].sum(1)/counts[draws].sum(1)
    return dict(mean=float(sums.sum()/counts.sum()), ci95=np.quantile(statistics,[.025,.975]).tolist(),
                cameras=len(names), positive_camera_fraction=float(np.mean(sums>0)),
                camera_differences=[dict(scene=k[0],q_id=k[1],rows=int(counts[i]),mean=float(sums[i]/counts[i]))
                                    for i,k in enumerate(names)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("results/round3_real_support_data_adjacent"))
    parser.add_argument("--out", type=Path, default=Path("results/round3_real_geometry_dev"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    examples, source_hashes, loading_audit = build_examples(args.data)
    train = [x for x in examples if x["split"] == "train"]
    val = [x for x in examples if x["split"] == "validation"]
    methods = list(examples[0]["kernels"])
    selected, all_rows, summaries, sweeps = {}, {}, {}, []
    for method in methods:
        for mode in ("normalise_kernel", "softmax_kernel"):
            key = f"{method}/{mode}"
            best = None
            for bandwidth in val[0]["kernels"][method]:
                for temperature in ([1.] if mode == "normalise_kernel" else TEMPERATURES):
                    rows = evaluate_configuration(val, method, bandwidth, mode, temperature)
                    score = float(np.mean([r["soft_ce"] for r in rows]))
                    sweeps.append(dict(method=method,mode=mode,bandwidth=bandwidth,temperature=temperature,validation_soft_ce=score))
                    if best is None or score < best[0]:
                        best = score, bandwidth, temperature
            _, bandwidth, temperature = best
            selected[key] = dict(bandwidth=bandwidth,temperature=temperature,selection="minimum aggregate validation soft CE")
            for split, data in (("train",train),("validation",val)):
                rows = evaluate_configuration(data, method, bandwidth, mode, temperature)
                all_rows[(key,split)] = rows
                for scene in sorted({r["scene"] for r in rows})+["all"]:
                    subset = rows if scene == "all" else [r for r in rows if r["scene"] == scene]
                    summaries[f"{key}/{split}/{scene}"] = summarise(subset)
            print(key,selected[key],summaries[f"{key}/validation/all"],flush=True)
    # Five independent covariance-shuffle repeats are averaged in metric space,
    # never ensembled as a new predictor. Keep their raw curves and choices.
    for mode in ("normalise_kernel", "softmax_kernel"):
        for split in ("train", "validation"):
            template = all_rows[(f"covshuffle_{SHUFFLE_SEEDS[0]}/{mode}",split)]
            repeats = [all_rows[(f"covshuffle_{seed}/{mode}",split)] for seed in SHUFFLE_SEEDS]
            averaged = [dict(row, **{metric:float(np.mean([rep[i][metric] for rep in repeats]))
                                    for metric in ("soft_ce","kl","top1_target_mass","top1_max_bin_accuracy")})
                        for i,row in enumerate(template)]
            all_rows[(f"covshuffle_mean/{mode}",split)] = averaged
            for scene in sorted({r["scene"] for r in averaged})+["all"]:
                subset = averaged if scene == "all" else [r for r in averaged if r["scene"]==scene]
                summaries[f"covshuffle_mean/{mode}/{split}/{scene}"] = summarise(subset)
    differences = {}
    for mode in ("normalise_kernel", "softmax_kernel"):
        for split in ("train", "validation"):
            full = all_rows[(f"full_bc/{mode}",split)]
            for control in ("fixed_key_cov","isotropic_same_area","isotropic_same_trace","covshuffle_mean","pixel_center_rbf"):
                other = all_rows[(f"{control}/{mode}",split)]
                for scene in ("all", "teatime", "waldo_kitchen"):
                    a = full if scene=="all" else [r for r in full if r["scene"]==scene]
                    b = other if scene=="all" else [r for r in other if r["scene"]==scene]
                    for metric in ("soft_ce","kl","top1_target_mass","top1_max_bin_accuracy"):
                        differences[f"full_minus_{control}/{mode}/{split}/{scene}/{metric}"] = cluster_difference(a,b,metric)
    for full_method, controls in [
        ("projected_query_bc", ["projected_query_isotropic_same_area", "projected_center_rbf"]),
        ("world_full_bc", ["world_isotropic_same_volume", "world_isotropic_same_trace", "world_plane_isotropic", "world_center_rbf"]),
    ]:
        for mode in ("normalise_kernel", "softmax_kernel"):
            for split in ("train", "validation"):
                full = all_rows[(f"{full_method}/{mode}",split)]
                for control in controls:
                    other = all_rows[(f"{control}/{mode}",split)]
                    for scene in ("all", "teatime", "waldo_kitchen"):
                        a = full if scene=="all" else [r for r in full if r["scene"]==scene]
                        b = other if scene=="all" else [r for r in other if r["scene"]==scene]
                        for metric in ("soft_ce","kl","top1_target_mass","top1_max_bin_accuracy"):
                            differences[f"{full_method}_minus_{control}/{mode}/{split}/{scene}/{metric}"] = cluster_difference(a,b,metric)
    output = dict(protocol=dict(data=str(args.data),loaded_splits=["train","validation"],test_files_opened=0,
                                position_units="image means divided by 24; image covariances divided by 24^2; world by scene scale",
                                target="conditional common-SfM-track distribution after invalid-key masking",
                                row_filter="original common track count >=4, q_valid, remaining key label mass >0",
                                normalise_kernel="BC / sum BC, implemented in log space",
                                softmax_kernel="softmax(temperature * BC)",temperatures=TEMPERATURES,
                                shape_hyperparameters="none; native query fixed image support",
                                pixel_rbf_widths=PIXEL_WIDTHS,world_rbf_widths=WORLD_WIDTHS,
                                covariance_shuffle_seeds=SHUFFLE_SEEDS,shuffle_aggregation="average metrics, no predictor ensemble",
                                selection="aggregate validation CE across both scenes; selected validation metrics are development diagnostics",
                                clustering="query camera, preserving its multiple key views and regions; 4,000 paired cluster bootstrap draws"),
                  selected=selected,summary=summaries,differences=differences,loading_audit=loading_audit,source_sha256=source_hashes)
    (args.out/"summary.json").write_text(json.dumps(output,indent=2,allow_nan=False))
    (args.out/"validation_sweep.json").write_text(json.dumps(sweeps,indent=2,allow_nan=False))
    with (args.out/"per_row.csv").open("w",newline="") as f:
        first = next(iter(all_rows.values()))[0]
        writer = csv.DictWriter(f,fieldnames=["method","mode",*first])
        writer.writeheader()
        for (key,split),rows in all_rows.items():
            method,mode = key.split("/")
            writer.writerows(dict(method=method,mode=mode,**r) for r in rows)
    source = Path(__file__).resolve()
    shutil.copy2(source,args.out/"source_snapshot.py")
    (args.out/"source_sha256.json").write_text(json.dumps({str(source):file_sha256(source)},indent=2))
    lines = ["# Real regional geometry development probe", "", "Only train and validation data were opened. Targets are sparse observed SfM-track transfer, not coverage truth. Validation-selected results are diagnostic and require held-out confirmation.", "", "| Method | score | val CE | val KL | top1 target mass | max-bin accuracy |", "|---|---|---:|---:|---:|---:|"]
    for method in [m for m in methods if not m.startswith("covshuffle_")]+["covshuffle_mean"]:
        for mode in ("normalise_kernel","softmax_kernel"):
            r=summaries[f"{method}/{mode}/validation/all"]
            lines.append(f"| {method} | {mode} | {r['soft_ce']:.4f} | {r['kl']:.4f} | {r['top1_target_mass']:.4f} | {r['top1_max_bin_accuracy']:.4f} |")
    lines += ["", "All methods use identical row masks. Invalid key candidates are masked, retained label mass is recorded and targets are renormalized. `summary.json` records every camera's paired differences and cluster confidence intervals; `per_row.csv` preserves individual observations. Covariance shuffles average five metrics, not predictions.", "", "Image means/covariances are normalized consistently by patch width 24 / 24^2. Fixed key covariance is 24^2/12 I before normalization. Same-area isotropic covariance preserves sqrt(det C), leaving scalar scale while removing direction/anisotropy. Query support is the fixed known image region unless the method explicitly says projected/world. No RGB training is performed."]
    (args.out/"README.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    main()
