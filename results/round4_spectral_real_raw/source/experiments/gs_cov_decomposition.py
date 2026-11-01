"""Attribute real regional-support kernels to GS shape versus center dispersion.

The native covariance is decomposed with exactly the original opaque center
z-buffer, opacity weights and disjoint primitive subsets. No kernel parameter
is fitted. Development opens only the two original scenes' train/validation
NPZs; confirmation requires a prewritten frozen protocol and opens only their
test pairs or the two external scenes. RGB/primitive IDs never enter scoring.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import time

import numpy as np

from experiments.real_support_data import (
    DEFAULT_SOURCES, fit_regions, load_scene, project_gaussian, region_indices, sha256,
)
from experiments.real_support_geometry_probe import (
    cluster_difference, log_bc, normalised_log_probability, row_metrics, summarise,
)

METHODS = ("full", "centeronly", "internalonly", "flooronly")
CONDITIONS = (("primary", False, False), ("alt", True, False),
              ("joint_primary", False, True), ("joint_alt", True, True))
DEV_SCENES = ("teatime", "waldo_kitchen")
EXTERNAL_SCENES = ("figurines", "temple")


def floor_covariance(matrix, eigen_floor):
    values, vectors = np.linalg.eigh((matrix + np.swapaxes(matrix, -1, -2)) * .5)
    return (vectors * np.maximum(values, eigen_floor)[..., None, :]) @ np.swapaxes(vectors, -1, -2)


def fit_regions_decomposed(means, covariances, opacities, K, R, t, *, width=192,
                           height=144, grid=(8, 6), opacity_threshold=.3,
                           near_surface_fraction=.015, min_points=12, eigen_floor=1e-7):
    """Same fit as real_support_data.fit_regions, retaining both raw moments.

    Covariance flooring is applied independently to each counterfactual, using
    the same fixed world-unit threshold. The *raw* total is exactly raw center
    dispersion + raw internal covariance; floored counterfactuals do not sum.
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
    raw_center = np.zeros((token_count, 3, 3))
    raw_internal = np.zeros_like(raw_center)
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
        out_mean[i] = mu
        raw_center[i] = np.einsum("n,ni,nj->ij", weights, centered, centered)
        raw_internal[i] = np.einsum("n,nij->ij", weights, covariances[point_ids])
    raw_total = raw_center + raw_internal
    result = {"mean": out_mean, "raw_center": raw_center, "raw_internal": raw_internal,
              "raw_total": raw_total, "valid": count >= min_points, "point_count": count,
              "zbuffer": zbuffer.reshape(height, width),
              "stats": {"front_and_inside": int(selected.sum()), "visible_centers": len(ids),
                        "valid_regions": int((count >= min_points).sum()),
                        "zbuffer_pixel_fraction": float(np.isfinite(zbuffer).mean())}}
    for name, raw in (("full", raw_total), ("centeronly", raw_center), ("internalonly", raw_internal)):
        result[name] = floor_covariance(raw, eigen_floor)
    result["flooronly"] = np.broadcast_to(np.eye(3) * eigen_floor, raw_total.shape).copy()
    return result


def covariance_statistics(fit):
    valid = fit["valid"]
    if not valid.any():
        return {"valid_regions": 0}
    total, internal = fit["full"][valid], fit["raw_internal"][valid]
    values, vectors = np.linalg.eigh(total)
    normal = vectors[..., 0]
    fraction = np.trace(internal, axis1=-2, axis2=-1) / np.trace(total, axis1=-2, axis2=-1)
    normal_fraction = np.einsum("ni,nij,nj->n", normal, internal, normal) / values[:, 0]
    inverse_root = (vectors * (1 / np.sqrt(values))[:, None, :]) @ vectors.swapaxes(-1, -2)
    whitened = inverse_root @ internal @ inverse_root
    generalized = np.linalg.eigvalsh((whitened + whitened.swapaxes(-1, -2)) / 2)
    return {"valid_regions": int(valid.sum()),
            "internal_trace_fraction_mean": float(fraction.mean()),
            "internal_trace_fraction_median": float(np.median(fraction)),
            "internal_normal_variance_fraction_mean": float(normal_fraction.mean()),
            "internal_normal_variance_fraction_median": float(np.median(normal_fraction)),
            "internal_generalized_eigenvalue_means": generalized.mean(0).tolist(),
            "mean_center_trace": float(np.trace(fit["raw_center"][valid], axis1=-2, axis2=-1).mean()),
            "mean_internal_trace": float(np.trace(internal, axis1=-2, axis2=-1).mean())}


def protocol():
    return dict(methods=list(METHODS), primary_comparison="full_minus_centeronly",
                primary_metric="soft_ce", score="BC / sum valid-key BC, implemented in log space",
                parameters_fitted=0, numerical_floor="(1e-4 * source normalization scale)^2 in original world units",
                flooring="eigenvalue clamp separately after forming each covariance counterfactual",
                mean="same opacity-weighted original-world center for every counterfactual",
                geometry="same opacity>=0.3, near-surface<=1.5%, minimum 12 primitive centers, even/odd subsets",
                target="independent sparse common SfM-track transfer distribution, not dense overlap truth",
                masks={"conditional":"original >=4 common tracks, q valid, >0 target mass on valid keys",
                       "strict95":"same but >=0.95 retained target mass"},
                conditions=[x[0] for x in CONDITIONS],
                primary_mask="conditional", primary_partition="primary",
                confidence="4,000 paired query-camera cluster bootstrap draws; within observed scenes only",
                interpretation="centeronly retains regional point-cloud shape; full-centeronly isolates internal GS covariance in this fixed exporter. flooronly is a numerical ablation, not a tuned point-kernel baseline.",
                development={"scenes":list(DEV_SCENES),"splits":["train","validation"]},
                confirmation={"in_scene_splits":["test"],"external_scenes":list(EXTERNAL_SCENES),"external_splits":["train","validation","test"]},
                selection="none; all methods and masks are declared before any run; no per-scene or test parameter choices")


def collect_scene(scene, data_root, destination, splits):
    """Load only explicitly selected pair NPZs; source geometry has no labels."""
    scene_data = data_root / scene
    manifest_path = scene_data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    selected_pairs = [p for p in manifest["pairs"] if p["split"] in splits]
    source = Path(DEFAULT_SOURCES[scene])
    _, frames, _, geometry, dataset_path, normalization = load_scene(source)
    eigen_floor = (1e-4 * normalization["scale"]) ** 2
    destination.mkdir(parents=True, exist_ok=True)
    used_ids = sorted({p[k] for p in selected_pairs for k in ("q_id", "k_id")})
    fits, camera_stats = {}, []
    source_hashes = {str(manifest_path):sha256(manifest_path)}
    for path in [source/"gaussians.npz",source/"source_model.json",source/"training_scene.json",
                 source/"run_status.json",dataset_path]:
        source_hashes[str(path)] = sha256(path)
    for iid in used_ids:
        frame = frames[iid]
        pairfits = []
        for parity in (0,1):
            args = (geometry["means"][parity::2], geometry["covariances"][parity::2],
                    geometry["opacities"][parity::2], frame["K"], np.asarray(frame["R"]), np.asarray(frame["t"]))
            kwargs = dict(opacity_threshold=manifest["opacity_threshold"],
                          near_surface_fraction=manifest["near_surface_fraction"],
                          min_points=manifest["minimum_fit_points"],eigen_floor=eigen_floor)
            fit = fit_regions_decomposed(*args,**kwargs)
            # Verify the copied fitting logic against the original executable,
            # independent of source-export float32 serialization.
            original = fit_regions(*args,**kwargs)
            np.testing.assert_array_equal(fit["valid"],original["valid"])
            np.testing.assert_array_equal(fit["point_count"],original["point_count"])
            np.testing.assert_array_equal(fit["mean"],original["mean"])
            np.testing.assert_allclose(fit["full"],original["cov"],rtol=1e-12,atol=eigen_floor*1e-7)
            np.testing.assert_array_equal(fit["raw_total"],fit["raw_center"]+fit["raw_internal"])
            pairfits.append(fit)
            camera_stats.append(dict(scene=scene,image_id=iid,name=frame["name"],parity=parity,
                                     K=frame["K"].tolist(),R=frame["R"],t=frame["t"],
                                     visibility=fit["stats"],covariance=covariance_statistics(fit)))
        fits[iid] = pairfits
    examples, audits = [], []
    max_serialized_mean_error = max_serialized_cov_error = 0.
    for pair in selected_pairs:
        original_path = scene_data / pair["file"]
        source_hashes[str(original_path)] = sha256(original_path)
        with np.load(original_path,allow_pickle=False) as f:
            # Do not open/read RGB feature fields: only evaluator labels and
            # fit verification fields are needed.
            original = {k:np.asarray(f[k]) for k in ["soft_labels","label_valid","track_counts"] +
                        [f"{s}_{field}{suffix}" for s in ("q","k") for field in ("mean","cov","valid")
                         for suffix in ("","_alt")]}
        qframe = frames[pair["q_id"]]
        K,R,t = qframe["K"],np.asarray(qframe["R"]),np.asarray(qframe["t"])
        arrays = {k:original[k] for k in ("soft_labels","label_valid","track_counts")}
        for alternate in (False,True):
            suffix = "_alt" if alternate else ""
            for side, iid, parity in (("q",pair["q_id"],int(alternate)),("k",pair["k_id"],1-int(alternate))):
                fit = fits[iid][parity]
                _,_,front = project_gaussian(fit["mean"],fit["full"],K,R,t)
                valid = fit["valid"] & front
                np.testing.assert_array_equal(valid,original[f"{side}_valid{suffix}"])
                np.testing.assert_allclose(fit["mean"],original[f"{side}_mean{suffix}"],rtol=1e-6,atol=1e-7)
                np.testing.assert_allclose(fit["full"],original[f"{side}_cov{suffix}"],rtol=1e-5,atol=eigen_floor*1e-2)
                max_serialized_mean_error = max(max_serialized_mean_error,float(np.max(abs(fit["mean"]-original[f"{side}_mean{suffix}"]))))
                max_serialized_cov_error = max(max_serialized_cov_error,float(np.max(abs(fit["full"]-original[f"{side}_cov{suffix}"]))))
                arrays[f"{side}_mean{suffix}"] = fit["mean"]
                arrays[f"{side}_valid{suffix}"] = valid
                arrays[f"{side}_point_count{suffix}"] = fit["point_count"]
                for field in (*METHODS,"raw_center","raw_internal","raw_total"):
                    arrays[f"{side}_{field}{suffix}"] = fit[field]
        pair_path = destination / pair["file"]
        np.savez_compressed(pair_path,**arrays)
        for condition,alternate,joint in CONDITIONS:
            suffix = "_alt" if alternate else ""
            qvalid = arrays["q_valid"+suffix].copy()
            kvalid = arrays["k_valid"+suffix].copy()
            if joint:
                qvalid &= arrays["q_valid"+ ("" if alternate else "_alt")]
                kvalid &= arrays["k_valid"+ ("" if alternate else "_alt")]
            targets = arrays["soft_labels"].astype(np.float64)
            retained = targets[:,kvalid].sum(-1)
            base_rows = arrays["label_valid"].astype(bool) & qvalid
            for mask in ("conditional","strict95"):
                rowvalid = base_rows & ((retained>0) if mask=="conditional" else (retained>=.95))
                audit = dict(scene=scene,split=pair["split"],file=pair["file"],q_id=pair["q_id"],k_id=pair["k_id"],
                             condition=condition,mask=mask,base_rows=int(base_rows.sum()),valid_rows=int(rowvalid.sum()),
                             valid_keys=int(kvalid.sum()),common_tracks=int(arrays["track_counts"].sum()),
                             mean_retained_mass=float(retained[base_rows].mean()) if base_rows.any() else None,
                             pair_sha256=sha256(pair_path))
                audits.append(audit)
                if not rowvalid.any() or not kvalid.any():
                    continue
                y = targets.copy()
                y[:,~kvalid] = 0
                y /= np.maximum(retained[:,None],1e-300)
                kernels = {method:log_bc(arrays["q_mean"+suffix],arrays[f"q_{method}{suffix}"],
                                         arrays["k_mean"+suffix],arrays[f"k_{method}{suffix}"])[rowvalid]
                           for method in METHODS}
                examples.append(dict(scene=scene,split=pair["split"],q_id=pair["q_id"],k_id=pair["k_id"],
                                     baseline_group=pair["baseline_group"],condition=condition,mask=mask,
                                     rows=np.flatnonzero(rowvalid),targets=y[rowvalid],retained=retained[rowvalid],
                                     kvalid=kvalid,kernels=kernels))
    return examples,audits,camera_stats,source_hashes,dict(scene=scene,eigen_floor=eigen_floor,
        pairs=len(selected_pairs),used_views=len(used_ids),source_export_mean_max_abs_error=max_serialized_mean_error,
        source_export_cov_max_abs_error=max_serialized_cov_error,unfloored_total_equals_sum=True,
        refit_matches_original_float64=True)


def evaluate(examples):
    rows=[]
    for example in examples:
        for method in METHODS:
            logp=normalised_log_probability(example["kernels"][method],example["kvalid"],"normalise_kernel")
            metrics=row_metrics(logp,example["targets"])
            for i,region in enumerate(example["rows"]):
                rows.append({k:example[k] for k in ("scene","split","q_id","k_id","baseline_group","condition","mask")} |
                            dict(query_region=int(region),method=method,retained_target_mass=float(example["retained"][i]),
                                 **{k:float(v[i]) for k,v in metrics.items()}))
    return rows


def write_results(out,phase,rows,audits,camera_stats,hashes,verification,protocol_hash):
    summaries,differences={},{}
    scenes=sorted({r["scene"] for r in rows})
    splitgroups=["all"]+sorted({r["split"] for r in rows})
    for condition,_,_ in CONDITIONS:
        for mask in ("conditional","strict95"):
            for scene in ["all"]+scenes:
                for split in splitgroups:
                    groups={method:[r for r in rows if r["condition"]==condition and r["mask"]==mask and r["method"]==method
                                    and (scene=="all" or r["scene"]==scene) and (split=="all" or r["split"]==split)]
                            for method in METHODS}
                    if not groups["full"]: continue
                    for method,subset in groups.items():
                        summaries[f"{condition}/{mask}/{scene}/{split}/{method}"]=summarise(subset)
                    for control in METHODS[1:]:
                        for metric in ("soft_ce","kl","top1_target_mass","top1_max_bin_accuracy"):
                            differences[f"{condition}/{mask}/{scene}/{split}/full_minus_{control}/{metric}"]=cluster_difference(groups["full"],groups[control],metric)
    output=dict(phase=phase,protocol_sha256=protocol_hash,protocol=protocol(),summary=summaries,differences=differences,
                loading_audit=audits,camera_statistics=camera_stats,source_sha256=hashes,verification=verification)
    (out/"summary.json").write_text(json.dumps(output,indent=2,allow_nan=False))
    with (out/"per_row.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    source=Path(__file__).resolve()
    shutil.copy2(source,out/"source_snapshot.py")
    (out/"source_sha256.json").write_text(json.dumps({str(source):sha256(source)},indent=2))
    lines=[f"# GS covariance decomposition: {phase}","",
           "All methods use identical regional centers, opacity weights, primitive-center visibility, key/row masks and numerical covariance floor. No kernel parameter is fitted. The targets are sparse SfM-track transfer, not dense overlap.","",
           "| Scene | split | covariance | rows | CE | KL | top1 target mass | max-bin accuracy |",
           "|---|---|---|---:|---:|---:|---:|---:|"]
    for scene in ["all"]+scenes:
        for split in splitgroups:
            for method in METHODS:
                key=f"primary/conditional/{scene}/{split}/{method}"
                if key not in summaries:continue
                r=summaries[key]
                lines.append(f"| {scene} | {split} | {method} | {r['rows']} | {r['soft_ce']:.5g} | {r['kl']:.5g} | {r['top1_target_mass']:.5g} | {r['top1_max_bin_accuracy']:.5g} |")
    lines += ["","Full minus center-only paired camera differences (negative CE favors internal GS covariance):","",
              "| partition | mask | scene | split | CE difference | camera-cluster 95% CI |","|---|---|---|---|---:|---|"]
    for condition,_,_ in CONDITIONS:
        for mask in ("conditional","strict95"):
            for scene in ["all"]+scenes:
                for split in splitgroups:
                    key=f"{condition}/{mask}/{scene}/{split}/full_minus_centeronly/soft_ce"
                    if key not in differences:continue
                    d=differences[key]
                    lines.append(f"| {condition} | {mask} | {scene} | {split} | {d['mean']:.5g} | [{d['ci95'][0]:.5g}, {d['ci95'][1]:.5g}] |")
    lines += ["","`centeronly` keeps the full regional covariance of primitive means; it is a shape-rich point-cloud control. `internalonly` keeps opacity-weighted individual GS covariances while retaining the same regional mean. `flooronly` uses the fixed tiny numerical floor; it is not a tuned center RBF competitor. Raw center + internal moments sum exactly before independent eigenvalue flooring. All fits are replayed against the original float64 exporter, and serialized original float32 arrays are checked separately. Camera bootstraps do not establish generalization to a population of scenes."]
    (out/"README.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"phase":phase,"scenes":scenes,"rows":len(rows),"verification":verification,
                      "primary_full_minus_centeronly":differences["primary/conditional/all/all/full_minus_centeronly/soft_ce"]},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=("development","confirmation"),default="development")
    parser.add_argument("--out",type=Path,default=Path("results/round3_cov_decomposition"))
    parser.add_argument("--development-data",type=Path,default=Path("results/round3_real_support_data_adjacent"))
    parser.add_argument("--external-data",type=Path,default=Path("results/round3_external_scenes"))
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    frozen_path=args.out/"frozen_protocol.json"
    if not frozen_path.exists():
        if args.phase!="development":raise ValueError("preregister protocol before confirmation")
        frozen_path.write_text(json.dumps(protocol(),indent=2))
    if json.loads(frozen_path.read_text())!=protocol():raise ValueError("frozen protocol mismatch")
    phaseout=args.out/args.phase
    if (phaseout/"summary.json").exists():raise ValueError("refusing to overwrite completed phase")
    phaseout.mkdir(parents=True,exist_ok=True)
    started=time.time()
    all_examples=[];audits=[];camera_stats=[];hashes={};verification=[]
    configs=[(s,args.development_data,("train","validation")) for s in DEV_SCENES] if args.phase=="development" else (
        [(s,args.development_data,("test",)) for s in DEV_SCENES]+[(s,args.external_data,("train","validation","test")) for s in EXTERNAL_SCENES])
    for scene,root,splits in configs:
        result=collect_scene(scene,root,phaseout/"pairs"/scene,splits)
        examples,audit,cameras,source_hashes,verify=result
        all_examples.extend(examples);audits.extend(audit);camera_stats.extend(cameras);hashes.update(source_hashes);verification.append(verify)
        print(f"{args.phase}: {scene}: {len(examples)} masked pair configurations",flush=True)
    rows=evaluate(all_examples)
    write_results(phaseout,args.phase,rows,audits,camera_stats,hashes,verification,sha256(frozen_path))
    (phaseout/"run_time.json").write_text(json.dumps({"seconds":time.time()-started},indent=2))


if __name__=="__main__":
    main()
