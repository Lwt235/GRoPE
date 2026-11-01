"""Frozen isometric-PE attribution: replace only key GS internal covariance.

This is an exploratory counterfactual of already trained checkpoints, not a
comparison between retrained architectures. Completion of all 45 training
records is a mandatory gate before any evaluation/checkpoint/pair is loaded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from experiments.real_support_data import project_gaussian
from experiments.region_pe_probe import evaluate, real_dataset
from models.gaussian_pe import GaussianPEMatcher

SEEDS = [11, 22, 33, 44, 55]
SCENES = ["teatime", "waldo_kitchen"]
METHOD = "isometric_shape"
BOOTSTRAP = 4000


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_digest(tensor):
    a = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str((a.shape,a.dtype)).encode()+a.tobytes()).hexdigest()


def state_digest(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(tensor_digest(value).encode())
    return h.hexdigest()


def protocol():
    return dict(method=METHOD,seeds=SEEDS,scenes=SCENES,split="test",partitions=["primary","alt"],
        training_gate="manifest contains elapsed_seconds and exactly 45 unique seed/method records",
        intervention="replace projected_kcov only: project floored centeronly world covariance at original serialized key mean using query-camera J_KR; divide by 24^2; add original loader 1e-6 I once",
        unchanged=["original image RGB","encoder and all checkpoint weights","query fixed rectangular covariance",
                   "query/key world means","key-to-query projected mean","query/key visibility masks","original >=95% retained-target-mass row mask","targets","model resolution 0.25 applied by model once"],
        floor="use exact existing decomposition centeronly arrays; do not refit or refloor",
        scoring="trained model probabilities with invalid candidates masked exactly as the original loader/evaluator",
        primary_metric="cross_entropy",difference="full minus centeronly; negative CE favors original full covariance",
        confidence="paired crossed bootstrap resampling five seeds and query-camera clusters independently, 4,000 draws",
        inference_device="cpu",tuning="none",
        scope="counterfactual frozen replay, not retraining; cannot rank the best attainable performance of alternative learned structures",
        archive_replay="original manifest aggregate metrics and per_scene pair metrics; original training did not archive token probabilities. New full and counterfactual probabilities are saved here.")


def require_training_complete(root):
    path=root/"manifest.json"
    if not path.exists():raise RuntimeError("training manifest does not yet exist")
    manifest=json.loads(path.read_text())
    records=manifest.get("records",[])
    expected={(seed,method) for seed in SEEDS for method in manifest.get("config",{}).get("methods","").split(",")}
    observed={(r["seed"],r["method"]) for r in records}
    if "elapsed_seconds" not in manifest or len(records)!=45 or len(observed)!=45 or expected!=observed:
        raise RuntimeError(f"training still incomplete: elapsed_seconds={'elapsed_seconds' in manifest}, records={len(records)}, unique={len(observed)}; no evaluation performed")
    if len(expected)!=45:raise ValueError("unexpected experiment matrix")
    if manifest["config"]["resolution"]!=.25:raise ValueError("unexpected model resolution")
    for seed in SEEDS:
        if (seed,METHOD) not in observed:raise ValueError("missing required checkpoint record")
    return manifest


def counterfactual_dataset(root,decomposition,scene,alternate,hashes):
    data,metadata=real_dataset(root,scene,"test",alternate)
    full={k:v.clone() for k,v in data.items()}
    cf={k:v.clone() for k,v in data.items()}
    suffix="_alt" if alternate else ""
    diagnostics=[]
    for index,pair in enumerate(metadata["pairs"]):
        source=Path(root)/scene/pair["file"]
        decomp=Path(decomposition)/"confirmation"/"pairs"/scene/pair["file"]
        hashes[str(source)]=sha256(source);hashes[str(decomp)]=sha256(decomp)
        if hashes[str(source)]!=pair["sha256"]:raise ValueError("source pair differs from loader manifest")
        with np.load(source,allow_pickle=False) as f:
            mean=f["k_mean"+suffix].astype(np.float64)
            cov=f["k_cov"+suffix].astype(np.float64)
            query_cov=f["q_pixel_cov"].copy()
            original_projection=f["k_projected_cov"+suffix].astype(np.float64)
            K,R,t=f["K_query"],f["R_query"],f["t_query"]
            key_valid=f["k_valid"+suffix].astype(bool)
        with np.load(decomp,allow_pickle=False) as f:
            center=f["k_centeronly"+suffix].copy()
            np.testing.assert_array_equal(f["k_valid"+suffix],key_valid)
            np.testing.assert_allclose(f["k_mean"+suffix],mean,rtol=1e-6,atol=1e-7)
        _,full_projection,_=project_gaussian(mean,cov,K,R,t)
        _,center_projection,_=project_gaussian(mean,center,K,R,t)
        # Agreement checks valid entries only; the original exporter replaces
        # invalid projected covariances with a fixed I placeholder.
        relative=np.linalg.norm(full_projection[key_valid]-original_projection[key_valid],axis=(-2,-1))/np.maximum(
            np.linalg.norm(original_projection[key_valid],axis=(-2,-1)),1e-20)
        if len(relative) and np.max(relative)>1e-4:raise ValueError("query-camera covariance projection did not reproduce original export")
        replacement=cf["projected_kcov"][index].clone()
        replacement[key_valid]=torch.as_tensor(center_projection[key_valid]/24**2+np.eye(2)*1e-6,dtype=replacement.dtype)
        cf["projected_kcov"][index]=replacement
        np.testing.assert_array_equal(data["projected_qcov"][index].numpy(),(query_cov/24**2).astype(np.float32))
        if torch.linalg.eigvalsh(replacement).min()<=0:raise ValueError("counterfactual covariance is not SPD")
        diagnostics.append(dict(scene=scene,alternate=alternate,file=pair["file"],q_id=pair["q_id"],k_id=pair["k_id"],
            valid_keys=int(key_valid.sum()),projection_relative_error_max=float(relative.max()) if len(relative) else None,
            full_trace_mean=float(data["projected_kcov"][index][key_valid].diagonal(dim1=-2,dim2=-1).sum(-1).mean()),
            centeronly_trace_mean=float(replacement[key_valid].diagonal(dim1=-2,dim2=-1).sum(-1).mean())))
    for key in full:
        if key!="projected_kcov":
            if not torch.equal(full[key],cf[key]):raise ValueError(f"intervention unexpectedly changed {key}")
    return full,cf,metadata,diagnostics


@torch.no_grad()
def probabilities(model,data):
    blocks=[]
    for start in range(0,len(data["q"]),32):
        part={k:v[start:start+32] for k,v in data.items()}
        score=model(part)[...,:-1]
        score=score.masked_fill(~part["key_valid"][...,None,:],-10000.)
        blocks.append(score.softmax(-1).cpu())
    return torch.cat(blocks)


def per_row(probability,data,metadata,seed,scene,alternate,condition):
    p=probability.double();y=data["soft_labels"].double()
    ce=-(y*p.clamp_min(1e-30).log()).sum(-1)
    entropy=-(y*y.clamp_min(1e-30).log()).sum(-1)
    mixture=(p+y)/2
    js=.5*(y*(y.clamp_min(1e-30)/mixture.clamp_min(1e-30)).log()+p*(p.clamp_min(1e-30)/mixture.clamp_min(1e-30)).log()).sum(-1)
    topmass=y.gather(-1,p.argmax(-1)[...,None])[...,0]
    accuracy=(p.argmax(-1)==y.argmax(-1)).double()
    metrics={"cross_entropy":ce,"kl":ce-entropy,"js":js,"top1_target_mass":topmass,"dominant_region_accuracy":accuracy}
    result=[]
    for i,pair in enumerate(metadata["pairs"]):
        for row in data["label_valid"][i].nonzero(as_tuple=False).flatten().tolist():
            result.append(dict(seed=seed,scene=scene,partition="alt" if alternate else "primary",condition=condition,
                q_id=pair["q_id"],k_id=pair["k_id"],query_region=row,file=pair["file"],
                **{k:float(v[i,row]) for k,v in metrics.items()}))
    return result


def paired_bootstrap(rows,metric,scene="all",partition="primary"):
    subset=[r for r in rows if r["partition"]==partition and (scene=="all" or r["scene"]==scene)]
    def key(r):return r["seed"],r["scene"],r["q_id"],r["k_id"],r["query_region"]
    full={key(r):r for r in subset if r["condition"]=="full"}
    center={key(r):r for r in subset if r["condition"]=="centeronly"}
    if full.keys()!=center.keys():raise ValueError("paired rows differ")
    clusters=sorted({(r["scene"],r["q_id"]) for r in full.values()})
    cluster_index={c:i for i,c in enumerate(clusters)}
    sums=np.zeros((len(SEEDS),len(clusters)))
    counts=np.zeros((len(SEEDS),len(clusters)),int)
    full_sum=np.zeros_like(sums);center_sum=np.zeros_like(sums)
    for k,a in full.items():
        si=SEEDS.index(a["seed"]);ci=cluster_index[(a["scene"],a["q_id"])];b=center[k]
        sums[si,ci]+=a[metric]-b[metric];counts[si,ci]+=1
        full_sum[si,ci]+=a[metric];center_sum[si,ci]+=b[metric]
    np.testing.assert_array_equal(counts,np.broadcast_to(counts[:1],counts.shape))
    camera_counts=counts[0]
    rng=np.random.default_rng(682304)
    seed_draws=rng.integers(0,len(SEEDS),(BOOTSTRAP,len(SEEDS)))
    cluster_draws=rng.integers(0,len(clusters),(BOOTSTRAP,len(clusters)))
    samples=sums[seed_draws[:,:,None],cluster_draws[:,None,:]].sum((1,2))/(len(SEEDS)*camera_counts[cluster_draws].sum(1))
    return dict(full_mean=float(full_sum.sum()/counts.sum()),centeronly_mean=float(center_sum.sum()/counts.sum()),
        full_minus_centeronly=float(sums.sum()/counts.sum()),ci95=np.quantile(samples,[.025,.975]).tolist(),
        seeds=len(SEEDS),query_cameras=len(clusters),rows_per_seed=int(camera_counts.sum()),
        per_seed=[dict(seed=seed,full=float(full_sum[i].sum()/counts[i].sum()),centeronly=float(center_sum[i].sum()/counts[i].sum()),
                       difference=float(sums[i].sum()/counts[i].sum())) for i,seed in enumerate(SEEDS)],
        per_camera=[dict(scene=c[0],q_id=c[1],rows_per_seed=int(camera_counts[i]),difference=float(sums[:,i].sum()/counts[:,i].sum())) for i,c in enumerate(clusters)])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training",type=Path,default=Path("results/round3_isometric_real_confirmation"))
    p.add_argument("--data",type=Path,default=Path("results/round3_real_support_data_adjacent"))
    p.add_argument("--decomposition",type=Path,default=Path("results/round3_cov_decomposition"))
    p.add_argument("--out",type=Path,default=Path("results/round3_isometric_cov_counterfactual"))
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    frozen=args.out/"frozen_protocol.json"
    if not frozen.exists():frozen.write_text(json.dumps(protocol(),indent=2))
    if json.loads(frozen.read_text())!=protocol():raise ValueError("counterfactual protocol changed")
    # Only this manifest may be inspected before the authoritative terminal gate.
    try:manifest=require_training_complete(args.training)
    except RuntimeError as error:
        (args.out/"waiting_status.json").write_text(json.dumps({"status":"waiting_for_completed_training","reason":str(error),"evaluation_started":False},indent=2))
        print(str(error),flush=True);return
    if (args.out/"summary.json").exists():raise ValueError("refusing to overwrite finished counterfactual")
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    started=time.time();hashes={};rows=[];diagnostics=[];records=[];replay=[]
    for rel in ["manifest.json","per_scene.npz"]:hashes[str(args.training/rel)]=sha256(args.training/rel)
    for rel in ["models/gaussian_pe.py","models/geometry.py","experiments/region_pe_probe.py","experiments/real_support_data.py"]:
        archived=args.training/"source"/rel
        expected=manifest["source_hashes"][rel]
        if sha256(rel)!=expected or sha256(archived)!=expected:raise ValueError(f"source differs from trained snapshot: {rel}")
        hashes[rel]=expected;hashes[str(archived)]=expected
    (args.out/"evaluation_started.json").write_text(json.dumps(dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        completed_training_manifest_sha256=hashes[str(args.training/'manifest.json')],frozen_protocol_sha256=sha256(frozen),
        expected_training_records=45,actual_training_records=len(manifest["records"])),indent=2))
    archived=np.load(args.training/"per_scene.npz",allow_pickle=False)
    cache={}
    for scene in SCENES:
        hashes[str(args.data/scene/"manifest.json")]=sha256(args.data/scene/"manifest.json")
        for alternate in (False,True):
            full,center,metadata,diag=counterfactual_dataset(args.data,args.decomposition,scene,alternate,hashes)
            cache[(scene,alternate)]=(full,center,metadata);diagnostics.extend(diag)
    probability_dir=args.out/"probabilities";probability_dir.mkdir(exist_ok=True)
    for seed in SEEDS:
        checkpoint_path=args.training/"checkpoints"/f"{seed}-{METHOD}.pt"
        hashes[str(checkpoint_path)]=sha256(checkpoint_path)
        checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=True)
        if checkpoint["method"]!=METHOD or checkpoint["resolution"]!=.25:raise ValueError("wrong checkpoint settings")
        model=GaussianPEMatcher(METHOD,checkpoint["features"],checkpoint["resolution"]).eval()
        model.load_state_dict(checkpoint["state_dict"]);before=state_digest(model)
        record=next(r for r in manifest["records"] if r["seed"]==seed and r["method"]==METHOD)
        for (scene,alternate),(full,center,metadata) in cache.items():
            condition_name=scene+("_alt" if alternate else "")
            original_metrics,original_per=evaluate(model,full)
            for metric,expected in record["test"][condition_name].items():
                np.testing.assert_allclose(original_metrics[metric],expected,rtol=1e-7,atol=1e-8)
            per_max=0.
            for metric,actual in original_per.items():
                expected=archived[f"{seed}/{METHOD}/{condition_name}/{metric}"]
                np.testing.assert_allclose(actual,expected,rtol=1e-7,atol=1e-8,equal_nan=True)
                finite=np.isfinite(actual)&np.isfinite(expected)
                if finite.any():per_max=max(per_max,float(np.max(abs(actual[finite]-expected[finite]))))
            full_probability=probabilities(model,full)
            center_probability=probabilities(model,center)
            # Independent full replay after the intervention also proves that
            # the replacement never mutated original tensors or model state.
            if not torch.equal(full_probability,probabilities(model,full)):raise ValueError("full replay mutated")
            if state_digest(model)!=before:raise ValueError("checkpoint state changed during inference")
            for condition,probability,data in [("full",full_probability,full),("centeronly",center_probability,center)]:
                rows.extend(per_row(probability,data,metadata,seed,scene,alternate,condition))
                metrics,_=evaluate(model,data)
                records.append(dict(seed=seed,scene=scene,partition="alt" if alternate else "primary",condition=condition,metrics=metrics))
            save=probability_dir/f"{seed}-{condition_name}.npz"
            np.savez_compressed(save,full=full_probability.numpy(),centeronly=center_probability.numpy(),
                                label_valid=full["label_valid"].numpy(),key_valid=full["key_valid"].numpy())
            replay.append(dict(seed=seed,scene=scene,alternate=alternate,aggregate_metrics_match=True,
                pair_metrics_max_abs_error=per_max,full_prediction_replay_exact=True,model_state_unchanged=True,
                full_probability_sha256=tensor_digest(full_probability),center_probability_sha256=tensor_digest(center_probability),
                probability_archive_sha256=sha256(save)))
        print(f"replayed seed {seed}",flush=True)
    differences={}
    for partition in ("primary","alt"):
        for scene in ["all"]+SCENES:
            for metric in ["cross_entropy","kl","js","top1_target_mass","dominant_region_accuracy"]:
                differences[f"{partition}/{scene}/{metric}"]=paired_bootstrap(rows,metric,scene,partition)
    if sha256(args.training/"manifest.json")!=hashes[str(args.training/"manifest.json")]:raise ValueError("training manifest changed during replay")
    output=dict(protocol=protocol(),protocol_sha256=sha256(frozen),records=records,differences=differences,
        replay_audit=replay,projection_audit=diagnostics,source_sha256=hashes,seconds=time.time()-started)
    (args.out/"summary.json").write_text(json.dumps(output,indent=2,allow_nan=False))
    with (args.out/"per_row.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    shutil.copy2(Path(__file__),args.out/"source_snapshot.py")
    (args.out/"source_sha256.json").write_text(json.dumps({str(Path(__file__)):sha256(__file__)},indent=2))
    lines=["# Frozen isometric PE: internal-GS covariance counterfactual","",
        "Only the key projected covariance is replaced with the regional covariance of primitive centers. This is a frozen checkpoint intervention, not retraining. Query covariance, all means, labels, masks, camera, content encoder and checkpoint parameters remain identical. Resolution 0.25 is added exactly once inside the unchanged model.","",
        "| partition | scene | full CE | centeronly CE | full−centeronly | crossed seed/camera 95% CI |","|---|---|---:|---:|---:|---|"]
    for partition in ("primary","alt"):
        for scene in ["all"]+SCENES:
            d=differences[f"{partition}/{scene}/cross_entropy"]
            lines.append(f"| {partition} | {scene} | {d['full_mean']:.6f} | {d['centeronly_mean']:.6f} | {d['full_minus_centeronly']:+.6f} | [{d['ci95'][0]:+.6f},{d['ci95'][1]:+.6f}] |")
    lines += ["","All original full aggregate and per-pair archived metrics match. The original experiment did not save token probabilities, so archive validation cannot establish bitwise identity to a historical token-probability file. This replay saves both probability tensors and verifies an independent repeated original-full forward pass exactly. All checkpoint and source hashes, row metrics, projection errors and seed/camera differences are retained in summary.json."]
    (args.out/"README.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({key:value for key,value in differences.items() if key.endswith('/cross_entropy')},indent=2),flush=True)


if __name__=="__main__":main()
