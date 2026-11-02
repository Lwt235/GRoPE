"""Post-result, fixed-checkpoint diagnosis of R50 zoom weakness; no training."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from pathlib import Path
import types
import numpy as np
import torch
from experiments import round50_train as P
from experiments.round50_data import OUT, centers, sha, write
from models.round50_local_frame import LocalFrameMatcher

ROLES=('rope','mlp','hrope','gs_frame','frame_identity')
VARIANTS=('baseline','angular_phase','no_rotary_value','rgb_permuted')
FOLDER=OUT/'zoom_diagnostic'


def plan():
    if (FOLDER/'plan.json').exists():raise FileExistsError('retain plan')
    write(FOLDER/'plan.json',dict(source_sha256=sha(__file__),roles=ROLES,seeds=P.SEEDS,
        train_regime='oracle',input_regime='oracle',conditions=['base','zoom'],variants=VARIANTS,
        trigger='R50 weak zoom generalization, near-tied anisotropy components, saturated base PCK8/AUC',
        scope='old procedural development, post-result diagnostic; no checkpoint selection, retraining or confirmation',
        angular_phase='replace only native phase positions by6*(pixel-31.5)/f; content geometry/intrinsics unchanged',
        no_rotary_value='keep geometry V/output transport; remove only rotary V/output transport. Conventional RoPE/MLP must be unchanged.',
        rgb_permuted='independent per-view/per-example fixed token permutations preserving all geometry/labels',
        decision='A fixed-checkpoint improvement motivates matched retraining, not promotion. Weak RGB sensitivity diagnoses limited appearance reliance. No parameter/angle/temperature sweep.'))
    (FOLDER/'source.py').write_bytes(Path(__file__).read_bytes())


def prepare(data,variant):
    modified=dict(data)
    if variant=='angular_phase':
        for side in ('q','k'):
            focal=data[side+'K'][:,0,0]*64
            modified[side+'pos']=6*(data[side+'pos']*8-31.5)/focal[:,None,None]
    if variant=='rgb_permuted':
        for side_id,side in enumerate(('q','k')):
            orders=np.stack([np.random.default_rng(20261102+1000*side_id+i).permutation(64) for i in range(len(data[side+'rgb']))])
            order=torch.as_tensor(orders,device=data[side+'rgb'].device)
            modified[side+'rgb']=data[side+'rgb'].gather(1,order[...,None].expand(-1,-1,192))
    return modified


@torch.no_grad()
def run(gpu):
    if P.read(FOLDER/'plan.json')['source_sha256']!=sha(__file__):raise ValueError('changed diagnostic source')
    P.check_lock(OUT);device=P.setup(gpu);rows=[];baseline_checks=[]
    for condition in ('base','zoom'):
        data=P.load(OUT,'dev',condition,device)
        metric_data={k:v.cpu().numpy() for k,v in data.items()}
        variants={v:prepare(data,v) for v in VARIANTS}
        for role in ROLES:
            for seed in P.SEEDS:
                model=LocalFrameMatcher(role).to(device)
                model.load_state_dict(torch.load(OUT/'runs/oracle'/role/str(seed)/'model.pt',map_location=device,weights_only=True));model.eval()
                original_transport=model.transport
                for variant in VARIANTS:
                    if variant=='no_rotary_value':
                        def transport(self,value,matrix,pos,operation):
                            out=original_transport(value,matrix,pos,operation)
                            return torch.cat((out[...,:24],value[...,24:]),-1) if operation in ('value','output') else out
                        model.transport=types.MethodType(transport,model)
                    else:model.transport=original_transport
                    d=variants[variant];arrays={k:[] for k in ('argmax','selected_logp','soft_xy')}
                    for start in range(0,len(d['qrgb']),16):
                        batch=P.subset(d,slice(start,start+16));logits=model(batch,'oracle');lp=logits.log_softmax(-1)
                        xy=torch.as_tensor(centers(),dtype=logits.dtype,device=device)
                        values=dict(argmax=logits.argmax(-1),selected_logp=lp.gather(-1,batch['qindices']),soft_xy=lp.exp()@xy)
                        for k,v in values.items():arrays[k].append(v.cpu().numpy())
                    pred={k:np.concatenate(v) for k,v in arrays.items()}
                    dst=FOLDER/'predictions'/role/str(seed)/f'{condition}_{variant}.npz'
                    if dst.exists():raise FileExistsError('retain diagnosis')
                    dst.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dst,**pred)
                    metrics,_=P.metrics_from_predictions(pred,metric_data)
                    rows.append(dict(role=role,seed=seed,condition=condition,variant=variant,**metrics))
                    if variant=='baseline':
                        with np.load(OUT/'predictions/oracle'/role/str(seed)/f'{condition}_oracle.npz') as old:
                            baseline_checks.append(dict(role=role,seed=seed,condition=condition,
                                argmax_changes=int((pred['argmax']!=old['argmax']).sum()),
                                logp_max_error=float(np.max(np.abs(pred['selected_logp']-old['selected_logp'])))))
                del model
            print('DIAG',condition,role,flush=True)
    tables=[]
    for role in ROLES:
        for condition in ('base','zoom'):
            for variant in VARIANTS:
                selected=[r for r in rows if (r['role'],r['condition'],r['variant'])==(role,condition,variant)]
                tables.append(dict(role=role,condition=condition,variant=variant,
                    **{m:float(np.mean([r[m] for r in selected])) for m in ('auc','pck8','epe','soft_epe','ce')}))
    P.check_lock(OUT)
    write(FOLDER/'result.json',dict(rows=rows,tables=tables,baseline_replay=baseline_checks,
        post_result_diagnostic=True,fresh_confirmation=False,trained=False))
    print('DIAGNOSTIC COMPLETE',len(rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','run']);p.add_argument('--gpu',type=int,default=2);a=p.parse_args()
    plan() if a.action=='plan' else run(a.gpu)
