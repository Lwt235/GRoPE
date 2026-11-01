"""Paired-seed training and transparent query-only depth metric sums."""
import argparse
import copy
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from models.round7_depth_attention import DepthAttention, FIELDS, CONFIGS, depth_loss
from experiments.round7_depth_data import sha, ROOT

SOURCES = ('models/round7_depth_attention.py','models/gaussian_quadratic_pe.py',
    'experiments/round7_depth_data.py','experiments/round7_depth_train.py',
    'experiments/round6_region_data.py','experiments/round5_rgbd_data.py')

def source_hashes(): return {p:sha(ROOT/p) for p in SOURCES}

def load_data(roots, split, device='cpu'):
    chunks={f:[] for f in FIELDS+('target','valid','boundary')};ids=[];sources=[]
    for root in roots:
        for dest in sorted((Path(root)/split).glob('*')):
            if not (dest/'manifest.json').exists(): continue
            m=json.loads((dest/'manifest.json').read_text())
            if m['split']!=split: raise ValueError('split mismatch')
            for filename,fields,digest in [('inputs.npz',FIELDS,m['input_sha256']),
                    ('labels.npz',('target','valid','boundary'),m['label_sha256'])]:
                if sha(dest/filename)!=digest: raise ValueError('changed '+str(dest/filename))
                with np.load(dest/filename) as arrays:
                    for f in fields: chunks[f].append(arrays[f])
            ids.extend(dict(scene=m['scene'],q_index=q,k_index=k,pair_id=f'{q}:{k}') for q,k in m['pairs'])
            sources.append(dict(path=str(dest),manifest_sha256=sha(dest/'manifest.json'),
                input_sha256=m['input_sha256'],label_sha256=m['label_sha256']))
    if not ids or len(set((d['scene'],d['pair_id']) for d in ids))!=len(ids):
        raise ValueError('empty or duplicate pairs')
    data={f:torch.from_numpy(np.concatenate(v)).to(device) for f,v in chunks.items()}
    return data,ids,sources

def batch(data,indices): return {f:v[indices] for f,v in data.items()}

def metric_rows(prediction,data,ids):
    pred=prediction.double(); valid=data['valid'];boundary=data['boundary']&valid
    target=torch.where(valid,data['target'],torch.ones_like(data['target'])).double()
    delta=(pred-target).abs();rel=delta/target; log=(pred.log()-target.log()).square()
    ratio=torch.maximum(pred/target,target/pred)
    arrays={'count':valid.sum(-1),'boundary_count':boundary.sum(-1),
        'absrel_sum':torch.where(valid,rel,0).sum(-1),
        'abs_sum':torch.where(valid,delta,0).sum(-1),
        'sq_sum':torch.where(valid,delta.square(),0).sum(-1),
        'log_sq_sum':torch.where(valid,log,0).sum(-1),
        'delta1_count':((ratio<1.25)&valid).sum(-1),
        'boundary_absrel_sum':torch.where(boundary,rel,0).sum(-1),
        'boundary_abs_sum':torch.where(boundary,delta,0).sum(-1)}
    cpu={f:v.detach().cpu().numpy() for f,v in arrays.items()}
    return [dict(identity,**{f:float(v[i]) for f,v in cpu.items()}) for i,identity in enumerate(ids)]

def macro(rows):
    by_scene={}
    for scene in sorted({r['scene'] for r in rows}):
        chosen=[r for r in rows if r['scene']==scene]
        sums={f:sum(r[f] for r in chosen) for f in ('count','boundary_count','absrel_sum',
             'abs_sum','sq_sum','log_sq_sum','delta1_count','boundary_absrel_sum','boundary_abs_sum')}
        n,b=sums['count'],sums['boundary_count']
        if n<=0: raise ValueError('scene without query depth labels')
        by_scene[scene]=dict(AbsRel=sums['absrel_sum']/n,MAE=sums['abs_sum']/n,
            RMSE=(sums['sq_sum']/n)**.5,log_RMSE=(sums['log_sq_sum']/n)**.5,
            delta1=sums['delta1_count']/n,boundary_AbsRel=sums['boundary_absrel_sum']/b if b else None,
            boundary_MAE=sums['boundary_abs_sum']/b if b else None,**sums)
    metrics=('AbsRel','MAE','RMSE','log_RMSE','delta1','boundary_AbsRel','boundary_MAE')
    return dict(metrics={f:float(np.mean([m[f] for m in by_scene.values() if m[f] is not None])) for f in metrics},
        by_scene=by_scene,total_count=sum(r['count'] for r in rows),total_positions=len(rows)*432)

@torch.no_grad()
def evaluate(model,data,ids,intervention=None):
    model.eval(); predictions=[]
    for start in range(0,len(ids),16):
        d=batch(data,slice(start,start+16))
        predictions.append(model(d,intervention=intervention))
    pred=torch.cat(predictions);rows=metric_rows(pred,data,ids)
    return dict(rows=rows,summary=macro(rows)),pred

def fit(role,seed,train,train_ids,val,val_ids,steps,out,data_sources):
    dest=out/role/str(seed)
    if dest.exists(): raise ValueError('refuse overwrite '+str(dest))
    dest.mkdir(parents=True);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model=DepthAttention(role).to(train['q'].device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    rng=np.random.default_rng(seed);start=time.time();history=[];best=float('inf');best_state=None
    source=source_hashes();sample_hash=__import__('hashlib').sha256()
    for step in range(steps+1):
        if step%100==0 or step==steps:
            result,_=evaluate(model,val,val_ids);score=result['summary']['metrics']['AbsRel']
            history.append(dict(step=step,validation=result['summary']))
            if score<best:
                best=score;best_step=step;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            print('VAL',role,seed,step,round(score,6),'sec',round(time.time()-start,1),flush=True)
        if step==steps:break
        model.train();indices=rng.integers(len(train_ids),size=8);sample_hash.update(indices.astype('<i8').tobytes())
        d=batch(train,indices);optimizer.zero_grad(set_to_none=True)
        pred=model(d);loss=depth_loss(pred,d['target'],d['valid'])
        if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
    model.load_state_dict(best_state);result,pred=evaluate(model,val,val_ids)
    interventions={}
    for mode in ('zero_v','reverse_v'):
        intervention,changed=evaluate(model,val,val_ids,mode)
        interventions[mode]=dict(summary=intervention['summary'],
            mean_absolute_output_change_m=float((changed-pred).abs().mean()))
    torch.save(dict(role=role,seed=seed,step=best_step,state_dict=best_state,source_hashes=source),dest/'best.pt')
    np.savez_compressed(dest/'val_predictions.npz',depth=pred.detach().cpu().numpy())
    (dest/'validation.json').write_text(json.dumps(result,indent=2))
    metadata=dict(role=role,seed=seed,steps=steps,best_step=best_step,best_validation_AbsRel=best,
        parameter_count=sum(p.numel() for p in model.parameters()),source_hashes=source,data_sources=data_sources,
        batch_sequence_sha256=sample_hash.hexdigest(),checkpoint_sha256=sha(dest/'best.pt'),
        validation_sha256=sha(dest/'validation.json'),prediction_sha256=sha(dest/'val_predictions.npz'),
        history=history,interventions=interventions,seconds=time.time()-start,
        device=str(train['q'].device),GPU=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        max_memory_allocated=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None)
    (dest/'run.json').write_text(json.dumps(metadata,indent=2))
    print('DONE',role,seed,best_step,best,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--roots',nargs='+',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--roles',nargs='+',choices=CONFIGS,default=list(CONFIGS))
    p.add_argument('--seeds',nargs='+',type=int,default=[11,22,33,44,55]);p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--device',default='cuda:0');a=p.parse_args();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    train,ti,ts=load_data(a.roots,'train',a.device);val,vi,vs=load_data(a.roots,'val',a.device)
    a.out.mkdir(parents=True,exist_ok=True)
    for role in a.roles:
        for seed in a.seeds:fit(role,seed,train,ti,val,vi,a.steps,a.out,dict(train=ts,val=vs))

if __name__=='__main__':main()
