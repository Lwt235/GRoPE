"""Multiscale region matching: identical inputs, paired seeds, visible-area loss."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from models.round6_measure_attention import Round6MeasureMatcher, METHODS, FIELDS

ROOT = Path(__file__).resolve().parents[1]
GRIDS = ('q8_k8', 'q8_k16', 'q16_k8', 'q16_k16')
LABEL_FIELDS = ('soft_labels', 'gt_valid', 'visible_fraction', 'sample_count',
                'point_gt_valid', 'point_pixcontinuous', 'point_key_index')
SOURCES = ('models/round6_measure_attention.py', 'models/gaussian_measure_pe.py',
           'models/round5_cross_attention.py', 'experiments/round6_train.py',
           'experiments/round6_region_data.py', 'experiments/round5_rgbd_data.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    return {p: sha(ROOT/p) for p in SOURCES}


def load_split(roots, split, device='cpu', limit=None):
    groups, identities, sources = {}, {}, []
    if isinstance(roots, (str, Path)):
        roots = [roots]
    for grid in GRIDS:
        arrays = {k: [] for k in FIELDS+LABEL_FIELDS}
        ids = []
        for root in roots:
            for directory in sorted((Path(root)/split).glob('*/'+grid)):
                m = json.loads((directory/'manifest.json').read_text())
                if m['split'] != split or m['grid'] != grid:
                    raise ValueError('manifest split/grid mismatch')
                for filename, key in [('inputs.npz', 'input_sha256'), ('labels.npz', 'label_sha256')]:
                    if sha(directory/filename) != m[key]:
                        raise ValueError('data cache hash mismatch: '+str(directory))
                n = len(m['pairs']) if limit is None else min(limit, len(m['pairs']))
                with np.load(directory/'inputs.npz') as data:
                    for k in FIELDS:
                        arrays[k].append(data[k][:n])
                with np.load(directory/'labels.npz') as data:
                    for k in LABEL_FIELDS:
                        arrays[k].append(data[k][:n])
                ids.extend(dict(scene=m['scene'], pair_id=f'{q}:{k}', q_index=q, k_index=k,
                    grid=grid, q_stride=m['label_metadata']['q_stride'],
                    k_stride=m['label_metadata']['k_stride']) for q, k in m['pairs'][:n])
                sources.append(dict(path=str(directory), scene=m['scene'], grid=grid,
                    manifest_sha256=sha(directory/'manifest.json'),
                    input_sha256=m['input_sha256'], label_sha256=m['label_sha256']))
        if not ids:
            raise ValueError(f'no {split}/{grid} inputs')
        if len({(r['scene'],r['pair_id']) for r in ids}) != len(ids):
            raise ValueError('duplicate scene/pair identity')
        groups[grid] = {k: torch.from_numpy(np.concatenate(v)).to(device) for k,v in arrays.items()}
        for k in FIELDS+('soft_labels','visible_fraction','sample_count','point_pixcontinuous'):
            groups[grid][k] = groups[grid][k].float()
        for k in ('gt_valid', 'point_gt_valid'):
            groups[grid][k] = groups[grid][k].bool()
        groups[grid]['point_key_index'] = groups[grid]['point_key_index'].long()
        identities[grid] = ids
    return groups, identities, sources


def batch(data, index):
    return {k: v[index] for k,v in data.items()}


def per_query(logits, d):
    lp = F.log_softmax(logits, dim=-1)
    y = d['soft_labels']
    ly = y.clamp_min(1e-30).log()
    ce = -(y*lp).sum(-1)
    entropy = -(y*ly).sum(-1)
    prob = lp.exp()
    lm = ((y+prob)*.5).clamp_min(1e-30).log()
    js = .5*(y*(ly-lm)+prob*(lp-lm)).sum(-1)
    return ce, entropy, js


def measure_rows(logits, d, ids):
    ce, entropy, js = per_query(logits, d)
    valid = d['gt_valid']
    mass = d['visible_fraction']*d['sample_count']
    if ((mass>0) != valid).any():
        raise ValueError('visible mass/validity mismatch')
    kgrid = d['kmean']*8
    predicted = kgrid.gather(1, logits.argmax(-1)[..., None].expand(-1,-1,2))
    soft = logits.softmax(-1)@kgrid
    err = (predicted-d['point_pixcontinuous']).norm(dim=-1)
    soft_err = (soft-d['point_pixcontinuous']).norm(dim=-1)
    point_valid = d['point_gt_valid']
    pce = -F.log_softmax(logits,-1).gather(-1,d['point_key_index'].clamp_min(0)[...,None]).squeeze(-1)
    threshold = ids[0]['k_stride']
    sums = dict(n_valid=valid.sum(-1), visible_mass=mass.sum(-1),
        ce_sum=(ce*mass).sum(-1), entropy_sum=(entropy*mass).sum(-1),
        kl_sum=((ce-entropy)*mass).sum(-1), js_sum=(js*mass).sum(-1),
        row_kl_sum=torch.where(valid,ce-entropy,0).sum(-1),
        n_point=point_valid.sum(-1), pck8_count=((err<=8)&point_valid).sum(-1),
        pck4_count=((err<=4)&point_valid).sum(-1),
        pck_stride_count=((err<=threshold)&point_valid).sum(-1),
        point_ce_sum=torch.where(point_valid,pce,0).sum(-1),
        point_epe_sum=torch.where(point_valid,err,0).sum(-1),
        point_soft_epe_sum=torch.where(point_valid,soft_err,0).sum(-1))
    # Sensitivities are predeclared; they never change training or primary masks.
    for lo, hi in [(0,.25),(.25,.5),(.5,.75),(.75,1.01)]:
        mask=valid&(d['visible_fraction']>=lo)&(d['visible_fraction']<hi)
        prefix=f'coverage_{lo}_{hi}'
        sums[prefix+'_mass']=torch.where(mask,mass,0).sum(-1)
        sums[prefix+'_kl']=torch.where(mask,(ce-entropy)*mass,0).sum(-1)
    values={k:v.cpu().tolist() for k,v in sums.items()}
    return [dict(identity,**{k:v[i] for k,v in values.items()}) for i,identity in enumerate(ids)]


@torch.no_grad()
def evaluate(model, groups, identities, batch_size=8, probability_dir=None):
    model.eval()
    rows=[]
    for grid,dataset in groups.items():
        ids=identities[grid]; saved={}
        for start in range(0,len(ids),batch_size):
            d=batch(dataset,slice(start,start+batch_size))
            inputs={k:d[k] for k in FIELDS}
            logits=model(inputs)
            if not torch.isfinite(logits).all():
                raise ValueError('nonfinite evaluation logits')
            rows.extend(measure_rows(logits,d,ids[start:start+batch_size]))
            if probability_dir is not None:
                probability_dir.mkdir(parents=True,exist_ok=True)
                for j,identity in enumerate(ids[start:start+batch_size]):
                    scene=identity['scene'];count=saved.get(scene,0)
                    if count<2:
                        np.savez_compressed(probability_dir/f'{grid}_{start+j:04d}.npz',
                            probability=logits[j].softmax(-1).cpu().numpy(),
                            **{k:d[k][j].cpu().numpy() for k in LABEL_FIELDS},
                            key_grid=(d['kmean'][j]*8).cpu().numpy(),
                            identity=json.dumps(identity))
                        saved[scene]=count+1
    return rows


def macro(rows, metric='kl_sum', denominator='visible_mass', grids=None):
    grouped={}
    for r in rows:
        if grids is not None and r['grid'] not in grids:continue
        acc=grouped.setdefault((r['scene'],r['grid']),[0.,0.])
        acc[0]+=r[metric];acc[1]+=r[denominator]
    if not grouped or any(n<=0 for _,n in grouped.values()):
        raise ValueError('empty scene/grid metric')
    return float(np.mean([s/n for s,n in grouped.values()]))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--data',type=Path,nargs='+',default=[ROOT/'results/round6_region_inputs'])
    p.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    p.add_argument('--seeds',type=int,nargs='+',default=[11])
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--eval-every',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--bandwidth',type=float,default=1.)
    p.add_argument('--placement',choices=['all','head'],default='all')
    p.add_argument('--train-grids',nargs='+',choices=GRIDS,default=list(GRIDS))
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--limit',type=int)
    args=p.parse_args();torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    args.out.mkdir(parents=True,exist_ok=True)
    train,trids,trsources=load_split(args.data,'train',args.device,args.limit)
    val,vids,vsources=load_split(args.data,'val',args.device,args.limit)
    sourcehash=source_hashes()
    for method in args.methods:
        for seed in args.seeds:
            run=args.out/f'{method}_{seed}'
            if run.exists():raise ValueError(f'refusing overwrite {run}')
            (run/'checkpoints').mkdir(parents=True)
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            if args.device.startswith('cuda'):torch.cuda.manual_seed_all(seed)
            model=Round6MeasureMatcher(method,args.bandwidth,args.placement).to(args.device)
            config={k:([str(x) for x in v] if k=='data' else str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()}
            (run/'config.json').write_text(json.dumps(dict(arguments=config,method=method,seed=seed,
                train_sources=trsources,val_sources=vsources,sourcehash=sourcehash,
                parameters=sum(x.numel() for x in model.parameters())),indent=2))
            optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
            rng=np.random.default_rng(seed+50000);best=math.inf;history=[];losses=[];started=time.time()
            for step in range(args.steps+1):
                if step%args.eval_every==0 or step==args.steps:
                    rows=evaluate(model,val,vids,args.batch_size)
                    score=macro(rows)
                    record=dict(step=step,val_kl=score,
                        val_kl_by_grid={g:macro(rows,grids=[g]) for g in GRIDS},
                        point_pck8=macro(rows,'pck8_count','n_point',['q8_k8']),
                        bandwidth=float(model.bandwidth),train_ce=float(np.mean(losses)) if losses else None,
                        seconds=time.time()-started)
                    history.append(record);losses=[]
                    if score<best:
                        best=score
                        torch.save(dict(model=model.state_dict(),method=method,seed=seed,step=step,
                            sourcehash=sourcehash),run/'checkpoints/best.pt')
                        (run/'val.json').write_text(json.dumps(dict(rows=rows,selected_step=step,input_sources=vsources),indent=2))
                    (run/'history.json').write_text(json.dumps(history,indent=2))
                    print(method,seed,record,flush=True)
                if step==args.steps:break
                grid=args.train_grids[int(rng.integers(len(args.train_grids)))]
                idx=torch.as_tensor(rng.integers(len(trids[grid]),size=args.batch_size),device=args.device)
                d=batch(train[grid],idx);model.train();optimizer.zero_grad(set_to_none=True)
                logits=model({k:d[k] for k in FIELDS})
                ce=-(d['soft_labels']*logits.log_softmax(-1)).sum(-1)
                mass=d['sample_count']*d['visible_fraction']
                loss=(ce*mass).sum()/mass.sum().clamp_min(1)
                if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(grad):raise ValueError('nonfinite gradient')
                optimizer.step();losses.append(float(loss.detach()))
                factor=.2+.8*.5*(1+math.cos(math.pi*(step+1)/args.steps))
                for group in optimizer.param_groups:group['lr']=args.lr*factor
            if source_hashes()!=sourcehash:raise ValueError('source changed during training')
            (run/'completed.json').write_text(json.dumps(dict(best_val_kl=best,
                seconds=time.time()-started,checkpoint_sha256=sha(run/'checkpoints/best.pt'),
                sourcehash=sourcehash),indent=2))


if __name__=='__main__':main()
