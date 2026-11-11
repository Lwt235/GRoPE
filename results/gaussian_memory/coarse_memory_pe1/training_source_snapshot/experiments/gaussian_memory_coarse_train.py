"""Matched finite all-layer query adaptation, two balanced memory capacities."""
import argparse
import hashlib
import json
import time
import numpy as np
import torch
from models.gaussian_memory_coarse_pe import CoarseReader, forward_query, descriptors
from models.gaussian_memory_native_pe import loss_for
from experiments.gaussian_memory_coarse_data import *
from experiments.gaussian_memory_coarse_cache import to_device
from experiments.gaussian_memory_native_run import model_load, scores, metric_depth


def digest(adapter):
    h=hashlib.sha256()
    for k,v in adapter.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def sequence(seed,p,steps):
    keys=[k for k in example_keys(p,'train') if k[3]==256]
    rng=np.random.default_rng(seed);indices=[]
    while len(indices)<steps*2:indices.extend(rng.permutation(len(keys)).tolist())
    return [[(*keys[int(i)][:3],CAPACITIES[s%2]) for i in indices[s*2:s*2+2]] for s in range(steps)]


def dataset(p,kind,keys):
    rows={}
    for key in keys:
        path=sample_path(*key);record=read(OUT/'cache'/(keyname(*key)+'.json'))
        assert record['plan_sha256']==sha256(OUT/'plan.json') and sha256(path)==record['sha256']
        cpu=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        z=descriptors(cpu,kind).cpu()
        rows[tuple(key)]=(cpu,z)
    return rows


def fit(model,data,p,kind,seed,steps,folder,pilot=False):
    folder.mkdir(parents=True,exist_ok=True);torch.manual_seed(seed)
    adapter=CoarseReader().cuda();optimizer=torch.optim.AdamW(adapter.parameters(),lr=p['lr'],weight_decay=p['weight_decay'])
    schedule=sequence(seed,p,steps)
    if pilot:
        keys=list(data);schedule=[[keys[(s*2)%len(keys)],keys[(s*2+1)%len(keys)]] for s in range(steps)]
    identity={'kind':kind,'seed':seed,'parameters':sum(t.numel() for t in adapter.parameters()),
        'plan_sha256':sha256(OUT/'plan.json'),'training_lock_sha256':sha256(OUT/'training_lock.json') if (OUT/'training_lock.json').exists() else None,
        'initial_state_sha256':digest(adapter),'schedule_sha256':hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),'pilot':pilot}
    first=0
    if (folder/'latest.pt').exists():
        old=load(folder/'latest.pt')
        for k,v in identity.items():assert old[k]==v
        adapter.load_state_dict(old['model']);optimizer.load_state_dict(old['optimizer']);first=old['step']
    else:save_tensor(folder/'initial.pt',dict(identity,model=adapter.state_dict()))
    logpath=folder/'training.jsonl'
    if logpath.exists():
        lines=logpath.read_text().splitlines();valid=[x for x in lines if json.loads(x)['step']<=first]
        if len(valid)!=len(lines):
            (folder/f'interrupted_after_{first}.jsonl').write_text('\n'.join(lines[len(valid):])+'\n');logpath.write_text('\n'.join(valid)+('\n' if valid else ''))
    begin=time.monotonic();torch.cuda.reset_peak_memory_stats()
    with logpath.open('a',buffering=1) as log:
        for step in range(first,steps):
            optimizer.zero_grad(set_to_none=True);total=0.
            for key in schedule[step]:
                cpu,z=data[tuple(key)];sample=to_device(cpu);zg=z.cuda(non_blocking=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    depth=forward_query(model,sample,adapter,kind,zg);loss=loss_for(depth,sample)
                if not torch.isfinite(loss):raise FloatingPointError(f'loss {kind}/{seed}/{step}')
                (loss/2).backward();total+=float(loss.detach())/2;del sample,zg,depth,loss
            norm=torch.nn.utils.clip_grad_norm_(adapter.parameters(),p['gradient_clip'])
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
            grads={n:None if t.grad is None else float(t.grad.norm()) for n,t in adapter.named_parameters()} if step in (0,1,19,steps-1) else None
            optimizer.step()
            log.write(json.dumps(dict(step=step+1,loss=total,gradient_norm=float(norm),samples=schedule[step],elapsed_s=time.monotonic()-begin,gradient_by_parameter=grads))+'\n')
            if (step+1)%25==0 or step+1==steps:
                torch.cuda.synchronize();rows=[json.loads(x) for x in logpath.read_text().splitlines()]
                progress=dict(identity,step=step+1,steps=steps,last20_loss=float(np.mean([r['loss'] for r in rows[-20:]])),elapsed_s=time.monotonic()-begin,peak_GiB=torch.cuda.max_memory_allocated()/2**30)
                write_json(folder/'progress.json',progress);print(json.dumps(progress),flush=True)
            if (step+1)%200==0 or step+1==steps:save_tensor(folder/'latest.pt',dict(identity,step=step+1,model=adapter.state_dict(),optimizer=optimizer.state_dict()))
    final=folder/f'step_{steps}.pt';save_tensor(final,dict(identity,step=steps,model=adapter.state_dict()))
    rows=[json.loads(x) for x in logpath.read_text().splitlines()]
    status=dict(identity,complete=True,steps=steps,checkpoint_sha256=sha256(final),first20=float(np.mean([r['loss'] for r in rows[:20]])),last20=float(np.mean([r['loss'] for r in rows[-20:]])),seconds_this_invocation=time.monotonic()-begin,peak_GiB=torch.cuda.max_memory_allocated()/2**30)
    write_json(folder/'status.json',status);del adapter,optimizer;torch.cuda.empty_cache();return status


def preflight(model,p):
    keys=[(rid,11,'persistent',n) for rid in p['training_ids'][:2] for n in CAPACITIES]
    results={}
    for kind in KINDS:
        data=dataset(p,kind,keys);folder=OUT/'preflight'/kind
        results[kind]=fit(model,data,p,kind,SEEDS[0],4,folder,True)
        rows=[json.loads(x) for x in (folder/'training.jsonl').read_text().splitlines()]
        g=rows[1]['gradient_by_parameter']
        assert g['q.0.b.weight']>0 and g['q.23.b.weight']>0 and g['relation.2.weight']>0
        del data;torch.cuda.empty_cache()
    write_json(OUT/'preflight/complete.json',dict(complete=True,resources=results,pilot_models_excluded=True,development_scores_used=False,plan_sha256=sha256(OUT/'plan.json')))


@torch.inference_mode()
def evaluate(model,p,kind,seed):
    checkpoint=OUT/f'{kind}_{seed}'/f'step_{p["steps"]}.pt';state=load(checkpoint)
    adapter=CoarseReader().cuda().eval();adapter.load_state_dict(state['model'])
    rows=[];pred=[];keys=[];begin=time.monotonic()
    for key in example_keys(p,'development'):
        cpu=load(sample_path(*key));s=to_device(cpu);z=descriptors(cpu,kind)
        with torch.autocast('cuda',dtype=torch.bfloat16):d=forward_query(model,s,adapter,kind,z,False)
        d=d.float().cpu();rows.append(dict(id=cpu['id'],place=cpu['place'],condition=cpu['condition'],capacity=cpu['capacity'],kind=kind,seed=seed,**scores(d,cpu)))
        pred.append(metric_depth(d));keys.append(list(key));del cpu,s,z,d
    path=CACHE/'evaluation'/f'{kind}_{seed}.pt';path.parent.mkdir(parents=True,exist_ok=True);save_tensor(path,dict(keys=keys,depth_metric=torch.stack(pred)))
    write_json(OUT/'evaluation'/f'{kind}_{seed}.json',dict(complete=True,rows=rows,prediction_path=str(path),prediction_sha256=sha256(path),checkpoint_sha256=sha256(checkpoint),plan_sha256=sha256(OUT/'plan.json'),seconds=time.monotonic()-begin))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['preflight','train','evaluate']);parser.add_argument('--kind',choices=KINDS);parser.add_argument('--seed',type=int,choices=SEEDS,default=SEEDS[0]);args=parser.parse_args()
    torch.set_num_threads(4);p=plan();verify(p);model=model_load(p)
    if args.action=='preflight':preflight(model,p);return
    if args.action=='train':
        assert read(OUT/'preflight/complete.json')['complete'] and (OUT/'training_lock.json').exists()
        folder=OUT/f'{args.kind}_{args.seed}'
        if not (folder/'status.json').exists():
            data=dataset(p,args.kind,example_keys(p,'train'));fit(model,data,p,args.kind,args.seed,p['steps'],folder);del data
    evaluate(model,p,args.kind,args.seed)


if __name__=='__main__':main()
