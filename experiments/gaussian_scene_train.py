"""Finite paired L/T/C/P/G query adaptation and exported final-checkpoint predictions."""
import argparse
import hashlib
import json
import time

import numpy as np
import torch

from experiments.gaussian_scene_data import OUT,CACHE,query_path,read,KINDS,SEEDS
from experiments.gaussian_scene_bind import load_tensor,ensure_token_centers
from experiments.gaussian_scene_cache import patch_path,geometry_path
from experiments.gaussian_scene_config import freeze,schedule
from experiments.gaussian_scene_preflight import metric_depth,objective
from experiments.gaussian_memory_native_run import model_load
from experiments.gaussian_memory_incremental_data import save_tensor,write_json
from experiments.gaussian_memory_setup import sha256
from models.gaussian_scene_native import forward_read
from models.gaussian_scene_memory import SceneReader,center_ray_descriptor,contribution_weights


def digest(module):
    h=hashlib.sha256()
    for name,tensor in module.state_dict().items():h.update(name.encode());h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def dataset(plan,kind,training):
    result={};fields={}
    for meta in plan['training_catalog'] if training else plan['catalog']:
        rid=meta['id']
        binding=read(OUT/'binding'/f'{rid}.json');cache=read(OUT/'query_cache'/f'{rid}.json')
        assert cache['plan_sha256']==sha256(OUT/'plan.json')
        assert all(sha256(path)==digest for path,digest in cache['files'].items())
        if kind!='L':
            used_path=str(CACHE/rid/('token_memory.pt' if kind=='T' else 'scene_memory.pt'))
            assert sha256(used_path)==binding['files'][used_path]
        field={} if kind=='L' else load_tensor(CACHE/rid/('token_memory.pt' if kind=='T' else 'scene_memory.pt'),'cuda')
        fields[rid]=field
        for q in plan['queries']:
            query=load_tensor(query_path(rid,q),'cuda');geometry=None
            if kind in ('T','P'):
                K=query['K'].clone();K[:2]/=14
                geometry=center_ray_descriptor(field['means'],query['c2w'],K,32,32,field['unit'])
            elif kind=='G':geometry=load_tensor(geometry_path(rid,q),'cuda')['weights']
            for c in plan['conditions']:
                patches=load_tensor(patch_path(rid,q,c),'cuda')['patches']
                sample=dict(patches=patches,features=patches[0] if kind=='L' else field['features'],
                            valid=torch.ones(1024,dtype=torch.bool,device='cuda') if kind=='L' else field['has_content'],
                            geometry=geometry,hw=(448,448),query=query,condition=c,place=meta['place'])
                result[(rid,q,c)]=sample
    return result,fields


def fit(model,p,kind,seed,folder):
    folder.mkdir(parents=True,exist_ok=True);torch.manual_seed(seed)
    reader=SceneReader(center_pe=kind in ('T','P')).cuda()
    optimizer=torch.optim.AdamW(reader.parameters(),lr=p['lr'],weight_decay=p['weight_decay'])
    order=schedule(p,seed);identity=dict(kind=kind,seed=seed,plan_sha256=sha256(OUT/'plan.json'),
        initial_state_sha256=digest(reader),schedule_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest(),
        parameters=sum(x.numel() for x in reader.parameters()))
    final=folder/'step_2000.pt';marker=folder/'status.json'
    if marker.exists():
        old=read(marker)
        assert all(old[key]==value for key,value in identity.items()) and old['complete']
        assert sha256(final)==old['checkpoint_sha256']
        return
    first=0;latest=folder/'latest.pt'
    if latest.exists():
        state=load_tensor(latest)
        assert all(state[key]==value for key,value in identity.items())
        reader.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);first=state['step']
    else:save_tensor(folder/'initial.pt',dict(identity,model=reader.state_dict()))
    data,fields=dataset(p,kind,True)
    logpath=folder/'training.jsonl'
    if logpath.exists():
        lines=logpath.read_text().splitlines();keep=[line for line in lines if json.loads(line)['step']<=first]
        if len(keep)!=len(lines):
            archive=folder/f'interrupted_after_{first}.jsonl'
            if archive.exists():archive=folder/f'interrupted_after_{first}_{time.time_ns()}.jsonl'
            archive.write_text('\n'.join(lines[len(keep):])+'\n')
            logpath.write_text('\n'.join(keep)+('\n' if keep else ''))
    torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    with logpath.open('a',buffering=1) as log:
        for step in range(first,p['steps']):
            optimizer.zero_grad(set_to_none=True);total=0.
            for key in order[step*2:step*2+2]:
                sample=data[tuple(key)];query=sample['query']
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    depth=forward_read(model,sample,reader,kind)
                    loss=objective(depth,query['target'],query['valid'])
                if not torch.isfinite(loss):raise FloatingPointError((kind,seed,step,'loss'))
                (loss/2).backward();total+=float(loss.detach())/2
                del depth,loss
            norm=torch.nn.utils.clip_grad_norm_(reader.parameters(),p['gradient_clip'])
            if not torch.isfinite(norm):raise FloatingPointError((kind,seed,step,'gradient'))
            optimizer.step()
            log.write(json.dumps(dict(step=step+1,loss=total,gradient_norm=float(norm),samples=order[step*2:step*2+2],elapsed_seconds=time.monotonic()-started))+'\n')
            if (step+1)%100==0 or step+1==p['steps']:
                save_tensor(latest,dict(identity,model=reader.state_dict(),optimizer=optimizer.state_dict(),step=step+1))
                print(dict(kind=kind,seed=seed,step=step+1,loss=total,elapsed_seconds=time.monotonic()-started),flush=True)
            if (step+1)%25==0:
                write_json(folder/'progress.json',dict(identity,step=step+1,total_steps=p['steps'],loss=total,
                    elapsed_seconds=time.monotonic()-started,peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3))
    save_tensor(final,dict(identity,model=reader.state_dict(),step=p['steps']))
    write_json(marker,dict(identity,complete=True,steps=p['steps'],checkpoint_sha256=sha256(final),
        final_state_sha256=digest(reader),seconds_this_session=time.monotonic()-started,
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3))


@torch.no_grad()
def evaluate(model,p,kind,seed,folder):
    marker=OUT/'evaluation'/f'{kind}_{seed}.json';final=folder/'step_2000.pt'
    if marker.exists():
        old=read(marker)
        assert old['complete'] and old['plan_sha256']==sha256(OUT/'plan.json')
        assert old['checkpoint_sha256']==sha256(final) and old['prediction_sha256']==sha256(old['prediction_path'])
        return
    state=load_tensor(final);reader=SceneReader(center_pe=kind in ('T','P')).cuda()
    reader.load_state_dict(state['model']);reader.eval()
    data,fields=dataset(p,kind,False);keys=[];predictions=[];started=time.monotonic()
    for key,sample in data.items():
        with torch.autocast('cuda',dtype=torch.bfloat16):depth=forward_read(model,sample,reader,kind,checkpoint_blocks=False)
        assert torch.isfinite(depth).all() and (depth>0).all()
        predictions.append(metric_depth(depth).float().cpu());keys.append(key)
    path=CACHE/'evaluation'/f'{kind}_{seed}.pt';path.parent.mkdir(parents=True,exist_ok=True)
    save_tensor(path,dict(keys=keys,depth_metric=torch.stack(predictions)))
    write_json(marker,dict(complete=True,kind=kind,seed=seed,predictions=len(keys),prediction_path=str(path),
        prediction_sha256=sha256(path),checkpoint_sha256=sha256(final),plan_sha256=sha256(OUT/'plan.json'),
        seconds=time.monotonic()-started))
    print(dict(evaluation_complete=f'{kind}_{seed}',predictions=len(keys)),flush=True)


@torch.no_grad()
def intervene(model,p,seed,folder):
    marker=OUT/'interventions'/f'G_{seed}.json';final=folder/'step_2000.pt'
    if marker.exists():
        old=read(marker)
        assert old['checkpoint_sha256']==sha256(final) and old['prediction_sha256']==sha256(old['prediction_path'])
        return
    reader=SceneReader().cuda();reader.load_state_dict(load_tensor(final)['model']);reader.eval()
    data,fields=dataset(p,'G',False);permutations={};wrong={}
    for rid,field in fields.items():
        generator=torch.Generator(device='cuda').manual_seed(p['interventions']['shuffle_seed'])
        permutations[rid]=torch.randperm(len(field['means']),generator=generator,device='cuda')
    keys=[];predictions=[];started=time.monotonic()
    for key,sample in data.items():
        rid,q,c=key;field=fields[rid];query=sample['query']
        if (rid,q) not in wrong:
            camera=query['c2w'].clone();camera[:3,3]+=camera[:3,0]*field['unit']*.5
            K=query['K'].clone();K[:2]/=14
            weight,_=contribution_weights(field,camera[None],K[None],32,32)
            wrong[(rid,q)]=weight[0].reshape(1024,-1)
        for name in p['interventions']['kinds']:
            changed=dict(sample)
            if name=='remove_memory':changed['valid']=torch.zeros_like(sample['valid'])
            elif name=='shuffle_binding':
                changed['features']=sample['features'][permutations[rid]]
                changed['valid']=sample['valid'][permutations[rid]]
            elif name=='wrong_camera':changed['geometry']=wrong[(rid,q)]
            with torch.autocast('cuda',dtype=torch.bfloat16):depth=forward_read(model,changed,reader,'G',checkpoint_blocks=False)
            assert torch.isfinite(depth).all() and (depth>0).all()
            keys.append((*key,name));predictions.append(metric_depth(depth).float().cpu())
    path=CACHE/'evaluation'/f'G_{seed}_interventions.pt'
    save_tensor(path,dict(keys=keys,depth_metric=torch.stack(predictions)))
    write_json(marker,dict(complete=True,seed=seed,predictions=len(keys),prediction_path=str(path),
        prediction_sha256=sha256(path),checkpoint_sha256=sha256(final),plan_sha256=sha256(OUT/'plan.json'),
        seconds=time.monotonic()-started,scope='Fixed-model dependencies, not matched retraining; remove also removes trained reader residual'))
    print(dict(interventions_complete=seed,predictions=len(keys)),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--kind',required=True,choices=KINDS)
    parser.add_argument('--seed',required=True,type=int,choices=SEEDS);args=parser.parse_args()
    torch.set_num_threads(4);p=freeze();model=model_load(p);folder=OUT/f'{args.kind}_{args.seed}'
    if args.kind=='T':ensure_token_centers(model,p)
    fit(model,p,args.kind,args.seed,folder);torch.cuda.empty_cache()
    evaluate(model,p,args.kind,args.seed,folder)
    if args.kind=='G':
        torch.cuda.empty_cache();intervene(model,p,args.seed,folder)


if __name__=='__main__':main()
