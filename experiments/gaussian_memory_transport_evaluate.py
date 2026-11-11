"""One fixed evaluation for all readers, sources and position functions."""
import hashlib
import json

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_transport_data import OUT, REGIMES, KERNELS, SEEDS, checked, load_record, load_memory, make_model, model_path, name
from experiments.gaussian_memory_oracle_data import input_seeds, model_path as previous_model_path
from experiments.gaussian_memory_surface_evaluate import metrics, baseline, correspondence, consistency
from models.gaussian_memory_surface import SurfaceMemory, camera_query
from models.gaussian_memory_transport import read_inputs


def conditions(reader):
    if reader=='local':return ('normal',)
    common=('normal','removed','shuffled','wrong_camera')
    if reader=='transport':return common+('anchor_removed','residual_removed','routing_frozen')
    if reader=='prior_residual':return common+('residual_removed',)
    return common


@torch.inference_mode()
def evaluate():
    plan=checked();torch.set_num_threads(4);nets={}
    for c in plan['cases']:
        net=make_model(c,plan).cuda().eval();net.load_state_dict(torch.load(model_path(c),map_location='cuda',weights_only=False)['model'])
        nets[name(c)]=net
    large={}
    for seed in SEEDS:
        net=SurfaceMemory('local',plan['old_local_width'],plan['center_variance']).cuda().eval()
        net.load_state_dict(torch.load(previous_model_path('local',seed),map_location='cuda',weights_only=False)['model']);large[seed]=net
    dest=OUT/'evaluation';assert not dest.exists();dest.mkdir();rows=[];full_rows=[];coherence=[];state_checks=[]
    for meta in plan['catalog']:
        rid=meta['id'];d=load_record(meta);raw,rec,target=d['raw'],d['rec'],d['target']
        keep=torch.tensor([r=='evaluation' for r in rec['roles']]);q=rec['query'][keep];uq=sorted(set(q.tolist()))
        x=rec['input_tokens'][keep].cuda();y=target['log_depth_m'][q].cuda();valid=target['valid'][q].cuda();mask=rec['mask'][keep].cuda()
        full_x=rec['teacher_tokens'][uq].cuda();full_y=target['log_depth_m'][uq].cuda();full_v=target['valid'][uq].cuda()
        camera,K,unit=raw['c2w'].cuda(),raw['K'].cuda(),raw['diagnostics']['unit_meters']
        gt,pairs,distances=correspondence(full_y,full_v,camera[uq],K[uq],unit)
        tag={'episode':rid,'scene':meta['row']['scene'],'split':meta['eval_split']}
        predictions={};full_predictions={};parts={};geometry={}
        generator=torch.Generator().manual_seed(int(hashlib.sha256(rid.encode()).hexdigest()[:8],16))
        perm=torch.randperm(256,generator=generator).cuda()
        for regime in REGIMES:
            memory=load_memory(rid,regime,'cuda');g=read_inputs(memory,camera,K,unit)
            shuffled=dict(g,key=torch.cat([g['key'][:,perm,:32],g['key'][...,32:]],-1),value=g['value'][:,perm])
            wrong=read_inputs(memory,camera,K,unit,camera[(torch.arange(13,device='cuda')+4)%13])
            geometry[regime]={'normal':g,'shuffled':shuffled,'wrong_camera':wrong}
            seeds=input_seeds(d,regime)
            state_checks.append({**tag,'regime':regime,'means_max_difference':float((memory['means']-seeds['seed_mean']).abs().max()),
                                 'covars_max_difference':float((memory['covars']-seeds['seed_chol']@seeds['seed_chol'].transpose(-1,-2)).abs().max())})
            for kernel in KERNELS:
                p=baseline(seeds,camera[q],K[q],unit,'seed_'+kernel,plan['center_variance'])
                rows.append(dict(tag,regime=regime,kernel=kernel,reader='baseline',seed=None,condition='normal',**metrics(p,y,valid,mask)))
                predictions[f'{regime}_{kernel}_baseline_normal']=p.cpu()
        geometry['none']={'normal':{'camera_query':camera_query(camera,K,unit)}}
        for c in plan['cases']:
            key=name(c);net=nets[key]
            for condition in conditions(c['reader']):
                source=geometry[c['regime']][condition if condition in ('shuffled','wrong_camera') else 'normal']
                inputs={k:v[q] for k,v in source.items()}
                p,components=net(x,inputs,condition,return_parts=True)
                rows.append(dict(tag,**c,condition=condition,**metrics(p,y,valid,mask)));predictions[key+'_'+condition]=p.cpu()
                if condition=='normal':
                    full=net(full_x,{k:v[uq] for k,v in source.items()});full_predictions[key]=full.cpu()
                    full_rows.append(dict(tag,**c,condition=condition,**metrics(full,full_y,full_v,torch.zeros_like(full_v))))
                    coherence.append(dict(tag,**c,condition=condition,gt_pair_distance_mean_m=float(np.mean(distances)),
                                          **consistency(full,gt,pairs,camera[uq],K[uq],unit)))
                    if c['reader']=='transport':parts[key]={k:v.cpu() for k,v in components.items()}
        for seed,net in large.items():
            c=dict(regime='none',kernel='center',reader='large_local',seed=seed);key=name(c)
            p=net.read(x,None,camera[q],K[q],unit);full=net.read(full_x,None,camera[uq],K[uq],unit)
            predictions[key+'_normal']=p.cpu();full_predictions[key]=full.cpu()
            rows.append(dict(tag,**c,condition='normal',**metrics(p,y,valid,mask)))
            full_rows.append(dict(tag,**c,condition='normal',**metrics(full,full_y,full_v,torch.zeros_like(full_v))))
            coherence.append(dict(tag,**c,condition='normal',gt_pair_distance_mean_m=float(np.mean(distances)),
                                  **consistency(full,gt,pairs,camera[uq],K[uq],unit)))
        torch.save({'episode':meta['row'],'split':meta['eval_split'],'query':q,'full_query':uq,'target':y.cpu(),'valid':valid.cpu(),'mask':mask.cpu(),
                    'predictions':predictions,'full_predictions':full_predictions,'components':parts,'rgb':raw['rgb'],'permutation':perm.cpu()},dest/(rid+'_maps.pt'))
        print(json.dumps({'evaluated':rid,'predictions':len(predictions)}),flush=True)
    write_json(dest/'scores.json',{'rows':rows,'full_query':full_rows,'consistency':coherence,'confirmation_claim':False})
    write_json(OUT/'state_checks.json',{'rows':state_checks})


if __name__=='__main__':evaluate()
