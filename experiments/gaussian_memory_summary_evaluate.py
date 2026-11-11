"""Identical fixed-model evaluation on used development and method-unseen places."""
import hashlib
import json

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_setup import sha256
from experiments.gaussian_memory_summary_data import OUT,SEEDS,KINDS,locked,load_record,model,path
from experiments.gaussian_memory_surface_evaluate import metrics
from experiments.gaussian_memory_surface_diagnostic import decompose
from models.gaussian_memory_transport import read_inputs


def conditions(kind):
    return ('normal',) if kind in ('local','large_local') else ('normal','removed','shuffled','wrong_camera')


@torch.inference_mode()
def evaluate():
    plan=locked();torch.set_num_threads(4);cache=json.loads((OUT/'cache_identity.json').read_text())
    for name,digest in cache['sha256'].items():assert sha256(name)==digest,name
    catalog=plan['catalog']+cache['catalog'];nets={}
    for kind in KINDS:
        for seed in SEEDS:
            net=model(kind,plan).cuda().eval();net.load_state_dict(torch.load(path(kind,seed),map_location='cuda',weights_only=False)['model']);nets[(kind,seed)]=net
    dest=OUT/'evaluation';assert not dest.exists();dest.mkdir()
    rows=[];full_rows=[];identities={};invariants=[]
    for meta in catalog:
        d=load_record(meta);raw,rec,target=d['raw'],d['rec'],d['target'];rid=meta['id']
        keep=torch.tensor([r=='evaluation' for r in rec['roles']]);q=rec['query'][keep];uq=sorted(set(q.tolist()))
        x=rec['input_tokens'][keep].cuda();y=target['log_depth_m'][q].cuda();valid=target['valid'][q].cuda();mask=rec['mask'][keep].cuda()
        full_x=rec['teacher_tokens'][uq].cuda();full_y=target['log_depth_m'][uq].cuda();full_v=target['valid'][uq].cuda()
        camera,K,unit=raw['c2w'].cuda(),raw['K'].cuda(),raw['diagnostics']['unit_meters']
        memory={k:v.cuda() for k,v in d['memory'].items()};g=read_inputs(memory,camera,K,unit)
        generator=torch.Generator().manual_seed(int(hashlib.sha256(rid.encode()).hexdigest()[:8],16))
        perm=torch.randperm(256,generator=generator).cuda()
        shuffled=dict(g,key=torch.cat([g['key'][:,perm,:32],g['key'][...,32:]],-1),value=g['value'][:,perm])
        wrong=read_inputs(memory,camera,K,unit,camera[(torch.arange(13,device='cuda')+4)%13])
        geometry={'normal':g,'shuffled':shuffled,'wrong_camera':wrong};predictions={};full_predictions={}
        tag={'episode':rid,'scene':meta['row']['scene'],'split':meta['eval_split']}
        for (kind,seed),net in nets.items():
            for condition in conditions(kind):
                inputs={k:v[q] for k,v in geometry.get(condition,g).items()}
                p=(net.read(x,None,camera[q],K[q],unit) if kind=='large_local' else net(x,inputs,condition))
                result=dict(tag,kind=kind,seed=seed,condition=condition,**metrics(p,y,valid,mask))
                if condition=='normal':
                    dec=decompose(p,y,valid);result.update({k:v for k,v in dec.items() if k!='log_mse'})
                    full=(net.read(full_x,None,camera[uq],K[uq],unit) if kind=='large_local' else net(full_x,{k:v[uq] for k,v in g.items()}))
                    dec=decompose(full,full_y,full_v)
                    full_rows.append(dict(tag,kind=kind,seed=seed,condition='normal',**metrics(full,full_y,full_v,torch.zeros_like(full_v)),
                                          **{k:v for k,v in dec.items() if k!='log_mse'}))
                    full_predictions[f'{kind}_{seed}']=full.cpu()
                    if kind=='global':
                        joint={k:(v[:,perm] if k in ('key','value','log_depth','log_weight') else v) for k,v in inputs.items()}
                        pp=net(x,joint);diff=float((pp-p).abs().max());assert diff<2e-5
                        invariants.append(dict(tag,seed=seed,joint_permutation_max_difference=diff))
                rows.append(result);predictions[f'{kind}_{seed}_{condition}']=p.cpu()
        maps=dest/(rid+'_maps.pt')
        torch.save({'episode':meta['row'],'split':meta['eval_split'],'query':q,'full_query':uq,'target':y.cpu(),'valid':valid.cpu(),'mask':mask.cpu(),
                    'full_target':full_y.cpu(),'full_valid':full_v.cpu(),'predictions':predictions,'full_predictions':full_predictions,
                    'rgb':raw['rgb'],'permutation':perm.cpu()},maps)
        identities[str(maps)]=sha256(maps)
        print(json.dumps({'evaluated':rid,'split':meta['eval_split'],'predictions':len(predictions)}),flush=True)
    write_json(dest/'scores.json',{'rows':rows,'full_query':full_rows,'invariants':invariants,'maps_sha256':identities,
               'old_clips':46,'method_unseen_places':20,'historically_untouched_claim':False})


if __name__=='__main__':evaluate()
