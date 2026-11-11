"""Matched and crossed geometry reads from explicitly saved scene states."""
import hashlib
import json

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_setup import sha256
from experiments.gaussian_memory_oracle_data import OUT, OLD, VARIANTS, SEEDS, model_path, checked, load_record, input_seeds
from experiments.gaussian_memory_surface_evaluate import metrics, correspondence, consistency, baseline
from models.gaussian_memory_surface import SurfaceMemory
from models.gaussian_memory import detached_memory


def case_name(train_geometry, read_geometry, variant, seed):
    return f'{train_geometry}_{read_geometry}_{variant}_{seed}'


@torch.inference_mode()
def evaluate():
    plan=checked();torch.set_num_threads(4);nets={}
    for regime in ('predicted','sensor'):
        for variant in VARIANTS:
            for seed in SEEDS:
                net=SurfaceMemory(variant,plan['local_width'],plan['center_variance']).cuda().eval()
                net.load_state_dict(torch.load(model_path(variant,seed,regime),map_location='cuda',weights_only=False)['model'])
                nets[(regime,variant,seed)]=net
    for seed in SEEDS:
        net=SurfaceMemory('local',plan['local_width'],plan['center_variance']).cuda().eval()
        net.load_state_dict(torch.load(model_path('local',seed),map_location='cuda',weights_only=False)['model']);nets[('none','local',seed)]=net
    dest=OUT/'evaluation';memdir=OUT/'memories';assert not dest.exists();dest.mkdir();memdir.mkdir()
    rows=[];full_rows=[];consistency_rows=[];identities={};states=[]
    for meta in plan['catalog']:
        d=load_record(meta);raw,rec,target=d['raw'],d['rec'],d['target'];keep=torch.tensor([r=='evaluation' for r in rec['roles']]);q=rec['query'][keep]
        x=rec['input_tokens'][keep].cuda();y=target['log_depth_m'][q].cuda();valid=target['valid'][q].cuda();hidden=rec['mask'][keep].cuda()
        camera=raw['c2w'][q].cuda();K=raw['K'][q].cuda();unit=raw['diagnostics']['unit_meters'];uq=sorted(set(q.tolist()))
        full_x=rec['teacher_tokens'][uq].cuda();full_y=target['log_depth_m'][uq].cuda();full_v=target['valid'][uq].cuda()
        full_camera,full_K=raw['c2w'][uq].cuda(),raw['K'][uq].cuda()
        gt,pairs,distances=correspondence(full_y,full_v,full_camera,full_K,unit)
        base={'episode':meta['id'],'scene':meta['row']['scene'],'split':meta['eval_split']}
        seed_inputs={g:input_seeds(d,g) for g in ('predicted','sensor')};predictions={};full_predictions={}
        generator=torch.Generator().manual_seed(int(hashlib.sha256(meta['id'].encode()).hexdigest()[:8],16));perm=torch.randperm(256,generator=generator).cuda()
        for geometry in ('predicted','sensor'):
            for variant in ('seed_center','seed_gaussian'):
                pred=baseline(seed_inputs[geometry],camera,K,unit,variant,plan['center_variance'])
                rows.append(dict(base,train_geometry='none',read_geometry=geometry,variant=variant,seed=None,condition='normal',**metrics(pred,y,valid,hidden)))
                predictions[f'none_{geometry}_{variant}']=pred.cpu()
        for (regime,variant,seed),net in nets.items():
            for geometry in (('none',) if variant=='local' else ('predicted','sensor')):
                name=case_name(regime,geometry,variant,seed);memory=None
                if variant!='local':
                    live=net.write(seed_inputs[geometry]);path=memdir/(meta['id']+'_'+name+'.pt')
                    torch.save({'memory':detached_memory(live),'train_geometry':regime,'read_geometry':geometry,
                                'model_sha256':sha256(model_path(variant,seed,regime)), 'reference_frames':[0,4,8,12],
                                'query_depth_used':False,'query_RGB_used':False,'reference_sensor_used':geometry=='sensor'},path)
                    identities[str(path)]=sha256(path);memory=torch.load(path,map_location='cuda',weights_only=False)['memory']
                    states.append(dict(base,train_geometry=regime,read_geometry=geometry,variant=variant,seed=seed,
                                       replay_max_diff=max(float((memory[k]-live[k]).abs().max()) for k in memory)))
                conditions=('normal','removed','shuffled','wrong_camera','initial_geometry') if regime==geometry=='sensor' else ('normal',)
                for condition in conditions:
                    current=memory;wrong_camera=None
                    if condition=='shuffled':current=dict(memory,features=memory['features'][perm])
                    elif condition=='wrong_camera':wrong_camera=raw['c2w'][(q+4)%13].cuda()
                    elif condition=='initial_geometry':
                        s=seed_inputs[geometry];current=dict(memory,means=s['seed_mean'],covars=s['seed_chol']@s['seed_chol'].transpose(-1,-2))
                    pred=net.read(x,current,camera,K,unit,condition,memory_c2w=wrong_camera)
                    tag=dict(base,train_geometry=regime,read_geometry=geometry,variant=variant,seed=seed,condition=condition)
                    rows.append(dict(tag,**metrics(pred,y,valid,hidden)));predictions[name+'_'+condition]=pred.cpu()
                    if condition=='normal':
                        full=net.read(full_x,memory,full_camera,full_K,unit);full_predictions[name]=full.cpu()
                        full_rows.append(dict(tag,**metrics(full,full_y,full_v,torch.zeros_like(full_v))))
                        consistency_rows.append(dict(tag,gt_pair_distance_mean_m=float(np.mean(distances)),
                                                     **consistency(full,gt,pairs,full_camera,full_K,unit)))
        torch.save({'episode':meta['row'],'split':meta['eval_split'],'query':q,'full_query':uq,'target':y.cpu(),
                    'valid':valid.cpu(),'mask':hidden.cpu(),'predictions':predictions,'full_predictions':full_predictions,
                    'rgb':raw['rgb'],'correspondences':pairs,'permutation':perm.cpu()},dest/(meta['id']+'_maps.pt'))
        print(json.dumps({'evaluated':meta['id'],'cases':len(predictions)}),flush=True)
    write_json(dest/'scores.json',{'rows':rows,'full_query':full_rows,'consistency':consistency_rows,'episodes':46,'confirmation_claim':False})
    write_json(OUT/'memory_identity.json',{'sha256':identities,'state_checks':states})


if __name__=='__main__':evaluate()
