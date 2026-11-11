"""All four fixed models on old development and two already-used clip cohorts."""
import json

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_setup import sha256
from experiments.gaussian_memory_direct_data import OUT, SEEDS, VARIANTS, model_path, checked_cache, load_record
from models.gaussian_memory_projection import projected_inputs
from models.gaussian_memory_direct_reference import DirectReferenceAdapter
from models.gaussian_memory_view_token import ViewTokenAdapter, normalized_error


def models(device='cuda'):
    result={}
    for variant in VARIANTS:
        for seed in SEEDS:
            ckpt=torch.load(model_path(variant,seed),weights_only=False,map_location=device)
            cls=ViewTokenAdapter if variant in ('local','center') else DirectReferenceAdapter
            model=cls(variant,ckpt['config']['center_variance']).to(device).eval()
            model.load_state_dict(ckpt['model']); result[(variant,seed)]=model
    return result


def error_row(error,mask):
    return {'hidden_mse':error[mask].mean().item(),'visible_mse':error[~mask].mean().item(),
            'all_patch_mse':error.mean().item()}


@torch.inference_mode()
def evaluate():
    plan=checked_cache(); torch.set_num_threads(4); nets=models()
    dest=OUT / 'evaluation'; assert not dest.exists(); dest.mkdir()
    paths=[model_path(v,s) for v in VARIANTS for s in SEEDS]
    lock={'models':{str(p):sha256(p) for p in paths},'model_selection':False,
          'all_data_previously_used':True,'plan_sha256':sha256(OUT / 'plan.json')}
    write_json(OUT / 'evaluation_lock.json',lock)
    rows=[]; preservation=[]
    for cohort in ('development','used_clip40','used_clip80'):
        records=[r for r in plan['catalog'] if r['cohort']==cohort]
        for idx,row in enumerate(records):
            d=load_record(row,'cuda'); raw,rec=d['raw'],d['rec']
            keep=torch.tensor([s=='evaluation' for s in rec['roles']]); query=rec['query'][keep]
            x,y=rec['input_tokens'][keep].cuda(),rec['teacher_tokens'][query].cuda()
            mask=rec['mask'][keep].cuda(); c2w,K=raw['c2w'][query].cuda(),raw['K'][query].cuda()
            split=cohort if cohort!='development' else ('heldout_view' if row['row']['split']=='fit' else 'development_scene')
            base={'episode':row['id'],'scene':row['row']['scene'],'cohort':cohort,'split':split}
            errors={'backbone':normalized_error(x,y).cpu()}
            rows.append(dict(base,variant='backbone',seed=None,condition='normal',**error_row(errors['backbone'].cuda(),mask)))
            wrong_row=next(r for r in records[idx+1:]+records[:idx+1] if r['row']['scene']!=row['row']['scene'])
            wrong=load_record(wrong_row,'cuda')
            original_g=projected_inputs(d['memory'],c2w,K)
            for variant in ('local','wide'):
                for seed in SEEDS:
                    net=nets[(variant,seed)]; error=normalized_error(net(x,original_g),y)
                    rows.append(dict(base,variant=variant,seed=seed,condition='normal',**error_row(error,mask)))
                    preservation.append(dict(base,variant=variant,seed=seed,clean_input_mse=normalized_error(net(y,original_g),y).mean().item()))
                    if seed==SEEDS[0]: errors[variant]=error.cpu()
            for variant,memory_key in [('center','memory'),('direct','direct_memory')]:
                memory=d[memory_key]
                permutation=torch.randperm(len(memory['means']),generator=torch.Generator().manual_seed(371+idx)).cuda()
                shuffled={k:v for k,v in memory.items()}; shuffled['features']=memory['features'][permutation]
                for condition in ('normal','removed','shuffled','swapped','wrong_camera'):
                    m=shuffled if condition=='shuffled' else wrong[memory_key] if condition=='swapped' else memory
                    camera=raw['c2w'][0].cuda().expand_as(c2w) if condition=='wrong_camera' else c2w
                    g=projected_inputs(m,camera,K)
                    for seed in SEEDS:
                        net=nets[(variant,seed)]; error=normalized_error(net(x,g,condition),y)
                        rows.append(dict(base,variant=variant,seed=seed,condition=condition,**error_row(error,mask)))
                        if condition=='normal':
                            preservation.append(dict(base,variant=variant,seed=seed,clean_input_mse=normalized_error(net(y,g),y).mean().item()))
                            if seed==SEEDS[0]: errors[variant]=error.cpu()
                    del g
            torch.save({'episode':row['row'],'query':query,'mask':mask.cpu(),'errors':errors,'rgb':raw['rgb']},dest / (row['id']+'_maps.pt'))
            print(json.dumps({'evaluated':row['id'],'split':split}),flush=True)
            del d,wrong,original_g
    write_json(dest / 'scores.json',{'rows':rows,'preservation':preservation,'model_sha256':lock['models'],
               'confirmation_claim':False,'episodes':42})
    print(json.dumps({'complete':True,'episodes':42,'score_rows':len(rows)}),flush=True)


if __name__=='__main__': evaluate()
