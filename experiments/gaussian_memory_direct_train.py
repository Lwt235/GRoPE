"""Six fixed-budget fits with matched sampling and frozen reference states."""
import json
import time

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_setup import sha256
from experiments.gaussian_memory_direct_data import OUT, SEEDS, NEW_VARIANTS, model_path, checked_cache, load_record
from models.gaussian_memory_projection import projected_inputs
from models.gaussian_memory_direct_reference import DirectReferenceAdapter
from models.gaussian_memory_view_token import normalized_error


def load_training(plan):
    data=[]
    for row in plan['catalog']:
        if row['cohort']!='development' or row['row']['split']!='fit': continue
        d=load_record(row,'cuda'); rec,raw=d['rec'],d['raw']
        keep=torch.tensor([r=='train' for r in rec['roles']]); query=rec['query'][keep]
        g=projected_inputs(d['direct_memory'],raw['c2w'].cuda(),raw['K'].cuda())
        # Keep 13 camera states once; do not replicate 2048-D keys for all masks.
        g={k:g[k] for k in ('query','key','value','distance2','log_weight')}
        data.append({'id':row['id'],'x':rec['input_tokens'][keep].cuda(),
                     'y':rec['teacher_tokens'][query].cuda(),'mask':rec['mask'][keep].cuda(),
                     'query':query,'geometry':g})
    assert len(data)==12
    return data


def train():
    plan=checked_cache(); torch.set_num_threads(4); data=load_training(plan)
    for seed in SEEDS:
        for variant in NEW_VARIANTS:
            dest=model_path(variant,seed).parent; assert not dest.exists(); dest.mkdir()
            torch.manual_seed(seed)
            model=DirectReferenceAdapter(variant,plan['center_variance']).cuda()
            if variant=='direct':
                initial=torch.load(model_path('center',seed,True),weights_only=False,map_location='cuda')['model']
                model.copy_common_initialization(initial)
            opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
            config={'variant':variant,'seed':seed,'steps':2000,'batch':2,'loss':'uniform_all_patches',
                    'lr':3e-4,'weight_decay':.01,'clip':5.,'center_variance':plan['center_variance'],
                    'width':model.width,'parameters':sum(p.numel() for p in model.parameters()),
                    'plan_sha256':sha256(OUT / 'plan.json'),
                    'reference_cache_sha256':sha256(OUT / 'reference_cache_identity.json'),
                    'initialization':'copy matching B Center initial modules and geometry columns' if variant=='direct' else 'fixed-seed wider MLP; zero output',
                    'used_reference_features':variant=='direct'}
            assert config['parameters']==(531652 if variant=='direct' else 530548)
            write_json(dest / 'config.json',config)
            torch.save({'model':model.state_dict(),'config':config},dest / 'initial.pt')
            rng=np.random.default_rng(seed); started=time.monotonic(); torch.cuda.reset_peak_memory_stats()
            with (dest / 'training.jsonl').open('w',buffering=1) as log:
                for step in range(1,2001):
                    d=data[int(rng.integers(len(data)))]; ids=rng.choice(len(d['x']),2,replace=False).tolist()
                    q=d['query'][ids].tolist()
                    g={k:v[q] for k,v in d['geometry'].items() if variant=='direct' or k=='query'}
                    prediction=model(d['x'][ids],g); error=normalized_error(prediction,d['y'][ids])
                    loss=error.mean(); mask=d['mask'][ids]
                    opt.zero_grad(set_to_none=True); loss.backward()
                    grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                    assert torch.isfinite(loss) and torch.isfinite(grad),(variant,seed,step)
                    opt.step()
                    row={'step':step,'episode':d['id'],'sample_ids':ids,'loss':loss.item(),
                         'hidden':error[mask].mean().item(),'visible':error[~mask].mean().item(),
                         'gradient_norm':grad.item(),'seconds':time.monotonic()-started}
                    log.write(json.dumps(row)+'\n')
                    if step==1 or step%500==0: print(json.dumps({'run':dest.name,**row}),flush=True)
                    if step%500==0: torch.save({'model':model.state_dict(),'config':config,'step':step},dest / f'step_{step:04d}.pt')
            torch.cuda.synchronize()
            write_json(dest / 'status.json',{'complete':True,'steps':2000,'seconds':time.monotonic()-started,
                       'peak_GiB':torch.cuda.max_memory_allocated()/1024**3})
            print(json.dumps({'finished':dest.name,'seconds':time.monotonic()-started}),flush=True)
            del model,opt
    write_json(OUT / 'training_status.json',{'complete':True,'new_runs':6,'reused_B_runs':6})


if __name__=='__main__': train()
