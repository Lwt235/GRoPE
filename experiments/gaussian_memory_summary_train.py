"""Three matched-budget global controls; lock every model before new extraction."""
import json
import time

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_surface_train import loss_function
from experiments.gaussian_memory_setup import sha256
from experiments.gaussian_memory_summary_data import OUT,OLD,SEEDS,KINDS,checked,load_record,model,path
from models.gaussian_memory_transport import read_inputs


def preflight(plan,data):
    d=data[0];x=d['x'][:2];q=d['q'][:2];g={k:v[q] for k,v in d['projections'].items()}
    net=model('global',plan).cuda()
    # A nonzero, already trained old-data head makes invariance meaningful.
    net.load_state_dict(torch.load(path('gaussian',SEEDS[0]),map_location='cuda',weights_only=False)['model'])
    permutation=torch.arange(255,-1,-1,device='cuda')
    permuted={k:(v[:,permutation] if k in ('key','value','log_depth','log_weight') else v) for k,v in g.items()}
    with torch.no_grad():
        a,parts=net(x,g,return_parts=True);b=net(x,permuted);removed=net(x,g,'removed')
    difference=float((a-b).abs().max());summary_variation=float((parts['summary']-parts['summary'][:,:1]).abs().max())
    assert difference<2e-5 and summary_variation==0 and torch.isfinite(removed).all()
    initial_checks=[]
    for seed in SEEDS:
        new=torch.load(path('global',seed,True),map_location='cpu',weights_only=False)['model']
        old=torch.load(path('gaussian',seed,True),map_location='cpu',weights_only=False)['model']
        assert new.keys()==old.keys() and all(torch.equal(new[k],old[k]) for k in old)
        initial_checks.append({'seed':seed,'exact_shared_parameters':True})
    net.zero_grad(set_to_none=True);loss=loss_function(net(x,g),d['y'][:2],d['valid'][:2]);loss.backward()
    gradients={n:float(p.grad.norm()) for n,p in net.named_parameters() if p.grad is not None}
    assert len(gradients)==len(list(net.parameters())) and all(np.isfinite(list(gradients.values())))
    write_json(OUT/'preflight.json',{'complete':True,'parameters':sum(p.numel() for p in net.parameters()),
        'joint_slot_permutation_max_difference':difference,'summary_patch_variation':summary_variation,
        'empty_memory_finite':True,'initialization_checks':initial_checks,'old_data_gradients':gradients})


def train():
    plan=checked();torch.set_num_threads(4);data=[]
    for rid in plan['training_ids']:
        meta=next(m for m in plan['catalog'] if m['id']==rid);d=load_record(meta)
        raw,rec,target=d['raw'],d['rec'],d['target'];keep=torch.tensor([r=='train' for r in rec['roles']]);q=rec['query'][keep]
        camera,K,unit=raw['c2w'].cuda(),raw['K'].cuda(),raw['diagnostics']['unit_meters']
        memory={k:v.cuda() for k,v in d['memory'].items()}
        with torch.no_grad():projections=read_inputs(memory,camera,K,unit)
        data.append({'id':rid,'x':rec['input_tokens'][keep].cuda(),'q':q.cuda(),
                     'y':target['log_depth_m'][q].cuda(),'valid':target['valid'][q].cuda(),'projections':projections})
    preflight(plan,data)
    for seed in SEEDS:
        dest=path('global',seed);assert not dest.exists()
        initial=torch.load(path('global',seed,True),map_location='cuda',weights_only=False)
        torch.manual_seed(seed);net=model('global',plan).cuda();net.load_state_dict(initial['model'])
        opt=torch.optim.AdamW(net.parameters(),lr=plan['lr'],weight_decay=plan['weight_decay'])
        rng=np.random.default_rng(seed+73);losses=[];gradients=[]
        torch.cuda.synchronize();start=time.monotonic();torch.cuda.reset_peak_memory_stats()
        with (dest.parent/'training.jsonl').open('w') as log:
            for step in range(1,2001):
                slot=int(rng.integers(len(data)));d=data[slot];ids=rng.choice(len(d['x']),2,replace=False).tolist();q=d['q'][ids]
                opt.zero_grad(set_to_none=True);pred=net(d['x'][ids],{k:v[q] for k,v in d['projections'].items()})
                loss=loss_function(pred,d['y'][ids],d['valid'][ids]);assert torch.isfinite(loss);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(net.parameters(),plan['gradient_clip']);assert torch.isfinite(norm)
                if step in (2,100,2000):gradients.append({'step':step,**{n:float(p.grad.norm()) for n,p in net.named_parameters()}})
                opt.step();losses.append(float(loss))
                log.write(json.dumps({'step':step,'slot':slot,'episode':d['id'],'sample_ids':ids,'loss':float(loss),'gradient_norm':float(norm)})+'\n')
                if step%500==0:
                    log.flush();print(json.dumps({'training':seed,'step':step,'loss_last100':float(np.mean(losses[-100:]))}),flush=True)
        torch.cuda.synchronize()
        torch.save({'model':net.state_dict(),'config':initial['config'],'plan_sha256':sha256(OUT/'plan.json')},dest)
        write_json(dest.parent/'status.json',{'complete':True,'steps':2000,'seed':seed,'seconds':time.monotonic()-start,
                   'peak_GiB':torch.cuda.max_memory_allocated()/1024**3,'first100':float(np.mean(losses[:100])),
                   'last100':float(np.mean(losses[-100:])),'gradients':gradients,'model_sha256':sha256(dest)})
        print(json.dumps({'fit_complete':seed}),flush=True)
        del net,opt;torch.cuda.empty_cache()
    write_json(OUT/'evaluation_lock.json',{'models':{str(path(k,s)):sha256(path(k,s)) for k in KINDS for s in SEEDS},
               'plan_sha256':sha256(OUT/'plan.json'),'new_data_extracted':False,'model_selection':False})


if __name__=='__main__':train()
