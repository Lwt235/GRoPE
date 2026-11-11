"""Fixed-budget paired readers over precomputed projections of frozen states."""
import json
import time

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_setup import sha256
from experiments.gaussian_memory_surface_train import loss_function
from experiments.gaussian_memory_transport_data import OUT, REGIMES, checked, load_record, load_memory, model_path, make_model, name
from models.gaussian_memory_surface import camera_query
from models.gaussian_memory_transport import read_inputs


def train():
    plan=checked(); torch.set_num_threads(4); data=[]
    for rid in plan['training_ids']:
        meta=next(m for m in plan['catalog'] if m['id']==rid); d=load_record(meta)
        raw,rec,target=d['raw'],d['rec'],d['target']; keep=torch.tensor([r=='train' for r in rec['roles']]); q=rec['query'][keep]
        camera,K,unit=raw['c2w'].cuda(),raw['K'].cuda(),raw['diagnostics']['unit_meters']
        with torch.inference_mode():
            projections={g:read_inputs(load_memory(rid,g,'cuda'),camera,K,unit) for g in REGIMES}
            projections['none']={'camera_query':camera_query(camera,K,unit)}
        # Ordinary detached tensors may be saved by Linear during backward.
        projections={g:{k:v.clone() for k,v in values.items()} for g,values in projections.items()}
        data.append({'id':rid,'x':rec['input_tokens'][keep].cuda(),'q':q.cuda(),
                     'y':target['log_depth_m'][q].cuda(),'valid':target['valid'][q].cuda(),'projections':projections})
    for c in plan['cases']:
        path=model_path(c); assert not path.exists(); initial=torch.load(model_path(c,True),map_location='cuda',weights_only=False)
        torch.manual_seed(c['seed']); net=make_model(c,plan).cuda();net.load_state_dict(initial['model'])
        opt=torch.optim.AdamW(net.parameters(),lr=plan['lr'],weight_decay=plan['weight_decay'])
        rng=np.random.default_rng(c['seed']+73);losses=[];gradient_checks=[]
        torch.cuda.synchronize();start=time.monotonic();torch.cuda.reset_peak_memory_stats()
        with (path.parent/'training.jsonl').open('w') as log:
            for step in range(1,2001):
                slot=int(rng.integers(len(data)));d=data[slot];ids=rng.choice(len(d['x']),2,replace=False).tolist();q=d['q'][ids]
                inputs={k:v[q] for k,v in d['projections'][c['regime']].items()}
                opt.zero_grad(set_to_none=True);prediction=net(d['x'][ids],inputs)
                loss=loss_function(prediction,d['y'][ids],d['valid'][ids]);assert torch.isfinite(loss);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(net.parameters(),plan['gradient_clip']);assert torch.isfinite(norm)
                if step in (2,100,2000):
                    gradient_checks.append({'step':step,'output':float(net.output.weight.grad.norm()),
                                            'key':float(net.key.weight.grad.norm()) if hasattr(net,'key') else None,
                                            'query':float(net.query.weight.grad.norm())})
                opt.step();losses.append(float(loss));log.write(json.dumps({'step':step,'slot':slot,'episode':d['id'],'sample_ids':ids,'loss':float(loss),'gradient_norm':float(norm)})+'\n')
                if step%500==0:log.flush();print(json.dumps({'training':name(c),'step':step,'loss_last100':float(np.mean(losses[-100:]))}),flush=True)
        torch.cuda.synchronize();seconds=time.monotonic()-start
        torch.save({'model':net.state_dict(),'config':initial['config'],'plan_sha256':sha256(OUT/'plan.json')},path)
        status={'complete':True,**c,'steps':2000,'seconds':seconds,'peak_GiB':torch.cuda.max_memory_allocated()/1024**3,
                'first100':float(np.mean(losses[:100])),'last100':float(np.mean(losses[-100:])),
                'gradients':gradient_checks,'model_sha256':sha256(path)}
        write_json(path.parent/'status.json',status); print(json.dumps({'fit_complete':name(c),**status}),flush=True)
        del net,opt;torch.cuda.empty_cache()
    write_json(OUT/'evaluation_lock.json',{'models':{str(model_path(c)):sha256(model_path(c)) for c in plan['cases']},
                                         'plan_sha256':sha256(OUT/'plan.json')})


if __name__=='__main__':train()
