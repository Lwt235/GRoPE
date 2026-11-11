"""Same architecture, initial state and samples; reference geometry changes."""
import json
import time

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_setup import sha256
from experiments.gaussian_memory_oracle_data import OUT, OLD, SEEDS, VARIANTS, checked, load_record, input_seeds, model_path
from experiments.gaussian_memory_surface_train import loss_function
from models.gaussian_memory_surface import SurfaceMemory


def train():
    plan = checked(True); torch.set_num_threads(4); data = []
    for rid in plan['training_ids']:
        meta = next(m for m in plan['catalog'] if m['id']==rid); d = load_record(meta)
        raw, rec, target = d['raw'], d['rec'], d['target']; keep = torch.tensor([r=='train' for r in rec['roles']]); q=rec['query'][keep]
        data.append({'id':rid, 'x':rec['input_tokens'][keep].cuda(), 'query':q.cuda(),
                     'y':target['log_depth_m'][q].cuda(), 'valid':target['valid'][q].cuda(),
                     'c2w':raw['c2w'].cuda(), 'K':raw['K'].cuda(), 'unit':raw['diagnostics']['unit_meters'],
                     'seeds':input_seeds(d,'sensor')})
    for seed in SEEDS:
        for variant in VARIANTS:
            directory = model_path(variant,seed).parent; assert not directory.exists(); directory.mkdir()
            initial_path = model_path(variant,seed,'predicted',True)
            initial = torch.load(initial_path,map_location='cuda',weights_only=False)
            torch.manual_seed(seed); net=SurfaceMemory(variant,plan['local_width'],plan['center_variance']).cuda()
            net.load_state_dict(initial['model']); opt=torch.optim.AdamW(net.parameters(),lr=plan['lr'],weight_decay=plan['weight_decay'])
            config={**initial['config'],'regime':'sensor','plan_sha256':sha256(OUT/'plan.json'),'initial_source_sha256':sha256(initial_path)}
            torch.save({'model':net.state_dict(),'config':config},model_path(variant,seed,initial=True)); write_json(directory/'config.json',config)
            rng=np.random.default_rng(seed+73); losses=[]; grads=[]; torch.cuda.synchronize(); start=time.monotonic();torch.cuda.reset_peak_memory_stats()
            with (directory/'training.jsonl').open('w') as log:
                for step in range(1,2001):
                    slot=int(rng.integers(len(data))); d=data[slot]; ids=rng.choice(len(d['x']),2,replace=False).tolist();q=d['query'][ids]
                    opt.zero_grad(set_to_none=True);pred=net(d['x'][ids],d['seeds'],d['c2w'][q],d['K'][q],d['unit'])
                    loss=loss_function(pred,d['y'][ids],d['valid'][ids]);assert torch.isfinite(loss);loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_(net.parameters(),5.);assert torch.isfinite(norm)
                    if step in (2,100,2000):
                        g=net.writer.net[-1].weight.grad;grads.append({'step':step,'center':float(g[:3].norm()),'shape':float(g[3:9].norm()),'opacity':float(g[9].norm()),'content':float(g[10:].norm())})
                    opt.step();losses.append(float(loss));log.write(json.dumps({'step':step,'slot':slot,'episode':d['id'],'sample_ids':ids,'loss':float(loss),'gradient_norm':float(norm)})+'\n')
                    if step%500==0:log.flush();print(json.dumps({'training':variant,'seed':seed,'step':step,'loss_last100':float(np.mean(losses[-100:]))}),flush=True)
            torch.cuda.synchronize();seconds=time.monotonic()-start
            torch.save({'model':net.state_dict(),'config':config},model_path(variant,seed))
            status={'complete':True,'steps':2000,'seconds':seconds,'peak_GiB':torch.cuda.max_memory_allocated()/1024**3,
                    'first100':float(np.mean(losses[:100])),'last100':float(np.mean(losses[-100:])),'writer_gradients':grads,
                    'model_sha256':sha256(model_path(variant,seed))};write_json(directory/'status.json',status)
            print(json.dumps({'fit_complete':variant,'seed':seed,**status}),flush=True);del net,opt;torch.cuda.empty_cache()
    paths=[model_path(v,s,r) for r in ('predicted','sensor') for v in VARIANTS for s in SEEDS]+[model_path('local',s) for s in SEEDS]
    write_json(OUT/'evaluation_lock.json',{'models':{str(p):sha256(p) for p in paths},'plan_sha256':sha256(OUT/'plan.json')})


if __name__=='__main__':train()
