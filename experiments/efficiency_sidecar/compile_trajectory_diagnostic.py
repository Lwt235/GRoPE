"""One fixed-weight diagnosis at the first large gradient-norm separation.

Replay old 35 steps, then compare eager/compiled at each identical parameter
state. Inspect final max-pooling winners; no model/optimizer/formula changes.
"""
from argparse import Namespace
import time
import numpy as np
import torch
import torch.nn.functional as F
from experiments.efficiency_sidecar import compile_gpu as runner
from experiments.efficiency_sidecar.check_cpu import old_inputs,state_hash
from experiments.efficiency_sidecar.followup_gpu import base_model,make_schedule,gather_cpu
from experiments.round65_run import augment
from experiments.round65_common import read,write,sha


def probe(model,inputs,y,seed):
    captured=[]
    handle=model.final_norm.register_forward_hook(lambda m,args,out:captured.append(out.detach().cpu()))
    model.zero_grad(set_to_none=True);model.train();torch.manual_seed(seed)
    logits=model(*inputs);loss=F.cross_entropy(logits,y,label_smoothing=.1);loss.backward()
    handle.remove()
    gradient=torch.cat([p.grad.detach().cpu().flatten() for p in model.parameters()])
    features=captured[0]
    return dict(logits=logits.detach().cpu(),loss=float(loss),gradient=gradient,features=features)


def compare(a,b):
    va,ia=a['features'].max(1);vb,ib=b['features'].max(1)
    top=a['features'].topk(2,dim=1).values
    gap=top[:,0]-top[:,1];changed=ia!=ib
    return dict(loss_difference=b['loss']-a['loss'],logit_max_abs=float((a['logits']-b['logits']).abs().max()),
        gradient_relative_l2=float((a['gradient']-b['gradient']).double().norm()/a['gradient'].double().norm().clamp_min(1e-12)),
        baseline_grad_norm=float(a['gradient'].norm()),candidate_grad_norm=float(b['gradient'].norm()),
        pooled_feature_max_abs=float((va-vb).abs().max()),all_feature_max_abs=float((a['features']-b['features']).abs().max()),
        winner_changes=int(changed.sum()),winner_total=changed.numel(),
        changed_winner_eager_margin=gap[changed].tolist(),eager_near_ties_under_1e_5=int((gap<1e-5).sum()))


def run():
    args=Namespace(stage='trajectory_diagnostic',batch=4,merge=False,mode='shared')
    r,out=runner.initialize(args)
    try:
        r['scope']='Post-result old-data diagnosis, step index35 selected from prior stability record; not fresh confirmation.'
        r['stability_sha256']=sha(runner.OUT/'stability.json')
        pool,target,r['input_identity']=old_inputs(16,2048)
        schedule,_=make_schedule(steps=36)
        base,ck=base_model('relative5');models=[runner.make(base,f) for f in (False,True)];opts=[runner.optimizer(m) for m in models]
        for i in range(35):
            if i%8==0:runner.resource(r)
            values={}
            for f in (i%2,1-i%2):
                loss,norm=runner.step(models[f],opts[f],pool,target,schedule[i],871000+i)
                values['candidate' if f else 'baseline']=dict(loss=float(loss),grad_norm=float(norm))
            r['measurements'].append(dict(step=i,**values))
        spec=schedule[35];values,y,_=gather_cpu(pool,target,spec)
        c,s,a,y=tuple(torch.tensor(v,device='cuda') for v in (*values,y));c,s=augment(c,s,spec[2],spec[3]);inputs=(c,s,a)
        actual=[probe(m,inputs,y,871035) for m in models]
        r['different_trajectory_states']=compare(*actual)
        r['fixed_state_comparisons']=[]
        for origin in (0,1):
            state={n:t.detach().cpu().clone() for n,t in models[origin].state_dict().items()}
            replicas=[runner.make(base,f) for f in (False,True)]
            for model in replicas:model.load_state_dict(state)
            assert state_hash(replicas[0])==state_hash(replicas[1])
            probes=[probe(m,inputs,y,871035) for m in replicas]
            r['fixed_state_comparisons'].append(dict(origin='compiled_trajectory' if origin else 'eager_trajectory',**compare(*probes)))
            del replicas,probes;runner.cleanup()
        print('DIAGNOSIS',r['different_trajectory_states'],r['fixed_state_comparisons'],flush=True)
        r.update(state='complete',elapsed_seconds=time.time()-r['started_unix'])
        write(out/'trajectory_diagnostic.json',r);write(runner.OUT/'trajectory_diagnostic.json',r)
    except BaseException as e:
        r.update(state='failed',error_type=type(e).__name__,error=str(e));write(out/'trajectory_diagnostic_failure.json',r);raise

if __name__=='__main__':run()
