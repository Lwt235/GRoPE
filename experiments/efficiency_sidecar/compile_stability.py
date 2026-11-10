"""Private 64-step old-pool stability check for the fixed compiled candidate."""
from argparse import Namespace
import time
import numpy as np
import torch
from experiments.efficiency_sidecar import compile_gpu as runner
from experiments.efficiency_sidecar.check_cpu import old_inputs
from experiments.efficiency_sidecar.followup_gpu import base_model,make_schedule
from experiments.round65_common import read,write,sha


def run():
    args=Namespace(stage='stability',batch=4,merge=False,mode='shared')
    runner.USE_MERGE=False
    r,out=runner.initialize(args)
    try:
        assert read(out/'correctness.json')['state']=='complete'
        r['correctness_sha256']=sha(out/'correctness.json')
        pool,target,r['input_identity']=old_inputs(16,2048)
        schedule,r['schedule_generation_ms_per_step']=make_schedule(steps=64)
        r['scope']='Old B4/N512 private 64-step paired diagnostic; not final accuracy confirmation.'
        for role in ('relative5','distance_mlp'):
            base,ck=base_model(role)
            models=[runner.make(base,f) for f in (False,True)]
            opts=[runner.optimizer(m) for m in models]
            for i,spec in enumerate(schedule):
                if i%8==0:runner.resource(r)
                pair={}
                for f in (i%2,1-i%2):
                    loss,norm=runner.step(models[f],opts[f],pool,target,spec,871000+i)
                    pair['candidate' if f else 'baseline']=dict(loss=float(loss),grad_norm=float(norm))
                assert all(np.isfinite(v['loss']) and np.isfinite(v['grad_norm']) for v in pair.values())
                r['measurements'].append(dict(role=role,step=i,**pair))
                if i%16==15:
                    write(out/'stability_progress.json',r);print('STABILITY',role,i+1,flush=True)
                time.sleep(.05)
            delta={n:float((p-dict(models[1].named_parameters())[n]).abs().max()) for n,p in models[0].named_parameters()}
            for m in models:m.eval()
            with torch.inference_mode():
                values=tuple(t[:4,:512].cuda() for t in pool)
                a=models[0](*values);b=models[1](*values)
                final_diff=float((a-b).abs().max())
            r.setdefault('final_models',[]).append(dict(role=role,max_parameter_abs=max(delta.values()),old_probe_logit_max_abs=final_diff,checkpoint=ck))
            del models,opts,loss,norm,values,a,b;runner.cleanup()
        r.update(state='complete',elapsed_seconds=time.time()-r['started_unix'])
        write(out/'stability.json',r);write(runner.OUT/'stability.json',r)
    except BaseException as e:
        r.update(state='failed',error_type=type(e).__name__,error=str(e));write(out/'stability_failure.json',r);raise

if __name__=='__main__':run()
