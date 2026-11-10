"""Matched hot-pool training-step timing, including cache gather/transfer/rotation."""
import gc
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.efficiency_sidecar import cache_geometry as cached
from experiments.efficiency_sidecar.followup_gpu import initialize, check_resources, finish, OUT, base_model, make_schedule, gather_cpu
from experiments.efficiency_sidecar.check_cpu import old_inputs, variant, tensor_hash
from experiments.efficiency_sidecar.adapters import quadratic_descriptors
from experiments.round65_run import augment
from experiments.round65_common import ROOT, read, sha, write
from models.round65_actual_gs import shape_features, vectorize


def run():
    cpu = read(OUT/'cache_cpu.json')
    assert cpu['state']=='complete'
    for p,d in cpu['source_sha256'].items(): assert sha(p)==d,p
    profile = read(OUT/'profile.json')
    assert profile['state']=='complete'
    r = initialize('cache_gpu')
    try:
        pool,target,r['input_identity']=old_inputs(16,2048)
        schedule,r['schedule_generation_ms_per_step']=make_schedule()
        start=time.perf_counter(); cache=cached.prepare(pool[1],precision=False)
        log_ms=(time.perf_counter()-start)*1000
        start=time.perf_counter(); cache['precision']=torch.linalg.inv(pool[1])
        inv_ms=(time.perf_counter()-start)*1000
        r['precompute_cpu_ms']=dict(log_matrix=log_ms,precision=inv_ms)
        r['cache_bytes']={k:v.numel()*v.element_size() for k,v in cache.items()}
        r['cache_sha256']={k:tensor_hash(v) for k,v in cache.items()}
        r['checks']=[]
        for role in ('relative5','distance_mlp'):
            check_resources(r)
            base,ck=base_model(role)
            names=('uncached','cache_log','cache_log_precision') if role=='relative5' else ('uncached','cache_log')
            # Full parameter gradient checks on four distinct rotations/token sets.
            for test_step in (0,8,16,24):
                spec=schedule[test_step]
                values,y,extra=gather_cpu(pool,target,spec,cache)
                c,s,a,y=[torch.tensor(v,device='cuda') for v in (*values,y)]
                gpu_cache={k:torch.tensor(v,device='cuda') for k,v in extra.items()}
                c,s=augment(c,s,spec[2],spec[3]);rc=cached.rotate(gpu_cache,spec[2])
                ce=float((shape_features(s)-vectorize(rc['log_matrix'])).abs().max())
                de=float((quadratic_descriptors(c,s)-cached.cached_descriptor(c,s,rc['precision'])).abs().max())
                ref_output=ref_grad=None
                for name in names:
                    model=variant(base,'geometry_factored').cuda().train()
                    torch.manual_seed(925000+test_step)
                    output=(model(c,s,a) if name=='uncached' else cached.forward(model,c,s,a,
                        log_matrix=rc['log_matrix'],precision=rc['precision'] if name=='cache_log_precision' else None))
                    loss=F.cross_entropy(output,y,label_smoothing=.1);loss.backward()
                    grad=torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).detach().cpu()
                    out=output.detach().cpu()
                    if name=='uncached':ref_output,ref_grad=out,grad
                    oe=float((out-ref_output).abs().max())
                    ge=float((grad.double()-ref_grad.double()).norm()/ref_grad.double().norm())
                    assert torch.isfinite(grad).all() and oe<2e-4 and ge<1e-4
                    r['checks'].append(dict(role=role,variant=name,step=test_step,logit_max_abs=oe,
                        gradient_relative_l2=ge,common_max_abs=ce,descriptor_max_abs=de,checkpoint=ck))
                    del model,output,loss,grad
                del c,s,a,y,gpu_cache,rc
                gc.collect();torch.cuda.empty_cache()
            # Keep the same optimizer history per variant, interleaving blocks.
            models={name:variant(base,'geometry_factored').cuda().train() for name in names}
            opts={name:torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01) for name,model in models.items()}
            for block in range(8):
                order=names[block%len(names):]+names[:block%len(names)]
                for name in order:
                    check_resources(r)
                    model,opt=models[name],opts[name]
                    fields=None if name=='uncached' else {'log_matrix':cache['log_matrix']}
                    if name=='cache_log_precision':fields=dict(fields,precision=cache['precision'])
                    for step in range(block*4,(block+1)*4):
                        spec=schedule[step]
                        torch.manual_seed(925100+step)
                        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                        start=time.perf_counter()
                        values,y,extra=gather_cpu(pool,target,spec,fields)
                        c,s,a,y=[torch.tensor(v,device='cuda') for v in (*values,y)]
                        gpu_cache={k:torch.tensor(v,device='cuda') for k,v in extra.items()}
                        c,s=augment(c,s,spec[2],spec[3])
                        rc=cached.rotate(gpu_cache,spec[2]) if gpu_cache else {}
                        model.zero_grad(set_to_none=True)
                        output=(model(c,s,a) if name=='uncached' else cached.forward(model,c,s,a,**rc))
                        loss=F.cross_entropy(output,y,label_smoothing=.1)
                        if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite loss.')
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                        opt.step();value=float(loss)
                        torch.cuda.synchronize()
                        r['measurements'].append(dict(role=role,variant=name,block=block,step=step,warmup=block==0,
                            wall_ms=(time.perf_counter()-start)*1000,loss=value,
                            peak_bytes=torch.cuda.max_memory_allocated(),additional_h2d_bytes=sum(v.nbytes for v in extra.values())))
                        model.zero_grad(set_to_none=True)
                        del c,s,a,y,gpu_cache,rc,output,loss
                        time.sleep(.05)
                    write(OUT/'cache_gpu_progress.json',r)
                print('CACHE_GPU',role,block,flush=True)
            del models,opts,model,opt
            gc.collect();torch.cuda.empty_cache()
        finish(r)
    except BaseException as e:
        r.update(state='failed',error_type=type(e).__name__,error=str(e))
        write(OUT/'cache_gpu_failure.json',r);raise


if __name__=='__main__':run()
