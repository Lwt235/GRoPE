"""Bounded private checks, matched full-step timing and optional short training."""
import argparse
import gc
import time
import numpy as np
import torch
import torch.nn.functional as F
from experiments.efficiency_sidecar.backward_common import initialize, check, finish, failure, OUT
from experiments.efficiency_sidecar.check_cpu import old_inputs, variant, state_hash
from experiments.efficiency_sidecar.followup_gpu import base_model, make_schedule, gather_cpu
from experiments.efficiency_sidecar.fused_hidden_grad import FanoutHidden, fuse_hidden_gradient
from experiments.round65_common import write
from experiments.round65_run import augment


def make(base, fused):
    model = variant(base, 'geometry_factored')
    return (fuse_hidden_gradient(model) if fused else model).cuda().train()


def optimizer(model):
    return torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)


def step(model,opt,pool,target,spec,seed):
    torch.manual_seed(seed)
    values,y,_=gather_cpu(pool,target,spec)
    c,s,a,y=tuple(torch.tensor(v,device='cuda') for v in (*values,y))
    c,s=augment(c,s,spec[2],spec[3])
    model.zero_grad(set_to_none=True)
    loss=F.cross_entropy(model(c,s,a),y,label_smoothing=.1)
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    opt.step()
    return loss.detach(),norm.detach()


def cleanup():
    gc.collect();torch.cuda.empty_cache();time.sleep(.1)


def correctness(r):
    # Exercise strided real-layout cotangents and partial use on CUDA.
    for used in ((0,), (1,3), (0,1,2,3)):
        x=torch.randn(2,7,9,96,device='cuda',requires_grad=True)
        coeff=[torch.randn(2,7,96,9,device='cuda').transpose(-1,-2) for _ in range(4)]
        slots=FanoutHidden.apply(x)
        actual,=torch.autograd.grad(sum((slots[s]*coeff[s]).sum() for s in used),x)
        torch.testing.assert_close(actual,sum(coeff[s] for s in used),atol=1e-6,rtol=1e-5)
    del x,coeff,slots,actual
    pool,y,r['input_identity']=old_inputs(4,512)
    inputs=tuple(t.cuda() for t in pool); y=y.cuda()
    for role in ('relative5','distance_mlp'):
        check(r)
        base,ck=base_model(role)
        states=[]
        for fused in (False,True):
            model=make(base,fused); opt=optimizer(model)
            initial=state_hash(model)
            torch.manual_seed(831)
            torch.cuda.reset_peak_memory_stats()
            out=model(*inputs)
            loss=F.cross_entropy(out,y,label_smoothing=.1)
            loss.backward()
            grad={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters()}
            assert all(bool(torch.isfinite(t).all()) for t in grad.values())
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            states.append(dict(initial=initial,output=out.detach().cpu(),gradient=grad,
                updated={n:p.detach().cpu().clone() for n,p in model.named_parameters()},
                peak_bytes=torch.cuda.max_memory_allocated()))
            del model,opt,out,loss,grad;cleanup()
        a,b=states
        assert a['initial']==b['initial']
        torch.testing.assert_close(a['output'],b['output'],atol=2e-6,rtol=2e-5)
        errors={}
        for key,atol,rtol in (('gradient',3e-6,3e-4),('updated',2e-5,2e-4)):
            maximum=0.;relative=0.
            for n in a[key]:
                torch.testing.assert_close(a[key][n],b[key][n],atol=atol,rtol=rtol,msg=lambda msg,n=n:f'{n}: {msg}')
                difference=(a[key][n]-b[key][n]).double()
                maximum=max(maximum,float(difference.abs().max()))
                relative=max(relative,float(difference.norm()/a[key][n].double().norm().clamp_min(1e-12)))
            errors[key]={'max_abs':maximum,'max_tensor_relative_l2':relative}
        r['measurements'].append(dict(role=role,checkpoint=ck,max_logit_abs=float((a['output']-b['output']).abs().max()),
            errors=errors,baseline_peak_bytes=a['peak_bytes'],candidate_peak_bytes=b['peak_bytes']))
        print('CHECK',r['measurements'][-1],flush=True)
        del states,a,b;cleanup()


def timing(r):
    pool,target,r['input_identity']=old_inputs(16,2048)
    schedule,r['schedule_generation_ms_per_step']=make_schedule(steps=8)
    r['scope']='Complete hot-pool step including gather, transfer, augmentation, loss, backward, clip and AdamW. No cold IO, validation or checkpoint writes.'
    labels=('baseline_a','candidate_a','baseline_b','candidate_b')
    for role in ('relative5','distance_mlp'):
        base,ck=base_model(role)
        for block in range(8):
            order=labels[block%4:]+labels[:block%4]
            for label in order:
                check(r)
                model=make(base,label.startswith('candidate'));opt=optimizer(model)
                for j in range(2):step(model,opt,pool,target,schedule[j],831000+block*20+j)
                model.zero_grad(set_to_none=True);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                wall=time.perf_counter();start.record()
                for j in range(2,8):loss,norm=step(model,opt,pool,target,schedule[j],831000+block*20+j)
                end.record();torch.cuda.synchronize()
                ms=(time.perf_counter()-wall)*1000/6
                r['measurements'].append(dict(role=role,block=block,label=label,wall_ms=ms,
                    cuda_span_ms=start.elapsed_time(end)/6,peak_bytes=torch.cuda.max_memory_allocated(),
                    final_loss=float(loss),final_grad_norm=float(norm),checkpoint=ck))
                del model,opt,loss,norm;cleanup()
            write(OUT/'timing_progress.json',r)
            print('TIMING',role,block,flush=True)


def stability(r):
    pool,target,r['input_identity']=old_inputs(16,2048)
    schedule,r['schedule_generation_ms_per_step']=make_schedule(steps=64)
    r['scope']='Private 64-step old-input stability; no scientific accuracy or convergence claim.'
    for role in ('relative5','distance_mlp'):
        base,ck=base_model(role)
        models=[make(base,f) for f in (False,True)];opts=[optimizer(m) for m in models]
        for i,spec in enumerate(schedule):
            if i%8==0:check(r)
            pair={}
            for f in (i%2,1-i%2):
                loss,norm=step(models[f],opts[f],pool,target,spec,841000+i)
                pair['candidate' if f else 'baseline']=dict(loss=float(loss),grad_norm=float(norm))
            assert all(np.isfinite(v['loss']) and np.isfinite(v['grad_norm']) for v in pair.values())
            r['measurements'].append(dict(role=role,step=i,**pair))
            if i%16==15:
                write(OUT/'stability_progress.json',r);print('STABILITY',role,i+1,flush=True)
            time.sleep(.05)
        delta={n:float((p-dict(models[1].named_parameters())[n]).abs().max()) for n,p in models[0].named_parameters()}
        r.setdefault('final_models',[]).append(dict(role=role,max_parameter_abs=max(delta.values()),checkpoint=ck))
        del models,opts,loss,norm;cleanup()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('correctness','timing','stability'));args=p.parse_args()
    r=initialize(args.stage)
    try:
        globals()[args.stage](r);finish(r)
    except BaseException as e:
        failure(r,e);raise
