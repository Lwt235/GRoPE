"""Bounded contraction study; unchanged frozen model copies."""
import argparse
import os
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from experiments.round65_common import ROOT, CONFIG, read, write, sha
from experiments.round65_run import epoch_batches
from experiments.efficiency_sidecar.benchmark_gpu import resource_check
from experiments.efficiency_sidecar.check_cpu import old_inputs, state_hash
from experiments.efficiency_sidecar.followup_gpu import base_model
from experiments.efficiency_sidecar.hidden_gpu import make as old_make, optimizer, step, cleanup
from experiments.efficiency_sidecar.relation_layout import adapt_layout

OUT=ROOT/'results/efficiency_sidecar/layout1'
USE_MERGE=False

def make(base,candidate):
    model=old_make(base,USE_MERGE)
    return adapt_layout(model) if candidate else model


def resource(r,initialized=True):
    if time.time()-r['started_unix']>600:raise TimeoutError('10 minute limit')
    snapshot=resource_check(r['gpu_uuid'],initialized=initialized,shared_peers=r['peers'])
    if r['mode']=='exclusive' and not initialized:
        card=next(s.split(',') for s in snapshot['cards'].splitlines() if s.split(',')[1].strip()==r['gpu_uuid'])
        if int(card[2])>512 or int(card[4])>10:raise RuntimeError('Exclusive mode requires idle GPU')
    r['resources'].append(snapshot)


def initialize(args):
    out=OUT/('b'+str(args.batch)+('_merge' if args.merge else ''))
    if (out/f'{args.stage}_execution.json').exists():raise FileExistsError('Preserve existing stage')
    main=read(ROOT/'results/native_source_diversity/attempt1/lock.json')
    assert main['config']==CONFIG and CONFIG['batch_size']==16 and CONFIG['tokens']==512
    for p,d in main['source_sha256'].items():
        if p.endswith('.py'):assert sha(p)==d,p
    for p,d in read(ROOT/'results/efficiency_sidecar/backward1/correctness_v2.json')['sources'].items():
        assert sha(ROOT/p)==d,p
    uuid=os.environ['CUDA_VISIBLE_DEVICES']
    queue=read(ROOT/'results/native_source_diversity/attempt1/queue_status.json')
    peers={x['pid']:x['start_ticks'] for x in queue['active'] if x['gpu']['uuid']==uuid} if args.mode=='shared' else {}
    if args.mode=='shared':assert peers
    r=dict(stage=args.stage,mode=args.mode,pid=os.getpid(),started_unix=time.time(),gpu_uuid=uuid,peers=peers,
        batch=args.batch,tokens=512,merge=args.merge,allocator_cap_bytes=12*2**30,resources=[],checks=[],measurements=[],
        sources={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')},
        plan_sha256=sha(Path(__file__).parent/'LAYOUT_PLAN.md'),main_lock_sha256=sha(ROOT/'results/native_source_diversity/attempt1/lock.json'),
        main_training_modified=False,scientific_confirmation=False)
    resource(r,False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.cuda.set_per_process_memory_fraction(r['allocator_cap_bytes']/torch.cuda.get_device_properties(0).total_memory)
    torch.ones(1,device='cuda').sum().item();resource(r)
    r.update(torch_version=torch.__version__,gpu_name=torch.cuda.get_device_name())
    write(out/f'{args.stage}_execution.json',r)
    return r,out


def correctness(r,out):
    pool,y,r['input_identity']=old_inputs(r['batch'],512)
    inputs=tuple(t.cuda() for t in pool);y=y.cuda()
    for role in ('relative5','distance_mlp'):
        base,ck=base_model(role);states=[]
        for fused in (False,True):
            resource(r)
            model=make(base,fused);opt=optimizer(model)
            initial=state_hash(model)
            torch.manual_seed(831);torch.cuda.reset_peak_memory_stats()
            logits=model(*inputs);loss=F.cross_entropy(logits,y,label_smoothing=.1);loss.backward()
            gradients={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters()}
            assert all(bool(torch.isfinite(t).all()) for t in gradients.values())
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            states.append(dict(initial=initial,logits=logits.detach().cpu(),gradients=gradients,
                updated={n:p.detach().cpu().clone() for n,p in model.named_parameters()},
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
            del model,opt,logits,loss,gradients;cleanup()
        a,b=states;assert a['initial']==b['initial']
        torch.testing.assert_close(a['logits'],b['logits'],atol=2e-6,rtol=2e-5)
        errors={}
        for key,atol,rtol in (('gradients',3e-6,3e-4),('updated',2e-5,2e-4)):
            maximum=relative=0.
            for n in a[key]:
                torch.testing.assert_close(a[key][n],b[key][n],atol=atol,rtol=rtol,msg=lambda msg,n=n:f'{n}: {msg}')
                diff=(a[key][n]-b[key][n]).double()
                maximum=max(maximum,float(diff.abs().max()))
                relative=max(relative,float(diff.norm()/a[key][n].double().norm().clamp_min(1e-12)))
            errors[key]=dict(max_abs=maximum,max_tensor_relative_l2=relative)
        r['checks'].append(dict(role=role,checkpoint=ck,logit_max_abs=float((a['logits']-b['logits']).abs().max()),errors=errors,
            baseline_memory={k:a[k] for k in ('peak_allocated_bytes','peak_reserved_bytes')},
            candidate_memory={k:b[k] for k in ('peak_allocated_bytes','peak_reserved_bytes')}))
        print('CHECK',r['checks'][-1],flush=True)
        del states,a,b;cleanup()
        write(out/'correctness_progress.json',r)


def timing(r,out):
    correct=read(out/'correctness.json');assert correct['state']=='complete'
    for p,d in correct['sources'].items():assert sha(ROOT/p)==d,p
    r['correctness_sha256']=sha(out/'correctness.json')
    pool,target,r['input_identity']=old_inputs(16,2048)
    rng=np.random.default_rng(926001);schedule=[]
    while len(schedule)<6:schedule.extend(epoch_batches(rng,16,batch_size=r['batch'],tokens=512))
    labels=('baseline_a','candidate_a','baseline_b','candidate_b')
    for role in ('relative5','distance_mlp'):
        base,ck=base_model(role)
        # Resident inference includes the whole model/geometry; training includes the input pipeline.
        resident=tuple(t[:r['batch'],:512].cuda() for t in pool)
        for mode in ('inference','training'):
            for block in range(8):
                order=labels[block%4:]+labels[:block%4]
                for label in order:
                    resource(r)
                    model=make(base,label.startswith('candidate')).train(mode=='training')
                    opt=optimizer(model) if mode=='training' else None
                    def execute(j):
                        if opt is not None:return step(model,opt,pool,target,schedule[j],861000+block*20+j)
                        with torch.inference_mode():return model(*resident)
                    for j in range(2):value=execute(j)
                    del value
                    model.zero_grad(set_to_none=True);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                    first,last=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    wall=time.perf_counter();first.record()
                    for j in range(2,6):value=execute(j)
                    last.record();torch.cuda.synchronize()
                    elapsed=(time.perf_counter()-wall)*1000/4
                    r['measurements'].append(dict(role=role,mode=mode,block=block,label=label,wall_ms=elapsed,cuda_span_ms=first.elapsed_time(last)/4,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        final_loss=float(value[0]) if opt is not None else None,checkpoint=ck))
                    del model,opt,value;cleanup()
                write(out/'timing_progress.json',r)
                print('TIMING',role,mode,block,flush=True)
        del resident;cleanup()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('correctness','timing'));parser.add_argument('--mode',choices=('exclusive','shared'),default='exclusive')
    parser.add_argument('--batch',type=int,choices=(4,16),default=4);parser.add_argument('--merge',action='store_true')
    args=parser.parse_args();USE_MERGE=args.merge;r,out=initialize(args)
    try:
        globals()[args.stage](r,out)
        r.update(state='complete',elapsed_seconds=time.time()-r['started_unix'])
        write(out/f'{args.stage}.json',r)
    except BaseException as e:
        r.update(state='failed',error_type=type(e).__name__,error=str(e),elapsed_seconds=time.time()-r['started_unix'])
        write(out/f'{args.stage}_failure.json',r);raise
