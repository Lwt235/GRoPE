"""Fixed-checkpoint full evaluation and a four-arm composition benchmark."""
import argparse
import os
from pathlib import Path
import time
from types import MethodType
import numpy as np
import torch
import torch.nn.functional as F
from experiments.round65_common import ROOT, CONFIG, read, write, sha
from experiments.round65_run import evaluation_batch, metrics, epoch_batches
from experiments.efficiency_sidecar.benchmark_gpu import resource_check
from experiments.efficiency_sidecar.check_cpu import variant, old_inputs, state_hash
from experiments.efficiency_sidecar.followup_gpu import base_model
from experiments.efficiency_sidecar.hidden_gpu import optimizer, step, cleanup
from experiments.efficiency_sidecar.compiled_runtime import adapt_compiled, compiled_block, OPTIONS
from experiments.efficiency_sidecar.fused_hidden_grad import fuse_hidden_gradient, LayerRelations
from models.factored_relation_runtime import ProjectedRelation

OUT=ROOT/'results/efficiency_sidecar/deployment1'


def combined_block(self,x,relative=None,centers=None,frequency=None):
    assert isinstance(relative,LayerRelations)
    relation=ProjectedRelation(relative.hidden[self._merge_slot],relative.weight)
    return compiled_block(self,x,relation,centers,frequency)


def make(base,arm):
    model=variant(base,'original' if arm=='original' else 'geometry_factored')
    if arm in ('compiled','combined'):adapt_compiled(model)
    if arm in ('fused','combined'):
        fuse_hidden_gradient(model)
        if arm=='combined':
            for layer in model.layers:layer.forward=MethodType(combined_block,layer)
    return model.cuda()


def resource(r,initialized=True):
    if time.time()-r['started_unix']>1200:raise TimeoutError('20 minute stage limit')
    r['resources'].append(dict(unix=time.time(),**resource_check(r['gpu_uuid'],initialized=initialized,shared_peers={})))


def initialize(args):
    dest=OUT/args.stage/args.role
    if (dest/'execution.json').exists():raise FileExistsError(dest)
    r=dict(stage=args.stage,role=args.role,pid=os.getpid(),started_unix=time.time(),
        gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],resources=[],measurements=[],checks=[],
        sources={str(p.relative_to(ROOT)):sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        model_sources={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'models').glob('*.py'))},
        plan_sha256=sha(Path(__file__).with_name('DEPLOYMENT_PLAN.md')),config=CONFIG,
        compile_options=OPTIONS,compiler_cache=os.environ['TORCHINDUCTOR_CACHE_DIR'],
        cpu_threads=1,allocator_cap_bytes=12*2**30,scientific_confirmation=False)
    resource(r,False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.cuda.set_per_process_memory_fraction(r['allocator_cap_bytes']/torch.cuda.get_device_properties(0).total_memory)
    torch.ones(1,device='cuda').sum().item()
    r.update(torch_version=torch.__version__,gpu_name=torch.cuda.get_device_name())
    resource(r);write(dest/'execution.json',r)
    return r,dest


def evaluation(r,dest):
    root=ROOT/'results/round65_actual_gs/attempt1/data'
    manifest=read(root/'manifest.json');row=next(x for x in manifest['partitions'] if x['split']=='val')
    for name,digest in row['hashes'].items():assert sha(root/'val'/name)==digest,name
    r['data_identity']=dict(manifest_sha256=sha(root/'manifest.json'),partition=row)
    r['batch']=CONFIG['evaluation_batch_size'];r['tokens']=512
    base,r['checkpoint']=base_model(r['role']);initial=state_hash(base)
    arms=('original','geometry_factored','compiled')
    orders=[('first',-1,a) for a in (arms if r['role']=='relative5' else arms[::-1])]
    labels=[(a,rep) for rep in ('a','b') for a in arms]
    for block in range(2):
        orders.extend((rep,block,arm) for arm,rep in (labels if block==0 else labels[::-1]))
    for rep,block,arm in orders:
        resource(r);start=time.perf_counter();model=make(base,arm).eval();torch.cuda.synchronize()
        setup=time.perf_counter()-start
        assert state_hash(model)==initial
        torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        bundle={k:np.load(root/'val'/(k+'.npy'),mmap_mode='r',allow_pickle=False)
                for k in ('centers','covariance','appearance','labels')}
        bundle['ids']=np.asarray(row['ids']);logits=[];first_batch=None;tail=None
        with torch.inference_mode():
            for pos in range(0,len(bundle['labels']),r['batch']):
                end=min(pos+r['batch'],len(bundle['labels']));before=time.perf_counter()
                inputs=evaluation_batch(bundle,pos,end,'base','cuda')
                logits.append(model(*inputs).cpu().numpy())
                duration=time.perf_counter()-before
                if pos==0:first_batch=duration
                if end-pos<r['batch']:tail=duration
        logits=np.concatenate(logits);score=metrics(logits,bundle['labels'])
        file=dest/f'{arm}_{block}_{rep}.npz'
        with file.open('wb') as f:
            np.savez(f,logits=logits,labels=bundle['labels'],ids=bundle['ids']);f.flush();os.fsync(f.fileno())
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        r['measurements'].append(dict(arm=arm,block=block,rep=rep,seconds=elapsed,setup_seconds=setup,
            total_seconds=setup+elapsed,first_batch_seconds=first_batch,tail_batch_seconds=tail,
            metrics=score,peak_allocated_bytes=torch.cuda.max_memory_allocated(),prediction_file=str(file),
            prediction_sha256=sha(file)))
        assert state_hash(model)==initial
        resource(r);write(dest/'progress.json',r)
        print('EVAL',r['role'],arm,block,rep,round(elapsed,3),score['oa'],flush=True)
        del model,bundle,inputs,logits;cleanup()


def combination(r,dest):
    r.update(batch=16,tokens=512)
    base,r['checkpoint']=base_model(r['role']);arms=('geometry_factored','fused','compiled','combined')
    pool,target,r['correctness_input_identity']=old_inputs(16,512)
    inputs=tuple(t.cuda() for t in pool);y=target.cuda();reference=None
    for arm in arms:
        resource(r);model=make(base,arm).train();opt=optimizer(model);torch.manual_seed(831)
        start=time.perf_counter();out=model(*inputs);loss=F.cross_entropy(out,y,label_smoothing=.1);loss.backward()
        torch.cuda.synchronize();first=time.perf_counter()-start
        grads={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters()}
        assert all(bool(torch.isfinite(g).all()) for g in grads.values())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        updated={n:p.detach().cpu().clone() for n,p in model.named_parameters()};model.eval()
        with torch.inference_mode():after=model(*inputs).cpu()
        current=dict(output=out.detach().cpu(),grads=grads,updated=updated,after=after)
        if reference is None:reference=current
        torch.testing.assert_close(current['output'],reference['output'],atol=2e-6,rtol=2e-5)
        torch.testing.assert_close(after,reference['after'],atol=2e-4,rtol=2e-4)
        grad_abs=grad_rel=update_abs=0.;exceptions=[]
        for n,g in grads.items():
            torch.testing.assert_close(g,reference['grads'][n],atol=3e-6,rtol=3e-4)
            diff=(g-reference['grads'][n]).double();grad_abs=max(grad_abs,float(diff.abs().max()))
            grad_rel=max(grad_rel,float(diff.norm()/reference['grads'][n].double().norm().clamp_min(1e-12)))
            ud=(updated[n]-reference['updated'][n]).abs();update_abs=max(update_abs,float(ud.max()))
            mask=ud>2e-5+2e-4*reference['updated'][n].abs()
            if bool(mask.any()):exceptions.append(dict(name=n,count=int(mask.sum()),max_abs=float(ud.max())))
        r['checks'].append(dict(arm=arm,first_train_seconds=first,logit_max_abs=float((current['output']-reference['output']).abs().max()),
            gradient_max_abs=grad_abs,gradient_max_tensor_relative_l2=grad_rel,update_max_abs=update_abs,
            update_exceptions=exceptions,after_update_logit_max_abs=float((after-reference['after']).abs().max())))
        print('CHECK',r['role'],arm,r['checks'][-1],flush=True);write(dest/'progress.json',r)
        del model,opt,out,loss,current,grads,updated,after;cleanup()
    del reference,inputs,y,pool,target;cleanup()
    pool,target,r['timing_input_identity']=old_inputs(16,2048)
    rng=np.random.default_rng(926001);schedule=[]
    while len(schedule)<6:schedule.extend(epoch_batches(rng,16,batch_size=16,tokens=512))
    labels=[(a,rep) for rep in ('a','b') for a in arms]
    for block in range(8):
        order=labels[block%8:]+labels[:block%8]
        for arm,rep in order:
            resource(r);model=make(base,arm).train();opt=optimizer(model)
            for j in range(2):value=step(model,opt,pool,target,schedule[j],861000+block*20+j)
            del value;model.zero_grad(set_to_none=True);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter()
            for j in range(2,6):value=step(model,opt,pool,target,schedule[j],861000+block*20+j)
            torch.cuda.synchronize();ms=(time.perf_counter()-start)*1000/4
            r['measurements'].append(dict(arm=arm,rep=rep,block=block,wall_ms=ms,final_loss=float(value[0]),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
            del model,opt,value;cleanup()
        resource(r);write(dest/'progress.json',r);print('TIMING',r['role'],block,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('evaluation','combination'))
    p.add_argument('role',choices=('relative5','distance_mlp'));args=p.parse_args()
    r,dest=initialize(args)
    try:
        globals()[args.stage](r,dest);r.update(state='complete',elapsed_seconds=time.time()-r['started_unix'])
        resource(r);write(dest/'result.json',r)
    except BaseException as e:
        r.update(state='failed',error_type=type(e).__name__,error=str(e),elapsed_seconds=time.time()-r['started_unix'])
        write(dest/'failure.json',r);raise
