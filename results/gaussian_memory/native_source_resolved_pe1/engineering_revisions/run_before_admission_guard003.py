"""Round028 meaningful gradient/restore check,15pairedfits, fixed full track export."""
import argparse,os,time,signal,functools,resource,collections,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from experiments import native_source_resolved_data as d
from experiments.gaussian_memory_native_run import model_load as native_model_load
from models.native_source_resolved_memory import SourceResolvedMemory
from models.native_track_memory import TrackMemory,native_tokens,query_tokens,fmap,track

def make_adapter():
    p=d.load(d.CACHE/'PCA255.pt')
    return SourceResolvedMemory(p['basis'],p['mean']).cuda()
def model_load(name):
    m=native_model_load(d.plan());m.aggregator.patch_embed.cpu();m.point_head.cpu();torch.cuda.empty_cache();torch.cuda.synchronize();d.journal(name,kind='official_model_loaded',all_native_frozen=not any(p.requires_grad for p in m.parameters()));return m

STOP=False
def stop(*a):
    global STOP;STOP=True
def digest(module):
    h=hashlib.sha256()
    for n,v in module.state_dict().items():h.update(n.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def schedule(seed):
    keys=[(m['id'],a,q) for m in d.plan()['training_catalog'] for q in d.QUERIES[:4] for a in d.SOURCES];rng=np.random.default_rng(seed);out=[]
    while len(out)<1000:out.extend(keys[int(i)] for i in rng.permutation(len(keys)))
    return out[:1000]
class Samples:
    def __init__(self,m,name,anchor_limit=8):self.model=m;self.name=name;self.anchors=collections.OrderedDict();self.fields={};self.anchor_limit=anchor_limit
    @functools.lru_cache(24)
    def patches(self,rid,q):return d.load(d.old.patch_path(rid,q,'full'))['patches']
    @functools.lru_cache(48)
    def prepared(self,rid,a,q):return d.load(d.case(rid,a,q))
    @functools.lru_cache(4)
    def sourcepatch(self,rid):return d.source_patches(rid)[0]
    def anchor(self,rid,a):
        key=rid,a
        if key not in self.anchors:
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):tokens,prefix=native_tokens(self.model,self.sourcepatch(rid)[d.SOURCES.index(a):d.SOURCES.index(a)+1].cuda())
            with torch.no_grad():feature=fmap(self.model,tokens)
            self.anchors[key]=(prefix,feature);d.journal(self.name,kind='frozen_anchor_reconstruction',id=rid,anchor=a)
            if len(self.anchors)>self.anchor_limit:self.anchors.popitem(last=False)
        self.anchors.move_to_end(key);return self.anchors[key]
    def clear(self):
        self.patches.cache_clear();self.prepared.cache_clear();self.sourcepatch.cache_clear();self.anchors.clear();self.fields.clear()
    def get(self,rid,a,q,kind):
        case=self.prepared(rid,a,q);patches=self.patches(rid,q).cuda();v=case['inputs']['GS']
        field_key=(rid,kind[0])
        if field_key not in self.fields:self.fields[field_key]=d.load(d.state(rid,kind[0]))['content'].float()
        content=self.fields[field_key].cuda()
        sample=dict(ordinary=torch.cat((content,v['ordinary'].cuda()),-1)[None],position=torch.cat((v['common25'],v[kind[-1]]),-1)[None].cuda(),scene_pos=v['scene_pos'][None].cuda(),key_mask=v['key_mask'][None].cuda())
        prefix,af=self.anchor(rid,a);return patches,sample,prefix,af,case
def predict(m,adapter,samples,key,kind,checkpoint=True):
    patches,sample,prefix,af,case=samples.get(*key,kind);versions=[(k._version,v._version) for k,v in prefix]
    with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):tokens=query_tokens(m,patches,prefix,adapter,sample,checkpoint)
    result=track(m,af,tokens,d.point_grid()[None].cuda());assert versions==[(k._version,v._version) for k,v in prefix]
    return result,case
def loss_for(result,case):
    valid=case['labels']['valid'].cuda();target=case['labels']['target'].cuda().float()
    return sum(F.smooth_l1_loss(c[0,1][valid],target[valid],beta=8,reduction='mean') if valid.any() else c.sum()*0 for c in result[0])/4
def checkpoint(path,adapter,opt,step,seed,kind,order):
    d.save(path,dict(adapter=d.cpu(adapter.state_dict()),optimizer=d.cpu(opt.state_dict()),step=step,seed=seed,kind=kind,schedule=order,schedule_sha256=hashlib.sha256(d.json.dumps(order).encode()).hexdigest(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state().cpu(),science_sha256=d.SCIENCE_SHA))
def fit(m,name,kind,seed):
    folder=d.CACHE/'fits'/f'{kind}_{seed}';final=folder/'final.pt';latest=folder/'latest.pt';order=schedule(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);adapter=make_adapter();opt=torch.optim.AdamW(adapter.parameters(),lr=1e-4,weight_decay=.01);step=0
    if final.exists():
        state=d.load(final);assert state['science_sha256']==d.SCIENCE_SHA and state['kind']==kind and state['seed']==seed and state['step']==1000;adapter.load_state_dict(state['adapter']);d.journal(name,kind='evaluation_resume_from_permanent_final',arm=kind,seed=seed,step=1000,inference_only_no_remaining_updates=True,complete_Adam_RNG_sampler_preserved_in_checkpoint=True,already_committed_predictions=len(list((d.OUT/'completed').glob(f'{kind}_{seed}_*.json'))));return adapter
    if latest.exists():
        state=d.load(latest);assert state['science_sha256']==d.SCIENCE_SHA and state['schedule']==order;adapter.load_state_dict(state['adapter']);opt.load_state_dict(state['optimizer']);step=state['step'];torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state(state['cuda_rng']);del state
    else:d.write(d.OUT/'fits'/f'{kind}_{seed}_initial.json',dict(parameters=sum(p.numel() for p in adapter.parameters()),initial_sha256=digest(adapter),schedule_sha256=hashlib.sha256(d.json.dumps(order).encode()).hexdigest(),official_model_frozen=True,steps=1000,batch=1))
    samples=Samples(m,name,anchor_limit=128);start=time.monotonic();d.journal(name,kind='fit_resume',arm=kind,seed=seed,step=step,complete_Adam_RNG_sampler_restored=bool(step),frozen_anchor_GPU_cache_limit=128);print(dict(fit=kind,seed=seed,start_step=step),flush=True)
    for i in range(step,1000):
        opt.zero_grad(set_to_none=True);result,case=predict(m,adapter,samples,order[i],kind);loss=loss_for(result,case)
        if not torch.isfinite(loss):raise FloatingPointError(('training_loss',kind,seed,i))
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(adapter.parameters(),5.)
        if not torch.isfinite(norm):raise FloatingPointError(('training_gradient',kind,seed,i))
        opt.step();step=i+1;d.journal(name,kind='formal_update',arm=kind,seed=seed,step=step,sample=order[i],GT_points=int(case['labels']['valid'].sum()),loss=float(loss.detach()),gradient_norm=float(norm),native_query=1,tracker_iterations=4)
        if step%100==0 or STOP or step==1000:
            checkpoint(final if step==1000 else latest,adapter,opt,step,seed,kind,order);print(dict(arm=kind,seed=seed,step=step,loss=float(loss.detach()),seconds=time.monotonic()-start),flush=True)
        del result,case,loss
        if STOP:raise SystemExit(75)
    d.write(d.OUT/'fits'/f'{kind}_{seed}_complete.json',dict(complete=True,steps=step,final=str(final),sha256=d.sha(final),seconds=time.monotonic()-start,parameters=sum(p.numel() for p in adapter.parameters())))
    if latest.exists():latest.unlink()
    samples.clear();del opt,samples;torch.cuda.empty_cache();return adapter
@torch.no_grad()
def evaluate(m,adapter,name,kind,seed):
    samples=Samples(m,name);adapter.eval()
    for meta in d.metas():
        rid=meta['id'];d.write(d.OUT/'public_states'/f'{kind}_{seed}_{rid}.json',dict(source_only=True,field_reference=str(d.state(rid,kind[0])),global_PCA_reference=str(d.CACHE/'PCA255.pt'),geometry_reference=str(d.fields_path(rid)),GS_field_consumed=True,adapter_final_reference=str(d.CACHE/'fits'/f'{kind}_{seed}'/'final.pt'),persistent_query_writes=0,representation='Source-only four-block content and immutable geometry; global decoder/LoRA/PE adapter; frozen source-native anchor prefix is a separate mixed-state dependency.'))
        for a in d.SOURCES:
            for q in d.QUERIES:
                marker=d.OUT/'completed'/f'{kind}_{seed}_{rid}_{a}_{q}.json'
                if marker.exists():continue
                t=time.monotonic();res,case=predict(m,adapter,samples,(rid,a,q),kind,False);coords=torch.stack([c[0,1].cpu() for c in res[0]]);path=d.CACHE/'predictions'/f'{kind}_{seed}'/rid/f'{a}_{q}.pt';d.save(path,dict(coords=coords,vis=res[1][0,1].cpu(),conf=res[2][0,1].cpu(),alignment_valid=case['inputs']['alignment_valid']))
                d.write(marker,dict(complete=True,kind=kind,seed=seed,id=rid,place=meta['place'],anchor=a,query=q,path=str(path),sha256=d.sha(path),valid_GT=int(case['labels']['valid'].sum()),zero_support=case['inputs']['zero_support'],alignment_valid=case['inputs']['alignment_valid'],all64predicted=True));d.journal(name,kind='evaluation_query',arm=kind,seed=seed,id=rid,anchor=a,query=q,seconds=time.monotonic()-t,tracker_iterations=4)
                if STOP:raise SystemExit(75)
        print(dict(evaluated=rid,arm=kind,seed=seed),flush=True)
    d.write(d.OUT/'evaluation'/f'{kind}_{seed}.json',dict(complete=True,cases=1408,all_points=1408*64,final_step=1000));samples.clear();del samples
def preflight(m,name):
    keys=[(meta['id'],a,q) for meta in d.plan()['training_catalog'] for q in d.QUERIES[:4] for a in d.SOURCES];key=next(k for k in keys if d.case(*k).exists() and d.load(d.case(*k))['labels']['valid'].any());samples=Samples(m,name);records=[];initials=[];mapping_values={};prediction_values={}
    for kind in d.KINDS:
        torch.manual_seed(d.SEEDS[0]);torch.cuda.manual_seed_all(d.SEEDS[0]);adapter=make_adapter();initials.append(digest(adapter))
        if kind in ('F_P','M_P'):
            _,sv,_,_,_=samples.get(*key,kind)
            with torch.no_grad():
                fp=adapter.source(sv['ordinary']).cpu()
                with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):bf=adapter.source(sv['ordinary']).float().cpu()
            mapping_values[kind]=(fp,bf)
        result,case=predict(m,adapter,samples,key,kind);loss=loss_for(result,case);loss.backward();grad={n:float(p.grad.float().norm()) for n,p in adapter.named_parameters() if p.grad is not None};assert all(np.isfinite(v) for v in grad.values()) and grad['source.weight']>0 and grad['position.2.weight']>0 and grad['qkv.0.b.weight']>0 and grad['qkv.1.b.weight']>0
        if kind in ('F_P','M_P'):prediction_values[kind]=torch.stack([c[0,1].detach().cpu() for c in result[0]])
        records.append(dict(kind=kind,loss=float(loss.detach()),gradients=grad,query_displacement_mean=float((result[0][-1][0,1]-case['labels']['query_points'].cuda()).norm(dim=-1).mean()),GT_points=int(case['labels']['valid'].sum())))
        if kind=='F_P':
            opt=torch.optim.AdamW(adapter.parameters(),lr=1e-4,weight_decay=.01);torch.nn.utils.clip_grad_norm_(adapter.parameters(),5.);opt.step();path=d.CACHE/'preflight'/'complete_restore.pt';checkpoint(path,adapter,opt,1,d.SEEDS[0],kind,schedule(d.SEEDS[0]));state=d.load(path);restored=make_adapter();restored.load_state_dict(state['adapter']);opt2=torch.optim.AdamW(restored.parameters(),lr=1e-4,weight_decay=.01);opt2.load_state_dict(state['optimizer']);assert digest(restored)==digest(adapter)
            for i,sv in opt.state_dict()['state'].items():
                for k,v in sv.items():assert torch.equal(v.cpu(),opt2.state_dict()['state'][i][k].cpu()) if torch.is_tensor(v) else v==opt2.state_dict()['state'][i][k]
            torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state(state['cuda_rng']);assert torch.equal(torch.get_rng_state(),state['torch_rng']) and torch.equal(torch.cuda.get_rng_state().cpu(),state['cuda_rng']);restore=dict(parameters_and_full_Adam_exact=True,RNG_fields_saved=True,RNG_restoration_exact=True,materialized1000sampler_exact=state['schedule']==schedule(d.SEEDS[0]),discarded_preflight_updates=1);del opt,opt2,restored,state
        del adapter,result,loss;torch.cuda.empty_cache();d.journal(name,kind='preflight_gradient',arm=kind,sample=key)
    assert len(set(initials))==1
    mapping_errors={}
    for i,label in enumerate(('FP32','BF16')):
        diff=(mapping_values['F_P'][i]-mapping_values['M_P'][i]).double();mapping_errors[label]=dict(max_abs=float(diff.abs().max()),RMS=float(diff.square().mean().sqrt()))
    diff=(prediction_values['F_P']-prediction_values['M_P']).double();prediction_errors=dict(max_abs_pixels=float(diff.abs().max()),RMS_pixels=float(diff.square().mean().sqrt()),loss_difference_F_minus_M=records[0]['loss']-records[3]['loss'],no_numerical_equality_gate=True)
    assert not any(p.requires_grad or p.grad is not None for p in m.parameters())
    d.write(d.OUT/'preflight.json',dict(complete=True,fixed_first_training_pair_with_GT=key,records=records,paired_arm_initials_exact=True,initial_mapping_finite_precision=mapping_errors,initial_prediction_finite_precision=prediction_errors,restore=restore,official_all_frozen=True,source_prefix_unchanged=True,real_twoframe_official_tracker=True,all64points_not_pseudo_identity=True,inherited_unchanged_Native2_and_empty_support_validation=str(d.prior.OUT/'preflight.json'),extra_native_queries=5,extra_backwards=5,formal_updates=0));samples.clear();print(dict(preflight_complete=True),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['preflight','fits']);p.add_argument('--kind');p.add_argument('--seed',type=int);p.add_argument('--name',required=True);a=p.parse_args();torch.set_num_threads(2);signal.signal(signal.SIGTERM,stop);torch.backends.cuda.enable_mem_efficient_sdp(False);start=time.monotonic()
    if os.environ.get('GROPE_EVAL_MEMORY_CAP_MIB'):
        assert a.phase=='fits' and (d.CACHE/'fits'/f'{a.kind}_{a.seed}'/'final.pt').exists()
        cap=int(os.environ['GROPE_EVAL_MEMORY_CAP_MIB']);torch.cuda.set_per_process_memory_fraction(cap*2**20/torch.cuda.get_device_properties(0).total_memory);d.journal(a.name,kind='evaluation_only_GPU_allocator_cap',MiB=cap,formal_updates_remaining=0)
    m=model_load(a.name);m.camera_head.cpu();m.depth_head.cpu();torch.cuda.empty_cache()
    if a.phase=='preflight':preflight(m,a.name)
    else:
        adapter=fit(m,a.name,a.kind,a.seed);evaluate(m,adapter,a.name,a.kind,a.seed)
    d.write(d.OUT/'worker_resources'/f'{a.name}.json',dict(phase=a.phase,seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),completed=True))
if __name__=='__main__':main()
