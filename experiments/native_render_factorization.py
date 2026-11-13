"""Fixed-weight, zero-update Render input factorization; prospective round_001."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np  # Compatible runtime requires this import before torch.
import torch

from experiments import native_coverage_data as d
from experiments.native_receiver_data import core_load
from experiments.native_receiver_runtime import model_load, forward_read
from experiments.gaussian_scene_preflight import metric_depth
from models.gaussian_scene_rendered import RenderedFeatureFusion

OUT = d.ROOT/'results/gaussian_memory/native_render_factorization1'
PLAN = d.ROOT/'results/gaussian_memory/autonomous_research_loop/round_001/plan.json'
ARMS = ('N','C','U','M')
STOP = False


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def cases():
    p=d.plan()
    return [(m,q,'train') for m in p['training_catalog'] for q in d.HELDOUT]+[(m,q,'development') for m in p['catalog'] for q in d.QUERIES]


def inputs(field, payload):
    # Original r remains unchanged. No epsilon or effect-dependent threshold.
    r=payload['render'];a=payload['effective_alpha'].reshape(1024,1).float()
    valid=field['has_content'].bool();f=field['features'].float()
    g=f[valid].mean(0) if bool(valid.any()) else torch.zeros(1024,dtype=torch.float32)
    n=g[None].expand_as(r).clone();nonzero=a[:,0]>0
    n[nonzero]=(r[nonzero].double()/a[nonzero].double()).float()
    nr=float(torch.linalg.vector_norm(r.double()));nn=float(torch.linalg.vector_norm(n.double()))
    s=nr/nn if nn else 0.
    out=dict(R=r,N=n,C=a*g[None],U=g[None].expand_as(r),M=(s*n.double()).float())
    assert r.shape==(1024,1024) and bool((a>=0).all()) and float(a.max())<=1.0001
    assert all(bool(torch.isfinite(v).all()) for v in out.values())
    stats=dict(alpha_zero_fraction=float((a==0).float().mean()),alpha_quantile_levels=[0,.25,.5,.75,.9,.99,1.],alpha_quantiles=torch.quantile(a.double(),torch.tensor([0,.25,.5,.75,.9,.99,1.],dtype=torch.float64)).tolist(),
        valid_slots=int(valid.sum()),s=s,r_norm=nr,n_norm=nn,
        norm_ratio={k:float(torch.linalg.vector_norm(v.double()))/nr if nr else None for k,v in out.items()},
        relative_difference_from_r={k:float(torch.linalg.vector_norm((v-r).double()))/nr if nr else None for k,v in out.items()})
    return out,stats


def expected_files(rid,q):
    return [d.scene_path(rid),d.query_path(rid,q),d.feature_path(rid,q)]+[d.patch_path(rid,q,c) for c in d.CONDITIONS]


def prepare():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    if (OUT/'configuration_lock.json').exists():
        verify();print('Existing configuration lock verified.',flush=True);return
    start=time.monotonic();p=d.read(PLAN)
    assert d.sha256(PLAN)=='36bf5b5e04836cab169834dfeb4068445f499f88d077c0fd9f0d176d27e0f43b'
    assert d.sha256(d.OUT/'plan.json')==p['fixed_parent']['plan_sha256']
    identity={};origins={};records={};asset_paths={}
    # Resolve hashes from authoritative parent cache markers, never inferred filenames.
    for meta,q,cohort in cases():
        rid=meta['id']
        marker=(d.OUT/'query_cache'/f'{rid}.json' if d.is_new(rid) else
            d.VIEW_OUT/'query_cache'/f'{rid}.json' if q in d.HELDOUT else
            d.FROZEN/'features'/f'{rid}.json')
        if str(marker) not in records:records[str(marker)]=d.read(marker)
        rec=records[str(marker)];assert rec['complete']
        expected=dict(rec.get('files',{}),**rec.get('inputs',{}))
        if 'source_path' in rec:expected[rec['source_path']]=rec['source_sha256']
        for path in expected_files(rid,q):
            key=str(path)
            if key in identity:continue
            h=d.sha256(path)
            if key in expected: assert h==expected[key],key
            elif path==d.scene_path(rid):
                assert h==d.read(d.VIEW_OUT/'identities.json')['scenes'][rid]['sha256']
            else:raise AssertionError(('missing parent cache identity',key,str(marker)))
            identity[key]=h;origins[key]=str(marker)
            stat=path.stat();asset_paths[key]=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)
    models=[]
    for m in p['source_and_model_identity']['models']:
        h=d.sha256(m['checkpoint_path']);assert h==m['checkpoint_sha256_from_completed_parent']
        models.append(dict(path=m['checkpoint_path'],sha256=h,seed=m['seed'],bytes=Path(m['checkpoint_path']).stat().st_size,mtime_ns=Path(m['checkpoint_path']).stat().st_mtime_ns))
    inherited=[]
    for rec in p['source_and_model_identity']['inherited_predictions']:
        assert d.sha256(rec['marker'])==rec['marker_sha256']
        assert d.sha256(rec['prediction_path'])==rec['prediction_sha256']
        inherited.append(rec)
    checks=[]
    for rid,q in [('fit_00',1),('confirm_12',1)]:
        field=d.load(d.scene_path(rid));payload=d.load(d.feature_path(rid,q));xx,stats=inputs(field,payload)
        a=payload['effective_alpha'].reshape(1024,1).double();take=a[:,0]>0
        # Separate NumPy formula check on the two prescribed training inputs.
        r=payload['render'].numpy().astype('float64');av=a.numpy();v=field['has_content'].numpy().astype(bool)
        g=field['features'].numpy().astype('float32')[v].mean(0,dtype='float32') if v.any() else np.zeros(1024,dtype='float32')
        n=np.broadcast_to(g,r.shape).copy();n[take.numpy()]=(r[take.numpy()]/av[take.numpy()]).astype('float32')
        np_error=float(np.max(np.abs(n-xx['N'].numpy())))
        assert np.allclose(n,xx['N'].numpy(),rtol=2e-6,atol=2e-6),np_error
        assert torch.equal(xx['R'],payload['render'])
        assert torch.allclose(xx['N'][take].double()*a[take],xx['R'][take].double(),rtol=1e-7,atol=1e-7)
        checks.append(dict(id=rid,query=q,finite_all_inputs=True,original_R_unchanged=True,numpy_normalization_max_abs=np_error,statistics=stats))
    source_names=['experiments/native_render_factorization.py','experiments/native_coverage_data.py','experiments/native_receiver_runtime.py','experiments/native_receiver_data.py','models/gaussian_scene_rendered.py','models/gaussian_scene_native.py']
    # Include all native and runtime code used by the fixed parent lock, without changing it.
    source_names+=list(d.read(d.OUT/'configuration_lock.json')['source_sha256'])
    source_hashes={name:d.sha256(d.ROOT/name) for name in sorted(set(source_names))}
    model_cfg=dict(task_path='/root/research_implementation_sol',creation_config=dict(model='gpt-6.1-sol',reasoning_effort='high',fork_turns='none',source='parent-reported explicit accepted spawn request'),followup_parameter_support=False,effective_model=None,effective_reasoning_effort=None,effective_independently_verified=False)
    write(OUT/'preflight_inputs.json',dict(complete=True,checks=checks,cache_semantics='Original prepare code computes effective=actual contribution*has_content, r=effective@features, effective_alpha=effective.sum; verified parent cache identities.',native_forwards=0))
    write(OUT/'configuration_lock.json',dict(complete=True,created_utc=now(),plan_path=str(PLAN),plan_sha256=d.sha256(PLAN),source_sha256=source_hashes,input_sha256=identity,input_parent_marker=origins,input_stats=asset_paths,models=models,inherited=inherited,preflight_sha256=d.sha256(OUT/'preflight_inputs.json'),model_config=model_cfg,seconds=time.monotonic()-start))
    print(dict(configuration_locked=True,input_assets=len(identity),models=len(models),seconds=time.monotonic()-start),flush=True)


def verify():
    lock=d.read(OUT/'configuration_lock.json');assert d.sha256(PLAN)==lock['plan_sha256']
    assert all(d.sha256(d.ROOT/n)==h for n,h in lock['source_sha256'].items())
    for path,v in lock['input_stats'].items():
        st=Path(path).stat();assert st.st_size==v['size'] and st.st_mtime_ns==v['mtime_ns'],path
    for m in lock['models']:
        st=Path(m['path']).stat();assert st.st_size==m['bytes'] and st.st_mtime_ns==m['mtime_ns']
    return lock


def journal(path,row):
    with path.open('a',buffering=1) as f:f.write(json.dumps(row)+'\n')


def charge():
    with (OUT/'forward_budget.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        path=OUT/'forward_budget.json';record=d.read(path) if path.exists() else dict(charged=0,maximum=4272)
        assert record['charged']<record['maximum'],'Native forward budget exhausted'
        record['charged']+=1;write(path,record)


def run(seeds):
    global STOP
    def stop(*_):
        global STOP
        STOP=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    torch.set_num_threads(4);lock=verify();run_id=f'{seeds[0]}_{time.time_ns()}'
    folder=OUT/'runtime';folder.mkdir(exist_ok=True);log=folder/f'{run_id}.jsonl';begin=time.monotonic()
    def record(**kw):journal(log,dict(utc=now(),elapsed_seconds=time.monotonic()-begin,**kw))
    with (folder/f'worker_{seeds[0]}.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.cuda.set_per_process_memory_fraction(8*2**30/torch.cuda.get_device_properties(0).total_memory)
        free,total=torch.cuda.mem_get_info();assert free>=14*2**30
        record(event='start',pid=os.getpid(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),free_bytes=free,cuda=True,seeds=seeds)
        forwards=0;complete=0
        try:
            model=model_load(d.plan());reader=RenderedFeatureFusion().cuda().eval()
            model.eval().requires_grad_(False);reader.requires_grad_(False)
            record(event='model_loaded',peak_GiB=torch.cuda.max_memory_allocated()/2**30)
            for seed in seeds:
                if STOP:raise SystemExit(75)
                t=time.monotonic();state=d.load(d.checkpoint_path('Render',seed));assert state['step']==2000
                core_load(model,state['native']);reader.load_state_dict(state['model']);del state
                record(event='weights_loaded',seed=seed,seconds=time.monotonic()-t)
                # The prescribed R replays check exact existing predictions only.
                for cohort in ('train','development'):
                    m=d.read(d.OUT/'evaluation'/f'{cohort}_Render_{seed}.json');old=d.load(m['prediction_path'])
                    saved={tuple(k):i for i,k in enumerate(old['keys'])}
                    for rid,q in ([('fit_00',1),('confirm_12',1)] if cohort=='train' else [('fidelity_00',2),('fidelity_02',11)]):
                        for c in d.CONDITIONS:
                            name=f'{seed}_{rid}_{q}_{c}';marker=OUT/'replay'/f'{name}.json'
                            if marker.exists():assert d.read(marker)['complete'];continue
                            if STOP:raise SystemExit(75)
                            sample=dict(patches=d.load(d.patch_path(rid,q,c))['patches'].cuda(),extra=d.load(d.feature_path(rid,q))['render'].cuda(),hw=(448,448))
                            charge();record(event='forward_start',purpose='baseline_replay',key=name);forwards+=1
                            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):pred=metric_depth(forward_read(model,sample,reader,'Render',checkpoint_blocks=False)).float().cpu()
                            difference=float((pred-old['depth_metric'][saved[rid,q,c]]).abs().max());assert difference<=1e-5,(name,difference)
                            write(marker,dict(complete=True,seed=seed,id=rid,query=q,condition=c,max_abs=difference,source=str(m['prediction_path']),source_sha256=m['prediction_sha256']))
                            record(event='forward_complete',purpose='baseline_replay',key=name,max_abs=difference)
                    del old
                for meta,q,cohort in cases():
                    rid=meta['id'];t=time.monotonic();field=d.load(d.scene_path(rid));payload=d.load(d.feature_path(rid,q));extra,stats=inputs(field,payload)
                    write(OUT/'input_statistics'/f'{rid}_{q}.json',dict(id=rid,query=q,cohort=cohort,**stats))
                    record(event='inputs_constructed',id=rid,query=q,seconds=time.monotonic()-t)
                    del field,payload
                    for c in d.CONDITIONS:
                        patches=d.load(d.patch_path(rid,q,c))['patches'].cuda()
                        for arm in ARMS:
                            name=f'{seed}_{rid}_{q}_{c}_{arm}';marker=OUT/'completed'/f'{name}.json';path=OUT/'predictions'/f'{name}.npy'
                            if marker.exists():
                                r=d.read(marker);assert r['complete'] and path.stat().st_size==r['bytes'];continue
                            if STOP:raise SystemExit(75)
                            free,_=torch.cuda.mem_get_info()
                            if free<6*2**30:record(event='resource_yield',free_bytes=free);raise SystemExit(75)
                            # Hard budget is summed across all workers, including prior attempts.
                            used=0.
                            for logfile in folder.glob('*.jsonl'):
                                lines=logfile.read_text().splitlines()
                                if lines:used+=json.loads(lines[-1])['elapsed_seconds']
                            if used>7200:raise RuntimeError('Hard 2 worker-hour budget reached')
                            if sum(x.stat().st_size for x in OUT.rglob('*') if x.is_file())>2*2**30:raise RuntimeError('Disk budget reached')
                            sample=dict(patches=patches,extra=extra[arm].cuda(),hw=(448,448));t=time.monotonic()
                            charge();record(event='forward_start',purpose='formal',key=name);forwards+=1
                            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):pred=metric_depth(forward_read(model,sample,reader,'Render',checkpoint_blocks=False)).float().cpu()
                            torch.cuda.synchronize();inference_seconds=time.monotonic()-t
                            assert pred.shape==(210,210) and torch.isfinite(pred).all() and (pred>0).all()
                            path.parent.mkdir(exist_ok=True);tmp=path.with_suffix('.tmp')
                            with tmp.open('wb') as f:np.save(f,pred.numpy(),allow_pickle=False)
                            tmp.replace(path)
                            write(marker,dict(complete=True,seed=seed,id=rid,query=q,condition=c,arm=arm,cohort=cohort,place=meta['place'],prediction_path=str(path),sha256=d.sha256(path),bytes=path.stat().st_size,inference_seconds=inference_seconds,lock_sha256=d.sha256(OUT/'configuration_lock.json')))
                            complete+=1;record(event='forward_complete',purpose='formal',key=name,seconds=inference_seconds,peak_GiB=torch.cuda.max_memory_allocated()/2**30)
                            del sample,pred
                        del patches
                    if complete%128==0:print(dict(seed=seed,new_this_run=complete,elapsed_seconds=time.monotonic()-begin),flush=True)
                    del extra
            record(event='complete',forwards=forwards,new_predictions_this_run=complete,peak_GiB=torch.cuda.max_memory_allocated()/2**30)
        except BaseException as exc:
            record(event='exit_incomplete',error=repr(exc),forwards=forwards,new_predictions_this_run=complete,peak_GiB=torch.cuda.max_memory_allocated()/2**30)
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run']);parser.add_argument('--seeds',type=int,nargs='+');a=parser.parse_args()
    if a.command=='prepare':prepare()
    else:run(a.seeds)
