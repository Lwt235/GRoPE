"""Round004: frozen Render covariance interventions, no training or new weights."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from experiments import native_coverage_data as d

OUT = d.ROOT/'results/gaussian_memory/native_covariance_role1'
PLAN = d.ROOT/'results/gaussian_memory/autonomous_research_loop/round_004/plan.json'
CACHE = Path(d.read(PLAN)['budget']['data_cache_root'])
ARMS = ('F', 'C_raw', 'C_match', 'S_match')
STOP = False


def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n'); tmp.replace(path)


def journal(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', buffering=1) as f: f.write(json.dumps(obj, allow_nan=False)+'\n')


def cases():
    p = d.read(PLAN)
    return [(m,q,'train') for m in p['data']['training_catalog'] for q in p['data']['training_queries']] + [(m,q,'development') for m in p['data']['development_catalog'] for q in p['data']['development_queries']]


def diskbytes(root): return sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0


def resources(previous=None, force=True):
    # Already materialized bytes are credited; concurrent atomic temporary writes are included.
    import shutil
    data_used=diskbytes(CACHE); project_used=diskbytes(OUT)
    free=shutil.disk_usage(CACHE if CACHE.exists() else CACHE.parent).free
    project_free=shutil.disk_usage(d.ROOT).free
    remaining=max(0, 3.4*2**30-data_used)+2*16*2**20
    h=free-remaining; reasons=[]
    if data_used>=4*2**30: reasons.append('data_hard_cap')
    if project_used>=2**30: reasons.append('project_hard_cap')
    if h<6*2**30 or free<5*2**30: reasons.append('data_headroom')
    if project_free<8*2**30: reasons.append('project_headroom')
    drop=0
    if previous:
        drop=max(0, previous['free_bytes']-free-max(0,data_used-previous['data_used_bytes']))
        if h<8*2**30 and drop>=.5*2**30: reasons.append('unexplained_drop')
    row=dict(utc=now(),epoch=time.time(),data_used_bytes=data_used,project_artifact_bytes=project_used,free_bytes=free,project_free_bytes=project_free,unmaterialized_and_transient_bytes=remaining,projected_headroom_bytes=h,unexplained_drop_bytes=drop,stop_reasons=reasons)
    journal(OUT/'resources/ledger.jsonl',row)
    return row


def check_stop():
    if STOP: raise SystemExit(75)


def charge(kind, purpose):
    caps={'native':4272,'rasterization':600}
    with (OUT/'calls.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        path=OUT/'calls.json';r=d.read(path) if path.exists() else dict(native=0,rasterization=0,by_purpose={})
        assert r[kind]<caps[kind],(kind,'hard budget')
        if kind=='native':
            cap={'formal':4224,'preflight':16,'recovery':32}[purpose]
            assert r['by_purpose'].get(purpose,0)<cap
        r[kind]+=1;r['by_purpose'][purpose]=r['by_purpose'].get(purpose,0)+1
        write(path,r)


def raw_path(rid,q): return CACHE/'rendered'/f'{rid}_{q}.pt'


def geometry(field):
    """CPU FP64 self+31 all-center scatter; no validity filtering or epsilon."""
    p=field['means'].numpy().astype(np.float64); s=field['scales'].numpy().astype(np.float64)
    assert np.isfinite(s).all() and (s>0).all()
    quat=field['rotations'].numpy().astype(np.float64); rot=Rotation.from_quat(quat[:,[1,2,3,0]]).as_matrix()
    cov=(rot*(s*s)[:,None,:])@rot.transpose(0,2,1)
    n=len(p);dist=((p[:,None]-p[None])**2).sum(-1);np.fill_diagonal(dist,np.inf)
    neighbors=np.concatenate((np.arange(n)[:,None],np.argsort(dist,axis=1,kind='stable')[:,:min(31,n-1)]),1)
    v=p[neighbors];v=v-v.mean(1,keepdims=True);B=np.einsum('nki,nkj->nij',v,v)/neighbors.shape[1]
    eig,U=np.linalg.eigh(B);delta=eig-eig[:,:1];trD=delta.sum(-1);t=(s*s).sum(-1);m=(s*s).min(-1)
    nonzero=trD>0; vals=np.broadcast_to(t[:,None]/3,(n,3)).copy()
    vals[nonzero]=m[nonzero,None]+(t[nonzero]-3*m[nonzero])[:,None]*delta[nonzero]/trD[nonzero,None]
    assert (vals>0).all()
    C=(U*vals[:,None,:])@U.transpose(0,2,1)
    # A right-handed basis is required for a quaternion; sign changes preserve covariance.
    U=U.copy();U[np.linalg.det(U)<0,:,2]*=-1
    cq=Rotation.from_matrix(U).as_quat()[:,[3,0,1,2]]
    cs=np.sqrt(vals);ss=np.sqrt(t/3)[:,None]*np.ones((n,3));sq=np.tile([1.,0.,0.,0.],(n,1))
    result={'C':dict(scales=torch.from_numpy(cs).float(),rotations=torch.from_numpy(cq).float()),'S':dict(scales=torch.from_numpy(ss).float(),rotations=torch.from_numpy(sq).float())}
    errors={}
    for name,entry in result.items():
        sc=entry['scales'].numpy().astype(np.float64);rr=Rotation.from_quat(entry['rotations'].numpy()[:,[1,2,3,0]]).as_matrix()
        actual=(rr*(sc*sc)[:,None,:])@rr.transpose(0,2,1)
        errors[name]=dict(trace_max_relative=float(np.max(abs(np.trace(actual,axis1=1,axis2=2)-t)/t)),min_axis_max_relative=float(np.max(abs(np.linalg.eigvalsh(actual)[nonzero,0]-m[nonzero])/m[nonzero])) if name=='C' and nonzero.any() else None,covariance_roundtrip_max_relative=float(np.max(np.linalg.norm(actual-(C if name=='C' else np.eye(3)[None]*t[:,None,None]/3),axis=(1,2))/t)))
        assert errors[name]['trace_max_relative']<2e-6
        if name=='C' and nonzero.any(): assert errors[name]['min_axis_max_relative']<2e-5
    stats=dict(N=n,K=neighbors.shape[1],valid_content=int(field['has_content'].sum()),degenerate_count=int((~nonzero).sum()),degenerate_indices=np.flatnonzero(~nonzero).tolist(),trace_quantiles=np.quantile(t,[0,.25,.5,.75,1]).tolist(),min_variance_quantiles=np.quantile(m,[0,.25,.5,.75,1]).tolist(),neighbor_distance_quantiles=np.quantile(np.sqrt(np.take_along_axis(dist,neighbors[:,1:],axis=1)),[0,.5,.9,1]).tolist(),invariant_errors=errors,all_centers_used=True,original_trace_minimum_privilege=True)
    arrays=dict(neighbors=neighbors,B=B,C=C,F=cov,trace=t,minimum=m,trD=trD,center_scales=cs,center_quat=cq,sphere_scales=ss)
    return result,stats,arrays


def fallback(field,query):
    p=field['means'].double();cam=query['c2w'].double();K=query['K'].double().clone();K[:2]/=14
    camera=(p-cam[:3,3])@cam[:3,:3];valid=field['has_content'].bool();take=valid&(camera[:,2]>0)
    yy,xx=torch.meshgrid(torch.arange(32,dtype=torch.float64)+.5,torch.arange(32,dtype=torch.float64)+.5,indexing='ij');patch=torch.stack((xx,yy),-1).reshape(-1,2)
    indices=torch.nonzero(take).flatten()
    if len(indices):
        pix=camera[indices]@K.T;pix=pix[:,:2]/pix[:,2:]
        nearest=((patch[:,None]-pix[None])**2).sum(-1).argmin(-1)
        ids=indices[nearest];mode='nearest_positive_depth_valid_center'
    else: ids=torch.full((1024,),-1,dtype=torch.int64);mode='valid_mean' if valid.any() else 'zero'
    return ids,mode


def make_inputs(field,raw):
    f=field['features'].float();valid=field['has_content'];ids=raw['fallback_indices'];mode=raw['fallback_mode']
    if mode=='nearest_positive_depth_valid_center': base=f[ids]
    elif mode=='valid_mean': base=f[valid].mean(0)[None].expand(1024,-1)
    else: base=torch.zeros((1024,1024))
    norms={};ns={}
    for name in ('C','S'):
        a=raw['a_'+name].reshape(-1,1);r=raw['r_'+name];nz=a[:,0]>0
        normalized=base.clone();normalized[nz]=(r[nz].double()/a[nz].double()).float()
        ns[name]=normalized
    aF=raw['a_F'].reshape(-1,1)
    out=dict(F=raw['r_F'],C_raw=raw['r_C'],C_match=aF*ns['C'],S_match=aF*ns['S'])
    for name in ('C_match','S_match'): out[name][aF[:,0]==0]=0
    assert all(x.dtype==torch.float32 and torch.isfinite(x).all() for x in out.values())
    return out


def lock():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'configuration_lock.json').exists(): return verify()
    p=d.read(PLAN)
    expected={'analysis_handoff.json':'88d3fd0ffddb58a72a49c3f2e1af46306a8842ed5beeb0709333f5f2d66c9818','plan.json':'d2391ea222d606a56a6c6490633fea4a5e4c171f754b5771e700358da21f659f'}
    for name,h in expected.items(): assert d.sha256(PLAN.parent/name)==h
    assert d.sha256(d.ROOT/'docs/native_covariance_role_execution_plan.md')=='8111d5a7b6dddff8a2ee619788347edd59d808941002c270c601a43eb5afb7ca'
    old=d.read(d.ROOT/'results/gaussian_memory/native_render_factorization1/configuration_lock.json')
    paths={}
    for meta,q,_ in cases():
        for path in [d.scene_path(meta['id']),d.query_path(meta['id'],q),d.feature_path(meta['id'],q)]+[d.patch_path(meta['id'],q,c) for c in d.CONDITIONS]:
            k=str(path);st=path.stat();prior=old['input_stats'][k]
            assert st.st_size==prior['size'] and st.st_mtime_ns==prior['mtime_ns'],k
            paths[k]=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns,sha256_inherited=old['input_sha256'][k])
    models=[]
    for entry in p['source_and_model_identity']['models']:
        assert d.sha256(entry['status_path'])==entry['status_sha256']
        status=d.read(entry['status_path']);assert status['complete'] and status['checkpoint_sha256']==entry['checkpoint_sha256_from_completed_parent']
        st=Path(entry['checkpoint_path']).stat();assert st.st_size==entry['checkpoint_bytes']
        models.append(dict(entry,bytes=st.st_size,mtime_ns=st.st_mtime_ns,hash_policy='authoritative completed-parent SHA + path/size, no large rehash; explicit implementation dispatch'))
    inherited=p['source_and_model_identity']['inherited_predictions']
    for r in inherited:
        assert d.sha256(r['marker'])==r['marker_sha256']
        old_record=d.read(r['marker']);assert old_record['complete'] and old_record['prediction_sha256']==r['prediction_sha256']
        assert Path(r['prediction_path']).exists()
    sources=['experiments/native_covariance_role.py','experiments/native_covariance_role_launch.py','experiments/native_covariance_role_score.py','experiments/native_covariance_role_delivery.py','experiments/native_coverage_data.py','experiments/native_receiver_runtime.py','experiments/native_receiver_data.py','experiments/gaussian_memory_native_run.py','models/gaussian_scene_native.py','models/gaussian_scene_rendered.py','models/gaussian_scene_memory.py','experiments/gaussian_scene_score.py']
    value=dict(complete=True,created_utc=now(),plan_sha256=d.sha256(PLAN),sources={x:d.sha256(d.ROOT/x) for x in sources},inputs=paths,models=models,inherited=inherited,model_config=dict(thread_uuid=os.environ.get('CODEX_THREAD_ID'),requested_model='gpt-6.1-sol',requested_effort='high',requested_service_tier='fast',parent_observed_client_selected=dict(model='gpt-6.1-sol',effort='high',approvals_reviewer='auto_review'),observation_source='parent dispatch, not independently re-read by role',backend_effective_model=None,backend_effective_effort=None,backend_service_tier=None),prior_authority='native_render_factorization1/configuration_lock.json',scientific_plan_modified=False)
    write(OUT/'configuration_lock.json',value);return value


def verify():
    r=d.read(OUT/'configuration_lock.json');assert d.sha256(PLAN)==r['plan_sha256']
    for name,h in r['sources'].items(): assert d.sha256(d.ROOT/name)==h,name
    for path,info in r['inputs'].items():
        st=Path(path).stat();assert st.st_size==info['bytes'] and st.st_mtime_ns==info['mtime_ns'],path
    for m in r['models']:
        st=Path(m['checkpoint_path']).stat();assert st.st_size==m['bytes'] and st.st_mtime_ns==m['mtime_ns']
    return r


@torch.no_grad()
def prepare():
    from models.gaussian_scene_memory import rasterize_field
    verify();torch.set_num_threads(4)
    for meta in d.read(PLAN)['data']['training_catalog']+d.read(PLAN)['data']['development_catalog']:
        check_stop();rid=meta['id'];field=d.load(d.scene_path(rid));start=time.monotonic();g,stats,arrays=geometry(field)
        gp=CACHE/'geometry'/f'{rid}.npz';gp.parent.mkdir(parents=True,exist_ok=True)
        space=resources();assert not space['stop_reasons'],space
        tmp=gp.with_suffix('.tmp')
        with tmp.open('wb') as ff: np.savez(ff,**arrays)
        tmp.replace(gp)
        write(OUT/'geometry'/f'{rid}.json',dict(complete=True,id=rid,statistics=stats,geometry_path=str(gp),sha256=d.sha256(gp),seconds=time.monotonic()-start))
        fieldgpu={k:v.cuda() if torch.is_tensor(v) else v for k,v in field.items()}
        fields=dict(F=fieldgpu,**{k:dict(fieldgpu,**{kk:vv.cuda() for kk,vv in v.items()}) for k,v in g.items()})
        colors=torch.cat((fieldgpu['features'].float()*fieldgpu['has_content'][:,None],fieldgpu['has_content'].float()[:,None]),-1)
        for _,q,cohort in [case for case in cases() if case[0]['id']==rid]:
            check_stop();marker=OUT/'preparation'/f'{rid}_{q}.json';path=raw_path(rid,q)
            if marker.exists():
                r=d.read(marker);assert r['complete'] and d.sha256(path)==r['sha256'];continue
            start=time.monotonic();query=d.load(d.query_path(rid,q));K=query['K'].cuda().clone();K[:2]/=14;camera=query['c2w'].cuda()
            ids,mode=fallback(field,query);raw=dict(fallback_indices=ids,fallback_mode=mode);seconds={}
            for name in ('F','C','S'):
                charge('rasterization','render_prepare');t=time.monotonic()
                rendered,alpha,info=rasterize_field(fields[name],camera[None],K[None],32,32,colors)
                torch.cuda.synchronize();seconds[name]=time.monotonic()-t
                raw['r_'+name]=rendered[0,:,:,:1024].reshape(1024,1024).float().cpu().contiguous();raw['a_'+name]=rendered[0,:,:,1024].reshape(1024).float().cpu().contiguous()
                assert (raw['a_'+name]>=0).all() and float(raw['a_'+name].max())<=1.0001
                del rendered,alpha,info
            inputs=make_inputs(field,raw);refnorm=float(raw['r_F'].double().norm());af=raw['a_F'].double();mass=float(af.sum());details={}
            for name in ('F','C','S'):
                aa=raw['a_'+name];zero=aa==0
                details[name]=dict(alpha_quantiles=torch.quantile(aa.double(),torch.tensor([0,.25,.5,.75,.9,.99,1.],dtype=torch.float64)).tolist(),alpha_zero_fraction=float(zero.double().mean()),fallback_pixels=int(zero.sum()),factual_alpha_mass_at_fallback=float(af[zero].sum()),factual_alpha_mass_fraction_at_fallback=float(af[zero].sum())/mass if mass else None)
            old=d.load(d.feature_path(rid,q))
            inpstats=dict(id=rid,query=q,cohort=cohort,effective_alpha=details,fallback_mode=mode,factual_alpha_total=mass,input_norm={k:float(v.double().norm()) for k,v in inputs.items()},norm_ratio={k:float(v.double().norm())/refnorm if refnorm else None for k,v in inputs.items()},relative_difference_from_F={k:float((v-raw['r_F']).double().norm())/refnorm if refnorm else None for k,v in inputs.items()},F_vs_old_cached_content_max_abs=float((raw['r_F']-old['render']).abs().max()),F_vs_old_cached_alpha_max_abs=float((raw['a_F']-old['effective_alpha'].reshape(-1)).abs().max()),render_seconds=seconds)
            path.parent.mkdir(parents=True,exist_ok=True);space=resources();assert not space['stop_reasons'],space
            tmp=path.with_suffix('.tmp');torch.save(raw,tmp);tmp.replace(path)
            write(marker,dict(complete=True,id=rid,query=q,path=str(path),sha256=d.sha256(path),bytes=path.stat().st_size,seconds=time.monotonic()-start,lock_sha256=d.sha256(OUT/'configuration_lock.json')))
            write(OUT/'input_statistics'/f'{rid}_{q}.json',inpstats)
            if rid in ('fit_00','confirm_12') and q==1:
                checks=[]
                for name in ('C','S'):
                    a=raw['a_'+name].numpy();nz=a>0;n=raw['r_'+name].numpy()[nz].astype(np.float64)/a[nz,None].astype(np.float64)
                    expected=raw['a_F'].numpy()[nz,None]*n.astype(np.float32)
                    err=float(np.max(abs(expected-inputs['C_match' if name=='C' else 'S_match'].numpy()[nz]))) if nz.any() else 0.
                    assert err==0;checks.append(dict(geometry=name,numpy_FP64_normalization_match_max_abs=err))
                write(OUT/'checks'/f'{rid}_input.json',dict(complete=True,checks=checks,geometry_invariants=stats['invariant_errors'],effective_not_total_alpha=True,colors='features*has_content,has_content',projection_fallback_ids=ids.tolist(),factual_zero_matched_inputs_zero=all(bool((inputs[k][raw['a_F']==0]==0).all()) for k in ('C_match','S_match'))))
            del raw,inputs,old
        del fields,fieldgpu,colors;print(dict(prepared=rid,seconds=time.monotonic()-start),flush=True)
    assert len(list((OUT/'preparation').glob('*.json')))==176
    write(OUT/'preparation_complete.json',dict(complete=True,queries=176,geometries=528,calls=d.read(OUT/'calls.json')))


def inference(seeds,preflight=False):
    from experiments.native_receiver_data import core_load
    from experiments.native_receiver_runtime import model_load,forward_read
    from experiments.gaussian_scene_preflight import metric_depth
    from models.gaussian_scene_rendered import RenderedFeatureFusion
    verify();torch.set_num_threads(4);runid=f'{os.getpid()}';log=OUT/'inference'/f'{runid}.jsonl'
    t=time.monotonic();model=model_load(d.plan());reader=RenderedFeatureFusion().cuda().eval();model.eval().requires_grad_(False);reader.requires_grad_(False)
    journal(log,dict(event='model_load',seconds=time.monotonic()-t,utc=now()))
    for seed in seeds:
        check_stop();t=time.monotonic();state=d.load(d.checkpoint_path('Render',seed));assert state['step']==2000
        core_load(model,state['native']);reader.load_state_dict(state['model']);del state
        journal(log,dict(event='checkpoint_load',seed=seed,seconds=time.monotonic()-t,utc=now()))
        selected=[v for v in cases() if v[0]['id'] in ('fit_00','confirm_12') and v[1]==1] if preflight else cases()
        for meta,q,cohort in selected:
            check_stop();rid=meta['id'];raw=d.load(raw_path(rid,q));field=d.load(d.scene_path(rid));extra=make_inputs(field,raw);del raw,field
            for condition in (('missing',) if preflight else d.CONDITIONS):
                patches=d.load(d.patch_path(rid,q,condition))['patches'].cuda()
                for arm in ARMS:
                    key=f'{seed}_{rid}_{q}_{condition}_{arm}';marker=OUT/('preflight' if preflight else 'completed')/f'{key}.json';path=CACHE/'predictions'/f'{key}.npy'
                    if marker.exists():
                        old=d.read(marker);assert old['complete'];assert preflight or (path.stat().st_size==old['bytes'] and d.sha256(path)==old['sha256']);continue
                    check_stop();charge('native','preflight' if preflight else 'formal');t=time.monotonic()
                    sample=dict(patches=patches,extra=extra[arm].cuda(),hw=(448,448))
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):pred=metric_depth(forward_read(model,sample,reader,'Render',checkpoint_blocks=False)).float().cpu()
                    torch.cuda.synchronize();seconds=time.monotonic()-t
                    assert pred.shape==(210,210) and torch.isfinite(pred).all() and (pred>0).all()
                    rec=dict(complete=True,seed=seed,id=rid,query=q,condition=condition,arm=arm,cohort=cohort,place=meta['place'],inference_seconds=seconds,peak_GiB=torch.cuda.max_memory_allocated()/2**30,lock_sha256=d.sha256(OUT/'configuration_lock.json'),input_sha256=d.read(OUT/'preparation'/f'{rid}_{q}.json')['sha256'])
                    if not preflight:
                        path.parent.mkdir(exist_ok=True);tmp=path.with_suffix('.tmp')
                        with tmp.open('wb') as f:np.save(f,pred.numpy(),allow_pickle=False)
                        tmp.replace(path);rec.update(prediction_path=str(path),sha256=d.sha256(path),bytes=path.stat().st_size)
                    else: rec.update(prediction_min=float(pred.min()),prediction_max=float(pred.max()))
                    write(marker,rec);journal(log,dict(event='forward_complete',key=key,seconds=seconds,preflight=preflight,utc=now()))
                    del sample,pred
                del patches
            del extra
            if q in (10,11):print(dict(seed=seed,completed_case=rid,preflight=preflight),flush=True)
    journal(log,dict(event='complete',utc=now(),all_parameters_frozen=all(not p.requires_grad for p in model.parameters()) and all(not p.requires_grad for p in reader.parameters())))


def main():
    global STOP
    def stop(*_):
        global STOP
        STOP=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    p=argparse.ArgumentParser();p.add_argument('command',choices=['lock','prepare','preflight','run']);p.add_argument('--seeds',type=int,nargs='+',default=[20261110]);a=p.parse_args()
    if a.command=='lock':lock()
    elif a.command=='prepare':prepare()
    else:inference(a.seeds,a.command=='preflight')


if __name__=='__main__':main()
