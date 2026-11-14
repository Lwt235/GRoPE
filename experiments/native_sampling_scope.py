"""Round021 metadata labels first, then fixed CPU-only joins of old query scores."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('MKL_THREADING_LAYER','GNU')
import argparse,collections,csv,datetime,fcntl,hashlib,json,resource,shutil,sys,time
from pathlib import Path
import numpy as np
import torch

ROOT=Path('/home/liwt/GRoPE');OUT=ROOT/'results/gaussian_memory/native_sampling_scope1'
BASE=ROOT/'results/gaussian_memory/native_feature_field_pe1';J=BASE/'J'
SCIENCE=ROOT/'results/gaussian_memory/autonomous_research_loop/round_021/science_plan.md'
SCIENCE_SHA='da48ea1a0d3d598145877de2b21139e1ddf2961ec4be5ffc2c9ac2f034f3f52b'
CONTRACT=ROOT/'results/gaussian_memory/autonomous_research_loop/round_006/data_contract.json'
QUERY_MANIFEST=ROOT/'results/gaussian_memory/camera_only_feature_pe1/input_manifest.json'
SEEDS=(20261110,20261111,20261112);METHODS=('P','S','G','Native','RasterFill')
PAIRS=[('G','P'),('S','P'),('G','S')]+[(a,b) for a in ('P','S','G') for b in ('Native','RasterFill')]
METRICS=('centered_log_mse','log_mse','log_bias','bias_squared','absrel','rmse_m')
START=time.monotonic();WALL_START=time.time();CPU_START=time.process_time()

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def finite(x):
    if isinstance(x,dict):return {str(k):finite(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [finite(v) for v in x]
    if isinstance(x,np.ndarray):return finite(x.tolist())
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    if isinstance(x,np.integer):return int(x)
    return x
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:json.dump(finite(obj),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(p)
def metaidentity(p):
    p=Path(p);s=p.stat();return dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
def ident(p):return dict(**metaidentity(p),sha256=sha(p))
def guards():
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    size=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) if OUT.exists() else 0
    assert time.monotonic()-START<3600 and rss<16*2**30 and size+8*2**20<.25*2**30
    assert not torch.cuda.is_initialized()
    return dict(utc=now(),pid=os.getpid(),elapsed_seconds=time.monotonic()-START,CPU_seconds=time.process_time()-CPU_START,peak_RSS_bytes=rss,new_storage_bytes=size,CUDA_initialized=False)
def loadmetadata(p):return torch.load(p,map_location='cpu',weights_only=True,mmap=True)
def metadatahash(*arrays):
    h=hashlib.sha256()
    for a in arrays:
        a=np.asarray(a);h.update(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes(order='C'))
    return h.hexdigest()

def project(means,cam,K):
    p=(means-cam[:3,3])@cam[:3,:3];z=p[:,2]
    assert np.isfinite(p).all() and K[0,0]>0 and K[1,1]>0 and np.array_equal(K[2],[0,0,1])
    xy=np.full((len(p),2),np.nan);positive=z>0;xy[positive]=(p[positive]@K.T)[:,:2]/z[positive,None]
    nu=np.zeros(len(p));nu[positive]=np.sqrt(K[0,0]*K[1,1])/z[positive]
    return xy,nu
def label_query(means,src_cam,src_K,query_cam,query_K,window,hw=(448,448)):
    src=np.zeros(len(means))
    for cam,K in zip(src_cam,src_K):
        xy,nu=project(means,cam,K);inside=(nu>0)&(xy[:,0]>=0)&(xy[:,0]<hw[1])&(xy[:,1]>=0)&(xy[:,1]<hw[0]);src=np.maximum(src,np.where(inside,nu,0))
    xy,nu=project(means,query_cam,query_K);top,bottom,left,right=window
    inwindow=(nu>0)&(xy[:,0]>=left)&(xy[:,0]<right)&(xy[:,1]>=top)&(xy[:,1]<bottom);eligible=inwindow&(src>0)
    values=np.log2(src[eligible]/nu[eligible]);r=float(np.median(values)) if values.size else None
    return dict(label='U' if r is None else 'H' if r>=1 else 'L',r_log2_source_over_query=r,valid_content_slots=len(means),source_frustum_slots=int((src>0).sum()),query_window_slots=int(inwindow.sum()),eligible_slots=int(eligible.sum()),ratio_log2_min=float(values.min()) if values.size else None,ratio_log2_max=float(values.max()) if values.size else None)
def validation():
    cam=np.eye(4);K=np.array([[100.,0,224],[0,100.,224],[0,0,1]])
    p=np.array([[0.,0,2.]])
    h=label_query(p,cam[None],(K*np.array([[2,1,1],[1,2,1],[1,1,1]]))[None],cam,K,[112,336,112,336])
    l=label_query(p,cam[None],K[None],cam,K,[112,336,112,336])
    outside=label_query(np.array([[4.,0,2.]]),cam[None],K[None],cam,K,[112,336,112,336])
    behind=label_query(-p,cam[None],K[None],cam,K,[112,336,112,336])
    assert h['label']=='H' and h['r_log2_source_over_query']==1 and l['label']=='L' and l['r_log2_source_over_query']==0 and outside['label']==behind['label']=='U'
    return dict(H_boundary=h,L_unit_ratio=l,U_no_window_center=outside,U_nonpositive_depth=behind,scope='Four analytic CPU geometry examples only; no threshold tuning or model calls.')

def labels():
    commit=OUT/'labels_complete.json'
    if commit.exists():
        c=read(commit);assert c['science_sha256']==SCIENCE_SHA and all(sha(x['path'])==x['sha256'] for x in c['artifacts']);return c
    c=read(CONTRACT);dev=c['development_catalog'];assert len(dev)==12 and len(c['fit_queries'])==len(c['other_queries'])==4
    original=read(ROOT/'results/gaussian_memory/gaussian_scene_memory1/input_plan.json');window=original['missing_rectangle_image'];assert window==[112,336,112,336]
    manifest=read(QUERY_MANIFEST)['inputs'];records=[];inputs=[];start=now()
    for m in dev:
        rid=m['id'];fm=read(J/'formation'/f'{rid}.json');assert fm['complete'] and fm['row']=='J';field=loadmetadata(fm['field_path'])
        valid=field['has_content'].numpy();means=field['means'].double().numpy()[valid];cams=field['source_c2w'].double().numpy();ks=field['source_intrinsic'].double().numpy();assert len(cams)==4 and list(field['source_ids'])==[0,3,6,9] and np.isfinite(means).all()
        inputs.append(dict(**metaidentity(fm['field_path']),published_field_sha256=fm['field_sha256'],authority=ident(J/'formation'/f'{rid}.json'),metadata_sha256=metadatahash(field['means'].numpy(),valid,cams,ks),consumed_fields=['means','has_content','source_c2w','source_intrinsic','source_ids'],no_Sigma_RGB_or_target_access=True))
        for q in c['evaluation_queries']:
            matches=[p for p in manifest if p.endswith(f'/{rid}/query_{q}.pt')];assert len(matches)==1;path=matches[0];query=loadmetadata(path);cam=query['c2w'].double().numpy();K=query['K'].double().numpy();hw=tuple(query['image_missing'].shape);assert hw==(448,448)
            rec=label_query(means,cams,ks,cam,K,window,hw);records.append(dict(id=rid,place=m['place'],query=q,query_group='fit' if q in c['fit_queries'] else 'other',**rec))
            inputs.append(dict(**metaidentity(path),published_sha256=manifest[path]['sha256_inherited'],authority=str(QUERY_MANIFEST),metadata_sha256=metadatahash(cam,K,np.array(hw),np.array(window)),consumed_fields=['c2w','K','image_missing.shape'],RGB_target_valid_error_not_accessed=True));del query
        del field;guards()
    assert len(records)==96 and len({(r['id'],r['query']) for r in records})==96
    counts={}
    for qg in ('fit','other'):
        rr=[r for r in records if r['query_group']==qg];counts[qg]=dict(queries=48,labels={l:sum(r['label']==l for r in rr) for l in ('H','L','U')},per_place=[dict(id=m['id'],place=m['place'],**{l:sum(r['id']==m['id'] and r['label']==l for r in rr) for l in ('H','L','U')}) for m in dev])
    payload=dict(science_sha256=SCIENCE_SHA,complete=True,threshold_octaves=1,window_image_halfopen=window,image_hw=[448,448],scope='All12fixed development places x8queries=96. No training subset or model/threshold scan.',formula='median of log2(max positive source-frustum sqrt(fx*fy)/z divided by query sqrt(fx*fy)/z) over valid centers in fixed query missing window and with a source rate; no qualified center ->U.',records=records,score_files_read_during_label_stage=0,consumed_identity_policy='Published original whole-file SHA referenced; only actually consumed center/mask/camera metadata independently hashed, no old large weights or prediction reads.')
    write(OUT/'sampling_labels.json',payload);write(OUT/'sampling_counts.json',counts);write(OUT/'label_inputs.json',dict(inputs=inputs,contract=ident(CONTRACT),query_manifest=ident(QUERY_MANIFEST),no_new_field_copies=True));write(OUT/'definition_validation.json',validation())
    with (OUT/'sampling_labels.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records);f.flush();os.fsync(f.fileno())
    files=[ident(OUT/n) for n in ('sampling_labels.json','sampling_counts.json','sampling_labels.csv','label_inputs.json','definition_validation.json')]
    result=dict(complete=True,science_sha256=SCIENCE_SHA,started_utc=start,labels_saved_before_any_score_join=True,score_reads=0,labels=96,artifacts=files,completed_utc=now());write(commit,result)
    for x in files:os.chmod(x['path'],0o444)
    print(dict(labels_complete=True,counts=counts),flush=True);return result

def stats(a,b,places,signed=False):
    good=np.isfinite(a).all(0)&np.isfinite(b).all(0);a=a[:,good];b=b[:,good];places=[p for p,g in zip(places,good) if g];n=len(places)
    result=dict(places=places,place_count=n,paired_seed_place_cells=3*n,interval_estimable=n>=2,interval_unavailable_reason=None if n>=2 else 'Fewer than two distinct places; seed resampling is not cross-place support.')
    if not n:return dict(result,test=None,reference=None,reduction_percent=None,interval95_percent=None,absolute=None,absolute_interval95=None,per_seed_percent=[None]*3,per_place_percent=[],per_seed_absolute=[None]*3,per_place_absolute=[])
    ma=float(a.mean());mb=float(b.mean());delta=a-b;result.update(test=ma,reference=mb,absolute=ma-mb,per_seed_absolute=delta.mean(1).tolist(),per_place_absolute=delta.mean(0).tolist(),per_seed_percent=None if signed else (100*(1-a.mean(1)/b.mean(1))).tolist(),per_place_percent=None if signed else (100*(1-a.mean(0)/b.mean(0))).tolist(),reduction_percent=None if signed or not mb else 100*(1-ma/mb),interval95_percent=None,absolute_interval95=None)
    if n>=2:
        rng=np.random.default_rng(20261113);ss=rng.integers(0,3,(5000,3,1));pp=rng.integers(0,n,(5000,1,n));aa=a[ss,pp].mean((1,2));bb=b[ss,pp].mean((1,2));result['absolute_interval95']=np.percentile(aa-bb,[2.5,97.5]).tolist()
        if not signed and mb>0 and (bb>0).all():result['interval95_percent']=np.percentile(100*(1-aa/bb),[2.5,97.5]).tolist()
    return finite(result)
def practical(v):return v['reduction_percent'] is not None and v['reduction_percent']>=1 and v['interval95_percent'] is not None and v['interval95_percent'][0]>0 and all(x is not None and x>0 for x in v['per_seed_percent'])
def harm(v):return v['reduction_percent'] is not None and v['reduction_percent']<=-1 and v['interval95_percent'] is not None and v['interval95_percent'][1]<0

def aggregate():
    commit=read(OUT/'labels_complete.json');assert commit['complete'] and all(sha(x['path'])==x['sha256'] for x in commit['artifacts']);labels_=read(OUT/'sampling_labels.json')['records'];lookup={(r['id'],r['query']):r for r in labels_};c=read(CONTRACT);dev=c['development_catalog'];dev_ids={m['id'] for m in dev};join_start=now()
    assert join_start>commit['completed_utc'];metricpath=J/'scoring/metrics.json';old=read(metricpath);selected=[];full=[]
    for row in old['native']:
        if row['id'] not in dev_ids or row['permuted']:continue
        if row['kind']=='Native_full':full.append(row);continue
        if row['kind'] not in METHODS:continue
        r=dict(row,sampling_label=lookup[row['id'],row['query']]['label']);assert r['condition']=='missing';selected.append(r)
    assert len(selected)==1056 and len(full)==96 and collections.Counter(r['kind'] for r in selected)==dict(P=288,S=288,G=288,Native=96,RasterFill=96)
    assert len({(r['kind'],r['seed'],r['id'],r['query']) for r in selected})==1056
    coverage={}
    for m in dev:
        for r in read(J/'preparation'/f'{m["id"]}.json')['coverage']:coverage[m['id'],r['query']]=r
    for r in selected:r.update(original_zero_all=coverage[r['id'],r['query']]['old_zero_all'],J_zero_all=coverage[r['id'],r['query']]['new_zero_all'])
    write(OUT/'joined_scores.json',dict(labels_commit=ident(OUT/'labels_complete.json'),labels=ident(OUT/'sampling_labels.json'),score_source=ident(metricpath),join_started_utc=join_start,missing_rows=selected,shared_Native_full_reference=full,unmodified_per_query_metrics=True,no_old_pixel_metric_recomputation=True))
    with (OUT/'joined_scores.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['method','seed','id','place','query','query_group','sampling_label','original_zero_all','J_zero_all','region','pixels',*METRICS,'prediction_path','published_prediction_sha256'])
        for r in selected:
            for reg,v in r['metrics'].items():w.writerow([r.get(k) for k in ('kind','seed','id','place','query','query_group','sampling_label','original_zero_all','J_zero_all')]+[reg,v['pixels'],*[v[k] for k in METRICS],r['path'],r['sha256']])
    comparisons={};summaries={};arrays={};regions=sorted(set.intersection(*[set(r['metrics']) for r in selected]));counterexamples={}
    for qg in ('fit','other'):
        for label in ('all','H','L','U'):
            rr=[r for r in selected if r['query_group']==qg and (label=='all' or r['sampling_label']==label)];metas=[m for m in dev if any(r['id']==m['id'] for r in rr)];places=[m['place'] for m in metas]
            cells={}
            for r in rr:cells.setdefault((r['kind'],r['seed'],r['id']),[]).append(r)
            for reg in regions:
                for metric in METRICS:
                    aa={};prefix=f'{qg}/{label}/{reg}/{metric}'
                    for method in METHODS:
                        a=np.full((3,len(metas)),np.nan)
                        for si,seed in enumerate(SEEDS):
                            for pi,m in enumerate(metas):
                                vals=[r['metrics'][reg][metric] for r in cells.get((method,None if method in ('Native','RasterFill') else seed,m['id']),[]) if r['metrics'][reg][metric] is not None]
                                if vals:a[si,pi]=np.mean(vals)
                        aa[method]=a;summaries[prefix+'/'+method]=dict(places=places,per_seed_place=a.tolist(),per_place=[float(np.nanmean(v)) if np.isfinite(v).any() else None for v in a.T],per_seed=[float(np.nanmean(v)) if np.isfinite(v).any() else None for v in a],mean=float(np.nanmean(a)) if np.isfinite(a).any() else None)
                    if reg=='roi' and metric=='centered_log_mse':arrays[qg,label]=dict(metas=metas,values=aa)
                    for a,b in PAIRS:comparisons[prefix+'/'+a+'-'+b]=stats(aa[a],aa[b],places,metric=='log_bias')
            for reg,metric in [('roi','centered_log_mse'),('whole','absrel')]:
                for a,b in PAIRS:
                    examples=[]
                    for lab in labels_:
                        if lab['query_group']!=qg or (label!='all' and lab['label']!=label):continue
                        test=[r['metrics'][reg][metric] for r in rr if (r['id'],r['query'],r['kind'])==(lab['id'],lab['query'],a)];ref=[r['metrics'][reg][metric] for r in rr if (r['id'],r['query'],r['kind'])==(lab['id'],lab['query'],b)];assert len(test)==3 and len(ref)==(1 if b in ('Native','RasterFill') else 3)
                        x,y=float(np.mean(test)),float(np.mean(ref));examples.append(dict(id=lab['id'],place=lab['place'],query=lab['query'],sampling_label=lab['label'],test=x,reference=y,test_minus_reference=x-y,reduction_percent=100*(1-x/y),test_per_seed=test,reference_per_seed=ref))
                    counterexamples[f'{qg}/{label}/{reg}/{metric}/{a}-{b}']=dict(worst_absolute=max(examples,key=lambda x:x['test_minus_reference']) if examples else None,worst_relative=min(examples,key=lambda x:x['reduction_percent']) if examples else None,all_queries=examples)
            guards()
    interactions={}
    for qg in ('fit','other'):
        h,l=arrays[qg,'H'],arrays[qg,'L'];common=[m for m in dev if m in h['metas'] and m in l['metas']];ah={k:v[:,[h['metas'].index(m) for m in common]] for k,v in h['values'].items()};al={k:v[:,[l['metas'].index(m) for m in common]] for k,v in l['values'].items()};n=len(common)
        def effect(g,p):return 100*(1-g.mean()/p.mean())
        result=dict(places=[m['place'] for m in common],place_count=n,interval_estimable=n>=2,estimate_percentage_points=None,interval95_percentage_points=None,per_seed_percentage_points=None,per_place_percentage_points=None,scope='Direct H-minusL G/P relative reduction on only places with both conditions; identical paired seed/place resampling of H and L, not difference of significance flags.')
        if n:
            gh,ph,gl,pl=ah['G'],ah['P'],al['G'],al['P'];result.update(estimate_percentage_points=effect(gh,ph)-effect(gl,pl),per_seed_percentage_points=(100*(1-gh.mean(1)/ph.mean(1))-100*(1-gl.mean(1)/pl.mean(1))).tolist(),per_place_percentage_points=(100*(1-gh.mean(0)/ph.mean(0))-100*(1-gl.mean(0)/pl.mean(0))).tolist())
            if n>=2:
                rng=np.random.default_rng(20261113);ss=rng.integers(0,3,(5000,3,1));pp=rng.integers(0,n,(5000,1,n));boot=100*(1-gh[ss,pp].mean((1,2))/ph[ss,pp].mean((1,2)))-100*(1-gl[ss,pp].mean((1,2))/pl[ss,pp].mean((1,2)));result['interval95_percentage_points']=np.percentile(boot,[2.5,97.5]).tolist()
        interactions[qg]=result
    original=read(J/'scoring/summary.json');replay=[]
    for a,b in PAIRS:
        ours=comparisons['fit/all/roi/centered_log_mse/'+a+'-'+b];prior=original['comparisons']['native/development12/fit/roi/centered_log_mse/'+a+'-'+b]
        err=max(abs(ours[k]-prior[k]) for k in ('test','reference','reduction_percent'));err=max(err,max(abs(x-y) for x,y in zip(ours['interval95_percent'],prior['interval95_percent'])));assert err<1e-11;replay.append(dict(pair=a+'-'+b,maximum_aggregate_difference=err,original=prior))
    flags={}
    for label in ('all','H','L','U'):
        cc={a+'-'+b:comparisons[f'fit/{label}/roi/centered_log_mse/{a}-{b}'] for a,b in PAIRS};flags[label]=dict(G_P_practical=practical(cc['G-P']),G_Native_practical=practical(cc['G-Native']),G_S_practical=practical(cc['G-S']),conditional_development_line_only=practical(cc['G-P']) and practical(cc['G-Native']),RasterFill_primary_material_harm=harm(cc['G-RasterFill']),RasterFill_missing_whole_AbsRel_material_harm=harm(comparisons[f'fit/{label}/whole/absrel/G-RasterFill']))
    write(OUT/'summary.json',dict(complete=True,summaries=summaries,comparisons=comparisons,interactions=interactions,bootstrap=dict(replicates=5000,seed=20261113,unit='Equal eligible queries within each place, equal places, paired original three seeds; fewer than2places cross-place interval unavailable.'),signed_bias='Signed means and absolute differences only; no percent error reduction on signed log_bias.',no_training_subset_or_threshold_scan=True))
    write(OUT/'method_decision.json',dict(complete=True,measured_flags=flags,primary='fit/H/roi/centered_log_mse/G-P',conditional_line_screen_H=flags['H']['conditional_development_line_only'],interaction=interactions['fit'],scientific_interpretation_reserved_for_original_Astra=True,new_fresh_confirmation=False,old_global_endpoint_unchanged=True,old_closed_training_not_reopened=True,method_selected=False,overall_goal_claim=False))
    write(OUT/'counterexamples.json',counterexamples);write(OUT/'unstratified_replay.json',dict(source=ident(J/'scoring/summary.json'),comparisons=replay,no_pixel_recomputation=True))
    zeros=[dict(lab,**{k:coverage[lab['id'],lab['query']][v] for k,v in [('original_zero_all','old_zero_all'),('J_zero_all','new_zero_all')]}) for lab in labels_ if coverage[lab['id'],lab['query']]['old_zero_all'] or coverage[lab['id'],lab['query']]['new_zero_all']]
    write(OUT/'zero_coverage_and_U.json',dict(all_development_zero_queries=zeros,all_U_labels=[r for r in labels_ if r['label']=='U'],all_retained=True,zero_is_not_U_definition=True,counts={qg:{lab:dict(query_count=sum(r['query_group']==qg and r['label']==lab for r in labels_),original_zero_all=sum(r['query_group']==qg and r['label']==lab and coverage[r['id'],r['query']]['old_zero_all'] for r in labels_),J_zero_all=sum(r['query_group']==qg and r['label']==lab and coverage[r['id'],r['query']]['new_zero_all'] for r in labels_)) for lab in ('H','L','U')} for qg in ('fit','other')}))
    write(OUT/'score_input_identity.json',dict(inputs=[ident(p) for p in [metricpath,J/'scoring/summary.json',J/'scoring/independent_recomputation.json',J/'configuration_lock.json',BASE/'report.md',BASE/'cost.json',BASE/'source_build_cost.json']],fixed_row='J',fixed_methods=list(METHODS),fixed_seeds=list(SEEDS),new_predictions=0,new_pixel_scores=0,raw_prediction_files_read=0,full_by_construction='96Native_full reference scores shared path; not learned noninferiority evidence.',labels_commit_sha256=sha(OUT/'labels_complete.json')))
    with (OUT/'all_comparisons.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['comparison','places','reduction_percent','interval95_percent','absolute','absolute_interval95','per_seed_percent','per_place_percent'])
        for key,v in comparisons.items():w.writerow([key,v['places'],v['reduction_percent'],v['interval95_percent'],v['absolute'],v['absolute_interval95'],v['per_seed_percent'],v['per_place_percent']])
    print(dict(aggregate_complete=True,flags=flags,interaction=interactions),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['labels','aggregate']);args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);assert sha(SCIENCE)==SCIENCE_SHA
    with (OUT/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);admit=guards();mem={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith(('MemAvailable:','MemTotal:'))};assert mem['MemAvailable']>18*2**30 and shutil.disk_usage(ROOT).free>.5*2**30
        write(OUT/f'{args.phase}_admission.json',dict(**admit,CPU_count=os.cpu_count(),load=os.getloadavg(),memory=mem,project_free_bytes=shutil.disk_usage(ROOT).free,thread_cap=1,CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],Python=sys.version,Torch=torch.__version__,NumPy=np.__version__))
        try:labels() if args.phase=='labels' else aggregate()
        except BaseException as error:
            write(OUT/f'{args.phase}_failure_{time.time_ns()}.json',dict(type=type(error).__name__,error=str(error),resource=guards()));raise
        write(OUT/f'{args.phase}_worker.json',dict(complete=True,**guards(),wall_started_epoch=WALL_START,wall_ended_epoch=time.time(),calls=dict(GPU=0,training=0,new_model_prediction=0,raster=0,download=0,old_pixel_metric_recomputation=0),scope='CPU process wall includes metadata/old score reads, aggregation and output I/O; actual exit is recorded externally after wait.'))
if __name__=='__main__':main()
