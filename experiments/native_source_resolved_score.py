"""Independent final-coordinate PCK/EPE, paired-place/seed outcomes and full failures."""
import json,time,math,collections,resource
from pathlib import Path
import numpy as np
import torch
from experiments import native_source_resolved_data as d
ALLKINDS=(*d.KINDS,"P","G","L")
COMPARISONS=[("F_G","F_P"),("F_P","M_P"),("M_G","M_P"),("F_G","M_G"),("F_G","F_S"),("F_S","F_P"),("F_G","P"),("F_G","G"),("F_G","L"),("F_G","Native2"),("F_G","Native4"),("F_S","P"),("F_S","L"),("F_S","Native4"),("F_P","P"),("F_P","L"),("F_P","Native4"),("M_P","P"),("M_G","P"),("M_G","L"),("M_G","Native4")]

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,(np.integer,np.bool_)):return x.item()
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    return x
def avg(values):
    v=[x for x in values if x is not None];return float(np.mean(v)) if v else None
def paired_interval(matrix):
    a=np.asarray(matrix,dtype=np.float64);valid=np.isfinite(a).all(0);a=a[:,valid];ns,np_=a.shape
    if not np_:return dict(mean=None,interval95=None,places=0,seeds=ns)
    rng=np.random.default_rng(20261114);s=rng.integers(0,ns,(5000,ns));p=rng.integers(0,np_,(5000,np_));b=a[s[:,:,None],p[:,None,:]].mean((1,2)) if np_>1 else None
    return dict(mean=float(a.mean()),interval95=np.percentile(b,[2.5,97.5]).tolist() if b is not None else None,places=np_,seeds=ns,per_seed=a.mean(1).tolist())
def paired_continuous(a,b):
    # A model failure is infinity, never an omitted label or a finite-subset CI.
    if any(x is not None and not np.isfinite(x) for row in a+b for x in row):
        return dict(mean=None,interval95=None,status='Nonfinite model-failure EPE retained; finite-subset CI deliberately undefined',failed_entries_A=sum(x is not None and not np.isfinite(x) for row in a for x in row),failed_entries_B=sum(x is not None and not np.isfinite(x) for row in b for x in row))
    delta=[[x-y if x is not None and y is not None else np.nan for x,y in zip(aa,bb)] for aa,bb in zip(a,b)]
    return dict(**paired_interval(delta),status='A minus B in448pixel units; lower is better')
def row_score(coords,gt,valid,initial_valid=True):
    coords=np.asarray(coords,dtype=np.float64);gt=np.asarray(gt,dtype=np.float64);mask=np.asarray(valid,bool);ok=np.isfinite(coords).all(-1)&bool(initial_valid);err=np.linalg.norm(coords-gt,axis=-1);err=np.where(ok,err,np.inf);values=err[mask];n=len(values)
    return dict(valid_GT=n,unlabelled_points=64-n,failed_valid_points=int((~ok&mask).sum()),PCK8=float((values<=8).mean()) if n else None,EPE=float(values.mean()) if n else None,EPE95=float(np.percentile(values,95)) if n and np.isfinite(values).all() else (float('inf') if n else None),EPE_median=float(np.median(values)) if n else None,EPE_infinite_due_model_failure=bool(n and not np.isfinite(values).all()),point_errors=err.tolist(),predicted_all64=True)
def score():
    t=time.monotonic();torch.set_num_threads(2);cases={};rows=[];max_recompute=0.;totalpred=0;independent_records=0
    def independently_recompute(pred,labels,result,initial_valid=True):
        nonlocal max_recompute,independent_records
        independent_records+=1
        pp=torch.as_tensor(pred).double();mask=labels['valid'];ee=(pp-labels['target'].double()).norm(dim=-1);ok=torch.isfinite(pp).all(-1)&bool(initial_valid);ee=torch.where(ok,ee,torch.inf)[mask]
        assert int((~ok&mask).sum())==result['failed_valid_points']
        if not len(ee):assert result['PCK8'] is None;return
        max_recompute=max(max_recompute,abs(float((ee<=8).double().mean())-result['PCK8']))
        if torch.isfinite(ee).all():
            for metric,value in [('EPE',ee.mean()),('EPE95',torch.quantile(ee,.95)),('EPE_median',torch.quantile(ee,.5))]:max_recompute=max(max_recompute,abs(float(value)-result[metric]))
        else:assert result['EPE_infinite_due_model_failure'] and math.isinf(result['EPE'])
    inherited_rows=d.read(d.prior.OUT/"scoring/viewpairs.json");inherited_index=collections.defaultdict(list)
    for r in inherited_rows:
        if r["kind"] in ("P","G","L","Native2","Native4"):inherited_index[r["id"],r["anchor"],r["query"]].append(r)
    trainids={m['id'] for m in d.plan()['training_catalog']};oldtrain={m['id'] for m in d.plan()['original_training_catalog']}
    for meta in d.metas():
        rid=meta['id']
        for a in d.SOURCES:
            for q in d.QUERIES:
                obj=d.load(d.case(rid,a,q));source_alignment_valid=bool(obj['inputs']['alignment_valid']);predicted_geometry_finite=bool(torch.isfinite(obj['inputs']['predicted_camera']).all() and torch.isfinite(obj['inputs']['predicted_K']).all());initializer_valid=source_alignment_valid and predicted_geometry_finite;cases[rid,a,q]=dict(labels=obj['labels'],inputs=dict(zero_support=obj['inputs']['zero_support'],alignment_valid=initializer_valid,source_alignment_valid=source_alignment_valid,predicted_geometry_finite=predicted_geometry_finite,valid_keys=int(obj['inputs']['GS']['key_mask'].sum())))
                cohort='train32' if rid in trainids else 'dev12';qset='original' if q in d.QUERIES[:4] else 'other'
                common=dict(id=rid,place=meta['place'],anchor=a,query=q,cohort=cohort,query_set=qset,training_origin='original12' if rid in oldtrain else ('added20' if rid in trainids else 'development12'),zero_GS_raster_support=obj['inputs']['zero_support'],zero_GS_attention_keys=not bool(obj['inputs']['GS']['key_mask'].any()),initializer_valid=initializer_valid,source_alignment_valid=source_alignment_valid,predicted_geometry_finite=predicted_geometry_finite)
                reference=inherited_index[rid,a,q]
                assert len(reference)==11 and all(r['valid_GT']==int(obj['labels']['valid'].sum()) for r in reference)
                rows.extend(reference)
                for kind in d.KINDS:
                    for seed in d.SEEDS:
                        p=d.CACHE/'predictions'/f'{kind}_{seed}'/rid/f'{a}_{q}.pt';pred=d.load(p)['coords'][-1].numpy();r=row_score(pred,obj['labels']['target'].numpy(),obj['labels']['valid'].numpy(),initializer_valid);rows.append(dict(**common,kind=kind,seed=seed,**r));totalpred+=1
                        independently_recompute(pred,obj['labels'],r,initializer_valid)
    assert totalpred==21120
    place=[];queryrows=[];sourcerows=[]
    for kind in (*ALLKINDS,'Native2','Native4'):
        for seed in d.SEEDS if kind in ALLKINDS else [None]:
            for meta in d.metas():
                rid=meta['id'];rr=[r for r in rows if r['id']==rid and r['kind']==kind and r['seed']==seed];qdata=[]
                for q in d.QUERIES:
                    v=[r for r in rr if r['query']==q];entry=dict(id=rid,kind=kind,seed=seed,query=q,defined_anchor_pairs=sum(r['PCK8'] is not None for r in v),**{k:avg([r[k] for r in v]) for k in ('PCK8','EPE','EPE95','EPE_median')});queryrows.append(entry);qdata.append(entry)
                for a in d.SOURCES:
                    for qset in ('original','other'):
                        v=[r for r in rr if r['anchor']==a and r['query_set']==qset];sourcerows.append(dict(id=rid,kind=kind,seed=seed,anchor=a,query_set=qset,defined_queries=sum(r['PCK8'] is not None for r in v),**{k:avg([r[k] for r in v]) for k in ('PCK8','EPE','EPE95','EPE_median')}))
                for qset in ('original','other','all8'):
                    qs=d.QUERIES[:4] if qset=='original' else d.QUERIES[4:] if qset=='other' else d.QUERIES;v=[r for r in qdata if r['query'] in qs];vr=[r for r in rr if r['query'] in qs]
                    place.append(dict(id=rid,place=meta['place'],cohort=vr[0]['cohort'],training_origin=vr[0]['training_origin'],kind=kind,seed=seed,query_set=qset,defined_queries=sum(r['PCK8'] is not None for r in v),defined_viewpairs=sum(r['PCK8'] is not None for r in vr),valid_GT_points=sum(r['valid_GT'] for r in vr),failed_valid_points=sum(r['failed_valid_points'] for r in vr),EPE_infinite_due_model_failure=any(r['EPE_infinite_due_model_failure'] for r in vr),**{k:avg([r[k] for r in v]) for k in ('PCK8','EPE','EPE95','EPE_median')}))
    place_index={(r['id'],r['kind'],r['seed'],r['query_set']):r for r in place}
    assert len(place_index)==len(place)
    comparisons={};continuous={};absolute={}
    cohort_ids=dict(dev12=[m['id'] for m in d.plan()['catalog']],train32=[m['id'] for m in d.plan()['training_catalog']],train_original12=sorted(oldtrain),train_added20=sorted(trainids-oldtrain))
    def get(cohort,qset,kind,seed,metric):
        return [place_index[rid,kind,seed,qset][metric] for rid in cohort_ids[cohort]]
    for cohort in cohort_ids:
        for qset in ('original','other','all8'):
            k=f'{cohort}/{qset}';absolute[k]={};comparisons[k]={};continuous[k]={}
            for kind in (*ALLKINDS,'Native2','Native4'):
                seeds=d.SEEDS if kind in ALLKINDS else [None];selected=[r for r in place if r['id'] in cohort_ids[cohort] and r['query_set']==qset and r['kind']==kind];absolute[k][kind]=dict(PCK8=paired_interval([[np.nan if x is None else x*100 for x in get(cohort,qset,kind,s,'PCK8')] for s in seeds]),EPE_place_mean=avg([r['EPE'] for r in selected]),EPE95_place_mean=avg([r['EPE95'] for r in selected]),failed_valid_points=sum(r['failed_valid_points'] for r in selected))
            for ka,kb in COMPARISONS:
                ma=[]
                for seed in d.SEEDS:
                    a=get(cohort,qset,ka,seed,'PCK8');b=get(cohort,qset,kb,seed if kb in ALLKINDS else None,'PCK8');ma.append([100*(x-y) if x is not None and y is not None else np.nan for x,y in zip(a,b)])
                comparisons[k][ka+'-'+kb]=paired_interval(ma)
                continuous[k][ka+'-'+kb]={metric:paired_continuous([get(cohort,qset,ka,s,metric) for s in d.SEEDS],[get(cohort,qset,kb,s if kb in ALLKINDS else None,metric) for s in d.SEEDS]) for metric in ('EPE','EPE95')}
    interactions={}
    for cohort in cohort_ids:
        for qset in ('original','other','all8'):
            matrix=[]
            for seed in d.SEEDS:
                fp=get(cohort,qset,'F_P',seed,'PCK8');fg=get(cohort,qset,'F_G',seed,'PCK8');mp=get(cohort,qset,'M_P',seed,'PCK8');mg=get(cohort,qset,'M_G',seed,'PCK8')
                matrix.append([100*((g-p)-(gg-pp)) if all(x is not None for x in (g,p,gg,pp)) else np.nan for g,p,gg,pp in zip(fg,fp,mg,mp)])
            interactions[cohort+'/'+qset]=paired_interval(matrix)
    primary=comparisons['dev12/original'];flags={}
    for kind in ('F_G','F_S'):
        pe=primary[kind+'-F_P'];pepass=pe['mean'] is not None and pe['mean']>=1 and pe['interval95'] is not None and pe['interval95'][0]>0
        net={ref:primary[kind+'-'+ref]['interval95'] is not None and primary[kind+'-'+ref]['interval95'][0]>0 for ref in ('P','L','Native4')}
        flags[kind]=dict(PE_increment=pepass,net_over_old_full_mean_P_L_Native4=net,development_candidate=pepass and all(net.values()),secondary_only=kind=='F_S')
    # Coverage is independent of arm/seed. Preserve each rejection stage and all
    # zero-label / zero-support cases before any endpoint aggregation.
    coverage=[];stages=collections.Counter()
    for (rid,a,q),obj in cases.items():
        lab=obj['labels'];iv=obj['inputs'];coverage.append(dict(id=rid,anchor=a,query=q,valid_GT=int(lab['valid'].sum()),unlabelled_points=64-int(lab['valid'].sum()),**iv))
        for stage in lab['stage_counts']:stages[stage['name']]+=stage['rejected']
    strata=[]
    for cohort in ('dev12','train32'):
        for qset in ('original','other'):
            for flag in ('zero_GS_raster_support','zero_GS_attention_keys','initializer_valid'):
                for value in (False,True):
                    for kind in (*ALLKINDS,'Native2','Native4'):
                        rr=[r for r in rows if r['cohort']==cohort and r['query_set']==qset and r[flag]==value and r['kind']==kind]
                        # Diagnostic strata retain the same anchor/query/place
                        # hierarchy. They are conditional, never replacement endpoints.
                        byplace=[]
                        for rid in sorted({r['id'] for r in rr}):
                            for seed in d.SEEDS if kind in ALLKINDS else [None]:
                                qr=[]
                                for q in (d.QUERIES[:4] if qset=='original' else d.QUERIES[4:]):
                                    qr.append(avg([r['PCK8'] for r in rr if r['id']==rid and r['seed']==seed and r['query']==q]))
                                byplace.append(avg(qr))
                        strata.append(dict(cohort=cohort,query_set=qset,flag=flag,value=value,kind=kind,pair_seed_records=len(rr),defined_records=sum(r['PCK8'] is not None for r in rr),valid_GT_points=sum(r['valid_GT'] for r in rr),PCK8=avg(byplace),failed_valid_points=sum(r['failed_valid_points'] for r in rr)))
    counterexamples=[]
    bykey={(r['kind'],r['seed'],r['id'],r['anchor'],r['query']):r for r in rows}
    for ka,kb in COMPARISONS:
        for r in rows:
            if r['kind']!=ka or r['PCK8'] is None:continue
            b=bykey[kb,r['seed'] if kb in ALLKINDS else None,r['id'],r['anchor'],r['query']]
            counterexamples.append(dict(A=ka,B=kb,seed=r['seed'],id=r['id'],cohort=r['cohort'],query_set=r['query_set'],anchor=r['anchor'],query=r['query'],PCK_difference_pp=(r['PCK8']-b['PCK8'])*100,PCK_A=r['PCK8'],PCK_B=b['PCK8'],EPE_A=r['EPE'],EPE_B=b['EPE'],valid_GT=r['valid_GT'],zero_GS_raster_support=r['zero_GS_raster_support']))
    counterexamples.sort(key=lambda r:r['PCK_difference_pp'])
    summary=dict(complete=True,absolute=absolute,comparisons_pp=comparisons,comparisons_continuous_pixels=continuous,representation_PE_interaction_pp=interactions,flags=flags,bootstrap=dict(replicates=5000,seed=20261114,method='Paired crossed resampling of seed IDs and place IDs; same indices for both arms, no point/viewpair independence. Native reference shared across seeds.'),coverage=dict(fixed_geometry_pairs=1408,new_model_predictions=21120,inherited_trained_reference_predictions=12672,inherited_native_reference_predictions=2816,all64predicted=True,total_source_grid_points=1408*64,valid_GT=sum(r['valid_GT'] for r in coverage),zero_GT_pairs=sum(not r['valid_GT'] for r in coverage),zero_GS_raster_pairs=sum(r['zero_support'] for r in coverage),zero_GS_attention_pairs=sum(not r['valid_keys'] for r in coverage),invalid_initializer_pairs=sum(not r['alignment_valid'] for r in coverage),rejection_stage_totals=dict(stages)),independent_NumPy_Torch_max_difference=max_recompute,not_fresh=True,scientific_disposition_reserved_for_Astra=True)
    summary['independent_NumPy_Torch_records']=independent_records
    summary['inherited_metric_scope']=dict(path=str(d.prior.OUT/'scoring/viewpairs.json'),sha256=d.sha(d.prior.OUT/'scoring/viewpairs.json'),old_metrics_recomputed=False,old_independent_scoring_reference=str(d.prior.OUT/'scoring/summary.json'),full_mean_P_G_L_not_strict_new_PCA_initialization_controls=True)
    summary['independent_NumPy_Torch_metrics']=['PCK8','EPE','EPE95','EPE_median','failed_valid_points','undefined_GT']
    summary['coverage'].update(invalid_source_alignment_pairs=sum(not r['source_alignment_valid'] for r in coverage),nonfinite_predicted_camera_intrinsics_pairs=sum(not r['predicted_geometry_finite'] for r in coverage),initializer_valid_definition='Valid source-only metric alignment AND finite predicted query camera/intrinsics; no query GT fallback. Native references do not consume GS alignment.')
    tails=[]
    for cohort in ('dev12','train32'):
        for qset in ('original','other'):
            for kind in (*ALLKINDS,'Native2','Native4'):
                for seed in d.SEEDS if kind in ALLKINDS else [None]:
                    rr=[r for r in rows if r['cohort']==cohort and r['query_set']==qset and r['kind']==kind and r['seed']==seed];errors=[]
                    for r in rr:
                        mask=cases[r['id'],r['anchor'],r['query']]['labels']['valid'].numpy();errors.extend(np.asarray(r['point_errors'])[mask].tolist())
                    e=np.asarray(errors);tails.append(dict(cohort=cohort,query_set=qset,kind=kind,seed=seed,labelled_points=len(e),model_failure_infinite_points=int(np.isinf(e).sum()),pooled_point_tail_is_descriptive_not_primary_equal_place_estimator=True,percentiles_pixels={str(p):(float(np.percentile(e,p)) if len(e) and np.isfinite(e).all() else None) for p in (50,90,95,99)},maximum_pixels=float(e.max()) if len(e) else None))
    d.write(d.OUT/'scoring/summary.json',clean(summary));d.write(d.OUT/'scoring/viewpairs.json',clean(rows));d.write(d.OUT/'scoring/query.json',clean(queryrows));d.write(d.OUT/'scoring/source.json',clean(sourcerows));d.write(d.OUT/'scoring/place.json',clean(place));d.write(d.OUT/'scoring/pooled_point_tails.json',clean(tails))
    d.write(d.OUT/'scoring/label_coverage_cohorts.json',d.read(d.prior.OUT/'scoring/label_coverage_cohorts.json'));d.write(d.OUT/'scoring/coverage.json',clean(coverage));d.write(d.OUT/'scoring/support_strata.json',clean(strata));d.write(d.OUT/'scoring/all_pairwise_counterexamples.json',clean(counterexamples))
    d.write(d.OUT/'CPU_scoring_runtime.json',dict(complete=True,wall_seconds=time.monotonic()-t,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,CUDA_initialized=torch.cuda.is_initialized(),new_model_calls=0));print(json.dumps(clean(dict(primary=primary,flags=flags)),indent=2),flush=True)
if __name__=='__main__':score()
