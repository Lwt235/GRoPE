"""Independent NumPy float64 pose/coordinate scoring; fixed paired place x seed bootstrap."""
import os,csv,math,time
import numpy as np
import torch
from experiments.native_pose_memory_data import *

def np_camera(pose,align):
    p=np.asarray(pose,dtype=np.float64);x,y,z,w=p[3:7];s=2/(x*x+y*y+z*z+w*w)
    R=np.array([[1-s*(y*y+z*z),s*(x*y-z*w),s*(x*z+y*w)],[s*(x*y+z*w),1-s*(x*x+z*z),s*(y*z-x*w)],[s*(x*z-y*w),s*(y*z+x*w),1-s*(x*x+y*y)]])
    A=np.array(align['rotation'],dtype=np.float64);scale=float(align['scale']);t=np.array(align['translation'],dtype=np.float64);out=np.eye(4);out[:3,:3]=A@R.T;out[:3,3]=scale*A@(-R.T@p[:3])+t;return out

def metrics(camera,target,valid=True):
    if not valid or not np.isfinite(camera).all():return dict(valid=False,rotation_deg=180.,center_m=1e30,success=0)
    delta=camera[:3,:3].T@target[:3,:3];s=np.linalg.norm([delta[2,1]-delta[1,2],delta[0,2]-delta[2,0],delta[1,0]-delta[0,1]])/2;c=(np.trace(delta)-1)/2;rot=np.degrees(np.arctan2(s,c));d=np.linalg.norm(camera[:3,3]-target[:3,3]);return dict(valid=True,rotation_deg=float(rot),center_m=float(d),success=int(rot<5 and d<.2))

def csvwrite(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)

def summarise(rows):
    groups={}
    for r in rows:groups.setdefault((r['seed'],r['id']),[]).append(r)
    weighted=lambda key:float(np.mean([np.mean([r[key] for r in g]) for g in groups.values()]))
    return dict(n=len(rows),success_percent=100*weighted('success'),query_weighted_success_percent=100*float(np.mean([r['success'] for r in rows])),rotation_mean=weighted('rotation_deg'),rotation_median=float(np.median([r['rotation_deg'] for r in rows])),rotation_p90=float(np.quantile([r['rotation_deg'] for r in rows],.9)),center_mean=weighted('center_m'),center_median=float(np.median([r['center_m'] for r in rows])),center_p90=float(np.quantile([r['center_m'] for r in rows],.9)),invalid=sum(not r['valid'] for r in rows))

def comparison(rows,a,b,metric='success',bootstrap_seed=20261113):
    keys=sorted({r['id'] for r in rows});mapping={(r['kind'],r['seed'],r['id'],r['query']):r for r in rows};per=[]
    sign=100 if metric=='success' else -1
    for seed in SEEDS:
        arr=[]
        for rid in keys:
            qs=sorted({r['query'] for r in rows if r['id']==rid});arr.append(np.mean([sign*(mapping[(a,seed,rid,q)][metric]-mapping[(b,seed,rid,q)][metric]) for q in qs]))
        per.append(arr)
    d=np.asarray(per);rng=np.random.default_rng(bootstrap_seed);sidx=rng.integers(0,3,(5000,3));pidx=rng.integers(0,len(keys),(5000,len(keys)));boot=d[sidx[:,:,None],pidx[:,None,:]].mean((1,2));ci=np.quantile(boot,[.025,.975]);means=d.mean(1);mean=float(d.mean())
    return dict(candidate=a,baseline=b,metric=metric,mean=mean,CI95=ci.tolist(),seed_means=means.tolist(),places=keys,place_seed_differences=d.tolist(),screen=bool(metric=='success' and mean>=1 and ci[0]>0 and (means>0).all()),bootstrap=dict(replicates=5000,seed=bootstrap_seed,paired_place_seed=True))

def main():
    assert all((OUT/'evaluation'/f'{k}_{s}.json').exists() for k in KINDS for s in SEEDS)
    rows=[];maxdiff=0.;original={m['id'] for m in plan()['original_training_catalog']};train={m['id'] for m in plan()['training_catalog']};cameras={}
    support={ (r['id'],r['query']):r for r in [__import__('json').loads(l) for l in (OUT/'preparation_queries.jsonl').read_text().splitlines()]}
    for meta in metas():
        rid=meta['id'];sc=load(sceneout(rid));a=sc['alignment'];valid=a['valid']
        for q in QUERIES:
            c=load(casepath(rid,q));target=np.array(c['target'],dtype=np.float64);initial=np_camera(c['initial_pose'],a);base=metrics(initial,target,valid);cov=support[(rid,q)]['coverage']
            common=dict(id=rid,place=meta['place'],split='train' if rid in train else 'dev',training_cohort='original12' if rid in original else ('added20' if rid in train else 'dev12'),query=q,query_group='fit' if q in QUERIES[:4] else 'other',coverage=cov,zero_support=cov==0,initial_rotation_deg=base['rotation_deg'],initial_center_m=base['center_m'],initial_success=base['success'],alignment_valid=valid)
            for seed in SEEDS:
                rows.append(dict(kind='Native',seed=seed,**common,**base))
                for kind in KINDS:
                    pred=load(CACHE/'predictions'/f'{kind}_{seed}'/rid/f'{q}.pt');cam=np_camera(pred['pose'],a);maxdiff=max(maxdiff,float(np.abs(cam-np.array(pred['camera'])).max()));r=metrics(cam,target,valid);rows.append(dict(kind=kind,seed=seed,**common,**r));cameras[(kind,seed,rid,q)]=cam.tolist()
    csvwrite(OUT/'per_prediction_metrics.csv',rows)
    strata={
        'dev_fit':lambda r:r['split']=='dev' and r['query_group']=='fit',
        'dev_other':lambda r:r['split']=='dev' and r['query_group']=='other',
        'train_fit':lambda r:r['split']=='train' and r['query_group']=='fit',
        'train_other':lambda r:r['split']=='train' and r['query_group']=='other',
        'original12_fit':lambda r:r['training_cohort']=='original12' and r['query_group']=='fit',
        'added20_fit':lambda r:r['training_cohort']=='added20' and r['query_group']=='fit',
        'zero_support':lambda r:r['zero_support'],
        'positive_coverage':lambda r:r['coverage']>0,
        'initial_success':lambda r:r['initial_success']==1,
        'initial_failure':lambda r:r['initial_success']==0,
        'coverage_low':lambda r:0<r['coverage']<.25,
        'coverage_medium':lambda r:.25<=r['coverage']<.75,
        'coverage_high':lambda r:r['coverage']>=.75,
        'initial_large_error':lambda r:r['initial_rotation_deg']>=15 or r['initial_center_m']>=1,
        'initial_moderate_error':lambda r:not r['initial_success'] and r['initial_rotation_deg']<15 and r['initial_center_m']<1}
    comparisons=[('G','P'),('S','P'),('G','S')]+[(k,b) for k in KINDS for b in ('L','Native') if k!=b]
    summary={};table=[];paired=[]
    for name,predicate in strata.items():
        subset=[r for r in rows if predicate(r)]
        if not subset:summary[name]=dict(n=0,empty=True);continue
        summary[name]=dict(by_kind={k:summarise([r for r in subset if r['kind']==k]) for k in ('Native',)+KINDS},comparisons=[])
        for k in ('Native',)+KINDS:table.append(dict(stratum=name,kind=k,**summary[name]['by_kind'][k]))
        for a,b in comparisons:
            comp=comparison(subset,a,b);summary[name]['comparisons'].append(comp)
            paired.append(dict(stratum=name,candidate=a,baseline=b,mean_pp=comp['mean'],CI_low_pp=comp['CI95'][0],CI_high_pp=comp['CI95'][1],seed1_pp=comp['seed_means'][0],seed2_pp=comp['seed_means'][1],seed3_pp=comp['seed_means'][2],screen=comp['screen']))
        summary[name]['continuous_comparisons']=[comparison(subset,a,b,metric) for a,b in comparisons for metric in ('rotation_deg','center_m')]
    write(OUT/'summary.json',summary);csvwrite(OUT/'aggregate_metrics.csv',table);csvwrite(OUT/'paired_comparisons.csv',paired)
    dev=[r for r in rows if strata['dev_fit'](r)];cs={ (x['candidate'],x['baseline']):x for x in summary['dev_fit']['comparisons']};eligible=[]
    if all(cs[x]['screen'] for x in [('G','P'),('G','L'),('G','Native')]):eligible.append('G')
    if all(cs[x]['screen'] for x in [('S','P'),('S','L'),('S','Native')]):eligible.append('S')
    mapping={(r['kind'],r['seed'],r['id'],r['query']):r for r in rows};cases=[]
    for rid in sorted({r['id'] for r in dev}):
        for q in QUERIES[:4]:
            a=[mapping[('G',s,rid,q)] for s in SEEDS];b=[mapping[('P',s,rid,q)] for s in SEEDS];cases.append(dict(id=rid,query=q,success_loss=sum(y['success']-x['success'] for x,y in zip(a,b)),center_harm=float(np.mean([x['center_m']-y['center_m'] for x,y in zip(a,b)])),rotation_harm=float(np.mean([x['rotation_deg']-y['rotation_deg'] for x,y in zip(a,b)]))))
    worst=max(cases,key=lambda r:(r['success_loss'],r['center_harm'],r['rotation_harm']));write(OUT/'counterexamples.json',dict(ranking=sorted(cases,key=lambda r:(r['success_loss'],r['center_harm']),reverse=True),maximum=worst,fixed=dict(id=plan()['catalog'][0]['id'],query=2)))
    write(OUT/'method_decision.json',dict(status='complete_development_measurement_pending_Astra',primary=cs[('G','P')],eligible_under_success_screens=eligible,anisotropy_screen=cs[('G','S')]['screen'],overall_goal_achieved=False,fresh=False,selection_owner='original Astra; tail and cost tradeoffs require scientific assessment',scientific_success_not_claimed=True))
    write(OUT/'independent_recomputation.json',dict(complete=True,new_predictions=4224,unique_Native=352,Native_repeated_for_pairing=1056,arithmetic='NumPy float64 raw XYZW pose -> extrinsics inverse -> source-only map similarity -> geodesic and metric center',maximum_saved_float32_camera_difference=maxdiff,all_cases_retained=True,thresholds=dict(rotation_strict_degrees=5,center_strict_m=.2),bootstrap_replicates=5000,bootstrap_seed=20261113))
    print(summary['dev_fit']['by_kind']);print(summary['dev_fit']['comparisons'][:3])
if __name__=='__main__':main()
