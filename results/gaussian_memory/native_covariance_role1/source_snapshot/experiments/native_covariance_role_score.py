"""Full fixed-mask scores, NumPy recomputation and prespecified paired contrasts."""
import csv
import time
import numpy as np
import torch
from experiments.native_covariance_role import OUT,PLAN,ARMS,cases,write,verify,d
from experiments.gaussian_scene_score import score_torch,score_numpy,regions,METRICS,json_finite


def collect():
    verify();rows=[];maximum=0.;audit=[];queries={};supports={};original={m['id'] for m in d.plan()['original_training_catalog']}
    expected={(seed,m['id'],q,c,arm) for seed in d.SEEDS for m,q,_ in cases() for c in d.CONDITIONS for arm in ARMS};seen=set()
    for meta,q,_ in cases():
        rid=meta['id'];queries[rid,q]=d.load(d.query_path(rid,q))
        if rid not in supports:
            path=(d.CACHE if d.is_new(rid) else d.VIEW_CACHE)/'evaluation_support'/f'{rid}.pt';supports[rid]=d.load(path)['labels']
    for path in sorted((OUT/'completed').glob('*.json')):
        r=d.read(path);key=(r['seed'],r['id'],r['query'],r['condition'],r['arm']);assert r['complete'] and key in expected and key not in seen;seen.add(key)
        assert d.sha256(r['prediction_path'])==r['sha256']
        pred=torch.from_numpy(np.load(r['prediction_path'],allow_pickle=False));query=queries[r['id'],r['query']];masks=regions(query,supports[r['id']][str(r['query'])])
        a=score_torch(pred,query['target'],masks);b=score_numpy(pred,query['target'],masks)
        for reg in a:
            for metric,value in a[reg].items():
                other=b[reg][metric];assert (value is None)==(other is None)
                if value is not None:maximum=max(maximum,abs(value-other))
        rows.append(dict(id=r['id'],place=r['place'],query=r['query'],condition=r['condition'],kind=r['arm'],seed=r['seed'],cohort=r['cohort'],training_group='original' if r['id'] in original else 'added' if r['cohort']=='train' else None,metrics=a,inherited=False))
        audit.append(dict(key=list(key),path=r['prediction_path'],sha256=r['sha256']))
    assert seen==expected and len(seen)==4224 and maximum<1e-9
    source=d.OUT/'scoring/metrics.json';inherited=[];target={(m['id'],q,c) for m,q,_ in cases() for c in d.CONDITIONS}
    for r in d.read(source)['rows']:
        if r['kind'] not in ('W_Local','W_Render') or (r['id'],r['query'],r['condition']) not in target:continue
        inherited.append(dict(r,kind='Local' if r['kind']=='W_Local' else 'Old_R',cohort='development' if r['cohort']=='development' else 'train',training_group='original' if r['id'] in original else 'added' if r['cohort']!='development' else None,inherited=True))
    assert len(inherited)==2112
    rows+=inherited;write(OUT/'scoring/metrics.json',dict(rows=rows))
    write(OUT/'scoring/independent_recomputation.json',dict(complete=True,new_predictions=4224,inherited_predictions=2112,new_max_abs_metric_difference=maximum,independent_backend='NumPy float64 residual/index/centering implementation',all_new_exported_predictions_recomputed=True,files=audit,inherited_source=str(source),inherited_source_sha256=d.sha256(source),inherited_prediction_sources=verify()['inherited'],inherited_independent_recomputation=str(d.OUT/'scoring/independent_recomputation.json'),new_confirmation=False))
    return rows


def contrast(a,b,indices):
    seed,place=indices;x=np.nanmean(a[seed,place],axis=(1,2));y=np.nanmean(b[seed,place],axis=(1,2))
    xm=float(np.nanmean(a));ym=float(np.nanmean(b));delta=a-b
    with np.errstate(invalid='ignore',divide='ignore'):
        samples=100*(x/y-1);perseed=100*(np.nanmean(a,1)/np.nanmean(b,1)-1);perplace=100*(np.nanmean(a,0)/np.nanmean(b,0)-1)
    return dict(test=xm,reference=ym,mean_absolute=xm-ym,interval95_absolute=np.nanpercentile(x-y,[2.5,97.5]).tolist(),relative_error_increase_percent=100*(xm/ym-1) if ym else None,interval95_percent=np.nanpercentile(samples,[2.5,97.5]).tolist(),per_seed_relative_percent=perseed.tolist(),per_place_relative_percent=perplace.tolist(),per_seed_absolute=np.nanmean(delta,1).tolist(),per_place_absolute=np.nanmean(delta,0).tolist(),positive_place_count=int((np.nanmean(delta,0)>0).sum()),negative_place_count=int((np.nanmean(delta,0)<0).sum()),paired_cells=int((np.isfinite(a)&np.isfinite(b)).sum()))


def summarize(rows):
    p=d.plan();summaries={};comparisons={};order={};methods=ARMS+('Old_R','Local')
    groups=dict(train32=p['training_catalog'],train_original=p['original_training_catalog'],train_added=p['added_training_catalog'],development12=p['catalog'])
    pairs=[('C_match','F'),('S_match','F'),('C_raw','C_match'),('C_raw','F'),('F','Old_R')]+[(k,'Local') for k in ARMS]+[('Old_R','Local')]
    for cohort,metas in groups.items():
        places=[m['place'] for m in metas];order[cohort]=places;lookup={place:i for i,place in enumerate(places)};rng=np.random.default_rng(20261113)
        indices=(rng.integers(0,3,(5000,3,1)),rng.integers(0,len(places),(5000,1,len(places))))
        rr=[r for r in rows if r['place'] in lookup]
        for condition in d.CONDITIONS:
            cells={}
            for r in rr:
                if r['condition']==condition:cells.setdefault((r['kind'],d.SEEDS.index(r['seed']),lookup[r['place']]),[]).append(r['metrics'])
            for reg in ['whole','roi','visible','roi_supported','roi_unsupported','whole_supported','whole_unsupported']:
                for metric in METRICS:
                    arrays={k:np.full((3,len(places)),np.nan) for k in methods}
                    for (k,si,pi),vals in cells.items():
                        v=[x[reg][metric] for x in vals if x[reg][metric] is not None]
                        if v:arrays[k][si,pi]=np.mean(v)
                    key=f'{cohort}/{condition}/{reg}/{metric}'
                    for k,a in arrays.items():summaries[key+'/'+k]=dict(mean=float(np.nanmean(a)),per_seed=np.nanmean(a,1).tolist(),per_place=np.nanmean(a,0).tolist())
                    if metric=='log_bias':continue
                    for a,b in pairs:comparisons[key+'/'+a+'-'+b]=contrast(arrays[a],arrays[b],indices)
    return json_finite(dict(complete=True,primary_endpoint='development12/missing/roi/centered_log_mse',summaries=summaries,comparisons=comparisons,place_order=order,bootstrap=dict(replicates=5000,seed=20261113,units='paired seeds/places; equal queries within place'),new_predictions=4224,inherited_predictions=2112,new_confirmation=False,scope='Frozen original Render conditional read-time geometry dependence; original shape remains in feature binding and factual coverage; not trained alternatives, Transformer PE gain or independent confirmation.'))


def decisions(summary):
    key=summary['primary_endpoint'];cmp=summary['comparisons'];flags={}
    for name in ('C_match','S_match','C_raw'):
        r=cmp[key+'/'+name+'-F'];flags[name]=dict(r,practical_dependence=r['relative_error_increase_percent']>=1 and r['interval95_percent'][0]>0 and all(v>0 for v in r['per_seed_relative_percent']))
    center=flags['C_match']['practical_dependence'];sphere=flags['S_match']['practical_dependence'];raw=flags['C_raw']['practical_dependence'];c=flags['C_match']
    improvement=c['relative_error_increase_percent']<=-1 and c['interval95_percent'][1]<0 and all(v<0 for v in c['per_seed_relative_percent'])
    if center and sphere:branch='both_matched_controls_degrade'
    elif improvement:branch='center_matched_practically_improves'
    elif sphere and not center:branch='sphere_only_practical_dependence'
    elif raw and not center and not sphere:branch='raw_only_practical_dependence'
    else:branch='limited_uncertain_or_mixed'
    return dict(complete=True,empirical_branch=branch,primary=flags,center_matched_practical_improvement=improvement,interpretation_deferred_to_same_Astra=True,method_adopted=False,O6_success=False,independent_confirmation=False,overall_goal_achieved=False,extra_scans_authorized=False)


def main():
    torch.set_num_threads(4);t=time.monotonic();rows=collect();s=summarize(rows);write(OUT/'scoring/summary.json',s);write(OUT/'method_decision.json',decisions(s))
    with (OUT/'scoring/all_summary_cells.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['scope_condition_region_metric_method','mean','per_seed','per_place'])
        for k,v in s['summaries'].items():w.writerow([k,v['mean'],v['per_seed'],v['per_place']])
    with (OUT/'scoring/all_comparisons.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['comparison','absolute','absolute_lo','absolute_hi','increase_percent','percent_lo','percent_hi','per_seed_percent'])
        for k,v in s['comparisons'].items():w.writerow([k,v['mean_absolute'],*v['interval95_absolute'],v['relative_error_increase_percent'],*v['interval95_percent'],v['per_seed_relative_percent']])
    write(OUT/'scoring/runtime.json',dict(complete=True,seconds=time.monotonic()-t));print(dict(complete=True,rows=len(rows),seconds=time.monotonic()-t),flush=True)


if __name__=='__main__':main()
