"""All fixed predictions, independent NumPy float64 scoring and paired summaries."""
import time
import numpy as np
import torch
from experiments.native_render_factorization import OUT,PLAN,ARMS,cases,write,verify
from experiments import native_coverage_data as d
from experiments.gaussian_scene_score import score_torch,score_numpy,regions,METRICS,json_finite,paired


def collect():
    verify();rows=[];audit=[];maximum=0.;queries={};supports={};original={m['id'] for m in d.plan()['original_training_catalog']}
    expected={(seed,m['id'],q,c,arm) for seed in d.SEEDS for m,q,cohort in cases() for c in d.CONDITIONS for arm in ARMS}
    markers=list((OUT/'completed').glob('*.json'));observed=set()
    for meta,q,cohort in cases():
        rid=meta['id'];queries[rid,q]=d.load(d.query_path(rid,q))
        if rid not in supports:
            path=(d.CACHE if d.is_new(rid) else d.VIEW_CACHE)/'evaluation_support'/f'{rid}.pt'
            supports[rid]=d.load(path)['labels']
    for marker in sorted(markers):
        r=d.read(marker);assert r['complete'];key=(r['seed'],r['id'],r['query'],r['condition'],r['arm']);assert key in expected and key not in observed;observed.add(key)
        path=r['prediction_path'];assert d.sha256(path)==r['sha256']
        pred=torch.from_numpy(np.load(path,allow_pickle=False));query=queries[r['id'],r['query']]
        masks=regions(query,supports[r['id']][str(r['query'])]);a=score_torch(pred,query['target'],masks);b=score_numpy(pred,query['target'],masks)
        for region in a:
            for metric,value in a[region].items():
                other=b[region][metric];assert (value is None)==(other is None)
                if value is not None:maximum=max(maximum,abs(value-other))
        rows.append(dict(id=r['id'],place=r['place'],query=r['query'],condition=r['condition'],kind=r['arm'],seed=r['seed'],cohort=r['cohort'],training_group='original' if r['id'] in original else 'added' if r['cohort']=='train' else None,metrics=a,inherited=False))
        audit.append(dict(key=list(key),path=path,sha256=r['sha256']))
    assert observed==expected and len(observed)==4224 and maximum<1e-9
    source=d.OUT/'scoring/metrics.json';inherited=[]
    target={(m['id'],q,c) for m,q,_ in cases() for c in d.CONDITIONS}
    for r in d.read(source)['rows']:
        if r['kind'] not in ('W_Local','W_Render') or (r['id'],r['query'],r['condition']) not in target:continue
        inherited.append(dict(r,kind='Local' if r['kind']=='W_Local' else 'R',cohort='development' if r['cohort']=='development' else 'train',training_group='original' if r['id'] in original else 'added' if r['cohort']!='development' else None,inherited=True))
    assert len(inherited)==2112
    rows+=inherited
    write(OUT/'scoring/metrics.json',dict(rows=rows))
    result=dict(complete=True,new_predictions=4224,inherited_predictions=2112,new_max_abs_metric_difference=maximum,independent_backend='Independent existing NumPy float64 residual/index/centering implementation',all_new_exported_predictions_recomputed=True,files=audit,inherited_source=str(source),inherited_source_sha256=d.sha256(source),inherited_prediction_sources=verify()['inherited'],inherited_independent_recomputation=str(d.OUT/'scoring/independent_recomputation.json'),new_confirmation=False)
    write(OUT/'independent_recomputation.json',result)
    return rows


def effect(delta,indices):
    seed,place=indices;boot=np.nanmean(delta[seed,place],axis=(1,2))
    return dict(mean_absolute=float(np.nanmean(delta)),interval95_absolute=np.nanpercentile(boot,[2.5,97.5]).tolist(),per_seed_absolute=np.nanmean(delta,axis=1).tolist(),per_place_absolute=np.nanmean(delta,axis=0).tolist(),positive_place_count=int((np.nanmean(delta,axis=0)>0).sum()),negative_place_count=int((np.nanmean(delta,axis=0)<0).sum()))


def summarize(rows):
    p=d.plan();p=dict(p,bootstrap=dict(seed=20261113,replicates=5000));summaries={};contrasts={};comparisons={};order={}
    groups=dict(train32=p['training_catalog'],train_original=p['original_training_catalog'],train_added=p['added_training_catalog'],development12=p['catalog'])
    methods=('R','N','C','U','M','Local')
    for cohort,metas in groups.items():
        places=[m['place'] for m in metas];order[cohort]=places;lookup={place:i for i,place in enumerate(places)}
        rng=np.random.default_rng(20261113);indices=(rng.integers(0,3,(5000,3,1)),rng.integers(0,len(places),(5000,1,len(places))))
        rr=[r for r in rows if r['place'] in lookup]
        for c in d.CONDITIONS:
            cells={}
            for r in rr:
                if r['condition']==c:cells.setdefault((r['kind'],d.SEEDS.index(r['seed']),lookup[r['place']]),[]).append(r['metrics'])
            for region in ['whole','roi','visible','roi_supported','roi_unsupported','whole_supported','whole_unsupported']:
                for metric in METRICS:
                    arrays={k:np.full((3,len(places)),np.nan) for k in methods}
                    for (kind,si,pi),values in cells.items():
                        vals=[v[region][metric] for v in values if v[region][metric] is not None]
                        if vals:arrays[kind][si,pi]=np.mean(vals)
                    key=f'{cohort}/{c}/{region}/{metric}'
                    for k,a in arrays.items():summaries[key+'/'+k]=dict(mean=float(np.nanmean(a)),per_seed=np.nanmean(a,axis=1).tolist(),per_place=np.nanmean(a,axis=0).tolist())
                    if metric=='log_bias':continue
                    for a,b in [(k,'R') for k in ARMS]+[(k,'Local') for k in ('R',)+ARMS]:comparisons[key+'/'+a+'-'+b]=dict(paired(arrays[a],arrays[b],p),absolute_effect=effect(arrays[a]-arrays[b],indices))
                    r,n,cc,u,m,l=[arrays[k] for k in methods]
                    ds=dict(D_address=cc-r,D_coverage=n-r,interaction=u-cc-n+r,U_minus_N=u-n,U_minus_C=u-cc,D_matched=m-r)
                    contrasts[key]={name:effect(delta,indices) for name,delta in ds.items()}
                    if cohort=='train32' and c=='missing' and region=='roi' and metric=='centered_log_mse':
                        gap=float(np.nanmean(l-r))
                        for name,delta in ds.items():contrasts[key][name]['baseline_gap_erased_ratio']=float(np.nanmean(delta))/gap
    return json_finite(dict(complete=True,primary_endpoint='train32/missing/roi/centered_log_mse',summaries=summaries,contrasts=contrasts,comparisons=comparisons,place_order=order,bootstrap=dict(replicates=5000,seed=20261113,units='paired seeds and places; queries equal within place'),inherited_predictions=2112,new_predictions=4224,new_confirmation=False,scope='Fixed Render weight conditional input dependence and sensitivity; not reoptimized methods, causal diagnosis of new attention, PE increment, or independent confirmation.'))


def main():
    torch.set_num_threads(4);begin=time.monotonic();rows=collect();write(OUT/'scoring/summary.json',summarize(rows));write(OUT/'scoring/runtime.json',dict(seconds=time.monotonic()-begin,complete=True))
    print(dict(complete=True,rows=len(rows),seconds=time.monotonic()-begin),flush=True)


if __name__=='__main__':main()
