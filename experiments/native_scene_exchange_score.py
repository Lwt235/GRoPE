"""Full predictions: separate NumPy64 recomputation, fixed paired bootstrap."""
import csv,warnings
from experiments import native_scene_exchange_data as d
from experiments.native_scene_exchange_data import np,torch,F
from experiments.gaussian_scene_score import score_torch,score_numpy,regions,METRICS
from experiments.camera_only_feature_score import contrast

def collect():
    folder=d.OUT/'scoring';folder.mkdir(exist_ok=True);dest=folder/'metrics.json'
    if dest.exists():return d.read(dest)
    records=sorted((d.OUT/'completed').glob('*.json'));assert len(records)==14784,len(records)
    train={m['id'] for m in d.plan()['training_catalog']};original={m['id'] for m in d.plan()['original_training_catalog']};masks={};targets={};allzero={};rows=[];maxdiff=0.;audit=[]
    for meta in d.metas():
        rid=meta['id'];support=d.load((d.old.CACHE if d.old.is_new(rid) else d.old.VIEW_CACHE)/'evaluation_support'/f'{rid}.pt')['labels']
        for q in d.old.EVAL_QUERIES:
            query=d.load(d.old.query_path(rid,q));a=d.load(d.old.feature_path(rid,q))['effective_alpha'].reshape(32,32);zero=F.interpolate((a<=1e-8)[None,None].float(),(210,210),mode='nearest')[0,0].bool();reg=regions(query,support[str(q)])
            reg.update(roi_zero_coverage=query['valid']&query['hidden']&zero,roi_unsupported_positive_coverage=query['valid']&query['hidden']&~support[str(q)]&~zero)
            masks[rid,q]=reg;targets[rid,q]=query['target'];allzero[rid,q]=bool((a<=1e-8).all())
    def add(r):
        nonlocal maxdiff
        rid,q=r['id'],r['query'];path=d.Path(r['path']);assert d.sha(path)==r['sha256'];pred=torch.from_numpy(np.load(path,allow_pickle=False));a=score_torch(pred,targets[rid,q],masks[rid,q]);b=score_numpy(pred,targets[rid,q],masks[rid,q])
        for region,vv in a.items():
            for k,v in vv.items():
                assert (v is None)==(b[region][k] is None)
                if v is not None:maxdiff=max(maxdiff,abs(v-b[region][k]))
        row=dict(r,kind=r['arm'],cohort='train' if rid in train else 'development',training_group='original' if rid in original else 'added' if rid in train else None,query_group='fit' if q in d.old.QUERIES else 'other',zero_coverage=allzero[rid,q],metrics=a);rows.append(row);audit.append(dict(path=str(path),sha256=r['sha256']))
    for i,p in enumerate(records):
        add(d.read(p))
        if i%500==0:print(dict(scored=i,of=14784),flush=True)
    for meta in d.metas():
        rid=meta['id']
        for q in d.old.EVAL_QUERIES:
            r=d.read(d.base.OUT/'baselines'/f'{rid}_{q}.json')
            for name,c in [('Native','missing'),('Native_full','full')]:
                add(dict(id=rid,place=meta['place'],query=q,condition=c,arm='Native',seed=None,**r['methods'][name]))
    result=dict(rows=rows,new_predictions=14784,inherited_unique_Native=704)
    d.write(dest,result);d.write(folder/'independent_recomputation.json',dict(complete=True,new_predictions=14784,inherited_unique_Native=704,independent_backend='NumPy float64 independent indexing/log residual/centering and Torchfloat64',maximum_metric_difference=maxdiff,raw_bias_decomposition='log_mse=centered_log_mse+bias_squared; all unmodified valid masks',files=audit,zero_coverage_queries=[dict(id=rid,query=q) for (rid,q),z in allzero.items() if z]))
    with (folder/'per_prediction_metrics.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['kind','seed','id','place','query','condition','cohort','training_group','query_group','zero_coverage','region','pixels',*METRICS])
        for r in rows:
            for reg,v in r['metrics'].items():writer.writerow([r[k] for k in ['kind','seed','id','place','query','condition','cohort','training_group','query_group','zero_coverage']]+[reg,v['pixels']]+[v[k] for k in METRICS])
    return result

def summarize(data):
    groups=dict(train32=d.plan()['training_catalog'],train_original12=d.plan()['original_training_catalog'],train_added20=d.plan()['added_training_catalog'],development12=d.plan()['catalog']);summaries={};comparisons={};interactions={};arrays={};rows=data['rows']
    methods=[*d.ARMS,'Native'];pairs=[(f'{row}_{a}',f'{row}_{b}') for row in ('R','U') for a,b in [('G','P'),('G','S'),('S','P')]]+[(f'U_{a}',f'R_{a}') for a in ('P','S','G')]+[(a,b) for a in d.ARMS for b in ('L','Native') if a!=b]
    for group,metas in groups.items():
        places=[m['place'] for m in metas];lookup={p:i for i,p in enumerate(places)}
        for qg,qs in [('fit',d.old.QUERIES),('other',d.old.HELDOUT),('pooled',d.old.EVAL_QUERIES)]:
            for condition in d.old.CONDITIONS:
                chosen=[r for r in rows if r['place'] in lookup and r['query'] in qs and r['condition']==condition]
                cells={}
                for r in chosen:cells.setdefault((r['kind'],r['seed'],r['place']),[]).append(r['metrics'])
                rng=np.random.default_rng(20261113);indices=(rng.integers(0,3,(5000,3,1)),rng.integers(0,len(places),(5000,1,len(places))))
                for region in chosen[0]['metrics']:
                    for metric in METRICS:
                        aa={}
                        for method in methods:
                            a=np.full((3,len(places)),np.nan)
                            for si,seed in enumerate(d.SEEDS):
                                for place,pi in lookup.items():
                                    vals=[v[region][metric] for v in cells.get((method,None if method=='Native' else seed,place),[]) if v[region][metric] is not None]
                                    if vals:a[si,pi]=np.mean(vals)
                            aa[method]=a;key=f'{group}/{qg}/{condition}/{region}/{metric}'
                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore',RuntimeWarning);summaries[key+'/'+method]=dict(mean=float(np.nanmean(a)) if np.isfinite(a).any() else None,per_seed=np.nanmean(a,1).tolist(),per_place=np.nanmean(a,0).tolist())
                        arrays[key]=aa
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore',RuntimeWarning)
                            for a,b in pairs:
                                if np.isfinite(aa[a]).any() and np.isfinite(aa[b]).any():comparisons[key+'/'+a+'-'+b]=contrast(aa[a],aa[b],indices)
                            I=(aa['U_P']-aa['U_G'])-(aa['R_P']-aa['R_G']);ss,pp=indices;boot=np.nanmean(I[ss,pp],(1,2))
                            interactions[key]=dict(mean_absolute=float(np.nanmean(I)) if np.isfinite(I).any() else None,interval95_absolute=np.nanpercentile(boot,[2.5,97.5]).tolist(),per_seed_absolute=np.nanmean(I,1).tolist(),per_place_absolute=np.nanmean(I,0).tolist(),positive_favors_feedback_PE=True)
    def clean(x):
        if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
        if isinstance(x,list):return [clean(v) for v in x]
        if isinstance(x,float) and not np.isfinite(x):return None
        return x
    summary=clean(dict(summaries=summaries,comparisons=comparisons,interactions=interactions,bootstrap=dict(draws=5000,seed=20261113,unit='paired seeds and places; queries averaged within place',all_data_used=True)))
    d.write(d.OUT/'scoring/summary.json',summary)
    primary='development12/fit/missing/roi/centered_log_mse';checks={}
    for pair in [a+'-'+b for a,b in pairs]:
        v=summary['comparisons'][primary+'/'+pair];checks[pair]=dict(practical_supported=v['reduction_percent']>=1 and v['interval95_percent'][0]>0 and all(x>0 for x in v['per_seed_percent']),**v)
    interaction=summary['interactions'][primary]
    d.write(d.OUT/'method_decision.json',dict(status='Measured fixed flags; scientific interpretation and any next decision reserved for original Astra',primary=primary,checks=checks,interaction=interaction,feedback_PE_interval_positive=interaction['interval95_absolute'][0]>0,overall_goal_achieved=False,fresh_confirmation=False,candidate_selected_by_implementation=False))
    lines=['# 全量配对表','', '所有原/其他query、训练原12/新增20、full/missing、支持及零覆盖分层均保留。正降误差偏向前者；log_bias百分比仅算术列，不作效用筛。','','| 分层/指标/比较 | 降误差 % | 95%区间 | 三seed % |','|---|---:|---|---|']
    for key,v in summary['comparisons'].items():lines.append(f"|{key}|{v['reduction_percent']}|{v['interval95_percent']}|{v['per_seed_percent']}|")
    (d.OUT/'report_tables.md').write_text('\n'.join(lines)+'\n')
    return summary

if __name__=='__main__':
    torch.set_num_threads(4);summarize(collect())
