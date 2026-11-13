"""Complete used-development score export, matched paired contrasts and interaction."""
import csv,warnings,time
import torch.nn.functional as F
from experiments import native_task_writer_data as d
from experiments.native_task_writer_data import torch,np
from experiments import camera_only_feature_data as baseline
from experiments.gaussian_scene_score import regions,score_torch,score_numpy,METRICS,json_finite
from experiments.camera_only_feature_score import contrast

def collect():
    out=d.OUT/'scoring';out.mkdir(exist_ok=True);rows=[];audit=[];maxdiff=0.;zeros=[];mapping={m['id']:m for m in d.metas()};orig={m['id'] for m in d.plan()['original_training_catalog']};train={m['id'] for m in d.plan()['training_catalog']};maskcache={};targets={}
    for meta in d.metas():
        rid=meta['id'];support=d.load((d.old.CACHE if d.old.is_new(rid) else d.old.VIEW_CACHE)/'evaluation_support'/f'{rid}.pt')['labels']
        for q in d.old.EVAL_QUERIES:
            query=d.load(d.old.query_path(rid,q));alpha=d.load(d.old.feature_path(rid,q))['effective_alpha'].reshape(32,32)
            zero=F.interpolate((alpha<=1e-8)[None,None].float(),(210,210),mode='nearest')[0,0].bool();reg=regions(query,support[str(q)])
            reg.update(roi_old_zero_coverage=query['valid']&query['hidden']&zero,roi_old_positive_coverage=query['valid']&query['hidden']&~zero)
            maskcache[rid,q]=reg;targets[rid,q]=query
            if bool((alpha<=1e-8).all()):zeros.append(dict(id=rid,query=q))
    def append(meta,key,pred,method,seed,newalpha=None):
        nonlocal maxdiff
        rid,q,c=key;query=targets[rid,q];masks=dict(maskcache[rid,q])
        if newalpha is not None:
            covered=F.interpolate((newalpha.reshape(32,32)>1e-8)[None,None].float(),(210,210),mode='nearest')[0,0].bool()
            masks.update(roi_new_render_supported=query['valid']&query['hidden']&covered,roi_new_render_unsupported=query['valid']&query['hidden']&~covered)
        a=score_torch(pred,query['target'],masks);b=score_numpy(pred,query['target'],masks)
        for region,v in a.items():
            for name,x in v.items():
                assert (x is None)==(b[region][name] is None)
                if x is not None:maxdiff=max(maxdiff,abs(x-b[region][name]))
        rows.append(dict(id=rid,place=meta['place'],query=q,condition=c,kind=method,seed=seed,cohort='train' if rid in train else 'development',training_group='original' if rid in orig else 'added' if rid in train else None,query_group='fit' if q in d.old.QUERIES else 'other',metrics=a,inherited=method=='Native'))
    for arm in d.ARMS:
        for seed in d.SEEDS:
            for meta in d.metas():
                rid=meta['id'];r=d.read(d.OUT/'evaluation'/f'{arm}_{seed}_{rid}.json');assert r['complete'] and r['predictions']==16 and d.sha(r['path'])==r['sha256'];payload=d.load(r['path'])
                expected={(rid,q,c) for q in d.old.EVAL_QUERIES for c in d.old.CONDITIONS};assert {tuple(k) for k in payload['keys']}==expected and len(payload['keys'])==16
                for key,pred in zip(payload['keys'],payload['depth_metric']):append(meta,key,pred,arm,seed,payload['alpha_patch'][payload['query_order'].index(key[1])])
                audit.append(dict(path=r['path'],sha256=r['sha256'],predictions=16))
            print(dict(scored=arm,seed=seed),flush=True)
    for meta in d.metas():
        rid=meta['id']
        for q in d.old.EVAL_QUERIES:
            r=d.read(baseline.OUT/'baselines'/f'{rid}_{q}.json')
            for name,c in [('Native','missing'),('Native_full','full')]:
                record=r['methods'][name];assert d.sha(record['path'])==record['sha256'];append(meta,(rid,q,c),torch.from_numpy(np.load(record['path'],allow_pickle=False)),'Native',None)
    inherited=d.read(d.old.OUT/'scoring/metrics.json');refs=[r for r in inherited['rows'] if r['kind'] in ('W_Local','W_Render')];assert len(refs)==4224
    for r in refs:
        rr=dict(r);rr.update(kind='Local' if r['kind']=='W_Local' else 'Render',cohort='train' if r['cohort'].startswith('train') else 'development',training_group='original' if r['id'] in orig else 'added' if r['id'] in train else None,query_group='fit' if r['query'] in d.old.QUERIES else 'other',inherited=True);rows.append(rr)
    assert len(rows)==12672+704+4224 and maxdiff<1e-9
    result=dict(rows=rows,new_predictions=12672,inherited_Native=704,inherited_Local_Render=4224);d.write(out/'metrics.json',result)
    d.write(out/'independent_recomputation.json',dict(complete=True,new_predictions=12672,inherited_Native_recomputed=704,maximum_metric_difference=maxdiff,independent_backend='NumPy float64 independent residual/indexing/centering',new_prediction_files=audit,old_Local_Render_authority=str(d.old.OUT/'scoring/independent_recomputation.json'),old_Local_Render_metrics_sha256=d.sha(d.old.OUT/'scoring/metrics.json'),zero_coverage_queries=zeros,new_render_strata='Arm-dependent descriptive strata; no fixed-mask between-arm causal comparisons'))
    with (out/'per_prediction_metrics.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['kind','seed','id','place','query','condition','cohort','training_group','query_group','region','pixels',*METRICS])
        for r in rows:
            for region,v in r['metrics'].items():writer.writerow([r.get(k) for k in ('kind','seed','id','place','query','condition','cohort','training_group','query_group')]+[region,v['pixels']]+[v[m] for m in METRICS])
    return result

def summarize(data):
    groups=dict(train32=d.plan()['training_catalog'],train_original12=d.plan()['original_training_catalog'],train_added20=d.plan()['added_training_catalog'],development12=d.plan()['catalog']);summaries={};comparisons={};interactions={};rows=data['rows'];methods=[*d.ARMS,'Native','Local','Render']
    pairs=[(f'{row}_{a}',f'{row}_{b}') for row in ('C','J') for a,b in [('G','P'),('G','S'),('S','P')]]+[(f'J_{k}',f'C_{k}') for k in ('P','S','G')]+[(a,b) for a in d.ARMS for b in ('Native','Local','Render')]
    for group,metas in groups.items():
        places=[m['place'] for m in metas];lookup={p:i for i,p in enumerate(places)}
        for qg,qs in [('fit',d.old.QUERIES),('other',d.old.HELDOUT),('pooled',d.old.EVAL_QUERIES)]:
            for condition in d.old.CONDITIONS:
                chosen=[r for r in rows if r['place'] in lookup and r['query'] in qs and r['condition']==condition];cells={}
                for r in chosen:cells.setdefault((r['kind'],r['seed'],r['place']),[]).append(r['metrics'])
                rng=np.random.default_rng(20261113);indices=(rng.integers(0,3,(5000,3,1)),rng.integers(0,len(places),(5000,1,len(places))))
                regions_all=sorted({region for r in chosen for region in r['metrics']})
                for region in regions_all:
                    for metric in METRICS:
                        aa={};key=f'{group}/{qg}/{condition}/{region}/{metric}'
                        for method in methods:
                            a=np.full((3,len(places)),np.nan)
                            for si,seed in enumerate(d.SEEDS):
                                for place,pi in lookup.items():
                                    vals=[v[region][metric] for v in cells.get((method,None if method=='Native' else seed,place),[]) if region in v and v[region][metric] is not None]
                                    if vals:a[si,pi]=np.mean(vals)
                            aa[method]=a
                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore',RuntimeWarning);summaries[key+'/'+method]=dict(mean=float(np.nanmean(a)) if np.isfinite(a).any() else None,per_seed=np.nanmean(a,1).tolist(),per_place=np.nanmean(a,0).tolist())
                        if region.startswith('roi_new_render_'):continue
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore',RuntimeWarning)
                            for a,b in pairs:
                                if np.isfinite(aa[a]).any() and np.isfinite(aa[b]).any():comparisons[key+'/'+a+'-'+b]=contrast(aa[a],aa[b],indices)
                            I=(aa['J_P']-aa['J_G'])-(aa['C_P']-aa['C_G']);ss,pp=indices;boot=np.nanmean(I[ss,pp],(1,2))
                            interactions[key]=dict(mean_absolute=float(np.nanmean(I)) if np.isfinite(I).any() else None,interval95_absolute=np.nanpercentile(boot,[2.5,97.5]).tolist(),per_seed_absolute=np.nanmean(I,1).tolist(),per_place_absolute=np.nanmean(I,0).tolist(),positive_favors_geometry_formation_PE=True)
    summary=json_finite(dict(summaries=summaries,comparisons=comparisons,interactions=interactions,bootstrap=dict(draws=5000,seed=20261113,unit='paired seeds and places; queries averaged within place',all_data_used=True),effect='Total training-chain effect; formed fields and native receiver can differ between arms. Not identical-field direct PE causality.'))
    d.write(d.OUT/'scoring/summary.json',summary);primary='development12/fit/missing/roi/centered_log_mse';checks={}
    for pair in [a+'-'+b for a,b in pairs]:
        v=summary['comparisons'][primary+'/'+pair];checks[pair]=dict(practical_supported=v['reduction_percent']>=1 and v['interval95_percent'][0]>0 and all(x>0 for x in v['per_seed_percent']),**v)
    interaction=summary['interactions'][primary];material={}
    for a,b in [(a,b) for a in d.ARMS for b in ('Native','Local','Render')]:
        v=summary['comparisons'][f'development12/fit/full/whole/absrel/{a}-{b}'];material[a+'-'+b]=dict(material_damage=v['reduction_percent']<=-1 and v['interval95_percent'][1]<0,**v)
    d.write(d.OUT/'method_decision.json',dict(status='Measured prespecified flags; scientific interpretation reserved for original Astra',primary=primary,checks=checks,interaction=interaction,geometry_PE_interaction_supported=interaction['interval95_absolute'][0]>0,geometry_formation_JG_CG_supported=checks['J_G-C_G']['practical_supported'],whole_full_AbsRel_material_damage=material,overall_goal_achieved=False,fresh_confirmation=False,candidate_selected_by_implementation=False))
    lines=['# 全部固定配对表','','同地点同种子，query先地点内平均；正降误差偏向前者。新render支持层是随arm变化的描述分层，不作匹配因果比较。','','| 分层/指标/比较 | 降误差% | 95%区间 | 三seed% |','|---|---:|---|---|']
    for key,v in summary['comparisons'].items():lines.append(f"|{key}|{v['reduction_percent']}|{v['interval95_percent']}|{v['per_seed_percent']}|")
    (d.OUT/'report_tables.md').write_text('\n'.join(lines)+'\n');return summary

if __name__=='__main__':
    torch.set_num_threads(4);start=time.monotonic();summarize(collect());d.write(d.OUT/'scoring/runtime.json',dict(complete=True,seconds=time.monotonic()-start,scope='CPU scoring/bootstrap/export mixed wall time'))
