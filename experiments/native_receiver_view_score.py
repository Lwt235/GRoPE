"""Matched old/new view scoring and place/seed paired transfer contrasts."""
from experiments.native_receiver_view_data import *
from experiments.gaussian_scene_score import (score_torch,score_numpy,regions,
    cube,paired,json_finite,METRICS)


def collect():
    p=plan();expected=read(OUT/'identities.json');queries={};supports={}
    places={m['id']:m['place'] for m in p['training_catalog']+p['catalog']}
    inherited_path=Path(p['inherited_metrics_path'])
    assert sha256(inherited_path)==p['inherited_metrics_sha256']
    methods=['A_Local','A_Render','F_Local','F_Render']
    previous={}
    for r in read(inherited_path)['rows']:
        if r['kind'] in methods:
            previous[r['kind'],r['seed'],r['id'],r['query'],r['condition']]=r
    assert len(previous)==2304
    rows=[];audit=[];maxdiff=old_diff=0.;source_conditions=[]
    for meta in p['training_catalog']+p['catalog']:
        rid=meta['id'];supports[rid]=load(CACHE/'evaluation_support'/f'{rid}.pt')['labels']
        for view,qs in [('old',OLD_QUERIES),('new',QUERIES)]:
            for q in qs:
                query=load(query_path(rid,q) if view=='new' else old_query_path(rid,q))
                queries[rid,q]=query
                valid=query['valid'];roi=valid&query['hidden'];support=supports[rid][str(q)]
                source_conditions.append(dict(id=rid,place=meta['place'],view=view,query=q,
                    cohort='train' if meta in p['training_catalog'] else 'development',
                    roi_support_fraction=float((roi&support).sum()/roi.sum()),
                    whole_support_fraction=float((valid&support).sum()/valid.sum())))
    for regime in REGIMES:
        prefix='A_' if regime=='adapted' else 'F_'
        for seed in SEEDS:
            for kind in KINDS:
                name=f'{regime}_{kind}_{seed}'
                records=[('new',read(OUT/'evaluation'/f'{name}.json'),None)]
                for cohort in ['train','development']:
                    path=FROZEN/'evaluation'/f'{kind}_{seed}.json' if regime=='frozen' and cohort=='development' else PARENT/'evaluation'/f'{regime}_{cohort}_{kind}_{seed}.json'
                    records.append(('old',read(path),cohort))
                for view,record,fixed_cohort in records:
                    assert record['complete']
                    assert sha256(record['prediction_path'])==record['prediction_sha256']
                    payload=load(record['prediction_path'])
                    assert len(payload['keys'])==len(set(tuple(k) for k in payload['keys']))
                    if view=='new':
                        assert record['plan_sha256']==sha256(OUT/'plan.json')
                        assert record['checkpoint']==expected['models'][name]
                        assert len(payload['keys'])==192
                    else:assert len(payload['keys'])==96
                    for i,(key,depth) in enumerate(zip(payload['keys'],payload['depth_metric'])):
                        rid,q,condition=key
                        cohort=fixed_cohort or payload['cohorts'][i]
                        metas=p['training_catalog'] if cohort=='train' else p['catalog']
                        assert rid in {m['id'] for m in metas}
                        assert q in (QUERIES if view=='new' else OLD_QUERIES)
                        query=queries[rid,q];masks=regions(query,supports[rid][str(q)])
                        a=score_torch(depth,query['target'],masks)
                        b=score_numpy(depth,query['target'],masks)
                        for region in a:
                            for metric,v in a[region].items():
                                other=b[region][metric]
                                assert (v is None)==(other is None)
                                if v is not None:maxdiff=max(maxdiff,abs(v-other))
                        if view=='old':
                            old=previous[prefix+kind,seed,rid,q,condition]
                            for region,values in old['metrics'].items():
                                for metric,v in values.items():
                                    other=a[region][metric]
                                    assert (v is None)==(other is None)
                                    if v is not None:old_diff=max(old_diff,abs(v-other))
                        rows.append(dict(id=rid,place=places[rid],query=q,condition=condition,
                            kind=prefix+kind,seed=seed,cohort=cohort,view=view,metrics=a,
                            inherited=view=='old'))
                    audit.append(dict(regime=regime,kind=kind,seed=seed,view=view,cohort=fixed_cohort,
                        predictions=len(payload['keys']),prediction_path=record['prediction_path'],
                        prediction_sha256=record['prediction_sha256']))
    assert len(rows)==4608 and maxdiff<1e-9 and old_diff<1e-9
    # Each model/cohort/condition has exactly four queries per place at both positions.
    for method in methods:
        for seed in SEEDS:
            for view in ['old','new']:
                selected=[r for r in rows if r['kind']==method and r['seed']==seed and r['view']==view]
                expected_keys={(m['id'],q,c) for m in p['training_catalog']+p['catalog']
                    for q in (QUERIES if view=='new' else OLD_QUERIES) for c in CONDITIONS}
                assert {(r['id'],r['query'],r['condition']) for r in selected}==expected_keys
    return rows,dict(complete=True,new_predictions=2304,inherited_predictions=2304,
        new_files=12,old_files=24,files=audit,max_abs_independent_difference=maxdiff,
        max_abs_inherited_score_replay_difference=old_diff,independent_backend='NumPy float64',
        inherited_training_support_regions='Additional evaluation-only support regions computed from unchanged old predictions; common inherited scores replay exactly.',
        new_confirmation=False),source_conditions


def absolute_paired(a,b,p):
    delta=a-b; rng=np.random.default_rng(p['bootstrap']['seed'])
    si=rng.integers(0,3,(5000,3,1));pi=rng.integers(0,delta.shape[1],(5000,1,delta.shape[1]))
    boot=np.nanmean(delta[si,pi],axis=(1,2))
    return dict(mean_absolute=float(np.nanmean(delta)),
        interval95_absolute=np.nanpercentile(boot,[2.5,97.5]).tolist(),
        per_seed_absolute=np.nanmean(delta,1).tolist(),per_place_absolute=np.nanmean(delta,0).tolist())


def summarize(rows,source_conditions):
    p=plan();summaries={};comparisons={};arrays={};transfer={};contrasts={}
    methods=['A_Local','A_Render','F_Local','F_Render']
    for cohort in ['train','development']:
        places=[m['place'] for m in p['training_catalog' if cohort=='train' else 'catalog']]
        for view in ['old','new']:
            rr=[r for r in rows if r['cohort']==cohort and r['view']==view]
            for condition in CONDITIONS:
                for region in ['whole','roi','visible','roi_supported','roi_unsupported','whole_supported','whole_unsupported']:
                    for metric in METRICS:
                        key=f'{cohort}/{view}/{condition}/{region}/{metric}'
                        arrays[key]={k:cube(rr,k,condition,region,metric,places) for k in methods}
                        for kind,a in arrays[key].items():
                            summaries[key+'/'+kind]=dict(mean=float(np.nanmean(a)),per_seed=np.nanmean(a,1).tolist(),per_place=np.nanmean(a,0).tolist())
                        if metric=='log_bias':continue
                        for a,b in [('A_Render','A_Local'),('F_Render','F_Local'),('A_Render','F_Render'),('A_Local','F_Local')]:
                            comparisons[key+'/'+a+'-'+b]=paired(arrays[key][a],arrays[key][b],p)
        for condition in CONDITIONS:
            for region in ['whole','roi','visible','roi_supported','roi_unsupported','whole_supported','whole_unsupported']:
                for metric in METRICS:
                    for prefix in ['A_','F_']:
                        old=arrays[f'{cohort}/old/{condition}/{region}/{metric}']
                        new=arrays[f'{cohort}/new/{condition}/{region}/{metric}']
                        od=old[prefix+'Render']-old[prefix+'Local'];nd=new[prefix+'Render']-new[prefix+'Local']
                        key=f'{cohort}/{condition}/{region}/{metric}/{prefix}Render-Local'
                        transfer[key]=absolute_paired(nd,od,p)
                        transfer[key].update(old_difference=float(np.nanmean(od)),new_difference=float(np.nanmean(nd)),
                            positive_means_advantage_erodes=True)
    # Matched seed draws, independent training/development place draws; new/old paired within place.
    for condition in CONDITIONS:
        for region in ['whole','roi','visible','roi_supported','roi_unsupported','whole_supported','whole_unsupported']:
            for metric in METRICS:
                for prefix in ['A_','F_']:
                    key=f'{condition}/{region}/{metric}/{prefix}Render-Local'
                    ds={}
                    for cohort in ['train','development']:
                        old=arrays[f'{cohort}/old/{condition}/{region}/{metric}'];new=arrays[f'{cohort}/new/{condition}/{region}/{metric}']
                        ds[cohort]=(new[prefix+'Render']-new[prefix+'Local'])-(old[prefix+'Render']-old[prefix+'Local'])
                    rng=np.random.default_rng(p['bootstrap']['seed']);si=rng.integers(0,3,(5000,3,1))
                    ti=rng.integers(0,12,(5000,1,12));di=rng.integers(0,12,(5000,1,12))
                    boot=np.nanmean(ds['train'][si,ti],axis=(1,2))-np.nanmean(ds['development'][si,di],axis=(1,2))
                    contrasts[key]=dict(mean_absolute=float(np.nanmean(ds['train'])-np.nanmean(ds['development'])),
                        interval95_absolute=np.nanpercentile(boot,[2.5,97.5]).tolist(),
                        per_seed_absolute=(np.nanmean(ds['train'],1)-np.nanmean(ds['development'],1)).tolist(),
                        positive_means_more_training_query_erosion=True)
    support_summary={}
    for cohort in ['train','development']:
        for view in ['old','new']:
            rr=[r for r in source_conditions if r['cohort']==cohort and r['view']==view]
            support_summary[cohort+'/'+view]={key:float(np.mean([r[key] for r in rr])) for key in ['roi_support_fraction','whole_support_fraction']}
    return json_finite(dict(complete=True,plan_sha256=sha256(OUT/'plan.json'),summaries=summaries,
        comparisons=comparisons,query_transfer=transfer,train_minus_development_transfer=contrasts,
        source_support=support_summary,new_confirmation=False,training_steps=0,
        scope='Fixed source memories and model weights; same-clip query hold-out, old places. O4/O5 evidence only; no O6 PE increment.'))


def main():
    torch.set_num_threads(4);begin=time.monotonic();rows,audit,source=collect();folder=OUT/'scoring'
    write_json(folder/'metrics.json',dict(rows=rows));write_json(folder/'independent_recomputation.json',audit)
    write_json(folder/'source_conditions.json',dict(queries=source))
    write_json(folder/'summary.json',summarize(rows,source))
    print(dict(scoring_complete=True,new_predictions=2304,inherited_predictions=2304,
        independent_max_abs=audit['max_abs_independent_difference'],old_replay_max_abs=audit['max_abs_inherited_score_replay_difference'],seconds=time.monotonic()-begin),flush=True)


if __name__=='__main__':main()
