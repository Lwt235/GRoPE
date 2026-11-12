"""Fixed-place net utility and paired coverage interactions, independently scored."""
import time
from experiments.native_coverage_data import *
from experiments.gaussian_scene_score import score_torch,score_numpy,regions,paired,json_finite,METRICS


def collect():
    p=plan();queries={};supports={};rows=[];audit=[];maximum=0.
    mapping={m['id']:m for m in p['training_catalog']+p['catalog']}
    original={m['id'] for m in p['original_training_catalog']};added={m['id'] for m in p['added_training_catalog']}
    inherited=Path(p['parent_metrics_path']);assert sha256(inherited)==p['parent_metrics_sha256']
    inherited_rows=[r for r in read(inherited)['rows'] if r['kind'] in ['A_Local','A_Render']]
    assert len(inherited_rows)==2304
    for r in inherited_rows:
        rows.append(dict(r,kind='N_'+r['kind'][2:],cohort='train_original' if r['cohort']=='train' else 'development',
            view='original' if r['query'] in QUERIES else 'heldout',inherited=True))
    for meta in p['training_catalog']+p['catalog']:
        rid=meta['id'];sp=(CACHE if rid in added else VIEW_CACHE)/'evaluation_support'/f'{rid}.pt'
        supports[rid]=load(sp)['labels']
        for q in EVAL_QUERIES:queries[rid,q]=load(query_path(rid,q))
    for seed in SEEDS:
        for kind in KINDS:
            status=read(OUT/f'{kind}_{seed}'/'status.json');assert status['complete'] and status['steps']==2000
            for source_cohort,metas in [('development',p['catalog']),('train',p['training_catalog'])]:
                marker=OUT/'evaluation'/f'{source_cohort}_{kind}_{seed}.json';record=read(marker)
                assert record['complete'] and record['plan_sha256']==sha256(OUT/'plan.json')
                assert record['checkpoint_sha256']==status['checkpoint_sha256']
                assert sha256(record['prediction_path'])==record['prediction_sha256']
                payload=load(record['prediction_path'])
                expected={(m['id'],q,c) for m in metas for q in EVAL_QUERIES for c in CONDITIONS}
                assert {tuple(k) for k in payload['keys']}==expected and len(payload['keys'])==len(expected)
                for key,depth in zip(payload['keys'],payload['depth_metric']):
                    rid,q,condition=key;query=queries[rid,q];masks=regions(query,supports[rid][str(q)])
                    a=score_torch(depth,query['target'],masks);b=score_numpy(depth,query['target'],masks)
                    for region in a:
                        for metric,v in a[region].items():
                            other=b[region][metric];assert (v is None)==(other is None)
                            if v is not None:maximum=max(maximum,abs(v-other))
                    rows.append(dict(id=rid,place=mapping[rid]['place'],query=q,condition=condition,kind='W_'+kind,seed=seed,
                        cohort=('train_original' if rid in original else 'train_added') if source_cohort=='train' else 'development',
                        view='original' if q in QUERIES else 'heldout',metrics=a,inherited=False))
                audit.append(dict(marker=str(marker),predictions=len(expected),prediction_sha256=record['prediction_sha256']))
    assert len(rows)==6528 and sum(r['predictions'] for r in audit)==4224 and maximum<1e-9
    return rows,dict(complete=True,new_predictions=4224,inherited_predictions=2304,
        independent_backend='NumPy float64',new_max_abs_metric_difference=maximum,
        inherited_source=str(inherited),inherited_source_sha256=sha256(inherited),files=audit,
        inherited_independent_recomputation=str(VIEW_OUT/'scoring/independent_recomputation.json'),new_confirmation=False)


def interaction(wr,wl,nr,nl,p):
    delta=(wr-wl)-(nr-nl);rng=np.random.default_rng(p['bootstrap']['seed'])
    si=rng.integers(0,3,(5000,3,1));pi=rng.integers(0,delta.shape[1],(5000,1,delta.shape[1]))
    boot=np.nanmean(delta[si,pi],axis=(1,2))
    return dict(mean_absolute=float(np.nanmean(delta)),interval95_absolute=np.nanpercentile(boot,[2.5,97.5]).tolist(),
        per_seed_absolute=np.nanmean(delta,1).tolist(),per_place_absolute=np.nanmean(delta,0).tolist(),
        narrow_difference=float(np.nanmean(nr-nl)),wide_difference=float(np.nanmean(wr-wl)),negative_favors_coverage=True)


def summarize(rows):
    p=plan();summaries={};comparisons={};interactions={}
    cohorts={'development':p['catalog'],'train_original':p['original_training_catalog'],
        'train_added':p['added_training_catalog'],'train_all':p['training_catalog']}
    methods=['N_Local','N_Render','W_Local','W_Render']
    for cohort,metas in cohorts.items():
        places=[m['place'] for m in metas];lookup={place:i for i,place in enumerate(places)}
        rr=[r for r in rows if (r['cohort'].startswith('train') if cohort=='train_all' else r['cohort']==cohort)]
        if cohort=='train_all':rr=[r for r in rr if r['kind'].startswith('W_')]
        for view in ['original','heldout','pooled']:
            selected=[r for r in rr if view=='pooled' or r['view']==view]
            for condition in CONDITIONS:
                group=[r for r in selected if r['condition']==condition]
                # Index once; no repeated full-row scans inside each metric/cell.
                cells={}
                for r in group:cells.setdefault((r['kind'],SEEDS.index(r['seed']),lookup[r['place']]),[]).append(r['metrics'])
                for region in ['whole','roi','visible','roi_supported','roi_unsupported','whole_supported','whole_unsupported']:
                    for metric in METRICS:
                        arrays={k:np.full((3,len(places)),np.nan) for k in methods}
                        for (kind,si,pi),values in cells.items():
                            v=[x[region][metric] for x in values if x[region][metric] is not None]
                            if v:arrays[kind][si,pi]=np.mean(v)
                        key=f'{cohort}/{view}/{condition}/{region}/{metric}'
                        for kind,a in arrays.items():
                            if np.isfinite(a).all() or np.isfinite(a).any():
                                summaries[key+'/'+kind]=dict(mean=float(np.nanmean(a)),per_seed=np.nanmean(a,1).tolist(),per_place=np.nanmean(a,0).tolist())
                        if metric=='log_bias':continue
                        for a,b in [('W_Render','W_Local'),('N_Render','N_Local'),('W_Render','N_Render'),('W_Local','N_Local')]:
                            if np.isfinite(arrays[a]).any() and np.isfinite(arrays[b]).any():comparisons[key+'/'+a+'-'+b]=paired(arrays[a],arrays[b],p)
                        if cohort in ['development','train_original']:
                            aa=[arrays[k] for k in ['W_Render','W_Local','N_Render','N_Local']]
                            interactions[key]=interaction(*aa,p)
    return json_finite(dict(complete=True,primary_endpoint='development/original/missing/roi/centered_log_mse',
        summaries=summaries,comparisons=comparisons,coverage_interactions=interactions,
        development_places=[m['place'] for m in p['catalog']],new_confirmation=False,
        scope='32vs12training places at fixed2000updates; joint coverage/sample change, same architecture; no O6 increment. Added training places are not held-out confirmation for expanded models.'))


def main():
    torch.set_num_threads(4);begin=time.monotonic();rows,audit=collect();folder=OUT/'scoring'
    write_json(folder/'metrics.json',dict(rows=rows));write_json(folder/'independent_recomputation.json',audit)
    write_json(folder/'summary.json',summarize(rows))
    print(dict(scoring_complete=True,new=4224,inherited=2304,independent_max_abs=audit['new_max_abs_metric_difference'],seconds=time.monotonic()-begin),flush=True)


if __name__=='__main__':main()
