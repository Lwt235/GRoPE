"""Fixed-place net utility and paired coverage interactions, independently scored."""
import time
from experiments.native_memory_context_data import *
PARENT_OUT=parent.OUT
PARENT_CACHE=parent.CACHE
from experiments.gaussian_scene_score import score_torch,score_numpy,regions,paired,json_finite,METRICS


def collect():
    p=plan();queries={};supports={};rows=[];audit=[];maximum=0.
    mapping={m['id']:m for m in p['training_catalog']+p['catalog']}
    original={m['id'] for m in p['original_training_catalog']};added={m['id'] for m in p['added_training_catalog']}
    inherited=PARENT_OUT/'scoring/metrics.json'
    inherited_hash=sha256(inherited)
    inherited_rows=[r for r in read(inherited)['rows'] if r['kind'] in ['W_Local','W_Render']]
    assert len(inherited_rows)==4224
    for r in inherited_rows:
        rows.append(dict(r,kind='Previous_L' if r['kind']=='W_Local' else 'Previous_R',inherited=True,background=True))
    for meta in p['training_catalog']+p['catalog']:
        rid=meta['id'];sp=(PARENT_CACHE if rid in added else VIEW_CACHE)/'evaluation_support'/f'{rid}.pt'
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
                    rows.append(dict(id=rid,place=mapping[rid]['place'],query=q,condition=condition,kind=kind,seed=seed,
                        cohort=('train_original' if rid in original else 'train_added') if source_cohort=='train' else 'development',
                        view='original' if q in QUERIES else 'heldout',metrics=a,inherited=False))
                audit.append(dict(marker=str(marker),predictions=len(expected),prediction_sha256=record['prediction_sha256']))
    assert len(rows)==10560 and sum(r['predictions'] for r in audit)==6336 and maximum<1e-9
    return rows,dict(complete=True,new_predictions=6336,inherited_predictions=4224,
        independent_backend='NumPy float64',new_max_abs_metric_difference=maximum,
        inherited_source=str(inherited),inherited_source_sha256=sha256(inherited),files=audit,
        inherited_independent_recomputation=str(PARENT_OUT/'scoring/independent_recomputation.json'),new_confirmation=False)


def summarize(rows):
    p=plan();summaries={};comparisons={}
    cohorts={'development':p['catalog'],'train_original':p['original_training_catalog'],
        'train_added':p['added_training_catalog'],'train_all':p['training_catalog']}
    methods=['P','S','G','Previous_L','Previous_R']
    for cohort,metas in cohorts.items():
        places=[m['place'] for m in metas];lookup={place:i for i,place in enumerate(places)}
        rr=[r for r in rows if (r['cohort'].startswith('train') if cohort=='train_all' else r['cohort']==cohort)]
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
                        for a,b in [('G','P'),('G','S'),('S','P'),('P','Previous_L'),('S','Previous_L'),('G','Previous_L'),('P','Previous_R'),('S','Previous_R'),('G','Previous_R')]:
                            if np.isfinite(arrays[a]).any() and np.isfinite(arrays[b]).any():comparisons[key+'/'+a+'-'+b]=paired(arrays[a],arrays[b],p)
    return json_finite(dict(complete=True,primary_endpoint='development/original/missing/roi/centered_log_mse',
        summaries=summaries,comparisons=comparisons,
        development_places=[m['place'] for m in p['catalog']],new_confirmation=False,
        scope='Same source/full geometry and writer; only4 source-neighborhood PE fields differ. Learned content may differ; original renderer weights/coverage remain. Previous_L/R are inherited same32-place/2000step references; no new R/Local fits. All old data with sensor/given cameras; training-place heldout views are not independent scene confirmation.'))


def main():
    torch.set_num_threads(4);begin=time.monotonic();rows,audit=collect();folder=OUT/'scoring'
    write(folder/'metrics.json',dict(rows=rows));write(folder/'independent_recomputation.json',audit)
    write(folder/'summary.json',summarize(rows))
    print(dict(scoring_complete=True,new=6336,inherited=4224,independent_max_abs=audit['new_max_abs_metric_difference'],seconds=time.monotonic()-begin),flush=True)


if __name__=='__main__':main()
