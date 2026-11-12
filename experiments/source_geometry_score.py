"""Paired PE increments and their interaction; all new predictions independently rescored."""
import argparse
import numpy as np
import torch

from experiments.source_geometry_data import *
from experiments.gaussian_scene_score import score_torch,score_numpy,regions,cube,paired,json_finite,METRICS

def collect(partial=False):
    p=plan();places={r['id']:r['place'] for r in p['catalog']};queries={};support={};rows=[];files=0;maximum=0.
    for mode in MODES:
        for seed in SEEDS:
            sources=[(branch_out(mode)/'evaluation'/f'{kind}_{seed}.json',kind,False) for kind in KINDS]
            sources += [(branch_out(mode)/'interventions'/f'G_{seed}.json','G',True)]
            for marker,kind,intervention in sources:
                if not marker.exists():
                    if partial:continue
                    raise FileNotFoundError(marker)
                record=read(marker);assert record['complete'] and record['plan_sha256']==sha256(branch_out(mode)/'plan.json')
                assert sha256(record['prediction_path'])==record['prediction_sha256']
                pred=load(record['prediction_path']);assert len(pred['keys'])==len(set(tuple(k) for k in pred['keys']))
                expected={(m['id'],q,c) for m in p['catalog'] for q in QUERIES for c in CONDITIONS}
                if intervention:expected={(*key,x) for key in expected for x in p['interventions']['kinds']}
                assert set(tuple(k) for k in pred['keys'])==expected
                assert pred['depth_metric'].shape==(len(expected),210,210)
                files+=1
                for key,depth in zip(pred['keys'],pred['depth_metric']):
                    rid,q,c=key[:3]
                    if (rid,q) not in queries:queries[(rid,q)]=load(query_path(rid,q))
                    if rid not in support:support[rid]=load(PARENT_CACHE/'evaluation_support'/f'{rid}.pt')['labels']
                    query=queries[(rid,q)];masks=regions(query,support[rid][str(q)])
                    a=score_torch(depth,query['target'],masks);b=score_numpy(depth,query['target'],masks)
                    for region in a:
                        for metric,value in a[region].items():
                            assert (value is None)==(b[region][metric] is None)
                            if value is not None:maximum=max(maximum,abs(value-b[region][metric]))
                    method=f'{kind}_{mode}'+('_'+key[3] if intervention else '')
                    rows.append(dict(id=rid,place=places[rid],query=q,condition=c,kind=method,seed=seed,metrics=a))
    assert maximum<1e-9
    new_count=len(rows)
    # Inherited metrics have already been independently recomputed; preserve provenance and don't count them as new.
    parent_rows=read(PARENT_OUT/'scoring/metrics.json')['rows'];reused=0
    for r in parent_rows:
        if r['kind'] not in ('L','T','P','G','native_current','official_history','render_depth'):continue
        item=dict(r);item['kind']='parent_'+r['kind'];item['inherited_parent_metric']=True;rows.append(item);reused+=1
    return rows,dict(complete=not partial,new_files=files,new_predictions=new_count,inherited_parent_rows=reused,
        max_abs_metric_difference=maximum,independent_backend='NumPy float64',
        parent_metrics_sha256=sha256(PARENT_OUT/'scoring/metrics.json'),parent_independent_check_sha256=sha256(PARENT_OUT/'scoring/independent_recomputation.json'))

def interaction(arrays,p):
    names=['P_pred','G_pred','P_sensor','G_sensor'];a,b,c,d=[arrays[n] for n in names]
    mean=lambda x:float(np.mean(x))
    delta_pred=mean(a-b);delta_sensor=mean(c-d);value=delta_sensor-delta_pred
    rng=np.random.default_rng(p['bootstrap']['seed']);n=p['bootstrap']['replicates'];si=rng.integers(0,3,(n,3,1));pi=rng.integers(0,a.shape[1],(n,1,a.shape[1]))
    aa,bb,cc,dd=[x[si,pi].mean((1,2)) for x in (a,b,c,d)]
    absolute=(cc-dd)-(aa-bb);relative=100*((cc-dd)/cc-(aa-bb)/aa)
    return dict(delta_pred_absolute=delta_pred,delta_sensor_absolute=delta_sensor,interaction_absolute=value,
        interval95_absolute=np.percentile(absolute,[2.5,97.5]).tolist(),
        interaction_relative_percentage_points=100*((mean(c)-mean(d))/mean(c)-(mean(a)-mean(b))/mean(a)),
        interval95_relative_percentage_points=np.percentile(relative,[2.5,97.5]).tolist(),
        per_seed_interaction_absolute=((c-d)-(a-b)).mean(1).tolist(),per_place_interaction_absolute=((c-d)-(a-b)).mean(0).tolist(),
        direction='Positive: the G-versus-P error reduction is larger with source sensor geometry',
        paired_seed_place_cells=36)

def summarize(rows,p):
    places=[r['place'] for r in p['catalog']];summary={};comparisons={};interactions={}
    methods=[f'{k}_{m}' for m in MODES for k in KINDS]+[f'G_{m}_{i}' for m in MODES for i in p['interventions']['kinds']]
    methods+=['parent_'+k for k in ('L','T','P','G','native_current','official_history','render_depth')]
    pairs=[('G_pred','P_pred'),('G_sensor','P_sensor'),('P_sensor','P_pred'),('G_sensor','G_pred')]
    pairs += [(f'{k}_{m}','parent_L') for m in MODES for k in KINDS]
    pairs += [(f'G_{m}',f'G_{m}_{i}') for m in MODES for i in p['interventions']['kinds']]
    for c in CONDITIONS:
        for region in ('whole','roi','visible','roi_supported','roi_unsupported','whole_supported','whole_unsupported'):
            for metric in METRICS:
                arrays={k:cube(rows,k,c,region,metric,places) for k in methods}
                for k,x in arrays.items():summary[f'{c}/{region}/{metric}/{k}']=dict(mean=float(np.nanmean(x)),per_seed=np.nanmean(x,1).tolist(),per_place=np.nanmean(x,0).tolist())
                if metric=='log_bias':continue
                for a,b in pairs:comparisons[f'{c}/{region}/{metric}/{a}-{b}']=paired(arrays[a],arrays[b],p)
                if all(np.isfinite(arrays[k]).all() for k in ['P_pred','G_pred','P_sensor','G_sensor']):interactions[f'{c}/{region}/{metric}']=interaction(arrays,p)
    return dict(complete=True,plan_sha256=sha256(OUT/'plan.json'),places=places,main_endpoint='missing/roi/centered_log_mse',summaries=summary,comparisons=comparisons,interactions=interactions,
        scope='Old-data controlled extra-source-geometry diagnostic, no RGB-only method promotion. Parent L is matched common adaptation background, not equal source sensor privilege. Other parent methods retain their original writer and are historical backgrounds.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');args=p.parse_args();torch.set_num_threads(4)
    rows,audit=collect(args.partial);folder=OUT/('partial_scoring' if args.partial else 'scoring');folder.mkdir(exist_ok=True)
    write_json(folder/'metrics.json',dict(rows=rows));write_json(folder/'independent_recomputation.json',audit)
    if not args.partial:
        assert audit['new_predictions']==2880 and audit['new_files']==18
        write_json(folder/'summary.json',json_finite(summarize(rows,plan())))
    print(audit,flush=True)

if __name__=='__main__':main()
