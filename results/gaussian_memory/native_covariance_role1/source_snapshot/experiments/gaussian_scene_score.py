"""Place-equal evaluation with a separate NumPy recomputation of every metric."""
import argparse
import json

import numpy as np
import torch

from experiments.gaussian_scene_data import OUT,CACHE,query_path,read,KINDS,SEEDS
from experiments.gaussian_scene_bind import load_tensor
from experiments.gaussian_scene_config import freeze
from experiments.gaussian_memory_incremental_data import write_json
from experiments.gaussian_memory_setup import sha256

METRICS=('log_mse','centered_log_mse','bias_squared','log_bias','absrel','rmse_m')


def regions(query,support,coverage=None):
    valid=query['valid'];roi=query['hidden']
    masks=dict(whole=valid,roi=valid&roi,visible=valid&~roi,
               roi_supported=valid&roi&support,roi_unsupported=valid&roi&~support,
               whole_supported=valid&support,whole_unsupported=valid&~support)
    if coverage is not None:masks['render_covered']=valid&coverage
    return masks


def score_torch(pred,target,masks):
    assert pred.shape==target.shape==(210,210)
    assert torch.isfinite(pred).all() and (pred>0).all()
    pred=pred.double();target=target.double();residual=(pred.clamp_min(1e-6)/target).log();error=pred-target
    result={}
    for name,mask in masks.items():
        count=int(mask.sum());values=dict(pixels=count)
        if count:
            r=residual[mask];e=error[mask]
            values.update(log_mse=float(r.square().mean()),centered_log_mse=float(r.var(unbiased=False)),
                          bias_squared=float(r.mean().square()),log_bias=float(r.mean()),
                          absrel=float((e.abs()/target[mask]).mean()),rmse_m=float(e.square().mean().sqrt()))
        else:values.update({key:None for key in METRICS})
        result[name]=values
    return result


def score_numpy(pred,target,masks):
    # Independent arithmetic, indexing and centering; no reuse of Torch residuals.
    a=pred.numpy().astype(np.float64);b=target.numpy().astype(np.float64)
    result={}
    for name,mask in masks.items():
        take=mask.numpy();n=int(np.count_nonzero(take));values=dict(pixels=n)
        if n:
            x=np.log(np.maximum(a[take],1e-6)/b[take]);delta=a[take]-b[take];mean=np.sum(x)/n
            values.update(log_mse=float(np.sum(x*x)/n),centered_log_mse=float(np.sum((x-mean)**2)/n),
                          bias_squared=float(mean*mean),log_bias=float(mean),absrel=float(np.sum(np.abs(delta)/b[take])/n),
                          rmse_m=float(np.sqrt(np.sum(delta*delta)/n)))
        else:values.update({key:None for key in METRICS})
        result[name]=values
    return result


def prediction_sources(p,partial=False):
    result=[]
    for seed in SEEDS:
        for kind in KINDS:
            path=OUT/'evaluation'/f'{kind}_{seed}.json'
            if not path.exists():
                if partial:continue
                raise FileNotFoundError(path)
            record=read(path);result.append((record,kind,seed,False))
        path=OUT/'interventions'/f'G_{seed}.json'
        if path.exists():result.append((read(path),'G',seed,True))
        elif not partial:raise FileNotFoundError(path)
    for name in ('native','renderer'):
        path=OUT/'references'/f'{name}.json'
        if path.exists():result.append((read(path),None,None,False))
        elif not partial:raise FileNotFoundError(path)
    return result


def collect(p,partial=False):
    rows=[];max_difference=0.;query_cache={};support_cache={};sources=prediction_sources(p,partial)
    places={m['id']:m['place'] for m in p['catalog']}
    for record,kind,seed,intervention in sources:
        assert record['complete'] and record['plan_sha256']==sha256(OUT/'plan.json')
        assert sha256(record['prediction_path'])==record['prediction_sha256']
        predictions=load_tensor(record['prediction_path']);assert len(set(tuple(k) for k in predictions['keys']))==len(predictions['keys'])
        for i,(key,pred) in enumerate(zip(predictions['keys'],predictions['depth_metric'])):
            rid,q,condition=key[:3]
            if (rid,q) not in query_cache:query_cache[(rid,q)]=load_tensor(query_path(rid,q))
            if rid not in support_cache:support_cache[rid]=load_tensor(CACHE/'evaluation_support'/f'{rid}.pt')['labels']
            query=query_cache[(rid,q)];support=support_cache[rid][str(q)]
            coverage=predictions['alpha'][i]>.0001 if 'alpha' in predictions else None
            masks=regions(query,support,coverage);values=score_torch(pred,query['target'],masks)
            independent=score_numpy(pred,query['target'],masks)
            for name,metrics in values.items():
                for metric,value in metrics.items():
                    other=independent[name][metric]
                    assert (value is None)==(other is None)
                    if value is not None:max_difference=max(max_difference,abs(value-other))
            assert values['whole']['pixels']>0 and values['roi']['pixels']>0
            method=(f'G_{key[3]}' if intervention else kind) if kind is not None else key[3]
            rows.append(dict(id=rid,place=places[rid],query=q,condition=condition,kind=method,seed=seed,metrics=values))
    assert max_difference<1e-9
    return rows,dict(complete=not partial,files=len(sources),predictions=len(rows),max_abs_metric_difference=max_difference,
                     independent_backend='NumPy float64',all_exported_predictions_recomputed=True)


def cube(rows,kind,condition,region,metric,places):
    values=np.full((3,len(places)),np.nan)
    for si,seed in enumerate(SEEDS):
        for pi,place in enumerate(places):
            group=[r['metrics'][region][metric] for r in rows if r['kind']==kind and r['condition']==condition and r['place']==place
                   and (r['seed']==seed or r['seed'] is None) and region in r['metrics']]
            group=[x for x in group if x is not None]
            if group:values[si,pi]=np.mean(group)
    return values


def paired(a,b,p):
    # Lower error is better. All comparisons keep the same sampled seed/place units.
    rng=np.random.default_rng(p['bootstrap']['seed'])
    seed=rng.integers(0,3,(p['bootstrap']['replicates'],3,1))
    place=rng.integers(0,a.shape[1],(p['bootstrap']['replicates'],1,a.shape[1]))
    with np.errstate(invalid='ignore',divide='ignore'):
        x=np.nanmean(a[seed,place],axis=(1,2));y=np.nanmean(b[seed,place],axis=(1,2))
        samples=100*(y-x)/y
    x=float(np.nanmean(a));y=float(np.nanmean(b))
    with np.errstate(invalid='ignore',divide='ignore'):
        effect=float(np.divide(100*(y-x),y))
    return dict(candidate=x,control=y,absolute_difference=x-y,relative_error_reduction_percent=effect,
                interval95_percent=np.nanpercentile(samples,[2.5,97.5]).tolist(),
                per_seed_relative_percent=(100*(np.nanmean(b,axis=1)-np.nanmean(a,axis=1))/np.nanmean(b,axis=1)).tolist(),
                per_place_relative_percent=(100*(np.nanmean(b,axis=0)-np.nanmean(a,axis=0))/np.nanmean(b,axis=0)).tolist(),
                paired_cells=int((np.isfinite(a)&np.isfinite(b)).sum()))


def json_finite(value):
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,dict):return {key:json_finite(item) for key,item in value.items()}
    if isinstance(value,list):return [json_finite(item) for item in value]
    return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4);p=freeze();rows,audit=collect(p,args.partial)
    folder=OUT/('partial_scoring' if args.partial else 'scoring');folder.mkdir(exist_ok=True)
    write_json(folder/'metrics.json',dict(rows=rows))
    write_json(folder/'independent_recomputation.json',audit)
    if not args.partial:
        from experiments.gaussian_scene_training_summary import main as summarize_training
        summarize_training()
        assert len(rows)==2592,(len(rows),'expected1440adapted+288reference+864intervention')
        places=[m['place'] for m in p['catalog']];comparisons={};summaries={}
        methods=list(KINDS)+['native_current','official_history','render_depth']+['G_'+name for name in p['interventions']['kinds']]
        for condition in p['conditions']:
            for region in ('whole','roi','visible','roi_supported','roi_unsupported','whole_supported','whole_unsupported'):
                for metric in METRICS:
                    arrays={kind:cube(rows,kind,condition,region,metric,places) for kind in methods}
                    for kind,array in arrays.items():
                        summaries[f'{condition}/{region}/{metric}/{kind}']=dict(mean=float(np.nanmean(array)),
                            per_seed=np.nanmean(array,axis=1).tolist(),per_place=np.nanmean(array,axis=0).tolist())
                    if metric=='log_bias':continue
                    for candidate,control in [('G','P'),('G','T'),('G','L'),('G','C'),('P','L'),('P','T'),('C','L'),('C','T'),
                                              ('G','native_current'),('G','official_history'),('G','render_depth')]+[('G','G_'+name) for name in p['interventions']['kinds']]:
                        comparisons[f'{condition}/{region}/{metric}/{candidate}-{control}']=paired(arrays[candidate],arrays[control],p)
        write_json(folder/'summary.json',json_finite(dict(complete=True,plan_sha256=sha256(OUT/'plan.json'),
            main_endpoint='missing/roi/centered_log_mse',places=places,summaries=summaries,comparisons=comparisons)))
    print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
