"""Independent scoring of every new native output and matched utility contrasts."""
import argparse

from experiments.rendered_feature_data import *
from experiments.gaussian_scene_score import (score_torch,score_numpy,regions,cube,
    paired,json_finite,METRICS)


def collect(partial=False):
    p=plan();rows=[];queries={};supports={};maximum=0.;files=[]
    places={m['id']:m['place'] for m in p['catalog']}
    expected={(m['id'],q,c) for m in p['catalog'] for q in QUERIES for c in CONDITIONS}
    sources=[]
    for seed in SEEDS:
        sources.extend((OUT/'evaluation'/f'{kind}_{seed}.json',kind,seed,False) for kind in KINDS)
        sources.append((OUT/'interventions'/f'Render_{seed}.json','Render_shuffle_binding',seed,True))
    for marker,kind,seed,intervention in sources:
        if not marker.exists():
            if partial:continue
            raise FileNotFoundError(marker)
        r=read(marker);checkpoint=OUT/f"{'Render' if intervention else kind}_{seed}"/'step_2000.pt'
        assert r['complete'] and r['plan_sha256']==sha256(OUT/'plan.json')
        assert sha256(checkpoint)==r['checkpoint_sha256']
        assert sha256(r['prediction_path'])==r['prediction_sha256']
        export=load(r['prediction_path']);keys=[tuple(k) for k in export['keys']]
        assert len(keys)==96 and len(set(keys))==96
        assert {k[:3] for k in keys}==expected
        if intervention:assert all(k[3]=='shuffle_binding' for k in keys)
        assert export['depth_metric'].shape==(96,210,210)
        for key,depth in zip(keys,export['depth_metric']):
            rid,q,c=key[:3]
            if (rid,q) not in queries:queries[rid,q]=load(query_path(rid,q))
            if rid not in supports:supports[rid]=load(PARENT_CACHE/'evaluation_support'/f'{rid}.pt')['labels']
            query=queries[rid,q];masks=regions(query,supports[rid][str(q)])
            a=score_torch(depth,query['target'],masks);b=score_numpy(depth,query['target'],masks)
            for region,metrics in a.items():
                for metric,value in metrics.items():
                    assert (value is None)==(b[region][metric] is None)
                    if value is not None:maximum=max(maximum,abs(value-b[region][metric]))
            rows.append(dict(id=rid,place=places[rid],query=q,condition=c,kind=kind,seed=seed,metrics=a))
        files.append(dict(kind=kind,seed=seed,intervention=intervention,predictions=96,
            marker=str(marker),prediction_sha256=r['prediction_sha256']))
    count=len(rows);inherited=[]
    origins=[(PARENT_OUT/'scoring/metrics.json',{'L':'parent_L','native_current':'native_current'}),
        (PROJECTION_OUT/'scoring/metrics.json',{'P':'previous_P','S':'previous_S','G':'previous_G'})]
    for path,mapping in origins:
        n=0
        for old in read(path)['rows']:
            if old['kind'] not in mapping:continue
            item=dict(old);item.update(kind=mapping[old['kind']],inherited_metric=True,inherited_from=str(path))
            rows.append(item);n+=1
        inherited.append(dict(path=str(path),sha256=sha256(path),rows=n))
    assert maximum<1e-9
    return rows,dict(complete=len(files)==12,new_files=len(files),new_predictions=count,
        files=files,inherited=inherited,max_abs_metric_difference=maximum,
        independent_backend='NumPy float64',scope='All new predictions rescored; inherited rows separately labeled, no fresh confirmation.')


def summarize(rows,complete=True):
    p=plan();places=[m['place'] for m in p['catalog']];summaries={};comparisons={}
    methods=list(KINDS)+['Render_shuffle_binding','parent_L','native_current','previous_P','previous_S','previous_G']
    pairs=[('Render','Local'),('Render','Global'),('Global','Local'),('Render','Render_shuffle_binding'),
        ('Local','parent_L'),('Render','previous_P'),('Render','native_current')]
    for condition in CONDITIONS:
        for region in ('whole','roi','visible','roi_supported','roi_unsupported','whole_supported','whole_unsupported'):
            for metric in METRICS:
                arrays={k:cube(rows,k,condition,region,metric,places) for k in methods}
                for kind,array in arrays.items():
                    if not np.isfinite(array).any():continue
                    summaries[f'{condition}/{region}/{metric}/{kind}']=dict(mean=float(np.nanmean(array)),
                        per_seed=np.nanmean(array,1).tolist(),per_place=np.nanmean(array,0).tolist())
                if metric=='log_bias':continue
                for a,b in pairs:
                    if not (np.isfinite(arrays[a]).any() and np.isfinite(arrays[b]).any()):continue
                    comparisons[f'{condition}/{region}/{metric}/{a}-{b}']=paired(arrays[a],arrays[b],p)
    return json_finite(dict(complete=complete,places=places,plan_sha256=sha256(OUT/'plan.json'),
        main_endpoint=p['main_endpoint'],summaries=summaries,comparisons=comparisons,
        primary_pairs=p['primary_pairs'],
        scope='Old sensor/joint-validity privileged data. New matched Local/Global/Render contrasts primary; inherited interface comparisons descriptive; shuffle is dependency, not utility. No O6 PE claim.'))


def main():
    a=argparse.ArgumentParser();a.add_argument('--partial',action='store_true');args=a.parse_args()
    torch.set_num_threads(4);rows,audit=collect(args.partial)
    folder=OUT/('partial_scoring' if args.partial else 'scoring');folder.mkdir(exist_ok=True)
    write_json(folder/'metrics.json',dict(rows=rows));write_json(folder/'independent_recomputation.json',audit)
    if not args.partial:assert audit['complete'] and audit['new_predictions']==1152
    write_json(folder/'summary.json',summarize(rows,audit['complete']))
    print({k:v for k,v in audit.items() if k not in ('files','inherited')},flush=True)


if __name__=='__main__':main()
