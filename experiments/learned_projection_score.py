"""All new exports independently rescored; matched place/seed comparisons."""
import argparse

from experiments.learned_projection_data import *
from experiments.gaussian_scene_score import score_torch,score_numpy,regions,cube,paired,json_finite,METRICS


def collect(partial=False):
    p=plan();rows=[];queries={};supports={};maximum=0.;files=[]
    places={m['id']:m['place'] for m in p['catalog']}
    expected={(m['id'],q,c) for m in p['catalog'] for q in QUERIES for c in CONDITIONS}
    for seed in SEEDS:
        for kind in KINDS:
            marker=OUT/'evaluation'/f'{kind}_{seed}.json'
            if not marker.exists():
                if partial:continue
                raise FileNotFoundError(marker)
            r=read(marker);checkpoint=OUT/f'{kind}_{seed}'/'step_2000.pt'
            assert r['complete'] and r['plan_sha256']==sha256(OUT/'plan.json')
            assert sha256(checkpoint)==r['checkpoint_sha256'] and sha256(r['prediction_path'])==r['prediction_sha256']
            export=load(r['prediction_path'])
            keys=[tuple(k) for k in export['keys']]
            assert len(keys)==96 and len(set(keys))==96 and set(keys)==expected
            assert export['depth_metric'].shape==(96,210,210)
            for key,depth in zip(keys,export['depth_metric']):
                rid,q,c=key
                if (rid,q) not in queries:queries[rid,q]=load(query_path(rid,q))
                if rid not in supports:supports[rid]=load(PARENT_CACHE/'evaluation_support'/f'{rid}.pt')['labels']
                query=queries[rid,q];masks=regions(query,supports[rid][str(q)])
                a=score_torch(depth,query['target'],masks);b=score_numpy(depth,query['target'],masks)
                for region,metrics in a.items():
                    for metric,value in metrics.items():
                        assert (value is None)==(b[region][metric] is None)
                        if value is not None:maximum=max(maximum,abs(value-b[region][metric]))
                rows.append(dict(id=rid,place=places[rid],query=q,condition=c,kind=kind,seed=seed,metrics=a))
            files.append(dict(kind=kind,seed=seed,predictions=96,marker=str(marker),prediction_sha256=r['prediction_sha256']))
    new_count=len(rows)
    origins=[(PARENT_OUT/'scoring/metrics.json',{'L':'parent_L','native_current':'native_current'}),
             (SOURCE_OUT.parent/'scoring/metrics.json',{'P_sensor':'old_P_sensor','G_sensor':'old_fixed_G_sensor'})]
    inherited=[]
    for path,mapping in origins:
        count=0
        for r in read(path)['rows']:
            if r['kind'] not in mapping:continue
            item=dict(r);item['kind']=mapping[r['kind']];item['inherited_metric']=True
            item['inherited_from']=str(path);rows.append(item);count+=1
        inherited.append(dict(path=str(path),sha256=sha256(path),rows=count))
    assert maximum<1e-9
    return rows,dict(complete=len(files)==9,new_files=len(files),new_predictions=new_count,
        files=files,inherited=inherited,max_abs_metric_difference=maximum,independent_backend='NumPy float64',
        scope='Old data sensor diagnostic; inherited rows are not additional evidence')


def summarize(rows):
    p=plan();places=[r['place'] for r in p['catalog']];summaries={};comparisons={}
    methods=list(KINDS)+['parent_L','native_current','old_P_sensor','old_fixed_G_sensor']
    pairs=[('G','P'),('G','S'),('S','P')]+[(k,'parent_L') for k in KINDS]
    pairs += [('G','old_fixed_G_sensor'),('P','old_P_sensor')]
    for condition in CONDITIONS:
        for region in ('whole','roi','visible','roi_supported','roi_unsupported','whole_supported','whole_unsupported'):
            for metric in METRICS:
                arrays={k:cube(rows,k,condition,region,metric,places) for k in methods}
                for kind,x in arrays.items():
                    summaries[f'{condition}/{region}/{metric}/{kind}']=dict(mean=float(np.nanmean(x)),
                        per_seed=np.nanmean(x,1).tolist(),per_place=np.nanmean(x,0).tolist())
                if metric=='log_bias':continue
                for a,b in pairs:
                    comparisons[f'{condition}/{region}/{metric}/{a}-{b}']=paired(arrays[a],arrays[b],p)
    return json_finite(dict(complete=True,places=places,plan_sha256=sha256(OUT/'plan.json'),
        main_endpoint='missing/roi/centered_log_mse',summaries=summaries,comparisons=comparisons,
        scope='Sensor/joint-validity privileged, old data; G/P and G/S are primary; L tests net utility; no new confirmation'))


def main():
    a=argparse.ArgumentParser();a.add_argument('--partial',action='store_true');args=a.parse_args()
    torch.set_num_threads(4);rows,audit=collect(args.partial)
    folder=OUT/('partial_scoring' if args.partial else 'scoring');folder.mkdir(exist_ok=True)
    write_json(folder/'metrics.json',dict(rows=rows));write_json(folder/'independent_recomputation.json',audit)
    if not args.partial:
        assert audit['complete'] and audit['new_predictions']==864
        write_json(folder/'summary.json',summarize(rows))
    print({k:v for k,v in audit.items() if k not in ('files','inherited')},flush=True)


if __name__=='__main__':main()
