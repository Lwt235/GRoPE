"""Independent all-output scoring, frozen controls and absolute paired interactions."""
import argparse
from experiments.native_receiver_data import *
from experiments.gaussian_scene_score import score_torch,score_numpy,regions,cube,paired,json_finite,METRICS


def collect(partial=False):
    p=plan();places={m['id']:m['place'] for m in p['training_catalog']+p['catalog']}
    rows=[];audit=[];queries={};supports={};maximum=0.
    markers=[]
    for seed in SEEDS:
        for kind in KINDS:
            markers.extend(OUT/'evaluation'/f'{regime}_{cohort}_{kind}_{seed}.json'
                for regime,cohort in [('adapted','development'),('adapted','train'),('frozen','train')])
        markers.append(OUT/'evaluation'/f'adapted_development_Render_{seed}_shuffle.json')
    for path in markers:
        if not path.exists():
            if partial:continue
            raise FileNotFoundError(path)
        marker=read(path);assert marker['complete'] and marker['plan_sha256']==sha256(OUT/'plan.json')
        assert sha256(marker['checkpoint_path'])==marker['checkpoint_sha256']
        assert sha256(marker['prediction_path'])==marker['prediction_sha256']
        payload=load(marker['prediction_path']);assert len(payload['keys'])==len(set(tuple(k) for k in payload['keys']))==96
        metas=p['training_catalog'] if marker['cohort']=='train' else p['catalog']
        expected={(m['id'],q,c) for m in metas for q in QUERIES for c in CONDITIONS}
        assert {tuple(k[:3]) for k in payload['keys']}==expected
        for key,depth in zip(payload['keys'],payload['depth_metric']):
            rid,q,condition=key[:3]
            if (rid,q) not in queries:queries[rid,q]=load(query_path(rid,q))
            query=queries[rid,q]
            if marker['cohort']=='development':
                if rid not in supports:supports[rid]=load(PARENT_CACHE/'evaluation_support'/f'{rid}.pt')['labels']
                masks=regions(query,supports[rid][str(q)])
            else:masks=dict(whole=query['valid'],roi=query['valid']&query['hidden'],visible=query['valid']&~query['hidden'])
            a=score_torch(depth,query['target'],masks);b=score_numpy(depth,query['target'],masks)
            for region,metrics in a.items():
                for metric,value in metrics.items():
                    assert (value is None)==(b[region][metric] is None)
                    if value is not None:maximum=max(maximum,abs(value-b[region][metric]))
            prefix='A_' if marker['regime']=='adapted' else 'F_'
            kind=prefix+marker['kind']+('_shuffle_binding' if marker['shuffle'] else '')
            rows.append(dict(id=rid,place=places[rid],query=q,condition=condition,kind=kind,seed=marker['seed'],cohort=marker['cohort'],metrics=a))
        audit.append(dict(marker=str(path),predictions=96,cohort=marker['cohort'],regime=marker['regime'],shuffle=marker['shuffle'],prediction_sha256=marker['prediction_sha256']))
    inherited=FROZEN/'scoring/metrics.json';n=0
    for old in read(inherited)['rows']:
        if old['kind'] not in KINDS:continue
        rows.append(dict(old,kind='F_'+old['kind'],cohort='development',inherited=True,inherited_from=str(inherited)));n+=1
    assert n==864 and maximum<1e-9
    return rows,dict(complete=len(audit)==30,new_files=len(audit),new_predictions=sum(r['predictions'] for r in audit),
        inherited_development_rows=n,inherited_metrics_sha256=sha256(inherited),files=audit,
        independent_backend='NumPy float64',max_abs_metric_difference=maximum,new_confirmation=False)


def interaction(ar,ac,fr,fc,p):
    delta=(ar-ac)-(fr-fc);rng=np.random.default_rng(p['bootstrap']['seed'])
    seeds=rng.integers(0,3,(5000,3,1));places=rng.integers(0,delta.shape[1],(5000,1,delta.shape[1]))
    boot=np.nanmean(delta[seeds,places],axis=(1,2))
    return json_finite(dict(mean_absolute=float(np.nanmean(delta)),interval95_absolute=np.nanpercentile(boot,[2.5,97.5]).tolist(),
        per_seed_absolute=np.nanmean(delta,1).tolist(),per_place_absolute=np.nanmean(delta,0).tolist(),
        paired_cells=int(np.isfinite(delta).sum()),negative_favors_adaptation=True,
        scope='Difference of matched errors, log-MSE/AbsRel units as named. Interaction improvement is not sufficient for net method utility.'))


def summarize(rows,complete):
    p=plan();summaries={};comparisons={};interactions={};methods=['A_'+k for k in KINDS]+['F_'+k for k in KINDS]+['A_Render_shuffle_binding']
    pairs=[('A_Render','A_Local'),('A_Render','A_Global'),('A_Global','A_Local'),
        ('F_Render','F_Local'),('F_Render','F_Global'),('A_Render','F_Render'),('A_Local','F_Local'),
        ('A_Global','F_Global'),('A_Render','A_Render_shuffle_binding')]
    for cohort in ['development','train']:
        rr=[r for r in rows if r['cohort']==cohort];places=[m['place'] for m in p['catalog' if cohort=='development' else 'training_catalog']]
        for condition in CONDITIONS:
            for region in (['whole','roi','visible','roi_supported','roi_unsupported','whole_supported','whole_unsupported'] if cohort=='development' else ['whole','roi','visible']):
                for metric in METRICS:
                    arrays={k:cube(rr,k,condition,region,metric,places) for k in methods}
                    key=f'{cohort}/{condition}/{region}/{metric}'
                    for k,a in arrays.items():
                        if np.isfinite(a).any():summaries[key+'/'+k]=dict(mean=float(np.nanmean(a)),per_seed=np.nanmean(a,1).tolist(),per_place=np.nanmean(a,0).tolist())
                    if metric=='log_bias':continue
                    for a,b in pairs:
                        if np.isfinite(arrays[a]).any() and np.isfinite(arrays[b]).any():comparisons[key+'/'+a+'-'+b]=paired(arrays[a],arrays[b],p)
                    for control in ['Local','Global']:
                        aa=[arrays[k] for k in ['A_Render','A_'+control,'F_Render','F_'+control]]
                        if all(np.isfinite(a).any() for a in aa):interactions[key+'/Render-'+control]=interaction(*aa,p)
    return json_finite(dict(complete=complete,main_endpoint='development/missing/roi/centered_log_mse',
        plan_sha256=sha256(OUT/'plan.json'),summaries=summaries,comparisons=comparisons,interactions=interactions,
        development_places=[m['place'] for m in p['catalog']],training_places=[m['place'] for m in p['training_catalog']],
        scope='Same old sensor-assisted static scenes. Primary within adapted Local/Global/Render utility and interaction with inherited frozen development row. Train set evaluates fitted inputs, no held-out or fresh claim. No O6 PE claim.'))


def main():
    a=argparse.ArgumentParser();a.add_argument('--partial',action='store_true');args=a.parse_args()
    torch.set_num_threads(4);rows,audit=collect(args.partial)
    if not args.partial:assert audit['complete'] and audit['new_predictions']==2880
    folder=OUT/('partial_scoring' if args.partial else 'scoring')
    write_json(folder/'metrics.json',dict(rows=rows));write_json(folder/'independent_recomputation.json',audit)
    write_json(folder/'summary.json',summarize(rows,audit['complete']))
    print({k:v for k,v in audit.items() if k!='files'},flush=True)


if __name__=='__main__':main()
