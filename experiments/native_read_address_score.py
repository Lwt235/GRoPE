"""Independent new predictions and complete fixed paired development contrasts."""
import csv,time,warnings
import torch.nn.functional as F
from experiments import native_read_address_data as d
from experiments.native_read_address_data import torch,np
from experiments.gaussian_scene_score import regions,score_torch,score_numpy,METRICS,json_finite
from experiments.camera_only_feature_score import contrast

def collect():
    out=d.OUT/'scoring';out.mkdir(exist_ok=True);rows=[];audit=[];maximum=0.;zeros=[];masks={};targets={};orig={m['id'] for m in d.plan()['original_training_catalog']};train={m['id'] for m in d.plan()['training_catalog']}
    for meta in d.metas():
        rid=meta['id'];labels=d.load((d.parent.old.CACHE if d.parent.old.is_new(rid) else d.parent.old.VIEW_CACHE)/'evaluation_support'/f'{rid}.pt')['labels']
        for q in d.parent.old.EVAL_QUERIES:
            query=d.load(d.parent.old.query_path(rid,q));alpha=d.load(d.parent.old.feature_path(rid,q))['effective_alpha'].reshape(32,32);zero=F.interpolate((alpha<=1e-8)[None,None].float(),(210,210),mode='nearest')[0,0].bool();reg=regions(query,labels[str(q)]);reg.update(roi_old_zero_coverage=query['valid']&query['hidden']&zero,roi_old_positive_coverage=query['valid']&query['hidden']&~zero);masks[rid,q]=reg;targets[rid,q]=query
            if (alpha<=1e-8).all():zeros.append(dict(id=rid,query=q))
    for arm in d.ARMS:
        for seed in d.SEEDS:
            for meta in d.metas():
                rid=meta['id'];marker=d.OUT/'evaluation'/f'{arm}_{seed}_{rid}.json';waiting=time.monotonic();last_notice=waiting-60
                while not marker.exists():
                    if time.monotonic()-last_notice>=60:print(dict(waiting_for_completed_evaluation=str(marker),no_model_calls=True),flush=True);last_notice=time.monotonic()
                    time.sleep(5)
                r=d.read(marker);assert r['complete'] and r['predictions']==16 and d.sha(r['path'])==r['sha256'];payload=d.load(r['path']);assert {tuple(k) for k in payload['keys']}=={(rid,q,c) for q in d.parent.old.EVAL_QUERIES for c in d.parent.old.CONDITIONS}
                for key,pred in zip(payload['keys'],payload['depth_metric']):
                    _,q,c=key;query=targets[rid,q];reg=dict(masks[rid,q]);alpha=payload['alpha_patch'][payload['query_order'].index(q)].reshape(32,32);covered=F.interpolate((alpha>1e-8)[None,None].float(),(210,210),mode='nearest')[0,0].bool();reg.update(roi_new_render_supported=query['valid']&query['hidden']&covered,roi_new_render_unsupported=query['valid']&query['hidden']&~covered)
                    a=score_torch(pred,query['target'],reg);b=score_numpy(pred,query['target'],reg)
                    for region,v in a.items():
                        for name,x in v.items():
                            assert (x is None)==(b[region][name] is None)
                            if x is not None:maximum=max(maximum,abs(x-b[region][name]))
                    rows.append(dict(id=rid,place=meta['place'],query=q,condition=c,kind=arm,seed=seed,cohort='train' if rid in train else 'development',training_group='original' if rid in orig else 'added' if rid in train else None,query_group='fit' if q in d.parent.old.QUERIES else 'other',metrics=a,inherited=False))
                audit.append(dict(path=r['path'],sha256=r['sha256'],predictions=16))
            print(dict(scored=arm,seed=seed),flush=True)
    inherited_path=d.parent.OUT/'scoring/metrics.json';inherited=d.read(inherited_path)['rows'];counts={}
    for kind,target in [('C_P','B'),('Native','Native'),('Local','Local'),('Render','Render')]:
        rr=[dict(r,kind=target,inherited=True) for r in inherited if r['kind']==kind];rows.extend(rr);counts[target]=len(rr)
    assert counts==dict(B=2112,Native=704,Local=2112,Render=2112) and len(rows)==13376 and maximum<1e-9
    data=dict(rows=rows,new_predictions=6336,inherited=counts);d.write(out/'metrics.json',data);d.write(out/'independent_recomputation.json',dict(complete=True,new_predictions=6336,maximum_metric_difference=maximum,independent_backend='NumPyfloat64 independent indexing/log residual/centering',new_prediction_files=audit,inherited_metric_authority=str(inherited_path),inherited_metric_sha256=d.sha(inherited_path),inherited_independent_audit=str(d.parent.OUT/'scoring/independent_recomputation.json'),inherited_counts=counts,zero_coverage_queries=zeros,support_scope='Actual full-shape content support shared by A/P/S/B for each seed. Shadow S support affects positional kernel only, not masks or content.'))
    with (out/'per_prediction_metrics.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['kind','seed','id','place','query','condition','cohort','training_group','query_group','region','pixels',*METRICS])
        for r in rows:
            for region,v in r['metrics'].items():writer.writerow([r.get(k) for k in ('kind','seed','id','place','query','condition','cohort','training_group','query_group')]+[region,v['pixels']]+[v[k] for k in METRICS])
    return data

def summarize(data):
    groups=dict(train32=d.plan()['training_catalog'],train_original12=d.plan()['original_training_catalog'],train_added20=d.plan()['added_training_catalog'],development12=d.plan()['catalog']);methods=['A','P','S','B','Native','Local','Render'];pairs=[('A','P'),('A','S'),('P','S')]+[(a,b) for a in d.ARMS for b in ('B','Native','Local','Render')];summaries={};comparisons={};rows=data['rows']
    for group,metas in groups.items():
        places=[m['place'] for m in metas];lookup={p:i for i,p in enumerate(places)}
        for qg,qs in [('fit',d.parent.old.QUERIES),('other',d.parent.old.HELDOUT),('pooled',d.parent.old.EVAL_QUERIES)]:
            for condition in d.parent.old.CONDITIONS:
                chosen=[r for r in rows if r['place'] in lookup and r['query'] in qs and r['condition']==condition];cells={}
                for r in chosen:cells.setdefault((r['kind'],r['seed'],r['place']),[]).append(r['metrics'])
                rng=np.random.default_rng(20261114);indices=(rng.integers(0,3,(5000,3,1)),rng.integers(0,len(places),(5000,1,len(places))))
                for region in sorted({k for r in chosen for k in r['metrics']}):
                    for metric in METRICS:
                        arrays={};key=f'{group}/{qg}/{condition}/{region}/{metric}'
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore',RuntimeWarning)
                            for method in methods:
                                a=np.full((3,len(places)),np.nan)
                                for si,seed in enumerate(d.SEEDS):
                                    for place,pi in lookup.items():
                                        vals=[v[region][metric] for v in cells.get((method,None if method=='Native' else seed,place),[]) if region in v and v[region][metric] is not None]
                                        if vals:a[si,pi]=np.mean(vals)
                                arrays[method]=a;summaries[key+'/'+method]=dict(mean=float(np.nanmean(a)) if np.isfinite(a).any() else None,per_seed=np.nanmean(a,1).tolist(),per_place=np.nanmean(a,0).tolist())
                            if metric=='log_bias':continue
                            for a,b in pairs:
                                if np.isfinite(arrays[a]).any() and np.isfinite(arrays[b]).any():comparisons[key+'/'+a+'-'+b]=contrast(arrays[a],arrays[b],indices)
    summary=json_finite(dict(summaries=summaries,comparisons=comparisons,bootstrap=dict(draws=5000,seed=20261114,unit='Paired seed/place; equal query average within each place'),scope='All old used development; fixed frozen content and B receiver shared between A/P/S; newgamma differs. A/B includes pairwise compute/scalar adaptation; slot-alias is same A, not another fit.',signed_bias='Signed means retained without percent error-reduction on signed bias.'))
    d.write(d.OUT/'scoring/summary.json',summary);primary='development12/fit/missing/roi/centered_log_mse';checks={}
    for a,b in pairs:
        v=summary['comparisons'][primary+'/'+a+'-'+b];checks[a+'-'+b]=dict(practical_supported=v['reduction_percent']>=1 and v['interval95_percent'][0]>0 and all(x>0 for x in v['per_seed_percent']),**v)
    material={}
    for a in d.ARMS:
        for b in ('B','Native','Local','Render'):
            v=summary['comparisons'][f'development12/fit/full/whole/absrel/{a}-{b}'];material[a+'-'+b]=dict(material_damage=v['reduction_percent']<=-1 and v['interval95_percent'][1]<0,**v)
    d.write(d.OUT/'method_decision.json',dict(status='Prespecified measured flags; original Astra interprets and selects scientific branch',primary=primary,checks=checks,required_generic_utility_flags_joint=all(checks[k]['practical_supported'] for k in ('A-P','A-B','A-Native','A-Local')),shape_address_A_S_supported=checks['A-S']['practical_supported'],whole_full_AbsRel_material_damage=material,all_data_used_development=True,fresh_confirmation=False,candidate_selected_by_implementation=False,overall_goal_achieved=False))
    lines=['# 全部固定分层配对表','','正降误差偏向前者；query先地点内等权；5000次配对seed/place区间。B=各seed原C/P。所有数据已用。','','|分层/指标/比较|降误差%|95%区间|三seed%|','|---|---:|---|---|']
    for k,v in comparisons.items():lines.append(f"|{k}|{v['reduction_percent']}|{v['interval95_percent']}|{v['per_seed_percent']}|")
    (d.OUT/'report_tables.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':
    torch.set_num_threads(4);start=time.monotonic();summarize(collect());d.write(d.OUT/'scoring/runtime.json',dict(complete=True,mixed_wall_seconds=time.monotonic()-start,scope='PureCPU independent scoring/bootstrap/export and any wait for completed evaluation markers; no CUDA/model invocation. Not a pure compute-time estimate.'))
