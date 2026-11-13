"""All new exports: independent float64 scoring and fixed seed/place bootstrap."""
import csv
import time
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from experiments import native_feature_field_data as d
from experiments.gaussian_scene_score import score_torch,score_numpy,regions,METRICS,json_finite


def feature_torch(pred,target,mask):
    a=pred.double()[mask];b=target.double()[mask];delta=a-b
    if len(a)==0:return dict(patches=0,raw_mse=None,spatial_mse=None,bias_squared=None,cosine=None)
    bias=delta.mean(0)
    return dict(patches=len(a),raw_mse=float(delta.square().mean()),spatial_mse=float((delta-bias).square().mean()),bias_squared=float(bias.square().mean()),cosine=float(F.cosine_similarity(a,b,dim=-1,eps=1e-8).mean()))

def feature_numpy(pred,target,mask):
    a=pred.numpy().astype(np.float64)[mask.numpy()];b=target.numpy().astype(np.float64)[mask.numpy()];n=len(a)
    if not n:return dict(patches=0,raw_mse=None,spatial_mse=None,bias_squared=None,cosine=None)
    error=a-b;mean=np.sum(error,axis=0)/n
    cosine=np.sum(a*b,axis=1)/(np.maximum(np.sqrt(np.sum(a*a,axis=1)),1e-8)*np.maximum(np.sqrt(np.sum(b*b,axis=1)),1e-8))
    return dict(patches=n,raw_mse=float(np.sum(error*error)/error.size),spatial_mse=float(np.sum((error-mean)**2)/error.size),bias_squared=float(np.sum(mean*mean)/len(mean)),cosine=float(np.sum(cosine)/n))

def compare_metrics(a,b):
    differences=[]
    for k,v in a.items():
        assert (v is None)==(b[k] is None)
        if v is not None:differences.append(abs(v-b[k]))
    return max(differences,default=0.)

def collect():
    dest=d.OUT/'scoring';dest.mkdir(exist_ok=True)
    if (dest/'metrics.json').exists() and (dest/'independent_recomputation.json').exists():return d.read(dest/'metrics.json')
    native_rows=[];feature_rows=[];max_native=0.;max_feature=0.;raw_decomposition=0.;queries={};supports={};coverage={};teachers={};masks={};audit=[]
    c=d.contract();train={m['id'] for m in c['training_catalog']};original=set(c['original_training_ids']);meta={m['id']:m for m in d.metas()}
    for m in d.metas():
        rid=m['id'];support_path=(d.old.CACHE if d.old.is_new(rid) else d.old.VIEW_CACHE)/'evaluation_support'/f'{rid}.pt'
        supports[rid]=d.load(support_path)['labels']
        for q in c['evaluation_queries']:
            query=d.load(d.old.query_path(rid,q));queries[rid,q]=query
            r=d.load(d.old.feature_path(rid,q));alpha=r['effective_alpha'].reshape(32,32);coverage[rid,q]=alpha
            reg=regions(query,supports[rid][str(q)])
            zero=F.interpolate((alpha<=1e-8)[None,None].float(),(210,210),mode='nearest')[0,0].bool()
            reg['roi_zero_coverage']=query['valid']&query['hidden']&zero
            reg['roi_unsupported_positive_coverage']=query['valid']&query['hidden']&~supports[rid][str(q)]&~zero
            masks[rid,q]=reg
            teachers[rid,q]=d.load(d.old.patch_path(rid,q,'full'))['patches'][0]
    def identity(rid,q,arm,seed,permuted=False):
        return dict(id=rid,place=meta[rid]['place'],query=q,kind=arm,seed=seed,permuted=permuted,cohort='train' if rid in train else 'development',training_group='original' if rid in original else 'added' if rid in train else None,query_group='fit' if q in c['fit_queries'] else 'other')
    def native_score(path,rid,q,row,sha):
        nonlocal max_native
        assert d.sha(path)==sha;prediction=torch.from_numpy(np.load(path,allow_pickle=False));target=queries[rid,q]['target'];a=score_torch(prediction,target,masks[rid,q]);b=score_numpy(prediction,target,masks[rid,q])
        for region in a:max_native=max(max_native,compare_metrics(a[region],b[region]))
        native_rows.append(dict(row,condition='full' if row['kind']=='Native_full' else 'missing',metrics=a,path=str(path),sha256=sha));audit.append(dict(path=str(path),sha256=sha,kind='depth'))
    f_roi=torch.zeros(32,32,dtype=torch.bool);f_roi[8:24,8:24]=True;f_roi=f_roi.flatten()
    def feature_score(prediction,rid,q,row,path=None,sha=None):
        nonlocal max_feature,raw_decomposition
        region_masks=dict(full=torch.ones(1024,dtype=torch.bool),roi=f_roi,zero_coverage=coverage[rid,q].flatten()<=1e-8,positive_coverage=coverage[rid,q].flatten()>1e-8)
        values={}
        for region,mask in region_masks.items():
            a=feature_torch(prediction,teachers[rid,q],mask);b=feature_numpy(prediction,teachers[rid,q],mask);max_feature=max(max_feature,compare_metrics(a,b));values[region]=a
            if a['raw_mse'] is not None:raw_decomposition=max(raw_decomposition,abs(a['raw_mse']-a['spatial_mse']-a['bias_squared']))
        feature_rows.append(dict(row,metrics=values,path=str(path) if path else None,sha256=sha))
    records=sorted((d.OUT/'completed').glob('*.json'));assert len(records)==3312,len(records)
    for i,p in enumerate(records):
        r=d.read(p);rid,q=r['id'],r['query'];row=identity(rid,q,r['arm'],r['seed'],r['permuted'])
        native_score(r['depth_path'],rid,q,row,r['depth_sha256']);assert d.sha(r['feature_path'])==r['feature_sha256']
        feature_score(torch.from_numpy(np.load(r['feature_path'],allow_pickle=False)),rid,q,row,r['feature_path'],r['feature_sha256']);audit.append(dict(path=r['feature_path'],sha256=r['feature_sha256'],kind='feature'))
        if i%300==0:print(dict(scored=i,of=3312),flush=True)
    base=list((d.OUT/'baselines').glob('*.json'));assert len(base)==352
    for p in sorted(base):
        r=d.read(p);rid,q=r['id'],r['query']
        for name,info in r['methods'].items():native_score(info['path'],rid,q,identity(rid,q,name,None),info['sha256'])
        b,_=d.cached_anchor(rid,q);feature_score(b,rid,q,identity(rid,q,'RasterFill',None))
    assert len(native_rows)==4368 and len(feature_rows)==3664
    result=dict(native=native_rows,feature=feature_rows,feature_exported=3312,feature_cached_RasterFill_references=352,native_scored=4368,new_native_exported=3664,inherited_Native=704)
    d.write(dest/'metrics.json',result)
    d.write(dest/'independent_recomputation.json',dict(complete=True,new_feature_predictions=3312,new_depth_predictions=3664,inherited_Native_predictions_rescored_under_new_coverage=704,cached_anchor_reference_scores=352,maximum_native_metric_difference=max_native,maximum_feature_metric_difference=max_feature,maximum_feature_raw_spatial_bias_decomposition_error=raw_decomposition,all_new_predictions_recomputed=True,independent_backend='NumPyfloat64 indexing, residual, centering and per-patch cosine; separate Torchfloat64 calculation',support_masks='Original saved geometric evaluation_support labels; unchanged. Zero coverage uses original effective a_F<=1e-8, nearest resize to210; unsupported-positive is original geometric unsupported with nonzero a_F, no tuned cutoff.',feature_cosine='Mean per-patch cosine, each vector norm clamped below1e-8; zero vectors contribute0.',scored_file_identities=audit,Native_identity_scope='unchanged inherited Native rescored only because row-dependent coverage strata changed; no repeated native inference',new_confirmation=False))
    return result

def contrast(a,b,indices,cosine=False):
    ss,pp=indices;aa=np.nanmean(a[ss,pp],axis=(1,2));bb=np.nanmean(b[ss,pp],axis=(1,2));ma=float(np.nanmean(a));mb=float(np.nanmean(b));delta=a-b
    if cosine:
        return dict(test=ma,reference=mb,absolute=ma-mb,absolute_interval95=np.nanpercentile(aa-bb,[2.5,97.5]).tolist(),reduction_percent=None,interval95_percent=[None,None],per_seed_percent=None,per_place_percent=None,per_seed_absolute=np.nanmean(delta,1).tolist(),per_place_absolute=np.nanmean(delta,0).tolist(),places_test_similarity_higher=int((np.nanmean(delta,0)>0).sum()),places_test_similarity_lower=int((np.nanmean(delta,0)<0).sum()),paired_cells=int((np.isfinite(a)&np.isfinite(b)).sum()),direction='Higher cosine similarity is better; absolute paired difference and interval only.',relative_not_applicable='Cosine is similarity, not error; zero RasterFill vectors have cosine0 and a percentage ratio is undefined. No clipping or epsilon ratio introduced.')
    with np.errstate(divide='ignore',invalid='ignore'):
        perseed=100*(1-np.nanmean(a,1)/np.nanmean(b,1));perplace=100*(1-np.nanmean(a,0)/np.nanmean(b,0));boot=100*(1-aa/bb)
    return dict(test=ma,reference=mb,absolute=ma-mb,absolute_interval95=np.nanpercentile(aa-bb,[2.5,97.5]).tolist(),reduction_percent=100*(1-ma/mb) if mb else None,interval95_percent=np.nanpercentile(boot,[2.5,97.5]).tolist(),per_seed_percent=perseed.tolist(),per_place_percent=perplace.tolist(),per_seed_absolute=np.nanmean(delta,1).tolist(),per_place_absolute=np.nanmean(delta,0).tolist(),places_test_error_lower=int((np.nanmean(delta,0)<0).sum()),places_test_error_higher=int((np.nanmean(delta,0)>0).sum()),paired_cells=int((np.isfinite(a)&np.isfinite(b)).sum()),direction='higher is better; use absolute difference for cosine' if cosine else 'positive reduction favors test')

def summarize(data):
    c=d.contract();groups={'train32':c['training_catalog'],'train_original12':[m for m in c['training_catalog'] if m['id'] in c['original_training_ids']],'train_added20':[m for m in c['training_catalog'] if m['id'] in c['added_training_ids']],'development12':c['development_catalog']}
    summaries={};comparisons={};arrays_saved={}
    for domain in ('native','feature'):
        rows=data[domain];methods=['P','S','G','Native','RasterFill','Native_full'] if domain=='native' else ['P','S','G','RasterFill']
        for group,metas in groups.items():
            places=[m['place'] for m in metas];lookup={p:i for i,p in enumerate(places)}
            for query_group,qs in [('fit',c['fit_queries']),('other',c['other_queries']),('pooled',c['evaluation_queries'])]:
                selected=[r for r in rows if not r['permuted'] and r['place'] in lookup and r['query'] in qs]
                regions_=list(selected[0]['metrics']);rng=np.random.default_rng(20261113);indices=(rng.integers(0,3,(5000,3,1)),rng.integers(0,len(places),(5000,1,len(places))))
                for reg in regions_:
                    metrics=[k for k in selected[0]['metrics'][reg] if k not in ('pixels','patches')]
                    for metric in metrics:
                        arr={method:np.full((3,len(places)),np.nan) for method in methods};cells={}
                        for row in selected:
                            if row['metrics'][reg][metric] is None:continue
                            seeds=range(3) if row['seed'] is None else [d.SEEDS.index(row['seed'])]
                            for si in seeds:cells.setdefault((row['kind'],si,lookup[row['place']]),[]).append(row['metrics'][reg][metric])
                        for (method,si,pi),values in cells.items():arr[method][si,pi]=np.mean(values)
                        prefix=f'{domain}/{group}/{query_group}/{reg}/{metric}'
                        for method,a in arr.items():
                            if np.isfinite(a).any():summaries[prefix+'/'+method]=dict(mean=float(np.nanmean(a)),per_seed=np.nanmean(a,1).tolist(),per_place=np.nanmean(a,0).tolist())
                        pairs=[('G','P'),('G','S'),('S','P')]+[(method,'RasterFill') for method in d.ARMS]
                        if domain=='native':pairs +=[(method,'Native') for method in d.ARMS]+[('RasterFill','Native')]
                        if metric!='log_bias':
                            for a,b in pairs:
                                if np.isfinite(arr[a]).any() and np.isfinite(arr[b]).any():comparisons[prefix+'/'+a+'-'+b]=contrast(arr[a],arr[b],indices,metric=='cosine')
                        if prefix=='native/development12/fit/roi/centered_log_mse':arrays_saved={k:v.tolist() for k,v in arr.items()}
        # Fixed144 binding cases vs their same-seed normalG counterparts.
        diag=[r for r in rows if r['permuted']];normal={(r['seed'],r['id'],r['query']):r for r in rows if not r['permuted'] and r['kind']=='G'}
        places=[m['place'] for m in c['development_catalog']];lookup={p:i for i,p in enumerate(places)};rng=np.random.default_rng(20261113);indices=(rng.integers(0,3,(5000,3,1)),rng.integers(0,12,(5000,1,12)))
        for reg in diag[0]['metrics']:
            for metric in diag[0]['metrics'][reg]:
                if metric in ('pixels','patches','log_bias'):continue
                a=np.full((3,12),np.nan);b=a.copy();cells={}
                for r in diag:
                    n=normal[r['seed'],r['id'],r['query']];x=r['metrics'][reg][metric];y=n['metrics'][reg][metric]
                    if x is not None and y is not None:cells.setdefault((d.SEEDS.index(r['seed']),lookup[r['place']]),[]).append((x,y))
                for (si,pi),values in cells.items():a[si,pi],b[si,pi]=np.mean(values,axis=0)
                if np.isfinite(a).any():comparisons[f'{domain}/binding_development_fit/{reg}/{metric}/G_permuted-G']=contrast(a,b,indices,metric=='cosine')
    # Counterexample is fixed by average three-seed primary error difference per case.
    nr=data['native'];cases=[]
    for m in c['development_catalog']:
        for q in c['fit_queries']:
            vals={arm:[r['metrics']['roi']['centered_log_mse'] for r in nr if r['id']==m['id'] and r['query']==q and r['kind']==arm and not r['permuted']] for arm in d.ARMS}
            assert all(len(v)==3 for v in vals.values());cases.append(dict(id=m['id'],place=m['place'],query=q,G_minus_P=float(np.mean(vals['G'])-np.mean(vals['P'])),per_arm={k:float(np.mean(v)) for k,v in vals.items()}))
    cases.sort(key=lambda x:x['G_minus_P'],reverse=True)
    result=json_finite(dict(complete=True,summaries=summaries,comparisons=comparisons,primary='native/development12/fit/roi/centered_log_mse',primary_arrays=arrays_saved,counterexamples=cases,bootstrap=dict(replicates=5000,seed=20261113,units='paired seed/place resampling; equal queries within place; fixed baselines common across seeds'),feature_is_secondary=True,full_condition='Native_full shared by construction; not learned fidelity/noninferiority',new_confirmation=False))
    return result

def decisions(s):
    def practical(r):return r['reduction_percent'] is not None and r['reduction_percent']>=1 and r['interval95_percent'][0]>0 and all(x>0 for x in r['per_seed_percent'])
    c=s['comparisons'];prefix=s['primary'];pe={k:dict(c[prefix+'/'+k],practical_screen=practical(c[prefix+'/'+k])) for k in ('G-P','G-S','S-P')}
    utility={k:dict(c[prefix+'/'+k+'-Native'],practical_screen=practical(c[prefix+'/'+k+'-Native'])) for k in d.ARMS}
    harms={}
    for arm in d.ARMS:
        harms[arm]={}
        for label,key in [('RasterFill_primary',prefix+'/'+arm+'-RasterFill'),('RasterFill_missing_whole_AbsRel','native/development12/fit/whole/absrel/'+arm+'-RasterFill')]:
            r=c[key];harms[arm][label]=dict(r,material_harm=r['reduction_percent'] is not None and r['reduction_percent']<=-1 and r['interval95_percent'][1]<0)
    return dict(complete=True,PE_empirical_flags=pe,native_net_utility_flags=utility,material_harm_flags=harms,at_most_one_candidate_by_original_Astra=True,scientific_candidate_selected=False,feature_hard_gate=False,method_adopted=False,independent_confirmation=False,overall_goal_achieved=False,no_extra_experiment_authorized=True)

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--row',required=True);args=parser.parse_args();d.context(args.row)
    torch.set_num_threads(4);warnings.filterwarnings('ignore',message='Mean of empty slice');warnings.filterwarnings('ignore',message='All-NaN slice');start=time.monotonic();data=collect();s=summarize(data);d.write(d.OUT/'scoring/summary.json',s);d.write(d.OUT/'method_decision.json',decisions(s))
    with (d.OUT/'scoring/all_summary_cells.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['cell','mean','per_seed','per_place'])
        for key,r in s['summaries'].items():w.writerow([key,r['mean'],r['per_seed'],r['per_place']])
    with (d.OUT/'scoring/all_comparisons.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['contrast','reduction_percent','relative_CI_low','relative_CI_high','absolute_test_minus_reference','absolute_CI_low','absolute_CI_high','per_seed_percent','per_place_percent','per_seed_absolute','per_place_absolute','direction'])
        for key,r in s['comparisons'].items():w.writerow([key,r['reduction_percent'],*r['interval95_percent'],r['absolute'],*r['absolute_interval95'],r['per_seed_percent'],r['per_place_percent'],r['per_seed_absolute'],r['per_place_absolute'],r['direction']])
    d.write(d.OUT/'scoring/runtime.json',dict(seconds=time.monotonic()-start,complete=True));print(dict(scored_native=4368,new_native=3664,inherited_native=704,scored_exported_features=3312,seconds=time.monotonic()-start),flush=True)

if __name__=='__main__':main()
