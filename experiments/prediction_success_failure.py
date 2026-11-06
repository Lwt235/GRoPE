"""Descriptive case audit of saved predictions; no model execution or new inputs."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from experiments.region_confirmation_metric_runtime import BufferedNpz
from experiments.round65_common import ROOT, read, write, sha, VARIANTS

OUT = ROOT / 'results/prediction_success_failure/attempt1'
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
REGION_FEATURES = ['center_ambiguity', 'candidate_shape_diversity', 'log_anisotropy', 'endpoint_asymmetry']
NATIVE_FEATURES = ['median_log_anisotropy', 'log_support_dispersion', 'median_floor_fraction']
IDENTITIES = {}
REPLAY_ERRORS = []


def bind(path, expected=None):
    path = Path(path).resolve()
    if str(path) not in IDENTITIES:
        IDENTITIES[str(path)] = sha(path)
    if expected is not None:
        assert IDENTITIES[str(path)] == expected, str(path)
    return path


def jread(path):
    bind(path)
    return read(path)


def load_npz(path, expected=None, fields=None):
    bind(path, expected)
    with BufferedNpz(path) as archive:
        return {key:archive[key] for key in (archive.files if fields is None else fields)}


def weights(groups, mask):
    w = np.zeros(len(groups),dtype=np.float64)
    present = np.unique(groups[mask])
    for group in present:
        select = mask & (groups == group)
        w[select] = 1. / (len(present) * select.sum())
    return w


def describe(a, b, groups, mask):
    """Seed-by-unit booleans; group-macro rates and raw events are kept separate."""
    assert a.dtype == b.dtype == np.bool_ and a.shape == b.shape and a.shape[1] == len(mask)
    if not mask.any():
        return None
    w = weights(groups,mask)
    states = dict(both_correct=a & b, corrected=a & ~b, regressed=~a & b, both_wrong=~a & ~b)
    events = {k:int(v[:,mask].sum()) for k,v in states.items()}
    rates = {k:float(100*(v@w).mean()) for k,v in states.items()}
    gain = 100*((a.astype(float)-b.astype(float))@w)
    assert sum(events.values()) == a.shape[0] * int(mask.sum())
    assert abs(sum(rates.values())-100) < 1e-9
    assert abs(rates['corrected']-rates['regressed']-gain.mean()) < 1e-9
    return dict(units=int(mask.sum()),seed_pairs=len(a),macro_groups=len(np.unique(groups[mask])),
        event_counts=events,macro_event_rates_pct=rates,mean_gain_pp=float(gain.mean()),
        paired_seed_gains_pp=gain.tolist(),left_accuracy_pct=float(100*(a@w).mean()),
        right_accuracy_pct=float(100*(b@w).mean()),
        unique_cases=dict(corrected_all_seeds=int(np.sum(states['corrected'].all(0)&mask)),
            regressed_all_seeds=int(np.sum(states['regressed'].all(0)&mask)),
            corrected_at_least_one_seed=int(np.sum(states['corrected'].any(0)&mask)),
            regressed_at_least_one_seed=int(np.sum(states['regressed'].any(0)&mask)),
            switches_direction_between_seeds=int(np.sum(states['corrected'].any(0)&states['regressed'].any(0)&mask)),
            both_wrong_all_seeds=int(np.sum(states['both_wrong'].all(0)&mask))))


def geometry_groups(a,b,groups,eligible,features,thresholds):
    records = []
    for key,value in features.items():
        threshold = thresholds[key]
        high = value >= threshold
        low = ~high
        lo,hi = describe(a,b,groups,eligible&low),describe(a,b,groups,eligible&high)
        within = []
        for group in np.unique(groups[eligible]):
            keep = eligible & (groups == group)
            l,h = describe(a,b,groups,keep&low),describe(a,b,groups,keep&high)
            if l and h:
                within.append(dict(group=str(group),low_units=l['units'],high_units=h['units'],
                    high_minus_low_net_pp=h['mean_gain_pp']-l['mean_gain_pp']))
        records.append(dict(feature=key,low=lo,high=hi,
            high_minus_low_net_pp=hi['mean_gain_pp']-lo['mean_gain_pp'] if hi and lo else None,
            within_group_differences=within,
            positive_within_groups=sum(r['high_minus_low_net_pp']>0 for r in within),
            negative_within_groups=sum(r['high_minus_low_net_pp']<0 for r in within)))
    return records


def region_geometry(cache):
    shape = cache['centers'].shape[:-1]
    anisotropy, asymmetry = [], []
    for start in range(0,len(cache['ids']),16):
        c = cache['centers'][start:start+16].astype(np.float64).reshape(-1,64,3)
        cov = cache['covariance'][start:start+16].astype(np.float64).reshape(-1,64,3,3)
        eig = np.linalg.eigvalsh(cov);inv = np.linalg.inv(cov)
        assert eig.min() > 0
        anisotropy.append(np.log(eig[...,2]/eig[...,0]))
        delta = c[:,:,None]-c[:,None,:]
        distance = np.square(delta).sum(-1)
        distance[:,np.arange(64),np.arange(64)] = np.inf
        near = np.argsort(distance,axis=-1,kind='stable')[...,:8]
        batch = np.arange(len(c))[:,None,None]
        dc = c[:,:,None]-c[batch,near]
        other_cov,other_inv = cov[batch,near],inv[batch,near]
        mi = np.einsum('btki,btij,btkj->btk',dc,inv,dc)
        mj = np.einsum('btki,btkij,btkj->btk',dc,other_inv,dc)
        ti = np.einsum('btij,btkji->btk',inv,other_cov)
        tj = np.einsum('btkij,btji->btk',other_inv,cov)
        value = (abs(np.log1p(mi)-np.log1p(mj))+abs(np.log1p(ti)-np.log1p(tj))).mean(-1)
        asymmetry.append(value)
    return dict(center_ambiguity=cache['geometry_scores'][...,0].reshape(-1),
        candidate_shape_diversity=cache['geometry_scores'][...,1].reshape(-1),
        log_anisotropy=np.concatenate(anisotropy).reshape(shape).reshape(-1),
        endpoint_asymmetry=np.concatenate(asymmetry).reshape(shape).reshape(-1))


def region_values(logits,cache):
    gathered = {k:[] for k in ('correct','pck','hard_epe','soft_epe','ce','picked','picked_overlap')}
    for start in range(0,len(logits),16):
        score = logits[start:start+16].astype(np.float64)
        truth = cache['overlap'][start:start+16].astype(np.float64)
        world = cache['world_centers'][start:start+16].astype(np.float64)
        pieces = {k:[] for k in gathered}
        for direction in range(2):
            x = score if direction == 0 else score.swapaxes(-1,-2)
            o = truth if direction == 0 else truth.swapaxes(-1,-2)
            picked = x.argmax(-1)
            overlap = np.take_along_axis(o,picked[...,None],-1)[...,0]
            target_world = world[:,1-direction][np.arange(len(x))[:,None],picked]
            hard = np.linalg.norm(world[:,direction]-target_world,axis=-1)
            z = x-x.max(-1,keepdims=True)
            logp = z-np.log(np.exp(z).sum(-1,keepdims=True))
            soft = np.linalg.norm(np.exp(logp)@world[:,1-direction]-world[:,direction],axis=-1)
            weight = o*(o>=.1);target = weight/np.maximum(weight.sum(-1,keepdims=True),1e-12)
            values = dict(correct=overlap>=.3,pck=hard<=.1,hard_epe=hard,soft_epe=soft,
                ce=-(target*logp).sum(-1),picked=picked,picked_overlap=overlap)
            for k,v in values.items():pieces[k].append(v)
        for k in gathered:gathered[k].append(np.stack(pieces[k],axis=1))
    return {k:np.concatenate(v).reshape(-1) for k,v in gathered.items()}


def example_rows(ids,a,b,mask,extra,limit=8):
    rows = []
    for name,condition in [('corrected_all_seeds',(a&~b).all(0)),('regressed_all_seeds',(~a&b).all(0))]:
        candidates = np.flatnonzero(condition&mask)
        candidates = sorted(candidates,key=lambda i:str(ids[i]))[:limit]
        for i in candidates:
            row = dict(type=name,id=str(ids[i]),index=int(i))
            row.update({k:np.asarray(v)[...,i].tolist() for k,v in extra.items()})
            rows.append(row)
    return rows


def save_events(name,ids,groups,eligible,a,b,features,left_prediction,right_prediction,seeds):
    path = OUT / 'events' / (name+'.npz');path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path,ids=ids,groups=groups,eligible=eligible,left_correct=a,right_correct=b,
        left_prediction=left_prediction,right_prediction=right_prediction,seeds=np.asarray(seeds),**features)
    return str(path.relative_to(OUT))


def region_analysis():
    dev_manifest = jread(ROOT/'results/round68_fixed_region/attempt1/data/manifest.json')
    fit = []
    for k in ('fit0','fit1'):
        r=dev_manifest['caches'][k]
        fit.append(region_geometry(load_npz(r['path'],r['sha256'],['centers','covariance','geometry_scores','ids'])))
    old_cut = jread(ROOT/'results/round68_fixed_region/attempt1/data/strata_thresholds.json')['thresholds']
    cuts = dict(zip(REGION_FEATURES[:2],old_cut))
    cuts.update({k:float(np.median(np.concatenate([r[k] for r in fit]))) for k in REGION_FEATURES[2:]})
    write(OUT/'region_thresholds.json',dict(thresholds=cuts,source='old fit0+fit1, before outcome grouping'))
    del fit
    old_records = jread(ROOT/'results/symmetric_relation/attempt1/result.json')['records']
    confirmed = jread(ROOT/'results/region_confirmation/attempt1/result.json')['records']
    confirmed_manifest = jread(ROOT/'results/region_confirmation/attempt1/data_manifest.json')
    results=[]
    for dataset,manifest in [('region_dev5',dev_manifest),('region_used9',confirmed_manifest)]:
        geometry = None
        for condition in ('base','sparse256'):
            meta=manifest['caches'][('eval_' if dataset=='region_dev5' else '')+condition]
            cache=load_npz(meta['path'],meta['sha256'])
            valid=np.stack([cache['overlap'].max(-1)>=.3,cache['overlap'].max(-2)>=.3],1).reshape(-1)
            groups=np.repeat(cache['scenes'],128)
            ids=np.asarray([str(i)+f':v{v}:q{q:02d}' for i in cache['ids'] for v in range(2) for q in range(64)])
            if geometry is None:geometry=region_geometry(cache)
            roles=['relative5','distance_mlp','fixed_isotropic5']+(['symmetric3'] if dataset=='region_dev5' else [])
            values={}
            for role in roles:
                for seed in SEEDS[:3]:
                    if dataset=='region_dev5':
                        row=next(r for r in old_records if r['case']=='region' and (r['role'],r['seed'],r['condition'])==(role,seed,condition))
                        path=row['prediction_path']
                    else:
                        row=next(r for r in confirmed if (r['role'],r['seed'],r['condition'])==(role,seed,condition))
                        path=ROOT/'results/region_confirmation/attempt1/predictions'/role/str(seed)/(condition+'.npz')
                    pred=load_npz(path,row['prediction_sha256'])
                    assert np.array_equal(pred['ids'],cache['ids'])
                    value=region_values(pred['logits'],cache);values[role,seed]=value
                    w=weights(groups,valid)
                    for key,old_key in [('correct','region_recall'),('pck','center_pck'),('soft_epe','soft_epe'),('ce','ce')]:
                        err=abs(float(value[key]@w)-row['metrics']['macro'][old_key]);REPLAY_ERRORS.append(err)
                        assert err<1e-8,(dataset,role,seed,key,err)
            a=np.stack([values['relative5',s]['correct'] for s in SEEDS[:3]])
            left_picks=np.stack([values['relative5',s]['picked'] for s in SEEDS[:3]])
            for control in roles[1:]:
                b=np.stack([values[control,s]['correct'] for s in SEEDS[:3]])
                row=dict(dataset=dataset,condition=condition,control=control,seeds=SEEDS[:3],
                    summary=describe(a,b,groups,valid),geometry=geometry_groups(a,b,groups,valid,geometry,cuts),
                    strata=[dict(stratum=g,summary=describe(a,b,groups,valid&(cache['strata'].reshape(-1)==g))) for g in range(4)],
                    by_scene=[dict(scene=str(g),summary=describe(a,b,groups,valid&(groups==g))) for g in np.unique(groups)])
                outcomes=dict(corrected=(a&~b),regressed=(~a&b),both_wrong=(~a&~b))
                row['center_differences_by_event']={}
                for outcome,mask in outcomes.items():
                    row['center_differences_by_event'][outcome]={}
                    for key in ('pck','hard_epe','soft_epe','ce'):
                        left=np.stack([values['relative5',s][key] for s in SEEDS[:3]])
                        right=np.stack([values[control,s][key] for s in SEEDS[:3]])
                        keep=mask&valid[None,:]
                        row['center_differences_by_event'][outcome][key]=float((left.astype(float)-right)[keep].mean()) if keep.any() else None
                right_picks=np.stack([values[control,s]['picked'] for s in SEEDS[:3]])
                row['events_path']=save_events(f'{dataset}_{condition}_{control}',ids,groups,valid,a,b,geometry,left_picks,right_picks,SEEDS[:3])
                if condition=='base' and control=='distance_mlp':
                    row['examples']=example_rows(ids,a,b,valid,dict(geometry=np.stack(list(geometry.values())),
                        five_picked=left_picks,distance_picked=right_picks,
                        five_overlap=np.stack([values['relative5',s]['picked_overlap'] for s in SEEDS[:3]]),
                        distance_overlap=np.stack([values[control,s]['picked_overlap'] for s in SEEDS[:3]]),
                        five_center_error=np.stack([values['relative5',s]['hard_epe'] for s in SEEDS[:3]]),
                        distance_center_error=np.stack([values[control,s]['hard_epe'] for s in SEEDS[:3]])))
                    fixed=np.stack([values['fixed_isotropic5',s]['correct'] for s in SEEDS[:3]])
                    rescue=a&~b&valid[None,:]
                    row['corrected_distance_events_fixed_also_wrong']=int((rescue&~fixed).sum())
                    row['corrected_distance_events_fixed_correct']=int((rescue&fixed).sum())
                results.append(row)
            print('REGION_CASES_COMPLETE',dataset,condition,int(valid.sum()),flush=True)
    return results


def native_geometry(covariance):
    values={k:[] for k in NATIVE_FEATURES}
    for start in range(0,len(covariance),64):
        eig=np.linalg.eigvalsh(np.asarray(covariance[start:start+64,:512]))
        assert eig.min()>0
        trace=eig.sum(-1)
        values['median_log_anisotropy'].extend(np.median(np.log(eig[...,2]/eig[...,0]),axis=1))
        values['log_support_dispersion'].extend(np.std(.5*np.log(trace/3),axis=1))
        values['median_floor_fraction'].extend(np.median(3e-8/trace,axis=1))
    return {k:np.asarray(v) for k,v in values.items()}


def native_analysis():
    base=ROOT/'results/round65_actual_gs/attempt1/data'
    manifest=jread(base/'manifest.json')
    fit_meta=next(p for p in manifest['partitions'] if p['split']=='fit')
    for name in ('covariance.npy','labels.npy'):bind(base/'fit'/name,fit_meta['hashes'][name])
    fit_labels=np.load(base/'fit/labels.npy',mmap_mode='r')
    fit_geometry=native_geometry(np.load(base/'fit/covariance.npy',mmap_mode='r'))
    cuts={k:{int(c):float(np.median(v[fit_labels==c])) for c in range(60)} for k,v in fit_geometry.items()}
    write(OUT/'native_thresholds.json',dict(thresholds_by_class=cuts,source='old fit, within taxonomy class'))
    sym=jread(ROOT/'results/symmetric_relation/attempt1/result.json')['records']
    replication=jread(ROOT/'results/native_seed_replication/attempt1/result.json')['records']
    official=jread(ROOT/'results/native_confirmation/attempt1/result.json')['records']
    membership={r['file_path']:r for r in jread(ROOT/'results/native_confirmation_metadata/attempt1/historical_overlap1/object_membership.json')['rows']}
    catalog=jread(ROOT/'results/round65_actual_gs/preparation/train_catalog.json')['rows']
    catalog+=jread(ROOT/'results/native_confirmation_metadata/attempt1/test_catalog.json')['rows']
    names={r['taxonomy_id']:r['category'] for r in catalog}
    write(OUT/'class_names.json',names)
    results=[]
    geometry_cache={}
    for dataset in ('native_dev512','native_dev256','native_used_test'):
        if dataset=='native_used_test':
            lock=jread(ROOT/'results/native_confirmation/attempt1/evaluation_lock.json')
            folder=Path(lock['cache_path']);meta=jread(folder/'manifest.json')
            ids=np.asarray(meta['ids']);labels=np.load(bind(folder/'labels.npy',meta['array_sha256']['labels.npy']))
            covpath=bind(folder/'covariance.npy',meta['array_sha256']['covariance.npy'])
            geometry=native_geometry(np.load(covpath,mmap_mode='r'))
            keep=np.asarray([membership[i]['historical_status']!='used_original_train' for i in ids])
            assert keep.sum()==3845
        else:
            meta=next(r for r in manifest['partitions'] if r['split']=='dev_eval')
            ids=np.asarray(meta['ids']);labels=np.load(bind(base/'dev_eval/labels.npy',meta['hashes']['labels.npy']))
            if not geometry_cache:
                covpath=bind(base/'dev_eval/covariance.npy',meta['hashes']['covariance.npy'])
                geometry_cache=native_geometry(np.load(covpath,mmap_mode='r'))
            geometry=geometry_cache;keep=np.ones(len(ids),dtype=bool)
        local_cuts={k:np.asarray([cuts[k][int(c)] for c in labels]) for k in NATIVE_FEATURES}
        controls=['distance_mlp','fixed_isotropic5']+(['symmetric3'] if dataset!='native_used_test' else [])
        all_seeds=SEEDS[:3] if dataset=='native_dev256' else SEEDS
        for condition in VARIANTS:
            predictions={};ces={};correct={}
            for role in ['relative5']+controls:
                seeds=all_seeds if role in ('relative5','distance_mlp') else SEEDS[:3]
                for seed in seeds:
                    if dataset=='native_used_test':
                        entry=next(r for r in official if (r['role'],r['seed'],r['variant'])==(role,seed,condition))
                        path=entry['prediction_path'];expected=entry['groups']['full_population']['metrics']
                    elif seed in SEEDS[3:]:
                        entry=next(r for r in replication if (r['role'],r['seed'],r['variant'])==(role,seed,condition))
                        path=ROOT/'results/native_seed_replication/attempt1/evaluation'/role/str(seed)/(condition+'.npz')
                        expected=entry['metrics']
                    else:
                        case='native256' if dataset=='native_dev256' else 'native512'
                        entry=next(r for r in sym if r['case']==case and (r['role'],r['seed'],r['condition'])==(role,seed,condition))
                        path=entry['prediction_path'];expected=entry['metrics']
                    data=load_npz(path,entry['prediction_sha256'])
                    assert np.array_equal(data['ids'],ids) and np.array_equal(data['labels'],labels)
                    logits=data['logits'].astype(np.float64)
                    predictions[role,seed]=logits.argmax(-1);correct[role,seed]=predictions[role,seed]==labels
                    maximum=logits.max(-1);logsum=maximum+np.log(np.exp(logits-maximum[:,None]).sum(-1))
                    ces[role,seed]=logsum-logits[np.arange(len(labels)),labels]
                    actual=float(correct[role,seed]@weights(labels,np.ones(len(labels),dtype=bool)))
                    err=abs(actual-expected['mca']);REPLAY_ERRORS.append(err);assert err<1e-8
            for control in controls:
                cohorts=[('old3',SEEDS[:3])]
                if control=='distance_mlp' and dataset!='native_dev256':cohorts+=[('new5',SEEDS[3:]),('all8',SEEDS)]
                for cohort,seeds in cohorts:
                    a=np.stack([correct['relative5',s] for s in seeds]);b=np.stack([correct[control,s] for s in seeds])
                    pa=np.stack([predictions['relative5',s] for s in seeds]);pb=np.stack([predictions[control,s] for s in seeds])
                    row=dict(dataset=dataset,condition=condition,control=control,cohort=cohort,seeds=seeds,
                        primary_group='no_known_historical_overlap' if dataset=='native_used_test' else 'dev_eval',
                        summary=describe(a,b,labels,keep),domains=[dict(source=domain,summary=describe(a,b,labels,keep&mask))
                            for domain,mask in [('MACGS',labels<30),('ModelNet',labels>=30)]],
                        by_class=[dict(label=int(c),category=names[int(c)],summary=describe(a,b,labels,keep&(labels==c)))
                            for c in np.unique(labels[keep])],
                        paired_ce_difference=[float((ces['relative5',s]-ces[control,s])[keep].mean()) for s in seeds])
                    if dataset=='native_used_test' and condition=='base':
                        row['full_population']=describe(a,b,labels,np.ones(len(ids),dtype=bool))
                        row['historical_reuse']=describe(a,b,labels,~keep)
                    # Only base512 gets geometry association; other conditions retain outcomes.
                    if condition=='base' and dataset!='native_dev256':
                        row['geometry']=geometry_groups(a,b,labels,keep,geometry,local_cuts)
                    primary_example=(control=='distance_mlp' and ((dataset=='native_dev512' and cohort=='all8') or
                        (dataset=='native_used_test' and cohort=='new5') or dataset=='native_dev256'))
                    if primary_example and condition==('sparse256' if dataset=='native_dev256' else 'base'):
                        row['examples']=example_rows(ids,a,b,keep,dict(label=labels,five_predictions=pa,distance_predictions=pb,
                            geometry=np.stack(list(geometry.values()))))
                    if cohort in ('all8','old3') and control=='distance_mlp' or (dataset=='native_used_test' and cohort=='new5'):
                        row['events_path']=save_events(f'{dataset}_{condition}_{control}_{cohort}',ids,labels,keep,a,b,geometry,pa,pb,seeds)
                    if cohort=='old3' and control=='distance_mlp':
                        fixed=np.stack([correct['fixed_isotropic5',s] for s in seeds]);rescue=a&~b&keep[None,:]
                        row['corrected_distance_events_fixed_also_wrong']=int((rescue&~fixed).sum())
                        row['corrected_distance_events_fixed_correct']=int((rescue&fixed).sum())
                    # Per true/predicted class transitions include every error, sorted later for display only.
                    if primary_example:
                        rescue=a&~b&keep[None,:];regress=~a&b&keep[None,:]
                        true=np.broadcast_to(labels,a.shape)
                        transitions=[]
                        for event,selection,pred in [('corrected',rescue,pb),('regressed',regress,pa)]:
                            codes=true[selection]*60+pred[selection]
                            unique,counts=np.unique(codes,return_counts=True)
                            transitions.extend(dict(event=event,true_class=int(code//60),wrong_class=int(code%60),
                                true_name=names[int(code//60)],wrong_name=names[int(code%60)],events=int(count))
                                for code,count in zip(unique,counts))
                        row['confusion_transitions']=transitions
                    results.append(row)
            print('NATIVE_CASES_COMPLETE',dataset,condition,len(ids),flush=True)
    return results


def main():
    if (OUT/'result.json').exists():raise ValueError('preserve previous analysis')
    started=time.time();bind(OUT/'plan.md');bind(__file__)
    # Exact, independently knowable event/weight identities before reading cases.
    a=np.array([[1,1,0,0]],dtype=bool);b=np.array([[1,0,1,0]],dtype=bool)
    assert describe(a,b,np.zeros(4),np.ones(4,dtype=bool))['event_counts']==dict(both_correct=1,corrected=1,regressed=1,both_wrong=1)
    assert describe(a,b,np.array([0,0,0,1]),np.ones(4,dtype=bool))['mean_gain_pp']==0
    region=region_analysis();write(OUT/'region_result.json',region)
    native=native_analysis();write(OUT/'native_result.json',native)
    write(OUT/'result.json',dict(region=region,native=native,source_and_input_sha256=IDENTITIES,
        reproduced_metric_cells=len(REPLAY_ERRORS),maximum_prior_metric_error=max(REPLAY_ERRORS),
        elapsed_seconds=time.time()-started,created_unix=time.time(),training_updates=0,new_inference=0,
        new_data_downloads=0,fresh_confirmation=False,causal_claim=False,actual_review_pending=True))
    print('PREDICTION_CASES_COMPLETE',len(region),len(native),max(REPLAY_ERRORS),time.time()-started,flush=True)


if __name__=='__main__':main()
