"""Independent NumPy subset/energy and all-row CI audit of sealed Round6.

No sealed implementation is imported or edited. Saved float32 probabilities do
not identify CE/KL wherever positive target mass has zero/subnormal probability;
those rows are marked unresolved. Independent Gaussian energy + stable logsumexp
reconstructs their metrics using ONLY the preselected saved input subset.
"""
from __future__ import annotations
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROLES=('full','fixed','isotropic','point','point_pck')
GRIDS=('q8_k8','q8_k16','q16_k8','q16_k16')
EXPECTED_LOCK='878ce25b0273333e9e58438261d1a58c7ba1fd28005aade37d9321986815cd92'
ENERGY_METRIC_TOLERANCE=.002
PROBABILITY_METRIC_TOLERANCE=.0002
PROBABILITY_ABSOLUTE_TOLERANCE=.0003
POINT_EPE_TOLERANCE=.001
SPECS={'kl':('kl_sum','visible_mass'),'ce':('ce_sum','visible_mass'),
       'js':('js_sum','visible_mass'),'query_kl':('row_kl_sum','n_valid'),
       'pck8':('pck8_count','n_point'),'pck4':('pck4_count','n_point'),
       'pck_stride':('pck_stride_count','n_point'),'point_ce':('point_ce_sum','n_point'),
       'point_epe':('point_epe_sum','n_point'),'point_soft_epe':('point_soft_epe_sum','n_point')}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def row_key(row):return (row['scene'],row['grid'],int(row['q_index']),int(row['k_index']))


def stable_log_probability(energy):
    shifted=np.asarray(energy,np.float64)-np.max(energy,axis=-1,keepdims=True)
    return shifted-np.log(np.exp(shifted).sum(-1,keepdims=True))


def gaussian_energy(inputs,config):
    """Direct negative Mahalanobis energy; row logdet cancels for uniform keys."""
    mean=np.asarray(inputs['q_to_k'],np.float64)
    C=np.asarray(inputs['q_to_kcov'],np.float64).copy()
    D=np.asarray(inputs['kcov'],np.float64).copy()
    method=config['method']
    if method=='point':C[:]=0;D[:]=0
    elif method=='fixed':C=np.asarray(inputs['qcov'],np.float64)
    elif method=='isotropic':
        C=np.eye(2)[None]*np.trace(C,axis1=-2,axis2=-1)[:,None,None]/2
        D=np.eye(2)[None]*np.trace(D,axis1=-2,axis2=-1)[:,None,None]/2
    elif method!='full':raise ValueError('unrecognized frozen method')
    if not np.array_equal(D,np.broadcast_to(D[0],D.shape)):
        raise ValueError('target covariance is not common across keys')
    h=config.get('bandwidth',config.get('width'))
    precision=np.linalg.inv(C+D[0]+np.eye(2)*h*h)
    delta=np.asarray(inputs['kmean'],np.float64)[None]-mean[:,None]
    energy=-.5*np.einsum('nki,nij,nkj->nk',delta,precision,delta)
    content=np.asarray(inputs['q'],np.float64)@np.asarray(inputs['k'],np.float64).T
    return energy+config['temperature']*content


def probability_metrics(probability,labels,log_probability=None):
    """NumPy metrics and explicit unresolved CE rows, not a log-clamp surrogate."""
    p=np.asarray(probability,np.float64);y=np.asarray(labels['soft_labels'],np.float64)
    if p.shape!=y.shape or not np.isfinite(p).all() or (p<0).any():raise ValueError('bad probability array')
    with np.errstate(divide='ignore',invalid='ignore'):
        ly=np.where(y>0,np.log(y),0)
        lp=np.where(p>0,np.log(p),-np.inf) if log_probability is None else np.asarray(log_probability,np.float64)
        entropy=-np.sum(y*ly,-1)
        ce=-np.sum(np.where(y>0,y*lp,0),-1)
        midpoint=(y+p)/2
        js=.5*np.sum(np.where(y>0,y*(ly-np.log(midpoint)),0),-1)
        js+=.5*np.sum(np.where(p>0,p*(np.log(p)-np.log(midpoint)),0),-1)
    unresolved=((y>0)&(p<=np.finfo(np.float32).tiny)).any(-1) if log_probability is None else np.zeros(len(y),bool)
    ce=np.where(unresolved,np.nan,ce)
    valid=np.asarray(labels['gt_valid'],bool)
    mass=np.asarray(labels['visible_fraction'],np.float64)*np.asarray(labels['sample_count'],np.float64)
    point=np.asarray(labels['point_gt_valid'],bool)
    target=np.asarray(labels['point_pixcontinuous'],np.float64)
    keygrid=np.asarray(labels['key_grid'],np.float64)
    argmax=p.argmax(-1)
    error=np.linalg.norm(keygrid[argmax]-target,axis=-1)
    soft_error=np.linalg.norm(p@keygrid-target,axis=-1)
    point_index=np.asarray(labels['point_key_index'],int)
    point_ce=-lp[np.arange(len(p)),np.maximum(point_index,0)]
    unresolved_point=point&(p[np.arange(len(p)),np.maximum(point_index,0)]<=np.finfo(np.float32).tiny) if log_probability is None else np.zeros(len(y),bool)
    def weighted(values,weights):
        keep=weights>0
        return float(np.sum(values[keep]*weights[keep])) if np.isfinite(values[keep]).all() else None
    result={'n_valid':int(valid.sum()),'visible_mass':float(mass.sum()),'n_point':int(point.sum()),
        'ce_sum':weighted(ce,mass),'entropy_sum':weighted(entropy,mass),'kl_sum':weighted(ce-entropy,mass),
        'js_sum':weighted(js,mass),'row_kl_sum':weighted(ce-entropy,valid.astype(float)),
        'point_ce_sum':None if unresolved_point.any() else float(np.sum(point_ce[point])),
        'point_epe_sum':float(np.sum(error[point])),
        'point_soft_epe_sum':float(np.sum(soft_error[point])),
        'pck8_count':int(np.sum((error<=8)&point)),'pck4_count':int(np.sum((error<=4)&point)),
        'pck_stride_count':int(np.sum((error<=labels['k_stride'])&point)),
        'unresolved_area_ce_rows':int((unresolved&valid).sum()),
        'unresolved_point_ce_rows':int(unresolved_point.sum()),
        'maximum_row_sum_error':float(np.max(np.abs(p.sum(-1)-1))),
        'argmax_exact_tie_rows':int(((p==p.max(-1,keepdims=True)).sum(-1)>1).sum())}
    return result


def pck_bounds(scores,labels,score_tolerance=0.):
    """Allow only explicitly reported top-score ambiguity and 1e-4px norm rounding."""
    scores=np.asarray(scores,np.float64)
    eligible=scores>=scores.max(-1,keepdims=True)-score_tolerance
    target=np.asarray(labels['point_pixcontinuous'],np.float64)
    keygrid=np.asarray(labels['key_grid'],np.float64)
    distance=np.linalg.norm(keygrid[None]-target[:,None],axis=-1)
    point=np.asarray(labels['point_gt_valid'],bool)
    bounds={}
    for name,threshold in [('pck8_count',8),('pck4_count',4),('pck_stride_count',labels['k_stride'])]:
        sure=(eligible & (distance>threshold-1e-4)).sum(-1)==0
        possible=(eligible & (distance<=threshold+1e-4)).any(-1)
        bounds[name]=[int((sure&point).sum()),int((possible&point).sum())]
    low=np.min(np.where(eligible,distance,np.inf),axis=-1)
    high=np.max(np.where(eligible,distance,-np.inf),axis=-1)
    bounds['point_epe_sum']=[float(low[point].sum()),float(high[point].sum())]
    return bounds,int(((eligible.sum(-1)>1)&point).sum())


def independent_ci(rows_by_role,scenes,reference):
    """Recompute ratios from raw counters, all20k scan draws and every frozen gate."""
    matrices={}
    for role,rows in rows_by_role.items():
        matrices[role]={}
        for metric,(num,den) in SPECS.items():
            cells=np.zeros((5,4))
            for i,scene in enumerate(scenes):
                for j,grid in enumerate(GRIDS):
                    selected=[row for row in rows if row['scene']==scene and row['grid']==grid]
                    denominator=np.sum([row[den] for row in selected],dtype=np.float64)
                    if denominator<=0:raise ValueError('unmeasurable frozen scene/grid')
                    cells[i,j]=np.sum([row[num] for row in selected],dtype=np.float64)/denominator
            matrices[role][metric]=cells
    draw=np.random.default_rng(20261030).integers(0,5,(20000,5))
    slices={'all':[0,1,2,3],'cross':[1,2],'same':[0,3],**{g:[i] for i,g in enumerate(GRIDS)}}
    maximum_error=0.;computed={};checks=0
    for role in ROLES[1:]:
        key='full_minus_'+role;computed[key]={}
        for metric in SPECS:
            conditions=slices if metric=='kl' else {'all':[0,1,2,3]} if metric in ['ce','js','query_kl'] else {'q8_k8':[0]}
            computed[key][metric]={}
            for condition,indices in conditions.items():
                x=matrices['full'][metric][:,indices].mean(-1)
                y=matrices[role][metric][:,indices].mean(-1)
                diff=x-y
                a=x[draw].mean(-1);b=y[draw].mean(-1)
                value={'full':float(x.mean()),'baseline':float(y.mean()),'difference':float(diff.mean()),
                    'ci95':np.percentile(a-b,[2.5,97.5]).tolist(),
                    'relative_difference':float(diff.mean()/y.mean()),
                    'relative_ci95':np.percentile((a-b)/b,[2.5,97.5]).tolist(),
                    'improved_scan_count':int((diff>0).sum() if metric.startswith('pck') else (diff<0).sum())}
                expected=reference['comparisons'][key][metric][condition] if metric=='kl' else reference['comparisons'][key][metric]
                for field in ['full','baseline','difference','ci95','relative_difference','relative_ci95']:
                    error=float(np.max(np.abs(np.asarray(value[field])-np.asarray(expected[field]))))
                    maximum_error=max(maximum_error,error);checks+=1
                    if error>1e-10:raise ValueError(f'independent CI mismatch {key}/{metric}/{condition}/{field}: {error}')
                if value['improved_scan_count']!=expected['improved_scan_count']:raise ValueError('scan direction count mismatch')
                if metric.startswith('pck'):
                    if abs(expected['difference_pp']-value['difference']*100)>1e-10:raise ValueError('percentage-point conversion mismatch')
                computed[key][metric][condition]=value
    baseline=reference['gates']['area_baseline'];primary=computed['full_minus_'+baseline]
    improvement={name:(primary['kl'][name]['relative_difference']<=-.02 and primary['kl'][name]['ci95'][1]<0)
                 for name in ['all','q8_k16','q16_k8']}
    same={name:primary['kl'][name]['relative_ci95'][1]<.02 for name in ['q8_k8','q16_k16']}
    point={role:computed['full_minus_'+role]['pck8']['q8_k8']['ci95'][0]>-.005 for role in [baseline,'point_pck']}
    stable=primary['kl']['all']['improved_scan_count']>=4
    gates={'kl_at_least_2pct_and_ci_upper_below_zero':improvement,
        'at_least_four_of_five_scans_improve_total_kl':stable,
        'same_scale_relative_ci_upper_below_2pct':same,'q8_k8_pck8_ci_lower_above_minus_0_5pp':point,
        'pass':all(improvement.values()) and stable and all(same.values()) and all(point.values())}
    for field,value in gates.items():
        if value!=reference['gates'][field]:raise ValueError('independent gate differs: '+field)
    return {'checks':checks,'maximum_absolute_ci_or_metric_error':maximum_error,'bootstrap':20000,
        'rng_seed':20261030,'sampling_units':5,'training_seed_replicates':0,'gates':gates,
        'computed':computed,'paired_draw_sha256':hashlib.sha256(draw.astype('<i8').tobytes()).hexdigest()}


def compare_subset_metrics(recomputed,expected,bounds,tolerance):
    errors={};compatible=True
    for name,value in recomputed.items():
        if name not in expected:continue
        if value is None:
            errors[name]={'status':'not identified from saved float32 probability'}
            continue
        if name in ['n_valid','n_point','visible_mass']:
            error=abs(value-expected[name]);ok=error==0
        elif name.startswith('pck'):
            error=value-expected[name];ok=bounds[name][0]<=expected[name]<=bounds[name][1]
        elif name=='point_epe_sum':
            denominator=max(expected['n_point'],1)
            error=(value-expected[name])/denominator
            ok=bounds[name][0]-POINT_EPE_TOLERANCE*denominator<=expected[name]<=bounds[name][1]+POINT_EPE_TOLERANCE*denominator
        else:
            denominator=expected['n_point'] if name.startswith('point_') else expected['n_valid'] if name=='row_kl_sum' else expected['visible_mass']
            error=(value-expected[name])/max(denominator,1)
            ok=abs(error)<=(POINT_EPE_TOLERANCE if name=='point_soft_epe_sum' else tolerance)
        errors[name]={'difference_normalized_per_measurement' if not name.startswith('pck') else 'count_difference':float(error),'compatible':bool(ok)}
        compatible &= bool(ok)
    return errors,compatible


def audit(args):
    started=time.time()
    if args.out.exists() and any(args.out.iterdir()):raise ValueError('use a fresh audit directory')
    lock_path=args.run/'lock.json'
    if sha(lock_path)!=EXPECTED_LOCK:raise ValueError('confirmation root lock differs from authorized hash')
    lock=json.loads(lock_path.read_text())
    completion=json.loads((args.run/'completion.json').read_text())
    if completion['lock_sha256']!=EXPECTED_LOCK:raise ValueError('test evaluation is not complete under the expected lock')
    source_checks={}
    repository=Path(__file__).resolve().parents[1]
    for name,digest in lock['sourcehash'].items():
        if sha(repository/name)!=digest:raise ValueError('sealed source changed: '+name)
        source_checks[name]=digest
    reference=json.loads(args.analysis.read_text())
    scenes=lock['expected_scenes']
    payloads={};tables={};rows_by_role={};input_sources={}
    for role in ROLES:
        path=args.run/role/'test.json'
        if sha(path)!=completion['roles'][role]:raise ValueError('completed test JSON changed')
        payload=json.loads(path.read_text());payloads[role]=payload
        if payload['lock_sha256']!=EXPECTED_LOCK:raise ValueError('role lock mismatch')
        table={row_key(row):row for row in payload['rows']}
        if len(table)!=len(payload['rows']) or len(table)!=5*4*96:raise ValueError('duplicate/incomplete row identities')
        tables[role]=table;rows_by_role[role]=payload['rows']
        for source in payload['input_sources']:
            key=source['scene'],source['grid']
            previous=input_sources.get(key)
            if previous and (source['input_sha256'],source['label_sha256'])!=(previous['input_sha256'],previous['label_sha256']):
                raise ValueError('role inputs/labels differ')
            input_sources[key]=source
    for role in ROLES[1:]:
        if tables[role].keys()!=tables['full'].keys():raise ValueError('method identity mismatch')
        for key,row in tables[role].items():
            full=tables['full'][key]
            for field in ['n_valid','visible_mass','n_point','entropy_sum']:
                if row[field]!=full[field]:raise ValueError('method supervision arithmetic differs')
    ci=independent_ci(rows_by_role,scenes,reference)
    archives=[];subset_by_condition={}
    for role in ROLES:
        paths=sorted((args.run/role/'predictions/test').glob('*.npz'))
        if len(paths)!=40:raise ValueError('expected exactly40 saved subset archives per role')
        for path in paths:
            with np.load(path,allow_pickle=False) as values:
                ident=json.loads(str(values['identity'].item()))
            key=row_key(ident)
            if key not in tables[role]:raise ValueError('probability subset identity not in complete row table')
            archives.append((role,path,key,ident))
            subset_by_condition.setdefault(key[:2],set()).add(key[2:])
    if len(subset_by_condition)!=20 or any(len(pairs)!=2 for pairs in subset_by_condition.values()):
        raise ValueError('subset must be fixed two pairs per named scan/grid')
    # Evaluate exactly40 unique input pairs. NPZ members store96 pairs, but only
    # selected rows are retained/evaluated; other input/label rows are not analyzed.
    inputs={};input_hashes={}
    for (scene,grid),pairs in subset_by_condition.items():
        source=input_sources[scene,grid];directory=Path(source['path'])
        path=directory/'inputs.npz'
        if sha(path)!=source['input_sha256']:raise ValueError('locked input hash changed')
        input_hashes[str(path)]=source['input_sha256']
        manifest=json.loads((directory/'manifest.json').read_text())
        indices={tuple(pair):i for i,pair in enumerate(manifest['pairs'])}
        ordered=sorted(pairs);positions=[indices[pair] for pair in ordered]
        fields=['q','k','q_to_k','q_to_kcov','qcov','kmean','kcov']
        with np.load(path,allow_pickle=False) as values:
            selected={name:values[name][positions].copy() for name in fields}
        for i,pair in enumerate(ordered):inputs[scene,grid,*pair]={name:value[i] for name,value in selected.items()}
    records=[];probability_hashes={};label_hashes={};all_compatible=True
    max_probability_error=0.;unresolved_total=0
    for role,path,key,ident in archives:
        probability_hashes[str(path)]=sha(path)
        with np.load(path,allow_pickle=False) as values:
            p=values['probability'].copy()
            labels={name:values[name].copy() for name in ['soft_labels','gt_valid','visible_fraction','sample_count',
                'point_gt_valid','point_pixcontinuous','point_key_index','key_grid']}
        digest=hashlib.sha256(b''.join(np.ascontiguousarray(labels[name]).tobytes() for name in sorted(labels))).hexdigest()
        if key in label_hashes and label_hashes[key]!=digest:raise ValueError('saved subset label/mask arrays differ by method')
        label_hashes[key]=digest
        labels['k_stride']=ident['k_stride']
        expected=tables[role][key]
        from_probability=probability_metrics(p,labels)
        prob_bounds,prob_ties=pck_bounds(p,labels)
        p_errors,p_ok=compare_subset_metrics(from_probability,expected,prob_bounds,PROBABILITY_METRIC_TOLERANCE)
        energy=gaussian_energy(inputs[key],payloads[role]['config'])
        lp=stable_log_probability(energy);energy_probability=np.exp(lp)
        reconstructed=probability_metrics(energy_probability,labels,lp)
        energy_bounds,energy_ties=pck_bounds(energy,labels,ENERGY_METRIC_TOLERANCE)
        e_errors,e_ok=compare_subset_metrics(reconstructed,expected,energy_bounds,ENERGY_METRIC_TOLERANCE)
        p_error=float(np.abs(energy_probability-p).max())
        max_probability_error=max(max_probability_error,p_error)
        all_compatible &= p_ok and e_ok and p_error<=PROBABILITY_ABSOLUTE_TOLERANCE
        unresolved_total+=from_probability['unresolved_area_ce_rows']
        records.append({'role':role,'identity':ident,'probability_sha256':probability_hashes[str(path)],
            'saved_probability_metrics':from_probability,'probability_metric_checks':p_errors,
            'direct_gaussian_energy_metrics':reconstructed,'energy_metric_checks':e_errors,
            'probability_argmax_tied_point_rows':prob_ties,'energy_near_tied_point_rows':energy_ties,
            'probability_pck_epe_bounds':prob_bounds,'energy_pck_epe_bounds':energy_bounds,
            'maximum_probability_error':p_error,'compatible':bool(p_ok and e_ok and p_error<=PROBABILITY_ABSOLUTE_TOLERANCE)})
    result={'complete':True,'scope':'fixed200 saved probability archives =2pairs/scan/grid/role;40unique input pairs, not a full prediction replay',
        'probability_and_independent_energy_compatible':bool(all_compatible),'subset_archives':len(records),
        'unique_input_pairs':len(inputs),'maximum_probability_error':max_probability_error,
        'unresolved_probability_area_ce_rows_total_across_roles':unresolved_total,
        'ce_recovery_note':'zero/subnormal probabilities with positive GT mass do not identify CE/KL; direct energy reconstruction is a separate check',
        'tolerances':{'energy_mean_metric_nats':ENERGY_METRIC_TOLERANCE,'probability_mean_metric_nats':PROBABILITY_METRIC_TOLERANCE,
            'probability_max_absolute':PROBABILITY_ABSOLUTE_TOLERANCE,'epe_pixels':POINT_EPE_TOLERANCE,
            'pck_distance_boundary_pixels':1e-4,'energy_near_top_score_gap':ENERGY_METRIC_TOLERANCE},
        'all_row_independent_ci':ci,'subset_records':records,'sealed_sources':source_checks,
        'input_sha256':input_hashes,'probability_sha256':probability_hashes,
        'audit_source_sha256':sha(__file__),'lock_sha256':EXPECTED_LOCK,'analysis_sha256':sha(args.analysis),
        'elapsed_seconds':time.time()-started,
        'limits':['Only40unique pairs have independent energy/probability verification, not all480.',
            'CI arithmetic audits all row tables, not the unarchived probabilities for other pairs.',
            'Five named scan/place groups, with no guarantee of physical-room independence; no training-seed repeats.',
            'Candidate/config/gates remain frozen even if their outcome is negative.']}
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({name:value for name,value in result.items() if name not in ['subset_records','all_row_independent_ci','sealed_sources','input_sha256','probability_sha256']},indent=2))
    print('INDEPENDENT_GATES',json.dumps(ci['gates'],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('results/round6_confirmation'))
    parser.add_argument('--analysis',type=Path,default=Path('results/round6_confirmation_analysis/summary.json'))
    parser.add_argument('--out',type=Path,default=Path('results/round6_probability_audit'))
    audit(parser.parse_args())
