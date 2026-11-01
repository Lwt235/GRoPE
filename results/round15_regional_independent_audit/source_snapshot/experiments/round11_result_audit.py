"""Independent references and artifact audit for four local Gaussian supports.

Reference support construction uses literal image rectangles and NumPy moments,
without calling the actual Round11 model or experiment runner.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
from scipy.special import softmax

from experiments import round10_partition_audit as R
from experiments import round10_cell_result_audit as BREF

METHODS=('full4','iso4')
EVAL_ROLES=('full4','iso4','iso4_matched')
TIMING_ROLES=('full4','iso4','quad16','full1')


def reference_four_support(inputs,parts,name):
    """Fit four actual 2x2 spatial strata, including projected pixel footprints.

    xy and within covariance are already projected image observations. Units
    become stride8 and stride8 squared. No GT field is accessed or accepted as a
    selector; every source pixel contributes its actual area once.
    """
    xy=np.asarray(inputs['q_to_k_xy'],np.float64).reshape(144,192,2)/8
    within=np.asarray(inputs['q_to_k_pixel_cov'],np.float64).reshape(144,192,2,2)/64
    bounds=parts[name+'_bounds']
    output={key:[] for key in ('mean','cov','mass','nodes','node_cov','pixel_only_node_cov','node_weights')}
    for x0,y0,x1,y1 in bounds:
        points=xy[y0:y1,x0:x1].reshape(-1,2)
        pixel=within[y0:y1,x0:x1].reshape(-1,2,2)
        mean=points.mean(0);delta=points-mean
        output['mean'].append(mean)
        output['cov'].append(delta.T@delta/len(points)+pixel.mean(0))
        output['mass'].append(len(points))
        nodes=[];local=[];pixel_only=[];weights=[]
        xedges=np.linspace(x0,x1,3,dtype=np.int64);yedges=np.linspace(y0,y1,3,dtype=np.int64)
        for by in range(2):
            for bx in range(2):
                block=xy[yedges[by]:yedges[by+1],xedges[bx]:xedges[bx+1]].reshape(-1,2)
                own=within[yedges[by]:yedges[by+1],xedges[bx]:xedges[bx+1]].reshape(-1,2,2)
                m=block.mean(0);d=block-m
                nodes.append(m);pixel_only.append(own.mean(0))
                local.append(d.T@d/len(block)+own.mean(0))
                weights.append(len(block)/len(points))
        output['nodes'].append(nodes);output['node_cov'].append(local)
        output['pixel_only_node_cov'].append(pixel_only);output['node_weights'].append(weights)
    result={key:np.asarray(value,np.float64) for key,value in output.items()}
    result.update(parent=np.asarray(parts[name+'_parent_g16']),
        features=R.reference_features(inputs['q_features'],bounds))
    return result


def mixture_moments(query):
    weights=query['node_weights']
    mean=np.sum(weights[:,:,None]*query['nodes'],axis=1)
    delta=query['nodes']-mean[:,None]
    covariance=np.sum(weights[:,:,None,None]*(query['node_cov']+delta[:,:,:,None]*delta[:,:,None,:]),axis=1)
    return mean,covariance


def reference_four_logits(query,key,bounds,method,width):
    covariance=query['node_cov'].copy()
    if method=='iso4':
        covariance=np.trace(covariance,axis1=-2,axis2=-1)[:,:,None,None]*np.eye(2)/2
    elif method!='full4':
        raise ValueError('only the two four-Gaussian candidate roles are defined')
    masses=BREF.reference_mass(query['nodes'],covariance,bounds,width)
    mass=np.sum(masses*query['node_weights'][:,:,None],axis=1)
    return np.log(mass+BREF.FLOOR*key['mass'][None]/64)


def numeric(data,out,device='cpu'):
    """Actual model vs independent rectangular slices; observations only."""
    import torch
    from models.round11_stratum_pe import prepare_gaussian_strata,gaussian_stratum_logits
    from models.round10_partition_pe import pool_content
    if out.exists():raise ValueError('refuse overwrite')
    out.mkdir(parents=True);started=time.time();torch.set_num_threads(4)
    paths=[Path(__file__).resolve(),Path(R.__file__).resolve(),
           Path(BREF.__file__).resolve(),R.ROOT/'models/round11_stratum_pe.py',
           R.ROOT/'models/round10_partition_pe.py',R.ROOT/'models/round10_cell_integral_fast.py',
           R.ROOT/'models/round10_cell_integral.py']
    sources={str(p):R.sha(p) for p in paths}
    R.write(out/'lock.json',dict(source_hashes=sources,created_unix=time.time(),
        scope='first metadata calibration pair per old training scan, allquerypartitions, no label files',
        tolerance_relative_to_1_plus_absolute_reference=1e-4,probability_tolerance=1e-4,
        geometry_scope='both methods, all4widths,T0/10, allquery/key rows in all9partitionconditions',device=device))
    parts=R.archive(data/'partitions.npz');first={}
    for path,meta in R.entries(data,'train'):first.setdefault(meta['scene'],(path,meta))
    if len(first)!=3:raise ValueError('expected3 old calibration scans')
    rows=[];conservation=[];probability_rows=[]
    for scene,(path,meta) in first.items():
        inputs=R.archive(path/'inputs.npz')
        xy=torch.as_tensor(inputs['q_to_k_xy'],device=device,dtype=torch.float32)/8
        pixel=torch.as_tensor(inputs['q_to_k_pixel_cov'],device=device,dtype=torch.float32)/64
        for name in R.PARTITIONS:
            owner=torch.as_tensor(parts[name+'_pixel_index'].ravel(),device=device,dtype=torch.long)
            bounds=torch.as_tensor(parts[name+'_bounds'],device=device,dtype=torch.long)
            parent=torch.as_tensor(parts[name+'_parent_g16'],device=device,dtype=torch.long)
            actual=prepare_gaussian_strata(xy,pixel,owner,bounds,parent)
            ref=reference_four_support(inputs,parts,name)
            actual_np={k:v.detach().cpu().numpy() for k,v in actual.items()}
            for field in ('mean','mass','parent','nodes','node_cov','node_weights'):
                delta=np.abs(actual_np[field]-ref[field])
                rows.append(dict(scene=scene,pair=meta['pair'],partition=name,field=field,
                    max_absolute_error=float(delta.max()),max_scaled_error=float((delta/(1+abs(ref[field]))).max())))
            qm,qc=mixture_moments(actual_np)
            conservation.append(dict(scene=scene,pair=meta['pair'],partition=name,
                mean_absolute_error=float(np.abs(qm-ref['mean']).max()),
                covariance_absolute_error=float(np.abs(qc-ref['cov']).max()),
                mean_scaled_error=float((np.abs(qm-ref['mean'])/(1+abs(ref['mean']))).max()),
                covariance_scaled_error=float((np.abs(qc-ref['cov'])/(1+abs(ref['cov']))).max()),
                covariance_min_eigenvalue=float(np.linalg.eigvalsh(actual_np['node_cov']).min())))
            atoms=torch.as_tensor(parts[name+'_pixel_index'][4::8,4::8].ravel(),device=device,dtype=torch.long)
            qfeatures=pool_content(torch.as_tensor(inputs['q_features'],device=device,dtype=torch.float32),atoms,len(bounds))
            for kname in R.PARTITIONS:
                kbounds=(parts[kname+'_bounds']-.5)/8
                katoms=torch.as_tensor(parts[kname+'_pixel_index'][4::8,4::8].ravel(),device=device,dtype=torch.long)
                keyref=R.reference_key(inputs,parts,kname)
                key=dict(cell_bounds=torch.as_tensor(kbounds,device=device,dtype=torch.float64),
                    mass=torch.as_tensor(keyref['mass'],device=device,dtype=torch.float32))
                kfeatures=pool_content(torch.as_tensor(inputs['k_features'],device=device,dtype=torch.float32),katoms,len(kbounds))
                c32=qfeatures@kfeatures.T;c64=ref['features']@keyref['features'].T
                for method in METHODS:
                    for width in R.WIDTHS:
                        with torch.no_grad():actual_score=gaussian_stratum_logits(actual,key,method,width)
                        reference_score=reference_four_logits(ref,keyref,kbounds,method,width)
                        for temperature in (0.,10.):
                            ap=(actual_score+temperature*c32).softmax(-1).cpu().numpy()
                            rp=softmax(reference_score+temperature*c64,axis=-1)
                            delta=np.abs(ap-rp)
                            probability_rows.append(dict(scene=scene,pair=meta['pair'],q=name,k=kname,
                                method=method,h=width,temperature=temperature,query_rows=len(ap),key_rows=ap.shape[1],
                                max_probability_error=float(delta.max()),mean_probability_error=float(delta.mean()),
                                topkey_disagreements=int((ap.argmax(-1)!=rp.argmax(-1)).sum())))
        print('R11_NUMERIC',scene,flush=True)
    maximum=max([r['max_scaled_error'] for r in rows]
                +[r[f] for r in conservation for f in ('mean_scaled_error','covariance_scaled_error')])
    max_probability=max(r['max_probability_error'] for r in probability_rows)
    result=dict(passed=maximum<1e-4 and max_probability<=1e-4,max_scaled_error=maximum,
        max_probability_error=max_probability,probability_rows=probability_rows,rows=rows,conservation=conservation,
        source_hashes=sources,inputs={str(p/'inputs.npz'):R.sha(p/'inputs.npz') for p,m in first.values()},
        labels_opened=False,gate_archives_opened=False,seconds=time.time()-started)
    if any(R.sha(p)!=d for p,d in sources.items()):raise ValueError('source changed during numeric check')
    R.write(out/'verification.json',result)
    if not result['passed']:raise ValueError('R11moment conservation/preparation check failed')
    return result


def rebuild_gates(candidates,baseline_summary,baseline_selection,timing,conditions):
    times={role:float(np.mean([r['total_ms'] for r in timing if r['method']==role])) for role in TIMING_ROLES}
    full=candidates['full4'];pn=baseline_selection['strong_point'];qn=baseline_selection['strong_quadrature']
    point=baseline_summary[pn];q=baseline_summary[qn]
    pg=1-full['kl']/point['kl'];qg=1-full['kl']/q['kl'];ratio=times['full4']/times[qn]
    gains={c:1-full['by_condition'][c]/q['by_condition'][c] for c in conditions}
    checks=dict(point_relative_gain_at_least_2pct=pg>=.02,
        both_scenes_improve_over_point=all(full['by_scene'][s]<point['by_scene'][s] for s in full['by_scene']),
        quadrature_accuracy_or_speed=(qg>=.02 and ratio<=1.25) or (qg>=-.01 and ratio<=.5),
        no_partition_worse_than_quadrature_by_2pct=min(gains.values())>=-.02,
        pck_loss_no_more_than_half_pp=full['pck8']>=baseline_summary['point_pck']['pck8']-.005)
    return dict(passed=all(checks.values()),checks=checks,point_relative_gain=pg,strong_point=pn,
        strong_quadrature=qn,quadrature_relative_gain=qg,full_to_quad_time_ratio=ratio,
        partition_gains=gains,incremental_pe_ms=times,
        pck_difference_pp=100*(full['pck8']-baseline_summary['point_pck']['pck8']),
        full1_relative_gain=1-full['kl']/baseline_summary['full']['kl'],
        gate_quadrature_envelope=min(baseline_summary[m]['kl'] for m in ('quad4','quad16')),
        timing_source='Round11 within-process paired remeasurement; no historical B timing used in gate',
        fresh_confirmation_authorized=False,general_research_goal_achieved=False)


def replay_pair(path,meta,parts,selection,predictions):
    inputs=R.archive(path/'inputs.npz');records=[];parents=BREF.PARENTS
    for qname in R.PARTITIONS:
        complete=reference_four_support(inputs,parts,qname)
        keep=np.isin(complete['parent'],parents)
        query={k:v[keep] for k,v in complete.items()}
        for kname in R.PARTITIONS:
            key=R.reference_key(inputs,parts,kname);bounds=(parts[kname+'_bounds']-.5)/8
            for role,config in selection['configurations'].items():
                score=reference_four_logits(query,key,bounds,config['method'],config['h'])
                prob=softmax(score+config['temperature']*(query['features']@key['features'].T),axis=-1)
                common=R.common_reference(prob,query,key,parts,kname)[parents]
                saved=predictions[role+'__'+qname+'_'+kname][parents]
                error=float(np.abs(common-saved).max())
                if error>1e-4:raise ValueError('Round11 independent probability replay differs')
                records.append(dict(scene=meta['scene'],pair=meta['pair'],q=qname,k=kname,role=role,
                    parents=parents.tolist(),query_rows=int(keep.sum()),max_common_probability_error=error))
    return records


def audit(data,runs,baseline,out,replay=True):
    if out.exists():raise ValueError('refuse overwrite')
    if not (runs/'completion.json').exists():raise ValueError('Round11 incomplete')
    out.mkdir(parents=True);started=time.time()
    paths=[Path(__file__).resolve(),Path(R.__file__).resolve(),Path(BREF.__file__).resolve()]
    source_hashes={str(p):R.sha(p) for p in paths}
    preflight=R.ROOT/'results/round11_audit_source_lock.json';frozen=R.read(preflight)
    for name,digest in frozen['source_hashes'].items():
        if R.sha(name)!=digest:raise ValueError('changed pre-results audit source')
    prior_a=R.ROOT/'results/round10_partition_audit/source_snapshot.py'
    prior_b=R.ROOT/'results/round10_cell_b_audit/source_snapshot.py'
    if R.sha(prior_a)!=R.sha(R.__file__) or R.sha(prior_b)!=R.sha(BREF.__file__):
        raise ValueError('independent reference helper changed after prior audit')
    R.write(out/'audit_lock.json',dict(source_hashes=source_hashes,created_unix=time.time(),
        preflight_source_lock_sha256=R.sha(preflight),run_completion_sha256=R.sha(runs/'completion.json'),
        replay_probability_tolerance=1e-4,fixed_replay_parents=BREF.PARENTS.tolist(),replay=replay))
    for p in paths:(out/(p.stem+'_snapshot.py')).write_bytes(p.read_bytes())
    complete=R.read(runs/'completion.json');lock=R.read(runs/'lock.json')
    if complete['fresh_test_read'] or complete['lock_sha256']!=R.sha(runs/'lock.json'):
        raise ValueError('wrong completion scope')
    if (runs/'lock.sha256').read_text().strip()!=R.sha(runs/'lock.json'):raise ValueError('changed lock')
    for name,digest in complete['files'].items():
        if R.sha(runs/name)!=digest:raise ValueError('changed completed artifact')
    for name,digest in lock['source_hashes'].items():
        if R.sha(R.ROOT/name)!=digest:raise ValueError('changed candidate source')
    if R.sha(data/'completion.json')!=lock['data_completion_sha256'] or R.sha(data/'partitions.npz')!=lock['partition_sha256']:
        raise ValueError('changed data')
    if lock['baseline_root']!=str(baseline.resolve()) or lock['baseline_recalibrated'] or lock['fixed_component_count']!=4:
        raise ValueError('wrong baseline/complexity protocol')
    for name,digest in lock['baseline_files'].items():
        if R.sha(baseline/name)!=digest:raise ValueError('changed frozen B evidence')
    bsummary=R.read(baseline/'summary.json');bselection=R.read(baseline/'selection.json')
    bverified=R.read(R.ROOT/'results/round10_cell_b_audit/verification.json')
    if not bverified['passed'] or bverified['run_completion_sha256']!=R.sha(baseline/'completion.json'):
        raise ValueError('B baselines no longer match independently audited evidence')
    if bverified['run_lock_sha256']!=R.sha(baseline/'lock.json') or bselection['strong_quadrature']!='quad16':
        raise ValueError('B baseline comparator mismatch')
    selection=R.read(runs/'selection.json');result=R.read(runs/'summary.json');selection_sha=R.sha(runs/'selection.json')
    if (runs/'selection.sha256').read_text().strip()!=selection_sha or result['selection_sha256']!=selection_sha:
        raise ValueError('changed candidate selection')
    if (selection['lock_sha256']!=R.sha(runs/'lock.json') or selection['created_unix']<lock['created_unix']
            or selection['candidate_gate_labels_read'] or selection['baseline_recalibrated']):
        raise ValueError('selection chronology/protocol mismatch')
    if selection['baseline_files']!=lock['baseline_files'] or result['baseline_files']!=lock['baseline_files']:
        raise ValueError('baseline chain mismatch')
    if selection['baseline_selection_sha256']!=R.sha(baseline/'selection.json'):
        raise ValueError('changed comparator selection')
    if result['source_hashes']!=lock['source_hashes']:raise ValueError('result source mismatch')
    for name,digest in selection['calibration_files'].items():
        if R.sha(runs/name)!=digest:raise ValueError('changed calibration evidence')
    R.compare_tree(result['baseline_summary'],bsummary['summary'],atol=0.)
    R.compare_tree(result['historical_B_failure_retained'],bsummary['gates'],atol=0.)
    if R.read(runs/'baseline_gate_rows.json')!=R.read(baseline/'gate_rows.json'):
        raise ValueError('B original metric rows changed')
    train=R.entries(data,'train');val=R.entries(data,'val')
    if len(train)!=48 or len(val)!=32:raise ValueError('wrong cohort')
    conditions=[q+'_'+k for q in R.PARTITIONS for k in R.PARTITIONS]
    names=[f'{m}_h{h}_t{t}' for m in METHODS for h in R.WIDTHS for t in R.TEMPERATURES]
    calibration={name:[] for name in names}
    for scene,split in lock['scenes'].items():
        if split!='train':continue
        directory=runs/'calibration'/scene.replace('/','__')
        shard=BREF.check_shard(directory,runs,'calibrate',scene)
        if shard['candidate_gate_labels_read']:raise ValueError('calibration opened candidate gate labels')
        table=R.read(directory/'rows.json')
        if list(table)!=names:raise ValueError('wrong configuration order/grid')
        for name,values in table.items():calibration[name]+=values
    expected_cal={(m['scene'],tuple(m['pair']),c) for _,m in train for c in conditions}
    expected_gate={(m['scene'],tuple(m['pair']),c) for _,m in val for c in conditions}
    gate=R.read(runs/'gate_rows.json')
    if set(gate)!=set(EVAL_ROLES):raise ValueError('wrong candidate gate roles')
    for table,population in ((calibration,expected_cal),(gate,expected_gate)):
        for name,rows in table.items():
            identity=list(map(R.row_identity,rows))
            if len(identity)!=len(set(identity)) or set(identity)!=population:raise ValueError('wrong metric population')
    cal_summary={name:R.aggregate(rows) for name,rows in calibration.items()}
    errors=R.compare_tree(selection['calibration_summaries'],cal_summary)
    chosen={m:min((n for n in names if calibration[n][0]['method']==m),key=lambda n:cal_summary[n]['kl']) for m in METHODS}
    chosen['iso4_matched']=chosen['full4']
    if chosen!=selection['selected_names']:raise ValueError('candidate selection mismatch')
    for m in EVAL_ROLES:
        row=calibration[chosen[m]][0]
        expected_config={k:('iso4' if m=='iso4_matched' and k=='method' else row[k]) for k in ('method','h','temperature')}
        if selection['configurations'][m]!=expected_config:
            raise ValueError('selected config fields mismatch')
    index={role:{R.row_identity(row):row for row in rows} for role,rows in gate.items()}
    for scene,split in lock['scenes'].items():
        shard=BREF.check_shard(runs/'evaluation'/scene.replace('/','__'),runs,'evaluate',scene,selection_sha)
        if shard['labels_read']!=(split=='val'):raise ValueError('wrong evaluation label policy')
    parts=R.archive(data/'partitions.npz');metric_errors=[];replays=[];arrays=0
    first={}
    for path,meta in val:first.setdefault(meta['scene'],path)
    for path,meta in val:
        if R.sha(path/'labels.npz')!=meta['label_sha256']:raise ValueError('changed labels')
        labels=R.archive(path/'labels.npz');counts=R.independent_counts(labels,parts)
        slug=meta['scene'].replace('/','__');predpath=runs/'evaluation'/slug/'predictions'/slug/(path.name+'.npz')
        if predpath.stat().st_mtime<selection['created_unix']:raise ValueError('predictions predate selection')
        predictions=R.archive(predpath)
        fields={m+'__'+c for m in EVAL_ROLES for c in conditions}|{m+'__point_argmax' for m in EVAL_ROLES}
        if set(predictions)!=fields:raise ValueError('wrong saved arrays')
        for m in EVAL_ROLES:
            for condition in conditions:
                metric=R.independent_metrics(predictions[m+'__'+condition],counts);arrays+=1
                row=index[m][(meta['scene'],tuple(meta['pair']),condition)]
                for field,value in metric.items():
                    delta=abs(value-row[field]);metric_errors.append(delta)
                    if delta>2e-7:raise ValueError('saved metric mismatch')
            centers=parts['g8_centers'][predictions[m+'__point_argmax']];valid=labels['g8_point_visible']
            correct=int(np.count_nonzero(valid&(np.linalg.norm(centers-labels['g8_point_xy'],axis=-1)<=8)))
            row=index[m][(meta['scene'],tuple(meta['pair']),'g8_g8')]
            if correct!=row['pck_count'] or int(valid.sum())!=row['n_point']:raise ValueError('PCK mismatch')
        if replay and path==first[meta['scene']]:replays+=replay_pair(path,meta,parts,selection,predictions)
        print('R11_AUDITED',meta['scene'],path.name,flush=True)
    candidate_summary={m:R.aggregate(rows) for m,rows in gate.items()}
    errors+=R.compare_tree(result['summary'],candidate_summary)
    timing=R.read(runs/'timing.json');identities=[(r['scene'],r['condition'],r['method']) for r in timing]
    expected_timing={(s,c,m) for s in lock['scenes'] for c in conditions for m in TIMING_ROLES}
    if len(identities)!=len(set(identities)) or set(identities)!=expected_timing:raise ValueError('wrong paired timing population')
    for r in timing:
        if r['repeats']!=15 or r['warmups']!=3 or r['total_ms']<=0 or not np.isfinite(r['total_ms']):
            raise ValueError('invalid paired timing')
        if r['baseline_config_frozen']!=(r['method'] not in METHODS):raise ValueError('timing baseline drift')
    rebuilt=rebuild_gates(candidate_summary,bsummary['summary'],bselection,timing,conditions)
    errors+=R.compare_tree(result['gates'],rebuilt)
    numerical=R.read(runs/'gate_numerical.json')
    keys=[(r['scene'],tuple(r['pair']),r['condition'],r['method'],r['h']) for r in numerical]
    expected_n={(m['scene'],tuple(m['pair']),c,config['method'],config['h']) for _,m in val for c in conditions for config in selection['configurations'].values()}
    if len(keys)!=len(set(keys)) or set(keys)!=expected_n:raise ValueError('wrong numerical diagnostic population')
    for r in numerical:
        if r['floor_fov_mass']!=BREF.FLOOR*432 or not 0<=r['raw_zero_fov_rows']<=r['raw_tiny_fov_rows']<=r['query_rows']:
            raise ValueError('numerical floor/count inconsistency')
    totals=dict(geometry_records=len(numerical),raw_zero_fov_rows=sum(r['raw_zero_fov_rows'] for r in numerical),
        raw_tiny_fov_rows=sum(r['raw_tiny_fov_rows'] for r in numerical),geometry_query_rows=sum(r['query_rows'] for r in numerical))
    R.compare_tree(result['numerical_totals'],totals,atol=0.)
    if any(R.sha(name)!=digest for name,digest in source_hashes.items()):raise ValueError('audit source changed')
    verification=dict(passed=True,scientific_gate_passed=rebuilt['passed'],checks=rebuilt['checks'],
        source_hashes=source_hashes,preflight_source_lock_sha256=R.sha(preflight),
        run_lock_sha256=R.sha(runs/'lock.json'),run_completion_sha256=R.sha(runs/'completion.json'),
        frozen_B_completion_sha256=R.sha(baseline/'completion.json'),frozen_B_audit_sha256=R.sha(R.ROOT/'results/round10_cell_b_audit/verification.json'),
        calibration_rows=sum(map(len,calibration.values())),saved_prediction_files=len(val),saved_common_probability_arrays=arrays,
        gate_rows=sum(map(len,gate.values())),maximum_saved_metric_error=max(metric_errors),maximum_aggregation_error=max(errors),
        replay_records=replays,maximum_replay_probability_error=max((r['max_common_probability_error'] for r in replays),default=None),
        replay_scope='first gate pair per old scan; all9conditions/3roles; fixed8g16parents with allchildren/allkeys',
        direct_metric_scope='aggregate frozen direct-partition metric rows, not full independent direct prediction replay',
        timing_scope='independent aggregation of newly paired synchronized medians; no independent clock measurement',
        numerical_totals=totals,seconds=time.time()-started,fresh_confirmation_opened=False)
    R.write(out/'verification.json',verification)
    print('R11_AUDIT_VERIFIED',verification['passed'],'SCIENTIFIC_GATE',verification['scientific_gate_passed'],flush=True)
    return verification


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('numeric','results'),required=True)
    parser.add_argument('--data',type=Path,default=R.ROOT/'results/round10_partition_data')
    parser.add_argument('--runs',type=Path,default=R.ROOT/'results/round11_stratum')
    parser.add_argument('--baseline',type=Path,default=R.ROOT/'results/round10_partition_b')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--skip-replay',action='store_true')
    args=parser.parse_args()
    if args.stage=='numeric':numeric(args.data,args.out,args.device)
    else:audit(args.data,args.runs,args.baseline,args.out,not args.skip_replay)


if __name__=='__main__':main()
