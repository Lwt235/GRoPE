"""Independent staged B artifact, probability, metric and selection audit.

Imports only NumPy reference helpers from the independent A auditor. Neither
experiment runner nor actual PE implementation is imported. The replay uses
SciPy CDFs with 64-node positive quadrature, and adaptive scalar SciPy checks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr, softmax

from experiments import round10_partition_audit as R

ROOT=R.ROOT
PARTITIONS=R.PARTITIONS
METHODS=R.METHODS
WIDTHS=R.WIDTHS
TEMPERATURES=R.TEMPERATURES
PARENTS=np.linspace(0,107,8).astype(np.int64)
FLOOR=1e-30


def normal_interval(lo,hi):
    return np.maximum(np.where(lo>=0,ndtr(-lo)-ndtr(-hi),ndtr(hi)-ndtr(lo)),0.)


def reference_mass(mean,cov,bounds,width,order=64):
    """Independent NumPy/SciPy positive integral, no Torch/actual operator."""
    shape=mean.shape[:-1]
    mean=np.asarray(mean,np.float64).reshape(-1,2)
    cov=np.asarray(cov,np.float64).reshape(-1,2,2)+width**2*np.eye(2)
    nodes,weights=np.polynomial.legendre.leggauss(order)
    mid=(bounds[:,0]+bounds[:,2])/2;half=(bounds[:,2]-bounds[:,0])/2
    x=mid[:,None]+half[:,None]*nodes
    output=np.empty((len(mean),len(bounds)),np.float64)
    for start in range(0,len(mean),16):
        m=mean[start:start+16];c=cov[start:start+16]
        sx=np.sqrt(c[:,0,0])[:,None,None]
        slope=(c[:,1,0]/c[:,0,0])[:,None,None]
        conditional_sigma=np.sqrt(c[:,1,1]-c[:,1,0]**2/c[:,0,0])[:,None,None]
        delta=x[None]-m[:,None,0,None]
        ymean=m[:,None,1,None]+slope*delta
        low=(bounds[None,:,1,None]-ymean)/conditional_sigma
        high=(bounds[None,:,3,None]-ymean)/conditional_sigma
        yprob=normal_interval(low,high)
        xdensity=np.exp(-.5*(delta/sx)**2)/(np.sqrt(2*np.pi)*sx)
        output[start:start+16]=half[None]*np.sum(weights*xdensity*yprob,axis=-1)
    return output.reshape(*shape,len(bounds))


def scalar_reference(mean,cov,bounds,width):
    c=np.asarray(cov)+width**2*np.eye(2)
    sx=np.sqrt(c[0,0]);slope=c[1,0]/c[0,0]
    sy=np.sqrt(c[1,1]-c[1,0]**2/c[0,0])
    def density(x):
        ym=mean[1]+slope*(x-mean[0])
        yp=float(normal_interval((bounds[1]-ym)/sy,(bounds[3]-ym)/sy))
        return np.exp(-.5*((x-mean[0])/sx)**2)/(np.sqrt(2*np.pi)*sx)*yp
    value,error=quad(density,bounds[0],bounds[2],epsabs=1e-280,epsrel=1e-10,limit=150)
    return value,error


def transformed_covariance(query,method):
    cov=query['cov'].copy()
    if method in ('point','center'):cov[:]=0
    elif method=='fixed':cov=query['mass'][:,None,None]*np.eye(2)/(64*12)
    elif method=='isotropic':cov=np.trace(cov,axis1=-2,axis2=-1)[:,None,None]*np.eye(2)/2
    elif method=='rotated':
        rot=np.array([[0.,-1.],[1.,0.]])
        cov=rot@cov@rot.T
    elif method!='full':raise ValueError(method)
    return cov


def reference_logits(query,key,bounds,method,width):
    if method.startswith('quad'):
        mass=reference_mass(query['nodes'],query['node_cov'],bounds,width)
        mass=np.sum(mass*query['node_weights'][:,:,None],axis=1)
    else:
        mass=reference_mass(query['mean'],transformed_covariance(query,method),bounds,width)
    return np.log(mass+FLOOR*key['mass'][None]/64)


def check_shard(directory,runs,stage,scene,selection_sha=None):
    record=R.read(directory/'completion.json')
    if (record['stage']!=stage or record['scene']!=scene
            or record['lock_sha256']!=R.sha(runs/'lock.json')):
        raise ValueError('wrong shard identity')
    if selection_sha is not None and record['selection_sha256']!=selection_sha:
        raise ValueError('shard selection mismatch')
    for path,digest in record['files'].items():
        if R.sha(directory/path)!=digest:raise ValueError('changed shard file: '+path)
    return record


def calibration(runs,lock,selection,conditions,train_entries):
    names=[f'{m}_h{h}_t{t}' for m in METHODS for h in WIDTHS for t in TEMPERATURES]
    rows={name:[] for name in names}
    for scene,split in lock['scenes'].items():
        if split!='train':continue
        directory=runs/'calibration'/scene.replace('/','__')
        record=check_shard(directory,runs,'calibrate',scene)
        if record['gate_labels_read']:raise ValueError('calibration opened gate labels')
        table=R.read(directory/'rows.json')
        if list(table)!=names:raise ValueError('calibration config/order mismatch')
        for name in names:rows[name]+=table[name]
    expected={(m['scene'],tuple(m['pair']),c) for _,m in train_entries for c in conditions}
    for name,values in rows.items():
        identities=list(map(R.row_identity,values))
        if len(set(identities))!=len(identities) or set(identities)!=expected:
            raise ValueError('wrong calibration row population: '+name)
    summary={name:R.aggregate(values) for name,values in rows.items()}
    errors=R.compare_tree(selection['calibration_summaries'],summary)
    selected={m:min((n for n in names if rows[n][0]['method']==m),key=lambda n:summary[n]['kl']) for m in METHODS}
    selected['point_pck']=max((n for n in names if rows[n][0]['method'] in ('point','center')),key=lambda n:summary[n]['pck8'])
    selected['rotated']=selected['full']
    if selected!=selection['selected_names']:raise ValueError('selected config mismatch')
    for role,config in selection['configurations'].items():
        chosen=rows[selected[role]][0]
        expected_config={k:('rotated' if role=='rotated' and k=='method' else chosen[k]) for k in ('method','h','temperature')}
        if config!=expected_config:raise ValueError('chosen config mismatch')
    for h in WIDTHS:
        for t in TEMPERATURES:
            errors+=R.compare_tree(summary[f'point_h{h}_t{t}'],summary[f'center_h{h}_t{t}'],atol=1e-12)
    strong_quad=min(('quad4','quad16'),key=lambda m:summary[selected[m]]['kl'])
    strong_point=min(('point','center'),key=lambda m:summary[selected[m]]['kl'])
    if selection['strong_quadrature']!=strong_quad or selection['strong_point']!=strong_point:
        raise ValueError('comparator mismatch')
    return rows,errors


def rebuild_gates(summary,selection,timing,conditions):
    times={m:float(np.mean([r['total_ms'] for r in timing if r['method']==m])) for m in METHODS}
    pointname=selection['strong_point'];quadname=selection['strong_quadrature']
    full,point,quadrature=summary['full'],summary[pointname],summary[quadname]
    pg=1-full['kl']/point['kl'];qg=1-full['kl']/quadrature['kl'];ratio=times['full']/times[quadname]
    gains={c:1-full['by_condition'][c]/quadrature['by_condition'][c] for c in conditions}
    checks=dict(point_relative_gain_at_least_2pct=pg>=.02,
        both_scenes_improve_over_point=all(full['by_scene'][s]<point['by_scene'][s] for s in full['by_scene']),
        quadrature_accuracy_or_speed=(qg>=.02 and ratio<=1.25) or (qg>=-.01 and ratio<=.5),
        no_partition_worse_than_quadrature_by_2pct=min(gains.values())>=-.02,
        pck_loss_no_more_than_half_pp=full['pck8']>=summary['point_pck']['pck8']-.005)
    return dict(passed=all(checks.values()),checks=checks,point_relative_gain=pg,
        strong_point=pointname,strong_quadrature=quadname,quadrature_relative_gain=qg,
        full_to_quad_time_ratio=ratio,partition_gains=gains,incremental_pe_ms=times,
        pck_difference_pp=100*(full['pck8']-summary['point_pck']['pck8']),
        gate_quadrature_envelope=min(summary[m]['kl'] for m in ('quad4','quad16')),
        fresh_confirmation_authorized=False,general_research_goal_achieved=False)


def verify_repair(runs,lock,selection):
    differences=[];comparisons=[]
    for h in WIDTHS:
        for t in TEMPERATURES:
            delta=R.compare_tree(selection['calibration_summaries'][f'point_h{h}_t{t}'],
                selection['calibration_summaries'][f'center_h{h}_t{t}'],atol=1e-12)
            differences+=delta
            comparisons.append(dict(h=h,temperature=t,max_absolute_difference=max(delta)))
    maximum=max(differences)
    repair=selection.get('selection_repair')
    if repair is None:
        if maximum!=0:raise ValueError('nonexact identity without declared repair')
        return dict(applied=False,maximum_identity_difference=maximum)
    path=runs/repair['lock_file'];record=R.read(path)
    if repair['lock_file']!='selection_repair_lock.json' or repair['tolerance']!=1e-12:
        raise ValueError('unexpected repair policy')
    if R.sha(path)!=repair['lock_sha256'] or R.sha(path)!=(runs/'selection_repair_lock.sha256').read_text().strip():
        raise ValueError('repair lock hash mismatch')
    if record['source_hashes']!=repair['source_hashes'] or record['base_lock_sha256']!=R.sha(runs/'lock.json'):
        raise ValueError('repair source/base chain mismatch')
    if set(record['source_hashes'])!={'experiments/round10_cell_selection_repair.py','docs/round10_selection_numerical_fix.md'}:
        raise ValueError('unexpected repair helper/protocol sources')
    for name,digest in record['source_hashes'].items():
        if R.sha(ROOT/name)!=digest:raise ValueError('changed repair helper/protocol')
    if record['calibration_files']!=selection['calibration_files']:
        raise ValueError('calibration changed during repair')
    if not lock['created_unix']<=record['created_unix']<=selection['created_unix']:
        raise ValueError('repair was not frozen before selection')
    if any(record[field] for field in ('candidate_scores_changed','selection_order_changed','effect_gates_changed')):
        raise ValueError('repair exceeds numerical guard scope')
    if (record['guard_diagnostic']['tolerance']!=1e-12
            or record['guard_diagnostic']['max_absolute_difference']!=maximum
            or record['guard_diagnostic']['comparisons']!=comparisons
            or selection['point_center_max_absolute_difference']!=maximum
            or selection['point_center_identity']!='within1e-12'):
        raise ValueError('repair identity diagnostic mismatch')
    return dict(applied=True,repair_lock_sha256=R.sha(path),tolerance=1e-12,
        maximum_identity_difference=maximum,source_hashes=record['source_hashes'],
        created_before_selection=True,candidate_scores_changed=False,effect_gates_changed=False)


def replay_pair(path,meta,parts,selection,predictions):
    inputs=R.archive(path/'inputs.npz');records=[];scalar_checks=[]
    for qname in PARTITIONS:
        supports={budget:R.reference_support(inputs,parts,qname,budget) for budget in (1,4,16)}
        mask=np.isin(supports[1]['parent'],PARENTS)
        supports={b:{k:v[mask] for k,v in q.items()} for b,q in supports.items()}
        for kname in PARTITIONS:
            key=R.reference_key(inputs,parts,kname);bounds=(parts[kname+'_bounds']-.5)/8
            cache={}
            for role,config in selection['configurations'].items():
                method=config['method'];budget=int(method[4:]) if method.startswith('quad') else 1
                query=supports[budget];gkey=(method,config['h'])
                if gkey not in cache:
                    cache[gkey]=reference_logits(query,key,bounds,method,config['h'])
                probability=softmax(cache[gkey]+config['temperature']*(query['features']@key['features'].T),axis=-1)
                common=R.common_reference(probability,query,key,parts,kname)[PARENTS]
                observed=predictions[role+'__'+qname+'_'+kname][PARENTS]
                delta=float(np.max(np.abs(common-observed)))
                if delta>1e-4:raise ValueError('independent B saved probability replay differs: '+str(delta))
                records.append(dict(scene=meta['scene'],pair=meta['pair'],q=qname,k=kname,
                    role=role,parents=PARENTS.tolist(),query_rows=int(mask.sum()),max_common_probability_error=delta))
            # The first retained full query and its closest native key are fixed
            # from geometry, independent of labels and result discrepancies.
            q=supports[1];closest=int(np.argmin(((key['mean']-q['mean'][0])**2).sum(-1)))
            width=selection['configurations']['full']['h']
            adaptive,err=scalar_reference(q['mean'][0],q['cov'][0],bounds[closest],width)
            fixed=reference_mass(q['mean'][:1],q['cov'][:1],bounds[closest:closest+1],width)[0,0]
            if abs(adaptive-fixed)>1e-10:raise ValueError('independent replay quadrature scalar check failed')
            scalar_checks.append(dict(scene=meta['scene'],q=qname,k=kname,key=closest,
                adaptive_value=adaptive,adaptive_error_bound=err,positive64_value=float(fixed),absolute_error=abs(adaptive-fixed)))
    return records,scalar_checks


def audit(data,runs,out,replay=True):
    if out.exists():raise ValueError('refuse overwrite')
    if not (runs/'completion.json').exists():raise ValueError('B run incomplete')
    out.mkdir(parents=True);started=time.time()
    checked={str(Path(__file__).resolve()):R.sha(__file__),str(Path(R.__file__).resolve()):R.sha(R.__file__)}
    preflight=ROOT/'results/round10_cell_audit_source_lock.json'
    preflight_record=R.read(preflight)
    for name,digest in preflight_record['source_hashes'].items():
        if R.sha(name)!=digest:raise ValueError('audit source differs from pre-results snapshot')
    helper_snapshot=ROOT/'results/round10_partition_audit/source_snapshot.py'
    if R.sha(helper_snapshot)!=R.sha(R.__file__):
        raise ValueError('canonical independent helper differs from its frozen A audit snapshot')
    checked[str(helper_snapshot)]=R.sha(helper_snapshot)
    # Freeze independent audit before reading any scores or gate labels.
    R.write(out/'audit_lock.json',dict(source_hashes=checked,run_completion_sha256=R.sha(runs/'completion.json'),
        preflight_source_lock_sha256=R.sha(preflight),
        fixed_replay_parents=PARENTS.tolist(),replay_probability_tolerance=1e-4,
        independent_reference='NumPy moments + SciPy CDF/positive64 integration + adaptive scalar quad',
        replay=replay,created_unix=time.time()))
    (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (out/'independent_helper_snapshot.py').write_bytes(Path(R.__file__).read_bytes())
    complete=R.read(runs/'completion.json');lock=R.read(runs/'lock.json')
    if complete['fresh_test_read'] or complete['lock_sha256']!=R.sha(runs/'lock.json'):
        raise ValueError('wrong completion scope or lock')
    if (runs/'lock.sha256').read_text().strip()!=R.sha(runs/'lock.json'):raise ValueError('lock changed')
    for name,digest in complete['files'].items():
        if R.sha(runs/name)!=digest:raise ValueError('changed completed B artifact '+name)
    for name,digest in lock['source_hashes'].items():
        if R.sha(ROOT/name)!=digest:raise ValueError('changed B frozen source')
    if R.sha(data/'completion.json')!=lock['data_completion_sha256'] or R.sha(data/'partitions.npz')!=lock['partition_sha256']:
        raise ValueError('changed data')
    selection=R.read(runs/'selection.json');selection_sha=R.sha(runs/'selection.json')
    result=R.read(runs/'summary.json')
    if (runs/'selection.sha256').read_text().strip()!=selection_sha or result['selection_sha256']!=selection_sha:
        raise ValueError('selection hash mismatch')
    if selection['lock_sha256']!=R.sha(runs/'lock.json') or selection['created_unix']<lock['created_unix']:
        raise ValueError('selection chronology/lock invalid')
    if result['source_hashes']!=lock['source_hashes'] or selection['gate_labels_read_by_B']:
        raise ValueError('source or staged label policy mismatch')
    for name,digest in selection['calibration_files'].items():
        if R.sha(runs/name)!=digest:raise ValueError('changed calibration selection artifact')
    repair_verification=verify_repair(runs,lock,selection)
    train=R.entries(data,'train');val=R.entries(data,'val')
    if len(train)!=48 or len(val)!=32:raise ValueError('wrong cohort')
    conditions=[q+'_'+k for q in PARTITIONS for k in PARTITIONS]
    train_rows,aggregation_errors=calibration(runs,lock,selection,conditions,train)
    gate_rows=R.read(runs/'gate_rows.json')
    if set(gate_rows)!=set(selection['configurations']):raise ValueError('wrong gate roles')
    expected={(m['scene'],tuple(m['pair']),c) for _,m in val for c in conditions}
    for role,values in gate_rows.items():
        identities=list(map(R.row_identity,values))
        if len(set(identities))!=len(identities) or set(identities)!=expected:raise ValueError('wrong gate population')
    index={role:{R.row_identity(row):row for row in rows} for role,rows in gate_rows.items()}
    for scene,split in lock['scenes'].items():
        directory=runs/'evaluation'/scene.replace('/','__')
        shard=check_shard(directory,runs,'evaluate',scene,selection_sha)
        if shard['labels_read']!=(split=='val'):raise ValueError('wrong evaluation label policy')
        if split=='val':R.compare_tree(R.read(directory/'rows.json'),
            {role:[row for row in rows if row['scene']==scene] for role,rows in gate_rows.items()},atol=0.)
    parts=R.archive(data/'partitions.npz');metric_errors=[];replays=[];scalars=[]
    first={}
    for path,meta in val:first.setdefault(meta['scene'],path)
    array_count=0
    for path,meta in val:
        if R.sha(path/'labels.npz')!=meta['label_sha256']:raise ValueError('changed labels')
        labels=R.archive(path/'labels.npz');counts=R.independent_counts(labels,parts)
        prediction_path=runs/'evaluation'/meta['scene'].replace('/','__')/'predictions'/meta['scene'].replace('/','__')/(path.name+'.npz')
        if prediction_path.stat().st_mtime<selection['created_unix']:raise ValueError('prediction predates selection')
        predictions=R.archive(prediction_path)
        expected_fields={role+'__'+c for role in gate_rows for c in conditions}|{role+'__point_argmax' for role in gate_rows}
        if set(predictions)!=expected_fields:raise ValueError('wrong saved arrays')
        for role in gate_rows:
            for condition in conditions:
                metric=R.independent_metrics(predictions[role+'__'+condition],counts);array_count+=1
                row=index[role][(meta['scene'],tuple(meta['pair']),condition)]
                for field,value in metric.items():
                    error=abs(value-row[field]);metric_errors.append(error)
                    if error>2e-7:raise ValueError('saved metric mismatch')
            centers=parts['g8_centers'][predictions[role+'__point_argmax']]
            valid=labels['g8_point_visible'];correct=int(np.count_nonzero(valid&(np.linalg.norm(centers-labels['g8_point_xy'],axis=-1)<=8)))
            row=index[role][(meta['scene'],tuple(meta['pair']),'g8_g8')]
            if correct!=row['pck_count'] or int(valid.sum())!=row['n_point']:raise ValueError('saved point metric mismatch')
        if replay and path==first[meta['scene']]:
            rr,ss=replay_pair(path,meta,parts,selection,predictions);replays+=rr;scalars+=ss
        print('B_AUDITED',meta['scene'],path.name,flush=True)
    aggregate={role:R.aggregate(rows) for role,rows in gate_rows.items()}
    aggregation_errors+=R.compare_tree(result['summary'],aggregate)
    timing=R.read(runs/'timing.json')
    expected_timing={(s,c,m) for s in lock['scenes'] for c in conditions for m in METHODS}
    keys={(r['scene'],r['condition'],r['method']) for r in timing}
    if keys!=expected_timing or len(keys)!=len(timing):raise ValueError('wrong timing population')
    if any(r['repeats']!=15 or not np.isfinite(r['total_ms']) or r['total_ms']<=0 for r in timing):raise ValueError('invalid timing')
    gates=rebuild_gates(aggregate,selection,timing,conditions)
    aggregation_errors+=R.compare_tree(result['gates'],gates)
    numeric=R.read(runs/'gate_numerical.json')
    numerical_keys=[(r['scene'],tuple(r['pair']),r['condition'],r['method'],r['h']) for r in numeric]
    expected_num={(m['scene'],tuple(m['pair']),c,config['method'],config['h']) for _,m in val for c in conditions for config in selection['configurations'].values()}
    if len(numerical_keys)!=len(set(numerical_keys)) or set(numerical_keys)!=expected_num:
        raise ValueError('wrong numerical diagnostic population')
    for r in numeric:
        if r['floor_fov_mass']!=FLOOR*432:raise ValueError('background floor area mismatch')
        if not 0<=r['raw_zero_fov_rows']<=r['raw_tiny_fov_rows']<=r['query_rows']:raise ValueError('invalid numerical counts')
    totals=dict(geometry_records=len(numeric),raw_zero_fov_rows=sum(r['raw_zero_fov_rows'] for r in numeric),
        raw_tiny_fov_rows=sum(r['raw_tiny_fov_rows'] for r in numeric),geometry_query_rows=sum(r['query_rows'] for r in numeric))
    R.compare_tree(result['numerical_totals'],totals,atol=0.)
    if any(R.sha(p)!=d for p,d in checked.items()):raise ValueError('audit source changed during run')
    verification=dict(passed=True,scientific_gate_passed=gates['passed'],checks=gates['checks'],
        preflight_source_lock_sha256=R.sha(preflight),
        source_hashes=checked,run_lock_sha256=R.sha(runs/'lock.json'),run_completion_sha256=R.sha(runs/'completion.json'),
        calibration_rows=sum(map(len,train_rows.values())),saved_prediction_files=len(val),
        saved_common_probability_arrays=array_count,gate_rows=sum(map(len,gate_rows.values())),
        maximum_saved_metric_error=max(metric_errors),maximum_aggregation_error=max(aggregation_errors),
        replay_scope='first metadata gate pair per scan; all selected roles and9conditions; fixed8parent query cells/alltheirchildren/allkeys',
        replay_records=replays,maximum_replay_probability_error=max((r['max_common_probability_error'] for r in replays),default=None),
        adaptive_scalar_checks=scalars,maximum_adaptive_check_error=max((r['absolute_error'] for r in scalars),default=None),
        selection_guard_repair=repair_verification,
        direct_kl_scope='aggregate frozen direct rows; not independently recomputed for every pair',
        timing_scope='independently aggregate recorded synchronized medians; clocks not remeasured',
        numerical_totals=totals,seconds=time.time()-started,fresh_confirmation_data_opened=False)
    R.write(out/'verification.json',verification)
    print('B_AUDIT_PASSED',gates['passed'],flush=True)
    return verification


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'results/round10_partition_data')
    parser.add_argument('--runs',type=Path,default=ROOT/'results/round10_partition_b')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--skip-replay',action='store_true')
    args=parser.parse_args();audit(args.data,args.runs,args.out,not args.skip_replay)


if __name__=='__main__':main()
