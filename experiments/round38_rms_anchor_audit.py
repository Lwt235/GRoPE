"""R38 independent geometry and saved-output arithmetic; no integration."""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import time
import numpy as np
import torch
from experiments import round38_rms_anchor_contract as C
from experiments import round38_rms_anchor_arithmetic as AR


def normalized_error(actual,expected):
    actual,expected=np.asarray(actual),np.asarray(expected)
    if actual.shape!=expected.shape or actual.dtype!=expected.dtype or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('independent geometry shape/dtype/finite mismatch')
    scale=max(1.,float(np.max(abs(actual))),float(np.max(abs(expected))))
    return float(np.max(abs(actual/scale-expected/scale)))


def literal_bounds(size):
    return np.array([[x,y,x+size,y+size] for y in range(0,144,size) for x in range(0,192,size)],np.int64)


def pair_check(observations,cache,bank,tolerance=1e-10):
    """Direct moments and per-center solves, independent of producer QR code."""
    if set(observations)!=set(C.OBSERVATION_FIELDS) or set(cache)!=set(C.CACHE_FIELDS):
        raise ValueError('exact input/cache field boundaries required')
    for name,dtype,shape in (('g24__a',np.float64,(48,2)),('g24__anchor_cov',np.float64,(48,2,2)),
        ('g24__tau32',np.float32,(48,)),('g24__q_features',np.float32,(48,128)),
        ('g8__k_features',np.float32,(432,128)),('g8__bounds_px',np.int64,(432,4))):
        value=cache[name]
        if value.dtype!=dtype or value.shape!=shape or not np.isfinite(value).all():raise ValueError('cache format differs: '+name)
    errors={};exact={}
    def compare(name,expected):errors[name]=normalized_error(bank[name],np.asarray(expected))
    def equal(name,expected):
        actual=bank[name];expected=np.asarray(expected)
        exact[name]=bool(actual.dtype==expected.dtype and actual.shape==expected.shape and np.array_equal(actual,expected))
    qbounds=literal_bounds(24);kbounds=literal_bounds(8)
    equal('query_bounds_px',qbounds);equal('source_pixel_count',np.full(48,576,np.int64))
    equal('key_bounds_px',kbounds);equal('key_bounds',(kbounds.astype(np.float64)-.5)/8)
    if not np.array_equal(cache['g8__bounds_px'],kbounds):raise ValueError('cache native cell ordering differs')
    points=np.asarray(observations['q_points3']);cov=np.asarray(observations['q_pixel_cov3'])
    kq,kk,T=(observations[n] for n in ('q_K','k_K','q_to_k_transform'))
    shapes={'q_points3':(27648,3),'q_pixel_cov3':(27648,3,3),'q_K':(3,3),'k_K':(3,3),'q_to_k_transform':(4,4)}
    for name,shape in shapes.items():
        value=observations[name]
        if value.shape!=shape or value.dtype!=np.float64 or not np.isfinite(value).all():raise ValueError('source geometry invalid: '+name)
    if (not np.array_equal(kq[2],[0.,0.,1.]) or not np.array_equal(kk[2],[0.,0.,1.])
            or not np.array_equal(T[3],[0.,0.,0.,1.])):raise ValueError('camera homogeneous row differs')
    depth=np.asarray(observations['q_predicted_depth'],np.float64)
    if depth.shape!=(144,192) or not np.issubdtype(observations['q_predicted_depth'].dtype,np.floating):raise ValueError('depth raster differs')
    depth=np.nan_to_num(depth,nan=10.,posinf=20.,neginf=.1).clip(.1,20.)
    inv=np.linalg.inv(kq);basis=inv[:,:2]/math.sqrt(12);B=basis@basis.T
    compare('source_within_basis',basis)
    H=np.array([[kk[0,0]/8,kk[0,1]/8,kk[0,2]/8],
                [kk[1,0]/8,kk[1,1]/8,kk[1,2]/8],[0.,0.,1.]],np.float64)
    compare('homogeneous_transform',H)
    R,t=T[:3,:3],T[:3,3]
    compare('camera_rotation_diagnostics',np.array([np.max(abs(R@R.T-np.eye(3))),np.linalg.det(R)]))
    values={k:[] for k in ('source_mean','source_covariance','source_covariance_direct','target_mean',
        'target_covariance','mean_homogeneous','covariance_homogeneous','anchor_query_xy','anchor_depth',
        'anchor_source','anchor_target','anchor_homogeneous','anchor_delta','anchor_covariance')}
    expected_front=[]
    for index,(x0,y0,x1,y1) in enumerate(qbounds):
        patch=points.reshape(144,192,3)[y0:y1,x0:x1].reshape(-1,3)
        mean=np.array([math.fsum(map(float,patch[:,j]))/576 for j in range(3)])
        scatter=patch-mean;within=B*(math.fsum(float(z)*float(z) for z in patch[:,2])/576)
        moment=np.einsum('ni,nj->ij',scatter,scatter)/576+within
        direct=np.einsum('ni,nj->ij',scatter,scatter)/576+cov.reshape(144,192,3,3)[y0:y1,x0:x1].mean((0,1))
        expected=(patch[:,2,None,None]**2)*B[None]
        errors['pixel_covariance_%02d'%index]=normalized_error(cov.reshape(144,192,3,3)[y0:y1,x0:x1].reshape(-1,3,3),expected)
        target_mean=R@mean+t;target_cov=R@moment@R.T;mh=H@target_mean;Qh=H@target_cov@H.T
        xy=np.array([(int(x0)+int(x1)-1)/2,(int(y0)+int(y1)-1)/2])
        ix,iy=map(int,np.floor(xy));fx,fy=xy-[ix,iy]
        z=math.fsum(float(depth[iy+dy,ix+dx])*(fx if dx else 1-fx)*(fy if dy else 1-fy)
            for dy in (0,1) for dx in (0,1))
        source=np.linalg.solve(kq,np.r_[xy,1.])*z;target=R@source+t;anchor=H@target;delta=mh-anchor
        second=Qh+np.outer(delta,delta)
        row=dict(source_mean=mean,source_covariance=moment,source_covariance_direct=direct,
            target_mean=target_mean,target_covariance=target_cov,mean_homogeneous=mh,covariance_homogeneous=Qh,
            anchor_query_xy=xy,anchor_depth=z,anchor_source=source,anchor_target=target,
            anchor_homogeneous=anchor,anchor_delta=delta,anchor_covariance=second)
        for name,value in row.items():values[name].append(value)
        expected_front.append(target[2]>0)
        for name,expected_cov in (('source_factor',moment),('target_factor',target_cov),('factor_homogeneous',Qh),('anchored_factor',second)):
            factor=bank[name][index]
            if factor.shape!=(3,3) or factor.dtype!=np.float64:raise ValueError('factor format differs')
            errors[name+'_%02d'%index]=normalized_error(factor@factor.T,expected_cov)
        errors['target_factor_mapping_%02d'%index]=normalized_error(bank['target_factor'][index],R@bank['source_factor'][index])
        errors['homogeneous_factor_mapping_%02d'%index]=normalized_error(bank['factor_homogeneous'][index],H@bank['target_factor'][index])
    for name,value in values.items():compare(name,np.asarray(value,np.float64))
    equal('anchor_strict_front',np.asarray(expected_front,bool))
    for actual,old in (('old_a2','g24__a'),('old_anchor_covariance2','g24__anchor_cov'),('tau32','g24__tau32'),
            ('q_features32','g24__q_features'),('k_features32','g8__k_features')):equal(actual,cache[old])
    ca=cache['g24__anchor_cov'];scale=np.maximum(1.,np.max(abs(ca),axis=(1,2)));norm=ca/scale[:,None,None]
    asym=np.max(abs(norm-norm.transpose(0,2,1)),axis=(1,2));compare('old_covariance_asymmetry_normalized',asym)
    if np.any(asym>1e-12):raise ValueError('old covariance asymmetry exceeds inherited bound')
    diagonal=torch.from_numpy(ca).diagonal(dim1=-2,dim2=-1);h2=torch.tensor(.25**2,dtype=torch.float64)
    s=torch.maximum(diagonal.amax(-1),h2);q=h2/s
    tau=(10.*(q/(q+(diagonal/s[:,None]).sum(-1)))).float().numpy()
    exact['tau_formula']=bool(np.array_equal(tau,cache['g24__tau32']))
    with torch.no_grad():
        content=torch.from_numpy(cache['g24__q_features'])@torch.from_numpy(cache['g8__k_features']).T
        bias=torch.from_numpy(cache['g24__tau32'])[:,None]*content
    equal('content32',content.numpy());equal('bias32',bias.numpy());equal('content_bias64',bias.double().numpy())
    return dict(passed=bool(all(exact.values()) and max(errors.values())<=tolerance),
        maximum_normalized_error=max(errors.values()),normalized_errors=errors,exact_checks=exact)


def query_check(bank,actual,index):
    expected=dict(input__mu=bank['anchor_homogeneous'][index],input__factor=bank['anchored_factor'][index],
        input__bounds=bank['key_bounds'],input__content_bias=bank['content_bias64'][index])
    if set(actual)!=set(expected):raise ValueError('model input allowlist differs')
    checks={n:C.array_sha(actual[n])==C.array_sha(v) for n,v in expected.items()}
    return dict(passed=all(checks.values()),canonical_input_bytes_exact=checks)


def load_arrays(path,digest,hashes,fields=None):
    if C.sha(path)!=digest:raise ValueError('archive byte hash differs')
    wanted=set(hashes) if fields is None else set(fields)
    if not wanted<=set(hashes):raise ValueError('missing authenticated fields')
    with np.load(path,allow_pickle=False) as archive:
        if fields is None and set(archive.files)!=wanted:raise ValueError('archive field population differs')
        result={n:archive[n] for n in sorted(wanted)}
    if any(C.array_sha(v)!=hashes[n] for n,v in result.items()):raise ValueError('canonical array hash differs')
    return result


def complete_inventory(directory):
    complete=C.read(directory/'completion.json');C.verify_inventory(directory,complete['files'],exclude=('completion.json',))
    return complete


def input_audit(run,out):
    run,out=Path(run),Path(out);lock,_=C.joint_locks(run,out);prepared=complete_inventory(run/'prepared')
    seal=C.read(run/'prepared_complete.json')
    if (seal['prepared_completion_sha256']!=C.sha(run/'prepared/completion.json') or not prepared['all_inputs_prepared']
        or not seal['all_inputs_prepared'] or seal['records_sha256']!=C.sha(run/'prepared/records.json')
        or seal['pair_records_sha256']!=C.sha(run/'prepared/pair_records.json')
        or seal['query_specs_sha256']!=C.sha(run/'query_specs.json')
        or not 0<=prepared['wall_seconds']<=C.GATES['preparation_max_phase_wall_seconds']):raise ValueError('complete prepared inputs required')
    pairs=C.read(run/'prepared/pair_records.json');rows=C.read(run/'prepared/records.json');specs=C.read(run/'query_specs.json')
    if [p['pair_id'] for p in pairs]!=[e['pair_id'] for e in lock['entries']] or [r['id'] for r in rows]!=[s['id'] for s in specs]:
        raise ValueError('complete ordered input population required')
    directory=out/'input_audit'
    if directory.exists():raise ValueError('preserve prior input audit')
    directory.mkdir();started=time.time();clock=time.perf_counter();deadline=clock+C.GATES['input_audit_max_phase_wall_seconds']
    C.write(directory/'started.json',dict(started_unix=started,**C.SCOPE));checks=[];pair_checks=[]
    for entry,pair in zip(lock['entries'],pairs):
        bank=None;reason=None
        try:
            if time.perf_counter()>=deadline:raise TimeoutError('input audit deadline')
            if not pair['passed']:raise ValueError('prepared pair failed')
            obs=load_arrays(entry['input_path'],entry['input_sha256'],pair['source_array_sha256'],C.OBSERVATION_FIELDS)
            cache=load_arrays(entry['cache_path'],entry['cache_sha256'],entry['cache_array_sha256'],C.CACHE_FIELDS)
            bank=load_arrays(run/pair['archive_path'],pair['archive_sha256'],pair['array_sha256'])
            check=pair_check(obs,cache,bank,C.GATES['input_normalized_absolute'])
        except Exception as error:reason=type(error).__name__+': '+str(error);check=dict(passed=False,failure_reason=reason)
        pair_checks.append(dict(pair_id=entry['pair_id'],**check))
        for row,spec in zip([r for r in rows if r['pair_id']==entry['pair_id']],[s for s in specs if s['pair_id']==entry['pair_id']]):
            try:
                if time.perf_counter()>=deadline:raise TimeoutError('input audit deadline')
                if not check['passed'] or bank is None:raise ValueError(reason or 'independent pair geometry failed')
                if any(row[k]!=v for k,v in spec.items()):raise ValueError('query identity differs')
                actual=load_arrays(run/row['archive_path'],row['archive_sha256'],row['array_sha256'])
                result=query_check(bank,actual,row['query_index'])
            except Exception as error:result=dict(passed=False,failure_reason=type(error).__name__+': '+str(error))
            checks.append(dict(id=spec['id'],**result))
    wall=time.perf_counter()-clock;passed=len(checks)==2304 and all(c['passed'] for c in checks) and wall<=C.GATES['input_audit_max_phase_wall_seconds']
    C.write(directory/'pair_checks.json',pair_checks);C.write(directory/'checks.json',checks)
    C.write(out/'input_verification.json',dict(passed=passed,checked_queries=len(checks),checked_pairs=len(pair_checks),
        producer_lock_sha256=C.sha(run/'lock.json'),audit_lock_sha256=C.sha(out/'lock.json'),
        prepared_complete_sha256=C.sha(run/'prepared_complete.json'),wall_seconds=wall,
        started_unix=started,finished_unix=time.time(),real_inputs_read=True,**C.SCOPE))
    C.write(directory/'completion.json',dict(passed=passed,files=C.inventory(directory),
        verification_sha256=C.sha(out/'input_verification.json'),finished_unix=time.time(),**C.SCOPE))


def audit(run,out):
    run,out=Path(run),Path(out);lock,_=C.joint_locks(run,out)
    complete_inventory(run/'prepared');forward=complete_inventory(run/'forward');input_complete=complete_inventory(out/'input_audit')
    input_v=C.read(out/'input_verification.json')
    if (input_complete['verification_sha256']!=C.sha(out/'input_verification.json')
        or not input_v['passed'] or input_v['checked_pairs']!=48 or input_v['checked_queries']!=2304
        or input_v['prepared_complete_sha256']!=C.sha(run/'prepared_complete.json')
        or not 0<=input_v['wall_seconds']<=C.GATES['input_audit_max_phase_wall_seconds']):
        raise ValueError('complete input verification binding required')
    seal=C.read(run/'forward_complete.json');rows=C.read(run/'forward/records.json');specs=C.read(run/'query_specs.json')
    if (seal['forward_completion_sha256']!=C.sha(run/'forward/completion.json') or seal['forward_records_sha256']!=C.sha(run/'forward/records.json')
        or seal['prepared_complete_sha256']!=C.sha(run/'prepared_complete.json')
        or seal['input_verification_sha256']!=C.sha(out/'input_verification.json')):
        raise ValueError('global production seal changed')
    if [r['id'] for r in rows]!=[s['id'] for s in specs] or len(rows)!=2304:raise ValueError('full production population required')
    if (out/'completion.json').exists() or (out/'checks.json').exists():raise ValueError('retain prior output audit')
    started=time.time();clock=time.perf_counter();deadline=clock+C.GATES['arithmetic_audit_max_phase_wall_seconds']
    C.write(out/'audit_started.json',dict(started_unix=started,**C.SCOPE));checks=[];total=0;trigger=None
    preps={r['id']:r for r in C.read(run/'prepared/records.json')}
    for index,(row,spec) in enumerate(zip(rows,specs)):
        try:
            if time.perf_counter()>=deadline:raise TimeoutError('arithmetic audit deadline')
            if any(row[k]!=v for k,v in spec.items()) or any(row.get(k)!=v for k,v in C.SCOPE.items()):raise ValueError('identity/scope differs')
            if row['index']!=index or row['budget_used_before']!=total:raise ValueError('ordered resource ledger differs')
            bank=load_arrays(run/row['archive_path'],row['archive_sha256'],row['array_sha256'])
            if trigger is not None:
                if (row['attempted'] or row['passed'] or bank or row['work'] or row['charged_cell_node_evaluations']
                    or row['started_unix'] is not None or row['finished_unix'] is not None
                    or row['not_attempted_reason']!='fail_fast_after_first_failed_scheduled_outcome'):
                    raise ValueError('fail-fast suffix fabricated work or outputs')
                arithmetic=dict(passed=False,status='not_attempted')
            else:
                prep=preps[row['id']]
                if row['input_path']!=prep['archive_path'] or row['input_sha256']!=prep['archive_sha256']:raise ValueError('prepared binding differs')
                if row['attempted']:
                    actual={n:bank[n] for n in C.OLD_FIELDS}
                    if any(C.array_sha(v)!=prep['array_sha256'][n] for n,v in actual.items()):raise ValueError('model input bytes differ')
                    if row['passed']:
                        expected=set(C.OLD_FIELDS)|{'argmax','log_raw_mass','log_raw_fov_mass','log_floored_mass',
                            'log_probability','probability','materialized_raw_indices','materialized_raw_mass'}
                        if set(bank)!=expected:raise ValueError('complete model array fields differ')
                        arithmetic=AR.production_check(row,bank,lock['policy'],lock['gates'])
                        if bank['argmax'].dtype!=np.int32 or bank['argmax'].shape!=() or int(bank['argmax'])!=int(bank['probability'].argmax()):raise ValueError('saved winner differs')
                    else:
                        partial=AR.partial_check(row,bank,lock['policy']);arithmetic=dict(passed=False,status='model_failed',partial=partial)
                else:
                    if bank or row['charged_cell_node_evaluations'] or row['work']:raise ValueError('pre-call failure has fabricated numerical output')
                    arithmetic=dict(passed=False,status='precall_or_delivery_failed')
                if not row['passed']:trigger=dict(index=index,id=row['id'],stage=row['failure_stage'],reason=row['failure_reason'])
            charge=row['charged_cell_node_evaluations']
            if type(charge) is not int or not 0<=charge<=C.POLICY['max_cell_node_evaluations']:raise ValueError('invalid per-call cost')
            if row['attempted'] and not row['work_unavailable'] and charge!=row['work']['cell_node_evaluations']:raise ValueError('recorded model work differs from charged cost')
            total+=charge
            if row['budget_used_after']!=total:raise ValueError('cost balance differs')
            result=dict(delivery_integrity_passed=True,arithmetic_passed=arithmetic['passed'],arithmetic=arithmetic)
        except Exception as error:result=dict(delivery_integrity_passed=False,arithmetic_passed=False,failure_reason=type(error).__name__+': '+str(error))
        checks.append(dict(id=spec['id'],**result))
    wall=time.perf_counter()-clock
    integrity=all(c['delivery_integrity_passed'] for c in checks)
    ledger_ok=(total==seal['charged_cell_node_evaluations']==forward['charged_cell_node_evaluations']
        and total<=C.GATES['production_max_phase_cell_node_evaluations']
        and trigger==seal['stop_trigger']==forward['stop_trigger']
        and seal['phase_wall_seconds']==forward['wall_seconds']
        and seal['phase_wall_passed']==forward['phase_wall_passed']==(0<=forward['wall_seconds']<=C.GATES['production_max_phase_wall_seconds'])
        and seal['phase_node_budget_passed']==forward['phase_node_budget_passed']==(not forward['work_unavailable'] and total<=C.GATES['production_max_phase_cell_node_evaluations']))
    passed=bool(integrity and ledger_ok and input_v['passed'] and seal['all_production_completed']
        and all(c['arithmetic_passed'] for c in checks) and wall<=C.GATES['arithmetic_audit_max_phase_wall_seconds'])
    C.write(out/'checks.json',checks)
    C.write(out/'forward_verification.json',dict(all_input_and_output_arithmetic_passed=passed,
        can_evaluate_train_triage=passed,independent_probability_reference=False,checked_pairs=48,checked_queries=len(checks),
        completed_queries=sum(r['passed'] for r in rows),delivery_integrity_passed=integrity,ledger_passed=ledger_ok,
        producer_lock_sha256=C.sha(run/'lock.json'),audit_lock_sha256=C.sha(out/'lock.json'),
        producer_forward_complete_sha256=C.sha(run/'forward_complete.json'),input_verification_sha256=C.sha(out/'input_verification.json'),
        checks_sha256=C.sha(out/'checks.json'),wall_seconds=wall,started_unix=started,finished_unix=time.time(),
        **{k:v for k,v in C.SCOPE.items() if k!='independent_probability_reference'}))
    C.write(out/'completion.json',dict(files=C.inventory(out),all_input_and_output_arithmetic_passed=passed,
        forward_verification_sha256=C.sha(out/'forward_verification.json'),finished_unix=time.time(),**C.SCOPE))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('freeze','inputs','outputs'))
    p.add_argument('--run',type=Path,default=C.DEFAULT_OUT);p.add_argument('--out',type=Path,default=C.DEFAULT_AUDIT)
    a=p.parse_args();C.configure_runtime()
    if a.stage=='freeze':C.freeze(a.out,'audit',producer_root=a.run)
    elif a.stage=='inputs':input_audit(a.run,a.out)
    else:audit(a.run,a.out)


if __name__=='__main__':main()
