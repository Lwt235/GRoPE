"""Independent Round12 saved-prediction, sensor-center and numerical audit.

No Round12 experiment/model function is imported. Geometry references use
NumPy literal slices and SciPy CDF integration from the prior frozen auditors.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy.special import softmax

from experiments import round10_partition_audit as AREF
from experiments import round10_cell_result_audit as BREF
from experiments import round11_result_audit as REF

ROOT=Path(__file__).resolve().parents[1]
SCENES=('harvard_c8/hv_c8_3','brown_bm_4/brown_bm_4')
ROLES=('full4','iso4','full1','point','point_pck','fixed','quad4','quad16')
STATES=('zero','depth09','depth11','ry_plus1','ry_minus1','tx_plus01','tx_minus01')
ORIGINAL=tuple((q,k) for q in ('g8','g16','mixed') for k in ('g8','g16','mixed'))
SCALE=(('g8','g24'),('g24','g8'),('g16','g24'),('g24','g16'),('g24','g24'))
CONFIGS={role:dict(method=method,h=h,temperature=t) for role,method,h,t in (
    ('full4','full4',.5,5.),('iso4','iso4',.5,5.),('full1','full',.25,0.),
    ('point','point',1.,5.),('point_pck','point',.5,5.),('fixed','fixed',.5,5.),
    ('quad4','quad4',.5,5.),('quad16','quad16',.5,5.))}
AUDIT_SOURCES=('experiments/round12_stress_result_audit.py','tests/test_round12_stress_result_audit.py',
    'experiments/round10_partition_audit.py','experiments/round10_cell_result_audit.py',
    'experiments/round11_result_audit.py','experiments/round6_region_data.py',
    'experiments/round5_rgbd_data.py')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def archive(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def verify_files(files,base=None):
    for name,digest in files.items():
        path=Path(name) if base is None else base/name
        if sha(path)!=digest:raise ValueError('artifact hash changed: '+str(path))


def freeze(lock_path):
    if lock_path.exists():raise ValueError('refuse overwrite audit source freeze')
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    sources={n:sha(ROOT/n) for n in AUDIT_SOURCES}
    snapshot=lock_path.with_name(lock_path.stem+'_source.py')
    if snapshot.exists():raise ValueError('refuse overwrite audit snapshot')
    snapshot.write_bytes(Path(__file__).read_bytes())
    write(lock_path,dict(created_unix=time.time(),source_hashes=sources,snapshot=str(snapshot),
        snapshot_sha256=sha(snapshot),stage='before opening new stress metrics or labels',
        scope='all17408 matrices/metrics/PCK; all32 independent sensor center labels; firstpair eachscene full4/quad16 SciPy',
        probability_tolerance=1e-4,metric_tolerance=1e-8))


def verify_audit_source(lock_path):
    lock=read(lock_path);verify_files(lock['source_hashes'],ROOT)
    if sha(lock['snapshot'])!=lock['snapshot_sha256'] or lock['snapshot_sha256']!=sha(__file__):
        raise ValueError('audit snapshot changed')
    return lock


def array_sha(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def perturb_reference(inputs,state):
    factor={'depth09':.9,'depth11':1.1}.get(state,1.)
    point=np.asarray(inputs['q_points3'],np.float64)*factor
    covariance=np.asarray(inputs['q_pixel_cov3'],np.float64)*factor**2
    transform=np.asarray(inputs['q_to_k_transform'],np.float64).copy()
    if state.startswith('ry_'):
        a=np.pi/180*(1 if state=='ry_plus1' else -1);c,s=np.cos(a),np.sin(a)
        transform[:3,:]=np.array([[c,0,s],[0,1,0],[-s,0,c]])@transform[:3,:]
    if state.startswith('tx_'):transform[0,3]+=.01 if state=='tx_plus01' else -.01
    point=point@transform[:3,:3].T+transform[:3,3]
    covariance=transform[:3,:3]@covariance@transform[:3,:3].T
    front=point[:,2]>=.1
    safe=point.copy();safe[:,2]=np.maximum(safe[:,2],.1)
    K=inputs['k_K'];hom=safe@K.T;den=hom[:,2]
    xy=(hom[:,:2]/den[:,None]).clip(-1024,1024)
    J=(K[:2][None]*den[:,None,None]-hom[:,:2,None]*K[2][None,None])/den[:,None,None]**2
    cov=J@covariance@J.transpose(0,2,1)
    eig,vec=np.linalg.eigh((cov+cov.transpose(0,2,1))/2)
    cov=(vec*np.clip(eig,0,16384)[:,None,:])@vec.transpose(0,2,1)
    effective={side:np.nan_to_num(np.asarray(inputs[side+'_predicted_depth'],np.float64),nan=10.,posinf=20.,neginf=.1).clip(.1,20.)*factor for side in ('q','k')}
    return dict(q_to_k_xy=xy,q_to_k_pixel_cov=cov,q_features=inputs['q_features'],k_features=inputs['k_features']),dict(
        transform=transform,front=int(front.sum()),effective={k:array_sha(v) for k,v in effective.items()})


def native_sample(depth,points):
    h,w=depth.shape;finite=np.isfinite(points).all(-1)
    safe=np.where(np.isfinite(points),points,0)
    inside=finite&(safe[:,0]>=-.5)&(safe[:,0]<w-.5)&(safe[:,1]>=-.5)&(safe[:,1]<h-.5)
    address=np.floor(np.clip(safe,[-.5,-.5],[w-.5,h-.5])+.5+1e-9).astype(int)
    address=np.clip(address,[0,0],[w-1,h-1]);value=depth[address[:,1],address[:,0]].astype(np.float64)
    known=inside&np.isfinite(value)&(value>0)
    return np.where(known,value,np.nan),known


def center_reference(centers,qd,kd,Kq,Kk,c2wq,c2wk):
    """Independent native depth sample, pinhole projection and status priority."""
    qnative=(centers+.5)*np.array([qd.shape[1]/192,qd.shape[0]/144])-.5
    z,known=native_sample(qd,qnative)
    rays=np.c_[qnative,np.ones(len(centers))]@np.linalg.inv(Kq).T
    point=rays*np.where(known,z,0)[:,None]
    T=np.linalg.inv(c2wk)@c2wq;point=point@T[:3,:3].T+T[:3,3]
    projected=np.isfinite(point).all(-1)&(point[:,2]>0)
    hom=point@Kk.T;den=np.where(projected,hom[:,2],1)
    knative=np.where(projected[:,None],hom[:,:2]/den[:,None],np.nan)
    target=(knative+.5)*np.array([192/kd.shape[1],144/kd.shape[0]])-.5
    inside=np.isfinite(target).all(-1)&(target[:,0]>=-.5)&(target[:,0]<191.5)&(target[:,1]>=-.5)&(target[:,1]<143.5)
    kz,kknown=native_sample(kd,knative);consistent=np.abs(point[:,2]-kz)<=.05+.05*kz
    status=np.zeros(len(centers),np.uint8);active=known.copy();status[~known]=1
    for accepted,code in ((np.isfinite(point).all(-1),6),(projected,2),(inside,3),(kknown,4),(consistent,5)):
        status[active&~accepted]=code;active&=accepted
    target=np.where((known&projected)[:,None],target,np.nan)
    return target,active,status


def decode_sensor(path):
    with Image.open(path) as image:bits=np.asarray(image).astype(np.uint32)
    return (((bits>>3)|(bits<<13))&65535).astype(np.float32)/np.float32(1000)


def check_centers(meta,labels,provenance,parts):
    # This helper resolves timestamps only; geometry/sampling above are independent.
    from experiments.round6_region_data import read_scene_selection
    selection=Path(meta['source_selection']);row=[r for r in read(selection)['scenes'] if r['scene']==meta['scene']][0]
    frames=read_scene_selection(selection.parent,row);qi,ki=meta['pair'];q,k=frames[qi],frames[ki]
    verify_files(meta['sensor_sources'])
    if set(meta['sensor_sources'])!={str(q.depth_path),str(k.depth_path)}:raise ValueError('wrong sensor pair')
    if provenance['sensor_sha256']!=meta['sensor_sources'] or provenance['query_rgb_index']!=qi or provenance['key_rgb_index']!=ki:
        raise ValueError('sensor provenance changed')
    centers=(parts['g24']['bounds'][:,:2]+parts['g24']['bounds'][:,2:]-1)/2
    xy,visible,status=center_reference(centers,decode_sensor(q.depth_path),decode_sensor(k.depth_path),q.K_native,k.K_native,q.c2w,k.c2w)
    np.testing.assert_allclose(labels['g24_point_xy'],xy,rtol=0,atol=1e-10,equal_nan=True)
    np.testing.assert_array_equal(labels['g24_point_visible'],visible)
    np.testing.assert_array_equal(labels['g24_point_status'],status)
    return len(centers)


def part_lookup(part):
    lookup=np.full((144,192),-1,int)
    for i,(x0,y0,x1,y1) in enumerate(part['bounds']):
        if np.any(lookup[y0:y1,x0:x1]>=0):raise ValueError('overlapping partition')
        lookup[y0:y1,x0:x1]=i
    if np.any(lookup<0):raise ValueError('incomplete partition')
    np.testing.assert_array_equal(lookup.ravel(),part['owner'])
    np.testing.assert_array_equal(lookup[4::8,4::8].ravel(),part['atoms'])
    np.testing.assert_array_equal(np.bincount(lookup.ravel()),part['mass'])
    return lookup


def counts_reference(labels,parts,lookups,qname,kname):
    good=labels['gt_visible'];xy=labels['gt_xy'][good]
    if not (np.isfinite(xy).all() and (xy[:,0]>=-.5).all() and (xy[:,0]<191.5).all() and (xy[:,1]>=-.5).all() and (xy[:,1]<143.5).all()):
        raise ValueError('visible GT outside image')
    pix=np.floor(xy+.5+1e-9).astype(int);pix=np.clip(pix,[0,0],[191,143])
    key=lookups[kname][pix[:,1],pix[:,0]];query=parts[qname]['owner'][good]
    direct=np.zeros((len(parts[qname]['mass']),len(parts[kname]['mass'])),np.int64)
    np.add.at(direct,(query,key),1)
    common=np.zeros((108,432),np.int64)
    np.add.at(common,(parts['g16']['owner'][good],lookups['g8'][pix[:,1],pix[:,0]]),1)
    return direct,common


def common_reference(probability,q,k):
    if np.any(q['parent']<0):raise ValueError('no exact common parent')
    lifted=probability[:,k['atoms']]/(k['mass'][k['atoms']]/64)[None]
    return np.stack([(lifted[q['parent']==i]*q['mass'][q['parent']==i,None]).sum(0)/256 for i in range(108)])


def kljs(probability,counts):
    mass=counts.sum(-1);truth=counts/np.maximum(mass,1)[:,None]
    lp=np.log(np.maximum(probability,1e-15));lt=np.log(np.maximum(truth,1e-15));lm=np.log(np.maximum((truth+probability)/2,1e-15))
    return float((counts*(lt-lp)).sum()),float((.5*(truth*(lt-lm)+probability*(lp-lm)).sum(-1)@mass)),int(mass.sum())


def metric_reference(probability,argmax,labels,parts,qname,kname,scenario,direct,common):
    if probability.dtype!=np.float64 or probability.shape!=direct.shape or not np.isfinite(probability).all() or np.any(probability<0):
        raise ValueError('invalid saved probabilities')
    if np.max(np.abs(probability.sum(-1)-1))>1e-12:raise ValueError('unnormalized probability')
    np.testing.assert_array_equal(argmax,probability.argmax(-1))
    direct_kl,js,mass=kljs(probability,direct);kl=direct_kl
    if scenario!='scale':
        kl,js,cmass=kljs(common_reference(probability,parts[qname],parts[kname]),common)
        if mass!=cmass:raise ValueError('different direct/common population')
    predicted=(parts[kname]['bounds'][:,:2]+parts[kname]['bounds'][:,2:]-1)[argmax]/2
    count=int(((np.linalg.norm(predicted-labels[qname+'_point_xy'],axis=-1)<=8)&labels[qname+'_point_visible']).sum())
    return dict(kl_sum=kl,js_sum=js,direct_kl_sum=direct_kl,mass=mass,pck_count=count,n_point=int(labels[qname+'_point_visible'].sum()))


def summary_reference(rows,scenario):
    conditions=[q+'_'+k for q,k in (SCALE if scenario=='scale' else ORIGINAL)]
    pcondition='g24_g8' if scenario=='scale' else 'g8_g8';output={}
    for role in ROLES:
        values=[r for r in rows if r['scenario']==scenario and r['role']==role]
        sc={};sp={}
        for scene in SCENES:
            sc[scene]={}
            for condition in conditions:
                selected=[r for r in values if r['scene']==scene and r['condition']==condition]
                if len(selected)!=16:raise ValueError('wrong scene/condition population')
                sc[scene][condition]=sum(r['kl_sum'] for r in selected)/sum(r['mass'] for r in selected)
            selected=[r for r in values if r['scene']==scene and r['condition']==pcondition]
            sp[scene]=sum(r['pck_count'] for r in selected)/sum(r['n_point'] for r in selected)
        bs={s:float(np.mean(list(v.values()))) for s,v in sc.items()}
        bc={c:float(np.mean([sc[s][c] for s in SCENES])) for c in conditions}
        output[role]=dict(kl=float(np.mean(list(bs.values()))),by_scene=bs,by_condition=bc,
            by_scene_condition=sc,pck8=float(np.mean(list(sp.values()))),pck8_by_scene=sp,
            pck_condition=pcondition,metric='direct' if scenario=='scale' else 'common16_to_8')
    return output


def gate_reference(summaries):
    scale=summaries['scale'];f,p,q=scale['full4'],scale['point'],scale['quad16']
    gain={c:1-f['by_condition'][c]/q['by_condition'][c] for c in f['by_condition']}
    checks=dict(point_relative_gain_at_least_2pct=1-f['kl']/p['kl']>=.02,
        both_scenes_improve_over_point=all(f['by_scene'][s]<p['by_scene'][s] for s in SCENES),
        no_condition_worse_than_quad16_by_2pct=min(gain.values())>=-.02,
        q24_to_k8_pck_loss_no_more_than_half_pp=f['pck8']>=scale['point_pck']['pck8']-.005)
    sg=dict(passed=all(checks.values()),checks=checks,point_relative_gain=1-f['kl']/p['kl'],
        quad16_relative_gain=1-f['kl']/q['kl'],condition_gains_vs_quad16=gain,
        pck_difference_pp=100*(f['pck8']-scale['point_pck']['pck8']))
    stress={}
    for state in STATES:
        f,q,p=(summaries[state][r] for r in ('full4','quad16','point_pck'))
        gain=1-f['kl']/q['kl'];difference=f['pck8']-p['pck8']
        checks=dict(kl_no_more_than_2pct_worse_than_quad16=gain>=-.02,
            pck_loss_no_more_than_half_pp=difference>=-.005,all_predictions_finite=True)
        stress[state]=dict(passed=all(checks.values()),checks=checks,quad16_relative_gain=gain,
            pck_difference_pp=100*difference,full4_absolute_kl=f['kl'],full4_absolute_pck=f['pck8'],
            full4_kl_change_from_zero=f['kl']-summaries['zero']['full4']['kl'],
            full4_pck_change_from_zero_pp=100*(f['pck8']-summaries['zero']['full4']['pck8']))
    return dict(passed=sg['passed'] and all(v['passed'] for v in stress.values()),scale=sg,stress=stress,
                general_research_goal_achieved=False,fresh_confirmation_authorized=False)


def compare(actual,expected,tolerance=1e-8):
    if isinstance(expected,dict):
        if set(actual)!=set(expected):raise ValueError('dictionary fields differ')
        return max([compare(actual[k],v,tolerance) for k,v in expected.items()]+[0.])
    if isinstance(expected,(bool,str)):
        if actual!=expected:raise ValueError('identity or decision differs')
        return 0.
    difference=abs(actual-expected)
    if not np.isfinite(difference) or difference>tolerance:raise ValueError('numeric result differs: '+str(difference))
    return float(difference)


def ref_parts(parts):
    output={}
    for name,p in parts.items():
        for dest,source in [('bounds','bounds'),('area','mass'),('parent_g16','parent')]:output[name+'_'+dest]=p[source]
        output[name+'_centers']=p['mean']*8;output[name+'_covariance']=p['cov']*64
    return output


def numeric_reference(inputs,parts,prediction_root,configs):
    rp=ref_parts(parts);maximum=0.;matrices=0
    for state in STATES:
        obs,_=perturb_reference(inputs,state)
        qnames=('g8','g16','mixed','g24') if state=='zero' else ('g8','g16','mixed')
        queries={q:{'full4':REF.reference_four_support(obs,rp,q),
                    'quad16':AREF.reference_support(obs,rp,q,samples=16)} for q in qnames}
        keys={k:AREF.reference_key(obs,rp,k) for k in qnames}
        for scenario,conditions in ([(state,ORIGINAL),('scale',SCALE)] if state=='zero' else [(state,ORIGINAL)]):
            saved=archive(prediction_root/(scenario+'.npz'))
            for qname,kname in conditions:
                key=keys[kname];bounds=(parts[kname]['bounds']-.5)/8
                for role in ('full4','quad16'):
                    query=queries[qname][role];index=np.linspace(0,len(query['mean'])-1,8).astype(int)
                    q={k:v[index] for k,v in query.items()};cfg=configs[role]
                    geometry=(REF.reference_four_logits(q,key,bounds,'full4',cfg['h']) if role=='full4'
                              else BREF.reference_logits(q,key,bounds,'quad16',cfg['h']))
                    probability=softmax(geometry+cfg['temperature']*(q['features']@key['features'].T),axis=-1)
                    error=float(np.abs(probability-saved[role+'__'+qname+'_'+kname][index]).max())
                    if error>1e-4:raise ValueError(f'SciPy probability mismatch {scenario} {role} {qname} {kname}: {error}')
                    maximum=max(maximum,error);matrices+=1
    return maximum,matrices


def audit(runs,out,source_lock):
    if out.exists():raise ValueError('refuse overwrite independent audit')
    verify_audit_source(source_lock)
    lock=read(runs/'lock.json');completion=read(runs/'completion.json');summary=read(runs/'summary.json')
    if sha(runs/'lock.json')!=(runs/'lock.sha256').read_text().strip() or completion['lock_sha256']!=sha(runs/'lock.json'):
        raise ValueError('broken stress lock')
    verify_files(lock['source_hashes'],ROOT);verify_files(lock['inputs']);verify_files(lock['upstream_metadata']);verify_files(lock['inherited']['files'])
    verify_files(completion['files'],runs)
    actual={str(p.relative_to(runs)) for p in runs.rglob('*') if p.is_file() and p!=runs/'completion.json'}
    if actual!=set(completion['files']):raise ValueError('unbound stress artifact')
    if sha(Path(lock['data_root'])/'completion.json')!=lock['data_completion_sha256'] or sha(runs/'partitions.npz')!=lock['partitions_sha256']:
        raise ValueError('upstream completion or partition changed')
    if lock['configurations']!=CONFIGS or lock['pairs']!=32 or lock['roles']!=list(ROLES):raise ValueError('changed predeclared cohort or controls')
    if summary['lock_sha256']!=sha(runs/'lock.json') or summary['configurations']!=CONFIGS:raise ValueError('summary identity')
    rawparts=archive(runs/'partitions.npz')
    parts={name:{key.split('__')[1]:value for key,value in rawparts.items() if key.startswith(name+'__')} for name in ('g8','g16','mixed','g24')}
    lookups={name:part_lookup(p) for name,p in parts.items()}
    entries=lock['entries'];identities=[(e['scene'],tuple(e['pair'])) for e in entries]
    if len(set(identities))!=32 or {s for s,p in identities}!=set(SCENES):raise ValueError('duplicate/wrong cohort')
    rows=[];metric_error=0.;label_count=0;probability_count=0;numeric_error=0.;numeric_count=0;geometry_count=0
    started=time.time();all_forward={}
    for scene in SCENES:
        directory=runs/'forward'/scene.replace('/','__');c=read(directory/'completion.json')
        if c['stage']!='forward' or c['scene']!=scene or c['lock_sha256']!=sha(runs/'lock.json') or c['sensor_decoded']:
            raise ValueError('forward chain')
        verify_files(c['files'],directory);all_forward[scene]=dict(completion_sha256=sha(directory/'completion.json'),finished_unix=c['finished_unix'])
    for scene in SCENES:
        slug=scene.replace('/','__');directory=runs/'evaluation'/slug;c=read(directory/'completion.json')
        if c['stage']!='evaluate' or c['scene']!=scene or c['lock_sha256']!=sha(runs/'lock.json') or c['forward_chain']!=all_forward:
            raise ValueError('label stage chain')
        if any(v['finished_unix']>c['started_unix'] for v in all_forward.values()):raise ValueError('labels read before both forwards')
        verify_files(c['files'],directory)
        saved_rows=read(directory/'rows.json');mapping={}
        for row in saved_rows:
            identity=(tuple(row['pair']),row['scenario'],row['condition'],row['role'])
            if identity in mapping:raise ValueError('duplicate metric identity')
            if row['scene']!=scene or row['config']!=CONFIGS[row['role']]:raise ValueError('metric configuration changed')
            mapping[identity]=row
        provenance={tuple(p['pair']):p for p in read(directory/'label_provenance.json')}
        geometry=read(runs/'forward'/slug/'geometry.json')
        geom_map={(tuple(x['pair']),x['state']):x for x in geometry}
        if len(geom_map)!=16*7:raise ValueError('geometry diagnostics population')
        numerical=read(runs/'forward'/slug/'numerical.json')
        if len(numerical)!=16*8*68:raise ValueError('numerical diagnostic population')
        if len({(tuple(x['pair']),x['scenario'],x['condition'],x['role']) for x in numerical})!=len(numerical):raise ValueError('duplicate numerical row')
        scene_entries=sorted([e for e in entries if e['scene']==scene],key=lambda e:Path(e['path']).name)
        for entry in scene_entries:
            source=Path(entry['path']);meta=read(source/'manifest.json');labels=archive(directory/'labels'/(source.name+'.npz'))
            if meta['pair']!=entry['pair'] or meta['scene']!=scene:raise ValueError('source pair mismatch')
            if sha(source/'labels.npz')!=meta['label_sha256']:raise ValueError('old GT identity changed')
            old=archive(source/'labels.npz')
            if set(labels)!=set(old)|{'g24_point_xy','g24_point_visible','g24_point_status'}:raise ValueError('label fields changed')
            for name in old:np.testing.assert_array_equal(labels[name],old[name])
            label_count+=check_centers(meta,labels,provenance[tuple(entry['pair'])],parts)
            inputs=archive(source/'inputs.npz')
            for state in STATES:
                obs,diagnostic=perturb_reference(inputs,state);saved=geom_map[(tuple(entry['pair']),state)]
                np.testing.assert_allclose(diagnostic['transform'],saved['transform'],atol=1e-14,rtol=0)
                if diagnostic['effective']!=saved['effective_depth_sha256'] or diagnostic['front']!=saved['predicted_in_front_count']:
                    raise ValueError('changed perturbation observations')
                if array_sha(obs['q_to_k_xy'])!=saved['projected_xy_sha256'] or array_sha(obs['q_to_k_pixel_cov'])!=saved['projected_cov_sha256']:
                    raise ValueError('changed projected observation hash')
                geometry_count+=1
            predroot=runs/'forward'/slug/source.name
            for scenario in ('scale',*STATES):
                predictions=archive(predroot/(scenario+'.npz'));conditions=SCALE if scenario=='scale' else ORIGINAL
                expected={r+'__'+q+'_'+k+s for r in ROLES for q,k in conditions for s in ('','__argmax')}
                if set(predictions)!=expected:raise ValueError('saved prediction population')
                for q,k in conditions:
                    direct,common=counts_reference(labels,parts,lookups,q,k)
                    for role in ROLES:
                        name=role+'__'+q+'_'+k;metric=metric_reference(predictions[name],predictions[name+'__argmax'],labels,parts,q,k,scenario,direct,common)
                        identity=(tuple(entry['pair']),scenario,q+'_'+k,role);row=mapping.pop(identity)
                        metric_error=max(metric_error,compare({k:row[k] for k in metric},metric))
                        metric.update(scene=scene,pair=entry['pair'],scenario=scenario,condition=q+'_'+k,role=role)
                        rows.append(metric);probability_count+=1
            if entry is scene_entries[0]:
                error,count=numeric_reference(inputs,parts,predroot,CONFIGS);numeric_error=max(numeric_error,error);numeric_count+=count
            print('AUDIT',scene,source.name,probability_count,flush=True)
        if mapping:raise ValueError('extra metric rows')
    if probability_count!=17408 or label_count!=1536 or geometry_count!=224:raise ValueError('audit population incomplete')
    summaries={s:summary_reference(rows,s) for s in ('scale',*STATES)}
    summary_error=compare(summary['summary'],summaries);gates=gate_reference(summaries)
    gate_error=compare(summary['gates'],gates)
    if summary['forward_chain']!=all_forward or summary['record_count']!=len(rows) or completion['record_count']!=len(rows):raise ValueError('final populations changed')
    out.mkdir(parents=True)
    write(out/'verification.json',dict(passed=True,scientific_gate_passed=gates['passed'],
        stress_lock_sha256=sha(runs/'lock.json'),stress_summary_sha256=sha(runs/'summary.json'),
        stress_completion_sha256=sha(runs/'completion.json'),audit_source_lock_sha256=sha(source_lock),
        probability_matrices=probability_count,argmax_vectors=probability_count,metric_rows=len(rows),
        independent_sensor_centers=label_count,independent_geometry_states=geometry_count,
        independent_SciPy_probability_matrices=numeric_count,maximum_probability_error=numeric_error,
        maximum_metric_error=metric_error,maximum_summary_error=summary_error,maximum_gate_numeric_error=gate_error,
        source_hashes={n:sha(ROOT/n) for n in AUDIT_SOURCES},seconds=time.time()-started,
        all_forward_before_labels=True,fresh_test_read=False,gates=gates))
    print(json.dumps(read(out/'verification.json')),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('freeze','audit'))
    p.add_argument('--runs',type=Path,default=ROOT/'results/round12_stress')
    p.add_argument('--out',type=Path,default=ROOT/'results/round12_stress_independent_audit')
    p.add_argument('--source-lock',type=Path,default=ROOT/'results/round12_stress_audit_source_lock.json')
    args=p.parse_args()
    if args.stage=='freeze':freeze(args.source_lock)
    else:audit(args.runs,args.out,args.source_lock)

if __name__=='__main__':main()
