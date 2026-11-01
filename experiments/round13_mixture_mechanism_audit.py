"""Independent NumPy/SciPy replay of the label-free Round13 mechanism probe.

No actual mechanism model/runner is imported. All dense pixels and all eight
fixed queries in all three scans receive independent GL64 CDF integration.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import softmax
from experiments import round10_cell_result_audit as REF

ROOT=Path(__file__).resolve().parents[1]
SCENES=('harvard_c3/hv_c3_1','harvard_c5/hv_c5_1','harvard_c6/hv_c6_1')
METHODS=('dense','full4','ou4','full1','quad16')
TEMPERATURES=(0.,5.)
QUERIES=np.array([0,6,13,20,26,33,40,47])
FIELDS=('q_to_k_xy','q_to_k_pixel_cov','q_features','k_features','q_predicted_depth','q_K','k_K','q_to_k_transform')
OWN_SOURCES=('experiments/round13_mixture_mechanism_audit.py','tests/test_round13_mixture_mechanism_audit.py',
    'experiments/round10_cell_result_audit.py','experiments/round10_partition_audit.py')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as file:
        while b:=file.read(8*1024*1024):digest.update(b)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x') as file:json.dump(value,file,indent=2,allow_nan=False)


def verify_files(files,base=None):
    for name,digest in files.items():
        p=Path(name) if base is None else base/name
        if sha(p)!=digest:raise ValueError('changed artifact '+str(p))


def grid(stride):
    return np.array([(x,y,x+stride,y+stride) for y in range(0,144,stride) for x in range(0,192,stride)],np.int64)


def predicted_center_reference(inputs,bounds):
    """Native q24 center has exactly four equal bilinear depth weights."""
    z=np.nan_to_num(np.asarray(inputs['q_predicted_depth'],np.float64),nan=10.,posinf=20.,neginf=.1).clip(.1,20.)
    center=(bounds[:,:2]+bounds[:,2:]-1)/2
    depths=[]
    for x,y in center:
        ix,iy=int(np.floor(x)),int(np.floor(y))
        if x-ix!=.5 or y-iy!=.5:raise ValueError('expected half-integer q24 center')
        depths.append(z[iy:iy+2,ix:ix+2].mean())
    rays=np.linalg.solve(np.asarray(inputs['q_K']),np.c_[center,np.ones(len(center))].T).T
    source=rays*np.asarray(depths)[:,None];T=np.asarray(inputs['q_to_k_transform'])
    target=(T@np.c_[source,np.ones(len(source))].T).T[:,:3]
    front=target[:,2]>=.1;safe=target.copy();safe[:,2]=np.maximum(safe[:,2],.1)
    hom=safe@inputs['k_K'].T;xy=np.clip(hom[:,:2]/hom[:,2,None],-1024,1024)
    fov=np.isfinite(xy).all(-1)&(xy[:,0]>=-.5)&(xy[:,0]<191.5)&(xy[:,1]>=-.5)&(xy[:,1]<143.5)
    return xy,front,fov


def component_reference(inputs,bounds,method):
    xy=np.asarray(inputs['q_to_k_xy'],np.float64).reshape(144,192,2)/8
    within=np.asarray(inputs['q_to_k_pixel_cov'],np.float64).reshape(144,192,2,2)/64
    means=[];covariances=[];weights=[];owners=[]
    for qi,(x0,y0,x1,y1) in enumerate(bounds):
        if method=='dense':
            positions=xy[y0:y1,x0:x1].reshape(-1,2);pixel=within[y0:y1,x0:x1].reshape(-1,2,2)
            means.extend(positions);covariances.extend(pixel);weights.extend(np.full(len(positions),1/len(positions)));owners.extend([qi]*len(positions))
            continue
        side={'full1':1,'full4':2,'quad16':4}[method];step=(x1-x0)//side
        for y in range(y0,y1,step):
            for x in range(x0,x1,step):
                positions=xy[y:y+step,x:x+step].reshape(-1,2);pixel=within[y:y+step,x:x+step].reshape(-1,2,2)
                mean=positions.mean(0);cov=pixel.mean(0)
                if method!='quad16':
                    residual=positions-mean;cov=cov+np.einsum('ni,nj->ij',residual,residual)/len(residual)
                means.append(mean);covariances.append(cov);weights.append(1/(side*side));owners.append(qi)
    return dict(mean=np.asarray(means),cov=np.asarray(covariances),weights=np.asarray(weights),owner=np.asarray(owners),queries=len(bounds))


def moments(bank):
    mean=[];covariance=[]
    for qi in range(bank['queries']):
        keep=bank['owner']==qi;w=bank['weights'][keep];m=bank['mean'][keep];c=bank['cov'][keep]
        mu=np.einsum('n,ni->i',w,m);d=m-mu
        mean.append(mu);covariance.append(np.einsum('n,nij->ij',w,c+d[:,:,None]*d[:,None,:]))
    return np.asarray(mean),np.asarray(covariance)


def smooth_reference(bank):
    mean,cov=moments(bank);rho2=2/3;owner=bank['owner']
    return dict(**{k:v.copy() if hasattr(v,'copy') else v for k,v in bank.items() if k not in ('mean','cov')},
        mean=np.sqrt(rho2)*bank['mean']+(1-np.sqrt(rho2))*mean[owner],
        cov=rho2*bank['cov']+(1-rho2)*cov[owner])


def content_reference(inputs,bounds):
    features=np.asarray(inputs['q_features'],np.float64).reshape(18,24,-1)
    pooled=np.array([features[y0//8:y1//8,x0//8:x1//8].reshape(-1,features.shape[-1]).mean(0) for x0,y0,x1,y1 in bounds])
    keys=np.asarray(inputs['k_features'],np.float64)
    pooled=pooled/np.maximum(np.sqrt(np.sum(pooled*pooled,-1)),1e-12)[:,None]
    keys=keys/np.maximum(np.sqrt(np.sum(keys*keys,-1)),1e-12)[:,None]
    return np.einsum('ic,jc->ij',pooled,keys)


def integral_reference(bank):
    bounds=(grid(8)-.5)/8
    # Independent SciPy CDF + positive GL64 quadrature, including all dense pixels.
    cells=REF.reference_mass(bank['mean'],bank['cov'],bounds,.5,order=64)
    mass=np.stack([(cells[bank['owner']==qi]*bank['weights'][bank['owner']==qi,None]).sum(0) for qi in range(bank['queries'])])
    fov=mass.sum(-1);corrected=mass+1e-30
    return corrected/corrected.sum(-1,keepdims=True),fov


def discrepancies(reference,probability):
    lp=np.log(np.maximum(reference,1e-300));lq=np.log(np.maximum(probability,1e-300));lm=np.log(np.maximum((reference+probability)/2,1e-300))
    return ((reference*(lp-lq)).sum(-1),
        .5*(reference*(lp-lm)+probability*(lq-lm)).sum(-1),np.abs(reference-probability).sum(-1))


def center_rows(probability,xy,valid):
    bounds=grid(8);centers=(bounds[:,:2]+bounds[:,2:]-1)/2;output=[]
    for i,p in enumerate(probability):
        argmax=int(p.argmax());row=dict(argmax=argmax,predicted_center_valid=bool(valid[i]))
        if valid[i]:
            cell=np.floor((xy[i]+.5+1e-9)/8).astype(int);cell=np.clip(cell,[0,0],[23,17]);j=int(cell[1]*24+cell[0])
            competitors=p.copy();competitors[j]=-np.inf;other=competitors.max()
            row.update(center_bin=j,center_probability=float(p[j]),center_probability_margin=float(p[j]-other),
                center_log_margin=float(np.log(p[j])-np.log(other)),center_bin_is_argmax=bool(j==argmax),
                mode_distance_to_predicted_center_px=float(np.sqrt(((centers[argmax]-xy[i])**2).sum())))
        else:
            row.update({k:None for k in ('center_bin','center_probability','center_probability_margin','center_log_margin','center_bin_is_argmax','mode_distance_to_predicted_center_px')})
        output.append(row)
    return output


def compare(actual,expected,tolerance=1e-9):
    if isinstance(expected,dict):
        if set(actual)!=set(expected):raise ValueError('wrong output fields')
        return max([compare(actual[k],v,tolerance) for k,v in expected.items()]+[0.])
    if isinstance(expected,list):
        if len(actual)!=len(expected):raise ValueError('wrong output population')
        return max([compare(a,b,tolerance) for a,b in zip(actual,expected)]+[0.])
    if expected is None or isinstance(expected,(bool,str)):
        if actual!=expected:raise ValueError('wrong identity/decision')
        return 0.
    error=float(abs(actual-expected))
    if not np.isfinite(error) or error>tolerance:raise ValueError('metric differs: '+str(error))
    return error


def summary_reference(rows):
    result=[]
    for method in METHODS:
        for temperature in TEMPERATURES:
            selected=[r for r in rows if r['method']==method and r['temperature']==temperature]
            if len(selected)!=24:raise ValueError('wrong denominator')
            valid=[r for r in selected if r['predicted_center_valid']]
            value=dict(method=method,temperature=temperature,query_count=24,predicted_center_count=len(valid),
                center_bin_argmax_count=sum(r['center_bin_is_argmax'] for r in valid),
                argmax_agreement_with_dense=float(np.mean([r['argmax_agrees_with_dense'] for r in selected])))
            for name in ('kl','js','l1'):value[name]=float(np.mean([r[name] for r in selected]))
            for name in ('center_probability_margin','center_log_margin','mode_distance_to_predicted_center_px'):
                value[name]=float(np.mean([r[name] for r in valid])) if valid else None
            result.append(value)
    return result


def freeze(source_lock,runs):
    if source_lock.exists():raise ValueError('refuse overwrite audit lock')
    source_lock.parent.mkdir(parents=True,exist_ok=True);snapshot=source_lock.with_name(source_lock.stem+'_source.py')
    if snapshot.exists():raise ValueError('refuse overwrite source copy')
    snapshot.write_bytes(Path(__file__).read_bytes())
    write(source_lock,dict(created_unix=time.time(),source_hashes={n:sha(ROOT/n) for n in OWN_SOURCES},
        source_snapshot=str(snapshot),source_snapshot_sha256=sha(snapshot),mechanism_lock_sha256=sha(runs/'lock.json'),
        stage='before auditor opens mechanism arrays/rows/summary; producer completion timing alone was disclosed',
        integration_scope='all3scenes/all8fixedqueries/all432keys, all4smallbanks and all576densepixelcomponents',
        probability_tolerance=1e-4,raw_fov_absolute_tolerance=1e-8,moment_mean_tolerance=1e-10,moment_cov_tolerance=1e-9,
        row_metric_tolerance=1e-9,labels_read=False,diagnosis_only=True,scientific_gate_passed=False))


def audit(runs,out,source_lock):
    if out.exists():raise ValueError('refuse overwrite audit')
    own=read(source_lock);verify_files(own['source_hashes'],ROOT)
    if sha(own['source_snapshot'])!=own['source_snapshot_sha256'] or own['source_snapshot_sha256']!=sha(__file__):raise ValueError('audit source copy changed')
    lock=read(runs/'lock.json');completion=read(runs/'completion.json')
    digest=sha(runs/'lock.json')
    if digest!=own['mechanism_lock_sha256'] or digest!=(runs/'lock.sha256').read_text().strip() or completion['lock_sha256']!=digest:
        raise ValueError('mechanism lock changed')
    verify_files(lock['source_hashes'],ROOT);verify_files(lock['source_hashes'],runs/'sources');verify_files(completion['files'],runs)
    actual={str(p.relative_to(runs)) for p in runs.rglob('*') if p.is_file() and p!=runs/'completion.json'}
    if actual!=set(completion['files']):raise ValueError('unbound completion artifacts')
    if tuple(lock['npz_fields_read'])!=FIELDS or tuple(lock['methods'])!=METHODS or tuple(lock['temperatures'])!=TEMPERATURES:
        raise ValueError('changed input allowlist or method/T population')
    if lock['h']!=.5 or lock['rho_squared']!=2/3 or lock['query_indices']!=QUERIES.tolist() or lock['query_partition']!='g24' or lock['key_partition']!='g8':
        raise ValueError('changed fixed geometry configuration')
    for field in ('labels_read','validation_inputs_read','fresh_test_read','scientific_gate_passed'):
        if lock[field] is not False:raise ValueError('incorrect diagnostic scope')
    data=Path(lock['data_root']);data_complete=read(data/'completion.json')
    if sha(data/'completion.json')!=lock['data_completion_sha256']:raise ValueError('source completion changed')
    if len(lock['cohort'])!=3 or tuple(e['scene'] for e in lock['cohort'])!=SCENES:raise ValueError('incorrect old-train cohort')
    stored_rows=read(runs/'rows.json');mapping={(r['scene'],r['query_index'],r['method'],r['temperature']):r for r in stored_rows}
    if len(mapping)!=240 or len(stored_rows)!=240:raise ValueError('wrong diagnostic population')
    stored_transitions=read(runs/'transitions.json');tm={(r['scene'],r['query_index'],r['method']):r for r in stored_transitions}
    if len(tm)!=120 or len(stored_transitions)!=120:raise ValueError('wrong transition population')
    timing=read(runs/'timing.json');time_map={(r['scene'],r['method']):r for r in timing}
    if len(time_map)!=15 or len(timing)!=15:raise ValueError('wrong timing metadata population')
    mc=read(runs/'moment_checks.json')
    if len(mc)!=9 or {(r['scene'],r['method']) for r in mc}!={(s,m) for s in SCENES for m in ('full4','ou4','full1')}:
        raise ValueError('wrong moment population')
    if any(not 0<=r['mean_max_error']<=1e-10 or not 0<=r['covariance_max_error']<=1e-9 for r in mc):raise ValueError('reported moments failed')
    rows=[];transitions=[];bounds=grid(24)[QUERIES];started=time.time();numeric=[];moments_report=[]
    max_row=max_center=max_content=max_bank=0.
    for item in lock['cohort']:
        members=[x for x in data_complete['manifests'] if x['scene']==item['scene'] and x['split']=='train']
        expected=min(members,key=lambda x:Path(x['path']).parent.name)
        if item['manifest']!=expected['path'] or item['manifest_sha256']!=expected['sha256'] or item['pair']!=[30,45]:raise ValueError('not fixed first train pair')
        if sha(item['manifest'])!=item['manifest_sha256'] or sha(item['inputs'])!=item['input_sha256']:raise ValueError('changed input identity')
        meta=read(item['manifest'])
        if meta['split']!='train' or meta['forward_uses_sensor'] or meta['input_sha256']!=item['input_sha256'] or meta['pair']!=item['pair']:
            raise ValueError('unexpected forward source')
        with np.load(item['inputs'],allow_pickle=False) as z:inputs={k:z[k] for k in FIELDS}
        with np.load(runs/(item['scene'].replace('/','__')+'.npz'),allow_pickle=False) as z:saved={k:z[k] for k in z.files}
        fields={'query_indices','query_bounds','key_bounds','predicted_center_xy','predicted_center_in_front','predicted_center_in_fov','content_scores'}
        fields|={m+s for m in METHODS for s in ('_geometry_probability','_raw_fov','_T0','_T5')}
        fields|={m+'_'+f for m in METHODS if m!='dense' for f in ('mean','cov','weights','owner')}
        if set(saved)!=fields:raise ValueError('extra or missing output arrays')
        np.testing.assert_array_equal(saved['query_indices'],QUERIES);np.testing.assert_array_equal(saved['query_bounds'],bounds)
        np.testing.assert_array_equal(saved['key_bounds'],grid(8)-.5)
        xy,front,fov=predicted_center_reference(inputs,bounds);valid=front&fov
        error=float(np.abs(xy-saved['predicted_center_xy']).max());max_center=max(max_center,error)
        if error>1e-10:raise ValueError('predicted ray center mismatch')
        np.testing.assert_array_equal(front,saved['predicted_center_in_front']);np.testing.assert_array_equal(fov,saved['predicted_center_in_fov'])
        content=content_reference(inputs,bounds);error=float(np.abs(content-saved['content_scores']).max());max_content=max(max_content,error)
        if error>1e-12:raise ValueError('shared pooled content mismatch')
        banks={m:component_reference(inputs,bounds,m) for m in ('dense','full4','full1','quad16')};banks['ou4']=smooth_reference(banks['full4'])
        dense_mean,dense_cov=moments(banks['dense'])
        for method in METHODS:
            bank=banks[method]
            if method!='dense':
                for name in ('mean','cov','weights','owner'):
                    actual_bank=saved[method+'_'+name];expected_bank=bank[name]
                    tolerance=1e-9 if name=='cov' else 1e-10 if name=='mean' else 1e-14
                    error=float(np.abs(actual_bank-expected_bank).max());max_bank=max(max_bank,error)
                    if error>tolerance:raise ValueError('literal support/OU bank mismatch')
                if method in ('full4','ou4','full1'):
                    actual_bank={k:saved[method+'_'+k] for k in ('mean','cov','weights','owner')};actual_bank['queries']=8
                    mean,cov=moments(actual_bank);me=float(np.abs(mean-dense_mean).max());ce=float(np.abs(cov-dense_cov).max())
                    if me>1e-10 or ce>1e-9:raise ValueError('independent moment conservation failed')
                    moments_report.append(dict(scene=item['scene'],method=method,mean_error=me,cov_error=ce))
            reference,fov_reference=integral_reference(bank);actual_p=saved[method+'_geometry_probability'];raw=saved[method+'_raw_fov']
            if actual_p.shape!=(8,432) or actual_p.dtype!=np.float64 or not np.isfinite(actual_p).all() or np.any(actual_p<=0):raise ValueError('invalid geometry probability')
            if np.max(abs(actual_p.sum(-1)-1))>1e-12:raise ValueError('unnormalized geometry probability')
            pe=float(np.abs(reference-actual_p).max());fe=float(np.abs(fov_reference-raw).max())
            if pe>1e-4 or fe>1e-8:raise ValueError(f'independent CDF integral mismatch {method}: {pe}, {fe}')
            numeric.append(dict(scene=item['scene'],method=method,probability_error=pe,raw_fov_error=fe,components=len(bank['mean']),queries=8))
            t=time_map.pop((item['scene'],method))
            if t['pair']!=item['pair'] or t['query_count']!=8 or t['component_count']!=len(bank['mean']):raise ValueError('timing identity')
            if t['raw_zero_fov_rows']!=int((raw==0).sum()) or t['raw_tiny_fov_rows']!=int((raw<1e-20).sum()):raise ValueError('FOV diagnostic changed')
            if any(not np.isfinite(t[k]) or t[k]<0 for k in ('prepare_ms','integration_ms')):raise ValueError('invalid descriptive timing')
        for temp in TEMPERATURES:
            dense=saved['dense_T'+str(int(temp))]
            for method in METHODS:
                p=saved[method+'_T'+str(int(temp))];expected=softmax(np.log(saved[method+'_geometry_probability'])+temp*content,axis=-1)
                np.testing.assert_allclose(p,expected,atol=1e-13,rtol=1e-12)
                kl,js,l1=discrepancies(dense,p);stats=center_rows(p,xy,valid)
                for i,qi in enumerate(QUERIES):
                    row=dict(scene=item['scene'],pair=item['pair'],query_index=int(qi),method=method,temperature=temp,
                        kl=float(kl[i]),js=float(js[i]),l1=float(l1[i]),argmax_agrees_with_dense=bool(p[i].argmax()==dense[i].argmax()),**stats[i])
                    old=mapping.pop((item['scene'],int(qi),method,temp));max_row=max(max_row,compare(old,row));rows.append(row)
        for method in METHODS:
            before=saved[method+'_T0'].argmax(-1);after=saved[method+'_T5'].argmax(-1)
            for i,qi in enumerate(QUERIES):
                row=dict(scene=item['scene'],pair=item['pair'],query_index=int(qi),method=method,
                    argmax_T0=int(before[i]),argmax_T5=int(after[i]),changed=bool(before[i]!=after[i]))
                compare(tm.pop((item['scene'],int(qi),method)),row);transitions.append(row)
        print('INDEPENDENT_MECHANISM_SCENE',item['scene'],'rows',len(rows),flush=True)
    if mapping or tm or time_map:raise ValueError('unused rows')
    summary=read(runs/'summary.json');summary_error=compare(summary['methods'],summary_reference(rows))
    counts={m:sum(r['changed'] for r in transitions if r['method']==m) for m in METHODS}
    compare(summary['transitions'],counts)
    if summary['rows']!=240 or summary['transitions_denominator_per_method']!=24 or completion['rows']!=240:
        raise ValueError('summary population changed')
    for record in (summary,completion):
        for field in ('labels_read','validation_inputs_read','fresh_test_read'):
            if record[field] is not False:raise ValueError('diagnostic scope changed')
    if summary['scientific_gate_passed'] is not False or summary['diagnosis_only'] is not True:raise ValueError('inappropriate gate claim')
    out.mkdir(parents=True)
    write(out/'verification.json',dict(passed=True,diagnosis_only=True,scientific_gate_passed=False,
        mechanism_lock_sha256=digest,mechanism_summary_sha256=sha(runs/'summary.json'),mechanism_completion_sha256=sha(runs/'completion.json'),
        audit_source_lock_sha256=sha(source_lock),query_metric_rows=len(rows),transition_rows=len(transitions),
        independent_integral_matrices=len(numeric),independent_dense_queries=24,dense_pixels_per_query=576,
        maximum_integral_probability_error=max(r['probability_error'] for r in numeric),maximum_raw_fov_error=max(r['raw_fov_error'] for r in numeric),
        maximum_bank_error=max_bank,maximum_predicted_center_error=max_center,maximum_content_error=max_content,
        maximum_row_metric_error=max_row,maximum_summary_error=summary_error,numerical=numeric,moments=moments_report,
        labels_read=False,validation_inputs_read=False,fresh_test_read=False,seconds=time.time()-started,
        interpretation='Independent numerical and artifact validity only. Predicted-pixel fidelity and predicted-center statistics are not sensor truth, PCK, candidate capability, or evidence passing any research gate. R12 failure remains.'))
    print(json.dumps({k:v for k,v in read(out/'verification.json').items() if k not in ('numerical','moments')}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('freeze','audit'))
    p.add_argument('--runs',type=Path,default=ROOT/'results/round13_mixture_mechanism')
    p.add_argument('--out',type=Path,default=ROOT/'results/round13_mixture_mechanism_independent_audit')
    p.add_argument('--source-lock',type=Path,default=ROOT/'results/round13_mixture_mechanism_audit_source_lock.json')
    args=p.parse_args()
    if args.stage=='freeze':freeze(args.source_lock,args.runs)
    else:audit(args.runs,args.out,args.source_lock)

if __name__=='__main__':main()
