"""Fixed old-TRAIN observations only: quartic production vs independent GL64/96.

No label or sensor archive is opened. The independent reference uses literal
NumPy moments/features and Cholesky-whitened Cartesian density quadrature.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import softmax
import torch
from models.round10_partition_pe import prepare_support,pool_content
from models.round15_quartic_tiled import quartic_cell_mass

ROOT=Path(__file__).resolve().parents[1]
SCENES=('harvard_c3/hv_c3_1','harvard_c5/hv_c5_1','harvard_c6/hv_c6_1')
STRIDES=(8,16,24)
WIDTHS=(.25,.5,1.,2.)
TEMPERATURES=(0.,5.,10.)
METHODS=('quartic_full','quartic_point')
FIELDS=('q_to_k_xy','q_to_k_pixel_cov','q_features','k_features')
PARENT=ROOT/'results/round15_quartic_tiled_numerical_audit_frozen'
PARENT_SHA='d4a7c4d376f2845c74ddeeea99f8e9ba5adb60346ec974a2dd333f6e2f6ec8a8'
SOURCE_NAMES=('ROUND15_TRAIN_NUMERICAL_PROTOCOL.md','experiments/round15_train_numerical.py',
    'tests/test_round15_train_numerical.py','models/round10_partition_pe.py',
    'models/round15_quartic_pe.py','models/round15_quartic_tiled.py')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as file:
        while block:=file.read(8*1024**2):digest.update(block)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x') as file:json.dump(value,file,indent=2,allow_nan=False)


def native_bounds(stride):
    return np.array([[x,y,x+stride,y+stride] for y in range(0,144,stride) for x in range(0,192,stride)],np.int64)


def selected_indices(stride):return np.linspace(0,len(native_bounds(stride))-1,8,dtype=np.int64)


def independent_support(inputs,stride,indices):
    xy=np.asarray(inputs['q_to_k_xy'],np.float64).reshape(144,192,2)/8
    within=np.asarray(inputs['q_to_k_pixel_cov'],np.float64).reshape(144,192,2,2)/64
    features=np.asarray(inputs['q_features'],np.float64).reshape(18,24,-1)
    means=[];covariances=[];pooled=[]
    for x0,y0,x1,y1 in native_bounds(stride)[indices]:
        points=xy[y0:y1,x0:x1].reshape(-1,2);mean=points.mean(0);delta=points-mean
        means.append(mean);covariances.append(delta.T@delta/len(points)+within[y0:y1,x0:x1].mean((0,1)))
        feature=features[y0//8:y1//8,x0//8:x1//8].mean((0,1))
        pooled.append(feature/max(np.linalg.norm(feature),1e-12))
    key=np.asarray(inputs['k_features'],np.float64)
    key=key/np.maximum(np.linalg.norm(key,axis=-1,keepdims=True),1e-12)
    return dict(mean=np.asarray(means),cov=np.asarray(covariances),content=np.asarray(pooled)@key.T)


def production_support(inputs,stride,indices):
    yy,xx=np.mgrid[:144,:192];owners=yy//stride*(192//stride)+xx//stride
    bounds=native_bounds(stride);count=len(bounds)
    query=prepare_support(torch.tensor(inputs['q_to_k_xy'],dtype=torch.float32)/8,
        torch.tensor(inputs['q_to_k_pixel_cov'],dtype=torch.float32)/64,
        torch.tensor(owners.ravel(),dtype=torch.long),torch.tensor(bounds),torch.full((count,),-1,dtype=torch.long))
    qf=pool_content(torch.tensor(inputs['q_features'],dtype=torch.float32),
                    torch.tensor(owners[4::8,4::8].ravel()),count)
    kf=pool_content(torch.tensor(inputs['k_features'],dtype=torch.float32),torch.arange(432),432)
    chosen=torch.tensor(indices)
    return dict(mean=query['mean'][chosen],cov=query['cov'][chosen],content=(qf@kf.T)[chosen])


def independent_mass(mean,covariance,bounds,width,order,key_chunk=16):
    """NumPy Cartesian positive GL rule; no model quadrature helper is called."""
    mean=np.asarray(mean,np.float64);covariance=np.asarray(covariance,np.float64)
    bounds=np.asarray(bounds,np.float64)
    if mean.ndim!=2 or mean.shape[1]!=2 or covariance.shape!=(len(mean),2,2):raise ValueError('reference shape')
    nodes,weights=np.polynomial.legendre.leggauss(order);product=weights[:,None]*weights[None,:]
    c=covariance+width**2*np.eye(2);L=np.linalg.cholesky(c)
    output=np.empty((len(mean),len(bounds)),np.float64)
    for q in range(len(mean)):
        norm=np.pi**2*L[q,0,0]*L[q,1,1]
        for start in range(0,len(bounds),key_chunk):
            boxes=bounds[start:start+key_chunk];middle=(boxes[:,:2]+boxes[:,2:])/2;half=(boxes[:,2:]-boxes[:,:2])/2
            x=middle[:,0,None]+half[:,0,None]*nodes
            y=middle[:,1,None]+half[:,1,None]*nodes
            u=(x-mean[q,0])/L[q,0,0]
            v=(y[:,None,:]-mean[q,1]-L[q,1,0]*u[:,:,None])/L[q,1,1]
            radial=u[:,:,None]**2+v*v
            values=np.exp(-radial*radial/(4*np.pi))
            output[q,start:start+len(boxes)]=(values*product).sum((1,2))*half.prod(-1)/norm
    return output


def probability(mass,content,temperature,bounds):
    area=np.prod(bounds[:,2:]-bounds[:,:2],axis=-1)
    return softmax(np.log(mass+1e-30*area[None])+temperature*content,axis=-1)


def row_comparison(actual,reference,reference64,hybrid):
    if any(p.ndim!=1 or not np.isfinite(p).all() or np.any(p<0) or abs(p.sum()-1)>1e-12
           for p in (actual,reference,reference64,hybrid)):raise ValueError('invalid probability row')
    a=int(actual.argmax());r=int(reference.argmax());ordered=np.sort(reference)
    return dict(max_probability_error=float(abs(actual-reference).max()),
        reference_convergence_error=float(abs(reference64-reference).max()),
        integration_only_error=float(abs(hybrid-reference).max()),
        production_support_content_error=float(abs(actual-hybrid).max()),
        actual_argmax=a,reference_argmax=r,argmax_changed=bool(a!=r),reference_top_gap=float(ordered[-1]-ordered[-2]),
        reference_probability_at_reference_winner=float(reference[r]),reference_probability_at_actual_winner=float(reference[a]),
        actual_probability_at_reference_winner=float(actual[r]),actual_probability_at_actual_winner=float(actual[a]),
        reference_winner_margin_against_actual=float(reference[r]-reference[a]),
        actual_winner_margin_against_reference=float(actual[a]-actual[r]))


def parent_contract():
    if sha(PARENT/'lock.json')!=PARENT_SHA or (PARENT/'lock.sha256').read_text().strip()!=PARENT_SHA:
        raise ValueError('changed frozen tiled numerical lock')
    lock=read(PARENT/'lock.json')
    for path,digest in lock['source_hashes'].items():
        if sha(ROOT/path)!=digest or sha(PARENT/'sources'/path)!=digest:raise ValueError('changed frozen numerical source')
    report=Path(lock['report'])
    if sha(report/'verification.json')!=lock['report_sha256'] or sha(report/'completion.json')!=lock['completion_sha256']:
        raise ValueError('changed independent numerical report')
    if not read(report/'verification.json')['passed']:raise ValueError('independent numerical report failed')
    return dict(lock_path=str(PARENT/'lock.json'),lock_sha256=PARENT_SHA,report=str(report),
                report_sha256=lock['report_sha256'],completion_sha256=lock['completion_sha256'])


def cohort_records(data):
    complete=read(data/'completion.json')
    if complete['pair_count']!=80 or complete['fresh_test_read']:raise ValueError('unexpected old development population')
    if sha(data/'contract.json')!=complete['contract_sha256']:raise ValueError('old data contract changed')
    output=[]
    for scene in SCENES:
        choices=[x for x in complete['manifests'] if x['scene']==scene and x['split']=='train']
        if len(choices)!=16:raise ValueError('expected16 train records')
        record=min(choices,key=lambda x:Path(x['path']).parent.name);manifest=Path(record['path'])
        if sha(manifest)!=record['sha256']:raise ValueError('train manifest changed')
        meta=read(manifest)
        if meta['split']!='train' or meta['scene']!=scene or meta['pair']!=[30,45] or meta['forward_uses_sensor']:
            raise ValueError('wrong fixed first training observation')
        output.append(dict(scene=scene,pair=meta['pair'],manifest_path=str(manifest),manifest_sha256=record['sha256'],
            input_path=str(manifest.parent/'inputs.npz'),input_sha256=meta['input_sha256']))
    return output


def freeze(out,data):
    if out.exists():raise ValueError('refuse overwrite preflight')
    data=data.resolve();parent=parent_contract();cohort=cohort_records(data)
    for e in cohort:
        if sha(e['input_path'])!=e['input_sha256']:raise ValueError('input changed')
    out.mkdir(parents=True)
    source_hashes={p:sha(ROOT/p) for p in SOURCE_NAMES}
    for p in SOURCE_NAMES:
        dest=out/'sources'/p;dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as f:f.write((ROOT/p).read_bytes())
    lock=dict(created_unix=time.time(),source_hashes=source_hashes,data_root=str(data),
        data_completion_sha256=sha(data/'completion.json'),cohort=cohort,npz_fields_read=list(FIELDS),
        strides=list(STRIDES),query_indices={str(s):selected_indices(s).tolist() for s in STRIDES},
        methods=list(METHODS),widths=list(WIDTHS),temperatures=list(TEMPERATURES),
        integration_order=16,reference_orders=[64,96],parent=parent,
        tolerances=dict(reference_probability=1e-8,actual_probability=1e-4,hybrid_probability=1e-4,
                        support_relative_to_one_plus_reference=1e-4,content_absolute=1e-5),
        geometry_rows=576,probability_rows=1728,labels_read=False,real_train_observations=True,
        validation_or_test_read=False,scientific_gate_passed=False,device='cpu')
    write(out/'lock.json',lock);(out/'lock.sha256').write_text(sha(out/'lock.json')+'\n')
    print('FROZEN_TRAIN_NUMERICAL',sha(out/'lock.json'),flush=True)


def verify_lock(out):
    lock=read(out/'lock.json')
    if sha(out/'lock.json')!=(out/'lock.sha256').read_text().strip():raise ValueError('preflight lock changed')
    if lock['source_hashes']!={p:sha(ROOT/p) for p in SOURCE_NAMES}:raise ValueError('preflight sources changed')
    for p,digest in lock['source_hashes'].items():
        if sha(out/'sources'/p)!=digest:raise ValueError('preflight source snapshot changed')
    if parent_contract()!=lock['parent']:raise ValueError('frozen numerical parent changed')
    if cohort_records(Path(lock['data_root']))!=lock['cohort'] or sha(Path(lock['data_root'])/'completion.json')!=lock['data_completion_sha256']:
        raise ValueError('fixed cohort changed')
    for e in lock['cohort']:
        if sha(e['input_path'])!=e['input_sha256']:raise ValueError('forward input changed')
    return lock


@torch.no_grad()
def run(out):
    lock=verify_lock(out)
    if (out/'started.json').exists() or (out/'completion.json').exists():raise ValueError('refuse overwrite or silent resume')
    torch.set_num_threads(4);started=time.time();write(out/'started.json',dict(started_unix=started,lock_sha256=sha(out/'lock.json')))
    bounds=(native_bounds(8).astype(float)-.5)/8;tensor_bounds=torch.tensor(bounds,dtype=torch.float64)
    rows=[];geometry=[];supports=[];failures=[]
    for e in lock['cohort']:
        with np.load(e['input_path'],allow_pickle=False) as archive:
            inputs={field:archive[field] for field in FIELDS}
        scene_dir=out/'arrays'/e['scene'].replace('/','__');scene_dir.mkdir(parents=True)
        for stride in STRIDES:
            indices=selected_indices(stride);reference=independent_support(inputs,stride,indices)
            actual=production_support(inputs,stride,indices)
            moment_errors={k:float((abs(actual[k].numpy()-reference[k])/(1+abs(reference[k]))).max()) for k in ('mean','cov')}
            content_error=float(abs(actual['content'].numpy()-reference['content']).max())
            support_passed=max(moment_errors.values())<=1e-4 and content_error<=1e-5
            supports.append(dict(scene=e['scene'],pair=e['pair'],stride=stride,query_indices=indices.tolist(),
                moment_errors=moment_errors,content_error=content_error,passed=support_passed))
            if not support_passed:failures.append(dict(kind='support',scene=e['scene'],stride=stride))
            np.savez_compressed(scene_dir/f'support_g{stride}.npz',indices=indices,
                **{'actual_'+k:v.numpy() for k,v in actual.items()},**{'reference_'+k:v for k,v in reference.items()})
            for width in WIDTHS:
                for method in METHODS:
                    acov=actual['cov'] if method=='quartic_full' else torch.zeros_like(actual['cov'])
                    rcov=reference['cov'] if method=='quartic_full' else np.zeros_like(reference['cov'])
                    production=quartic_cell_mass(actual['mean'],acov,tensor_bounds,bandwidth=width,quadrature_order=16)
                    hybrid=quartic_cell_mass(torch.tensor(reference['mean']),torch.tensor(rcov),tensor_bounds,bandwidth=width,quadrature_order=16).numpy()
                    ref64=independent_mass(reference['mean'],rcov,bounds,width,64)
                    ref96=independent_mass(reference['mean'],rcov,bounds,width,96)
                    pmass=production.numpy()
                    if any(not np.isfinite(m).all() or np.any(m<0) for m in (pmass,hybrid,ref64,ref96)):raise ValueError('invalid raw geometry mass')
                    arrays=dict(actual_raw=pmass,hybrid_raw=hybrid,reference64_raw=ref64,reference96_raw=ref96)
                    for qi,native_index in enumerate(indices):
                        values=dict(scene=e['scene'],pair=e['pair'],stride=stride,query_index=int(native_index),method=method,h=width)
                        for name,mass in (('actual',pmass[qi]),('reference',ref96[qi])):
                            fov=float(mass.sum());values.update({name+'_raw_fov':fov,name+'_zero_fov':bool(fov==0),
                                name+'_tiny_fov':bool(fov<1e-20),name+'_zero_cells':int((mass==0).sum()),
                                name+'_floor_fraction':float(432e-30/(fov+432e-30))})
                        values['raw_max_absolute_error']=float(abs(pmass[qi]-ref96[qi]).max());geometry.append(values)
                    for temperature in TEMPERATURES:
                        # Reproduce production float32 content multiplication before
                        # its addition to float64 geometry, exactly as prior runners.
                        p=(torch.log(production+1e-30)+temperature*actual['content']).softmax(-1).numpy()
                        p64=probability(ref64,reference['content'],temperature,bounds)
                        p96=probability(ref96,reference['content'],temperature,bounds)
                        ph=probability(hybrid,reference['content'],temperature,bounds)
                        arrays.update({f'actual_T{temperature:g}':p,f'hybrid_T{temperature:g}':ph,
                                       f'reference64_T{temperature:g}':p64,f'reference96_T{temperature:g}':p96})
                        for qi,native_index in enumerate(indices):
                            row=row_comparison(p[qi],p96[qi],p64[qi],ph[qi])
                            row.update(scene=e['scene'],pair=e['pair'],stride=stride,query_index=int(native_index),method=method,h=width,temperature=temperature)
                            row['passed']=bool(row['reference_convergence_error']<=1e-8 and row['max_probability_error']<=1e-4 and row['integration_only_error']<=1e-4)
                            if not row['passed']:failures.append({k:row[k] for k in ('scene','stride','query_index','method','h','temperature','max_probability_error','reference_convergence_error','integration_only_error')})
                            rows.append(row)
                    np.savez_compressed(scene_dir/f'g{stride}_{method}_h{width:g}.npz',**arrays)
                print('TRAIN_NUMERICAL',e['scene'],'stride',stride,'h',width,'rows',len(rows),flush=True)
    if len(rows)!=1728 or len(geometry)!=576 or len(supports)!=9:raise ValueError('unexpected predeclared row population')
    verify_lock(out)
    write(out/'probability_rows.json',rows);write(out/'geometry_rows.json',geometry);write(out/'support_rows.json',supports)
    result=dict(passed=not failures,lock_sha256=sha(out/'lock.json'),source_hashes=lock['source_hashes'],
        reference_converged=all(r['reference_convergence_error']<=1e-8 for r in rows),
        max_reference_convergence_error=max(r['reference_convergence_error'] for r in rows),
        max_probability_error=max(r['max_probability_error'] for r in rows),
        max_integration_only_error=max(r['integration_only_error'] for r in rows),
        max_production_support_content_error=max(r['production_support_content_error'] for r in rows),
        argmax_change_count=sum(r['argmax_changed'] for r in rows),
        actual_zero_fov_rows=sum(r['actual_zero_fov'] for r in geometry),actual_tiny_fov_rows=sum(r['actual_tiny_fov'] for r in geometry),
        reference_zero_fov_rows=sum(r['reference_zero_fov'] for r in geometry),reference_tiny_fov_rows=sum(r['reference_tiny_fov'] for r in geometry),
        probability_rows=len(rows),geometry_rows=len(geometry),support_sets=len(supports),failures=failures,
        labels_read=False,validation_or_test_read=False,real_train_observations=True,scientific_gate_passed=False,
        seconds=time.time()-started,interpretation='Numerical preflight only. Argmax disagreement is not GT correctness. All raw/tiny-FOV rows retained; no labels or effectiveness scores were accessed.')
    write(out/'summary.json',result)
    files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
    write(out/'completion.json',dict(passed=result['passed'],lock_sha256=sha(out/'lock.json'),summary_sha256=sha(out/'summary.json'),
        files=files,finished_unix=time.time(),labels_read=False,validation_or_test_read=False))
    print(json.dumps(result,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('freeze','run'))
    p.add_argument('--out',type=Path,default=ROOT/'results/round15_train_numerical')
    p.add_argument('--data',type=Path,default=ROOT/'results/round10_partition_data');args=p.parse_args()
    if args.stage=='freeze':freeze(args.out,args.data)
    else:run(args.out)


if __name__=='__main__':main()
