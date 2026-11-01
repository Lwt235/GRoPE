"""Fixed old-train-only, label-free mixture/content mechanism diagnosis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.round11_compression_diagnostic import (
    cohort, component_bank, mixture_moments, integrate, discrepancy, sha, read, write,
)
from models.round13_moment_smoothing import smooth_gaussian_strata, RHO_SQUARED

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('dense', 'full4', 'ou4', 'full1', 'quad16')
TEMPERATURES = (0., 5.)
QUERIES = np.linspace(0, 47, 8).astype(np.int64)
FIELDS = ('q_to_k_xy', 'q_to_k_pixel_cov', 'q_features', 'k_features',
          'q_predicted_depth', 'q_K', 'k_K', 'q_to_k_transform')
FILES = ('ROUND13_MECHANISM_PROTOCOL.md', 'models/round13_moment_smoothing.py',
         'experiments/round13_mixture_mechanism.py',
         'experiments/round11_compression_diagnostic.py',
         'models/round10_cell_integral_fast.py', 'models/round10_cell_integral.py',
         'tests/test_round13_moment_smoothing.py', 'tests/test_round13_mixture_mechanism.py')


def source_hashes():
    return {name: sha(ROOT/name) for name in FILES}


def spatial_bounds(stride):
    return np.asarray([[x,y,x+stride,y+stride] for y in range(0,144,stride)
                       for x in range(0,192,stride)], np.int64)


def predicted_centers(inputs, bounds):
    """A separate center-ray projection, using RGB depth only; never area mean."""
    depth = np.nan_to_num(np.asarray(inputs['q_predicted_depth'], np.float64),
                         nan=10., posinf=20., neginf=.1).clip(.1,20.)
    if depth.shape != (144,192): raise ValueError('unexpected predicted depth raster')
    points = (np.asarray(bounds)[:,:2]+np.asarray(bounds)[:,2:]-1)/2
    ij = np.floor(points).astype(np.int64)
    frac = points-ij
    z = np.zeros(len(points), np.float64)
    for dy in (0,1):
        for dx in (0,1):
            weight = (frac[:,0] if dx else 1-frac[:,0])*(frac[:,1] if dy else 1-frac[:,1])
            z += depth[ij[:,1]+dy,ij[:,0]+dx]*weight
    rays = np.column_stack([points,np.ones(len(points))])@np.linalg.inv(inputs['q_K']).T
    source = rays*z[:,None]
    transform = np.asarray(inputs['q_to_k_transform'], np.float64)
    target = source@transform[:3,:3].T+transform[:3,3]
    in_front = target[:,2] >= .1
    safe = target.copy(); safe[:,2] = np.maximum(safe[:,2], .1)
    projected = safe@np.asarray(inputs['k_K'], np.float64).T
    xy = (projected[:,:2]/projected[:,2,None]).clip(-1024,1024)
    in_fov = (np.isfinite(xy).all(-1)&(xy[:,0]>=-.5)&(xy[:,0]<191.5)
              &(xy[:,1]>=-.5)&(xy[:,1]<143.5))
    return xy, in_front, in_fov


def content_scores(qfeatures, kfeatures, bounds):
    qfeatures = np.asarray(qfeatures, np.float64)
    kfeatures = np.asarray(kfeatures, np.float64)
    if qfeatures.ndim != 2 or len(qfeatures) != 432 or qfeatures.shape != kfeatures.shape:
        raise ValueError('expected paired base8 feature fields')
    raster = qfeatures.reshape(18,24,-1)
    pooled = np.stack([raster[y0//8:y1//8,x0//8:x1//8].mean((0,1)) for x0,y0,x1,y1 in bounds])
    pooled /= np.maximum(np.linalg.norm(pooled,axis=-1,keepdims=True),1e-12)
    keys = kfeatures/np.maximum(np.linalg.norm(kfeatures,axis=-1,keepdims=True),1e-12)
    return pooled@keys.T


def add_content(probability, content, temperature):
    logits = np.log(np.asarray(probability, np.float64))+temperature*content
    unnormalized = np.exp(logits-logits.max(-1,keepdims=True))
    return unnormalized/unnormalized.sum(-1,keepdims=True)


def smooth_bank(bank):
    n = bank['queries']
    if len(bank['mean']) != n*4: raise ValueError('four ordered strata per query required')
    expected = torch.arange(n,device=bank['owner'].device).repeat_interleave(4)
    if not torch.equal(expected,bank['owner']): raise ValueError('unexpected component order')
    smoothed = smooth_gaussian_strata(dict(nodes=bank['mean'].reshape(n,4,2),
        node_cov=bank['cov'].reshape(n,4,2,2), node_weights=bank['weights'].reshape(n,4)))
    return {**bank,'mean':smoothed['nodes'].reshape(-1,2),'cov':smoothed['node_cov'].reshape(-1,2,2)}


def center_statistics(probability, xy, valid):
    probability = np.asarray(probability,np.float64)
    argmax = probability.argmax(-1)
    key_centers = (spatial_bounds(8)[:,:2]+spatial_bounds(8)[:,2:]-1)/2
    native = np.floor((xy+.5+1.e-9)/8).astype(np.int64)
    native = np.clip(native,[0,0],[23,17]); native = native[:,1]*24+native[:,0]
    out = []
    for index in range(len(probability)):
        row = dict(argmax=int(argmax[index]),predicted_center_valid=bool(valid[index]))
        if valid[index]:
            central = int(native[index]); rest=np.delete(probability[index],central)
            row.update(center_bin=central,center_probability=float(probability[index,central]),
                center_probability_margin=float(probability[index,central]-rest.max()),
                center_log_margin=float(np.log(probability[index,central])-np.log(rest.max())),
                center_bin_is_argmax=bool(central==argmax[index]),
                mode_distance_to_predicted_center_px=float(np.linalg.norm(key_centers[argmax[index]]-xy[index])))
        else:
            row.update({k:None for k in ('center_bin','center_probability','center_probability_margin',
                'center_log_margin','center_bin_is_argmax','mode_distance_to_predicted_center_px')})
        out.append(row)
    return out


def freeze(data, out):
    if out.exists(): raise ValueError('refuse overwrite')
    selected = cohort(data)
    if not np.array_equal(QUERIES,[0,6,13,20,26,33,40,47]): raise ValueError('changed fixed queries')
    lock = dict(source_hashes=source_hashes(),cohort=selected,
        data_completion_sha256=sha(data/'completion.json'),data_root=str(data.resolve()),
        query_indices=QUERIES.tolist(),query_partition='g24',key_partition='g8',
        methods=METHODS,h=.5,temperatures=TEMPERATURES,rho_squared=RHO_SQUARED,
        npz_fields_read=FIELDS,labels_read=False,validation_inputs_read=False,fresh_test_read=False,
        diagnosis_only=True,scientific_gate_passed=False,created_unix=time.time())
    out.mkdir(parents=True); write(out/'lock.json',lock)
    (out/'lock.sha256').write_text(sha(out/'lock.json')+'\n')
    snapshots=out/'sources';snapshots.mkdir()
    for relative,digest in lock['source_hashes'].items():
        destination=snapshots/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes((ROOT/relative).read_bytes())
        if sha(destination)!=digest: raise ValueError('snapshot mismatch')
    return lock


def verify(out):
    lock=read(out/'lock.json')
    if (out/'lock.sha256').read_text().strip()!=sha(out/'lock.json'): raise ValueError('changed lock')
    if source_hashes()!=lock['source_hashes']: raise ValueError('changed source')
    for relative,digest in lock['source_hashes'].items():
        if sha(out/'sources'/relative)!=digest: raise ValueError('changed source snapshot')
    if sha(Path(lock['data_root'])/'completion.json')!=lock['data_completion_sha256']:
        raise ValueError('changed input completion')
    for item in lock['cohort']:
        if sha(item['inputs'])!=item['input_sha256'] or sha(item['manifest'])!=item['manifest_sha256']:
            raise ValueError('changed selected forward input')
    return lock


def summarize(rows):
    result=[]
    for method in METHODS:
        for temperature in TEMPERATURES:
            selected=[r for r in rows if r['method']==method and r['temperature']==temperature]
            if len(selected)!=24: raise ValueError('incomplete diagnostic population')
            valid=[r for r in selected if r['predicted_center_valid']]
            metrics={name:float(np.mean([r[name] for r in selected])) for name in ('kl','js','l1')}
            metrics.update(argmax_agreement_with_dense=float(np.mean([r['argmax_agrees_with_dense'] for r in selected])),
                query_count=len(selected),predicted_center_count=len(valid),
                center_bin_argmax_count=sum(r['center_bin_is_argmax'] for r in valid))
            for name in ('center_probability_margin','center_log_margin','mode_distance_to_predicted_center_px'):
                metrics[name]=float(np.mean([r[name] for r in valid])) if valid else None
            result.append(dict(method=method,temperature=temperature,**metrics))
    return result


def run(out, device):
    lock=verify(out)
    if (out/'started.json').exists(): raise ValueError('refuse partial/complete run overwrite')
    write(out/'started.json',dict(lock_sha256=sha(out/'lock.json'),device=device,
        torch=torch.__version__,numpy=np.__version__,started_unix=time.time()))
    torch.set_num_threads(4); started=time.time()
    bounds=spatial_bounds(24)[QUERIES]
    key_bounds=torch.as_tensor((spatial_bounds(8)-.5)/8,device=device,dtype=torch.float64)
    rows=[];moment_checks=[];timing=[];transitions=[]
    files={name:sha(out/name) for name in ('lock.json','lock.sha256','started.json')}
    files.update({'sources/'+relative:sha(out/'sources'/relative) for relative in lock['source_hashes']})
    def sync():
        if device.startswith('cuda'): torch.cuda.synchronize(torch.device(device))
    with torch.no_grad():
        for item in lock['cohort']:
            # Exact allowlist: neither labels.npz nor arbitrary archive fields are loaded.
            with np.load(item['inputs'],allow_pickle=False) as archive:
                values={name:archive[name].copy() for name in FIELDS}
            xy=torch.as_tensor(values['q_to_k_xy'].reshape(144,192,2)/8,device=device,dtype=torch.float64)
            cov=torch.as_tensor(values['q_to_k_pixel_cov'].reshape(144,192,2,2)/64,device=device,dtype=torch.float64)
            centers,front,fov=predicted_centers(values,bounds); valid=front&fov
            content=content_scores(values['q_features'],values['k_features'],bounds)
            banks={}; probability={}; saved=dict(query_indices=QUERIES,query_bounds=bounds,
                key_bounds=(spatial_bounds(8)-.5),predicted_center_xy=centers,
                predicted_center_in_front=front,predicted_center_in_fov=fov,content_scores=content)
            for method in METHODS:
                sync();tick=time.perf_counter()
                bank=smooth_bank(banks['full4']) if method=='ou4' else component_bank(xy,cov,bounds,method)
                sync();prepare=(time.perf_counter()-tick)*1000; banks[method]=bank
                sync();tick=time.perf_counter(); p,raw=integrate(bank,key_bounds,.5);sync()
                seconds=(time.perf_counter()-tick)*1000
                probability[method]=p.cpu().numpy();raw=raw.cpu().numpy()
                saved[method+'_geometry_probability']=probability[method];saved[method+'_raw_fov']=raw
                # Small moment banks are also retained for independent transform audit.
                if method!='dense':
                    for name in ('mean','cov','weights','owner'):saved[method+'_'+name]=bank[name].cpu().numpy()
                timing.append(dict(scene=item['scene'],pair=item['pair'],method=method,prepare_ms=prepare,
                    integration_ms=seconds,component_count=len(bank['mean']),query_count=8,
                    raw_zero_fov_rows=int((raw==0).sum()),raw_tiny_fov_rows=int((raw<1e-20).sum())))
            dense_mean,dense_cov=mixture_moments(banks['dense'])
            for method in ('full4','ou4','full1'):
                mean,covariance=mixture_moments(banks[method])
                me=float((mean-dense_mean).abs().max());ce=float((covariance-dense_cov).abs().max())
                if me>1e-10 or ce>1e-9:raise ValueError('global moment preservation failed')
                moment_checks.append(dict(scene=item['scene'],method=method,mean_max_error=me,covariance_max_error=ce))
            argmax_by_temperature={}
            for temperature in TEMPERATURES:
                final={m:add_content(p,content,temperature) for m,p in probability.items()}
                argmax_by_temperature[temperature]={m:p.argmax(-1) for m,p in final.items()}
                for method,p in final.items():
                    saved[method+'_T'+str(int(temperature))]=p
                    kl,js,l1=discrepancy(final['dense'],p)
                    center_rows=center_statistics(p,centers,valid)
                    for position,index in enumerate(QUERIES):
                        rows.append(dict(scene=item['scene'],pair=item['pair'],query_index=int(index),
                            method=method,temperature=temperature,kl=float(kl[position]),js=float(js[position]),l1=float(l1[position]),
                            argmax_agrees_with_dense=bool(p[position].argmax()==final['dense'][position].argmax()),
                            **center_rows[position]))
            for method in METHODS:
                before=argmax_by_temperature[0.][method];after=argmax_by_temperature[5.][method]
                for position,index in enumerate(QUERIES):
                    transitions.append(dict(scene=item['scene'],pair=item['pair'],query_index=int(index),method=method,
                        argmax_T0=int(before[position]),argmax_T5=int(after[position]),
                        changed=bool(before[position]!=after[position])))
            path=out/(item['scene'].replace('/','__')+'.npz');np.savez_compressed(path,**saved)
            files[path.name]=sha(path)
            print('MECHANISM_SCENE_COMPLETE',item['scene'],flush=True)
    summary=dict(methods=summarize(rows),transitions={m:sum(r['changed'] for r in transitions if r['method']==m) for m in METHODS},
        transitions_denominator_per_method=24,rows=len(rows),seconds=time.time()-started,
        diagnosis_only=True,scientific_gate_passed=False,labels_read=False,validation_inputs_read=False,fresh_test_read=False,
        limitations='Predicted-pixel fidelity and predicted-center alignment only. No PCK, physical truth, efficiency or research gate.')
    for name,value in (('rows',rows),('summary',summary),('timing',timing),('moment_checks',moment_checks),('transitions',transitions)):
        path=out/(name+'.json');write(path,value);files[path.name]=sha(path)
    verify(out)
    write(out/'completion.json',dict(lock_sha256=sha(out/'lock.json'),files=files,rows=len(rows),
        labels_read=False,validation_inputs_read=False,fresh_test_read=False,completed_unix=time.time()))
    print('MECHANISM_COMPLETE',summary,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','run'))
    parser.add_argument('--out',type=Path,default=ROOT/'results/round13_mixture_mechanism')
    parser.add_argument('--data',type=Path,default=ROOT/'results/round10_partition_data')
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if args.stage=='freeze':
        freeze(args.data,args.out);print('MECHANISM_FROZEN',sha(args.out/'lock.json'),flush=True)
    else:run(args.out,args.device)


if __name__=='__main__':main()
