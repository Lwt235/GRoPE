"""Forward-only Gaussian compression diagnosis against predicted pixel mixtures.

No sensor labels, visibility, scene scores, training or candidate selection are
used. The reference is the same predicted geometry at full pixel resolution,
not physical truth. All probability mixtures normalize only after integration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from models.round10_cell_integral_fast import gaussian_cell_mass_fast

ROOT=Path(__file__).resolve().parents[1]
SCENES=('harvard_c3/hv_c3_1','harvard_c5/hv_c5_1','harvard_c6/hv_c6_1')
PARTITIONS=('g8','g16','mixed')
METHODS=('full1','quad4','quad16','full4','iso4')
WIDTHS=(.25,.5,1.,2.)
FLOOR_DENSITY=1e-30
TINY_FOV=1e-20


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False))


def sources():
    paths=[Path(__file__),ROOT/'models/round10_cell_integral_fast.py',ROOT/'models/round10_cell_integral.py']
    return {str(path.resolve()):sha(path) for path in paths}


def cohort(data):
    completion=read(data/'completion.json')
    if completion['fresh_test_read']:raise ValueError('unexpected fresh data')
    chosen=[]
    for scene in SCENES:
        members=[item for item in completion['manifests'] if item['scene']==scene and item['split']=='train']
        if len(members)!=16:raise ValueError('missing old calibration scan')
        item=min(members,key=lambda item:Path(item['path']).parent.name)
        path=Path(item['path'])
        if sha(path)!=item['sha256']:raise ValueError('changed selected manifest')
        meta=read(path)
        if meta['scene']!=scene or meta['split']!='train' or meta['forward_uses_sensor']:
            raise ValueError('invalid calibration forward metadata')
        input_path=path.parent/'inputs.npz'
        if sha(input_path)!=meta['input_sha256']:raise ValueError('changed predicted observations')
        chosen.append(dict(scene=scene,pair=meta['pair'],manifest=str(path),manifest_sha256=item['sha256'],
                           inputs=str(input_path),input_sha256=meta['input_sha256']))
    return chosen


def moment(points,within):
    mean=points.mean(0)
    centered=points-mean
    covariance=centered.T@centered/len(points)+within.mean(0)
    return mean,covariance


def component_bank(xy,pixel_cov,bounds,method):
    """Fit fixed spatial strata; every method sees exactly the same pixels."""
    means,covariances,weights,owners=[],[],[],[]
    for query,(x0,y0,x1,y1) in enumerate(bounds):
        count=(x1-x0)*(y1-y0)
        if method=='dense':
            points=xy[y0:y1,x0:x1].reshape(-1,2)
            within=pixel_cov[y0:y1,x0:x1].reshape(-1,2,2)
            means.extend(points.unbind(0));covariances.extend(within.unbind(0))
            weights.extend([1/count]*count);owners.extend([query]*count)
            continue
        side=1 if method=='full1' else 4 if method=='quad16' else 2
        if method not in (*METHODS,):raise ValueError('unknown diagnostic method')
        if (x1-x0)%side or (y1-y0)%side:raise ValueError('strata must have equal pixel area')
        sx,sy=(x1-x0)//side,(y1-y0)//side
        for by in range(side):
            for bx in range(side):
                points=xy[y0+by*sy:y0+(by+1)*sy,x0+bx*sx:x0+(bx+1)*sx].reshape(-1,2)
                within=pixel_cov[y0+by*sy:y0+(by+1)*sy,x0+bx*sx:x0+(bx+1)*sx].reshape(-1,2,2)
                if method in ('quad4','quad16'):
                    mean,covariance=points.mean(0),within.mean(0)
                else:
                    mean,covariance=moment(points,within)
                    if method=='iso4':
                        covariance=torch.eye(2,device=xy.device,dtype=xy.dtype)*covariance.diagonal().mean()
                means.append(mean);covariances.append(covariance);weights.append(len(points)/count);owners.append(query)
    return dict(mean=torch.stack(means),cov=torch.stack(covariances),
        weights=xy.new_tensor(weights),owner=torch.tensor(owners,device=xy.device,dtype=torch.long),
        queries=len(bounds))


def mixture_moments(bank):
    means=bank['mean'].new_zeros((bank['queries'],2))
    means.index_add_(0,bank['owner'],bank['mean']*bank['weights'][:,None])
    delta=bank['mean']-means[bank['owner']]
    second=bank['cov']+delta[:,:,None]*delta[:,None,:]
    cov=bank['mean'].new_zeros((bank['queries'],2,2))
    cov.index_add_(0,bank['owner'],second*bank['weights'][:,None,None])
    return means,cov


def integrate(bank,key_bounds,width):
    cells=gaussian_cell_mass_fast(bank['mean'],bank['cov'],key_bounds,bandwidth=width,
        quadrature_order=32,query_chunk_size=256,key_chunk_size=432)
    mass=cells.new_zeros((bank['queries'],len(key_bounds)))
    mass.index_add_(0,bank['owner'],cells*bank['weights'][:,None])
    raw_fov=mass.sum(-1)
    area=(key_bounds[:,2:]-key_bounds[:,:2]).prod(-1)
    corrected=mass+FLOOR_DENSITY*area[None]
    probability=corrected/corrected.sum(-1,keepdim=True)
    return probability,raw_fov


def discrepancy(reference,candidate):
    p=np.asarray(reference,np.float64);q=np.asarray(candidate,np.float64)
    if not np.isfinite(p).all() or not np.isfinite(q).all() or np.any(p<0) or np.any(q<0):
        raise ValueError('invalid conditional probability')
    if max(np.abs(p.sum(-1)-1).max(),np.abs(q.sum(-1)-1).max())>1e-10:
        raise ValueError('conditional probability mass mismatch')
    logp=np.log(np.maximum(p,1e-300));logq=np.log(np.maximum(q,1e-300));mid=(p+q)/2
    kl=np.sum(p*(logp-logq),axis=-1)
    js=.5*np.sum(p*(logp-np.log(np.maximum(mid,1e-300)))+q*(logq-np.log(np.maximum(mid,1e-300))),axis=-1)
    l1=np.abs(p-q).sum(-1)
    return kl,js,l1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'results/round10_partition_data')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();torch.set_num_threads(4)
    if args.out.exists():raise ValueError('refuse overwrite')
    selected=cohort(args.data)
    with np.load(args.data/'partitions.npz',allow_pickle=False) as archive:
        bounds={name:archive[name+'_bounds'].copy() for name in PARTITIONS}
    indices={name:np.linspace(0,len(bounds[name])-1,16).astype(np.int64).tolist() for name in PARTITIONS}
    if any(len(set(value))!=16 for value in indices.values()):raise ValueError('duplicate chosen query')
    frozen=sources();args.out.mkdir(parents=True)
    protocol=dict(scope='forward-only compression diagnostic; no research effectiveness gate',
        source_hashes=frozen,data_completion_sha256=sha(args.data/'completion.json'),
        partition_sha256=sha(args.data/'partitions.npz'),cohort=selected,query_indices=indices,
        query_partition_names=PARTITIONS,key_partition='g8',key_count=432,widths=WIDTHS,temperature=0,
        reference='equal mixture of every actual RGB-predicted projected pixel Gaussian, with its own projected within-pixel covariance',
        methods=dict(full1='one full moment Gaussian per selected region',
            quad4='2x2 stratum means and average within-pixel covariance, no within-stratum scatter',
            quad16='4x4 stratum means and average within-pixel covariance, no within-stratum scatter',
            full4='2x2 stratum Gaussian FULL moments including within-stratum scatter',
            iso4='same as full4 with each component covariance replaced by its half-trace identity'),
        information='same complete predicted pixel field for all methods; no sensor visibility or labels',
        integration='same bounded-domain float64 rectangle integral, order32; source mixture before FOV normalization',
        query_semantics='conditional region-density mixture; not exact source-area averaging of separately normalized atom queries',
        floor_density=FLOOR_DENSITY,tiny_fov_threshold=TINY_FOV,
        floor_policy='add same positive background density times actual key area after mixture; retain all rows, report zero/tiny FOV',
        metrics='per-query KL(reference||method), JS and L1; row means then equal scene/partition/width means; no sensor truth',
        timing='one warmup and three synchronized repeats; prepare and integrate separately, sum medians; includes all selected queries and full target bank; excludes shared RGB prediction/data transfer',
        interpretation='measures representation fidelity to predicted pixel mixture only; no correctness, generalization, or downstream success claim',
        no_labels_opened=True,no_validation_inputs_opened=True,
        device=args.device,torch=torch.__version__,numpy=np.__version__,created_unix=time.time())
    if args.device.startswith('cuda'):
        protocol['device_name']=torch.cuda.get_device_name(torch.device(args.device))
    write(args.out/'protocol.json',protocol)
    (args.out/'protocol.sha256').write_text(sha(args.out/'protocol.json')+'\n')
    snapshots=args.out/'source_snapshots';snapshots.mkdir()
    for source,digest in frozen.items():
        destination=snapshots/Path(source).name
        destination.write_bytes(Path(source).read_bytes())
        if sha(destination)!=digest:raise ValueError('source snapshot mismatch')
    print('FROZEN_PROTOCOL',sha(args.out/'protocol.json'),flush=True)
    started=time.time();rows=[];timing=[];moment_checks=[];distribution_files=[]
    def sync():
        if args.device.startswith('cuda'):torch.cuda.synchronize(torch.device(args.device))
    key_bounds=torch.as_tensor((bounds['g8'].astype(np.float64)-.5)/8,device=args.device,dtype=torch.float64)
    with torch.no_grad():
        for item in selected:
            # The only NPZ values read are these two permitted predicted fields.
            if sha(item['inputs'])!=item['input_sha256']:raise ValueError('changed forward input before read')
            with np.load(item['inputs'],allow_pickle=False) as archive:
                xy=torch.as_tensor(archive['q_to_k_xy'].reshape(144,192,2)/8,device=args.device,dtype=torch.float64)
                within=torch.as_tensor(archive['q_to_k_pixel_cov'].reshape(144,192,2,2)/64,device=args.device,dtype=torch.float64)
            for name in PARTITIONS:
                query_bounds=bounds[name][indices[name]]
                banks={};prepare_times={}
                for method in ('dense',*METHODS):
                    measured=[]
                    for repeat in range(4):
                        sync();tick=time.perf_counter();bank=component_bank(xy,within,query_bounds,method);sync()
                        if repeat:measured.append((time.perf_counter()-tick)*1000)
                    banks[method]=bank;prepare_times[method]=measured
                dense_mean,dense_cov=mixture_moments(banks['dense'])
                for method in METHODS:
                    mean,covariance=mixture_moments(banks[method])
                    mean_error=float((mean-dense_mean).abs().max())
                    covariance_error=float((covariance-dense_cov).abs().max())
                    trace_error=float((covariance.diagonal(dim1=-2,dim2=-1).sum(-1)-dense_cov.diagonal(dim1=-2,dim2=-1).sum(-1)).abs().max())
                    if mean_error>1e-10 or (method in ('full1','full4') and covariance_error>1e-9) or (method=='iso4' and trace_error>1e-9):
                        raise ValueError('Gaussian component moment conservation failed')
                    moment_checks.append(dict(scene=item['scene'],partition=name,method=method,
                        mean_max_error=mean_error,covariance_max_difference=covariance_error,trace_max_difference=trace_error))
                for width in WIDTHS:
                    saved={};probabilities={};fov={}
                    for method,bank in banks.items():
                        measured=[]
                        for repeat in range(4):
                            sync();tick=time.perf_counter();p,raw=integrate(bank,key_bounds,width);sync()
                            if repeat:measured.append((time.perf_counter()-tick)*1000)
                        probability=p.cpu().numpy();raw_fov=raw.cpu().numpy()
                        probabilities[method]=probability;fov[method]=raw_fov
                        saved[method+'_probability']=probability;saved[method+'_raw_fov']=raw_fov
                        sizes=np.bincount(bank['owner'].cpu().numpy(),minlength=16)
                        timing.append(dict(scene=item['scene'],partition=name,h=width,method=method,
                            prepare_ms=float(np.median(prepare_times[method])),integration_ms=float(np.median(measured)),
                            total_ms=float(np.median(prepare_times[method])+np.median(measured)),
                            prepare_repeats_ms=prepare_times[method],integration_repeats_ms=measured,
                            queries=16,components_total=int(sizes.sum()),components_per_query=sizes.tolist(),
                            stored_component_tensor_bytes=sum(value.numel()*value.element_size() for value in bank.values() if torch.is_tensor(value)),
                            raw_zero_fov_rows=int((raw_fov==0).sum()),raw_tiny_fov_rows=int((raw_fov<TINY_FOV).sum())))
                    for method in METHODS:
                        kl,js,l1=discrepancy(probabilities['dense'],probabilities[method])
                        for position,query_index in enumerate(indices[name]):
                            x0,y0,x1,y1=query_bounds[position]
                            rows.append(dict(scene=item['scene'],pair=item['pair'],partition=name,query_index=query_index,
                                native_width=int(x1-x0),native_height=int(y1-y0),native_area=int((x1-x0)*(y1-y0)),h=width,method=method,
                                kl=float(kl[position]),js=float(js[position]),l1=float(l1[position]),
                                reference_raw_fov=float(fov['dense'][position]),method_raw_fov=float(fov[method][position])))
                    saved['query_indices']=np.asarray(indices[name],np.int64);saved['query_bounds']=query_bounds
                    directory=args.out/'predicted_distributions'/item['scene'].replace('/','__');directory.mkdir(parents=True,exist_ok=True)
                    destination=directory/(name+'_h'+str(width)+'.npz')
                    np.savez_compressed(destination,**saved)
                    distribution_files.append(dict(path=str(destination),sha256=sha(destination)))
                print('COMPRESSED',item['scene'],name,flush=True)
    by_condition=[]
    for method in METHODS:
        for partition in PARTITIONS:
            for width in WIDTHS:
                subset=[row for row in rows if row['method']==method and row['partition']==partition and row['h']==width]
                by_condition.append(dict(method=method,partition=partition,h=width,queries=len(subset),
                    **{metric:float(np.mean([row[metric] for row in subset])) for metric in ('kl','js','l1')}))
    by_size=[]
    for method in METHODS:
        for area in sorted({row['native_area'] for row in rows}):
            per_scene=[]
            for scene in SCENES:
                subset=[row for row in rows if row['method']==method and row['native_area']==area and row['scene']==scene]
                per_scene.append({metric:float(np.mean([row[metric] for row in subset])) for metric in ('kl','js','l1')})
            by_size.append(dict(method=method,native_area=area,**{metric:float(np.mean([s[metric] for s in per_scene])) for metric in ('kl','js','l1')}))
    method_summary={method:{metric:float(np.mean([row[metric] for row in by_condition if row['method']==method]))
                           for metric in ('kl','js','l1')} for method in METHODS}
    summary=dict(methods=method_summary,by_condition=by_condition,by_native_area=by_size,
        per_query_rows=len(rows),reference_distributions=36*16,probability_archives=len(distribution_files),
        diagnosis_only=True,research_effectiveness_passed=False,sensor_labels_read=False,
        interpretation=protocol['interpretation'],seconds=time.time()-started)
    if sources()!=frozen:raise ValueError('source changed during diagnostic')
    if any(sha(item['inputs'])!=item['input_sha256'] for item in selected):raise ValueError('input changed during diagnostic')
    write(args.out/'rows.json',rows);write(args.out/'timing.json',timing);write(args.out/'moment_checks.json',moment_checks)
    write(args.out/'summary.json',summary)
    completion=dict(protocol_sha256=sha(args.out/'protocol.json'),source_hashes=frozen,distribution_files=distribution_files,
        files={name:sha(args.out/name) for name in ('rows.json','timing.json','moment_checks.json','summary.json')},
        seconds=time.time()-started,no_labels_opened=True,no_validation_inputs_opened=True)
    write(args.out/'completion.json',completion)
    print('COMPRESSION_DIAGNOSTIC',method_summary,flush=True)


if __name__=='__main__':main()
