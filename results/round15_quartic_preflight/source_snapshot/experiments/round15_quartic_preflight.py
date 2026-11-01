"""Frozen synthetic-only numerical-source binding and operator cost preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from models.round15_quartic_pe import quartic_cell_mass as quartic48
from models.round15_quartic_tiled import quartic_cell_mass
from models.round10_cell_integral_fast import gaussian_cell_mass_fast

ROOT=Path(__file__).resolve().parents[1]
SOURCES=('ROUND15_NUMERICAL_PROTOCOL.md','models/round15_quartic_pe.py',
    'models/round15_quartic_tiled.py','tests/test_round15_quartic_tiled.py',
    'tests/test_round15_quartic_pe.py','experiments/round15_quartic_preflight.py',
    'tests/test_round15_quartic_preflight.py','models/round10_cell_integral_fast.py',
    'models/round10_cell_integral.py')
ROLES=('quartic','full1','full4','quad16','quartic48_reference')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def synthetic_geometry(stride):
    if stride not in (8,16,24):raise ValueError('fixed synthetic scales only')
    A=np.array([[1.05,.10],[-.05,.95]],np.float64);translation=np.array([.10,-.15])
    bounds=np.array([[x,y,x+stride,y+stride] for y in range(0,144,stride)
                     for x in range(0,192,stride)],np.float64)
    means=(bounds[:,:2]+bounds[:,2:]-1)/16
    basecov=(stride**2/(12*64))*(A@A.T)
    fullcov=np.broadcast_to(basecov,(len(means),2,2)).copy()
    key=np.array([[x-.5,y-.5,x+7.5,y+7.5] for y in range(0,144,8)
                  for x in range(0,192,8)],np.float64)/8
    roles={}
    for role,n in (('quartic',1),('full1',1),('full4',2),('quad16',4),('quartic48_reference',1)):
        step=stride/n
        offsets=np.array([[(x+.5)*step-stride/2,(y+.5)*step-stride/2]
                          for y in range(n) for x in range(n)])/8
        nodes=(means[:,None]+offsets[None])@A.T+translation
        variance=(stride**2 if n==1 else step**2 if role=='full4' else 1.)/(12*64)
        cov=np.broadcast_to(variance*(A@A.T),(len(means),n*n,2,2)).copy()
        roles[role]=dict(means=nodes,cov=cov,weights=np.full((len(means),n*n),1/(n*n)))
    return dict(stride=stride,key_bounds=key,roles=roles)


def freeze(out,report):
    if out.exists():raise ValueError('refuse freeze overwrite')
    verification=read(report/'verification.json');completion=read(report/'completion.json')
    if not verification['passed'] or verification['quadrature_order'] not in (16,24):
        raise ValueError('passed declared numerical order required')
    if verification['maximum_tile_side']!=1. or verification['internal_floor_used']:
        raise ValueError('expected numerical-only native-key tiling report')
    if verification['real_observations_read'] or verification['labels_read']:
        raise ValueError('numeric report must be synthetic only')
    for name,digest in verification['source_hashes'].items():
        if sha(ROOT/name)!=digest:raise ValueError('numeric source changed')
    for name,digest in completion['files'].items():
        if sha(report/name)!=digest:raise ValueError('numeric report incomplete')
    out.mkdir(parents=True);snap=out/'source_snapshot';snap.mkdir()
    hashes={name:sha(ROOT/name) for name in SOURCES}
    for name in SOURCES:
        dest=snap/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
    data={}
    for stride in (8,16,24):
        g=synthetic_geometry(stride);arrays={'key_bounds':g['key_bounds']}
        for role,v in g['roles'].items():
            for name,a in v.items():arrays[role+'_'+name]=a
        path=out/f'synthetic_q{stride}.npz';np.savez_compressed(path,**arrays)
        data[str(stride)]=dict(file=path.name,sha256=sha(path),arrays={k:array_sha(v) for k,v in arrays.items()})
    lock=dict(created_unix=time.time(),source_hashes=hashes,synthetic=data,order=verification['quadrature_order'],
        numerical_report=str(report.resolve()),numerical_report_sha256=sha(report/'verification.json'),
        numerical_completion_sha256=sha(report/'completion.json'),labels_read=False,real_inputs_read=False)
    write(out/'lock.json',lock);(out/'lock.sha256').write_text(sha(out/'lock.json')+'\n')
    print('FROZEN',sha(out/'lock.json'),flush=True)


def verify(out):
    lock=read(out/'lock.json')
    if sha(out/'lock.json')!=(out/'lock.sha256').read_text().strip():raise ValueError('lock hash mismatch')
    for name,digest in lock['source_hashes'].items():
        if sha(ROOT/name)!=digest or sha(out/'source_snapshot'/name)!=digest:raise ValueError('source mutation')
    report=Path(lock['numerical_report'])
    for name,key in (('verification.json','numerical_report_sha256'),('completion.json','numerical_completion_sha256')):
        if sha(report/name)!=lock[key]:raise ValueError('numerical provenance changed')
    for record in lock['synthetic'].values():
        if sha(out/record['file'])!=record['sha256']:raise ValueError('synthetic geometry changed')
    return lock


def run(out,device):
    lock=verify(out);target=out/('cost_'+device.replace(':','_')+'.json')
    started=target.with_suffix('.started.json')
    if target.exists() or started.exists():raise ValueError('refuse performance run overwrite')
    if device!='cpu' and not device.startswith('cuda:'):raise ValueError('explicit CPU or CUDA device required')
    torch.set_num_threads(4);cuda=device.startswith('cuda:')
    if cuda:
        torch.cuda.set_device(device)
        # Availability/other-job check belongs before invoking this CLI. No
        # external job is stopped or displaced by this preflight.
        torch.cuda.synchronize()
    write(started,dict(created_unix=time.time(),device=device,lock_sha256=sha(out/'lock.json')))
    warmups,repeats=(3,15) if cuda else (1,3)
    records=[]
    with torch.no_grad():
        for stride in (8,16,24):
            with np.load(out/lock['synthetic'][str(stride)]['file']) as z:
                arrays={k:z[k] for k in z.files}
            boxes=torch.as_tensor(arrays['key_bounds'],device=device)
            for role in ROLES:
                means=torch.as_tensor(arrays[role+'_means'],device=device)
                cov=torch.as_tensor(arrays[role+'_cov'],device=device)
                weights=torch.as_tensor(arrays[role+'_weights'],device=device)
                def forward():
                    if role=='quartic':
                        mass=quartic_cell_mass(means,cov,boxes,bandwidth=.5,quadrature_order=lock['order'],
                                               query_chunk_size=16,key_chunk_size=32)
                    elif role=='quartic48_reference':
                        mass=quartic48(means,cov,boxes,bandwidth=.5,quadrature_order=48,
                                       query_chunk_size=16,key_chunk_size=32)
                    else:
                        mass=gaussian_cell_mass_fast(means,cov,boxes,bandwidth=.5,quadrature_order=32,
                                                      query_chunk_size=256,key_chunk_size=432)
                    return (mass*weights[...,None]).sum(-2)
                for _ in range(warmups):forward()
                if cuda:torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                times=[];value=None
                for _ in range(repeats):
                    if cuda:torch.cuda.synchronize()
                    start=time.perf_counter();value=forward()
                    if cuda:torch.cuda.synchronize()
                    times.append(time.perf_counter()-start)
                if not bool(torch.isfinite(value).all()) or not bool((value>=0).all()):raise ValueError('invalid synthetic probabilities')
                row=dict(stride=stride,role=role,queries=len(means),keys=len(boxes),nodes=means.shape[-2],
                    raw_seconds=times,median_seconds=float(np.median(times)),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated() if cuda else None,
                    output_sha256=array_sha(value.cpu().numpy()),minimum_fov_mass=float(value.sum(-1).min()),
                    maximum_fov_mass=float(value.sum(-1).max()))
                records.append(row);print('COST',stride,role,row['median_seconds'],flush=True)
    ratios={}
    for stride in (8,16,24):
        rows={r['role']:r for r in records if r['stride']==stride}
        ratios[str(stride)]={r:rows['quartic']['median_seconds']/rows[r]['median_seconds'] for r in ROLES[1:]}
    verify(out)
    write(target,dict(created_unix=time.time(),device=device,torch_version=torch.__version__,
        numpy_version=np.__version__,platform=platform.platform(),processor=platform.processor(),
        cuda_device=torch.cuda.get_device_name() if cuda else None,threads=4,warmups=warmups,repeats=repeats,
        order=lock['order'],lock_sha256=sha(out/'lock.json'),records=records,quartic_time_ratio=ratios,
        labels_read=False,real_inputs_read=False,scientific_gate_passed=False,
        scope='Synthetic positional integration and node reduction only; not old end-to-end task gate timing.'))
    print('COMPLETE',target,sha(target),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('freeze','run'))
    p.add_argument('--out',type=Path,default=ROOT/'results/round15_quartic_preflight')
    p.add_argument('--report',type=Path);p.add_argument('--device',default='cpu')
    a=p.parse_args()
    if a.stage=='freeze':freeze(a.out,a.report)
    else:run(a.out,a.device)


if __name__=='__main__':main()
