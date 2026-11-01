"""Forward-only numerical audit on the first metadata-ordered training pair."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import _mvn
import torch

from experiments.round10_partition_data import PARTITIONS, partition_bounds, partition_moments, projected_3d_moments
from models.round10_cell_integral import gaussian_cell_mass
from models.round10_cell_integral_fast import gaussian_cell_mass_fast


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('results/round10_partition_data'))
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise ValueError('refusing overwrite')
    torch.set_num_threads(4)
    path=sorted((args.data/'train').glob('*/*/inputs.npz'))[0]
    if path.parts[-3]!='harvard_c3__hv_c3_1' or not path.parts[-2].startswith('000_'):
        raise ValueError('unexpected metadata-first training pair')
    with np.load(path) as archive:
        required=('q_to_k_xy','q_to_k_pixel_cov','q_points3','q_pixel_cov3','q_to_k_transform','k_K')
        inputs={key:archive[key].copy() for key in required}
    rows=[];scipy_rows=[]
    for representation in ('projected3d','projected_pixel_moments'):
        for qname in PARTITIONS:
            bounds=partition_bounds(qname)
            if representation=='projected3d':
                mean,cov,_=projected_3d_moments(inputs,bounds)
            else:
                mean,cov=partition_moments(inputs['q_to_k_xy'],inputs['q_to_k_pixel_cov'],bounds)
            mean,cov=mean/8,cov/64
            qm,qc=(torch.from_numpy(a) for a in (mean,cov))
            for kname in PARTITIONS:
                kb=(partition_bounds(kname).astype(np.float64)-.5)/8
                boxes=torch.from_numpy(kb)
                kwargs=dict(bandwidth=.25)
                times={};values={}
                for name,fn in [('robust24',lambda:gaussian_cell_mass(qm,qc,boxes,**kwargs)),
                                ('fast16',lambda:gaussian_cell_mass_fast(qm,qc,boxes,quadrature_order=16,**kwargs)),
                                ('fast32',lambda:gaussian_cell_mass_fast(qm,qc,boxes,quadrature_order=32,**kwargs))]:
                    started=time.perf_counter();values[name]=fn().numpy();times[name]=time.perf_counter()-started
                ref=values['robust24']
                row=dict(representation=representation,query=qname,key=kname,query_count=len(mean),key_count=len(kb),
                    max_cell_side=float((kb[:,2:]-kb[:,:2]).max()),bandwidth=.25,
                    covariance_max_eigenvalue=float(np.linalg.eigvalsh(cov).max()),seconds=times,
                    reference_min_FOV_mass=float(ref.sum(-1).min()),reference_zero_FOV_rows=int((ref.sum(-1)==0).sum()))
                for name in ('fast16','fast32'):
                    estimate=values[name]
                    row[name+'_max_abs_mass_error']=float(np.abs(estimate-ref).max())
                    valid=(ref.sum(-1)>1e-30)&(estimate.sum(-1)>1e-30)
                    row[name+'_max_probability_error_above_FOV_1e30']=float(np.abs(estimate[valid]/estimate[valid].sum(-1,keepdims=True)-ref[valid]/ref[valid].sum(-1,keepdims=True)).max())
                rows.append(row)
                # Fixed query ordinals plus trace extrema, and the three closest
                # cell centers. Selection uses only forward means/covariances.
                trace=np.trace(cov,axis1=-2,axis2=-1)
                queries=sorted(set(np.linspace(0,len(mean)-1,5,dtype=int).tolist()+[int(trace.argmin()),int(trace.argmax())]))
                centers=(kb[:,:2]+kb[:,2:])/2
                for qi in queries:
                    keys=np.argsort(np.square(centers-mean[qi]).sum(-1),kind='stable')[:3]
                    for ki in keys:
                        value,status=_mvn.mvnun(kb[ki,:2],kb[ki,2:],mean[qi],cov[qi]+.25**2*np.eye(2),100000,1e-13)
                        if status:raise ValueError('SciPy reference failed')
                        scipy_rows.append(dict(representation=representation,query=qname,key=kname,qi=int(qi),ki=int(ki),
                            scipy=float(value),fast32=float(values['fast32'][qi,ki]),abs_error=abs(float(values['fast32'][qi,ki])-value)))
                print(representation,qname,kname,'mass',row['fast32_max_abs_mass_error'],
                      'prob',row['fast32_max_probability_error_above_FOV_1e30'],flush=True)
    root=Path(__file__).resolve().parents[1]
    source_paths=[Path(__file__).resolve(),root/'models/round10_cell_integral.py',root/'models/round10_cell_integral_fast.py',root/'experiments/round10_partition_data.py']
    summary=dict(forward_path=str(path.resolve()),forward_sha256=sha(path),
        field_names_read=list(required),labels_read=False,sensor_images_read=False,test_inputs_read=False,
        selection='lexicographically first allowlisted training pair; all declared query/key partitions',
        bandwidth=.25,bandwidth_reason='smallest allowed bandwidth gives largest cell/conditional-sigma ratio',
        source_hashes={str(p):sha(p) for p in source_paths},rows=rows,scipy_rows=scipy_rows,
        checks=dict(fast32_max_abs_mass_error=max(r['fast32_max_abs_mass_error'] for r in rows),
            fast32_max_probability_error_above_FOV_1e30=max(r['fast32_max_probability_error_above_FOV_1e30'] for r in rows),
            fast16_max_abs_mass_error=max(r['fast16_max_abs_mass_error'] for r in rows),
            scipy_max_abs_error=max(r['abs_error'] for r in scipy_rows),
            query_key_mass_count=sum(r['query_count']*r['key_count'] for r in rows),scipy_comparison_count=len(scipy_rows)),
        timing_scope='one CPU call per configuration, four threads, not a controlled throughput comparison')
    summary['passed']=summary['checks']['fast32_max_abs_mass_error']<1e-10 and summary['checks']['fast32_max_probability_error_above_FOV_1e30']<1e-8 and summary['checks']['scipy_max_abs_error']<1e-10
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary['checks']),flush=True)
    if not summary['passed']:raise ValueError('numerical audit failed')


if __name__=='__main__':main()
