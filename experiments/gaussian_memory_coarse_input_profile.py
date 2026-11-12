"""Describe support sizes/anisotropy of fixed inputs, without method scores."""
import numpy as np
import torch
from experiments.gaussian_memory_coarse_data import *


def main():
    p=plan();rows=[]
    for split in ('train','development'):
        for key in example_keys(p,split):
            s=torch.load(sample_path(*key),map_location='cpu',mmap=True,weights_only=False)
            covariance=s['source_covariances'][~s['source_special']].numpy().astype(np.float64)
            eigen=np.linalg.eigvalsh(covariance);radius=np.sqrt(eigen.sum(-1));counts=s['member_counts'].numpy()
            rows.append(dict(key=list(key),split=split,place=s['place'],capacity=key[3],condition=key[2],radius_q10_q50_q90=np.percentile(radius,[10,50,90]).tolist(),radius_q90_over_q10=float(np.percentile(radius,90)/np.percentile(radius,10)),median_axis_variance_ratio=float(np.median(eigen[:,-1]/eigen[:,0])),member_count_q10_q50_q90=np.percentile(counts,[10,50,90]).tolist()))
    summaries={}
    for split in ('train','development'):
        for n in CAPACITIES:
            for condition in p['conditions']:
                group=[r for r in rows if r['split']==split and r['capacity']==n and r['condition']==condition]
                summaries[f'{split}_{n}_{condition}']=dict(examples=len(group),median_within_example_radius_q90_over_q10=float(np.median([r['radius_q90_over_q10'] for r in group])),median_axis_variance_ratio=float(np.median([r['median_axis_variance_ratio'] for r in group])))
    write_json(OUT/'input_profile.json',dict(complete=True,rows=rows,summaries=summaries,scope='Descriptive fixed input regime only; no method score, stratification selection, training or decision threshold change',all_data_previously_used=True,new_confirmation=False,plan_sha256=sha256(OUT/'plan.json'),source_sha256=sha256(ROOT/'experiments/gaussian_memory_coarse_input_profile.py')))
    print({k:v for k,v in summaries.items() if k.startswith('development')},flush=True)


if __name__=='__main__':main()
