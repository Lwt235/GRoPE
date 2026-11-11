"""Separate within-patch and between-patch spread in the frozen memberships."""
import json

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_setup import sha256
from experiments.gaussian_memory_oracle_data import OUT, CACHE, checked


def main():
    plan=checked();torch.set_num_threads(4);rows=[]
    for meta in plan['catalog']:
        direct=torch.load(meta['direct'],map_location='cpu',weights_only=False)['memory']
        sensor=torch.load(CACHE/(meta['id']+'.pt'),map_location='cpu',weights_only=False)
        binding=torch.load(OUT/'bindings'/(meta['id']+'.pt'),map_location='cpu',weights_only=False);near,weights=binding['near'],binding['weights'].double()
        raw=torch.load(meta['raw'],map_location='cpu',weights_only=False);unit=raw['diagnostics']['unit_meters'];row={'episode':meta['id'],'scene':meta['row']['scene'],'split':meta['eval_split']}
        for regime in ('predicted','sensor'):
            means=direct['means'].double();cov=direct['covars'].double()
            if regime=='sensor':
                means=torch.where(sensor['replace'][:,None],sensor['patch_sensor_mean'].double(),means)
                cov=torch.where(sensor['replace'][:,None,None],sensor['patch_sensor_cov'].double(),cov)
            center=(weights[...,None]*means[near]).sum(1);delta=means[near]-center[:,None]
            within=(weights*cov[near].diagonal(dim1=-2,dim2=-1).sum(-1)).sum(-1)*unit**2
            between=(weights*delta.square().sum(-1)).sum(-1)*unit**2
            row[regime]={'mean_within_patch_trace_m2':float(within.mean()),'mean_between_patch_trace_m2':float(between.mean()),
                         'between_fraction_of_total_trace':float(between.sum()/(within+between).sum())}
        rows.append(row)
    selected=[r for r in rows if r['split'].startswith('used_clip')]
    summary={g:{k:float(np.mean([r[g][k] for r in selected])) for k in selected[0][g]} for g in ('predicted','sensor')}
    write_json(OUT/'geometry_diagnostic.json',{'rows':rows,'primary_summary':summary,'purpose':'Check the predeclared fixed-membership limitation, without query labels or outcome selection.',
               'post_training_model_modified':False,'source_sha256':sha256(__file__)})
    print(json.dumps(summary))


if __name__=='__main__':main()
