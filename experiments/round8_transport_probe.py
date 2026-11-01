"""Nonlearned old-validation A@depth checks, separated from trainable evidence."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from models.gaussian_quadratic_pe import GaussianQuadraticMatcher
from models.round8_oracle_attention import replace_routing
from experiments.round8_oracle_train import load_development,sources_now
from experiments.round7_depth_data import ROOT,sha
from experiments.round7_depth_train import batch,metric_rows,macro


@torch.no_grad()
def run(out,lockpath,device='cpu'):
    if out.exists():raise ValueError('refuse overwrite transport probe')
    lock=json.loads(lockpath.read_text())
    if sources_now()!=lock['source_hashes'] or lock['test_access_authorized']:raise ValueError('diagnostic source lock mismatch')
    data,ids,sources,oracle_sources=load_development('val',device)
    if sources!=lock['data_sources']['val'] or oracle_sources!=lock['data_sources']['oracle_val']:
        raise ValueError('changed development data')
    model=GaussianQuadraticMatcher('full',.5,5.).to(device)
    results={r:[] for r in ('pred_pred','gt_route','gt_value','gt_both')};start=time.time()
    for i in range(0,len(ids),16):
        d=batch(data,slice(i,i+16));pred=model(d).softmax(-1)
        route=replace_routing(pred,d['oracle_indices'],d['oracle_weights'],d['oracle_valid'])
        for role,attn,z in [('pred_pred',pred,d['k_value_depth']),('gt_route',route,d['k_value_depth']),
                ('gt_value',pred,d['k_oracle_value_depth']),('gt_both',route,d['k_oracle_value_depth'])]:
            results[role].append((attn@z[...,None]).squeeze(-1))
    predictions={role:torch.cat(values) for role,values in results.items()}
    predictions['query_raw']=data['q_depth']
    predictions['direct_source_prediction']=data['direct_corresponding_pred_depth']
    predictions['direct_source_sensor']=data['direct_corresponding_sensor_depth']
    masks={'all_query_valid':data['valid'],
           'oracle_eligible':data['valid']&data['oracle_valid'],
           'predicted_fallback':data['valid']&~data['oracle_valid']}
    summary={};out.mkdir(parents=True)
    np.savez_compressed(out/'predictions.npz',**{r:p.cpu().numpy() for r,p in predictions.items()})
    for role,pred in predictions.items():
        summary[role]={}
        for subset,valid in masks.items():
            if role.startswith('direct_') and subset!='oracle_eligible':continue
            d=dict(data,valid=valid);rows=metric_rows(pred,d,ids);s=macro(rows)
            summary[role][subset]=s
            (out/f'{role}_{subset}.json').write_text(json.dumps(dict(rows=rows,summary=s),indent=2))
    result=dict(scope='fixed untrained visual/full-PE versus sensor-route direct A@z; old val only',
        diagnostic_only=True,learned_seed_replicates=0,pair_count=len(ids),summary=summary,
        lock_sha256=sha(lockpath),source_hashes=lock['source_hashes'],
        source_paths=dict(val=sources,oracle_val=oracle_sources),prediction_sha256=sha(out/'predictions.npz'),
        eligible=int(masks['oracle_eligible'].sum()),fallback=int(masks['predicted_fallback'].sum()),
        seconds=time.time()-start,
        caution='Direct source sensor uses GT-consistency-selected correspondence; geometry/quantization sanity only, not independent evidence or a mathematical upper bound.')
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    for r,s in summary.items():print(r,{k:v['metrics']['AbsRel'] for k,v in s.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'results/round8_transport_probe')
    p.add_argument('--lock',type=Path,default=ROOT/'results/round8_oracle_development/lock.json')
    p.add_argument('--device',default='cpu');a=p.parse_args();torch.set_num_threads(1)
    run(a.out,a.lock,a.device)
