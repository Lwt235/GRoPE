"""Unadapted references and resident-source full two-pass query timing."""
import argparse
import json
import time
import numpy as np
import torch
from experiments.gaussian_memory_coarse_data import *
from experiments.gaussian_memory_coarse_cache import to_device
from experiments.gaussian_memory_native_run import model_load,state_new,step,metric_depth,scores
from models.gaussian_memory_coarse_pe import CoarseReader,forward_query,patch_tokens,query_geometry,descriptors


@torch.inference_mode()
def references(model,p):
    rows=[];pred=[];keys=[]
    for meta in p['catalog']:
        rid=meta['id']
        for c in p['conditions']:
            inp=load(input_path(rid,11,c))
            d,_=step(model,inp['images'][-1],11,state_new(model),True)
            rows.append(dict(id=rid,place=meta['place'],condition=c,capacity=0,kind='single_view',seed=None,**scores(d,inp)))
            pred.append(metric_depth(d));keys.append([rid,c,0,'single_view'])
            for n in CAPACITIES:
                s=load(sample_path(rid,11,c,n));d=s['first_pass_depth']
                rows.append(dict(id=rid,place=meta['place'],condition=c,capacity=n,kind='coarse_frozen',seed=None,**scores(d,inp)))
                pred.append(metric_depth(d));keys.append([rid,c,n,'coarse_frozen'])
    path=CACHE/'evaluation/references.pt';path.parent.mkdir(parents=True,exist_ok=True);save_tensor(path,dict(keys=keys,depth_metric=torch.stack(pred)))
    write_json(OUT/'evaluation/references.json',dict(complete=True,rows=rows,prediction_path=str(path),prediction_sha256=sha256(path),plan_sha256=sha256(OUT/'plan.json'),single_view='Unadapted official model, current RGB only; not a matched trained no-memory ablation'))


@torch.inference_mode()
def timing(model,p):
    seed=SEEDS[0];records=[];replay_max=0.
    for kind in KINDS:
        adapter=CoarseReader().cuda().eval();checkpoint=OUT/f'{kind}_{seed}'/'step_2000.pt';adapter.load_state_dict(load(checkpoint)['model'])
        for rid in p['training_ids'][:2]:
            for c in p['conditions']:
                inp=load(input_path(rid,11,c))
                for n in CAPACITIES:
                    cpu=load(sample_path(rid,11,c,n));s=to_device(cpu);camera=[tuple(t.cuda() for t in pair) for pair in cpu['camera_state']]
                    # Drop offline audit tensors from the resident query view.
                    s={k:v for k,v in s.items() if k not in ('source_patch_means','source_patch_covariances','membership','fps_indices','camera_state')}
                    times=[];parts=[]
                    for repeat in range(4):
                        torch.cuda.synchronize();begin=time.perf_counter()
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            image=inp['images'][-1].cuda();s['patch_tokens']=patch_tokens(model,image)
                            d,out=forward_query(model,s,checkpoint_blocks=False,return_tokens=True)
                        with torch.autocast('cuda',enabled=False):poses,_=model.camera_head(out,past_key_values_camera=list(camera),use_cache=True)
                        mu,cov=query_geometry(d,poses[-1][0,0],s['native_hw'],s['geometry_unit'],s['geometry_origin'])
                        replay_max=max(replay_max,float((d.cpu()-cpu['first_pass_depth']).abs().max()))
                        s['query_means']=mu;s['query_covariances']=cov;z=descriptors(s,kind)
                        torch.cuda.synchronize();mid=time.perf_counter()
                        with torch.autocast('cuda',dtype=torch.bfloat16):result=forward_query(model,s,adapter,kind,z,False)
                        torch.cuda.synchronize();end=time.perf_counter()
                        if repeat:times.append(end-begin);parts.append([mid-begin,end-mid])
                        del d,out,poses,mu,cov,z,result
                    records.append(dict(kind=kind,id=rid,condition=c,capacity=n,seconds=times,median_seconds=float(np.median(times)),first_pass_median_seconds=float(np.median(parts,axis=0)[0]),second_pass_median_seconds=float(np.median(parts,axis=0)[1])))
                    del cpu,s,camera;torch.cuda.empty_cache()
        del adapter;torch.cuda.empty_cache()
    write_json(OUT/'runtime.json',dict(complete=True,gpu=torch.cuda.get_device_name(),records=records,median_seconds={str(n):{k:float(np.median([v for r in records if r['kind']==k and r['capacity']==n for v in r['seconds']])) for k in KINDS} for n in CAPACITIES},first_pass_cache_replay_max_abs=replay_max,scope='Current image transfer/backbone, frozen coarse depth+camera+query geometry/descriptor, adapted second query/depth. Sources resident; model/disk loading excluded. Not bounded-memory construction or all-task VGGT timing.',plan_sha256=sha256(OUT/'plan.json'),checkpoint_sha256={k:sha256(OUT/f'{k}_{seed}'/'step_2000.pt') for k in KINDS}))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['references','timing']);a=parser.parse_args();torch.set_num_threads(4);p=plan();verify(p);model=model_load(p)
    references(model,p) if a.action=='references' else timing(model,p)


if __name__=='__main__':main()
