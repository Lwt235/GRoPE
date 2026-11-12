"""Measured full source construction and uncached native query costs."""
import argparse
import os
import subprocess
import time

import numpy as np
import torch

from experiments.gaussian_scene_data import OUT,CACHE,source_path,query_path,query_image,read,KINDS
from experiments.gaussian_scene_bind import load_tensor,bind_one
from experiments.gaussian_scene_config import freeze
from experiments.gaussian_memory_native_run import model_load
from experiments.gaussian_memory_incremental_data import save_tensor,write_json
from experiments.gaussian_memory_setup import sha256
from models.gaussian_scene_native import prepare_read,forward_read
from models.gaussian_scene_memory import SceneReader,field_bytes


def live():
    uuid=os.environ.get('CUDA_VISIBLE_DEVICES','0')
    state=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=uuid,memory.used,memory.free,utilization.gpu',
                                  '--format=csv,noheader,nounits'],text=True,timeout=15).strip()
    apps=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-compute-apps=pid,used_gpu_memory',
                                 '--format=csv,noheader,nounits'],text=True,timeout=15).strip()
    return dict(gpu=state,compute_apps=apps,own_pid=os.getpid())


def construction(p):
    marker=OUT/'cost/full_construction.json'
    if marker.exists():return
    from experiments.gaussian_scene_anysplat import model_load as builder_load,initialize
    from experiments.gaussian_scene_build import fit_sources
    before=live();torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    builder=builder_load();torch.cuda.synchronize();builder_load_seconds=time.monotonic()-started
    started=time.monotonic();model=model_load(p);torch.cuda.synchronize();native_load_seconds=time.monotonic()-started
    records=[]
    for meta in p['training_catalog'][:2]:
        rid=meta['id'];bid=f'cost_{rid}_v1';folder=CACHE/bid
        if (folder/'anysplat_source_init.pt').exists():raise RuntimeError('Partial cost trial exists; preserve and explicitly version a new trial instead of silently timing reuse')
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();begin=time.monotonic()
        source=load_tensor(source_path(rid));source['id']=bid;folder.mkdir(parents=True,exist_ok=True)
        save_tensor(source_path(bid),source)
        stage=time.monotonic();initialize(builder,dict(meta,id=bid));torch.cuda.synchronize();init_seconds=time.monotonic()-stage
        stage=time.monotonic();raw=load_tensor(folder/'anysplat_source_init.pt');fit_sources(raw,source,folder)
        torch.cuda.synchronize();fit_seconds=time.monotonic()-stage;del raw
        stage=time.monotonic();bind_one(model,dict(meta,id=bid));torch.cuda.synchronize();bind_seconds=time.monotonic()-stage
        records.append(dict(id=rid,benchmark_id=bid,total_source_to_saved_scene_and_token_seconds=time.monotonic()-begin,
            initialization_with_serialization_seconds=init_seconds,fit_with_loading_serialization_seconds=fit_seconds,
            binding_plus_token_control_and_serialization_seconds=bind_seconds,
            peak_allocated_GiB=max(read(OUT/'source_initialization'/f'{bid}.json')['peak_allocated_GiB'],
                                   read(folder/'source_fit.json')['peak_allocated_GiB'],
                                   read(OUT/'binding'/f'{bid}.json')['peak_allocated_GiB']),
            scene_state_bytes=field_bytes(load_tensor(folder/'scene_memory.pt')),
            source_input_sha256=sha256(source_path(rid))))
    write_json(marker,dict(complete=True,cold_builder_load_seconds=builder_load_seconds,cold_native_load_seconds=native_load_seconds,
        records=records,live_before=before,live_after=live(),plan_sha256=sha256(OUT/'plan.json'),
        scope='Whole implemented RGB-to-saved-memory path including disk/hash and token-control overhead; two old fit clips, same fixed configuration. Shared GPU if other PIDs present; no exclusive speed claim.'))


@torch.no_grad()
def queries(p):
    marker=OUT/'cost/full_query.json'
    if marker.exists():return
    before=live();started=time.monotonic();model=model_load(p);torch.cuda.synchronize();load_seconds=time.monotonic()-started
    records=[]
    for meta in p['training_catalog'][:2]:
        rid=meta['id'];query=load_tensor(query_path(rid,2),'cuda')
        for kind in KINDS:
            field={};load_start=time.monotonic()
            if kind!='L':field=load_tensor(CACHE/rid/('token_memory.pt' if kind=='T' else 'scene_memory.pt'),'cuda')
            torch.cuda.synchronize();field_load_seconds=time.monotonic()-load_start
            reader=SceneReader(center_pe=kind in ('T','P')).cuda()
            final=OUT/f'{kind}_{p["seeds"][0]}'/'step_2000.pt'
            reader.load_state_dict(load_tensor(final)['model']);reader.eval()
            for condition in p['conditions']:
                image=query_image(query,condition);samples=[];geometry_times=[];native_times=[]
                torch.cuda.reset_peak_memory_stats()
                for index in range(7):
                    torch.cuda.synchronize();begin=time.monotonic()
                    with torch.autocast('cuda',dtype=torch.bfloat16):sample=prepare_read(model,image,query['c2w'],query['K'],field,kind)
                    torch.cuda.synchronize();prepared=time.monotonic()
                    with torch.autocast('cuda',dtype=torch.bfloat16):depth=forward_read(model,sample,reader,kind,checkpoint_blocks=False)
                    torch.cuda.synchronize();end=time.monotonic()
                    assert torch.isfinite(depth).all() and (depth>0).all()
                    if index>=2:
                        samples.append(end-begin);geometry_times.append(prepared-begin);native_times.append(end-prepared)
                    del depth,sample
                records.append(dict(id=rid,kind=kind,condition=condition,query=2,
                    source_state_tensor_bytes=field_bytes(field),field_load_seconds=field_load_seconds,
                    total_query_seconds=samples,current_encoder_and_geometry_seconds=geometry_times,
                    native_and_four_reader_calls_seconds=native_times,
                    peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3))
            del reader,field;torch.cuda.empty_cache()
    write_json(marker,dict(complete=True,records=records,cold_native_load_seconds=load_seconds,
        live_before=before,live_after=live(),plan_sha256=sha256(OUT/'plan.json'),
        scope='Uncached current image encoder, query geometry, all native blocks and DPT; field resident during warm query, load separately; no GT used. Shared GPU if other PIDs present; no exclusive speed claim.'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['construction','query']);args=parser.parse_args()
    torch.set_num_threads(4);p=freeze()
    (construction if args.action=='construction' else queries)(p)


if __name__=='__main__':main()
