"""Cache current-image encodings and source-field query geometry, never a history prefix."""
import time

import numpy as np
import torch

from experiments.gaussian_scene_data import OUT,CACHE,QUERIES,CONDITIONS,query_path,query_image,read
from experiments.gaussian_scene_bind import bind_one,load_tensor
from experiments.gaussian_scene_config import freeze
from experiments.gaussian_memory_native_run import model_load
from experiments.gaussian_memory_incremental_data import save_tensor,write_json
from experiments.gaussian_memory_setup import sha256
from models.gaussian_memory_coarse_pe import patch_tokens
from models.gaussian_scene_memory import contribution_weights


def patch_path(rid,q,c):return CACHE/rid/f'patch_{q}_{c}.pt'
def geometry_path(rid,q):return CACHE/rid/f'geometry_{q}.pt'


@torch.no_grad()
def cache_one(model,meta):
    rid=meta['id'];marker=OUT/'query_cache'/f'{rid}.json'
    if marker.exists():
        record=read(marker)
        assert record['plan_sha256']==sha256(OUT/'plan.json')
        assert all(sha256(path)==digest for path,digest in record['files'].items())
        return
    field=load_tensor(CACHE/rid/'scene_memory.pt','cuda');files={};times=[]
    for q in QUERIES:
        query=load_tensor(query_path(rid,q),'cuda')
        K=query['K'].clone();K[:2]/=14
        torch.cuda.synchronize();begin=time.monotonic()
        weights,alpha=contribution_weights(field,query['c2w'][None],K[None],32,32)
        torch.cuda.synchronize()
        geometry=dict(weights=weights[0].reshape(1024,-1).float().cpu(),alpha=alpha[0].cpu())
        save_tensor(geometry_path(rid,q),geometry);files[str(geometry_path(rid,q))]=sha256(geometry_path(rid,q))
        times.append(dict(query=q,geometry_seconds=time.monotonic()-begin))
        for c in CONDITIONS:
            # A source geometry repair must not change an already verified current-image input.
            if not patch_path(rid,q,c).exists():
                with torch.autocast('cuda',dtype=torch.bfloat16):patches=patch_tokens(model,query_image(query,c))
                save_tensor(patch_path(rid,q,c),dict(patches=patches.cpu()))
            files[str(patch_path(rid,q,c))]=sha256(patch_path(rid,q,c))
        del query,weights,alpha
    write_json(marker,dict(complete=True,id=rid,plan_sha256=sha256(OUT/'plan.json'),files=files,
        saved_scene_sha256=sha256(CACHE/rid/'scene_memory.pt'),source_state_unchanged=True,
        four_queries_same_saved_state=True,query_cache_contains_history_prefix=False,times=times))
    print(dict(query_cache_ready=rid),flush=True)


def main():
    p=freeze();model=model_load(p)
    for meta in p['training_catalog']+p['catalog']:
        bind_one(model,meta);cache_one(model,meta)


if __name__=='__main__':main()
