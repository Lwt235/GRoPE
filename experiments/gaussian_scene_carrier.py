"""Source/coarse visual preservation and independent re-observation diagnostics."""
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.gaussian_scene_data import OUT,CACHE,source_path,query_path,read
from experiments.gaussian_scene_bind import load_tensor
from experiments.gaussian_scene_build import registered_field
from experiments.gaussian_scene_cache import patch_path,geometry_path
from experiments.gaussian_scene_config import freeze
from experiments.gaussian_memory_native_run import model_load
from experiments.gaussian_memory_incremental_data import save_tensor,write_json
from experiments.gaussian_memory_setup import sha256
from models.gaussian_scene_memory import rasterize_field,contribution_weights,field_to_device
from models.gaussian_memory_coarse_pe import patch_tokens


@torch.no_grad()
def feature_metrics(weights,field,target):
    alpha=weights.sum(-1)
    rendered=(weights@field['features'].float())/alpha[:,None].clamp_min(1e-8)
    mse=(rendered-target.float()).square().mean(-1)
    cosine=F.cosine_similarity(rendered,target.float(),dim=-1)
    mask=alpha>.0001
    return dict(covered_patches=int(mask.sum()),all_patch_mse=float(mse.mean()),
                covered_patch_mse=float(mse[mask].mean()) if mask.any() else None,
                covered_patch_cosine=float(cosine[mask].mean()) if mask.any() else None),mse.reshape(32,32).cpu(),cosine.reshape(32,32).cpu()


@torch.no_grad()
def main():
    torch.set_num_threads(4);p=freeze();model=model_load(p);records=[];manifests={}
    for meta in p['training_catalog']+p['catalog']:
        rid=meta['id'];marker=OUT/'carrier'/f'{rid}.json'
        if marker.exists():
            old=read(marker);assert old['complete'] and sha256(old['maps_path'])==old['maps_sha256'];records.append(old);continue
        folder=CACHE/rid;source=load_tensor(source_path(rid),'cuda');field=load_tensor(folder/'scene_memory.pt','cuda')
        source_hash=sha256(folder/'scene_memory.pt')
        raw=load_tensor(folder/'anysplat_source_init.pt')
        rawfield,_,_,registration=registered_field(raw,{key:(val.cpu() if torch.is_tensor(val) else val) for key,val in source.items()})
        rawfield=field_to_device(rawfield,'cuda');initial=load_tensor(folder/'coarse_initial.pt','cuda')
        maps={};rgb_metrics=[];feature_records=[];torch.cuda.synchronize();started=time.monotonic()
        # Same DC color convention and same standard renderer for each geometry state.
        for v,index in enumerate(p['sources']):
            K=source['K'][v:v+1].clone();K[:,:2]*=210/448
            target=F.interpolate(source['rgb'][v:v+1].float()/255,(210,210),mode='area')[0].permute(1,2,0)
            rgbrow=dict(view='source',frame=index)
            for name,state in [('raw_dc',rawfield),('coarse_initial',initial),('coarse_final',field)]:
                rgb,alpha,_=rasterize_field(state,source['c2w'][v:v+1],K,210,210)
                rgbrow[name]=dict(rgb_mse=float((rgb[0]-target).square().mean()),coverage_fraction=float((alpha>.0001).float().mean()))
                maps[f'source_{index}_{name}']=rgb[0].cpu()
            rgb_metrics.append(rgbrow)
            Kf=source['K'][v:v+1].clone();Kf[:,:2]/=14
            weights,_=contribution_weights(field,source['c2w'][v:v+1],Kf,32,32)
            with torch.autocast('cuda',dtype=torch.bfloat16):teacher=patch_tokens(model,source['rgb'][v].float()/255)[0]
            metrics,mse,cosine=feature_metrics(weights[0].reshape(1024,-1),field,teacher)
            feature_records.append(dict(view='source',frame=index,**metrics))
            maps[f'source_{index}_feature_mse']=mse;maps[f'source_{index}_feature_cosine']=cosine
        if meta in p['catalog']:
            for q in p['queries']:
                query=load_tensor(query_path(rid,q),'cuda')
                target=F.interpolate(query['rgb'][None].float()/255,(210,210),mode='area')[0].permute(1,2,0)
                rgbrow=dict(view='query',frame=q)
                for name,state in [('raw_dc',rawfield),('coarse_initial',initial),('coarse_final',field)]:
                    rgb,alpha,_=rasterize_field(state,query['c2w'][None],query['K_metric'][None],210,210)
                    rgbrow[name]=dict(rgb_mse=float((rgb[0]-target).square().mean()),coverage_fraction=float((alpha>.0001).float().mean()))
                    maps[f'query_{q}_{name}']=rgb[0].cpu()
                rgb_metrics.append(rgbrow)
                weights=load_tensor(geometry_path(rid,q),'cuda')['weights'];teacher=load_tensor(patch_path(rid,q,'full'),'cuda')['patches'][0]
                metrics,mse,cosine=feature_metrics(weights,field,teacher)
                feature_records.append(dict(view='query',frame=q,**metrics))
                maps[f'query_{q}_feature_mse']=mse;maps[f'query_{q}_feature_cosine']=cosine
        torch.cuda.synchronize()
        assert sha256(folder/'scene_memory.pt')==source_hash
        path=folder/'carrier_maps.pt';save_tensor(path,maps)
        result=dict(complete=True,id=rid,split='development' if meta in p['catalog'] else 'training',
            source_state_sha256=source_hash,source_state_unchanged=True,maps_path=str(path),maps_sha256=sha256(path),
            rgb_metrics=rgb_metrics,feature_metrics=feature_records,source_registration=registration,
            seconds=time.monotonic()-started,plan_sha256=sha256(OUT/'plan.json'),
            feature_comparison='alpha-normalized rendered1024d feature vs independently frozen current full-image encoder, covered threshold1e-4',
            raw_reference='Full source initialization geometry/DC under common registered coordinates; not official full-SH renderer',
            scope='Carrier preservation and re-observation, not native utility or PE effectiveness')
        write_json(marker,result);records.append(result);print(dict(carrier_ready=rid),flush=True)
        del source,field,raw,rawfield,initial,maps;torch.cuda.empty_cache()
    write_json(OUT/'carrier/summary.json',dict(complete=True,records=records,plan_sha256=sha256(OUT/'plan.json')))


if __name__=='__main__':main()
