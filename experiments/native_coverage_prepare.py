"""Identical sensor construction and binding on added scenes; isolated caches."""
import argparse
import time
from experiments.native_coverage_data import *
from experiments.gaussian_memory_native_run import model_load
from models.gaussian_memory_coarse_pe import patch_tokens
from models.gaussian_scene_memory import lift_source_features,field_bytes,contribution_weights
from experiments.gaussian_scene_data import query_image


def construct(meta):
    import experiments.source_geometry_build as builder
    builder.branch_cache=lambda mode:CACHE/'scenes'
    builder.branch_out=lambda mode:OUT
    builder.source_path=source_pair_path
    rid=meta['id'];source=load(source_pair_path(rid),'cuda')
    depths,valid,indices,candidate=builder.sampled_inputs(source)
    builder.fit_one(source,'sensor',depths,valid,indices,candidate)


@torch.no_grad()
def bind_cache(model,meta):
    rid=meta['id'];marker=OUT/'query_cache'/f'{rid}.json'
    if marker.exists():
        r=read(marker);assert all(sha256(k)==v for k,v in r['files'].items());return
    begin=time.monotonic();folder=CACHE/'scenes'/rid
    source=load(source_rgb_path(rid),'cuda');field=load(folder/'coarse_geometry.pt','cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):
        features=torch.cat([patch_tokens(model,image.float()/255) for image in source['rgb']]).reshape(4,32,32,1024)
    K=source['K'].clone();K[:,:2]/=14
    field['features'],field['mass'],field['has_content']=lift_source_features(field,features,source['c2w'],K)
    field['source_ids']=torch.tensor(plan()['sources'],device='cuda',dtype=torch.int32)
    assert field_bytes(field)==4319652 and field['features'].shape==(2048,1024)
    save_tensor(scene_path(rid),field)
    digest=sha256(scene_path(rid));files={str(scene_path(rid)):digest}
    for q in EVAL_QUERIES:
        query=load(query_path(rid,q),'cuda');k=query['K'].clone();k[:2]/=14
        weights,alpha=contribution_weights(field,query['c2w'][None],k[None],32,32)
        effective=weights[0].reshape(1024,-1).float()*field['has_content'][None]
        rendered=effective@field['features'].float()
        save_tensor(feature_path(rid,q),dict(render=rendered.cpu(),effective_alpha=effective.sum(-1).cpu()))
        files[str(feature_path(rid,q))]=sha256(feature_path(rid,q))
        files[str(query_path(rid,q))]=sha256(query_path(rid,q))
        for condition in CONDITIONS:
            with torch.autocast('cuda',dtype=torch.bfloat16):patches=patch_tokens(model,query_image(query,condition))
            save_tensor(patch_path(rid,q,condition),dict(patches=patches.cpu()))
            files[str(patch_path(rid,q,condition))]=sha256(patch_path(rid,q,condition))
    assert sha256(scene_path(rid))==digest
    write_json(marker,dict(complete=True,id=rid,files=files,source_only_scene=True,queries_used_in_binding=False,
        content_valid=int(field['has_content'].sum()),scene_tensor_bytes=field_bytes(field),seconds=time.monotonic()-begin))
    print(dict(bound_and_cached=rid,seconds=time.monotonic()-begin),flush=True)


def main():
    a=argparse.ArgumentParser();a.add_argument('--shard',type=int,default=0);a.add_argument('--shards',type=int,default=2);args=a.parse_args()
    torch.set_num_threads(4);p=plan();rows=[m for i,m in enumerate(p['added_training_catalog']) if i%args.shards==args.shard]
    for meta in rows:construct(meta)
    model=model_load(p)
    for meta in rows:bind_cache(model,meta)
    write_json(OUT/'preparation'/f'{args.shard}.json',dict(complete=True,fields=len(rows)))


if __name__=='__main__':main()
