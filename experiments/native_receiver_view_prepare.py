"""Encode new queries and render unchanged saved content; no learning."""
import argparse
from experiments.native_receiver_view_data import *
from experiments.gaussian_scene_data import query_image
from experiments.gaussian_scene_reference import support as old_support
from experiments.gaussian_memory_native_run import model_load
from models.gaussian_memory_coarse_pe import patch_tokens
from models.gaussian_scene_memory import contribution_weights


@torch.no_grad()
def support(meta,p):
    # Reuse the exact old evaluation-only support implementation, isolated paths.
    import experiments.gaussian_scene_reference as reference
    saved=(reference.CACHE,reference.OUT,reference.query_path)
    reference.CACHE=CACHE;reference.OUT=OUT
    reference.query_path=lambda rid,q: query_path(rid,q) if q in QUERIES else old_query_path(rid,q)
    try: old_support(meta,dict(p,queries=list(QUERIES)+list(OLD_QUERIES)))
    finally: reference.CACHE,reference.OUT,reference.query_path=saved
    # Original helper uses CACHE/evaluation_support; our compact public path is separate.
    saved_path=CACHE/'evaluation_support'/f'{meta["id"]}.pt'
    assert saved_path.exists()


@torch.no_grad()
def prepare_one(model,meta,p):
    rid=meta['id'];marker=OUT/'query_cache'/f'{rid}.json'
    if marker.exists():
        record=read(marker);assert all(sha256(k)==v for k,v in record['files'].items());return
    expected=read(OUT/'identities.json')['scenes'][rid]
    assert sha256(expected['path'])==expected['sha256']
    field=load(expected['path'],'cuda');features=field['features'].float()
    files={};times=[];torch.cuda.synchronize();begin=time.monotonic()
    for q in QUERIES:
        query=load(query_path(rid,q),'cuda');K=query['K'].clone();K[:2]/=14
        start=time.monotonic();weights,alpha=contribution_weights(field,query['c2w'][None],K[None],32,32)
        effective=weights[0].reshape(1024,-1).float()*field['has_content'][None]
        rendered=effective@features
        assert rendered.shape==(1024,1024) and torch.isfinite(rendered).all()
        path=feature_path(rid,q);save_tensor(path,dict(render=rendered.cpu(),effective_alpha=effective.sum(-1).cpu()))
        files[str(path)]=sha256(path)
        for c in CONDITIONS:
            with torch.autocast('cuda',dtype=torch.bfloat16):patches=patch_tokens(model,query_image(query,c))
            path=patch_path(rid,q,c);save_tensor(path,dict(patches=patches.cpu()));files[str(path)]=sha256(path)
        torch.cuda.synchronize();times.append(dict(query=q,seconds=time.monotonic()-start))
        files[str(query_path(rid,q))]=sha256(query_path(rid,q))
        del query,weights,alpha,effective,rendered,patches
    support(meta,p)
    sp=CACHE/'evaluation_support'/f'{rid}.pt';files[str(sp)]=sha256(sp)
    assert sha256(expected['path'])==expected['sha256']
    torch.cuda.synchronize()
    write_json(marker,dict(complete=True,id=rid,files=files,source_sha256=expected['sha256'],
        plan_sha256=sha256(OUT/'plan.json'),source_state_unchanged=True,
        encoding_rendering_support_seconds=time.monotonic()-begin,times=times))
    print(dict(cache_ready=rid,seconds=time.monotonic()-begin),flush=True)


@torch.no_grad()
def replay_cache(model,p):
    from experiments.rendered_feature_data import feature_path as old_feature,patch_path as old_patch
    records=[]
    for meta in [p['training_catalog'][0],p['catalog'][0]]:
        rid=meta['id'];q=2;query=load(old_query_path(rid,q),'cuda')
        field=load(SOURCE_CACHE/rid/'scene_memory.pt','cuda');K=query['K'].clone();K[:2]/=14
        weights,_=contribution_weights(field,query['c2w'][None],K[None],32,32)
        actual=(weights[0].reshape(1024,-1).float()*field['has_content'][None])@field['features'].float()
        expected=load(old_feature(rid,q),'cuda')['render']
        with torch.autocast('cuda',dtype=torch.bfloat16):patches=patch_tokens(model,query_image(query,'missing'))
        cached=load(old_patch(rid,q,'missing'),'cuda')['patches']
        patch_error=float((patches.float()-cached.float()).abs().max())
        render_error=float((actual-expected).abs().max())
        assert patch_error==0 and render_error<2e-4,(rid,patch_error,render_error)
        records.append(dict(id=rid,query=q,patch_max_abs=patch_error,render_max_abs=render_error))
    write_json(OUT/'cache_replay.json',dict(complete=True,records=records,training=False))


def main():
    a=argparse.ArgumentParser();a.add_argument('--shard',type=int,default=0);a.add_argument('--shards',type=int,default=1)
    args=a.parse_args();torch.set_num_threads(4);p=plan();model=model_load(p)
    assert torch.cuda.is_available();torch.cuda.reset_peak_memory_stats()
    if args.shard==0:replay_cache(model,p)
    rows=p['training_catalog']+p['catalog']
    for i,meta in enumerate(rows):
        if i%args.shards==args.shard:prepare_one(model,meta,p)
    write_json(OUT/'preparation'/f'shard_{args.shard}.json',dict(complete=True,shard=args.shard,
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,device=torch.cuda.get_device_name(0)))


if __name__=='__main__':main()
