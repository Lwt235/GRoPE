"""Query-camera-only descriptors; original44 fields, source bindings and patches reused."""
import argparse
import time
from experiments.native_input_pe_data import *
from models.gaussian_scene_memory import center_ray_descriptor,contribution_weights,field_bytes
from models.gaussian_scene_learned_projection import spherical_field


@torch.no_grad()
def prepare(meta):
    rid=meta['id'];marker=OUT/'geometry'/f'{rid}.json'
    if marker.exists():
        r=read(marker);assert r['complete'] and sha256(r['source_path'])==r['source_sha256']
        assert all(sha256(k)==v for k,v in r['files'].items());return
    begin=time.monotonic();source=scene_path(rid);h=sha256(source);field=load(source,'cuda');sphere=spherical_field(field)
    assert field_bytes(field)==4319652
    assert torch.allclose(field['scales'].float().square().sum(-1),sphere['scales'].square().sum(-1),rtol=1e-5)
    files={};inputs={};timings=[]
    for q in EVAL_QUERIES:
        query=load(query_path(rid,q),'cuda');K=query['K'].clone();K[:2]/=14
        inputs[str(query_path(rid,q))]=sha256(query_path(rid,q));t=time.monotonic()
        center=center_ray_descriptor(field['means'],query['c2w'],K,32,32,field['unit'])
        fw,fa=contribution_weights(field,query['c2w'][None],K[None],32,32)
        sw,sa=contribution_weights(sphere,query['c2w'][None],K[None],32,32)
        payload=dict(center=center.bfloat16(),full=torch.log1p(2048*fw[0].reshape(1024,2048).float()).bfloat16(),
            sphere=torch.log1p(2048*sw[0].reshape(1024,2048).float()).bfloat16())
        assert payload['center'].shape==(1024,2048,7)
        assert all(torch.isfinite(v).all() for v in payload.values())
        path=geometry_path(rid,q);path.parent.mkdir(parents=True,exist_ok=True)
        save_tensor(path,{k:v.cpu() for k,v in payload.items()});files[str(path)]=sha256(path)
        for c in CONDITIONS:inputs[str(patch_path(rid,q,c))]=sha256(patch_path(rid,q,c))
        timings.append(dict(query=q,seconds=time.monotonic()-t,full_alpha_mean=float(fa.mean()),sphere_alpha_mean=float(sa.mean())))
    assert sha256(source)==h
    write_json(marker,dict(complete=True,id=rid,place=meta['place'],source_path=str(source),source_sha256=h,
        files=files,inputs=inputs,timings=timings,seconds=time.monotonic()-begin,
        scene_tensor_bytes=field_bytes(field),source_unchanged=True,content_unchanged=True,
        geometry_inputs='saved field and given query camera/intrinsics/unit only; no query RGB, depth or GT support'))
    print(dict(geometry_complete=rid,seconds=time.monotonic()-begin),flush=True)


def main():
    a=argparse.ArgumentParser();a.add_argument('--shard',type=int,default=0);a.add_argument('--shards',type=int,default=2);args=a.parse_args()
    p=initialize();torch.set_num_threads(4)
    for i,meta in enumerate(p['training_catalog']+p['catalog']):
        if i%args.shards==args.shard:prepare(meta)
    write_json(OUT/'preparation'/f'{args.shard}.json',dict(complete=True,shard=args.shard,shards=args.shards))


if __name__=='__main__':main()
