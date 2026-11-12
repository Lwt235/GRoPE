"""Full fresh-query timing from prepared RGB and resident source state."""
import time

from experiments.learned_projection_data import *
from experiments.gaussian_scene_data import query_image
from experiments.gaussian_memory_native_run import model_load
from models.gaussian_memory_coarse_pe import patch_tokens
from models.gaussian_scene_memory import contribution_weights,center_ray_descriptor
from models.gaussian_scene_learned_projection import LearnedProjectionReader,spherical_field
from models.gaussian_scene_native import forward_read


@torch.no_grad()
def fresh_sample(model,image,query,field,kind):
    with torch.autocast('cuda',dtype=torch.bfloat16):patches=patch_tokens(model,image)
    K=query['K'].clone();K[:2]/=14
    center=center_ray_descriptor(field['means'],query['c2w'],K,32,32,field['unit'])
    t=torch.zeros_like(center[...,:1])
    if kind!='P':
        state=spherical_field(field) if kind=='S' else field
        w,_=contribution_weights(state,query['c2w'][None],K[None],32,32)
        t=torch.log1p(2048*w[0].reshape(1024,2048).float())[...,None]
    return dict(patches=patches,features=field['features'],valid=field['has_content'],
        geometry=torch.cat((center,t),-1).bfloat16(),hw=(448,448))


@torch.no_grad()
def main():
    torch.set_num_threads(4);p=plan();start=time.monotonic();model=model_load(p)
    torch.cuda.synchronize();cold=time.monotonic()-start
    readers={}
    for kind in KINDS:
        reader=LearnedProjectionReader().cuda();reader.load_state_dict(load(OUT/f'{kind}_{SEEDS[0]}'/'step_2000.pt')['model'])
        reader.eval();readers[kind]=reader
    rows=[]
    for meta in p['training_catalog'][:2]:
        rid=meta['id'];field=load(SOURCE_CACHE/rid/'scene_memory.pt','cuda');query=load(query_path(rid,2),'cuda')
        for condition in CONDITIONS:
            image=query_image(query,condition)
            for kind in KINDS:
                samples=[];torch.cuda.reset_peak_memory_stats()
                for repeat in range(7):
                    torch.cuda.synchronize();begin=time.monotonic()
                    sample=fresh_sample(model,image,query,field,kind)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        depth=forward_read(model,sample,readers[kind],kind,checkpoint_blocks=False)
                    torch.cuda.synchronize();elapsed=time.monotonic()-begin
                    if repeat>=2:samples.append(elapsed)
                    assert torch.isfinite(depth).all() and (depth>0).all()
                    del sample,depth
                rows.append(dict(id=rid,query=2,condition=condition,kind=kind,warmup=2,retained=5,
                    seconds=samples,median_seconds=float(np.median(samples)),
                    peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30))
    size=sum(f.stat().st_size for f in CACHE.glob('*/geometry_*.pt'))
    write_json(OUT/'cost.json',dict(complete=True,cold_native_load_seconds=cold,queries=rows,
        development_geometry_cache_bytes=size,source_scene_tensor_bytes=p['source_state_bytes'],
        scope='Prepared GPU current RGB and resident source scene to fresh encoder, actual query geometry, all24 native layers/four reads/DPT. No image I/O; shared GPU, no exclusive speed claim. Geometry cache is evaluation acceleration, not persistent deployable scene size.',
        source_construction='Reused unchanged source_geometry_control1/sensor; inherited cost excludes predictor generation, sensor acquisition, raw preprocessing and cold load'))
    print(dict(cost_complete=True,records=len(rows)),flush=True)


if __name__=='__main__':main()
