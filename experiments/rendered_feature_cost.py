"""Fresh encoder and actual rendered-feature queries; cache excluded from timing."""
import time

from experiments.rendered_feature_data import *
from experiments.gaussian_scene_data import query_image
from experiments.gaussian_memory_native_run import model_load
from models.gaussian_scene_rendered import RenderedFeatureFusion, forward_read, fresh_sample


@torch.no_grad()
def main():
    torch.set_num_threads(4);p=plan();begin=time.monotonic();model=model_load(p)
    torch.cuda.synchronize();cold=time.monotonic()-begin;readers={}
    for kind in KINDS:
        reader=RenderedFeatureFusion().cuda()
        reader.load_state_dict(load(OUT/f'{kind}_{SEEDS[0]}'/'step_2000.pt')['model'])
        readers[kind]=reader.eval()
    rows=[]
    for meta in p['training_catalog'][:2]:
        rid=meta['id'];field=load(SOURCE_CACHE/rid/'scene_memory.pt','cuda')
        query=load(query_path(rid,2),'cuda')
        for condition in CONDITIONS:
            image=query_image(query,condition)
            for kind in KINDS:
                times=[];torch.cuda.reset_peak_memory_stats()
                for repeat in range(7):
                    torch.cuda.synchronize();begin=time.monotonic()
                    sample=fresh_sample(model,image,query,field,kind)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        depth=forward_read(model,sample,readers[kind],kind,checkpoint_blocks=False)
                    torch.cuda.synchronize();elapsed=time.monotonic()-begin
                    assert torch.isfinite(depth).all() and (depth>0).all()
                    if repeat>=2:times.append(elapsed)
                    del sample,depth
                rows.append(dict(id=rid,query=2,condition=condition,kind=kind,warmup=2,retained=5,
                    seconds=times,median_seconds=float(np.median(times)),
                    peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30))
        del field,query,image;torch.cuda.empty_cache()
    size=sum(f.stat().st_size for f in CACHE.glob('*/features_*.pt'))
    source_cost=ROOT/'results/gaussian_memory/source_geometry_control1/cost_description.json'
    write_json(OUT/'cost.json',dict(complete=True,cold_native_load_seconds=cold,queries=rows,
        development_feature_cache_bytes=size,source_scene_tensor_bytes=p['source_state_bytes'],
        source_construction_reused_from=str(source_cost),source_construction_sha256=sha256(source_cost),
        scope='Prepared GPU RGB plus resident field to fresh encoder, actual rendering (Global/Render), one fusion, all24 native layers and DPT. No image I/O/cold loading. Shared GPU observations, no exclusive speed ranking.',
        scene_scope='Saved scene payload excludes common backbone, learned fusion and temporary rendering. Local accesses no source field. Derived feature cache is development acceleration, not deployed scene memory.',
        construction_scope='Inherited unchanged sensor construction; excludes predictor generation, sensor acquisition, raw preprocessing and cold loading. Not RGB end-to-end cost.'))
    print(dict(cost_complete=True,records=len(rows)),flush=True)


if __name__=='__main__':main()
