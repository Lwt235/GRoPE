"""Fresh encoder+real feature renderer+adapted full native receiver timings."""
from experiments.native_receiver_data import *
from experiments.native_receiver_runtime import model_load,forward_read
from experiments.gaussian_scene_data import query_image
from models.gaussian_scene_rendered import RenderedFeatureFusion,fresh_sample


@torch.no_grad()
def main():
    torch.set_num_threads(4);p=plan();begin=time.monotonic();model=model_load(p,offload_unused=False)
    sync_cuda();cold=time.monotonic()-begin;reader=RenderedFeatureFusion().cuda();records=[];weight_loading=[]
    for kind in KINDS:
        sync_cuda();load_begin=time.monotonic()
        state=load(checkpoint_path(kind,SEEDS[0]));core_load(model,state['native']);reader.load_state_dict(state['model']);del state
        sync_cuda();weight_loading.append(dict(kind=kind,seconds=time.monotonic()-load_begin,
            scope='Adapted native+fusion checkpoint read and transfer, excluded from warm query timing.'))
        for meta in p['training_catalog'][:2]:
            field=load(SOURCE_CACHE/meta['id']/'scene_memory.pt','cuda');query=load(query_path(meta['id'],2),'cuda')
            for condition in CONDITIONS:
                image=query_image(query,condition);seconds=[];torch.cuda.reset_peak_memory_stats()
                for repeat in range(7):
                    sync_cuda();begin=time.monotonic();sample=fresh_sample(model,image,query,field,kind)
                    depth=forward_read(model,sample,reader,kind,checkpoint_blocks=False)
                    sync_cuda();elapsed=time.monotonic()-begin
                    assert torch.isfinite(depth).all() and (depth>0).all()
                    if repeat>=2:seconds.append(elapsed)
                    del sample,depth
                records.append(dict(kind=kind,id=meta['id'],query=2,condition=condition,warmup=2,retained=5,
                    seconds=seconds,median_seconds=float(np.median(seconds)),peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30))
            del field,query,image;torch.cuda.empty_cache()
    inherited=FROZEN/'cost_description.json'
    write_json(OUT/'cost.json',dict(complete=True,queries=records,cold_native_load_seconds=cold,
        adapted_weight_loading=weight_loading,
        source_scene_tensor_bytes=p['source_state_bytes'],inherited_construction=str(inherited),
        inherited_construction_sha256=sha256(inherited),native_receiver_trainable_parameters=read(OUT/'preflight.json')['native_parameters'],
        deployment_weights_scope='Adapted full receiver replaces original common native weights; fusion787712 parameters; image encoder unchanged. Scene payload separate from common model and development cache.',
        timing_scope='Prepared current GPU RGB and resident saved field to fresh image encoder, actual feature/coverage renderer, one fusion, all24native blocks and DPT. No cache in timed queries; excludes I/O and cold model loading.',
        contention_scope='Shared GPU and variable contention; no exclusive ranking or cross-round speedup claim.',
        model_sha256=sha256(p['model_path'])))
    print('native receiver fresh-query cost complete',flush=True)


if __name__=='__main__':main()
