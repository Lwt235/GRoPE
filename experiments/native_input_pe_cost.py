"""Full warm query: current encoder, real PE construction, one read, native layers/DPT."""
import time
from experiments.native_input_pe_data import *
from experiments.native_receiver_runtime import model_load
from experiments.native_receiver_data import core_load
from models.gaussian_scene_input_pe import InputPEReader,forward_read
from models.gaussian_scene_memory import center_ray_descriptor,contribution_weights
from models.gaussian_scene_learned_projection import spherical_field
from models.gaussian_memory_coarse_pe import patch_tokens
from experiments.gaussian_scene_data import query_image


@torch.no_grad()
def bias_description(reader,sample,kind):
    """Fixed training inputs, descriptive logit ablation only; no native-output tuning."""
    geometry=sample['geometry'];valid=sample['valid']
    with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
        q=reader.query(reader.query_norm(sample['patches'])).reshape(1,-1,4,64).transpose(1,2)
        k=reader.key(reader.memory_norm(sample['features'])).reshape(1,-1,4,64).transpose(1,2)
        actual=reader.position_bias(kind,geometry,special_tokens=0).float()
        removed=geometry.clone();removed[...,7]=0
        center=reader.position_bias(kind,removed,special_tokens=0).float()
    qk=(q.float()@k.float().transpose(-1,-2))[0]/8
    a=actual[...,valid];c=center[...,valid];delta=a-c
    centered_delta=delta-delta.mean(-1,keepdim=True)
    centered_center=c-c.mean(-1,keepdim=True)
    lp=torch.log_softmax(qk[...,valid]+a,-1);lq=torch.log_softmax(qk[...,valid]+c,-1)
    probability=lp.exp();support=geometry[...,7][:,valid]>0
    return dict(extra_column_weight_norm=float(reader.center_pe[0].weight[:,7].norm()),
        extra_row_centered_bias_rms=float(centered_delta.square().mean().sqrt()),
        center_row_centered_bias_rms=float(centered_center.square().mean().sqrt()),
        descriptive_attention_KL_after_zeroing_extra=float((probability*(lp-lq)).sum(-1).mean()),
        nonzero_extra_pair_fraction=float(support.float().mean()),
        attention_mass_on_nonzero_extra=float((probability*support[None]).sum(-1).mean()))


@torch.no_grad()
def fresh(model,query,field,kind,condition):
    with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
        patches=patch_tokens(model,query_image(query,condition))
    sample=dict(patches=patches,hw=(448,448))
    if kind!='L':
        K=query['K'].clone();K[:2]/=14
        center=center_ray_descriptor(field['means'],query['c2w'],K,32,32,field['unit']).bfloat16()
        extra=torch.zeros_like(center[...,:1])
        if kind in ('S','G'):
            rf=spherical_field(field) if kind=='S' else field
            w,_=contribution_weights(rf,query['c2w'][None],K[None],32,32)
            extra=torch.log1p(2048*w[0].reshape(1024,2048).float()).bfloat16()[...,None]
        sample.update(features=field['features'],valid=field['has_content'],geometry=torch.cat((center,extra),-1).contiguous())
    return sample


@torch.no_grad()
def main():
    p=plan();torch.set_num_threads(4);begin=time.monotonic();model=model_load(p,offload_unused=False)
    torch.cuda.synchronize();cold=time.monotonic()-begin;records=[];diagnostics=[];reader=InputPEReader().cuda().eval()
    for kind in KINDS:
        state=load(checkpoint_path(kind,SEEDS[0]));core_load(model,state['native']);reader.load_state_dict(state['model']);del state
        for meta in [p['original_training_catalog'][0],p['added_training_catalog'][0]]:
            rid=meta['id'];query=load(query_path(rid,2),'cuda');field=load(scene_path(rid),'cuda') if kind!='L' else None
            for condition in CONDITIONS:
                values=[];errors=[];torch.cuda.reset_peak_memory_stats()
                cached=to_device(sample_cpu(kind,rid,2,condition),'cuda')
                expected=forward_read(model,cached,reader,kind,checkpoint_blocks=False)
                for i in range(5):
                    torch.cuda.synchronize();t=time.monotonic();sample=fresh(model,query,field,kind,condition)
                    depth=forward_read(model,sample,reader,kind,checkpoint_blocks=False)
                    torch.cuda.synchronize();elapsed=time.monotonic()-t
                    error=float((depth-expected).abs().max());assert error<2e-4
                    if i>=2:values.append(elapsed);errors.append(error)
                    del sample,depth
                records.append(dict(kind=kind,id=rid,condition=condition,seconds=values,cached_prediction_max_abs=max(errors),
                    peak_GiB=torch.cuda.max_memory_allocated()/2**30))
                if kind in ('S','G') and condition=='missing':
                    diagnostics.append(dict(kind=kind,id=rid,query=2,seed=SEEDS[0],**bias_description(reader,cached,kind)))
                del cached,expected
            del query,field;torch.cuda.empty_cache()
    write_json(OUT/'cost.json',dict(complete=True,records=records,cold_native_load_seconds=cold,
        summary={k:dict(median_ms=1000*float(np.median([v for r in records if r['kind']==k for v in r['seconds']])),
            peak_GiB=max(r['peak_GiB'] for r in records if r['kind']==k)) for k in KINDS},
        query_scope='Preprocessed448RGB resident; encoder plus actual center/sphere/full projection PE construction, single input read and all native24 frame/global/DPT. Disk/JPEG preprocessing, camera estimation and source construction excluded; given camera. Shared GPU, no exclusive speed claim.',
        source_construction_reference=str(PARENT_OUT/'cost_description.json'),scene_tensor_bytes=4319652,
        source_scope='Unchanged44saved fields; inherited construction costs retain original sensor/registration/I/O exclusions. No rebuild timed as zero-cost scene construction.'))
    write_json(OUT/'pe_description.json',dict(complete=True,records=diagnostics,
        scope='Predeclared first seed, first original/added training clip q2 missing input. Centered logit and float32-softmax descriptions; no native depth ablation, development selection, retraining or parameter adjustment. KL describes attention distribution, not downstream causal utility. Timings exclude these diagnostics.'))
    print('Full warm query costs and fresh-vs-cached predictions checked.',flush=True)


if __name__=='__main__':main()
