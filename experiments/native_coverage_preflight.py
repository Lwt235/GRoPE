"""Two representative old/added inputs; reuse prior unchanged resume-format checks."""
import time
from experiments.native_coverage_data import *
from experiments.native_receiver_data import enable_receiver,core_state,core_load,optimizers
from experiments.native_receiver_runtime import model_load,forward_read
from models.gaussian_scene_rendered import RenderedFeatureFusion
from experiments.gaussian_scene_preflight import objective


def main():
    torch.set_num_threads(4);p=plan();model=model_load(p);core=enable_receiver(model)
    initial={n:t.detach().cpu() for n,t in core_state(model).items()};records=[]
    for meta in [p['original_training_catalog'][0],p['added_training_catalog'][0]]:
        rid=meta['id'];q=2;query=load(query_path(rid,q),'cuda')
        patches=load(patch_path(rid,q,'missing'),'cuda')['patches'];extra=load(feature_path(rid,q),'cuda')['render']
        for kind in KINDS:
            core_load(model,initial);model.zero_grad(set_to_none=True);torch.manual_seed(SEEDS[0])
            reader=RenderedFeatureFusion().cuda();old=load(FROZEN/f'{kind}_{SEEDS[0]}'/'initial.pt')['model']
            assert all(torch.equal(t.cpu(),old[n]) for n,t in reader.state_dict().items())
            opts=optimizers(reader,core,p);sample=dict(patches=patches,extra=extra if kind=='Render' else None,hw=(448,448))
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();begin=time.monotonic()
            for opt in opts:opt.zero_grad(set_to_none=True)
            for _ in range(2):
                with torch.autocast('cuda',dtype=torch.bfloat16):depth=forward_read(model,sample,reader,kind);loss=objective(depth,query['target'],query['valid'])
                assert torch.isfinite(loss);(loss/2).backward();del depth
            gradients={group:[float(getattr(model.aggregator,group)[i].attn.qkv.weight.grad.norm()) for i in range(24)] for group in ['frame_blocks','global_blocks']}
            assert all(np.isfinite(v) and v>0 for vv in gradients.values() for v in vv)
            assert any(t.grad is not None and float(t.grad.norm())>0 for t in model.depth_head.parameters())
            assert all(t.grad is None for t in model.aggregator.patch_embed.parameters())
            norm=torch.nn.utils.clip_grad_norm_(list(reader.parameters())+core,5.,foreach=False);assert torch.isfinite(norm)
            for opt in opts:opt.step()
            torch.cuda.synchronize();records.append(dict(id=rid,kind=kind,loss=float(loss),gradient_norm=float(norm),
                native_parameters=sum(t.numel() for t in core),peak_GiB=torch.cuda.max_memory_allocated()/2**30,seconds=time.monotonic()-begin))
            for opt in opts:opt.zero_grad(set_to_none=True)
            model.zero_grad(set_to_none=True);del opt,opts,reader,sample,loss,norm;torch.cuda.empty_cache()
    # Same native/fusion/two-Adam serialization was already actually restored.
    resume=read(BASE/'checkpoint_preflight.json');assert resume['complete'] and resume['prediction_max_abs']==0
    write_json(OUT/'preflight.json',dict(complete=True,records=records,pilot_weights_discarded=True,
        inherited_exact_resume_check=str(BASE/'checkpoint_preflight.json'),inherited_resume_sha256=sha256(BASE/'checkpoint_preflight.json'),
        no_repeated_large_restore_io=True,structural_and_optimizer_format_unchanged=True))
    print(dict(preflight_complete=True,peak_GiB=max(r['peak_GiB'] for r in records)),flush=True)


if __name__=='__main__':main()
