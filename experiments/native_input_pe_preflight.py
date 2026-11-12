"""Old-training-only checks of one residual, all native gradients and new PE parameters."""
import time
from experiments.native_input_pe_data import *
from experiments.native_receiver_data import enable_receiver,core_state,core_load,optimizers,digest
from experiments.native_receiver_runtime import model_load
from models.gaussian_scene_input_pe import InputPEReader,forward_read
from models.gaussian_scene_native import forward_read as baseline_forward
from experiments.gaussian_scene_preflight import objective


def main():
    torch.set_num_threads(4);p=plan();RAM.mkdir(mode=0o700,exist_ok=True)
    assert RAM.stat().st_uid==os.getuid() and RAM.resolve().parent==Path('/dev/shm')
    model=model_load(p);core=enable_receiver(model)
    initial={n:t.detach().cpu().clone() for n,t in core_state(model).items()};records=[];identities={}
    for meta in [p['original_training_catalog'][0],p['added_training_catalog'][0]]:
        rid=meta['id'];query=load(query_path(rid,2),'cuda')
        for kind in KINDS:
            core_load(model,initial);model.zero_grad(set_to_none=True);torch.manual_seed(SEEDS[0]);reader=InputPEReader().cuda()
            assert sum(t.numel() for t in reader.parameters())==1054672
            identities[kind]=digest(reader)
            sample=to_device(sample_cpu(kind,rid,2,'missing'),'cuda')
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
                expected=baseline_forward(model,sample,reader=None,checkpoint_blocks=False)
                actual=forward_read(model,sample,reader,kind,checkpoint_blocks=False)
            error=float((expected-actual).abs().max());assert error==0
            opts=optimizers(reader,core,p);steps=[];torch.cuda.reset_peak_memory_stats();begin=time.monotonic()
            for step in range(3):
                for opt in opts:opt.zero_grad(set_to_none=True)
                trace={}
                with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
                    depth=forward_read(model,sample,reader,kind,trace=trace);loss=objective(depth,query['target'],query['valid'])
                assert torch.isfinite(loss);loss.backward()
                assert trace.get('input',0)>0
                if step==0:
                    gradients={group:[float(getattr(model.aggregator,group)[i].attn.qkv.weight.grad.norm()) for i in range(24)] for group in ['frame_blocks','global_blocks']}
                    assert all(np.isfinite(v) and v>0 for vv in gradients.values() for v in vv)
                    assert any(t.grad is not None and float(t.grad.norm())>0 for t in model.depth_head.parameters())
                    assert all(t.grad is None for t in model.aggregator.patch_embed.parameters())
                pe_grad=reader.center_pe[0].weight.grad
                extra=0. if pe_grad is None else float(pe_grad[:,7].float().norm())
                if kind in ('L','P'):assert extra==0
                if step==2 and kind in ('S','G'):assert extra>0
                norm=torch.nn.utils.clip_grad_norm_(list(reader.parameters())+core,p['gradient_clip'],foreach=False);assert torch.isfinite(norm)
                for opt in opts:opt.step()
                steps.append(dict(step=step+1,loss=float(loss),extra_column_gradient=extra,input_gradient=trace['input']))
            # Exercise new reader and reader-Adam serialization; native/two-Adam format has prior actual full restore.
            with torch.no_grad():reference=forward_read(model,sample,reader,kind,checkpoint_blocks=False)
            path=RAM/'preflight_reader.pt';save_tensor(path,dict(model=reader.state_dict(),optimizer=opts[0].state_dict()))
            restored=load(path);reader.load_state_dict(restored['model']);opts[0].load_state_dict(restored['optimizer'])
            with torch.no_grad():replay=forward_read(model,sample,reader,kind,checkpoint_blocks=False)
            restore_error=float((reference-replay).abs().max());assert restore_error==0;path.unlink()
            torch.cuda.synchronize();records.append(dict(id=rid,kind=kind,initial_native_max_abs=error,
                new_reader_optimizer_restore_max_abs=restore_error,steps=steps,native_gradients=gradients,
                native_parameters=sum(t.numel() for t in core),reader_parameters=1054672,
                peak_GiB=torch.cuda.max_memory_allocated()/2**30,seconds=time.monotonic()-begin))
            for opt in opts:opt.zero_grad(set_to_none=True)
            model.zero_grad(set_to_none=True);del opts,opt,reader,sample,depth,loss,expected,actual,reference,replay,restored;torch.cuda.empty_cache()
        assert len(set(identities.values()))==1
    prior=parent.BASE/'checkpoint_preflight.json';resume=read(prior)
    assert resume['complete'] and resume['prediction_max_abs']==0
    write_json(OUT/'preflight.json',dict(complete=True,records=records,common_reader_initialization_identical=True,
        pilot_weights_discarded=True,queries_from_training_only=True,formal_native_parameters=637383906,
        inherited_native_full_two_adam_restore=str(prior),inherited_restore_sha256=sha256(prior),
        new_reader_and_reader_adam_restore_exercised=True))
    print(dict(preflight_complete=True,peak_GiB=max(r['peak_GiB'] for r in records)),flush=True)


if __name__=='__main__':main()
