"""Full-receiver gradients/resource/restore checks on two old training clips only."""
from experiments.native_receiver_data import *
from experiments.native_receiver_runtime import model_load,forward_read
from experiments.gaussian_scene_preflight import objective
from models.gaussian_scene_rendered import RenderedFeatureFusion
from models.gaussian_scene_rendered import forward_read as original_forward_read


def main():
    torch.set_num_threads(4);p=initialize();model=model_load(p);records=[]
    core=enable_receiver(model)
    names=[n for n,t in model.named_parameters() if t.requires_grad]
    assert all(n.startswith(PREFIXES) for n in names)
    assert all(not t.requires_grad for t in model.aggregator.patch_embed.parameters())
    initial_core={n:t.detach().cpu() for n,t in core_state(model).items()}
    parameters=sum(t.numel() for t in core)
    pilot_path=CACHE/'pilot_restore.pt';pilot_path.parent.mkdir(parents=True,exist_ok=True)
    for meta in p['training_catalog'][:2]:
        rid=meta['id'];query=load(query_path(rid,2),'cuda')
        patches=load(patch_path(rid,2,'missing'),'cuda')['patches']
        payload=load(feature_path(rid,2),'cuda')
        for kind in KINDS:
            core_load(model,initial_core);model.zero_grad(set_to_none=True)
            torch.manual_seed(SEEDS[0]);reader=RenderedFeatureFusion().cuda()
            saved=load(FROZEN/f'{kind}_{SEEDS[0]}'/'initial.pt')['model']
            assert all(torch.equal(t.cpu(),saved[n]) for n,t in reader.state_dict().items())
            sample=dict(patches=patches,extra=None if kind=='Local' else payload['global' if kind=='Global' else 'render'],hw=(448,448))
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                base=original_forward_read(model,sample,reader=None,checkpoint_blocks=False)
                zero=forward_read(model,sample,reader,kind,checkpoint_blocks=False)
            error=float((base-zero).abs().max());assert error<1e-4
            opts=optimizers(reader,core,p);steps=[];torch.cuda.reset_peak_memory_stats()
            for step in range(2):
                for opt in opts:opt.zero_grad(set_to_none=True)
                sync_cuda();begin=time.monotonic()
                for micro in range(2):
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        depth=forward_read(model,sample,reader,kind)
                        loss=objective(depth,query['target'],query['valid'])
                    assert torch.isfinite(loss);(loss/2).backward()
                    del depth
                block_grad={}
                for group in ('frame_blocks','global_blocks'):
                    block_grad[group]=[float(getattr(model.aggregator,group)[i].attn.qkv.weight.grad.norm()) for i in range(24)]
                assert all(np.isfinite(v) and v>0 for a in block_grad.values() for v in a)
                head_gradient=sum(float(t.grad.norm()) for t in model.depth_head.parameters() if t.grad is not None)
                assert head_gradient>0
                assert all(t.grad is None for t in model.aggregator.patch_embed.parameters())
                fusion_gradient={n:float(t.grad.norm()) for n,t in reader.named_parameters() if t.grad is not None}
                norm=torch.nn.utils.clip_grad_norm_(list(reader.parameters())+core,p['gradient_clip'],foreach=False)
                assert torch.isfinite(norm)
                for opt in opts:opt.step()
                sync_cuda()
                steps.append(dict(step=step+1,loss=float(loss.detach()),seconds=time.monotonic()-begin,
                    native_block_gradients=block_grad,depth_head_gradient_sum=head_gradient,
                    fusion_gradients=fusion_gradient,gradient_norm=float(norm)))
            # One representative real native+fusion+AdamW resume, not only a toy module.
            restore_error=None;checkpoint_bytes=None
            if not records:
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    expected=forward_read(model,sample,reader,kind,checkpoint_blocks=False)
                # Gradients are not part of AdamW resume state and are cleared
                # before the next update anyway. Release them before slow I/O.
                for opt in opts:opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                begin=time.monotonic()
                save_tensor(pilot_path,dict(native=core_state(model),model=reader.state_dict(),optimizer=[o.state_dict() for o in opts],step=2))
                checkpoint_bytes=pilot_path.stat().st_size
                for opt in opts:
                    opt.zero_grad(set_to_none=True)
                    opt.state.clear()
                state=load(pilot_path);core_load(model,state['native']);reader.load_state_dict(state['model'])
                for opt,saved_opt in zip(opts,state['optimizer']):opt.load_state_dict(saved_opt)
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    got=forward_read(model,sample,reader,kind,checkpoint_blocks=False)
                restore_error=float((expected-got).abs().max());assert restore_error==0
                write_json(OUT/'checkpoint_preflight.json',dict(complete=True,bytes=checkpoint_bytes,
                    seconds=time.monotonic()-begin,prediction_max_abs=restore_error,
                    optimizer_groups=[len(o.param_groups) for o in opts],step=state['step']))
                del state,expected,got;pilot_path.unlink()
            record=dict(id=rid,kind=kind,initial_native_max_abs=error,native_parameters=parameters,
                fusion_parameters=sum(t.numel() for t in reader.parameters()),steps=steps,
                peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,restore_error=restore_error)
            records.append(record);write_json(OUT/'preflight_progress.json',dict(complete=False,records=records))
            print(dict(preflight=kind,id=rid,peak_GiB=record['peak_allocated_GiB'],seconds=[s['seconds'] for s in steps]),flush=True)
            for opt in opts:opt.zero_grad(set_to_none=True)
            model.zero_grad(set_to_none=True)
            del opts,reader,base,zero,loss,norm,sample;torch.cuda.empty_cache()
        del query,patches,payload;torch.cuda.empty_cache()
    write_json(OUT/'preflight.json',dict(complete=True,records=records,native_parameters=parameters,
        trainable_native_names=names,pilot_weights_discarded=True,old_training_only=True,
        native_parameters_not_low_rank=True,source_content_unchanged=True))
    print('full receiver preflight complete',flush=True)


if __name__=='__main__':main()
