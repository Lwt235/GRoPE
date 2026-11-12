"""Exact saved-field rendering and old-training-only native gradient checks."""
import argparse
import time

from experiments.rendered_feature_data import *
from experiments.gaussian_scene_preflight import objective, official_current
from experiments.gaussian_scene_data import query_image
from experiments.gaussian_memory_native_run import model_load
from models.gaussian_scene_rendered import RenderedFeatureFusion, forward_read, fresh_sample


@torch.no_grad()
def prepare_one(meta):
    rid=meta['id']; marker=OUT/'features'/f'{rid}.json'
    if marker.exists():
        old=read(marker)
        assert old['complete'] and sha256(old['source_path'])==old['source_sha256']
        assert all(sha256(k)==v for k,v in old['files'].items())
        return
    source=SOURCE_CACHE/rid/'scene_memory.pt'; source_hash=sha256(source)
    binding=read(SOURCE_OUT/'binding'/f'{rid}.json')
    assert binding['source_only'] and binding['files'][str(source)]==source_hash
    field=load(source,'cuda'); valid=field['has_content']; assert bool(valid.any())
    features=field['features'].float(); average=features[valid].mean(0)
    indices=torch.where(valid.cpu())[0]
    generator=torch.Generator().manual_seed(20261111)
    permutation=indices[torch.randperm(len(indices),generator=generator)]
    shuffled=features.clone(); shuffled[indices.to('cuda')]=features[permutation.to('cuda')]
    parent=read(SOURCE_OUT/'query_cache'/f'{rid}.json')
    files={}; inputs={}; timings=[]
    for q in QUERIES:
        query=query_path(rid,q); inputs[str(query)]=sha256(query)
        original=SOURCE_CACHE/rid/f'geometry_{q}.pt'
        assert sha256(original)==parent['files'][str(original)]
        weights=load(original,'cuda')['weights'].float()
        torch.cuda.synchronize(); begin=time.monotonic()
        effective=weights*valid[None]
        alpha=effective.sum(-1,keepdim=True)
        rendered=effective@features
        global_context=alpha*average[None]
        shuffled_context=effective@shuffled
        torch.cuda.synchronize()
        assert rendered.shape==global_context.shape==(1024,1024)
        assert bool(torch.isfinite(rendered).all()) and bool(torch.isfinite(global_context).all())
        assert float(alpha.min())>=0 and float(alpha.max())<=1.0001
        payload=dict(render=rendered.cpu(),global_context=global_context.cpu(),
            shuffle=shuffled_context.cpu(),effective_alpha=alpha.cpu(),valid_indices=indices,
            shuffled_source_indices=permutation)
        # The public dataset key is deliberately independent of the Python keyword.
        payload['global']=payload.pop('global_context')
        path=feature_path(rid,q); path.parent.mkdir(parents=True,exist_ok=True)
        save_tensor(path,payload); files[str(path)]=sha256(path)
        inputs[str(original)]=sha256(original)
        timings.append(dict(query=q,derived_feature_seconds=time.monotonic()-begin))
        for condition in CONDITIONS:
            path=patch_path(rid,q,condition)
            assert sha256(path)==parent['files'][str(path)]
            inputs[str(path)]=parent['files'][str(path)]
    assert sha256(source)==source_hash
    write_json(marker,dict(complete=True,id=rid,place=meta['place'],source_path=str(source),
        source_sha256=source_hash,files=files,inputs=inputs,timings=timings,
        source_only_saved_state_reused=True,source_content_unchanged=True,
        shuffle_validity_and_geometry_unchanged=True,
        scope='Derived query-view feature cache; no query RGB or depth enters scene state or rendering.'))
    print(dict(features_complete=rid),flush=True)


def preflight(model,p):
    records=[]
    for meta in p['training_catalog'][:2]:
        rid=meta['id']; field=load(SOURCE_CACHE/rid/'scene_memory.pt','cuda')
        query=load(query_path(rid,2),'cuda'); image=query_image(query,'missing')
        reference=official_current(model,image); payload=load(feature_path(rid,2),'cuda')
        patches=load(patch_path(rid,2,'missing'),'cuda')['patches']
        for kind in KINDS:
            torch.manual_seed(SEEDS[0]); reader=RenderedFeatureFusion().cuda()
            assert sum(x.numel() for x in reader.parameters())==787712
            extra=None if kind=='Local' else payload['global' if kind=='Global' else 'render']
            sample=dict(patches=patches,extra=extra,hw=(448,448))
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                initial=forward_read(model,sample,reader,kind,checkpoint_blocks=False)
            error=float((initial-reference).abs().max()); assert error<1e-4
            fresh=fresh_sample(model,image,query,field,kind)
            patch_difference=float((fresh['patches'].float()-patches.float()).abs().max())
            assert patch_difference==0
            rendered_difference=0. if kind=='Local' else float((fresh['extra']-extra).abs().max())
            assert rendered_difference<2e-4,(rid,kind,rendered_difference)
            optimizer=torch.optim.AdamW(reader.parameters(),lr=p['lr'],weight_decay=p['weight_decay'])
            steps=[];torch.cuda.reset_peak_memory_stats()
            for step in range(2):
                optimizer.zero_grad(set_to_none=True);trace={};begin=time.monotonic()
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    depth=forward_read(model,sample,reader,kind,trace=trace)
                    loss=objective(depth,query['target'],query['valid'])
                loss.backward()
                gradients={n:float(x.grad.float().norm()) for n,x in reader.named_parameters() if x.grad is not None}
                assert torch.isfinite(loss) and trace.get('input',0)>0 and gradients['output.weight']>0
                if step:assert gradients['input.weight']>0
                norm=torch.nn.utils.clip_grad_norm_(reader.parameters(),p['gradient_clip'])
                assert torch.isfinite(norm);optimizer.step();torch.cuda.synchronize()
                steps.append(dict(step=step+1,loss=float(loss),gradients=gradients,
                    input_residual_gradient=trace['input'],seconds=time.monotonic()-begin))
            records.append(dict(id=rid,kind=kind,initial_native_max_abs=error,
                fresh_patch_max_abs=patch_difference,cached_vs_actual_renderer_max_abs=rendered_difference,
                steps=steps,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30))
            del reader,optimizer,sample,fresh,depth,loss,initial
        del field,query,payload,patches,reference;torch.cuda.empty_cache()
    write_json(OUT/'preflight.json',dict(complete=True,records=records,
        old_training_inputs_only=True,pilot_weights_discarded=True,parameters=787712))


def main():
    a=argparse.ArgumentParser();a.add_argument('--pilot',action='store_true');args=a.parse_args()
    torch.set_num_threads(4);p=initialize_plan()
    rows=p['training_catalog'][:2] if args.pilot else p['training_catalog']+p['catalog']
    for meta in rows:prepare_one(meta)
    if args.pilot:
        preflight(model_load(p),p)
        names=['models/gaussian_scene_rendered.py','models/gaussian_scene_memory.py',
            'models/gaussian_scene_native.py','experiments/rendered_feature_data.py',
            'experiments/rendered_feature_prepare.py','experiments/rendered_feature_train.py',
            'experiments/gaussian_scene_train.py','experiments/gaussian_scene_config.py']
        write_json(OUT/'configuration_lock.json',dict(plan_sha256=sha256(OUT/'plan.json'),
            preflight_sha256=sha256(OUT/'preflight.json'),source_sha256={n:sha256(ROOT/n) for n in names},
            fixed_before_development_outputs=True,formal_training_started=False))
    else:
        write_json(OUT/'features_complete.json',dict(complete=True,fields=len(rows),queries_per_field=4,
            plan_sha256=sha256(OUT/'plan.json'),source_fields_not_rebuilt=True,content_not_rebound=True))


if __name__=='__main__':main()
