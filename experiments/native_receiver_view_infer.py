"""Inference-only reuse of all twelve fixed native/fusion model states."""
import argparse
import signal

from experiments.native_receiver_view_data import *
from experiments.native_receiver_data import core_load,restore_native
from experiments.native_receiver_runtime import model_load,forward_read
from experiments.gaussian_scene_preflight import metric_depth
from models.gaussian_scene_rendered import RenderedFeatureFusion

STOP=False


def request_stop(*args):
    global STOP
    STOP=True


def samples(p):
    for cohort,metas in [('train',p['training_catalog']),('development',p['catalog'])]:
        for meta in metas:
            rid=meta['id'];record=read(OUT/'query_cache'/f'{rid}.json')
            assert record['complete'] and record['plan_sha256']==sha256(OUT/'plan.json')
            assert all(sha256(k)==v for k,v in record['files'].items())
            for q in QUERIES:
                patches={c:load(patch_path(rid,q,c))['patches'] for c in CONDITIONS}
                extra=load(feature_path(rid,q))['render']
                yield cohort,rid,q,patches,extra


@torch.no_grad()
def replay(model,reader,kind,seed,regime,p):
    from experiments.rendered_feature_data import feature_path as old_feature,patch_path as old_patch
    from experiments.native_receiver_data import CACHE as OLD_CACHE
    records=[]
    for cohort,meta in [('train',p['training_catalog'][0]),('development',p['catalog'][0])]:
        rid=meta['id'];q=2;c='missing'
        sample=dict(patches=load(old_patch(rid,q,c),'cuda')['patches'],hw=(448,448),
            extra=load(old_feature(rid,q),'cuda')['render'] if kind=='Render' else None)
        pred=metric_depth(forward_read(model,sample,reader,kind,checkpoint_blocks=False)).float().cpu()
        if regime=='frozen' and cohort=='development':marker=read(FROZEN/'evaluation'/f'{kind}_{seed}.json')
        else:marker=read(PARENT/'evaluation'/f'{regime}_{cohort}_{kind}_{seed}.json')
        payload=load(marker['prediction_path']);index=[tuple(k) for k in payload['keys']].index((rid,q,c))
        expected=payload['depth_metric'][index]
        error=float((pred-expected).abs().max())
        assert error<1e-5,(kind,seed,regime,cohort,error)
        records.append(dict(cohort=cohort,id=rid,query=q,condition=c,max_abs_depth_difference=error))
    return records


@torch.no_grad()
def infer(model,p,regime,kind,seed):
    name=f'{regime}_{kind}_{seed}';marker=OUT/'evaluation'/f'{name}.json'
    if marker.exists():
        r=read(marker);assert r['complete'] and sha256(r['prediction_path'])==r['prediction_sha256'];return
    identity=read(OUT/'identities.json')['models'][name]
    assert sha256(identity['path'])==identity['sha256']
    torch.cuda.synchronize();begin=time.monotonic();state=load(identity['path'])
    if regime=='adapted':core_load(model,state['native'])
    # Each shard runs its frozen jobs first, then adapted. Frozen jobs all share
    # exactly the original pretrained receiver; no adapted state enters them.
    reader=RenderedFeatureFusion().cuda().eval();reader.load_state_dict(state['model']);del state
    torch.cuda.synchronize();loading=time.monotonic()-begin
    replays=replay(model,reader,kind,seed,regime,p)
    torch.cuda.reset_peak_memory_stats();predictions=[];keys=[];timings=[];cohorts=[]
    torch.cuda.synchronize();start=time.monotonic()
    for cohort,rid,q,patches,extra in samples(p):
        for condition in CONDITIONS:
            if STOP:raise SystemExit(75)
            sample=dict(patches=patches[condition].cuda(),extra=extra.cuda() if kind=='Render' else None,hw=(448,448))
            torch.cuda.synchronize();t=time.monotonic()
            depth=forward_read(model,sample,reader,kind,checkpoint_blocks=False)
            torch.cuda.synchronize();timings.append(time.monotonic()-t)
            assert torch.isfinite(depth).all() and (depth>0).all()
            predictions.append(metric_depth(depth).float().cpu());keys.append((rid,q,condition));cohorts.append(cohort)
            del depth,sample
    assert len(predictions)==192
    path=CACHE/'predictions'/f'{name}.pt';path.parent.mkdir(parents=True,exist_ok=True)
    save_tensor(path,dict(keys=keys,cohorts=cohorts,depth_metric=torch.stack(predictions)))
    write_json(marker,dict(complete=True,kind=kind,seed=seed,regime=regime,predictions=192,
        prediction_path=str(path),prediction_sha256=sha256(path),checkpoint=identity,
        plan_sha256=sha256(OUT/'plan.json'),replays=replays,weight_loading_seconds=loading,
        inference_export_seconds=time.monotonic()-start,cached_native_forward_seconds=timings,
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,training_steps=0))
    print(dict(evaluation_complete=name,predictions=192,seconds=time.monotonic()-start),flush=True)
    del reader,predictions;torch.cuda.empty_cache()


def main():
    a=argparse.ArgumentParser();a.add_argument('--shard',type=int,default=0);a.add_argument('--shards',type=int,default=3)
    args=a.parse_args();signal.signal(signal.SIGTERM,request_stop)
    torch.set_num_threads(4);p=plan();model=model_load(p)
    # Partition seeds, preserving identical pretrained receiver for frozen jobs.
    first=True
    for index,seed in enumerate(SEEDS):
        if index%args.shards!=args.shard:continue
        if not first:restore_native(model,p)
        for regime in ('frozen','adapted'):
            for kind in KINDS:infer(model,p,regime,kind,seed)
        first=False
    write_json(OUT/'inference'/f'shard_{args.shard}.json',dict(complete=True,training_steps=0))


if __name__=='__main__':main()
