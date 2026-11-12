"""Reuse the paired optimizer and native export loop with input-level fusion."""
import argparse
import fcntl

from experiments.rendered_feature_data import *
from experiments import gaussian_scene_train as trainer
from models.gaussian_scene_rendered import RenderedFeatureFusion, forward_read
from experiments.gaussian_scene_preflight import metric_depth


def configure():
    trainer.OUT=OUT;trainer.CACHE=CACHE;trainer.dataset=dataset
    trainer.SceneReader=RenderedFeatureFusion;trainer.forward_read=forward_read
    return plan()


@torch.no_grad()
def intervene(model,p,seed,folder):
    marker=OUT/'interventions'/f'Render_{seed}.json'; final=folder/'step_2000.pt'
    if marker.exists():
        r=read(marker)
        assert r['complete'] and r['checkpoint_sha256']==sha256(final)
        assert sha256(r['prediction_path'])==r['prediction_sha256']
        return
    import time
    reader=RenderedFeatureFusion().cuda();reader.load_state_dict(load(final)['model']);reader.eval()
    data,_=dataset(p,'Render',False); keys=[];predictions=[];cached={};begin=time.monotonic()
    for key in data:
        rid,q,c=key; sample=data[key]
        if (rid,q) not in cached:cached[rid,q]=load(feature_path(rid,q))['shuffle']
        sample['extra']=cached[rid,q].to('cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            depth=forward_read(model,sample,reader,'Render',checkpoint_blocks=False)
        assert torch.isfinite(depth).all() and (depth>0).all()
        keys.append((*key,'shuffle_binding'));predictions.append(metric_depth(depth).float().cpu())
    path=CACHE/'evaluation'/f'Render_{seed}_interventions.pt'
    save_tensor(path,dict(keys=keys,depth_metric=torch.stack(predictions)))
    write_json(marker,dict(complete=True,seed=seed,predictions=len(keys),prediction_path=str(path),
        prediction_sha256=sha256(path),checkpoint_sha256=sha256(final),plan_sha256=sha256(OUT/'plan.json'),
        seconds=time.monotonic()-begin,scope='Fixed trained-model valid-slot binding dependency; not net utility.'))


def main():
    a=argparse.ArgumentParser();a.add_argument('--kind',required=True,choices=KINDS)
    a.add_argument('--seed',required=True,type=int,choices=SEEDS);args=a.parse_args()
    torch.set_num_threads(4);p=configure();folder=OUT/f'{args.kind}_{args.seed}'
    folder.mkdir(parents=True,exist_ok=True)
    assert read(OUT/'preflight.json')['complete'] and read(OUT/'features_complete.json')['complete']
    record=read(OUT/'configuration_lock.json')
    assert record['plan_sha256']==sha256(OUT/'plan.json')
    assert all(sha256(ROOT/n)==h for n,h in record['source_sha256'].items())
    with (folder/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        model=trainer.model_load(p)
        trainer.fit(model,p,args.kind,args.seed,folder);torch.cuda.empty_cache()
        trainer.evaluate(model,p,args.kind,args.seed,folder)
        if args.kind=='Render':
            torch.cuda.empty_cache();intervene(model,p,args.seed,folder)


if __name__=='__main__':main()
