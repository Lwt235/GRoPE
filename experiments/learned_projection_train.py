"""Use the established paired trainer with one learned eight-input PE."""
import argparse
import fcntl

from experiments.learned_projection_data import *
from experiments import gaussian_scene_train as trainer
from models.gaussian_scene_learned_projection import LearnedProjectionReader


def dataset(p,kind,training):
    result={};fields={}
    for meta in p['training_catalog'] if training else p['catalog']:
        rid=meta['id'];record=read(OUT/'geometry'/f'{rid}.json')
        assert record['complete'] and sha256(record['source_path'])==record['source_sha256']
        field=load(record['source_path'],'cuda');fields[rid]=field
        for q in QUERIES:
            path=geometry_path(rid,q);assert sha256(path)==record['files'][str(path)]
            geometry=geometry_tensor(load(path,'cuda'),kind)
            query=load(query_path(rid,q),'cuda')
            assert sha256(query_path(rid,q))==record['inputs'][str(query_path(rid,q))]
            for c in CONDITIONS:
                path=patch_path(rid,q,c);assert sha256(path)==record['inputs'][str(path)]
                result[(rid,q,c)]=dict(patches=load(path,'cuda')['patches'],features=field['features'],
                    valid=field['has_content'],geometry=geometry,hw=(448,448),query=query,condition=c,place=meta['place'])
    return result,fields


def configure():
    trainer.OUT=OUT;trainer.CACHE=CACHE;trainer.dataset=dataset;trainer.SceneReader=LearnedProjectionReader
    return plan()


def main():
    a=argparse.ArgumentParser();a.add_argument('--kind',required=True,choices=KINDS)
    a.add_argument('--seed',required=True,type=int,choices=SEEDS);args=a.parse_args()
    torch.set_num_threads(4);p=configure();folder=OUT/f'{args.kind}_{args.seed}';folder.mkdir(parents=True,exist_ok=True)
    assert read(OUT/'preflight.json')['complete'] and read(OUT/'geometry_complete.json')['complete']
    lock_record=read(OUT/'configuration_lock.json')
    assert lock_record['plan_sha256']==sha256(OUT/'plan.json')
    assert all(sha256(ROOT/n)==h for n,h in lock_record['source_sha256'].items())
    with (folder/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        model=trainer.model_load(p)
        trainer.fit(model,p,args.kind,args.seed,folder);torch.cuda.empty_cache()
        trainer.evaluate(model,p,args.kind,args.seed,folder)


if __name__=='__main__':main()
