"""Exact cached tensors staged from CPU per sample; scientific trainer unchanged."""
from collections.abc import Mapping
import argparse
import fcntl

from experiments.learned_projection_data import *
from experiments import learned_projection_train as original
from experiments import gaussian_scene_train as trainer


def to_device(value, device):
    if torch.is_tensor(value):return value.to(device)
    if isinstance(value,dict):return {k:to_device(v,device) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return type(value)(to_device(v,device) for v in value)
    return value


class StagedSamples(Mapping):
    def __init__(self, cpu_samples):self.cpu_samples=cpu_samples
    def __len__(self):return len(self.cpu_samples)
    def __iter__(self):return iter(self.cpu_samples)
    def __getitem__(self,key):return to_device(self.cpu_samples[key],'cuda')


def dataset(p,kind,training):
    result={};fields={}
    for meta in p['training_catalog'] if training else p['catalog']:
        rid=meta['id'];record=read(OUT/'geometry'/f'{rid}.json')
        assert record['complete'] and sha256(record['source_path'])==record['source_sha256']
        field=load(record['source_path']);fields[rid]=field
        for q in QUERIES:
            path=geometry_path(rid,q);assert sha256(path)==record['files'][str(path)]
            geometry=geometry_tensor(load(path),kind);query=load(query_path(rid,q))
            assert sha256(query_path(rid,q))==record['inputs'][str(query_path(rid,q))]
            for c in CONDITIONS:
                path=patch_path(rid,q,c);assert sha256(path)==record['inputs'][str(path)]
                result[rid,q,c]=dict(patches=load(path)['patches'],features=field['features'],
                    valid=field['has_content'],geometry=geometry,hw=(448,448),query=query,condition=c,place=meta['place'])
    return StagedSamples(result),fields


def main():
    a=argparse.ArgumentParser();a.add_argument('--kind',required=True,choices=KINDS)
    a.add_argument('--seed',required=True,type=int,choices=SEEDS);args=a.parse_args()
    torch.set_num_threads(4);p=original.configure();trainer.dataset=dataset
    proof=read(OUT/'memory_staging/validation.json');assert proof['complete']
    assert proof['source_sha256']==sha256(ROOT/'experiments/learned_projection_stream_train.py')
    lock=read(OUT/'configuration_lock.json');assert lock['plan_sha256']==sha256(OUT/'plan.json')
    assert all(sha256(ROOT/n)==h for n,h in lock['source_sha256'].items())
    folder=OUT/f'{args.kind}_{args.seed}'
    with (folder/'worker.lock').open('a') as worker:
        fcntl.flock(worker,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_json(folder/'staged_resume.json',dict(kind=args.kind,seed=args.seed,staging='CPU exact cached tensors; CUDA per sample',
            source_sha256=proof['source_sha256'],validation_sha256=sha256(OUT/'memory_staging/validation.json'),
            previous_peak_allocated_GiB=read(folder/'progress.json').get('peak_allocated_GiB') if (folder/'progress.json').exists() else None,
            initial_and_optimizer_checkpoint_preserved=True,scientific_trainer_unchanged=True))
        model=trainer.model_load(p)
        trainer.fit(model,p,args.kind,args.seed,folder);torch.cuda.empty_cache()
        trainer.evaluate(model,p,args.kind,args.seed,folder)


if __name__=='__main__':main()
