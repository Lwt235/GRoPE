"""Run the unchanged paired trainer with isolated geometry-specific state and manifests."""
import argparse
import fcntl

import numpy as np
import torch

from experiments.source_geometry_data import *
from experiments import gaussian_scene_train as trainer

def configure(mode):
    trainer.OUT=branch_out(mode);trainer.CACHE=branch_cache(mode)
    trainer.query_path=query_path;trainer.patch_path=patch_path
    trainer.geometry_path=lambda rid,q:geometry_path(mode,rid,q)
    return plan(mode)

def main():
    a=argparse.ArgumentParser();a.add_argument('--mode',required=True,choices=MODES)
    a.add_argument('--kind',required=True,choices=KINDS);a.add_argument('--seed',required=True,type=int,choices=SEEDS);args=a.parse_args()
    torch.set_num_threads(4);p=configure(args.mode);folder=branch_out(args.mode)/f'{args.kind}_{args.seed}';folder.mkdir(parents=True,exist_ok=True)
    # One lock per actual fit prevents accidental duplicate dispatch across restarts.
    with (folder/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        model=trainer.model_load(p)
        trainer.fit(model,p,args.kind,args.seed,folder);torch.cuda.empty_cache()
        trainer.evaluate(model,p,args.kind,args.seed,folder)
        if args.kind=='G':torch.cuda.empty_cache();trainer.intervene(model,p,args.seed,folder)

if __name__=='__main__':main()
