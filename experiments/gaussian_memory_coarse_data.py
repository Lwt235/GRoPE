"""Finite coarse-memory experiment identities; reuse only previously used inputs."""
import argparse
import shutil
from pathlib import Path
import numpy as np
import torch
from experiments.gaussian_memory_native_pe_data import ROOT, DATA, CACHE as PARENT_CACHE, OUT as PARENT_OUT, load, read, write_json, save_tensor, sha256, input_path

OUT = ROOT / 'results/gaussian_memory/coarse_memory_pe1'
CACHE = DATA / 'coarse_memory_pe_v1'
KINDS = ('center', 'sphere', 'gaussian')
CAPACITIES = (256, 1024)
SEEDS = (20261110, 20261111, 20261112)
SOURCES = ('models/gaussian_memory_coarse_pe.py', 'experiments/gaussian_memory_coarse_data.py', 'experiments/gaussian_memory_coarse_cache.py', 'experiments/gaussian_memory_coarse_train.py', 'models/gaussian_memory_native_pe.py', 'experiments/gaussian_memory_native_run.py')


def plan(): return read(OUT / 'plan.json')
def keyname(rid, q, condition, capacity): return f'{rid}_q{q}_{condition}_{capacity}'
def sample_path(rid, q, condition, capacity): return CACHE / rid / f'q{q}_{condition}_{capacity}.pt'
def example_keys(p, split):
    catalog = p['training_catalog'] if split == 'train' else p['catalog']
    return [(m['id'], q, c, n) for m in catalog for q in (p['queries'] if split == 'train' else (11,)) for c in p['conditions'] for n in CAPACITIES]


def verify(p):
    lock = OUT / 'training_lock.json'
    for name, digest in (read(lock)['source_sha256'] if lock.exists() else p['source_sha256']).items(): assert sha256(ROOT / name) == digest, name


def prepare():
    old = read(PARENT_OUT / 'plan.json')
    OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    p = {k: old[k] for k in ('training_catalog','catalog','training_ids','evaluation_ids','queries','conditions','model_path','model_sha256','official_source_sha256','github_commit','hf_revision','lr','weight_decay','gradient_clip','bootstrap','fixed_examples')}
    p.update(experiment='coarse_memory_pe1', kinds=list(KINDS), capacities=list(CAPACITIES), seeds=list(SEEDS), steps=2000, effective_batch=2,
        source_sha256={s:sha256(ROOT/s) for s in SOURCES}, parent_plan_sha256=sha256(PARENT_OUT/'plan.json'),
        document_sha256=sha256(ROOT/'docs/gaussian_memory_coarse_budget_plan.md'), training_examples=192, development_examples=48,
        expected_adapted_predictions=432, all_data_previously_used=True, new_confirmation=False,
        primary='256-slot persistent ROI centered log-MSE: gaussian vs center; sphere matched-trace control',
        secondary='raw/bias/whole/full-input, 1024 capacity, cost; no GT geometry input',
        rank=8, adapt_layers=list(range(24)), adaptation='query Q and output only; frozen historical K/V and all original weights',
        training_loss='full raw log-MSE + centered log-MSE', final_checkpoint_only=True,
        query_prefix='current RGB backbone only; both passes read coarse memory starting at global0',
        source_build='static full native source replay allowed, released before query; not bounded-memory online construction',
        input_records=old['records'])
    if (OUT/'plan.json').exists(): assert read(OUT/'plan.json') == p
    else: write_json(OUT/'plan.json',p)
    for s in SOURCES:
        dest=OUT/'source_snapshot'/s; dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists(): shutil.copy2(ROOT/s,dest)
    write_json(OUT/'execution_status.json',dict(state='implementation_and_preflight',finite_goal=True,new_confirmation=False))
    print({'prepared':p['experiment'],'train':192,'dev':48},flush=True)


if __name__=='__main__': prepare()
