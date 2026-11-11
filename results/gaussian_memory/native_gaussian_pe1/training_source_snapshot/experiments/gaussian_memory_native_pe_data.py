"""Prospective identities and local RGB/mask inputs for native Gaussian PE."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from experiments.gaussian_memory_setup import ROOT, DATA, sha256
from experiments.gaussian_memory_incremental_data import sensor_target, write_json, save_tensor, SEEDS, QUERIES
from experiments.gaussian_memory_native_data import load, read, HISTORY

OUT = ROOT / 'results/gaussian_memory/native_gaussian_pe1'
CACHE = DATA / 'native_gaussian_pe_v1'
TRAIN_PARENT = ROOT / 'results/gaussian_memory/history_visibility_train1'
NATIVE_PARENT = ROOT / 'results/gaussian_memory/native_history1'
TRAIN_CACHE = DATA / 'history_visibility_training_v1'
NATIVE_CACHE = DATA / 'native_history_v1'
KINDS = ('native', 'center', 'sphere', 'gaussian')
CONDITIONS = ('full', 'persistent')
SOURCE_FILES = ('models/gaussian_memory_native_pe.py', 'experiments/gaussian_memory_native_pe_data.py',
                'experiments/gaussian_memory_native_pe_cache.py', 'experiments/gaussian_memory_native_pe_train.py',
                'experiments/gaussian_memory_native_run.py', 'experiments/gaussian_memory_native_data.py')


def source_ids(q): return [f for f in HISTORY if f < q]


def input_path(rid, q, condition): return CACHE / rid / f'q{q}_{condition}_input.pt'
def sample_path(rid, q, condition): return CACHE / rid / f'q{q}_{condition}.pt'
def sample_key(rid, q, condition): return f'{rid}/q{q}_{condition}'


def prepare_inputs():
    train = read(TRAIN_PARENT / 'plan.json'); native = read(NATIVE_PARENT / 'plan.json')
    training = train['training_catalog']; evaluation = native['catalog']
    assert len(training) == len(evaluation) == 12
    assert not {m['place'] for m in training} & {m['place'] for m in evaluation}
    for split, catalog, queries in [('train', training, QUERIES), ('development', evaluation, (11,))]:
        for meta in catalog:
            rid = meta['id']; marker = OUT / 'inputs' / f'{rid}.json'
            (CACHE / rid).mkdir(parents=True, exist_ok=True)
            if marker.exists():
                rec = read(marker)
                for p, digest in rec['files'].items(): assert sha256(p) == digest
                continue
            files = {}; raw_files = {}; mask_files = {}; rgb = {}
            for f in (*HISTORY, *queries):
                if f in rgb: continue
                frame = meta['row']['frames'][f]
                assert sha256(frame['rgb_path']) == frame['rgb_sha256']
                raw_files[frame['rgb_path']] = frame['rgb_sha256']
                with Image.open(frame['rgb_path']) as image:
                    image = image.convert('RGB'); w, h = image.size
                    shape = (518, round(h * 518 / w / 14) * 14); assert shape == (518, 392)
                    rgb[f] = torch.from_numpy(np.asarray(image.resize(shape, Image.Resampling.BICUBIC)).copy()).permute(2, 0, 1)
            for q in queries:
                if split == 'train':
                    mask_file = TRAIN_CACHE / rid / f'q{q}_masks.pt'
                    old = read(TRAIN_PARENT / 'inputs' / f'{rid}.json')
                    assert sha256(mask_file) == old['files'][mask_file.name]
                    patches = load(mask_file)['patch_masks'].bool()
                    metric_masks = patches.reshape(12, 15, 20).repeat_interleave(14, 1).repeat_interleave(14, 2)
                    masks = torch.stack([torch.from_numpy(np.asarray(Image.fromarray(p.numpy()).resize((518, 392), Image.Resampling.NEAREST)).copy()) for p in metric_masks])
                else:
                    mask_file = NATIVE_CACHE / rid / 'inputs.pt'
                    old = native['records'][rid]; assert sha256(mask_file) == old['sha256']
                    saved = load(mask_file); metric_masks = saved['metric_masks']; masks = saved['source_masks']
                mask_files[str(mask_file)] = sha256(mask_file)
                frame = meta['row']['frames'][q]; assert sha256(frame['depth_path']) == frame['depth_sha256']
                raw_files[frame['depth_path']] = frame['depth_sha256']
                target, valid = sensor_target(frame)
                for condition in CONDITIONS:
                    ids = source_ids(q) + [q]; images = []
                    for f in ids:
                        image = rgb[f].float() / 255.
                        if condition == 'persistent' and (f in (4, 6, 8, 10) or f == q):
                            fill = torch.tensor([.485, .456, .406])[:, None, None]
                            image = torch.where(masks[f][None], fill, image)
                        images.append(image)
                    value = {'images': torch.stack(images), 'frame_ids': ids, 'query': q, 'id': rid,
                             'condition': condition, 'place': meta['place'], 'split': split,
                             'target': target, 'valid': valid, 'hidden': metric_masks[q].bool()}
                    path = input_path(rid, q, condition); save_tensor(path, value); files[str(path)] = sha256(path)
            write_json(marker, {'complete': True, 'id': rid, 'split': split, 'files': files,
                                'raw_files': raw_files, 'mask_files': mask_files})
            print(json.dumps({'input_complete': rid, 'split': split, 'examples': len(queries) * 2}), flush=True)


def prepare_plan():
    train = read(TRAIN_PARENT / 'plan.json'); native = read(NATIVE_PARENT / 'plan.json')
    training, evaluation = train['training_catalog'], native['catalog']
    records = {m['id']: read(OUT / 'inputs' / f'{m["id"]}.json') for m in training + evaluation}
    for r in records.values():
        for p, digest in r['files'].items(): assert sha256(p) == digest
    plan = {'experiment': 'native_gaussian_pe1', 'training_catalog': training, 'catalog': evaluation,
            'training_ids': [m['id'] for m in training], 'evaluation_ids': [m['id'] for m in evaluation],
            'queries': list(QUERIES), 'conditions': list(CONDITIONS), 'seeds': list(SEEDS), 'kinds': list(KINDS),
            'records': records, 'model_path': native['model_path'], 'model_sha256': native['model_sha256'],
            'official_source_sha256': native['official_source_sha256'], 'github_commit': native['github_commit'],
            'hf_revision': native['hf_revision'], 'training_examples': 96, 'development_examples': 24,
            'steps': 2000, 'effective_batch': 2, 'lr': 1e-4, 'weight_decay': .01, 'gradient_clip': 5.,
            'rank': 8, 'adapt_layers': [20, 21, 22, 23], 'geometry_frozen': True, 'writer_frozen': True,
            'training_loss': 'full valid raw log-MSE + full centered log-MSE; equal weights',
            'final_checkpoint_only': True, 'new_confirmation': False, 'all_data_previously_used': True,
            'expected_predictions': 312, 'fixed_examples': ['fidelity_00', 'fidelity_01', 'fidelity_02'],
            'primary': 'gaussian vs center and sphere, persistent ROI raw log-MSE',
            'spatial_secondary': 'persistent ROI centered log-MSE; raw scale/bias and normal input retained',
            'bootstrap': {'draws': 5000, 'seed': 20261111, 'unit': 'paired place and paired model seed'},
            'source_sha256': {n: sha256(ROOT / n) for n in SOURCE_FILES},
            'document_sha256': sha256(ROOT / 'docs/gaussian_memory_native_pe_plan.md'),
            'point3r_reference_sha256': sha256(OUT / 'references/pos_embed_con.py'),
            'parent_plan_sha256': sha256(NATIVE_PARENT / 'plan.json'),
            'training_parent_plan_sha256': sha256(TRAIN_PARENT / 'plan.json')}
    path = OUT / 'plan.json'
    if path.exists(): assert read(path) == plan
    else: write_json(path, plan)
    for n, digest in plan['source_sha256'].items():
        p = OUT / 'source_snapshot' / n; p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists(): assert sha256(p) == digest
        else: shutil.copy2(ROOT / n, p)
    write_json(OUT / 'execution_status.json', {'state': 'prepared', 'finite_goal': True, 'new_confirmation': False})
    return plan


def verify(plan):
    lock = OUT / 'training_lock.json'
    expected = read(lock)['source_sha256'] if lock.exists() else plan['source_sha256']
    for n, digest in expected.items(): assert sha256(ROOT / n) == digest, n


def load_plan(): return read(OUT / 'plan.json')


def examples(plan, split):
    catalog = plan['training_catalog'] if split == 'train' else plan['catalog']
    return [(m['id'], q, c) for m in catalog for q in (plan['queries'] if split == 'train' else (11,)) for c in CONDITIONS]


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['inputs', 'plan']); args = p.parse_args()
    prepare_inputs() if args.action == 'inputs' else prepare_plan()
