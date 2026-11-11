"""Fixed local inputs for a pretrained native historical-memory diagnostic."""
import argparse
import json
import shutil

import numpy as np
import torch
from PIL import Image

from experiments.gaussian_memory_setup import ROOT, DATA, sha256
from experiments.gaussian_memory_incremental_data import sensor_target, write_json, save_tensor
from experiments.gaussian_memory_native_setup import DEST, STORE, COMMIT

OUT = ROOT / 'results/gaussian_memory/native_history1'
CACHE = DATA / 'native_history_v1'
PARENT = ROOT / 'results/gaussian_memory/history_visibility_control1'
OLD_CACHE = DATA / 'history_visibility_v1'
RECENT = (0, 4, 6, 8, 10)
HISTORY = (0, 1, 2, 3, 4, 6, 8, 10)
EXTRA = (1, 2, 3)
CONDITIONS = ('full', 'persistent')
ARMS = ('recent', 'history', 'history_mask', 'history_drop')
SOURCES = ('experiments/gaussian_memory_native_data.py',
           'experiments/gaussian_memory_native_run.py',
           'experiments/gaussian_memory_native_setup.py',
           'experiments/gaussian_memory_raw_history_run.py',
           'experiments/gaussian_memory_incremental_data.py')


def read(path): return json.loads(path.read_text())


def load(path): return torch.load(path, map_location='cpu', weights_only=False)


def make_clip(meta):
    rid = meta['id']; folder = CACHE / rid; folder.mkdir(parents=True, exist_ok=True)
    old = read(PARENT / 'inputs' / f'{rid}.json')
    mask_path = OLD_CACHE / rid / 'masks.pt'
    assert sha256(mask_path) == old['files']['masks.pt']
    masks = load(mask_path)['patch_masks'].cpu().bool()
    # Masks were defined on the old 20x15 patch grid. Preserve physical regions.
    pixels = masks.reshape(12, 15, 20).repeat_interleave(14, 1).repeat_interleave(14, 2)
    rgb = []; dimensions = []; identities = {str(mask_path): sha256(mask_path)}
    for index in (*HISTORY, 11):
        f = meta['row']['frames'][index]
        for kind in ('rgb', 'depth'):
            path = f[kind + '_path']; identities[path] = sha256(path)
            assert identities[path] == f[kind + '_sha256']
        with Image.open(f['rgb_path']) as image:
            image = image.convert('RGB'); width, height = image.size
            shape = (518, round(height * 518 / width / 14) * 14)
            assert shape[1] <= 518, 'Current fixed SUN3D cohort should require no crop.'
            dimensions.append({'frame': index, 'raw_wh': [width, height], 'input_wh': list(shape)})
            rgb.append(torch.from_numpy(np.asarray(image.resize(shape, Image.Resampling.BICUBIC)).copy()).permute(2, 0, 1))
    assert len({tuple(r['input_wh']) for r in dimensions}) == 1
    image = torch.stack(rgb); height, width = image.shape[-2:]
    mapped = torch.stack([torch.from_numpy(np.asarray(Image.fromarray(p.numpy()).resize((width, height), Image.Resampling.NEAREST)).copy()) for p in pixels])
    target, valid = sensor_target(meta['row']['frames'][11])
    sample = {'rgb_uint8': image, 'frame_ids': list((*HISTORY, 11)),
              'source_masks': mapped, 'metric_masks': pixels, 'target': target, 'valid': valid,
              'hidden': pixels[11], 'dimensions': dimensions,
              'metric_shape': [210, 280], 'native_shape': [height, width]}
    save_tensor(folder / 'inputs.pt', sample)
    rec = {'complete': True, 'id': rid, 'place': meta['place'], 'path': str(folder / 'inputs.pt'),
           'sha256': sha256(folder / 'inputs.pt'), 'identities': identities,
           'parent_record_sha256': sha256(PARENT / 'inputs' / f'{rid}.json'),
           'native_shape': [height, width], 'mask_counts_original': masks.sum(-1).tolist(),
           'mask_pixels_native': mapped.sum((1, 2)).tolist(), 'mask_generation_uses_query_depth': False}
    write_json(OUT / 'inputs' / f'{rid}.json', rec)
    return rec


def inputs():
    CACHE.mkdir(parents=True, exist_ok=True)
    parent = read(PARENT / 'plan.json')
    for meta in parent['catalog']:
        marker = OUT / 'inputs' / f'{meta["id"]}.json'
        if marker.exists():
            record = read(marker); assert sha256(record['path']) == record['sha256']
        else: record = make_clip(meta)
        print(json.dumps({'input_complete': meta['id'], 'shape': record['native_shape']}), flush=True)


def prepare():
    parent = read(PARENT / 'plan.json'); setup = read(OUT / 'setup.json')
    assert setup['complete'] and setup['github_commit'] == COMMIT
    records = {m['id']: read(OUT / 'inputs' / f'{m["id"]}.json') for m in parent['catalog']}
    for r in records.values(): assert r['complete'] and sha256(r['path']) == r['sha256']
    runtime = sorted(p for p in (DEST / 'src/streamvggt').rglob('*.py'))
    assert runtime
    plan = {'experiment': 'native_history1', 'catalog': parent['catalog'], 'records': records,
            'setup_sha256': sha256(OUT / 'setup.json'), 'model_path': setup['model_path'],
            'model_sha256': setup['model_sha256'], 'hf_revision': setup['hf_revision'], 'github_commit': COMMIT,
            'recent': RECENT, 'history': HISTORY, 'extra': EXTRA, 'query': 11, 'arms': ARMS,
            'conditions': CONDITIONS, 'expected_predictions': 96, 'new_training_steps': 0,
            'all_data_previously_used': True, 'fresh_confirmation': False,
            'mask_fill': [.485, .456, .406], 'dtype': 'BF16 native attention; FP32 official heads',
            'metric_definition': 'Raw native depth resized bilinearly to existing 210x280 sensor grid; no scale calibration or fitted alignment.',
            'primary': 'persistent ROI log-MSE: history versus recent; history_mask provides same-length content evidence.',
            'drop_definition': 'Remove EXTRA source entries from all aggregator and camera-head caches before final query. Later cached states retain indirect history.',
            'camera_iterations': 4, 'bootstrap': {'draws': 5000, 'seed': 20261111, 'unit': 'paired place only; one pretrained checkpoint'},
            'fixed_examples': ['fidelity_00', 'fidelity_01', 'fidelity_02'],
            'counterexample': 'largest history minus recent persistent ROI error; alphabetical tie break',
            'document_sha256': sha256(ROOT / 'docs/gaussian_memory_native_route_plan.md'),
            'source_sha256': {n: sha256(ROOT / n) for n in SOURCES},
            'official_source_sha256': {str(p.relative_to(DEST)): sha256(p) for p in runtime},
            'source_support_records': {m['id']: sha256(OUT / 'sources' / f'{m["id"]}.json') for m in parent['catalog']},
            'isolated_transformers_version': '4.51.3', 'parent_plan_sha256': sha256(PARENT / 'plan.json')}
    plan['evaluation_protocol_sha256'] = sha256(OUT / 'evaluation_protocol.json')
    path = OUT / 'plan.json'
    if path.exists(): assert read(path) == json.loads(json.dumps(plan))
    else: write_json(path, plan)
    for name, digest in plan['source_sha256'].items():
        destination = OUT / 'source_snapshot' / name; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination); assert sha256(destination) == digest
    write_json(OUT / 'input_checks.json', {'complete': True, 'places': 12, 'inputs': 12,
        'model_sha256': sha256(plan['model_path']), 'source_files': len(runtime), 'plan_sha256': sha256(path)})
    print(json.dumps({'prepared': True, 'outputs': 96}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['inputs', 'prepare'])
    args = parser.parse_args(); globals()[args.action]()
