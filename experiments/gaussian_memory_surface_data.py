"""Query sensor labels are isolated from reference-only writer inputs."""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from experiments.gaussian_memory_setup import ROOT, DATA, sha256
from experiments.gaussian_memory_attention_train import write_json
from experiments.round5_rgbd_data import _listing, sun3d_timestamp, load_sensor_depth

OUT = ROOT / 'results/gaussian_memory/surface1'
CACHE = DATA / 'surface_v1'
SEEDS = (20261110, 20261111, 20261112)
VARIANTS = ('local', 'fixed', 'center', 'gaussian')


def model_path(variant, seed, initial=False):
    return OUT / f'{variant}_{seed}' / ('initial.pt' if initial else 'step_2000.pt')


def sensor_path(frame):
    rgb = Path(frame['rgb_path']); directory = rgb.parent.parent
    if 'depth_path' in frame:
        path = Path(frame['depth_path'])
        assert sha256(path) == frame['depth_sha256'], path
    else:
        names = _listing(directory / 'depth_index.html', 'png')
        times = np.array([sun3d_timestamp(n) for n in names], dtype='int64')
        nearest = int(np.abs(times - sun3d_timestamp(rgb.name)).argmin())
        path = directory / 'depth' / names[nearest]
    assert path.is_file(), path
    return path, abs(sun3d_timestamp(path.name) - sun3d_timestamp(rgb.name))


def coarse_depth(path):
    native = load_sensor_depth(path)
    depth = np.asarray(Image.fromarray(native).resize((280, 210), Image.Resampling.NEAREST)).copy()
    samples = depth.reshape(15, 14, 20, 14).transpose(0, 2, 1, 3).reshape(300, 196)
    valid = np.isfinite(samples) & (samples > .1) & (samples < 10.)
    fraction = valid.mean(-1)
    # A sentinel row prevents an all-NaN warning; zero-coverage rows stay invalid.
    temporary = np.where(valid, samples, np.nan)
    temporary[fraction == 0, 0] = 1.
    median = np.nanmedian(temporary, axis=-1).astype('float32')
    return median, fraction >= .5, fraction.astype('float32')


def prepare():
    OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'plan.json').exists()
    old_path = ROOT / 'results/gaussian_memory/coverage1/plan.json'
    old = json.loads(old_path.read_text()); identities = {}; diagnostics = []; samples = []
    for meta in old['catalog']:
        rec = torch.load(meta['tokens'], map_location='cpu', weights_only=False)
        role = torch.tensor([r == 'train' for r in rec['roles']])
        used = torch.ones(len(role), dtype=torch.bool) if meta['id'] in old['training_ids'] else ~role
        queries = sorted(set(rec['query'][used].tolist()))
        assert not set(queries) & {0, 4, 8, 12}
        depth = torch.ones(13, 300); valid = torch.zeros(13, 300, dtype=torch.bool); fractions = torch.zeros(13, 300)
        sources = []
        for q in queries:
            frame = meta['row']['frames'][q]; path, gap = sensor_path(frame)
            d, v, f = coarse_depth(path)
            assert v.any(), (meta['id'], q, 'no supervised patch')
            depth[q] = torch.from_numpy(d); valid[q] = torch.from_numpy(v); fractions[q] = torch.from_numpy(f)
            sources.append({'query': q, 'path': str(path), 'sha256': sha256(path), 'time_gap_raw': gap,
                            'valid_patches': int(v.sum()), 'valid_depth_range_m': [float(d[v].min()), float(d[v].max())]})
        dest = CACHE / (meta['id'] + '.pt'); assert not dest.exists()
        torch.save({'depth_m': depth, 'log_depth_m': depth.log(), 'valid': valid, 'fractions': fractions,
                    'query_ids': queries, 'sources': sources, 'supervision_only': True,
                    'writer_sensor_used': False, 'reference_rgb_only': True}, dest)
        identities[str(dest)] = sha256(dest)
        for key in ('raw', 'tokens'):
            identities[meta[key]] = sha256(meta[key])
        diagnostics.append({'id': meta['id'], 'scene': meta['row']['scene'], 'queries': len(queries),
                            'valid_fraction': float(valid[queries].float().mean()), 'sources': sources})
        if meta['id'] in old['training_ids']:
            for q in sorted(set(rec['query'][role].tolist())):
                samples.extend(depth[q][valid[q]].log().tolist())
        print(json.dumps({'labels': meta['id'], 'queries': len(queries), 'valid_fraction': diagnostics[-1]['valid_fraction']}), flush=True)
    sources = ['models/gaussian_memory_surface.py', 'models/gaussian_memory.py', 'models/gaussian_memory_projection.py',
               'models/gaussian_memory_attention.py', 'experiments/round5_rgbd_data.py']
    sources += ['experiments/gaussian_memory_surface_' + n + '.py' for n in ('data', 'train', 'evaluate', 'run')]
    plan = {'catalog': old['catalog'], 'training_ids': old['training_ids'], 'primary_places': old['primary_places'],
            'seeds': SEEDS, 'variants': VARIANTS, 'steps': 2000, 'batch': 2, 'lr': 3e-4, 'weight_decay': .01,
            'center_variance': old['center_variance'], 'local_width': 191, 'gradient_clip': 5.,
            'target': 'sensor_patch_median_log_depth_m', 'valid_range_m': [.1, 10.], 'valid_fraction_minimum': .5,
            'input_sha256': identities, 'source_sha256': {p: sha256(ROOT / p) for p in sources},
            'document_sha256': sha256(ROOT / 'docs/gaussian_memory_surface_plan.md'),
            'previous_catalog_sha256': sha256(old_path), 'training_log_depth_mean': float(np.mean(samples)),
            'query_RGB_in_writer': False, 'sensor_in_writer': False, 'sensor_in_reader': False,
            'confirmation_claim': False, 'data_downloads': False}
    write_json(OUT / 'plan.json', plan)
    write_json(OUT / 'target_identity.json', {'diagnostics': diagnostics, 'targets': len(diagnostics)})


def checked(inputs=False):
    plan = json.loads((OUT / 'plan.json').read_text())
    for p, digest in plan['source_sha256'].items():
        assert sha256(ROOT / p) == digest, p
    assert sha256(ROOT / 'docs/gaussian_memory_surface_plan.md') == plan['document_sha256']
    if inputs:
        for p, digest in plan['input_sha256'].items():
            assert sha256(p) == digest, p
    return plan


def load_record(meta):
    return {'meta': meta, 'raw': torch.load(meta['raw'], map_location='cpu', weights_only=False),
            'rec': torch.load(meta['tokens'], map_location='cpu', weights_only=False),
            'target': torch.load(CACHE / (meta['id'] + '.pt'), map_location='cpu', weights_only=False)}


if __name__ == '__main__':
    prepare()
