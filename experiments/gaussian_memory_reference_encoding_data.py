"""Reference-only S=1 features with exactly the frozen S=4 geometry."""
import argparse
import hashlib
import json
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.gaussian_memory_setup import ROOT, DATA, COMMIT, sha256
from experiments.gaussian_memory_data import REF
from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_view_data import load_backbone
from experiments.gaussian_memory_coverage_data import OUT as COVERAGE, SEEDS, checked as coverage_checked, model_path as coverage_model_path

OUT = ROOT / 'results/gaussian_memory/reference_encoding1'
REF_CACHE = DATA / 'reference_encoding_v1'
VARIANTS = ('wide', 's4', 's1')


def model_path(variant, seed, initial=False):
    if variant in ('wide', 's4'):
        return coverage_model_path('broad', 'wide' if variant == 'wide' else 'direct', seed, initial)
    return OUT / f's1_{seed}' / ('initial.pt' if initial else 'step_2000.pt')


def prepare():
    coverage = coverage_checked()
    OUT.mkdir(parents=True, exist_ok=True); assert not (OUT / 'plan.json').exists()
    assert json.loads((COVERAGE / 'status.json').read_text())['complete']
    sources = ['experiments/gaussian_memory_reference_encoding_' + name + '.py' for name in ('data', 'train', 'evaluate')]
    sources += ['models/gaussian_memory_direct_reference.py', 'models/gaussian_memory_view_token.py',
                'models/gaussian_memory_attention.py', 'models/gaussian_memory_projection.py',
                'experiments/gaussian_memory_view_data.py']
    identities = [COVERAGE / 'plan.json', COVERAGE / 'cache_identity.json', COVERAGE / 'evaluation/scores.json',
                  DATA / 'source_identity.json', DATA / 'model.safetensors', ROOT / 'third_party/vggt/PINNED_COMMIT']
    identities += [model_path(v, s, ini) for v in ('wide', 's4') for s in SEEDS for ini in (False, True)]
    identities += [r[k] for r in coverage['catalog'] for k in ('raw', 'tokens', 'direct')]
    hashes = {str(path): sha256(path) for path in identities}
    assert hashes[str(DATA / 'model.safetensors')] == 'f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e'
    assert (ROOT / 'third_party/vggt/PINNED_COMMIT').read_text().strip() == COMMIT
    plan = {'source_sha256': {p: sha256(ROOT / p) for p in sources}, 'input_sha256': hashes,
            'document_sha256': sha256(ROOT / 'docs/gaussian_memory_reference_encoding_plan.md'),
            'catalog': coverage['catalog'], 'training_ids': coverage['training_ids'],
            'primary_places': coverage['primary_places'], 'center_variance': coverage['center_variance'],
            'seeds': SEEDS, 'variants': VARIANTS, 'steps': 2000, 'batch': 2, 'new_runs': 3, 'reused_models': 6,
            'reference_frames': REF, 'reference_slots': 1200, 'feature_dim': 2048,
            'only_change': 'reference content from S=4 joint to S=1 independent VGGT encoding',
            'geometry_source': 'unchanged S=4 Direct memories',
            'feature_normalization': 'per-token LayerNorm before half storage',
            'shuffle': 'within each reference image, permute features only among valid slots',
            'confirmation_claim': False, 'no_new_downloads': True}
    write_json(OUT / 'plan.json', plan)
    print(json.dumps({'prepared': True, 'new_runs': 3, 'episodes': len(plan['catalog']), 'training_scenes': 12}), flush=True)


def checked(inputs=False):
    plan = json.loads((OUT / 'plan.json').read_text())
    for path, digest in plan['source_sha256'].items():
        assert sha256(ROOT / path) == digest, path
    assert sha256(ROOT / 'docs/gaussian_memory_reference_encoding_plan.md') == plan['document_sha256']
    if inputs:
        for path, digest in plan['input_sha256'].items():
            assert sha256(path) == digest, path
    return plan


@torch.inference_mode()
def reference_memory(model, reference_images, frozen_geometry):
    """This interface has four reference RGBs and geometry; no query or teacher."""
    assert reference_images.shape == (4, 3, 210, 280)
    assert set(frozen_geometry) == {'means', 'covars', 'opacities'}
    patches = []
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for batch in reference_images.split(2):
            tokens, start = model.aggregator(batch[:, None])
            assert tokens[-1].shape[1] == 1
            patches.append(tokens[-1][:, 0, start:].float())
    context = torch.cat(patches).reshape(1200, 2048)
    memory = {k: v.detach().cpu().clone() for k, v in frozen_geometry.items()}
    memory['features'] = F.layer_norm(context, (2048,)).half().cpu()
    assert all(torch.isfinite(v).all() for v in memory.values())
    return memory


def spatial_permutation(memory, episode):
    """Preserve each image's valid feature multiset and all geometry/masks."""
    seed = int.from_bytes(hashlib.sha256(('reference-encoding-shuffle-v1:' + episode).encode()).digest()[:8], 'little')
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.arange(1200)
    valid = memory['opacities'].detach().cpu() > 0
    for view in range(4):
        ids = torch.arange(view * 300, (view + 1) * 300)[valid[view * 300:(view + 1) * 300]]
        permutation[ids] = ids[torch.randperm(len(ids), generator=generator)]
    return permutation.to(memory['features'].device)


def extract():
    plan = checked(True); torch.set_num_threads(4)
    REF_CACHE.mkdir(parents=True, exist_ok=True)
    model = load_backbone(); started = time.monotonic(); hashes, records = {}, []
    for meta in plan['catalog']:
        raw = torch.load(meta['raw'], map_location='cpu', weights_only=False)
        old = torch.load(meta['direct'], map_location='cpu', weights_only=False)['memory']
        ref_images = raw['rgb'][REF].permute(0, 3, 1, 2).float().cuda() / 255.
        geometry = {k: old[k] for k in ('means', 'covars', 'opacities')}
        memory = reference_memory(model, ref_images, geometry)
        assert all(torch.equal(memory[k], old[k]) for k in geometry)
        # This is a post-construction source comparison. Only the four reference
        # entries are used; no cached query target enters reference_memory.
        rec = torch.load(meta['tokens'], map_location='cpu', weights_only=False)
        cached_refs = rec['teacher_tokens'][REF].reshape(1200, 2048).float()
        cached_features = F.layer_norm(cached_refs, (2048,)).half()
        reference_cache_mse = float((memory['features'].float() - cached_features.float()).square().mean())
        difference = float((memory['features'].float() - old['features'].float()).square().mean())
        permutation = spatial_permutation(memory, meta['id']).cpu()
        path = REF_CACHE / (meta['id'] + '.pt'); assert not path.exists()
        torch.save({'memory': memory, 'episode': meta['id'], 'reference_frames': REF,
                    'source_raw_sha256': sha256(meta['raw']), 'source_geometry_sha256': sha256(meta['direct']),
                    'reference_sequence_length': 1, 'query_images_used': False, 'teacher_used': False,
                    'permutation': permutation}, path)
        hashes[str(path)] = sha256(path)
        records.append({'episode': meta['id'], 'geometry_equal': True, 'valid_slots': int((memory['opacities'] > 0).sum()),
                        's1_vs_existing_reference_cache_mse': reference_cache_mse,
                        's1_vs_s4_feature_mse': difference,
                        'tensor_bytes': sum(v.numel() * v.element_size() for v in memory.values())})
        print(json.dumps({'cached': meta['id'], 'reference_cache_mse': reference_cache_mse,
                          'seconds': time.monotonic() - started}), flush=True)
    write_json(OUT / 'reference_cache_identity.json', {'sha256': hashes, 'records': records,
               'seconds': time.monotonic() - started, 'peak_GiB': torch.cuda.max_memory_allocated() / 1024**3,
               'no_query_input_to_reference_builder': True})


def checked_cache():
    plan = checked()
    for path, digest in json.loads((OUT / 'reference_cache_identity.json').read_text())['sha256'].items():
        assert sha256(path) == digest, path
    return plan


def load_record(meta, device='cpu'):
    result = {'meta': meta, 'raw': torch.load(meta['raw'], map_location='cpu', weights_only=False),
              'rec': torch.load(meta['tokens'], map_location='cpu', weights_only=False)}
    for variant, path in [('s4', meta['direct']), ('s1', REF_CACHE / (meta['id'] + '.pt'))]:
        saved = torch.load(path, map_location=device, weights_only=False)
        result[variant] = {k: v.float() for k, v in saved['memory'].items()}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['prepare', 'extract'])
    args = parser.parse_args(); {'prepare': prepare, 'extract': extract}[args.command]()
