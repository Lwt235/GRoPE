"""Query-first multiview teacher targets; immutable reference-only S1 memories."""
import argparse
import json
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.gaussian_memory_setup import ROOT, DATA, sha256
from experiments.gaussian_memory_data import REF
from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_view_data import load_backbone
from experiments.gaussian_memory_reference_encoding_data import OUT as PREVIOUS, REF_CACHE, SEEDS, model_path as previous_model_path, checked as previous_checked, load_record as previous_record
from models.gaussian_memory_view_token import normalized_error

OUT = ROOT / 'results/gaussian_memory/context_supervision1'
TARGET_CACHE = DATA / 'context_supervision_v1'
VARIANTS = ('wide', 'content', 'spatial')


def model_path(variant, seed, regime='context', initial=False):
    if regime == 'single':
        assert variant in ('wide', 'spatial')
        return previous_model_path('wide' if variant == 'wide' else 's1', seed, initial)
    return OUT / f'{variant}_{seed}' / ('initial.pt' if initial else 'step_2000.pt')


def initial_source(variant, seed):
    return previous_model_path('wide' if variant == 'wide' else 's1', seed, True)


def prepare():
    previous = previous_checked()
    assert json.loads((PREVIOUS / 'status.json').read_text())['complete']
    OUT.mkdir(parents=True, exist_ok=True); assert not (OUT / 'plan.json').exists()
    sources = ['models/gaussian_memory_context.py', 'models/gaussian_memory_direct_reference.py',
               'models/gaussian_memory_view_token.py', 'models/gaussian_memory_attention.py',
               'models/gaussian_memory_projection.py', 'experiments/gaussian_memory_view_data.py',
               'third_party/vggt/vggt/models/aggregator.py']
    sources += ['experiments/gaussian_memory_context_' + n + '.py' for n in ('data', 'train', 'evaluate')]
    inputs = [PREVIOUS / p for p in ('plan.json', 'reference_cache_identity.json', 'evaluation/scores.json')]
    inputs += [DATA / 'model.safetensors', DATA / 'source_identity.json', ROOT / 'third_party/vggt/PINNED_COMMIT']
    inputs += [initial_source(v, s) for v in ('wide', 'spatial') for s in SEEDS]
    inputs += [model_path(v, s, 'single') for v in ('wide', 'spatial') for s in SEEDS]
    inputs += [r[k] for r in previous['catalog'] for k in ('raw', 'tokens', 'direct')]
    inputs += [REF_CACHE / (r['id'] + '.pt') for r in previous['catalog']]
    hashes = {str(p): sha256(p) for p in inputs}
    assert hashes[str(DATA / 'model.safetensors')] == 'f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e'
    plan = {'source_sha256': {p: sha256(ROOT / p) for p in sources}, 'input_sha256': hashes,
            'document_sha256': sha256(ROOT / 'docs/gaussian_memory_context_supervision_plan.md'),
            'catalog': previous['catalog'], 'training_ids': previous['training_ids'],
            'primary_places': previous['primary_places'], 'center_variance': previous['center_variance'],
            'seeds': SEEDS, 'variants': VARIANTS, 'steps': 2000, 'batch': 2, 'new_runs': 9,
            'reused_models_for_target_diagnostic': 6, 'teacher_order': ['query'] + REF,
            'sequence_length_control': ['query'] * 5, 'teacher_view_index': 0,
            'student_query_source': 'unchanged masked S1 cache', 'student_reference_source': str(REF_CACHE),
            'only_training_target_change': 'S1 full-query to S5 query-first full-query+4-reference context',
            'content_control': 'same complete reference content and query-valid mask; zero key geometry and center distance',
            'shuffle': 'features within each reference image and each query-visible subset',
            'confirmation_claim': False, 'no_new_downloads': True}
    write_json(OUT / 'plan.json', plan)
    print(json.dumps({'prepared': True, 'episodes': len(plan['catalog']), 'new_runs': 9}), flush=True)


def checked(inputs=False):
    plan = json.loads((OUT / 'plan.json').read_text())
    for p, h in plan['source_sha256'].items():
        assert sha256(ROOT / p) == h, p
    assert sha256(ROOT / 'docs/gaussian_memory_context_supervision_plan.md') == plan['document_sha256']
    if inputs:
        for p, h in plan['input_sha256'].items():
            assert sha256(p) == h, p
    return plan


def target_queries(plan, meta, rec):
    return sorted({int(q) for q, role in zip(rec['query'], rec['roles'])
                   if meta['id'] in plan['training_ids'] or role == 'evaluation'})


@torch.inference_mode()
def query_teacher(model, query_images, reference_images, repeat_query=False):
    """Teacher only: query is always the first view and no other query enters."""
    assert reference_images.shape == (4, 3, 210, 280)
    assert query_images.ndim == 4 and query_images.shape[1:] == (3, 210, 280)
    references = query_images[:, None].expand(-1, 4, -1, -1, -1) if repeat_query else reference_images[None].expand(len(query_images), -1, -1, -1, -1)
    sequence = torch.cat([query_images[:, None], references], dim=1)
    assert sequence.shape[1] == 5
    with torch.autocast('cuda', dtype=torch.bfloat16):
        tokens, start = model.aggregator(sequence)
    return tokens[-1][:, 0, start:].half().cpu()


def extract():
    plan = checked(True); torch.set_num_threads(4)
    TARGET_CACHE.mkdir(parents=True, exist_ok=True)
    started = time.monotonic(); model = load_backbone(); hashes, records = {}, []
    for meta in plan['catalog']:
        raw = torch.load(meta['raw'], map_location='cpu', weights_only=False)
        rec = torch.load(meta['tokens'], map_location='cpu', weights_only=False)
        ids = target_queries(plan, meta, rec)
        assert not set(ids) & set(REF)
        images = raw['rgb'].permute(0, 3, 1, 2).float().cuda() / 255.
        reference_images = images[REF]
        target = torch.zeros((13, 300, 2048), dtype=torch.float16)
        self_target = torch.zeros_like(target)
        for begin in range(0, len(ids), 2):
            queries = ids[begin:begin + 2]
            target[queries] = query_teacher(model, images[queries], reference_images)
            self_target[queries] = query_teacher(model, images[queries], reference_images, repeat_query=True)
        assert torch.isfinite(target).all() and torch.isfinite(self_target).all()
        real = F.layer_norm(target[ids].float(), (2048,))
        repeat = F.layer_norm(self_target[ids].float(), (2048,))
        change = real - repeat
        global_change = change.mean(1, keepdim=True)
        path = TARGET_CACHE / (meta['id'] + '.pt'); assert not path.exists()
        torch.save({'episode': meta['id'], 'query_ids': ids, 'context_targets': target,
                    'repeat_query_targets': self_target, 'teacher_order': ['query'] + REF,
                    'teacher_query_position': 0, 'supervision_only': True,
                    'other_query_images_used': False, 'student_reference_modified': False,
                    'source_raw_sha256': sha256(meta['raw']), 'source_tokens_sha256': sha256(meta['tokens']),
                    'student_reference_sha256': sha256(REF_CACHE / (meta['id'] + '.pt'))}, path)
        hashes[str(path)] = sha256(path)
        record = {'episode': meta['id'], 'queries': ids, 'split': meta['eval_split'],
                  'real_reference_vs_repeat_query_mse': float(change.square().mean()),
                  'spatially_centered_reference_change_mse': float((change - global_change).square().mean()),
                  'global_reference_change_mse': float(global_change.square().mean()),
                  's1_vs_context_target_mse': float(normalized_error(rec['teacher_tokens'][ids], target[ids]).mean()),
                  's1_vs_repeat_query_target_mse': float(normalized_error(rec['teacher_tokens'][ids], self_target[ids]).mean())}
        records.append(record)
        print(json.dumps({'cached': meta['id'], 'queries': len(ids), 'seconds': time.monotonic() - started}), flush=True)
        del images, target, self_target, raw, rec
    write_json(OUT / 'target_cache_identity.json', {'sha256': hashes, 'records': records,
               'seconds': time.monotonic() - started, 'peak_GiB': torch.cuda.max_memory_allocated() / 1024**3,
               'reference_cache_unchanged': True, 'teacher_only_supervision': True})


def checked_cache():
    plan = checked()
    for p, h in json.loads((OUT / 'target_cache_identity.json').read_text())['sha256'].items():
        assert sha256(p) == h, p
    return plan


def load_record(meta, device='cpu'):
    d = previous_record(meta, device)
    d['target'] = torch.load(TARGET_CACHE / (meta['id'] + '.pt'), map_location='cpu', weights_only=False)
    return d


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['prepare', 'extract'])
    a = p.parse_args(); {'prepare': prepare, 'extract': extract}[a.command]()
