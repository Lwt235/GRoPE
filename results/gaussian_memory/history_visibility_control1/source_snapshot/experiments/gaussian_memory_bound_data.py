"""Reuse the verified causal inputs; cache only the new slot-wise priors."""
import argparse
import json
import time

import numpy as np
import torch

from experiments.gaussian_memory_setup import ROOT, DATA, sha256
from experiments.gaussian_memory_incremental_data import (
    CACHE as BASE_CACHE, OUT as BASE_OUT, SEEDS, QUERIES, write_json, save_tensor, device_tree)
from models.gaussian_memory_bound_read import projected_slot_weights, attention_prior

OUT = ROOT / 'results/gaussian_memory/bound_read1'
CACHE = DATA / 'bound_read_v1'
KINDS = ('local', 'content', 'coverage', 'bound')


def prepare():
    previous = json.loads((BASE_OUT / 'plan.json').read_text())
    assert json.loads((BASE_OUT / 'phase_completion.json').read_text())['experiment_complete']
    stats = json.loads((BASE_OUT / 'training_statistics.json').read_text())
    plan = {k: previous[k] for k in ('catalog', 'training_ids', 'primary_places', 'input_sha256', 'backbone_identity')}
    plan.update(schema=1, experiment='bound_read1', seeds=list(SEEDS), kinds=list(KINDS),
                steps=2000, effective_batch=2, microbatch=1, lr=3e-4, weight_decay=.01, gradient_clip=5.,
                parameters=137472, memory_slots=256, content_dim=32, insert_global_index=11,
                parent_plan_sha256=sha256(BASE_OUT / 'plan.json'),
                parent_completion_sha256=sha256(BASE_OUT / 'phase_completion.json'),
                document_sha256=sha256(ROOT / 'docs/gaussian_memory_bound_read_plan.md'),
                shared_cache=str(BASE_CACHE), prior_cache=str(CACHE),
                reference=[0, 1, 2, 3], write_frames=[4, 6, 8, 10], query_frames=list(QUERIES),
                all_data_previously_used=True, confirmation_claim=False, known_camera=True,
                initial_memory_fixed=True, query_frames_never_written=True,
                center_variance=stats['center_variance'],
                center_statistics_sha256=sha256(BASE_OUT / 'training_statistics.json'),
                scale_rule=previous['scale_rule'], opacity=.7, projection_sigma_cutoff=3.,
                projection_blur_patch_variance=.1, output_bias=False,
                primary_comparisons=['bound_vs_coverage', 'bound_vs_local', 'bound_vs_content', 'coverage_vs_content'],
                center_trigger='bound mean lower than coverage and at least two of three paired seeds positive',
                error_probe_trigger='bound mean lower than coverage and local; at least two paired seeds positive for each')
    target = OUT / 'plan.json'
    if target.exists():
        assert json.loads(target.read_text()) == plan, 'Scientific plan differs; preserve and record a revision before changing it.'
    else:
        write_json(target, plan)
    return plan


def load_plan():
    return json.loads((OUT / 'plan.json').read_text())


def prior_path(rid, query, center=False):
    return CACHE / rid / f'q{query}{"_center" if center else ""}.pt'


@torch.no_grad()
def prepare_priors(center=False):
    plan = prepare(); start = time.monotonic(); done = []; worst_mass_error = 0.
    source_hash = sha256(ROOT / 'models/gaussian_memory_bound_read.py')
    for meta in plan['catalog']:
        rid = meta['id']; folder = CACHE / rid; folder.mkdir(parents=True, exist_ok=True)
        source = json.loads((BASE_CACHE / rid / 'identity.json').read_text())
        identity_path = folder / ('center_identity.json' if center else 'identity.json')
        expected = {'base_identity_sha256': sha256(BASE_CACHE / rid / 'identity.json'),
                    'initial_sha256': source['sha256']['initial.pt'], 'model_source_sha256': source_hash,
                    'center_variance': plan['center_variance'] if center else None}
        if identity_path.exists():
            old = json.loads(identity_path.read_text())
            assert all(old[k] == v for k, v in expected.items())
            assert all(sha256(folder / k) == v for k, v in old['sha256'].items())
            done.append(rid); worst_mass_error = max(worst_mass_error, old['max_mass_error']); continue
        assert sha256(BASE_CACHE / rid / 'initial.pt') == expected['initial_sha256']
        initial = torch.load(BASE_CACHE / rid / 'initial.pt', map_location='cpu', weights_only=False)
        hashes = {}; error = 0.
        for query in QUERIES:
            name = f'q{query}_c0.pt'
            assert sha256(BASE_CACHE / rid / name) == source['sha256'][name]
            sample = torch.load(BASE_CACHE / rid / name, map_location='cpu', weights_only=False)
            projected = projected_slot_weights(initial['memory'], sample['c2w'], sample['K'],
                                               plan['center_variance'] if center else None)
            mass_error = (projected['coverage'] + projected['transmittance'] - 1).abs().max().item()
            assert mass_error < 1e-5 and torch.isfinite(projected['weights']).all()
            error = max(error, mass_error)
            target = prior_path(rid, query, center)
            save_tensor(target, projected); hashes[target.name] = sha256(target)
        write_json(identity_path, {**expected, 'sha256': hashes, 'max_mass_error': error})
        done.append(rid); worst_mass_error = max(worst_mass_error, error)
        if len(done) % 10 == 0:
            print(json.dumps({'prior_cache_clips': len(done), 'center': center, 'seconds': time.monotonic() - start}), flush=True)
    write_json(OUT / f'priors_{"center" if center else "gaussian"}_complete.json',
               {'complete': True, 'clips': done, 'max_mass_error': worst_mass_error,
                'seconds': time.monotonic() - start, 'model_source_sha256': source_hash})


def load_priors(ids, center=False, device='cuda'):
    result = {}
    for rid in ids:
        identity = json.loads((CACHE / rid / ('center_identity.json' if center else 'identity.json')).read_text())
        assert identity['model_source_sha256'] == sha256(ROOT / 'models/gaussian_memory_bound_read.py')
        for query in QUERIES:
            path = prior_path(rid, query, center)
            assert sha256(path) == identity['sha256'][path.name]
            result[(rid, query)] = device_tree(torch.load(path, map_location='cpu', weights_only=False), device)
    return result


def inputs_for(sample, initial, kind, priors):
    content = sample['local_content'] if kind == 'local' else initial['memory']['features']
    projected = priors[(sample['id'], sample['query'])]
    return content, attention_prior(projected, kind, len(content))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--center', action='store_true'); args = p.parse_args()
    torch.set_num_threads(4); prepare_priors(args.center)
