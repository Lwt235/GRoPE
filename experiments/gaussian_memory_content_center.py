"""One missing matched control; reuse the exact shared-writer training/evaluator."""
import argparse
import json
import time
import traceback

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_setup import ROOT, sha256
from experiments import gaussian_memory_joint_write as joint
from experiments import gaussian_memory_summary_data as previous
from models.gaussian_memory_compact_writer import CompactGaussianWriter

OLD = joint.OUT
RECENT = ROOT / 'results/gaussian_memory/write_confirmation1'
OUT = ROOT / 'results/gaussian_memory/content_center1'
DOC = ROOT / 'docs/gaussian_memory_content_center_plan.md'
SEEDS = joint.SEEDS
NEW = ('content_center',)
KINDS = NEW + joint.KINDS
WRITERS = NEW + joint.NEW
BASE_LOAD = previous.load_record


def load_record(meta):
    # The previous loader uses 'confirmation' to select a cache layout, not
    # scientific freshness. Keep that storage adapter separate from our label.
    storage_meta = dict(meta, eval_split='confirmation') if meta['id'].startswith('fresh_') else meta
    return BASE_LOAD(storage_meta)


def model_path(kind, seed, initial=False):
    filename = 'initial.pt' if initial else 'step_2000.pt'
    if kind in WRITERS:
        return (OUT if kind in NEW else OLD) / f'{kind}_{seed}' / filename
    return previous.path(kind, seed, initial)


def make_models(kind, plan):
    if kind in WRITERS:
        return previous.model(kind.split('_')[-1], plan), CompactGaussianWriter(kind.startswith('full'))
    return previous.model(kind, plan), None


def checked(locked=False):
    plan = json.loads((OUT / 'plan.json').read_text())
    assert sha256(DOC) == plan['document_sha256']
    for name, digest in plan['source_sha256'].items():
        assert sha256(ROOT / name) == digest, name
    for name, digest in plan['old_models_sha256'].items():
        assert sha256(name) == digest, name
    for name, digest in plan['reused_artifacts_sha256'].items():
        assert sha256(name) == digest, name
    if locked:
        lock = json.loads((OUT / 'evaluation_lock.json').read_text())
        assert sha256(OUT / 'plan.json') == lock['plan_sha256']
        for name, digest in lock['models'].items():
            assert sha256(name) == digest, name
    return plan


def prepare():
    old = joint.checked(True)
    recent = json.loads((RECENT / 'cache_identity.json').read_text())
    assert json.loads((RECENT / 'status.json').read_text())['complete']
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'plan.json').exists()
    catalog = old['catalog'] + [{**m, 'eval_split': 'used_writer19'} for m in recent['catalog']]
    assert len(catalog) == 85 and len({m['id'] for m in catalog}) == 85
    identities = {**old['input_sha256'], **recent['sha256']}
    reused = {}
    for directory in (OLD, RECENT):
        for filename in ('plan.json', 'evaluation_lock.json', 'evaluation/scores.json'):
            path = directory / filename
            reused[str(path)] = sha256(path)
    sources = dict(old['source_sha256'])
    for name in ('experiments/gaussian_memory_content_center.py', 'experiments/gaussian_memory_joint_write.py',
                 'models/gaussian_memory_compact_writer.py'):
        sources[name] = sha256(ROOT / name)
    plan = {**{k: old[k] for k in ('training_ids', 'primary_places', 'steps', 'batch', 'lr',
                                  'weight_decay', 'center_variance', 'gradient_clip')},
            'catalog': catalog, 'seeds': SEEDS, 'new_kinds': NEW, 'all_kinds': KINDS,
            'input_sha256': identities, 'source_sha256': sources,
            'old_models_sha256': {str(model_path(k, s)): sha256(model_path(k, s))
                                  for k in KINDS if k not in NEW for s in SEEDS},
            'reused_artifacts_sha256': reused, 'document_sha256': sha256(DOC),
            'confirmation_claim': False, 'primary': '51 used non-training places, clips averaged within place',
            'only_change': 'Content Gaussian Mahalanobis position distance to existing fixed-variance center distance',
            'figure_cases': ['fit_00', 'transfer_00', 'confirm_00', 'confirm_15', 'fresh_00', 'fresh_19']}
    for seed in SEEDS:
        initial_path = model_path('content_gaussian', seed, True)
        initial = torch.load(initial_path, map_location='cpu', weights_only=False)
        reader, writer = make_models('content_center', plan)
        reader.load_state_dict(initial['model']); writer.load_state_dict(initial['writer'])
        config = dict(initial['config'], kind='content_center', paired_initial=str(initial_path))
        assert not writer.geometry and reader.kernel == 'center'
        dest = model_path('content_center', seed, True); dest.parent.mkdir()
        torch.save({'model': reader.state_dict(), 'writer': writer.state_dict(), 'config': config}, dest)
        identities[str(dest)] = sha256(dest); identities[str(initial_path)] = sha256(initial_path)
    write_json(OUT / 'plan.json', plan)
    write_json(OUT / 'status.json', {'complete': False, 'pending': ['train', 'evaluate', 'checks', 'analysis', 'review', 'report']})
    print(json.dumps({'prepared': True, 'fits': 3, 'reused_models': 24, 'clips': 85,
                      'parameters': config['reader_parameters'] + config['writer_parameters'], 'confirmation_claim': False}), flush=True)


def configure_joint(evaluation=False):
    # These process-local adapters leave every historical source and result intact.
    joint.OUT = OUT; joint.NEW = NEW; joint.KINDS = NEW if evaluation else KINDS
    joint.checked = checked; joint.model_path = model_path; joint.make_models = make_models
    joint.previous.load_record = load_record


def train():
    configure_joint()
    joint.train()


def old_maps_path(rid):
    return (RECENT if rid.startswith('fresh_') else OLD) / 'evaluation' / f'{rid}_maps.pt'


def merged_maps(rid):
    old = torch.load(old_maps_path(rid), map_location='cpu', weights_only=False)
    new = torch.load(OUT / 'evaluation' / f'{rid}_maps.pt', map_location='cpu', weights_only=False)
    for key in ('query', 'target', 'valid', 'mask', 'full_target', 'full_valid'):
        assert torch.equal(old[key].cpu(), new[key].cpu()), (rid, key)
    return {**new, 'predictions': {**old['predictions'], **new['predictions']},
            'full_predictions': {**old['full_predictions'], **new['full_predictions']}}


def evaluate():
    configure_joint(evaluation=True)
    joint.evaluate()
    score_path = OUT / 'evaluation/scores.json'
    new = json.loads(score_path.read_text())
    score_path.rename(OUT / 'evaluation/new_scores.json')
    combined = {k: list(new[k]) for k in ('rows', 'full_query', 'updates')}
    combined.update({k: new[k] for k in ('maps_sha256', 'memory_sha256', 'counts')})
    combined['reused_sources'] = []
    for directory in (OLD, RECENT):
        source = directory / 'evaluation/scores.json'; data = json.loads(source.read_text())
        combined['reused_sources'].append({'path': str(source), 'sha256': sha256(source)})
        for field in ('rows', 'full_query', 'updates'):
            combined[field].extend([dict(r, split='used_writer19') if directory == RECENT else r for r in data[field]])
    combined['confirmation_claim'] = False
    combined['reused_old_predictions_exact'] = True
    write_json(score_path, combined)
    print(json.dumps({'new_models_evaluated': 3, 'reused_models': 24, 'clips': len(new['counts']),
                      'new_saved_memories': len(new['memory_sha256']), 'confirmation_claim': False}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=('prepare', 'train', 'evaluate'))
    args = parser.parse_args(); start = time.monotonic()
    try:
        globals()[args.stage]()
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / f'failure_{args.stage}_{int(time.time())}.json',
                   {'traceback': traceback.format_exc(), 'stage': args.stage, 'scientific_result': False})
        raise
    write_json(OUT / f'{args.stage}_complete.json', {'complete': True, 'seconds': time.monotonic() - start})
