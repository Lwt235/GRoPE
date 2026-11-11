"""Proportional checks of scene separation, pairing, replay and independent scoring."""
import json

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_setup import ROOT, sha256
from experiments.gaussian_memory_coverage_data import OUT, OLD_OUT, SEEDS, VARIANTS, model_path, checked_cache, checked, load_record
from experiments.gaussian_memory_view_data import TOKEN_CACHE
from models.gaussian_memory_attention import geometry_inputs
from models.gaussian_memory_direct_reference import DirectReferenceAdapter
from models.gaussian_memory_view_token import ViewTokenAdapter


def normalize(a):
    return (a - a.mean(-1, keepdims=True)) / np.sqrt(a.var(-1, keepdims=True) + 1e-5)


def main():
    plan = checked(True); checked_cache(); torch.set_num_threads(4)
    lock = json.loads((OUT / 'evaluation_lock.json').read_text())
    assert lock['plan_sha256'] == sha256(OUT / 'plan.json')
    for path, digest in lock['models'].items():
        assert sha256(path) == digest, path
    training = [next(r for r in plan['catalog'] if r['id'] == rid) for rid in plan['training_ids']]
    places = {r['row']['scene'].split('/')[0] for r in training}
    assert len(places) == 12 and not places & set(plan['primary_places'])
    train_frame_paths = {f['rgb_path'] for r in training for f in r['row']['frames']}
    test_frame_paths = {f['rgb_path'] for r in plan['catalog'] if r['eval_split'].startswith('used_clip') for f in r['row']['frames']}
    assert not train_frame_paths & test_frame_paths
    for slot, meta in enumerate(training):
        rec = torch.load(meta['tokens'], map_location='cpu', weights_only=False)
        old = torch.load(TOKEN_CACHE / (plan['narrow_training_ids'][slot] + '.pt'), map_location='cpu', weights_only=False)
        keep = torch.tensor([r == 'train' for r in rec['roles']]); old_keep = torch.tensor([r == 'train' for r in old['roles']])
        assert keep.sum() == 24
        assert torch.equal(rec['query'][keep], old['query'][old_keep])
        assert torch.equal(rec['mask'][keep], old['mask'][old_keep])
        assert not set(rec['query'][keep].tolist()) & set(meta['row']['reference'])
        assert not set(rec['query'][keep].tolist()) & set(rec['query'][~keep].tolist())
    runs = []
    narrow_slot = {rid: i for i, rid in enumerate(plan['narrow_training_ids'])}
    for seed in SEEDS:
        shared = None
        for variant in VARIANTS:
            logs = {}
            for regime in ('narrow', 'broad'):
                directory = model_path(regime, variant, seed).parent
                log = [json.loads(s) for s in (directory / 'training.jsonl').read_text().splitlines()]
                assert len(log) == 2000 and [r['step'] for r in log] == list(range(1, 2001))
                assert all(np.isfinite([r['loss'], r['gradient_norm']]).all() for r in log)
                logs[regime] = [(narrow_slot[r['episode']] if regime == 'narrow' else r['slot'], r['sample_ids']) for r in log]
            assert logs['narrow'] == logs['broad']
            if shared is None:
                shared = logs['broad']
            else:
                assert shared == logs['broad']
            old = torch.load(model_path('narrow', variant, seed, True), map_location='cpu', weights_only=False)['model']
            new = torch.load(model_path('broad', variant, seed, True), map_location='cpu', weights_only=False)['model']
            assert all(torch.equal(old[k], new[k]) for k in old)
            assert torch.count_nonzero(new['output.weight']) == 0 and torch.count_nonzero(new['output.bias']) == 0
            status = json.loads((model_path('broad', variant, seed).parent / 'status.json').read_text())
            assert status['complete']
            runs.append(dict(variant=variant, seed=seed, **status))
    scores = json.loads((OUT / 'evaluation/scores.json').read_text())
    old = json.loads((OLD_OUT / 'evaluation/scores.json').read_text())
    def key(row):
        return row['episode'], row['variant'], row['seed'], row['condition']
    index = {key(r): r for r in old['rows']}
    selected = [r for r in scores['rows'] if r['regime'] == 'narrow' and r['split'].startswith('used_clip')]
    replay = max(abs(r[m] - index[key(r)][m]) for r in selected for m in ('hidden_mse', 'visible_mse', 'all_patch_mse'))
    assert replay < 1e-6
    accounting = max(abs(r['all_patch_mse'] - (70 * r['hidden_mse'] + 230 * r['visible_mse']) / 300) for r in scores['rows'])
    assert accounting < 1e-7
    independent = {}
    with torch.inference_mode():
        for rid in ('fit_00', 'transfer_08', 'fidelity_07'):
            meta = next(r for r in plan['catalog'] if r['id'] == rid)
            d = load_record(meta); raw, rec = d['raw'], d['rec']
            sample = rec['roles'].index('evaluation'); q = int(rec['query'][sample])
            x = rec['input_tokens'][sample:sample + 1]
            teacher = rec['teacher_tokens'][q].float().numpy()
            maps = torch.load(OUT / 'evaluation' / (rid + '_maps.pt'), map_location='cpu', weights_only=False)
            for regime in ('narrow', 'broad'):
                for variant in VARIANTS:
                    cls = ViewTokenAdapter if variant in ('local', 'center') else DirectReferenceAdapter
                    ckpt = torch.load(model_path(regime, variant, SEEDS[0]), map_location='cpu', weights_only=False)
                    net = cls(variant, ckpt['config']['center_variance']).eval(); net.load_state_dict(ckpt['model'])
                    memory = d['direct_memory'] if variant == 'direct' else d['memory']
                    g = geometry_inputs(memory, raw['c2w'][q:q + 1], raw['K'][q:q + 1])
                    prediction = net(x, g).numpy()[0]
                    error = ((normalize(prediction) - normalize(teacher)) ** 2).mean(-1)
                    diff = float(np.max(np.abs(error - maps['errors'][regime + '_' + variant][0].numpy())))
                    assert diff < 1e-5, (rid, regime, variant, diff)
                    independent[rid + '/' + regime + '/' + variant] = diff
    for meta in plan['catalog']:
        if not meta.get('new_geometry'):
            continue
        direct = torch.load(meta['direct'], map_location='cpu', weights_only=False)
        gaussian = torch.load(meta['memory'], map_location='cpu', weights_only=False)
        assert not direct['query_images_used'] and not direct['teacher_used'] and not gaussian['query_images_used']
        assert direct['reference_frames'] == [0, 4, 8, 12]
        assert direct['memory']['features'].shape == (1200, 2048)
        assert gaussian['memory']['features'].shape == (256, 32)
    result = {'complete': True, 'runs': runs, 'training_scenes': 12, 'primary_evaluation_scenes': 12,
              'train_test_place_and_frame_disjoint': True, 'all_initial_weights_paired': True,
              'all_training_masks_and_queries_paired': True, 'all_2000_sampling_sequences_paired': True,
              'narrow_primary_replay_max_difference': replay, 'all_patch_accounting_difference': accounting,
              'independent_numpy_max_differences': independent,
              'new_reference_only_boundary_checked': True, 'fixed_input_and_model_identities_checked': True,
              'training_seconds': sum(r['seconds'] for r in runs), 'training_peak_GiB': max(r['peak_GiB'] for r in runs),
              'auxiliary_source_sha256': {p: sha256(ROOT / p) for p in
                   ['experiments/gaussian_memory_coverage_report.py', 'experiments/gaussian_memory_coverage_checks.py',
                    'experiments/gaussian_memory_coverage_run.py']}}
    write_json(OUT / 'completion_checks.json', result)
    concise = {k: v for k, v in result.items() if k not in ('runs', 'auxiliary_source_sha256', 'independent_numpy_max_differences')}
    concise['independent_max'] = max(independent.values())
    print(json.dumps(concise), flush=True)


if __name__ == '__main__':
    main()
