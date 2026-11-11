"""Bounded checks of independent reference construction, matched fits and scoring."""
import json

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_setup import ROOT, sha256
from experiments.gaussian_memory_coverage_data import OUT as COVERAGE
from experiments.gaussian_memory_reference_encoding_data import OUT, REF_CACHE, SEEDS, VARIANTS, model_path, checked_cache, checked, load_record, spatial_permutation
from models.gaussian_memory_attention import geometry_inputs
from models.gaussian_memory_direct_reference import DirectReferenceAdapter


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
    train_paths = {f['rgb_path'] for r in training for f in r['row']['frames']}
    primary_paths = {f['rgb_path'] for r in plan['catalog'] if r['eval_split'].startswith('used_clip') for f in r['row']['frames']}
    assert not train_paths & primary_paths
    for meta in training:
        rec = torch.load(meta['tokens'], map_location='cpu', weights_only=False)
        keep = torch.tensor([role == 'train' for role in rec['roles']])
        assert keep.sum() == 24
        train_queries = set(rec['query'][keep].tolist())
        assert not train_queries & set(meta['row']['reference'])
        assert not train_queries & set(rec['query'][~keep].tolist())
    runs = []
    for seed in SEEDS:
        paired = None
        for variant in VARIANTS:
            directory = model_path(variant, seed).parent
            log = [json.loads(s) for s in (directory / 'training.jsonl').read_text().splitlines()]
            assert len(log) == 2000 and [r['step'] for r in log] == list(range(1, 2001))
            assert all(np.isfinite([r['loss'], r['gradient_norm']]).all() for r in log)
            sequence = [(r['episode'], r['slot'], r['sample_ids']) for r in log]
            if paired is None:
                paired = sequence
            else:
                assert sequence == paired
        old = torch.load(model_path('s4', seed, True), map_location='cpu', weights_only=False)['model']
        new = torch.load(model_path('s1', seed, True), map_location='cpu', weights_only=False)['model']
        assert old.keys() == new.keys() and all(torch.equal(old[k], new[k]) for k in old)
        assert torch.count_nonzero(new['output.weight']) == 0 and torch.count_nonzero(new['output.bias']) == 0
        status = json.loads((model_path('s1', seed).parent / 'status.json').read_text())
        assert status['complete']
        runs.append(dict(seed=seed, **status))
    geometry_checked, moved_slots = [], []
    for meta in plan['catalog']:
        old = torch.load(meta['direct'], map_location='cpu', weights_only=False)['memory']
        saved = torch.load(REF_CACHE / (meta['id'] + '.pt'), map_location='cpu', weights_only=False)
        new = saved['memory']
        assert not saved['query_images_used'] and not saved['teacher_used']
        assert saved['reference_frames'] == [0, 4, 8, 12] and saved['reference_sequence_length'] == 1
        assert saved['source_raw_sha256'] == sha256(meta['raw'])
        assert saved['source_geometry_sha256'] == sha256(meta['direct'])
        assert new['features'].shape == (1200, 2048)
        assert all(torch.isfinite(v).all() for v in new.values())
        assert all(torch.equal(old[k], new[k]) for k in ('means', 'covars', 'opacities'))
        permutation = spatial_permutation(new, meta['id']).cpu(); ids = torch.arange(1200)
        valid = new['opacities'] > 0
        assert torch.equal(permutation, saved['permutation'])
        assert torch.equal(permutation.sort().values, ids)
        assert torch.equal(permutation // 300, ids // 300)
        assert torch.equal(permutation[~valid], ids[~valid])
        assert torch.equal(permutation[valid].sort().values, ids[valid])
        geometry_checked.append(meta['id']); moved_slots.append(int((permutation != ids).sum()))
    scores = json.loads((OUT / 'evaluation/scores.json').read_text())
    assert len(scores['rows']) == 46 * 22 and len(scores['preservation']) == 46 * 9
    old_scores = json.loads((COVERAGE / 'evaluation/scores.json').read_text())
    def key(row):
        return row['episode'], row['variant'], row['seed'], row.get('condition', 'normal')
    index = {key(r): r for r in old_scores['rows'] if r['regime'] == 'broad'}
    selected = [r for r in scores['rows'] if r['variant'] in ('wide', 's4') and r['condition'] in ('normal', 'removed')]
    replay = max(abs(r[m] - index[key(dict(r, variant='direct' if r['variant'] == 's4' else 'wide'))][m])
                 for r in selected for m in ('hidden_mse', 'visible_mse', 'all_patch_mse'))
    clean_index = {key(r): r for r in old_scores['preservation'] if r['regime'] == 'broad'}
    clean_replay = max(abs(r['clean_input_mse'] - clean_index[key(dict(r, variant='direct' if r['variant'] == 's4' else 'wide'))]['clean_input_mse'])
                       for r in scores['preservation'] if r['variant'] in ('wide', 's4'))
    assert replay < 1e-6 and clean_replay < 1e-6
    accounting = max(abs(r['all_patch_mse'] - (70 * r['hidden_mse'] + 230 * r['visible_mse']) / 300) for r in scores['rows'])
    assert accounting < 1e-7
    independent = {}
    with torch.inference_mode():
        for rid in ('fit_00', 'transfer_08', 'fidelity_07'):
            meta = next(r for r in plan['catalog'] if r['id'] == rid)
            d = load_record(meta); raw, rec = d['raw'], d['rec']
            sample = rec['roles'].index('evaluation'); q = int(rec['query'][sample])
            x = rec['input_tokens'][sample:sample + 1]; teacher = rec['teacher_tokens'][q].float().numpy()
            maps = torch.load(OUT / 'evaluation' / (rid + '_maps.pt'), map_location='cpu', weights_only=False)
            for variant in VARIANTS:
                ckpt = torch.load(model_path(variant, SEEDS[0]), map_location='cpu', weights_only=False)
                net = DirectReferenceAdapter('wide' if variant == 'wide' else 'direct', ckpt['config']['center_variance']).eval()
                net.load_state_dict(ckpt['model'])
                memory = d['s4'] if variant == 'wide' else d[variant]
                for condition in (('normal',) if variant == 'wide' else ('normal', 'removed', 'shuffled')):
                    m = memory
                    if condition == 'shuffled':
                        m = dict(memory, features=memory['features'][spatial_permutation(memory, rid)])
                    g = geometry_inputs(m, raw['c2w'][q:q + 1], raw['K'][q:q + 1])
                    prediction = net(x, g, condition).numpy()[0]
                    error = ((normalize(prediction) - normalize(teacher)) ** 2).mean(-1)
                    name = variant if condition == 'normal' else variant + '_' + condition
                    difference = float(np.max(np.abs(error - maps['errors'][name][0].numpy())))
                    assert difference < 1e-5, (rid, variant, condition, difference)
                    independent[rid + '/' + name] = difference
    cache = json.loads((OUT / 'reference_cache_identity.json').read_text())
    ratios = [r['s1_vs_existing_reference_cache_mse'] / r['s1_vs_s4_feature_mse'] for r in cache['records']]
    result = {'complete': True, 'runs': runs, 'training_scenes': 12, 'primary_evaluation_scenes': 12,
              'train_primary_place_and_frame_disjoint': True, 'identical_query_target_mask_cache_paths': True,
              'all_initial_s1_s4_weights_equal': True, 'all_2000_sampling_sequences_paired': True,
              'reference_only_builder_boundary_reviewed': True, 'geometry_equal_episodes': geometry_checked,
              'within_view_valid_only_permutations_checked': True, 'moved_slots_range': [min(moved_slots), max(moved_slots)],
              'wide_s4_score_replay_max_difference': replay, 'wide_s4_clean_replay_max_difference': clean_replay,
              'all_patch_accounting_difference': accounting, 'independent_numpy_max_differences': independent,
              'reference_cache_mse_range': [min(r['s1_vs_existing_reference_cache_mse'] for r in cache['records']),
                                            max(r['s1_vs_existing_reference_cache_mse'] for r in cache['records'])],
              'reference_cache_mse_to_s1_s4_difference_ratio_range': [min(ratios), max(ratios)],
              'fixed_input_and_model_identities_checked': True,
              'new_training_seconds': sum(r['seconds'] for r in runs),
              'new_training_peak_GiB': max(r['peak_GiB'] for r in runs),
              'auxiliary_source_sha256': {p: sha256(ROOT / p) for p in
                   ['experiments/gaussian_memory_reference_encoding_report.py', 'experiments/gaussian_memory_reference_encoding_checks.py',
                    'experiments/gaussian_memory_reference_encoding_run.py', 'experiments/gaussian_memory_coverage_report.py']}}
    write_json(OUT / 'completion_checks.json', result)
    concise = {k: v for k, v in result.items() if k not in ('runs', 'auxiliary_source_sha256', 'independent_numpy_max_differences', 'geometry_equal_episodes')}
    concise['independent_max'] = max(independent.values())
    print(json.dumps(concise), flush=True)


if __name__ == '__main__':
    main()
