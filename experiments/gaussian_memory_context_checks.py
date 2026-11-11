"""Bounded checks of teacher boundaries, matched fits and spatial controls."""
import json

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_setup import ROOT, sha256
from experiments.gaussian_memory_reference_encoding_data import OUT as PREVIOUS, REF_CACHE
from experiments.gaussian_memory_context_data import OUT, TARGET_CACHE, SEEDS, VARIANTS, model_path, initial_source, checked_cache, checked, load_record, target_queries
from models.gaussian_memory_attention import geometry_inputs
from models.gaussian_memory_context import ContextAdapter, permute_visible_content


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
    runs = []
    for seed in SEEDS:
        old_logs = [json.loads(s) for s in (model_path('spatial', seed, 'single').parent / 'training.jsonl').read_text().splitlines()]
        expected = [(r['episode'], r['slot'], r['sample_ids']) for r in old_logs]
        for variant in VARIANTS:
            directory = model_path(variant, seed).parent
            logs = [json.loads(s) for s in (directory / 'training.jsonl').read_text().splitlines()]
            assert len(logs) == 2000 and [r['step'] for r in logs] == list(range(1, 2001))
            assert all(np.isfinite([r['loss'], r['gradient_norm']]).all() for r in logs)
            assert [(r['episode'], r['slot'], r['sample_ids']) for r in logs] == expected
            old = torch.load(initial_source(variant, seed), map_location='cpu', weights_only=False)['model']
            new = torch.load(model_path(variant, seed, initial=True), map_location='cpu', weights_only=False)['model']
            assert old.keys() == new.keys() and all(torch.equal(old[k], new[k]) for k in old)
            assert torch.count_nonzero(new['output.weight']) == 0 and torch.count_nonzero(new['output.bias']) == 0
            status = json.loads((directory / 'status.json').read_text())
            assert status['complete']
            runs.append(dict(variant=variant, seed=seed, **status))
    target_count = 0
    for meta in plan['catalog']:
        target = torch.load(TARGET_CACHE / (meta['id'] + '.pt'), map_location='cpu', weights_only=False)
        rec = torch.load(meta['tokens'], map_location='cpu', weights_only=False)
        assert target['query_ids'] == target_queries(plan, meta, rec)
        assert not set(target['query_ids']) & {0, 4, 8, 12}
        assert target['teacher_order'] == ['query', 0, 4, 8, 12] and target['teacher_query_position'] == 0
        assert target['supervision_only'] and not target['other_query_images_used'] and not target['student_reference_modified']
        assert target['source_raw_sha256'] == sha256(meta['raw']) and target['source_tokens_sha256'] == sha256(meta['tokens'])
        assert target['student_reference_sha256'] == sha256(REF_CACHE / (meta['id'] + '.pt'))
        for name in ('context_targets', 'repeat_query_targets'):
            value = target[name]
            assert value.shape == (13, 300, 2048) and torch.isfinite(value).all()
            assert value[target['query_ids']].abs().sum() > 0
        if meta['id'] in plan['training_ids']:
            keep = torch.tensor([r == 'train' for r in rec['roles']])
            assert int(keep.sum()) == 24
            assert not set(rec['query'][keep].tolist()) & set(rec['query'][~keep].tolist())
        target_count += len(target['query_ids'])
        maps = torch.load(OUT / 'evaluation' / (meta['id'] + '_maps.pt'), map_location='cpu', weights_only=False)
        ps, valid = maps['permutations'], maps['valid_slots']
        ids = torch.arange(1200)
        for p, v in zip(ps, valid):
            assert torch.equal(p.sort().values, ids)
            assert torch.equal(p // 300, ids // 300)
            assert torch.equal(p[~v], ids[~v]) and torch.equal(p[v].sort().values, ids[v])
    assert target_count == 360
    scores = json.loads((OUT / 'evaluation/scores.json').read_text())
    assert len(scores['rows']) == 46 * 28 and len(scores['full_query']) == 46 * 15
    old_scores = json.loads((PREVIOUS / 'evaluation/scores.json').read_text())
    def key(row):
        return row['episode'], row['variant'], row['seed']
    index = {key(r): r for r in old_scores['rows'] if r['condition'] == 'normal' and r['variant'] in ('s1', 'wide')}
    clean_index = {key(r): r for r in old_scores['preservation'] if r['variant'] in ('s1', 'wide')}
    selected = [r for r in scores['rows'] if r['regime'] == 'single']
    replay = max(abs(r['original_s1_all_mse'] - index[key(dict(r, variant='s1' if r['variant'] == 'spatial' else 'wide'))]['all_patch_mse']) for r in selected)
    clean_replay = max(abs(r['full_query_s1_change_mse'] - clean_index[key(dict(r, variant='s1' if r['variant'] == 'spatial' else 'wide'))]['clean_input_mse'])
                       for r in scores['full_query'] if r['regime'] == 'single')
    assert replay < 1e-6 and clean_replay < 1e-6
    accounting = max(abs(r['all_patch_mse'] - (70 * r['hidden_mse'] + 230 * r['visible_mse']) / 300) for r in scores['rows'])
    assert accounting < 1e-7
    invariance = scores['content_permutation_invariance']; assert len(invariance) == 138
    normalized_max = max(r['max_normalized_difference'] for r in invariance)
    assert normalized_max < 2e-5, normalized_max
    raw_max = max(r['max_output_difference'] for r in invariance)
    independent = {}
    with torch.inference_mode():
        for rid in ('fit_00', 'transfer_08', 'fidelity_07'):
            meta = next(r for r in plan['catalog'] if r['id'] == rid)
            d = load_record(meta); raw, rec = d['raw'], d['rec']
            sample = rec['roles'].index('evaluation'); q = int(rec['query'][sample])
            x = rec['input_tokens'][sample:sample + 1]
            teacher = d['target']['context_targets'][q].float().numpy()
            maps = torch.load(OUT / 'evaluation' / (rid + '_maps.pt'), map_location='cpu', weights_only=False)
            g = geometry_inputs(d['s1'], raw['c2w'][q:q + 1], raw['K'][q:q + 1])
            shuffled, _ = permute_visible_content(g, rid, [q])
            for variant in VARIANTS:
                ckpt = torch.load(model_path(variant, SEEDS[0]), map_location='cpu', weights_only=False)
                net = ContextAdapter(variant, ckpt['config']['center_variance']).eval(); net.load_state_dict(ckpt['model'])
                for condition in (('normal',) if variant == 'wide' else ('normal', 'removed', 'shuffled')):
                    prediction = net(x, shuffled if condition == 'shuffled' else g, condition).numpy()[0]
                    error = ((normalize(prediction) - normalize(teacher)) ** 2).mean(-1)
                    name = 'context_' + variant + ('' if condition == 'normal' else '_' + condition)
                    difference = float(np.max(np.abs(error - maps['errors'][name][0].numpy())))
                    assert difference < 1e-5, (rid, variant, condition, difference)
                    independent[rid + '/' + name] = difference
    diagnostic_decomposition = max(abs(r['real_reference_vs_repeat_query_mse'] - r['spatially_centered_reference_change_mse'] - r['global_reference_change_mse']) for r in scores['teacher_diagnostics'])
    assert diagnostic_decomposition < 1e-6
    result = {'complete': True, 'runs': runs, 'training_scenes': 12, 'primary_evaluation_scenes': 12,
              'train_primary_place_and_frame_disjoint': True, 'identical_query_input_and_mask_cache_paths': True,
              'all_initial_weights_match_previous': True, 'all_2000_sampling_sequences_paired': True,
              'reference_cache_immutable_and_teacher_not_used_by_student': True,
              'teacher_query_first_and_one_query_per_sequence_source_reviewed': True,
              'actual_unique_queries_targeted': target_count, 'query_valid_within_view_permutations_checked': True,
              'previous_original_target_replay_max_difference': replay, 'previous_clean_replay_max_difference': clean_replay,
              'content_shuffle_normalized_output_max_difference': normalized_max,
              'content_shuffle_raw_output_max_difference': raw_max, 'all_patch_accounting_difference': accounting,
              'teacher_change_decomposition_difference': diagnostic_decomposition,
              'independent_numpy_max_differences': independent, 'fixed_input_and_model_identities_checked': True,
              'new_training_seconds': sum(r['seconds'] for r in runs),
              'new_training_peak_GiB': max(r['peak_GiB'] for r in runs),
              'auxiliary_source_sha256': {p: sha256(ROOT / p) for p in
                   ['experiments/gaussian_memory_context_report.py', 'experiments/gaussian_memory_context_checks.py',
                    'experiments/gaussian_memory_context_run.py', 'experiments/gaussian_memory_coverage_report.py']}}
    write_json(OUT / 'completion_checks.json', result)
    concise = {k: v for k, v in result.items() if k not in ('runs', 'auxiliary_source_sha256', 'independent_numpy_max_differences')}
    concise['independent_max'] = max(independent.values())
    print(json.dumps(concise), flush=True)


if __name__ == '__main__':
    main()
