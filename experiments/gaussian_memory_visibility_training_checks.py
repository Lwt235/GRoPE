"""Checks of causal construction, matched fitting, and independent NumPy scores."""
import argparse
import json
from collections import Counter

import numpy as np
import torch

from experiments.gaussian_memory_visibility_training_data import (
    ROOT, OUT, CACHE, BASE_CACHE, EVAL_CACHE, PARENT, EARLY, LATE, QUERIES, KINDS,
    SEEDS, POLICIES, load_plan, verify_sources, sha256, write_json)
from experiments.gaussian_memory_visibility_checks import cloud_numpy, index_numpy
from experiments.gaussian_memory_incremental_data import sensor_target
from experiments.gaussian_memory_incremental_train import schedule
from experiments.gaussian_memory_view_data import rectangle


def input_checks(plan):
    verify_sources(plan); psha = sha256(OUT / 'plan.json')
    assert not {m['place'] for m in plan['training_catalog']} & {m['place'] for m in plan['catalog']}
    for p, h in plan['input_sha256'].items(): assert sha256(p) == h, p
    records = []; states_count = 0; replay_count = 0; windows = 0
    for meta in plan['training_catalog']:
        rid = meta['id']; rec = json.loads((OUT / 'inputs' / f'{rid}.json').read_text())
        assert rec['complete'] and rec['plan_sha256'] == psha
        for n, h in rec['files'].items(): assert sha256(CACHE / rid / n) == h
        row = meta['row']; cameras = np.asarray(row['c2w_metric'], float); K = np.asarray(row['K'], float)
        early = []
        for f in EARLY:
            d, v = sensor_target(row['frames'][f])
            early.append(cloud_numpy(d.numpy().astype(float), v.numpy(), cameras[f], K[f]))
        early = np.concatenate(early)
        for q in QUERIES:
            roi_patches = rectangle(rid, q, 0).numpy()
            roi = roi_patches.reshape(15, 20).repeat(14, 0).repeat(14, 1)
            keep, uv, _ = index_numpy(early, cameras[q], K[q])
            selected = early[np.flatnonzero(keep)[roi[uv[:, 1], uv[:, 0]]]]
            masks = np.zeros((12, 300), bool)
            for f in list(EARLY) + [f for f in LATE if f < q]:
                _, uv, _ = index_numpy(selected, cameras[f], K[f])
                masks[f, uv[:, 1] // 14 * 20 + uv[:, 0] // 14] = True
            masks[q] = roi_patches
            saved = torch.load(CACHE / rid / f'q{q}_masks.pt', map_location='cpu', weights_only=False)
            mismatch = int(np.count_nonzero(saved['patch_masks'].numpy() != masks))
            assert mismatch == 0 and saved['selected_early_points'] == len(selected), (rid, q, mismatch)
            states = torch.load(CACHE / rid / f'q{q}_states.pt', map_location='cpu', weights_only=False)
            assert set(states) == {'full/recent', 'full/history', 'persistent/recent', 'persistent/history'}
            for key, m in states.items():
                used = m['sources'].any(0).nonzero().flatten().tolist()
                allowed = [f for f in LATE if f < q] + (list(EARLY) if key.endswith('/history') else [])
                assert set(used).issubset(allowed) and all(f < q for f in used)
                assert m['features'].shape == (256, 32)
                assert all(torch.isfinite(m[k]).all() for k in ('means', 'covars', 'features'))
                assert torch.linalg.eigvalsh(m['covars']).min() > 0
                states_count += 1
            reference = torch.load(BASE_CACHE / rid / f'q{q}_c0.pt', map_location='cpu', weights_only=False)
            for c in range(3):
                s = torch.load(CACHE / rid / f'q{q}_c{c}.pt', map_location='cpu', weights_only=False)
                assert s['metric_scale'] == reference['metric_scale'] and s['input_frames'] == reference['input_frames']
                for key in ('target', 'valid', 'c2w', 'K'): assert torch.equal(s[key], reference[key]), (rid, q, c, key)
                assert np.array_equal(s['hidden'].numpy(), roi)
                if c < 2:
                    original = torch.load(BASE_CACHE / rid / f'q{q}_c{c}.pt', map_location='cpu', weights_only=False)
                    for key in ('prefix', 'early4', 'local_content', 'query_depth_native', 'rgb'):
                        assert torch.equal(s[key], original[key]), (rid, q, c, key)
            records.append({'id': rid, 'query': q, 'mask_mismatch_patches': mismatch, 'selected_points': len(selected)})
        windows += rec['new_backbone_inferences']; replay_count += len(rec['replays'])
    assert states_count == 192 and replay_count == 4 and windows == 172
    result = {'complete': True, 'plan_sha256': psha, 'states_checked': states_count,
              'independent_masks': records, 'full_and_query_only_replays': replay_count,
              'new_backbone_windows': windows, 'training_and_evaluation_places_disjoint': True,
              'query_targets_only_used_for_training_loss_and_scoring': True}
    write_json(OUT / 'input_checks.json', result)
    return result


def fitting_checks(plan):
    reports = []; examples = [[rid, q, c, p] for rid in plan['training_ids'] for q in QUERIES for c in range(3) for p in POLICIES]
    for seed in SEEDS:
        inits = {}; rows = {}; sequence = schedule(seed, len(examples), 2000)
        expected = [[examples[int(i)] for i in step] for step in sequence]
        for kind in KINDS:
            folder = OUT / f'{kind}_{seed}'
            status = json.loads((folder / 'status.json').read_text())
            assert status['complete'] and status['steps'] == 2000 and not status['pilot']
            assert status['checkpoint_sha256'] == sha256(folder / 'step_2000.pt')
            inits[kind] = torch.load(folder / 'initial.pt', map_location='cpu', weights_only=False)
            rows[kind] = [json.loads(l) for l in (folder / 'training.jsonl').read_text().splitlines()]
            assert len(rows[kind]) == 2000
            assert [r['step'] for r in rows[kind]] == list(range(1, 2001))
            assert [r['samples'] for r in rows[kind]] == expected
            assert all(np.isfinite([r['loss'], r['gradient_norm']]).all() for r in rows[kind])
            assert rows[kind][1]['gradient_by_parameter']['q.weight'] > 0
            assert rows[kind][-1]['gradient_by_parameter']['out.weight'] > 0
        for kind in KINDS[1:]:
            assert inits[kind]['schedule_sha256'] == inits['content']['schedule_sha256']
            assert all(torch.equal(t, inits[kind]['model'][n]) for n, t in inits['content']['model'].items())
        counts = Counter(f'{sample[2]}/{sample[3]}' for step in expected for sample in step)
        reports.append({'seed': seed, 'paired_initializations_exact': True, 'paired_steps_exact': 2000,
                        'samples_per_condition_policy': dict(counts)})
    return reports


def score_checks(plan):
    maximum = {}; count = 0; parent_count = 0
    for meta in plan['catalog']:
        rid = meta['id']; rec = json.loads((OUT / 'evaluation' / f'{rid}.json').read_text())
        assert rec['complete'] and sha256(rec['prediction_path']) == rec['prediction_sha256']
        assert rec['source_sha256'] == sha256(ROOT / 'experiments/gaussian_memory_visibility_training_evaluate.py')
        export = torch.load(rec['prediction_path'], map_location='cpu', weights_only=False)
        samples = {c: torch.load(EVAL_CACHE / rid / f'q11_c{c}.pt', map_location='cpu', weights_only=False) for c in range(3)}
        keys = {(c, f, s, p) for c in range(3) for f in ('new_content', 'new_soft', 'new_center') for s in SEEDS for p in POLICIES}
        assert len(rec['rows']) == 54 and len(export['keys']) == 54 and set(map(tuple, export['keys'])) == keys
        for i, r in enumerate(rec['rows']):
            assert export['keys'][i] == [r['condition'], r['family'], r['seed'], r['policy']]
            pred = export['prediction_native'][i].numpy().astype(float); s = samples[r['condition']]
            assert np.isfinite(pred).all() and (pred > 0).all()
            t = s['target'].numpy().astype(float); valid = s['valid'].numpy(); roi = s['hidden'].numpy(); scale = s['metric_scale']
            residual = np.log(np.maximum(pred, 1e-6)) + np.log(scale) - np.log(t)
            x = residual[valid]; error = pred[valid] * scale - t[valid]
            calc = {'log_mse': float(np.mean(x*x)), 'centered_log_mse': float(np.mean((x-x.mean())**2)),
                'bias_squared': float(x.mean()**2), 'absrel': float(np.mean(np.abs(error) / t[valid])),
                'rmse_m': float(np.sqrt(np.mean(error**2))), 'hidden_log_mse': float(np.mean(residual[valid & roi]**2)),
                'visible_log_mse': float(np.mean(residual[valid & ~roi]**2))}
            for key, value in calc.items():
                maximum[key] = max(maximum.get(key, 0.), abs(value - r[key]))
                assert np.isclose(value, r[key], rtol=1e-5, atol=1e-5), (rid, i, key)
            assert r['valid_pixels'] == int(valid.sum()); count += 1
        for name in ('inputs', 'evaluation'):
            path = PARENT / name / f'{rid}.json'
            assert sha256(path) == plan['reused_record_sha256'][str(path)]
        parent = json.loads((PARENT / 'evaluation' / f'{rid}.json').read_text())
        assert sha256(parent['prediction_path']) == parent['prediction_sha256']
        parent_count += len(parent['rows'])
    assert count == 648 and parent_count == 1224
    return {'predictions_independently_scored': count, 'parent_rows_identity_checked': parent_count,
            'max_score_difference': maximum}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--inputs', action='store_true'); args = p.parse_args()
    torch.set_num_threads(4); plan = load_plan(); verify_sources(plan)
    if args.inputs:
        r = input_checks(plan); print(json.dumps({'input_checks': True, 'states': r['states_checked']})); return
    inputs = json.loads((OUT / 'input_checks.json').read_text())
    assert inputs['complete'] and inputs['plan_sha256'] == sha256(OUT / 'plan.json')
    result = {'complete': True, 'input_checks_sha256': sha256(OUT / 'input_checks.json'),
              'training': fitting_checks(plan), **score_checks(plan),
              'checker_source_sha256': sha256(ROOT / 'experiments/gaussian_memory_visibility_training_checks.py')}
    write_json(OUT / 'independent_checks.json', result); print(json.dumps(result))


if __name__ == '__main__': main()
