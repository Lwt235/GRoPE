"""Independent raw-label scoring, fixed model pairing, and parent reuse checks."""
import json
import re
from pathlib import Path

import numpy as np
import torch

from experiments.gaussian_memory_setup import ROOT, sha256
from experiments.gaussian_memory_soft_data import OUT, PARENT_OUT, BASE_CACHE, SEEDS, QUERIES, load_plan, source_identity, write_json
from experiments.gaussian_memory_incremental_data import sensor_target
from experiments.gaussian_memory_incremental_states import STATE_OUT
from experiments.gaussian_memory_view_data import rectangle


def expected_cases(decision):
    kinds = ['soft'] + (['soft_center'] if decision['center_control_triggered'] else [])
    result = {'static': [(f'{k}_{s}', 12) for k in kinds for s in SEEDS],
              'intervention': [(f'soft_{SEEDS[0]}_{name}', 12) for name in ('content_permuted', 'uniform_prior')],
              'integrated': [(f'{k}_{s}_{policy}', 12) for k in ('content', 'soft') for s in SEEDS
                             for policy in ('corrective_clean', 'rebuild_clean')],
              'retention': [(f'{k}_{SEEDS[0]}', 3) for k in ('content', 'soft')]}
    if decision['center_control_triggered']:
        result['integrated'] += [(f'soft_center_{s}_corrective_clean', 12) for s in SEEDS]
    return kinds, result


def training_checks(kinds):
    identities = source_identity(); summaries = []; matches = []
    for name, digest in identities.items(): assert sha256(OUT / 'source_snapshot' / name) == digest
    for s in SEEDS:
        parent = PARENT_OUT / f'content_{s}'
        old = torch.load(parent / 'initial.pt', map_location='cpu', weights_only=False)
        old_rows = [json.loads(x) for x in (parent / 'training.jsonl').read_text().splitlines()]
        for k in kinds:
            folder = OUT / f'{k}_{s}'; status = json.loads((folder / 'status.json').read_text())
            assert status['complete'] and status['steps'] == 2000 and not status['pilot']
            assert status['checkpoint_sha256'] == sha256(folder / 'step_2000.pt')
            saved = torch.load(folder / 'step_2000.pt', map_location='cpu', weights_only=False)
            assert saved['source_sha256'] == identities and saved['kind'] == k and saved['seed'] == s
            assert all(torch.isfinite(t).all() for t in saved['model'].values())
            initial = torch.load(folder / 'initial.pt', map_location='cpu', weights_only=False)
            assert initial['source_sha256'] == identities
            assert initial['schedule_sha256'] == old['schedule_sha256'] == saved['schedule_sha256']
            assert set(initial['model']) == set(old['model'])
            assert all(torch.equal(initial['model'][n], old['model'][n]) for n in old['model'])
            rows = [json.loads(x) for x in (folder / 'training.jsonl').read_text().splitlines()]
            assert [r['step'] for r in rows] == list(range(1, 2001))
            assert all(a['samples'] == b['samples'] for a, b in zip(rows, old_rows)) and len(old_rows) == len(rows)
            assert all(np.isfinite(r['loss']) and np.isfinite(r['gradient_norm']) for r in rows)
            segments = []; previous = 0.
            for r in rows:
                if r['elapsed_s'] < previous: segments.append(previous)
                previous = r['elapsed_s']
            segments.append(previous)
            summaries.append({**status, 'training_loop_seconds': float(sum(segments)), 'elapsed_segments': len(segments)})
        matches.append({'seed': s, 'new_kinds': kinds, 'parent': f'content_{s}',
                        'initialization_equal': True, 'all_2000_sampling_steps_equal': True})
    write_json(OUT / 'training_summary.json', {'fits': len(summaries), 'steps': sum(x['steps'] for x in summaries),
        'sum_worker_hours': sum(x['training_loop_seconds'] for x in summaries) / 3600,
        'peak_allocated_GiB': max(x['peak_GiB'] for x in summaries), 'per_fit': summaries,
        'scope': 'shared GPU training loops, excludes waiting/extraction/evaluation; reused fits excluded'})
    write_json(OUT / 'formal_training_pairing.json', {'matched': matches, 'all_final_tensors_finite': True})
    return matches


def main():
    plan = load_plan(); decision = json.loads((OUT / 'conditional_decision.json').read_text())
    kinds, expected = expected_cases(decision); matches = training_checks(kinds)
    metadata = {m['id']: m for m in plan['catalog']}; targets = {}; identities = {}; hashes = {}
    def digest(path):
        path = str(path)
        if path not in hashes: hashes[path] = sha256(path)
        return hashes[path]
    protocol_hash = digest(OUT / 'evaluation_protocol.json')
    evaluator_hash = digest(ROOT / 'experiments/gaussian_memory_soft_evaluate.py')
    phase_counts = {}; record_counts = {}; max_error = {}; total = 0
    for phase, cases in expected.items():
        for case, rows_expected in cases:
            kind, seed = re.match(r'(soft_center|soft|content)_(\d+)', case).groups()
            checkpoint = (PARENT_OUT if kind == 'content' else OUT) / f'{kind}_{seed}/step_2000.pt'
            for rid in metadata:
                path = OUT / 'evaluation' / phase / case / f'{rid}.json'; record = json.loads(path.read_text())
                assert record['complete'] and len(record['rows']) == rows_expected
                assert record['evaluation_protocol_sha256'] == protocol_hash
                assert record['evaluation_source_sha256'] == evaluator_hash
                assert record['checkpoint_sha256'] == digest(checkpoint)
                if rid not in identities:
                    identities[rid] = json.loads((BASE_CACHE / rid / 'identity.json').read_text())
                    targets[rid] = {q: tuple(x.numpy() for x in sensor_target(metadata[rid]['row']['frames'][q])) for q in QUERIES}
                identity = identities[rid]; sources = record['sources']
                assert sources['causal_cache_identity_sha256'] == digest(BASE_CACHE / rid / 'identity.json')
                assert sources['initial_sha256'] == identity['sha256']['initial.pt']
                if 'prior_identity_path' in sources:
                    assert sources['prior_identity_sha256'] == digest(sources['prior_identity_path'])
                if 'state_trajectories_sha256' in sources:
                    assert sources['state_trajectories_sha256'] == digest(STATE_OUT / rid / 'trajectories.pt')
                    parent_record = PARENT_OUT / f'evaluation/integrated/bound_{SEEDS[0]}_corrective_clean/{rid}.json'
                    assert sources['state_trajectories_sha256'] == json.loads(parent_record.read_text())['sources']['state_trajectories_sha256']
                prediction_path = Path(record['prediction_path']); assert digest(prediction_path) == record['prediction_sha256']
                export = torch.load(prediction_path, map_location='cpu', weights_only=False)
                assert len(export['prediction_native']) == rows_expected
                scales = {(d['window'][-1], d['condition']): d.get('scale') for d in identity['diagnostics'] if d['kind'] == 'query'}
                expected_keys = [(5, 0, t) for t in (0, 1, 4)] if phase == 'retention' else [(q, c, None) for q in QUERIES for c in range(3)]
                for index, row in enumerate(record['rows']):
                    q, c = row['query'], row['condition']; eq, ec, et = expected_keys[index]
                    assert (q, c) == (eq, ec) and row['id'] == rid
                    assert export['sample_keys'][index] == [q, c] and export['state_keys'][index] == row.get('state_key')
                    if phase == 'retention':
                        assert row['state_key'] == ['corrective_clean', et] and row['retrospective_retention']
                    elif phase == 'integrated':
                        policy = 'corrective_clean' if case.endswith('corrective_clean') else 'rebuild_clean'
                        assert row['state_key'] == [policy, QUERIES.index(q) + 1]
                    target, valid = targets[rid][q]; target = target.astype('float64')
                    pred = export['prediction_native'][index].numpy().astype('float64')
                    assert np.isfinite(pred).all() and (pred > 0).all()
                    assert int(valid.sum()) == row['valid_pixels']
                    scale = scales[q, c]; assert (scale is not None) == row['calibration_valid']
                    residual = np.log(np.maximum(pred, 1e-6)) - np.log(target)
                    if scale is not None: residual += np.log(scale)
                    r = residual[valid]; calc = {'centered_log_mse': float(np.mean((r - r.mean())**2))}
                    if scale is not None:
                        error = pred[valid] * scale - target[valid]
                        mask = np.zeros((210, 280), dtype=bool) if c == 0 else rectangle(rid, q, c - 1).numpy().reshape(15, 20).repeat(14, 0).repeat(14, 1)
                        calc.update(log_mse=float(np.mean(r*r)), bias_squared=float(r.mean()**2),
                                    absrel=float(np.mean(np.abs(error) / target[valid])), rmse_m=float(np.sqrt(np.mean(error*error))))
                        for metric, keep in [('hidden_log_mse', valid & mask), ('visible_log_mse', valid & ~mask)]:
                            if keep.any(): calc[metric] = float(np.mean(residual[keep]**2))
                    for metric, value in calc.items():
                        delta = abs(value - row[metric]); max_error[metric] = max(max_error.get(metric, 0.), delta)
                        assert np.isclose(value, row[metric], rtol=1e-5, atol=1e-5), (path, index, metric, value, row[metric])
                    total += 1; phase_counts[phase] = phase_counts.get(phase, 0) + 1
                record_counts[phase] = record_counts.get(phase, 0) + 1
            print(json.dumps({'independently_checked': f'{phase}/{case}', 'predictions': total}), flush=True)
    reused = json.loads((OUT / 'baseline_reuse.json').read_text()); assert reused['complete']; reused_rows = 0
    for case, clips in reused['parent_records'].items():
        assert set(clips) == set(metadata)
        for rid, record_hash in clips.items():
            record = json.loads((OUT / 'evaluation/static' / case / f'{rid}.json').read_text())
            assert digest(record['soft_reused_from']) == record_hash == record['soft_reused_record_sha256']
            old = json.loads(Path(record['soft_reused_from']).read_text()); assert record['rows'] == old['rows']
            reused_rows += len(record['rows'])
    result = {'complete': True, 'exported_predictions_recomputed': total, 'by_phase': phase_counts,
        'records_by_phase': record_counts, 'reused_parent_rows': reused_rows,
        'float64_numpy_max_abs_difference': max_error, 'matched_training': matches,
        'scientific_sources_match_snapshot': True, 'old_state_identities_match': True,
        'new_confirmation': False, 'no_row_or_clip_exclusions': True}
    write_json(OUT / 'independent_checks.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4); main()
