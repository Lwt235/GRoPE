"""One label-free log-scale projection, uniformly applied to all fixed controls."""
import json
import time
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import torch

from experiments.gaussian_memory_native_anchor_data import (
    ROOT, OUT, CACHE, PARENT, BASE_CACHE, COUNTS, QUERIES,
    load_plan, sha256, write_json, save_tensor)
from experiments.gaussian_memory_view_data import rectangle
from experiments.gaussian_memory_incremental_evaluate import score


@torch.no_grad()
def anchor_depth(memory_depth, native_depth, unmasked_rgb):
    """Only prediction tensors and the synthetic RGB mask cross this boundary."""
    assert memory_depth.shape == native_depth.shape == unmasked_rgb.shape
    assert memory_depth.ndim == 3 and unmasked_rgb.dtype == torch.bool
    assert torch.isfinite(memory_depth).all() and (memory_depth > 0).all()
    assert torch.isfinite(native_depth).all() and (native_depth > 0).all()
    count = unmasked_rgb.sum((-2, -1)); assert (count > 0).all()
    r = memory_depth.log() - native_depth.log()
    delta = (r * unmasked_rgb).sum((-2, -1)) / count
    result = native_depth * torch.exp(r - delta[:, None, None])
    assert torch.isfinite(result).all() and (result > 0).all()
    return result, delta


def rgb_hidden(rid, query, condition):
    if condition == 0:
        return torch.zeros((210, 280), dtype=torch.bool)
    return rectangle(rid, query, condition - 1).reshape(15, 20).repeat_interleave(14, 0).repeat_interleave(14, 1)


def load_samples(rid, keys, identity_digest):
    identity_path = BASE_CACHE / rid / 'identity.json'
    assert sha256(identity_path) == identity_digest
    identity = json.loads(identity_path.read_text()); samples = {}; inputs = {}
    for q, c in sorted(keys):
        name = f'q{q}_c{c}.pt'; path = BASE_CACHE / rid / name
        assert sha256(path) == identity['sha256'][name]
        s = torch.load(path, map_location='cpu', weights_only=False)
        mask = rgb_hidden(rid, q, c); assert torch.equal(mask, s['hidden'])
        samples[q, c] = {k: s[k] for k in ('id', 'query', 'condition', 'query_depth_native',
                                         'target', 'valid', 'hidden', 'metric_scale')}
        inputs[name] = identity['sha256'][name]
        del s
    return samples, inputs


def main():
    start = time.monotonic(); plan = load_plan(); plan_hash = sha256(OUT / 'plan.json')
    for name, digest in plan['source_sha256'].items(): assert sha256(ROOT / name) == digest
    assert torch.cuda.is_available()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    pair_by_id = {rid: p for p in plan['pairs'] for rid in (p['id'], p['early_id'])}
    counts = Counter(); record_count = 0; transform_seconds = 0.; native_rows = 0
    max_anchor_residual = 0.; max_spatial_change = 0.; min_depth = float('inf')
    for rid in [m['id'] for m in plan['catalog'] + plan['early_catalog']]:
        tasks = [t for t in plan['records'] if t['id'] == rid]
        keys = {tuple(k) for t in tasks for k in t['sample_keys']}
        identity_digest = pair_by_id[rid]['inputs'][rid]['identity_sha256']
        samples, inputs = load_samples(rid, keys, identity_digest)
        for task in tasks:
            assert sha256(task['path']) == task['sha256']
            old = json.loads(open(task['path']).read())
            assert sha256(task['prediction_path']) == task['prediction_sha256']
            export = torch.load(task['prediction_path'], map_location='cpu', weights_only=False)
            indices = task['indices']; sample_keys = task['sample_keys']
            assert [export['sample_keys'][i] for i in indices] == sample_keys
            memory = export['prediction_native'][indices].float().cuda()
            native = torch.stack([samples[tuple(k)]['query_depth_native'] for k in sample_keys]).float().cuda()
            # The support mask is generated from IDs and the RGB corruption schedule alone.
            unmasked = torch.stack([~rgb_hidden(rid, *k) for k in sample_keys]).cuda()
            torch.cuda.synchronize(); tick = time.monotonic()
            prediction, delta = anchor_depth(memory, native, unmasked)
            torch.cuda.synchronize(); transform_seconds += time.monotonic() - tick
            residual = (prediction.log() - native.log()) * unmasked
            max_anchor_residual = max(max_anchor_residual, float((residual.sum((-2, -1)) / unmasked.sum((-2, -1))).abs().max()))
            min_depth = min(min_depth, float(prediction.min()))
            prediction = prediction.cpu(); shifts = delta.cpu(); supports = unmasked.sum((-2, -1)).cpu()
            rows = []
            for i, key in enumerate(sample_keys):
                row = score(prediction[i], samples[tuple(key)])
                for name in ('state_key', 'retrospective_retention'):
                    if name in old['rows'][i]: row[name] = old['rows'][i][name]
                row.update(anchor_log_shift=float(shifts[i]), anchor_unmasked_pixels=int(supports[i]))
                max_spatial_change = max(max_spatial_change, abs(row['centered_log_mse'] - old['rows'][i]['centered_log_mse']))
                rows.append(row)
            path = CACHE / 'predictions' / task['phase'] / task['case'] / f'{rid}.pt'
            path.parent.mkdir(parents=True, exist_ok=True)
            save_tensor(path, {'prediction_native': prediction, 'sample_keys': sample_keys,
                              'anchor_log_shift': shifts,
                              'state_keys': [r.get('state_key') for r in rows]})
            record = {'complete': True, 'rows': rows, 'prediction_path': str(path),
                'prediction_sha256': sha256(path), 'checkpoint_sha256': task['checkpoint_sha256'],
                'parent_record': task['path'], 'parent_record_sha256': task['sha256'],
                'parent_prediction_path': task['prediction_path'], 'parent_prediction_sha256': task['prediction_sha256'],
                'parent_prediction_indices': indices, 'native_cache_identity_sha256': identity_digest,
                'query_cache_sha256': inputs, 'plan_sha256': plan_hash,
                'formula_source_sha256': plan['source_sha256']['experiments/gaussian_memory_native_anchor_evaluate.py']}
            write_json(OUT / 'evaluation' / task['phase'] / task['case'] / f'{rid}.json', record)
            counts[task['phase']] += len(rows); record_count += 1
            del memory, native, unmasked, export, prediction, delta
        if rid.startswith('fidelity_'):
            info = next(n for n in plan['native_records'] if n['id'] == rid)
            assert sha256(info['path']) == info['sha256']
            old = json.loads(open(info['path']).read())
            write_json(OUT / 'evaluation/reused/native' / f'{rid}.json',
                {'complete': True, 'rows': old['rows'], 'parent_record': info['path'],
                 'parent_record_sha256': info['sha256'], 'plan_sha256': plan_hash,
                 'prediction_source': 'Unmodified native query cache; no transform required.'})
            native_rows += len(old['rows'])
        print(json.dumps({'clip_complete': rid, 'transformed': sum(counts.values()), 'by_phase': dict(counts)}), flush=True)
    assert dict(counts) == COUNTS and native_rows == 144
    assert max_anchor_residual < 1e-5 and max_spatial_change < 1e-5
    runtime = {'complete': True, 'completed_utc': datetime.now(timezone.utc).isoformat(),
        'transformed_predictions': sum(counts.values()), 'by_phase': dict(counts), 'records': record_count,
        'reused_native_rows': native_rows, 'new_training_steps': 0, 'new_download_bytes': 0,
        'new_backbone_inferences': 0, 'worker_seconds': time.monotonic() - start,
        'transform_compute_seconds': transform_seconds,
        'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 1024**3,
        'GPU': torch.cuda.get_device_name(0), 'max_unmasked_mean_log_change': max_anchor_residual,
        'max_centered_score_change': max_spatial_change, 'minimum_exported_depth': min_depth,
        'scope': 'One shared-cap worker; cached prediction transform, not end-to-end inference throughput.'}
    write_json(OUT / 'runtime_costs.json', runtime)
    print(json.dumps(runtime), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
