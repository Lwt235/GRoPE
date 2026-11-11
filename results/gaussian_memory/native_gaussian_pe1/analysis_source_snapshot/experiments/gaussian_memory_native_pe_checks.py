"""Independent geometry and exported-pixel checks, separate from the training loss."""
import argparse
import hashlib
import json
import numpy as np
import torch

from experiments.gaussian_memory_native_pe_data import (
    ROOT, OUT, CACHE, load, read, sha256, write_json, save_tensor, load_plan, verify,
    examples, input_path, sample_path, sample_key, NATIVE_PARENT)


def metric_numpy(pred, sample):
    p = np.asarray(pred, np.float64); t = sample['target'].numpy().astype(np.float64)
    valid = sample['valid'].numpy(); hidden = sample['hidden'].numpy()
    residual = np.log(np.maximum(p, 1e-6) / t); selected = residual[valid]; roi = residual[valid & hidden]
    error = p[valid] - t[valid]; roierror = p[valid & hidden] - t[valid & hidden]
    return {'log_mse': float(np.mean(selected**2)), 'hidden_log_mse': float(np.mean(roi**2)),
            'visible_log_mse': float(np.mean(residual[valid & ~hidden]**2)),
            'centered_log_mse': float(np.mean((selected - np.mean(selected))**2)),
            'bias_squared': float(np.mean(selected)**2), 'log_bias': float(np.mean(selected)),
            'absrel': float(np.mean(np.abs(error) / t[valid])),
            'hidden_centered_log_mse': float(np.mean((roi - np.mean(roi))**2)),
            'hidden_bias_squared': float(np.mean(roi)**2),
            'hidden_absrel': float(np.mean(np.abs(roierror) / t[valid & hidden])),
            'rmse_m': float(np.sqrt(np.mean(error**2))), 'valid_pixels': int(valid.sum()),
            'hidden_valid_pixels': int((valid & hidden).sum())}


def geometry_numpy(s):
    d = s['geometry_depths'].numpy().astype(np.float64); ext = s['predicted_extrinsics'].numpy(); intr = s['predicted_intrinsics'].numpy()
    count, h, w = d.shape; y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    points = []
    for f in range(count):
        cam = np.stack(((x - intr[f, 0, 2]) / intr[f, 0, 0] * d[f],
                        (y - intr[f, 1, 2]) / intr[f, 1, 1] * d[f], d[f]), -1)
        points.append((cam - ext[f, :3, 3]) @ ext[f, :3, :3])
    points = np.stack(points); flat = d[0].ravel(); unit = np.partition(flat, (len(flat) - 1) // 2)[(len(flat) - 1) // 2]
    origin = np.mean(points[0], axis=(0, 1)); points = (points - origin) / unit
    patches = points.reshape(count, h // 14, 14, w // 14, 14, 3).transpose(0, 1, 3, 2, 4, 5).reshape(count, -1, 196, 3)
    mu = patches.mean(-2); delta = patches - mu[..., None, :]
    cov = np.einsum('spni,spnj->spij', delta, delta) / 196 + np.eye(3) * 1e-6
    actual_mu = s['means'].numpy().reshape(count, -1, 3)[:, 5:]
    actual_cov = s['covariances'].numpy().reshape(count, -1, 3, 3)[:, 5:]
    return {'center_max_abs': float(np.max(np.abs(actual_mu - mu))),
            'covariance_max_abs': float(np.max(np.abs(actual_cov - cov))),
            'unit_abs': abs(s['geometry_unit'] - unit),
            'origin_max_abs': float(np.max(np.abs(s['geometry_origin'].numpy() - origin)))}


def input_checks():
    plan = load_plan(); verify(plan)
    assert sha256(plan['model_path']) == plan['model_sha256']
    for name, digest in plan['official_source_sha256'].items():
        assert sha256(ROOT / 'third_party/streamvggt' / name) == digest, name
    train_places = {m['place'] for m in plan['training_catalog']}; dev_places = {m['place'] for m in plan['catalog']}
    assert not train_places & dev_places
    cache_records = []; geometry = []; replay = []
    from experiments.gaussian_memory_native_run import metric_depth
    for split in ('train', 'development'):
        for key in examples(plan, split):
            rid, q, c = key; path = sample_path(*key)
            rec = read(OUT / 'cache' / (sample_key(*key).replace('/', '_') + '.json'))
            assert sha256(path) == rec['sha256'] and rec['input_sha256'] == sha256(input_path(*key))
            s = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
            assert all(f < q and f not in (5, 7, 9, 11) for f in s['source_ids'])
            assert s['source_ids'][0] == 0
            n = s['tokens_per_frame']; assert n == 1041
            assert all(m['k'].shape == m['v'].shape == (1, 16, len(s['source_ids']) * n, 64) for m in s['memory'])
            assert all(m['norm'].shape == (1, len(s['source_ids']) * n, 1024) for m in s['memory'])
            assert s['means'].shape == ((len(s['source_ids']) + 1) * n, 3)
            assert set(s['early']) == {4, 11, 17}
            assert torch.isfinite(s['covariances']).all() and torch.linalg.eigvalsh(s['covariances'].double()).min() > 0
            cache_records.append({'key': key, 'bytes': path.stat().st_size, 'memory_bytes': rec['memory_bytes']})
            if (split == 'train' and rid in plan['training_ids'][:2] and q == 11 and c == 'persistent') or (split == 'development' and rid in plan['fixed_examples'][:2]):
                g = geometry_numpy(s); assert g['center_max_abs'] < 2e-6 and g['covariance_max_abs'] < 2e-6
                geometry.append({'key': key, **g})
            if split == 'development':
                record = read(NATIVE_PARENT / 'evaluation' / f'{rid}.json')
                old = load(record['prediction_path']); index = old['keys'].index([c, 'history'])
                delta = metric_depth(s['baseline_depth']) - old['depth_metric'][index]
                replay.append({'key': key, 'max_abs': float(delta.abs().max()), 'rms': float(delta.square().mean().sqrt())})
    write_json(OUT / 'independent_input_checks.json', {'complete': True, 'train_development_place_disjoint': True,
        'examples': len(cache_records), 'model_sha256': plan['model_sha256'], 'cache_records': cache_records,
        'geometry_numpy': geometry, 'official_history_replays': replay,
        'old_history_replay_max_abs': max(r['max_abs'] for r in replay),
        'geometry_reads_query_labels': False, 'all_data_previously_used': True,
        'source_sha256': sha256(ROOT / 'experiments/gaussian_memory_native_pe_checks.py')})


def result_checks():
    plan = load_plan(); verify(plan); rows = []; errors = []; checkpoints = []
    for seed in plan['seeds']:
        signatures = []
        for kind in plan['kinds']:
            folder = OUT / f'{kind}_{seed}'; status = read(folder / 'status.json'); checkpoint = folder / f'step_{plan["steps"]}.pt'
            assert status['complete'] and not status['pilot'] and status['steps'] == plan['steps']
            assert status['checkpoint_sha256'] == sha256(checkpoint)
            assert status['training_lock_sha256'] == sha256(OUT / 'training_lock.json')
            log = [json.loads(l) for l in (folder / 'training.jsonl').read_text().splitlines()]
            assert len(log) == plan['steps'] and all(np.isfinite(r['loss']) and np.isfinite(r['gradient_norm']) for r in log)
            assert all(len(r['samples']) == 2 for r in log)
            assert all(s[0] in plan['training_ids'] and s[1] in plan['queries'] and s[2] in plan['conditions'] for r in log for s in r['samples'])
            sample_digest = hashlib.sha256(json.dumps([r['samples'] for r in log], separators=(',', ':')).encode()).hexdigest()
            initial = load(folder / 'initial.pt')['model']; h = hashlib.sha256()
            for name, value in initial.items(): h.update(name.encode()); h.update(value.numpy().tobytes())
            assert h.hexdigest() == status['initial_state_sha256']
            signatures.append((h.hexdigest(), status['schedule_sha256'], sample_digest, status['parameters']))
            checkpoints.append({'kind': kind, 'seed': seed, 'steps': len(log), 'checkpoint_sha256': sha256(checkpoint),
                                'first20': status['first20'], 'last20': status['last20'], 'peak_GiB': status['peak_GiB']})
            rec = read(OUT / 'evaluation' / f'{kind}_{seed}.json'); assert rec['checkpoint_sha256'] == sha256(checkpoint)
            assert sha256(rec['prediction_path']) == rec['prediction_sha256']
            saved = load(rec['prediction_path'])
            assert saved['keys'] == [list(key) for key in examples(plan, 'development')]
            assert len(rec['rows']) == 24 and len(saved['depth_metric']) == 24
            for key, pred, recorded in zip(saved['keys'], saved['depth_metric'], rec['rows']):
                sample = load(input_path(*key)); m = metric_numpy(pred.numpy(), sample)
                assert all(np.isfinite(v) for v in m.values()) and np.isfinite(pred.numpy()).all() and pred.min() > 0
                diffs = {k: abs(v - recorded[k]) for k, v in m.items()}
                assert max(diffs.values()) < 1e-10, (key, diffs)
                errors.append(max(diffs.values()))
                rows.append(dict(id=key[0], condition=key[2], kind=kind, seed=seed, place=recorded['place'], **m))
        assert len(set(signatures)) == 1, (seed, signatures)
    assert len(rows) == 288
    baseline_preds = []; baseline_keys = []
    for key in examples(plan, 'development'):
        sample = load(sample_path(*key)); from experiments.gaussian_memory_native_run import metric_depth
        pred = metric_depth(sample['baseline_depth']); m = metric_numpy(pred.numpy(), sample)
        rows.append(dict(id=key[0], condition=key[2], kind='official', seed=None, place=sample['place'], **m))
        baseline_preds.append(pred); baseline_keys.append(key)
    path = CACHE / 'evaluation/official.pt'; save_tensor(path, {'keys': baseline_keys, 'depth_metric': torch.stack(baseline_preds)})
    write_json(OUT / 'independent_result_checks.json', {'complete': True, 'metric_rows': len(rows), 'adapted_rows': 288,
        'max_metric_abs': max(errors), 'checkpoints': checkpoints, 'all_training_finite': True,
        'paired_initialization_and_schedule_exact': True, 'baseline_path': str(path), 'baseline_sha256': sha256(path),
        'plan_sha256': sha256(OUT / 'plan.json'), 'source_sha256': sha256(ROOT / 'experiments/gaussian_memory_native_pe_checks.py')})
    write_json(OUT / 'independent_rows.json', rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['inputs', 'results']); a = p.parse_args()
    input_checks() if a.action == 'inputs' else result_checks()
