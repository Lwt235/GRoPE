"""Native-output evaluation with reusable predictions and causal state snapshots."""
import argparse
import json
import time

import numpy as np
import torch

from experiments.gaussian_memory_setup import ROOT, sha256
from experiments.gaussian_memory_view_data import load_backbone
from experiments.gaussian_memory_incremental_data import (
    OUT, CACHE, SEEDS, KINDS, QUERIES, load_plan, write_json, save_tensor, device_tree)
from experiments.gaussian_memory_incremental_train import inputs_for
from experiments.gaussian_memory_incremental_states import STATE_OUT, AMPLITUDES, track_name
from models.gaussian_memory_vggt_adapter import InternalMemoryAdapter, native_tail, padded_geometry

PREDICTIONS = CACHE / 'predictions'


def evaluation_protocol():
    value = {'created_before_formal_native_evaluation': True,
             'primary': 'place-equal full/mask0/mask1 mixed log-MSE, CG vs C and CG vs L',
             'cohorts': ['old12', 'used_summary20', 'used_writer19'],
             'secondary': ['full', 'masked', 'hidden', 'visible', 'centered_log_mse', 'bias_squared', 'absrel', 'rmse_m'],
             'seeds': SEEDS, 'bootstrap': 'paired place and seed resampling, 5000 draws; patches not independent scenes',
             'center_control_trigger': 'combined mean log-MSE lower than content on primary51 and at least two of three paired seeds positive; CI is reported and not required for this diagnostic control',
             'center_control_claim': 'additional developmental shape control, not promotion or fresh confirmation',
             'clean_integration': 'content and combined, three fixed seeds, corrective/prefix; static scores reused',
             'error_and_retention_probe': 'combined first fixed seed only, full input, all signed amplitudes/strategies; diagnostic scope',
             'fixed_figure_ids': ['transfer_00', 'confirm_00', 'fresh_00'],
             'posthoc_counterexample': 'one largest combined-minus-local primary place, clearly labeled descriptive',
             'all_data_previously_used': True}
    dest = OUT / 'evaluation_protocol.json'
    if dest.exists(): assert json.loads(dest.read_text()) == json.loads(json.dumps(value))
    else: write_json(dest, value)
    return value


def scalar_mean(x):
    return float(x.mean()) if x.numel() else None


def score(depth_native, sample):
    depth_native = depth_native.float().cpu()
    target = sample['target'].float().cpu(); valid = sample['valid'].cpu(); hidden = sample['hidden'].cpu()
    assert torch.isfinite(depth_native).all() and (depth_native > 0).all()
    raw = depth_native.clamp_min(1e-6).log() - target.log()
    metric = sample['metric_scale'] is not None
    residual = raw + (np.log(sample['metric_scale']) if metric else 0.)
    r = residual[valid]
    result = {'id': sample['id'], 'query': sample['query'], 'condition': sample['condition'],
              'valid_pixels': int(valid.sum()), 'calibration_valid': metric,
              'log_mse': scalar_mean(r.square()) if metric else None,
              'centered_log_mse': scalar_mean((r - r.mean()).square()),
              'bias_squared': float(r.mean().square()) if metric else None,
              'hidden_log_mse': scalar_mean(residual[valid & hidden].square()) if metric else None,
              'visible_log_mse': scalar_mean(residual[valid & ~hidden].square()) if metric else None}
    if metric:
        prediction = depth_native * sample['metric_scale']
        error = prediction[valid] - target[valid]
        result.update(absrel=scalar_mean(error.abs() / target[valid]), rmse_m=float(error.square().mean().sqrt()))
        if 'selected_region' in sample:
            region = sample['selected_region'].cpu() & valid
            result['selected_region_log_mse'] = scalar_mean(residual[region].square())
            result['selected_region_valid_pixels'] = int(region.sum())
    else:
        result.update(absrel=None, rmse_m=None)
    return result


def save_case_record(phase, case, rid, rows, predictions, checkpoint_path=None):
    folder = OUT / 'evaluation' / phase / case
    folder.mkdir(parents=True, exist_ok=True)
    pred_folder = PREDICTIONS / phase / case; pred_folder.mkdir(parents=True, exist_ok=True)
    target = pred_folder / (rid + '.pt')
    save_tensor(target, {'prediction_native': torch.stack(predictions),
                        'sample_keys': [[r['query'], r['condition']] for r in rows],
                        'state_keys': [r.get('state_key') for r in rows]})
    record = {'complete': True, 'rows': rows, 'prediction_path': str(target),
              'prediction_sha256': sha256(target),
              'checkpoint_sha256': sha256(checkpoint_path) if checkpoint_path is not None else None}
    write_json(folder / (rid + '.json'), record)


def load_adapter(kind, seed):
    path = OUT / f'{kind}_{seed}' / 'step_2000.pt'
    state = torch.load(path, map_location='cpu', weights_only=False)
    adapter = InternalMemoryAdapter().cuda().eval().requires_grad_(False)
    adapter.load_state_dict(state['model'])
    return adapter, state['statistics'], path


@torch.no_grad()
def predict(backbone, adapter, sample_cpu, initial_gpu, kind, statistics, replacement=None):
    sample = device_tree(sample_cpu)
    pe, content = inputs_for(sample, initial_gpu, kind, statistics, replacement)
    if replacement is not None and kind in ('geometry', 'combined'):
        # Match the cached training/static path: raw clipping, then RMS, then clipping.
        raw = padded_geometry(replacement, sample['c2w'], sample['K'])
        pe = (raw / raw.new_tensor(statistics['geometry_rms'])).clamp(-20, 20)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        depth, _ = native_tail(backbone, sample, adapter, pe, content)
    return depth.float().cpu()


def baseline():
    evaluation_protocol(); plan = load_plan()
    for meta in plan['catalog']:
        rid = meta['id']; destination = OUT / 'evaluation/static/native' / (rid + '.json')
        if destination.exists(): continue
        rows = []
        for q in QUERIES:
            for c in range(3):
                sample = torch.load(CACHE / rid / f'q{q}_c{c}.pt', map_location='cpu', weights_only=False)
                rows.append(score(sample['query_depth_native'], sample))
        write_json(destination, {'complete': True, 'rows': rows, 'prediction_source': 'unchanged native cache',
                   'cache_identity_sha256': sha256(CACHE / rid / 'identity.json')})
    write_json(OUT / 'baseline_complete.json', {'complete': True, 'clips': len(plan['catalog'])})


def static(kind=None, seed=None):
    evaluation_protocol(); plan = load_plan(); backbone = load_backbone()
    start = time.monotonic()
    for k in ([kind] if kind else KINDS):
        for s in ([seed] if seed is not None else SEEDS):
            adapter, statistics, path = load_adapter(k, s); case = f'{k}_{s}'
            for index, meta in enumerate(plan['catalog']):
                rid = meta['id']
                if (OUT / 'evaluation/static' / case / (rid + '.json')).exists(): continue
                initial = device_tree(torch.load(CACHE / rid / 'initial.pt', map_location='cpu', weights_only=False))
                rows, predictions = [], []
                for q in QUERIES:
                    for c in range(3):
                        sample = torch.load(CACHE / rid / f'q{q}_c{c}.pt', map_location='cpu', weights_only=False)
                        prediction = predict(backbone, adapter, sample, initial, k, statistics)
                        rows.append(score(prediction, sample)); predictions.append(prediction)
                save_case_record('static', case, rid, rows, predictions, path)
                write_json(OUT / 'evaluation_progress.json', {'stage': 'static', 'case': case,
                           'clips_complete': index + 1, 'clips': len(plan['catalog']), 'seconds': time.monotonic() - start})
                if (index + 1) % 10 == 0:
                    print(json.dumps({'evaluation': case, 'clips': index + 1, 'seconds': round(time.monotonic() - start, 2)}), flush=True)
            del adapter
    write_json(OUT / f'static_{kind or "all"}_{seed or "all"}_complete.json', {'complete': True, 'seconds': time.monotonic() - start})


def integrated(kind=None, seed=None):
    evaluation_protocol(); plan = load_plan(); backbone = load_backbone(); start = time.monotonic()
    for k in ([kind] if kind else ('content', 'combined')):
        for s in ([seed] if seed is not None else SEEDS):
            adapter, statistics, path = load_adapter(k, s)
            for policy in ('corrective_clean', 'rebuild_clean'):
                case = f'{k}_{s}_{policy}'
                for index, meta in enumerate(plan['catalog']):
                    rid = meta['id']
                    if (OUT / 'evaluation/integrated' / case / (rid + '.json')).exists(): continue
                    initial = device_tree(torch.load(CACHE / rid / 'initial.pt', map_location='cpu', weights_only=False))
                    tracks = torch.load(STATE_OUT / rid / 'trajectories.pt', map_location='cpu', weights_only=False)
                    states = tracks[policy]['states']
                    rows, predictions = [], []
                    for ti, q in enumerate(QUERIES, 1):
                        memory = device_tree(states[ti])
                        for c in range(3):
                            sample = torch.load(CACHE / rid / f'q{q}_c{c}.pt', map_location='cpu', weights_only=False)
                            prediction = predict(backbone, adapter, sample, initial, k, statistics, memory)
                            row = score(prediction, sample); row['state_key'] = [policy, ti]
                            rows.append(row); predictions.append(prediction)
                    save_case_record('integrated', case, rid, rows, predictions, path)
                    write_json(OUT / 'evaluation_progress.json', {'stage': 'integrated', 'case': case,
                               'clips_complete': index + 1, 'seconds': time.monotonic() - start})
                    if (index + 1) % 20 == 0:
                        print(json.dumps({'integration': case, 'clips': index + 1, 'seconds': round(time.monotonic() - start, 2)}), flush=True)
            del adapter
    write_json(OUT / f'integrated_{kind or "all"}_{seed or "all"}_complete.json', {'complete': True, 'seconds': time.monotonic() - start})


def error_probe():
    evaluation_protocol(); plan = load_plan(); backbone = load_backbone(); start = time.monotonic()
    adapter, statistics, path = load_adapter('combined', SEEDS[0])
    policies = ['average_clean'] + [track_name(st, a) for a in AMPLITUDES for st in ('static', 'average', 'corrective')]
    for meta in plan['catalog']:
        rid = meta['id']; case = 'combined_20261110'
        if ((OUT / 'evaluation/error_probe' / case / (rid + '.json')).exists()
                and (OUT / 'evaluation/retention' / case / (rid + '.json')).exists()): continue
        initial = device_tree(torch.load(CACHE / rid / 'initial.pt', map_location='cpu', weights_only=False))
        tracks = torch.load(STATE_OUT / rid / 'trajectories.pt', map_location='cpu', weights_only=False)
        samples = {q: torch.load(CACHE / rid / f'q{q}_c0.pt', map_location='cpu', weights_only=False) for q in QUERIES}
        regions = torch.load(STATE_OUT / rid / 'selected_region.pt', map_location='cpu', weights_only=False)['query_patch_masks']
        for qi, q in enumerate(QUERIES):
            samples[q]['selected_region'] = regions[qi].reshape(15, 20).repeat_interleave(14, 0).repeat_interleave(14, 1)
        rows, predictions = [], []
        for policy in policies:
            for ti, q in enumerate(QUERIES, 1):
                memory = device_tree(tracks[policy]['states'][ti])
                prediction = predict(backbone, adapter, samples[q], initial, 'combined', statistics, memory)
                row = score(prediction, samples[q]); row['state_key'] = [policy, ti]
                rows.append(row); predictions.append(prediction)
        save_case_record('error_probe', case, rid, rows, predictions, path)
        # Same early read-only view, revisited with later clean states; not causal-time performance.
        rows, predictions = [], []
        for policy in ('static_clean', 'average_clean', 'corrective_clean', 'rebuild_clean'):
            for ti in (0, 1, 2, 3, 4):
                memory = device_tree(tracks[policy]['states'][ti])
                prediction = predict(backbone, adapter, samples[5], initial, 'combined', statistics, memory)
                row = score(prediction, samples[5]); row.update(state_key=[policy, ti], retrospective_retention=True)
                rows.append(row); predictions.append(prediction)
        save_case_record('retention', case, rid, rows, predictions, path)
        write_json(OUT / 'evaluation_progress.json', {'stage': 'error_and_retention', 'last_id': rid,
                   'seconds': time.monotonic() - start})
        print(json.dumps({'error_probe': rid, 'seconds': round(time.monotonic() - start, 2)}), flush=True)
    write_json(OUT / 'error_probe_complete.json', {'complete': True, 'seconds': time.monotonic() - start,
               'fixed_seed': SEEDS[0], 'condition': 'full', 'scope': 'developmental mechanism probe'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('protocol', 'baseline', 'static', 'integrated', 'error_probe'))
    p.add_argument('--kind', choices=KINDS + ('center',)); p.add_argument('--seed', type=int)
    args = p.parse_args(); torch.set_num_threads(4)
    if args.stage == 'protocol': evaluation_protocol()
    elif args.stage in ('static', 'integrated'): globals()[args.stage](args.kind, args.seed)
    else: globals()[args.stage]()
