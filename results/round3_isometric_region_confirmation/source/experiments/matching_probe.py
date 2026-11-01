"""Train tiny attention matchers on independent synthetic scenes.

These are mechanism ablations, not WorldRoPE/HRoPE reproductions. Same content
network, batches, optimizer, steps and validation checkpoint rule for each method.
Only scalar nonnegative geometry weights are added, with a random descriptor
capacity control. Test results never select checkpoints or weights.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from models.geometry import rotary
from experiments.synthetic import SCENARIOS, make_dataset


METHODS = ['appearance', 'center', 'reprojection', 'rope3d_axis',
           'expected_axis_support', 'expected_oblique_support',
           'expected_axis_uncertainty_oracle', 'center_normal', 'center_scale',
           'center_logcov', 'center_projected_logcov', 'center_bounded_logcov',
           'center_random_control']


class Matcher(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.method = method
        self.content = nn.Sequential(nn.Linear(16, 48), nn.GELU(), nn.Linear(48, 32))
        self.logtemp = nn.Parameter(torch.tensor(2.))
        # Register even for the appearance control: equal parameter count.
        self.center_weight = nn.Parameter(torch.tensor(-1.))
        self.shape_weight = nn.Parameter(torch.tensor(-1.))
        # Same frequency norms for axis and oblique variants; differ in direction.
        norms = torch.logspace(0, 1.3, 16)
        axes = torch.eye(3)[torch.arange(16) % 3]
        gen = torch.Generator().manual_seed(771)
        directions = F.normalize(torch.randn(16, 3, generator=gen), dim=-1)
        self.register_buffer('axis', axes * norms[:, None])
        self.register_buffer('oblique', directions * norms[:, None])

    def forward(self, data):
        q = F.normalize(self.content(data['q']), dim=-1)
        k = F.normalize(self.content(data['k']), dim=-1)
        name = self.method
        if name.startswith('rope3d_') or name.startswith('expected_'):
            cq = ck = None
            if 'support' in name:
                cq, ck = data['cq'], data['ck']
            elif 'uncertainty' in name:
                cq, ck = data['uqcov'], data['ukcov']
            freq = self.oblique if 'oblique' in name else self.axis
            q = rotary(q, data['mq'], freq, cq)
            k = rotary(k, data['mk'], freq, ck)
        score = self.logtemp.exp().clamp(max=50) * (q @ k.transpose(-1, -2))
        if name.startswith('center'):
            score = score - F.softplus(self.center_weight) * data['center']
        if name == 'reprojection':
            score = score - F.softplus(self.center_weight) * data['reprojection']
        which = {'center_normal': 'normal', 'center_scale': 'scale',
                 'center_logcov': 'shape', 'center_projected_logcov': 'projected',
                 'center_random_control': 'random', 'center_bounded_logcov': 'shape',
                 'rope3d_axis_normal': 'normal', 'rope3d_axis_scale': 'scale',
                 'rope3d_axis_logcov': 'shape', 'rope3d_axis_projected_logcov': 'projected',
                 'rope3d_axis_bounded_logcov': 'shape', 'rope3d_axis_moments': 'moments',
                 'rope3d_axis_random_control': 'random',
                 'expected_axis_uncertainty_oracle_logcov': 'shape'}
        if name in which:
            distance = data[which[name]]
            if 'bounded' in name:
                distance = distance.clamp(max=2.)
            score = score - F.softplus(self.shape_weight) * distance
        return score


def subset(data, index):
    return {k: v[index] for k, v in data.items()}


@torch.no_grad()
def evaluate(model, data, batch=32):
    hits, losses, recall5 = [], [], []
    for start in range(0, len(data['q']), batch):
        part = subset(data, slice(start, start+batch))
        score = model(part)
        truth = part['labels']
        hits.append((score.argmax(-1) == truth).float().mean(-1).cpu())
        recall5.append((score.topk(5, dim=-1).indices == truth[..., None]).any(-1).float().mean(-1).cpu())
        losses.append(F.cross_entropy(score.flatten(0, 1), truth.flatten(), reduction='none').reshape_as(truth).mean(-1).cpu())
    return {'r1': torch.cat(hits).numpy(), 'r5': torch.cat(recall5).numpy(), 'loss': torch.cat(losses).numpy()}


def digest(data):
    h = hashlib.sha256()
    for key in sorted(data):
        h.update(key.encode())
        h.update(data[key].contiguous().numpy().tobytes())
    return h.hexdigest()


def aggregate(raw, seeds, methods, scenarios):
    rows = []
    for scenario in scenarios:
        base = np.stack([raw[f'{s}/center/{scenario}/r1'] for s in seeds])
        for method in methods:
            values = np.stack([raw[f'{s}/{method}/{scenario}/r1'] for s in seeds])
            r5 = np.stack([raw[f'{s}/{method}/{scenario}/r5'] for s in seeds])
            means = values.mean(1)
            delta = values-base
            rng = np.random.default_rng(991)
            # Hierarchical paired bootstrap: resample seeds, then shared scenes.
            boots = []
            for _ in range(1000):
                si = rng.integers(0, len(seeds), len(seeds))
                ci = rng.integers(0, values.shape[1], (len(seeds), values.shape[1]))
                boots.append(delta[si[:, None], ci].mean())
            lo, hi = np.quantile(boots, [.025, .975])
            rows.append({'scenario': scenario, 'method': method,
                         'r1_mean': float(means.mean()), 'r1_seed_std': float(means.std(ddof=1)) if len(seeds)>1 else 0.,
                         'r5_mean': float(r5.mean()), 'delta_vs_center': float(delta.mean()),
                         'paired_bootstrap_95_low': float(lo), 'paired_bootstrap_95_high': float(hi)})
    return rows


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    methods = args.methods.split(',') if args.methods else METHODS
    seeds = [int(s) for s in args.seeds.split(',')]
    manifest = {'config': vars(args), 'methods': methods, 'scenarios': SCENARIOS,
                'torch': torch.__version__, 'numpy': np.__version__, 'python': platform.python_version(),
                'cuda_available': torch.cuda.is_available(), 'device': str(device),
                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'gpu_name': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
                'source_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for folder in ['models', 'experiments'] for p in sorted(Path(folder).glob('*.py'))},
                'dataset_hashes': {}, 'selection': [],
                'limitations': ['Synthetic paired support, no real images or geometry predictor.',
                                'No official HRoPE or WorldRoPE implementation in this experiment.',
                                'Uncertainty oracle receives injected position error covariance.',
                                'No multiple-comparison correction; bootstrap intervals exploratory.']}
    for filename, expected_hash in manifest['source_hashes'].items():
        source = Path(filename)
        target = out / 'source' / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        assert hashlib.sha256(target.read_bytes()).hexdigest() == expected_hash
    raw, histories = {}, []
    started = time.time()
    for seed in seeds:
        cpu_train = make_dataset(100000+seed, args.train_scenes)
        cpu_val = make_dataset(200000+seed, args.val_scenes)
        cpu_test = {s: make_dataset(300000+seed, args.test_scenes, scenario=s) for s in SCENARIOS}
        manifest['dataset_hashes'][str(seed)] = {'train': digest(cpu_train), 'val': digest(cpu_val),
                                               'test': {s: digest(d) for s, d in cpu_test.items()}}
        train = {k: v.to(device) for k, v in cpu_train.items()}
        val = {k: v.to(device) for k, v in cpu_val.items()}
        tests = {s: {k: v.to(device) for k, v in d.items()} for s, d in cpu_test.items()}
        batch_gen = torch.Generator().manual_seed(400000+seed)
        batches = torch.randint(0, args.train_scenes, (args.steps, args.batch), generator=batch_gen).to(device)
        for method in methods:
            torch.manual_seed(seed)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            model = Matcher(method).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.0001)
            best, best_state, best_step = -1., None, None
            train_start = time.time()
            for step in range(args.steps):
                part = subset(train, batches[step])
                logits = model(part)
                loss = F.cross_entropy(logits.flatten(0, 1), part['labels'].flatten())
                if not torch.isfinite(loss):
                    raise RuntimeError(f'nonfinite loss: {seed} {method} {step}')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad = nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                if (step+1) % args.val_every == 0 or step+1 == args.steps:
                    metrics = evaluate(model, val)
                    val_r1 = float(metrics['r1'].mean())
                    histories.append({'seed': seed, 'method': method, 'step': step+1,
                                      'train_loss': loss.item(), 'grad_norm': float(grad),
                                      'val_r1': val_r1, 'val_loss': float(metrics['loss'].mean())})
                    if val_r1 > best:
                        best, best_step = val_r1, step+1
                        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(best_state)
            for scenario, test in tests.items():
                scores = evaluate(model, test)
                for metric, values in scores.items():
                    raw[f'{seed}/{method}/{scenario}/{metric}'] = values
            checkpoint = out / 'checkpoints' / f'{seed}-{method}.pt'
            checkpoint.parent.mkdir(exist_ok=True)
            torch.save({'state_dict': model.cpu().state_dict(), 'method': method, 'seed': seed,
                        'best_step': best_step, 'args': vars(args)}, checkpoint)
            record = {'seed': seed, 'method': method, 'step': best_step, 'val_r1': best,
                      'center_weight': F.softplus(model.center_weight).item(),
                      'shape_weight': F.softplus(model.shape_weight).item(),
                      'parameters': sum(p.numel() for p in model.parameters()),
                      'train_and_eval_seconds': time.time()-train_start,
                      'peak_cuda_allocated_mb': torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None}
            manifest['selection'].append(record)
            print(json.dumps({**record, 'test_id_r1': float(raw[f'{seed}/{method}/id/r1'].mean())}), flush=True)
            np.savez_compressed(out / 'per_scene.npz', **raw)
            (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    rows = aggregate(raw, seeds, methods, SCENARIOS)
    with (out/'summary.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out/'summary.json').write_text(json.dumps(rows, indent=2)+'\n')
    (out/'training_history.json').write_text(json.dumps(histories, indent=2)+'\n')
    manifest['elapsed_seconds'] = time.time()-started
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'COMPLETE {out} elapsed={manifest["elapsed_seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--out', default='results/pilot')
    parser.add_argument('--seeds', default='11,22,33,44,55')
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--val-every', type=int, default=50)
    parser.add_argument('--train-scenes', type=int, default=512)
    parser.add_argument('--val-scenes', type=int, default=96)
    parser.add_argument('--test-scenes', type=int, default=192)
    parser.add_argument('--methods', default='')
    run(parser.parse_args())
