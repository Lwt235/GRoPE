"""Paired GPU checks and whole-model costs of an algebraically equal PE path."""
import argparse
import copy
import gc
import os
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments import pe_runtime_diagnostic as previous
from experiments.round65_common import ROOT, read, write, sha
from experiments.round73_native_relative5 import local_dependencies
from experiments.region_confirmation_metric_runtime import BufferedNpz
from models.factored_relation_runtime import factorize

OUT = ROOT / 'results/factored_relation_runtime/attempt1'
UNITS = ROOT / 'results/factored_relation_runtime/preparation/units.json'
ROLES = previous.ROLES
IMPLEMENTATIONS = ('original', 'factored')


def freeze():
    old = previous.checked()
    units = read(UNITS)
    assert units['passed']
    for path, digest in units['source_sha256'].items():
        assert sha(path) == digest
    assert not (OUT / 'plan.json').exists()
    sources = dict(old['sources'])
    for path in local_dependencies([Path(__file__), ROOT / 'tests/test_factored_relation_runtime.py']) + [UNITS, previous.OUT / 'plan.json', previous.OUT / 'review.json']:
        sources[str(path)] = sha(path)
    write(OUT / 'plan.json', dict(created_unix=time.time(), sources=sources, checkpoints=old['checkpoints'],
        hypothesis='Moving the final bias-free relation projection onto queries preserves the model function while reducing realized pairwise activation cost.',
        competing_explanations=['FP32 reorder changes outputs or gradients materially.', 'Larger per-head hidden contractions negate the apparent storage/MAC saving.'],
        comparison=dict(tasks=['region', 'native'], roles=ROLES, implementations=IMPLEMENTATIONS,
            checkpoints='Same seed11 selected models as the preceding cost diagnostic.',
            replay='All 64 old regional validation pairs and all 765 old native validation objects, original batch sizes, original predictions retained.',
            backward='Same old first16 validation examples, original loss, identical dropout seed; compare every parameter gradient and unchanged state.',
            cost='batch16, region64 tokens per view/native512 primitives; original FP32 network/FP64 geometry; TF32 off.',
            warmup_calls=5, blocks=8, calls_per_block=3, order='Cyclic order across all eight role/implementation combinations.'),
        numerical_screen=dict(old_validation_max_logit_error=2e-4, prediction_changes=0,
            full_batch_loss_difference=1e-4, global_gradient_relative_norm=1e-4,
            position_gradient_relative_norm=1e-4,
            interpretation='Screen a possible future implementation at existing precision; not bitwise equivalence or a proof of identical long training trajectories. Retain discrepancies and costs even if this screen fails.'),
        decision='If outputs/gradients agree and measured costs improve, retain the adapter as an implementation option for prospective runs. Apply it fairly to all compatible controls. Otherwise keep the original implementation. Never replace frozen fit states or confirmation predictions.',
        scope='Fixed observed geometry only, no geometry-input gradients, no BF16/RGB/NVS claim. Inference or forward+loss+backward; excludes data loading, optimizer, and full training-cache VRAM. One batch, one seed, one GPU.',
        resource='One whole-idle GPU with >=34GiB free; CUDA probe and occupancy recheck. Preserve all active native fits.',
        new_fits=0, optimizer_updates=0, fresh_data_access=False, scientific_candidate_changed=False, goal_complete=False))
    (OUT / 'plan.sha256').write_text(sha(OUT / 'plan.json') + '\n')
    print('FACTORED_PLAN_FROZEN', sha(OUT / 'plan.json'), flush=True)


def checked():
    plan = read(OUT / 'plan.json')
    assert sha(OUT / 'plan.json') == (OUT / 'plan.sha256').read_text().strip()
    for path, digest in plan['sources'].items():
        assert sha(path) == digest, path
    for item in plan['checkpoints'].values():
        assert sha(item['path']) == item['sha256']
    return plan


def validation_cache(task):
    root = previous.ROOTS[task + '_old']
    manifest = read(root / 'data/manifest.json')
    if task == 'region':
        row = manifest['caches']['val']
        assert sha(row['path']) == row['sha256']
        with BufferedNpz(row['path']) as z:
            return {k: z[k] for k in z.files}
    row = next(r for r in manifest['partitions'] if r['split'] == 'val')
    result = {}
    for name in ('centers', 'covariance', 'appearance', 'labels'):
        path = root / 'data/val' / (name + '.npy')
        assert sha(path) == row['hashes'][name + '.npy']
        result[name] = np.load(path, mmap_mode='r', allow_pickle=False)
    result['ids'] = np.asarray(row['ids'])
    return result


def make_model(task, role):
    if task == 'native':
        return previous.CompactNativeClassifier(role) if role == 'relative5' else previous.NativeGSClassifier(role)
    return previous.CompactMatcher(role) if role == 'relative5' else previous.FixedRegionMatcher(role)


def backward_check(task, models, data, target):
    records = []
    for role in ROLES:
        losses, gradients = [], []
        for implementation in IMPLEMENTATIONS:
            model = models[role, implementation]
            model.train()
            model.zero_grad(set_to_none=True)
            torch.manual_seed(731109)
            output = model(*data)
            loss = F.cross_entropy(output, target, label_smoothing=.1) if task == 'native' else previous.region_engine.loss_function(output, target)
            loss.backward()
            losses.append(float(loss.detach()))
            gradients.append({name: None if p.grad is None else p.grad.detach().cpu().clone() for name, p in model.named_parameters()})
        cells, totals = [], dict(all=[0., 0.], position=[0., 0.])
        for name in gradients[0]:
            a, b = gradients[0][name], gradients[1][name]
            if a is None or b is None:
                assert a is b, (task, role, name)
                continue
            assert bool(torch.isfinite(a).all() and torch.isfinite(b).all()), (task, role, name)
            delta = (a.double() - b.double())
            d2, a2 = float(delta.square().sum()), float(a.double().square().sum())
            cells.append(dict(parameter=name, maximum_absolute_error=float(delta.abs().max()),
                relative_norm=(d2 / max(a2, 1e-30))**.5, original_norm=a2**.5))
            for group in ['all'] + (['position'] if name.startswith('position.') else []):
                totals[group][0] += d2
                totals[group][1] += a2
        global_error = (totals['all'][0] / max(totals['all'][1], 1e-30))**.5
        pe_error = (totals['position'][0] / max(totals['position'][1], 1e-30))**.5
        records.append(dict(task=task, role=role, losses=losses, loss_difference=abs(losses[0] - losses[1]),
            global_gradient_relative_norm=global_error, position_gradient_relative_norm=pe_error,
            screen_passed=abs(losses[0] - losses[1]) < 1e-4 and max(global_error, pe_error) < 1e-4,
            parameters=cells))
        print('FACTORED_BACKWARD', task, role, records[-1]['screen_passed'], global_error, pe_error, flush=True)
    return records


def run():
    plan = checked()
    assert not (OUT / 'execution.json').exists()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    before = previous.gpu_snapshot()
    row = next(line.split(',') for line in before.splitlines() if line.split(',')[1].strip() == visible)
    assert int(row[2]) <= 512 and int(row[3]) >= 34 * 1024 and int(row[4]) <= 10
    previous.native_engine.configure()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    probe = torch.ones(4, device='cuda')
    assert float(probe.sum()) == 4
    torch.cuda.synchronize()
    del probe
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True, timeout=10)
    assert all(int(line.split(',')[1]) == os.getpid() for line in apps.splitlines() if line.split(',')[0].strip() == visible)
    identity = dict(pid=os.getpid(), started_unix=time.time(), gpu_uuid=visible, gpu_name=torch.cuda.get_device_name(),
        torch_version=torch.__version__, gpu_before=before, plan_sha256=sha(OUT / 'plan.json'))
    write(OUT / 'execution.json', identity)
    blocks, replay, backward, input_identity = [], [], [], {}
    try:
        for task in ('region', 'native'):
            print('FACTORED_INPUT', task, flush=True)
            data, target, input_identity[task] = previous.input_batch(task)
            full = validation_cache(task)
            models, initial = {}, {}
            for role in ROLES:
                checkpoint = plan['checkpoints'][f'{task}/{role}']
                original = make_model(task, role)
                original.load_state_dict(torch.load(checkpoint['path'], map_location='cpu', weights_only=True)['state_dict'])
                alternative = factorize(copy.deepcopy(original), task)
                for implementation, model in zip(IMPLEMENTATIONS, (original, alternative)):
                    model = model.cuda().eval()
                    models[role, implementation] = model
                    initial[role, implementation] = previous.state_digest(model)
                assert initial[role, 'original'] == initial[role, 'factored']
                outputs = []
                for implementation in IMPLEMENTATIONS:
                    evaluator = previous.native_engine.evaluate if task == 'native' else previous.region_engine.evaluate
                    logits, _ = evaluator(models[role, implementation], full)
                    outputs.append(torch.from_numpy(logits))
                with BufferedNpz(checkpoint['validation']) as z:
                    assert np.array_equal(z['ids'], full['ids'])
                    expected = torch.from_numpy(z['logits'])
                    replay_error = float((outputs[0] - expected).abs().max())
                assert replay_error < 2e-4 and bool(torch.equal(outputs[0].argmax(-1), expected.argmax(-1)))
                error = float((outputs[0] - outputs[1]).abs().max())
                changes = int((outputs[0].argmax(-1) != outputs[1].argmax(-1)).sum())
                replay.append(dict(task=task, role=role, examples=len(full['ids']),
                    original_stored_replay_error=replay_error, maximum_logit_error=error,
                    prediction_changes=changes, screen_passed=error < 2e-4 and changes == 0))
                write(OUT / 'replay_progress.json', dict(records=replay, observed_unix=time.time()))
                print('FACTORED_REPLAY', task, role, len(full['ids']), error, changes, flush=True)
            backward.extend(backward_check(task, models, data, target))
            write(OUT / 'backward_progress.json', dict(records=backward))
            combinations = [(role, implementation) for role in ROLES for implementation in IMPLEMENTATIONS]
            for mode in ('inference', 'forward_backward'):
                def step(key):
                    model = models[key]
                    model.zero_grad(set_to_none=True)
                    if mode == 'inference':
                        with torch.no_grad():
                            model(*data)
                    else:
                        output = model(*data)
                        loss = F.cross_entropy(output, target, label_smoothing=.1) if task == 'native' else previous.region_engine.loss_function(output, target)
                        loss.backward()

                for model in models.values():
                    model.train(mode == 'forward_backward')
                    model.zero_grad(set_to_none=True)
                for key in combinations:
                    for _ in range(5):
                        step(key)
                for block in range(8):
                    order = combinations[block:] + combinations[:block]
                    for role, implementation in order:
                        for model in models.values():
                            model.zero_grad(set_to_none=True)
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        baseline = torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats()
                        torch.manual_seed(731110 + block)
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        wall = time.perf_counter()
                        start.record()
                        for _ in range(3):
                            step((role, implementation))
                        end.record()
                        torch.cuda.synchronize()
                        blocks.append(dict(task=task, role=role, implementation=implementation, mode=mode, block=block,
                            cuda_ms_per_call=start.elapsed_time(end) / 3,
                            wall_ms_per_call=(time.perf_counter() - wall) * 1000 / 3,
                            baseline_allocated_bytes=baseline, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            incremental_peak_bytes=torch.cuda.max_memory_allocated() - baseline))
                    write(OUT / 'cost_progress.json', dict(last=blocks[-1], completed_blocks=len(blocks), observed_unix=time.time()))
                    print('FACTORED_COST', task, mode, block, len(blocks), flush=True)
            for key, model in models.items():
                assert previous.state_digest(model) == initial[key], key
            del models, original, alternative, model, full, data, target, logits, outputs, expected
            gc.collect()
            torch.cuda.empty_cache()
        summary = []
        for task in ('region', 'native'):
            for mode in ('inference', 'forward_backward'):
                for role in ROLES:
                    for implementation in IMPLEMENTATIONS:
                        group = [r for r in blocks if (r['task'], r['mode'], r['role'], r['implementation']) == (task, mode, role, implementation)]
                        times = [r['cuda_ms_per_call'] for r in group]
                        summary.append(dict(task=task, mode=mode, role=role, implementation=implementation,
                            median_cuda_ms=statistics.median(times), min_cuda_ms=min(times), max_cuda_ms=max(times),
                            peak_gib=max(r['peak_allocated_bytes'] for r in group) / 2**30,
                            incremental_peak_gib=max(r['incremental_peak_bytes'] for r in group) / 2**30))
        write(OUT / 'result.json', dict(execution=identity, completed_unix=time.time(), replay=replay, backward=backward,
            all_numerical_screens_passed=all(r['screen_passed'] for r in replay + backward),
            blocks=blocks, summary=summary, input_identity=input_identity, model_states_unchanged=True,
            optimizer_updates=0, gpu_after=previous.gpu_snapshot(), scope=plan['scope'], review_pending=True,
            scientific_effectiveness_claim=False, current_fits_or_predictions_changed=False, goal_complete=False))
        print('FACTORED_COMPLETE', len(blocks), 'blocks', flush=True)
    except BaseException as error:
        write(OUT / 'failure.json', dict(error_type=type(error).__name__, message=str(error),
            completed_cost_blocks=len(blocks), completed_replays=len(replay), observed_unix=time.time()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze', 'run'))
    globals()[parser.parse_args().action]()
