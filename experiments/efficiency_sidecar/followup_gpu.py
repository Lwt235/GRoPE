"""Noise-controlled recheck and segmented pipeline profile on private models."""
import argparse
import gc
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.efficiency_sidecar.benchmark_gpu import resource_check
from experiments.efficiency_sidecar.check_cpu import old_inputs, model_for, variant, state_hash
from experiments.efficiency_sidecar.adapters import quadratic_descriptors
from experiments.round65_common import ROOT, read, write, sha
from experiments.round65_run import augment, epoch_batches
from models.round65_actual_gs import common_features
from models.round69_compact_relation import descriptors

OUT = ROOT / 'results/efficiency_sidecar/followup1'


def initialize(stage):
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / f'{stage}.json').exists() or (OUT / f'{stage}_execution.json').exists():
        raise FileExistsError('Preserve completed or interrupted stage.')
    for file in ('result.json', 'dependencies.json'):
        identity = read(ROOT / 'results/efficiency_sidecar/cpu1' / file)
        for p, d in identity['sources'].items():
            assert sha(ROOT / p) == d, p
    visible = os.environ['CUDA_VISIBLE_DEVICES']
    queue = read(ROOT / 'results/native_source_diversity/attempt1/queue_status.json')
    peers = {r['pid']: r['start_ticks'] for r in queue['active'] if r['gpu']['uuid'] == visible}
    assert peers, 'Select a recorded project training GPU.'
    snapshot = resource_check(visible, shared_peers=peers)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.cuda.set_per_process_memory_fraction(6*2**30/torch.cuda.get_device_properties(0).total_memory)
    torch.ones(1, device='cuda').sum().item()
    resource_check(visible, initialized=True, shared_peers=peers)
    record = dict(stage=stage, started_unix=time.time(), pid=os.getpid(), gpu_uuid=visible,
        torch_version=torch.__version__, gpu_name=torch.cuda.get_device_name(), resources=snapshot,
        peer_start_ticks=peers, allocator_cap_bytes=6*2**30, cpu_threads=1,
        shared_gpu=True, scientific_models_updated=False, measurements=[], resource_samples=[],
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in Path(__file__).parent.glob('*.py')},
        plan_sha256=sha(Path(__file__).parent / 'FOLLOWUP_PLAN.md'))
    write(OUT / f'{stage}_execution.json', record)
    return record


def check_resources(record):
    if time.time()-record['started_unix'] > 600:
        raise TimeoutError('Ten-minute stage budget reached.')
    observed = resource_check(record['gpu_uuid'], initialized=True,
                              shared_peers=record['peer_start_ticks'])
    record['resource_samples'].append(dict(observed_unix=time.time(), **observed))


def base_model(role):
    lock = read(ROOT / 'results/modelnet_cross_source/attempt1/lock.json')
    ck = next(m for m in lock['models'] if m['role'] == role and m['seed'] == 11)
    assert sha(ck['path']) == ck['sha256']
    model = model_for(role)
    model.load_state_dict(torch.load(ck['path'], map_location='cpu', weights_only=True)['state_dict'])
    return model, ck


def finish(record):
    record.update(state='complete', elapsed_seconds=time.time()-record['started_unix'])
    write(OUT / f"{record['stage']}.json", record)


def replicate():
    r = initialize('replicate')
    try:
        cpu, target, r['input_identity'] = old_inputs(4, 512)
        inputs, target = tuple(t.cuda() for t in cpu), target.cuda()
        base, r['checkpoint'] = base_model('relative5')
        labels = ('baseline_a', 'candidate_a', 'baseline_b', 'candidate_b')
        for mode in ('inference', 'optimizer_step'):
            for block in range(8):
                order = labels[block % 4:] + labels[:block % 4]
                for label in order:
                    check_resources(r)
                    name = 'factored' if label.startswith('baseline') else 'geometry_factored'
                    model = variant(base, name).cuda().train(mode != 'inference')
                    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01) if mode != 'inference' else None
                    def step():
                        model.zero_grad(set_to_none=True)
                        if opt is None:
                            with torch.no_grad():
                                model(*inputs)
                        else:
                            loss = F.cross_entropy(model(*inputs), target, label_smoothing=.1)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                            opt.step()
                    torch.manual_seed(924000+block)
                    for _ in range(2): step()
                    model.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    wall = time.perf_counter(); a.record()
                    for _ in range(5): step()
                    b.record(); torch.cuda.synchronize()
                    r['measurements'].append(dict(mode=mode, block=block, label=label, implementation=name,
                        cuda_ms=a.elapsed_time(b)/5, wall_ms=(time.perf_counter()-wall)*1000/5,
                        peak_bytes=torch.cuda.max_memory_allocated()))
                    del model, opt
                    gc.collect(); torch.cuda.empty_cache(); time.sleep(.15)
                write(OUT / 'replicate_progress.json', r)
                print('REPLICATE', mode, block, flush=True)
        finish(r)
    except BaseException as e:
        r.update(state='failed', error_type=type(e).__name__, error=str(e))
        write(OUT / 'replicate_failure.json', r)
        raise


def make_schedule(count=16, steps=32):
    rng = np.random.default_rng(924001)
    result = []
    start = time.perf_counter()
    while len(result) < steps:
        result.extend(epoch_batches(rng, count, batch_size=4, tokens=512))
    return result[:steps], (time.perf_counter()-start)*1000/steps


def gather_cpu(pool, target, spec, cache=None):
    index, subset, quaternion, jitter = spec
    values = [np.array(t.numpy()[index[:, None], subset], copy=True) for t in pool]
    y = np.array(target.numpy()[index], copy=True)
    extra = {} if cache is None else {k: np.array(t.numpy()[index[:, None], subset], copy=True) for k, t in cache.items()}
    return values, y, extra


def descriptor(model, name, c, s):
    if model.role == 'distance_mlp':
        return model.position.descriptors(c, s)
    return quadratic_descriptors(c, s) if name == 'geometry_factored' else descriptors('relative5', c, s)


def network(model, content, relative):
    x = model.stem(content)
    for layer in model.layers:
        x = layer(x, relative)
    x = model.final_norm(x)
    return model.classifier(torch.cat((x.mean(1), x.max(1).values), -1))


def profile():
    r = initialize('profile')
    try:
        pool, target, r['input_identity'] = old_inputs(16, 2048)
        schedule, r['schedule_generation_ms_per_step'] = make_schedule()
        r['scope'] = 'Old validation hot-memory pool, fixed schedule; no source mixing, cold IO, checkpointing or validation.'
        r['split_forward_checks'] = []
        for role in ('relative5', 'distance_mlp'):
            base, ck = base_model(role)
            for name in ('original', 'factored', 'geometry_factored'):
                check_resources(r)
                probe = variant(base, name).cuda().eval()
                batch = tuple(t[:4, :512].cuda() for t in pool)
                with torch.no_grad():
                    a = probe(*batch)
                    b = network(probe, common_features(*batch), probe.position.mlp(descriptor(probe, name, *batch[:2])))
                error = float((a-b).abs().max())
                assert error < 2e-4
                r['split_forward_checks'].append(dict(role=role, implementation=name, max_logit_abs=error, checkpoint=ck))
                del probe, a, b, batch
                for segmented in (True, False):
                    check_resources(r)
                    model = variant(base, name).cuda().train()
                    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
                    for step, spec in enumerate(schedule):
                        if step % 8 == 0: check_resources(r)
                        torch.manual_seed(924100+step)
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        stage = {}
                        def call(key, fn):
                            if not segmented: return fn()
                            torch.cuda.synchronize()
                            t = time.perf_counter(); value = fn(); torch.cuda.synchronize()
                            stage[key] = (time.perf_counter()-t)*1000
                            return value
                        values, y, _ = call('cpu_gather', lambda: gather_cpu(pool, target, spec))
                        c, s, a, y = call('h2d', lambda: tuple(torch.tensor(v, device='cuda') for v in (*values, y)))
                        c, s = call('augment', lambda: augment(c, s, spec[2], spec[3]))
                        model.zero_grad(set_to_none=True)
                        if segmented:
                            common = call('common_geometry', lambda: common_features(c, s, a))
                            desc = call('pe_descriptor', lambda: descriptor(model, name, c, s))
                            relation = call('pe_mlp', lambda: model.position.mlp(desc))
                            output = call('network_forward', lambda: network(model, common, relation))
                        else:
                            output = model(c, s, a)
                        loss = call('loss', lambda: F.cross_entropy(output, y, label_smoothing=.1))
                        if not bool(torch.isfinite(loss)): raise ValueError('Nonfinite loss.')
                        call('backward', loss.backward)
                        def update():
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                            opt.step()
                        call('optimizer', update)
                        value = float(loss)
                        torch.cuda.synchronize()
                        r['measurements'].append(dict(role=role, implementation=name, segmented=segmented,
                            step=step, warmup=step<4, wall_ms=(time.perf_counter()-start)*1000,
                            stage_ms=stage, loss=value))
                        if segmented: del common, desc, relation
                        del c, s, a, y, output, loss
                        time.sleep(.05)
                    del model, opt
                    gc.collect(); torch.cuda.empty_cache()
                    write(OUT / 'profile_progress.json', r)
                    print('PROFILE', role, name, segmented, flush=True)
        finish(r)
    except BaseException as e:
        r.update(state='failed', error_type=type(e).__name__, error=str(e))
        write(OUT / 'profile_failure.json', r)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('replicate', 'profile'))
    args = parser.parse_args()
    globals()[args.stage]()
