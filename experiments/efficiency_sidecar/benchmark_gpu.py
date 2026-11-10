"""Explicit opt-in GPU benchmark; optionally share only with recorded project jobs.

No polling daemon, training launch, global patch or checkpoint writes. Select an
GPU by UUID via CUDA_VISIBLE_DEVICES. Sharing requires --share-project-gpu.
"""
import argparse
import gc
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.efficiency_sidecar.check_cpu import old_inputs, model_for, variant, VARIANTS, ROLES
from experiments.round65_common import ROOT, read, write, sha


def resource_check(uuid, *, initialized=False, shared_peers=None):
    queue = read(ROOT / 'results/native_source_diversity/attempt1/queue_status.json')
    post = read(ROOT / 'results/native_source_diversity/attempt1/postprocess_status.json')
    if shared_peers is None and (queue['active'] or queue['pending'] or queue['state'] != 'ready_for_evaluation'):
        raise RuntimeError('Main training still has work; sidecar GPU work is deferred.')
    if shared_peers is None and (post['state'] not in ('ready_for_scientific_review', 'complete') or post.get('active_child')):
        raise RuntimeError('Main evaluation/postprocessing still has priority.')
    cards = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True, timeout=10)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                   '--format=csv,noheader,nounits'], text=True, timeout=10)
    card = next((line.split(',') for line in cards.splitlines() if line.split(',')[1].strip() == uuid), None)
    if card is None:
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES to one existing GPU UUID.')
    allowed = {os.getpid()} if initialized else set()
    if shared_peers is not None:
        allowed.update(shared_peers)
        for pid, start_ticks in shared_peers.items():
            stat = Path(f'/proc/{pid}/stat')
            if stat.exists() and int(stat.read_text().rsplit(')', 1)[1].split()[19]) != start_ticks:
                raise RuntimeError('Shared peer PID was reused; stop this sidecar.')
    if any(int(line.split(',')[1]) not in allowed for line in apps.splitlines()
           if line.split(',')[0].strip() == uuid):
        raise RuntimeError('Another process occupies this GPU; stop only this sidecar.')
    if shared_peers is not None:
        if int(card[3]) < (8 if initialized else 24) * 1024:
            raise RuntimeError('Insufficient headroom for the bounded shared pilot.')
    elif not initialized and (int(card[2]) > 512 or int(card[3]) < 24 * 1024 or int(card[4]) > 10):
        raise RuntimeError('This benchmark requires an entirely idle GPU with >=24 GiB free.')
    return dict(cards=cards, apps=apps, main_queue_state=queue['state'], main_postprocess_state=post['state'])


def run(destination, share_project_gpu=False):
    cpu = read(ROOT / 'results/efficiency_sidecar/cpu1/result.json')
    assert cpu['state'] == 'complete'
    for name, digest in cpu['sources'].items():
        assert sha(ROOT / name) == digest, name
    dependencies = read(ROOT / 'results/efficiency_sidecar/cpu1/dependencies.json')
    for name, digest in dependencies['sources'].items():
        assert sha(ROOT / name) == digest, name
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    peers = None
    if share_project_gpu:
        queue = read(ROOT / 'results/native_source_diversity/attempt1/queue_status.json')
        peers = {r['pid']: r['start_ticks'] for r in queue['active'] if r['gpu']['uuid'] == visible}
        if not peers:
            raise RuntimeError('Sharing is limited to a GPU with a recorded active project training job.')
    snapshot = resource_check(visible, shared_peers=peers)  # Before CUDA initialization.
    if destination.exists():
        raise FileExistsError('Preserve existing benchmark output.')
    destination.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    cap = (6 if share_project_gpu else 20) * 2**30
    torch.cuda.set_per_process_memory_fraction(cap / torch.cuda.get_device_properties(0).total_memory)
    probe = torch.ones(1, device='cuda')
    torch.cuda.synchronize()
    del probe
    resource_check(visible, initialized=True, shared_peers=peers)
    record = dict(started_unix=time.time(), pid=os.getpid(), gpu_uuid=visible,
        gpu_name=torch.cuda.get_device_name(), torch_version=torch.__version__, initial_resources=snapshot,
        precision='FP32 network, FP64 geometry, TF32 disabled', threads=1,
        benchmark_source_sha256=sha(__file__), cpu_result_sha256=sha(ROOT / 'results/efficiency_sidecar/cpu1/result.json'),
        dependency_identity_sha256=sha(ROOT / 'results/efficiency_sidecar/cpu1/dependencies.json'),
        sharing_authorized=share_project_gpu, shared_peer_start_ticks=peers, allocator_cap_bytes=cap,
        shared_timing_is_exploratory=share_project_gpu,
        resource_samples=[], pause_between_cells_seconds=.15 if share_project_gpu else 0,
        timing_scope='Resident-batch forward / loss+backward / loss+backward+clip+AdamW; excludes data IO and transfers.',
        no_scientific_fit_updated=True, no_new_confirmation=True, measurements=[], checks=[])
    write(destination / 'execution.json', record)
    try:
        data, labels, record['input_identity'] = old_inputs()
        lock = read(ROOT / 'results/modelnet_cross_source/attempt1/lock.json')
        checkpoints = {m['role']: m for m in lock['models'] if m['seed'] == 11 and m['role'] in ROLES}
        for role in ROLES:
            ck = checkpoints[role]
            assert sha(ck['path']) == ck['sha256']
            base = model_for(role)
            saved = torch.load(ck['path'], map_location='cpu', weights_only=True)['state_dict']
            base.load_state_dict(saved)
            cases = ((4, 512), (1, 1024)) if share_project_gpu else ((1, 512), (16, 512), (4, 128), (4, 256), (4, 512), (4, 1024))
            for batch, tokens in cases:
                resource_check(visible, initialized=True, shared_peers=peers)
                inputs = tuple(t[:batch, :tokens].cuda() for t in data)
                target = labels[:batch].cuda()
                # Paired checkpoint outputs and all parameter gradients on this GPU.
                reference_output = reference_gradient = None
                for name in VARIANTS:
                    resource_check(visible, initialized=True, shared_peers=peers)
                    model = variant(base, name).cuda().train()
                    torch.manual_seed(921112)
                    output = model(*inputs)
                    loss = F.cross_entropy(output, target, label_smoothing=.1)
                    loss.backward()
                    gradient = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).detach().cpu()
                    out = output.detach().cpu()
                    if name == 'original':
                        reference_output, reference_gradient = out, gradient
                    logit_error = float((out-reference_output).abs().max())
                    grad_error = float((gradient.double()-reference_gradient.double()).norm() / reference_gradient.double().norm())
                    record['checks'].append(dict(role=role, variant=name, batch=batch, tokens=tokens,
                        logit_max_abs=logit_error, gradient_relative_l2=grad_error))
                    assert logit_error < 2e-4 and grad_error < 1e-4
                    del model, output, loss, gradient
                    gc.collect()
                    torch.cuda.empty_cache()
                for mode in ('inference', 'forward_backward', 'optimizer_step'):
                    for block in range(4):
                        order = list(VARIANTS[block:]) + list(VARIANTS[:block])
                        for name in order:
                            observed = resource_check(visible, initialized=True, shared_peers=peers)
                            record['resource_samples'].append(dict(observed_unix=time.time(), role=role,
                                batch=batch, tokens=tokens, mode=mode, block=block, variant=name, **observed))
                            if time.time() - record['started_unix'] > 600:
                                raise TimeoutError('Bounded 10-minute sidecar budget reached; preserve partial results.')
                            model = variant(base, name).cuda().train(mode != 'inference')
                            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01) if mode == 'optimizer_step' else None
                            def step():
                                model.zero_grad(set_to_none=True)
                                if mode == 'inference':
                                    with torch.no_grad():
                                        model(*inputs)
                                else:
                                    F.cross_entropy(model(*inputs), target, label_smoothing=.1).backward()
                                    if optimizer is not None:
                                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                                        optimizer.step()
                            torch.manual_seed(921113 + block)
                            for _ in range(2):
                                step()
                            model.zero_grad(set_to_none=True)
                            torch.cuda.synchronize()
                            baseline = torch.cuda.memory_allocated()
                            torch.cuda.reset_peak_memory_stats()
                            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            wall = time.perf_counter()
                            start.record()
                            for _ in range(3):
                                step()
                            end.record()
                            torch.cuda.synchronize()
                            record['measurements'].append(dict(role=role, variant=name, batch=batch, tokens=tokens,
                                mode=mode, block=block, cuda_ms=start.elapsed_time(end)/3,
                                wall_ms=(time.perf_counter()-wall)*1000/3,
                                baseline_bytes=baseline, peak_bytes=torch.cuda.max_memory_allocated()))
                            del model, optimizer
                            gc.collect()
                            torch.cuda.empty_cache()
                            if share_project_gpu:
                                time.sleep(.15)
                        write(destination / 'progress.json', record)
                    print('BENCHMARK', role, batch, tokens, mode, flush=True)
                del inputs, target
        record.update(state='complete', elapsed_seconds=time.time()-record['started_unix'])
        write(destination / 'result.json', record)
    except BaseException as error:
        record.update(state='stopped', error_type=type(error).__name__, error=str(error))
        write(destination / 'partial.json', record)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/efficiency_sidecar/gpu1')
    parser.add_argument('--share-project-gpu', action='store_true',
                        help='User-authorized bounded shared pilot: 6 GiB cap, smaller batches, throttled cells.')
    args = parser.parse_args()
    run(args.output, args.share_project_gpu)
