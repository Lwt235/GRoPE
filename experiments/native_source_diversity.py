"""Unchanged native classifiers with fixed, class-preserving source diversity."""
import argparse
import ast
import hashlib
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments import round65_run as runner
from experiments.native_source_diversity_data import OUT, DATA, OLD_DATA, PLAN, check as check_data
from experiments.native_source_diversity_sampling import SourceSchedule, gather, selection_ce, CLASSES
from experiments.round65_common import ROOT, CONFIG, read, write, sha
from models.round65_actual_gs import NativeGSClassifier
from models.round73_native_relative5 import CompactNativeClassifier
from models.native_shape_attribution import NativeFixedIsotropic

ROLES = ('relative5', 'distance_mlp', 'fixed_isotropic5')
SEEDS = (11, 22, 33)
CAP_BYTES = 14 * 2**30
FIELDS = ('centers', 'covariance', 'appearance', 'labels')


def model_for(role):
    if role == 'relative5':
        return CompactNativeClassifier(role)
    if role == 'distance_mlp':
        return NativeGSClassifier(role)
    if role == 'fixed_isotropic5':
        return NativeFixedIsotropic(role)
    raise ValueError(role)


def source_files():
    pending = [Path(__file__), ROOT / 'experiments/native_source_diversity_preflight.py',
               ROOT / 'experiments/native_source_diversity_queue.py',
               ROOT / 'tests/test_native_source_diversity_sampling.py']
    found = set()
    while pending:
        path = pending.pop().resolve()
        if path in found:
            continue
        found.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names = ([n.name for n in node.names] if isinstance(node, ast.Import) else
                     [node.module] + [node.module + '.' + n.name for n in node.names]
                     if isinstance(node, ast.ImportFrom) and node.module else [])
            for name in names:
                candidate = ROOT / (name.replace('.', '/') + '.py')
                if name.split('.')[0] in ('models', 'experiments') and candidate.is_file():
                    pending.append(candidate)
    return sorted(found)


def freeze():
    check_data()
    assert not (OUT / 'lock.json').exists()
    ready = read(OUT / 'cache_ready.json')
    assert ready['objects'] == 2657 and ready['counts_by_partition'] == dict(fit=2133, val=262, dev_eval=262)
    assert ready['manifest_sha256'] == sha(DATA / 'cache/manifest.json')
    assert ready['catalog_sha256'] == sha(OUT / 'catalog.json')
    assert read(OUT / 'data_runtime.json')['state'] == 'complete'
    cpu, cuda = read(OUT / 'cpu_checks.json'), read(OUT / 'cuda_preflight.json')
    assert cpu['passed'] and cuda['passed']
    for path, digest in cpu['source_sha256'].items():
        assert sha(ROOT / path) == digest
    assert cuda['source_sha256'] == sha(ROOT / 'experiments/native_source_diversity_preflight.py')
    assert cuda['sampler_source_sha256'] == sha(ROOT / 'experiments/native_source_diversity_sampling.py')
    old_models = [m for m in read(ROOT / 'results/modelnet_cross_source/attempt1/lock.json')['models']
                  if m['role'] in ROLES and m['seed'] in SEEDS]
    assert len(old_models) == 9
    for model in old_models:
        assert sha(model['path']) == model['sha256']
    paths = source_files() + [PLAN, OUT / 'preparation_lock.json', OUT / 'catalog.json',
        OUT / 'cache_ready.json', OUT / 'cpu_checks.json', OUT / 'cuda_preflight.json',
        OLD_DATA / 'manifest.json', DATA / 'cache/manifest.json']
    record = dict(created_unix=time.time(), source_sha256={str(p): sha(p) for p in paths},
        config=CONFIG, roles=list(ROLES), seeds=list(SEEDS), old_models=old_models,
        old_run_sha256={str(Path(m['path']).parent / 'run.json'): sha(Path(m['path']).parent / 'run.json') for m in old_models},
        preparation_lock_sha256=sha(OUT / 'preparation_lock.json'),
        source_replacement_probability=.5, source_rng_seed_sequence_prefix=92000,
        original_slots_per_epoch=6227, steps_per_epoch=390, total_steps=46800,
        selected_by='class_then_source_equal_validation_CE', evaluation_conditions=['base'],
        old_native_validation_objects=765, additional_validation_objects=262,
        old_native_development_objects=765, additional_development_objects=262,
        primary_classes=list(CLASSES), bootstrap_replicates=10000, primary_bootstrap_seed=92001,
        maximum_gpu_jobs=6, maximum_shared_gpu_jobs=2, shared_minimum_free_mib=16384,
        allocator_cap_bytes=CAP_BYTES, cpu_batch_gather=True, new_fits=9,
        confirmation_data_reused=False, fresh_confirmation=False, goal_complete=False)
    write(OUT / 'lock.json', record)
    (OUT / 'lock.sha256').write_text(sha(OUT / 'lock.json') + '\n')
    print('SOURCE_DIVERSITY_TRAINING_SEALED', sha(OUT / 'lock.json'), flush=True)


def check():
    lock = read(OUT / 'lock.json')
    assert sha(OUT / 'lock.json') == (OUT / 'lock.sha256').read_text().strip()
    assert lock['config'] == CONFIG and lock['roles'] == list(ROLES) and lock['seeds'] == list(SEEDS)
    for name, digest in lock['source_sha256'].items():
        assert sha(name) == digest, name
    for name, digest in lock['old_run_sha256'].items():
        assert sha(name) == digest, name
    return lock


def old_bundle(split):
    manifest = read(OLD_DATA / 'manifest.json')
    record = next(r for r in manifest['partitions'] if r['split'] == split)
    directory = OLD_DATA / split
    for name, digest in record['hashes'].items():
        assert sha(directory / name) == digest
    bundle = {key: np.load(directory / (key + '.npy'), mmap_mode='r', allow_pickle=False) for key in FIELDS}
    assert len(bundle['labels']) == record['count']
    bundle['ids'] = np.asarray(record['ids'])
    bundle['sources'] = np.where(bundle['labels'] < 30, 'MACGS', 'ModelNet')
    return bundle


def extra_bundle():
    directory = DATA / 'cache'
    manifest = read(directory / 'manifest.json')
    for name, digest in manifest['array_sha256'].items():
        assert sha(directory / name) == digest
    rows = read(OUT / 'catalog.json')['rows']
    bundle = {key: np.load(directory / (key + '.npy'), mmap_mode='r', allow_pickle=False) for key in FIELDS}
    assert manifest['ids'] == [r['file_path'] for r in rows]
    assert np.array_equal(bundle['labels'], [r['taxonomy_id'] for r in rows])
    bundle['ids'] = np.asarray(manifest['ids'])
    bundle['sources'] = np.full(len(rows), 'ModelNet')
    return bundle, rows


def merged_population(split, extra, rows):
    if split not in ('val', 'dev_eval'):
        raise ValueError('Only fixed selection/evaluation populations may be merged.')
    if split == 'dev_eval' and not (OUT / 'selected.json').exists():
        raise ValueError('Seal all nine models before opening development predictions.')
    old = old_bundle(split)
    chosen = np.array([i for i, row in enumerate(rows) if row['split'] == split])
    result = {key: np.concatenate([old[key], extra[key][chosen]], axis=0) for key in FIELDS}
    result['ids'] = np.concatenate([old['ids'], extra['ids'][chosen]])
    result['sources'] = np.concatenate([old['sources'], extra['sources'][chosen]])
    result['original_native'] = np.arange(len(result['labels'])) < len(old['labels'])
    assert len(result['labels']) == 1027 and len(set(result['ids'])) == 1027
    return result


def setup_cuda():
    runner.configure()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    torch.cuda.set_per_process_memory_fraction(CAP_BYTES / torch.cuda.get_device_properties(0).total_memory)


def train(role, seed):
    lock = check()
    assert role in ROLES and seed in SEEDS
    setup_cuda()
    dest = OUT / 'runs' / role / str(seed)
    assert not (dest / 'run.json').exists()
    dest.mkdir(parents=True, exist_ok=True)
    write(dest / f'environment_{time.time_ns()}.json', dict(pid=os.getpid(), started_unix=time.time(),
        torch_version=torch.__version__, numpy_version=np.__version__, gpu=torch.cuda.get_device_name(),
        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
        CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'), allocator_cap_bytes=CAP_BYTES,
        lock_sha256=sha(OUT / 'lock.json')))
    old = old_bundle('fit')
    extra, rows = extra_bundle()
    val = merged_population('val', extra, rows)
    assert len(old['labels']) == lock['original_slots_per_epoch']
    count, steps_per_epoch = len(old['labels']), lock['steps_per_epoch']
    assert math.ceil(count / CONFIG['batch_size']) == steps_per_epoch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = model_for(role).cuda()
    initial_shared = runner.shared_digest(model)
    initial_position = {n: p.detach().cpu().clone() for n, p in model.position.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    rng = np.random.default_rng(seed + 65000)
    schedule = SourceSchedule(seed, rows)
    raw_stream = hashlib.sha256(('R65-sampler-v1:' + str(seed)).encode()).hexdigest()
    effective_stream = hashlib.sha256(('native-source-diversity-v1:' + str(seed)).encode()).hexdigest()
    completed = step = 0
    history, training = [], []
    best, best_epoch, best_state, best_logits = float('inf'), None, None, None
    first_gradients, elapsed_before, peak_before, resumed = {}, 0., 0, False
    resume = dest / 'resume.pt'
    lock_sha = sha(OUT / 'lock.json')
    if resume.exists():
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        assert (saved['role'], saved['seed'], saved['lock_sha256']) == (role, seed, lock_sha)
        completed, step = saved['epoch_completed'], saved['step']
        assert step == completed * steps_per_epoch
        model.load_state_dict(saved['state_dict']); optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state = saved['sampler_state']
        schedule = SourceSchedule(seed, rows, state=saved['source_sampler_state'])
        raw_stream, effective_stream = saved['raw_sampling_sha256'], saved['effective_sampling_sha256']
        history, training = saved['validation'], saved['training']
        best, best_epoch, best_state, best_logits = saved['best'], saved['best_epoch'], saved['best_state'], saved['best_logits']
        first_gradients, elapsed_before, peak_before = saved['first_gradients'], saved['seconds'], saved['peak_allocated']
        resumed = True
        write(dest / f'resume_{time.time_ns()}.json', dict(snapshot_sha256=sha(resume), epoch_completed=completed, steps=step))
        runner.atomic_save(dest / 'best.pt', dict(role=role, seed=seed, epoch=best_epoch, state_dict=best_state))
        np.savez(dest / 'validation.npz', logits=best_logits, labels=val['labels'], ids=val['ids'], sources=val['sources'])
    started = time.monotonic()
    deadline = started + CONFIG['shard_seconds']
    torch.cuda.reset_peak_memory_stats()

    def select(epoch):
        nonlocal best, best_epoch, best_state, best_logits
        logits, metrics = runner.evaluate(model, val, 'base')
        criterion = selection_ce(logits, val['labels'], val['sources'])
        history.append(dict(epoch=epoch, step=step, selection_ce=criterion, **metrics))
        if criterion < best:
            best, best_epoch, best_logits = criterion, epoch, logits
            best_state = {n: p.detach().cpu().clone() for n, p in model.state_dict().items()}
            runner.atomic_save(dest / 'best.pt', dict(role=role, seed=seed, epoch=epoch, state_dict=best_state))
            np.savez(dest / 'validation.npz', logits=logits, labels=val['labels'], ids=val['ids'], sources=val['sources'])
        write(dest / 'history.json', dict(validation=history, training=training))
        print('VAL', role, seed, epoch, criterion, metrics['mca'], flush=True)

    def save():
        runner.atomic_save(resume, dict(role=role, seed=seed, lock_sha256=lock_sha,
            epoch_completed=completed, step=step, state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
            optimizer=optimizer.state_dict(), sampler_state=rng.bit_generator.state,
            source_sampler_state=schedule.state(), raw_sampling_sha256=raw_stream,
            effective_sampling_sha256=effective_stream, validation=history, training=training,
            best=best, best_epoch=best_epoch, best_state=best_state, best_logits=best_logits,
            first_gradients=first_gradients, seconds=elapsed_before + time.monotonic() - started,
            peak_allocated=max(peak_before, torch.cuda.max_memory_allocated())))

    try:
        if not resumed:
            select(0); save()
        for epoch in range(completed, CONFIG['epochs']):
            total_loss, replaced = 0., {str(c): 0 for c in CLASSES}
            for index, subset, quaternion, jitter in runner.epoch_batches(rng, count):
                raw_stream = runner.stream_update(raw_stream, index, subset, quaternion, jitter)
                mapped = schedule.map(index, old['labels'])
                values = gather(old, extra, mapped, subset)
                assert np.array_equal(values['labels'], old['labels'][index])
                effective_stream = runner.stream_update(effective_stream, index, mapped, subset, quaternion, jitter)
                for label in CLASSES:
                    replaced[str(label)] += int(((values['labels'] == label) & (mapped[:, 0] == 1)).sum())
                c, s, a = [torch.tensor(values[key], device='cuda') for key in ('centers', 'covariance', 'appearance')]
                labels = torch.tensor(values['labels'], device='cuda')
                c, s = runner.augment(c, s, quaternion, jitter)
                torch.manual_seed(seed * 10000000 + step)
                model.train(); optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:
                    group['lr'] = runner.learning_rate(step, steps_per_epoch)
                loss = F.cross_entropy(model(c, s, a), labels, label_smoothing=CONFIG['label_smoothing'])
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss.')
                loss.backward()
                if step == 0:
                    first_gradients = {n: None if p.grad is None else float(p.grad.norm()) for n, p in model.named_parameters()}
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['clip_norm'], error_if_nonfinite=True)
                optimizer.step(); total_loss += float(loss) * len(index); step += 1
            completed = epoch + 1
            assert step == completed * steps_per_epoch
            training.append(dict(epoch=completed, step=step, loss=total_loss / count,
                raw_sampling_sha256=raw_stream, effective_sampling_sha256=effective_stream,
                ModelNet_replacements_by_class=replaced, last_gradient_norm=float(norm)))
            if completed % CONFIG['validation_every'] == 0:
                select(completed)
            save()
            write(dest / 'progress.json', dict(role=role, seed=seed, epoch=completed, step=step,
                best_epoch=best_epoch, best_validation_ce=best, loss=total_loss / count,
                peak_allocated=torch.cuda.max_memory_allocated(), updated_unix=time.time()))
            if time.monotonic() > deadline and completed < CONFIG['epochs']:
                print('SHARD_COMPLETE', role, seed, completed, flush=True)
                return 75
        model.load_state_dict(best_state)
        changes = {n: float((p.detach().cpu() - initial_position[n]).norm()) for n, p in model.position.named_parameters()}
        record = dict(role=role, seed=seed, epochs=completed, steps=step, best_epoch=best_epoch,
            best_selection_ce=best, initial_shared_sha256=initial_shared,
            shared_parameters_changed=runner.shared_digest(model) != initial_shared,
            first_gradients=first_gradients, position_parameter_changes=changes,
            raw_sampling_sha256=raw_stream, effective_sampling_sha256=effective_stream,
            parameters=sum(p.numel() for p in model.parameters()), position_parameters=sum(p.numel() for p in model.position.parameters()),
            seconds=elapsed_before + time.monotonic() - started,
            peak_allocated=max(peak_before, torch.cuda.max_memory_allocated()), resumed=resumed,
            lock_sha256=lock_sha, optimizer_updates=step,
            hashes={name: sha(dest / name) for name in ('best.pt', 'validation.npz', 'history.json', 'resume.pt')})
        write(dest / 'run.json', record)
        print('FIT_COMPLETE', role, seed, record['seconds'], flush=True)
        return 0
    except BaseException as error:
        write(dest / f'failure_{time.time_ns()}.json', dict(error_type=type(error).__name__, reason=str(error),
            step=step, last_complete_epoch=completed, resume_exists=resume.exists()))
        raise


def select_all():
    lock = check()
    assert not (OUT / 'selected.json').exists()
    records = []
    for seed in SEEDS:
        paired = []
        for role in ROLES:
            directory = OUT / 'runs' / role / str(seed)
            run = read(directory / 'run.json')
            assert run['epochs'] == 120 and run['steps'] == 46800
            for name, digest in run['hashes'].items():
                assert sha(directory / name) == digest
            history = read(directory / 'history.json')['validation']
            best = min(history, key=lambda row: row['selection_ce'])
            assert best['epoch'] == run['best_epoch'] and best['selection_ce'] == run['best_selection_ce']
            original = next(m for m in lock['old_models'] if (m['role'], m['seed']) == (role, seed))
            assert run['raw_sampling_sha256'] == read(Path(original['path']).parent / 'run.json')['sampling_sha256']
            paired.append(run)
            records.append(dict(role=role, seed=seed, selected_epoch=run['best_epoch'],
                path=str(directory / 'best.pt'), sha256=sha(directory / 'best.pt'), run_sha256=sha(directory / 'run.json')))
        for field in ('initial_shared_sha256', 'raw_sampling_sha256', 'effective_sampling_sha256'):
            assert len({r[field] for r in paired}) == 1, field
    write(OUT / 'selected.json', dict(records=records, lock_sha256=sha(OUT / 'lock.json'),
        completed_unix=time.time(), new_development_scores_read=False))
    print('ALL_MODELS_SELECTED', len(records), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze', 'train', 'select_all'))
    parser.add_argument('--role', choices=ROLES)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.action == 'train':
        raise SystemExit(train(args.role, args.seed))
    globals()[args.action]()
