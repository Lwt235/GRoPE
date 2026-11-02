"""Paired native point-cloud Transformer experiment. Official test is inaccessible."""
import argparse
import hashlib
import math
from pathlib import Path
import random
import shutil
import time
import unittest
import numpy as np
import torch
import torch.nn.functional as F

from models.round48_point_transformer import make_model, normalize_points, ROLES
from experiments.round48_common import (ROOT, OUT, SEEDS, CONFIG, VARIANTS,
    sha, read, write, checked, source_paths)
from experiments.round48_acquire import OUT as ACQUISITION, ARCHIVE, EXPECTED_SHA
from experiments.round48_data import load, evaluation_points


def configure():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def freeze(out):
    if out.exists():
        raise ValueError('preserve existing R48 attempt')
    acquisition = read(ACQUISITION/'completion.json')
    if not acquisition['completed'] or acquisition['array_members_opened']:
        raise ValueError('closed verified archive required')
    if sha(ARCHIVE) != EXPECTED_SHA:
        raise ValueError('archive identity')
    sources = {name: sha(ROOT/name) for name in source_paths()}
    out.mkdir(parents=True)
    for name in sources:
        target = out/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    lock = dict(sources=sources, config=CONFIG, roles=ROLES, seeds=SEEDS,
                primary='conditional', variants=VARIANTS, archive_sha256=EXPECTED_SHA,
                acquisition_plan_sha256=sha(ACQUISITION/'plan.json'),
                acquisition_completion_sha256=sha(ACQUISITION/'completion.json'),
                official_test_access=False, prior_confirmation_scenes_access=False,
                data_arrays_opened_at_freeze=False, frozen_unix=time.time())
    write(out/'lock.json', lock)
    (out/'lock.sha256').write_text(sha(out/'lock.json')+'\n')
    print('FROZEN', sha(out/'lock.json'), len(sources), 'source files', flush=True)


def preflight(out):
    checked(out)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(
        str(ROOT/'tests'), pattern='test_round48_point_pe.py'))
    write(out/'preflight.json', dict(passed=result.wasSuccessful(), tests=result.testsRun,
                                    real_data_arrays_read=False))
    if not result.wasSuccessful():
        raise SystemExit(1)


def shared_digest(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if not name.startswith('position_field.'):
            h.update(name.encode())
            h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def score_sample(model, points, path):
    model.eval()
    x = normalize_points(points[:1])
    positions = x*12
    h = model.stem(x)
    layer = model.layers[0]
    q, k, _ = layer._project(h)
    q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    features = layer.attention_norm(h)
    probability = model.scores(layer, q, k, features, positions).softmax(-1)
    ids = torch.linspace(0, x.shape[1]-1, min(8, x.shape[1]), device=x.device).long()
    values = dict(query_ids=ids, q=q[0, :, ids], k=k[0],
                  features=features[0, ids], qmean=positions[0, ids],
                  kmean=positions[0], probability=probability[0, :, ids])
    np.savez_compressed(path, **{k: v.cpu().numpy() for k, v in values.items()})


def smoke(out):
    from experiments.round48_audit import check_sample
    checked(out)
    if not read(out/'preflight.json')['passed']:
        raise ValueError('CPU preflight required')
    rows = []
    for role in ROLES:
        torch.manual_seed(4817)
        points = torch.randn(2, 37, 3, device='cuda')
        labels = torch.tensor([3, 19], device='cuda')
        torch.manual_seed(4811)
        model = make_model(role).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        model.train()
        loss = F.cross_entropy(model(points), labels)
        loss.backward()
        gradients = {n: None if p.grad is None else float(p.grad.norm())
                     for n, p in model.position_field.named_parameters()}
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        path = out/('smoke_'+role+'.npz')
        score_sample(model, points, path)
        numeric = check_sample(role, model.state_dict(), path)
        if not torch.isfinite(loss) or numeric['warning']:
            raise ValueError('substantive CUDA smoke discrepancy: '+str(numeric))
        rows.append(dict(role=role, loss=float(loss), gradients=gradients, numeric=numeric))
    timing = []
    # Synthetic throughput only: no architecture/optimizer search or real arrays.
    for role in ('rope', 'conditional'):
        torch.manual_seed(4819)
        model = make_model(role).cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        points = torch.randn(CONFIG['batch_size'], CONFIG['points'], 3, device='cuda')
        labels = torch.arange(CONFIG['batch_size'], device='cuda') % 40
        durations = []
        for i in range(15):
            torch.cuda.synchronize()
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(points), labels).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            if i >= 5:
                durations.append(time.perf_counter()-started)
        timing.append(dict(role=role, median_step_seconds=float(np.median(durations)),
                           repeats=10, warmup=5))
    write(out/'smoke.json', dict(passed=True, rows=rows, synthetic_timing=timing,
                                gpu=torch.cuda.get_device_name(), real_arrays_read=False))
    print('SMOKE', timing, flush=True)


@torch.no_grad()
def evaluate(model, points, labels):
    model.eval()
    logits, total_ce, correct = [], 0., 0
    for start in range(0, len(points), CONFIG['evaluation_batch_size']):
        end = start+CONFIG['evaluation_batch_size']
        prediction = model(points[start:end])
        total_ce += float(F.cross_entropy(prediction, labels[start:end], reduction='sum'))
        correct += int((prediction.argmax(-1) == labels[start:end]).sum())
        logits.append(prediction.cpu().numpy())
    return np.concatenate(logits), dict(ce=total_ce/len(points), oa=correct/len(points),
                                       correct=correct, count=len(points))


def learning_rate(step, steps_per_epoch):
    warmup = CONFIG['warmup_epochs']*steps_per_epoch
    total = CONFIG['epochs']*steps_per_epoch
    if step < warmup:
        return CONFIG['learning_rate']*(step+1)/warmup
    factor = .5*(1+math.cos(math.pi*(step-warmup)/max(1, total-warmup-1)))
    return CONFIG['final_learning_rate']+(CONFIG['learning_rate']-CONFIG['final_learning_rate'])*factor


def fit(role, seed, train_points, train_labels, val_points, val_labels, val_ids, out, deadline):
    dest = out/'runs'/role/str(seed)
    if dest.exists():
        raise ValueError('preserve existing run')
    dest.mkdir(parents=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = make_model(role).cuda()
    initial_shared = shared_digest(model)
    initial_position = {n: p.detach().cpu().clone() for n, p in model.position_field.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'],
                                 weight_decay=CONFIG['weight_decay'])
    rng = np.random.default_rng(seed+48000)
    stream_hash = hashlib.sha256()
    history, training_history = [], []
    best, best_epoch, best_state = float('inf'), None, None
    count = len(train_points)
    steps_per_epoch = math.ceil(count/CONFIG['batch_size'])
    step = 0
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(CONFIG['epochs']+1):
        if epoch % CONFIG['validation_every'] == 0:
            logits, stats = evaluate(model, val_points, val_labels)
            history.append(dict(epoch=epoch, step=step, **stats))
            if stats['ce'] < best:
                best, best_epoch = stats['ce'], epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(dict(role=role, seed=seed, epoch=epoch, state_dict=best_state), dest/'best.pt')
                np.savez_compressed(dest/'validation.npz', logits=logits,
                                    labels=val_labels.cpu().numpy(), ids=val_ids)
            write(dest/'history.json', dict(validation=history, training=training_history))
            print('VAL', role, seed, epoch, stats['ce'], stats['oa'], flush=True)
        if epoch == CONFIG['epochs']:
            break
        order = rng.permutation(count)
        stream_hash.update(order.astype('<i8').tobytes())
        epoch_loss = 0.
        for begin in range(0, count, CONFIG['batch_size']):
            index = order[begin:begin+CONFIG['batch_size']]
            subset = np.stack([rng.choice(2048, CONFIG['points'], replace=False) for _ in index])
            angle = rng.uniform(0, 2*np.pi, len(index)).astype(np.float32)
            jitter = rng.normal(0, CONFIG['jitter_std'], (len(index), CONFIG['points'], 3))
            jitter = jitter.clip(-CONFIG['jitter_clip'], CONFIG['jitter_clip']).astype(np.float32)
            for value in (index.astype('<i8'), subset.astype('<i8'), angle, jitter):
                stream_hash.update(value.tobytes())
            ii = torch.as_tensor(index, device='cuda')
            jj = torch.as_tensor(subset, device='cuda')
            x = train_points[ii[:, None], jj]
            angles = torch.as_tensor(angle, device='cuda')
            c, s = angles.cos()[:, None], angles.sin()[:, None]
            x = torch.stack((c*x[..., 0]+s*x[..., 2], x[..., 1],
                             -s*x[..., 0]+c*x[..., 2]), -1)
            x = x+torch.as_tensor(jitter, device='cuda')
            y = train_labels[ii]
            # Common dropout stream, independent of positional parameter initialization.
            torch.manual_seed(seed*100000000+step)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group['lr'] = learning_rate(step, steps_per_epoch)
            loss = F.cross_entropy(model(x), y, label_smoothing=CONFIG['label_smoothing'])
            if not torch.isfinite(loss):
                raise ValueError('nonfinite loss')
            loss.backward()
            if step == 0:
                first_gradients = {n: None if p.grad is None else float(p.grad.norm())
                                   for n, p in model.named_parameters()}
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['clip_norm'],
                                                 error_if_nonfinite=True)
            optimizer.step()
            epoch_loss += float(loss)*len(index)
            step += 1
        training_history.append(dict(epoch=epoch+1, loss=epoch_loss/count,
                                     steps=step, last_gradient_norm=float(norm)))
        if torch.cuda.max_memory_allocated() > CONFIG['memory_ceiling_bytes']:
            raise MemoryError('declared allocation ceiling')
        if time.monotonic() > deadline:
            write(dest/'interrupted.json', dict(epoch=epoch+1, reason='bounded seed-shard budget',
                                               partial_run=True, completed=False))
            raise TimeoutError('eight GPU-hour seed-shard budget')
    model.load_state_dict(best_state)
    score_sample(model, val_points, dest/'score_sample.npz')
    changes = {n: float((p.detach().cpu()-initial_position[n]).norm())
               for n, p in model.position_field.named_parameters()}
    record = dict(role=role, seed=seed, epochs=CONFIG['epochs'], steps=step,
        best_epoch=best_epoch, best_val_ce=best, initial_shared_sha256=initial_shared,
        batch_augmentation_sha256=stream_hash.hexdigest(), first_gradients=first_gradients,
        shared_parameters_changed=shared_digest(model) != initial_shared,
        position_parameter_changes=changes,
        parameters=sum(p.numel() for p in model.parameters()),
        positional_parameters=sum(p.numel() for p in model.position_field.parameters()),
        seconds=time.monotonic()-start, peak_allocated=torch.cuda.max_memory_allocated(),
        hashes={name: sha(dest/name) for name in ('best.pt', 'history.json', 'validation.npz', 'score_sample.npz')})
    write(dest/'run.json', record)
    print('DONE', role, seed, record['seconds'], flush=True)


def train(out, seeds):
    checked(out)
    if not read(out/'smoke.json')['passed']:
        raise ValueError('CUDA smoke required')
    train_data, val = load('fit', out), load('val', out)
    train_points = torch.as_tensor(train_data['points'], device='cuda')
    train_labels = torch.as_tensor(train_data['labels'], device='cuda')
    val_points = torch.as_tensor(evaluation_points(val, 'base'), device='cuda')
    val_labels = torch.as_tensor(val['labels'], device='cuda')
    for seed in seeds:
        if seed not in SEEDS:
            raise ValueError('undeclared seed')
        start = time.monotonic()
        write(out/f'seed_{seed}_started.json', dict(unix=time.time(), gpu=torch.cuda.get_device_name()))
        offset = SEEDS.index(seed)
        try:
            for role in ROLES[offset:]+ROLES[:offset]:
                fit(role, seed, train_points, train_labels, val_points, val_labels, val['ids'],
                    out, start+CONFIG['shard_seconds'])
        except BaseException as error:
            write(out/f'seed_{seed}_failure.json', dict(error=repr(error), unix=time.time()))
            raise
        write(out/f'seed_{seed}_complete.json', dict(completed=True, seconds=time.monotonic()-start))


@torch.no_grad()
def export(out, seeds):
    checked(out)
    seal = read(out/'audit_training/completion.json')
    if len(seal['records']) != len(ROLES)*len(SEEDS):
        raise ValueError('all selected checkpoints must be sealed')
    data = load('dev_eval', out)
    labels = torch.as_tensor(data['labels'], device='cuda')
    for seed in seeds:
        for role in ROLES:
            dest = out/'runs'/role/str(seed)
            saved = next(r for r in seal['records'] if r['role'] == role and r['seed'] == seed)
            if sha(dest/'best.pt') != saved['checkpoint_sha256']:
                raise ValueError('selected checkpoint changed')
            if (dest/'dev_eval.json').exists():
                raise ValueError('preserve existing evaluation')
            state = torch.load(dest/'best.pt', weights_only=True, map_location='cpu')['state_dict']
            model = make_model(role).cuda()
            model.load_state_dict(state)
            arrays = dict(labels=data['labels'], ids=data['ids'])
            metrics = {}
            for variant in VARIANTS:
                points = torch.as_tensor(evaluation_points(data, variant), device='cuda')
                arrays[variant], metrics[variant] = evaluate(model, points, labels)
                del points
            np.savez_compressed(dest/'dev_eval.npz', **arrays)
            write(dest/'dev_eval.json', dict(role=role, seed=seed, metrics=metrics,
                checkpoint_sha256=saved['checkpoint_sha256'], predictions_sha256=sha(dest/'dev_eval.npz'),
                official_test_access=False, fresh_confirmation=False))
            print('EXPORT', role, seed, metrics['base'], flush=True)
        write(out/f'export_{seed}_complete.json', dict(completed=True, official_test_access=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('freeze', 'preflight', 'smoke', 'train', 'export'))
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    args = parser.parse_args()
    configure()
    if args.stage in ('train', 'export'):
        globals()[args.stage](args.out, args.seeds)
    else:
        globals()[args.stage](args.out)


if __name__ == '__main__':
    main()
