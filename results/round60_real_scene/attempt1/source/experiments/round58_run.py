"""Freeze, check, fit and export the one prospective R58 experiment."""
import argparse
import hashlib
import math
import os
from pathlib import Path
import shutil
import time
import unittest
import numpy as np
import torch
from models.round58_support_pe import SupportMatcher
from experiments.round58_common import (ROOT, OUT, OLD_DATA, CONFIG, ROLES, SEEDS,
                                         CONDITIONS, SOURCES, sha, read, write, checked)
from experiments.round58_data import load_cache, loss_function


def configure():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def shared_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not name.startswith('position.'):
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def freeze():
    if OUT.exists():
        raise ValueError('preserve existing attempt')
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern='test_round58_support_pe.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError('unit preflight failed before freeze')
    manifest_path = OLD_DATA/'manifest.json'
    if sha(manifest_path) != (OLD_DATA/'manifest.sha256').read_text().strip():
        raise ValueError('old data manifest identity')
    manifest = read(manifest_path)
    sources = {name: sha(ROOT/name) for name in SOURCES}
    OUT.mkdir(parents=True)
    for name in sources:
        dest = OUT/'source'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, dest)
    lock = dict(sources=sources, config=CONFIG, roles=ROLES, seeds=SEEDS, conditions=CONDITIONS,
                input_manifest_sha256=sha(manifest_path),
                input_data={x['split']: {k: x[k] for k in ('path', 'sha256', 'count')} for x in manifest['partitions']},
                frozen_unix=time.time(), primary='gaussian', new_fits=18,
                official_test_access=False, fresh_confirmation=False)
    write(OUT/'lock.json', lock)
    (OUT/'lock.sha256').write_text(sha(OUT/'lock.json')+'\n')
    write(OUT/'units.json', dict(passed=True, tests=result.testsRun, sources=sources))
    print('FROZEN', sha(OUT/'lock.json'), flush=True)


def model_inputs(cache, index, device='cuda'):
    return tuple(torch.as_tensor(cache[k][index], device=device) for k in ('features', 'centers', 'covariance'))


@torch.no_grad()
def metric_torch(logits, overlap, world):
    """One Torch path, checked against separate NumPy arithmetic after fitting."""
    logits, overlap, world = [torch.as_tensor(x, dtype=torch.float64) for x in (logits, overlap, world)]
    result, counts, all_ce = {}, [], []
    for reverse in (False, True):
        s, o = (logits.transpose(-1, -2), overlap.transpose(-1, -2)) if reverse else (logits, overlap)
        a, b = (world[:, 1], world[:, 0]) if reverse else (world[:, 0], world[:, 1])
        weight = o*(o >= .1)
        label = weight/weight.sum(-1, keepdim=True).clamp_min(1e-12)
        valid = o.max(-1).values >= .3
        count = int(valid.sum())
        counts.append(count)
        probability = s.softmax(-1)
        chosen = s.argmax(-1, keepdim=True)
        picked_overlap = o.gather(-1, chosen).squeeze(-1)
        distances = (a[:, :, None]-b[:, None]).norm(dim=-1)
        hard = distances.gather(-1, chosen).squeeze(-1)
        soft = (probability@b-a).norm(dim=-1)
        ce = -(label*s.log_softmax(-1)).sum(-1)
        values = dict(region_recall=(picked_overlap >= .3).double(), overlap=picked_overlap,
                      center_pck=(hard <= .1).double(), hard_epe=hard, soft_epe=soft, ce=ce,
                      oracle_center_pck=(distances.min(-1).values <= .1).double(),
                      oracle_overlap=o.max(-1).values,
                      overlap_oracle_center_pck=(distances.gather(-1, o.argmax(-1, keepdim=True)).squeeze(-1) <= .1).double(),
                      center_oracle_region_recall=(o.gather(-1, distances.argmin(-1, keepdim=True)).squeeze(-1) >= .3).double())
        for key, value in values.items():
            result.setdefault(key, []).append(float((value*valid).sum()))
        if count:
            all_ce.append(float((ce*valid).sum())/count)
    total = sum(counts)
    return dict(**{k: sum(v)/max(1, total) for k, v in result.items()},
                selection_ce=float(np.mean(all_ce)) if all_ce else 0.,
                valid_counts=counts, eligible_fraction=total/(2*logits.shape[0]*logits.shape[1]))


@torch.no_grad()
def evaluate(model, cache):
    model.eval()
    scores = []
    for begin in range(0, len(cache['ids']), CONFIG['eval_batch_size']):
        index = slice(begin, begin+CONFIG['eval_batch_size'])
        scores.append(model(*model_inputs(cache, index)).cpu().numpy())
    scores = np.concatenate(scores)
    return scores, metric_torch(scores, cache['overlap'], cache['world_centers'])


def preflight():
    checked()
    from experiments.round58_audit import audit_cache_sample
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    cache = load_cache('fit0')
    cache_checks = [audit_cache_sample(OUT/'data'/f'{n}_sample.npz') for n in ('fit0', 'fit1', 'val')]
    rows, shared = [], None
    for role in ROLES:
        torch.manual_seed(11)
        model = SupportMatcher(role).cuda().train()
        digest = shared_digest(model)
        if shared is None:
            shared = digest
        if digest != shared:
            raise ValueError('unpaired shared initialization')
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        data = model_inputs(cache, slice(0, 2))
        loss = loss_function(model(*data), torch.as_tensor(cache['overlap'][:2], device='cuda'))
        loss.backward()
        gradients = {n: None if p.grad is None else float(p.grad.norm()) for n, p in model.position.named_parameters()}
        if any(x is None or not np.isfinite(x) or x == 0 for x in gradients.values()):
            raise ValueError('invalid PE gradient '+role)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            baseline = model(*data)
            # Rotate/translate each view independently; do not use any relative pose.
            rotation = torch.tensor([[[0., -1, 0], [1, 0, 0], [0, 0, 1]],
                                     [[1., 0, 0], [0, 0, -1], [0, 1, 0]]], device='cuda')
            moved = data[1]@rotation.transpose(-1, -2)+torch.tensor([.7, -.3, .2], device='cuda')
            covariance = rotation[:, None]@data[2]@rotation[:, None].transpose(-1, -2)
            changed = model(data[0], moved, covariance)
            error = float((baseline.softmax(-1)-changed.softmax(-1)).abs().max())
            permuted = model(data[0].flip(-2), data[1].flip(-2), data[2].flip(-3)).flip(-2, -1)
            permutation_error = float((baseline-permuted).abs().max())
        if not torch.isfinite(loss) or error > .002 or permutation_error > .002:
            raise ValueError('substantive CUDA preflight issue')
        rows.append(dict(role=role, loss=float(loss), gradients=gradients,
                         rigid_probability_error=error, permutation_logit_error=permutation_error,
                         parameters=sum(p.numel() for p in model.parameters()),
                         pe_parameters=sum(p.numel() for p in model.position.parameters())))
    write(OUT/'preflight.json', dict(passed=True, rows=rows, cache_checks=cache_checks,
                                    gpu=torch.cuda.get_device_name(), unix=time.time()))
    print('PREFLIGHT passed', flush=True)


def learning_rate(step, steps_per_epoch):
    warmup = CONFIG['warmup_epochs']*steps_per_epoch
    total = CONFIG['epochs']*steps_per_epoch
    if step < warmup:
        return CONFIG['lr']*(step+1)/warmup
    fraction = .5*(1+math.cos(math.pi*(step-warmup)/max(1, total-warmup-1)))
    return CONFIG['final_lr']+(CONFIG['lr']-CONFIG['final_lr'])*fraction


def train(role, seed):
    checked()
    if role not in ROLES or seed not in SEEDS or not read(OUT/'preflight.json')['passed']:
        raise ValueError('scope/preflight')
    destination = OUT/'runs'/role/str(seed)
    if destination.exists():
        raise ValueError('preserve existing fit')
    destination.mkdir(parents=True)
    fitting = [load_cache('fit0'), load_cache('fit1')]
    validation = load_cache('val')
    torch.manual_seed(seed)
    model = SupportMatcher(role).cuda()
    initial = shared_digest(model)
    initial_position = {n: p.detach().cpu().clone() for n, p in model.position.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    rng = np.random.default_rng(seed+58000)
    stream = hashlib.sha256()
    count = len(fitting[0]['ids'])
    steps_per_epoch = math.ceil(count/CONFIG['batch_size'])
    history = dict(training=[], validation=[])
    best, selected, step = float('inf'), None, 0
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(CONFIG['epochs']+1):
        if epoch % CONFIG['validation_every'] == 0:
            scores, stats = evaluate(model, validation)
            history['validation'].append(dict(epoch=epoch, steps=step, **stats))
            if stats['selection_ce'] < best:
                best, selected = stats['selection_ce'], epoch
                torch.save(dict(role=role, seed=seed, epoch=epoch,
                                state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()}), destination/'best.pt')
                np.savez_compressed(destination/'validation.npz', logits=scores, ids=validation['ids'])
            write(destination/'history.json', history)
            print('VAL', role, seed, epoch, best, stats['region_recall'], stats['center_pck'], flush=True)
        if epoch == CONFIG['epochs']:
            break
        cache = fitting[epoch % 2]
        order = rng.permutation(count)
        stream.update(np.array([epoch % 2], dtype='<i8').tobytes())
        stream.update(order.astype('<i8').tobytes())
        loss_sum = 0.
        model.train()
        for begin in range(0, count, CONFIG['batch_size']):
            index = order[begin:begin+CONFIG['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group['lr'] = learning_rate(step, steps_per_epoch)
            loss = loss_function(model(*model_inputs(cache, index)),
                                 torch.as_tensor(cache['overlap'][index], device='cuda'))
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite training loss')
            loss.backward()
            if step == 0:
                gradients = {n: None if p.grad is None else float(p.grad.norm()) for n, p in model.position.named_parameters()}
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['clip'], error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss)*len(index)
            step += 1
        history['training'].append(dict(epoch=epoch+1, steps=step, loss=loss_sum/count))
        if time.monotonic()-start > CONFIG['max_fit_seconds']:
            write(destination/'failure.json', dict(reason='eight-hour fit bound', epoch=epoch+1, steps=step))
            raise TimeoutError('bounded fit time')
    changes = {n: float((p.detach().cpu()-initial_position[n]).norm()) for n, p in model.position.named_parameters()}
    write(destination/'run.json', dict(role=role, seed=seed, epochs=CONFIG['epochs'], steps=step,
          selected_epoch=selected, best_val_ce=best, seconds=time.monotonic()-start,
          initial_shared_sha256=initial, stream_sha256=stream.hexdigest(),
          shared_parameters_changed=shared_digest(model) != initial,
          first_gradients=gradients, final_position_changes=changes,
          parameters=sum(p.numel() for p in model.parameters()),
          pe_parameters=sum(p.numel() for p in model.position.parameters()),
          peak_cuda_bytes=torch.cuda.max_memory_allocated(),
          data_sha256={n: read(OUT/'data'/f'{n}.json')['sha256'] for n in ('fit0', 'fit1', 'val')},
          hashes={n: sha(destination/n) for n in ('best.pt', 'validation.npz', 'history.json')},
          completed_unix=time.time()))
    print('TRAIN complete', role, seed, step, flush=True)


def export(seed):
    checked()
    selected = read(OUT/'selected.json')
    if seed not in SEEDS or len(selected['runs']) != 18:
        raise ValueError('selection seal')
    for condition in CONDITIONS:
        cache = load_cache('eval_'+condition)
        for role in ROLES:
            record = next(x for x in selected['runs'] if x['role'] == role and x['seed'] == seed)
            checkpoint = OUT/'runs'/role/str(seed)/'best.pt'
            if sha(checkpoint) != record['checkpoint_sha256']:
                raise ValueError('selected checkpoint changed')
            model = SupportMatcher(role).cuda()
            model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['state_dict'])
            destination = OUT/'evaluation'/role/str(seed)
            destination.mkdir(parents=True, exist_ok=True)
            path = destination/(condition+'.npz')
            if path.exists():
                raise ValueError('preserve existing prediction')
            scores, metrics = evaluate(model, cache)
            np.savez_compressed(path, logits=scores, ids=cache['ids'])
            write(path.with_suffix('.json'), dict(role=role, seed=seed, condition=condition,
                  metrics=metrics, checkpoint_sha256=record['checkpoint_sha256'],
                  predictions_sha256=sha(path), cache_sha256=read(OUT/'data'/('eval_'+condition+'.json'))['sha256'],
                  selection_sha256=sha(OUT/'selected.json'), exported_unix=time.time()))
            print('EXPORTED', role, seed, condition, metrics['region_recall'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('freeze', 'preflight', 'train', 'export'))
    parser.add_argument('--role', choices=ROLES)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    configure()
    if args.stage == 'train':
        train(args.role, args.seed)
    elif args.stage == 'export':
        export(args.seed)
    else:
        globals()[args.stage]()
