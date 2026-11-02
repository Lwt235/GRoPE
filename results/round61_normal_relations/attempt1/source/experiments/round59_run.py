"""R59 uses the unchanged R58 optimization loop, with explicit scoped adapters."""
import argparse
from contextlib import contextmanager
import shutil
import time
import unittest
import numpy as np
import torch
from models.round59_relation_controls import make_model
from experiments import round58_run as original
from experiments.round59_common import (ROOT, OUT, REFERENCE_OUT, CONFIG, SEEDS, CONDITIONS,
    NEW_ROLES, REFERENCE_ROLES, ROLES, SOURCES, CACHE_NAMES, DIAGNOSTIC,
    checked, read, write, sha, load_cache, run_path, prediction_path, reference_record)


@contextmanager
def fitting_adapter():
    changes = dict(OUT=OUT, ROLES=NEW_ROLES, SupportMatcher=make_model, checked=checked, load_cache=load_cache)
    saved = {name: getattr(original, name) for name in changes}
    try:
        for name, value in changes.items():
            setattr(original, name, value)
        yield
    finally:
        for name, value in saved.items():
            setattr(original, name, value)


def freeze():
    from experiments.round58_common import checked as checked58
    checked58()
    if OUT.exists():
        raise ValueError('preserve existing R59 attempt')
    completion = read(REFERENCE_OUT/'completion.json')
    if not completion['completed'] or not read(REFERENCE_OUT/'queue_status.json')['all_children_exit_zero']:
        raise ValueError('R58 must be complete')
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern='test_round59_relation_controls.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError('R59 unit preflight failed')
    sources = {name: sha(ROOT/name) for name in SOURCES}
    records = {str(REFERENCE_OUT/name): sha(REFERENCE_OUT/name) for name in
               ('lock.json', 'completion.json', 'selected.json', 'metrics/summary.json', 'queue_status.json')}
    references, paired = [], {}
    for role in REFERENCE_ROLES:
        for seed in SEEDS:
            path = run_path(role, seed)
            run = read(path/'run.json')
            if (run['role'], run['seed'], run['epochs'], run['steps']) != (role, seed, 60, 14880):
                raise ValueError('reference run identity')
            identity = [run['initial_shared_sha256'], run['stream_sha256']]
            if paired.setdefault(str(seed), identity) != identity:
                raise ValueError('reference pairing')
            hashes = dict(run['hashes'], **{'run.json': sha(path/'run.json')})
            for name, digest in hashes.items():
                if sha(path/name) != digest:
                    raise ValueError('changed reference '+str(path/name))
            predictions = {}
            for condition in CONDITIONS:
                prediction = prediction_path(role, seed, condition)
                meta = read(prediction.with_suffix('.json'))
                if sha(prediction) != meta['predictions_sha256'] or meta['checkpoint_sha256'] != hashes['best.pt']:
                    raise ValueError('reference prediction identity')
                predictions[condition] = dict(path=str(prediction), sha256=meta['predictions_sha256'],
                    record_sha256=sha(prediction.with_suffix('.json')))
            references.append(dict(role=role, seed=seed, path=str(path), hashes=hashes, predictions=predictions,
                initial_shared_sha256=identity[0], stream_sha256=identity[1], reused=True))
    caches = {}
    for name in CACHE_NAMES:
        meta = REFERENCE_OUT/'data'/f'{name}.json'
        data = REFERENCE_OUT/'data'/f'{name}.npz'
        info = read(meta)
        if sha(data) != info['sha256']:
            raise ValueError('cache identity '+name)
        caches[name] = dict(path=str(data), sha256=info['sha256'], count=info['count'])
        records[str(meta)] = sha(meta)
    OUT.mkdir(parents=True)
    for name in sources:
        target = OUT/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    (OUT/'data').mkdir()
    for name in CACHE_NAMES:
        shutil.copyfile(REFERENCE_OUT/'data'/f'{name}.json', OUT/'data'/f'{name}.json')
    lock = dict(sources=sources, config=CONFIG, roles=ROLES, new_roles=NEW_ROLES, seeds=SEEDS,
                conditions=CONDITIONS, diagnostic=DIAGNOSTIC, references=references, paired=paired,
                caches=caches, input_records=records, primary='invariant_mlp', new_fits=12,
                reference_fits=18, frozen_unix=time.time(), fresh_confirmation=False,
                candidate_chosen_after_R58_development_results=True)
    write(OUT/'lock.json', lock)
    (OUT/'lock.sha256').write_text(sha(OUT/'lock.json')+'\n')
    write(OUT/'units.json', dict(passed=True, tests=result.testsRun, sources=sources))
    print('FROZEN', sha(OUT/'lock.json'), '12 new fits;18 exact references', flush=True)


def reframed(data):
    features, centers, covariance = data
    r = centers.new_tensor([[[0., -1, 0], [1, 0, 0], [0, 0, 1]], [[1., 0, 0], [0, 0, -1], [0, 1, 0]]])
    c = centers@r.transpose(-1, -2)+centers.new_tensor([.7, -.3, .2])
    s = r[:, None]@covariance@r[:, None].transpose(-1, -2)
    return features, c, s


def preflight():
    lock = checked()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    cache = load_cache('fit0')
    data = original.model_inputs(cache, slice(0, 2))
    truth = torch.as_tensor(cache['overlap'][:2], device='cuda')
    rows = []
    for role in NEW_ROLES:
        torch.manual_seed(11)
        model = make_model(role).cuda()
        if original.shared_digest(model) != reference_record(lock, 'invariant_mlp', 11)['initial_shared_sha256']:
            raise ValueError('unpaired new common initialization')
        opt = torch.optim.AdamW(model.parameters(), lr=.001)
        loss = original.loss_function(model(*data), truth)
        loss.backward()
        gradients = {n: None if p.grad is None else float(p.grad.norm()) for n, p in model.position.named_parameters()}
        if any(v is None or v == 0 or not np.isfinite(v) for v in gradients.values()):
            raise ValueError('PE gradient '+role)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        model.eval()
        with torch.no_grad():
            base = model(*data)
            rigid = model(*reframed(data))
            permuted = model(data[0].flip(-2), data[1].flip(-2), data[2].flip(-3)).flip(-2, -1)
            rigid_error = float((base.softmax(-1)-rigid.softmax(-1)).abs().max())
            order_error = float((base-permuted).abs().max())
        if not torch.isfinite(loss) or order_error > .002 or (role != 'matrix_mlp' and rigid_error > .002):
            raise ValueError('substantive invariant/CUDA discrepancy')
        rows.append(dict(role=role, loss=float(loss), gradients=gradients, rigid_probability_error=rigid_error,
            invariant_by_construction=role != 'matrix_mlp', permutation_logit_error=order_error,
            parameters=sum(p.numel() for p in model.parameters()), pe_parameters=sum(p.numel() for p in model.position.parameters())))
    write(OUT/'preflight.json', dict(passed=True, rows=rows, gpu=torch.cuda.get_device_name(), unix=time.time()))
    print('PREFLIGHT four CUDA updates and geometry behavior checked', flush=True)


def train(role, seed):
    lock = checked()
    if role not in NEW_ROLES:
        raise ValueError('never refit a reference')
    with fitting_adapter():
        original.train(role, seed)
    run = read(run_path(role, seed)/'run.json')
    ref = reference_record(lock, 'invariant_mlp', seed)
    for name in ('initial_shared_sha256', 'stream_sha256'):
        if run[name] != ref[name]:
            raise ValueError('actual fitting pairing mismatch '+name)
    print('PAIRED actual initialization/order', role, seed, flush=True)


@torch.no_grad()
def evaluate(model, cache, reframe=False):
    if not reframe:
        return original.evaluate(model, cache)
    model.eval()
    values = []
    for begin in range(0, len(cache['ids']), CONFIG['eval_batch_size']):
        data = original.model_inputs(cache, slice(begin, begin+CONFIG['eval_batch_size']))
        values.append(model(*reframed(data)).cpu().numpy())
    values = np.concatenate(values)
    return values, original.metric_torch(values, cache['overlap'], cache['world_centers'])


def export(seed):
    checked()
    selected = read(OUT/'selected.json')
    if seed not in SEEDS or len(selected['runs']) != 30:
        raise ValueError('selection scope')
    for condition in tuple(CONDITIONS)+(DIAGNOSTIC,):
        cache_name = 'eval_base' if condition == DIAGNOSTIC else 'eval_'+condition
        cache = load_cache(cache_name)
        roles = ROLES if condition == DIAGNOSTIC else NEW_ROLES
        for role in roles:
            record = next(x for x in selected['runs'] if (x['role'], x['seed']) == (role, seed))
            checkpoint = run_path(role, seed)/'best.pt'
            if sha(checkpoint) != record['checkpoint_sha256']:
                raise ValueError('selected checkpoint identity')
            model = make_model(role).cuda()
            model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['state_dict'])
            path = prediction_path(role, seed, condition)
            if path.exists():
                raise ValueError('preserve existing predictions')
            path.parent.mkdir(parents=True, exist_ok=True)
            logits, metrics = evaluate(model, cache, reframe=condition == DIAGNOSTIC)
            np.savez_compressed(path, logits=logits, ids=cache['ids'])
            write(path.with_suffix('.json'), dict(role=role, seed=seed, condition=condition, metrics=metrics,
                checkpoint_sha256=record['checkpoint_sha256'], predictions_sha256=sha(path),
                cache_sha256=read(OUT/'data'/f'{cache_name}.json')['sha256'],
                selection_sha256=sha(OUT/'selected.json'), exported_unix=time.time(), reused=False))
            print('EXPORTED', role, seed, condition, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('freeze', 'preflight', 'train', 'export'))
    parser.add_argument('--role', choices=NEW_ROLES)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    original.configure()
    if args.stage == 'train':
        train(args.role, args.seed)
    elif args.stage == 'export':
        export(args.seed)
    else:
        globals()[args.stage]()
