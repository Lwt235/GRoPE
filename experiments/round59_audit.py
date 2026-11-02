"""R59 checkpoint/reuse checks and independent metric arithmetic."""
import argparse
import time
import numpy as np
import torch
from models.round59_relation_controls import make_model
from experiments.round58_audit import metrics_numpy
from experiments.round59_run import reframed
from experiments.round59_common import (OUT, REFERENCE_OUT, ROLES, NEW_ROLES, REFERENCE_ROLES,
    SEEDS, CONDITIONS, DIAGNOSTIC, checked, read, write, sha, load_cache,
    run_path, prediction_path, reference_record)


def training():
    lock = checked()
    if (OUT/'selected.json').exists():
        raise ValueError('selection already sealed')
    validation = load_cache('val')
    inputs = tuple(torch.from_numpy(validation[k][:2]) for k in ('features', 'centers', 'covariance'))
    records = []
    for role in ROLES:
        for seed in SEEDS:
            path = run_path(role, seed)
            run, history = read(path/'run.json'), read(path/'history.json')
            if (run['role'], run['seed'], run['epochs'], run['steps']) != (role, seed, 60, 14880):
                raise ValueError('run identity/budget')
            ref = reference_record(lock, 'invariant_mlp', seed)
            if any(run[k] != ref[k] for k in ('initial_shared_sha256', 'stream_sha256')):
                raise ValueError('actual pairing')
            for name, digest in run['hashes'].items():
                if sha(path/name) != digest:
                    raise ValueError('changed run artifact')
            if any(run['data_sha256'][name] != lock['caches'][name]['sha256'] for name in ('fit0', 'fit1', 'val')):
                raise ValueError('fit data identity')
            if role in REFERENCE_ROLES:
                old = reference_record(lock, role, seed)
                if sha(path/'run.json') != old['hashes']['run.json']:
                    raise ValueError('changed exact reference')
            best = min(history['validation'], key=lambda x: x['selection_ce'])
            if (best['epoch'], best['selection_ce']) != (run['selected_epoch'], run['best_val_ce']):
                raise ValueError('selection rule')
            with np.load(path/'validation.npz', allow_pickle=False) as f:
                if not np.array_equal(f['ids'], validation['ids']):
                    raise ValueError('validation ID ordering')
                metrics, _ = metrics_numpy(f['logits'], validation['overlap'], validation['world_centers'])
                saved_sample = f['logits'][:2].copy()
            ce_error = abs(metrics['selection_ce']-run['best_val_ce'])
            model = make_model(role).eval()
            checkpoint = torch.load(path/'best.pt', map_location='cpu', weights_only=True)
            if (checkpoint['role'], checkpoint['seed'], checkpoint['epoch']) != (role, seed, run['selected_epoch']):
                raise ValueError('checkpoint identity')
            model.load_state_dict(checkpoint['state_dict'])
            with torch.no_grad():
                baseline = model(*inputs)
                rigid = model(*reframed(inputs))
                permuted = model(inputs[0].flip(-2), inputs[1].flip(-2), inputs[2].flip(-3)).flip(-2, -1)
                replay_error = float((baseline-torch.from_numpy(saved_sample)).abs().max())
                rigid_error = float((baseline.softmax(-1)-rigid.softmax(-1)).abs().max())
                order_error = float((baseline-permuted).abs().max())
                changes = int((baseline.argmax(-1) != rigid.argmax(-1)).sum())
            if ce_error > 1e-6 or replay_error > .002 or order_error > .002 or (role != 'matrix_mlp' and rigid_error > .002):
                raise ValueError('substantive implementation mismatch '+role)
            if not run['shared_parameters_changed']:
                raise ValueError('no common network learning')
            records.append(dict(role=role, seed=seed, reused=role in REFERENCE_ROLES,
                selected_epoch=run['selected_epoch'], checkpoint_sha256=sha(path/'best.pt'),
                run_sha256=sha(path/'run.json'), validation_ce_error=ce_error, replay_logit_error=replay_error,
                rigid_probability_error=rigid_error, rigid_winner_changes=changes,
                permutation_logit_error=order_error, invariant_by_construction=role != 'matrix_mlp'))
    write(OUT/'selected.json', dict(runs=records, sealed_unix=time.time(), lock_sha256=sha(OUT/'lock.json'),
        new_models=12, reused_models=18, fresh_confirmation=False))
    print('SEALED 30 selected models;12 new,18 reused', flush=True)


def metrics():
    lock = checked()
    selected = read(OUT/'selected.json')
    rows, diagnostics, errors = [], [], []
    for condition in tuple(CONDITIONS)+(DIAGNOSTIC,):
        cache_name = 'eval_base' if condition == DIAGNOSTIC else 'eval_'+condition
        cache = load_cache(cache_name)
        for role in ROLES:
            for seed in SEEDS:
                path = prediction_path(role, seed, condition)
                rec = read(path.with_suffix('.json'))
                reused = role in REFERENCE_ROLES and condition in CONDITIONS
                if reused:
                    frozen = reference_record(lock, role, seed)['predictions'][condition]
                    if sha(path) != frozen['sha256'] or sha(path.with_suffix('.json')) != frozen['record_sha256']:
                        raise ValueError('changed reference prediction')
                elif rec['exported_unix'] < selected['sealed_unix'] or rec['selection_sha256'] != sha(OUT/'selected.json'):
                    raise ValueError('new prediction precedes selection')
                if sha(path) != rec['predictions_sha256'] or rec['cache_sha256'] != lock['caches'][cache_name]['sha256']:
                    raise ValueError('prediction/cache identity')
                checkpoint = next(x for x in selected['runs'] if (x['role'], x['seed']) == (role, seed))['checkpoint_sha256']
                if rec['checkpoint_sha256'] != checkpoint:
                    raise ValueError('prediction checkpoint identity')
                with np.load(path, allow_pickle=False) as f:
                    if not np.array_equal(f['ids'], cache['ids']):
                        raise ValueError('prediction ID order')
                    actual, objects = metrics_numpy(f['logits'], cache['overlap'], cache['world_centers'])
                    if condition == DIAGNOSTIC:
                        with np.load(prediction_path(role, seed, 'base'), allow_pickle=False) as base:
                            probability = np.exp(f['logits']-f['logits'].max(-1, keepdims=True))
                            probability /= probability.sum(-1, keepdims=True)
                            p0 = np.exp(base['logits']-base['logits'].max(-1, keepdims=True))
                            p0 /= p0.sum(-1, keepdims=True)
                            diagnostic = dict(max_probability_difference=float(abs(probability-p0).max()),
                                winner_changes=int((f['logits'].argmax(-1) != base['logits'].argmax(-1)).sum()))
                for key, value in actual.items():
                    if key == 'valid_counts':
                        if value != rec['metrics'][key]:
                            raise ValueError('count disagreement')
                    else:
                        error = abs(value-rec['metrics'][key]);errors.append(error)
                        if error > 1e-6:
                            raise ValueError('substantive metric disagreement')
                dest = OUT/'metrics'/role/str(seed)
                dest.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest/(condition+'_objects.npz'), **objects, ids=cache['ids'])
                item = dict(role=role, seed=seed, condition=condition, reused=reused, **actual)
                if condition == DIAGNOSTIC:
                    diagnostics.append(dict(item, **diagnostic))
                else:
                    rows.append(item)
    write(OUT/'metrics/summary.json', dict(rows=rows, diagnostics=diagnostics, groups=len(rows),
        diagnostic_groups=len(diagnostics), models=30, new_fits=12, reused_fits=18,
        max_metric_error=max(errors), fresh_confirmation=False, goal_complete=False))
    print('METRICS',len(rows),'plus',len(diagnostics),'reframe diagnostics;error',max(errors),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('training', 'metrics'))
    torch.set_num_threads(4)
    globals()[parser.parse_args().stage]()
