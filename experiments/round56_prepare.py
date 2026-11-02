"""Freeze the new experiment and verify exact, already-used R48 references."""
import argparse
from pathlib import Path
import shutil
import time
import unittest
import numpy as np
import torch
import torch.nn.functional as F
from models.round48_point_transformer import make_model as make_reference
from models.round56_native_joint import make_model
from experiments.round48_common import checked as checked48
from experiments.round48_data import load, evaluation_points
from experiments.round48_train import configure, shared_digest
from experiments.round56_common import (OUT, ROOT, REFERENCE_OUT, CONFIG, VARIANTS,
    NEW_ROLES, REFERENCE_ROLES, ROLES, SEEDS, PRIMARY, sha, read, write, checked,
    source_paths, reference)


def freeze(out=OUT):
    if out.exists():
        raise ValueError('preserve existing R56 attempt')
    old = checked48(REFERENCE_OUT)
    status = read(REFERENCE_OUT/'queue_status.json')
    if status['state'] != 'complete' or status['active'] or not status['all_children_exit_zero']:
        raise ValueError('R48 must already be completed')
    if old['config'] != CONFIG:
        raise ValueError('old training configuration changed')
    manifest = REFERENCE_OUT/'data/manifest.json'
    if sha(manifest) != manifest.with_suffix('.sha256').read_text().strip():
        raise ValueError('old manifest changed')
    references, paired = [], {}
    for role in REFERENCE_ROLES:
        for seed in SEEDS:
            path = REFERENCE_OUT/'runs'/role/str(seed)
            run, prediction = read(path/'run.json'), read(path/'dev_eval.json')
            if run['role'] != role or run['seed'] != seed or run['epochs'] != 120 or run['steps'] != 29760:
                raise ValueError('incomplete/misidentified reference')
            identity = [run['initial_shared_sha256'], run['batch_augmentation_sha256']]
            if paired.setdefault(str(seed), identity) != identity:
                raise ValueError('R48 references not paired')
            hashes = dict(run['hashes'], **{'run.json':sha(path/'run.json'),
                'dev_eval.json':sha(path/'dev_eval.json'), 'dev_eval.npz':prediction['predictions_sha256']})
            if prediction['checkpoint_sha256'] != hashes['best.pt']:
                raise ValueError('R48 selection/prediction mismatch')
            for name, digest in hashes.items():
                if sha(path/name) != digest:
                    raise ValueError('changed R48 reference '+str(path/name))
            references.append(dict(role=role, seed=seed, path=str(path), hashes=hashes,
                initial_shared_sha256=identity[0], batch_augmentation_sha256=identity[1],
                original_lock_sha256=sha(REFERENCE_OUT/'lock.json'), reused=True))
    sources = {name:sha(ROOT/name) for name in source_paths()}
    inputs = {str(path):sha(path) for path in (REFERENCE_OUT/'lock.json', manifest,
        REFERENCE_OUT/'audit_training/completion.json', REFERENCE_OUT/'queue_status.json')}
    out.mkdir(parents=True)
    for name in sources:
        target = out/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    (out/'data').mkdir()
    for name in ('manifest.json', 'manifest.sha256'):
        shutil.copyfile(REFERENCE_OUT/'data'/name, out/'data'/name)
    lock = dict(sources=sources, config=CONFIG, roles=ROLES, new_roles=NEW_ROLES,
        seeds=SEEDS, primary=PRIMARY, variants=VARIANTS, references=references,
        paired_reference_identity=paired, input_identities=inputs,
        copied_manifest_sha256=sha(out/'data/manifest.json'), frozen_unix=time.time(),
        official_test_access=False, fresh_confirmation=False, new_fits=9, reused_fits=30,
        diagnostics=['isotropic_trace','rotate_z90'], diagnostics_roles=['joint_conditional','conditional'])
    write(out/'lock.json', lock)
    (out/'lock.sha256').write_text(sha(out/'lock.json')+'\n')
    print('FROZEN', sha(out/'lock.json'), len(sources), 'sources; 30 references; 9 new fits', flush=True)


def preflight(out=OUT):
    lock = checked(out)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(
        str(ROOT/'tests'), pattern='test_round56_native_joint.py'))
    if not result.wasSuccessful():
        write(out/'preflight.json', dict(passed=False, tests=result.testsRun))
        raise SystemExit(1)
    val, fit = load('val', out), load('fit', out)
    x = torch.from_numpy(evaluation_points(val, 'base')[:2])
    labels = torch.from_numpy(val['labels'][:2])
    rows = []
    for rec in lock['references']:
        role, seed = rec['role'], rec['seed']
        path = Path(rec['path'])
        for name, digest in rec['hashes'].items():
            if sha(path/name) != digest:
                raise ValueError('reference changed before preflight')
        torch.manual_seed(seed)
        m = make_model(role).eval()
        if type(m) is not type(make_reference(role)) or shared_digest(m) != rec['initial_shared_sha256']:
            raise ValueError('reference factory/shared initialization mismatch')
        state = torch.load(path/'best.pt', map_location='cpu', weights_only=True)['state_dict']
        old_model = make_reference(role).eval()
        m.load_state_dict(state)
        old_model.load_state_dict(state)
        with torch.no_grad():
            error = float((m(x)-old_model(x)).abs().max())
        if error != 0:
            raise ValueError('reused reference function changed')
        rows.append(dict(role=role, seed=seed, function_error=error))
    gradients = []
    for role in NEW_ROLES:
        for seed in SEEDS:
            torch.manual_seed(seed)
            model = make_model(role)
            ref = reference('rope', seed, lock)
            if shared_digest(model) != ref['initial_shared_sha256']:
                raise ValueError('new model backbone initialization changed')
        torch.manual_seed(11)
        model = make_model(role)
        loss = F.cross_entropy(model(x), labels, label_smoothing=CONFIG['label_smoothing'])
        loss.backward()
        values = {n: None if p.grad is None else float(p.grad.norm()) for n,p in model.named_parameters()}
        if not torch.isfinite(loss) or any(v is not None and not np.isfinite(v) for v in values.values()):
            raise ValueError('invalid real-input gradient')
        if role != 'joint_fixed' and not all(v is not None and v > 0 for n,v in values.items() if n.startswith('position_field.')):
            raise ValueError('regional predictor has no initial gradient')
        gradients.append(dict(role=role, loss=float(loss), gradients=values))
    write(out/'preflight.json', dict(passed=True, tests=result.testsRun, references=rows,
        real_val_gradient_checks=gradients, fit_count=len(fit['ids']), val_count=len(val['ids']),
        dev_eval_arrays_read=False, official_test_access=False, unix=time.time()))
    print('PREFLIGHT', result.testsRun, 'tests,30 exact reference functions,3 real-input gradients', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('freeze','preflight'))
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    configure()
    globals()[args.stage](args.out)
