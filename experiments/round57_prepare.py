"""Freeze R57 and exact R56/R48 references before reading new task results."""
import argparse
from pathlib import Path
import shutil
import time
import numpy as np
import torch
import torch.nn.functional as F
from models.round56_native_joint import make_model as make_reference
from models.round57_gaussian_pair import make_model
from experiments.round56_common import checked as checked56
from experiments.round48_data import load, evaluation_points
from experiments.round48_train import configure, shared_digest
from experiments.round57_common import (OUT, ROOT, REFERENCE_OUT, CONFIG, VARIANTS,
    NEW_ROLES, REFERENCE_ROLES, ROLES, SEEDS, PRIMARY, DIAGNOSTIC_ROLES, INTERVENTIONS,
    sha, read, write, checked, source_paths, reference, model_path)


def freeze(out=OUT):
    if out.exists():
        raise ValueError('preserve existing R57 attempt')
    old = checked56(REFERENCE_OUT)
    status = read(REFERENCE_OUT/'queue_status.json')
    completion = read(REFERENCE_OUT/'completion.json')
    if status['state'] != 'complete' or status['active'] or not status['all_children_exit_zero'] or not completion['completed']:
        raise ValueError('R56 must already be completed')
    if old['config'] != CONFIG:
        raise ValueError('reference configuration changed')
    unit_path = ROOT/'results/round57_gaussian_pair/preparation1/unit_passed.json'
    units = read(unit_path)
    if units['exit_code'] != 0 or units['tests'] != 4:
        raise ValueError('actual successful bounded unit-test record required')
    for name, digest in units['source_hashes'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('tested scientific source changed: '+name)
    manifest = REFERENCE_OUT/'data/manifest.json'
    if sha(manifest) != manifest.with_suffix('.sha256').read_text().strip():
        raise ValueError('reference data manifest changed')
    references, paired = [], {}
    for role in REFERENCE_ROLES:
        for seed in SEEDS:
            path = model_path(role, seed, out)
            prediction_path = REFERENCE_OUT/'evaluation'/role/(str(seed)+'.npz')
            prediction_record_path = prediction_path.with_suffix('.json')
            run, prediction = read(path/'run.json'), read(prediction_record_path)
            if (run['role'], run['seed'], run['epochs'], run['steps']) != (role, seed, 120, 29760):
                raise ValueError('incomplete/misidentified reference')
            identity = [run['initial_shared_sha256'], run['batch_augmentation_sha256']]
            if paired.setdefault(str(seed), identity) != identity:
                raise ValueError('references are not paired')
            hashes = dict(run['hashes'], **{'run.json':sha(path/'run.json')})
            for name, digest in hashes.items():
                if sha(path/name) != digest:
                    raise ValueError('changed reference artifact '+str(path/name))
            if prediction['checkpoint_sha256'] != hashes['best.pt'] or sha(prediction_path) != prediction['predictions_sha256']:
                raise ValueError('reference prediction/model identity')
            references.append(dict(role=role, seed=seed, path=str(path), hashes=hashes,
                prediction_path=str(prediction_path), prediction_record_path=str(prediction_record_path),
                prediction_sha256=sha(prediction_path), prediction_record_sha256=sha(prediction_record_path),
                initial_shared_sha256=identity[0], batch_augmentation_sha256=identity[1],
                original_lock_sha256=sha(path.parents[2]/'lock.json'), reused=True))
    sources = {name:sha(ROOT/name) for name in source_paths()}
    inputs = {str(path):sha(path) for path in (REFERENCE_OUT/'lock.json', manifest,
        REFERENCE_OUT/'audit_training/completion.json', REFERENCE_OUT/'queue_status.json',
        REFERENCE_OUT/'completion.json', REFERENCE_OUT/'metrics/summary.json',
        REFERENCE_OUT/'metrics/verification.json', unit_path)}
    out.mkdir(parents=True)
    for name in sources:
        target = out/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    (out/'data').mkdir()
    for name in ('manifest.json', 'manifest.sha256'):
        shutil.copyfile(REFERENCE_OUT/'data'/name, out/'data'/name)
    shutil.copyfile(unit_path, out/'unit_tests.json')
    lock = dict(sources=sources, config=CONFIG, roles=ROLES, new_roles=NEW_ROLES,
        seeds=SEEDS, primary=PRIMARY, variants=VARIANTS, references=references,
        paired_reference_identity=paired, input_identities=inputs,
        copied_manifest_sha256=sha(out/'data/manifest.json'), frozen_unix=time.time(),
        official_test_access=False, fresh_confirmation=False, new_fits=9, reused_fits=39,
        diagnostics=INTERVENTIONS, diagnostics_roles=DIAGNOSTIC_ROLES)
    write(out/'lock.json', lock)
    (out/'lock.sha256').write_text(sha(out/'lock.json')+'\n')
    print('FROZEN',sha(out/'lock.json'),len(sources),'sources;39 references;9 new fits',flush=True)


def preflight(out=OUT):
    lock = checked(out)
    units = read(out/'unit_tests.json')
    if units['exit_code'] != 0 or units['tests'] != 4:
        raise ValueError('missing actual successful unit tests')
    for name, digest in units['source_hashes'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('tested scientific source changed: '+name)
    val, fit = load('val', out), load('fit', out)
    x = torch.from_numpy(evaluation_points(val, 'base')[:2])
    labels = torch.from_numpy(val['labels'][:2])
    rows = []
    for rec in lock['references']:
        role, seed = rec['role'], rec['seed'];path = Path(rec['path'])
        for name, digest in rec['hashes'].items():
            if sha(path/name) != digest:
                raise ValueError('reference changed before preflight')
        torch.manual_seed(seed);model = make_model(role).eval()
        if type(model) is not type(make_reference(role)) or shared_digest(model) != rec['initial_shared_sha256']:
            raise ValueError('reference factory/initialization changed')
        state = torch.load(path/'best.pt',map_location='cpu',weights_only=True)['state_dict']
        original = make_reference(role).eval();model.load_state_dict(state);original.load_state_dict(state)
        with torch.no_grad():
            error = float((model(x)-original(x)).abs().max())
        if error != 0:
            raise ValueError('reference function changed')
        rows.append(dict(role=role,seed=seed,function_error=error))
    gradients = []
    for role in NEW_ROLES:
        for seed in SEEDS:
            torch.manual_seed(seed);model = make_model(role)
            if shared_digest(model) != reference('rope',seed,lock)['initial_shared_sha256']:
                raise ValueError('unpaired new-model shared initialization')
        torch.manual_seed(11);model = make_model(role)
        loss = F.cross_entropy(model(x),labels,label_smoothing=CONFIG['label_smoothing']);loss.backward()
        values = {n:None if p.grad is None else float(p.grad.norm()) for n,p in model.named_parameters()}
        if not torch.isfinite(loss) or any(v is not None and not np.isfinite(v) for v in values.values()):
            raise ValueError('nonfinite real-input gradient')
        if not values['position_field.predictor.weight'] > 0:
            raise ValueError('new position predictor has no weight gradient')
        # A common center shift is a gauge: mean bias can have zero initial
        # derivative. Do not mistake this mathematical null direction for failure.
        gradients.append(dict(role=role,loss=float(loss),gradients=values,
                              initial_common_center_bias_null_direction_expected=True))
    write(out/'preflight.json',dict(passed=True,tests=units['tests'],unit_tests_reused_without_changes=True,references=rows,
        real_val_gradient_checks=gradients,fit_count=len(fit['ids']),val_count=len(val['ids']),
        dev_eval_arrays_read=False,official_test_access=False,unix=time.time()))
    print('PREFLIGHT',units['tests'],'tests;39 exact reference functions;3 real-input gradients',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','preflight'))
    parser.add_argument('--out',type=Path,default=OUT);args=parser.parse_args()
    configure();globals()[args.stage](args.out)
