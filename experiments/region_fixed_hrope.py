"""One old-data control: HRoPE with a fixed H=K and unchanged rotary transport."""
import argparse
import os
from pathlib import Path
import time
import numpy as np
import torch

from experiments import round68_fixed_region_run as engine
from experiments.region_hrope_control import check as check_parent, configure, INPUT_FIELDS
from experiments.region_hrope_data import CAMERA, load_cache
from experiments.region_geometry_controls import dependencies
from experiments.round65_common import ROOT, SEEDS, read, write, sha
from experiments.round58_data import loss_function
from experiments.round58_run import shared_digest
from models.region_hrope_control import CameraRegionMatcher
from models.region_fixed_hrope import FixedHRegionMatcher

BASE = ROOT / 'results/region_fixed_hrope'
PREP = BASE / 'preparation'
OUT = BASE / 'attempt1'
DEV = ROOT / 'results/region_hrope_control/attempt1'
PLAN = ROOT / 'docs/region_fixed_hrope_plan.md'
ROLES = ('fixed_hrope',)
COMPARISONS = [('predicted_hrope', 'fixed_hrope'), ('relative5', 'fixed_hrope'),
               ('fixed_hrope', 'distance_mlp'), ('fixed_hrope', 'geotransformer_rpe')]


def freeze():
    check_parent()
    assert not OUT.exists()
    assert read(DEV / 'review.json')['scientific_review_complete']
    assert read(PREP / 'cpu_checks.json')['passed']
    paths = dependencies([Path(__file__), ROOT / 'models/region_fixed_hrope.py',
        ROOT / 'tests/test_region_fixed_hrope.py', ROOT / 'experiments/region_fixed_hrope_pose.py',
        ROOT / 'experiments/region_fixed_hrope_queue.py', ROOT / 'experiments/region_fixed_hrope_report.py'])
    sources = {str(p): sha(p) for p in paths}
    for p in (PLAN, PREP / 'cpu_checks.json', CAMERA / 'manifest.json', DEV / 'selected.json',
              DEV / 'result.json', DEV / 'review.json', DEV / 'lock.json'):
        sources[str(p)] = sha(p)
    old = read(DEV / 'result.json')
    receipts = [r['receipt'] for r in old['records']]
    assert len(receipts) == 17
    for r in receipts:
        assert sha(r['path']) == r['sha256']
    pose = read(DEV / 'pose_analysis_lock.json')
    write(OUT / 'lock.json', dict(created_unix=time.time(), source_sha256=sources,
        settings=dict(engine.SETTINGS, roles=ROLES, input_fields=INPUT_FIELDS, conditions=['base']),
        seeds=list(SEEDS), roles=ROLES, new_fits=3, comparisons=COMPARISONS,
        bootstrap_seeds=list(range(97001, 97005)), bootstrap_replicates=10000,
        cache=pose['cache'], reused_pose_receipts=receipts,
        parent_lock_sha256=sha(DEV / 'lock.json'), parent_review_sha256=sha(DEV / 'review.json'),
        hypothesis='Learned token homographies add alignment utility beyond the identical fixed-H rotary transport.',
        competing_explanation='Fixed H=K and rotary transport already explain the useful encoding effect.',
        consensus_radius_pair_units=.1, maximum_consensus_refits=5,
        alignment_rmse_threshold_m=.2, fresh_confirmation=False,
        completed_confirmation_cohort_accessed=False, goal_complete=False))
    (OUT / 'lock.sha256').write_text(sha(OUT / 'lock.json')+'\n')
    write(OUT / 'pose_analysis_lock.json', dict(created_unix=time.time(),
        study_lock_sha256=sha(OUT / 'lock.json'), cache=pose['cache'], diagnostics=receipts,
        comparisons=COMPARISONS, bootstrap_seeds=list(range(97001, 97005)),
        consensus_radius_pair_units=.1, maximum_consensus_refits=5,
        reused_solver_preflight_sha256=pose['reused_solver_cpu_gpu_preflight_sha256'],
        source_sha256={str(ROOT / 'experiments/region_fixed_hrope_pose.py'):
                       sha(ROOT / 'experiments/region_fixed_hrope_pose.py')},
        fresh_confirmation=False, new_fits=3, goal_complete=False))
    print('FIXED_H_CONTROL_FROZEN', sha(OUT / 'lock.json'), flush=True)


def check():
    check_parent()
    lock = read(OUT / 'lock.json')
    assert sha(OUT / 'lock.json') == (OUT / 'lock.sha256').read_text().strip()
    for path, digest in lock['source_sha256'].items():
        assert sha(path) == digest, path
    return lock


def bind():
    engine.OUT = OUT
    engine.ROLES = ROLES
    engine.FixedRegionMatcher = FixedHRegionMatcher
    engine.check = check
    engine.configure = configure
    engine.load_cache = load_cache
    engine.SETTINGS = dict(engine.SETTINGS, roles=ROLES, input_fields=INPUT_FIELDS, conditions=['base'])


def preflight():
    check(); configure(); bind()
    assert not (OUT / 'preflight.json').exists()
    cache = load_cache('fit0')
    index = np.arange(32)
    torch.manual_seed(11)
    original = CameraRegionMatcher('predicted_hrope').cuda()
    torch.manual_seed(11)
    model = FixedHRegionMatcher().cuda()
    assert shared_digest(original) == shared_digest(model)
    with torch.no_grad():
        a, b = original(*engine.inputs(cache, index)), model(*engine.inputs(cache, index))
        error = float((a-b).abs().max())
        assert error < 2e-5
        assert torch.equal(a.argmax(-1), b.argmax(-1))
    del original, a, b
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    losses = []
    before = model.position.raw.clone()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        logits = model(*engine.inputs(cache, index))
        loss = loss_function(logits, torch.as_tensor(cache['overlap'][index], device='cuda'))
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        camera_gradient = float(model.stem[0].weight.grad[:, 20:].norm())
        assert camera_gradient > 0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step(); losses.append(float(loss.detach()))
    assert torch.equal(before, model.position.raw)
    write(OUT / 'preflight.json', dict(passed=True, completed_unix=time.time(),
        device=torch.cuda.get_device_name(), CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],
        batch_size=32, initial_hrope_max_logit_error=error, prediction_changes=0,
        common_initialization_equal=True, constant_head_unchanged=True, discarded_updates=2,
        losses=losses, camera_gradient_norm=camera_gradient,
        parameters=sum(p.numel() for p in model.parameters()), pe_parameters=0,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), lock_sha256=sha(OUT / 'lock.json')))
    print('FIXED_H_PREFLIGHT_PASSED', losses, flush=True)


def selections():
    check(); bind()
    assert not (OUT / 'selected.json').exists()
    val = load_cache('val')
    records, pairing = [], []
    for seed in SEEDS:
        folder = OUT / 'runs/fixed_hrope' / str(seed)
        run, history = read(folder / 'run.json'), read(folder / 'history.json')
        assert run['epochs'] == 60 and run['steps'] == 1080 and len(history['training']) == 60
        for name, digest in run['hashes'].items():
            assert sha(folder / name) == digest
        streams = [r['stream_sha256'] for r in history['training']]
        for role in ('relative5', 'distance_mlp', 'geotransformer_rpe', 'fixed_isotropic5', 'predicted_hrope'):
            old = DEV / 'runs' / role / str(seed)
            assert run['initial_shared_sha256'] == read(old / 'run.json')['initial_shared_sha256']
            assert streams == [r['stream_sha256'] for r in read(old / 'history.json')['training']]
        best = min(history['validation'], key=lambda r: r['selection_ce'])
        assert best['epoch'] == run['selected_epoch']
        with np.load(folder / 'validation.npz', allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], val['ids'])
            values = engine.query_values(z['logits'], val)
        ce = [float(values['ce'][:, d][values['valid'][:, d]].mean()) for d in (0, 1)]
        error = abs(float(np.mean(ce))-run['best_val_ce'])
        assert error < 1e-10
        records.append(dict(role='fixed_hrope', seed=seed, selected_epoch=run['selected_epoch'],
            checkpoint_sha256=sha(folder / 'best.pt'), run_sha256=sha(folder / 'run.json'),
            validation_ce_error=error))
        pairing.append(dict(seed=seed,all_six_common_initializations_equal=True,all60_augmented_streams_equal=True))
    write(OUT / 'selected.json', dict(records=records, pairing=pairing, all3_selected_before_eval=True,
        lock_sha256=sha(OUT / 'lock.json')))
    print('FIXED_H_ALL3_SELECTED', flush=True)


def export(seed):
    check(); configure(); bind()
    selection = read(OUT / 'selected.json')
    assert len(selection['records']) == 3 and selection['all3_selected_before_eval']
    choice = next(r for r in selection['records'] if r['seed'] == seed)
    folder = OUT / 'runs/fixed_hrope' / str(seed)
    assert sha(folder / 'best.pt') == choice['checkpoint_sha256']
    model = FixedHRegionMatcher().cuda()
    model.load_state_dict(torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)['state_dict'])
    val, cache = load_cache('val'), load_cache('eval_base')
    logits, _ = engine.evaluate(model, val)
    with np.load(folder / 'validation.npz', allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'], val['ids'])
        error = float(np.abs(logits-z['logits']).max())
        changes = int((logits.argmax(-1) != z['logits'].argmax(-1)).sum())
        assert error < 2e-4 and changes == 0
    logits, independent = engine.evaluate(model, cache)
    metrics = engine.grouped_metrics(logits, cache)
    metric_error = max(abs(independent[k]-metrics['micro'][k]) for k in ('region_recall','center_pck','soft_epe','ce'))
    assert metric_error < 1e-10
    path = OUT / 'evaluation/fixed_hrope' / str(seed) / 'base.npz'
    assert not path.exists(); path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, logits=logits, ids=cache['ids'])
    write(path.with_suffix('.json'), dict(role='fixed_hrope', seed=seed, path=str(path), sha256=sha(path),
        checkpoint_sha256=choice['checkpoint_sha256'], metrics=metrics,
        validation_replay=dict(maximum_logit_error=error, prediction_changes=changes),
        independent_metric_error=metric_error, lock_sha256=sha(OUT / 'lock.json')))
    print('FIXED_H_EXPORTED', seed, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze','preflight','train','selections','export'))
    parser.add_argument('--seed', type=int, choices=SEEDS)
    args = parser.parse_args(); bind()
    if args.action == 'train':
        engine.train('fixed_hrope', args.seed)
    elif args.action == 'export':
        export(args.seed)
    else:
        globals()[args.action]()
