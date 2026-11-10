"""Fixed regional relations on the same rotary backbone: old-data development."""
import argparse
import os
from pathlib import Path
import time
import numpy as np
import torch

from experiments import round68_fixed_region_run as engine
from experiments.region_fixed_hrope import check as check_parent, OUT as FIXED
from experiments.region_hrope_control import configure, INPUT_FIELDS, OUT as DEV
from experiments.region_hrope_data import CAMERA, load_cache
from experiments.region_geometry_controls import dependencies
from experiments.round65_common import ROOT, SEEDS, read, write, sha
from experiments.round58_data import loss_function
from experiments.round58_run import shared_digest
from models.region_rotary_relation import ROLES, RotaryRelationMatcher

BASE = ROOT/'results/region_rotary_relation'
PREP = BASE/'preparation'
OUT = BASE/'attempt1'
PLAN = ROOT/'docs/region_rotary_relation_plan.md'
COMPARISONS = [('relative5','distance_mlp'), ('relative5','fixed_hrope'),
    ('relative5','geotransformer_rpe'), ('relative5','predicted_hrope'), ('relative5','fixed_isotropic5')]


def freeze():
    check_parent()
    assert not OUT.exists()
    assert read(FIXED/'review.json')['scientific_review_complete']
    cpu = read(PREP/'cpu_checks.json')
    assert cpu['passed'] and cpu['tests_passed'] == 3
    for path, digest in cpu['source_sha256'].items():
        assert sha(path) == digest
    paths = dependencies([Path(__file__), ROOT/'models/region_rotary_relation.py',
        ROOT/'tests/test_region_rotary_relation.py', ROOT/'experiments/region_rotary_relation_pose.py',
        ROOT/'experiments/region_rotary_relation_queue.py', ROOT/'experiments/region_rotary_relation_report.py'])
    sources = {str(p):sha(p) for p in paths}
    for p in (PLAN, PREP/'cpu_checks.json', CAMERA/'manifest.json', DEV/'selected.json',
              FIXED/'result.json', FIXED/'review.json', FIXED/'lock.json'):
        sources[str(p)] = sha(p)
    receipts = [r['receipt'] for r in read(FIXED/'result.json')['records']
                if r['role'] in ('fixed_hrope','predicted_hrope','identity','gt_nearest')]
    assert len(receipts) == 8
    for r in receipts:
        assert sha(r['path']) == r['sha256']
    pose = read(FIXED/'pose_analysis_lock.json')
    write(OUT/'lock.json', dict(created_unix=time.time(), source_sha256=sources,
        settings=dict(engine.SETTINGS, roles=ROLES, input_fields=INPUT_FIELDS, conditions=['base']),
        seeds=list(SEEDS), roles=ROLES, new_fits=12, reused_fits=6, comparisons=COMPARISONS,
        primary_comparisons=COMPARISONS[:2], bootstrap_seeds=list(range(98001,98006)), bootstrap_replicates=10000,
        cache=pose['cache'], reused_pose_receipts=receipts,
        parent_lock_sha256=sha(FIXED/'lock.json'), parent_review_sha256=sha(FIXED/'review.json'),
        hypothesis='Fixed Gaussian relations retain alignment utility conditional on a matched 2D rotary backbone.',
        competing_explanation='Prior relation utility overlaps with the image rotary cue or generic self-attention preferences.',
        consensus_radius_pair_units=.1, maximum_consensus_refits=5, alignment_rmse_threshold_m=.2,
        fresh_confirmation=False, completed_confirmation_cohort_accessed=False, goal_complete=False))
    (OUT/'lock.sha256').write_text(sha(OUT/'lock.json')+'\n')
    write(OUT/'pose_analysis_lock.json', dict(created_unix=time.time(), study_lock_sha256=sha(OUT/'lock.json'),
        cache=pose['cache'], diagnostics=receipts, comparisons=COMPARISONS,
        bootstrap_seeds=list(range(98001,98006)), consensus_radius_pair_units=.1, maximum_consensus_refits=5,
        reused_solver_preflight_sha256=pose['reused_solver_preflight_sha256'],
        source_sha256={str(ROOT/'experiments/region_rotary_relation_pose.py'):sha(ROOT/'experiments/region_rotary_relation_pose.py')},
        fresh_confirmation=False, new_fits=12, goal_complete=False))
    print('ROTARY_RELATION_FROZEN', sha(OUT/'lock.json'), flush=True)


def check():
    check_parent()
    lock = read(OUT/'lock.json')
    assert sha(OUT/'lock.json') == (OUT/'lock.sha256').read_text().strip()
    for path, digest in lock['source_sha256'].items():
        assert sha(path) == digest, path
    return lock


def bind():
    engine.OUT = OUT
    engine.ROLES = ROLES
    engine.FixedRegionMatcher = RotaryRelationMatcher
    engine.check = check
    engine.configure = configure
    engine.load_cache = load_cache
    engine.SETTINGS = dict(engine.SETTINGS, roles=ROLES, input_fields=INPUT_FIELDS, conditions=['base'])


def preflight():
    check(); configure(); bind()
    assert not (OUT/'preflight.json').exists()
    cache = load_cache('fit0'); index = np.arange(32)
    rows, common = [], None
    for role in ROLES:
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(11)
        model = RotaryRelationMatcher(role).cuda().train()
        digest = shared_digest(model)
        common = digest if common is None else common
        assert digest == common
        constant = model.fixed_transform.raw.clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        losses = []
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            logits = model(*engine.inputs(cache, index))
            loss = loss_function(logits, torch.as_tensor(cache['overlap'][index], device='cuda'))
            assert torch.isfinite(loss)
            loss.backward()
            gradients = {n:float(p.grad.norm()) if p.grad is not None else None for n,p in model.position.named_parameters()}
            assert all(v is not None and np.isfinite(v) for v in gradients.values()) and any(v > 0 for v in gradients.values())
            camera_gradient = float(model.stem[0].weight.grad[:,20:].norm())
            assert np.isfinite(camera_gradient) and camera_gradient > 0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); losses.append(float(loss.detach()))
        assert torch.equal(constant, model.fixed_transform.raw)
        rows.append(dict(role=role, losses=losses, pe_gradients=gradients, constant_transform_unchanged=True,
            camera_gradient_norm=camera_gradient, parameters=sum(p.numel() for p in model.parameters()),
            pe_parameters=sum(p.numel() for p in model.position.parameters()), peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        print('ROTARY_RELATION_PREFLIGHT',role,losses,rows[-1]['peak_allocated_bytes'],flush=True)
        del model, optimizer, logits, loss
        torch.cuda.empty_cache()
    write(OUT/'preflight.json',dict(passed=True,completed_unix=time.time(),rows=rows,batch_size=32,
        discarded_updates_per_model=2,common_initialization_sha256=common,device=torch.cuda.get_device_name(),
        CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],lock_sha256=sha(OUT/'lock.json')))


def selections():
    check(); bind()
    assert not (OUT/'selected.json').exists()
    val = load_cache('val'); records, pairing = [], []
    for seed in SEEDS:
        for role in ROLES:
            folder = OUT/'runs'/role/str(seed)
            run, history = read(folder/'run.json'), read(folder/'history.json')
            assert run['epochs'] == 60 and run['steps'] == 1080 and len(history['training']) == 60
            for name,digest in run['hashes'].items():
                assert sha(folder/name) == digest
            for old in (DEV/'runs'/role/str(seed), DEV/'runs/predicted_hrope'/str(seed), FIXED/'runs/fixed_hrope'/str(seed)):
                assert run['initial_shared_sha256'] == read(old/'run.json')['initial_shared_sha256']
                assert [r['stream_sha256'] for r in history['training']] == [r['stream_sha256'] for r in read(old/'history.json')['training']]
            best = min(history['validation'],key=lambda r:r['selection_ce'])
            assert best['epoch'] == run['selected_epoch']
            with np.load(folder/'validation.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],val['ids']); values = engine.query_values(z['logits'],val)
            ce = [float(values['ce'][:,d][values['valid'][:,d]].mean()) for d in (0,1)]
            error = abs(float(np.mean(ce))-run['best_val_ce']); assert error < 1e-10
            records.append(dict(role=role,seed=seed,selected_epoch=run['selected_epoch'],checkpoint_sha256=sha(folder/'best.pt'),
                run_sha256=sha(folder/'run.json'),validation_ce_error=error))
            pairing.append(dict(role=role,seed=seed,common_initialization_equal=True,all60_augmented_streams_equal=True))
    write(OUT/'selected.json',dict(records=records,pairing=pairing,all12_selected_before_eval=True,lock_sha256=sha(OUT/'lock.json')))
    print('ROTARY_RELATION_ALL12_SELECTED',flush=True)


def export(seed):
    check(); configure(); bind()
    selection = read(OUT/'selected.json')
    assert len(selection['records']) == 12 and selection['all12_selected_before_eval']
    val, cache = load_cache('val'), load_cache('eval_base')
    for role in ROLES:
        choice = next(r for r in selection['records'] if (r['role'],r['seed']) == (role,seed))
        folder = OUT/'runs'/role/str(seed)
        assert sha(folder/'best.pt') == choice['checkpoint_sha256']
        model = RotaryRelationMatcher(role).cuda()
        model.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=True)['state_dict'])
        logits,_ = engine.evaluate(model,val)
        with np.load(folder/'validation.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],val['ids'])
            error = float(np.abs(logits-z['logits']).max())
            changes = int((logits.argmax(-1)!=z['logits'].argmax(-1)).sum())
            assert error < 2e-4 and changes == 0
        logits,independent = engine.evaluate(model,cache)
        metrics = engine.grouped_metrics(logits,cache)
        metric_error = max(abs(independent[k]-metrics['micro'][k]) for k in ('region_recall','center_pck','soft_epe','ce'))
        assert metric_error < 1e-10
        path = OUT/'evaluation'/role/str(seed)/'base.npz'
        assert not path.exists(); path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,logits=logits,ids=cache['ids'])
        write(path.with_suffix('.json'),dict(role=role,seed=seed,path=str(path),sha256=sha(path),
            checkpoint_sha256=choice['checkpoint_sha256'],metrics=metrics,validation_replay=dict(maximum_logit_error=error,prediction_changes=changes),
            independent_metric_error=metric_error,lock_sha256=sha(OUT/'lock.json')))
        print('ROTARY_RELATION_EXPORTED',role,seed,flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=('freeze','preflight','train','selections','export'))
    parser.add_argument('--role',choices=ROLES)
    parser.add_argument('--seed',type=int,choices=SEEDS)
    args = parser.parse_args(); bind()
    if args.action == 'train': engine.train(args.role,args.seed)
    elif args.action == 'export': export(args.seed)
    else: globals()[args.action]()
