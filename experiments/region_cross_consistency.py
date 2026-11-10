"""Nine old-data fits of a compact, learned-reference cross positional term."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch

from experiments import round68_fixed_region_run as engine
from experiments.region_rotary_relation import OUT as ROTARY, check as check_parent
from experiments.region_fixed_hrope import OUT as FIXED
from experiments.region_hrope_control import configure, INPUT_FIELDS, OUT as DEV
from experiments.region_hrope_data import CAMERA, load_cache
from experiments.region_geometry_controls import dependencies
from experiments.round65_common import ROOT, SEEDS, read, write, sha
from models.region_cross_consistency import ROLES, CrossConsistencyMatcher

BASE = ROOT/'results/region_cross_consistency'
PREP = BASE/'preparation'
OUT = BASE/'attempt1'
PLAN = ROOT/'docs/region_cross_consistency_plan.md'
DIAGNOSTIC = ROOT/'results/region_relation_consistency/diagnostic1'
SCALES = DIAGNOSTIC/'scales.json'
REUSED_ROLES = ('relative5','distance_mlp','geotransformer_rpe','fixed_isotropic5','predicted_hrope','fixed_hrope')
COMPARISONS = [('cross_full5',r) for r in ('cross_distance','fixed_hrope','cross_sphere5',
    'predicted_hrope','relative5','geotransformer_rpe','distance_mlp','fixed_isotropic5')]
BOOTSTRAP_SEEDS = list(range(99601,99609))
MINIMUM_GAIN = {r: (0 if r in ('cross_sphere5','relative5','fixed_isotropic5') else 1) for _,r in COMPARISONS}


def owner(role):
    return OUT if role in ROLES else FIXED if role == 'fixed_hrope' else DEV if role == 'predicted_hrope' else ROTARY


def model_factory(role):
    return CrossConsistencyMatcher(role, read(SCALES)['scales'][role.removeprefix('cross_')]['scale'])


def freeze():
    check_parent()
    assert not OUT.exists(), 'Preserve existing formal work.'
    assert read(ROTARY/'review.json')['scientific_review_complete']
    assert read(DIAGNOSTIC/'review.json')['scientific_review_complete']
    assert read(PREP/'status.json')['state'] == 'complete'
    cpu, gpu = read(PREP/'cpu_checks.json'), read(PREP/'gpu_preflight.json')
    assert cpu['passed'] and cpu['tests_passed'] == 3 and gpu['passed'] and gpu['all_weights_discarded']
    for path,digest in {**cpu['source_sha256'], **read(PREP/'preflight_started.json')['source_sha256']}.items():
        assert sha(path) == digest, path
    paths = dependencies([Path(__file__), ROOT/'experiments/round68_fixed_region_run.py',
        ROOT/'tests/test_region_cross_consistency.py', ROOT/'experiments/region_cross_consistency_pose.py',
        ROOT/'experiments/region_cross_consistency_queue.py', ROOT/'experiments/region_cross_consistency_report.py'])
    sources = {str(p):sha(p) for p in paths}
    for p in (PLAN, ROOT/'docs/region_cross_consistency_design.md', SCALES,
              PREP/'cpu_checks.json', PREP/'gpu_preflight.json', PREP/'preflight_started.json',
              CAMERA/'manifest.json', ROTARY/'result.json', ROTARY/'review.json', ROTARY/'lock.json',
              DIAGNOSTIC/'result.json', DIAGNOSTIC/'review.json'):
        sources[str(p)] = sha(p)
    receipts = [r['receipt'] for r in read(ROTARY/'result.json')['records']]
    assert len(receipts) == 20
    identities = {}
    for r in receipts:
        assert sha(r['path']) == r['sha256']
        identities[r['path']] = r['sha256']
        if r['input_prediction']:
            prediction = r['input_prediction']
            assert sha(prediction['path']) == prediction['sha256']
            identities[prediction['path']] = prediction['sha256']
    for role in REUSED_ROLES:
        for seed in SEEDS:
            folder = owner(role)
            for p in (folder/'runs'/role/str(seed)/'run.json',folder/'runs'/role/str(seed)/'history.json',
                      folder/'evaluation'/role/str(seed)/'base.json'):
                identities[str(p)] = sha(p)
    pose = read(ROTARY/'pose_analysis_lock.json')
    settings = dict(engine.SETTINGS, roles=ROLES, input_fields=INPUT_FIELDS, conditions=['base'])
    write(OUT/'lock.json',dict(created_unix=time.time(),source_sha256=sources,reused_artifact_sha256=identities,
        settings=settings,seeds=list(SEEDS),roles=ROLES,new_fits=9,reused_fits=18,pose_groups=29,
        comparisons=COMPARISONS,primary_comparisons=COMPARISONS[:2],minimum_gain_pp=MINIMUM_GAIN,
        bootstrap_seeds=BOOTSTRAP_SEEDS,bootstrap_replicates=10000,cache=pose['cache'],
        reused_pose_receipts=receipts,scale_sha256=sha(SCALES),reused_gpu_preflight_sha256=sha(PREP/'gpu_preflight.json'),
        hypothesis='Learned cross-attention references make five Gaussian relations useful beyond distance and fixed rotary transport.',
        competing_explanation='Incorrect and unmatched references erase the truth-assisted signal, or distance/rotary suffice.',
        coefficient_initialization=0.,coefficients_signed=True,reference_gradient_detached=False,
        consensus_radius_pair_units=.1,maximum_consensus_refits=5,alignment_rmse_threshold_m=.2,
        fresh_confirmation=False,completed_confirmation_cohort_accessed=False,goal_complete=False))
    (OUT/'lock.sha256').write_text(sha(OUT/'lock.json')+'\n')
    write(OUT/'pose_analysis_lock.json',dict(created_unix=time.time(),study_lock_sha256=sha(OUT/'lock.json'),
        cache=pose['cache'],diagnostics=receipts,comparisons=COMPARISONS,bootstrap_seeds=BOOTSTRAP_SEEDS,
        minimum_gain_pp=MINIMUM_GAIN,consensus_radius_pair_units=.1,maximum_consensus_refits=5,
        reused_solver_preflight_sha256=pose['reused_solver_preflight_sha256'],
        source_sha256={str(ROOT/'experiments/region_cross_consistency_pose.py'):sha(ROOT/'experiments/region_cross_consistency_pose.py')},
        fresh_confirmation=False,new_fits=9,goal_complete=False))
    write(OUT/'preflight.json',dict(passed=True,reused=True,source_path=str(PREP/'gpu_preflight.json'),
        sha256=sha(PREP/'gpu_preflight.json'),cpu_checks_sha256=sha(PREP/'cpu_checks.json'),
        all_weights_discarded=True,lock_sha256=sha(OUT/'lock.json')))
    print('CROSS_CONSISTENCY_FROZEN',sha(OUT/'lock.json'),flush=True)


def check():
    check_parent()
    lock = read(OUT/'lock.json')
    assert sha(OUT/'lock.json') == (OUT/'lock.sha256').read_text().strip()
    for group in ('source_sha256','reused_artifact_sha256'):
        for path,digest in lock[group].items():
            assert sha(path) == digest,path
    return lock


def bind():
    engine.OUT = OUT
    engine.ROLES = ROLES
    engine.FixedRegionMatcher = model_factory
    engine.check = check
    engine.configure = configure
    engine.load_cache = load_cache
    engine.SETTINGS = dict(engine.SETTINGS,roles=ROLES,input_fields=INPUT_FIELDS,conditions=['base'])


def selections():
    check();bind()
    assert not (OUT/'selected.json').exists()
    val = load_cache('val'); records,pairing = [],[]
    for seed in SEEDS:
        for role in ROLES:
            folder = OUT/'runs'/role/str(seed)
            run,history = read(folder/'run.json'),read(folder/'history.json')
            assert run['epochs'] == 60 and run['steps'] == 1080 and len(history['training']) == 60
            assert run['lock_sha256'] == sha(OUT/'lock.json')
            for name,digest in run['hashes'].items():assert sha(folder/name) == digest
            for old_role in REUSED_ROLES:
                old = owner(old_role)/'runs'/old_role/str(seed)
                assert run['initial_shared_sha256'] == read(old/'run.json')['initial_shared_sha256']
                assert [r['stream_sha256'] for r in history['training']] == [r['stream_sha256'] for r in read(old/'history.json')['training']]
            best = min(history['validation'],key=lambda r:r['selection_ce'])
            assert best['epoch'] == run['selected_epoch']
            with np.load(folder/'validation.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],val['ids']);values = engine.query_values(z['logits'],val)
            ce = [float(values['ce'][:,d][values['valid'][:,d]].mean()) for d in (0,1)]
            error = abs(float(np.mean(ce))-run['best_val_ce']);assert error < 1e-10
            state = torch.load(folder/'best.pt',map_location='cpu',weights_only=True)['state_dict']
            expected = torch.tensor(read(SCALES)['scales'][role.removeprefix('cross_')]['scale'],dtype=torch.float32)
            torch.testing.assert_close(state['position.scale'],expected,atol=0,rtol=0)
            records.append(dict(role=role,seed=seed,selected_epoch=run['selected_epoch'],
                checkpoint_sha256=sha(folder/'best.pt'),run_sha256=sha(folder/'run.json'),validation_ce_error=error,
                selected_coefficients=state['position.coefficients'].tolist(),fixed_scale=expected.tolist()))
            pairing.append(dict(role=role,seed=seed,all_reused_common_initializations_equal=True,all60_augmented_streams_equal=True))
    write(OUT/'selected.json',dict(records=records,pairing=pairing,all9_selected_before_eval=True,lock_sha256=sha(OUT/'lock.json')))
    print('CROSS_CONSISTENCY_ALL9_SELECTED',flush=True)


def export(seed):
    check();configure();bind()
    selection = read(OUT/'selected.json')
    assert len(selection['records']) == 9 and selection['all9_selected_before_eval']
    val,cache = load_cache('val'),load_cache('eval_base')
    for role in ROLES:
        choice = next(r for r in selection['records'] if (r['role'],r['seed']) == (role,seed))
        folder = OUT/'runs'/role/str(seed)
        assert sha(folder/'best.pt') == choice['checkpoint_sha256']
        model = model_factory(role).cuda()
        model.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=True)['state_dict'])
        logits,_ = engine.evaluate(model,val)
        with np.load(folder/'validation.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],val['ids'])
            error = float(np.abs(logits-z['logits']).max());changes = int((logits.argmax(-1)!=z['logits'].argmax(-1)).sum())
            assert error < 2e-4 and changes == 0
        logits,independent = engine.evaluate(model,cache)
        metrics = engine.grouped_metrics(logits,cache)
        metric_error = max(abs(independent[k]-metrics['micro'][k]) for k in ('region_recall','center_pck','soft_epe','ce'))
        assert metric_error < 1e-10
        path = OUT/'evaluation'/role/str(seed)/'base.npz'
        assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,logits=logits,ids=cache['ids'])
        write(path.with_suffix('.json'),dict(role=role,seed=seed,path=str(path),sha256=sha(path),
            checkpoint_sha256=choice['checkpoint_sha256'],metrics=metrics,
            validation_replay=dict(maximum_logit_error=error,prediction_changes=changes),
            independent_metric_error=metric_error,lock_sha256=sha(OUT/'lock.json')))
        print('CROSS_CONSISTENCY_EXPORTED',role,seed,flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=('freeze','train','selections','export'))
    parser.add_argument('--role',choices=ROLES)
    parser.add_argument('--seed',type=int,choices=SEEDS)
    args = parser.parse_args();bind()
    if args.action == 'train':engine.train(args.role,args.seed)
    elif args.action == 'export':export(args.seed)
    else:globals()[args.action]()
