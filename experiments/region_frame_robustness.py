"""Frozen existing-model comparison under independent local frame changes."""
import argparse
import hashlib
from pathlib import Path
import time
import numpy as np
import torch

from experiments import round68_fixed_region_run as engine
from experiments.region_rotary_relation import OUT as PARENT, check as check_parent
from experiments.region_fixed_hrope import OUT as FIXED
from experiments.region_hrope_control import OUT as DEV, INPUT_FIELDS, configure
from experiments.region_hrope_data import load_cache, CAMERA
from experiments.region_geometry_controls import dependencies
from experiments.region_frame_transform import frames,reframe,SALT
from experiments.round65_common import ROOT,SEEDS,read,write,sha
from models.region_rotary_relation import RotaryRelationMatcher
from models.region_hrope_control import CameraRegionMatcher
from models.region_fixed_hrope import FixedHRegionMatcher
from models.round69_compact_relation import descriptors

BASE=ROOT/'results/region_frame_robustness'
PREP=BASE/'preparation'
OUT=BASE/'attempt1'
PLAN=ROOT/'docs/region_frame_robustness_plan.md'
ROLES=('relative5','distance_mlp','geotransformer_rpe','fixed_isotropic5','predicted_hrope','fixed_hrope')
COMPARISONS=[('relative5_reframed',r+'_reframed') for r in
    ('distance_mlp','fixed_hrope','geotransformer_rpe','predicted_hrope','fixed_isotropic5')]


def owner(role):
    return FIXED if role=='fixed_hrope' else DEV if role=='predicted_hrope' else PARENT


def model_factory(role):
    if role=='fixed_hrope':return FixedHRegionMatcher()
    if role=='predicted_hrope':return CameraRegionMatcher(role)
    return RotaryRelationMatcher(role)


def digest_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def digest_state(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def freeze():
    check_parent();assert not OUT.exists()
    assert read(PARENT/'review.json')['scientific_review_complete']
    cpu=read(PREP/'cpu_checks.json');assert cpu['passed'] and cpu['tests_passed']==3
    for p,d in cpu['source_sha256'].items():assert sha(p)==d,p
    sources={str(p):sha(p) for p in dependencies([Path(__file__),ROOT/'experiments/round68_fixed_region_run.py',
        ROOT/'experiments/region_frame_robustness_pose.py',ROOT/'experiments/region_frame_robustness_queue.py',
        ROOT/'experiments/region_frame_robustness_report.py',ROOT/'tests/test_region_frame_transform.py'])}
    for p in (PLAN,PREP/'cpu_checks.json',PARENT/'result.json',PARENT/'review.json',CAMERA/'manifest.json'):
        sources[str(p)]=sha(p)
    old=read(PARENT/'result.json');identities={};models=[]
    for record in old['records']:
        receipt=record['receipt'];assert sha(receipt['path'])==receipt['sha256']
        identities[receipt['path']]=receipt['sha256']
        prediction=receipt['input_prediction']
        if prediction:
            assert sha(prediction['path'])==prediction['sha256']
            identities[prediction['path']]=prediction['sha256']
    for seed in SEEDS:
        for role in ROLES:
            folder=owner(role);checkpoint=folder/'runs'/role/str(seed)/'best.pt'
            prediction=read(folder/'evaluation'/role/str(seed)/'base.json')
            run=read(folder/'runs'/role/str(seed)/'run.json')
            assert sha(checkpoint)==prediction['checkpoint_sha256']==run['hashes']['best.pt']
            identities[str(checkpoint)]=sha(checkpoint)
            for p in (folder/'evaluation'/role/str(seed)/'base.json',folder/'runs'/role/str(seed)/'run.json'):
                identities[str(p)]=sha(p)
            models.append(dict(role=role+'_reframed',base_role=role,seed=seed,checkpoint_path=str(checkpoint),
                checkpoint_sha256=sha(checkpoint),selected_epoch=run['selected_epoch'],base_prediction=prediction))
    pose=read(PARENT/'pose_analysis_lock.json')
    write(OUT/'lock.json',dict(created_unix=time.time(),source_sha256=sources,reused_artifact_sha256=identities,
        cache=pose['cache'],models=models,roles=ROLES,seeds=list(SEEDS),new_fits=0,
        new_prediction_groups=18,new_pose_groups=18,total_pose_groups=38,frame_salt=SALT,
        frame_distribution='Independent normalized Gaussian xyzw quaternion and U(-.5,.5)^3 translation for each ID/view.',
        comparisons=COMPARISONS,bootstrap_seeds=list(range(99801,99806)),
        interaction_bootstrap_seeds=list(range(99901,99906)),change_bootstrap_seeds=list(range(100001,100007)),
        bootstrap_replicates=10000,consensus_radius_pair_units=.1,maximum_consensus_refits=5,
        reused_pose_receipts=[r['receipt'] for r in old['records']],
        new_confirmation_access=False,fresh_confirmation=False,goal_complete=False))
    (OUT/'lock.sha256').write_text(sha(OUT/'lock.json')+'\n')
    write(OUT/'selected.json',dict(records=models,all18_selected_before_eval=True,original_checkpoint_selection_reused=True,
        new_selection_or_training=False,lock_sha256=sha(OUT/'lock.json')))
    write(OUT/'pose_analysis_lock.json',dict(created_unix=time.time(),study_lock_sha256=sha(OUT/'lock.json'),
        cache=pose['cache'],diagnostics=[r['receipt'] for r in old['records']],comparisons=COMPARISONS,
        bootstrap_seeds=list(range(99801,99806)),consensus_radius_pair_units=.1,maximum_consensus_refits=5,
        interaction_bootstrap_seeds=list(range(99901,99906)),change_bootstrap_seeds=list(range(100001,100007)),
        source_sha256={str(ROOT/'experiments/region_frame_robustness_pose.py'):sha(ROOT/'experiments/region_frame_robustness_pose.py')},
        fresh_confirmation=False,new_fits=0,goal_complete=False))
    print('FRAME_ROBUSTNESS_FROZEN',sha(OUT/'lock.json'),flush=True)


def check():
    check_parent();lock=read(OUT/'lock.json')
    assert sha(OUT/'lock.json')==(OUT/'lock.sha256').read_text().strip()
    for group in ('source_sha256','reused_artifact_sha256'):
        for p,d in lock[group].items():assert sha(p)==d,p
    return lock


def prepare():
    check();assert not (OUT/'data_checks.json').exists();torch.set_num_threads(4)
    rows=[]
    for name in ('fit0','eval_base'):
        base=load_cache(name)
        if name=='fit0':base={k:v[:8] for k,v in base.items()}
        changed,r,t=reframe(base)
        for key in set(base)-{'centers','covariance'}:np.testing.assert_array_equal(base[key],changed[key])
        inv=np.einsum('bvji,bvnj->bvni',r,changed['centers'].astype(np.float64)-t[:,:,None])
        center_error=float(np.abs(inv-base['centers']).max())
        back=r[:,:,None].swapaxes(-1,-2)@changed['covariance']@r[:,:,None]
        cov_error=float(np.abs(back-base['covariance']).max())
        assert center_error<2e-6 and cov_error<1e-10
        before=descriptors('relative5',torch.from_numpy(base['centers'][:8]),torch.from_numpy(base['covariance'][:8]))
        after=descriptors('relative5',torch.from_numpy(changed['centers'][:8]),torch.from_numpy(changed['covariance'][:8]))
        phi_error=float((before-after).abs().max());assert phi_error<2e-4
        spectrum_error=float(np.abs(np.linalg.eigvalsh(changed['covariance'])-np.linalg.eigvalsh(base['covariance'].astype(np.float64))).max())
        assert spectrum_error<1e-10
        rows.append(dict(cache=name,pairs=len(base['ids']),center_roundtrip_max_error=center_error,
            covariance_roundtrip_max_error=cov_error,first8_float32_relation_max_error=phi_error,spectrum_max_error=spectrum_error,
            unchanged_fields=sorted(set(base)-{'centers','covariance'}),
            changed_centers_sha256=digest_array(changed['centers']),changed_covariance_sha256=digest_array(changed['covariance'])))
        if name=='eval_base':np.savez(OUT/'frames.npz',ids=base['ids'],rotation=r,translation=t)
    write(OUT/'data_checks.json',dict(passed=True,rows=rows,completed_unix=time.time(),lock_sha256=sha(OUT/'lock.json'),
        frames_sha256=sha(OUT/'frames.npz'),model_outcomes_accessed=False,new_confirmation_access=False))
    print('FRAME_ROBUSTNESS_DATA_CHECKS_PASSED',flush=True)


def export(seed):
    lock=check();checks=read(OUT/'data_checks.json');assert checks['passed']
    configure();engine.SETTINGS=dict(engine.SETTINGS,input_fields=INPUT_FIELDS,eval_batch_size=16)
    base=load_cache('eval_base');changed,r,t=reframe(base)
    row=next(x for x in checks['rows'] if x['cache']=='eval_base')
    assert digest_array(changed['centers'])==row['changed_centers_sha256']
    assert digest_array(changed['covariance'])==row['changed_covariance_sha256']
    with np.load(OUT/'frames.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'],base['ids']);np.testing.assert_array_equal(z['rotation'],r);np.testing.assert_array_equal(z['translation'],t)
    for spec in [x for x in lock['models'] if x['seed']==seed]:
        path=OUT/'evaluation'/spec['role']/str(seed)/'base.npz';assert not path.exists()
        model=model_factory(spec['base_role']).cuda()
        model.load_state_dict(torch.load(spec['checkpoint_path'],map_location='cpu',weights_only=True)['state_dict'])
        before_state=digest_state(model)
        original,_=engine.evaluate(model,base)
        with np.load(spec['base_prediction']['path'],allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],base['ids'])
            replay_error=float(np.abs(original-z['logits']).max())
            replay_changes=int((original.argmax(-1)!=z['logits'].argmax(-1)).sum())
            assert replay_error<2e-4 and replay_changes==0
        logits,independent=engine.evaluate(model,changed)
        metrics=engine.grouped_metrics(logits,base)
        metric_error=max(abs(independent[k]-metrics['micro'][k]) for k in ('region_recall','center_pck','soft_epe','ce'))
        assert metric_error<1e-10 and digest_state(model)==before_state
        changed_queries=[];valid_queries=[]
        for axis in (1,2):
            a,b=logits,original;truth=base['overlap']
            if axis==1:a,b,truth=a.swapaxes(1,2),b.swapaxes(1,2),truth.swapaxes(1,2)
            changed_queries.append(a.argmax(-1)!=b.argmax(-1));valid_queries.append(truth.max(-1)>=.3)
        altered=np.stack(changed_queries,1);valid=np.stack(valid_queries,1)
        sensitivity=dict(all_query_top1_changed_fraction=float(altered.mean()),valid_query_top1_changed_fraction=float(altered[valid].mean()),
            changed_query_count=int(altered.sum()),valid_changed_query_count=int(altered[valid].sum()),
            total_queries=int(altered.size),valid_queries=int(valid.sum()),
            maximum_absolute_logit_change=float(np.abs(logits-original).max()),logit_rms_change=float(np.sqrt(np.square(logits.astype(float)-original).mean())))
        path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,logits=logits,ids=base['ids'])
        write(path.with_suffix('.json'),dict(role=spec['role'],base_role=spec['base_role'],seed=seed,path=str(path),sha256=sha(path),
            checkpoint_sha256=spec['checkpoint_sha256'],metrics=metrics,sensitivity=sensitivity,
            base_replay=dict(maximum_logit_error=replay_error,prediction_changes=replay_changes),
            independent_metric_error=metric_error,weights_unchanged=True,frames_sha256=sha(OUT/'frames.npz'),lock_sha256=sha(OUT/'lock.json')))
        print('FRAME_ROBUSTNESS_EXPORTED',spec['role'],seed,flush=True)
        del model;torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('freeze','prepare','export'))
    parser.add_argument('--seed',type=int,choices=SEEDS);args=parser.parse_args()
    if args.action=='export':export(args.seed)
    else:globals()[args.action]()
