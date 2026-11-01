"""CPU gradient checks, actual-renderer smoke, and matched image evaluation."""
import argparse
from pathlib import Path
import time
import unittest
import numpy as np
import torch
from scipy.special import logsumexp
from models.round41_gs_readout import GSReadout
from experiments.round41_gs_data import query,VAL_VIEWS
from experiments.round41_gs_audit import numpy_logits,arr
from experiments.round42_render_train import (ROOT,OUT,ROLES,OBJECTIVES,SCENES,SEEDS,
    checked,load_scene,read,write,sha)
from experiments.round42_render_core import (fused_colors,color_gradient,image_loss,
    accumulate_color_gradient)


def preflight(out):
    checked(out)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(
        str(ROOT/'tests'),pattern='test_round42_render_gradient.py'))
    write(out/'preflight.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,
        scene_arrays_read=False,gradient_reference='separate monolithic stack/mean graph, CPU float64'))
    if not result.wasSuccessful():raise SystemExit(1)


def smoke(out):
    checked(out)
    if not read(out/'preflight.json')['passed']:raise ValueError('CPU preflight required')
    if (out/'smoke.json').exists():raise ValueError('retain smoke')
    data=load_scene('teatime',out);rows=[]
    for objective in OBJECTIVES:
        torch.manual_seed(17);model=GSReadout('decoupled').cuda();initial=model.qk.weight.detach().clone()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        optimizer.zero_grad(set_to_none=True);colors,counts=fused_colors(model,data,0)
        loss,g=color_gradient(colors,data,0,counts,objective)
        directional=None
        if objective=='image':
            length=g.double().norm();direction=g/length.to(g.dtype);epsilon=.01
            with torch.no_grad():
                plus=float(image_loss(colors+epsilon*direction,data,0))
                minus=float(image_loss(colors-epsilon*direction,data,0))
            numeric=(plus-minus)/(2*epsilon);analytic=float((g.double()*direction.double()).sum())
            relative=abs(numeric-analytic)/max(abs(analytic),1e-12)
            directional=dict(epsilon=epsilon,analytic=analytic,finite_difference=numeric,relative_error=relative)
            if not np.isfinite(relative) or relative>.005:raise ValueError('renderer color derivative needs diagnosis '+str(directional))
        stats=accumulate_color_gradient(model,data,0,g,counts)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        change=float((model.qk.weight-initial).norm())
        if not torch.isfinite(loss) or not change>0:raise ValueError('actual-context gradient inactive')
        rows.append(dict(objective=objective,loss=float(loss),gradient_norm=float(norm),qk_change=change,
            renderer_directional_check=directional,**stats))
    write(out/'smoke.json',dict(passed=True,rows=rows,target_rgb_read=False,gpu=torch.cuda.get_device_name()))
    print('SMOKE',rows,flush=True)


@torch.no_grad()
def audit(out):
    checked(out);dest=out/'audit'
    if dest.exists():raise ValueError('retain audit')
    dest.mkdir();records=[];paired={};warnings=[]
    for scene in SCENES:
        data=load_scene(scene,out,device='cpu')
        for objective in OBJECTIVES:
            for role in ROLES:
                for seed in SEEDS:
                    path=out/'runs'/objective/scene/role/str(seed);meta=read(path/'run.json')
                    for f,k in (('best.pt','checkpoint_sha256'),('history.json','history_sha256'),
                            ('colors.npz','colors_sha256'),('predictions.npz','predictions_sha256'),('score_samples.npz','samples_sha256')):
                        if sha(path/f)!=meta[k]:raise ValueError('changed artifact')
                    identity=(meta['initial_qk_sha256'],meta['batch_sha256'])
                    key=(scene,seed)
                    if key in paired and paired[key]!=identity:raise ValueError('factorial not paired')
                    paired[key]=identity
                    history=read(path/'history.json')['validation']
                    if len(history)!=6 or min(history,key=lambda r:r['mse'])['step']!=meta['best_step']:
                        raise ValueError('image selection rule mismatch')
                    if any([r['target'] for r in row['views']]!=list(VAL_VIEWS) for row in history):raise ValueError('selection-view identity')
                    state=torch.load(path/'best.pt',map_location='cpu',weights_only=True)['state_dict']
                    model=GSReadout(role);model.load_state_dict(state)
                    with np.load(path/'score_samples.npz',allow_pickle=False) as z:
                        k=int(z['view']);ids=torch.from_numpy(z['primitive_ids']);p=z['probability'];rgb=z['rgb']
                    keydata=data['views'][k];q=query(data,k,ids,model)
                    ref=numpy_logits(q,keydata,state,role);refp=np.exp(ref-logsumexp(ref,axis=-1,keepdims=True))
                    pe=float(abs(refp-p).max());re=float(abs(refp@arr(keydata['rgb'])-rgb).max())
                    numeric=dict(p_max_error=pe,rgb_max_error=re,winner_changes=int(np.sum(p.argmax(-1)!=refp.argmax(-1))))
                    if pe>5e-5 or re>2e-5:warnings.append(dict(scene=scene,objective=objective,role=role,seed=seed,**numeric))
                    with np.load(path/'colors.npz',allow_pickle=False) as z:
                        if not np.array_equal(z['counts'],data['all_count'].numpy()) or not np.isfinite(z['colors']).all():raise ValueError('fusion outputs')
                    with np.load(path/'predictions.npz',allow_pickle=False) as z:
                        frames=read(ROOT/'results/round9_context_data'/scene/'target_cameras.json')['frames']
                        if z['target_ids'].tolist()!=[f['id'] for f in frames] or z['target_splits'].tolist()!=[f['split'] for f in frames]:raise ValueError('target identity')
                        if z['predictions'].shape!=(len(meta['variants']),8,144,192,3) or not np.isfinite(z['predictions']).all():raise ValueError('incomplete rendered population')
                    records.append(dict(scene=scene,objective=objective,role=role,seed=seed,path=str(path),
                        run_sha256=sha(path/'run.json'),numeric=numeric))
                    print('AUDIT',scene,objective,role,seed,flush=True)
    if len(records)!=60:raise ValueError('full factorial required')
    write(dest/'completion.json',dict(records=records,numeric_warnings=warnings,paired=True,target_rgb_read=False,
        independently_authored=False,separate_numpy_formulas=True,unix=time.time()))


def metrics(out):
    from experiments.round9_analysis import read_target,frame_metrics
    checked(out);audit=read(out/'audit/completion.json');dest=out/'metrics'
    if dest.exists() or len(audit['records'])!=60:raise ValueError('retain metrics; all60 sealed runs required')
    dest.mkdir();truths={};frames={};rows=[];hashes={};max_error=0.
    for scene in SCENES:
        frames[scene]=read(ROOT/'results/round9_context_data'/scene/'target_cameras.json')['frames']
        truths[scene]=[]
        for frame in frames[scene]:
            truth,digest=read_target(frame);truths[scene].append(truth);hashes[frame['image_path']]=digest
    def add(scene,objective,role,seed,variant,preds):
        nonlocal max_error
        if len(preds)!=8:raise ValueError('all target frames required')
        for i,(pred,truth) in enumerate(zip(preds,truths[scene])):
            met=frame_metrics(pred,truth);delta=(pred.astype(np.float64)-truth).ravel();replay=float(delta@delta/delta.size)
            max_error=max(max_error,abs(replay-met['mse']))
            if not np.isclose(replay,met['mse'],rtol=1e-10,atol=1e-12):raise ValueError('MSE replay')
            rows.append(dict(scene=scene,objective=objective,role=role,seed=seed,variant=variant,
                split=frames[scene][i]['split'],id=frames[scene][i]['id'],**met))
    for rec in audit['records']:
        path=Path(rec['path']);meta=read(path/'run.json')
        if sha(path/'run.json')!=rec['run_sha256'] or sha(path/'predictions.npz')!=meta['predictions_sha256']:raise ValueError('changed sealed prediction')
        with np.load(path/'predictions.npz',allow_pickle=False) as z:
            for variant,preds in zip(z['variants'].tolist(),z['predictions']):add(rec['scene'],rec['objective'],rec['role'],rec['seed'],variant,preds)
    for scene in SCENES:
        path=ROOT/'results/round9_gs_readout'/scene
        if sha(path/'predictions.npz')!=read(path/'run.json')['prediction_sha256']:raise ValueError('changed external reference')
        with np.load(path/'predictions.npz',allow_pickle=False) as z:
            if z['target_ids'].tolist()!=[f['id'] for f in frames[scene]]:raise ValueError('external targets')
            for role,preds in zip(z['methods'].tolist(),z['predictions']):add(scene,'external','r9_'+role,0,'primary',preds)
    def summary(obj,role,seed,variant='primary'):
        by_scene={}
        for scene in SCENES:
            chosen=[r for r in rows if (r['objective'],r['role'],r['seed'],r['variant'],r['scene'])==(obj,role,seed,variant,scene)]
            if len(chosen)!=8:raise ValueError('full image population missing')
            by_scene[scene]={s:float(np.mean([r['mse'] for r in chosen if s=='all' or r['split']==s])) for s in ('gate','calibration','all')}
        return dict(by_scene=by_scene,macro={s:float(np.mean([r[s] for r in by_scene.values()])) for s in ('gate','calibration','all')})
    table={obj:{role:{str(seed):summary(obj,role,seed) for seed in SEEDS} for role in ROLES} for obj in OBJECTIVES}
    external={role:summary('external',role,0) for role in sorted({r['role'] for r in rows if r['objective']=='external'})}
    means={obj:{role:float(np.mean([r['macro']['gate'] for r in vals.values()])) for role,vals in roles.items()} for obj,roles in table.items()}
    effects={control:{str(seed):(table['image']['decoupled'][str(seed)]['macro']['gate']-table['image'][control][str(seed)]['macro']['gate'])-(table['center']['decoupled'][str(seed)]['macro']['gate']-table['center'][control][str(seed)]['macro']['gate']) for seed in SEEDS} for control in ('point','full')}
    strongest=min([r for r in external if r.startswith(('r9_point_','r9_fixed_')) or r=='r9_bilinear'],key=lambda r:external[r]['macro']['gate'])
    checks=[]
    for control,required in (('point',.02),('full',.02),(strongest,.02),('r9_original_gs',0.)):
        ext=control in external;den=external[control]['macro']['gate'] if ext else means['image'][control]
        delta=means['image']['decoupled']/den-1
        wins=sum(table['image']['decoupled'][str(seed)]['macro']['gate']<(external[control] if ext else table['image'][control][str(seed)])['macro']['gate'] for seed in SEEDS)
        scenes={scene:float(np.mean([table['image']['decoupled'][str(seed)]['by_scene'][scene]['gate']-(external[control] if ext else table['image'][control][str(seed)])['by_scene'][scene]['gate'] for seed in SEEDS])) for scene in SCENES}
        ok=(delta<=-required if required else delta<0) and wins>=4 and all(v<0 for v in scenes.values())
        checks.append(dict(control=control,relative_mse_change=delta,winning_seeds=wins,scene_differences=scenes,passed=bool(ok)))
    interventions={obj:{variant:{str(seed):summary(obj,'decoupled',seed,variant) for seed in SEEDS} for variant in ('zero_shape','reverse_v')} for obj in OBJECTIVES}
    write(dest/'frame_metrics.json',rows)
    write(dest/'summary.json',dict(by_objective_role_seed=table,gate_mse_means=means,external=external,
        difference_in_gaps=effects,checks=checks,promising_development_only=all(c['passed'] for c in checks),
        interventions=interventions,target_sha256=hashes,goal_complete=False,fresh_confirmation=False))
    write(dest/'verification.json',dict(frame_rows=len(rows),pixels_per_frame=27648,max_mse_replay_error=max_error,
        metric_replay_passed=True,numeric_warnings=audit['numeric_warnings']))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('preflight','smoke','audit','metrics'))
    p.add_argument('--out',type=Path,default=OUT/'attempt1');a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    globals()[a.stage](a.out)


if __name__=='__main__':main()
