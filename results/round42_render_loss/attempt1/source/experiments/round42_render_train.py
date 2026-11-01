"""Matched loss-control training with fixed R41 PE models and actual rendering."""
import argparse
import hashlib
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from models.round41_gs_readout import GSReadout
from experiments import round41_gs_data as D
from experiments.round41_gs_train import infer
from experiments.round42_render_core import (fused_colors,accumulate_color_gradient,
    color_gradient,center_loss,image_loss,render_image)

ROOT=D.ROOT;OUT=ROOT/'results/round42_render_loss';PREVIOUS=D.OUT/'attempt1'
ROLES=('point','full','decoupled');OBJECTIVES=('center','image')
SCENES=D.SCENES;SEEDS=D.SEEDS
SOURCES=('ROUND42_RENDER_LOSS_CONTROL.md','experiments/round42_render_core.py',
    'experiments/round42_render_train.py','experiments/round42_render_audit.py',
    'tests/test_round42_render_gradient.py')
read,write,sha=D.read,D.write,D.sha


def freeze(out):
    old=D.checked(PREVIOUS)
    if out.exists():raise ValueError('preserve attempt')
    sources=dict(old['sources']);sources.update({str(ROOT/p):sha(ROOT/p) for p in SOURCES})
    data={str(PREVIOUS/'lock.json'):sha(PREVIOUS/'lock.json')}
    for scene in SCENES:
        rec=read(PREVIOUS/'prepared'/scene/'completion.json')
        if sha(rec['path'])!=rec['sha256']:raise ValueError('changed prepared scene')
        data[rec['path']]=rec['sha256']
        data[str(PREVIOUS/'prepared'/scene/'completion.json')]=sha(PREVIOUS/'prepared'/scene/'completion.json')
    out.mkdir(parents=True)
    for name in sources:
        p=Path(name)
        if p.is_relative_to(ROOT):
            dst=out/'source'/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
    write(out/'lock.json',dict(sources=sources,data=data,previous=str(PREVIOUS),roles=ROLES,
        objectives=OBJECTIVES,seeds=SEEDS,steps=100,unix=time.time(),fresh_data=False))
    (out/'lock.sha256').write_text(sha(out/'lock.json')+'\n')


def checked(out):
    lock=read(out/'lock.json')
    if sha(out/'lock.json')!=(out/'lock.sha256').read_text().strip():raise ValueError('changed lock')
    for name,digest in lock['sources'].items():
        if sha(name)!=digest:raise ValueError('changed source '+name)
    return lock


def load_scene(scene,out,device='cuda:0'):
    lock=checked(out)
    for name,digest in lock['data'].items():
        if '/'+scene+'/' in name and sha(name)!=digest:raise ValueError('changed input')
    data=D.load_scene(scene,PREVIOUS,device=device)
    path=ROOT/'results/round9_context_data'/scene/'context.npz'
    old=read(PREVIOUS/'lock.json')
    if sha(path)!=old['data'][str(path)]:raise ValueError('changed context camera archive')
    with np.load(path,allow_pickle=False) as z:
        data['cameras']=[{k:torch.from_numpy(z[k][i].copy()).to(device=device,dtype=torch.float32)
                         for k in ('K','R','t')} for i in range(24)]
    return data


@torch.no_grad()
def evaluate(model,data):
    model.eval();colors,_=fused_colors(model,data,D.VAL_VIEWS[0])
    rows=[]
    for target in D.VAL_VIEWS:
        pred=render_image(colors,data,target)
        truth=data['views'][target]['rgb'].reshape(144,192,3)
        mse=float((pred.double()-truth.double()).square().mean())
        rows.append(dict(target=target,mse=mse,pixels=27648))
    return dict(mse=float(np.mean([r['mse'] for r in rows])),views=rows)


def fit(scene,role,objective,seed,data,out):
    dest=out/'runs'/objective/scene/role/str(seed)
    if dest.exists():raise ValueError('preserve run')
    dest.mkdir(parents=True);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=GSReadout(role).cuda();initial=model.qk.weight.detach().cpu().clone()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    targets=np.asarray(D.TRAIN_VIEWS)[np.random.default_rng(seed).integers(len(D.TRAIN_VIEWS),size=100)]
    histories=[];losses=[];best=float('inf');started=time.time();torch.cuda.reset_peak_memory_stats()
    for step in range(101):
        if step%20==0:
            result=evaluate(model,data);histories.append(dict(step=step,**result))
            if result['mse']<best:
                best,best_step=result['mse'],step
                state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            print('VAL',scene,objective,role,seed,step,best,flush=True)
        if step==100:break
        model.train();target=int(targets[step]);optimizer.zero_grad(set_to_none=True)
        colors,counts=fused_colors(model,data,target)
        loss,g=color_gradient(colors,data,target,counts,objective)
        if not torch.isfinite(loss) or not torch.isfinite(g).all():raise ValueError('nonfinite image/color derivative')
        with torch.no_grad():
            other=float(image_loss(colors,data,target) if objective=='center' else center_loss(colors,data,target,counts))
        stats=accumulate_color_gradient(model,data,target,g,counts)
        if step==0:first_grad={n:float(p.grad.norm()) if p.grad is not None else None for n,p in model.named_parameters()}
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        losses.append(dict(step=step+1,target=target,loss=float(loss),other_objective_loss=other,
            gradient_norm=float(norm),**stats))
        if torch.cuda.max_memory_allocated()>16*(1<<30):raise MemoryError('16GiB allocation ceiling')
    model.load_state_dict(state);model.eval()
    torch.save(dict(role=role,objective=objective,seed=seed,step=best_step,state_dict=state),dest/'best.pt')
    write(dest/'history.json',dict(validation=histories,training=losses))
    meta=dict(role=role,objective=objective,seed=seed,best_step=best_step,
        selection_objective='four context selection images, full-pixel MSE',context_selection_mse=best,
        parameters=sum(p.numel() for p in model.parameters()),
        batch_sha256=hashlib.sha256(targets.astype('<i8').tobytes()).hexdigest(),target_view_sequence=targets.tolist(),
        initial_qk_sha256=hashlib.sha256(initial.numpy().tobytes()).hexdigest(),first_gradient=first_grad,
        qk_weight_change=float((state['qk.weight']-initial).norm()),
        shape_strength=float(torch.nn.functional.softplus(state['strength_raw'])) if 'strength_raw' in state else None,
        training_seconds=time.time()-started,training_peak_bytes=torch.cuda.max_memory_allocated(),
        checkpoint_sha256=sha(dest/'best.pt'),history_sha256=sha(dest/'history.json'))
    write(dest/'training.json',meta);infer(scene,model,data,dest,meta)


def train(scene,seeds,out):
    checked(out)
    if not read(out/'preflight.json')['passed'] or not read(out/'smoke.json')['passed']:raise ValueError('checks required')
    data=load_scene(scene,out)
    for seed in seeds:
        started=time.time();tag=scene+'_'+str(seed)
        write(out/(tag+'_started.json'),dict(unix=started,scene=scene,seed=seed,gpu=torch.cuda.get_device_name()))
        cells=[(objective,role) for objective in OBJECTIVES for role in ROLES]
        offset=SEEDS.index(seed);cells=cells[offset:]+cells[:offset]
        try:
            for objective,role in cells:
                fit(scene,role,objective,seed,data,out)
                if time.time()-started>3*3600:raise TimeoutError('3GPU-hour seed shard ceiling')
        except BaseException as e:
            write(out/(tag+'_failure.json'),dict(error=repr(e),seconds=time.time()-started));raise
        write(out/(tag+'_complete.json'),dict(completed=True,seconds=time.time()-started,goal_complete=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('freeze','train'))
    p.add_argument('--out',type=Path,default=OUT/'attempt1');p.add_argument('--scene',choices=SCENES)
    p.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS));a=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.stage=='freeze':freeze(a.out);return
    if not torch.cuda.is_available():raise RuntimeError('CUDA training required')
    if a.scene is None or not set(a.seeds)<=set(SEEDS):p.error('scene and frozen seeds required')
    torch.use_deterministic_algorithms(True);train(a.scene,a.seeds,a.out)


if __name__=='__main__':main()
