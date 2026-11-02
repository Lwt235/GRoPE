"""Bounded independent operator and stored-metric replay; no selection/tuning."""
import argparse
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiments import round53_train as P
from experiments.round53_data import OUT, ROOT, load, write, read, sha
from models.round53_ray_frame import make_model, camera_pose, center_rays
from models.round52_camera_frame import camera_lift
from tests.test_round53_ray_frame import numpy_ray_frame
from experiments.round52_postcheck import full_frame as projection_full_frame
from tests.test_round52_camera_frame import numpy_operator, full_operator
from experiments.round51_postcheck import full_frame as previous_full_frame
from models.round50_local_frame import rotate
from tests.test_round50_local_frame import numpy_frame
from experiments.round50_postcheck import rotmat, quadratic_bias


def plan():
    if (OUT/'postcheck/plan.json').exists():raise FileExistsError('retain plan')
    sources=['experiments/round53_postcheck.py','experiments/round53_postcheck_execute.py',
        'tests/test_round53_ray_frame.py','experiments/round52_postcheck.py',
        'experiments/round51_postcheck.py','tests/test_round52_camera_frame.py',
        'tests/test_round50_local_frame.py','experiments/round51_train.py',
        'experiments/round51_data.py','experiments/round50_postcheck.py',
        'experiments/round50_train.py','experiments/round50_data.py']
    write(OUT/'postcheck/plan.json',dict(sources={p:sha(ROOT/p) for p in sources},
        scope='all30 selected models: first validation pair/first cross-attention direction, whole validation CE CPU replay, all60 saved metric groups',
        warning_probability=5e-5,warning_validation_ce=5e-6,
        action='interpret probability winners, transported feature differences and metric changes; never reject solely for a tiny numerical discrepancy',
        confirmation_access=False))
    for path in sources:
        dest=OUT/'postcheck/source'/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/path).read_bytes())


def full_frame(model,features,K,xy,geometry):
    if model.experiment_role=='camera_only':
        return projection_full_frame(model,features,K,xy,geometry)
    if model.experiment_role=='hrope':
        return previous_full_frame(model,features,K,xy)
    # Independent NumPy camera/ray inversion and full frame construction.
    camera=np.eye(4);camera[:3,:3]=geometry[0,:9].reshape(3,3)
    camera[:3,3]=geometry[0,9:12]
    image=np.column_stack((xy*np.array([8/192,8/144]),np.ones(len(xy))))
    rays=np.linalg.solve(K,image.T).T
    g=numpy_ray_frame(model,features,rays)
    a=np.linalg.inv(camera@g)
    full=np.zeros((len(features),48,48))
    width=24 if model.experiment_role=='ray_gaussian_hybrid' else 48
    for start in range(0,width,4):full[:,start:start+4,start:start+4]=a
    if width==24:full[:,24:,24:]=rotmat(xy,24).transpose(0,2,1)
    return full


def production_matrices(model,features,K,xy,geometry):
    if model.experiment_role=='camera_only':
        return model.matrices(features,K,camera_lift(geometry,K))
    if model.experiment_role=='hrope':return model.matrices(features,K)
    camera=camera_pose(geometry)
    return model.ray_matrices(features,center_rays(xy,K),(camera,torch.linalg.inv(camera)))


@torch.no_grad()
def operator(role,seed,data):
    path=OUT/'runs'/role/str(seed);old=read(path/'run.json')
    if sha(path/'model.pt')!=old['model_sha256']:raise ValueError('checkpoint changed')
    model=make_model(role);model.load_state_dict(torch.load(path/'model.pt',map_location='cpu',weights_only=True));model.eval()
    ce=P.validation(model,data);d=P.subset(data,slice(0,1))
    # Reproduce only the common input arithmetic, including inherited +/- .5.
    q=model.input_projection(torch.cat(((d['q']*math.sqrt(128)+.5)-.5,d['qgeom']),-1))
    k=model.input_projection(torch.cat(((d['k']*math.sqrt(128)+.5)-.5,d['kgeom']),-1))
    layer=model.layers[0];fq,fk=layer.attention_norm(q),layer.attention_norm(k)
    qq,_,_=layer._project(q);_,kk,kv=layer._project(k);qq,kk=F.normalize(qq,dim=-1),F.normalize(kk,dim=-1)
    q0,k0,v0=[t.numpy()[0].astype(np.float64) for t in (qq,kk,kv)]
    qxy,kxy=d['qpos'],d['kpos'];aq=None
    if role=='nope':qxy,kxy=torch.zeros_like(qxy),torch.zeros_like(kxy)
    if role in ('rope','mlp','nope'):
        if role=='mlp':qxy=qxy+4*torch.tanh(model.field(fq)/4);kxy=kxy+4*torch.tanh(model.field(fk)/4)
        qq,kk=rotate(qq,qxy),rotate(kk,kxy)
    elif role!='gaussian_bias':
        aq,iq=production_matrices(model,fq,d['qK'],qxy,d['qgeom']);ak,ik=production_matrices(model,fk,d['kK'],kxy,d['kgeom'])
        qq=model.transport(qq,aq.transpose(-1,-2),qxy,'query');kk=model.transport(kk,ik,kxy,'key');kv=model.transport(kv,ik,kxy,'value')
    temp=float(layer.log_temperature.clamp(max=math.log(60.)).exp());scores=temp*(qq@kk.transpose(-1,-2))
    if role=='gaussian_bias':scores=scores+model.field.bias(fq,qxy,kxy).to(scores.dtype)
    prob=scores.softmax(-1);value=prob@kv
    if aq is not None:value=model.transport(value,aq,qxy,'output')
    prob,value=prob.numpy()[0],value.numpy()[0]
    model.double();xn,yn=qxy.numpy()[0].astype(float),kxy.numpy()[0].astype(float)
    if role in ('rope','mlp','nope'):
        nq=np.einsum('nij,hnj->hni',rotmat(xn,48),q0);nk=np.einsum('nij,hnj->hni',rotmat(yn,48),k0);nv=v0
    elif role=='gaussian_bias':nq,nk,nv=q0,k0,v0
    else:
        a=full_frame(model,fq.numpy()[0].astype(float),d['qK'].numpy()[0],xn,d['qgeom'].numpy()[0])
        b=full_frame(model,fk.numpy()[0].astype(float),d['kK'].numpy()[0],yn,d['kgeom'].numpy()[0]);bi=np.linalg.inv(b)
        nq=np.einsum('nji,hnj->hni',a,q0);nk=np.einsum('nij,hnj->hni',bi,k0);nv=np.einsum('nij,hnj->hni',bi,v0)
    ns=temp*(nq@nk.swapaxes(-1,-2))
    if role=='gaussian_bias':ns+=quadratic_bias(model,fq.numpy()[0].astype(float),xn,yn)
    ns-=ns.max(-1,keepdims=True);reference=np.exp(ns);reference/=reference.sum(-1,keepdims=True)
    out=reference@nv
    if role not in ('rope','mlp','nope','gaussian_bias'):out=np.einsum('nij,hnj->hni',a,out)
    pe=float(np.max(np.abs(reference-prob)));ve=float(np.max(np.abs(out-value)))
    return dict(role=role,seed=seed,probability_max_error=pe,transport_max_error=ve,
        winner_changes=int((reference.argmax(-1)!=prob.argmax(-1)).sum()),validation_ce_error=ce-old['val_ce'],
        warning=bool(pe>5e-5 or abs(ce-old['val_ce'])>5e-6))


def replay_metrics(data,ids):
    grid=np.array([[8*x+3.5,8*y+3.5] for y in range(18) for x in range(24)],dtype=np.float64)
    valid=data['gt_valid'].numpy();gt=data['pixcontinuous'].numpy().astype(float);w=data['key_weights'].numpy().astype(float)
    records=[]
    for role in P.ROLES:
        for seed in P.SEEDS:
            for variant in ('baseline','rgb_permuted'):
                path=OUT/'evaluation'/role/f'{seed}_{variant}.json';old=read(path)
                if sha(path.with_suffix('.npz'))!=old['prediction_sha256']:raise ValueError('saved predictions changed')
                with np.load(path.with_suffix('.npz')) as z:
                    e=np.sqrt(np.sum((grid[z['argmax']]-gt)**2,axis=-1))
                    s=np.sqrt(np.sum((z['soft_xy'].astype(float)-gt)**2,axis=-1))
                    ce=-np.sum(z['target_logp'].astype(float)*w,axis=-1)
                values=dict(ce=ce,pck4=e<=4,pck8=e<=8,auc8=np.maximum(1-e/8,0),
                    epe_argmax=e,epe_soft=s,tail16=e>16,soft_tail16=s>16)
                by_scene=[]
                for scene in sorted({x['scene'] for x in ids}):
                    idx=np.array([i for i,x in enumerate(ids) if x['scene']==scene]);mask=valid[idx]
                    by_scene.append({k:float(v[idx][mask].mean()) for k,v in values.items()})
                metrics={k:float(np.mean([r[k] for r in by_scene])) for k in P.METRICS}
                deltas={k:metrics[k]-old['metrics'][k] for k in P.METRICS}
                records.append(dict(role=role,seed=seed,variant=variant,metric_replay_differences=deltas))
    return records


def run():
    plan_=read(OUT/'postcheck/plan.json')
    for path,digest in plan_['sources'].items():
        if sha(ROOT/path)!=digest:raise ValueError('postcheck source changed')
    P.check();P.setup()
    if len(read(OUT/'selection.json')['records'])!=30:raise ValueError('seal all selections')
    val,_=load(OUT,'val',torch.device('cpu'));records=[]
    for seed in P.SEEDS:
        for role in P.ROLES:
            r=operator(role,seed,val);records.append(r);print('CHECK',role,seed,r,flush=True)
    data,ids=load(OUT,'old_eval',torch.device('cpu'));metrics=replay_metrics(data,ids)
    P.check();write(OUT/'postcheck/result.json',dict(records=records,metrics=metrics,
        warnings=sum(r['warning'] for r in records),confirmation_access=False,
        scope='same-author independent NumPy operator, not a second complete network or lab'))
    print('POSTCHECK COMPLETE',len(records),'models',len(metrics),'metric groups',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','run']);a=p.parse_args()
    plan() if a.action=='plan' else run()
