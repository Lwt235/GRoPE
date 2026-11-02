"""Bounded post-training operator replay and resolution ceiling, no tuning."""
import argparse
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiments import round50_train as P
from experiments.round50_data import OUT, ROOT, centers, sha, write, CONDITIONS
from tests.test_round50_local_frame import numpy_frame
from models.round50_local_frame import LocalFrameMatcher, rotate


def plan():
    path=OUT/'postcheck/plan.json'
    if path.exists():raise FileExistsError('retain plan')
    write(path,dict(source_sha256=sha(__file__),scope='all48 selected models, first val pair / first cross-attention direction',
        check='independent float64 matrices/rotations/quadratic score, probability and value transport; CPU validation replay',
        warning_probability=5e-5,warning_validation_ce=5e-6,
        action='report output/winner impact; do not overwrite predictions or reject on tiny differences',
        extras='model-independent nearest-grid metric ceiling; stratified summaries and plots',
        confirmation_access=False))
    (OUT/'postcheck/source.py').write_bytes(Path(__file__).read_bytes())


def rotmat(xy,width):
    omega=np.logspace(-1,.6,width//4)
    phases=(xy[...,None]*omega).reshape(len(xy),-1)
    a=np.broadcast_to(np.eye(width),(len(xy),width,width)).copy()
    for i in range(width//2):
        c,s=np.cos(phases[:,i]),np.sin(phases[:,i])
        a[:,2*i,2*i]=c;a[:,2*i,2*i+1]=-s;a[:,2*i+1,2*i]=s;a[:,2*i+1,2*i+1]=c
    return a


def full_frame(model,features,K,xy):
    a=numpy_frame(model,features[None],K[None])[0];d=a.shape[-1]
    full=np.broadcast_to(np.eye(48),(len(features),48,48)).copy()
    for j in range(0,24,d):full[:,j:j+d,j:j+d]=a
    full[:,24:,24:]=rotmat(xy,24).transpose(0,2,1)
    return full


def quadratic_bias(model,features,x,y):
    layer=model.field.predictor
    raw=(features@layer.weight.detach().numpy().T+layer.bias.detach().numpy()).reshape(len(x),2,5).transpose(1,0,2)
    mean=4*np.tanh(raw[...,:2]/4);shape=3*np.tanh(raw[...,2:]/3)
    l=np.zeros((2,len(x),2,2));l[...,0,0]=np.exp(shape[...,0]);l[...,1,1]=np.exp(shape[...,1]);l[...,1,0]=shape[...,2]
    precision=l@l.swapaxes(-1,-2)
    delta=y[None,None]-x[None,:,None]-mean[:,:,None]
    return -.5*np.einsum('hnki,hnij,hnkj->hnk',delta,precision/16,delta)


@torch.no_grad()
def check_one(role,seed,regime,data):
    path=OUT/'runs'/regime/role/str(seed)
    model=LocalFrameMatcher(role);model.load_state_dict(torch.load(path/'model.pt',map_location='cpu',weights_only=True));model.eval()
    ce=P.validation(model,data,regime);old=P.read(path/'run.json')
    d=P.subset(data,slice(0,1));suffix='geom' if regime=='oracle' else 'noisy'
    q=model.input_projection(torch.cat((d['qrgb']-.5,d['q'+suffix]),-1));k=model.input_projection(torch.cat((d['krgb']-.5,d['k'+suffix]),-1))
    layer=model.layers[0];fq,fk=layer.attention_norm(q),layer.attention_norm(k)
    qq,_,_=layer._project(q);_,kk,kv=layer._project(k)
    qq,kk=F.normalize(qq,dim=-1),F.normalize(kk,dim=-1)
    q0,k0,v0=[t.numpy()[0].astype(np.float64) for t in (qq,kk,kv)]
    qxy,kxy=d['qpos'],d['kpos'];aq=None
    if role in ('rope','mlp'):
        if role=='mlp':qxy=qxy+4*torch.tanh(model.field(fq)/4);kxy=kxy+4*torch.tanh(model.field(fk)/4)
        qq,kk=rotate(qq,qxy),rotate(kk,kxy)
    elif role!='gaussian_bias':
        aq,iq=model.matrices(fq,d['qK']);ak,ik=model.matrices(fk,d['kK'])
        qq=model.transport(qq,aq.transpose(-1,-2),qxy,'query');kk=model.transport(kk,ik,kxy,'key');kv=model.transport(kv,ik,kxy,'value')
    temperature=float(layer.log_temperature.clamp(max=math.log(60.)).exp())
    score=temperature*(qq@kk.transpose(-1,-2))
    if role=='gaussian_bias':score=score+model.field.bias(fq,qxy,kxy).to(score.dtype)
    actual_p=score.softmax(-1);actual_v=actual_p@kv
    if aq is not None:actual_v=model.transport(actual_v,aq,qxy,'output')
    actual_p=actual_p.numpy()[0];actual_v=actual_v.numpy()[0]
    # NumPy implementation receives identical pre-operator features and parameters.
    model.double();xn,yn=qxy.numpy()[0].astype(float),kxy.numpy()[0].astype(float)
    if role in ('rope','mlp'):
        rq,rk=rotmat(xn,48),rotmat(yn,48)
        nq=np.einsum('nij,hnj->hni',rq,q0);nk=np.einsum('nij,hnj->hni',rk,k0);nv=v0
    elif role=='gaussian_bias':nq,nk,nv=q0,k0,v0
    else:
        a=full_frame(model,fq.numpy()[0].astype(float),d['qK'].numpy()[0],xn)
        b=full_frame(model,fk.numpy()[0].astype(float),d['kK'].numpy()[0],yn)
        bi=np.linalg.inv(b)
        nq=np.einsum('nji,hnj->hni',a,q0);nk=np.einsum('nij,hnj->hni',bi,k0);nv=np.einsum('nij,hnj->hni',bi,v0)
    ns=temperature*(nq@nk.swapaxes(-1,-2))
    if role=='gaussian_bias':ns+=quadratic_bias(model,fq.numpy()[0].astype(float),xn,yn)
    ns-=ns.max(-1,keepdims=True);prob=np.exp(ns);prob/=prob.sum(-1,keepdims=True)
    value=prob@nv
    if role not in ('rope','mlp','gaussian_bias'):value=np.einsum('nij,hnj->hni',a,value)
    p_error=float(np.max(np.abs(actual_p-prob)));v_error=float(np.max(np.abs(actual_v-value)))
    field={}
    if role in ('gs_frame','gs_iso'):
        a,_=model.matrices(fq.double(),d['qK'].double());eig=torch.linalg.eigvalsh(a[...,:3,:3]).numpy()
        ratio=eig.max(-1)/eig.min(-1)
        field=dict(mean_axis_ratio=float(ratio.mean()),max_axis_ratio=float(ratio.max()))
    return dict(role=role,seed=seed,regime=regime,probability_max_error=p_error,transport_max_error=v_error,
        probability_winner_changes=int((actual_p.argmax(-1)!=prob.argmax(-1)).sum()),validation_ce_error=ce-old['val_ce'],
        warning=bool(p_error>5e-5 or abs(ce-old['val_ce'])>5e-6),field=field)


def run():
    plan_=P.read(OUT/'postcheck/plan.json')
    if sha(__file__)!=plan_['source_sha256']:raise ValueError('postcheck source changed')
    P.check_lock(OUT)
    if len(P.read(OUT/'sealed_models.json')['records'])!=48:raise ValueError('all model choices must be sealed')
    P.setup();data=P.load(OUT,'val','base',torch.device('cpu'));records=[]
    for regime in P.REGIMES:
        for seed in P.SEEDS:
            for role in P.ROLES:
                record=check_one(role,seed,regime,data);records.append(record)
            print('CHECK',regime,seed,'models',len(records),flush=True)
    ceiling=[]
    for c in CONDITIONS:
        with np.load(OUT/'data'/f'dev_{c}.npz',allow_pickle=False) as z:
            xy=z['qtarget'];valid=z['qvalid'];count=valid.sum(-1)
            nearest=np.clip(np.round((xy-3.5)/8),0,7)*8+3.5
            error=np.linalg.norm(xy-nearest,axis=-1)
            ceiling.append(dict(condition=c,auc=float(((np.clip(1-error/8,0,1)*valid).sum(-1)/count).mean()),
                epe=float(((error*valid).sum(-1)/count).mean()),pck4=float((((error<=4)*valid).sum(-1)/count).mean()),
                eligible=int(valid.sum()),total=int(valid.size),shape_stratum_counts=[int((valid&(z['qanisotropy']>=lo)&(z['qanisotropy']<hi)).sum()) for lo,hi in ((0,1.3),(1.3,1.8),(1.8,np.inf))]))
    P.check_lock(OUT)
    write(OUT/'postcheck/result.json',dict(records=records,grid_resolution_ceiling=ceiling,
        warnings=sum(r['warning'] for r in records),source_hashes_unchanged=True,
        scope='independent operator formula by the same author; not a second complete network or laboratory'))
    print('POSTCHECK COMPLETE',len(records),'warnings',sum(r['warning'] for r in records),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','run']);a=p.parse_args()
    plan() if a.action=='plan' else run()
