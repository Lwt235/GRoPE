"""Independent full-grid geometry/relation arithmetic and R64 selection/metrics."""
import argparse
import hashlib
import time
import numpy as np
from scipy.special import logsumexp,erf
import torch
from experiments.round55_train import tensor_hash,common,subset
from experiments.round55_optimization import sampling
from experiments.round64_common import (OUT,CONFIG,ROLES,NEW_ROLES,REFERENCE_ROLES,SEEDS,
    METRICS,checked,load,read,write,sha,run_path,prediction_path,pixels)
from models.round64_depth_field import make_model,select_inputs
from tests.test_round62_rgb_relations import numpy_relations


def numpy_fields(model,features,data,prefix):
    """Depth differentiation independently ordered by positions; NumPy gradient."""
    position=data[prefix+'pos'].numpy()[0].astype(float)
    intrinsics=data[prefix+'K'].numpy()[0].astype(float)
    head=model.field['predictor']
    raw=features@head.weight.detach().numpy().astype(float).T+head.bias.detach().numpy().astype(float)
    z=1/30+np.logaddexp(0,raw[:,0]+np.log(np.expm1(29/30)))
    order=np.lexsort((position[:,0],position[:,1]));inverse=np.argsort(order)
    grid=z[order].reshape(18,24)
    dv,du=np.gradient(grid,8.,8.,edge_order=1)
    du,dv=du.ravel()[inverse],dv.ravel()[inverse]
    ray=np.column_stack((position*np.array([8/192,8/144]),np.ones(len(position))))@np.linalg.inv(intrinsics).T
    fx,fy=intrinsics[0,0]*192,intrinsics[1,1]*144
    return z,ray,du,dv,fx,fy


def numpy_geometry(fields,data,prefix,surface):
    z,ray,du,dv,fx,fy=fields
    mean=ray*z[:,None]
    ju=np.zeros_like(ray);jv=np.zeros_like(ray)
    ju[:,0]=z/fx;jv[:,1]=z/fy
    if surface:
        ju+=ray*du[:,None];jv+=ray*dv[:,None]
    covariance=5.25*(np.einsum('ni,nj->nij',ju,ju)+np.einsum('ni,nj->nij',jv,jv))+(1/300)**2*np.eye(3)
    geometry=data[prefix+'geom'].numpy()[0,0].astype(float)
    r,t=geometry[:9].reshape(3,3),geometry[9:12]
    return mean@r.T+t,r[None]@covariance@r.T[None]


@torch.no_grad()
def operator_check(model,data):
    sample=select_inputs(subset(data,slice(0,1)),model.variant)
    q=model.input_projection(torch.cat(((sample['q']*np.sqrt(128)+.5)-.5,sample['qgeom']),-1))
    k=model.input_projection(torch.cat(((sample['k']*np.sqrt(128)+.5)-.5,sample['kgeom']),-1))
    fq,fk=model.layers[0].attention_norm(q),model.layers[0].attention_norm(k)
    qfields=numpy_fields(model,fq.numpy()[0].astype(float),sample,'q')
    kfields=numpy_fields(model,fk.numpy()[0].astype(float),sample,'k')
    qi=np.array([0,1,23,24,199,287,408,431]);ki=np.array([431,409,300,230,120,45,22,0])
    geometry_errors={}
    if model.variant=='depth_local_mlp':
        endpoints=[]
        for fields,index in ((qfields,qi),(kfields,ki)):
            z,ray,du,dv,fx,fy=fields
            endpoints.append(np.column_stack((z,fx*du,fy*dv,ray[:,:2]))[index])
        a,b=endpoints
        inputs=np.concatenate((np.broadcast_to(a[:,None],(8,8,5)),np.broadcast_to(b[None],(8,8,5))),-1)
    else:
        geoms=[]
        for fields,features,prefix in ((qfields,fq,'q'),(kfields,fk,'k')):
            expected=numpy_geometry(fields,sample,prefix,model.variant=='depth_surface')
            actual=model.geometry(features,sample,prefix)
            for name,a,b in zip(('mean','covariance'),actual,expected):
                error=float(np.abs(a.numpy()[0]-b).max());scale=float(np.abs(b).max())
                geometry_errors[prefix+name]=error
                if error>1e-5*(1+scale):raise ValueError('substantive geometry discrepancy')
            geoms.append(expected)
        (qm,qs),(km,ks)=geoms
        inputs=numpy_relations(qm[qi],qs[qi],km[ki],ks[ki])
    first,last=model.field['relation'][0],model.field['relation'][2]
    z=inputs@first.weight.detach().numpy().astype(float).T+first.bias.detach().numpy().astype(float)
    expected=(z*.5*(1+erf(z/np.sqrt(2))))@last.weight.detach().numpy().astype(float).T
    actual=model.relative(fq,fk,sample)[0].numpy()[0][qi][:,ki]
    error=float(np.abs(actual-expected).max());scale=float(np.abs(expected).max())
    if error>1e-4*(1+scale):raise ValueError('substantive independent relation discrepancy')
    return dict(relation_max_error=error,reference_max_abs=scale,geometry_errors=geometry_errors,
        scope='first layer,full-grid NumPy depth/derivatives/geometry;8x8 relation MLP;not whole-network rewrite')


def selections():
    lock=checked()
    if (OUT/'selected.json').exists():raise ValueError('preserve selections')
    data,_=load('val','cpu');rows=[]
    valid=data['gt_valid'].numpy();index=data['key_indices'].numpy().clip(0);weights=data['key_weights'].numpy()
    for role in NEW_ROLES:
        for seed in SEEDS:
            path=run_path(role,seed);run=read(path/'run.json');history=read(path/'history.json')
            ref=next(r for r in lock['references'] if r['seed']==seed)
            if (run['role'],run['seed'],run['steps'])!=(role,seed,5000):raise ValueError('fit scope')
            if run['shared_init_sha256']!=ref['shared_init_sha256'] or run['shared_final_sha256']==run['shared_init_sha256']:
                raise ValueError('shared state identity/learning')
            if run['sampling_sha256']!=ref['sampling_sha256']:raise ValueError('actual data stream')
            if run['lock_sha256']!=sha(OUT/'lock.json') or run['data_manifest_sha256']!=sha(OUT/'data/manifest.json'):
                raise ValueError('source/data identity')
            for name,key in [('model.pt','model_sha256'),('validation.npz','validation_sha256'),('history.json','history_sha256')]:
                if sha(path/name)!=run[key]:raise ValueError('fit artifact identity')
            for split,h in run['observed_data_sha256'].items():
                if sha(OUT/'data'/('observed_'+split+'.npz'))!=h:raise ValueError('fit observed identity')
            if [r['step'] for r in history]!=list(range(0,5001,100)):
                raise ValueError('validation/budget coverage')
            if not all(np.isfinite(v) and v>0 for v in run['field_changes'].values()):
                raise ValueError('positional field not updated')
            best=min(history,key=lambda r:r['val_ce'])
            if (best['step'],best['val_ce'])!=(run['best_step'],run['val_ce']):raise ValueError('selection rule')
            with np.load(path/'validation.npz',allow_pickle=False) as z:logits=z['logits'].astype(np.float64)
            logp=logits-logsumexp(logits,axis=-1,keepdims=True)
            ce=-(np.take_along_axis(logp,index,-1)*weights).sum(-1)[valid].mean()
            ce_error=abs(float(ce)-run['val_ce'])
            if ce_error>1e-5:raise ValueError('validation CE discrepancy')
            model=make_model(role).eval();model.load_state_dict(torch.load(path/'model.pt',map_location='cpu',weights_only=True))
            with torch.no_grad():
                replay=model(select_inputs(subset(data,slice(0,1)),role)).numpy()
                replay_error=float(np.abs(replay-logits[:1]).max())
                op=operator_check(model,data)
            if replay_error>1e-3:raise ValueError('substantive CPU replay difference')
            rows.append(dict(role=role,seed=seed,run_sha256=sha(path/'run.json'),checkpoint_sha256=sha(path/'model.pt'),
                validation_ce_error=ce_error,replay_logit_error=replay_error,operator=op))
            print('CHECKED',role,seed,ce_error,replay_error,flush=True)
    write(OUT/'selected.json',dict(runs=rows,reused_models=24,sealed_unix=time.time(),
        lock_sha256=sha(OUT/'lock.json'),fresh_confirmation=False,goal_complete=False))


def independent_metrics(pred,data,ids):
    grid=pixels().astype(np.float64);rows=[]
    for i,identity in enumerate(ids):
        mask=data['gt_valid'][i].numpy();gt=data['pixcontinuous'][i].numpy()[mask].astype(float)
        chosen=grid[pred['argmax'][i][mask]]
        hard=np.sqrt(np.sum((chosen-gt)**2,axis=1))
        soft=np.sqrt(np.sum((pred['soft_xy'][i][mask].astype(float)-gt)**2,axis=1))
        ce=-np.sum(pred['target_logp'][i][mask].astype(float)*data['key_weights'][i].numpy()[mask],axis=1)
        values=dict(ce=ce,pck4=(hard<=4),pck8=(hard<=8),auc8=np.maximum(1-hard/8,0),
                    epe_argmax=hard,epe_soft=soft,tail16=(hard>16),soft_tail16=(soft>16))
        rows.append(dict(identity,n_valid=int(mask.sum()),frame_gap=identity['k_index']-identity['q_index'],
                         **{k:float(v.sum()) for k,v in values.items()}))
    scenes={}
    for scene in sorted({r['scene'] for r in rows}):
        selected=[r for r in rows if r['scene']==scene];n=sum(x['n_valid'] for x in selected)
        if not n:raise ValueError('no valid query in old scene')
        scenes[scene]={k:sum(x[k] for x in selected)/n for k in METRICS}
    return rows,scenes,{k:float(np.mean([s[k] for s in scenes.values()])) for k in METRICS}


def metrics():
    lock=checked();selection=read(OUT/'selected.json');data,ids=load('old_eval','cpu')
    if len(selection['runs'])!=9:raise ValueError('selection scope')
    records=[];max_error=0.
    for role in ROLES:
        for seed in SEEDS:
            for variant in ('baseline','rgb_permuted'):
                path=prediction_path(role,seed,variant);meta=read(path.with_suffix('.json'))
                if role in REFERENCE_ROLES:
                    old=next(r for r in lock['reference_predictions'] if (r['role'],r['seed'],r['variant'])==(role,seed,variant))
                    if sha(path)!=old['sha256'] or sha(path.with_suffix('.json'))!=old['metadata_sha256']:raise ValueError('reference prediction changed')
                else:
                    chosen=next(r for r in selection['runs'] if (r['role'],r['seed'])==(role,seed))
                    if meta['checkpoint_sha256']!=chosen['checkpoint_sha256'] or meta['selection_sha256']!=sha(OUT/'selected.json'):
                        raise ValueError('selected prediction identity')
                    if meta['exported_unix']<selection['sealed_unix'] or sha(path)!=meta['prediction_sha256']:
                        raise ValueError('prediction time/identity')
                with np.load(path,allow_pickle=False) as z:pred={k:z[k] for k in z.files}
                rows,scenes,values=independent_metrics(pred,data,ids)
                if len(rows)!=len(meta['rows']):raise ValueError('prediction population')
                for a,b in zip(rows,meta['rows']):
                    for k in ('scene','pair_id','q_index','k_index','n_valid'):
                        if a[k]!=b[k]:raise ValueError('prediction identity/population')
                error=max(abs(values[k]-meta['metrics'][k]) for k in METRICS)
                max_error=max(max_error,error)
                if error>1e-5:raise ValueError('metric discrepancy')
                records.append(dict(role=role,seed=seed,variant=variant,metrics=values,by_scene=scenes,
                    reused=role in REFERENCE_ROLES,privileged_diagnostic=role.startswith('observed_'),
                    rows=rows,checkpoint_sha256=meta['checkpoint_sha256']))
                print('METRICS',role,seed,variant,error,flush=True)
    if len(records)!=66:raise ValueError('metric scope')
    write(OUT/'metrics/summary.json',dict(records=records,groups=66,new_groups=18,reused_groups=48,
        max_metric_error=max_error,all_groups_independently_recomputed=True,
        fresh_confirmation=False,goal_complete=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('selections','metrics'))
    torch.set_num_threads(4)
    args=parser.parse_args()
    if args.stage=='selections':selections()
    else:metrics()
