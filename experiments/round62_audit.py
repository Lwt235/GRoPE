"""Separate arithmetic, selection identities, and new prediction metrics."""
import argparse
import hashlib
import time
import numpy as np
from scipy.special import logsumexp,erf
import torch
from experiments.round55_train import tensor_hash,common,subset
from experiments.round55_optimization import sampling
from experiments.round62_common import (OUT,CONFIG,ROLES,NEW_ROLES,REFERENCE_ROLES,SEEDS,
    METRICS,checked,load,read,write,sha,run_path,prediction_path,pixels)
from models.round62_rgb_relations import make_model,select_inputs,relation_inputs
from tests.test_round62_rgb_relations import numpy_relations


def numpy_geometry(model,features,data,prefix):
    """Independent first-layer prediction/rays/pose construction, NumPy only."""
    position=data[prefix+'pos'].numpy()[0]
    intrinsics=data[prefix+'K'].numpy()[0].astype(float)
    if model.variant.startswith('observed_'):
        mean=data[prefix+'observed_mean'].numpy()[0].astype(float)
        covariance=data[prefix+'observed_cov'].numpy()[0].astype(float)
    else:
        head=model.field['predictor']
        raw=features@head.weight.detach().numpy().T+head.bias.detach().numpy()
        z=1/30+np.logaddexp(0,raw[:,0]+np.log(np.expm1(29/30)))
        ray=np.column_stack((position*np.array([8/192,8/144]),np.ones(len(position))))@np.linalg.inv(intrinsics).T
        mean=ray*z[:,None]
        covariance=[]
        if model.variant=='predicted_full':
            for x in raw:
                l=np.diag(np.array([.025,.030,.010])*np.exp(2*np.tanh(x[1:4]/2)))
                l[np.tril_indices(3,-1)]=.03*np.tanh(x[4:7])
                covariance.append(l@l.T+(1/300)**2*np.eye(3))
            covariance=np.asarray(covariance)
    if model.variant.endswith('_point'):
        covariance=np.broadcast_to((.025**2+(1/300)**2)*np.eye(3),(len(mean),3,3)).copy()
    geometry=data[prefix+'geom'].numpy()[0,0].astype(float)
    r,t=geometry[:9].reshape(3,3),geometry[9:12]
    return mean@r.T+t,r[None]@covariance@r.T[None]


@torch.no_grad()
def operator_check(model,data):
    # Keep the actual feature stem; independently compute its new PE map.
    sample={k:(v[:1,:8] if k not in ('qK','kK') else v[:1]) for k,v in select_inputs(data,model.variant).items()}
    q=model.input_projection(torch.cat(((sample['q']*np.sqrt(128)+.5)-.5,sample['qgeom']),-1))
    k=model.input_projection(torch.cat(((sample['k']*np.sqrt(128)+.5)-.5,sample['kgeom']),-1))
    fq,fk=model.layers[0].attention_norm(q),model.layers[0].attention_norm(k)
    aq,ak=fq.numpy()[0].astype(float),fk.numpy()[0].astype(float)
    if model.variant=='camera_mlp':
        first,last=model.field[0],model.field[2]
        z=aq@first.weight.detach().numpy().T+first.bias.detach().numpy()
        raw=(z*.5*(1+erf(z/np.sqrt(2))))@last.weight.detach().numpy().T+last.bias.detach().numpy()
        expected=4*np.tanh(raw/4)
        actual=(4*torch.tanh(model.field(fq)/4)).numpy()[0]
        error=float(np.abs(expected-actual).max())
        if error>1e-5:raise ValueError('offset MLP discrepancy')
        return dict(offset_max_error=error,scope='independent NumPy first-layer2D offset')
    if model.variant=='free_relation':
        h=model.field['predictor'];w=h.weight.detach().numpy().astype(float);b=h.bias.detach().numpy().astype(float)
        a=np.column_stack((aq@w.T+b,sample['qpos'].numpy()[0]/12))
        b=np.column_stack((ak@w.T+b,sample['kpos'].numpy()[0]/12))
        inputs=np.concatenate((np.broadcast_to(a[:,None],(len(a),len(b),9)),np.broadcast_to(b[None],(len(a),len(b),9))),-1)
    else:
        qm,qs=numpy_geometry(model,aq,sample,'q');km,ks=numpy_geometry(model,ak,sample,'k')
        inputs=numpy_relations(qm,qs,km,ks)
    first,last=model.field['relation'][0],model.field['relation'][2]
    z=inputs@first.weight.detach().numpy().T+first.bias.detach().numpy()
    expected=(z*.5*(1+erf(z/np.sqrt(2))))@last.weight.detach().numpy().T
    actual=model.relative(fq,fk,sample)[0].numpy()[0]
    error=float(np.abs(actual-expected).max())
    scale=float(np.abs(expected).max())
    # Actual network is float32; NumPy geometry and MLP reference use float64.
    if error>1e-4*(1+scale):raise ValueError('substantive independent relation discrepancy')
    return dict(relation_max_error=error,reference_max_abs=scale,
                scope='first layer,8x8tokens;NumPy prediction/geometry/MLP;not whole-network rewrite')


def selections():
    lock=checked()
    if (OUT/'selected.json').exists():raise ValueError('preserve selections')
    data,_=load('val','cpu',observed=True);rows=[]
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
    write(OUT/'selected.json',dict(runs=rows,reused_models=12,sealed_unix=time.time(),
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
    if len(selection['runs'])!=18:raise ValueError('selection scope')
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
    if len(records)!=60:raise ValueError('metric scope')
    write(OUT/'metrics/summary.json',dict(records=records,groups=60,new_groups=36,reused_groups=24,
        max_metric_error=max_error,all_groups_independently_recomputed=True,
        fresh_confirmation=False,goal_complete=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('selections','metrics'))
    torch.set_num_threads(4)
    args=parser.parse_args()
    if args.stage=='selections':selections()
    else:metrics()
