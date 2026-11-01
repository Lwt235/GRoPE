"""Independent NumPy depth metrics and real validation model-path checks.

Default scope is OLD development validation only. No fresh test discovery,
glob, label loading, or test replay is implemented in this auditor.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
VALIDATION_ROOTS=(ROOT/'results/round7_depth_old/val',ROOT/'results/round7_depth_bm4/val')
OLD_SCENES=('harvard_c8/hv_c8_3','brown_bm_4/brown_bm_4')
SEEDS=(11,22,33,44,55)
ROLES=('full','fixed','isotropic','point','point_pck','singleview')
SUM_FIELDS=('count','boundary_count','absrel_sum','abs_sum','sq_sum','log_sq_sum',
    'delta1_count','boundary_absrel_sum','boundary_abs_sum')
SOURCES=('experiments/round7_depth_audit.py','tests/test_round7_depth_audit.py',
    'models/round7_depth_attention.py','models/gaussian_quadratic_pe.py',
    'experiments/round7_depth_data.py','experiments/round7_depth_train.py',
    'experiments/round6_region_data.py','experiments/round5_rgbd_data.py')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numpy_rows(prediction,target,valid,boundary,ids):
    """Natural-log errors, strict delta1<1.25, query-only validity population."""
    prediction=np.asarray(prediction,dtype=np.float64)
    target=np.asarray(target,dtype=np.float64);valid=np.asarray(valid,dtype=bool)
    boundary=np.asarray(boundary,dtype=bool)&valid
    if prediction.shape!=target.shape or valid.shape!=target.shape or len(ids)!=len(prediction):
        raise ValueError('prediction/label/identity shape mismatch')
    if not np.isfinite(prediction).all() or np.any(prediction<=0):raise ValueError('invalid predicted depth')
    if np.any(valid&(~np.isfinite(target)|(target<=0))):raise ValueError('invalid positive query target')
    rows=[]
    for index,identity in enumerate(ids):
        # Selecting valid values BEFORE arithmetic prevents invalid NaN/zero contamination.
        p,t=prediction[index,valid[index]],target[index,valid[index]]
        delta=np.abs(p-t);rel=delta/t;log=np.log(p)-np.log(t)
        pb,tb=prediction[index,boundary[index]],target[index,boundary[index]]
        row=dict(identity,count=float(len(p)),boundary_count=float(len(pb)),
            absrel_sum=float(rel.sum()),abs_sum=float(delta.sum()),sq_sum=float((delta**2).sum()),
            log_sq_sum=float((log**2).sum()),delta1_count=float((np.maximum(p/t,t/p)<1.25).sum()),
            boundary_absrel_sum=float((np.abs(pb-tb)/tb).sum()),boundary_abs_sum=float(np.abs(pb-tb).sum()))
        rows.append(row)
    return rows


def numpy_macro(rows,positions_per_pair=432):
    grouped=defaultdict(list)
    for row in rows:grouped[row['scene']].append(row)
    scene_metrics={}
    for scene,group in sorted(grouped.items()):
        sums={name:sum(row[name] for row in group) for name in SUM_FIELDS}
        n,b=sums['count'],sums['boundary_count']
        if n<=0:raise ValueError('scene without query labels')
        scene_metrics[scene]=dict(AbsRel=sums['absrel_sum']/n,MAE=sums['abs_sum']/n,
            RMSE=float(np.sqrt(sums['sq_sum']/n)),log_RMSE=float(np.sqrt(sums['log_sq_sum']/n)),
            delta1=sums['delta1_count']/n,boundary_AbsRel=sums['boundary_absrel_sum']/b if b else None,
            boundary_MAE=sums['boundary_abs_sum']/b if b else None,**sums)
    names=('AbsRel','MAE','RMSE','log_RMSE','delta1','boundary_AbsRel','boundary_MAE')
    return dict(metrics={name:float(np.mean([s[name] for s in scene_metrics.values() if s[name] is not None])) for name in names},
        by_scene=scene_metrics,total_count=sum(r['count'] for r in rows),total_positions=len(rows)*positions_per_pair)


def boundary_from_labels(target,valid):
    """Independent four-neighbor sensor-boundary reconstruction."""
    z=np.asarray(target,dtype=np.float64).reshape(-1,18,24);v=np.asarray(valid).reshape(z.shape)
    boundary=np.zeros_like(v)
    for row in range(18):
        for col in range(24):
            for rr,cc in ((row+1,col),(row,col+1)):
                if rr>=18 or cc>=24:continue
                edge=v[:,row,col]&v[:,rr,cc]&(
                    np.abs(z[:,row,col]-z[:,rr,cc])>.1+.1*np.minimum(z[:,row,col],z[:,rr,cc]))
                boundary[:,row,col]|=edge;boundary[:,rr,cc]|=edge
    return boundary.reshape(np.asarray(valid).shape)


def load_old_validation(sources,forward=False):
    from models.round7_depth_attention import FIELDS
    arrays=defaultdict(list);ids=[];manifests=[]
    for source in sources:
        directory=Path(source['path']).resolve()
        if not any(directory.parent==root.resolve() for root in VALIDATION_ROOTS):raise ValueError('only exact old validation directories permitted')
        manifest_path=directory/'manifest.json';manifest=json.loads(manifest_path.read_text())
        if manifest['split']!='val' or manifest['scene'] not in OLD_SCENES:raise ValueError('not an allowed old validation scene')
        for name,expected in [('manifest.json',source['manifest_sha256']),('inputs.npz',source['input_sha256']),('labels.npz',source['label_sha256'])]:
            if sha(directory/name)!=expected:raise ValueError('changed old validation source')
        with np.load(directory/'labels.npz',allow_pickle=False) as values:
            for name in ('target','valid','boundary'):arrays[name].append(values[name].copy())
        if forward:
            with np.load(directory/'inputs.npz',allow_pickle=False) as values:
                if set(values.files)!=set(FIELDS):raise ValueError('unexpected input field, possible label leakage')
                for name in FIELDS:arrays[name].append(values[name].copy())
        ids.extend(dict(scene=manifest['scene'],q_index=q,k_index=k,pair_id=f'{q}:{k}') for q,k in manifest['pairs'])
        manifests.append(manifest)
    if set(row['scene'] for row in ids)!=set(OLD_SCENES) or len(ids)!=128:raise ValueError('requires both complete64-pair old validation groups')
    if len({(r['scene'],r['pair_id']) for r in ids})!=128:raise ValueError('duplicate validation identity')
    return {name:np.concatenate(value) for name,value in arrays.items()},ids,manifests


def compare_tree(a,b):
    """Return all numeric mismatches and maximum numeric absolute error."""
    errors=[];maximum=0.
    def visit(x,y,path):
        nonlocal maximum
        if isinstance(x,dict) and isinstance(y,dict):
            if set(x)!=set(y):errors.append(path+':keys');return
            for key in x:visit(x[key],y[key],path+'/'+str(key))
        elif isinstance(x,list) and isinstance(y,list):
            if len(x)!=len(y):errors.append(path+':length');return
            for i,(xx,yy) in enumerate(zip(x,y)):visit(xx,yy,path+'/'+str(i))
        elif isinstance(x,(int,float)) and isinstance(y,(int,float)):
            error=abs(x-y);maximum=max(maximum,error)
            if not np.isfinite(error) or error>1e-8:errors.append(path+':numeric')
        elif x!=y:errors.append(path+':value')
    visit(a,b,'')
    return dict(max_absolute_error=maximum,mismatches=errors)


def numpy_attention(inputs,state,method,bandwidth,temperature):
    """Independent Gaussian energy in float64; no five-channel implementation import."""
    q,k=inputs['q'].astype(np.float64),inputs['k'].astype(np.float64)
    weight=state['qk.weight'].detach().cpu().numpy().astype(np.float64)
    q,k=q@weight.T,k@weight.T
    q/=np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1e-12)
    k/=np.maximum(np.linalg.norm(k,axis=-1,keepdims=True),1e-12)
    qc=inputs['q_to_kcov'].astype(np.float64).copy();kc=inputs['kcov'].astype(np.float64).copy()
    if method=='point':qc.fill(0);kc.fill(0)
    elif method=='fixed':qc=inputs['qcov'].astype(np.float64)
    elif method=='isotropic':
        qc=np.trace(qc,axis1=-2,axis2=-1)[...,None,None]*.5*np.eye(2)
        kc=np.trace(kc,axis1=-2,axis2=-1)[...,None,None]*.5*np.eye(2)
    elif method!='full':raise ValueError(method)
    if not np.array_equal(kc,np.broadcast_to(kc[:,:1],kc.shape)):raise ValueError('nonuniform native key covariance')
    total=qc+kc[:,:1]+bandwidth**2*np.eye(2)
    delta=inputs['q_to_k'].astype(np.float64)[:,:,None,:]-inputs['kmean'].astype(np.float64)[:,None,:,:]
    energy=-.5*np.einsum('bqki,bqij,bqkj->bqk',delta,np.linalg.inv(total),delta)
    logits=temperature*(q@k.transpose(0,2,1))+energy
    exp=np.exp(logits-logits.max(-1,keepdims=True))
    return exp/exp.sum(-1,keepdims=True)


def real_model_path(run,arrays,ids):
    from models.round7_depth_attention import DepthAttention,FIELDS,depth_loss
    checkpoint=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
    model=DepthAttention(checkpoint['role']);model.load_state_dict(checkpoint['state_dict']);model.eval()
    indices=np.array([0,32,64,96])
    d={name:torch.from_numpy(arrays[name][indices].copy()) for name in arrays}
    with torch.no_grad():
        base,attention,values=model(d,return_attention=True)
        injected=dict(d,target=torch.full_like(d['target'],float('nan')),
            valid=~d['valid'],boundary=~d['boundary'],gt_valid=torch.zeros_like(d['valid']),
            visible_fraction=torch.full_like(d['target'],float('nan')),sensor_depth=torch.full_like(d['target'],float('nan')))
        changed=model(injected)
        zero=model(d,intervention='zero_v');reverse=model(d,intervention='reverse_v')
    injected_exact=torch.equal(base,changed)
    # The source-value/key-depth paths receive gradients through the actual attention operation.
    model.zero_grad(set_to_none=True)
    grad_data={name:value.clone() for name,value in d.items()}
    grad_data['k_value_depth'].requires_grad_(True)
    pred,attention,values=model(grad_data,return_attention=True);attention.retain_grad();values.retain_grad()
    loss=depth_loss(pred,grad_data['target'],grad_data['valid']);loss.backward()
    gradients={name:dict(finite=bool(torch.isfinite(param.grad).all()) if param.grad is not None else False,
        norm=float(param.grad.double().norm()) if param.grad is not None else None) for name,param in model.named_parameters()}
    intermediates={name:dict(finite=bool(torch.isfinite(value.grad).all()) if value.grad is not None else False,
        norm=float(value.grad.double().norm()) if value.grad is not None else None) for name,value in
        [('attention',attention),('values',values),('k_value_depth',grad_data['k_value_depth'])]}
    model.zero_grad(set_to_none=True)
    missing_target=d['target'].clone();missing_valid=d['valid'].clone();missing_valid[:,::2]=False;missing_target[~missing_valid]=float('nan')
    missing_pred=model(d);missing_loss=depth_loss(missing_pred,missing_target,missing_valid);missing_loss.backward()
    missing_finite=bool(torch.isfinite(missing_loss)) and all(x.grad is not None and torch.isfinite(x.grad).all() for x in model.parameters())
    reference_inputs={name:arrays[name][indices] for name in FIELDS}
    if checkpoint['role']=='singleview':
        reference_inputs=dict(reference_inputs,k=reference_inputs['q'],kmean=reference_inputs['qmean'],
            kcov=reference_inputs['qcov'],q_to_k=reference_inputs['qmean'],q_to_kcov=reference_inputs['qcov'])
    reconstructed=numpy_attention(reference_inputs,checkpoint['state_dict'],model.method,model.bandwidth,model.temperature)
    attention_error=float(np.max(np.abs(reconstructed-attention.detach().numpy())))
    all_parameter_gradients=all(record['finite'] and record['norm']>0 for record in gradients.values())
    # singleview deliberately ignores external key-depth; its self V uses q_value_depth.
    required=('attention','values') if checkpoint['role']=='singleview' else ('attention','values','k_value_depth')
    all_intermediate_gradients=all(intermediates[name]['finite'] and intermediates[name]['norm']>0 for name in required)
    output=dict(role=checkpoint['role'],seed=checkpoint['seed'],indices=indices.tolist(),identities=[ids[i] for i in indices],
        label_and_mask_injection_bitwise_equal=injected_exact,loss=float(loss.detach()),
        gradients=gradients,intermediate_gradients=intermediates,all_parameter_gradients_finite_nonzero=all_parameter_gradients,
        required_intermediate_gradients_finite_nonzero=all_intermediate_gradients,missing_gt_loss_finite=missing_finite,
        missing_gt_loss=float(missing_loss.detach()),independent_attention_max_abs_error=attention_error,
        zero_v_mean_absolute_depth_change=float((zero-base).abs().mean()),
        reverse_v_mean_absolute_depth_change=float((reverse-base).abs().mean()),
        base=numpy_macro(numpy_rows(base.numpy(),d['target'].numpy(),d['valid'].numpy(),d['boundary'].numpy(),[ids[i] for i in indices])),
        zero_v=numpy_macro(numpy_rows(zero.numpy(),d['target'].numpy(),d['valid'].numpy(),d['boundary'].numpy(),[ids[i] for i in indices])),
        reverse_v=numpy_macro(numpy_rows(reverse.numpy(),d['target'].numpy(),d['valid'].numpy(),d['boundary'].numpy(),[ids[i] for i in indices])))
    output['pass']=all((injected_exact,all_parameter_gradients,all_intermediate_gradients,missing_finite,
        attention_error<2e-4,output['zero_v_mean_absolute_depth_change']>0,output['reverse_v_mean_absolute_depth_change']>0))
    return output


def replay_checkpoint(run,arrays,ids):
    from models.round7_depth_attention import DepthAttention
    checkpoint=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
    model=DepthAttention(checkpoint['role']);model.load_state_dict(checkpoint['state_dict']);model.eval()
    predictions=[]
    with torch.no_grad():
        for start in range(0,len(ids),16):
            d={name:torch.from_numpy(value[start:start+16]) for name,value in arrays.items()}
            predictions.append(model(d).numpy())
    actual=np.concatenate(predictions)
    with np.load(run/'val_predictions.npz',allow_pickle=False) as values:saved=values['depth']
    error=float(np.max(np.abs(actual-saved)))
    return dict(role=checkpoint['role'],seed=checkpoint['seed'],max_absolute_depth_error_m=error,
        tolerance_m=2e-4,pass_=error<2e-4,summary=numpy_macro(numpy_rows(actual,arrays['target'],arrays['valid'],arrays['boundary'],ids)))


def audit(development,out):
    started=time.time()
    if out.exists():raise ValueError('refusing overwrite audit output')
    if development.resolve()!=ROOT/'results/round7_depth_development':raise ValueError('only named old-development run directory permitted')
    out.mkdir(parents=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    sourcehash={name:sha(ROOT/name) for name in SOURCES}
    report=dict(scope='all30 saved old validation predictions; fixed10 checkpoint replay; six seed11 real-model gradient/intervention checks; NO fresh test access',
        sourcehash=sourcehash,metric_tolerance=1e-8,replay_tolerance_m=2e-4,attention_tolerance=2e-4,
        source_paths=[],metrics=[],model_path=[],checkpoint_replay=[])
    (out/'protocol.json').write_text(json.dumps(report,indent=2))
    arrays=ids=None;common_sources=None;populations=[];checks=[]
    for role in ROLES:
        for seed in SEEDS:
            run=development/role/str(seed);metadata=json.loads((run/'run.json').read_text())
            if metadata['role']!=role or metadata['seed']!=seed or metadata['steps']!=1000:raise ValueError('unexpected development run')
            for name,digest in metadata['source_hashes'].items():
                if sha(ROOT/name)!=digest:raise ValueError('training/model source differs from saved run')
            for name,key in [('best.pt','checkpoint_sha256'),('val_predictions.npz','prediction_sha256'),('validation.json','validation_sha256')]:
                if sha(run/name)!=metadata[key]:raise ValueError('changed development output')
            if arrays is None:
                common_sources=metadata['data_sources']['val'];arrays,ids,manifests=load_old_validation(common_sources,True)
                checks.append(dict(name='boundary labels independently reconstruct',pass_=bool(np.array_equal(boundary_from_labels(arrays['target'],arrays['valid']),arrays['boundary']))))
            elif metadata['data_sources']['val']!=common_sources:raise ValueError('different validation populations across roles/seeds')
            with np.load(run/'val_predictions.npz',allow_pickle=False) as values:pred=values['depth'].copy()
            rows=numpy_rows(pred,arrays['target'],arrays['valid'],arrays['boundary'],ids)
            summary=numpy_macro(rows);stored=json.loads((run/'validation.json').read_text())
            comparison=compare_tree(dict(rows=rows,summary=summary),stored)
            report['metrics'].append(dict(role=role,seed=seed,rows=len(rows),comparison=comparison,
                checkpoint_sha256=metadata['checkpoint_sha256'],prediction_sha256=metadata['prediction_sha256'],summary=summary))
            (out/f'{role}_{seed}_numpy_rows.json').write_text(json.dumps(dict(rows=rows,summary=summary),indent=2))
            populations.append((summary['total_count'],tuple(r['count'] for r in rows),tuple(r['boundary_count'] for r in rows)))
            if seed==11:
                path=real_model_path(run,arrays,ids);report['model_path'].append(path)
            if seed==11 or role=='full':report['checkpoint_replay'].append(replay_checkpoint(run,arrays,ids))
            print('AUDIT',role,seed,'metric error',comparison['max_absolute_error'],'sec',round(time.time()-started,2),flush=True)
    checks.append(dict(name='all30 identical query and boundary populations',pass_=all(p==populations[0] for p in populations)))
    # Show the all-query task includes positions rejected by cross-view visibility labels.
    not_covisible={}
    for scene,old in [(OLD_SCENES[0],'results/round6_region_inputs'),(OLD_SCENES[1],'results/round6_new_region_inputs')]:
        path=ROOT/old/'val'/scene.replace('/','__')/'q8_k8/labels.npz'
        with np.load(path,allow_pickle=False) as values:co=values['point_gt_valid']
        chosen=np.array([i for i,r in enumerate(ids) if r['scene']==scene])
        not_covisible[scene]=dict(query_valid=int(arrays['valid'][chosen].sum()),
            query_valid_not_crossvisible=int((arrays['valid'][chosen]&~co).sum()),
            crossvisible_not_queryvalid=int((co&~arrays['valid'][chosen]).sum()),prior_label_sha256=sha(path))
    checks.append(dict(name='population includes query-valid cross-invisible positions',pass_=all(r['query_valid_not_crossvisible']>0 and r['crossvisible_not_queryvalid']==0 for r in not_covisible.values())))
    checks.append(dict(name='source unchanged during audit',pass_=sourcehash=={name:sha(ROOT/name) for name in SOURCES}))
    report.update(checks=checks,query_only_population=not_covisible,data_sources=common_sources,
        elapsed_seconds=time.time()-started,total_pair_rows=sum(r['rows'] for r in report['metrics']))
    report['pass']=all(not r['comparison']['mismatches'] for r in report['metrics']) and all(r['pass'] for r in report['model_path']) and all(r['pass_'] for r in report['checkpoint_replay']) and all(r['pass_'] for r in checks)
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    print('PASS',report['pass'],'sec',report['elapsed_seconds'],flush=True)
    if not report['pass']:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--development',type=Path,default=ROOT/'results/round7_depth_development')
    p.add_argument('--out',type=Path,default=ROOT/'results/round7_depth_audit_development')
    a=p.parse_args();audit(a.development,a.out)
