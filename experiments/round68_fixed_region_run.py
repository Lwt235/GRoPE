"""Five matched regional-PE probes with fixed physical reference queries."""
import argparse
import hashlib
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import torch
from experiments import round65_queue as scheduler
from experiments.round65_common import ROOT,SEEDS,read,write,sha
from experiments.round65_run import configure,atomic_save,stream_update
from experiments.round65_data import quaternion_rotation
from experiments.round58_data import loss_function
from experiments.round58_run import shared_digest,metric_torch
from experiments.round60_audit import audit_sample
from experiments.round68_fixed_region_data import OUT,check as data_check,load_cache
from models.round68_fixed_region import FixedRegionMatcher,ROLES

SETTINGS=dict(epochs=60,batch_size=32,eval_batch_size=16,lr=.001,final_lr=.00001,
    warmup_epochs=3,weight_decay=.01,clip=1.,validation_every=3,
    input_fields=['features','centers','covariance','permutation'],
    conditions=['base','sparse256'],seeds=SEEDS,roles=ROLES)
MODULE='experiments.round68_fixed_region_run'


def plan():
    data_check()
    if (OUT/'lock.json').exists():raise ValueError('preserve previous fitting plan')
    manifest=read(OUT/'data/manifest.json')
    if not manifest['fixed_sparse_reference_verified']:raise ValueError('fixed reference required')
    data_review=read(OUT/'data_review.json')
    if data_review['source_sha256']!=sha(Path(__file__)) or data_review['manifest_sha256']!=sha(OUT/'data/manifest.json'):
        raise ValueError('data review provenance changed')
    write(OUT/'lock.json',dict(settings=SETTINGS,source_sha256=sha(Path(__file__)),
        metric_test_sha256=sha(ROOT/'tests/test_round68_metrics.py'),
        preparation_plan_sha256=sha(OUT/'preparation_plan.json'),
        data_manifest_sha256=sha(OUT/'data/manifest.json'),created_unix=time.time(),
        data_review_sha256=sha(OUT/'data_review.json'),
        hypothesis='Gaussian relation organization adds matching utility beyond same complete content and raw geometry MLP',
        competing_explanations=['distance/content suffice','raw MLP suffices','normal/scale suffice','shape nuisance'],
        fresh_confirmation=False,goal_complete=False))
    (OUT/'lock.sha256').write_text(sha(OUT/'lock.json')+'\n')
    print('R68_FIT_PLAN',sha(OUT/'lock.json'),flush=True)


def check():
    data_check();p=read(OUT/'lock.json')
    if sha(OUT/'lock.json')!=(OUT/'lock.sha256').read_text().strip():raise ValueError('lock changed')
    if p['source_sha256']!=sha(Path(__file__)):raise ValueError('fitting source changed')
    if p['metric_test_sha256']!=sha(ROOT/'tests/test_round68_metrics.py'):raise ValueError('metric test changed')
    if p['preparation_plan_sha256']!=sha(OUT/'preparation_plan.json'):raise ValueError('preparation plan changed')
    if p['data_manifest_sha256']!=sha(OUT/'data/manifest.json'):raise ValueError('data manifest changed')
    return p


def inputs(cache,index,device='cuda',quaternion=None,translation=None):
    values=[torch.as_tensor(cache[k][index],device=device) for k in SETTINGS['input_fields']]
    if quaternion is not None:
        rotation=torch.as_tensor(quaternion_rotation(quaternion),device=device,dtype=torch.float64)
        values[1]=values[1]@rotation.to(values[1]).transpose(-1,-2)+torch.as_tensor(translation,device=device,dtype=values[1].dtype)[:,:,None]
        r=rotation[:,:,None]
        values[2]=r@values[2].double()@r.transpose(-1,-2)
    return tuple(values)


def query_values(logits,cache):
    x=np.asarray(logits,dtype=np.float64);o=np.asarray(cache['overlap'],dtype=np.float64)
    world=np.asarray(cache['world_centers'],dtype=np.float64)
    values={}
    for direction in range(2):
        score=x if direction==0 else x.swapaxes(-1,-2)
        truth=o if direction==0 else o.swapaxes(-1,-2)
        a,b=world[:,direction],world[:,1-direction]
        z=score-score.max(-1,keepdims=True);logp=z-np.log(np.exp(z).sum(-1,keepdims=True));prob=np.exp(logp)
        picked=score.argmax(-1)[...,None]
        valid=truth.max(-1)>=.3
        overlap=np.take_along_axis(truth,picked,-1)[...,0]
        distance=np.linalg.norm(a[:,:,None]-b[:,None],axis=-1)
        hard=np.take_along_axis(distance,picked,-1)[...,0]
        weight=truth*(truth>=.1);target=weight/np.maximum(weight.sum(-1,keepdims=True),1e-12)
        row=dict(valid=valid,region_recall=(overlap>=.3).astype(float),center_pck=(hard<=.1).astype(float),
                 soft_epe=np.linalg.norm(prob@b-a,axis=-1),ce=-(target*logp).sum(-1))
        for key,value in row.items():values.setdefault(key,[]).append(value)
    return {key:np.stack(value,axis=1) for key,value in values.items()}


def aggregate(values,mask=None):
    valid=values['valid'];mask=np.ones_like(valid,dtype=bool) if mask is None else np.broadcast_to(mask,valid.shape)
    selected=valid&mask;count=int(selected.sum());all_count=int(mask.sum())
    stats={key:float(value[selected].mean()) if count else None for key,value in values.items() if key!='valid'}
    stats.update(valid_queries=count,total_queries=all_count,coverage=count/max(1,all_count),
        valid_and_correct_fraction=float((values['region_recall']*selected).sum()/max(1,all_count)))
    return stats


def grouped_metrics(logits,cache):
    values=query_values(logits,cache)
    def grouping(mask=None):
        by_scene={scene:aggregate(values,(cache['scenes']==scene)[:,None,None] &
                    (np.ones_like(values['valid'],dtype=bool) if mask is None else mask))
                  for scene in sorted(set(cache['scenes'].tolist()))}
        eligible=[row for row in by_scene.values() if row['valid_queries']]
        macro={key:float(np.mean([r[key] for r in eligible])) if eligible else None
               for key in ('region_recall','center_pck','soft_epe','ce')}
        return dict(micro=aggregate(values,mask),by_scene=by_scene,macro=macro,evaluable_scenes=len(eligible))
    overall=grouping()
    overall['strata']={str(group):grouping(cache['strata']==group) for group in range(4)}
    return overall


@torch.no_grad()
def evaluate(model,cache):
    model.eval();logits=[]
    for start in range(0,len(cache['ids']),SETTINGS['eval_batch_size']):
        logits.append(model(*inputs(cache,slice(start,start+SETTINGS['eval_batch_size']))).cpu().numpy())
    logits=np.concatenate(logits)
    return logits,metric_torch(logits,cache['overlap'],cache['world_centers'])


def audit_data():
    rows=[]
    for name in ('fit0','fit1','val','eval_base'):
        path=OUT/'data'/(name+'_sample.npz')
        with np.load(path,allow_pickle=False) as z:
            adapted={('token_'+key[len('reference_'):]) if key.startswith('reference_') else key:z[key]
                     for key in z.files if not key.startswith('sparse_')}
        target=OUT/'checks'/(name+'_legacy_sample.npz');target.parent.mkdir(exist_ok=True)
        np.savez_compressed(target,**adapted)
        rows.append(audit_sample(target))
    with np.load(OUT/'data/eval_base_sample.npz',allow_pickle=False) as z:
        a={key:z[key] for key in z.files}
    worst=0.
    for b in range(len(a['points'])):
        for v in range(2):
            keep=a['sparse_retained'][b,v]
            if keep.sum()!=256 or not keep[a['reference_anchors'][b,v]].all():raise ValueError('thinning changes anchors')
            for t in range(64):
                ids=a['reference_neighbors'][b,v,t];points=a['points'][b,v,ids[keep[ids]]].astype(np.float64)
                center=points.mean(0);delta=points-center
                cov=delta.T@delta/len(points)+.02**2*np.eye(3)
                worst=max(worst,float(abs(center-a['sparse_centers'][b,v,t]).max()),float(abs(cov-a['sparse_covariance'][b,v,t]).max()))
    if worst>2e-5:raise ValueError('substantive sparse moment discrepancy')
    base,sparse=load_cache('eval_base'),load_cache('eval_sparse256')
    for key in ('ids','scenes','overlap','world_centers','strata'):
        if not np.array_equal(base[key],sparse[key]):raise ValueError('reference query task changed')
    return dict(base_checks=rows,sparse_direct_moment_max_error=worst,full_sparse_reference_exact=True)


def preflight():
    lock=check();configure()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    if (OUT/'preflight.json').exists():raise ValueError('preserve completed preflight')
    if sha(OUT/'data_review.json')!=lock['data_review_sha256']:raise ValueError('data review changed')
    audit=read(OUT/'data_review.json')['checks'];cache=load_cache('fit0');common=None;rows=[]
    valid=(cache['overlap'].max(-1)>=.3).any(-1);index=np.flatnonzero(valid)[:4]
    for role in ROLES:
        torch.manual_seed(11);model=FixedRegionMatcher(role).cuda().train()
        digest=shared_digest(model)
        if common is None:common=digest
        if digest!=common:raise ValueError('different common initialization')
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        logits=model(*inputs(cache,index));truth=torch.as_tensor(cache['overlap'][index],device='cuda')
        loss=loss_function(logits,truth);loss.backward()
        grads={name:float(parameter.grad.norm()) if parameter.grad is not None else None for name,parameter in model.position.named_parameters()}
        if any(value is None or not np.isfinite(value) for value in grads.values()) or not any(value>0 for value in grads.values()):
            raise ValueError('invalid PE gradient')
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        sub={key:value[index] for key,value in cache.items()}
        actual=aggregate(query_values(logits.detach().cpu().numpy(),sub))
        independent=metric_torch(logits.detach().cpu(),sub['overlap'],sub['world_centers'])
        if max(abs(actual[k]-independent[k]) for k in ('region_recall','center_pck','soft_epe','ce'))>1e-10:
            raise ValueError('independent metric disagreement')
        rows.append(dict(role=role,loss=float(loss),gradients=grads,parameters=sum(p.numel() for p in model.parameters()),
            pe_parameters=sum(p.numel() for p in model.position.parameters())))
        del model,opt
    write(OUT/'preflight.json',dict(passed=True,rows=rows,data_checks=audit,common_initialization_exact=True,
        gpu=torch.cuda.get_device_name(),lock_sha256=sha(OUT/'lock.json')))
    print('R68_PREFLIGHT_COMPLETE',flush=True)


def rate(step,per_epoch):
    warm=3*per_epoch;total=60*per_epoch
    if step<warm:return .001*(step+1)/warm
    return .00001+.5*(.001-.00001)*(1+math.cos(math.pi*(step-warm)/(total-warm-1)))


def train(role,seed):
    check();configure()
    if role not in ROLES or seed not in SEEDS or not read(OUT/'preflight.json')['passed']:raise ValueError('scope/preflight')
    dest=OUT/'runs'/role/str(seed)
    if dest.exists():raise ValueError('preserve previous fit')
    dest.mkdir(parents=True)
    fitting=[load_cache('fit0'),load_cache('fit1')];val=load_cache('val')
    if not np.array_equal(fitting[0]['ids'],fitting[1]['ids']):raise ValueError('different fit populations')
    torch.manual_seed(seed);model=FixedRegionMatcher(role).cuda();initial=shared_digest(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    rng=np.random.default_rng(seed+68000);stream=hashlib.sha256(b'R68-input-stream-v1').hexdigest()
    count=len(fitting[0]['ids']);per_epoch=math.ceil(count/32)
    history=dict(training=[],validation=[]);best=float('inf');best_epoch=None;step=0
    started=time.time();torch.cuda.reset_peak_memory_stats()
    try:
        for epoch in range(61):
            if epoch%3==0:
                logits,stats=evaluate(model,val);history['validation'].append(dict(epoch=epoch,step=step,**stats))
                if stats['selection_ce']<best:
                    best,best_epoch=stats['selection_ce'],epoch
                    atomic_save(dest/'best.pt',dict(role=role,seed=seed,epoch=epoch,state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}))
                    np.savez_compressed(dest/'validation.npz',logits=logits,ids=val['ids'])
                write(dest/'history.json',history);print('VAL',role,seed,epoch,stats['selection_ce'],stats['region_recall'],flush=True)
            if epoch==60:break
            cache=fitting[epoch%2];order=rng.permutation(count);total_loss=0.;model.train()
            stream=stream_update(stream,np.array([epoch%2],dtype=np.int64),order)
            for begin in range(0,count,32):
                index=order[begin:begin+32];q=rng.normal(size=(len(index),2,4));t=rng.uniform(-.5,.5,size=(len(index),2,3))
                stream=stream_update(stream,index,q,t)
                optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:group['lr']=rate(step,per_epoch)
                loss=loss_function(model(*inputs(cache,index,quaternion=q,translation=t)),torch.as_tensor(cache['overlap'][index],device='cuda'))
                if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();step+=1;total_loss+=float(loss)*len(index)
            history['training'].append(dict(epoch=epoch+1,step=step,loss=total_loss/count,stream_sha256=stream,last_gradient_norm=float(norm)))
            write(dest/'history.json',history)
        write(dest/'run.json',dict(role=role,seed=seed,epochs=60,steps=step,selected_epoch=best_epoch,
            best_val_ce=best,initial_shared_sha256=initial,final_stream_sha256=stream,
            parameters=sum(p.numel() for p in model.parameters()),pe_parameters=sum(p.numel() for p in model.position.parameters()),
            seconds=time.time()-started,peak_allocated=torch.cuda.max_memory_allocated(),
            hashes={name:sha(dest/name) for name in ('best.pt','validation.npz','history.json')},
            lock_sha256=sha(OUT/'lock.json')))
        print('R68_FIT_COMPLETE',role,seed,best_epoch,flush=True)
    except BaseException as error:
        write(dest/'failure.json',dict(type=type(error).__name__,error=str(error),step=step,epoch=epoch));raise


def selections():
    check();val=load_cache('val');records=[];pairing=[]
    if (OUT/'selected.json').exists():raise ValueError('preserve selected models')
    for seed in SEEDS:
        initial=None;streams=None
        for role in ROLES:
            dest=OUT/'runs'/role/str(seed);run=read(dest/'run.json');history=read(dest/'history.json')
            if run['epochs']!=60 or run['steps']!=1080 or len(history['training'])!=60:
                raise ValueError('changed fit budget')
            actual=[v['stream_sha256'] for v in history['training']]
            if initial is None:initial=run['initial_shared_sha256'];streams=actual
            if initial!=run['initial_shared_sha256'] or streams!=actual:
                raise ValueError('unpaired common initialization or augmentation stream')
            for name,digest in run['hashes'].items():
                if sha(dest/name)!=digest:raise ValueError('fit artifact changed')
            best=min(history['validation'],key=lambda r:r['selection_ce'])
            if best['epoch']!=run['selected_epoch']:raise ValueError('selection rule changed')
            with np.load(dest/'validation.npz',allow_pickle=False) as z:
                if not np.array_equal(z['ids'],val['ids']):raise ValueError('validation membership')
                values=query_values(z['logits'],val)
            ce=[]
            for direction in range(2):
                valid=values['valid'][:,direction]
                if valid.any():ce.append(float(values['ce'][:,direction][valid].mean()))
            error=abs(float(np.mean(ce))-run['best_val_ce'])
            if error>1e-10:raise ValueError('independent selection metric')
            records.append(dict(role=role,seed=seed,selected_epoch=run['selected_epoch'],
                checkpoint_sha256=sha(dest/'best.pt'),run_sha256=sha(dest/'run.json'),validation_ce_error=error))
        pairing.append(dict(seed=seed,all_five_common_initializations_equal=True,all60_augmented_streams_equal=True))
    write(OUT/'selected.json',dict(records=records,pairing=pairing,all15_selected_before_eval=True,
        lock_sha256=sha(OUT/'lock.json')))
    print('R68_SELECTIONS_COMPLETE',[(r['role'],r['seed'],r['selected_epoch']) for r in records],flush=True)


def export(seed):
    check();configure();selected=read(OUT/'selected.json');val=load_cache('val')
    if seed not in SEEDS or len(selected['records'])!=15:raise ValueError('finish15 selections')
    caches={condition:load_cache('eval_'+condition) for condition in ('base','sparse256')}
    for role in ROLES:
        choice=next(r for r in selected['records'] if r['role']==role and r['seed']==seed)
        run=OUT/'runs'/role/str(seed);dest=OUT/'evaluation'/role/str(seed)
        if dest.exists():raise ValueError('preserve export')
        if sha(run/'best.pt')!=choice['checkpoint_sha256']:raise ValueError('selected weights changed')
        model=FixedRegionMatcher(role).cuda()
        model.load_state_dict(torch.load(run/'best.pt',map_location='cpu',weights_only=True)['state_dict'])
        logits,_=evaluate(model,val)
        with np.load(run/'validation.npz',allow_pickle=False) as z:
            error=float(abs(logits-z['logits']).max());changes=int((logits.argmax(-1)!=z['logits'].argmax(-1)).sum())
        if error>2e-4 or changes:raise ValueError('selected validation replay changed task decisions')
        dest.mkdir(parents=True);records=[]
        for condition,cache in caches.items():
            logits,independent=evaluate(model,cache);metrics=grouped_metrics(logits,cache)
            metric_error=max(abs(independent[k]-metrics['micro'][k]) for k in ('region_recall','center_pck','soft_epe','ce'))
            if metric_error>1e-10:raise ValueError('independent evaluation metric')
            path=dest/(condition+'.npz');np.savez_compressed(path,logits=logits,ids=cache['ids'])
            records.append(dict(role=role,seed=seed,condition=condition,metrics=metrics,
                prediction_sha256=sha(path),independent_metric_error=metric_error))
            print('R68_EXPORT',role,seed,condition,metrics['macro'],flush=True)
        write(dest/'record.json',dict(records=records,checkpoint_sha256=choice['checkpoint_sha256'],
            selection_sha256=sha(OUT/'selected.json'),manifest_sha256=sha(OUT/'data/manifest.json'),
            validation_replay=dict(maximum_logit_error=error,prediction_changes=changes)))
        del model;torch.cuda.empty_cache()


def contrasts(lookup):
    comparisons=[]
    for condition in ('base','sparse256'):
        for control in (r for r in ROLES if r!='gaussian_relation'):
            for stratum in (None,0,1,2,3):
                pairs=[];scene_effects=[];scenes=None
                for seed in SEEDS:
                    g,d=(lookup[role,seed,condition]['metrics'] for role in ('gaussian_relation',control))
                    if stratum is not None:g,d=g['strata'][str(stratum)],d['strata'][str(stratum)]
                    if g['macro']['region_recall'] is None or d['macro']['region_recall'] is None:continue
                    pairs.append(dict(seed=seed,region_gain_pp=100*(g['macro']['region_recall']-d['macro']['region_recall']),
                        center_gain_pp=100*(g['macro']['center_pck']-d['macro']['center_pck']),
                        soft_epe=g['macro']['soft_epe']-d['macro']['soft_epe'],ce=g['macro']['ce']-d['macro']['ce']))
                    scenes=[name for name in g['by_scene'] if g['by_scene'][name]['valid_queries']]
                    scene_effects.append([100*(g['by_scene'][name]['region_recall']-d['by_scene'][name]['region_recall']) for name in scenes])
                if len(pairs)!=3:
                    comparisons.append(dict(condition=condition,control=control,stratum=stratum,available=False));continue
                average=np.mean(scene_effects,axis=0)
                draw=np.random.default_rng(68001).integers(0,len(average),(10000,len(average)))
                ci=np.quantile(average[draw].mean(1),[.025,.975])
                mean={key:float(np.mean([r[key] for r in pairs])) for key in pairs[0] if key!='seed'}
                comparisons.append(dict(condition=condition,control=control,stratum=stratum,available=True,
                    paired=pairs,mean=mean,scene_mean_region_gains_pp=dict(zip(scenes,average.tolist())),
                    paired_scene_bootstrap_95_pp=ci.tolist(),
                    development_lead=bool(mean['region_gain_pp']>=1 and all(r['region_gain_pp']>0 for r in pairs)),
                    center_mean_loss_exceeds1pp=bool(mean['center_gain_pp'] < -1)))
    return comparisons


def report():
    check();selected=read(OUT/'selected.json');rows=[];worst=0.
    if (OUT/'result.json').exists():raise ValueError('preserve completed report')
    for choice in selected['records']:
        dest=OUT/'evaluation'/choice['role']/str(choice['seed']);meta=read(dest/'record.json')
        if meta['checkpoint_sha256']!=choice['checkpoint_sha256'] or meta['selection_sha256']!=sha(OUT/'selected.json'):
            raise ValueError('export identities')
        for row in meta['records']:
            if sha(dest/(row['condition']+'.npz'))!=row['prediction_sha256']:raise ValueError('prediction bytes changed')
            rows.append(row);worst=max(worst,row['independent_metric_error'])
    if len(rows)!=30:raise ValueError('expected30 groups')
    lookup={(r['role'],r['seed'],r['condition']):r for r in rows};comparisons=contrasts(lookup)
    lead=all(r['development_lead'] for r in comparisons if r['condition']=='base' and r['stratum'] is None
             and r['control'] in ('distance_mlp','direct_mlp'))
    write(OUT/'result.json',dict(records=rows,comparisons=comparisons,base_development_lead=lead,
        independent_metric_max_error=worst,lock_sha256=sha(OUT/'lock.json'),
        scientific_review_pending=True,visual_review_pending=True,fresh_confirmation=False,goal_complete=False))
    lines=['# R68 fixed-region cross-view mechanism study','',
        'All methods receive complete20-channel region content. Fifteen paired from-scratch fits; '
        'base and sparse inputs use exactly the same reference queries, labels and strata. '
        'Sparse observations use known full-density region membership. These are used SUN3D development scenes, not native optimized3DGS.','',
        '| Condition | Method | Region R@1 % | Center PCK % | Soft EPE | CE |',
        '|---|---|---:|---:|---:|---:|']
    for condition in ('base','sparse256'):
        for role in ROLES:
            m=[lookup[role,seed,condition]['metrics']['macro'] for seed in SEEDS]
            lines.append(f'| {condition} | {role} | '+
                ' | '.join(f'{np.mean([v[k] for v in m])*(100 if k in ("region_recall","center_pck") else 1):.4f}'
                           for k in ('region_recall','center_pck','soft_epe','ce'))+' |')
    lines+=['','| Condition | Gaussian minus control | Seed region gains pp | Mean pp | Scene95% pp | Center change pp | Lead |',
        '|---|---|---|---:|---|---:|---|']
    for row in comparisons:
        if row['stratum'] is not None or not row['available']:continue
        lo,hi=row['paired_scene_bootstrap_95_pp'];m=row['mean']
        lines.append(f"| {row['condition']} | {row['control']} | "+' / '.join(f"{p['region_gain_pp']:+.3f}" for p in row['paired'])+
            f" | {m['region_gain_pp']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {m['center_gain_pp']:+.3f} | {row['development_lead']} |")
    lines+=['','Four input-defined strata:0 low ambiguity/low shape diversity;1 low/high;2 high/low;3 high/high.','',
        '| Condition | Control | Stratum | Region gain pp | Center gain pp |',
        '|---|---|---|---:|---:|']
    for row in comparisons:
        if row['stratum'] is None:continue
        if not row['available']:
            lines.append(f"| {row['condition']} | {row['control']} | {row['stratum']} | unavailable | unavailable |")
        else:
            lines.append(f"| {row['condition']} | {row['control']} | {row['stratum']} | {row['mean']['region_gain_pp']:+.3f} | {row['mean']['center_gain_pp']:+.3f} |")
    lines+=['','Coverage and counts are in data/*json and each result group. Both conditions preserve their exact '
        'reference population; valid_and_correct_fraction is valid-and-correct divided by all queries, not unmatched-query rejection accuracy. '
        'Scene intervals condition on three selected seeds and only five scenes; they do not establish broad generalization. '
        'Input strata are association diagnostics, not causal proof. Scientific and visual review pending.','','![Comparison](comparison.png)','']
    (OUT/'report.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    full=[r for r in comparisons if r['stratum'] is None]
    for y,row in enumerate(full):
        axes[0].plot(row['paired_scene_bootstrap_95_pp'],[y,y],color='black')
        axes[0].scatter([r['region_gain_pp'] for r in row['paired']],y+np.array([-.1,0,.1]),color='tab:blue')
    axes[0].axvline(0,color='grey');axes[0].set(yticks=range(len(full)),
        yticklabels=[r['condition']+' vs '+r['control'] for r in full],xlabel='Gaussian region R@1 change (pp)')
    axes[0].invert_yaxis()
    for control in ('distance_mlp','direct_mlp','normal_scale'):
        values=[r for r in comparisons if r['condition']=='base' and r['control']==control and r['stratum'] is not None and r['available']]
        axes[1].plot([r['stratum'] for r in values],[r['mean']['region_gain_pp'] for r in values],marker='o',label=control)
    axes[1].axhline(0,color='grey');axes[1].set(xticks=range(4),xticklabels=['low/low','low/high','high/low','high/high'],
        xlabel='Center ambiguity / shape diversity',ylabel='Gaussian region R@1 change (pp)');axes[1].legend(fontsize=8)
    fig.suptitle('R68 | matched content and fixed physical reference regions | used development')
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig)
    print('R68_REPORT_COMPLETE',lead,flush=True)


def launch(stage,job,gpu):
    tag=stage if job is None else stage+'_'+'_'.join(map(str,job));key=tag+'_'+str(time.time_ns())
    command=[sys.executable,'-m',MODULE,stage]
    if stage=='train':command+=['--role',job[0],'--seed',str(job[1])]
    elif stage=='export':command+=['--seed',str(job[0])]
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_DEVICE_ORDER='PCI_BUS_ID',
        CUDA_VISIBLE_DEVICES='' if gpu is None else gpu['uuid'],MPLCONFIGDIR='/tmp/grope-r68-matplotlib')
    stream=(OUT/(key+'.log')).open('x')
    process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    meta=dict(tag=key,stage=stage,job=job,pid=process.pid,gpu=gpu,command=command,started_unix=time.time(),
        start_ticks=(Path('/proc')/str(process.pid)/'stat').read_text().split()[21])
    scheduler.ACTIVE[key]=dict(process=process,stream=stream,meta=meta)
    print('LAUNCHED',tag,process.pid,None if gpu is None else gpu['index'],flush=True)


def queue():
    check()
    # The fixed scientific comparison is independent of R67 outcomes. Coordinate
    # allocations so two controllers never compete while R67 starts GPU jobs.
    peer_path=ROOT/'results/round67_training_path/attempt1/queue_status.json'
    def peer_allows_allocation():
        peer=read(peer_path)
        return (peer['state']=='complete' or
                (peer['state']=='running' and peer.get('stage')=='train' and not peer.get('pending')))
    if (OUT/'queue_status.json').exists():raise ValueError('inspect existing controller before restart')
    scheduler.OUT=OUT;scheduler.INFO.update(pid=os.getpid(),started_unix=time.time(),
        start_ticks=Path('/proc/self/stat').read_text().split()[21],lock_sha256=sha(OUT/'lock.json'))
    def stop(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    phases=[('preflight',[None]),('train',[(ROLES[(j+i)%5],seed) for j in range(5) for i,seed in enumerate(SEEDS)]),
            ('selections',[None]),('export',[(seed,) for seed in SEEDS]),('report',[None])]
    try:
        for stage,tasks in phases:
            pending=list(tasks);previous=set()
            while pending or scheduler.ACTIVE:
                if scheduler.collect():raise ValueError('unexpected retry')
                if not pending and not scheduler.ACTIVE:break
                if stage in ('selections','report'):
                    if pending:launch(stage,pending.pop(0),None)
                    scheduler.record('running',stage=stage,pending=pending)
                else:
                    if not peer_allows_allocation():
                        previous=set()
                        scheduler.record('waiting_for_peer_gpu_phase',stage=stage,pending=pending)
                        time.sleep(5);continue
                    rows=scheduler.occupancy();available={r['uuid'] for r in rows if r['free']};stable=available&previous
                    owned={x['meta']['gpu']['uuid'] for x in scheduler.ACTIVE.values()}
                    for gpu in rows:
                        if not pending or len(scheduler.ACTIVE)>=2 or not peer_allows_allocation():break
                        if gpu['uuid'] in stable-owned:
                            launch(stage,pending.pop(0),gpu);owned.add(gpu['uuid'])
                    previous=available-owned
                    scheduler.record('running' if scheduler.ACTIVE else 'waiting_for_gpu',stage=stage,pending=pending,occupancy=rows)
                time.sleep(5)
        scheduler.record('complete',task_execution_completed=True,scientific_review_pending=True,goal_complete=False)
    except BaseException as error:
        scheduler.stop_owned();scheduler.record('stopped',error_type=type(error).__name__,error=str(error),goal_complete=False);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('plan','preflight','train','selections','export','report','queue'))
    parser.add_argument('--role',choices=ROLES);parser.add_argument('--seed',type=int);args=parser.parse_args()
    if args.action=='train':train(args.role,args.seed)
    elif args.action=='export':export(args.seed)
    else:globals()[args.action]()
