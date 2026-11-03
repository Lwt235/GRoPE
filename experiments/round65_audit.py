"""Independent metric/operator calculations and selected-checkpoint checks."""
import argparse
import math
from pathlib import Path
import numpy as np
from scipy.special import logsumexp,erf
import torch
from models.round65_actual_gs import NativeGSClassifier
from experiments.round65_common import OUT,CONFIG,ROLES,SEEDS,VARIANTS,checked,read,sha,write
from experiments.round65_data import load
from experiments.round65_run import configure,evaluate,evaluation_batch,shared_digest


def independent_metrics(logits,labels):
    x=np.asarray(logits,dtype=np.float64);y=np.asarray(labels,dtype=np.int64)
    if x.shape!=(len(y),60) or not np.isfinite(x).all():raise ValueError('invalid exported logits')
    correct=np.argmax(x,axis=1)==y
    classes=[float(correct[y==i].mean()) if np.any(y==i) else None for i in range(60)]
    ce=float((logsumexp(x,axis=1)-x[np.arange(len(y)),y]).mean())
    return dict(ce=ce,oa=float(correct.mean()),mca=float(np.mean([v for v in classes if v is not None])),
                class_accuracy=classes,count=len(y))


def independent_relation(centers,covariance):
    c,s=np.asarray(centers,dtype=np.float64),np.asarray(covariance,dtype=np.float64)
    precision=np.linalg.inv(s);spectrum=.5*np.log(np.maximum(np.linalg.eigvalsh(s),1e-15)/.04)
    out=np.empty((len(c),len(c),11),dtype=np.float64)
    for i in range(len(c)):
        for j in range(len(c)):
            delta=c[i]-c[j]
            out[i,j]=np.concatenate(([np.linalg.norm(delta)/.2],spectrum[i],spectrum[j],
                np.log1p([max(0,delta@precision[i]@delta),max(0,delta@precision[j]@delta),
                          max(0,np.trace(precision[i]@s[j])),max(0,np.trace(precision[j]@s[i]))])))
    return out


def independent_projection(inputs,position):
    state={k:v.detach().cpu().numpy() for k,v in position.state_dict().items()}
    x=inputs@state['mlp.0.weight'].T+state['mlp.0.bias']
    x=.5*x*(1+erf(x/np.sqrt(2)))
    return x@state['mlp.2.weight'].T


@torch.no_grad()
def selections(out=OUT):
    configure();checked(out)
    if not torch.cuda.is_available():raise RuntimeError('selected validation replay uses an idle GPU')
    if (out/'selected.json').exists():raise ValueError('preserve existing selection seal')
    val=load('val',out);count_fit=next(r['count'] for r in read(out/'data/manifest.json')['partitions'] if r['split']=='fit')
    expected_steps=CONFIG['epochs']*math.ceil(count_fit/CONFIG['batch_size'])
    expected_epochs=list(range(0,CONFIG['epochs']+1,CONFIG['validation_every']))
    records=[];initialization={};streams={};epoch_streams={}
    for seed in SEEDS:
        for role in ROLES:
            path=out/'runs'/role/str(seed);run=read(path/'run.json');history=read(path/'history.json')
            if run['role']!=role or run['seed']!=seed or run['epochs']!=CONFIG['epochs'] or run['steps']!=expected_steps:
                raise ValueError('declared training budget mismatch')
            if run['lock_sha256']!=sha(out/'lock.json') or run['data_sha256']!=sha(out/'data/manifest.json'):
                raise ValueError('run source/data identity')
            for name,digest in run['hashes'].items():
                if sha(path/name)!=digest:raise ValueError('training artifact changed '+name)
            if [r['epoch'] for r in history['validation']]!=expected_epochs or len(history['training'])!=CONFIG['epochs']:
                raise ValueError('incomplete training/validation history')
            best=min(history['validation'],key=lambda r:r['ce'])
            if best['epoch']!=run['best_epoch'] or best['ce']!=run['best_val_ce']:
                raise ValueError('selection rule mismatch')
            batch_steps=[r['step'] for r in history['training']]
            if batch_steps!=[i*math.ceil(count_fit/CONFIG['batch_size']) for i in range(1,CONFIG['epochs']+1)]:
                raise ValueError('epoch step accounting')
            epoch_digest=[r['stream_sha256'] for r in history['training']]
            for mapping,value in ((initialization,run['initial_shared_sha256']),(streams,run['sampling_sha256']),(epoch_streams,epoch_digest)):
                if mapping.setdefault(seed,value)!=value:raise ValueError('unpaired initialization or actual sampling')
            if not run['shared_parameters_changed'] or any(v<=0 for v in run['position_parameter_changes'].values()):
                raise ValueError('inactive trainable parameters')
            torch.manual_seed(seed);model=NativeGSClassifier(role)
            if shared_digest(model)!=run['initial_shared_sha256']:raise ValueError('initialization replay mismatch')
            checkpoint=torch.load(path/'best.pt',map_location='cpu',weights_only=True)
            if (checkpoint['role'],checkpoint['seed'],checkpoint['epoch'])!=(role,seed,run['best_epoch']):
                raise ValueError('selected checkpoint identity')
            model.load_state_dict(checkpoint['state_dict']);model.cuda().eval()
            with np.load(path/'validation.npz',allow_pickle=False) as z:
                saved={k:z[k] for k in z.files}
            if not np.array_equal(saved['ids'],val['ids']) or not np.array_equal(saved['labels'],val['labels']):
                raise ValueError('selected validation input identity')
            stats=independent_metrics(saved['logits'],saved['labels'])
            ce_error=abs(stats['ce']-run['best_val_ce'])
            replay,replay_stats=evaluate(model,val)
            gpu_error=float(np.max(np.abs(replay-saved['logits'])))
            if ce_error>1e-8 or not np.allclose(replay,saved['logits'],atol=2e-4,rtol=2e-4):
                raise ValueError('selected validation arithmetic/replay discrepancy')
            model.cpu();args=evaluation_batch(val,0,2,'base','cpu')
            cpu_logits=model(*args).numpy()
            cpu_error=float(np.max(np.abs(cpu_logits-saved['logits'][:2])))
            cpu_close=bool(np.allclose(cpu_logits,saved['logits'][:2],atol=2e-4,rtol=2e-4))
            operator_error=None
            if role=='gaussian_relation':
                c,s=args[0][:1,:12],args[1][:1,:12]
                expected=independent_projection(independent_relation(c[0].numpy(),s[0].numpy()),model.position)
                actual=model.position(c,s)[0].numpy();operator_error=float(np.max(np.abs(expected-actual)))
                if not np.allclose(expected,actual,atol=2e-4,rtol=2e-4):
                    raise ValueError('independent relation mismatch')
            if not cpu_close:raise ValueError('CPU replay needs discrepancy review before evaluation')
            records.append(dict(role=role,seed=seed,best_epoch=run['best_epoch'],validation_metrics=stats,
                independent_ce_error=ce_error,gpu_logit_error=gpu_error,cpu_logit_error=cpu_error,
                cpu_prediction_disagreements=int(np.sum(cpu_logits.argmax(1)!=saved['logits'][:2].argmax(1))),
                independent_operator_error=operator_error,checkpoint_sha256=sha(path/'best.pt'),run_sha256=sha(path/'run.json')))
            print('SELECTION_CHECKED',role,seed,ce_error,gpu_error,cpu_error,flush=True)
            del model;torch.cuda.empty_cache()
    write(out/'selected.json',dict(records=records,all_fifteen_selected_before_development_evaluation=True,
        lock_sha256=sha(out/'lock.json'),data_sha256=sha(out/'data/manifest.json'),confirmation_access=False))


def bootstrap(class_seed_differences):
    values=np.asarray(class_seed_differences,dtype=float).mean(axis=0)
    rng=np.random.default_rng(CONFIG['bootstrap_seed'])
    samples=values[rng.integers(0,len(values),size=(CONFIG['bootstrap_replicates'],len(values)))].mean(1)
    return np.quantile(samples,[.025,.975]).tolist()


def metrics(out=OUT):
    checked(out);seal=read(out/'selected.json');data=load('dev_eval',out)
    records=[];lookup={};worst=0.
    for role in ROLES:
        for seed in SEEDS:
            selected=next(r for r in seal['records'] if r['role']==role and r['seed']==seed)
            dest=out/'evaluation'/role/str(seed);meta=read(dest/'record.json')
            if meta['checkpoint_sha256']!=selected['checkpoint_sha256'] or meta['selection_sha256']!=sha(out/'selected.json'):
                raise ValueError('export selection identity')
            if meta['data_sha256']!=sha(out/'data/manifest.json') or [r['variant'] for r in meta['records']]!=list(VARIANTS):
                raise ValueError('export data/variant identity')
            for entry in meta['records']:
                variant=entry['variant'];path=dest/(variant+'.npz')
                if sha(path)!=entry['sha256']:raise ValueError('prediction bytes changed')
                with np.load(path,allow_pickle=False) as z:
                    logits,labels,ids=z['logits'],z['labels'],z['ids']
                if not np.array_equal(labels,data['labels']) or not np.array_equal(ids,data['ids']):
                    raise ValueError('evaluation identity/order mismatch')
                stats=independent_metrics(logits,labels)
                differences=[abs(stats[k]-entry['metrics'][k]) for k in ('ce','oa','mca')]
                differences+=np.abs(np.asarray(stats['class_accuracy'])-np.asarray(entry['metrics']['class_accuracy'])).tolist()
                error=max(differences);worst=max(worst,error)
                if error>1e-8:raise ValueError('independent classification metric mismatch')
                domains={name:independent_metrics(logits[mask],labels[mask]) for name,mask in
                         (('MACGS',labels<30),('ModelNet',labels>=30))}
                record=dict(role=role,seed=seed,variant=variant,metrics=stats,domains=domains,
                            metric_error=error,prediction_sha256=entry['sha256'])
                records.append(record);lookup[(role,seed,variant)]=record
    comparisons=[]
    for variant in VARIANTS:
        for control in ROLES[1:]:
            class_diff=np.array([np.array(lookup[('gaussian_relation',s,variant)]['metrics']['class_accuracy'])-
                                 np.array(lookup[(control,s,variant)]['metrics']['class_accuracy']) for s in SEEDS])*100
            paired=class_diff.mean(1);interval=bootstrap(class_diff)
            comparisons.append(dict(control=control,variant=variant,mca_gain_pp=float(paired.mean()),
                paired_seed_gains_pp=paired.tolist(),paired_class_bootstrap_95_pp=interval,
                classes_with_positive_mean=int(np.sum(class_diff.mean(0)>0)),
                development_gain_criterion=bool(np.all(paired>0) and paired.mean()>=CONFIG['minimum_gain_pp'] and interval[0]>0)))
    favorable=all(c['development_gain_criterion'] for c in comparisons if c['variant']=='base')
    write(out/'metrics/summary.json',dict(records=records,comparisons=comparisons,groups=len(records),
        independent_metric_max_error=worst,base_development_lead=favorable,
        goal_complete=False,fresh_confirmation=False,requires_scientific_and_plot_review=True))
    print('METRICS_COMPLETE',len(records),worst,'base_development_lead',favorable,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('selections','metrics'))
    parser.add_argument('--out',type=Path,default=OUT);args=parser.parse_args()
    (selections if args.action=='selections' else metrics)(args.out)
