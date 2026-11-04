"""Single fixed warm-start comparison; reuse established data and arithmetic."""
import argparse
import copy
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from experiments import round65_run as runner, round65_queue as scheduler
from experiments.round65_common import ROOT, OUT as OLD, CONFIG, SEEDS, VARIANTS, read, write, sha
from experiments.round65_data import load
from experiments.round66_logshape import specification
from experiments.round66_classification_diagnostic import observations, summarize
from models.round65_actual_gs import NativeGSClassifier
from models.round66_logshape import LogShapeClassifier
from models.round67_training_path import ROLES, attach_shape, make_model

OUT = ROOT/'results/round67_training_path/attempt1'
MODULE = 'experiments.round67_training_path'
SOURCES = ('docs/round67_training_path_plan.md', 'models/round67_training_path.py',
           'experiments/round67_training_path.py', 'experiments/round66_classification_diagnostic.py')
SETTINGS = dict(epochs=20, learning_rate=1e-4, final_learning_rate=1e-5,
    validation_every=5, selection='earliest_strict_minimum_val_ce_including_epoch0',
    initialization='same_seed_distance_terminal120_with_optimizer_and_sampler',
    roles=ROLES, seeds=SEEDS, variants=VARIANTS)
DIAGNOSTIC = ROOT/'results/round66_logshape/classification_diagnostic1'


def original_run(seed):
    return OLD/'runs/distance_mlp'/str(seed)


def plan():
    specification()
    if OUT.exists():raise ValueError('preserve prior R67 attempt')
    review=read(DIAGNOSTIC/'review.json')
    if review['scientific_review_pending'] or review['visual_review_pending']:
        raise ValueError('finish existing classification diagnostic')
    parents={}
    for seed in SEEDS:
        run=original_run(seed); meta=read(run/'run.json')
        if sha(run/'resume.pt')!=meta['hashes']['resume.pt']:
            raise ValueError('original distance checkpoint changed')
        parents[str(seed)]=dict(checkpoint_sha256=sha(run/'resume.pt'),run_sha256=sha(run/'run.json'))
    lock=dict(created_unix=time.time(), settings=SETTINGS, original_config=CONFIG,
        sources={name:sha(ROOT/name) for name in SOURCES},
        data_manifest_sha256=sha(OLD/'data/manifest.json'), parent_checkpoints=parents,
        diagnostic_review_sha256=sha(DIAGNOSTIC/'review.json'), used_development_only=True,
        hypothesis='late introduction of shape can add utility after distance learning',
        alternatives=['probability gain without decision gain','no marginal utility','poor generalization'],
        goal_complete=False)
    write(OUT/'lock.json',lock)
    (OUT/'lock.sha256').write_text(sha(OUT/'lock.json')+'\n')
    print('R67_PLAN',sha(OUT/'lock.json'),flush=True)


def check():
    specification()
    p=read(OUT/'lock.json')
    if sha(OUT/'lock.json')!=(OUT/'lock.sha256').read_text().strip():raise ValueError('lock changed')
    for name,digest in p['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('source changed '+name)
    if p['data_manifest_sha256']!=sha(OLD/'data/manifest.json'):raise ValueError('data changed')
    if p['diagnostic_review_sha256']!=sha(DIAGNOSTIC/'review.json'):raise ValueError('prior diagnostic changed')
    return p


def build(role,seed,device='cuda'):
    p=check(); path=original_run(seed)/'resume.pt'
    if sha(path)!=p['parent_checkpoints'][str(seed)]['checkpoint_sha256']:
        raise ValueError('parent checkpoint changed')
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved['epoch_completed']!=120 or saved['step']!=46800:
        raise ValueError('expected fixed trained distance state')
    torch.manual_seed(seed)
    model=NativeGSClassifier('distance_mlp').to(device)
    model.load_state_dict(saved['state_dict'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
    optimizer.load_state_dict(saved['optimizer'])
    if role==ROLES[1]:attach_shape(model,optimizer)
    elif role!=ROLES[0]:raise ValueError('undeclared role')
    return model,optimizer,saved


def preflight():
    check();runner.configure()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    if (OUT/'preflight.json').exists():raise ValueError('preserve prior preflight')
    base,bo,saved=build(ROLES[0],11)
    shape,so,_=build(ROLES[1],11)
    c,s,a=runner.synthetic_inputs(2,23,'cuda');y=torch.tensor([3,59],device='cuda')
    base.eval();shape.eval()
    with torch.no_grad():
        initial_error=float((base(c,s,a)-shape(c,s,a)).abs().max())
    if initial_error>1e-6:raise ValueError('initial function mismatch')
    bp,sp=dict(base.named_parameters()),dict(shape.named_parameters())
    for name,parameter in bp.items():
        if not torch.equal(parameter,sp[name]):raise ValueError('common weight mismatch')
        for key,value in bo.state[parameter].items():
            other=so.state[sp[name]][key]
            if torch.is_tensor(value):
                if not torch.equal(value,other):raise ValueError('common optimizer state mismatch')
            elif value!=other:raise ValueError('common optimizer scalar mismatch')
    new=shape.position.mlp[0].shape_weight
    if new in so.state and so.state[new]:raise ValueError('new shape has old optimizer history')
    new.requires_grad_(False)
    for model,opt in ((base,bo),(shape,so)):
        torch.manual_seed(67011);model.train();opt.zero_grad(set_to_none=True)
        F.cross_entropy(model(c,s,a),y,label_smoothing=.1).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step()
    update_error=max(float((parameter-dict(shape.named_parameters())[name]).abs().max())
                     for name,parameter in base.named_parameters())
    if update_error>2e-6:raise ValueError('inactive-shape common update mismatch')
    del base,bo,shape,so,saved
    shape,so,_=build(ROLES[1],11)
    with torch.no_grad():
        shape.position.mlp[0].shape_weight.copy_(torch.sin(torch.arange(96,device='cuda')[:,None])*.005)
    equivalent=LogShapeClassifier('log_shape_distance').cuda()
    state={k:v for k,v in shape.state_dict().items() if not k.endswith('shape_weight')}
    state['position.mlp.0.weight']=torch.cat((state['position.mlp.0.weight'],shape.position.mlp[0].shape_weight),1)
    equivalent.load_state_dict(state);equivalent.eval();shape.eval()
    with torch.no_grad():formula_error=float((equivalent(c,s,a)-shape(c,s,a)).abs().max())
    if formula_error>2e-5:raise ValueError('split layer differs materially from original R66')
    del equivalent,shape,so
    shape,so,_=build(ROLES[1],11)
    data=load('fit',OLD)
    c,s,a=runner.evaluation_batch(data,0,16,'base','cuda')
    y=torch.tensor(np.array(data['labels'][:16]),device='cuda',dtype=torch.long)
    torch.cuda.reset_peak_memory_stats();shape.train();so.zero_grad(set_to_none=True)
    loss=F.cross_entropy(shape(c,s,a),y,label_smoothing=.1);loss.backward()
    gradient=float(shape.position.mlp[0].shape_weight.grad.norm())
    if not np.isfinite(gradient) or gradient<=0:raise ValueError('inactive/nonfinite new shape gradient')
    torch.nn.utils.clip_grad_norm_(shape.parameters(),1.,error_if_nonfinite=True);so.step()
    new_step=float(so.state[shape.position.mlp[0].shape_weight]['step'])
    old_step=float(so.state[shape.position.mlp[0].weight]['step'])
    if new_step!=1 or old_step!=46801:raise ValueError('incorrect separate Adam clocks')
    peak=torch.cuda.max_memory_allocated()
    if peak>CONFIG['memory_ceiling_bytes']:raise MemoryError('preflight exceeds common memory cap')
    write(OUT/'preflight.json',dict(passed=True,initial_function_max_error=initial_error,
        common_weight_and_optimizer_state_exact=True,inactive_update_max_error=update_error,
        original_R66_formula_max_error=formula_error,shape_gradient=gradient,new_adam_step=new_step,
        common_adam_step=old_step,peak_allocated=peak,gpu=torch.cuda.get_device_name(),lock_sha256=sha(OUT/'lock.json')))
    print('R67_PREFLIGHT_COMPLETE',initial_error,update_error,formula_error,gradient,flush=True)


def rate(step,total):
    return SETTINGS['final_learning_rate']+.5*(SETTINGS['learning_rate']-SETTINGS['final_learning_rate'])*(1+math.cos(math.pi*step/(total-1)))


def fit(role,seed):
    check();runner.configure()
    if not read(OUT/'preflight.json')['passed']:raise ValueError('preflight missing')
    dest=OUT/'runs'/role/str(seed)
    if (dest/'run.json').exists():raise ValueError('preserve completed fit')
    dest.mkdir(parents=True,exist_ok=True)
    model,opt,parent=build(role,seed)
    data,val=load('fit',OLD),load('val',OLD)
    tensors={k:torch.tensor(np.array(data[k]),device='cuda') for k in ('centers','covariance','appearance','labels')}
    count=len(data['labels']);steps_per_epoch=math.ceil(count/CONFIG['batch_size']);total=20*steps_per_epoch
    rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(parent['sampler_state'])
    stream=parent['stream_sha256'];base_step=parent['step']
    completed=step=0;history=[];training=[];best=float('inf');best_epoch=None
    best_state=best_logits=None;seconds_before=0.;started=time.monotonic()
    initial_common=runner.shared_digest(model);torch.cuda.reset_peak_memory_stats()
    if (dest/'resume.pt').exists():
        own=torch.load(dest/'resume.pt',map_location='cpu',weights_only=False)
        if own['lock_sha256']!=sha(OUT/'lock.json') or own['role']!=role or own['seed']!=seed:
            raise ValueError('own resume identity mismatch')
        model.load_state_dict(own['state_dict']);opt.load_state_dict(own['optimizer'])
        rng.bit_generator.state=own['sampler_state'];stream=own['stream_sha256']
        completed,step=own['epoch_completed'],own['step']
        history,training=own['validation'],own['training']
        best,best_epoch,best_state,best_logits=own['best'],own['best_epoch'],own['best_state'],own['best_logits']
        seconds_before=own['seconds']
        if step!=completed*steps_per_epoch:raise ValueError('resume budget mismatch')

    def select(epoch):
        nonlocal best,best_epoch,best_state,best_logits
        logits,metrics=runner.evaluate(model,val)
        history.append(dict(epoch=epoch,step=step,**metrics))
        if metrics['ce']<best:
            best,best_epoch=metrics['ce'],epoch;best_logits=logits
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            runner.atomic_save(dest/'best.pt',dict(role=role,seed=seed,epoch=epoch,state_dict=best_state))
            np.savez_compressed(dest/'validation.npz',logits=logits,labels=val['labels'],ids=val['ids'])
        write(dest/'history.json',dict(validation=history,training=training))
        print('VAL',role,seed,epoch,metrics['ce'],metrics['mca'],flush=True)

    def save():
        runner.atomic_save(dest/'resume.pt',dict(role=role,seed=seed,lock_sha256=sha(OUT/'lock.json'),
            epoch_completed=completed,step=step,state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
            optimizer=opt.state_dict(),sampler_state=rng.bit_generator.state,stream_sha256=stream,
            validation=history,training=training,best=best,best_epoch=best_epoch,best_state=best_state,
            best_logits=best_logits,seconds=seconds_before+time.monotonic()-started))
        write(dest/'history.json',dict(validation=history,training=training))

    try:
        if not history:select(0);save()
        for epoch in range(completed,20):
            loss_sum=0.
            for index,subset,quaternion,jitter in runner.epoch_batches(rng,count):
                stream=runner.stream_update(stream,index,subset,quaternion,jitter)
                ii=torch.as_tensor(index,device='cuda');jj=torch.as_tensor(subset,device='cuda')
                c,s,a=(tensors[k][ii[:,None],jj] for k in ('centers','covariance','appearance'))
                c,s=runner.augment(c,s,quaternion,jitter);labels=tensors['labels'][ii]
                torch.manual_seed(seed*10000000+base_step+step);model.train();opt.zero_grad(set_to_none=True)
                for group in opt.param_groups:group['lr']=rate(step,total)
                loss=F.cross_entropy(model(c,s,a),labels,label_smoothing=CONFIG['label_smoothing'])
                if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),CONFIG['clip_norm'],error_if_nonfinite=True)
                opt.step();loss_sum+=float(loss)*len(index);step+=1
            completed=epoch+1
            training.append(dict(epoch=completed,step=step,loss=loss_sum/count,stream_sha256=stream,last_gradient_norm=float(norm)))
            if completed%5==0:select(completed)
            if torch.cuda.max_memory_allocated()>CONFIG['memory_ceiling_bytes']:
                raise MemoryError('fixed memory ceiling exceeded')
            save();print('EPOCH',role,seed,completed,loss_sum/count,flush=True)
        write(dest/'run.json',dict(role=role,seed=seed,epochs=completed,steps=step,
            best_epoch=best_epoch,best_val_ce=best,initial_common_sha256=initial_common,
            parent_checkpoint_sha256=sha(original_run(seed)/'resume.pt'),
            final_stream_sha256=stream,parameters=sum(p.numel() for p in model.parameters()),
            position_parameters=sum(p.numel() for p in model.position.parameters()),
            seconds=seconds_before+time.monotonic()-started,peak_allocated=torch.cuda.max_memory_allocated(),
            hashes={name:sha(dest/name) for name in ('best.pt','resume.pt','history.json','validation.npz')},
            lock_sha256=sha(OUT/'lock.json')))
        print('FIT_COMPLETE',role,seed,best_epoch,flush=True)
    except BaseException as error:
        write(dest/f'failure_{time.time_ns()}.json',dict(type=type(error).__name__,error=str(error),step=step,epoch_completed=completed))
        raise


def selections():
    check()
    if (OUT/'selected.json').exists():raise ValueError('preserve selection')
    rows=[];pairing=[]
    for seed in SEEDS:
        histories=[];runs=[]
        for role in ROLES:
            directory=OUT/'runs'/role/str(seed);run=read(directory/'run.json')
            history=read(directory/'history.json');histories.append(history);runs.append(run)
            if run['epochs']!=20 or run['steps']!=7800 or len(history['training'])!=20:
                raise ValueError('incomplete fixed budget')
            if [r['epoch'] for r in history['validation']]!=[0,5,10,15,20]:
                raise ValueError('changed validation schedule')
            selected=min(history['validation'],key=lambda r:r['ce'])
            if selected['epoch']!=run['best_epoch']:raise ValueError('wrong selected epoch')
            for name,digest in run['hashes'].items():
                if sha(directory/name)!=digest:raise ValueError('run artifact changed')
            with np.load(directory/'validation.npz',allow_pickle=False) as z:
                independent=summarize(observations(z['logits'],z['labels']))
            if abs(independent['ce']-selected['ce'])>1e-10:raise ValueError('selection CE mismatch')
            rows.append(dict(role=role,seed=seed,best_epoch=selected['epoch'],
                validation_metrics=independent,checkpoint_sha256=sha(directory/'best.pt'),
                run_sha256=sha(directory/'run.json'),selected_unchanged_parent=selected['epoch']==0))
        if runs[0]['initial_common_sha256']!=runs[1]['initial_common_sha256']:
            raise ValueError('common initialization mismatch')
        streams=[[r['stream_sha256'] for r in h['training']] for h in histories]
        if streams[0]!=streams[1]:raise ValueError('actual augmented streams not paired')
        initial_error=max(abs(histories[0]['validation'][0][key]-histories[1]['validation'][0][key])
                          for key in ('ce','oa','mca'))
        if initial_error>1e-7:raise ValueError('warm-start initial validation differs')
        pairing.append(dict(seed=seed,all20_stream_hashes_equal=True,initial_validation_metric_max_error=initial_error))
    write(OUT/'selected.json',dict(records=rows,pairing=pairing,
        all_six_selected_before_development_evaluation=True,lock_sha256=sha(OUT/'lock.json')))
    print('SELECTIONS_COMPLETE',[(r['role'],r['seed'],r['best_epoch']) for r in rows],flush=True)


def export(seed):
    check();runner.configure();selected=read(OUT/'selected.json')
    if len(selected['records'])!=6:raise ValueError('finish all six selections first')
    data,val=load('dev_eval',OLD),load('val',OLD)
    for role in ROLES:
        entry=next(r for r in selected['records'] if r['role']==role and r['seed']==seed)
        run=OUT/'runs'/role/str(seed);dest=OUT/'evaluation'/role/str(seed)
        if dest.exists():raise ValueError('preserve previous export')
        if sha(run/'best.pt')!=entry['checkpoint_sha256']:raise ValueError('selected model changed')
        model=make_model(role).cuda()
        model.load_state_dict(torch.load(run/'best.pt',map_location='cpu',weights_only=True)['state_dict'])
        logits,_=runner.evaluate(model,val)
        with np.load(run/'validation.npz',allow_pickle=False) as z:
            error=float(abs(logits-z['logits']).max())
            changes=int((logits.argmax(1)!=z['logits'].argmax(1)).sum())
        if error>2e-5 or changes:raise ValueError('selected validation replay mismatch')
        dest.mkdir(parents=True);rows=[]
        for variant in VARIANTS:
            logits,original=runner.evaluate(model,data,variant)
            independent=summarize(observations(logits,data['labels']))
            if any(abs(original[k]-independent[k])>1e-10 for k in ('ce','oa','mca')):
                raise ValueError('independent metrics disagree')
            path=dest/(variant+'.npz')
            np.savez_compressed(path,logits=logits,labels=data['labels'],ids=data['ids'])
            rows.append(dict(role=role,seed=seed,variant=variant,metrics=independent,prediction_sha256=sha(path)))
            print('EXPORT',role,seed,variant,independent['mca'],flush=True)
        write(dest/'record.json',dict(records=rows,checkpoint_sha256=entry['checkpoint_sha256'],
            selection_sha256=sha(OUT/'selected.json'),data_sha256=sha(OLD/'data/manifest.json'),
            validation_replay=dict(maximum_logit_error=error,prediction_changes=changes)))
        del model;torch.cuda.empty_cache()


def report():
    check()
    if (OUT/'result.json').exists():raise ValueError('preserve completed report')
    selected=read(OUT/'selected.json');rows=[];predictions={}
    worst=0.
    for entry in selected['records']:
        role,seed=entry['role'],entry['seed'];dest=OUT/'evaluation'/role/str(seed)
        meta=read(dest/'record.json')
        if meta['checkpoint_sha256']!=entry['checkpoint_sha256'] or meta['selection_sha256']!=sha(OUT/'selected.json'):
            raise ValueError('export identity mismatch')
        for row in meta['records']:
            path=dest/(row['variant']+'.npz')
            if sha(path)!=row['prediction_sha256']:raise ValueError('prediction changed')
            with np.load(path,allow_pickle=False) as z:
                expected=runner.metrics(z['logits'],z['labels'])
                predictions[role,seed,row['variant']]={k:z[k] for k in ('logits','labels','ids')}
            worst=max(worst,max(abs(expected[k]-row['metrics'][k]) for k in ('ce','oa','mca')))
            rows.append(row)
    if len(rows)!=36 or worst>1e-10:raise ValueError('incomplete or inconsistent evaluation')
    lookup={(r['role'],r['seed'],r['variant']):r for r in rows}
    indices=np.random.default_rng(67001).integers(0,60,(10000,60));contrasts=[]
    for variant in VARIANTS:
        classes=[];seed_gains=[];secondary=[]
        for seed in SEEDS:
            d,g=(lookup[role,seed,variant]['metrics'] for role in ROLES)
            dp,gp=(predictions[role,seed,variant] for role in ROLES)
            if any(not np.array_equal(dp[key],gp[key]) for key in ('labels','ids')):
                raise ValueError('unpaired evaluation')
            classes.append([b['accuracy']-a['accuracy'] for a,b in zip(d['classes'],g['classes'])])
            seed_gains.append(100*(g['mca']-d['mca']))
            secondary.append(dict(seed=seed,ce=g['ce']-d['ce'],macro_ce=g['macro_ce']-d['macro_ce'],
                oa_pp=100*(g['oa']-d['oa']),domains={name:{key:g['domains'][name][key]-d['domains'][name][key]
                    for key in ('ce','macro_ce','oa','mca')} for name in ('MACGS','ModelNet')}))
        average=np.mean(classes,axis=0);interval=np.quantile(100*average[indices].mean(1),[.025,.975])
        mean=float(np.mean(seed_gains))
        contrasts.append(dict(variant=variant,paired_seed_gains_pp=seed_gains,mean_gain_pp=mean,
            paired_class_bootstrap_95_pp=interval.tolist(),secondary=secondary,
            development_lead=bool(mean>=.5 and min(seed_gains)>0 and interval[0]>0)))
    permutation=[]
    for role in ROLES:
        for seed in SEEDS:
            a,b=(predictions[role,seed,v]['logits'] for v in ('base','permuted'))
            permutation.append(dict(role=role,seed=seed,maximum_logit_error=float(abs(a-b).max()),
                prediction_changes=int((a.argmax(1)!=b.argmax(1)).sum())))
    write(OUT/'result.json',dict(records=rows,comparisons=contrasts,permutation=permutation,
        independent_metric_max_error=worst,primary_development_lead=contrasts[0]['development_lead'],
        lock_sha256=sha(OUT/'lock.json'),scientific_review_pending=True,visual_review_pending=True,
        fresh_confirmation=False,goal_complete=False))
    lines=['# R67: late introduction of Gaussian shape','',
        'Matched terminal120 distance parents; two roles each get the same20 additional epochs. '
        'Used full60-class development data. This does not replace the original R66 result.','',
        '| Method | Base mCA seeds11/22/33 % | Mean % | CE | Class-average CE | Selected added epochs |',
        '|---|---|---:|---:|---:|---|']
    for role in ROLES:
        m=[lookup[role,s,'base']['metrics'] for s in SEEDS]
        epochs=[next(r['best_epoch'] for r in selected['records'] if r['role']==role and r['seed']==s) for s in SEEDS]
        lines.append('| '+role+' | '+' / '.join(f"{100*v['mca']:.3f}" for v in m)+
            f" | {100*np.mean([v['mca'] for v in m]):.3f} | {np.mean([v['ce'] for v in m]):.5f} | {np.mean([v['macro_ce'] for v in m]):.5f} | {epochs} |")
    lines+=['','| Condition | Paired mCA changes pp | Mean pp | Paired-class95% pp | Lead |',
        '|---|---|---:|---|---|']
    for r in contrasts:
        lo,hi=r['paired_class_bootstrap_95_pp']
        lines.append('| '+r['variant']+' | '+' / '.join(f'{v:+.3f}' for v in r['paired_seed_gains_pp'])+
            f" | {r['mean_gain_pp']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {r['development_lead']} |")
    lines+=['','All classes, sources and secondary scores are retained in result.json. '
        'Bootstrap is over classes after seed averaging, not a seed-population interval. '
        'Source categories are disjoint. Stage0 selection means no accepted adaptation. '
        'Scientific and actual visual review pending.','','![Comparison](comparison.png)','']
    (OUT/'report.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for y,r in enumerate(contrasts):
        axes[0].plot(r['paired_class_bootstrap_95_pp'],[y,y],color='black')
        axes[0].scatter(r['paired_seed_gains_pp'],y+np.array([-.1,0,.1]),color='tab:blue')
        axes[0].scatter([r['mean_gain_pp']],[y],color='black',marker='D')
    axes[0].axvline(0,color='grey');axes[0].set(yticks=range(len(contrasts)),yticklabels=list(VARIANTS),
        xlabel='Shape after distance minus distance continuation (pp)',title='Paired mCA changes');axes[0].invert_yaxis()
    for role in ROLES:
        histories=[read(OUT/'runs'/role/str(seed)/'history.json')['validation'] for seed in SEEDS]
        axes[1].plot([v['epoch'] for v in histories[0]],np.mean([[v['ce'] for v in h] for h in histories],axis=0),label=role)
    axes[1].set(xlabel='Additional epoch',ylabel='Validation CE',title='Fixed additional budget');axes[1].legend(fontsize=8)
    fig.suptitle('R67 | used development data | same pretrained distance parents')
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig)
    print('R67_REPORT_COMPLETE',contrasts[0]['development_lead'],flush=True)


def launch(stage,job,gpu):
    tag=stage if job is None else stage+'_'+'_'.join(map(str,job));key=tag+'_'+str(time.time_ns())
    command=[sys.executable,'-m',MODULE,stage]
    if stage=='train':command+=['--role',job[0],'--seed',str(job[1])]
    elif stage=='export':command+=['--seed',str(job[0])]
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_DEVICE_ORDER='PCI_BUS_ID',
        CUDA_VISIBLE_DEVICES='' if gpu is None else gpu['uuid'],MPLCONFIGDIR='/tmp/grope-r67-matplotlib')
    stream=(OUT/(key+'.log')).open('x')
    process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    meta=dict(tag=key,stage=stage,job=job,pid=process.pid,gpu=gpu,command=command,started_unix=time.time(),
              start_ticks=(Path('/proc')/str(process.pid)/'stat').read_text().split()[21])
    scheduler.ACTIVE[key]=dict(process=process,stream=stream,meta=meta)
    print('LAUNCHED',tag,process.pid,None if gpu is None else gpu['index'],flush=True)


def queue():
    check()
    if (OUT/'queue_status.json').exists():raise ValueError('inspect original controller before restart')
    scheduler.OUT=OUT
    scheduler.INFO.update(pid=os.getpid(),started_unix=time.time(),lock_sha256=sha(OUT/'lock.json'),
                          start_ticks=Path('/proc/self/stat').read_text().split()[21])
    def stop(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    phases=[('preflight',[None]),('train',[(role,seed) for seed in SEEDS for role in ROLES]),
            ('selections',[None]),('export',[(seed,) for seed in SEEDS]),('report',[None])]
    try:
        for stage,tasks in phases:
            pending=list(tasks);previous=set()
            while pending or scheduler.ACTIVE:
                retries=scheduler.collect()
                if retries:raise ValueError('unexpected retry code in fixed short run')
                if not pending and not scheduler.ACTIVE:break
                if stage in ('selections','report'):
                    if pending:launch(stage,pending.pop(0),None)
                    scheduler.record('running',stage=stage,pending=pending)
                else:
                    rows=scheduler.occupancy();available={r['uuid'] for r in rows if r['free']}
                    stable=available&previous;owned={v['meta']['gpu']['uuid'] for v in scheduler.ACTIVE.values()}
                    for gpu in rows:
                        if not pending or len(scheduler.ACTIVE)>=4:break
                        if gpu['uuid'] in stable-owned:
                            launch(stage,pending.pop(0),gpu);owned.add(gpu['uuid'])
                    previous=available-owned
                    scheduler.record('running' if scheduler.ACTIVE else 'waiting_for_gpu',stage=stage,pending=pending,occupancy=rows)
                time.sleep(5)
        scheduler.record('complete',task_execution_completed=True,scientific_review_pending=True,goal_complete=False)
    except BaseException as error:
        scheduler.stop_owned();scheduler.record('stopped',error_type=type(error).__name__,error=str(error),goal_complete=False)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('plan','preflight','train','selections','export','report','queue'))
    parser.add_argument('--role',choices=ROLES);parser.add_argument('--seed',type=int);args=parser.parse_args()
    if args.action=='train':fit(args.role,args.seed)
    elif args.action=='export':export(args.seed)
    else:globals()[args.action]()
