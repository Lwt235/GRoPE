"""Nine fixed-budget fits that remove columns from the reviewed R68 relation."""
import argparse
import os
from pathlib import Path
import shutil
import signal
import time
import numpy as np
import torch
from experiments import round68_fixed_region_run as engine
from experiments.round65_common import ROOT,SEEDS,read,write,sha
from experiments.round68_fixed_region_data import load_cache
from models.round69_compact_relation import CompactMatcher,ROLES

PARENT=engine.OUT
PARENT_CHECK=engine.check
OUT=ROOT/'results/round69_compact_relation/attempt1'
MODULE='experiments.round69_compact_relation'
CONDITIONS=('base','sparse256')
CONTROLS=('gaussian_relation','distance_mlp','direct_mlp')


def parent_check():
    previous=engine.OUT;engine.OUT=PARENT
    try:PARENT_CHECK()
    finally:engine.OUT=previous


def plan():
    parent_check()
    if OUT.exists():raise ValueError('preserve previous attempt')
    review=read(PARENT/'review.json')
    if not review['scientific_review_complete']:raise ValueError('review parent results first')
    names=('docs/round69_compact_relation_plan.md','models/round69_compact_relation.py',
           'experiments/round69_compact_relation.py','tests/test_round69_compact_relation.py')
    OUT.mkdir(parents=True)
    shutil.copy2(PARENT/'data_review.json',OUT/'data_review.json')
    write(OUT/'lock.json',dict(created_unix=time.time(),sources={n:sha(ROOT/n) for n in names},
        parent_lock_sha256=sha(PARENT/'lock.json'),parent_result_sha256=sha(PARENT/'result.json'),
        parent_review_sha256=sha(PARENT/'review.json'),parent_selection_sha256=sha(PARENT/'selected.json'),
        data_review_sha256=sha(OUT/'data_review.json'),settings=engine.SETTINGS,
        fixed_controls=CONTROLS,comparison_count=18,new_fits=9,
        hypothesis='Displacement-shape coupling retains the observed Gaussian relation utility without absolute spectra',
        fresh_confirmation=False,goal_complete=False))
    (OUT/'lock.sha256').write_text(sha(OUT/'lock.json')+'\n')
    print('R69_LOCK',sha(OUT/'lock.json'),flush=True)


def check():
    parent_check();p=read(OUT/'lock.json')
    if sha(OUT/'lock.json')!=(OUT/'lock.sha256').read_text().strip():raise ValueError('lock changed')
    for name,digest in p['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('source changed: '+name)
    for name,key in [('lock.json','parent_lock_sha256'),('result.json','parent_result_sha256'),
                     ('review.json','parent_review_sha256'),('selected.json','parent_selection_sha256')]:
        if sha(PARENT/name)!=p[key]:raise ValueError('reference changed: '+name)
    if sha(OUT/'data_review.json')!=p['data_review_sha256']:raise ValueError('data review changed')
    return p


def selections():
    check();val=load_cache('val');records=[];pairing=[]
    if (OUT/'selected.json').exists():raise ValueError('preserve selection')
    for seed in SEEDS:
        reference=read(PARENT/'runs/gaussian_relation'/str(seed)/'run.json')
        ref_history=read(PARENT/'runs/gaussian_relation'/str(seed)/'history.json')
        ref_streams=[v['stream_sha256'] for v in ref_history['training']]
        for role in ROLES:
            dest=OUT/'runs'/role/str(seed);run=read(dest/'run.json');history=read(dest/'history.json')
            if run['epochs']!=60 or run['steps']!=1080 or len(history['training'])!=60:raise ValueError('fit budget')
            if run['initial_shared_sha256']!=reference['initial_shared_sha256']:raise ValueError('initialization changed')
            if [v['stream_sha256'] for v in history['training']]!=ref_streams:raise ValueError('actual streams differ')
            for name,digest in run['hashes'].items():
                if sha(dest/name)!=digest:raise ValueError('fit artifact changed')
            best=min(history['validation'],key=lambda v:v['selection_ce'])
            if best['epoch']!=run['selected_epoch']:raise ValueError('selection rule')
            with np.load(dest/'validation.npz',allow_pickle=False) as z:
                if not np.array_equal(z['ids'],val['ids']):raise ValueError('validation membership')
                values=engine.query_values(z['logits'],val)
            ce=[]
            for d in (0,1):
                valid=values['valid'][:,d]
                if valid.any():ce.append(float(values['ce'][:,d][valid].mean()))
            error=abs(np.mean(ce)-run['best_val_ce'])
            if error>1e-10:raise ValueError('selection metric')
            records.append(dict(role=role,seed=seed,selected_epoch=best['epoch'],
                checkpoint_sha256=sha(dest/'best.pt'),run_sha256=sha(dest/'run.json'),validation_ce_error=error))
        pairing.append(dict(seed=seed,all_three_common_initializations_equal_parent=True,
                            all60_actual_streams_equal_parent=True))
    write(OUT/'selected.json',dict(records=records,pairing=pairing,all_nine_selected_before_eval=True,
        lock_sha256=sha(OUT/'lock.json')))
    print('R69_SELECTIONS_COMPLETE',flush=True)


def export(seed):
    check();engine.configure();selected=read(OUT/'selected.json');val=load_cache('val')
    if len(selected['records'])!=9 or seed not in SEEDS:raise ValueError('all selections required')
    caches={condition:load_cache('eval_'+condition) for condition in CONDITIONS}
    for role in ROLES:
        choice=next(v for v in selected['records'] if v['role']==role and v['seed']==seed)
        run=OUT/'runs'/role/str(seed);dest=OUT/'evaluation'/role/str(seed)
        if dest.exists():raise ValueError('preserve prior export')
        if sha(run/'best.pt')!=choice['checkpoint_sha256']:raise ValueError('checkpoint identity')
        model=CompactMatcher(role).cuda()
        model.load_state_dict(torch.load(run/'best.pt',map_location='cpu',weights_only=True)['state_dict'])
        logits,_=engine.evaluate(model,val)
        with np.load(run/'validation.npz',allow_pickle=False) as z:
            error=float(abs(logits-z['logits']).max());changes=int((logits.argmax(-1)!=z['logits'].argmax(-1)).sum())
        if error>2e-4 or changes:raise ValueError('selected replay changes task decisions')
        dest.mkdir(parents=True);records=[]
        for condition,cache in caches.items():
            logits,independent=engine.evaluate(model,cache);metrics=engine.grouped_metrics(logits,cache)
            delta=max(abs(independent[k]-metrics['micro'][k]) for k in ('region_recall','center_pck','soft_epe','ce'))
            if delta>1e-10:raise ValueError('independent evaluation metrics')
            path=dest/(condition+'.npz');np.savez_compressed(path,logits=logits,ids=cache['ids'])
            records.append(dict(role=role,seed=seed,condition=condition,metrics=metrics,
                prediction_sha256=sha(path),independent_metric_error=delta))
        write(dest/'record.json',dict(records=records,checkpoint_sha256=choice['checkpoint_sha256'],
            selection_sha256=sha(OUT/'selected.json'),manifest_sha256=sha(PARENT/'data/manifest.json'),
            validation_replay=dict(maximum_logit_error=error,prediction_changes=changes)))
        print('R69_EXPORT',role,seed,flush=True);del model;torch.cuda.empty_cache()


def report():
    check()
    if (OUT/'result.json').exists():raise ValueError('preserve prior report')
    rows=list(read(PARENT/'result.json')['records']);selected=read(OUT/'selected.json')
    for row in rows:
        path=PARENT/'evaluation'/row['role']/str(row['seed'])/(row['condition']+'.npz')
        if sha(path)!=row['prediction_sha256']:raise ValueError('reference prediction changed')
    for choice in selected['records']:
        dest=OUT/'evaluation'/choice['role']/str(choice['seed']);meta=read(dest/'record.json')
        if meta['checkpoint_sha256']!=choice['checkpoint_sha256'] or meta['selection_sha256']!=sha(OUT/'selected.json'):
            raise ValueError('export identity')
        for row in meta['records']:
            if sha(dest/(row['condition']+'.npz'))!=row['prediction_sha256']:raise ValueError('prediction changed')
            rows.append(row)
    if len(rows)!=48:raise ValueError('expected30 reused and18 new groups')
    lookup={(r['role'],r['seed'],r['condition']):r['metrics'] for r in rows};comparisons=[]
    for role in ROLES:
        for condition in CONDITIONS:
            for control in CONTROLS:
                pairs=[];scene_deltas=[];scenes=None
                for seed in SEEDS:
                    a,b=lookup[role,seed,condition],lookup[control,seed,condition]
                    pair={key:a['macro'][key]-b['macro'][key] for key in ('region_recall','center_pck','soft_epe','ce')}
                    pair['region_recall']*=100;pair['center_pck']*=100;pairs.append(dict(seed=seed,**pair))
                    scenes=list(a['by_scene']);scene_deltas.append([100*(a['by_scene'][s]['region_recall']-b['by_scene'][s]['region_recall']) for s in scenes])
                average=np.mean(scene_deltas,axis=0)
                draw=np.random.default_rng(69001).integers(0,len(scenes),(10000,len(scenes)))
                interval=np.quantile(average[draw].mean(1),[.025,.975])
                comparisons.append(dict(role=role,condition=condition,control=control,paired=pairs,
                    mean={k:float(np.mean([v[k] for v in pairs])) for k in pairs[0] if k!='seed'},
                    scene_mean_region_gains_pp=dict(zip(scenes,average.tolist())),paired_scene_bootstrap_95_pp=interval.tolist()))
    qualification={}
    for role in ('metric3','relative5'):
        chosen=[r for r in comparisons if r['role']==role]
        qualification[role]=bool(all(
            (r['mean']['region_recall']>=-1 and r['mean']['center_pck']>=-1) if r['control']=='gaussian_relation'
            else (r['mean']['region_recall']>0 and all(v['region_recall']>0 for v in r['paired'])) for r in chosen))
    preferred=next((role for role in ('metric3','relative5') if qualification[role]),'gaussian_relation')
    result=dict(records=rows,comparisons=comparisons,compact_descriptive_qualification=qualification,
        preferred_development_candidate=preferred,lock_sha256=sha(OUT/'lock.json'),
        independent_metric_max_error=max(r['independent_metric_error'] for r in rows),
        reused_groups=30,new_groups=18,scientific_review_pending=True,visual_review_pending=True,
        fresh_confirmation=False,goal_complete=False)
    write(OUT/'result.json',result)
    lines=['# R69 fixed relation-column decomposition','',
        'Nine new paired fits, thirty identity-verified reused R68 evaluation groups. Used development only.',
        'Descriptive compact candidate: '+preferred+'. This is not a formal noninferiority conclusion.','',
        '| Condition | Method | Region R@1 % | Center PCK % | Soft EPE | CE |','|---|---|---:|---:|---:|---:|']
    allroles=('distance_mlp','direct_mlp','normal_scale','gaussian_relation','shuffled_gaussian')+ROLES
    for condition in CONDITIONS:
        for role in allroles:
            values=[lookup[role,s,condition]['macro'] for s in SEEDS]
            lines.append(f'| {condition} | {role} | '+' | '.join(f'{np.mean([v[k] for v in values])*(100 if k in ("region_recall","center_pck") else 1):.4f}' for k in ('region_recall','center_pck','soft_epe','ce'))+' |')
    lines+=['','| Condition | Candidate minus control | Paired region gains pp | Mean pp | Scene95% pp | Center pp |','|---|---|---|---:|---|---:|']
    for r in comparisons:
        lo,hi=r['paired_scene_bootstrap_95_pp']
        lines.append(f"| {r['condition']} | {r['role']} - {r['control']} | "+' / '.join(f"{v['region_recall']:+.3f}" for v in r['paired'])+
                     f" | {r['mean']['region_recall']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {r['mean']['center_pck']:+.3f} |")
    lines+=['','No search or additional fitting is authorized by a failed simplification. Retain full11 if required. '
        'Known original region membership under thinning; unchanged20422 valid out of61440 queries. '
        'Scene intervals condition on the selected three seeds. Scientific and visual review pending.','','![Comparison](comparison.png)','']
    (OUT/'report.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ax,condition in zip(axes,CONDITIONS):
        selected=[r for r in comparisons if r['condition']==condition]
        for y,r in enumerate(selected):
            ax.plot(r['paired_scene_bootstrap_95_pp'],[y,y],color='black')
            ax.scatter([v['region_recall'] for v in r['paired']],y+np.array([-.12,0,.12]),s=15)
        ax.axvline(0,color='grey');ax.set(yticks=range(len(selected)),yticklabels=[r['role']+' vs '+r['control'] for r in selected],xlabel='Region R@1 change (pp)',title=condition);ax.invert_yaxis()
    fig.suptitle('R69 | relative inputs vs absolute shape descriptors | used development')
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig)
    print('R69_REPORT_COMPLETE',preferred,flush=True)


def queue():
    check();scheduler=engine.scheduler
    if (OUT/'queue_status.json').exists():raise ValueError('inspect existing controller')
    scheduler.OUT=OUT;scheduler.INFO.update(pid=os.getpid(),started_unix=time.time(),
        start_ticks=Path('/proc/self/stat').read_text().split()[21],lock_sha256=sha(OUT/'lock.json'))
    def stop(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def permitted():
        peer=read(ROOT/'results/round67_training_path/attempt1/queue_status.json')
        return peer['state']=='complete' or (peer['state']=='running' and peer.get('stage')=='train' and not peer.get('pending'))
    tasks=[('preflight',[None]),('train',[(ROLES[(i+j)%3],s) for j in range(3) for i,s in enumerate(SEEDS)]),
           ('selections',[None]),('export',[(s,) for s in SEEDS]),('report',[None])]
    try:
        for stage,jobs in tasks:
            pending=list(jobs);previous=set()
            while pending or scheduler.ACTIVE:
                if scheduler.collect():raise ValueError('unexpected retry')
                if not pending and not scheduler.ACTIVE:break
                if stage in ('selections','report'):
                    if pending:engine.launch(stage,pending.pop(0),None)
                    scheduler.record('running',stage=stage,pending=pending)
                else:
                    if not permitted():
                        previous=set();scheduler.record('waiting_for_peer_gpu_phase',stage=stage,pending=pending);time.sleep(5);continue
                    rows=scheduler.occupancy();available={v['uuid'] for v in rows if v['free']};stable=available&previous
                    owned={v['meta']['gpu']['uuid'] for v in scheduler.ACTIVE.values()}
                    for gpu in rows:
                        if not pending or len(scheduler.ACTIVE)>=2 or not permitted():break
                        if gpu['uuid'] in stable-owned:engine.launch(stage,pending.pop(0),gpu);owned.add(gpu['uuid'])
                    previous=available-owned
                    scheduler.record('running' if scheduler.ACTIVE else 'waiting_for_gpu',stage=stage,pending=pending,occupancy=rows)
                time.sleep(5)
        scheduler.record('complete',task_execution_completed=True,scientific_review_pending=True,goal_complete=False)
    except BaseException as error:
        scheduler.stop_owned();scheduler.record('stopped',error_type=type(error).__name__,error=str(error),goal_complete=False);raise


def bind():
    engine.OUT=OUT;engine.ROLES=ROLES;engine.FixedRegionMatcher=CompactMatcher;engine.check=check;engine.MODULE=MODULE
    engine.SETTINGS=dict(engine.SETTINGS,roles=ROLES)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('plan','preflight','train','selections','export','report','queue'))
    parser.add_argument('--role',choices=ROLES);parser.add_argument('--seed',type=int);args=parser.parse_args();bind()
    if args.action=='train':engine.train(args.role,args.seed)
    elif args.action=='preflight':engine.preflight()
    elif args.action=='export':export(args.seed)
    else:globals()[args.action]()
