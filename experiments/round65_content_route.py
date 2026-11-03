"""Matched retraining ablation: covariance in content versus explicit PE."""
import argparse
import time
import numpy as np  # This host requires NumPy before Torch.
import torch
from experiments import round65_pilot as runner
from experiments import round65_cohort_pilot as scheduler
from experiments.round65_common import ROOT, OUT, CONFIG, SEEDS, VARIANTS, checked, read, write, sha
from experiments.round65_run import shared_digest
from experiments.round65_audit import bootstrap, independent_metrics
from models.round65_content_route import ContentRouteClassifier, ROLES

DEST = ROOT/'results/round65_actual_gs/content_route1'
PREVIOUS = ROOT/'results/round65_actual_gs/pilot3'
MODULE = 'experiments.round65_content_route'
SOURCES = ('experiments/round65_content_route.py', 'models/round65_content_route.py',
           'tests/test_round65_content_route.py', 'experiments/round65_pilot.py',
           'experiments/round65_cohort_pilot.py')


def plan():
    checked()
    if (DEST/'plan.json').exists():
        raise ValueError('preserve existing content-route plan')
    old = read(PREVIOUS/'plan.json')
    if not read(PREVIOUS/'review.json')['actual_plot_viewed']:
        raise ValueError('complete previous result interpretation first')
    records = ('plan.json', 'result.json', 'review.json', 'selected.json',
               'fit_data.json', 'val_data.json', 'dev_eval_data.json')
    prior = {name: sha(PREVIOUS/name) for name in records}
    write(DEST/'plan.json', dict(created_unix=time.time(),
        sources={name: sha(ROOT/name) for name in SOURCES}, parent_lock_sha256=sha(OUT/'lock.json'),
        previous_record_hashes=prior, rows=old['rows'], classes=old['classes'], counts=old['counts'],
        roles=ROLES, seeds=SEEDS, variants=VARIANTS, config=CONFIG,
        hypothesis='shape in the common token content may make covariance relations redundant; removing that route may reveal where shape information helps',
        competing_explanations=['shape is not useful for this classification cohort',
            'shape through any ordinary learned PE suffices, without special Gaussian invariants',
            'Gaussian pair descriptors are less useful than center distance even with shape removed from content'],
        intervention='remove six log-covariance content channels from training and evaluation; retain center3/appearance4 and every explicit PE unchanged',
        initialization='construct each original13D model then retain stem columns0,1,2,9,10,11,12; all other shared and PE tensors, and RNG state, match its seed/role baseline',
        capacity='576 fewer active parameters in every ablated role; no unused padding; identical reduced stem across the three roles',
        fixed_comparison='three roles x three seeds x two content routes; nine prior fits reused, nine new fits trained from scratch;120epochs/16200steps and all data/augmentations/selection fixed',
        primary_metric='base512 development mean-class accuracy; full-minus-removed within role and Gaussian-minus-control within each route',
        secondary='all six original conditions, plus change in each Gaussian-control gap between content routes',
        decisions=dict(routing='if Gaussian advantage appears only after content removal, treat it as information routing; require directMLP comparison before attributing special structure',
            no_shape_value='if removing shape changes little for all roles and distance remains competitive, do not add shape modules for this task',
            negative='if full Gaussian relations still lose, preserve that failure and reconsider the descriptor rather than increasing budgets',
            inconclusive='retain uncertainty and do not select a new architecture from secondary conditions'),
        uncertainty='paired class bootstrap10000/seed65001 after averaging paired seeds; report all seed signs; descriptive used-development diagnostic, no universal pass gate',
        limitations=['same25MACGS classes,264evaluation objects already inspected',
            'distance without shape has less geometry information; a gain over it alone is not a matched-information PE superiority claim',
            'directMLP retains the same shape information in PE and remains the strong structural control',
            'class-size and nonidentical physical-instance grouping limitations persist',
            'retraining changes learned adaptation, unlike inference-only destructive intervention'],
        fresh_confirmation=False, parent_experiment_changed=False, full_controller_resumed=False, goal_complete=False))
    print('CONTENT_ROUTE_PLAN', old['counts'], ROLES, SEEDS, flush=True)


def specification():
    checked(); p = read(DEST/'plan.json')
    if p['parent_lock_sha256'] != sha(OUT/'lock.json'):
        raise ValueError('parent experiment changed')
    for name, digest in p['sources'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('content-route source changed '+name)
    for name, digest in p['previous_record_hashes'].items():
        if sha(PREVIOUS/name) != digest:
            raise ValueError('completed baseline record changed '+name)
    if (p['config'] != CONFIG or p['roles'] != list(ROLES) or p['seeds'] != list(SEEDS)
            or p['variants'] != list(VARIANTS) or p['rows'] != read(PREVIOUS/'plan.json')['rows']):
        raise ValueError('fixed comparison changed')
    return p


def reuse_split(split):
    specification()
    if split not in ('fit', 'val', 'dev_eval'):
        raise ValueError(split)
    if split == 'dev_eval' and not (DEST/'selected.json').exists():
        raise ValueError('select all nine fits before evaluation')
    target = DEST/(split+'_data.json')
    if target.exists():
        raise ValueError('preserve cache reference')
    previous = PREVIOUS/target.name; d = read(previous)
    if sha(d['path']) != d['sha256']:
        raise ValueError('original cache changed')
    write(target, dict(d, plan_sha256=sha(DEST/'plan.json'),
        reused_from=str(previous), original_record_sha256=sha(previous)))
    print('REUSED_CACHE', split, len(d['records']), flush=True)


def bind_runner():
    runner.DEST = DEST; runner.ROLES = ROLES; runner.SEEDS = SEEDS
    runner.EPOCHS = CONFIG['epochs']; runner.VARIANTS = VARIANTS
    runner.specification = specification; runner.prepare_split = reuse_split
    runner.NativeGSClassifier = ContentRouteClassifier


def seal():
    bind_runner(); runner.seal()
    selected = read(DEST/'selected.json')
    if len(selected['records']) != 9:
        raise ValueError('incomplete ablation selection')
    selected.pop('all_ten_selected_before_development_evaluation')
    selected['all_nine_selected_before_development_evaluation'] = True
    for entry in selected['records']:
        role, seed = entry['role'], entry['seed']
        current = DEST/'runs'/role/str(seed); previous = PREVIOUS/'runs'/role/str(seed)
        old_run = read(previous/'run.json')
        baseline = next(r for r in read(PREVIOUS/'selected.json')['records'] if (r['role'],r['seed']) == (role,seed))
        if sha(previous/'run.json') != baseline['run_sha256'] or sha(previous/'history.json') != old_run['hashes']['history.json']:
            raise ValueError('baseline run/history changed')
        streams = lambda d: [r['stream_sha256'] for r in read(d/'history.json')['training']]
        if streams(current) != streams(previous):
            raise ValueError('unpaired original training inputs')
        torch.manual_seed(seed)
        if shared_digest(ContentRouteClassifier(role)) != read(current/'run.json')['initial_shared_sha256']:
            raise ValueError('reduced-stem initialization mismatch')
    selected['all_original_120_epoch_streams_matched'] = True
    write(DEST/'selected.json', selected)


def contrasts(metrics, classes):
    """Preserve every paired sign; positive means the named left side wins."""
    expected = {(content,role,seed,variant) for content in ('full','removed')
                for role in ROLES for seed in SEEDS for variant in VARIANTS}
    if set(metrics) != expected:
        raise ValueError('incomplete108-group factorial comparison')
    result = []
    def values(content, role, variant):
        return 100*np.array([[metrics[content,role,s,variant]['class_accuracy'][c]
                              for c in classes] for s in SEEDS])
    def add(kind, label, variant, delta):
        paired = delta.mean(1); interval = bootstrap(delta)
        result.append(dict(kind=kind,label=label,variant=variant,paired_seed_gains_pp=paired.tolist(),
            mean_gain_pp=float(paired.mean()), paired_class_bootstrap_95_pp=interval))
    for variant in VARIANTS:
        for role in ROLES:
            add('content_full_minus_removed', role, variant,
                values('full',role,variant)-values('removed',role,variant))
        for control in ROLES[1:]:
            gaps = {}
            for content in ('full','removed'):
                gaps[content] = values(content,ROLES[0],variant)-values(content,control,variant)
                add('gaussian_minus_control_'+content,control,variant,gaps[content])
            add('gaussian_gap_removed_minus_full',control,variant,gaps['removed']-gaps['full'])
    return result


def report():
    p = specification(); rows = []
    labels = np.array([r['taxonomy_id'] for r in p['rows']['dev_eval']])
    ids = np.array([r['file_path'] for r in p['rows']['dev_eval']])
    for seed in SEEDS:
        d = read(DEST/('evaluation_'+str(seed)+'.json'))
        if d['selection_sha256'] != sha(DEST/'selected.json') or d['data_sha256'] != read(DEST/'dev_eval_data.json')['sha256']:
            raise ValueError('evaluation provenance changed')
        rows.extend(d['rows'])
    expected = {(r,s,v) for r in ROLES for s in SEEDS for v in VARIANTS}
    if len(rows) != 54 or {(r['role'],r['seed'],r['variant']) for r in rows} != expected:
        raise ValueError('incomplete54-group export')
    baseline = [r for r in read(PREVIOUS/'result.json')['rows'] if r['role'] in ROLES]
    metrics = {}; logits = {}
    for content, directory, source in [('full',PREVIOUS,baseline),('removed',DEST,rows)]:
        for row in source:
            path = directory/'evaluation'/row['role']/str(row['seed'])/(row['variant']+'.npz')
            if sha(path) != row['sha256']:
                raise ValueError('prediction artifact changed')
            with np.load(path,allow_pickle=False) as z:
                if not np.array_equal(z['labels'],labels) or not np.array_equal(z['ids'],ids):
                    raise ValueError('comparison membership/order mismatch')
                m = independent_metrics(z['logits'],labels)
                if any(abs(m[k]-row['metrics'][k])>1e-8 for k in ('ce','oa','mca')):
                    raise ValueError('independent metric discrepancy')
                key = content,row['role'],row['seed'],row['variant']; metrics[key] = m
                if content == 'removed' and row['variant'] in ('base','permuted'):
                    logits[key] = z['logits']
    differences = contrasts(metrics,p['classes'])
    permutation = []
    for role in ROLES:
        for seed in SEEDS:
            a,b = (logits['removed',role,seed,v] for v in ('base','permuted'))
            permutation.append(dict(role=role,seed=seed,maximum_logit_error=float(abs(a-b).max()),
                prediction_disagreements=int((a.argmax(1)!=b.argmax(1)).sum())))
    lines = ['# Native-GS content-routing ablation','',
        'Covariance is removed only from common token content during retraining and evaluation. '
        'The three explicit PEs are unchanged. Nine new fits are paired with nine completed pilot3 fits. '
        'Same25-class development objects, seeds11/22/33,120epochs and six conditions. '
        'This diagnoses information routing; it is not fresh confirmation or a proposed superior PE.','',
        '| PE | Content | Base mCA seeds11/22/33 (%) | Mean (%) |', '|---|---|---:|---:|']
    for role in ROLES:
        for content in ('full','removed'):
            a = [100*metrics[content,role,s,'base']['mca'] for s in SEEDS]
            lines.append('| '+role+' | '+content+' | '+' / '.join(f'{x:.2f}' for x in a)+f' | {np.mean(a):.2f} |')
    lines += ['', '| Contrast | Role/control | Paired gains (pp) | Mean (pp) | Paired-class95% interval |', '|---|---|---:|---:|---:|']
    for d in differences:
        if d['variant']=='base':
            gains=' / '.join(f'{x:+.2f}' for x in d['paired_seed_gains_pp']);lo,hi=d['paired_class_bootstrap_95_pp']
            lines.append(f"| {d['kind']} | {d['label']} | {gains} | {d['mean_gain_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    lines += ['', '| PE/content | '+' | '.join(VARIANTS)+' |', '|---|'+'---:|'*len(VARIANTS)]
    for role in ROLES:
        for content in ('full','removed'):
            a = [100*np.mean([metrics[content,role,s,v]['mca'] for s in SEEDS]) for v in VARIANTS]
            lines.append('| '+role+'/'+content+' | '+' | '.join(f'{x:.2f}' for x in a)+' |')
    lines += ['', 'The shape-free distance model deliberately has less geometric information; '
        'beating it alone does not establish special Gaussian structure. DirectMLP retains the same '
        'shape information in PE. Class bootstrap intervals describe this used development cohort. '
        'No new evaluation result changes the prior pilot3 conclusion or historical RGB/transfer failures.',
        '', '![Content-routing comparison](content_route_summary.png)','']
    (DEST/'report.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for role in ROLES:
        for content,directory,style in [('full',PREVIOUS,'-'),('removed',DEST,'--')]:
            histories=[read(directory/'runs'/role/str(s)/'history.json') for s in SEEDS]
            label=role+'/'+content
            axes[0,0].plot(range(5,121,5),np.mean([[x['ce'] for x in h['validation'][1:]] for h in histories],0),style,label=label)
            axes[0,1].plot(range(6),[100*np.mean([metrics[content,role,s,v]['mca'] for s in SEEDS]) for v in VARIANTS],style,marker='o',label=label)
    axes[0,0].set(xlabel='Epoch',ylabel='Validation CE',title='Matched retraining');axes[0,0].legend(fontsize=6)
    axes[0,1].set(xticks=range(6),xticklabels=VARIANTS,ylabel='Mean-class accuracy (%)',title='All fixed conditions');axes[0,1].legend(fontsize=6)
    for ax,content in zip(axes[1],('full','removed')):
        x=np.arange(3)
        for j,s in enumerate(SEEDS):
            ax.bar(x+(j-1)*.25,[100*metrics[content,r,s,'base']['mca'] for r in ROLES],.25,label='seed'+str(s))
        ax.set(xticks=x,xticklabels=['Gaussian','Direct MLP','Distance'],ylim=(0,100),ylabel='Base mCA (%)',title='Shape in content: '+content);ax.legend(fontsize=7)
    fig.savefig(DEST/'content_route_summary.png',dpi=150);plt.close(fig)
    write(DEST/'result.json',dict(rows=rows,comparisons=differences,permutation_diagnostics=permutation,
        baseline_result_sha256=sha(PREVIOUS/'result.json'),fresh_confirmation=False,goal_complete=False,
        actual_plot_review_pending=True))
    print('CONTENT_ROUTE_REPORT_COMPLETE',flush=True)


def queue():
    # Reuse the idle-GPU scheduler without modifying its frozen source.
    scheduler.DEST=DEST;scheduler.ROLES=ROLES;scheduler.SEEDS=SEEDS;scheduler.MODULE=MODULE
    scheduler.specification=specification;scheduler.bind_runner=bind_runner
    scheduler.seal=seal;scheduler.report=report
    scheduler.queue()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('plan','queue','train','export','report'))
    parser.add_argument('--role',choices=ROLES);parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    if args.action in ('train','export'):
        bind_runner()
        if args.action=='train':runner.train(args.role,args.seed)
        else:runner.export(args.seed)
    else:
        {'plan':plan,'queue':queue,'report':report}[args.action]()
