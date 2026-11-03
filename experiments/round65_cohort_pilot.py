"""Expanded native-GS development cohort; unchanged R65 model/training recipe."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
from experiments import round65_pilot as runner
from experiments.round65_common import (ROOT, OUT, PREP, CACHE_ROOT, CONFIG, ROLES,
    SEEDS, VARIANTS, checked, read, write, sha)
from experiments.round65_data import split_rows
from experiments.round65_queue import occupancy
from experiments.round65_audit import bootstrap, independent_metrics

DEST = ROOT/'results/round65_actual_gs/pilot3'
CACHE = CACHE_ROOT/'pilot3'
PREVIOUS = ROOT/'results/round65_actual_gs/pilot2'
MODULE = 'experiments.round65_cohort_pilot'


def plan():
    checked()
    if (DEST/'plan.json').exists():
        raise ValueError('preserve existing cohort plan')
    snapshot = read(DEST/'cohort_snapshot.json')
    if snapshot['catalog_sha256'] != sha(PREP/'train_catalog.json'):
        raise ValueError('cohort catalog changed')
    original = split_rows(read(PREP/'train_catalog.json')['rows'])
    for split, rows in snapshot['rows'].items():
        if rows != [r for r in original[split] if r['taxonomy_id'] in snapshot['classes']]:
            raise ValueError('expanded cohort alters original partition membership/order')
    if not read(PREVIOUS/'review.json')['actual_plot_viewed']:
        raise ValueError('finish earlier result interpretation first')
    write(DEST/'plan.json', dict(created_unix=time.time(), source_sha256=sha(__file__),
        runner_sha256=sha(runner.__file__), parent_lock_sha256=sha(OUT/'lock.json'),
        snapshot_sha256=sha(DEST/'cohort_snapshot.json'),
        preceding_result_sha256=sha(PREVIOUS/'result.json'),
        rows=snapshot['rows'], classes=snapshot['classes'], counts=snapshot['counts'],
        roles=ROLES, seeds=SEEDS, variants=VARIANTS, epochs=CONFIG['epochs'],
        config=CONFIG, model_output_classes=60, per_class_sampling_cap=None,
        cohort_rule=snapshot['selection'],
        hypothesis='the exact Gaussian relation PE offers a stable advantage over ordinary and center-distance MLPs across a broader native-GS category set',
        competing_explanations=['generic learned center relations explain the previous PE gains',
            'small-cohort class composition or seed variability explains apparent shape-structure benefits',
            'sampling sensitivity persists when class and object coverage increase'],
        fixed_comparison='allfive unchanged models, three paired seeds,120epochs,512tokens, same13D content and optimizer; full original split membership within25complete classes',
        primary_metric='development-evaluation mean-class accuracy, base512',
        decision=dict(development_lead='retain a broader development lead only if allthree paired seeds improve versus every control, mean>=.5pp and the paired-class95% bootstrap lower bound is positive',
            no_lead='record failures against strong controls; do not tune on these evaluation outputs or declare general feasibility from weaker comparisons',
            robustness='report allsix fixed conditions; secondary conditions never replace the primary comparison'),
        uncertainty='reuse R65 paired-class bootstrap:10000replicates,seed65001, average paired seeds within each of25classes; development uncertainty only',
        limitations=['availability-selected25classes, one MACGS source, not the full60class/two-source experiment',
            'class sizes differ; some validation/evaluation classes contain only2objects',
            'old five-class objects overlap prior development work; all newly decoded entries were already declared development',
            'nonidentical MACGS physical-instance grouping is unknown; no fresh confirmation'],
        all_data_development=True, full_r65_changed=False, fresh_confirmation=False, goal_complete=False))
    print('COHORT_PLAN', snapshot['classes'], snapshot['counts'], flush=True)


def specification():
    checked(); p = read(DEST/'plan.json')
    if (p['source_sha256'] != sha(__file__) or p['runner_sha256'] != sha(runner.__file__)
            or p['parent_lock_sha256'] != sha(OUT/'lock.json')
            or p['snapshot_sha256'] != sha(DEST/'cohort_snapshot.json')):
        raise ValueError('cohort source/configuration identity changed')
    if p['config'] != CONFIG or p['roles'] != list(ROLES) or p['seeds'] != list(SEEDS) or p['variants'] != list(VARIANTS):
        raise ValueError('cohort settings changed')
    if p['rows'] != read(DEST/'cohort_snapshot.json')['rows']:
        raise ValueError('cohort membership changed')
    return p


def bind_runner():
    # Process-local experiment context; the reused source is unchanged.
    runner.DEST = DEST
    runner.CACHE = CACHE
    runner.SEEDS = SEEDS
    runner.EPOCHS = CONFIG['epochs']
    runner.VARIANTS = VARIANTS
    runner.specification = specification


def seal():
    bind_runner(); runner.seal()
    selected = read(DEST/'selected.json')
    if len(selected['records']) != len(ROLES)*len(SEEDS):
        raise ValueError('incomplete cohort selection')
    selected.pop('all_ten_selected_before_development_evaluation')
    selected['all_fifteen_selected_before_development_evaluation'] = True
    write(DEST/'selected.json', selected)


def report():
    p = specification(); classes = p['classes']; rows = []
    labels = np.array([r['taxonomy_id'] for r in p['rows']['dev_eval']])
    ids = np.array([r['file_path'] for r in p['rows']['dev_eval']])
    for seed in SEEDS:
        result = read(DEST/('evaluation_'+str(seed)+'.json'))
        if (result['selection_sha256'] != sha(DEST/'selected.json')
                or result['data_sha256'] != read(DEST/'dev_eval_data.json')['sha256']):
            raise ValueError('evaluation provenance changed')
        rows.extend(result['rows'])
    expected = {(role, seed, variant) for role in ROLES for seed in SEEDS for variant in VARIANTS}
    if len(rows) != len(expected) or {(r['role'], r['seed'], r['variant']) for r in rows} != expected:
        raise ValueError('incomplete90-group evaluation')
    metrics, logits_lookup = {}, {}
    for row in rows:
        path = DEST/'evaluation'/row['role']/str(row['seed'])/(row['variant']+'.npz')
        if sha(path) != row['sha256']:
            raise ValueError('exported prediction identity changed')
        with np.load(path, allow_pickle=False) as z:
            if not np.array_equal(z['labels'], labels) or not np.array_equal(z['ids'], ids):
                raise ValueError('evaluation object membership/order changed')
            independent = independent_metrics(z['logits'], labels)
            if any(abs(independent[k]-row['metrics'][k]) > 1e-8 for k in ('ce', 'oa', 'mca')):
                raise ValueError('independent metric discrepancy')
            key = row['role'], row['seed'], row['variant']
            metrics[key] = independent
            if row['variant'] in ('base', 'permuted'):
                logits_lookup[key] = z['logits']
    comparisons = []
    for variant in VARIANTS:
        for control in ROLES[1:]:
            diff = np.array([[100*(metrics['gaussian_relation', seed, variant]['class_accuracy'][c]-
                                  metrics[control, seed, variant]['class_accuracy'][c]) for c in classes] for seed in SEEDS])
            paired = diff.mean(1); interval = bootstrap(diff)
            comparisons.append(dict(variant=variant, control=control,
                paired_seed_gains_pp=paired.tolist(), mean_gain_pp=float(paired.mean()),
                paired_class_bootstrap_95_pp=interval,
                classes_with_positive_mean=int((diff.mean(0)>0).sum()),
                development_lead=bool(np.all(paired>0) and paired.mean()>=CONFIG['minimum_gain_pp'] and interval[0]>0)))
    primary = [r for r in comparisons if r['variant']=='base']
    lead = all(r['development_lead'] for r in primary)
    permutation = []
    for role in ROLES:
        for seed in SEEDS:
            base, perm = logits_lookup[role, seed, 'base'], logits_lookup[role, seed, 'permuted']
            permutation.append(dict(role=role, seed=seed,
                maximum_logit_error=float(np.abs(base-perm).max()),
                prediction_disagreements=int((base.argmax(1)!=perm.argmax(1)).sum())))
    conclusion = ('Gaussian relations meet the predeclared broader-development lead criterion against allfour controls.' if lead else
                  'The predeclared broader-development Gaussian lead criterion is not established against allfour controls.')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.3), constrained_layout=True)
    best_epochs = {}
    for role in ROLES:
        histories = [read(DEST/'runs'/role/str(s)/'history.json') for s in SEEDS]
        axes[0, 0].plot(range(1, CONFIG['epochs']+1), np.mean([[r['loss'] for r in h['training']] for h in histories], axis=0), label=role)
        axes[0, 1].plot(range(0, CONFIG['epochs']+1, 5), np.mean([[r['ce'] for r in h['validation']] for h in histories], axis=0), label=role)
        best_epochs[role] = [read(DEST/'runs'/role/str(s)/'run.json')['best_epoch'] for s in SEEDS]
    axes[0, 0].set(xlabel='Epoch', ylabel='Smoothed training CE', title='Training: three-seed means')
    axes[0, 1].set(xlabel='Epoch', ylabel='Validation CE', title='Validation: three-seed means')
    for ax in axes[0]:
        ax.legend(fontsize=7)
    x = np.arange(len(ROLES))
    for j, seed in enumerate(SEEDS):
        axes[1, 0].bar(x+(j-1)*.25, [100*metrics[role, seed, 'base']['mca'] for role in ROLES], .25, label='seed'+str(seed))
    axes[1, 0].set(xticks=x, xticklabels=['Gaussian','Direct MLP','Distance','RoPE','NoPE'],
        ylabel='Mean-class accuracy (%)', ylim=(0,100), title='Development evaluation: base512')
    axes[1, 0].legend(fontsize=8)
    for role in ROLES:
        axes[1, 1].plot(range(len(VARIANTS)), [100*np.mean([metrics[role,s,v]['mca'] for s in SEEDS]) for v in VARIANTS], marker='o', label=role)
    axes[1, 1].set(xticks=range(len(VARIANTS)), xticklabels=['base512','sparse256','dense1024','jitter02','tilt','perm'],
        ylabel='Mean-class accuracy (%)', ylim=(0,100), title='All fixed evaluation conditions')
    axes[1, 1].legend(fontsize=7)
    fig.suptitle(f"Native GS development: {len(classes)} classes, {p['counts']['fit']} fit / {p['counts']['val']} val / {p['counts']['dev_eval']} eval")
    fig.savefig(DEST/'cohort_summary.png', dpi=150); plt.close(fig)
    lines = ['# R65 expanded-cohort development verification', '', conclusion, '',
        f"All {len(classes)} complete categories from the availability snapshot were retained, with "
        f"{p['counts']['fit']} fit / {p['counts']['val']} validation / {p['counts']['dev_eval']} evaluation objects. "
        'Original partitions and all objects within each selected category were retained. '
        'Allfive unchanged models, seeds11/22/33,120epochs,512tokens, and the same13D attributes. '
        'This is a MACGS-only development experiment, not the full60class/two-source R65 study.', '',
        '| Method | Base mCA seeds11/22/33 (%) | Base mCA mean (%) | OA mean (%) | CE mean |',
        '|---|---:|---:|---:|---:|']
    for role in ROLES:
        base = [metrics[role,s,'base'] for s in SEEDS]
        numbers = ' / '.join(f"{100*r['mca']:.2f}" for r in base)
        lines.append(f"| {role} | {numbers} | {100*np.mean([r['mca'] for r in base]):.2f} | "
            f"{100*np.mean([r['oa'] for r in base]):.2f} | {np.mean([r['ce'] for r in base]):.4f} |")
    lines += ['', 'Gaussian minus control, primary mean-class accuracy:', '',
        '| Control | Paired seed gains (pp) | Mean gain (pp) | Paired-class95% interval (pp) | Development lead |',
        '|---|---:|---:|---:|---|']
    for r in primary:
        numbers = ' / '.join(f'{v:+.2f}' for v in r['paired_seed_gains_pp'])
        lo, hi = r['paired_class_bootstrap_95_pp']
        lines.append(f"| {r['control']} | {numbers} | {r['mean_gain_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {r['development_lead']} |")
    lines += ['', '| Method | '+ ' | '.join(VI for VI in VARIANTS)+' |',
        '|---|'+ '|'.join(['---:']*len(VARIANTS))+'|']
    for role in ROLES:
        values = [100*np.mean([metrics[role,s,v]['mca'] for s in SEEDS]) for v in VARIANTS]
        lines.append('| '+role+' | '+' | '.join(f'{v:.2f}' for v in values)+' |')
    lines += ['', 'All15 selections precede evaluation. Training budgets, actual input streams, common '
        'initialization, source/data identities and validation checkpoint selection were checked. '
        'Selected validation predictions were replayed, and classification metrics independently recomputed. '
        'Permutation discrepancy records are in result.json.', '',
        'The interval resamples the selected classes after averaging paired seeds within class. '
        'It describes uncertainty within this development cohort, not independent confirmation. '
        'Class sizes vary; the smallest evaluation class has only two objects. '
        'The earlier five-class objects overlap prior development work, and nonidentical MACGS '
        'physical-instance grouping remains unknown. Raw accuracy cannot be treated as a paired '
        'improvement over the easier five-class task. Historical RGB/transfer/sampling failures remain.', '',
        '![Expanded cohort curves and results](cohort_summary.png)', '']
    (DEST/'report.md').write_text('\n'.join(lines))
    write(DEST/'result.json', dict(rows=rows, comparisons=comparisons,
        permutation_diagnostics=permutation, best_validation_epochs=best_epochs,
        broader_development_lead=lead, conclusion=conclusion, fresh_confirmation=False,
        full_r65_changed=False, goal_complete=False, actual_plot_review_pending=True))
    print('COHORT_REPORT_COMPLETE', conclusion, flush=True)


def queue():
    specification(); bind_runner()
    if (DEST/'queue_status.json').exists():
        raise ValueError('inspect existing controller instead of starting a duplicate')
    active, done = {}, []
    info = dict(pid=os.getpid(), started_unix=time.time(), plan_sha256=sha(DEST/'plan.json'),
        start_ticks=Path('/proc/self/stat').read_text().split()[21])

    def status(state, **extra):
        write(DEST/'queue_status.json', dict(info, state=state, observed_unix=time.time(),
            active=[v['metadata'] for v in active.values()], completed=done, **extra))

    def stop(signum, frame):
        raise SystemExit(128+signum)

    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    try:
        status('preparing'); runner.prepare_split('fit'); runner.prepare_split('val')
        for stage, jobs in [('train', [(r, s) for r in ROLES for s in SEEDS]), ('export', [(s,) for s in SEEDS])]:
            if stage == 'export':
                status('selecting'); seal(); runner.prepare_split('dev_eval')
            pending, previous = list(jobs), set()
            while pending or active:
                for pid, item in list(active.items()):
                    code = item['process'].poll()
                    if code is None:
                        continue
                    item['stream'].close(); done.append(dict(item['metadata'], exit_code=code)); del active[pid]
                    print('COHORT_EXIT', pid, code, flush=True)
                    if code:
                        raise RuntimeError('cohort child failure '+str(pid))
                if not pending and not active:
                    break
                try:
                    observed = occupancy()
                except (subprocess.SubprocessError, ValueError) as error:
                    previous = set(); status('gpu_observation_error', stage=stage, pending=pending, error=str(error))
                    time.sleep(10); continue
                free = {r['uuid'] for r in observed if r['free']}
                stable = free & previous
                for gpu in observed:
                    if not pending or len(active) >= 4:
                        break
                    if gpu['uuid'] not in stable:
                        continue
                    job = pending.pop(0)
                    command = [sys.executable, '-u', '-m', MODULE, stage]
                    if stage == 'train':
                        command += ['--role', job[0], '--seed', str(job[1])]
                    else:
                        command += ['--seed', str(job[0])]
                    env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
                        CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=gpu['uuid'])
                    log = DEST/(stage+'_'+'_'.join(map(str, job))+'.log'); stream = log.open('x')
                    try:
                        child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    except BaseException:
                        stream.close(); raise
                    metadata = dict(stage=stage, job=job, pid=child.pid, gpu=gpu,
                        command=command, log=str(log), started_unix=time.time())
                    active[child.pid] = dict(process=child, stream=stream, metadata=metadata)
                    print('COHORT_LAUNCH', job, child.pid, 'GPU', gpu['index'], flush=True)
                    stable.remove(gpu['uuid'])
                previous = free-{v['metadata']['gpu']['uuid'] for v in active.values()}
                status('running' if active else 'waiting_for_gpu', stage=stage, pending=pending, occupancy=observed)
                time.sleep(5 if active else 15)
        status('reporting'); report(); status('complete', scientific_review_pending=True)
    except BaseException as error:
        for item in active.values():
            if item['process'].poll() is None:
                os.killpg(item['process'].pid, signal.SIGTERM)
        for item in active.values():
            try:
                code = item['process'].wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(item['process'].pid, signal.SIGKILL); code = item['process'].wait(timeout=5)
            item['stream'].close(); done.append(dict(item['metadata'], exit_code=code, stopped_by_driver=True))
        active.clear(); status('stopped', error_type=type(error).__name__, error=str(error)); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('plan', 'queue', 'train', 'export', 'report'))
    parser.add_argument('--role', choices=ROLES); parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    if args.action in ('train', 'export'):
        bind_runner()
        if args.action == 'train':
            runner.train(args.role, args.seed)
        else:
            runner.export(args.seed)
    else:
        {'plan': plan, 'queue': queue, 'report': report}[args.action]()
