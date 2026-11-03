"""Fixed-cohort, 120-epoch follow-up to the completed 20-epoch R65 pilot.

Reuse its exact conversion, arrays, training implementation and model controls.
The full R65 experiment and the earlier pilot's files remain unchanged.
"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
from experiments import round65_pilot as prior
from experiments.round65_common import ROOT, OUT, CONFIG, ROLES, checked, read, sha, write
from experiments.round65_queue import occupancy

PREVIOUS = ROOT/'results/round65_actual_gs/pilot1'
DEST = ROOT/'results/round65_actual_gs/pilot2'
SEEDS = (11, 22, 33)
EPOCHS = 120
VARIANTS = ('base', 'sparse256', 'tilt')


def plan():
    checked()
    old = read(PREVIOUS/'plan.json')
    if old['pilot_source_sha256'] != sha(prior.__file__) or old['epochs'] != 20 or old['seeds'] != [11, 22]:
        raise ValueError('prior pilot source or specification changed')
    if not (PREVIOUS/'review.json').exists():
        raise ValueError('previous pilot review required')
    hashes = {n: sha(PREVIOUS/n) for n in ('plan.json', 'result.json', 'review.json')}
    for split in ('fit', 'val', 'dev_eval'):
        name = split+'_data.json'; record = read(PREVIOUS/name)
        if record['plan_sha256'] != hashes['plan.json'] or sha(record['path']) != record['sha256']:
            raise ValueError('previous data identity '+split)
        hashes[name] = sha(PREVIOUS/name)
    DEST.mkdir(parents=True, exist_ok=False)
    write(DEST/'plan.json', dict(created_unix=time.time(), source_sha256=sha(__file__),
        reused_runner_sha256=sha(prior.__file__), parent_lock_sha256=sha(OUT/'lock.json'),
        previous_record_hashes=hashes, rows=old['rows'], classes=old['classes'],
        roles=ROLES, seeds=SEEDS, epochs=EPOCHS, variants=VARIANTS,
        tokens=512, cache_tokens=2048, batch_size=16, model_output_classes=60,
        start_from='new random initialization; never resume a 20-epoch selected checkpoint',
        changed_factors=['20 to120 epochs with cosine endpoint adjusted accordingly', 'additional seed33'],
        fixed_factors=['all240/60/60 object identities and arrays', 'allfive models and13D content',
            'initialization and actual input stream for shared seeds11/22', 'AdamW and5epoch warmup',
            'minimum validation CE selection every5epochs; fixed base/sparse256/tilt evaluations'],
        hypothesis='the weak and unstable pilot result was partly limited by its short training schedule; the longer schedule may yield a stable Gaussian relation advantage',
        competing_explanations=['generic MLP or center information benefits equally or more from longer training',
            'sampling sensitivity persists after learning', 'small source-specific cohort limits inference despite training'],
        comparison='rank allfive methods across allthree seeds; compare20vs120 only on paired seeds11/22',
        decisions=dict(stable_lead='expand the original fixed data design; this remains development evidence',
            unstable_or_negative='retain negative comparisons and diagnose the remaining issue; do not tune on these evaluation outputs',
            density_failure='report as sampling-sensitivity evidence, not a numerical correctness failure'),
        limitations=['same five MACGS classes,60 evaluation objects, already used development data',
            'longer duration and longer cosine schedule are a joint budget intervention, not isolated step count',
            'three seeds do not establish population-level generality',
            'nonidentical MACGS physical-instance grouping remains unknown'],
        all_data_development=True, fresh_confirmation=False, full_controller_resumed=False,
        parent_experiment_changed=False, goal_complete=False))
    print('BUDGET_PILOT_PLAN', {s: len(v) for s, v in old['rows'].items()}, EPOCHS, SEEDS, flush=True)


def specification():
    checked(); p = read(DEST/'plan.json')
    if (p['source_sha256'] != sha(__file__) or p['reused_runner_sha256'] != sha(prior.__file__)
            or p['parent_lock_sha256'] != sha(OUT/'lock.json')):
        raise ValueError('budget pilot source identity changed')
    if p['epochs'] != EPOCHS or p['seeds'] != list(SEEDS) or p['roles'] != list(ROLES) or p['variants'] != list(VARIANTS):
        raise ValueError('budget pilot configuration changed')
    for name, digest in p['previous_record_hashes'].items():
        if sha(PREVIOUS/name) != digest:
            raise ValueError('previous pilot record changed '+name)
    if p['rows'] != read(PREVIOUS/'plan.json')['rows']:
        raise ValueError('budget comparison changed the original cohort')
    return p


def reuse_split(split):
    specification()
    if split not in ('fit', 'val', 'dev_eval'):
        raise ValueError('undeclared development partition')
    if split == 'dev_eval' and not (DEST/'selected.json').exists():
        raise ValueError('all15 checkpoints must be selected before evaluation')
    target = DEST/(split+'_data.json')
    if target.exists():
        raise ValueError('preserve existing pilot cache reference')
    old = read(PREVIOUS/target.name)
    if sha(old['path']) != old['sha256']:
        raise ValueError('original cached arrays changed')
    write(target, dict(old, plan_sha256=sha(DEST/'plan.json'),
        reused_from=str(PREVIOUS/target.name), original_record_sha256=sha(PREVIOUS/target.name)))
    print('REUSED_EXACT_CACHE', split, len(old['records']), flush=True)


def bind_runner():
    # Process-local parameters only: preserve the previous runner source and
    # all previous artifacts. The training/selection implementation is reused.
    prior.DEST = DEST
    prior.SEEDS = SEEDS
    prior.EPOCHS = EPOCHS
    prior.VARIANTS = VARIANTS
    prior.specification = specification


def seal():
    bind_runner(); prior.seal()
    selected = read(DEST/'selected.json')
    if len(selected['records']) != len(ROLES)*len(SEEDS):
        raise ValueError('incomplete budget-pilot selection')
    selected.pop('all_ten_selected_before_development_evaluation')
    selected['all_fifteen_selected_before_development_evaluation'] = True
    comparisons = []
    for seed in (11, 22):
        for role in ROLES:
            previous_run = read(PREVIOUS/'runs'/role/str(seed)/'run.json')
            current_run = read(DEST/'runs'/role/str(seed)/'run.json')
            previous_history = read(PREVIOUS/'runs'/role/str(seed)/'history.json')
            current_history = read(DEST/'runs'/role/str(seed)/'history.json')
            if previous_run['initial_shared_sha256'] != current_run['initial_shared_sha256']:
                raise ValueError('initialization differs from the short pilot')
            if ([r['stream_sha256'] for r in previous_history['training']] !=
                    [r['stream_sha256'] for r in current_history['training'][:20]]):
                raise ValueError('first20 epochs differ in data order, sampling or augmentation')
            comparisons.append(dict(role=role, seed=seed, same_initialization_and_first20_input_stream=True))
    selected['paired_short_pilot_checks'] = comparisons
    write(DEST/'selected.json', selected)


def report():
    p = specification(); rows = []
    for seed in SEEDS:
        r = read(DEST/('evaluation_'+str(seed)+'.json'))
        if r['selection_sha256'] != sha(DEST/'selected.json') or r['data_sha256'] != read(DEST/'dev_eval_data.json')['sha256']:
            raise ValueError('evaluation identity')
        rows.extend(r['rows'])
    expected = {(role, seed, variant) for role in ROLES for seed in SEEDS for variant in VARIANTS}
    if len(rows) != len(expected) or {(r['role'], r['seed'], r['variant']) for r in rows} != expected:
        raise ValueError('incomplete45-condition evaluation')
    for r in rows:
        path = DEST/'evaluation'/r['role']/str(r['seed'])/(r['variant']+'.npz')
        if sha(path) != r['sha256']:
            raise ValueError('evaluation artifact changed')
    metrics = {(r['role'], r['seed'], r['variant']): r['metrics'] for r in rows}
    old_metrics = {(r['role'], r['seed'], r['variant']): r['metrics'] for r in read(PREVIOUS/'result.json')['rows']}
    differences = {variant: {control: [100*(metrics['gaussian_relation', s, variant]['mca']-metrics[control, s, variant]['mca'])
                        for s in SEEDS] for control in ROLES[1:]} for variant in VARIANTS}
    paired_budget = {role: {v: [100*(metrics[role, s, v]['mca']-old_metrics[role, s, v]['mca'])
                        for s in (11, 22)] for v in VARIANTS} for role in ROLES}
    stable = all(all(x > 0 for x in values) for values in differences['base'].values())
    conclusion = ('A consistent base-condition development lead over every control in allthree seeds.' if stable else
        'No consistent base-condition Gaussian lead over every control across allthree seeds.')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2), constrained_layout=True)
    best_epochs = {}
    for role in ROLES:
        histories = [read(DEST/'runs'/role/str(s)/'history.json') for s in SEEDS]
        axes[0, 0].plot(range(1, EPOCHS+1), np.mean([[r['loss'] for r in h['training']] for h in histories], axis=0), label=role)
        axes[0, 1].plot(range(0, EPOCHS+1, 5), np.mean([[r['ce'] for r in h['validation']] for h in histories], axis=0), label=role)
        best_epochs[role] = [read(DEST/'runs'/role/str(s)/'run.json')['best_epoch'] for s in SEEDS]
    axes[0, 0].set(xlabel='Epoch', ylabel='Smoothed training CE', title='Same240 training objects, three-seed mean')
    axes[0, 1].set(xlabel='Epoch', ylabel='Validation CE', title='Same60 validation objects, three-seed mean')
    axes[0, 0].legend(fontsize=8); axes[0, 1].legend(fontsize=8)
    x = np.arange(len(ROLES)); labels = ['Gaussian', 'Direct MLP', 'Distance', 'RoPE', 'NoPE']
    for i, seed in enumerate(SEEDS):
        axes[1, 0].bar(x+(i-1)*.25, [100*metrics[r, seed, 'base']['mca'] for r in ROLES], .25, label='seed'+str(seed))
    for i, variant in enumerate(VARIANTS):
        axes[1, 1].bar(x+(i-1)*.25, [100*np.mean([metrics[r, s, variant]['mca'] for s in SEEDS]) for r in ROLES], .25, label=variant)
    for ax in axes[1]:
        ax.set(xticks=x, xticklabels=labels, ylabel='Mean-class accuracy (%)', ylim=(0, 100)); ax.legend(fontsize=8)
    axes[1, 0].set_title('Base condition: same60 development evaluation objects')
    axes[1, 1].set_title('Fixed sampling/rotation conditions: three-seed means')
    fig.savefig(DEST/'budget_summary.png', dpi=150); plt.close(fig)
    text = ['# R65 fixed-cohort training-budget verification', '', conclusion, '',
        'The same five MACGS classes and exact240/60/60 input arrays from pilot1 were used. '
        'Allfive models were trained from scratch for120epochs, seeds11/22/33, with512tokens. '
        'The20epoch cosine schedule was replaced by the original120epoch schedule; this is a joint '
        'duration/schedule intervention. No architecture, attributes, data split, optimizer, augmentation '
        'or checkpoint-selection rule changed. These evaluation objects were already used in pilot1.', '',
        '| Method | Base mCA seeds11 /22 /33 (%) | Base mean (%) | Base CE mean | Sparse256 mean (%) | Tilt mean (%) |',
        '|---|---:|---:|---:|---:|---:|']
    for role in ROLES:
        base = [metrics[role, s, 'base'] for s in SEEDS]
        per_seed = ' / '.join(f"{100*v['mca']:.2f}" for v in base)
        text.append(f"| {role} | {per_seed} | {100*np.mean([v['mca'] for v in base]):.2f} | "
            f"{np.mean([v['ce'] for v in base]):.4f} | "
            f"{100*np.mean([metrics[role,s,'sparse256']['mca'] for s in SEEDS]):.2f} | "
            f"{100*np.mean([metrics[role,s,'tilt']['mca'] for s in SEEDS]):.2f} |")
    text += ['', 'Paired20-to120epoch comparison uses only the same seeds11/22:', '',
        '| Method | 20epoch base mean (%) | 120epoch base mean (%) | Change (pp) |', '|---|---:|---:|---:|']
    for role in ROLES:
        a = 100*np.mean([old_metrics[role, s, 'base']['mca'] for s in (11, 22)])
        b = 100*np.mean([metrics[role, s, 'base']['mca'] for s in (11, 22)])
        text.append(f'| {role} | {a:.2f} | {b:.2f} | {b-a:+.2f} |')
    text += ['', 'Gaussian minus each control, base mCA (percentage points, seeds11/22/33):', '']
    for control, values in differences['base'].items():
        text.append('- '+control+': '+' / '.join(f'{v:+.2f}' for v in values))
    text += ['', 'Selected validation epochs:', '']
    text += ['- '+role+': '+str(values) for role, values in best_epochs.items()]
    text += ['', 'All15 fits completed equal budgets, with paired initialization and actual data streams. '
        'For shared seeds11/22, the initial weights and first20epoch sampling/augmentation streams match pilot1. '
        'Selected validation outputs were replayed, and metrics were independently recomputed. '
        'The main comparison is against direct MLP with identical native attributes; secondary conditions '
        'must not replace a negative primary result.', '',
        'Scope: five classes, one source,60 previously used development-evaluation objects. '
        'No unused confirmation was accessed, and no population-level significance or general effectiveness is claimed. '
        'Historical RGB/transfer/robustness negatives remain. The original full R65 experiment is unchanged.', '',
        '![Budget comparison](budget_summary.png)', '']
    (DEST/'report.md').write_text('\n'.join(text))
    write(DEST/'result.json', dict(rows=rows, gaussian_minus_controls_pp=differences,
        paired_budget_change_pp=paired_budget, best_validation_epochs=best_epochs,
        stable_base_development_lead=stable, conclusion=conclusion,
        fresh_confirmation=False, goal_complete=False, actual_plot_review_pending=True))
    print('BUDGET_PILOT_REPORT_COMPLETE', conclusion, flush=True)


def queue():
    specification(); bind_runner()
    if (DEST/'queue_status.json').exists():
        raise ValueError('inspect existing controller; never start a duplicate')
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
        status('preparing'); reuse_split('fit'); reuse_split('val')
        for stage, jobs in [('train', [(r, s) for r in ROLES for s in SEEDS]), ('export', [(s,) for s in SEEDS])]:
            if stage == 'export':
                status('selecting'); seal(); reuse_split('dev_eval')
            pending, previous = list(jobs), set()
            while pending or active:
                for pid, item in list(active.items()):
                    code = item['process'].poll()
                    if code is None:
                        continue
                    item['stream'].close(); done.append(dict(item['metadata'], exit_code=code)); del active[pid]
                    print('BUDGET_EXIT', pid, code, flush=True)
                    if code:
                        raise RuntimeError('budget-pilot child failure '+str(pid))
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
                    command = [sys.executable, '-u', '-m', 'experiments.round65_budget_pilot', stage]
                    if stage == 'train':
                        command += ['--role', job[0], '--seed', str(job[1])]
                    else:
                        command += ['--seed', str(job[0])]
                    env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
                        CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=gpu['uuid'])
                    log = DEST/(stage+'_'+'_'.join(map(str, job))+'.log'); stream = log.open('x')
                    try:
                        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                            stderr=subprocess.STDOUT, start_new_session=True)
                    except BaseException:
                        stream.close(); raise
                    metadata = dict(stage=stage, job=job, pid=child.pid, gpu=gpu, command=command,
                        log=str(log), started_unix=time.time())
                    active[child.pid] = dict(process=child, stream=stream, metadata=metadata)
                    print('BUDGET_LAUNCH', job, child.pid, 'GPU', gpu['index'], flush=True)
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
            prior.train(args.role, args.seed)
        else:
            prior.export(args.seed)
    else:
        {'plan': plan, 'queue': queue, 'report': report}[args.action]()
