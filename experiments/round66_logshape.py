"""One fixed development comparison, reusing the frozen R65 training engine."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np  # NumPy before Torch on this host.
import torch
from scipy.linalg import logm
from experiments import round65_run as runner, round65_audit as audit
from experiments import round65_queue as scheduler
from experiments.round65_common import (ROOT, OUT as PREVIOUS, PREP, CONFIG, SEEDS,
    VARIANTS, ROLES as OLD_ROLES, checked, read, write, sha)
from models.round65_actual_gs import NativeGSClassifier
from models.round66_logshape import ROLES, LogShapeClassifier

DEST = ROOT/'results/round66_logshape/attempt1'
MODULE = 'experiments.round66_logshape'
SOURCES = ('ROUND66_LOGSHAPE_PROTOCOL.md', 'models/round66_logshape.py',
           'experiments/round66_logshape.py', 'tests/test_round66_logshape.py')
ALL_ROLES = ROLES + OLD_ROLES
STRONG = ('direct_mlp', 'distance_mlp')


def freeze():
    parent = checked()
    for name in ('review.json', 'checkpoint_policy1/review.json'):
        review = read(PREVIOUS/name)
        if review['scientific_review_pending'] or review['visual_review_pending']:
            raise ValueError('finish earlier scientific and plot review first')
    if DEST.exists():
        raise ValueError('preserve prior round66 directory')
    names = ('lock.json', 'selected.json', 'metrics/summary.json', 'review.json',
             'checkpoint_policy1/review.json', 'data/manifest.json')
    prior = {name: sha(PREVIOUS/name) for name in names}
    for role in OLD_ROLES:
        for seed in SEEDS:
            for suffix in ('run.json', 'history.json'):
                name = f'runs/{role}/{seed}/{suffix}'
                prior[name] = sha(PREVIOUS/name)
    sources = dict(parent['sources'])
    sources.update({name: sha(ROOT/name) for name in SOURCES})
    lock = dict(config=CONFIG, roles=ROLES, seeds=SEEDS, variants=VARIANTS,
        sources=sources, catalog_sha256=parent['catalog_sha256'],
        previous_record_sha256=prior, created_unix=time.time(),
        protocol='ROUND66_LOGSHAPE_PROTOCOL.md', primary_role=ROLES[0],
        equal_capacity_control=ROLES[1], strong_controls=STRONG,
        reuse_original_ce_selected_baselines=True,
        hypothesis='relative Gaussian log-shape difference adds useful attention context to center distance',
        alternatives=['spectra suffice', 'common content already supplies shape', 'shape differences are nuisance'],
        decision='stop this version on unstable strong-control gains; no checkpoint/scale/epoch rescue',
        complete_scientific_objective_preserved=True, fresh_confirmation=False)
    write(DEST/'lock.json', lock)
    (DEST/'lock.sha256').write_text(sha(DEST/'lock.json')+'\n')
    (DEST/'data').symlink_to((PREVIOUS/'data').resolve(), target_is_directory=True)
    print('FROZEN', sha(DEST/'lock.json'), 'six fits; original full60 development cache', flush=True)


def specification(out=DEST):
    if Path(out) != DEST:
        raise ValueError('R66 output identity')
    checked(PREVIOUS)
    lock = checked(DEST)
    if lock['roles'] != list(ROLES) or lock['seeds'] != list(SEEDS) or lock['variants'] != list(VARIANTS):
        raise ValueError('fixed comparison changed')
    for name, digest in lock['previous_record_sha256'].items():
        if sha(PREVIOUS/name) != digest:
            raise ValueError('completed baseline record changed '+name)
    if (DEST/'data').resolve() != (PREVIOUS/'data').resolve():
        raise ValueError('full cache identity')
    return lock


def bind():
    specification()
    runner.NativeGSClassifier = LogShapeClassifier
    runner.ROLES = ROLES
    runner.checked = specification
    audit.NativeGSClassifier = LogShapeClassifier
    audit.ROLES = ROLES
    audit.checked = specification


def independent_shape(covariance, spectrum_only=False):
    """Small CPU reference: direct pair subtraction; SciPy logm for full SPD."""
    covariance = np.asarray(covariance, dtype=np.float64)
    if spectrum_only:
        features = [.5*np.log(np.linalg.eigvalsh(s)/.04) for s in covariance]
    else:
        features = [.5*logm(s/.04) for s in covariance]
    return np.array([[np.log1p(np.square(a-b).sum()) for b in features] for a in features])


def preflight():
    bind()
    runner.preflight(DEST/'preflight1')
    result = read(DEST/'preflight1/result.json')
    result['lock_sha256'] = sha(DEST/'lock.json')
    comparisons = []
    c, s, a = runner.synthetic_inputs(2, 17, 'cuda')
    for seed in SEEDS:
        torch.manual_seed(seed)
        baseline = NativeGSClassifier('distance_mlp').cuda().eval()
        with torch.no_grad():
            expected = baseline(c, s, a)
        for role in ROLES:
            torch.manual_seed(seed)
            model = LogShapeClassifier(role).cuda().eval()
            if runner.shared_digest(model) != runner.shared_digest(baseline):
                raise ValueError('common initialization mismatch')
            with torch.no_grad():
                actual = model(c, s, a)
            error = float((actual-expected).abs().max())
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
            model.train()
            loss = torch.nn.functional.cross_entropy(model(c, s, a), torch.tensor([3, 59], device='cuda'))
            loss.backward()
            gradient = float(model.position.mlp[0].weight.grad[:, 1].norm())
            if not np.isfinite(gradient) or gradient <= 0:
                raise ValueError('new shape input is inactive')
            comparisons.append(dict(role=role, seed=seed,
                initial_distance_logit_max_error=error, new_column_gradient=gradient))
            del model
        del baseline
    result['initial_distance_comparisons'] = comparisons
    write(DEST/'cuda_preflight.json', result)
    print('PREFLIGHT_COMPLETE', flush=True)


def selections():
    bind()
    audit.selections(DEST)
    selected = read(DEST/'selected.json')
    # Correct the legacy report label; the underlying check already used len(ROLES).
    selected.pop('all_fifteen_selected_before_development_evaluation')
    selected['all_six_selected_before_development_evaluation'] = True
    paired = []
    for entry in selected['records']:
        role, seed = entry['role'], entry['seed']
        current = DEST/'runs'/role/str(seed)
        previous = PREVIOUS/'runs/distance_mlp'/str(seed)
        old_run = read(previous/'run.json')
        if read(current/'run.json')['initial_shared_sha256'] != old_run['initial_shared_sha256']:
            raise ValueError('old/new shared initialization mismatch')
        streams = lambda p: [r['stream_sha256'] for r in read(p/'history.json')['training']]
        if streams(current) != streams(previous):
            raise ValueError('old/new120-epoch input stream mismatch')
        paired.append(dict(role=role, seed=seed, all120_epoch_streams_match=True))
    c, s, _ = runner.synthetic_inputs(1, 9, 'cpu')
    errors = []
    for role in ROLES:
        model = LogShapeClassifier(role)
        actual = model.position.descriptors(c, s)[0, :, :, 1].numpy()
        reference = independent_shape(s[0].numpy(), role == 'spectrum_distance')
        error = float(np.max(np.abs(actual-reference)))
        if not np.allclose(actual, reference, atol=2e-5, rtol=2e-5):
            raise ValueError('independent log-shape reference discrepancy')
        errors.append(dict(role=role, maximum_error=error))
    selected.update(old_baseline_pairing=paired, independent_shape_reference=errors,
                    goal_complete=False)
    write(DEST/'selected.json', selected)


def contrasts(lookup):
    expected = {(r, s, v) for r in ALL_ROLES for s in SEEDS for v in VARIANTS}
    if set(lookup) != expected:
        raise ValueError('incomplete126-group comparison')
    result = []
    pairs = [(ROLES[0], r) for r in (ROLES[1], *OLD_ROLES)]
    pairs += [(ROLES[1], r) for r in STRONG]
    for variant in VARIANTS:
        for candidate, control in pairs:
            left = np.array([lookup[candidate, s, variant]['metrics']['class_accuracy'] for s in SEEDS], dtype=float)
            right = np.array([lookup[control, s, variant]['metrics']['class_accuracy'] for s in SEEDS], dtype=float)
            if left.shape != (3, 60) or right.shape != (3, 60) or not np.isfinite(left).all() or not np.isfinite(right).all():
                raise ValueError('invalid paired class metrics')
            difference = 100*(left-right)
            paired = difference.mean(1)
            interval = audit.bootstrap(difference)
            result.append(dict(candidate=candidate, control=control, variant=variant,
                mean_gain_pp=float(paired.mean()), paired_seed_gains_pp=paired.tolist(),
                paired_class_bootstrap_95_pp=interval,
                development_gain_criterion=bool(paired.mean() >= CONFIG['minimum_gain_pp']
                                               and np.all(paired > 0) and interval[0] > 0)))
    return result


def report():
    specification()
    if (DEST/'result.json').exists():
        raise ValueError('preserve completed report')
    manifest = read(DEST/'data/manifest.json')
    partition = next(x for x in manifest['partitions'] if x['split'] == 'dev_eval')
    labels = np.load(DEST/'data/dev_eval/labels.npy', allow_pickle=False)
    ids = np.asarray(partition['ids'])
    rows, worst, logits = [], 0., {}
    for source, roles in ((DEST, ROLES), (PREVIOUS, OLD_ROLES)):
        selected = read(source/'selected.json')
        for role in roles:
            for seed in SEEDS:
                entry = next(r for r in selected['records'] if (r['role'], r['seed']) == (role, seed))
                directory = source/'evaluation'/role/str(seed)
                meta = read(directory/'record.json')
                if (meta['checkpoint_sha256'] != entry['checkpoint_sha256']
                        or meta['selection_sha256'] != sha(source/'selected.json')
                        or meta['data_sha256'] != sha(DEST/'data/manifest.json')
                        or [r['variant'] for r in meta['records']] != list(VARIANTS)):
                    raise ValueError('prediction provenance mismatch')
                for r in meta['records']:
                    variant = r['variant']; path = directory/(variant+'.npz')
                    if sha(path) != r['sha256']:
                        raise ValueError('prediction bytes changed')
                    with np.load(path, allow_pickle=False) as z:
                        if not np.array_equal(z['labels'], labels) or not np.array_equal(z['ids'], ids):
                            raise ValueError('unpaired prediction membership/order')
                        x = z['logits']
                    m = audit.independent_metrics(x, labels)
                    error = max(abs(m[k]-r['metrics'][k]) for k in ('ce', 'oa', 'mca'))
                    error = max(error, float(np.max(np.abs(np.asarray(m['class_accuracy'])-r['metrics']['class_accuracy']))))
                    worst = max(worst, error)
                    if error > 1e-8:
                        raise ValueError('independent metric mismatch')
                    domains = {name: audit.independent_metrics(x[mask], labels[mask])
                               for name, mask in (('MACGS', labels < 30), ('ModelNet', labels >= 30))}
                    rows.append(dict(role=role, seed=seed, variant=variant, metrics=m, domains=domains,
                                     prediction_sha256=r['sha256'], reused=source == PREVIOUS))
                    if role in ROLES and variant in ('base', 'permuted'):
                        logits[role, seed, variant] = x
    lookup = {(r['role'], r['seed'], r['variant']): r for r in rows}
    differences = contrasts(lookup)
    if len(rows) != len(lookup):
        raise ValueError('duplicate prediction group')
    permutation = []
    for role in ROLES:
        for seed in SEEDS:
            a, b = (logits[role, seed, v] for v in ('base', 'permuted'))
            permutation.append(dict(role=role, seed=seed, maximum_logit_error=float(abs(a-b).max()),
                prediction_changes=int((a.argmax(1) != b.argmax(1)).sum())))
    favorable = all(r['development_gain_criterion'] for r in differences
                    if r['candidate'] == ROLES[0] and r['control'] in STRONG and r['variant'] == 'base')
    result = dict(lock_sha256=sha(DEST/'lock.json'), records=rows, comparisons=differences,
        groups=len(rows), new_groups=36, reused_groups=90, independent_metric_max_error=worst,
        permutation=permutation, primary_development_lead=favorable,
        scientific_review_pending=True, visual_review_pending=True,
        fresh_confirmation=False, goal_complete=False)
    write(DEST/'result.json', result)
    lines = ['# R66: center distance plus one log-shape distance', '',
        'Full60-class used development data. Six new120-epoch fits; five original CE-selected baselines reused. '
        'No officialTEST or fresh confirmation. Scientific interpretation and actual plot review pending.', '',
        '| Method | Base mCA seeds11/22/33 % | Mean % | CE |', '| --- | --- | ---: | ---: |']
    for role in ALL_ROLES:
        values = [100*lookup[role, s, 'base']['metrics']['mca'] for s in SEEDS]
        ce = np.mean([lookup[role, s, 'base']['metrics']['ce'] for s in SEEDS])
        lines.append('| '+role+' | '+' / '.join(f'{x:.3f}' for x in values)+f' | {np.mean(values):.3f} | {ce:.5f} |')
    lines += ['', f'Primary development criterion met: **{favorable}**. This does not establish the global research goal.', '',
        '| Condition | Candidate minus control | Seed differences pp | Mean pp | Paired-class95% pp | Criterion |',
        '| --- | --- | --- | ---: | --- | --- |']
    for r in differences:
        lo, hi = r['paired_class_bootstrap_95_pp']
        values = ' / '.join(f'{x:+.3f}' for x in r['paired_seed_gains_pp'])
        lines.append(f"| {r['variant']} | {r['candidate']} - {r['control']} | {values} | {r['mean_gain_pp']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {r['development_gain_criterion']} |")
    lines += ['', 'All common content still contains full covariance. The full/spectrum control isolates '
        'orientation information in this PE, not in the whole network. Class intervals average paired seeds '
        'before resampling and are not seed-population intervals. Small class counts, unknown physical-instance '
        'grouping and the earlier geometry/transfer/RGB failures remain. Tilt is within the training rotation distribution.', '',
        f'Independent metric maximum error: {worst:.4g}. All per-source/class metrics are in result.json.', '',
        '![Fixed comparisons](comparison.png)', '']
    (DEST/'report.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    base = [r for r in differences if r['variant'] == 'base' and r['control'] in (*STRONG, ROLES[1])]
    for y, r in enumerate(base):
        axes[0].plot(r['paired_class_bootstrap_95_pp'], [y, y], color='black')
        axes[0].scatter(r['paired_seed_gains_pp'], y+np.array([-.1, 0, .1]), color='tab:blue', s=22)
        axes[0].scatter([r['mean_gain_pp']], [y], color='black', marker='D', s=25)
    axes[0].axvline(0, color='grey', lw=1)
    axes[0].set(yticks=range(len(base)), yticklabels=[r['candidate']+' minus '+r['control'] for r in base],
                xlabel='Base mCA difference (pp)', title='Dots: paired seeds; lines: class95% interval')
    axes[0].invert_yaxis(); axes[0].grid(axis='x', alpha=.2)
    for role in (*ROLES, *STRONG):
        source = DEST if role in ROLES else PREVIOUS
        histories = [read(source/'runs'/role/str(s)/'history.json')['validation'] for s in SEEDS]
        values = np.array([[v['ce'] for v in h] for h in histories])
        axes[1].plot([v['epoch'] for v in histories[0]], values.mean(0), label=role)
    axes[1].set(xlabel='Epoch', ylabel='Validation CE', title='Same120 epochs and CE selection')
    axes[1].legend(fontsize=8); axes[1].grid(alpha=.2)
    fig.suptitle('R66 | Used full60-class development; no fresh confirmation')
    fig.tight_layout(); fig.savefig(DEST/'comparison.png', dpi=160); plt.close(fig)
    print('REPORT_COMPLETE', len(rows), 'primary_lead', favorable, flush=True)


def launch(stage, job, gpu):
    tag = stage if job is None else stage+'_'+'_'.join(map(str, job))
    key = tag+'_'+str(time.time_ns())
    command = [sys.executable, '-m', MODULE, stage]
    if stage == 'train':
        command += ['--role', job[0], '--seed', str(job[1])]
    elif stage == 'export':
        command += ['--seed', str(job[0])]
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_DEVICE_ORDER='PCI_BUS_ID',
               CUDA_VISIBLE_DEVICES='' if gpu is None else gpu['uuid'])
    probe = None
    if gpu is not None:
        code = ('import numpy; import torch,json; assert torch.cuda.is_available(); '
                'assert torch.cuda.device_count()==1; x=torch.ones(4,device="cuda"); '
                'assert float(x.sum())==4; torch.cuda.synchronize(); '
                'print(json.dumps(dict(device=torch.cuda.get_device_name(),cuda=True)))')
        probe = json.loads(subprocess.check_output([sys.executable, '-c', code], env=env,
                                                   text=True, timeout=60))
        # Reobserve after the short probe; do not use a GPU occupied in the meantime.
        current = next(r for r in scheduler.occupancy() if r['uuid'] == gpu['uuid'])
        if not current['free']:
            return False
    stream = (DEST/(key+'.log')).open('x')
    try:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    except BaseException:
        stream.close(); raise
    meta = dict(tag=key, stage=stage, job=job, pid=process.pid, gpu=gpu, cuda_probe=probe,
                command=command, started_unix=time.time(),
                start_ticks=Path(f'/proc/{process.pid}/stat').read_text().split()[21])
    scheduler.ACTIVE[key] = dict(process=process, stream=stream, meta=meta)
    print('LAUNCHED', key, process.pid, None if gpu is None else gpu['index'], flush=True)
    return True


def queue():
    specification()
    if (DEST/'queue_status.json').exists():
        raise ValueError('inspect existing controller before any restart')
    scheduler.OUT = DEST
    scheduler.INFO.update(pid=os.getpid(), started_unix=time.time(), lock_sha256=sha(DEST/'lock.json'),
                          start_ticks=Path('/proc/self/stat').read_text().split()[21])
    def stop(signum, frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    jobs = [(ROLES[(j+i) % len(ROLES)], s) for j in range(len(ROLES)) for i, s in enumerate(SEEDS)]
    phases = [('preflight', [None]), ('train', jobs), ('selections', [None]),
              ('export', [(s,) for s in SEEDS]), ('report', [None])]
    try:
        for stage, tasks in phases:
            pending = list(tasks); previous = set()
            while pending or scheduler.ACTIVE:
                pending.extend(scheduler.collect())
                if not pending and not scheduler.ACTIVE:
                    break
                if stage == 'report':
                    if pending:
                        launch(stage, pending.pop(0), None)
                    scheduler.record('running', stage=stage, pending=pending)
                else:
                    rows = scheduler.occupancy()
                    available = {r['uuid'] for r in rows if r['free']}
                    stable = available & previous
                    owned = {item['meta']['gpu']['uuid'] for item in scheduler.ACTIVE.values()}
                    for gpu in rows:
                        if not pending or len(scheduler.ACTIVE) >= 6:
                            break
                        if gpu['uuid'] in stable-owned and launch(stage, pending[0], gpu):
                            pending.pop(0); owned.add(gpu['uuid'])
                    previous = available-owned
                    scheduler.record('running' if scheduler.ACTIVE else 'waiting_for_gpu',
                                     stage=stage, pending=pending, occupancy=rows)
                time.sleep(5)
        scheduler.record('complete', task_execution_completed=True,
                         scientific_review_pending=True, goal_complete=False)
    except BaseException as error:
        scheduler.stop_owned()
        scheduler.record('stopped', error_type=type(error).__name__, error=str(error), goal_complete=False)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze', 'preflight', 'train', 'selections', 'export', 'report', 'queue'))
    parser.add_argument('--role', choices=ROLES); parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    if args.action == 'freeze': freeze()
    elif args.action == 'queue': queue()
    elif args.action == 'preflight': preflight()
    elif args.action == 'selections': selections()
    elif args.action == 'report': report()
    else:
        bind()
        if args.action == 'train': raise SystemExit(runner.fit(args.role, args.seed, DEST))
        else: runner.export(args.seed, DEST)
