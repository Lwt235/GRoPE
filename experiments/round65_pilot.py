"""Small, balanced development experiment using already downloaded native GS.

Reuses the frozen R65 models/conversion without changing the full experiment.
Only this pilot is authorized to run; the original controller stays stopped.
"""
import argparse
import hashlib
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np  # Compatible host import order.
import torch
import torch.nn.functional as F
from experiments.round65_common import (ROOT, OUT, PREP, RAW, CACHE_ROOT, CONFIG,
    ROLES, checked, read, sha, write)
from experiments.round65_data import split_rows, read_vertices, native_arrays
from experiments.round65_run import (configure, shared_digest, atomic_save,
    epoch_batches, stream_update, augment, evaluate)
from experiments.round65_audit import independent_metrics
from experiments.round65_queue import occupancy
from models.round65_actual_gs import NativeGSClassifier

DEST = ROOT/'results/round65_actual_gs/pilot1'
CACHE = CACHE_ROOT/'pilot1'
SEEDS = (11, 22)
EPOCHS = 20
COUNTS = dict(fit=48, val=12, dev_eval=12)
VARIANTS = ('base', 'sparse256', 'tilt')


def select_rows(partitions, available):
    """Keep original split membership and ordering; select by availability only."""
    groups = {s: {c: [r for r in rows if r['taxonomy_id'] == c and r['file_path'] in available]
                  for c in range(60)} for s, rows in partitions.items()}
    eligible = [c for c in range(60) if all(len(groups[s][c]) >= n for s, n in COUNTS.items())]
    classes = eligible[:5]
    if len(classes) != 5:
        raise ValueError('five classes with 48/12/12 complete files required')
    selected = {s: [r for c in classes for r in groups[s][c][:n]] for s, n in COUNTS.items()}
    ids = [r['file_path'] for rows in selected.values() for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('pilot partitions overlap')
    return classes, selected


def plan():
    checked()
    catalog = read(PREP/'train_catalog.json')
    available = {r['file_path'] for r in catalog['rows']
                 if (RAW/r['file_path']).is_file() and (RAW/r['file_path']).stat().st_size == r['size']}
    classes, rows = select_rows(split_rows(catalog['rows']), available)
    DEST.mkdir(parents=True, exist_ok=False)
    write(DEST/'plan.json', dict(created_unix=time.time(), pilot_source_sha256=sha(__file__),
        parent_lock_sha256=sha(OUT/'lock.json'), catalog_sha256=sha(PREP/'train_catalog.json'),
        classes=classes, rows=rows, available_files=len(available),
        selection='first five taxonomy IDs with 48 fit / 12 val / 12 dev_eval available; retain original split ordering',
        roles=ROLES, seeds=SEEDS, epochs=EPOCHS, per_class=COUNTS, variants=VARIANTS,
        model_output_classes=60, tokens=CONFIG['tokens'], cache_tokens=CONFIG['cache_tokens'],
        schedule='same AdamW, five-epoch warmup; cosine endpoint compressed to epoch20',
        checkpoint_selection='earliest minimum validation CE at epochs0,5,10,15,20',
        hypothesis='Gaussian relation structure supplies a useful classification bias beyond identical native attributes in direct MLP',
        competing_explanations=['ordinary MLP already extracts the useful attributes',
            'center geometry suffices', 'small-cohort optimization or source-specific signals determine rankings'],
        decisions=dict(stable_lead='retain as a preliminary lead and run complete R65 before generality claims',
            no_lead='record the negative/uncertain result; no tuning on pilot eval or automatic full-experiment cancellation',
            implementation_failure='preserve failure, diagnose arithmetic or data handling; do not rank invalid fits'),
        limitations=['availability selected, only five classes; source coverage determined by chosen IDs',
            'two seeds, 60 evaluation objects, 20 epochs; no statistical or general-effectiveness claim',
            'nonidentical MACGS physical-instance grouping is unknown',
            'some fitting objects previously used in the fit-only feasibility diagnostic'],
        all_data_development=True, confirmation_access=False, full_controller_resumed=False,
        parent_experiment_settings_changed=False, goal_complete=False))
    print('PILOT_PLAN', classes, {s: len(v) for s, v in rows.items()}, flush=True)


def specification():
    checked()
    p = read(DEST/'plan.json')
    if p['pilot_source_sha256'] != sha(__file__) or p['parent_lock_sha256'] != sha(OUT/'lock.json'):
        raise ValueError('pilot or parent source changed; preserve and document any fix')
    return p


def prepare_split(split):
    p = specification()
    if split == 'dev_eval' and not (DEST/'selected.json').exists():
        raise ValueError('select all ten models before evaluation input decoding')
    target = CACHE/(split+'.npz')
    if target.exists() or (DEST/(split+'_data.json')).exists():
        raise ValueError('preserve existing pilot data')
    CACHE.mkdir(parents=True, exist_ok=True)
    arrays = {k: [] for k in ('centers', 'covariance', 'appearance', 'indices')}
    records = []
    for i, row in enumerate(p['rows'][split]):
        path = RAW/row['file_path']
        if path.stat().st_size != row['size'] or sha(path) != row['sha256']:
            raise ValueError('pilot source identity mismatch '+row['file_path'])
        values, stats = native_arrays(read_vertices(path), row['file_path'], CONFIG['cache_tokens'])
        for key in arrays:
            arrays[key].append(values[key])
        records.append(dict(path=row['file_path'], source_sha256=row['sha256'], **stats))
        if (i+1) % 30 == 0:
            print('PILOT_CONVERTED', split, i+1, flush=True)
    np.savez(target, **{k: np.stack(v) for k, v in arrays.items()},
             labels=np.array([r['taxonomy_id'] for r in p['rows'][split]], dtype=np.int64),
             ids=np.array([r['file_path'] for r in p['rows'][split]]))
    write(DEST/(split+'_data.json'), dict(path=str(target), sha256=sha(target),
        plan_sha256=sha(DEST/'plan.json'), records=records, confirmation_access=False))


def load(split):
    p = specification()
    if split == 'dev_eval' and not (DEST/'selected.json').exists():
        raise ValueError('all models must be selected before evaluation')
    record = read(DEST/(split+'_data.json'))
    if record['plan_sha256'] != sha(DEST/'plan.json') or sha(record['path']) != record['sha256']:
        raise ValueError('pilot cache identity')
    with np.load(record['path'], allow_pickle=False) as z:
        bundle = {k: z[k] for k in z.files}
    if bundle['ids'].tolist() != [r['file_path'] for r in p['rows'][split]]:
        raise ValueError('pilot cache input order')
    if bundle['labels'].tolist() != [r['taxonomy_id'] for r in p['rows'][split]]:
        raise ValueError('pilot cache labels')
    return bundle


def rate(step, steps_per_epoch):
    warm, total = CONFIG['warmup_epochs']*steps_per_epoch, EPOCHS*steps_per_epoch
    if step < warm:
        return CONFIG['learning_rate']*(step+1)/warm
    progress = (step-warm)/max(1, total-warm-1)
    return CONFIG['final_learning_rate']+.5*(CONFIG['learning_rate']-CONFIG['final_learning_rate'])*(1+math.cos(math.pi*progress))


def train(role, seed):
    specification(); configure()
    if role not in ROLES or seed not in SEEDS or not torch.cuda.is_available():
        raise ValueError('undeclared model or CUDA unavailable')
    dest = DEST/'runs'/role/str(seed)
    dest.mkdir(parents=True, exist_ok=False)
    data, val = load('fit'), load('val')
    tensors = {k: torch.tensor(data[k], device='cuda') for k in ('centers', 'covariance', 'appearance', 'labels')}
    torch.manual_seed(seed)
    model = NativeGSClassifier(role).cuda()
    initial = shared_digest(model)
    initial_position = {n: v.detach().cpu().clone() for n, v in model.position.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    rng = np.random.default_rng(seed+65000)
    stream = hashlib.sha256(('R65-pilot1:'+str(seed)).encode()).hexdigest()
    count = len(data['labels']); per_epoch = math.ceil(count/CONFIG['batch_size'])
    history, training = [], []
    best, best_epoch, step = float('inf'), None, 0
    started = time.monotonic(); torch.cuda.reset_peak_memory_stats()

    def validate(epoch):
        nonlocal best, best_epoch
        logits, stats = evaluate(model, val)
        history.append(dict(epoch=epoch, step=step, **stats))
        if stats['ce'] < best:
            best, best_epoch = stats['ce'], epoch
            atomic_save(dest/'best.pt', dict(role=role, seed=seed, epoch=epoch,
                state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()}))
            np.savez(dest/'validation.npz', logits=logits, labels=val['labels'], ids=val['ids'])
        write(dest/'history.json', dict(validation=history, training=training))
        print('PILOT_VAL', role, seed, epoch, stats['ce'], stats['mca'], flush=True)

    try:
        validate(0)
        for epoch in range(EPOCHS):
            total = 0.
            for index, subset, quaternion, jitter in epoch_batches(rng, count):
                stream = stream_update(stream, index, subset, quaternion, jitter)
                ii = torch.as_tensor(index, device='cuda'); jj = torch.as_tensor(subset, device='cuda')
                c, s, a = (tensors[k][ii[:, None], jj] for k in ('centers', 'covariance', 'appearance'))
                c, s = augment(c, s, quaternion, jitter)
                torch.manual_seed(seed*10000000+step)
                model.train(); optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:
                    group['lr'] = rate(step, per_epoch)
                loss = F.cross_entropy(model(c, s, a), tensors['labels'][ii], label_smoothing=CONFIG['label_smoothing'])
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite pilot training loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['clip_norm'], error_if_nonfinite=True)
                optimizer.step(); total += float(loss)*len(index); step += 1
            training.append(dict(epoch=epoch+1, step=step, loss=total/count,
                stream_sha256=stream, gradient_norm=float(norm)))
            if (epoch+1) % CONFIG['validation_every'] == 0:
                validate(epoch+1)
            write(dest/'history.json', dict(validation=history, training=training))
        torch.cuda.synchronize()
        changes = {n: float((v.detach().cpu()-initial_position[n]).norm()) for n, v in model.position.named_parameters()}
        write(dest/'run.json', dict(role=role, seed=seed, epochs=EPOCHS, steps=step,
            best_epoch=best_epoch, best_val_ce=best, initial_shared_sha256=initial,
            shared_parameters_changed=shared_digest(model) != initial,
            final_position_parameter_changes=changes, sampling_sha256=stream,
            seconds=time.monotonic()-started, peak_allocated=torch.cuda.max_memory_allocated(),
            parameters=sum(v.numel() for v in model.parameters()), gpu=torch.cuda.get_device_name(),
            plan_sha256=sha(DEST/'plan.json'), fit_data_sha256=read(DEST/'fit_data.json')['sha256'],
            val_data_sha256=read(DEST/'val_data.json')['sha256'],
            hashes={n: sha(dest/n) for n in ('best.pt', 'validation.npz', 'history.json')}))
        print('PILOT_FIT_COMPLETE', role, seed, flush=True)
    except BaseException as error:
        write(dest/'failure.json', dict(error_type=type(error).__name__, message=str(error), step=step))
        raise


def seal():
    p = specification(); val = load('val')
    if (DEST/'selected.json').exists():
        raise ValueError('preserve completed selection')
    records = []; paired = {}
    for seed in SEEDS:
        for role in ROLES:
            d = DEST/'runs'/role/str(seed); r = read(d/'run.json'); h = read(d/'history.json')
            if (r['role'], r['seed'], r['epochs'], r['steps']) != (role, seed, EPOCHS, EPOCHS*math.ceil(len(p['rows']['fit'])/16)):
                raise ValueError('pilot budget identity')
            if r['plan_sha256'] != sha(DEST/'plan.json'):
                raise ValueError('pilot run plan identity')
            for split in ('fit', 'val'):
                if r[split+'_data_sha256'] != read(DEST/(split+'_data.json'))['sha256']:
                    raise ValueError('pilot run data identity')
            for name, digest in r['hashes'].items():
                if sha(d/name) != digest:
                    raise ValueError('pilot artifact changed')
            if [x['epoch'] for x in h['validation']] != list(range(0, EPOCHS+1, 5)) or len(h['training']) != EPOCHS:
                raise ValueError('pilot history incomplete')
            selected = min(h['validation'], key=lambda x: x['ce'])
            if selected['epoch'] != r['best_epoch'] or selected['ce'] != r['best_val_ce']:
                raise ValueError('pilot checkpoint selection mismatch')
            pairing = (r['initial_shared_sha256'], [x['stream_sha256'] for x in h['training']])
            if paired.setdefault(seed, pairing) != pairing:
                raise ValueError('unmatched initialization or actual training inputs')
            with np.load(d/'validation.npz', allow_pickle=False) as z:
                if not np.array_equal(z['ids'], val['ids']) or not np.array_equal(z['labels'], val['labels']):
                    raise ValueError('validation identity mismatch')
                independent = independent_metrics(z['logits'], z['labels'])
            if abs(independent['ce']-r['best_val_ce']) > 1e-8:
                raise ValueError('independent validation CE disagrees')
            if not r['shared_parameters_changed'] or any(v <= 0 for v in r['final_position_parameter_changes'].values()):
                raise ValueError('inactive pilot trainable parameters')
            records.append(dict(role=role, seed=seed, best_epoch=r['best_epoch'],
                checkpoint_sha256=sha(d/'best.pt'), run_sha256=sha(d/'run.json')))
    write(DEST/'selected.json', dict(records=records, plan_sha256=sha(DEST/'plan.json'),
        all_ten_selected_before_development_evaluation=True, confirmation_access=False))
    print('PILOT_SELECTIONS_CHECKED', len(records), flush=True)


@torch.no_grad()
def export(seed):
    specification(); configure()
    if seed not in SEEDS or not torch.cuda.is_available():
        raise ValueError('undeclared evaluation seed or no CUDA')
    selected = read(DEST/'selected.json')
    bundle, val = load('dev_eval'), load('val')
    rows = []
    for role in ROLES:
        d = DEST/'runs'/role/str(seed)
        entry = next(x for x in selected['records'] if x['role'] == role and x['seed'] == seed)
        if sha(d/'best.pt') != entry['checkpoint_sha256'] or sha(d/'run.json') != entry['run_sha256']:
            raise ValueError('selected pilot model changed')
        checkpoint = torch.load(d/'best.pt', map_location='cpu', weights_only=True)
        if (checkpoint['role'], checkpoint['seed'], checkpoint['epoch']) != (role, seed, entry['best_epoch']):
            raise ValueError('checkpoint metadata differs from selection')
        model = NativeGSClassifier(role).cuda().eval(); model.load_state_dict(checkpoint['state_dict'])
        replay, _ = evaluate(model, val)
        with np.load(d/'validation.npz', allow_pickle=False) as z:
            error = float(np.max(np.abs(replay-z['logits'])))
            if not np.allclose(replay, z['logits'], atol=2e-4, rtol=2e-4):
                raise ValueError('selected validation replay differs')
        target = DEST/'evaluation'/role/str(seed); target.mkdir(parents=True, exist_ok=False)
        for variant in VARIANTS:
            logits, stats = evaluate(model, bundle, variant)
            independent = independent_metrics(logits, bundle['labels'])
            if any(abs(stats[k]-independent[k]) > 1e-8 for k in ('ce', 'oa', 'mca')):
                raise ValueError('independent evaluation metrics disagree')
            path = target/(variant+'.npz')
            np.savez(path, logits=logits, labels=bundle['labels'], ids=bundle['ids'])
            rows.append(dict(role=role, seed=seed, variant=variant, metrics=independent,
                sha256=sha(path), validation_replay_max_error=error))
            print('PILOT_EVAL', role, seed, variant, independent['mca'], flush=True)
        del model; torch.cuda.empty_cache()
    write(DEST/('evaluation_'+str(seed)+'.json'), dict(rows=rows,
        selection_sha256=sha(DEST/'selected.json'), data_sha256=read(DEST/'dev_eval_data.json')['sha256']))


def report():
    p = specification(); rows = []
    for seed in SEEDS:
        r = read(DEST/('evaluation_'+str(seed)+'.json'))
        if r['selection_sha256'] != sha(DEST/'selected.json') or r['data_sha256'] != read(DEST/'dev_eval_data.json')['sha256']:
            raise ValueError('evaluation identity')
        rows.extend(r['rows'])
    expected = {(role, seed, variant) for role in ROLES for seed in SEEDS for variant in VARIANTS}
    if {(r['role'], r['seed'], r['variant']) for r in rows} != expected or len(rows) != len(expected):
        raise ValueError('pilot evaluation coverage')
    for r in rows:
        path = DEST/'evaluation'/r['role']/str(r['seed'])/(r['variant']+'.npz')
        if sha(path) != r['sha256']:
            raise ValueError('evaluation artifact identity')
    metrics = {(r['role'], r['seed'], r['variant']): r['metrics'] for r in rows}
    differences = {control: [100*(metrics['gaussian_relation', s, 'base']['mca']-metrics[control, s, 'base']['mca'])
                            for s in SEEDS] for control in ROLES[1:]}
    leads = all(all(x > 0 for x in values) for values in differences.values())
    conclusion = ('A preliminary lead over every control in both seeds; complete R65 is still required.' if leads else
                  'No consistent lead over all controls in both seeds; this pilot does not establish a Gaussian-structure advantage.')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3), constrained_layout=True)
    for role in ROLES:
        histories = [read(DEST/'runs'/role/str(s)/'history.json') for s in SEEDS]
        axes[0].plot(range(1, EPOCHS+1), np.mean([[r['loss'] for r in h['training']] for h in histories], axis=0), label=role)
    axes[0].set(xlabel='Epoch', ylabel='Smoothed training CE (2-seed mean)', title='Small-cohort training')
    axes[0].legend(fontsize=7)
    x = np.arange(len(ROLES))
    for j, seed in enumerate(SEEDS):
        axes[1].bar(x+(j-.5)*.36, [100*metrics[r, seed, 'base']['mca'] for r in ROLES], .36, label='seed '+str(seed))
    axes[1].set(xticks=x, xticklabels=['Gaussian', 'Direct MLP', 'Distance', 'RoPE', 'NoPE'],
                ylabel='Mean-class accuracy (%)', ylim=(0, 100), title='60 development-evaluation objects')
    axes[1].legend(); fig.savefig(DEST/'pilot_summary.png', dpi=150); plt.close(fig)
    text = ['# R65 small-cohort pilot', '', conclusion, '',
        'Data: 5 availability-selected MACGS classes (bag, bottle, train, telephone, pistol); '
        '240 fit / 60 validation / 60 development evaluation. Original partitions retained. '
        'All five unchanged models, seeds11/22, 20epochs, 512tokens. The 60-output classifier is unchanged. '
        'All inputs include center, covariance, opacity and DC appearance; these are full-attribute classification results.', '',
        '| Method | Base mCA seed11 / seed22 (%) | Base CE mean | Sparse256 mCA mean (%) | Tilt mCA mean (%) |',
        '|---|---:|---:|---:|---:|']
    for role in ROLES:
        base = [metrics[role, s, 'base'] for s in SEEDS]
        text.append(f"| {role} | {100*base[0]['mca']:.2f} / {100*base[1]['mca']:.2f} | "
                    f"{np.mean([v['ce'] for v in base]):.4f} | "
                    f"{100*np.mean([metrics[role,s,'sparse256']['mca'] for s in SEEDS]):.2f} | "
                    f"{100*np.mean([metrics[role,s,'tilt']['mca'] for s in SEEDS]):.2f} |")
    text += ['', 'Gaussian minus control, base mCA percentage points (seed11 / seed22):', '']
    text += [f'- {control}: {values[0]:+.2f} / {values[1]:+.2f}' for control, values in differences.items()]
    text += ['', 'All ten fits completed the same budget and paired input streams; initialization and '
        'checkpoint selection were checked. Selected validation logits were replayed and classification '
        'metrics recomputed independently. These checks establish implementation consistency, not general efficacy.', '',
        'Limits: two seeds, five classes, one reconstruction source, only 60 evaluation objects and a short '
        'training budget. No hypothesis test or universal pass is claimed. Nonidentical MACGS physical-instance '
        'grouping remains unknown. No fresh confirmation was accessed. This pilot does not erase RGB, transfer '
        'or other historical negative results, modify full R65 settings, or resume its controller.', '',
        '![Pilot curves and paired results](pilot_summary.png)', '']
    (DEST/'report.md').write_text('\n'.join(text))
    write(DEST/'result.json', dict(rows=rows, gaussian_minus_controls_pp=differences,
        preliminary_consistent_lead=leads, conclusion=conclusion, fresh_confirmation=False,
        parent_experiment_changed=False, goal_complete=False, actual_plot_review_pending=True))
    print('PILOT_REPORT_COMPLETE', conclusion, flush=True)


def queue():
    specification()
    if (DEST/'queue_status.json').exists():
        raise ValueError('inspect existing pilot controller rather than duplicating it')
    active, done = {}, []
    info = dict(pid=os.getpid(), started_unix=time.time(), plan_sha256=sha(DEST/'plan.json'))

    def status(state, **extra):
        write(DEST/'queue_status.json', dict(info, state=state, observed_unix=time.time(),
            active=[v['metadata'] for v in active.values()], completed=done, **extra))

    def stop(signum, frame):
        raise SystemExit(128+signum)

    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    try:
        status('preparing')
        prepare_split('fit'); prepare_split('val')
        stages = [('train', [(r, s) for r in ROLES for s in SEEDS]), ('export', [(s,) for s in SEEDS])]
        for stage, jobs in stages:
            if stage == 'export':
                status('selecting'); seal(); prepare_split('dev_eval')
            pending, previous = list(jobs), set()
            while pending or active:
                for pid, item in list(active.items()):
                    code = item['process'].poll()
                    if code is None:
                        continue
                    item['stream'].close(); done.append(dict(item['metadata'], exit_code=code)); del active[pid]
                    print('PILOT_EXIT', pid, code, flush=True)
                    if code:
                        raise RuntimeError('pilot child failure '+str(pid))
                if not pending and not active:
                    break
                observed = occupancy(); free = {r['uuid'] for r in observed if r['free']}
                stable = free & previous
                for gpu in observed:
                    if not pending or len(active) >= 4:
                        break
                    if gpu['uuid'] not in stable:
                        continue
                    job = pending.pop(0)
                    command = [sys.executable, '-u', '-m', 'experiments.round65_pilot', stage]
                    if stage == 'train':
                        command += ['--role', job[0], '--seed', str(job[1])]
                    else:
                        command += ['--seed', str(job[0])]
                    env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
                        CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=gpu['uuid'])
                    log = DEST/(stage+'_'+'_'.join(map(str, job))+'.log')
                    stream = log.open('x')
                    try:
                        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                            stderr=subprocess.STDOUT, start_new_session=True)
                    except BaseException:
                        stream.close(); raise
                    metadata = dict(stage=stage, job=job, pid=child.pid, gpu=gpu,
                        command=command, log=str(log), started_unix=time.time())
                    active[child.pid] = dict(process=child, stream=stream, metadata=metadata)
                    print('PILOT_LAUNCH', job, child.pid, 'GPU', gpu['index'], flush=True)
                    stable.remove(gpu['uuid'])
                previous = free-{v['metadata']['gpu']['uuid'] for v in active.values()}
                status('running' if active else 'waiting_for_gpu', stage=stage, pending=pending, occupancy=observed)
                time.sleep(3)
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
        active.clear(); status('stopped', error_type=type(error).__name__, error=str(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('plan', 'queue', 'train', 'export', 'report'))
    parser.add_argument('--role', choices=ROLES); parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    if args.action == 'train':
        train(args.role, args.seed)
    elif args.action == 'export':
        export(args.seed)
    else:
        {'plan': plan, 'queue': queue, 'report': report}[args.action]()
