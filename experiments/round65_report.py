"""Report completed R65 development results; never fit or open raw/test data."""
import csv
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.round65_common import ROOT, OUT, ROLES, SEEDS, VARIANTS, checked, read, write, sha

LABELS = dict(gaussian_relation='Gaussian relation', direct_mlp='Direct MLP',
              distance_mlp='Distance MLP', center_rope='Center RoPE', nope='NoPE')
DOMAINS = ('MACGS', 'ModelNet')


def completed_queue(queue):
    """Exit75 is an intentional training shard, provided that same fit finishes."""
    if queue['state'] != 'complete' or queue.get('active'):
        raise ValueError('original queue is not complete')
    successes = []
    for row in queue['completed']:
        job = tuple(row['job']) if row['job'] is not None else None
        if row['exit_code'] == 0:
            successes.append((row['stage'], job))
        elif row['exit_code'] != 75 or row['stage'] != 'train':
            raise ValueError('failed child in completed queue')
    expected = [('data', None), ('selections', None), ('metrics', None)]
    expected += [('train', (role, seed)) for role in ROLES for seed in SEEDS]
    expected += [('export', (seed,)) for seed in SEEDS]
    if len(successes) != len(expected) or set(successes) != set(expected):
        raise ValueError('incomplete or duplicate successful stages')
    for row in queue['completed']:
        if row['exit_code'] == 75 and ('train', tuple(row['job'])) not in successes:
            raise ValueError('training shard has no completed fit')


def aggregate(summary):
    rows = summary['records']
    lookup = {(r['role'], r['seed'], r['variant']): r for r in rows}
    expected = {(r, s, v) for r in ROLES for s in SEEDS for v in VARIANTS}
    if len(rows) != len(expected) or set(lookup) != expected:
        raise ValueError('report requires all90 development groups')
    contrasts = {(c['control'], c['variant']): c for c in summary['comparisons']}
    if len(summary['comparisons']) != 24 or set(contrasts) != {(r, v) for r in ROLES[1:] for v in VARIANTS}:
        raise ValueError('report requires all24 fixed contrasts')
    means = {}
    for role in ROLES:
        means[role] = {}
        for variant in VARIANTS:
            items = [lookup[role, seed, variant] for seed in SEEDS]
            means[role][variant] = dict(
                metrics={m: float(np.mean([r['metrics'][m] for r in items])) for m in ('mca', 'oa', 'ce')},
                domains={d: {m: float(np.mean([r['domains'][d][m] for r in items]))
                             for m in ('mca', 'oa', 'ce')} for d in DOMAINS})
    return lookup, contrasts, means


def render(dest, summary, selection, runs, histories, conversion, scope='INTERNAL DEVELOPMENT'):
    """Pure postprocessing; a caller may provide explicitly labelled synthetic fixtures."""
    lookup, contrasts, means = aggregate(summary)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=False)
    write(dest/'tables.json', dict(aggregate=means, comparisons=summary['comparisons'],
        conversion=conversion, runs=runs, scope=scope, base_development_lead=summary['base_development_lead'],
        fresh_confirmation=False, goal_complete=False))
    with (dest/'class_accuracy.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['scope', 'role', 'seed', 'variant', 'class_id', 'domain', 'accuracy'])
        for role in ROLES:
            for seed in SEEDS:
                for variant in VARIANTS:
                    for category, accuracy in enumerate(lookup[role, seed, variant]['metrics']['class_accuracy']):
                        writer.writerow([scope, role, seed, variant, category,
                                         DOMAINS[int(category >= 30)], accuracy])
    lines = ['# R65: native optimized Gaussian parameters as Transformer PE', '',
        f'Scope: **{scope}**. Five methods, three seeds, six conditions; all90 groups reported.',
        'All roles receive the same center/log-covariance/opacity/DC content. No RGB parameter predictor.',
        'Internal development split of official TRAIN; no published TEST reproduction or fresh confirmation.',
        'Scientific interpretation and actual image review remain pending.', '',
        'The fixed base development lead criterion is '+('MET.' if summary['base_development_lead'] else 'NOT MET.'),
        'This flag does not establish the full research goal or override earlier negative results.', '',
        '## Base condition', '',
        '| Method | Mean-class accuracy % | Overall accuracy % | CE | MACGS mCA % | ModelNet mCA % |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for role in ROLES:
        x = means[role]['base']; m = x['metrics']
        lines.append(f"| {LABELS[role]} | {100*m['mca']:.3f} | {100*m['oa']:.3f} | {m['ce']:.5f} | "
                     f"{100*x['domains']['MACGS']['mca']:.3f} | {100*x['domains']['ModelNet']['mca']:.3f} |")
    lines += ['', '## All fixed contrasts', '',
        'Gaussian minus control, mean-class accuracy in percentage points; positive favors Gaussian.',
        'Intervals resample classes after averaging paired seeds (10,000 samples); they are not independent-seed confidence intervals.', '',
        '| Condition | Control | Mean gain pp | Seeds11/22/33 pp | Class bootstrap95% pp | Positive class means | Fixed criterion |',
        '| --- | --- | ---: | --- | --- | ---: | --- |']
    for variant in VARIANTS:
        for role in ROLES[1:]:
            c = contrasts[role, variant]
            seeds = '/'.join(f'{v:+.3f}' for v in c['paired_seed_gains_pp'])
            lo, hi = c['paired_class_bootstrap_95_pp']
            lines.append(f"| {variant} | {LABELS[role]} | {c['mca_gain_pp']:+.3f} | {seeds} | "
                         f"[{lo:+.3f}, {hi:+.3f}] | {c['classes_with_positive_mean']}/60 | {c['development_gain_criterion']} |")
    lines += ['', '## Per-seed results and training cost', '',
        '| Method | Seed | mCA % | OA % | CE | Selected epoch | Params / PE params | Peak GiB | Elapsed h |',
        '| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |']
    for run in runs:
        role, seed = run['role'], run['seed']; m = lookup[role, seed, 'base']['metrics']
        lines.append(f"| {LABELS[role]} | {seed} | {100*m['mca']:.3f} | {100*m['oa']:.3f} | {m['ce']:.5f} | "
                     f"{run['best_epoch']} | {run['parameters']} / {run['position_parameters']} | "
                     f"{run['peak_allocated']/2**30:.3f} | {run['seconds']/3600:.3f} |")
    lines += ['', 'Peak allocated memory and elapsed time are from shared-host fits, not a controlled speed comparison.',
        'Complete per-class/seed/condition accuracies are in class_accuracy.csv; every aggregate and domain metric is in tables.json.', '',
        '## Data conversion and interpretation', '',
        'The covariance floor is1e-8 I. The table describes its trace contribution; it is not a thin-axis distortion bound.', '',
        '| Split | Objects | Sampled with replacement | Median of object median floor fractions | Largest object maximum floor fraction |',
        '| --- | ---: | ---: | ---: | ---: |']
    for c in conversion:
        lines.append(f"| {c['split']} | {c['count']} | {c['replacement_count']} | "
                     f"{c['median_object_median_floor_fraction']:.6g} | {c['max_object_max_floor_fraction']:.6g} |")
    lines += ['',
        'A gain over center-only PE controls alone does not establish a Gaussian advantage over the direct MLP.',
        'A source-specific or density-specific gain retains that scope. Permutation is an implementation diagnostic.',
        'Tilt follows full SO(3) training and therefore is not unseen-rotation-distribution transfer.',
        'ModelNet objects may overlap earlier development objects. Nonidentical MACGS physical instances are not independently grouped.',
        'R62/R64 RGB failures and earlier direct-transfer/sparse failures remain unresolved; this experiment cannot erase them.',
        'Any fresh confirmation requires a prospective unused-data plan after scientific review; no such data is read here.', '',
        f"Independent metric maximum discrepancy: {summary['independent_metric_max_error']:.4g}.",
        f"Selected GPU replay maximum logit discrepancy: {max(r['gpu_logit_error'] for r in selection['records']):.4g}; "
        f"CPU replay: {max(r['cpu_logit_error'] for r in selection['records']):.4g}.", '']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for col, domain in enumerate((None, *DOMAINS)):
        ax = axes[col]
        for y, role in enumerate(ROLES[1:]):
            if domain is None:
                c = contrasts[role, 'base']; values = c['paired_seed_gains_pp']
                ax.plot(c['paired_class_bootstrap_95_pp'], [y, y], color='black', lw=2)
                mean = c['mca_gain_pp']
            else:
                values = [100*(lookup['gaussian_relation', s, 'base']['domains'][domain]['mca']-
                               lookup[role, s, 'base']['domains'][domain]['mca']) for s in SEEDS]
                mean = float(np.mean(values))
            ax.scatter(values, y+np.array([-.13, 0, .13]), s=28, color='tab:blue')
            ax.scatter([mean], [y], s=35, color='black', marker='D')
        ax.axvline(0, color='grey', lw=1); ax.grid(axis='x', alpha=.2)
        ax.set_yticks(range(4), [LABELS[r] for r in ROLES[1:]]); ax.invert_yaxis()
        ax.set_xlabel('Gaussian minus control mCA (pp)')
        ax.set_title('All60 classes; class bootstrap95%' if domain is None else domain+'; descriptive seed differences')
    fig.suptitle(scope+' | Base condition; dots=paired seeds, diamonds=means')
    fig.tight_layout(); fig.savefig(dest/'primary_contrasts.png', dpi=160); plt.close(fig)

    gains = np.array([[contrasts[r, v]['mca_gain_pp'] for r in ROLES[1:]] for v in VARIANTS])
    limit = max(float(np.max(np.abs(gains))), .01)
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(gains, cmap='RdBu', vmin=-limit, vmax=limit, aspect='auto')
    for i, variant in enumerate(VARIANTS):
        for j, role in enumerate(ROLES[1:]):
            passed = contrasts[role, variant]['development_gain_criterion']
            ax.text(j, i, f'{gains[i,j]:+.3f}'+(' *' if passed else ''), ha='center', va='center',
                    color='white' if abs(gains[i,j])>.55*limit else 'black')
    ax.set_xticks(range(4), [LABELS[r] for r in ROLES[1:]]); ax.set_yticks(range(6), VARIANTS)
    ax.set_title(scope+' | All fixed conditions\n* = frozen gain criterion met for this contrast')
    fig.colorbar(im, ax=ax, label='Gaussian minus control mCA (pp)')
    fig.tight_layout(); fig.savefig(dest/'condition_gains.png', dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for role in ROLES:
        items = [histories[role, s]['validation'] for s in SEEDS]
        epochs = [r['epoch'] for r in items[0]]
        if any([r['epoch'] for r in h] != epochs for h in items):
            raise ValueError('unmatched validation history epochs')
        for ax, metric, scale in ((axes[0], 'ce', 1.), (axes[1], 'mca', 100.)):
            values = scale*np.array([[r[metric] for r in h] for h in items])
            mean, sd = values.mean(0), values.std(0, ddof=1)
            ax.plot(epochs, mean, label=LABELS[role]); ax.fill_between(epochs, mean-sd, mean+sd, alpha=.12)
            ax.set_xlabel('Epoch'); ax.grid(alpha=.2)
    axes[0].set_ylabel('Validation CE'); axes[1].set_ylabel('Validation mCA (%)')
    axes[0].legend(fontsize=8); fig.suptitle(scope+' | Mean +/- seed SD; checkpoints selected by CE')
    fig.tight_layout(); fig.savefig(dest/'validation_curves.png', dpi=160); plt.close(fig)

    class_gain = np.array([np.mean([np.array(lookup['gaussian_relation', s, 'base']['metrics']['class_accuracy'])-
        np.array(lookup[r, s, 'base']['metrics']['class_accuracy']) for s in SEEDS], axis=0)*100 for r in ROLES[1:]])
    limit = max(float(np.max(np.abs(class_gain))), .01)
    fig, ax = plt.subplots(figsize=(14, 3.8))
    im = ax.imshow(class_gain, cmap='RdBu', vmin=-limit, vmax=limit, aspect='auto')
    ax.axvline(29.5, color='black', lw=1); ax.set_yticks(range(4), [LABELS[r] for r in ROLES[1:]])
    ax.set_xticks(range(0, 60, 5)); ax.set_xlabel('Class ID; 0-29 MACGS, 30-59 ModelNet')
    ax.set_title(scope+' | All60 classes, paired-seed mean differences')
    fig.colorbar(im, ax=ax, label='Gaussian minus control accuracy (pp)')
    fig.tight_layout(); fig.savefig(dest/'class_gains.png', dpi=160); plt.close(fig)
    for filename in ('primary_contrasts.png', 'condition_gains.png', 'validation_curves.png', 'class_gains.png'):
        lines += [f'![{filename}]({dest.resolve()/filename})', '']
    (dest/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    checked(); queue = read(OUT/'queue_status.json'); completed_queue(queue)
    summary = read(OUT/'metrics/summary.json'); selection = read(OUT/'selected.json')
    if len(selection['records']) != 15:
        raise ValueError('all selected models required')
    source_paths = [OUT/'metrics/summary.json', OUT/'selected.json', OUT/'data/manifest.json', OUT/'queue_status.json']
    runs, histories, conversion = [], {}, []
    for selected in selection['records']:
        role, seed = selected['role'], selected['seed']; directory = OUT/'runs'/role/str(seed)
        if sha(directory/'run.json') != selected['run_sha256']:
            raise ValueError('selected run metadata changed')
        run = read(directory/'run.json')
        if sha(directory/'history.json') != run['hashes']['history.json']:
            raise ValueError('selected training history changed')
        runs.append({k: run[k] for k in ('role', 'seed', 'best_epoch', 'parameters', 'position_parameters', 'seconds', 'peak_allocated')})
        histories[role, seed] = read(directory/'history.json')
        source_paths.extend([directory/'run.json', directory/'history.json'])
    for partition in read(OUT/'data/manifest.json')['partitions']:
        path = OUT/'data'/partition['split']/'conversion.json'
        if sha(path) != partition['hashes']['conversion.json']:
            raise ValueError('conversion record changed')
        rows = read(path); source_paths.append(path)
        conversion.append(dict(split=partition['split'], count=len(rows),
            replacement_count=sum(r['sampled_with_replacement'] for r in rows),
            median_object_median_floor_fraction=float(np.median([r['floor_fraction_quantiles'][1] for r in rows])),
            max_object_max_floor_fraction=max(r['floor_fraction_quantiles'][3] for r in rows)))
    inputs = {str(p.relative_to(OUT)): sha(p) for p in source_paths}
    dest = OUT/'report1'; render(dest, summary, selection, runs, histories, conversion)
    write(dest/'generated.json', dict(generated_unix=time.time(), source_sha256=sha(__file__),
        lock_sha256=sha(OUT/'lock.json'), input_sha256=inputs,
        hashes={p.name: sha(p) for p in dest.iterdir() if p.is_file()},
        scientific_review_pending=True, visual_review_pending=True, goal_complete=False))
    print('R65 REPORT GENERATED; scientific and actual image review pending', flush=True)


if __name__ == '__main__':
    main()
