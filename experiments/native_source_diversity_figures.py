"""Read-only figures for execution review and the eventual fixed comparison.

Does not change the sealed evaluation, select models, or access raw/test data.
Training snapshots are explicitly incomplete and never substitute for efficacy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.round65_common import ROOT, read, write, sha

OUT = ROOT / 'results/native_source_diversity/attempt1'
ROLES = ('relative5', 'distance_mlp', 'fixed_isotropic5')
SEEDS = (11, 22, 33)
LABELS = {'relative5': 'Full five', 'distance_mlp': 'Distance', 'fixed_isotropic5': 'Fixed sphere'}
COLORS = {'relative5': '#1769aa', 'distance_mlp': '#d17b18', 'fixed_isotropic5': '#468453'}
GROUPS = ('primary_seven', 'seven_MACGS', 'seven_ModelNet', 'original_60', 'original_other53')
GROUP_LABELS = ('Primary: seven classes, two sources', 'Seven classes: MACGS',
                'Seven classes: ModelNet', 'Original 60 classes', 'Original other 53 classes')


def runtime_annotations():
    records = []
    for role in ROLES:
        for seed in SEEDS:
            folder = OUT / 'runs' / role / str(seed)
            paths = sorted(folder.glob('execution_amendment_*.json'))
            if not paths:
                continue
            receipts = [read(path) for path in paths]
            implementations = sorted({r['implementation'] for r in receipts})
            if implementations == ['original']:
                continue
            records.append(dict(role=role, seed=seed, implementations=implementations,
                receipts_sha256={str(path): sha(path) for path in paths}))
    labels = ', '.join(f"{LABELS[r['role']]} seed {r['seed']}" for r in records)
    note = (labels + ': optimized runtime; full-run equivalence unverified') if records else ''
    return records, note


def curves(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.8), sharex=True, sharey='row', layout='constrained')
    records, epochs = [], []
    for column, seed in enumerate(SEEDS):
        present = []
        for role in ROLES:
            path = OUT / 'runs' / role / str(seed) / 'history.json'
            if not path.exists():
                continue
            payload = path.read_bytes()
            history = json.loads(payload)
            saved = destination / f'{role}_{seed}_history.json'
            saved.write_bytes(payload)
            training, validation = history['training'], history['validation']
            if training:
                x, y = [r['epoch'] for r in training], [r['loss'] for r in training]
                axes[0, column].plot(x, y, color=COLORS[role], label=LABELS[role], lw=1.8)
                epochs.extend(x)
            if validation:
                x, y = [r['epoch'] for r in validation], [r['selection_ce'] for r in validation]
                axes[1, column].plot(x, y, 'o-', color=COLORS[role], label=LABELS[role], lw=1.6, ms=3)
                epochs.extend(x)
            present.append(role)
            records.append(dict(role=role, seed=seed, original_path=str(path), snapshot=str(saved),
                history_sha256=hashlib.sha256(payload).hexdigest(), training_epochs=len(training),
                latest_validation_epoch=validation[-1]['epoch'] if validation else None))
        for row in range(2):
            axis = axes[row, column]
            axis.grid(alpha=.17)
            if not present:
                axis.text(.5, .5, 'Not started', ha='center', va='center', transform=axis.transAxes, color='.45')
        axes[0, column].set_title('Seed ' + str(seed))
        axes[1, column].set_xlabel('Completed training cycle (budget: 120)')
    last = max(epochs, default=0)
    upper = min(120, max(10, int(np.ceil(last / 5) * 5)))
    for axis in axes.flat:
        axis.set_xlim(0, upper)
    axes[0, 0].set_ylabel('Training CE\n(with label smoothing)')
    axes[1, 0].set_ylabel('Validation CE\n(class/source weighted)')
    handles, labels = axes[1, 0].get_legend_handles_labels()
    if handles:
        axes[1, 0].legend(handles, labels, fontsize=9, loc='upper right')
    runtime_records, runtime_note = runtime_annotations()
    all_fits_complete = (len(records) == len(ROLES) * len(SEEDS)
                         and all(r['training_epochs'] == 120 for r in records))
    title = ('Training complete: all nine fits; no development scores shown' if all_fits_complete
             else 'Execution snapshot — incomplete fits; no development scores')
    fig.suptitle(title + ('\n' + runtime_note if runtime_note else ''), fontsize=12)
    path = destination / 'training_curves.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    write(destination / 'snapshot.json', dict(created_unix=time.time(), histories=records,
        training_lock_sha256=sha(OUT / 'lock.json'), renderer_sha256=sha(__file__),
        runtime_annotations=runtime_records,
        figure_sha256=sha(path), all_fits_complete=all_fits_complete,
        development_scores_read=False, completed_scientific_result=False))
    print('TRAINING_CURVES_SNAPSHOT', str(path), flush=True)


def result_figures():
    result = read(OUT / 'result.json')
    checked = read(OUT / 'independent_checks.json')
    assert checked['passed'] and checked['result_sha256'] == sha(OUT / 'result.json')
    assert result['fresh_confirmation'] is False
    runtime_records, runtime_note = runtime_annotations()
    destination = OUT / 'figures'
    destination.mkdir(exist_ok=False)
    groups = GROUPS[:4]
    titles = ('Primary: seven classes\ntwo sources equally weighted', 'Seven classes: MACGS\nclass-balanced',
              'Seven classes: ModelNet\nclass-balanced', 'Original 60 classes\nclass-balanced')
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.4), sharey=True, layout='constrained')
    for axis, group, title in zip(axes, groups, titles, strict=True):
        for position, role in enumerate(ROLES):
            for family, offset, alpha in [('background', -.18, .28), ('trained', .18, .9)]:
                values = [100 * next(r['groups'][group]['source_equal_accuracy'] for r in result['records']
                    if (r['family'], r['role'], r['seed']) == (family, role, seed)) for seed in SEEDS]
                axis.bar(position + offset, np.mean(values), width=.33, color=COLORS[role], alpha=alpha,
                         label=('Original training' if family == 'background' else 'Source-diversity training')
                         if position == 0 else None)
                axis.scatter(position + offset + np.linspace(-.055, .055, 3), values, s=13,
                             color=COLORS[role], edgecolor='white', linewidth=.3, zorder=3)
        axis.set_xticks(range(3), ['Full five', 'Distance', 'Sphere'], rotation=25, ha='right')
        axis.set_title(title, fontsize=10)
        axis.grid(axis='y', alpha=.17)
        axis.set_ylim(0, 100)
        axis.set_axisbelow(True)
    axes[0].set_ylabel('Accuracy (%) — full 60-class output')
    axes[0].legend(fontsize=8, loc='lower left')
    title = 'Used development objects: absolute task performance'
    fig.suptitle(title + ('\n' + runtime_note if runtime_note else ''), fontsize=12)
    absolute = destination / 'absolute_accuracy.png'
    fig.savefig(absolute, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.1), sharex=True, sharey=True, layout='constrained')
    limits = []
    for axis, right in zip(axes, ROLES[1:], strict=True):
        for position, group in enumerate(GROUPS):
            row = next(r for r in result['comparisons'] if
                       (r['family'], r['group'], r['right']) == ('trained', group, right))
            lo, hi = row['intervals_95_pp']['crossed']
            mean = row['mean_gain_pp']
            seeds = row['paired_seed_gains_pp']
            limits.extend([lo, hi, *seeds])
            color = COLORS['relative5'] if position == 0 else '.42'
            axis.plot([lo, hi], [position, position], color=color, lw=2)
            axis.plot(mean, position, 'o', color=color, ms=6)
            axis.scatter(seeds, position + np.linspace(-.1, .1, 3), s=15, marker='x', color=color, zorder=3)
            axis.annotate(f'{mean:+.2f}', (mean, position), xytext=(0, -17), textcoords='offset points',
                          ha='center', fontsize=9, color=color)
        axis.axvline(0, color='.5', ls='--', lw=1)
        axis.set_title('Full five − ' + LABELS[right], fontsize=11)
        axis.set_xlabel('Accuracy gain (percentage points)')
        axis.set_yticks(range(5), GROUP_LABELS)
        axis.set_ylim(4.6, -.6)
        axis.grid(axis='x', alpha=.17)
    padding = max(1., (max(limits) - min(limits)) * .08)
    axes[0].set_xlim(min(limits) - padding, max(limits) + padding)
    fig.suptitle('Fixed source-diversity comparison: means, paired seeds, 95% seed/object intervals\n'
                 'Blue is the primary population; other groups describe scope' +
                 ('\n' + runtime_note if runtime_note else ''), fontsize=11)
    effects = destination / 'paired_effects.png'
    fig.savefig(effects, dpi=180)
    plt.close(fig)
    write(destination / 'figure_manifest.json', dict(created_unix=time.time(),
        result_sha256=sha(OUT / 'result.json'), independent_checks_sha256=sha(OUT / 'independent_checks.json'),
        renderer_sha256=sha(__file__), figures={p.name: sha(p) for p in (absolute, effects)},
        runtime_annotations=runtime_records,
        visual_review_pending=True, scientific_review_pending=True))
    print('SOURCE_DIVERSITY_RESULT_FIGURES', str(destination), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('curves', 'results'))
    parser.add_argument('--destination', type=Path)
    args = parser.parse_args()
    if args.action == 'curves':
        if args.destination is None:
            parser.error('curves requires a new --destination directory')
        curves(args.destination)
    else:
        result_figures()
