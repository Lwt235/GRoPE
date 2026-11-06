"""Describe the completed fixed replication without changing its decision rule."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BASE = HERE.parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(records, seeds, variant, domain=None):
    lookup = {(r['role'], r['seed'], r['variant']): r for r in records}
    values = {}
    for role in ('relative5', 'distance_mlp'):
        rows = [lookup[role, seed, variant] for seed in seeds]
        values[role] = [r['metrics'] if domain is None else r['domains'][domain] for r in rows]
    out = dict(seeds=seeds, variant=variant, domain=domain or 'all')
    for metric, scale in (('mca', 100), ('oa', 100), ('ce', 1)):
        left = np.array([r[metric] for r in values['relative5']]) * scale
        right = np.array([r[metric] for r in values['distance_mlp']]) * scale
        delta = left - right
        out[metric] = dict(relative5_mean=float(left.mean()), distance_mean=float(right.mean()),
            mean_difference=float(delta.mean()), paired_differences=delta.tolist(),
            favorable_pairs=int(np.sum(delta < 0 if metric == 'ce' else delta > 0)),
            units='pp for differences, percent for absolute values' if scale == 100 else 'cross entropy')
    return out


def main():
    assert not (HERE / 'analysis.json').exists(), 'Preserve the completed review analysis.'
    result, selected, queue = [read(BASE / n) for n in ('result.json', 'selected.json', 'queue_status.json')]
    assert queue['state'] == 'complete'
    assert result['selection_sha256'] == sha(BASE / 'selected.json')
    assert result['lock_sha256'] == sha(BASE / 'lock.json')
    assert selected['all_ten_selected_before_development_evaluation']
    assert len(selected['records']) == 10
    new_seeds = [44, 55, 66, 77, 88]
    variants = [r['variant'] for r in result['cohorts']['new5']]
    cohorts = {name: rows[0]['seeds'] for name, rows in result['cohorts'].items()}
    bases = {name: summary(result['records'], seeds, 'base') for name, seeds in cohorts.items()}
    conditions = [summary(result['records'], new_seeds, v) for v in variants]
    domains = [summary(result['records'], new_seeds, v, d)
               for v in variants for d in ('MACGS', 'ModelNet')]
    primary = next(r for r in result['cohorts']['new5'] if r['variant'] == 'base')
    class_delta = np.asarray(primary['class_mean_gain_pp'])
    # Numerical zero only; this is descriptive counting, never a new pass criterion.
    class_signs = dict(positive=int((class_delta > 1e-9).sum()),
                      negative=int((class_delta < -1e-9).sum()),
                      tied=int((abs(class_delta) <= 1e-9).sum()), numerical_zero_pp=1e-9)
    for row in conditions:
        expected = next(r for r in result['cohorts']['new5'] if r['variant'] == row['variant'])
        np.testing.assert_allclose(row['mca']['paired_differences'], expected['paired_seed_gains_pp'],
                                   atol=1e-10, rtol=1e-10)
    selected_rows = []
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), sharex=True, sharey='row', layout='constrained')
    for col, seed in enumerate(new_seeds):
        for role, color in (('relative5', 'tab:blue'), ('distance_mlp', 'tab:orange')):
            folder = BASE / 'runs' / role / str(seed)
            history, run = read(folder / 'history.json'), read(folder / 'run.json')
            assert run['epochs'] == 120 and run['steps'] == 46800
            choice = next(r for r in selected['records'] if (r['role'], r['seed']) == (role, seed))
            train, val = history['training'], history['validation']
            axes[0, col].plot([r['epoch'] for r in train], [r['loss'] for r in train], color=color, label=role)
            axes[1, col].plot([r['epoch'] for r in val], [r['ce'] for r in val], color=color, label=role)
            best = next(r for r in val if r['epoch'] == choice['best_epoch'])
            axes[1, col].scatter([best['epoch']], [best['ce']], marker='*', color=color, s=75)
            selected_rows.append(dict(role=role, seed=seed, best_epoch=choice['best_epoch'],
                validation_ce=best['ce'], validation_mca=best['mca'],
                gpu_logit_error=choice['gpu_logit_error'], cpu_logit_error=choice['cpu_logit_error'],
                cpu_prediction_disagreements=choice['cpu_prediction_disagreements']))
        axes[0, col].set_title(f'Paired seed {seed}')
        axes[1, col].set_xlabel('Epoch; star = selected by validation CE')
    axes[0, 0].set_ylabel('Training loss')
    axes[1, 0].set_ylabel('Validation CE (used for selection)')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('All ten fixed-budget fits; training and old validation only')
    fig.savefig(HERE / 'training_curves.png', dpi=140)
    plt.close(fig)
    output = dict(created_unix=time.time(), source_sha256=sha(Path(__file__)),
        result_sha256=sha(BASE / 'result.json'), selection_sha256=sha(BASE / 'selected.json'),
        lock_sha256=sha(BASE / 'lock.json'), cohorts_base=bases, new5_conditions=conditions,
        new5_domains=domains, selected_models=selected_rows, new5_class_signs=class_signs,
        original_primary_summary=primary, original_new_replication_signal=result['new_replication_signal'],
        original_R73_criterion_passed=result['original_R73_criterion_passed'],
        fresh_confirmation=False, official_test_access=False, new_training=False,
        scientific_review_complete=False, visual_review_complete=False, goal_complete=False)
    (HERE / 'analysis.json').write_text(json.dumps(output, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(primary=primary, new_signal=result['new_replication_signal'],
                          class_signs=class_signs, selections=selected_rows), indent=2))


if __name__ == '__main__':
    main()
