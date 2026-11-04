"""Prospective, output-only source-error decomposition for the full R65 run."""
import argparse
import time
import numpy as np
from experiments.round65_common import ROOT, OUT, PREP, ROLES, SEEDS, VARIANTS, checked, read, write, sha

PLAN = OUT/'source_confusion_plan1.json'
DEST = OUT/'source_diagnostic1'
SOURCES = ('experiments/round65_source_diagnostic.py', 'tests/test_round65_source_diagnostic.py')
EVENTS = ('accuracy', 'cross_source_error', 'within_source_error', 'source_known_accuracy')


def decompose(logits, labels):
    x, y = np.asarray(logits), np.asarray(labels)
    if (y.ndim != 1 or x.shape != (len(y), 60) or not np.isfinite(x).all()
            or not np.issubdtype(y.dtype, np.integer)
            or not np.array_equal(np.unique(y), np.arange(60))):
        raise ValueError('requires finite full60-class predictions and integer labels')
    prediction = x.argmax(1)
    correct = prediction == y
    same_source = (prediction >= 30) == (y >= 30)
    restricted = np.empty_like(prediction)
    for offset in (0, 30):
        mask = (y >= offset) & (y < offset+30)
        restricted[mask] = offset+x[mask, offset:offset+30].argmax(1)
    events = dict(accuracy=correct, cross_source_error=~same_source,
        within_source_error=same_source & ~correct, source_known_accuracy=restricted == y)
    classes = {key: [float(value[y == c].mean()) for c in range(60)]
               for key, value in events.items()}
    domains = {}
    for name, offset in (('MACGS', 0), ('ModelNet', 30)):
        mask = (y >= offset) & (y < offset+30)
        domains[name] = dict(count=int(mask.sum()),
            mca={key: float(np.mean(value[offset:offset+30])) for key, value in classes.items()},
            object_mean={key: float(value[mask].mean()) for key, value in events.items()})
    return dict(class_rates=classes, mca={key: float(np.mean(value)) for key, value in classes.items()},
        object_mean={key: float(value.mean()) for key, value in events.items()}, domains=domains)


def paired_contrasts(lookup):
    expected = {(r, s, v) for r in ROLES for s in SEEDS for v in VARIANTS}
    if set(lookup) != expected:
        raise ValueError('requires all90 fixed groups')
    results = []
    for variant in VARIANTS:
        for control in ROLES[1:]:
            terms = []
            for seed in SEEDS:
                g, c = (lookup[r, seed, variant]['mca'] for r in ('gaussian_relation', control))
                terms.append(dict(seed=seed, gain_pp=100*(g['accuracy']-c['accuracy']),
                    cross_source_error_reduction_pp=100*(c['cross_source_error']-g['cross_source_error']),
                    within_source_error_reduction_pp=100*(c['within_source_error']-g['within_source_error']),
                    source_known_gain_pp=100*(g['source_known_accuracy']-c['source_known_accuracy'])))
            results.append(dict(control=control, variant=variant, paired=terms,
                mean={key: float(np.mean([r[key] for r in terms])) for key in terms[0] if key != 'seed'}))
    return results


def plan():
    checked()
    if PLAN.exists():
        raise ValueError('preserve existing diagnostic plan')
    if (OUT/'selected.json').exists() or (OUT/'evaluation').exists() or (OUT/'metrics').exists():
        raise ValueError('this prospective plan must precede full-run development predictions')
    write(PLAN, dict(created_unix=time.time(), sources={name: sha(ROOT/name) for name in SOURCES},
        full_validation_plan_sha256=sha(OUT/'full_validation_plan_20261104.json'),
        data_manifest_sha256=sha(OUT/'data/manifest.json'), roles=ROLES, seeds=SEEDS, variants=VARIANTS,
        hypothesis='A Gaussian advantage in the combined60-class task could arise mainly from fewer confusions between the two reconstruction sources.',
        competing_explanation='The advantage also appears when distinguishing classes within each source; source membership alone does not account for it.',
        motivation='Source taxonomies are disjoint30-class sets and complete-data conversion metadata show different covariance scales. This motivates interpretation, not a demonstrated shortcut.',
        fixed_comparison='Reuse all90 frozen evaluation outputs; report Gaussian-minus-each-control paired seeds in allsix conditions.',
        decomposition='Per object:correct + wrong-source + wrong-class-within-source = 1. Average within class, then across classes. Gaussian accuracy gain equals the two control-minus-Gaussian error reductions in percentage points.',
        source_known_diagnostic='Restrict argmax to the30 labels of the known true source. This uses extra source information and is a diagnostic score, not the60-way primary metric or a new model.',
        decisions=dict(source_limited='If positive gains mainly reduce cross-source errors while within-source and source-known comparisons offer little benefit, limit interpretation and seek appropriate within-source or transfer evidence.',
            within_source='Gains that also survive source-known scoring weaken the source-membership-only explanation but do not prove a unique Gaussian mechanism.',
            negative='A primary failure remains a failure; this decomposition does not promote a secondary diagnostic into the primary claim.'),
        limitations=['Source and category sets are confounded; these are error categories, not causal estimates of a source shortcut.',
            'Do not divide contributions by a small or negative net gain or report them as causal percentages.',
            'No new fitting, model selection, parameter tuning, data access or scientific pass gate.',
            'Used TRAIN development data only; no fresh confirmation.'],
        original_protocol_changed=False, fresh_confirmation=False, goal_complete=False))
    print('SOURCE_DIAGNOSTIC_PLAN', sha(PLAN), flush=True)


def report():
    checked(); p = read(PLAN)
    for name, digest in p['sources'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('diagnostic source changed')
    if (p['full_validation_plan_sha256'] != sha(OUT/'full_validation_plan_20261104.json')
            or p['data_manifest_sha256'] != sha(OUT/'data/manifest.json')):
        raise ValueError('full-run identity changed')
    if read(OUT/'queue_status.json')['state'] != 'complete':
        raise ValueError('finish the original full validation before this diagnostic')
    summary = read(OUT/'metrics/summary.json')
    partition = next(r for r in read(OUT/'data/manifest.json')['partitions'] if r['split'] == 'dev_eval')
    ids = np.asarray(partition['ids'])
    catalog = {r['file_path']: r['taxonomy_id'] for r in read(PREP/'train_catalog.json')['rows']}
    labels = np.asarray([catalog[name] for name in ids])
    lookup = {}; records = []
    for row in summary['records']:
        key = row['role'], row['seed'], row['variant']
        if key in lookup:
            raise ValueError('duplicate prediction group')
        path = OUT/'evaluation'/row['role']/str(row['seed'])/(row['variant']+'.npz')
        if sha(path) != row['prediction_sha256']:
            raise ValueError('prediction bytes changed')
        with np.load(path, allow_pickle=False) as z:
            if not np.array_equal(z['ids'], ids) or not np.array_equal(z['labels'], labels):
                raise ValueError('unmatched prediction membership/order')
            d = decompose(z['logits'], z['labels'])
        if (abs(d['mca']['accuracy']-row['metrics']['mca']) > 1e-8
                or abs(d['object_mean']['accuracy']-row['metrics']['oa']) > 1e-8):
            raise ValueError('primary metric discrepancy')
        lookup[key] = d
        records.append(dict(role=key[0], seed=key[1], variant=key[2], **d))
    contrasts = paired_contrasts(lookup)
    DEST.mkdir(exist_ok=False)
    write(DEST/'result.json', dict(records=records, contrasts=contrasts,
        plan_sha256=sha(PLAN), original_summary_sha256=sha(OUT/'metrics/summary.json'),
        original_primary_criterion=summary['base_development_lead'], interpretation_pending=True,
        fresh_confirmation=False, goal_complete=False))
    lines = ['# Full R65 source-error diagnostic', '',
        'Output-only decomposition of the original90 groups. Primary results remain the original60-way classification. '
        'Source-known scoring uses the true source to restrict labels and is diagnostic only. '
        'Different source/category sets prevent causal identification of a source shortcut.', '',
        '| Method | Base mCA % | Cross-source error % | Within-source error % | Source-known mCA % |',
        '|---|---:|---:|---:|---:|']
    for role in ROLES:
        values = [100*np.mean([lookup[role, seed, 'base']['mca'][key] for seed in SEEDS]) for key in EVENTS]
        lines.append('| '+role+' | '+' | '.join(f'{x:.3f}' for x in values)+' |')
    lines += ['', 'Gaussian minus control: gain = cross-source error reduction + within-source error reduction. '
        'Report signed percentage points, not a fraction of a possibly small net gain.', '',
        '| Control | Seed | Gain pp | Cross-source contribution pp | Within-source contribution pp | Source-known gain pp |',
        '|---|---:|---:|---:|---:|---:|']
    for row in contrasts:
        if row['variant'] == 'base':
            for item in row['paired']:
                values = [item[k] for k in ('gain_pp', 'cross_source_error_reduction_pp', 'within_source_error_reduction_pp', 'source_known_gain_pp')]
                lines.append(f"| {row['control']} | {item['seed']} | "+' | '.join(f'{x:+.3f}' for x in values)+' |')
    lines += ['', 'All source/class/seed/condition records are in result.json. Scientific interpretation is pending. '
        'This diagnostic cannot establish effectiveness when the original strong-control comparison fails.', '']
    (DEST/'report.md').write_text('\n'.join(lines))
    print('SOURCE_DIAGNOSTIC_COMPLETE; interpretation pending', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('plan', 'report'))
    args = parser.parse_args(); (plan if args.action == 'plan' else report)()
