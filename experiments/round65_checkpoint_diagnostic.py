"""One fixed final120-versus-selected diagnostic on already used development data."""
import argparse
import hashlib
import os
from pathlib import Path
import time
import numpy as np
import torch
from experiments.round65_common import ROOT, OUT, ROLES, SEEDS, VARIANTS, CONFIG, checked, read, write, sha
from experiments.round65_data import load
from experiments.round65_run import configure, evaluate
from experiments.round65_audit import independent_metrics, bootstrap
from models.round65_actual_gs import NativeGSClassifier

DEST = OUT/'checkpoint_policy1'
PLAN = DEST/'plan.json'
SOURCES = ('experiments/round65_checkpoint_diagnostic.py', 'tests/test_round65_checkpoint_diagnostic.py')
INPUTS = ('lock.json', 'data/manifest.json', 'metrics/summary.json', 'selected.json', 'review.json')


def final_state(saved, entry, lock_sha, data_sha):
    expected = entry['role'], entry['seed'], 120, 46800, lock_sha, data_sha, entry['sampling_sha256']
    actual = tuple(saved.get(k) for k in ('role', 'seed', 'epoch_completed', 'step',
                                         'lock_sha256', 'data_sha256', 'stream_sha256'))
    if actual != expected or not saved.get('state_dict'):
        raise ValueError('final checkpoint identity/budget mismatch')
    return saved['state_dict']  # Deliberately not saved['best_state'].


def state_digest(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        a = value.detach().cpu().contiguous().numpy()
        h.update(key.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def contrast(values):
    x = np.asarray(values, dtype=float)
    if x.shape != (3, 60) or not np.isfinite(x).all():
        raise ValueError('three paired seeds and all60 classes required')
    paired = x.mean(1); interval = bootstrap(x)
    return dict(paired_pp=paired.tolist(), mean_pp=float(paired.mean()), class_bootstrap_95_pp=interval,
        fixed_criterion=bool(np.all(paired > 0) and paired.mean() >= CONFIG['minimum_gain_pp'] and interval[0] > 0))


def compare_policies(baseline, final):
    expected = {(r, s, v) for r in ROLES for s in SEEDS for v in VARIANTS}
    if set(baseline) != expected or set(final) != expected:
        raise ValueError('both policies need all90 groups')
    def classes(policy, role, variant):
        return 100*np.asarray([policy[role, seed, variant]['class_accuracy'] for seed in SEEDS])
    changes, gaps, interactions = [], [], []
    for variant in VARIANTS:
        for role in ROLES:
            changes.append(dict(role=role, variant=variant,
                **contrast(classes(final, role, variant)-classes(baseline, role, variant)),
                ce_changes=[final[role, seed, variant]['ce']-baseline[role, seed, variant]['ce'] for seed in SEEDS]))
        for control in ROLES[1:]:
            old = classes(baseline, ROLES[0], variant)-classes(baseline, control, variant)
            new = classes(final, ROLES[0], variant)-classes(final, control, variant)
            gaps.append(dict(control=control, variant=variant, **contrast(new)))
            interactions.append(dict(control=control, variant=variant, **contrast(new-old)))
    return dict(policy_changes=changes, final_gaussian_gaps=gaps, gaussian_gap_changes=interactions)


def plan():
    checked()
    if PLAN.exists():
        raise ValueError('preserve the existing diagnostic')
    if read(OUT/'queue_status.json')['state'] != 'complete' or read(OUT/'review.json')['scientific_review_pending']:
        raise ValueError('finish and interpret the original study first')
    entries = []
    for role in ROLES:
        for seed in SEEDS:
            path = OUT/'runs'/role/str(seed); run = read(path/'run.json')
            if (run['epochs'], run['steps']) != (120, 46800):
                raise ValueError('incomplete final checkpoint')
            for name in ('resume.pt', 'best.pt', 'history.json'):
                if sha(path/name) != run['hashes'][name]:
                    raise ValueError('original artifact changed')
            entries.append(dict(role=role, seed=seed, original_best_epoch=run['best_epoch'],
                sampling_sha256=run['sampling_sha256'], run_sha256=sha(path/'run.json'),
                hashes={name: run['hashes'][name] for name in ('resume.pt', 'best.pt', 'history.json')}))
    write(PLAN, dict(created_unix=time.time(), sources={name: sha(ROOT/name) for name in SOURCES},
        inputs={name: sha(OUT/name) for name in INPUTS}, entries=entries, roles=ROLES, seeds=SEEDS, variants=VARIANTS,
        hypothesis='Selecting by validation CE rather than using the shared final120 checkpoint contributes to the Gaussian mCA gap against strong controls.',
        competing_explanation='The Gaussian strong-control gap persists at the uniform final checkpoint; checkpoint policy does not account for the missing gain.',
        prior_observations='The complete original90-group results, validation curves, selected epochs, source diagnostic and lower Gaussian CE were already inspected. This is an adaptive development diagnostic, not fresh confirmation.',
        fixed_comparison='Allfive roles, allthree seeds and allsix conditions. Original CE-selected outputs are reused; exactly one new policy uses state_dict from resume.pt at epoch120/step46800. No best_state substitution, per-role policy choice, checkpoint sweep, new optimization or new data.',
        procedure='Freeze all15 checkpoint identities; replay final validation metrics for all15 before any new development export. Recompute all90 final groups independently. Report final-minus-selected per role, final Gaussian-minus-control gaps, and paired differences of those gaps.',
        decisions=dict(persistent='If final120 still lacks a stable strong-control advantage, retain this implementation as a negative benchmark and leave checkpoint search for task/descriptor redesign.',
            favorable='A favorable final120 result is a policy-sensitive development finding; it does not rewrite the original negative result or establish the global goal. Freeze the policy before appropriate further tests.',
            mixed='If effects depend on seed, source or condition, retain that limitation and do not select a favorable hybrid policy.'),
        primary='class-balanced mCA; same0.5pp/allthree-positive/class-bootstrap rule for descriptive final contrasts',
        secondary='OA, CE, source summaries, actual permutation predictions; no calibration claim from CE alone',
        new_training_runs=0, new_evaluation_groups=90, original_primary_result_changed=False,
        fresh_confirmation=False, goal_complete=False))
    print('CHECKPOINT_POLICY_PLAN', sha(PLAN), flush=True)


def check_plan():
    checked(); p = read(PLAN)
    for name, digest in p['sources'].items():
        if sha(ROOT/name) != digest: raise ValueError('diagnostic source changed '+name)
    for name, digest in p['inputs'].items():
        if sha(OUT/name) != digest: raise ValueError('original input changed '+name)
    for entry in p['entries']:
        path = OUT/'runs'/entry['role']/str(entry['seed'])
        if sha(path/'run.json') != entry['run_sha256']: raise ValueError('run identity changed')
        for name, digest in entry['hashes'].items():
            if sha(path/name) != digest: raise ValueError('checkpoint identity changed')
    return p


def checkpoint(entry, p):
    path = OUT/'runs'/entry['role']/str(entry['seed'])/'resume.pt'
    saved = torch.load(path, map_location='cpu', weights_only=False)
    return final_state(saved, entry, p['inputs']['lock.json'], p['inputs']['data/manifest.json'])


@torch.no_grad()
def seal():
    p = check_plan(); configure()
    if not torch.cuda.is_available(): raise RuntimeError('idle CUDA device required')
    if (DEST/'sealed.json').exists(): raise ValueError('preserve final-checkpoint seal')
    val = load('val'); records = []
    for entry in p['entries']:
        state = checkpoint(entry, p); path = OUT/'runs'/entry['role']/str(entry['seed'])
        best = torch.load(path/'best.pt', map_location='cpu', weights_only=True)['state_dict']
        same = state_digest(state) == state_digest(best)
        if entry['original_best_epoch'] == 120 and not same: raise ValueError('epoch120 policies must share weights')
        model = NativeGSClassifier(entry['role']).cuda().eval(); model.load_state_dict(state)
        logits, stats = evaluate(model, val); independent = independent_metrics(logits, val['labels'])
        logged = read(path/'history.json')['validation'][-1]
        if logged['epoch'] != 120: raise ValueError('missing final validation')
        error = max(abs(independent[k]-logged[k]) for k in ('ce', 'oa', 'mca'))
        if error > 1e-8 or not np.allclose(independent['class_accuracy'], logged['class_accuracy'], atol=1e-12, rtol=0):
            raise ValueError('final checkpoint validation replay mismatch')
        records.append(dict(role=entry['role'], seed=entry['seed'], state_sha256=state_digest(state),
            checkpoint_sha256=entry['hashes']['resume.pt'], identical_to_selected=same,
            validation_metrics=stats, independent_logged_metric_error=error))
        print('FINAL_SEALED', entry['role'], entry['seed'], error, same, flush=True)
        del model, state, best; torch.cuda.empty_cache()
    write(DEST/'sealed.json', dict(records=records, plan_sha256=sha(PLAN), all_fifteen_before_development=True))


@torch.no_grad()
def export(seed):
    p = check_plan(); configure(); sealed = read(DEST/'sealed.json')
    if seed not in SEEDS or len(sealed['records']) != 15 or sealed['plan_sha256'] != sha(PLAN):
        raise ValueError('complete final-checkpoint seal required')
    if not torch.cuda.is_available(): raise RuntimeError('idle CUDA device required')
    bundle = load('dev_eval')
    for role in ROLES:
        entry = next(r for r in p['entries'] if (r['role'], r['seed']) == (role, seed))
        record = next(r for r in sealed['records'] if (r['role'], r['seed']) == (role, seed))
        state = checkpoint(entry, p)
        if state_digest(state) != record['state_sha256']: raise ValueError('sealed final weights changed')
        dest = DEST/'evaluation'/role/str(seed); dest.mkdir(parents=True, exist_ok=False)
        model = NativeGSClassifier(role).cuda().eval(); model.load_state_dict(state); rows = []
        for variant in VARIANTS:
            logits, stats = evaluate(model, bundle, variant)
            target = dest/(variant+'.npz')
            np.savez(target, logits=logits, labels=bundle['labels'], ids=bundle['ids'])
            same_error = None
            if record['identical_to_selected']:
                with np.load(OUT/'evaluation'/role/str(seed)/(variant+'.npz'), allow_pickle=False) as old:
                    same_error = float(np.max(np.abs(logits-old['logits'])))
                    if not np.allclose(logits, old['logits'], atol=2e-4, rtol=2e-4):
                        raise ValueError('identical-checkpoint control discrepancy')
            rows.append(dict(variant=variant, sha256=sha(target), metrics=stats, identical_policy_logit_error=same_error))
            print('FINAL_EXPORT', role, seed, variant, flush=True)
        write(dest/'record.json', dict(role=role, seed=seed, records=rows, seal_sha256=sha(DEST/'sealed.json'),
            checkpoint_sha256=entry['hashes']['resume.pt']))
        del model, state; torch.cuda.empty_cache()


def report():
    p = check_plan(); seal_record = read(DEST/'sealed.json')
    if (DEST/'result.json').exists(): raise ValueError('preserve diagnostic result')
    original = read(OUT/'metrics/summary.json'); baseline = {(r['role'], r['seed'], r['variant']):r['metrics'] for r in original['records']}
    original_records = {(r['role'], r['seed'], r['variant']):r for r in original['records']}
    final = {}; records = []; maximum_error = 0.; permutation = []
    for entry in p['entries']:
        role, seed = entry['role'], entry['seed']; dest = DEST/'evaluation'/role/str(seed); meta = read(dest/'record.json')
        if meta['checkpoint_sha256'] != entry['hashes']['resume.pt'] or meta['seal_sha256'] != sha(DEST/'sealed.json'):
            raise ValueError('export identity mismatch')
        if [r['variant'] for r in meta['records']] != list(VARIANTS): raise ValueError('incomplete variants')
        outputs = {}
        for row in meta['records']:
            variant = row['variant']; key = role, seed, variant; target = dest/(variant+'.npz')
            old_path = OUT/'evaluation'/role/str(seed)/(variant+'.npz')
            if sha(target) != row['sha256'] or sha(old_path) != original_records[key]['prediction_sha256']:
                raise ValueError('prediction bytes changed')
            with np.load(target, allow_pickle=False) as z, np.load(old_path, allow_pickle=False) as old:
                if not np.array_equal(z['ids'], old['ids']) or not np.array_equal(z['labels'], old['labels']):
                    raise ValueError('unpaired evaluation objects/order')
                logits, labels = z['logits'], z['labels']; stats = independent_metrics(logits, labels)
                domains = {name:independent_metrics(logits[mask], labels[mask]) for name, mask in
                           (('MACGS', labels < 30), ('ModelNet', labels >= 30))}
            error = max(abs(stats[k]-row['metrics'][k]) for k in ('ce', 'oa', 'mca'))
            if error > 1e-8 or not np.allclose(stats['class_accuracy'], row['metrics']['class_accuracy'], atol=1e-12, rtol=0):
                raise ValueError('independent metric discrepancy')
            maximum_error = max(maximum_error, error); final[key] = stats; outputs[variant] = logits
            records.append(dict(role=role, seed=seed, variant=variant, metrics=stats, domains=domains,
                prediction_sha256=row['sha256'], identical_policy_logit_error=row['identical_policy_logit_error']))
        permutation.append(dict(role=role, seed=seed, changes=int(np.sum(outputs['base'].argmax(1) != outputs['permuted'].argmax(1))),
            max_logit_difference=float(np.max(np.abs(outputs['base']-outputs['permuted'])))))
    comparisons = compare_policies(baseline, final)
    favorable = all(r['fixed_criterion'] for r in comparisons['final_gaussian_gaps'] if r['variant'] == 'base')
    result = dict(records=records, **comparisons, groups=len(records), final_base_descriptive_criterion=favorable,
        independent_metric_max_error=maximum_error, permutation=permutation, plan_sha256=sha(PLAN),
        seal_sha256=sha(DEST/'sealed.json'), original_primary_result_changed=False,
        fresh_confirmation=False, goal_complete=False, scientific_review_pending=True, visual_review_pending=True)
    write(DEST/'result.json', result)
    lines = ['# Final120 versus CE-selected checkpoints: used development data', '',
        'All15 existing fits and allsix conditions. No new training. This adaptive diagnostic does not rewrite the original negative result or constitute fresh confirmation.', '',
        '| Method | Selected mCA % | Final120 mCA % | Selected CE | Final120 CE |', '|---|---:|---:|---:|---:|']
    for role in ROLES:
        values = [np.mean([policy[role, seed, 'base'][metric] for seed in SEEDS])*(100 if metric == 'mca' else 1)
                  for metric in ('mca', 'ce') for policy in (baseline, final)]
        lines.append('| '+role+' | '+' | '.join(f'{v:.5f}' for v in values)+' |')
    lines += ['', 'Final Gaussian-minus-control gaps. Intervals resample classes after averaging the three paired seeds.', '',
        '| Condition | Control | Final gap pp | Paired seeds pp | Class95% pp | Change from selected gap pp |', '|---|---|---:|---|---|---:|']
    for gap, change in zip(comparisons['final_gaussian_gaps'], comparisons['gaussian_gap_changes']):
        lines.append(f"| {gap['variant']} | {gap['control']} | {gap['mean_pp']:+.4f} | "+'/'.join(f'{v:+.4f}' for v in gap['paired_pp'])+' | '+str(gap['class_bootstrap_95_pp'])+f" | {change['mean_pp']:+.4f} |")
    lines += ['', f'Final base descriptive criterion: {favorable}. Scientific and actual image review pending. Full role changes, source metrics, seed differences and gap interactions are in result.json.', '']
    (DEST/'report.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, role in enumerate(ROLES):
        row = next(r for r in comparisons['policy_changes'] if r['role'] == role and r['variant'] == 'base')
        axes[0].scatter(row['paired_pp'], np.full(3, i), color='tab:blue', s=20)
        axes[0].scatter(row['mean_pp'], i, color='black', marker='D', s=25)
    axes[0].set_yticks(range(len(ROLES)), ROLES); axes[0].set_xlabel('Final120 minus selected mCA (pp)')
    for i, control in enumerate(ROLES[1:]):
        old = next(r for r in original['comparisons'] if r['control'] == control and r['variant'] == 'base')
        new = next(r for r in comparisons['final_gaussian_gaps'] if r['control'] == control and r['variant'] == 'base')
        for offset, color, label, value, interval in [(-.12, 'gray', 'CE-selected', old['mca_gain_pp'], old['paired_class_bootstrap_95_pp']),
                                                     (.12, 'tab:orange', 'Final120', new['mean_pp'], new['class_bootstrap_95_pp'])]:
            axes[1].plot(interval, [i+offset]*2, color=color)
            axes[1].scatter(value, i+offset, color=color, label=label if i == 0 else None)
    axes[1].set_yticks(range(4), ROLES[1:]); axes[1].set_xlabel('Gaussian minus control mCA (pp)'); axes[1].legend()
    for ax in axes: ax.axvline(0, color='black', linewidth=.7); ax.invert_yaxis(); ax.grid(axis='x', alpha=.2)
    fig.suptitle('USED DEVELOPMENT | Fixed final120 policy | Base condition')
    fig.tight_layout(); fig.savefig(DEST/'checkpoint_policy.png', dpi=160); plt.close(fig)
    print('CHECKPOINT_POLICY_COMPLETE', len(records), favorable, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('plan', 'seal', 'export', 'report'))
    parser.add_argument('--seed', type=int, choices=SEEDS); args = parser.parse_args()
    if args.action == 'export' and args.seed is None: parser.error('export requires --seed')
    tag = args.action+(''+str(args.seed) if args.seed is not None else '')
    status = DEST/'workers'/(tag+'.json')
    if args.action != 'plan':
        if status.exists(): raise ValueError('inspect existing worker before restarting')
        ticks = (Path('/proc')/str(os.getpid())/'stat').read_text().rsplit(')', 1)[1].split()[19]
        identity = dict(pid=os.getpid(), start_ticks=ticks, action=args.action, seed=args.seed, started_unix=time.time())
        write(status, dict(identity, state='running'))
    try:
        {'plan':plan, 'seal':seal, 'export':lambda:export(args.seed), 'report':report}[args.action]()
    except BaseException as error:
        if args.action != 'plan': write(status, dict(identity, state='failed', error=repr(error), finished_unix=time.time()))
        raise
    if args.action != 'plan': write(status, dict(identity, state='complete', exit_code=0, finished_unix=time.time()))
