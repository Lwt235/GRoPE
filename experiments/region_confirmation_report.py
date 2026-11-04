"""Complete fixed-cohort reporting, paired uncertainty and independent metrics."""
from pathlib import Path
import time

import numpy as np

from experiments import round68_fixed_region_run as engine
from experiments.region_confirmation_common import (
    OUT, ROLES, SEEDS, CONDITIONS, UTILITY_CONTROLS, read, write, sha, preserved, check)
from experiments.region_confirmation_data import load

METRICS = ('region_recall', 'center_pck', 'soft_epe', 'ce')


def population_metrics(logits, cache, planned_scenes):
    result = engine.grouped_metrics(logits, cache)
    empty = dict.fromkeys(METRICS)
    empty.update(valid_queries=0, total_queries=0, coverage=0., valid_and_correct_fraction=0.)
    for group in [result] + list(result['strata'].values()):
        for scene in planned_scenes:
            if scene not in group['by_scene']:
                group['by_scene'][scene] = dict(empty)
        group['by_scene'] = {s: group['by_scene'][s] for s in planned_scenes}
    return result


def independent_error(logits, cache, metrics):
    errors = []
    for scene in (None, *metrics['by_scene']):
        index = np.ones(len(cache['ids']), bool) if scene is None else cache['scenes'] == scene
        expected = metrics['micro'] if scene is None else metrics['by_scene'][scene]
        if not expected['valid_queries']:
            continue
        actual = engine.metric_torch(logits[index], cache['overlap'][index], cache['world_centers'][index])
        errors.extend(abs(actual[k] - expected[k]) for k in METRICS)
    return float(max(errors, default=0.))


def compare(rows, candidate, control, condition, group=None):
    lookup = {(r['role'], r['seed'], r['condition']): r['metrics'] for r in rows}
    pairs, scene_gains = [], []
    observed_scenes = None
    for seed in SEEDS:
        a, b = lookup[candidate, seed, condition], lookup[control, seed, condition]
        if group is not None:
            a, b = a['strata'][str(group)], b['strata'][str(group)]
        values = {k: None if a['macro'][k] is None or b['macro'][k] is None else
                  (a['macro'][k] - b['macro'][k]) * (100 if k in METRICS[:2] else 1) for k in METRICS}
        pairs.append(dict(seed=seed, **values))
        scenes = [s for s in a['by_scene'] if a['by_scene'][s]['valid_queries'] and b['by_scene'][s]['valid_queries']]
        if observed_scenes is None:
            observed_scenes = scenes
        assert scenes == observed_scenes
        scene_gains.append([100 * (a['by_scene'][s]['region_recall'] - b['by_scene'][s]['region_recall']) for s in scenes])
    mean = {k: None if any(p[k] is None for p in pairs) else float(np.mean([p[k] for p in pairs])) for k in METRICS}
    scene_interval, crossed_interval, averages = None, None, {}
    if observed_scenes:
        array = np.asarray(scene_gains)
        averages = dict(zip(observed_scenes, array.mean(0).tolist()))
        rng = np.random.default_rng(78001)
        scene_draw = rng.integers(0, len(observed_scenes), (10000, len(observed_scenes)))
        scene_interval = np.quantile(array.mean(0)[scene_draw].mean(1), [.025, .975]).tolist()
        seed_draw = rng.integers(0, len(SEEDS), (10000, len(SEEDS)))
        crossed_interval = np.quantile(array[seed_draw[:, :, None], scene_draw[:, None, :]].mean((1, 2)), [.025, .975]).tolist()
    return dict(candidate=candidate, control=control, condition=condition, stratum=group, paired=pairs, mean=mean,
        scene_mean_region_gains_pp=averages, evaluable_scenes=observed_scenes,
        paired_scene_bootstrap95_pp=scene_interval, crossed_seed_scene_bootstrap95_pp=crossed_interval)


def positive_seeds(row):
    return all(p['region_recall'] is not None and p['region_recall'] > 0 for p in row['paired'])


def report():
    lock = check()
    assert not (OUT / 'result.json').exists()
    manifest = read(OUT / 'data_manifest.json')
    caches = {c: load(c) for c in CONDITIONS}
    for k in ('ids', 'scenes', 'overlap', 'world_centers', 'strata', 'permutation'):
        assert np.array_equal(caches['base'][k], caches['sparse256'][k]), k
    rows, errors = [], []
    for role in ROLES:
        for seed in SEEDS:
            dest = OUT / 'predictions' / role / str(seed)
            identity = read(dest / 'identity.json')
            assert identity['checkpoint_sha256'] == lock['checkpoints'][f'{role}/{seed}']['sha256']
            assert identity['evaluation_lock_sha256'] == sha(OUT / 'evaluation_lock.json')
            assert identity['data_manifest_sha256'] == sha(OUT / 'data_manifest.json')
            for condition in CONDITIONS:
                path = dest / (condition + '.npz')
                item = next(r for r in identity['records'] if r['condition'] == condition)
                assert sha(path) == item['prediction_sha256']
                cache = caches[condition]
                with np.load(path, allow_pickle=False) as z:
                    assert np.array_equal(z['ids'], cache['ids'])
                    metrics = population_metrics(z['logits'], cache, lock['planned_scenes'])
                    error = independent_error(z['logits'], cache, metrics)
                assert error < 1e-10, (role, seed, condition, error)
                errors.append(error)
                rows.append(dict(item, metrics=metrics, independent_metric_error=error))
    assert len(rows) == 48
    comparisons = [compare(rows, 'relative5', role, condition) for condition in CONDITIONS for role in ROLES if role != 'relative5']
    strata = [compare(rows, 'relative5', role, condition, group) for group in range(4)
              for condition in CONDITIONS for role in ROLES if role != 'relative5']
    radial = [compare(rows, 'fixed_isotropic5', 'distance_mlp', c) for c in CONDITIONS]
    by_control = {r['control']: r for r in comparisons if r['condition'] == 'base'}
    utility = {role: bool(positive_seeds(by_control[role]) and by_control[role]['mean']['region_recall'] >= 1)
               for role in UTILITY_CONTROLS}
    shape = by_control['fixed_isotropic5']
    covariance = bool(positive_seeds(shape) and shape['paired_scene_bootstrap95_pp'] and shape['paired_scene_bootstrap95_pp'][0] > 0)
    compact = {r['condition']: bool(r['mean']['region_recall'] is not None and r['mean']['region_recall'] >= -1
                 and r['mean']['center_pck'] >= -1) for r in comparisons if r['control'] == 'gaussian_relation'}
    all_valid = all(v['valid_queries'] > 0 for v in manifest['caches']['base']['by_scene'].values())
    result = dict(records=rows, comparisons=comparisons, strata_comparisons=strata, fixed_minus_distance=radial,
        planned_scenes=lock['planned_scenes'], available_scenes=lock['scenes'], unavailable=lock['unavailable'],
        data_coverage=manifest['caches'], technical_pair_exclusions=manifest['excluded'],
        decision=dict(scope='Available members of the predeclared cohort; missing members are not replaced.',
            utility_pattern_by_control_on_available_scenes=utility,
            all_utility_patterns_on_available_scenes=all(utility.values()),
            covariance_pattern_on_available_scenes=covariance, compact_engineering_tolerance=compact,
            complete_planned_cohort_evaluated=bool(lock['complete_cohort'] and all_valid and not manifest['excluded']),
            center_metrics_require_joint_interpretation=True),
        independent_metric_max_error=max(errors), source_lock_sha256=sha(OUT / 'evaluation_lock.json'),
        data_manifest_sha256=sha(OUT / 'data_manifest.json'), fresh_cohort_now_used=True,
        new_fits=0, scientific_review_pending=True, visual_review_pending=True, goal_complete=False)
    preserved(OUT / 'result.json', result)
    def fmt(value, digits=4):
        return 'missing' if value is None else f'{value:.{digits}f}'
    lines = ['# Frozen regional confirmation', '',
        f"Available scenes: {len(lock['scenes'])}/{len(lock['planned_scenes'])}; 24 existing models; zero new fits.",
        'Missing members remain missing. This is not complete ten-place confirmation or evidence of general PE effectiveness.', '',
        'Unavailable: ' + '; '.join(r['scene'] + ': ' + r['message'] for r in lock['unavailable']), '',
        '| Condition | Method | Region R@1 % | Center PCK % | Soft EPE | CE |',
        '|---|---|---:|---:|---:|---:|']
    lookup = {(r['role'], r['seed'], r['condition']): r['metrics']['macro'] for r in rows}
    for condition in CONDITIONS:
        for role in ROLES:
            values = {k: [lookup[role, s, condition][k] for s in SEEDS] for k in METRICS}
            means = {k: None if any(v is None for v in values[k]) else np.mean(values[k]) for k in METRICS}
            entries = [None if means[k] is None else means[k] * (100 if k in METRICS[:2] else 1) for k in METRICS]
            lines.append(f'| {condition} | {role} | ' + ' | '.join(fmt(v) for v in entries) + ' |')
    lines += ['', '| Condition | relative5 minus control | Seed differences pp | Mean pp | Scene 95% pp | PCK pp |',
              '|---|---|---|---:|---|---:|']
    for r in comparisons:
        interval = r['paired_scene_bootstrap95_pp']
        ci = 'missing' if interval is None else '[' + ', '.join(fmt(x, 3) for x in interval) + ']'
        lines.append(f"| {r['condition']} | {r['control']} | " + ' / '.join(fmt(p['region_recall'], 3) for p in r['paired']) +
                     f" | {fmt(r['mean']['region_recall'], 3)} | {ci} | {fmt(r['mean']['center_pck'], 3)} |")
    lines += ['', f'Independent micro/scene metric maximum error: {max(errors):.3g}.',
        'Scene intervals condition on three selected seeds. Crossed intervals, all strata, scenes, seeds and coverage are retained in result.json.',
        'Sparse inputs retain known original regions and physical queries. Technical missingness is not silently removed from the planned population.',
        'Scientific interpretation and actual figure review are pending.', '', '![Comparison](comparison.png)', '', '![Paired differences](paired.png)', '']
    (OUT / 'report.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#2f6f9f' if r == 'relative5' else '#d29539' if r == 'fixed_isotropic5' else '#939b9f' for r in ROLES]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, condition in zip(axes, CONDITIONS):
        values = np.array([[lookup[r, s, condition]['region_recall'] for s in SEEDS] for r in ROLES], float) * 100
        ax.barh(np.arange(len(ROLES)), values.mean(1), color=colors)
        for i in range(len(ROLES)):
            ax.scatter(values[i], [i] * 3, s=18, color='black', zorder=3)
        ax.set_yticks(np.arange(len(ROLES)), ROLES)
        ax.invert_yaxis()
        ax.set(xlabel='Scene-equal region R@1 (%)', title=condition)
    fig.suptitle(f"Regional confirmation: {len(lock['scenes'])}/10 available places; dots = fixed training seeds")
    fig.savefig(OUT / 'comparison.png', dpi=160)
    plt.close(fig)
    controls = [r for r in ROLES if r not in ('relative5', 'gaussian_relation')]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, condition in zip(axes, CONDITIONS):
        for i, control in enumerate(controls):
            row = next(r for r in comparisons if r['condition'] == condition and r['control'] == control)
            values = [p['region_recall'] for p in row['paired']]
            if any(v is None for v in values):
                continue
            lo, hi = row['paired_scene_bootstrap95_pp']
            ax.plot([lo, hi], [i, i], color='#2f6f9f', lw=3)
            ax.scatter(values, [i] * 3, color=['#2b6b9b', '#b56120', '#477d41'], zorder=3)
        ax.axvline(0, color='black', lw=.8)
        ax.set_yticks(np.arange(len(controls)), controls)
        ax.invert_yaxis()
        ax.set(xlabel='relative5 minus control (pp)', title=condition)
    fig.suptitle('Paired seeds and conditional scene intervals; all predeclared comparisons retained')
    fig.savefig(OUT / 'paired.png', dpi=160)
    plt.close(fig)
    print('REGION_REPORT_COMPLETE', 48, 'groups; review pending', flush=True)
