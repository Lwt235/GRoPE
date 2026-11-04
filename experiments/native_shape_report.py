"""Native variable-covariance attribution across all original conditions."""
import numpy as np
from experiments.native_shape_attribution import OUT, NATIVE, OLD, ROLES, SEEDS, VARIANTS, check, audit
from experiments.round65_common import read, write, sha


def report():
    check()
    assert not (OUT / 'result.json').exists()
    selected = read(OUT / 'selected.json')
    assert selected['all_three_selected_before_development_evaluation'] and len(selected['records']) == 3
    manifest = read(OUT / 'data/manifest.json')
    partition = next(r for r in manifest['partitions'] if r['split'] == 'dev_eval')
    ids, labels = np.asarray(partition['ids']), np.load(OUT / 'data/dev_eval/labels.npy')
    rows = [dict(r, reused=True) for r in read(NATIVE / 'result.json')['records']]
    assert len(rows) == 108
    for r in rows:
        root = NATIVE if r['role'] == 'relative5' else OLD
        path = root / 'evaluation' / r['role'] / str(r['seed']) / (r['variant'] + '.npz')
        assert sha(path) == r['prediction_sha256']
    worst, permutation_logits = 0., {}
    for choice in selected['records']:
        seed = choice['seed']
        dest = OUT / 'evaluation' / ROLES[0] / str(seed)
        meta = read(dest / 'record.json')
        assert meta['selection_sha256'] == sha(OUT / 'selected.json')
        assert meta['data_sha256'] == sha(OUT / 'data/manifest.json')
        assert meta['checkpoint_sha256'] == choice['checkpoint_sha256']
        assert [r['variant'] for r in meta['records']] == list(VARIANTS)
        for r in meta['records']:
            path = dest / (r['variant'] + '.npz')
            assert sha(path) == r['sha256']
            with np.load(path) as z:
                assert np.array_equal(z['ids'], ids) and np.array_equal(z['labels'], labels)
                logits = z['logits']
            metrics = audit.independent_metrics(logits, labels)
            error = max(abs(metrics[k] - r['metrics'][k]) for k in ('mca', 'oa', 'ce'))
            error = max(error, float(np.max(np.abs(np.asarray(metrics['class_accuracy']) - r['metrics']['class_accuracy']))))
            assert error < 1e-8
            worst = max(worst, error)
            domains = {name:audit.independent_metrics(logits[mask], labels[mask])
                       for name, mask in (('MACGS', labels < 30), ('ModelNet', labels >= 30))}
            rows.append(dict(role=ROLES[0], seed=seed, variant=r['variant'], metrics=metrics,
                             domains=domains, prediction_sha256=r['sha256'], reused=False))
            if r['variant'] in ('base', 'permuted'):
                permutation_logits[seed, r['variant']] = logits
    lookup = {(r['role'], r['seed'], r['variant']):r for r in rows}
    assert len(lookup) == len(rows) == 126
    comparisons = []
    for left, right in (('relative5', ROLES[0]), (ROLES[0], 'distance_mlp')):
        for variant in VARIANTS:
            a, b = [np.asarray([lookup[role, s, variant]['metrics']['class_accuracy'] for s in SEEDS]) for role in (left, right)]
            delta = 100 * (a - b)
            paired = delta.mean(1)
            interval = audit.bootstrap(delta)
            domains = {name:[100 * (lookup[left, s, variant]['domains'][name]['mca'] -
                                   lookup[right, s, variant]['domains'][name]['mca']) for s in SEEDS]
                       for name in ('MACGS', 'ModelNet')}
            comparisons.append(dict(left=left, right=right, variant=variant,
                mean_gain_pp=float(paired.mean()), paired_seed_gains_pp=paired.tolist(),
                paired_class_bootstrap_95_pp=interval, class_mean_gain_pp=delta.mean(0).tolist(),
                domains=domains, paired_oa_gain_pp=[100 * (lookup[left, s, variant]['metrics']['oa'] -
                    lookup[right, s, variant]['metrics']['oa']) for s in SEEDS],
                paired_ce_difference=[lookup[left, s, variant]['metrics']['ce'] - lookup[right, s, variant]['metrics']['ce'] for s in SEEDS]))
    primary = next(r for r in comparisons if r['left'] == 'relative5' and r['variant'] == 'base')
    permutation = []
    for seed in SEEDS:
        a, b = permutation_logits[seed, 'base'], permutation_logits[seed, 'permuted']
        permutation.append(dict(seed=seed, maximum_logit_error=float(abs(a - b).max()),
                                prediction_changes=int((a.argmax(1) != b.argmax(1)).sum())))
    result = dict(records=rows, comparisons=comparisons, permutation=permutation,
        primary_covariance_positive_pattern=bool(all(v > 0 for v in primary['paired_seed_gains_pp']) and primary['paired_class_bootstrap_95_pp'][0] > 0),
        original_R73_criterion_unchanged=True, original_R73_result_sha256=sha(NATIVE / 'result.json'),
        new_groups=18, reused_groups=108, independent_metric_max_error=worst,
        lock_sha256=sha(OUT / 'lock.json'), selection_sha256=sha(OUT / 'selected.json'),
        scientific_review_pending=True, visual_review_pending=True, fresh_confirmation=False, goal_complete=False)
    write(OUT / 'result.json', result)
    lines = ['# 原生 3DGS 固定球形协方差对照（待实际审阅）', '',
        '三个新拟合，原五维及 R65 控制按身份复用；不改写原 R73 结论，不是官方 TEST 或新数据确认。', '',
        '| 方法 | Base mCA 三种子 % | 平均 mCA % | OA % | CE |', '| --- | --- | ---: | ---: | ---: |']
    for role in ('distance_mlp', ROLES[0], 'relative5', 'direct_mlp', 'gaussian_relation'):
        m = [lookup[role, s, 'base']['metrics'] for s in SEEDS]
        lines.append(f'| {role} | ' + ' / '.join(f"{100 * r['mca']:.3f}" for r in m) +
                     f" | {100 * np.mean([r['mca'] for r in m]):.3f} | {100 * np.mean([r['oa'] for r in m]):.3f} | {np.mean([r['ce'] for r in m]):.5f} |")
    lines += ['', '| 条件 | 左减右 | 三种子 mCA 差 pp | 平均 pp | 类别区间 pp |', '| --- | --- | --- | ---: | --- |']
    for r in comparisons:
        lo, hi = r['paired_class_bootstrap_95_pp']
        values = ' / '.join(f'{v:+.3f}' for v in r['paired_seed_gains_pp'])
        lines.append(f"| {r['variant']} | {r['left']} − {r['right']} | {values} | {r['mean_gain_pp']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines += ['', '类别区间条件于三个已选种子；MACGS 与 ModelNet 类别不同，来源组结果不是跨域因果证明。',
              '完整形状仍在共同内容中；该对照只检验 PE 中变化协方差的增量作用。全部条件、类别、OA/CE 与排列结果均保留。']
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for i, r in enumerate(v for v in comparisons if v['variant'] == 'base'):
        axes[0].scatter(r['paired_seed_gains_pp'], [i] * 3)
        axes[0].plot(r['paired_class_bootstrap_95_pp'], [i] * 2, color='black')
    axes[0].axvline(0, color='gray')
    axes[0].set(yticks=[0, 1], yticklabels=['relative5 − fixed', 'fixed − Distance'], xlabel='mCA difference (pp)', title='Base: paired seeds; conditional class interval')
    for i, variant in enumerate(VARIANTS):
        r = next(r for r in comparisons if r['left'] == 'relative5' and r['variant'] == variant)
        axes[1].scatter([i] * 3, r['paired_seed_gains_pp'])
        axes[1].plot([i - .2, i + .2], [r['mean_gain_pp']] * 2, color='black')
    axes[1].axhline(0, color='gray')
    axes[1].set(xticks=range(len(VARIANTS)), xticklabels=VARIANTS, ylabel='relative5 minus fixed covariance (pp)')
    axes[1].tick_params(axis='x', rotation=25)
    fig.savefig(OUT / 'comparison.png', dpi=150)
    plt.close(fig)
    print('NATIVE_SHAPE_REPORTED', result['primary_covariance_positive_pattern'], flush=True)
