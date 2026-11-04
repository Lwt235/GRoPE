"""Complete fixed-covariance attribution report, without replacing old claims."""
from pathlib import Path
import numpy as np
from experiments.region_shape_attribution import OUT, CANDIDATE, PARENT, ROLES, SEEDS, CONDITIONS, check, engine, data_module
from experiments.round65_common import read, write, sha

METRICS = ('region_recall', 'center_pck', 'soft_epe', 'ce')


def report():
    check()
    assert not (OUT / 'result.json').exists()
    selected = read(OUT / 'selected.json')
    assert selected['all_three_selected_before_eval']
    rows = [r for r in read(CANDIDATE / 'result.json')['records']
            if r['role'] in ('relative5', 'distance_mlp')]
    assert len(rows) == 12
    for row in rows:
        root = CANDIDATE if row['role'] == 'relative5' else PARENT
        path = root / 'evaluation' / row['role'] / str(row['seed']) / (row['condition'] + '.npz')
        assert sha(path) == row['prediction_sha256']
    verification = []
    caches = {c: data_module.load_cache('eval_' + c) for c in CONDITIONS}
    for choice in selected['records']:
        dest = OUT / 'evaluation' / ROLES[0] / str(choice['seed'])
        meta = read(dest / 'record.json')
        assert meta['selection_sha256'] == sha(OUT / 'selected.json')
        assert meta['checkpoint_sha256'] == choice['checkpoint_sha256']
        for row in meta['records']:
            cache, path = caches[row['condition']], dest / (row['condition'] + '.npz')
            assert sha(path) == row['prediction_sha256']
            with np.load(path) as z:
                assert np.array_equal(z['ids'], cache['ids'])
                independent = engine.metric_torch(z['logits'], cache['overlap'], cache['world_centers'])
                grouped = engine.grouped_metrics(z['logits'], cache)
            error = max(abs(independent[k] - row['metrics']['micro'][k]) for k in METRICS)
            error = max(error, max(abs(grouped['macro'][k] - row['metrics']['macro'][k]) for k in METRICS))
            assert error < 1e-10 and grouped['micro']['valid_queries'] == 20422
            verification.append(dict(seed=choice['seed'], condition=row['condition'],
                independent_metric_error=error, valid_queries=20422, selected_replay=meta['validation_replay']))
            rows.append(row)
    lookup = {(r['role'], r['seed'], r['condition']): r['metrics'] for r in rows}
    assert len(rows) == len(lookup) == 18
    comparisons = []
    for left, right in (('relative5', ROLES[0]), (ROLES[0], 'distance_mlp')):
        for condition in CONDITIONS:
            for stratum in (None, 0, 1, 2, 3):
                paired, effects = [], []
                for seed in SEEDS:
                    a, b = lookup[left, seed, condition], lookup[right, seed, condition]
                    if stratum is not None:
                        a, b = a['strata'][str(stratum)], b['strata'][str(stratum)]
                    paired.append(dict(seed=seed, **{k:(a['macro'][k] - b['macro'][k]) *
                        (100 if k in METRICS[:2] else 1) for k in METRICS}))
                    scenes = list(a['by_scene'])
                    assert set(scenes) == set(b['by_scene'])
                    effects.append([100 * (a['by_scene'][s]['region_recall'] - b['by_scene'][s]['region_recall']) for s in scenes])
                scene_mean = np.mean(effects, axis=0)
                draws = np.random.default_rng(76001).integers(0, len(scenes), (10000, len(scenes)))
                interval = np.quantile(scene_mean[draws].mean(1), [.025, .975]).tolist()
                comparisons.append(dict(left=left, right=right, condition=condition, stratum=stratum,
                    paired=paired, mean={k:float(np.mean([r[k] for r in paired])) for k in METRICS},
                    seed_scene_region_gain_pp=effects, scene_mean_region_gain_pp=dict(zip(scenes, scene_mean.tolist())),
                    paired_scene_bootstrap_95_pp=interval))
    primary = next(r for r in comparisons if r['left'] == 'relative5' and r['condition'] == 'base' and r['stratum'] is None)
    result = dict(records=rows, comparisons=comparisons, new_fits=3, new_evaluation_groups=6,
        reused_evaluation_groups=12, independent_verification=verification,
        lock_sha256=sha(OUT / 'lock.json'), selections_sha256=sha(OUT / 'selected.json'),
        source_sha256=sha(Path(__file__)),
        decision=dict(primary_covariance_positive_pattern=bool(all(r['region_recall'] > 0 for r in primary['paired'])
            and primary['paired_scene_bootstrap_95_pp'][0] > 0), automatic_promotion=False),
        interpretation_limits=['Old data, three seeds; scene intervals condition on seeds.',
            'Full covariance remains in common content. The control removes its additional PE dependence only.',
            'The control keeps five-input optimization parameterization; an independently trained two-input MLP is not equivalent in optimization.',
            'Preserve native, RGB and fresh projection mixed outcomes; no global goal completion.'],
        scientific_review_pending=True, visual_review_pending=True, fresh_confirmation=False, goal_complete=False)
    write(OUT / 'result.json', result)
    lines = ['# 区域匹配固定球形协方差对照（待实际审阅）', '',
        '仅把 PE 输入中的两端协方差固定为 .2²I；模型、完整共同内容、参数量、初始化和旧训练预算不变。', '',
        '| 条件 | 方法 | 区域 R@1 % | 中心 PCK % | Soft EPE | CE |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    roles = ('distance_mlp', ROLES[0], 'relative5')
    for condition in CONDITIONS:
        for role in roles:
            metrics = [np.mean([lookup[role, s, condition]['macro'][k] for s in SEEDS]) *
                       (100 if k in METRICS[:2] else 1) for k in METRICS]
            lines.append(f'| {condition} | {role} | ' + ' | '.join(f'{v:.5f}' for v in metrics) + ' |')
    lines += ['', '| 条件 / 左减右 | 三种子 R@1 差 pp | 平均差 pp | 中心 PCK 差 pp | 场景区间 pp |',
              '| --- | --- | ---: | ---: | --- |']
    for row in comparisons:
        if row['stratum'] is not None:
            continue
        lo, hi = row['paired_scene_bootstrap_95_pp']
        values = ' / '.join(f"{r['region_recall']:+.3f}" for r in row['paired'])
        lines.append(f"| {row['condition']} / {row['left']} − {row['right']} | {values} | "
                     f"{row['mean']['region_recall']:+.3f} | {row['mean']['center_pck']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines += ['', '完整逐种子、逐场景与四分组结果在 result.json。该对照不删除共同内容中的形状，也不构成新数据确认。',
              '没有额外最低效应门槛；连续报告差异和不确定性，不因均值接近而宣称正式等效。']
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout='constrained')
    for ax, condition in zip(axes, CONDITIONS):
        for i, role in enumerate(roles):
            values = [100 * lookup[role, s, condition]['macro']['region_recall'] for s in SEEDS]
            ax.scatter([i] * 3, values)
            ax.plot([i - .2, i + .2], [np.mean(values)] * 2, color='black')
        ax.set_xticks(range(len(roles)), roles, rotation=20, ha='right')
        ax.set_title(condition)
        ax.set_ylabel('Scene-equal region R@1 (%)')
        ax.grid(axis='y', alpha=.2)
    fig.savefig(OUT / 'comparison.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout='constrained')
    primary_rows = [r for r in comparisons if r['left'] == 'relative5' and r['stratum'] is None]
    for ax, metric in zip(axes, METRICS[:2]):
        for i, row in enumerate(primary_rows):
            ax.scatter([i] * 3, [r[metric] for r in row['paired']])
            ax.plot([i - .2, i + .2], [row['mean'][metric]] * 2, color='black')
        ax.axhline(0, color='gray')
        ax.set_xticks([0, 1], CONDITIONS)
        ax.set_ylabel('relative5 minus fixed covariance (pp)')
        ax.set_title(metric)
    fig.savefig(OUT / 'paired_metrics.png', dpi=150)
    plt.close(fig)
    print('REGION_SHAPE_REPORTED', result['decision'], flush=True)
