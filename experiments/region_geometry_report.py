"""Complete paired comparison, with original candidate and controls unchanged."""
from pathlib import Path
import numpy as np

from experiments.region_geometry_controls import OUT, PARENT, CANDIDATE, ROLES, SEEDS, CONDITIONS, check
from experiments import round68_fixed_region_run as engine
from experiments.round65_common import read, write, sha
from experiments.round68_fixed_region_data import load_cache

METRICS = ('region_recall', 'center_pck', 'soft_epe', 'ce')
OLD_ROLES = ('distance_mlp', 'direct_mlp', 'normal_scale', 'gaussian_relation',
             'shuffled_gaussian', 'relative5', 'metric3', 'spectrum7')
ALL_ROLES = OLD_ROLES + ROLES


def report():
    check()
    assert not (OUT/'result.json').exists()
    selected = read(OUT/'selected.json')
    rows = list(read(CANDIDATE/'result.json')['records'])
    assert len(rows) == 48
    for row in rows:
        root = CANDIDATE if row['role'] in ('relative5', 'metric3', 'spectrum7') else PARENT
        path = root/'evaluation'/row['role']/str(row['seed'])/(row['condition']+'.npz')
        assert sha(path) == row['prediction_sha256']
    verification = []
    caches = {condition:load_cache('eval_'+condition) for condition in CONDITIONS}
    for choice in selected['records']:
        dest = OUT/'evaluation'/choice['role']/str(choice['seed'])
        meta = read(dest/'record.json')
        assert meta['checkpoint_sha256'] == choice['checkpoint_sha256']
        assert meta['selection_sha256'] == sha(OUT/'selected.json')
        for row in meta['records']:
            path = dest/(row['condition']+'.npz')
            assert sha(path) == row['prediction_sha256']
            cache = caches[row['condition']]
            with np.load(path) as z:
                assert np.array_equal(z['ids'], cache['ids'])
                independent = engine.metric_torch(z['logits'], cache['overlap'], cache['world_centers'])
                grouped = engine.grouped_metrics(z['logits'], cache)
            error = max(abs(independent[k]-row['metrics']['micro'][k]) for k in METRICS)
            error = max(error, max(abs(grouped['macro'][k]-row['metrics']['macro'][k]) for k in METRICS))
            assert error < 1e-10 and grouped['micro']['valid_queries'] == 20422
            verification.append(dict(role=row['role'], seed=row['seed'], condition=row['condition'],
                independent_metric_error=error, valid_queries=20422,
                selected_replay=meta['validation_replay']))
            rows.append(row)
    lookup = {(x['role'], x['seed'], x['condition']):x['metrics'] for x in rows}
    assert len(rows) == len(lookup) == 60
    comparisons = []
    for control in (role for role in ALL_ROLES if role != 'relative5'):
        for condition in CONDITIONS:
            for stratum in (None, 0, 1, 2, 3):
                pairs, effects = [], []
                for seed in SEEDS:
                    left, right = (lookup[r, seed, condition] for r in ('relative5', control))
                    if stratum is not None:
                        left, right = left['strata'][str(stratum)], right['strata'][str(stratum)]
                    pair = {k:(left['macro'][k]-right['macro'][k])*(100 if k in METRICS[:2] else 1)
                            for k in METRICS}
                    pairs.append(dict(seed=seed, **pair))
                    scenes = list(left['by_scene'])
                    effects.append([100*(left['by_scene'][s]['region_recall']-
                                         right['by_scene'][s]['region_recall']) for s in scenes])
                scene_mean = np.mean(effects, axis=0)
                draw = np.random.default_rng(74001).integers(0, len(scenes), (10000, len(scenes)))
                interval = np.quantile(scene_mean[draw].mean(1), [.025, .975]).tolist()
                mean = {k:float(np.mean([r[k] for r in pairs])) for k in METRICS}
                comparisons.append(dict(candidate='relative5', control=control, condition=condition,
                    stratum=stratum, paired=pairs, mean=mean,
                    seed_scene_region_gain_pp=effects,
                    scene_mean_region_gain_pp=dict(zip(scenes, scene_mean.tolist())),
                    paired_scene_bootstrap_95_pp=interval,
                    descriptive_region_lead=bool(mean['region_recall'] >= 1 and
                                                 all(r['region_recall'] > 0 for r in pairs))))
    parameters = []
    for role in ALL_ROLES:
        root = OUT if role in ROLES else CANDIDATE if role in ('relative5', 'metric3', 'spectrum7') else PARENT
        for seed in SEEDS:
            run = read(root/'runs'/role/str(seed)/'run.json')
            parameters.append(dict(role=role, seed=seed, parameters=run['parameters'],
                                   pe_parameters=run['pe_parameters']))
    result = dict(records=rows, comparisons=comparisons, parameters=parameters,
        new_fits=6, reused_evaluation_groups=48, new_evaluation_groups=12,
        independent_verification=verification, lock_sha256=sha(OUT/'lock.json'),
        selections_sha256=sha(OUT/'selected.json'), source_sha256=sha(Path(__file__)),
        decision=dict(comparison_scope_extension=True, candidate_changed=False,
            base_lead_over_both_new_controls=all(r['descriptive_region_lead'] for r in comparisons
                if r['control'] in ROLES and r['condition'] == 'base' and r['stratum'] is None)),
        fresh_confirmation=False, goal_complete=False, scientific_review_pending=True,
        visual_review_pending=True,
        limits=['Used five scenes and three paired seeds; not fresh confirmation.',
                'Common full content and backbone; geometric controls have different PE capacities, reported explicitly.',
                'Official positional operator adapted to common backbone; not full GeoTransformer or published benchmark scores.',
                'Sparse observations preserve original region members, queries and labels.',
                'Previous native/RGB/fresh negative results are not changed.'])
    write(OUT/'result.json', result)
    lines = ['# 当前固定查询下的几何编码对照（待实际审阅）', '',
             '五维候选不变；六次新对照拟合，旧 R68/R69 结果按身份复用。', '',
             '| 条件 | 方法 | 区域 R@1 % | 中心 PCK % | Soft EPE | CE | PE 参数 |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for condition in CONDITIONS:
        for role in ALL_ROLES:
            values = [lookup[role, seed, condition]['macro'] for seed in SEEDS]
            metrics = [np.mean([r[k] for r in values])*(100 if k in METRICS[:2] else 1) for k in METRICS]
            count = next(r['pe_parameters'] for r in parameters if r['role'] == role)
            lines.append(f'| {condition} | {role} | '+' | '.join(f'{x:.4f}' for x in metrics)+f' | {count} |')
    lines += ['', '| 条件 / 对照 | 五维减对照三种子 R@1 pp | 平均 pp | 中心 PCK pp | 场景区间 pp |',
              '| --- | --- | ---: | ---: | --- |']
    for row in comparisons:
        if row['control'] not in ROLES or row['stratum'] is not None:
            continue
        paired = ' / '.join(f"{r['region_recall']:+.3f}" for r in row['paired'])
        lo, hi = row['paired_scene_bootstrap_95_pp']
        lines.append(f"| {row['condition']} / {row['control']} | {paired} | {row['mean']['region_recall']:+.3f} | {row['mean']['center_pck']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines += ['', '完整种子、场景、四分组、容量和指标回算见 result.json。',
              '基础用途描述沿用 R68 的平均至少 1 pp 且三种子正向；稀疏与中心结果单独报告。',
              '这次补对照不能覆盖 RGB、原生基元或新场景的负结果。不是整套 GeoTransformer 复现。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    plot(result, lookup)
    print('REGION_GEOMETRY_REPORTED', result['decision'], flush=True)


def plot(result, lookup):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    roles = ('distance_mlp', 'direct_mlp', 'normal_scale', 'geotransformer_rpe', 'angle_descriptor_mlp', 'relative5')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    for ax, condition in zip(axes, CONDITIONS):
        for i, role in enumerate(roles):
            values = [100*lookup[role, seed, condition]['macro']['region_recall'] for seed in SEEDS]
            ax.scatter([i]*3, values)
            ax.plot([i-.2, i+.2], [np.mean(values)]*2, color='black')
        ax.set_xticks(range(len(roles)), roles, rotation=30, ha='right')
        ax.set_title(condition)
        ax.set_ylabel('Scene-equal region R@1 (%)')
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Fixed queries and full common content; dots = seeds, lines = means')
    fig.savefig(OUT/'comparison.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout='constrained')
    rows = [r for r in result['comparisons'] if r['control'] in ROLES and r['stratum'] is None]
    for ax, metric in zip(axes, ('region_recall', 'center_pck')):
        for i, row in enumerate(rows):
            ax.scatter([i]*3, [p[metric] for p in row['paired']])
            ax.plot([i-.2, i+.2], [row['mean'][metric]]*2, color='black')
        ax.axhline(0, color='gray')
        ax.set_xticks(range(len(rows)), [r['control']+' / '+r['condition'] for r in rows], rotation=30, ha='right')
        ax.set_ylabel('relative5 minus control (pp)')
        ax.set_title(metric)
    fig.savefig(OUT/'paired_metrics.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    from experiments.region_geometry_controls import bind
    bind()
    report()
