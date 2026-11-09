"""Development-only closing control; reuse metric definitions, not old scores."""
import numpy as np

from experiments import normalized_precision as exp
from experiments import round65_audit as native_audit
from experiments.mean_precision_report import METRICS, region_metrics, scene_summary
from experiments.native_seed_replication_report import paired_summary
from experiments.prediction_success_failure import describe, region_values
from experiments.round65_common import SEEDS, VARIANTS, read, write, sha

ROLES = ('relative5', 'distance_mlp', 'trace_isotropic5', 'mean_precision5', 'normalized_precision5')
PAIRS = (('relative5', 'normalized_precision5'), ('normalized_precision5', 'distance_mlp'),
         ('normalized_precision5', 'trace_isotropic5'), ('normalized_precision5', 'mean_precision5'),
         ('relative5', 'distance_mlp'))


def records():
    engine = exp.engine
    engine.bind('region')
    caches = {c: engine.region.data_module.load_cache('eval_' + c) for c in engine.region.CONDITIONS}
    data = exp.NATIVE_PRIOR['native512']['distance_mlp'] / 'data'
    labels = np.load(data / 'dev_eval/labels.npy', allow_pickle=False)
    ids = np.asarray(next(p['ids'] for p in read(data / 'manifest.json')['partitions'] if p['split'] == 'dev_eval'))
    rows, permutation = [], []
    for case in exp.CASES:
        mapping = exp.REGION_PRIOR if case == 'region' else exp.NATIVE_PRIOR[case]
        conditions = engine.region.CONDITIONS if case == 'region' else VARIANTS
        manifest = (exp.REGION_PRIOR['distance_mlp'] / 'data/manifest.json'
                    if case == 'region' else data / 'manifest.json')
        for role in ROLES:
            root = mapping[role] if role in mapping else exp.OUT / case
            selection = read(root / 'selected.json')
            for seed in SEEDS:
                choice = next(r for r in selection['records'] if (r['role'], r['seed']) == (role, seed))
                directory = root / 'evaluation' / role / str(seed)
                meta = read(directory / 'record.json')
                assert meta['selection_sha256'] == sha(root / 'selected.json')
                assert meta['checkpoint_sha256'] == choice['checkpoint_sha256']
                assert sha(root / 'runs' / role / str(seed) / 'best.pt') == choice['checkpoint_sha256']
                assert meta['manifest_sha256' if case == 'region' else 'data_sha256'] == sha(manifest)
                condition_key = 'condition' if case == 'region' else 'variant'
                assert [r[condition_key] for r in meta['records']] == list(conditions)
                saved = {}
                for entry in meta['records']:
                    condition = entry[condition_key]
                    path = directory / (condition + '.npz')
                    digest = entry['prediction_sha256' if case == 'region' else 'sha256']
                    assert sha(path) == digest
                    with np.load(path, allow_pickle=False) as archive:
                        assert np.array_equal(archive['ids'], caches[condition]['ids'] if case == 'region' else ids)
                        if case != 'region':
                            assert np.array_equal(archive['labels'], labels)
                        logits = archive['logits']
                    assert np.isfinite(logits).all()
                    if case == 'region':
                        cache = caches[condition]
                        metrics = region_metrics(logits, cache)
                        independent = engine.region.engine.metric_torch(logits, cache['overlap'], cache['world_centers'])
                        error = max(abs(independent[k] - metrics['micro'][k]) for k in METRICS)
                        for group in (None, '0', '1', '2', '3'):
                            a, b = metrics, entry['metrics']
                            if group is not None:
                                a, b = a['strata'][group], b['strata'][group]
                            for scope in ('micro', 'macro'):
                                for key in ('region_recall', 'center_pck', 'soft_epe', 'ce'):
                                    if a[scope][key] is None:
                                        assert b[scope][key] is None
                                    else:
                                        error = max(error, abs(a[scope][key] - b[scope][key]))
                        assert error < 1e-10 and metrics['micro']['valid_queries'] == 20422
                        extra = dict(validation_replay=meta['validation_replay'])
                    else:
                        metrics = native_audit.independent_metrics(logits, labels)
                        error = max(abs(metrics[k] - entry['metrics'][k]) for k in ('mca', 'oa', 'ce'))
                        error = max(error, float(np.max(np.abs(np.asarray(metrics['class_accuracy']) - entry['metrics']['class_accuracy']))))
                        assert error < 1e-8
                        extra = dict(domains={name: native_audit.independent_metrics(logits[mask], labels[mask])
                                     for name, mask in (('MACGS', labels < 30), ('ModelNet', labels >= 30))})
                        if condition in ('base', 'permuted'):
                            saved[condition] = logits
                    rows.append(dict(case=case, role=role, seed=seed, condition=condition, metrics=metrics,
                        reused=role in mapping, prediction_path=str(path), prediction_sha256=digest,
                        independent_metric_error=error, **extra))
                if case != 'region':
                    a, b = saved['base'], saved['permuted']
                    permutation.append(dict(role=role, seed=seed, maximum_logit_error=float(np.abs(a-b).max()),
                                            prediction_changes=int((a.argmax(1) != b.argmax(1)).sum())))
    assert len(rows) == 120 and sum(r['reused'] for r in rows) == 96
    return rows, permutation


def comparisons_for(rows):
    lookup = {(r['case'], r['role'], r['seed'], r['condition']): r for r in rows}
    assert len(lookup) == len(rows) == 120
    comparisons = []
    for condition in exp.engine.region.CONDITIONS:
        for left, right in PAIRS:
            for group in (None, '0', '1', '2', '3'):
                paired, effects = [], []
                for seed in SEEDS:
                    a, b = [lookup['region', role, seed, condition]['metrics'] for role in (left, right)]
                    if group is not None:
                        a, b = a['strata'][group], b['strata'][group]
                    scenes = [s for s in sorted(a['by_scene']) if a['by_scene'][s]['valid_queries']]
                    assert scenes == [s for s in sorted(b['by_scene']) if b['by_scene'][s]['valid_queries']]
                    paired.append(dict(seed=seed, **{k: (a['macro'][k]-b['macro'][k]) *
                                  (100 if k in METRICS[:2] else 1) for k in METRICS}))
                    effects.append([100*(a['by_scene'][s]['region_recall']-b['by_scene'][s]['region_recall']) for s in scenes])
                random_seed = 87001 + len(comparisons)
                comparisons.append(dict(case='region', left=left, right=right, condition=condition,
                    stratum=group, primary=condition == 'base' and group is None, scenes=scenes,
                    bootstrap_seed=random_seed, paired=paired,
                    mean={k: float(np.mean([p[k] for p in paired])) for k in METRICS},
                    **scene_summary(effects, random_seed)))
    for condition in VARIANTS:
        for left, right in PAIRS:
            a, b = [[lookup['native512', role, seed, condition] for seed in SEEDS] for role in (left, right)]
            differences = 100*(np.asarray([r['metrics']['class_accuracy'] for r in a])-
                              np.asarray([r['metrics']['class_accuracy'] for r in b]))
            random_seed = 87001 + len(comparisons)
            comparisons.append(dict(case='native512', left=left, right=right, condition=condition,
                primary=condition == 'base', bootstrap_seed=random_seed,
                **paired_summary(differences, 10000, random_seed),
                paired_oa_gain_pp=[100*(x['metrics']['oa']-y['metrics']['oa']) for x,y in zip(a,b)],
                paired_ce_difference=[x['metrics']['ce']-y['metrics']['ce'] for x,y in zip(a,b)],
                domains={name: [100*(x['domains'][name]['mca']-y['domains'][name]['mca']) for x,y in zip(a,b)]
                         for name in ('MACGS', 'ModelNet')}))
    assert len(comparisons) == 80
    return lookup, comparisons


def error_counts(lookup, comparisons):
    result = []
    for case in exp.CASES:
        if case == 'region':
            cache = exp.engine.region.data_module.load_cache('eval_base')
            groups = np.repeat(cache['scenes'], 128)
            overlap = cache['overlap']
            eligible = np.stack((overlap.max(-1) >= .3, overlap.max(-2) >= .3), axis=1).reshape(-1)
        else:
            data = exp.NATIVE_PRIOR[case]['distance_mlp'] / 'data'
            labels = np.load(data / 'dev_eval/labels.npy', allow_pickle=False)
            groups, eligible = labels, np.ones(len(labels), dtype=bool)
        correct = {}
        for role in ('relative5', 'normalized_precision5', 'distance_mlp'):
            values = []
            for seed in SEEDS:
                with np.load(lookup[case, role, seed, 'base']['prediction_path'], allow_pickle=False) as z:
                    logits = z['logits']
                values.append(region_values(logits, cache)['correct'] if case == 'region' else logits.argmax(1) == labels)
            correct[role] = np.stack(values)
        for left, right in PAIRS[:2]:
            stats = describe(correct[left], correct[right], groups, eligible)
            reference = next(r for r in comparisons if r['case'] == case and r['primary']
                             and r['left'] == left and r['right'] == right)
            assert abs(stats['mean_gain_pp'] - reference['mean_gain_pp']) < 1e-9
            result.append(dict(case=case, left=left, right=right, stats=stats))
    return result


def figures(comparisons):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout='constrained')
    labels = ['F − N', 'N − Distance', 'N − T', 'N − Q', 'F − Distance']
    for axis, case in zip(axes, exp.CASES):
        for i, (left, right) in enumerate(PAIRS):
            row = next(r for r in comparisons if r['case'] == case and r['primary']
                       and r['left'] == left and r['right'] == right)
            axis.scatter([i]*3, row['paired_seed_gains_pp'], color='steelblue', alpha=.75)
            axis.plot([i-.18, i+.18], [row['mean_gain_pp']]*2, color='black')
        axis.axhline(0, color='gray', linewidth=.8)
        axis.set(xticks=range(5), xticklabels=labels, title=case,
                 ylabel=('R@1' if case == 'region' else 'mCA') + ' difference (pp)')
        axis.tick_params(axis='x', rotation=20)
    fig.savefig(exp.OUT / 'primary_comparison.png', dpi=150)
    plt.close(fig)


def report():
    exp.configure()
    exp.engine.check()
    exp.engine.selected_globally()
    assert not (exp.OUT / 'result.json').exists()
    exp.engine.native.configure()
    rows, permutation = records()
    lookup, comparisons = comparisons_for(rows)
    result = dict(records=rows, comparisons=comparisons, primary=[r for r in comparisons if r['primary']],
        error_counts=error_counts(lookup, comparisons), permutation=permutation,
        independent_metric_max_error=max(r['independent_metric_error'] for r in rows),
        new_fits=6, new_prediction_groups=24, reused_prediction_groups=96,
        lock_sha256=sha(exp.OUT / 'lock.json'), selected_sha256=sha(exp.OUT / 'selected.json'),
        scientific_review_pending=True, visual_review_pending=True, automatic_promotion=False,
        candidate='relative5', fresh_confirmation=False, official_test_access=False, goal_complete=False)
    write(exp.OUT / 'result.json', result)
    lines = ['# 平均精度归一化收尾控制（自动汇总，待实际审阅）', '',
        '六次新拟合统一按原验证指标选模，24 组新预测与 96 组旧预测比较。', '',
        '| 任务 | 方法 | 三种子指标 % | 均值 % |', '| --- | --- | --- | ---: |']
    for case in exp.CASES:
        for role in ROLES:
            metrics = [lookup[case, role, seed, 'base']['metrics'] for seed in SEEDS]
            values = [100*(m['macro']['region_recall'] if case == 'region' else m['mca']) for m in metrics]
            lines.append(f'| {case} | {role} | ' + ' / '.join(f'{v:.3f}' for v in values) + f' | {np.mean(values):.3f} |')
    lines += ['', '| 任务 | 左减右 | 三种子差 pp | 均值 pp | 种子×场景/类别区间 pp |',
              '| --- | --- | --- | ---: | --- |']
    for row in result['primary']:
        interval = row['crossed_seed_scene_bootstrap_95_pp' if row['case'] == 'region' else 'crossed_seed_class_bootstrap_95_pp']
        paired = ' / '.join(f'{v:+.3f}' for v in row['paired_seed_gains_pp'])
        lines.append(f"| {row['case']} | {row['left']} − {row['right']} | {paired} | "
                     f"{row['mean_gain_pp']:+.3f} | [{interval[0]:+.3f}, {interval[1]:+.3f}] |")
    lines += ['', 'F=完整五维，N=平均精度归一化，T=各自大小球形，Q=方向无关平均精度。',
        'N 保留方向与归一化谱权重，自关系沿用 F；共同内容仍含完整协方差。',
        '完整扰动、分组、中心/OA/CE、来源和错误交换结果保存在 result.json。',
        '本轮属于已用开发数据收尾检验，主候选仍为 F，不自动推广或启动新确认。',
        '原生强对照不确定性、RGB 和全查询可靠性负结果保持有效。']
    (exp.OUT / 'report.md').write_text('\n'.join(lines) + '\n')
    figures(comparisons)
    print('NORMALIZED_PRECISION_REPORTED', [(r['case'], r['left'], r['right'], r['mean_gain_pp'])
                                            for r in result['primary']], flush=True)
