"""Paired development comparisons for the fixed symmetric-relation experiment."""
import numpy as np

from experiments import symmetric_relation as exp
from experiments import round65_audit as native_audit
from experiments.native_seed_replication_report import paired_summary
from experiments.round65_common import SEEDS, VARIANTS, read, write, sha

METRICS = ('region_recall', 'center_pck', 'hard_epe', 'soft_epe', 'ce')
NATIVE_ROLES = ('relative5', 'distance_mlp', 'fixed_isotropic5', 'symmetric3', 'jeffreys_mlp')
REGION_ROLES = ('relative5', 'distance_mlp', 'fixed_isotropic5', 'jeffreys_mlp',
                'geotransformer_rpe', 'symmetric3')


def region_metrics(logits, cache):
    """Preserve the old grouping, adding hard EPE to every method from predictions."""
    engine = exp.region.engine
    values = engine.query_values(logits, cache)
    world = np.asarray(cache['world_centers'], dtype=np.float64)
    hard = []
    for direction in (0, 1):
        scores = logits if direction == 0 else logits.swapaxes(-1, -2)
        target = world[:, 1 - direction]
        picked = target[np.arange(len(target))[:, None], scores.argmax(-1)]
        hard.append(np.linalg.norm(picked - world[:, direction], axis=-1))
    values['hard_epe'] = np.stack(hard, axis=1)

    def grouping(mask=None):
        all_mask = np.ones_like(values['valid'], dtype=bool) if mask is None else mask
        scenes = {scene: engine.aggregate(values, (cache['scenes'] == scene)[:, None, None] & all_mask)
                  for scene in sorted(set(cache['scenes'].tolist()))}
        eligible = [row for row in scenes.values() if row['valid_queries']]
        macro = {key: float(np.mean([row[key] for row in eligible])) if eligible else None for key in METRICS}
        return dict(micro=engine.aggregate(values, mask), by_scene=scenes, macro=macro,
                    evaluable_scenes=len(eligible))

    result = grouping()
    result['strata'] = {str(group): grouping(cache['strata'] == group) for group in range(4)}
    return result


def scene_summary(differences, seed, replicates=10000):
    delta = np.asarray(differences, dtype=np.float64)
    assert delta.ndim == 2 and min(delta.shape) > 0 and np.isfinite(delta).all()
    rng = np.random.default_rng(seed)
    paired = delta.mean(1)
    si = rng.integers(0, len(delta), (replicates, len(delta)))
    ci = rng.integers(0, delta.shape[1], (replicates, delta.shape[1]))
    return dict(mean_gain_pp=float(paired.mean()), paired_seed_gains_pp=paired.tolist(),
        seed_scene_gains_pp=delta.tolist(), scene_mean_gain_pp=delta.mean(0).tolist(),
        positive_seed_pairs=int((paired > 0).sum()), seed_pairs=len(delta),
        paired_seed_bootstrap_95_pp=np.quantile(paired[si].mean(1), [.025, .975]).tolist(),
        paired_scene_bootstrap_95_pp=np.quantile(delta.mean(0)[ci].mean(1), [.025, .975]).tolist(),
        crossed_seed_scene_bootstrap_95_pp=np.quantile(
            delta[si[:, :, None], ci[:, None, :]].mean((1, 2)), [.025, .975]).tolist())


def native_records():
    data = exp.NATIVE_PRIOR['native512']['distance_mlp'] / 'data'
    partition = next(p for p in read(data / 'manifest.json')['partitions'] if p['split'] == 'dev_eval')
    ids = np.asarray(partition['ids'])
    labels = np.load(data / 'dev_eval/labels.npy', allow_pickle=False)
    rows, permutation = [], []
    for case in exp.CASES[1:]:
        for role in NATIVE_ROLES:
            reused = role in exp.NATIVE_PRIOR[case]
            root = exp.NATIVE_PRIOR[case][role] if reused else exp.OUT / case
            selection = read(root / 'selected.json')
            for seed in SEEDS:
                choice = next(r for r in selection['records'] if (r['role'], r['seed']) == (role, seed))
                directory = root / 'evaluation' / role / str(seed)
                meta = read(directory / 'record.json')
                assert meta['selection_sha256'] == sha(root / 'selected.json')
                assert meta['checkpoint_sha256'] == choice['checkpoint_sha256']
                assert sha(root / 'runs' / role / str(seed) / 'best.pt') == choice['checkpoint_sha256']
                assert meta['data_sha256'] == sha(data / 'manifest.json')
                assert [r['variant'] for r in meta['records']] == list(VARIANTS)
                saved = {}
                for entry in meta['records']:
                    variant = entry['variant']
                    path = directory / (variant + '.npz')
                    assert sha(path) == entry['sha256']
                    with np.load(path, allow_pickle=False) as archive:
                        assert np.array_equal(archive['ids'], ids) and np.array_equal(archive['labels'], labels)
                        logits = archive['logits']
                    metrics = native_audit.independent_metrics(logits, labels)
                    error = max(abs(metrics[k] - entry['metrics'][k]) for k in ('mca', 'oa', 'ce'))
                    error = max(error, float(np.max(np.abs(np.asarray(metrics['class_accuracy']) - entry['metrics']['class_accuracy']))))
                    assert error < 1e-8
                    domains = {name: native_audit.independent_metrics(logits[mask], labels[mask]) for name, mask in
                               (('MACGS', labels < 30), ('ModelNet', labels >= 30))}
                    rows.append(dict(case=case, role=role, seed=seed, condition=variant, metrics=metrics,
                        domains=domains, reused=reused, prediction_path=str(path),
                        prediction_sha256=entry['sha256'], independent_metric_error=error))
                    if variant in ('base', 'permuted'):
                        saved[variant] = logits
                a, b = saved['base'], saved['permuted']
                permutation.append(dict(case=case, role=role, seed=seed,
                    maximum_logit_error=float(np.abs(a - b).max()),
                    prediction_changes=int((a.argmax(1) != b.argmax(1)).sum())))
    assert len(rows) == 180
    return rows, permutation


def region_records():
    exp.bind('region')
    caches = {c: exp.region.data_module.load_cache('eval_' + c) for c in exp.region.CONDITIONS}
    manifest = exp.REGION_PRIOR['distance_mlp'] / 'data/manifest.json'
    rows = []
    for role in REGION_ROLES:
        reused = role in exp.REGION_PRIOR
        root = exp.REGION_PRIOR[role] if reused else exp.OUT / 'region'
        selected = read(root / 'selected.json')
        for seed in SEEDS:
            choice = next(r for r in selected['records'] if (r['role'], r['seed']) == (role, seed))
            directory = root / 'evaluation' / role / str(seed)
            meta = read(directory / 'record.json')
            assert meta['selection_sha256'] == sha(root / 'selected.json')
            assert meta['checkpoint_sha256'] == choice['checkpoint_sha256']
            assert sha(root / 'runs' / role / str(seed) / 'best.pt') == choice['checkpoint_sha256']
            assert meta['manifest_sha256'] == sha(manifest)
            assert [r['condition'] for r in meta['records']] == list(exp.region.CONDITIONS)
            for entry in meta['records']:
                condition = entry['condition']
                path = directory / (condition + '.npz')
                assert sha(path) == entry['prediction_sha256']
                cache = caches[condition]
                with np.load(path, allow_pickle=False) as archive:
                    assert np.array_equal(archive['ids'], cache['ids'])
                    logits = archive['logits']
                metrics = region_metrics(logits, cache)
                independent = exp.region.engine.metric_torch(logits, cache['overlap'], cache['world_centers'])
                error = max(abs(independent[k] - metrics['micro'][k]) for k in METRICS)
                for group in (None, '0', '1', '2', '3'):
                    a, b = metrics, entry['metrics']
                    if group is not None:
                        a, b = a['strata'][group], b['strata'][group]
                    for name in ('micro', 'macro'):
                        for key in ('region_recall', 'center_pck', 'soft_epe', 'ce'):
                            if a[name][key] is None:
                                assert b[name][key] is None
                            else:
                                error = max(error, abs(a[name][key] - b[name][key]))
                assert error < 1e-10 and metrics['micro']['valid_queries'] == 20422
                rows.append(dict(case='region', role=role, seed=seed, condition=condition, metrics=metrics,
                    reused=reused, prediction_path=str(path), prediction_sha256=entry['prediction_sha256'],
                    independent_metric_error=error, validation_replay=meta['validation_replay']))
    assert len(rows) == 36
    return rows


def comparisons_for(rows):
    lookup = {(r['case'], r['role'], r['seed'], r['condition']): r for r in rows}
    assert len(lookup) == len(rows) == 216
    comparisons = []
    # Fixed ordering: task, condition, comparison; subgroups follow each regional comparison.
    for condition in exp.region.CONDITIONS:
        for control in REGION_ROLES[:-1]:
            for group in (None, '0', '1', '2', '3'):
                paired, effects = [], []
                for seed in SEEDS:
                    a, b = [lookup['region', role, seed, condition]['metrics'] for role in ('symmetric3', control)]
                    if group is not None:
                        a, b = a['strata'][group], b['strata'][group]
                    scenes = [s for s in sorted(a['by_scene']) if a['by_scene'][s]['valid_queries']]
                    assert scenes == [s for s in sorted(b['by_scene']) if b['by_scene'][s]['valid_queries']]
                    paired.append(dict(seed=seed, **{k: (a['macro'][k] - b['macro'][k]) *
                        (100 if k in METRICS[:2] else 1) for k in METRICS}))
                    effects.append([100 * (a['by_scene'][s]['region_recall'] - b['by_scene'][s]['region_recall']) for s in scenes])
                random_seed = 79001 + len(comparisons)
                comparisons.append(dict(case='region', left='symmetric3', right=control,
                    condition=condition, stratum=group, primary=condition == 'base' and group is None,
                    bootstrap_seed=random_seed, scenes=scenes, paired=paired,
                    mean={k: float(np.mean([p[k] for p in paired])) for k in METRICS},
                    **scene_summary(effects, random_seed)))
    pairs = [('symmetric3', r) for r in ('relative5', 'distance_mlp', 'fixed_isotropic5', 'jeffreys_mlp')]
    pairs += [('jeffreys_mlp', r) for r in ('relative5', 'distance_mlp', 'fixed_isotropic5')]
    for case in exp.CASES[1:]:
        for condition in VARIANTS:
            for left, right in pairs:
                a, b = [[lookup[case, role, seed, condition] for seed in SEEDS] for role in (left, right)]
                differences = 100 * (np.asarray([r['metrics']['class_accuracy'] for r in a]) -
                                     np.asarray([r['metrics']['class_accuracy'] for r in b]))
                random_seed = 79001 + len(comparisons)
                comparisons.append(dict(case=case, left=left, right=right, condition=condition,
                    primary=condition == ('base' if case == 'native512' else 'sparse256'),
                    bootstrap_seed=random_seed, **paired_summary(differences, 10000, random_seed),
                    paired_oa_gain_pp=[100 * (x['metrics']['oa'] - y['metrics']['oa']) for x, y in zip(a, b)],
                    paired_ce_difference=[x['metrics']['ce'] - y['metrics']['ce'] for x, y in zip(a, b)],
                    domains={name: [100 * (x['domains'][name]['mca'] - y['domains'][name]['mca']) for x, y in zip(a, b)]
                             for name in ('MACGS', 'ModelNet')}))
    assert len(comparisons) == 134
    return lookup, comparisons


def figures(comparisons):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout='constrained')
    for axis, case in zip(axes, exp.CASES):
        for i, control in enumerate(('relative5', 'distance_mlp', 'fixed_isotropic5', 'jeffreys_mlp')):
            row = next(r for r in comparisons if r['case'] == case and r['primary'] and
                       r['left'] == 'symmetric3' and r['right'] == control)
            axis.scatter([i] * len(SEEDS), row['paired_seed_gains_pp'])
            axis.plot([i - .2, i + .2], [row['mean_gain_pp']] * 2, color='black')
        axis.axhline(0, color='gray', linewidth=.8)
        axis.set(xticks=range(4), xticklabels=['relative5', 'Distance', 'Fixed sphere', 'Jeffreys'],
                 ylabel=('Region R@1' if case == 'region' else 'mCA') + ' difference (pp)', title=case + ': symmetric3 minus control')
        axis.tick_params(axis='x', rotation=25)
    fig.savefig(exp.OUT / 'primary_comparison.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout='constrained', sharex=True)
    for axes_row, case in zip(axes, exp.CASES[1:]):
        for axis, control in zip(axes_row, ('relative5', 'distance_mlp', 'fixed_isotropic5')):
            for left, offset, color in (('symmetric3', -.12, 'tab:blue'), ('jeffreys_mlp', .12, 'tab:orange')):
                selected = [next(r for r in comparisons if r['case'] == case and r['condition'] == c and
                                 r['left'] == left and r['right'] == control) for c in VARIANTS]
                x = np.arange(len(VARIANTS)) + offset
                axis.plot(x, [r['mean_gain_pp'] for r in selected], color=color, marker='o', label=left)
                for j, row in enumerate(selected):
                    axis.scatter([x[j]] * len(SEEDS), row['paired_seed_gains_pp'], color=color, alpha=.45, s=16)
            axis.axhline(0, color='gray', linewidth=.8)
            axis.set(xticks=range(len(VARIANTS)), xticklabels=VARIANTS, ylabel='mCA difference (pp)',
                     title=case + ': minus ' + control)
            axis.tick_params(axis='x', rotation=35)
            axis.legend(fontsize=8)
    fig.savefig(exp.OUT / 'native_conditions.png', dpi=150)
    plt.close(fig)


def report():
    exp.check()
    exp.selected_globally()
    assert not (exp.OUT / 'result.json').exists()
    exp.native.configure()
    native, permutation = native_records()
    rows = region_records() + native
    lookup, comparisons = comparisons_for(rows)
    assert sum(not r['reused'] for r in rows) == 78 and sum(r['reused'] for r in rows) == 138
    result = dict(records=rows, comparisons=comparisons,
        primary=[r for r in comparisons if r['primary']], permutation=permutation,
        independent_metric_max_error=max(r['independent_metric_error'] for r in rows),
        new_fits=15, new_prediction_groups=78, reused_prediction_groups=138,
        lock_sha256=sha(exp.OUT / 'lock.json'), selected_sha256=sha(exp.OUT / 'selected.json'),
        scientific_review_pending=True, visual_review_pending=True, automatic_promotion=False,
        fresh_confirmation=False, official_test_access=False, goal_complete=False,
        limitations=['Three paired seeds on reused development data; no fresh confirmation.',
            'Region inputs retain known members and fixed eligible queries; native data are optimized GS primitives.',
            'Native training and validation counts change together; budgets match updates and objects, not FLOPs.',
            'Scene/class intervals are descriptive sensitivity analyses; no formal equivalence or universal efficacy claim.',
            'Repeated slots match initialization and parameter count; no measured speed or three-column optimizer equivalence.'])
    write(exp.OUT / 'result.json', result)
    lines = ['# 对称高斯关系：固定开发实验（自动汇总，待实际审阅）', '',
        '15 次新拟合均已按旧验证集选模并统一固定，之后导出开发预测；复用 138 组已绑定身份的控制预测。', '',
        '| 主读数 | 方法 | 三种子指标 % | 均值 % |', '| --- | --- | --- | ---: |']
    for case in exp.CASES:
        condition = 'sparse256' if case == 'native256' else 'base'
        for role in REGION_ROLES if case == 'region' else NATIVE_ROLES:
            metrics = [lookup[case, role, s, condition]['metrics'] for s in SEEDS]
            values = [100 * (m['macro']['region_recall'] if case == 'region' else m['mca']) for m in metrics]
            lines.append(f'| {case}/{condition} | {role} | ' + ' / '.join(f'{v:.3f}' for v in values) + f' | {np.mean(values):.3f} |')
    lines += ['', '| 主读数 | 左减右 | 三种子差 pp | 均值 pp | 种子×场景/类别区间 pp |',
              '| --- | --- | --- | ---: | --- |']
    for row in result['primary']:
        key = 'crossed_seed_scene_bootstrap_95_pp' if row['case'] == 'region' else 'crossed_seed_class_bootstrap_95_pp'
        lo, hi = row[key]
        values = ' / '.join(f'{v:+.3f}' for v in row['paired_seed_gains_pp'])
        lines.append(f"| {row['case']}/{row['condition']} | {row['left']} − {row['right']} | {values} | {row['mean_gain_pp']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines += ['', '所有条件、逐种子/类别/来源/场景/四分组、中心定位及 OA/CE 见 result.json。区域 hard EPE 从所有旧/新保存预测按同一规则补算，未重新训练对照。',
              '三个独立量在配对训练中使用五个重复槽；没有额外模块，不声称本轮获得速度收益。',
              '均值接近不表示等效，区间跨零不表示没有作用。图与完整结果须实际审阅后再作设计决定；既有确认与 RGB/跨域负结果不改写。']
    (exp.OUT / 'report.md').write_text('\n'.join(lines) + '\n')
    figures(comparisons)
    print('SYMMETRIC_RELATION_REPORTED', [(r['case'], r['left'], r['right'], r['mean_gain_pp']) for r in result['primary']], flush=True)
