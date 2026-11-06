"""Fixed 2x2 token-count comparison on the original development objects only."""
import numpy as np

from experiments import round65_audit as audit
from experiments.native_matched256 import OUT, PRIOR, ROLES, SEEDS, VARIANTS, check
from experiments.native_seed_replication_report import paired_summary
from experiments.round65_common import read, write, sha


def report():
    check()
    assert not (OUT / 'result.json').exists()
    seal = read(OUT / 'selected.json')
    assert seal['all_nine_selected_before_development_evaluation']
    partition = next(p for p in read(OUT / 'data/manifest.json')['partitions'] if p['split'] == 'dev_eval')
    ids = np.asarray(partition['ids'])
    labels = np.load(OUT / 'data/dev_eval/labels.npy', allow_pickle=False)
    rows, permutation, maximum_error = [], [], 0.
    for train_tokens in (256, 512):
        for role in ROLES:
            root = OUT if train_tokens == 256 else PRIOR[role]
            selected = read(root / 'selected.json')
            for seed in SEEDS:
                choice = next(r for r in selected['records'] if (r['role'], r['seed']) == (role, seed))
                directory = root / 'evaluation' / role / str(seed)
                meta = read(directory / 'record.json')
                assert meta['selection_sha256'] == sha(root / 'selected.json')
                assert meta['checkpoint_sha256'] == choice['checkpoint_sha256']
                assert meta['data_sha256'] == sha(OUT / 'data/manifest.json')
                assert [r['variant'] for r in meta['records']] == list(VARIANTS)
                saved = {}
                for entry in meta['records']:
                    variant = entry['variant']
                    path = directory / (variant + '.npz')
                    assert sha(path) == entry['sha256']
                    with np.load(path, allow_pickle=False) as z:
                        assert np.array_equal(z['ids'], ids) and np.array_equal(z['labels'], labels)
                        logits = z['logits']
                    metrics = audit.independent_metrics(logits, labels)
                    error = max(abs(metrics[k] - entry['metrics'][k]) for k in ('mca', 'oa', 'ce'))
                    error = max(error, float(np.max(np.abs(np.asarray(metrics['class_accuracy']) - entry['metrics']['class_accuracy']))))
                    assert error < 1e-8
                    maximum_error = max(maximum_error, error)
                    domains = {name: audit.independent_metrics(logits[mask], labels[mask]) for name, mask in
                               (('MACGS', labels < 30), ('ModelNet', labels >= 30))}
                    rows.append(dict(train_tokens=train_tokens, role=role, seed=seed, variant=variant,
                        metrics=metrics, domains=domains, reused=train_tokens == 512,
                        prediction_sha256=entry['sha256'], prediction_path=str(path)))
                    if variant in ('base', 'permuted'):
                        saved[variant] = logits
                a, b = saved['base'], saved['permuted']
                permutation.append(dict(train_tokens=train_tokens, role=role, seed=seed,
                    maximum_logit_error=float(np.abs(a - b).max()), prediction_changes=int((a.argmax(1) != b.argmax(1)).sum())))
    lookup = {(r['train_tokens'], r['role'], r['seed'], r['variant']): r for r in rows}
    assert len(lookup) == len(rows) == 108

    def delta(train, left, right, variant):
        return 100 * (np.asarray([lookup[train, left, s, variant]['metrics']['class_accuracy'] for s in SEEDS]) -
                      np.asarray([lookup[train, right, s, variant]['metrics']['class_accuracy'] for s in SEEDS]))

    comparisons = []
    for train in (256, 512):
        for variant in VARIANTS:
            for left, right in (('relative5', 'distance_mlp'), ('relative5', 'fixed_isotropic5'), ('fixed_isotropic5', 'distance_mlp')):
                comparisons.append(dict(train_tokens=train, left=left, right=right, variant=variant,
                    **paired_summary(delta(train, left, right, variant), 10000, 78001 + len(comparisons)),
                    paired_oa_gain_pp=[100 * (lookup[train, left, s, variant]['metrics']['oa'] - lookup[train, right, s, variant]['metrics']['oa']) for s in SEEDS],
                    paired_ce_difference=[lookup[train, left, s, variant]['metrics']['ce'] - lookup[train, right, s, variant]['metrics']['ce'] for s in SEEDS],
                    domains={name: [100 * (lookup[train, left, s, variant]['domains'][name]['mca'] -
                        lookup[train, right, s, variant]['domains'][name]['mca']) for s in SEEDS] for name in ('MACGS', 'ModelNet')}))
    interactions = []
    for control in ('distance_mlp', 'fixed_isotropic5'):
        for variant in VARIANTS:
            values = delta(256, 'relative5', control, variant) - delta(512, 'relative5', control, variant)
            interactions.append(dict(control=control, variant=variant,
                contrast='(relative5-control)train256 - (relative5-control)train512',
                **paired_summary(values, 10000, 78101 + len(interactions))))
    primary = [r for r in comparisons if r['train_tokens'] == 256 and r['variant'] == 'sparse256' and r['left'] == 'relative5']
    result = dict(records=rows, comparisons=comparisons, interactions=interactions, primary=primary,
        permutation=permutation, independent_metric_max_error=maximum_error,
        new_fits=9, new_groups=54, reused_groups=54, selected_sha256=sha(OUT / 'selected.json'),
        lock_sha256=sha(OUT / 'lock.json'), fresh_confirmation=False, official_test_access=False,
        scientific_review_pending=True, visual_review_pending=True, goal_complete=False,
        limitations=['Three paired seeds and reused development objects; no new-data confirmation.',
            'New models select by val256 CE; reused models selected by val512 CE. Training and validation count change together.',
            'Matched update/object budgets, not matched FLOPs or total token exposure.',
            'Do not replace the official base confirmation or RGB/transfer negatives with these diagnostic results.'])
    write(OUT / 'result.json', result)
    lines = ['# 原生 256 基元训练对照（自动汇总，待实际审阅）', '',
        '九次新拟合；原 765 个开发评价对象，三个配对种子。所有九模型先按 val256 CE 固定，再导出开发预测。', '',
        '| 训练基元 | 评价基元 | 方法 | 三种子 mCA % | 平均 mCA % | 平均 OA % | 平均 CE |',
        '| ---: | ---: | --- | --- | ---: | ---: | ---: |']
    for train in (256, 512):
        for variant, count in (('sparse256', 256), ('base', 512)):
            for role in ROLES:
                stats = [lookup[train, role, s, variant]['metrics'] for s in SEEDS]
                values = ' / '.join(f"{100 * m['mca']:.3f}" for m in stats)
                lines.append(f'| {train} | {count} | {role} | {values} | ' +
                    f"{100 * np.mean([m['mca'] for m in stats]):.3f} | {100 * np.mean([m['oa'] for m in stats]):.3f} | {np.mean([m['ce'] for m in stats]):.5f} |")
    lines += ['', '| 训练基元 | 条件 | 左减右 | 三种子差 pp | 平均 pp | 种子×类别区间 pp |',
              '| ---: | --- | --- | --- | ---: | --- |']
    for r in comparisons:
        lo, hi = r['crossed_seed_class_bootstrap_95_pp']
        values = ' / '.join(f'{v:+.3f}' for v in r['paired_seed_gains_pp'])
        lines.append(f"| {r['train_tokens']} | {r['variant']} | {r['left']} − {r['right']} | {values} | {r['mean_gain_pp']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines += ['', '稀疏评价中，训练数量改变后的优势变化（新 256 训练减旧 512 训练）：']
    for r in interactions:
        if r['variant'] == 'sparse256':
            lines.append(f"- 相对 {r['control']}：{r['mean_gain_pp']:+.3f} pp；三对差 {r['paired_seed_gains_pp']}。")
    lines += ['', '新流程同时匹配训练与验证数量，不能单独把差异全归给训练数量。区间只描述旧数据上少量种子的敏感性；完整来源、OA、CE、类别和置换记录见 result.json。',
              '本次没有修改五维公式或采用关系投影重排，没有读取官方 TEST。官方基础确认不充分及 RGB/跨域负结果保持不变。']
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for axis, control in zip(axes, ('distance_mlp', 'fixed_isotropic5')):
        ticks = []
        for i, (train, variant, label) in enumerate(((512, 'base', '512 to 512'), (512, 'sparse256', '512 to 256'),
                                                    (256, 'base', '256 to 512'), (256, 'sparse256', '256 to 256'))):
            r = next(r for r in comparisons if r['train_tokens'] == train and r['variant'] == variant and r['left'] == 'relative5' and r['right'] == control)
            axis.scatter([i] * len(SEEDS), r['paired_seed_gains_pp'])
            axis.plot([i - .2, i + .2], [r['mean_gain_pp']] * 2, color='black')
            ticks.append(label)
        axis.axhline(0, color='gray')
        axis.set(xticks=range(4), xticklabels=ticks, ylabel='mCA difference (pp)', title='relative5 minus ' + control)
        axis.tick_params(axis='x', rotation=20)
    fig.savefig(OUT / 'comparison.png', dpi=150)
    plt.close(fig)
    print('MATCHED256_REPORTED', [(r['right'], r['mean_gain_pp']) for r in primary], flush=True)
