"""Paired method and training-strategy contrasts on the same old objects."""
import numpy as np

from experiments import native_count_augmentation as exp
from experiments import round65_audit as audit
from experiments.round65_common import read, write, sha

REGIMES = ('fixed512', 'mixed256512')


def paired_summary(differences, seed, replicates=10000):
    delta = np.asarray(differences, dtype=np.float64)
    assert delta.shape == (3, 60) and np.isfinite(delta).all()
    rng = np.random.default_rng(seed)
    si = rng.integers(0, 3, (replicates, 3))
    ci = rng.integers(0, 60, (replicates, 60))
    paired = delta.mean(1)
    return dict(mean_gain_pp=float(paired.mean()), paired_seed_gains_pp=paired.tolist(),
        class_mean_gain_pp=delta.mean(0).tolist(), positive_seed_pairs=int((paired > 0).sum()),
        paired_seed_bootstrap_95_pp=np.quantile(paired[si].mean(1), [.025, .975]).tolist(),
        paired_class_bootstrap_95_pp=np.quantile(delta.mean(0)[ci].mean(1), [.025, .975]).tolist(),
        crossed_seed_class_bootstrap_95_pp=np.quantile(
            delta[si[:, :, None], ci[:, None, :]].mean((1, 2)), [.025, .975]).tolist())


def contrast_specs():
    def gap(regime, variant, sign=1):
        return [(sign, regime, 'relative5', variant), (-sign, regime, 'distance_mlp', variant)]
    rows = []
    for regime in REGIMES:
        for variant in exp.VARIANTS:
            rows.append(dict(kind='method_gap', regime=regime, variant=variant,
                terms=gap(regime, variant)))
    for variant in exp.VARIANTS:
        rows.append(dict(kind='gap_change', variant=variant,
            terms=gap('mixed256512', variant) + gap('fixed512', variant, -1)))
    for role in exp.ROLES:
        for variant in exp.VARIANTS:
            rows.append(dict(kind='training_effect', role=role, variant=variant,
                terms=[(1, 'mixed256512', role, variant), (-1, 'fixed512', role, variant)]))
    for regime in REGIMES:
        rows.append(dict(kind='sparse_extra_gap', regime=regime,
            terms=gap(regime, 'sparse256') + gap(regime, 'base', -1)))
    rows.append(dict(kind='sparse_extra_gap_change',
        terms=gap('mixed256512', 'sparse256') + gap('mixed256512', 'base', -1) +
              gap('fixed512', 'sparse256', -1) + gap('fixed512', 'base')))
    assert len(rows) == 18
    return rows


def contrasts(records, replicates=10000):
    lookup = {(r['regime'], r['role'], r['seed'], r['variant']): r for r in records}
    assert len(lookup) == len(records) == 36
    results = []
    for i, spec in enumerate(contrast_specs()):
        values = np.zeros((3, 60), dtype=np.float64)
        oa, ce = np.zeros(3), np.zeros(3)
        for sign, regime, role, variant in spec['terms']:
            for j, seed in enumerate(exp.SEEDS):
                stats = lookup[regime, role, seed, variant]['metrics']
                values[j] += sign * 100 * np.asarray(stats['class_accuracy'])
                oa[j] += sign * 100 * stats['oa']
                ce[j] += sign * stats['ce']
        results.append(dict(**spec, bootstrap_seed=84001 + i,
            **paired_summary(values, 84001 + i, replicates), paired_oa_gain_pp=oa.tolist(),
            paired_ce_difference=ce.tolist(), source_seed_gains_pp={
                'MACGS': values[:, :30].mean(1).tolist(), 'ModelNet': values[:, 30:].mean(1).tolist()}))
    return results


def error_exchange(a, b, labels):
    left, right = a.argmax(1) == labels, b.argmax(1) == labels
    corrected = np.asarray([(left & ~right)[labels == c].mean() for c in range(60)])
    regressed = np.asarray([(~left & right)[labels == c].mean() for c in range(60)])
    return dict(corrected_percent=float(100 * corrected.mean()),
        regressed_percent=float(100 * regressed.mean()),
        net_gain_pp=float(100 * (corrected - regressed).mean()),
        corrected_objects=int((left & ~right).sum()), regressed_objects=int((~left & right).sum()))


def report():
    exp.check()
    assert not (exp.OUT / 'result.json').exists()
    assert read(exp.OUT / 'selected.json')['all_six_selected_before_development_evaluation']
    partition = next(r for r in read(exp.OUT / 'data/manifest.json')['partitions'] if r['split'] == 'dev_eval')
    ids = np.asarray(partition['ids'])
    labels = np.load(exp.OUT / 'data/dev_eval/labels.npy', allow_pickle=False)
    assert len(labels) == 765 and np.array_equal(np.unique(labels), np.arange(60))
    rows, logits_by_key, maximum_error = [], {}, 0.
    for regime in REGIMES:
        for role in exp.ROLES:
            root = exp.PRIOR[role] if regime == 'fixed512' else exp.OUT
            selected = read(root / 'selected.json')
            for seed in exp.SEEDS:
                choice = next(r for r in selected['records'] if (r['role'], r['seed']) == (role, seed))
                path = root / 'evaluation' / role / str(seed)
                meta = read(path / 'record.json')
                assert meta['selection_sha256'] == sha(root / 'selected.json')
                assert meta['checkpoint_sha256'] == choice['checkpoint_sha256']
                assert sha(root / 'runs' / role / str(seed) / 'best.pt') == choice['checkpoint_sha256']
                assert meta['data_sha256'] == sha(exp.OUT / 'data/manifest.json')
                declared = [r['variant'] for r in meta['records']]
                assert len(set(declared)) == len(declared) and all(v in declared for v in exp.VARIANTS)
                if regime == 'mixed256512':
                    assert declared == list(exp.VARIANTS)
                for variant in exp.VARIANTS:
                    entry = next(r for r in meta['records'] if r['variant'] == variant)
                    file = path / (variant + '.npz')
                    assert sha(file) == entry['sha256']
                    with np.load(file, allow_pickle=False) as z:
                        assert np.array_equal(z['ids'], ids) and np.array_equal(z['labels'], labels)
                        logits = z['logits']
                    metrics = audit.independent_metrics(logits, labels)
                    error = max(abs(metrics[k] - entry['metrics'][k]) for k in ('mca', 'oa', 'ce'))
                    error = max(error, float(np.max(np.abs(np.asarray(metrics['class_accuracy']) - entry['metrics']['class_accuracy']))))
                    assert error < 1e-8
                    maximum_error = max(maximum_error, error)
                    domains = {name: audit.independent_metrics(logits[mask], labels[mask]) for name, mask in
                        (('MACGS', labels < 30), ('ModelNet', labels >= 30))}
                    rows.append(dict(regime=regime, role=role, seed=seed, variant=variant,
                        metrics=metrics, domains=domains, reused=regime == 'fixed512',
                        prediction_path=str(file), prediction_sha256=entry['sha256']))
                    logits_by_key[regime, role, seed, variant] = logits
    comparison = contrasts(rows)
    exchanges = []
    for regime in REGIMES:
        for variant in exp.VARIANTS:
            for seed in exp.SEEDS:
                a, b = [logits_by_key[regime, role, seed, variant] for role in exp.ROLES]
                exchanges.append(dict(regime=regime, variant=variant, seed=seed, **error_exchange(a, b, labels)))
    result = dict(records=rows, contrasts=comparison, error_exchanges=exchanges,
        new_fits=6, new_prediction_groups=18, reused_prediction_groups=18,
        independent_metric_max_error=maximum_error, lock_sha256=sha(exp.OUT / 'lock.json'),
        selected_sha256=sha(exp.OUT / 'selected.json'), fresh_confirmation=False, official_test_access=False,
        scientific_review_pending=True, visual_review_pending=True, goal_complete=False,
        limitations=['Three paired seeds on old development objects; all effects are development evidence.',
            'The mixed policy changes token exposure and FLOPs, while keeping updates and validation policy fixed.',
            'Bootstrap intervals are descriptive with few seeds; small gaps do not establish equivalence.',
            'Source groups have different categories; this is not a source intervention.',
            'No covariance mechanism, universal strong-PE superiority, or fresh confirmation is established by this experiment.'])
    write(exp.OUT / 'result.json', result)
    lines = ['# 原生同等数量增强：自动汇总，待实际审阅', '',
        '六次新拟合，固定完整五维与 Distance；原 765 个开发对象。旧固定训练按身份复用。', '',
        '| 训练 | 评价 | 方法 | 三种子 mCA % | 平均 mCA % | OA % | CE |',
        '| --- | --- | --- | --- | ---: | ---: | ---: |']
    lookup = {(r['regime'], r['role'], r['seed'], r['variant']): r for r in rows}
    for regime in REGIMES:
        for variant in exp.VARIANTS:
            for role in exp.ROLES:
                m = [lookup[regime, role, s, variant]['metrics'] for s in exp.SEEDS]
                scores = ' / '.join(f"{100*x['mca']:.3f}" for x in m)
                lines.append(f'| {regime} | {variant} | {role} | {scores} | ' +
                    f"{100*np.mean([x['mca'] for x in m]):.3f} | {100*np.mean([x['oa'] for x in m]):.3f} | {np.mean([x['ce'] for x in m]):.5f} |")
    lines += ['', '| 对比 | 训练/方法 | 条件 | 三种子差 pp | 均值 pp | 种子×类别区间 pp |',
        '| --- | --- | --- | --- | ---: | --- |']
    for r in comparison:
        gains = ' / '.join(f'{v:+.3f}' for v in r['paired_seed_gains_pp'])
        lo, hi = r['crossed_seed_class_bootstrap_95_pp']
        lines.append(f"| {r['kind']} | {r.get('regime', r.get('role', 'mixed−fixed'))} | {r.get('variant', '256−512')} | {gains} | {r['mean_gain_pp']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines += ['', 'method_gap 为 F−Distance；gap_change 为混合训练减固定训练的差距变化。',
        'training_effect 为同一方法混合减固定；sparse_extra_gap 为 256 差距减 512 差距；最后一行为该差距的策略间变化。',
        '必须结合各方法的绝对变化区分 Distance 追平与 F 退步。基础、稀疏和密集分别报告。',
        '完整来源、OA、CE、纠正/新错误、身份见 result.json。尚待实际训练与图表审阅，未宣称独立确认或总体目标完成。']
    (exp.OUT / 'report.md').write_text('\n'.join(lines) + '\n')
    plot(result)
    print('COUNT_AUGMENTATION_REPORTED', len(rows), len(comparison), maximum_error, flush=True)


def plot(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    for ax, regime in zip(axes, REGIMES):
        for i, variant in enumerate(exp.VARIANTS):
            row = next(r for r in result['contrasts'] if r['kind'] == 'method_gap' and r['regime'] == regime and r['variant'] == variant)
            ax.scatter(np.full(3, i), row['paired_seed_gains_pp'], label='Paired seeds' if i == 0 else None)
            ax.plot([i-.18, i+.18], [row['mean_gain_pp']]*2, color='black', label='Mean' if i == 0 else None)
        ax.axhline(0, color='gray', linewidth=1)
        ax.set(xticks=range(3), xticklabels=['512', '256', '1024'], xlabel='Evaluation tokens',
            ylabel='F minus Distance (mCA pp)', title=regime)
        ax.legend()
    fig.savefig(exp.OUT / 'comparison.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    for ax, role in zip(axes, exp.ROLES):
        for seed in exp.SEEDS:
            history = read(exp.OUT / 'runs' / role / str(seed) / 'history.json')
            ax.plot([r['epoch'] for r in history['validation']], [r['ce'] for r in history['validation']], label=f'seed {seed}')
        ax.set(xlabel='Epoch', ylabel='Validation CE (512 tokens)', title=role)
        ax.legend()
    fig.savefig(exp.OUT / 'training.png', dpi=150)
    plt.close(fig)
