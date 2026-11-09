"""Independent paired arithmetic for the fixed native-region PE experiment."""
import numpy as np

from experiments.round65_audit import independent_metrics
from experiments.round65_common import SEEDS, VARIANTS, read, write, sha
from models.native_region_pe import ROLES

PAIRS = (('relative5', 'centers_only'), ('relative5', 'distance_mlp'),
         ('relative5', 'geotransformer_rpe'), ('centers_only', 'distance_mlp'),
         ('centers_only', 'geotransformer_rpe'))


def paired_summary(differences, seed, replicates=10000):
    delta = np.asarray(differences, dtype=np.float64)
    if delta.shape != (3, 60) or not np.isfinite(delta).all():
        raise ValueError('expected three paired seeds and all 60 classes')
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


def contrasts(records, replicates=10000):
    lookup = {(r['role'], r['seed'], r['variant']): r['metrics'] for r in records}
    assert len(lookup) == len(records) == len(ROLES) * len(SEEDS) * len(VARIANTS)
    result = []
    for variant in VARIANTS:
        for left, right in PAIRS:
            a, b = ([lookup[role, seed, variant] for seed in SEEDS] for role in (left, right))
            delta = 100 * np.asarray([np.asarray(x['class_accuracy']) - y['class_accuracy'] for x, y in zip(a, b)])
            seed = 87001 + len(result)
            result.append(dict(left=left, right=right, variant=variant, bootstrap_seed=seed,
                **paired_summary(delta, seed, replicates),
                paired_oa_gain_pp=[100 * (x['oa'] - y['oa']) for x, y in zip(a, b)],
                paired_ce_difference=[x['ce'] - y['ce'] for x, y in zip(a, b)],
                source_seed_gains_pp=dict(MACGS=delta[:, :30].mean(1).tolist(),
                                         ModelNet=delta[:, 30:].mean(1).tolist())))
    return result


def report():
    from experiments import native_region_pe as exp
    exp.check()
    assert not (exp.OUT / 'result.json').exists()
    selected = read(exp.OUT / 'selected.json')
    assert selected['all_twelve_selected_before_development_evaluation']
    partition = next(r for r in read(exp.OUT / 'data/manifest.json')['partitions'] if r['split'] == 'dev_eval')
    ids = np.asarray(partition['ids'])
    labels = np.load(exp.OUT / 'data/dev_eval/labels.npy', allow_pickle=False)
    assert len(labels) == 765 and np.array_equal(np.unique(labels), np.arange(60))
    records, predictions, maximum_error, run_records = [], {}, 0., []
    for role in ROLES:
        for seed in SEEDS:
            choice = next(r for r in selected['records'] if (r['role'], r['seed']) == (role, seed))
            path = exp.OUT / 'evaluation' / role / str(seed)
            meta = read(path / 'record.json')
            run = exp.OUT / 'runs' / role / str(seed)
            assert sha(run / 'best.pt') == choice['checkpoint_sha256'] == meta['checkpoint_sha256']
            assert sha(run / 'run.json') == choice['run_sha256']
            assert meta['selection_sha256'] == sha(exp.OUT / 'selected.json')
            assert meta['data_sha256'] == sha(exp.OUT / 'data/manifest.json')
            assert [r['variant'] for r in meta['records']] == list(VARIANTS)
            run_records.append(read(run / 'run.json'))
            for entry in meta['records']:
                variant = entry['variant']
                file = path / (variant + '.npz')
                assert sha(file) == entry['sha256']
                with np.load(file, allow_pickle=False) as z:
                    assert np.array_equal(z['ids'], ids) and np.array_equal(z['labels'], labels)
                    logits = z['logits']
                metrics = independent_metrics(logits, labels)
                error = max(abs(metrics[k] - entry['metrics'][k]) for k in ('mca', 'oa', 'ce'))
                error = max(error, float(np.max(np.abs(np.asarray(metrics['class_accuracy']) - entry['metrics']['class_accuracy']))))
                assert error < 1e-8
                maximum_error = max(maximum_error, error)
                records.append(dict(role=role, seed=seed, variant=variant, metrics=metrics,
                    domains={name: independent_metrics(logits[mask], labels[mask]) for name, mask in
                        (('MACGS', labels < 30), ('ModelNet', labels >= 30))},
                    prediction_path=str(file), prediction_sha256=entry['sha256']))
                predictions[role, seed, variant] = logits
    comparison = contrasts(records)
    exchanges = []
    for variant in VARIANTS:
        for left, right in PAIRS:
            for seed in SEEDS:
                a, b = (predictions[role, seed, variant].argmax(1) == labels for role in (left, right))
                corrected = np.asarray([(a & ~b)[labels == c].mean() for c in range(60)])
                regressed = np.asarray([(~a & b)[labels == c].mean() for c in range(60)])
                exchanges.append(dict(variant=variant, left=left, right=right, seed=seed,
                    corrected_percent=float(100 * corrected.mean()), regressed_percent=float(100 * regressed.mean()),
                    net_gain_pp=float(100 * (corrected - regressed).mean()),
                    corrected_objects=int((a & ~b).sum()), regressed_objects=int((~a & b).sum())))
    base = {r['right']: r for r in comparison if r['variant'] == 'base' and r['left'] == 'relative5'}
    signal = lambda r: r['mean_gain_pp'] >= .5 and r['positive_seed_pairs'] == 3
    result = dict(records=records, contrasts=comparison, error_exchanges=exchanges, training=run_records,
        development_signals=dict(
            both_strong_controls=all(signal(base[k]) for k in ('distance_mlp', 'geotransformer_rpe')),
            extra_physical_covariance=signal(base['centers_only']),
            joint_intervals_exclude_zero={k: r['crossed_seed_class_bootstrap_95_pp'][0] > 0 for k, r in base.items()}),
        new_fits=12, new_prediction_groups=72, independent_metric_max_error=maximum_error,
        lock_sha256=sha(exp.OUT / 'lock.json'), selected_sha256=sha(exp.OUT / 'selected.json'),
        fresh_confirmation=False, official_test_access=False, scientific_review_pending=True,
        visual_review_pending=True, goal_complete=False,
        limitations=['Three paired seeds and repeatedly used development objects; no new-data confirmation.',
            'All roles share full region content; centers_only removes physical covariance only from PE.',
            'Overlapping uniform groups are not a lossless rendering or density representation.',
            'Raw input permutation can change FPS groups; post-pooling permutation is checked separately.',
            'GeoTransformer is a positional-operator adaptation with greater PE capacity, not its full system.',
            'Bootstrap intervals are descriptive; no multiplicity correction, equivalence, or general efficacy claim.',
            'Source and category are confounded; pooling versus the old 512-token task is not a PE attribution.'])
    write(exp.OUT / 'result.json', result)
    lines = ['# 原生高斯区域 PE：自动汇总，待实际审阅', '',
        '固定五维与三个控制，12次完整训练、72组新预测；旧765个开发对象，无新确认访问。', '',
        '| 条件 | 方法 | 三种子 mCA % | 平均 mCA % | OA % | CE |',
        '| --- | --- | --- | ---: | ---: | ---: |']
    lookup = {(r['role'], r['seed'], r['variant']): r['metrics'] for r in records}
    for variant in VARIANTS:
        for role in ROLES:
            m = [lookup[role, seed, variant] for seed in SEEDS]
            scores = ' / '.join(f"{100 * x['mca']:.3f}" for x in m)
            lines.append(f'| {variant} | {role} | {scores} | ' +
                f"{100 * np.mean([x['mca'] for x in m]):.3f} | {100 * np.mean([x['oa'] for x in m]):.3f} | {np.mean([x['ce'] for x in m]):.5f} |")
    lines += ['', '| 比较 | 条件 | 三种子差 pp | 平均差 pp | 联合种子×类别95%区间 pp |',
              '| --- | --- | --- | ---: | --- |']
    for row in comparison:
        gains = ' / '.join(f'{v:+.3f}' for v in row['paired_seed_gains_pp'])
        lo, hi = row['crossed_seed_class_bootstrap_95_pp']
        lines.append(f"| {row['left']} − {row['right']} | {row['variant']} | {gains} | {row['mean_gain_pp']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines += ['', '主判断使用 base 的 Ffull−Distance/Geo 及 Ffull−centers_only；辅助条件不替换主条件。',
        'permuted 是分组前逆序，可能改变FPS区域；模型分组后的排列检查见 selected.json。',
        '全部训练曲线、参数/显存/时间、来源、OA/CE、纠正与新增错误均保留。小差异不证明等效。',
        '实际科学与图表审阅尚未完成，自动开发信号不代表总体目标达成。']
    (exp.OUT / 'report.md').write_text('\n'.join(lines) + '\n')
    plot(result, exp.OUT)
    print('NATIVE_REGION_REPORTED', len(records), len(comparison), maximum_error, flush=True)


def plot(result, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, control in zip(axes, ROLES[1:]):
        rows = [r for r in result['contrasts'] if r['left'] == 'relative5' and r['right'] == control]
        for i, row in enumerate(rows):
            lo, hi = row['crossed_seed_class_bootstrap_95_pp']
            ax.plot([i, i], [lo, hi], color='tab:blue', alpha=.6)
            ax.scatter([i], [row['mean_gain_pp']], color='black', marker='D', s=28)
            ax.scatter(i + np.linspace(-.15, .15, 3), row['paired_seed_gains_pp'], s=22, color='tab:orange')
        ax.axhline(0, color='gray', linewidth=1)
        ax.set_xticks(range(len(VARIANTS)))
        ax.set_xticklabels(VARIANTS, rotation=35, ha='right')
        ax.set_title('Full five − ' + control)
        ax.set_ylabel('mCA difference (pp)')
    fig.suptitle('Paired seeds (orange), mean (black), descriptive seed × class interval (blue)')
    fig.tight_layout()
    fig.savefig(out / 'paired_effects.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(len(ROLES), len(SEEDS), figsize=(14, 12), sharex=True)
    for i, role in enumerate(ROLES):
        for j, seed in enumerate(SEEDS):
            history = read(out / 'runs' / role / str(seed) / 'history.json')
            ax = axes[i, j]
            ax.plot([r['epoch'] for r in history['training']], [r['loss'] for r in history['training']], label='train smoothed CE')
            ax.plot([r['epoch'] for r in history['validation']], [r['ce'] for r in history['validation']], label='val CE')
            ax.set_title(f'{role} / seed {seed}')
            ax.set_xlabel('epoch')
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'training_curves.png', dpi=130)
    plt.close(fig)
