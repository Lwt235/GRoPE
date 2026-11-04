"""Separate new-seed replication from original and pooled development evidence."""
import numpy as np

from experiments.round65_common import read, write, sha
from experiments import round65_audit as audit
from experiments.native_seed_replication import OUT, PLAN, SEEDS, OLD_SEEDS, ROLES, VARIANTS, parent, check


def paired_summary(class_differences, replicates, seed):
    delta = np.asarray(class_differences, dtype=np.float64)
    assert delta.ndim == 2 and delta.shape[1] == 60 and np.isfinite(delta).all()
    rng = np.random.default_rng(seed)
    paired = delta.mean(1)
    si = rng.integers(0, len(delta), (replicates, len(delta)))
    ci = rng.integers(0, 60, (replicates, 60))
    seed_means = paired[si].mean(1)
    class_means = delta.mean(0)[ci].mean(1)
    crossed = delta[si[:, :, None], ci[:, None, :]].mean((1, 2))
    return dict(mean_gain_pp=float(paired.mean()), paired_seed_gains_pp=paired.tolist(),
        sample_seed_std_pp=float(paired.std(ddof=1)), positive_seed_pairs=int((paired > 0).sum()),
        seed_pairs=len(delta), paired_seed_bootstrap_95_pp=np.quantile(seed_means, [.025, .975]).tolist(),
        paired_class_bootstrap_95_pp=np.quantile(class_means, [.025, .975]).tolist(),
        crossed_seed_class_bootstrap_95_pp=np.quantile(crossed, [.025, .975]).tolist(),
        class_mean_gain_pp=delta.mean(0).tolist())


def cohort_summaries(records, replicates=10000, seed=75001):
    lookup = {(r['role'], r['seed'], r['variant']):r for r in records}
    assert len(lookup) == len(records) == 96
    cohorts = {}
    for j, (name, seeds) in enumerate((('old3', OLD_SEEDS), ('new5', SEEDS), ('all8', OLD_SEEDS + SEEDS))):
        rows = []
        for i, variant in enumerate(VARIANTS):
            left = np.asarray([lookup['relative5', s, variant]['metrics']['class_accuracy'] for s in seeds])
            right = np.asarray([lookup['distance_mlp', s, variant]['metrics']['class_accuracy'] for s in seeds])
            rows.append(dict(variant=variant, seeds=seeds,
                **paired_summary(100 * (left - right), replicates, seed + 100*j + i)))
        cohorts[name] = rows
    return cohorts


def replication_signal(cohorts):
    primary = next(r for r in cohorts['new5'] if r['variant'] == 'base')
    return bool(primary['mean_gain_pp'] >= .5 and primary['paired_seed_bootstrap_95_pp'][0] > 0)


def report():
    check()
    assert not (OUT / 'result.json').exists()
    plan, seal = read(PLAN), read(OUT / 'selected.json')
    assert seal['all_ten_selected_before_development_evaluation'] and len(seal['records']) == 10
    manifest = read(OUT / 'data/manifest.json')
    partition = next(r for r in manifest['partitions'] if r['split'] == 'dev_eval')
    labels = np.load(OUT / 'data/dev_eval/labels.npy', allow_pickle=False)
    ids = np.asarray(partition['ids'])
    original = read(parent.OUT / 'result.json')
    assert read(parent.OUT / 'review.json')['result_sha256'] == sha(parent.OUT / 'result.json')
    rows = [dict(r, reused=True) for r in original['records'] if r['role'] in ROLES]
    assert len(rows) == 36
    worst, permutation, stored = 0., [], {}
    for role in ROLES:
        for seed in SEEDS:
            choice = next(r for r in seal['records'] if (r['role'], r['seed']) == (role, seed))
            path = OUT / 'evaluation' / role / str(seed)
            meta = read(path / 'record.json')
            assert meta['checkpoint_sha256'] == choice['checkpoint_sha256']
            assert meta['selection_sha256'] == sha(OUT / 'selected.json')
            assert meta['data_sha256'] == sha(OUT / 'data/manifest.json')
            assert [r['variant'] for r in meta['records']] == list(VARIANTS)
            for record in meta['records']:
                variant = record['variant']
                f = path / (variant + '.npz')
                assert sha(f) == record['sha256']
                with np.load(f, allow_pickle=False) as z:
                    assert np.array_equal(z['ids'], ids) and np.array_equal(z['labels'], labels)
                    logits = z['logits']
                metrics = audit.independent_metrics(logits, labels)
                error = max(abs(metrics[k] - record['metrics'][k]) for k in ('mca', 'oa', 'ce'))
                error = max(error, float(abs(np.asarray(metrics['class_accuracy']) - record['metrics']['class_accuracy']).max()))
                assert error < 1e-8
                worst = max(worst, error)
                domains = {name:audit.independent_metrics(logits[mask], labels[mask]) for name, mask in
                           (('MACGS', labels < 30), ('ModelNet', labels >= 30))}
                rows.append(dict(role=role, seed=seed, variant=variant, metrics=metrics, domains=domains,
                    reused=False, prediction_sha256=record['sha256'], independent_metric_error=error))
                if variant in ('base', 'permuted'):
                    stored[role, seed, variant] = logits
            a, b = stored[role, seed, 'base'], stored[role, seed, 'permuted']
            permutation.append(dict(role=role, seed=seed, maximum_logit_error=float(abs(a-b).max()),
                prediction_changes=int((a.argmax(-1) != b.argmax(-1)).sum())))
    cohorts = cohort_summaries(rows, plan['analysis']['bootstrap_replicates'], plan['analysis']['bootstrap_seed'])
    result = dict(records=rows, cohorts=cohorts, new_replication_signal=replication_signal(cohorts),
        original_R73_criterion_passed=original['primary_development_lead'],
        original_R73_distance_base=next(r for r in original['comparisons'] if r['control'] == 'distance_mlp' and r['variant'] == 'base'),
        reused_result_sha256=sha(parent.OUT / 'result.json'), new_groups=60, reused_groups=36,
        independent_metric_max_error=worst, permutation=permutation, selection_sha256=sha(OUT / 'selected.json'),
        lock_sha256=sha(OUT / 'lock.json'), plan_sha256=sha(PLAN), fresh_confirmation=False,
        official_test_access=False, goal_complete=False, scientific_review_pending=True, visual_review_pending=True,
        caveats=['All seeds use the same old 765 development objects; no new-data confirmation.',
                 'The new-five primary signal does not rewrite original R73 failure or establish the overall goal.',
                 'Old-three and all-eight summaries are contextual; primary decision uses new-five only.',
                 'Small-seed percentile intervals are approximate and conditional on used data.',
                 'Class/source sets differ; source subgroup gains are not cross-domain transfer.',
                 'Other R73 controls retain their original three-seed results; no eight-seed claim for them.'])
    write(OUT / 'result.json', result)
    lookup = {(r['role'], r['seed'], r['variant']):r for r in rows}
    lines = ['# 原生五维的固定配对复现（待实际审阅）', '',
             '新增五个种子、每个种子两种原有模型，共十次固定预算拟合。旧三种子不替换、不重新选模；官方 TEST 未使用。', '',
             '| 结果组 | 五维 mCA % | Distance mCA % | 配对差 pp | 种子区间 pp | 正向种子 |',
             '| --- | ---: | ---: | ---: | --- | --- |']
    for name in ('old3', 'new5', 'all8'):
        row = next(r for r in cohorts[name] if r['variant'] == 'base')
        absolute = [100*np.mean([lookup[role, s, 'base']['metrics']['mca'] for s in row['seeds']]) for role in ROLES]
        lo, hi = row['paired_seed_bootstrap_95_pp']
        lines.append(f"| {name} | {absolute[0]:.3f} | {absolute[1]:.3f} | {row['mean_gain_pp']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {row['positive_seed_pairs']}/{row['seed_pairs']} |")
    lines += ['', f"事前固定的新五组复现信号：{result['new_replication_signal']}。原 R73 完整条件：{result['original_R73_criterion_passed']}，保持原结论。", '',
              '| 新五组条件 | 五个配对差 pp | 平均差 pp | 种子区间 pp | 种子×类别区间 pp |',
              '| --- | --- | ---: | --- | --- |']
    for row in cohorts['new5']:
        lo, hi = row['paired_seed_bootstrap_95_pp']; a, b = row['crossed_seed_class_bootstrap_95_pp']
        gains = ' / '.join(f'{x:+.3f}' for x in row['paired_seed_gains_pp'])
        lines.append(f"| {row['variant']} | {gains} | {row['mean_gain_pp']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | [{a:+.3f}, {b:+.3f}] |")
    lines += ['', '全部 OA、CE、类别、来源、排列检查与三种结果组的完整六条件统计见 result.json。',
              '区间是旧数据下的开发不确定性描述，不是新数据确认；合并八组不覆盖新五组主比较，也不改写旧三组的固定判据。',
              '这批拟合固定为十次；不得按结果继续追加种子、改主指标或删条件来获得通过。实际图表审阅与后续解释另行记录。']
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n')
    plot(result)
    print('NATIVE_REPLICATION_REPORTED', result['new_replication_signal'], 'metric error', worst, flush=True)


def plot(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    all_base = next(r for r in result['cohorts']['all8'] if r['variant'] == 'base')
    axes[0].scatter(range(8), all_base['paired_seed_gains_pp'], c=['gray']*3 + ['tab:blue']*5)
    axes[0].axhline(0, color='black', linewidth=.8); axes[0].axvline(2.5, color='gray', linestyle='--')
    axes[0].set(xticks=range(8), xticklabels=list(OLD_SEEDS + SEEDS), xlabel='Paired seed (gray=old, blue=new)',
                ylabel='relative5 - Distance mCA (pp)', title='Every fixed seed pair')
    for i, name in enumerate(('old3', 'new5', 'all8')):
        r = next(r for r in result['cohorts'][name] if r['variant'] == 'base')
        axes[1].plot(r['paired_seed_bootstrap_95_pp'], [i, i], color='black')
        axes[1].scatter([r['mean_gain_pp']], [i], color='tab:blue' if name == 'new5' else 'gray')
    axes[1].axvline(0, color='gray'); axes[1].set(yticks=range(3), yticklabels=['old3 context', 'new5 PRIMARY', 'all8 context'],
        xlabel='Mean paired mCA difference (pp)', title='Paired-seed percentile intervals; used objects')
    fig.savefig(OUT / 'replication.png', dpi=150); plt.close(fig)
    matrix = np.asarray([r['paired_seed_gains_pp'] for r in result['cohorts']['new5']]).T
    fig, ax = plt.subplots(figsize=(9, 4.5), layout='constrained')
    limit = max(.5, float(abs(matrix).max()))
    im = ax.imshow(matrix, cmap='coolwarm', vmin=-limit, vmax=limit)
    ax.set(xticks=range(6), xticklabels=VARIANTS, yticks=range(5), yticklabels=SEEDS,
           title='New five pairs, every fixed condition: relative5 - Distance mCA (pp)')
    for i in range(5):
        for j in range(6):
            ax.text(j, i, f'{matrix[i,j]:+.2f}', ha='center', va='center')
    fig.colorbar(im, ax=ax); fig.savefig(OUT / 'conditions.png', dpi=150); plt.close(fig)

