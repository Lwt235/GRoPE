"""Report the scalar control with its actual scope and all paired outcomes."""
from pathlib import Path
import numpy as np
from experiments.region_jeffreys_control import OUT, bind, parent
from experiments.round65_common import ROOT, SEEDS, read, write, sha

METRICS = ('region_recall', 'center_pck', 'soft_epe', 'ce')
CONDITIONS = ('base', 'sparse256')


def report():
    bind()
    parent.check()
    assert not (OUT / 'result.json').exists()
    selected = read(OUT / 'selected.json')
    assert selected['all_three_selected_before_eval']
    rows = [r for r in read(parent.CANDIDATE / 'result.json')['records']
            if r['role'] in ('relative5', 'distance_mlp')]
    fixed = ROOT / 'results/region_shape_attribution/attempt1'
    rows += [r for r in read(fixed / 'result.json')['records'] if r['role'] == 'fixed_isotropic5']
    roots = dict(relative5=parent.CANDIDATE, distance_mlp=parent.PARENT, fixed_isotropic5=fixed)
    assert len(rows) == 18
    for row in rows:
        path = roots[row['role']] / 'evaluation' / row['role'] / str(row['seed']) / (row['condition'] + '.npz')
        assert sha(path) == row['prediction_sha256']
    caches = {c: parent.data_module.load_cache('eval_' + c) for c in CONDITIONS}
    verification = []
    for choice in selected['records']:
        dest = OUT / 'evaluation/jeffreys_mlp' / str(choice['seed'])
        meta = read(dest / 'record.json')
        assert meta['selection_sha256'] == sha(OUT / 'selected.json')
        assert meta['checkpoint_sha256'] == choice['checkpoint_sha256']
        for row in meta['records']:
            path = dest / (row['condition'] + '.npz')
            cache = caches[row['condition']]
            assert sha(path) == row['prediction_sha256']
            with np.load(path) as archive:
                assert np.array_equal(archive['ids'], cache['ids'])
                independent = parent.engine.metric_torch(archive['logits'], cache['overlap'], cache['world_centers'])
                grouped = parent.engine.grouped_metrics(archive['logits'], cache)
            error = max(abs(independent[k] - row['metrics']['micro'][k]) for k in METRICS)
            error = max(error, max(abs(grouped['macro'][k] - row['metrics']['macro'][k]) for k in METRICS))
            assert error < 1e-10 and grouped['micro']['valid_queries'] == 20422
            verification.append(dict(seed=choice['seed'], condition=row['condition'],
                independent_metric_error=error, valid_queries=20422, replay=meta['validation_replay']))
            rows.append(row)
    lookup = {(r['role'], r['seed'], r['condition']): r['metrics'] for r in rows}
    assert len(rows) == len(lookup) == 24
    comparisons = []
    for left, right in (('relative5', 'jeffreys_mlp'), ('jeffreys_mlp', 'distance_mlp'),
                        ('jeffreys_mlp', 'fixed_isotropic5')):
        for condition in CONDITIONS:
            for stratum in (None, 0, 1, 2, 3):
                paired, effects = [], []
                for seed in SEEDS:
                    a, b = lookup[left, seed, condition], lookup[right, seed, condition]
                    if stratum is not None:
                        a, b = a['strata'][str(stratum)], b['strata'][str(stratum)]
                    paired.append(dict(seed=seed, **{k: (a['macro'][k] - b['macro'][k]) *
                        (100 if k in METRICS[:2] else 1) for k in METRICS}))
                    scenes = list(a['by_scene'])
                    assert set(scenes) == set(b['by_scene'])
                    effects.append([100 * (a['by_scene'][s]['region_recall'] - b['by_scene'][s]['region_recall']) for s in scenes])
                scene_mean = np.mean(effects, axis=0)
                draws = np.random.default_rng(78001).integers(0, len(scenes), (10000, len(scenes)))
                comparisons.append(dict(left=left, right=right, condition=condition, stratum=stratum,
                    paired=paired, mean={k: float(np.mean([r[k] for r in paired])) for k in METRICS},
                    seed_scene_region_gain_pp=effects, scene_mean_region_gain_pp=dict(zip(scenes, scene_mean.tolist())),
                    paired_scene_bootstrap_95_pp=np.quantile(scene_mean[draws].mean(1), [.025, .975]).tolist()))
    primary = next(r for r in comparisons if r['left'] == 'relative5' and r['condition'] == 'base' and r['stratum'] is None)
    result = dict(records=rows, comparisons=comparisons, independent_verification=verification,
        new_fits=3, new_evaluation_groups=6, reused_evaluation_groups=18,
        lock_sha256=sha(OUT / 'lock.json'), selections_sha256=sha(OUT / 'selected.json'),
        analysis_sha256=sha(Path(__file__)),
        decision=dict(separate_components_positive_pattern=bool(all(r['region_recall'] > 0 for r in primary['paired'])
            and primary['paired_scene_bootstrap_95_pp'][0] > 0), automatic_promotion=False),
        limitations=['Old data, three seeds, scene intervals conditional on seeds.',
            'Only this log1p Jeffreys readout is tested; not a proof that scalar encodings universally fail.',
            'The repeated scalar preserves registered five-slot parameterization; it has only two independent inputs.',
            'No fresh scene, native classification or RGB utility follows automatically.'],
        scientific_review_pending=True, fresh_confirmation=False, goal_complete=False)
    write(OUT / 'result.json', result)
    lines = ['# 区域关系：可学习高斯散度对照', '',
        '保留普通距离，将四个形状关系分量合成双向 KL 之和 J，再以 log1p(J) 的四个重复槽输入原五输入 MLP。只有两个独立输入；完整初始状态、共同内容和预算不变。', '',
        '| 条件 | 方法 | R@1 % | 中心 PCK % | Soft EPE | CE |', '| --- | --- | ---: | ---: | ---: | ---: |']
    roles = ('distance_mlp', 'fixed_isotropic5', 'jeffreys_mlp', 'relative5')
    for condition in CONDITIONS:
        for role in roles:
            values = [np.mean([lookup[role, s, condition]['macro'][k] for s in SEEDS]) *
                      (100 if k in METRICS[:2] else 1) for k in METRICS]
            lines.append(f'| {condition} | {role} | ' + ' | '.join(f'{v:.5f}' for v in values) + ' |')
    lines += ['', '| 条件 / 左减右 | 三种子 R@1 差 pp | 平均差 pp | 中心 PCK 差 pp | 场景区间 pp |',
              '| --- | --- | ---: | ---: | --- |']
    for row in comparisons:
        if row['stratum'] is not None:
            continue
        differences = ' / '.join(f"{r['region_recall']:+.3f}" for r in row['paired'])
        lo, hi = row['paired_scene_bootstrap_95_pp']
        lines.append(f"| {row['condition']} / {row['left']} − {row['right']} | {differences} | "
                     f"{row['mean']['region_recall']:+.3f} | {row['mean']['center_pck']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines += ['', '完整五场景、四分组、次指标与核查见 result.json。连续报告全部差异，均值接近或区间跨零不构成等效证明。',
              '该控制改善了对 R58 固定 sin/cos 读出的可比性，仍不能单独证明所有标量压缩必然失效。',
              '当前表已生成，解释需读取全量配对结果后完成；候选、原生归因及 RGB 结论未更改。']
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n')
    print('REGION_JEFFREYS_REPORTED', result['decision'], flush=True)
