"""Compose the fixed native confirmation summaries from identity-bound exports."""
import hashlib
from pathlib import Path

import numpy as np

from experiments.native_confirmation_analysis import paired_accuracy_summary, confirmation_decision
from experiments.native_confirmation_runtime import summarize_logits
from experiments.round65_common import ROOT, VARIANTS, read, sha
from experiments.native_stream_cache import atomic_json

OLD_SEEDS = (11, 22, 33)
NEW_SEEDS = (44, 55, 66, 77, 88)


def comparison_specs():
    specs = [dict(left='relative5', right='distance_mlp', cohort=name, seeds=list(seeds))
             for name, seeds in (('old3', OLD_SEEDS), ('new5', NEW_SEEDS),
                                 ('all8', OLD_SEEDS + NEW_SEEDS))]
    specs += [dict(left='relative5', right=role, cohort='old3', seeds=list(OLD_SEEDS))
              for role in ('fixed_isotropic5', 'direct_mlp', 'gaussian_relation', 'center_rope', 'nope')]
    return specs


def joint_interval_required(spec, variant, group):
    if group not in ('full_population', 'no_known_historical_source_overlap'):
        return False
    return ((spec['cohort'] == 'new5' and spec['right'] == 'distance_mlp')
            or (spec['cohort'] == 'old3' and spec['right'] == 'fixed_isotropic5')
            or (variant == 'base' and spec['right'] == 'distance_mlp'
                and spec['cohort'] in ('old3', 'all8')))


def collect_exports(directory, models, variants, ids, labels, groups,
                    cache_identity_sha256, cache_manifest_sha256):
    """Read only declared models; retain every object, seed and condition."""
    directory = Path(directory)
    ids, labels = np.asarray(ids), np.asarray(labels)
    if len(set(ids.tolist())) != len(ids) or labels.shape != ids.shape:
        raise ValueError('invalid expected population')
    model_keys = [(m['role'], m['seed']) for m in models]
    if len(set(model_keys)) != len(model_keys) or len(set(variants)) != len(variants):
        raise ValueError('duplicated model or condition')
    group_identity = {k: hashlib.sha256(np.asarray(v, dtype=np.bool_).tobytes()).hexdigest()
                      for k, v in groups.items()}
    runtime_sha = sha(ROOT / 'experiments/native_confirmation_runtime.py')
    records, predictions = [], {}
    for model in models:
        for variant in variants:
            path = directory / model['role'] / str(model['seed']) / (variant + '.npz')
            saved = read(path.with_suffix('.json'))
            expected = dict(checkpoint=model, variant=variant, cache_identity_sha256=cache_identity_sha256,
                            cache_manifest_sha256=cache_manifest_sha256,
                            runtime_sha256=runtime_sha, group_identity=group_identity)
            if saved['identity'] != expected or saved['optimizer_updates'] != 0:
                raise ValueError('prediction identity differs: ' + str(path))
            if sha(path) != saved['prediction_sha256']:
                raise ValueError('prediction bytes changed: ' + str(path))
            with np.load(path, allow_pickle=False) as archive:
                if not np.array_equal(archive['ids'], ids) or not np.array_equal(archive['labels'], labels):
                    raise ValueError('prediction population or order changed')
                logits = archive['logits']
            metrics = summarize_logits(logits, labels, groups)
            error = 0.
            for group, summary in metrics.items():
                prior = saved['groups'][group]
                if (summary['count'], summary['classes']) != (prior['count'], prior['classes']):
                    raise ValueError('prediction group changed')
                if summary['metrics'] is not None:
                    error = max(error, *(abs(summary['metrics'][k] - prior['metrics'][k])
                                         for k in ('mca', 'oa', 'ce')))
            if error > 1e-8:
                raise ValueError('export and report metrics disagree')
            key = model['role'], model['seed'], variant
            predictions[key] = logits.argmax(1)
            records.append(dict(role=model['role'], seed=model['seed'], variant=variant,
                prediction_path=str(path), prediction_sha256=saved['prediction_sha256'],
                groups=metrics, independent_metric_difference=error))
    return records, predictions


def aggregate_comparisons(records, predictions, labels, groups, uncertainty,
                          *, variants=VARIANTS, specs=None):
    """The same reducer also accepts explicit old-data fixtures for validation."""
    labels = np.asarray(labels)
    lookup = {(r['role'], r['seed'], r['variant']): r for r in records}
    if len(lookup) != len(records) or set(lookup) != set(predictions):
        raise ValueError('prediction/metric grid differs')
    comparisons = []
    for spec in comparison_specs() if specs is None else specs:
        for variant in variants:
            left = np.stack([predictions[spec['left'], s, variant] for s in spec['seeds']])
            right = np.stack([predictions[spec['right'], s, variant] for s in spec['seeds']])
            for group, mask in groups.items():
                mask = np.asarray(mask)
                if mask.shape != labels.shape or mask.dtype != np.bool_ or not mask.any():
                    raise ValueError('confirmation requires nonempty aligned populations')
                stats = paired_accuracy_summary(left[:, mask], right[:, mask], labels[mask],
                    replicates=uncertainty['replicates'], seed=uncertainty['base_seed'],
                    chunk_size=uncertainty['chunk_size'],
                    object_bootstrap=joint_interval_required(spec, variant, group))
                a = [lookup[spec['left'], s, variant]['groups'][group]['metrics'] for s in spec['seeds']]
                b = [lookup[spec['right'], s, variant]['groups'][group]['metrics'] for s in spec['seeds']]
                expected = 100 * (np.array([x['mca'] for x in a]) - np.array([x['mca'] for x in b]))
                if not np.allclose(stats['paired_seed_gains_pp'], expected, atol=1e-10, rtol=0):
                    raise ValueError('paired class endpoint differs from exported metrics')
                comparisons.append(dict(**spec, variant=variant, group=group, **stats,
                    left_seed_oa_pct=[100*x['oa'] for x in a], right_seed_oa_pct=[100*x['oa'] for x in b],
                    left_seed_ce=[x['ce'] for x in a], right_seed_ce=[x['ce'] for x in b],
                    paired_ce_difference=[x['ce']-y['ce'] for x, y in zip(a, b)]))
    return comparisons


def report(out, seal, plan, rows, groups, acquisition):
    out = Path(out)
    ids = [r['file_path'] for r in rows]
    labels = np.asarray([r['taxonomy_id'] for r in rows])
    records, predictions = collect_exports(out / 'exports', seal['models'], plan['models']['conditions'],
        ids, labels, groups, acquisition['cache_identity_sha256'], acquisition['cache_manifest_sha256'])
    comparisons = aggregate_comparisons(records, predictions, labels, groups, plan['uncertainty'])
    primary = next(r for r in comparisons if r['cohort'] == 'new5' and r['right'] == 'distance_mlp'
                   and r['variant'] == 'base' and r['group'] == plan['population']['primary_group'])
    if len(records) != 186 or len(comparisons) != 240:
        raise ValueError('incomplete fixed confirmation grid')
    result = dict(records=records, comparisons=comparisons, primary=primary,
        decision=confirmation_decision(primary), evaluation_lock_sha256=sha(out / 'evaluation_lock.json'),
        protocol_lock_sha256=seal['protocol_lock_sha256'], acquisition_sha256=sha(out / 'acquisition.json'),
        new_fits=0, optimizer_updates=0, scientific_review_pending=True, visual_review_pending=True,
        goal_complete=False, official_objects=6090, no_known_historical_overlap_objects=3845,
        historical_source_reused_objects=2245)
    atomic_json(out / 'result.json', result)
    lines = ['# 原生高斯 PE：固定模型的未用对象确认（待实际审阅）', '',
        '全部 31 个模型来自原验证选模，六条件、186 组预测，无新训练、适配或集成。'
        '官方 6,090 个对象完整保留，其中 2,245 个底层对象有历史使用记录；'
        '主比较使用其余 3,845 个无已知历史 ID 重叠对象，不能保证没有几何近重复。', '',
        '| 人群 | 种子组 | 五维 mCA % | Distance mCA % | 差 pp | 种子×对象区间 pp |',
        '| --- | --- | ---: | ---: | ---: | --- |']
    for group in ('no_known_historical_source_overlap', 'full_population'):
        for cohort in ('new5', 'old3', 'all8'):
            r = next(x for x in comparisons if x['group'] == group and x['cohort'] == cohort
                     and x['right'] == 'distance_mlp' and x['variant'] == 'base')
            low, high = r['crossed_seed_object_bootstrap_95_pp']
            lines.append(f"| {group} | {cohort} | {np.mean(r['left_seed_mca_pct']):.3f} | "
                f"{np.mean(r['right_seed_mca_pct']):.3f} | {r['mean_gain_pp']:+.3f} | [{low:+.3f}, {high:+.3f}] |")
    lines += ['', f"事前主判断：{result['decision']['outcome']}。"
        '旧三组、合并八组或完整官方人群不能覆盖新五组主比较。', '',
        '所有种子、条件、来源、历史复用分组、OA/CE、类别与配对差见 result.json。'
        '区间是小种子数下的近似不确定性描述；本结果不覆盖 RGB、跨域和既有负结果，'
        '单个原生任务通过也不代表整体研究目标完成。']
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    plot(out, comparisons)
    return result


def plot(out, comparisons):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = [r for r in comparisons if r['cohort'] == 'new5' and r['right'] == 'distance_mlp'
            and r['group'] == 'no_known_historical_source_overlap']
    matrix = np.array([next(r for r in data if r['variant'] == v)['paired_seed_gains_pp'] for v in VARIANTS]).T
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout='constrained')
    limit = max(.5, float(np.abs(matrix).max()))
    image = axes[0].imshow(matrix, cmap='coolwarm', vmin=-limit, vmax=limit)
    axes[0].set(xticks=range(6), xticklabels=VARIANTS, yticks=range(5), yticklabels=NEW_SEEDS,
                title='Unused-source objects: every new seed pair (mCA pp)')
    for i in range(5):
        for j in range(6):
            axes[0].text(j, i, f'{matrix[i,j]:+.2f}', ha='center', va='center')
    fig.colorbar(image, ax=axes[0])
    for i, row in enumerate(data):
        axes[1].plot(row['crossed_seed_object_bootstrap_95_pp'], [i, i], color='black')
        axes[1].scatter(row['mean_gain_pp'], i, color='tab:blue')
    axes[1].axvline(0, color='gray')
    axes[1].set(yticks=range(6), yticklabels=[r['variant'] for r in data],
                xlabel='relative5 - Distance mCA (pp)', title='Paired seed × within-class object intervals')
    fig.savefig(out / 'confirmation.png', dpi=150)
    plt.close(fig)
