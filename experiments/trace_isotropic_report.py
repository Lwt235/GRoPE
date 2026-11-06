"""Local-scale versus ellipsoid attribution from four matched positional encodings."""
import numpy as np

from experiments import trace_isotropic as exp
from experiments import round65_audit as native_audit
from experiments.native_seed_replication_report import paired_summary
from experiments.round65_common import SEEDS, VARIANTS, read, write, sha

METRICS = ('region_recall', 'center_pck', 'hard_epe', 'soft_epe', 'ce')
ROLES = ('relative5', 'distance_mlp', 'fixed_isotropic5', 'trace_isotropic5')
NATIVE_ROLES = REGION_ROLES = ROLES
PAIRS = (('relative5','trace_isotropic5'), ('trace_isotropic5','fixed_isotropic5'),
         ('trace_isotropic5','distance_mlp'), ('relative5','distance_mlp'), ('relative5','fixed_isotropic5'))
from experiments.prediction_success_failure import describe, region_values, example_rows


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
    result['center_ambiguity'] = {name: grouping((cache['strata'] >= 2) == high)
                                  for name, high in (('low',False),('high',True))}
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
    assert len(rows) == 72
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
    assert len(rows) == 24
    return rows


def comparisons_for(rows):
    lookup = {(r['case'], r['role'], r['seed'], r['condition']): r for r in rows}
    assert len(lookup) == len(rows) == 96
    comparisons = []
    for condition in exp.region.CONDITIONS:
        for left, right in PAIRS:
            for group in (None, '0', '1', '2', '3'):
                paired, effects = [], []
                for seed in SEEDS:
                    a, b = [lookup['region', role, seed, condition]['metrics'] for role in (left, right)]
                    if group is not None:
                        a, b = a['strata'][group], b['strata'][group]
                    scenes = [s for s in sorted(a['by_scene']) if a['by_scene'][s]['valid_queries']]
                    assert scenes == [s for s in sorted(b['by_scene']) if b['by_scene'][s]['valid_queries']]
                    paired.append(dict(seed=seed, **{k: (a['macro'][k] - b['macro'][k]) *
                        (100 if k in METRICS[:2] else 1) for k in METRICS}))
                    effects.append([100*(a['by_scene'][s]['region_recall']-b['by_scene'][s]['region_recall']) for s in scenes])
                random_seed = 81001 + len(comparisons)
                comparisons.append(dict(case='region', left=left, right=right,
                    condition=condition, stratum=group, primary=condition == 'base' and group is None,
                    bootstrap_seed=random_seed, scenes=scenes, paired=paired,
                    mean={k:float(np.mean([p[k] for p in paired])) for k in METRICS},
                    **scene_summary(effects, random_seed)))
    for condition in VARIANTS:
        for left, right in PAIRS:
            a, b = [[lookup['native512', role, seed, condition] for seed in SEEDS] for role in (left, right)]
            differences = 100*(np.asarray([r['metrics']['class_accuracy'] for r in a])-
                               np.asarray([r['metrics']['class_accuracy'] for r in b]))
            random_seed = 81001 + len(comparisons)
            comparisons.append(dict(case='native512', left=left, right=right, condition=condition,
                primary=condition == 'base', bootstrap_seed=random_seed,
                **paired_summary(differences, 10000, random_seed),
                paired_oa_gain_pp=[100*(x['metrics']['oa']-y['metrics']['oa']) for x,y in zip(a,b)],
                paired_ce_difference=[x['metrics']['ce']-y['metrics']['ce'] for x,y in zip(a,b)],
                domains={name:[100*(x['domains'][name]['mca']-y['domains'][name]['mca']) for x,y in zip(a,b)]
                         for name in ('MACGS','ModelNet')}))
    assert len(comparisons) == 80
    return lookup, comparisons


def error_analysis(lookup, comparisons):
    """Keep every prediction and seed-level event, including unchanged outcomes."""
    records = []
    event_dir = exp.OUT / 'events'
    event_dir.mkdir(exist_ok=True)
    data = exp.NATIVE_PRIOR['native512']['distance_mlp'] / 'data'
    labels = np.load(data / 'dev_eval/labels.npy', allow_pickle=False)
    native_ids = np.asarray(next(p['ids'] for p in read(data / 'manifest.json')['partitions'] if p['split']=='dev_eval'))
    for case in exp.CASES:
        conditions = exp.region.CONDITIONS if case == 'region' else VARIANTS
        for condition in conditions:
            if case == 'region':
                cache = exp.region.data_module.load_cache('eval_'+condition)
                ids = np.asarray([f'{i}|{direction}|{token}' for i in cache['ids'] for direction in (0,1) for token in range(64)])
                groups = np.repeat(cache['scenes'],128)
                overlap = cache['overlap']
                eligible = np.stack((overlap.max(-1)>=.3,overlap.max(-2)>=.3),axis=1).reshape(-1)
                strata = cache['strata'].reshape(-1)
                masks = dict(all=eligible, **{f'stratum_{i}':eligible&(strata==i) for i in range(4)},
                             ambiguity_low=eligible&(strata<2), ambiguity_high=eligible&(strata>=2))
                extra = dict(strata=strata)
            else:
                ids, groups, eligible = native_ids, labels, np.ones(len(labels),dtype=bool)
                masks = dict(all=eligible,MACGS=labels<30,ModelNet=labels>=30)
                extra = dict(labels=labels)
            correct, picked, center = {}, {}, {}
            for role in ROLES:
                rv = []
                for seed in SEEDS:
                    with np.load(lookup[case,role,seed,condition]['prediction_path'],allow_pickle=False) as z:
                        logits = z['logits']
                    rv.append(region_values(logits,cache) if case == 'region' else
                              dict(correct=logits.argmax(1)==labels,picked=logits.argmax(1)))
                correct[role] = np.stack([v['correct'] for v in rv])
                picked[role] = np.stack([v['picked'] for v in rv])
                if case == 'region':
                    center[role] = {k:np.stack([v[k] for v in rv]) for k in ('pck','hard_epe','soft_epe','ce','picked_overlap')}
            event_path = event_dir / f'{case}_{condition}.npz'
            event_data = dict(ids=ids,groups=groups,eligible=eligible,seeds=np.asarray(SEEDS),**extra)
            for role in ROLES:
                event_data[role+'_correct'], event_data[role+'_picked'] = correct[role], picked[role]
                for key, value in center.get(role,{}).items(): event_data[role+'_'+key] = value
            for left,right in PAIRS:
                a,b=correct[left],correct[right]
                key=left+'__minus__'+right
                event_data[key+'_corrected_seeds']=(a&~b).sum(0).astype(np.uint8)
                event_data[key+'_regressed_seeds']=(~a&b).sum(0).astype(np.uint8)
                stats={name:describe(a,b,groups,mask) for name,mask in masks.items()}
                reference=next(r for r in comparisons if r['case']==case and r['condition']==condition
                               and r['left']==left and r['right']==right and r.get('stratum') is None)
                assert abs(stats['all']['mean_gain_pp']-reference['mean_gain_pp']) < 1e-9
                samples=example_rows(ids,a,b,eligible,dict(group=groups,left_prediction=picked[left],
                                    right_prediction=picked[right],**extra))
                records.append(dict(case=case,condition=condition,left=left,right=right,groups=stats,
                                    examples=samples,event_path=str(event_path),
                                    example_selection='first eight IDs lexicographically in each all-seed direction'))
            np.savez_compressed(event_path,**event_data)
    assert len(records)==40
    return records


def figures(comparisons):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    labels=['Full - local sphere','Local - fixed sphere','Local - Distance']
    for axis,case in zip(axes,exp.CASES):
        for i,(left,right) in enumerate(PAIRS[:3]):
            row=next(r for r in comparisons if r['case']==case and r['primary'] and r['left']==left and r['right']==right)
            axis.scatter([i]*len(SEEDS),row['paired_seed_gains_pp'])
            axis.plot([i-.2,i+.2],[row['mean_gain_pp']]*2,color='black')
        axis.axhline(0,color='gray',linewidth=.8)
        axis.set(xticks=range(3),xticklabels=labels,ylabel=('Region R@1' if case=='region' else 'mCA')+' difference (pp)',title=case)
        axis.tick_params(axis='x',rotation=15)
    fig.savefig(exp.OUT/'primary_comparison.png',dpi=150)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    for axis,case in zip(axes,exp.CASES):
        conditions=exp.region.CONDITIONS if case=='region' else VARIANTS
        for (left,right),label in zip(PAIRS[:3],labels):
            rows=[next(r for r in comparisons if r['case']==case and r['condition']==c and r['left']==left
                       and r['right']==right and r.get('stratum') is None) for c in conditions]
            axis.plot(range(len(rows)),[r['mean_gain_pp'] for r in rows],marker='o',label=label)
        axis.axhline(0,color='gray',linewidth=.8)
        axis.set(xticks=range(len(conditions)),xticklabels=conditions,ylabel='Difference (pp)',title=case)
        axis.tick_params(axis='x',rotation=25)
        axis.legend(fontsize=8)
    fig.savefig(exp.OUT/'conditions.png',dpi=150)
    plt.close(fig)


def report():
    exp.check()
    exp.selected_globally()
    assert not (exp.OUT/'result.json').exists()
    exp.native.configure()
    native,permutation=native_records()
    rows=region_records()+native
    lookup,comparisons=comparisons_for(rows)
    assert sum(not r['reused'] for r in rows)==24 and sum(r['reused'] for r in rows)==72
    events=error_analysis(lookup,comparisons)
    write(exp.OUT/'error_analysis.json',dict(records=events,fresh_confirmation=False))
    result=dict(records=rows,comparisons=comparisons,primary=[r for r in comparisons if r['primary']],
        permutation=permutation,independent_metric_max_error=max(r['independent_metric_error'] for r in rows),
        new_fits=6,new_prediction_groups=24,reused_prediction_groups=72,
        lock_sha256=sha(exp.OUT/'lock.json'),selected_sha256=sha(exp.OUT/'selected.json'),
        scientific_review_pending=True,visual_review_pending=True,automatic_promotion=False,
        fresh_confirmation=False,official_test_access=False,goal_complete=False,
        limitations=['Three paired seeds on reused development data; no fresh confirmation.',
            'Sphericalization jointly removes orientation and axis ratios; it does not isolate orientation alone.',
            'Original anisotropic geometry remains available to common content in every method.',
            'Intervals describe seed/scene/class sensitivity; near-equal means are not a formal equivalence result.',
            'Trace matches mean squared scale, not determinant, support volume or pairwise Mahalanobis scale.'])
    write(exp.OUT/'result.json',result)
    lines=['# 各自大小球形：局部尺度与椭球关系（自动汇总，待实际审阅）','',
        '六次新拟合按原验证 CE 选模并统一固定，随后导出 24 组旧开发预测，与 72 组既有预测比较。','',
        '| 任务 | 方法 | 三种子指标 % | 均值 % |','| --- | --- | --- | ---: |']
    for case in exp.CASES:
        for role in ROLES:
            ms=[lookup[case,role,s,'base']['metrics'] for s in SEEDS]
            values=[100*(m['macro']['region_recall'] if case=='region' else m['mca']) for m in ms]
            lines.append(f'| {case} | {role} | '+' / '.join(f'{v:.3f}' for v in values)+f' | {np.mean(values):.3f} |')
    lines+=['','| 任务 | 左减右 | 三种子差 pp | 均值 pp | 种子×场景/类别区间 pp |','| --- | --- | --- | ---: | --- |']
    for r in result['primary']:
        lo,hi=r['crossed_seed_scene_bootstrap_95_pp' if r['case']=='region' else 'crossed_seed_class_bootstrap_95_pp']
        values=' / '.join(f'{v:+.3f}' for v in r['paired_seed_gains_pp'])
        lines.append(f"| {r['case']} | {r['left']} − {r['right']} | {values} | {r['mean_gain_pp']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines+=['','| 任务 | 左减右 | 纠正率 % | 新增错误率 % | 全种子纠正/退步个数 |','| --- | --- | ---: | ---: | --- |']
    for r in events:
        if r['condition']!='base' or (r['left'],r['right']) not in PAIRS[:3]:continue
        v=r['groups']['all'];rates=v['macro_event_rates_pct'];u=v['unique_cases']
        lines.append(f"| {r['case']} | {r['left']} − {r['right']} | {rates['corrected']:.3f} | {rates['regressed']:.3f} | {u['corrected_all_seeds']} / {u['regressed_all_seeds']} |")
    lines+=['','完整条件、四分组、高/低中心歧义组、逐场景/类别/来源及中心定位、OA/CE 见 result.json；',
        '纠正/退步、不同种子反转、全部逐查询/对象事件和固定 ID 案例见 error_analysis.json 与 events/。',
        '本轮是旧数据上的机制归因。各自大小球形保留平均平方尺度；去掉方向与非球形的联合信息，不代表只去掉朝向。',
        '共同内容一直保留完整协方差。区间跨零不等于等效，本轮也不能改写已有确认的不确定性或 RGB/跨域负结果。']
    (exp.OUT/'report.md').write_text('\n'.join(lines)+'\n')
    figures(comparisons)
    print('TRACE_ISOTROPIC_REPORTED',[(r['case'],r['left'],r['right'],r['mean_gain_pp']) for r in result['primary']],flush=True)


