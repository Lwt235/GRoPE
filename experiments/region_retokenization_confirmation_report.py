"""All frozen predictions before reporting; missing population stays explicit."""
import time
import numpy as np
import torch

from experiments.region_retokenization_confirmation import OUT, ROLES, SEEDS, CONDITIONS, check, load, preserved
from experiments.region_retokenization import metrics, paired_summary, METRICS, load_arrays
from experiments.round58_run import metric_torch
from experiments.round65_common import read, sha


def population_metrics(logits, cache, planned_scenes):
    result=metrics(logits,cache)
    missing={s:dict.fromkeys(METRICS) for s in planned_scenes if s not in result['by_scene']}
    for row in missing.values():row.update(valid_queries=0,total_queries=0,unobserved=True)
    result['by_scene'].update(missing)
    result['by_scene']={s:result['by_scene'][s] for s in planned_scenes}
    result['missing_scenes']=list(missing)
    return result


def independent_error(logits, cache, actual):
    error=0.
    for scene,expected in [(None,actual['micro'])]+list(actual['by_scene'].items()):
        index=slice(None) if scene is None else cache['scenes']==scene
        if not expected['total_queries']:continue
        independent=metric_torch(logits[index],cache['overlap'][index],cache['world_centers'][index])
        assert sum(independent['valid_counts'])==expected['valid_queries']
        for metric in ('region_recall','center_pck','hard_epe','soft_epe','ce'):
            if expected[metric] is not None:error=max(error,abs(independent[metric]-expected[metric]))
        error=max(error,abs(independent['eligible_fraction']-expected['coverage']),
                  abs(independent['eligible_fraction']*independent['region_recall']-expected['valid_and_correct_fraction']))
    return error


def comparison(rows,condition,control,metric,seed):
    lookup={(r['condition'],r['role'],r['seed']):r['metrics'] for r in rows}
    a=[lookup[condition,'relative5',s] for s in SEEDS]
    b=[lookup[condition,control,s] for s in SEEDS]
    scenes=[s for s,v in a[0]['by_scene'].items() if v[metric] is not None]
    delta=[[100*(x['by_scene'][s][metric]-y['by_scene'][s][metric]) for s in scenes] for x,y in zip(a,b)]
    secondary={k:[None if x['macro'][k] is None else x['macro'][k]-y['macro'][k] for x,y in zip(a,b)] for k in METRICS}
    summary=(paired_summary(delta,seed) if scenes else
             dict(mean_gain_pp=None,paired_seed_gains_pp=[None]*3,crossed_seed_scene_bootstrap_95_pp=None,
                  positive_seed_pairs=0,bootstrap_seed=seed))
    return dict(condition=condition,left='relative5',right=control,metric=metric,scenes=scenes,
                primary=condition=='retokenized256' and control!='fixed_isotropic5' and metric=='region_recall',
                paired_metrics=secondary,**summary)


def report():
    lock=check();state=read(OUT/'status.json')
    assert state['stage']=='complete' and len(state['completed'])==36
    assert not (OUT/'result.json').exists()
    manifest=read(OUT/'data_manifest.json')
    caches={c:load(c) for c in CONDITIONS};records=[]
    for item in state['completed']:
        assert sha(item['path'])==item['sha256']
        selected=next(m for m in lock['models'] if (m['role'],m['seed'])==(item['role'],item['seed']))
        assert item['checkpoint_sha256']==selected['checkpoint_sha256']
        assert item['data_manifest_sha256']==sha(OUT/'data_manifest.json')
        assert item['evaluation_lock_sha256']==sha(OUT/'evaluation_lock.json')
        cache=caches[item['condition']];prediction=load_arrays(item['path'])
        assert np.array_equal(prediction['ids'],cache['ids'])
        actual=population_metrics(prediction['logits'],cache,lock['planned_scenes'])
        error=independent_error(prediction['logits'],cache,actual)
        assert error<1e-10,error
        records.append(dict(item,metrics=actual,independent_metric_error=error))
    expected={(c,r,s) for c in CONDITIONS for r in ROLES for s in SEEDS}
    assert {(r['condition'],r['role'],r['seed']) for r in records}==expected
    comparisons=[]
    for condition in CONDITIONS:
        for control in ROLES[1:]:
            for metric in ('region_recall','valid_and_correct_fraction'):
                comparisons.append(comparison(records,condition,control,metric,89001+len(comparisons)))
    primary={c['right']:c for c in comparisons if c['condition']=='retokenized256' and c['metric']=='region_recall'}
    def pattern(c,minimum):
        return bool(c['mean_gain_pp'] is not None and c['mean_gain_pp']>=minimum and
                    c['positive_seed_pairs']==3 and c['crossed_seed_scene_bootstrap_95_pp'][0]>0)
    decision=dict(utility_numeric_pattern={r:pattern(primary[r],1.) for r in ROLES[1:3]},
                  covariance_numeric_pattern=pattern(primary['fixed_isotropic5'],0.),
                  complete_planned_cohort_evaluated=False,
                  center_and_all_query_results_require_joint_review=True,
                  scope='Only available members of the predeclared cohort; missing scenes are not replaced.')
    result=dict(records=records,comparisons=comparisons,decision=decision,
        evaluation_lock_sha256=sha(OUT/'evaluation_lock.json'),data_manifest_sha256=sha(OUT/'data_manifest.json'),
        planned_scenes=lock['planned_scenes'],available_scenes=lock['available_scenes'],unavailable=lock['unavailable'],
        technical_pair_exclusions=manifest['excluded'],data_coverage=manifest['caches'],new_fits=0,
        prediction_groups=36,fresh_cohort_now_used=True,scientific_review_pending=True,visual_review_pending=True,
        independent_metric_max_error=max(r['independent_metric_error'] for r in records),goal_complete=False)
    preserved(OUT/'result.json',result)
    lookup={(r['condition'],r['role'],r['seed']):r['metrics']['macro'] for r in records}
    def mean(condition,role,key):
        values=[lookup[condition,role,s][key] for s in SEEDS]
        return None if any(v is None for v in values) else float(np.mean(values))
    def fmt(value,scale=1.):return 'missing' if value is None else f'{scale*value:.3f}'
    lines=['# 区域重构未用场景确认：自动汇总，待实际审阅','',
           '预定十个地点，九个元数据可用；缺失地点不替换。12 个旧模型，零训练。',
           '指标为可评价场景宏平均；新数据现已使用，不能再次当作 fresh。','',
           '| 条件 | 方法 | 条件 R@1 % | 全查询正确 % | 有效覆盖 % | PCK % | hard EPE | soft EPE | CE |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for condition in CONDITIONS:
        for role in ROLES:
            values=[fmt(mean(condition,role,k),100 if k in METRICS[:4] else 1) for k in METRICS]
            lines.append('| '+condition+' | '+role+' | '+' | '.join(values)+' |')
    lines+=['','| 条件 | F 减对照 | 指标 | 三种子差 pp | 均值 pp | 联合区间 pp |',
            '| --- | --- | --- | --- | ---: | --- |']
    for c in comparisons:
        values=' / '.join(fmt(x) for x in c['paired_seed_gains_pp'])
        ci=c['crossed_seed_scene_bootstrap_95_pp']
        interval='missing' if ci is None else '['+', '.join(fmt(x) for x in ci)+']'
        lines.append(f"| {c['condition']} | {c['right']} | {c['metric']} | {values} | {fmt(c['mean_gain_pp'])} | {interval} |")
    lines+=['','自动数值模式（尚需定位/绝对输出及范围审阅）：'+str(decision),
            f"独立总体/逐场景指标最大差：{result['independent_metric_max_error']:.3g}。",
            '完整缺失、覆盖、技术排除、逐场景和种子差保留在 result.json。',
            'EPE 是配对归一化场景单位；更高条件召回不自动代表总体更耐稀疏。','']
    (OUT/'report.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,metric,label in zip(axes,('region_recall','valid_and_correct_fraction'),
                              ('Conditional R@1 gain (pp)','Correct / all queries gain (pp)')):
        cs=[c for c in comparisons if c['condition']=='retokenized256' and c['metric']==metric]
        for i,c in enumerate(cs):
            if c['mean_gain_pp'] is None:continue
            ax.scatter([i]*3,c['paired_seed_gains_pp'])
            ax.plot([i-.2,i+.2],[c['mean_gain_pp']]*2,color='black')
        ax.axhline(0,color='gray',lw=.8)
        ax.set_xticks(range(3),['F - Distance','F - GeoTransformer','F - fixed'],rotation=18)
        ax.set_ylabel(label)
    fig.suptitle('Retokenized confirmation: 9 of 10 planned places available')
    fig.savefig(OUT/'confirmation_comparison.png',dpi=160);plt.close(fig)
    print('REPORT_COMPLETE',len(records),decision,flush=True)


if __name__=='__main__':
    torch.set_num_threads(4)
    report()
