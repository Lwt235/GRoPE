"""Generate the fixed report tables only after the complete finite evaluation."""
import argparse
import statistics

from experiments.gaussian_scene_data import OUT, KINDS, SEEDS, read
from experiments.gaussian_memory_setup import sha256
from experiments.gaussian_memory_incremental_data import write_json


def number(value, digits=6):
    return '未定义' if value is None else f'{value:.{digits}f}'


def table(columns, rows):
    return '\n'.join(['| '+' | '.join(columns)+' |',
                      '| '+' | '.join(['---']*len(columns))+' |']+
                     ['| '+' | '.join(map(str, row))+' |' for row in rows])


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    plan=read(OUT/'plan.json')
    scoring=read(OUT/'scoring/summary.json')
    audit=read(OUT/'scoring/independent_recomputation.json')
    training=read(OUT/'training_summary.json')
    assert scoring['complete'] and audit['complete'] and training['complete']
    assert audit['predictions']==2592 and len(training['completed'])==15
    assert scoring['plan_sha256']==sha256(OUT/'plan.json')
    carrier=read(OUT/'carrier/aggregate.json')
    conditions=read(OUT/'source_conditions.json')
    references=read(OUT/'references/summary.json')
    build=read(OUT/'cost/full_construction.json')
    query=read(OUT/'cost/full_query.json')
    assert all(item['complete'] for item in (carrier,conditions,references,build,query))
    summary=scoring['summaries']; comparisons=scoring['comparisons']

    def mean(kind, condition='missing', region='roi', metric='centered_log_mse'):
        return summary[f'{condition}/{region}/{metric}/{kind}']['mean']

    def effects(keys):
        rows=[]
        for condition,region,metric,pair in keys:
            value=comparisons[f'{condition}/{region}/{metric}/{pair}']
            effects=value['per_place_relative_percent']
            rows.append((f'{condition}/{region}/{metric}/{pair}',
                         number(value['relative_error_reduction_percent'],3),
                         '['+', '.join(number(x,3) for x in value['interval95_percent'])+']',
                         ', '.join(number(x,3) for x in value['per_seed_relative_percent']),
                         f'{sum(x>0 for x in effects if x is not None)}/{sum(x is not None for x in effects)}',
                         value['paired_cells']))
        return table(['条件/区域/指标/候选−对照','相对降误差%','配对95%区间%','三个种子%','改善地点','配对单元'],rows)

    sections=['# gaussian_scene_memory1：固定结果表',
              '所有数据均为旧数据；给定相机、静态四来源、逐场景300步拟合、冻结骨干/DPT。'
              '本表只整理预定结果，不代替实际阅图、方法取舍或目标完成验收。'
              '误差越低越好，相对降误差为正表示候选更好；地点和查询等权，种子配对。',
              '## 五组与固定参照',
              table(['组别','缺失ROI空间log-MSE','缺失ROI原始log-MSE','缺失ROI偏差平方','完整输入ROI空间log-MSE','缺失全图AbsRel','完整输入全图AbsRel'],
                    [(kind,number(mean(kind)),number(mean(kind,metric='log_mse')),
                      number(mean(kind,metric='bias_squared')),number(mean(kind,condition='full')),
                      number(mean(kind,region='whole',metric='absrel')),
                      number(mean(kind,condition='full',region='whole',metric='absrel')))
                     for kind in list(KINDS)+['native_current','official_history','render_depth']]),
              'L为无历史同容量适配；T为同字节原始历史token；C/P/G共用同一场景内容。'
              '原生当前图像与官方历史均未适配；官方历史不匹配状态预算或给定相机输入，直接渲染不使用查询RGB。',
              '## 预定主比较与完整输入辅助比较',
              effects([(c,'roi','centered_log_mse',pair) for c in ('missing','full')
                       for pair in ('G-P','G-L','G-T','G-C','P-L','P-T','C-L','C-T')]),
              '## 原始误差的分解',
              '下列三列均为对照减候选；负值表示候选误差更高。原始差等于空间差加偏差平方差，这是代数分解，不是唯一因果机制。',
              table(['候选−对照','原始log-MSE下降','空间log-MSE下降','偏差平方下降'],
                    [(pair,number(mean(pair[-1],metric='log_mse')-mean(pair[0],metric='log_mse')),
                      number(mean(pair[-1])-mean(pair[0])),
                      number(mean(pair[-1],metric='bias_squared')-mean(pair[0],metric='bias_squared')))
                     for pair in ('G-P','G-L','G-T','G-C','P-L','C-L')]),
              '## 事前来源支持分层',
              f"ROI来源支持比例{number(100*conditions['place_equal_roi_support_fraction'],2)}%；"
              '标签仅作评价，所有地点仍保留，不据分层重新定义主终点。',
              effects([('missing',region,'centered_log_mse',pair)
                       for region in ('roi_supported','roi_unsupported')
                       for pair in ('G-P','G-T','G-L')]),
              '## 固定G模型的干预',
              effects([('missing','roi',metric,'G-G_'+intervention)
                       for metric in ('centered_log_mse','log_mse')
                       for intervention in plan['interventions']['kinds']]),
              '固定模型干预只检验依赖性；移除记忆同时移除了训练读取残差，不能替代匹配训练的L。',
              '## 保存场景的保真',
              table(['集合/视图','原始GS/DC RGB MSE','2048初始','300步后','全patch特征MSE','覆盖部分cos','零覆盖视角'],
                    [(group,number(value['rgb_mse']['raw_dc']),number(value['rgb_mse']['coarse_initial']),
                      number(value['rgb_mse']['coarse_final']),
                      number(value['feature']['all_patch_mse']['mean_over_estimable_places']),
                      number(value['feature']['covered_patch_cosine']['mean_over_estimable_places']),
                      f"{len(value['zero_covered_views'])}/{value['views']}")
                     for group,value in carrier['summary'].items()]),
              '原始参照为共同相机坐标下的DC渲染，非官方全SH复现；特征cos只在有覆盖部分定义。'
              'RGB/特征保真和高alpha均不代替原生几何用途。',
              '## 完整成本与状态']
    costrows=[]
    for kind in KINDS:
        rows=[row for row in query['records'] if row['kind']==kind]
        samples=[v for row in rows for v in row['total_query_seconds']]
        costrows.append((kind,rows[0]['source_state_tensor_bytes'],
                         number(1000*statistics.median(samples),2),
                         f'{number(1000*min(samples),2)}–{number(1000*max(samples),2)}',
                         number(max(row['peak_allocated_GiB'] for row in rows),3)))
    sections += [table(['组别','持久tensor字节','完整暖查询中位ms','样本范围ms','查询峰值GiB'],costrows),
                 '暖查询重新计算当前编码器、几何、所有原生层和DPT，状态驻留；加载成本另见JSON。'
                 '共享GPU的波动不支持独占加速或细微排序。',
                 f"完整来源构建秒数：{', '.join(number(row['total_source_to_saved_scene_and_token_seconds'],3) for row in build['records'])}；"
                 f"另计构建器/原生模型冷加载{number(build['cold_builder_load_seconds'],3)}/{number(build['cold_native_load_seconds'],3)}秒。",
                 f"15次训练保留优化循环共{number(training['retained_loop_worker_hours'],3)} worker小时；"
                 f"中断后废弃循环另计{number(training['discarded_loop_worker_hours'],3)}小时。"
                 '这是日志循环时长，不含模型/数据加载或暂停墙钟，不是独占性能。',
                 f"独立NumPy复算{audit['predictions']}预测，最大指标差{audit['max_abs_metric_difference']:.3g}。",
                 '## 证据文件',
                 '[全部指标及配对区间](scoring/summary.json) · [来源到场景](carrier/aggregate.json) · '
                 '[深度参照及覆盖分母](references/summary.json) · [完整构建](cost/full_construction.json) · '
                 '[完整查询](cost/full_query.json) · [训练及中断工作](training_summary.json)']
    target=OUT/'report_tables.md'
    target.write_text('\n\n'.join(sections)+'\n')
    write_json(OUT/'report_tables_identity.json',dict(
        plan_sha256=sha256(OUT/'plan.json'), report_sha256=sha256(target),
        input_sha256={str(path.relative_to(OUT)):sha256(path) for path in [
            OUT/'scoring/summary.json',OUT/'scoring/independent_recomputation.json',
            OUT/'training_summary.json',OUT/'carrier/aggregate.json',OUT/'source_conditions.json',
            OUT/'references/summary.json',OUT/'cost/full_construction.json',OUT/'cost/full_query.json']},
        numerical_tables_complete=True, scientific_decision_not_automated=True, goal_complete=False))
    print(target,flush=True)


if __name__=='__main__':
    main()
