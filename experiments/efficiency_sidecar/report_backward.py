"""Report all measurements, including negative/noisy outcomes."""
import statistics as st
import time
from experiments.round65_common import ROOT,read,write,sha
from experiments.efficiency_sidecar.backward_common import OUT


def run():
    correctness=read(OUT/'correctness_v2.json'); timing=read(OUT/'timing.json')
    assert correctness['state']==timing['state']=='complete'
    summary=[]
    for role in ('relative5','distance_mlp'):
        rows=[x for x in timing['measurements'] if x['role']==role]
        gains=[];aa=[];bb=[]
        for block in range(8):
            d={x['label']:x['wall_ms'] for x in rows if x['block']==block}
            assert len(d)==4
            gains.append(100*(1-(d['candidate_a']+d['candidate_b'])/(d['baseline_a']+d['baseline_b'])))
            aa.append(100*(1-d['baseline_b']/d['baseline_a']))
            bb.append(100*(1-d['candidate_b']/d['candidate_a']))
        summary.append(dict(role=role,paired_block_time_reduction_percent=gains,
            median_paired_time_reduction_percent=st.median(gains),positive_blocks=sum(g>0 for g in gains),
            baseline_duplicate_drift_percent=aa,candidate_duplicate_drift_percent=bb,
            baseline_median_ms=st.median(x['wall_ms'] for x in rows if x['label'].startswith('baseline')),
            candidate_median_ms=st.median(x['wall_ms'] for x in rows if x['label'].startswith('candidate')),
            baseline_peak_gib=max(x['peak_bytes'] for x in rows if x['label'].startswith('baseline'))/2**30,
            candidate_peak_gib=max(x['peak_bytes'] for x in rows if x['label'].startswith('candidate'))/2**30))
    stability=read(OUT/'stability.json') if (OUT/'stability.json').exists() else None
    stable=[]
    if stability:
        assert stability['state']=='complete'
        for role in ('relative5','distance_mlp'):
            rows=[x for x in stability['measurements'] if x['role']==role]
            stable.append(dict(role=role,steps=len(rows),
                max_loss_abs=max(abs(x['candidate']['loss']-x['baseline']['loss']) for x in rows),
                final_baseline_loss=rows[-1]['baseline']['loss'],final_candidate_loss=rows[-1]['candidate']['loss'],
                max_grad_norm_abs=max(abs(x['candidate']['grad_norm']-x['baseline']['grad_norm']) for x in rows),
                final_parameter_max_abs=next(x['max_parameter_abs'] for x in stability['final_models'] if x['role']==role)))
    main=read(ROOT/'results/native_source_diversity/attempt1/lock.json')
    checked={p:sha(p)==d for p,d in main['source_sha256'].items() if p.endswith('.py')}
    assert all(checked.values())
    for stage in (correctness,timing,stability):
        if stage:
            for p,d in stage['sources'].items():assert sha(ROOT/p)==d,p
    result=dict(timing=summary,stability=stable,shared_gpu=True,fresh_scientific_confirmation=False,
        decision='Retain optional fused backward candidate; do not auto-enable main training. Confirm full batch memory and unshared speed before adoption.',
        main_source_count=len(checked),main_frozen_python_unchanged=True,
        result_sha256={n:sha(OUT/f'{n}.json') for n in ('operators','correctness_v2','timing')})
    if stability:result['result_sha256']['stability']=sha(OUT/'stability.json')
    write(OUT/'summary.json',result)
    table='\n'.join(f"| {x['role']} | {x['baseline_median_ms']:.2f} → {x['candidate_median_ms']:.2f} | {x['median_paired_time_reduction_percent']:.2f}% | {x['positive_blocks']}/8 | {x['baseline_peak_gib']:.3f} → {x['candidate_peak_gib']:.3f} |" for x in summary)
    noise='\n'.join(f"- {x['role']}：配对改善范围 {min(x['paired_block_time_reduction_percent']):.2f}% 至 {max(x['paired_block_time_reduction_percent']):.2f}%；A/A 漂移 {min(x['baseline_duplicate_drift_percent']):.2f}% 至 {max(x['baseline_duplicate_drift_percent']):.2f}%；B/B 漂移 {min(x['candidate_duplicate_drift_percent']):.2f}% 至 {max(x['candidate_duplicate_drift_percent']):.2f}%。" for x in summary)
    numerical='\n'.join(f"| {x['role']} | {x['max_logit_abs']:.3g} | {x['errors']['gradient']['max_abs']:.3g} | {x['errors']['gradient']['max_tensor_relative_l2']:.3g} | {x['errors']['updated']['max_abs']:.3g} |" for x in correctness['measurements'])
    trajectory='\n'.join(f"| {x['role']} | {x['steps']} | {x['max_loss_abs']:.3g} | {x['max_grad_norm_abs']:.3g} | {x['final_parameter_max_abs']:.3g} |" for x in stable)
    text=f'''# 反向累加优化：实现验证及完整步骤结果

本轮完成算子定位、单项优化、CPU/GPU 数值核对、完整步骤配对计时及私有短程训练。
保留融合梯度累加作为可选实现；没有接入主训练，也没有改变五维关系公式。
收益属于五维与 Distance 共有的执行优化，不能用来证明五维位置编码比 Distance 更有效。

## 问题、假设与实现

四层注意力读取同一份成对关系隐藏特征 z，原反向过程通过多次 add/add_ 汇总四层梯度。
B4/N512/hidden96 时，一份 FP32 张量为384 MiB。旧 profile 的三次五维反向中，
该尺寸的 aten::add/add_ 共37.191ms，约12.397ms/步；Distance也有相同瓶颈。
这是带 profiler 的定位值，不是整步可节省时间；底层 CUDA kernel 与 aten 时间是重复归因，不能相加。
profile 的父事件遍历未成功归集大多数 backward 事件，判断来自完整算子形状与代码数据流，
不采用不完整的 backward_events 列表计算比例。

假设：把四份梯度一次相加，可以减少大中间量读写；竞争解释是收益被调度、其他算子及额外显存抵消。
若完整步骤无稳定正向信号，则保留负结果并停止扩展；若有信号，则只保留为可选工程方案。
三次二元累加理论读写9份张量，融合为4读1写，共5份；并非省去全部反向开销。

实现仅在私有模型添加四个共享存储视图，并以一个 Triton kernel 合并四份梯度。
直接支持实际非连续梯度布局，不先做四次 contiguous 复制。前向公式、FP32网络、FP64几何、
参数数目、损失、优化器都保留。GPU目前明确限FP32一阶反向；不宣称支持高阶导或学习几何。
推理不执行这一融合反向算子，因此本轮不声称新增推理加速。

首次GPU编译遇到 Triton 3.2 嵌套 constexpr 元组兼容问题，在模型检查前退出。
改为显式stride常量后通过；失败记录及原代码保存在 preflight_v1 与 correctness_failure.json，未抹去。

## 实现正确性

CPU包括双精度 gradcheck、未使用分支、非连续梯度、重复反向，以及两个完整模型的FP32/FP64全参数梯度核对。
GPU使用旧seed11权重、旧validation输入，B4/N512；固定随机数，比较输出、全部参数梯度及一步AdamW更新。

| 方法 | 输出最大绝对差 | 梯度最大绝对差 | 单参数张量最大相对L2差 | 一步参数最大绝对差 |
| --- | ---: | ---: | ---: | ---: |
{numerical}

浮点累加顺序改变，故不要求梯度逐位一致。差异通过预定数值核对；这只证明当前实现的有限测试范围。

## 完整训练步骤计时

B4/N512，旧16对象×2048基元CPU热池，相同取样、旋转、jitter及dropout种子。
包括CPU gather、H2D、增强、模型、损失、反向、梯度裁剪和AdamW；调度提前生成，
不含冷盘IO、正式选模或checkpoint写入。对照已包含投影重排和上一轮几何重排。
每方法8块，每块对照A/A和候选B/B共四个标签，循环平衡执行顺序；
每标签从同一checkpoint重建优化器，预热2步，计时6步，不删不利测量。
两方法共512个私有更新步骤，其中384步计时。

| 方法 | 整步中位耗时ms | 配对块耗时下降中位 | 正向块数 | 峰值显存GiB |
| --- | ---: | ---: | ---: | ---: |
{table}

“配对改善”先对每块两份重复标签平均，再算候选相对对照的耗时下降，最后取8块中位数；
因此不等于表中两个全局中位数之比。峰值是每个单模型测试单元的最大进程分配量。

{noise}

重复噪声有时大于候选收益，不能把单次差异当作效果。平衡配对结果支持继续保留候选，
但重复块来自同一GPU时段，不是独立资源复现，暂不声称稳定固定百分比加速。
CUDA事件计时包含主机提交空隙及共享调度等待，不等于纯kernel时间；整步wall-time是主口径。

## 64步配对稳定性

两个私有模型均从原旧权重起步，使用相同64步输入和随机数，并交替先后顺序。
所有记录的loss和裁剪前梯度范数有限；下表为工程轨迹检查，不是准确率或收敛优势。

| 方法 | 配对步数 | loss最大绝对差 | 梯度范数最大绝对差 | 最终参数最大绝对差 |
| --- | ---: | ---: | ---: | ---: |
{trajectory}

检查仅覆盖旧小池短程运行，不能替代完整训练的终点精度确认。没有写入新的科学checkpoint。

## 取舍与下一步

融合要同时保留多份分支梯度，节省带宽的同时增加显存。当前小批次实测开销见上表；
不能线性外推为正式B16也能放入相同显存上限，不宜默认叠加更多模块。
下一步只需在稳定资源下复测实际批次的时间/峰值显存，确认收益是否值得这部分内存成本；
若不值得，则保留此前几何重排，停止这个分支。缓存无可靠净收益的旧结论保持不变。

测试运行于GPU1，与已获授权的本项目任务共享，nice15、1CPU线程、6GiB自身分配上限。
共享计算可能降低主任务吞吐，不能声称零资源干扰；没有暂停或修改主任务。
主训练冻结Python文件共{len(checked)}个，结束时哈希核对均未变化。
没有访问新科学确认指标，也没有把效率结果当作核心研究效果证据。

[完整计时](timing.json) · [数值核对](correctness_v2.json) · [短程轨迹](stability.json) · [汇总](summary.json)
'''
    (OUT/'report.md').write_text(text)
    print(table);print(trajectory)


if __name__=='__main__':run()
