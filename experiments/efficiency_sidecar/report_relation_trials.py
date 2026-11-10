"""Report complete negative/positive engineering trials without erasing failures."""
from pathlib import Path
import statistics as st
from experiments.round65_common import ROOT,read,write,sha


def summarize(folder):
    data=read(folder/'timing.json');assert data['state']=='complete'
    for p,d in data['sources'].items():assert sha(ROOT/p)==d,p
    rows=[]
    for role in ('relative5','distance_mlp'):
        for mode in ('inference','training'):
            selected=[r for r in data['measurements'] if r['role']==role and r['mode']==mode]
            assert len(selected)==32
            gains=[];aa=[];bb=[]
            for block in range(8):
                items={r['label']:r for r in selected if r['block']==block}
                if mode=='training':
                    for prefix in ('baseline','candidate'):
                        assert items[prefix+'_a']['final_loss']==items[prefix+'_b']['final_loss']
                d={k:v['wall_ms'] for k,v in items.items()}
                gains.append(100*(1-(d['candidate_a']+d['candidate_b'])/(d['baseline_a']+d['baseline_b'])))
                aa.append(100*(1-d['baseline_b']/d['baseline_a']));bb.append(100*(1-d['candidate_b']/d['candidate_a']))
            result=dict(role=role,mode=mode,paired_time_reduction_percent=gains,median_time_reduction_percent=st.median(gains),
                positive_blocks=sum(x>0 for x in gains),baseline_duplicate_drift_percent=aa,candidate_duplicate_drift_percent=bb)
            for prefix in ('baseline','candidate'):
                group=[r for r in selected if r['label'].startswith(prefix)]
                result[prefix]=dict(median_wall_ms=st.median(x['wall_ms'] for x in group),
                    peak_allocated_gib=max(x['peak_allocated_bytes'] for x in group)/2**30)
            rows.append(result)
    check=read(folder/'correctness.json')
    result=dict(batch=data['batch'],merge=data['merge'],timing=rows,numerical_checks=check['checks'],
        original_parameter_update_gate_passed=False,shared_gpu=True,scientific_effect_claim=False,
        timing_sha256=sha(folder/'timing.json'),correctness_sha256=sha(folder/'correctness.json'))
    write(folder/'summary.json',result)
    return result


def run():
    root=ROOT/'results/efficiency_sidecar'
    all_results={}
    for name in ('contraction1','layout1'):
        for folder in sorted((root/name).glob('b*')):
            if (folder/'timing.json').exists():all_results[str(folder.relative_to(root))]=summarize(folder)
    main=read(ROOT/'results/native_source_diversity/attempt1/lock.json')
    checked={p:sha(p)==d for p,d in main['source_sha256'].items() if p.endswith('.py')};assert all(checked.values())
    table=[];details=[]
    for key,result in all_results.items():
        for row in result['timing']:
            a,b=row['baseline'],row['candidate'];g=row['paired_time_reduction_percent']
            table.append(f"| {key} | {row['role']} | {row['mode']} | {a['median_wall_ms']:.2f} → {b['median_wall_ms']:.2f} | {row['median_time_reduction_percent']:+.2f}% | {row['positive_blocks']}/8 | {a['peak_allocated_gib']:.3f} → {b['peak_allocated_gib']:.3f} |")
            aa=row['baseline_duplicate_drift_percent'];bb=row['candidate_duplicate_drift_percent']
            details.append(f"- {key}/{row['role']}/{row['mode']}：配对改善范围[{min(g):.2f},{max(g):.2f}]%；A/A漂移[{min(aa):.2f},{max(aa):.2f}]%；B/B漂移[{min(bb):.2f},{max(bb):.2f}]%。")
    text=f'''# 共享资源下的新训练/推理优化尝试

本次尝试两种不改变模型公式/参数/精度的执行方案。所有测试只用旧数据和旧权重私有副本。
没有修改主科学训练；已核对其{len(checked)}个冻结Python文件均未改变。

## 两个候选及固定比较

1. contraction1：用Triton直接归约两头位置项，hidden梯度融合两个头贡献；query梯度仍用PyTorch einsum。
2. layout1：把共享hidden一次性转为物理B,I,K,J布局，逻辑维度仍为B,I,J,K；仍用原PyTorch矩阵乘法。

对照均为geometry_factored。候选只增加上述单项，未默认叠加旧的融合梯度方案。
参数、FP32网络/FP64几何、TF32关闭、损失与优化器不变；不混入近似激活或低精度改动。
专用算子的CPU双精度gradcheck和两个完整模型梯度核对通过。

## 数值诊断：保留失败，但区分它对模型的影响

两个候选的原始GPU检查均在逐参数“一步AdamW更新差”门槛失败；不是所有检查均通过。
原输出/全参数梯度核对通过，更新差主要由约1e-8至1e-7的梯度残差被Adam的归一化放大。
最初“全部是softmax不敏感key偏置”的解释不成立：还存在少量output/FFN权重例外，已保留记录。
改变key偏置的控制实验支持其近似函数不敏感性，但不能解释这些其他权重，更不能证明完整训练精度不变。

未修改梯度、优化器或公式来强行过关。随后分别记录全部异常坐标与更新后模型输出，
将这些候选用于工程速度筛选，而不是把原始参数检查改写为通过。
两种B4候选在旧输入上的更新后输出最大差均约1.5e-5以内；这个局部结果不等于长期等效。
原 failure.json、诊断及计时分别保留。诊断记录中的二维异常indices按坐标顺序展开，
例如[5,77]表示一个二维坐标，不是两个独立参数。

## 完整计时

每方法/模式8块，A/A与B/B四标签循环平衡；每标签预热2步、计时4步。
训练包含CPU旧16对象热池取样、H2D、增强、全模型、loss、反向、裁剪与AdamW；
推理输入已在GPU，计入整个模型及几何计算，不含输入传输和冷盘IO。
每单项候选有128个计时单元、256个计时训练更新和256次计时推理前向。
两方法共用公平实现，不能把执行优化解释成五维相对Distance的科学优势。

下表“耗时下降”为正表示加速、负表示变慢；配对块中位不等于两个全局时间中位之比。

| 候选/批次 | 方法 | 模式 | 整体中位ms | 配对耗时下降中位 | 正向块 | 峰值allocated GiB |
| --- | --- | --- | ---: | ---: | ---: | ---: |
{chr(10).join(table)}

同实现噪声完整保留：

{chr(10).join(details)}

所有GPU结果来自共享时段。重复块不是独立资源复现；CUDA span也包含共享调度与提交空隙。
显存减少和速度改善是不同指标，不能因为一个指标改善就称“优化成功”。

## 本轮决定

两条候选在B4上没有建立提速收益，不推广、不继续放大到B16，也不叠加旧融合梯度“挽救”结果。
省显存的专用核仅保留为研究原型，并未证明比现有激活重算更值得采用。
这次没有获得新的可采用训练/推理加速；保留先前几何重排和融合梯度候选。

## 执行边界

主任务保持原状，没有新确认数据访问或科学checkpoint写入。
新专用核仅验证两头、FP32、一阶反向，不支持高阶梯度主张。
若B4速度无收益，则按计划停止该候选的B16、组合和短程训练扩展；不为负结果继续堆叠改动。
此前几何重排与融合梯度的已有结果不受这两种独立候选成败影响。
'''
    destination=root/'relation_trials1';destination.mkdir(exist_ok=True)
    (destination/'report.md').write_text(text)
    write(destination/'summary.json',dict(results=all_results,main_source_count=len(checked),main_frozen_python_unchanged=True))
    print('\n'.join(table))

if __name__=='__main__':run()
