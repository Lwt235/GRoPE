"""Summarize full-batch measurements without treating shared timing as exclusive."""
import argparse
import statistics as st
from experiments.round65_common import ROOT,read,write,sha


def run(mode):
    out=ROOT/'results/efficiency_sidecar/fullbatch1'/mode
    check,timing=(read(out/f'{s}.json') for s in ('correctness','timing'))
    assert check['state']==timing['state']=='complete'
    main=read(ROOT/'results/native_source_diversity/attempt1/lock.json')
    source_check={p:sha(p)==d for p,d in main['source_sha256'].items() if p.endswith('.py')}
    assert all(source_check.values())
    for stage in (check,timing):
        for p,d in stage['sources'].items():assert sha(ROOT/p)==d,p
    summaries=[]
    for role in ('relative5','distance_mlp'):
        rows=[x for x in timing['measurements'] if x['role']==role]
        assert len(rows)==32
        gains=[];aa=[];bb=[]
        for block in range(8):
            records={x['label']:x for x in rows if x['block']==block}
            for prefix in ('baseline','candidate'):
                assert records[prefix+'_a']['final_loss']==records[prefix+'_b']['final_loss']
                assert records[prefix+'_a']['final_grad_norm']==records[prefix+'_b']['final_grad_norm']
            d={k:v['wall_ms'] for k,v in records.items()}
            gains.append(100*(1-(d['candidate_a']+d['candidate_b'])/(d['baseline_a']+d['baseline_b'])))
            aa.append(100*(1-d['baseline_b']/d['baseline_a']))
            bb.append(100*(1-d['candidate_b']/d['candidate_a']))
        stats={}
        for prefix in ('baseline','candidate'):
            selected=[x for x in rows if x['label'].startswith(prefix)]
            ms=st.median(x['wall_ms'] for x in selected)
            stats[prefix]=dict(median_step_ms=ms,objects_per_second=16000/ms,
                peak_allocated_gib=max(x['peak_allocated_bytes'] for x in selected)/2**30,
                peak_reserved_gib=max(x['peak_reserved_bytes'] for x in selected)/2**30)
        summaries.append(dict(role=role,**stats,paired_time_reduction_percent=gains,median_paired_time_reduction_percent=st.median(gains),
            positive_blocks=sum(g>0 for g in gains),baseline_duplicate_drift_percent=aa,candidate_duplicate_drift_percent=bb))
    result=dict(mode=mode,actual_batch=16,tokens=512,results=summaries,
        correctness=check['checks'],main_frozen_python_unchanged=True,main_source_count=len(source_check),
        exclusive_replication_completed=mode=='exclusive',new_scientific_effect_claim=False,
        source_sha256={s:sha(out/f'{s}.json') for s in ('correctness','timing')})
    write(out/'summary.json',result)
    table=[];noise=[]
    for x in summaries:
        a,b=x['baseline'],x['candidate'];g=x['paired_time_reduction_percent'];aa=x['baseline_duplicate_drift_percent'];bb=x['candidate_duplicate_drift_percent']
        table.append(f"| {x['role']} | {a['median_step_ms']:.2f} → {b['median_step_ms']:.2f} | {x['median_paired_time_reduction_percent']:.2f}% | {x['positive_blocks']}/8 | {a['peak_allocated_gib']:.3f} → {b['peak_allocated_gib']:.3f} | {a['peak_reserved_gib']:.3f} → {b['peak_reserved_gib']:.3f} |")
        noise.append(f"- {x['role']}：配对耗时下降 {min(g):.2f}%–{max(g):.2f}%；同实现A/A漂移 {min(aa):.2f}%–{max(aa):.2f}%，B/B漂移 {min(bb):.2f}%–{max(bb):.2f}%。")
    numerical=[]
    for c in check['checks']:
        numerical.append(f"- {c['role']}：输出最大差 {c['logit_max_abs']:.3g}；梯度最大绝对差 {c['errors']['gradients']['max_abs']:.3g}、单参数张量最大相对L2差 {c['errors']['gradients']['max_tensor_relative_l2']:.3g}；一步参数最大差 {c['errors']['updated']['max_abs']:.3g}。")
    scope=('当前没有空闲卡，仅完成共享卡正式批次验证。稳定资源下的独占复测尚未执行。' if mode=='shared' else '本次为没有其他GPU计算进程的独占复测。')
    text=f'''# 正式批次下的融合反向取舍

{scope}
实际训练配置和冻结锁均为B16/N512，本轮不再用B4近似正式批次。
比较已包含几何重排的实现与仅增加融合反向的实现；并非直接对比主训练原版。

## 正确性与成本

复用旧seed11权重、旧validation前16对象；参数、公式、FP32网络/FP64几何保持不变。
两个模型的输出、全部参数梯度及一步AdamW更新都通过沿用上一轮的数值检查。

{chr(10).join(numerical)}

完整步骤包括旧CPU热池取样、H2D、旋转/jitter、前后向、loss、裁剪及AdamW。
固定输入schedule和随机数；每标签重新初始化相同权重/优化器，预热2步、计时4步。
每方法8块，A/A与B/B四标签循环平衡，共384步私有更新，其中256步计时。
A/A、B/B每块的最终loss和梯度范数均逐位相同，配对检查通过。

| 方法 | 整步中位ms | 配对耗时下降中位 | 正向块 | 峰值allocated GiB | 峰值reserved GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(table)}

配对改善先在每块对两份标签取平均、求相对下降，再取8块中位数；不等于两个全局中位数之比。
allocated为张量分配峰值，reserved还包含PyTorch缓存块，两者不可相加；驱动上下文另有开销。
峰值为每种实现所有计时单元的最大值，不以减小batch掩盖显存代价。

## 计时噪声及解释

{chr(10).join(noise)}

共享时延取决于同卡工作负载，不能把正向配对块写成独占资源复现，或直接推算主训练可节省多少小时。
wall-time是完整步骤主口径；CUDA span亦包含调度/提交空隙，并非纯kernel耗时。
只用了旧16对象热池；不含冷盘IO、主任务双来源混合调度、验证及checkpoint写入。
正式batch相同不等于完整训练流水完全相同。

本轮检查的问题是“正式batch能否运行及其时间/显存取舍”，没有再重复上一轮64步短程训练。
此前短程稳定性检查仍只支持有限工程稳定性，不保证完整训练终点精度不变。
它是五维与Distance共有的执行优化，不是五维特有优势，也不改变推理主路径。

## 决定与未完成项

保留完整测量，不修改当前科学训练；仅凭共享卡短测不设为默认实现。
下一步的必要信息是稳定资源下复测相同B16/N512比较。已有命令支持exclusive模式，
启动及组间检查外部GPU进程；出现其他任务只停止自身，不暂停主任务或自动抢占资源。
若独占结果收益不值得额外显存，结束这个优化分支，不增加模块补救。
本轮没有后台等待器，空卡时可用相同脚本在exclusive目录执行correctness、timing两阶段。

本次测试GPU为 `{timing["gpu_uuid"]}`，资源模式 `{mode}`，自身allocator上限12GiB、nice15、单CPU线程，
相较此前6GiB上限的变化是B16所需资源适配。主训练冻结Python文件{len(source_check)}个结束时均未变。
没有读取本轮主科学实验得分，也没有写入科学checkpoint。

[数值及显存检查](correctness.json) · [全部计时](timing.json) · [机器可读汇总](summary.json)
'''
    (out/'report.md').write_text(text)
    print('\n'.join(table));print('\n'.join(noise))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('shared','exclusive'),default='shared');run(parser.parse_args().mode)
