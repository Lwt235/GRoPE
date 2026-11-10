"""Render the executed three-step efficiency study and its negative findings."""
import statistics as st
import time

from experiments.round65_common import ROOT, read, write, sha
from experiments.efficiency_sidecar.followup_gpu import OUT


def run():
    rep, profile, cache = [read(OUT/f'{name}.json') for name in ('replicate','profile','cache_gpu')]
    assert all(r['state']=='complete' for r in (rep,profile,cache))
    rep_summary=[]
    for mode in ('inference','optimizer_step'):
        gains,aa,bb=[],[],[]
        wall_gains,wall_aa,wall_bb=[],[],[]
        for block in range(8):
            d={x['label']:x['cuda_ms'] for x in rep['measurements'] if x['mode']==mode and x['block']==block}
            gains.append(100*(1-(d['candidate_a']+d['candidate_b'])/(d['baseline_a']+d['baseline_b'])))
            aa.append(100*(1-d['baseline_b']/d['baseline_a']))
            bb.append(100*(1-d['candidate_b']/d['candidate_a']))
            w={x['label']:x['wall_ms'] for x in rep['measurements'] if x['mode']==mode and x['block']==block}
            wall_gains.append(100*(1-(w['candidate_a']+w['candidate_b'])/(w['baseline_a']+w['baseline_b'])))
            wall_aa.append(100*(1-w['baseline_b']/w['baseline_a']))
            wall_bb.append(100*(1-w['candidate_b']/w['candidate_a']))
        rep_summary.append(dict(mode=mode,paired_block_gain_percent=gains,median_gain_percent=st.median(gains),
            baseline_duplicate_drift_percent=aa,candidate_duplicate_drift_percent=bb,
            positive_blocks=sum(x>0 for x in gains),wall_paired_block_gain_percent=wall_gains,
            wall_median_gain_percent=st.median(wall_gains),wall_baseline_duplicate_drift_percent=wall_aa,
            wall_candidate_duplicate_drift_percent=wall_bb))
    profile_summary=[]
    for role in ('relative5','distance_mlp'):
        for name in ('original','factored','geometry_factored'):
            rows=[x for x in profile['measurements'] if x['role']==role and x['implementation']==name and not x['warmup']]
            seg=[x for x in rows if x['segmented']]
            profile_summary.append(dict(role=role,implementation=name,
                median_unsplit_wall_ms=st.median(x['wall_ms'] for x in rows if not x['segmented']),
                median_segment_wall_ms={k:st.median(x['stage_ms'][k] for x in seg) for k in seg[0]['stage_ms']}))
    cache_summary=[]
    for role in ('relative5','distance_mlp'):
        names=['uncached','cache_log']+(['cache_log_precision'] if role=='relative5' else [])
        group={name:[x for x in cache['measurements'] if x['role']==role and x['variant']==name and not x['warmup']] for name in names}
        original=st.median(x['wall_ms'] for x in group['uncached'])
        for name in names:
            rows=group[name]; median=st.median(x['wall_ms'] for x in rows)
            gains=[]
            for b in range(1,8):
                a=st.mean(x['wall_ms'] for x in group['uncached'] if x['block']==b)
                v=st.mean(x['wall_ms'] for x in rows if x['block']==b)
                gains.append(100*(1-v/a))
            prep=0 if name=='uncached' else cache['precompute_cpu_ms']['log_matrix']
            if name=='cache_log_precision':prep+=cache['precompute_cpu_ms']['precision']
            saved=original-median
            cache_summary.append(dict(role=role,variant=name,median_step_wall_ms=median,
                median_time_reduction_percent=100*(1-median/original),paired_block_gain_percent=gains,
                positive_blocks=sum(v>0 for v in gains),one_time_prepare_ms=prep,
                optimistic_point_estimate_break_even_steps=prep/saved if saved>0 else None,
                break_even_is_not_reliably_established=True))
    summary=dict(replication=rep_summary,profile=profile_summary,cache=cache_summary,
        decision='Retain polynomial geometry candidate; do not default-enable per-token caches; no main training changes.',
        shared_gpu=True,fresh_confirmation=False)
    write(OUT/'summary.json',summary)
    rep_lines=[]
    for x in rep_summary:
        g,a,b=x['paired_block_gain_percent'],x['baseline_duplicate_drift_percent'],x['candidate_duplicate_drift_percent']
        rep_lines.append(f"| {x['mode']} | {x['median_gain_percent']:.2f}% | [{min(g):.2f}%, {max(g):.2f}%] | {x['positive_blocks']}/8 | [{min(a):.2f}%, {max(a):.2f}%] | [{min(b):.2f}%, {max(b):.2f}%] |")
    selected=next(x for x in profile_summary if x['role']=='relative5' and x['implementation']=='geometry_factored')
    parts='\n'.join(f'| {k} | {v:.3f} |' for k,v in selected['median_segment_wall_ms'].items())
    cache_lines=[]
    for x in cache_summary:
        br=x['optimistic_point_estimate_break_even_steps']
        cache_lines.append(f"| {x['role']} | {x['variant']} | {x['median_step_wall_ms']:.3f} | {x['median_time_reduction_percent']:.2f}% | {x['positive_blocks']}/7 | {'—' if br is None else f'{br:.1f}'} |")
    main=read(ROOT/'results/native_source_diversity/attempt1/lock.json')
    checked={p:sha(p)==d for p,d in main['source_sha256'].items() if p.endswith('.py')}
    assert all(checked.values())
    for r in (rep,profile,cache):
        for p,d in r['source_sha256'].items():assert sha(ROOT/p)==d,p
    review=dict(reviewed_unix=time.time(),stages_complete=True,main_frozen_python_sources_unchanged=True,
        main_source_count=len(checked),gpu_cache_numerical_checks=len(cache['checks']),
        maximum_cache_logit_error=max(x['logit_max_abs'] for x in cache['checks']),
        maximum_cache_gradient_relative_l2=max(x['gradient_relative_l2'] for x in cache['checks']),
        max_cache_peak_allocated_bytes=max(x['peak_bytes'] for x in cache['measurements']),
        no_new_scientific_accuracy_claim=True,shared_runtime_not_publication_final=True,
        result_sha256={k:sha(OUT/f'{k}.json') for k in ('replicate','profile','cache_gpu')})
    write(OUT/'review.json',review)
    text=f'''# 效率探索第二步：完整执行与判断

**三步计划均已执行。保留新的几何计算重排；暂不默认采用单基元缓存。**
只使用旧 validation 前16对象、旧 seed11 权重的私有副本，在 GPU 1 与本项目已有训练共享。
单CPU线程、nice15、6 GiB分配上限。没有修改主训练源码、配置、checkpoint或调度。
主任务冻结 Python 文件共 {len(checked)} 个，核对均未变化。
三个GPU阶段分别耗时 {rep['elapsed_seconds']:.1f}/{profile['elapsed_seconds']:.1f}/{cache['elapsed_seconds']:.1f} 秒；
均正常结束。计算竞争可能影响主训练吞吐，不能声称零减速。

## 1. 交错复核保留正向信号，也暴露共享计时波动

固定 B4/N512，对照是已经使用 query 投影重排的五维模型，候选仅增加多项式几何计算。
每块包含两份相同对照 A/A 和两份相同候选 B/B；四标签循环平衡，8块，不删波动较大的测量。
下表采用 CUDA-event 计时。改善是每块两份候选平均时间相对两份对照平均时间的减少，再取8块中位数。
重复漂移是完全相同路径的第二份相对第一份耗时减少；负数表示更慢。

| 模式 | 配对改善中位 | 配对改善范围 | 正向块 | A/A重复漂移范围 | B/B重复漂移范围 |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rep_lines)}

Wall-time 口径的配对改善中位数分别为推理 {rep_summary[0]['wall_median_gain_percent']:.2f}%、
含优化器步骤 {rep_summary[1]['wall_median_gain_percent']:.2f}%。
推理的完全相同候选 B/B 在 wall-time 下漂移范围为
[{min(rep_summary[0]['wall_candidate_duplicate_drift_percent']):.2f}%, {max(rep_summary[0]['wall_candidate_duplicate_drift_percent']):.2f}%]，
因此不能笼统说重复噪声只有几个百分点。先前进度更新中的“几个百分点”只概括了对照A/A和训练模式，
没有覆盖推理候选的偶发较大波动。共享时序干扰仍需保留。

本次两模式8块均改善，支持继续保留候选。训练步骤约11%的本次结果不同于前次约4%，
说明幅度依赖共享负载和具体测量。重复块是同一GPU时段的工程重复，不是独立数据或种子确认；
仍不能宣称所有训练均稳定快11%。这里只更新内存中的私有模型，且不含数据管线。

## 2. 完整步骤拆分：反向仍占较大成本

固定32个 B4/N512 batch，包含旧输入池取样、阻塞H2D复制、旋转和jitter、前后向、梯度裁剪与AdamW。
同一调度与随机数公平用于原版、factored、新几何路径及 Distance。
原版前向与拆分前向的logit核对通过。
以下为五维新几何路径的同步分段中位数，单位ms：

| 阶段 | 分段中位耗时 |
| --- | ---: |
{parts}

这些阶段逐段同步，改变流水重叠且受其他进程调度影响，不能相加作为真实训练精确占比。
另测不插入分段同步的整步：该组中位数为 {selected['median_unsplit_wall_ms']:.2f} ms。
输入是CPU热内存中的16对象池，调度提前生成；调度生成约
{profile['schedule_generation_ms_per_step']:.3f} ms/步，未含在整步计时。
没有双来源混合、冷盘IO、原batch16、选模和写checkpoint，不能直接替代主训练吞吐。

这一分解支持两个方向：PE几何确有开销可减少；共同形状特征只有几毫秒，
反向传播仍是几十毫秒，消除特征分解不会自动带来大幅训练加速。
另外，Distance 的 factored 与 geometry_factored 实际计算相同，但分段进行的整步测量仍有波动；
因此不把本轮各方法顺序测得的绝对时间差全部归因于实现。

## 3. 单基元缓存：正确，但净收益尚不可靠

精确沿用原实现：L=V diag[0.5 log(max(λ,1e−15)/0.04)] Vᵀ；P直接取原协方差的FP64逆，
没有把特征值floor混用于求逆，也没有额外对称化。增强时只旋转L/P，中心扰动后的成对关系实时重算。
同时比较无缓存、只缓存L、L+P；Distance仅缓存共同内容所需的L。
缓存测试同时保留多个私有模型和优化器，原始 peak_bytes 记录是整个测试进程的峰值，
不能直接用于每种缓存方案的独立显存比较。

CPU的4组随机取样/旋转检查通过；GPU另有 {len(cache['checks'])} 个检查，最大logit差
{review['maximum_cache_logit_error']:.3g}，最大全参数梯度相对L2差
{review['maximum_cache_gradient_relative_l2']:.3g}。这支持当前固定几何下的数值正确性，
不保证学习几何、缩放或修改协方差后仍能直接复用缓存，也不是长期精度确认。

下表包含额外缓存的CPU取样、传输和旋转。每方法32步，剔除预定首4步预热，
7个配对块×4步。方法交错，但三变体在7个非预热块中并非完全平衡；仍属短时共享初测。
中位耗时改善与正向块数是不同汇总：前者比较所有非预热步的中位数，后者比较每块平均。

| 方法 | 缓存方案 | 整步中位ms | 相对无缓存的中位改善 | 正向配对块 | 乐观点估计回本步数 |
| --- | --- | ---: | ---: | ---: | ---: |
{chr(10).join(cache_lines)}

只缓存L在五维中有多个块退步，L+P仍有负向块；Distance缓存后中位时间反而增加。
因此不推广为默认实现，也不根据小幅均值收益继续增加缓存层。
回本步数仅用一次性CPU预计算成本除以正向中位节省；净节省不稳定，**实际回本尚未建立**。

一次性CPU预计算该16×2048基元池：L为 {cache['precompute_cpu_ms']['log_matrix']:.2f} ms，
P另需 {cache['precompute_cpu_ms']['precision']:.2f} ms；各占 {cache['cache_bytes']['log_matrix']/2**20:.2f} MiB。
每基元缓存L增加72字节，L+P增加144字节，额外传输已经计入步骤。
若缓存原6227对象×2048基元的L+P，单这些缓存理论上约 {6227*2048*144/2**30:.2f} GiB；
没有实际构造整套缓存，不把小池构建时间直接当作全数据成本。
新对象首次推理仍要构建缓存；本轮不声称冷启动推理加速。

## 交付与下一步边界

现有 `adapters.py` 的几何重排继续作为可选实现，激活重算仍为显存受限时的选项。
`cache_geometry.py` 留作可复用原型和负结果证据，不接入主任务。
若继续追求更明显训练收益，应优先研究反向中的成对中间量和算子执行，而非盲目增加几何缓存。
论文效率结果还需稳定资源、实际batch规模和更完整数据管线下复测；本次不新增科学效果主张。

[复核原始记录](replicate.json) · [分段记录](profile.json) · [缓存记录](cache_gpu.json) ·
[汇总数据](summary.json) · [核对](review.json)
'''
    (OUT/'report.md').write_text(text)
    print(OUT/'report.md')


if __name__=='__main__':run()
