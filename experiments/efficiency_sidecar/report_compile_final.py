"""Summarize bounded partial compilation without overclaiming shared timing."""
import statistics as st
from experiments.round65_common import ROOT,read,write,sha

OUT=ROOT/'results/efficiency_sidecar/compile1'


def summarize(folder):
    data=read(folder/'timing.json');check=read(folder/'correctness.json')
    assert data['state']==check['state']=='complete'
    for record in (data,check):
        for p,d in record['sources'].items():assert sha(ROOT/p)==d,p
    results=[]
    for role in ('relative5','distance_mlp'):
        for mode in ('inference','training'):
            rows=[r for r in data['measurements'] if r['role']==role and r['mode']==mode]
            assert len(rows)==32
            gain=[];aa=[];bb=[]
            for block in range(8):
                items={r['label']:r for r in rows if r['block']==block}
                if mode=='training':
                    for prefix in ('baseline','candidate'):assert items[prefix+'_a']['final_loss']==items[prefix+'_b']['final_loss']
                d={k:v['wall_ms'] for k,v in items.items()}
                gain.append(100*(1-(d['candidate_a']+d['candidate_b'])/(d['baseline_a']+d['baseline_b'])))
                aa.append(100*(1-d['baseline_b']/d['baseline_a']));bb.append(100*(1-d['candidate_b']/d['candidate_a']))
            result=dict(role=role,mode=mode,paired_gain_percent=gain,median_gain_percent=st.median(gain),positive_blocks=sum(g>0 for g in gain),
                baseline_duplicate_drift_percent=aa,candidate_duplicate_drift_percent=bb)
            for prefix in ('baseline','candidate'):
                group=[r for r in rows if r['label'].startswith(prefix)]
                result[prefix]=dict(median_wall_ms=st.median(x['wall_ms'] for x in group),
                    peak_allocated_gib=max(x['peak_allocated_bytes'] for x in group)/2**30,
                    peak_reserved_gib=max(x['peak_reserved_bytes'] for x in group)/2**30)
            results.append(result)
    result=dict(batch=data['batch'],results=results,checks=check['checks'],shared_gpu=True,
        correctness_sha256=sha(folder/'correctness.json'),timing_sha256=sha(folder/'timing.json'))
    write(folder/'summary.json',result);return result


def run():
    batches=[summarize(folder) for folder in sorted(OUT.glob('b*')) if (folder/'timing.json').exists()]
    main=read(ROOT/'results/native_source_diversity/attempt1/lock.json')
    checked={p:sha(p)==d for p,d in main['source_sha256'].items() if p.endswith('.py')};assert all(checked.values())
    table=[];noise=[];numerical=[];cold=[]
    for batch in batches:
        for x in batch['results']:
            a,b=x['baseline'],x['candidate'];g=x['paired_gain_percent'];aa=x['baseline_duplicate_drift_percent'];bb=x['candidate_duplicate_drift_percent']
            table.append(f"| {batch['batch']} | {x['role']} | {x['mode']} | {a['median_wall_ms']:.2f} → {b['median_wall_ms']:.2f} | {x['median_gain_percent']:+.2f}% | {x['positive_blocks']}/8 | {a['peak_allocated_gib']:.3f} → {b['peak_allocated_gib']:.3f} |")
            noise.append(f"- B{batch['batch']}/{x['role']}/{x['mode']}：配对改善[{min(g):.2f},{max(g):.2f}]%；A/A漂移[{min(aa):.2f},{max(aa):.2f}]%；B/B漂移[{min(bb):.2f},{max(bb):.2f}]%。")
        for x in batch['checks']:
            cold.append(f"| {batch['batch']} | {x['role']} | {x['candidate']['first_train_seconds']:.3f} | {x['candidate']['first_eval_seconds']:.3f} |")
            n=sum(len(item['coordinates']) for item in x['update_tolerance_exceptions'])
            numerical.append(f"- B{batch['batch']}/{x['role']}：logit最大差{x['logit_max_abs']:.3g}，梯度最大绝对差{x['errors']['gradients']['max_abs']:.3g}、最大单参数张量相对L2差{x['errors']['gradients']['max_tensor_relative_l2']:.3g}；一步参数差最大{x['errors']['updated']['max_abs']:.3g}，逐元素旧update容差外{n}个坐标；更新后eval logit最大差{x['after_update_logit_max_abs']:.3g}。")
    stability=read(OUT/'stability.json') if (OUT/'stability.json').exists() else None
    stability_text='本轮尚未执行64步配对稳定性，不能据此声称长期训练精度不变。'
    if stability:
        assert stability['state']=='complete'
        for p,d in stability['sources'].items():assert sha(ROOT/p)==d,p
        stability_text='旧输入64步配对稳定性已完成，所有loss/梯度范数有限；详见stability.json。这不是完整训练终点精度确认。\n'
        for role in ('relative5','distance_mlp'):
            rows=[x for x in stability['measurements'] if x['role']==role]
            final=next(x for x in stability['final_models'] if x['role']==role)
            stability_text+=f"\n- {role}：最大loss差{max(abs(x['baseline']['loss']-x['candidate']['loss']) for x in rows):.3g}；最终参数最大差{final['max_parameter_abs']:.3g}，旧probe输出最大差{final['old_probe_logit_max_abs']:.3g}。"
    summary=dict(batches=batches,stability_completed=bool(stability),main_python_source_count=len(checked),main_sources_unchanged=True,shared_gpu=True)
    write(OUT/'summary.json',summary)
    text=f'''# 局部torch.compile：训练与推理验证

测试位置首层Linear+GELU和四层attention block的Inductor编译，几何计算和分类头保持eager。
对照为geometry_factored；未叠加先前融合梯度、专用contraction或布局重排。
参数、公式、FP32网络/FP64几何、TF32关闭、损失和优化器不变。
fullgraph=True、dynamic=False；CUDA Graph和max-autotune显式关闭，编译线程1。
未修改主训练；{len(checked)}个主任务冻结Python文件结束时均未变化。

## 数值检查

同权重、同输入、同dropout种子，输出和全部参数梯度按原容差核对；
一步AdamW更新例外全部登记，再检查更新后eval输出。这一处理在COMPILE_PLAN中事前规定，
没有把更新例外隐藏为逐参数严格等价，也没有裁零小梯度或改变优化器。

{chr(10).join(numerical)}

{stability_text}

## 首次成本与稳定运行分开

以下是候选首次调用耗时，含编译、缓存加载与实际执行，并非纯编译时间。
B4五维首先运行，Distance复用通用block编译缓存；B16也复用已有缓存基础，
不能把各行当作独立的全冷启动实验。缓存目录在/tmp/grope_compile1_inductor和/tmp/grope_compile1_triton。

| Batch | 方法 | 首次训练前后向秒 | 首次eval前向秒 |
| --- | --- | ---: | ---: |
{chr(10).join(cold)}

每个稳定计时标签预热2次、随后4次计时；8块A/A、B/B循环平衡，共128个单元/批次。
两种方法均执行相同编译。每标签重建相同模型权重与优化器，训练A/A、B/B最终loss逐位一致。
推理采用GPU驻留输入，包含全模型和几何计算；训练包含CPU热池gather、H2D、增强和完整优化步骤。
调度提前生成，不含冷盘IO、正式双来源混合、验证或checkpoint写入。
配对改善为每块两份候选/对照的平均耗时比，再取8块中位数；不是全局中位数之比。

| Batch | 方法 | 模式 | 整体中位ms | 配对耗时下降 | 正向块 | 峰值allocated GiB |
| --- | --- | --- | ---: | ---: | ---: | ---: |
{chr(10).join(table)}

{chr(10).join(noise)}

共享GPU结果只作工程筛选；重复块不是独立资源复现。编译成本不能从稳态时间中消失：
短作业必须计入首次调用，实际回本取决于稳态净收益，不从噪声中的微小差强算回本。
本轮没有改变核心PE研究结论；适用范围是当前本机PyTorch2.6、两种模型及已测形状。

## 当前采用边界

保留局部编译作为可选执行候选，不接入正在进行的科学训练。
推理、训练时延和显存分别判断；不能因为Distance推理明显改善，就宣称五维训练也有同幅度收益。
后续若考虑CUDA Graph或数据预取，应另做匹配比较，不把编译收益重复计入。
同样，若与已有融合梯度组合，必须测真实组合的时间与内存，不能相加百分比。

[固定计划](../../../experiments/efficiency_sidecar/COMPILE_PLAN.md) · [汇总](summary.json)
'''
    (OUT/'report.md').write_text(text)
    print('\n'.join(table))

if __name__=='__main__':run()
