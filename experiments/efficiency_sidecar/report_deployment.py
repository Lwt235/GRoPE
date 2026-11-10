"""Recompute adoption evidence from raw outputs, without rerunning GPU work."""
from pathlib import Path
import statistics as st
import numpy as np
from experiments.round65_common import ROOT, read, write, sha
from experiments.round65_run import metrics

OUT=ROOT/'results/efficiency_sidecar/deployment1'


def main():
    q=read(OUT/'queue_status.json');assert q['state']=='complete'
    review=dict(state='complete',evaluation={},combination={},sources={},resource_snapshots=0,
        full_training_endpoint_equivalence_established=False,new_scientific_confirmation=False)
    for stage in ('evaluation','combination'):
        for role in ('relative5','distance_mlp'):
            path=OUT/stage/role/'result.json';r=read(path);assert r['state']=='complete'
            review['sources'][str(path.relative_to(ROOT))]=sha(path)
            for name,digest in r['sources'].items():assert sha(ROOT/name)==digest,name
            for name,digest in r['model_sources'].items():assert sha(ROOT/name)==digest,name
            for snap in r['resources']:
                processes=[int(line.split(',')[1]) for line in snap['apps'].splitlines()
                    if line.split(',')[0].strip()==r['gpu_uuid']]
                assert set(processes).issubset({r['pid']})
                review['resource_snapshots']+=1
            rows=r['measurements']
            if stage=='evaluation':
                arrays={};summary={};comparisons={}
                for row in rows:
                    assert sha(row['prediction_file'])==row['prediction_sha256']
                    with np.load(row['prediction_file'],allow_pickle=False) as data:
                        logits=data['logits'].copy();labels=data['labels'].copy();ids=data['ids'].copy()
                    computed=metrics(logits,labels)
                    for key in ('ce','oa','mca'):assert abs(computed[key]-row['metrics'][key])<1e-12
                    if row['arm'] in arrays:
                        a=arrays[row['arm']];assert np.array_equal(logits,a[0]) and np.array_equal(ids,a[2])
                    else:arrays[row['arm']]=(logits,labels,ids)
                for arm,(logits,labels,ids) in arrays.items():
                    first=next(x for x in rows if x['arm']==arm and x['block']==-1)
                    warm=[x for x in rows if x['arm']==arm and x['block']>=0]
                    summary[arm]=dict(first_seconds=first['total_seconds'],first_evaluation_seconds=first['seconds'],
                        first_batch_seconds=first['first_batch_seconds'],tail_batch_seconds=first['tail_batch_seconds'],
                        warm_seconds=st.median(x['seconds'] for x in warm),
                        peak_gib=max(x['peak_allocated_bytes'] for x in rows if x['arm']==arm)/2**30,
                        metrics=metrics(logits,labels),count=len(labels),repeat_outputs_identical=True)
                for baseline,candidate in (('original','geometry_factored'),('geometry_factored','compiled'),('original','compiled')):
                    a,b=arrays[baseline],arrays[candidate];assert np.array_equal(a[1],b[1]) and np.array_equal(a[2],b[2])
                    changed=np.flatnonzero(a[0].argmax(-1)!=b[0].argmax(-1))
                    ratios=[];drifts=[]
                    for block in range(2):
                        aa=[x['seconds'] for x in rows if x['arm']==baseline and x['block']==block]
                        bb=[x['seconds'] for x in rows if x['arm']==candidate and x['block']==block]
                        ratios.append(100*(1-sum(bb)/sum(aa)))
                        drifts.extend(100*abs(v[1]-v[0])/st.mean(v) for v in (aa,bb))
                    delta=summary[baseline]['warm_seconds']-summary[candidate]['warm_seconds']
                    extra=summary[candidate]['first_seconds']-summary[baseline]['first_seconds']
                    comparisons[baseline+'->'+candidate]=dict(logit_max_abs=float(np.max(np.abs(a[0]-b[0]))),
                        changed_predictions=len(changed),changed_ids=a[2][changed].tolist(),
                        oa_delta_pp=100*(summary[candidate]['metrics']['oa']-summary[baseline]['metrics']['oa']),
                        mca_delta_pp=100*(summary[candidate]['metrics']['mca']-summary[baseline]['metrics']['mca']),
                        ce_delta=summary[candidate]['metrics']['ce']-summary[baseline]['metrics']['ce'],
                        warm_block_reductions_pct=ratios,median_warm_reduction_pct=st.median(ratios),
                        median_same_arm_abs_drift_pct=st.median(drifts),
                        approximate_additional_warm_evaluations_to_amortize=max(0,extra/delta) if delta>0 else None)
                review[stage][role]=dict(arms=summary,comparisons=comparisons)
            else:
                summary={};comparisons={}
                for arm in ('geometry_factored','fused','compiled','combined'):
                    armrows=[x for x in rows if x['arm']==arm]
                    summary[arm]=dict(median_ms=st.median(x['wall_ms'] for x in armrows),
                        peak_gib=max(x['peak_allocated_bytes'] for x in armrows)/2**30)
                    for block in range(8):
                        pair=[x for x in armrows if x['block']==block];assert len(pair)==2
                        assert pair[0]['final_loss']==pair[1]['final_loss'],(role,arm,block)
                for baseline,candidate in (('geometry_factored','fused'),('geometry_factored','compiled'),
                    ('geometry_factored','combined'),('fused','combined'),('compiled','combined')):
                    ratios=[];drifts=[]
                    for block in range(8):
                        a=[x['wall_ms'] for x in rows if x['arm']==baseline and x['block']==block]
                        b=[x['wall_ms'] for x in rows if x['arm']==candidate and x['block']==block]
                        ratios.append(100*(1-sum(b)/sum(a)))
                        drifts.extend(100*abs(v[1]-v[0])/st.mean(v) for v in (a,b))
                    comparisons[baseline+'->'+candidate]=dict(median_reduction_pct=st.median(ratios),
                        block_reductions_pct=ratios,positive_blocks=sum(v>0 for v in ratios),
                        median_same_arm_abs_drift_pct=st.median(drifts))
                review[stage][role]=dict(arms=summary,comparisons=comparisons,correctness=r['checks'])
    write(OUT/'review.json',review)
    lines=['# 完整评价与组合优化验证','',
        '旧seed11模型、旧验证集765对象，单张RTX A6000独占采样，CPU1、网络FP32/几何FP64、TF32关闭。',
        '数据/权重身份核对、输入准备之外的模型读取和Python启动不计入下表；评价耗时包括缓存映射、gather/H2D、完整前向、输出回传、指标和预测写盘/fsync。首次总耗时另计私有模型适配和上卡。没有清空操作系统页缓存，不称冷磁盘性能。',
        '','## 完整评价（B4/N512，末批B1）','',
        '| 模型 | 实现 | 首次适配＋评价(s) | 暖评价中位(s) | 峰值分配(GiB) | OA / mCA |',
        '| --- | --- | ---: | ---: | ---: | --- |']
    for role,data in review['evaluation'].items():
        for arm,v in data['arms'].items():
            lines.append(f"| {role} | {arm} | {v['first_seconds']:.3f} | {v['warm_seconds']:.3f} | {v['peak_gib']:.3f} | {100*v['metrics']['oa']:.4f}% / {100*v['metrics']['mca']:.4f}% |")
    for role,data in review['evaluation'].items():
        c=data['comparisons']['geometry_factored->compiled']
        lines.extend(['',f"{role}：编译相对几何重排的暖评价配对节时 {c['median_warm_reduction_pct']:.2f}%，两块结果 {c['warm_block_reductions_pct']}；同实现重复绝对漂移中位 {c['median_same_arm_abs_drift_pct']:.2f}%。预测改变 {c['changed_predictions']}/765，最大logit差 {c['logit_max_abs']:.3g}。首评额外开销约需 {c['approximate_additional_warm_evaluations_to_amortize']} 次暖评价摊销（描述性估计）。"])
    lines.extend(['','## 四臂完整训练步（B16/N512）','',
        '| 模型 | 实现 | 中位ms | 峰值GiB |','| --- | --- | ---: | ---: |'])
    for role,data in review['combination'].items():
        for arm,v in data['arms'].items():lines.append(f"| {role} | {arm} | {v['median_ms']:.3f} | {v['peak_gib']:.3f} |")
    for role,data in review['combination'].items():
        lines.append('')
        for name,c in data['comparisons'].items():
            lines.append(f"- {role} {name}：配对节时 {c['median_reduction_pct']:.2f}%，{c['positive_blocks']}/8块正向，重复绝对漂移中位 {c['median_same_arm_abs_drift_pct']:.2f}%。")
    lines.extend(['','## 解释边界','',
        f"核对 {review['resource_snapshots']} 次资源快照，无其他同卡计算进程；其他卡仍有工作。完整评价各臂重复输出相同，所有指标从保存预测重算。训练每块同臂重复末步损失完全一致。",
        '组合通过同权重前向、全参数梯度及一步后输出核对；AdamW参数更新例外在review.json保留。该检查不替代完整训练效果验证，也不推翻此前编译64步轨迹分叉。',
        '本轮未训练新科学模型、未读取新确认集、未改变主任务。完整训练终点配对随下一项确需训练的科学任务安排。原模型无几何输入梯度的适用限制保留。'])
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print(OUT/'report.md')


if __name__=='__main__':main()
