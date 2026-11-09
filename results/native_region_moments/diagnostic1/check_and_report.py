"""Independent input arithmetic review and descriptive diagnostic report."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.linalg import eigvalsh

OUT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
lock=read(OUT/'lock.json');result=read(OUT/'result.json')
assert sha(OUT/'lock.json')==result['lock_sha256']==(OUT/'lock.sha256').read_text().strip()
for p,d in lock['source_sha256'].items():assert sha(Path(p))==d,p
assert read(OUT/'terminal.json')['status']=='complete'
assert sha(OUT/'arrays.npz')==result['arrays_sha256']
with np.load(OUT/'arrays.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
assert result['objects']==120 and result['regions']==7680 and result['pairs']==483840
assert not result['task_scores_read'] and not result['utility_evidence']
assert result['peak_allocated_bytes']<=3*2**30
errors=dict(raw_second_moment=0.,generalized_eigenvalues=0.,descriptor=0.,neighbor_distances=0.)
eye=np.eye(3);epsilon=lock['epsilon'];mask=~np.eye(64,dtype=bool)
for i in range(120):
    c=arrays['input_centers'][i].astype(np.float64);s=arrays['input_covariances'][i]
    anchors=arrays['anchors'][i];groups=arrays['groups'][i]
    # Independent greedy FPS and sorted Euclidean neighbor distances.
    nearest=np.full(512,np.inf);chosen=0;expected=[]
    for _ in range(64):
        expected.append(chosen)
        nearest=np.minimum(nearest,((c-c[chosen])**2).sum(-1))
        chosen=int(nearest.argmax())
    np.testing.assert_array_equal(anchors,expected)
    d2=((c[anchors,None]-c[None])**2).sum(-1)
    selected_d2=np.take_along_axis(d2,groups,axis=-1)
    errors['neighbor_distances']=max(errors['neighbor_distances'],float(abs(np.sort(d2,-1)[:,:32]-selected_d2).max()))
    grouped=c[groups];mean=grouped.mean(1)
    second=(s[groups]+grouped[...,None]*grouped[...,None,:]).mean(1)
    total=second-mean[...,None]*mean[...,None,:]
    errors['raw_second_moment']=max(errors['raw_second_moment'],float(abs(total-arrays['total'][i]).max()))
    np.testing.assert_allclose(total,arrays['total'][i],rtol=1e-9,atol=1e-10)
    physical=(s[groups]-epsilon*eye).mean(1)
    generalized=np.stack([eigvalsh(a,b) for a,b in zip(physical,arrays['total'][i])])
    errors['generalized_eigenvalues']=max(errors['generalized_eigenvalues'],float(abs(generalized-arrays['directional_fraction_eigenvalues'][i]).max()))
    for name,cov in [('full',arrays['total'][i]),('centers',arrays['between'][i]+epsilon*eye)]:
        delta=mean[:,None]-mean[None]
        inv=np.linalg.inv(cov)
        metric=np.einsum('ijd,ide,ije->ij',delta,inv,delta)
        trace=np.einsum('ide,jed->ij',inv,cov)
        phi=np.log1p(np.maximum(np.stack((metric,metric.T,trace,trace.T),-1),0))[mask]
        errors['descriptor']=max(errors['descriptor'],float(abs(phi-arrays[name+'_shape_descriptor'][i]).max()))
    assert arrays['shape_descriptor_signed_difference'][i,:,:2].max()<1e-8
assert max(errors.values())<1e-7,errors

# Recompute context-median summaries from retained arrays, not report numbers.
statistics={
 'trace_physical_fraction':arrays['trace_physical_fraction'],
 'trace_floor_fraction':arrays['trace_floor_fraction'],
 'directional_fraction_mean':arrays['directional_fraction_eigenvalues'].mean(-1),
 'directional_fraction_min':arrays['directional_fraction_eigenvalues'][...,0],
 'directional_fraction_max':arrays['directional_fraction_eigenvalues'][...,-1],
 'shape_descriptor_relative_l2':arrays['shape_descriptor_relative_l2']}
for k,name in enumerate(('metric_i','metric_j','trace_i','trace_j')):
    statistics[name+'_absolute_difference']=abs(arrays['shape_descriptor_signed_difference'][...,k])
medians={k:np.median(v,axis=1) for k,v in statistics.items()}
for k,v in medians.items():
    np.testing.assert_allclose(np.quantile(v,lock['quantile_probabilities']),result['context_median_summary'][k]['quantiles'],rtol=1e-12,atol=1e-12)
checks=dict(passed=True,completed_unix=time.time(),errors=errors,objects=120,groups=7680,
            independent_generalized_eigenproblems=7680,source_files_checked=len(lock['source_sha256']),
            script_sha256=sha(Path(__file__)),result_sha256=sha(OUT/'result.json'))
with (OUT/'execution_checks.json').open('x') as f:json.dump(checks,f,indent=2);f.write('\n')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
keys=['trace_physical_fraction','directional_fraction_mean','directional_fraction_max']
axes[0].boxplot([100*medians[k] for k in keys],tick_labels=['Total variance','Mean direction','Strongest direction'])
axes[0].set_ylabel('Native shape contribution (%)')
axes[0].tick_params(axis='x',labelrotation=15)
keys=['metric_i_absolute_difference','metric_j_absolute_difference','trace_i_absolute_difference','trace_j_absolute_difference']
axes[1].boxplot([medians[k] for k in keys],tick_labels=['metric i','metric j','trace i','trace j'])
axes[1].set_ylabel('Absolute log1p descriptor change')
fig.suptitle('Object medians: 120 old fit objects; no task scores')
fig.savefig(OUT/'moment_contributions.png',dpi=160);plt.close(fig)
lines=['# 原生高斯区域矩：输入可行性结果','',
       '2026-11-09。120 个旧训练对象、7,680 个局部组、483,840 个非自身区域对。零训练，未读取验证、评价或确认集分数。',
       '**区域总方差主要来自中心散布，但原生协方差仍对部分方向及五维输入产生可测影响；这不是任务收益证据。**','',
       '| 对象内中位数的分布 | 5% 分位 | 中位数 | 95% 分位 |','| --- | ---: | ---: | ---: |']
labels=[('trace_physical_fraction','原生形状占总方差 %',100),
        ('directional_fraction_mean','方向归一化贡献均值 %',100),
        ('directional_fraction_max','最强方向贡献 %',100),
        ('shape_descriptor_relative_l2','四维形状关系相对 L2 改变 %',100),
        ('metric_i_absolute_difference','log1p 位移度量绝对差',1),
        ('trace_i_absolute_difference','log1p 协方差关系绝对差',1)]
for key,label,scale in labels:
    q=result['context_median_summary'][key]['quantiles']
    lines.append(f'| {label} | {q[0]*scale:.4f} | {q[2]*scale:.4f} | {q[4]*scale:.4f} |')
lines+=['','这里每个对象先对全部固定组/区域对取中位数，再在 120 个对象间取分位数；不是置信区间。双端量使用全部有向对，其分布按转置对称，故表中只列一端。',
 '','固定构造为 512 个原基元 → 64 个 FPS 区域，每区 32 个中心近邻，等权 `C = 中心散布 B + 平均原生协方差 W + 原 1e-8 I`；floor 只出现一次。分组、中心与同一五维公式均相同，对照仅把 W 从 PE 几何中移除。',
 '','典型组的原生形状只占总方差约 3.26%，约 96.74% 来自中心散布（floor 占比可忽略）。但最强归一化方向的典型贡献约 12.37%，说明不能只看 trace 就宣布物理协方差完全消失。四个形状关系输入的联合相对变化中位数约 2.62%，95% 对象分位约 6.48%；它是输入改变幅度，不是准确率贡献率。',
 '','本结果不支持把区域化后的全部效果称为原生高斯形状的收益。若后续进行学习验证，必须比较同一聚合内容、同一骨干下的 F(full)、F(centers-only PE) 和强中心/角度 PE。只有 F(full) 比这些对照仍有可靠额外收益，才支持物理基元形状通过区域 PE 提供额外用途。若两种 F 相近且都优于 Distance，结论应主要归于区域关系组织；若连强对照用途也不成立，则结束该输入接法。',
 '','本轮不晋升区域化，也不自动启动训练：输入差异大小无法决定收益，等权与重叠 KNN 组不保持整个对象的渲染密度，矩闭合还丢失高阶和多峰信息。已有原生分类、区域重构、RGB 与可靠输出的混合/负结果不变。后续若开展上述有限学习比较，需要先固定其预算、初始化、选模和决策，不能拿聚合前后的总分差直接归因于 PE。',
 '','合并矩不是新公式；[Hierarchical 3D Gaussians 式(3)–(4)](https://arxiv.org/html/2406.12080v1#S4.SS1) 已使用组内协方差加中心散布，其贡献权重与本项固定等权不同。[ShapeSplat §4.2–4.3](https://arxiv.org/html/2408.10906v1#S4) 已研究基元分组与可学习特征池化，本项未复现其方法或性能。仓库旧静态混合矩实验也保留，不把同一矩恒等式当作新发现。',
 '',f"四项实现测试通过；独立 NumPy 原始二阶矩、邻域距离/锚点、SciPy 广义特征值和完整描述符复算通过。最大误差分别见 execution_checks.json：{errors}。这些核对保证本轮统计的含义，不构成用途确认。实际 GPU 峰值约 {result['peak_allocated_bytes']/2**20:.2f} MiB。", 
 '','图表和最终解释仍需实际查看，审阅状态以 review.json 为准。完整目标尚未完成。',
 '','[事前方案](../../../docs/native_region_moments_plan.md) · [固定身份](lock.json) · [原始结果](result.json) · [独立复算](execution_checks.json) · [图](moment_contributions.png)','']
assert not (OUT/'report.md').exists()
(OUT/'report.md').write_text('\n'.join(lines))
print('MOMENT_DIAGNOSTIC_CHECKS_PASSED',errors,flush=True)
