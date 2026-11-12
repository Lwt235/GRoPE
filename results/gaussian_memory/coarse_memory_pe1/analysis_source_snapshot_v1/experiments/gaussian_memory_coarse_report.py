"""Readable results and ordinary experiment figures, preserving all places."""
import csv
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.gaussian_memory_coarse_data import *


def figures(a,p):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    kinds=['official','single_view','coarse_frozen','center','sphere','gaussian'];labels=['Official dense','Single view','Frozen coarse','Center','Sphere','Gaussian']
    colors=['#999999','#c6c6c6','#e0b065','#3e7fb1','#58a978','#c86357']
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for ax,metric,title in [(axes[0,0],'hidden_log_mse','Persistent ROI raw log-MSE'),(axes[0,1],'hidden_centered_log_mse','Persistent ROI spatial log-MSE')]:
        x=np.arange(len(kinds));width=.36
        for shift,n in enumerate(CAPACITIES):
            vals=[a['tables'][str(n)]['persistent'][k][metric] for k in kinds]
            ax.bar(x+(shift-.5)*width,vals,width,label=f'{n} slots',color=colors,alpha=1 if shift==0 else .45,edgecolor=colors,hatch=None if shift==0 else '//')
        ax.set_xticks(x,labels,rotation=22,ha='right');ax.set_title(title);ax.legend(fontsize=8)
    ax=axes[1,0];pairs=['gaussian_vs_center','gaussian_vs_sphere','sphere_vs_center'];ys=np.arange(3)
    bound=5.
    for shift,n in enumerate(CAPACITIES):
        vals=[a['comparisons'][str(n)]['persistent'][pair]['hidden_centered_log_mse'] for pair in pairs]
        x=np.array([v['relative_gain_percent'] for v in vals]);ci=np.array([v['relative_gain_percent_95'] for v in vals]);bound=max(bound,float(abs(ci).max())*1.15)
        ax.errorbar(x,ys+(shift-.5)*.18,xerr=np.stack((x-ci[:,0],ci[:,1]-x)),fmt='o',capsize=3,label=f'{n} slots')
    ax.set_yticks(ys,['Gaussian - Center','Gaussian - Sphere','Sphere - Center']);ax.axvline(0,c='black',lw=.8);ax.set_xlim(-bound,bound);ax.set_xlabel('ROI spatial error reduction (%), paired 95% interval');ax.legend();ax.set_title('Positive means lower error; minimum display range +/-5%')
    ax=axes[1,1]
    for index,k in enumerate(kinds[2:]):
        x=[a['budgets'][str(n)]['MiB'] for n in CAPACITIES];y=[a['tables'][str(n)]['persistent'][k]['hidden_centered_log_mse'] for n in CAPACITIES]
        ax.plot(x,y,'o-',label=labels[index+2],color=colors[index+2])
    ax.set_xlabel('Persistent source state (MiB, camera/metadata included)');ax.set_ylabel('ROI spatial log-MSE');ax.set_title('Static pooled memory; construction peak counted separately');ax.legend(fontsize=8)
    for suffix in ('png','pdf'):fig.savefig(folder/f'summary.{suffix}',dpi=160)
    plt.close(fig)
    selected=a['fixed_example_ids'];seed=SEEDS[0];lookup={}
    for kind in ('center','gaussian'):
        data=load(CACHE/'evaluation'/f'{kind}_{seed}.pt')
        for key,pred in zip(data['keys'],data['depth_metric']):lookup[(kind,*key)]=pred.numpy()
    fig,axes=plt.subplots(len(selected),5,figsize=(16,3.1*len(selected)),squeeze=False,layout='constrained')
    image_error=None;depth_image=None
    for row,rid in enumerate(selected):
        inp=load(input_path(rid,11,'persistent'));target=inp['target'].numpy();mask=inp['valid'].numpy()&inp['hidden'].numpy()
        rgb=inp['images'][-1].permute(1,2,0).numpy();b=lookup['center',rid,11,'persistent',256];d=lookup['gaussian',rid,11,'persistent',256]
        eb=np.log(np.maximum(b[mask],1e-6)/target[mask]);ed=np.log(np.maximum(d[mask],1e-6)/target[mask]);diff=np.full_like(target,np.nan);diff[mask]=(eb-eb.mean())**2-(ed-ed.mean())**2
        vmax=max(.01,float(np.nanpercentile(abs(diff),98)))
        axes[row,0].imshow(rgb);axes[row,0].set_ylabel(rid)
        for col,img in ((1,np.where(inp['valid'].numpy(),target,np.nan)),(2,b),(3,d)):
            depth_image=axes[row,col].imshow(img,vmin=0,vmax=8,cmap='viridis')
        image_error=axes[row,4].imshow(diff,cmap='RdBu',vmin=-vmax,vmax=vmax)
        axes[row,4].set_xlabel(f'Color range +/-{vmax:.3g}')
        for ax in axes[row]:ax.set_xticks([]);ax.set_yticks([])
    for ax,title in zip(axes[0],['Current masked RGB','Sensor depth (m)','Center raw depth','Gaussian raw depth','ROI spatial squared-log error reduction']):ax.set_title(title,fontsize=10)
    fig.colorbar(depth_image,ax=axes[:,1:4].ravel().tolist(),shrink=.65,label='Target: m; model: raw output value (no GT scale alignment)')
    # Each error panel has its stated symmetric range; do not attach a misleading shared scale.
    fig.suptitle(f'256 slots, fixed seed {seed}; first three IDs plus largest mean spatial regression')
    for suffix in ('png','pdf'):fig.savefig(folder/f'cases.{suffix}',dpi=150)
    plt.close(fig)
    write_json(OUT/'figure_manifest.json',dict(files={str(x.relative_to(OUT)):sha256(x) for x in folder.iterdir()},minimum_comparison_display_percent=5,minimum_error_color_range=.01,selection=read(OUT/'figure_selection.json'),viewed=False))


def render():
    p=plan();a=read(OUT/'analysis.json');checks=read(OUT/'independent_result_checks.json');runtime=read(OUT/'runtime.json') if (OUT/'runtime.json').exists() else None
    intro=(OUT/'interpretation.md').read_text() if (OUT/'interpretation.md').exists() else '**Interpretation pending result review.**'
    lines=['# coarse_memory_pe1：有限场景记忆中的范围与形状','',intro,'','## 设计与证据边界','',
        '三组共享来源、FPS/最近中心成员、聚合K/V、成员数补偿、训练容量和实际样本顺序，只比较强中心、球形范围、完整协方差。256/1024两档容量等量训练，9次2000步，最终检查点；432份适配输出，72份新无训练参照，另复用24份父轮已独立评分的官方输出。所有地点已用，无新确认。',
        '', '查询两遍都从当前RGB骨干特征开始，全部24个global层读取粗记忆；没有读取完整历史的query prefix或早期DPT特征旁路。源writer和DPT冻结，24层Q/输出rank8适配与共享关系映射合计786,784参数。第一遍预测相机/深度生成查询几何，三组共用，查询真值不进入PE。该实验是受限接收端训练，不是全模型训练。',
        '', '源构建允许暂时使用完整源K/V，查询前释放；这是静态构建后的读取预算比较，尚未实现全程固定内存或在线融合。高斯表示观测支持区域，不是光度优化后的3DGS。K/V仍按层保留，不能称最终单特征紧凑架构。',
        '', 'B与C/D是固定容量的描述选择，非严格嵌套特征消融；C/D为同迹形状控制。跨容量的第一遍几何也可能改变，容量交互不能唯一归因形状。官方dense和single-view均未作本轮匹配适配，只是质量/成本背景；PE主证据来自同预算B/C/D。',
        '', '主终点事前固定为256槽持续遮挡ROI去均值log-MSE；原始误差、均值偏差平方、全图、完整输入和成本均保留。统计分解不证明唯一物理尺度病因，模型原始输出未按查询真值对齐。','', '## 主表','', '| 槽数 | 条件 | 方法 | ROI原始误差 | ROI空间误差 | ROI偏差平方 | 全图原始误差 |','|---|---|---|---:|---:|---:|---:|']
    flat=[]
    for n in CAPACITIES:
        for c in p['conditions']:
            for k,row in a['tables'][str(n)][c].items():
                lines.append(f"| {n} | {c} | {k} | {row['hidden_log_mse']:.6f} | {row['hidden_centered_log_mse']:.6f} | {row['hidden_bias_squared']:.6f} | {row['log_mse']:.6f} |")
                flat.append(dict(capacity=n,condition=c,kind=k,**row))
    lines+=['','## 持续遮挡配对比较','','正数表示候选误差更低；5000次地点和种子配对重采样。三个种子和十二旧地点不是一般分布保证。区间跨零不证明等效，极小正区间也不自动构成实用收益。','','| 槽数 | 比较 | ROI空间降误差% [95%区间] | ROI原始降误差% [95%区间] | 空间改善地点/种子 |','|---|---|---:|---:|---|']
    for n in CAPACITIES:
        for pair,data in a['comparisons'][str(n)]['persistent'].items():
            s=data['hidden_centered_log_mse'];r=data['hidden_log_mse'];sc=s['relative_gain_percent_95'];rc=r['relative_gain_percent_95']
            lines.append(f"| {n} | {pair} | {s['relative_gain_percent']:+.6f} [{sc[0]:+.6f}, {sc[1]:+.6f}] | {r['relative_gain_percent']:+.6f} [{rc[0]:+.6f}, {rc[1]:+.6f}] | {s['positive_places']}/12; {s['positive_seeds']}/3 |")
    lines+=['','![Summary](figures/summary.png)','','## 成本与实现检查','',f"训练循环合计{a['training_worker_hours']:.3f} worker小时、峰值已分配{a['training_peak_GiB']:.3f}GiB。432份适配输出及72份新参照独立NumPy评分，最大指标差{checks['max_metric_abs']:.3g}。逐种子初始化、计划采样与实际采样日志配对一致，每拟合两档容量各2000个微样本。",'','矩合并与NumPy独立复算通过。无位置变更的全层查询在相同零global掩码下，与官方模型输出完全一致；不加该零掩码时BF16不同注意力内核导致相对RMS约0.0582%，已保留诊断而未更改科学算子。该检查不代替任务效用。','']
    for n in CAPACITIES:
        b=a['budgets'][str(n)];lines.append(f"- {n}槽：完整持久状态{b['MiB']:.3f}MiB，包含特殊token和相机缓存。完整协方差元数据可省的上界不足再容纳一个全24层K/V槽，因此同上限中心组不能多加一槽。静态源构建峰值{b['static_construction_peak_GiB']:.3f}GiB、中位源重放{b['source_build_seconds_median']:.3f}s。")
    if runtime:
        lines+=['','两遍查询实测（固定两个训练片段，两条件，两容量，首种子，预热后各三次，源驻GPU，排除模型/磁盘加载）：','','| 槽数 | Center ms | Sphere ms | Gaussian ms |','|---|---:|---:|---:|']
        for n in CAPACITIES:
            t=runtime['median_seconds'][str(n)];lines.append(f"| {n} | {t['center']*1000:.1f} | {t['sphere']*1000:.1f} | {t['gaussian']*1000:.1f} |")
        lines+=['',runtime['scope']]
    lines+=['','## 示例与反例','',f"预定前三ID加最大空间退步例，去重后为{a['fixed_example_ids']}；最大退步地点{a['worst_place']}。图用固定首种子，选例按三种子平均；全部地点和反例保留。误差图仅展示ROI内去均值平方log误差差，正值蓝色为高斯改善，各行色域已标明，至少±0.01以免放大小数差异。",'','![Cases](figures/cases.png)','','[固定计划](../../../docs/gaussian_memory_coarse_budget_plan.md) · [完整分析](analysis.json) · [独立评分](independent_result_checks.json) · [训练身份](training_lock.json)']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    with (OUT/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    figures(a,p);write_json(OUT/'report_manifest.json',dict(complete=(OUT/'interpretation.md').exists() and runtime is not None,files={name:sha256(OUT/name) for name in ('report.md','metrics.csv','analysis.json','independent_result_checks.json','figure_manifest.json')},plan_sha256=sha256(OUT/'plan.json')))


if __name__=='__main__':render()
