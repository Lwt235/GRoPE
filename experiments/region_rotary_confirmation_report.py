"""Complete descriptive tables and plots; scientific review remains separate."""
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.round65_common import SEEDS,read,write,sha
from experiments.region_rotary_confirmation import OUT,ROLES

LABELS=dict(relative5='Full five + rotary',distance_mlp='Distance + rotary',geotransformer_rpe='Geo PE + rotary',
    fixed_isotropic5='Sphere + rotary',predicted_hrope='Predicted HRoPE',fixed_hrope='Fixed H=K')


def main():
    result=read(OUT/'result.json');checks=read(OUT/'independent_checks.json');data=read(OUT/'data_manifest.json')
    assert checks['passed'] and checks['result_sha256']==sha(OUT/'result.json')
    assert not (OUT/'tables.md').exists()
    records={(r['role'],r['seed']):r for r in result['records']}
    pairs=data['cache']['pairs'];places=len(result['scenes'])
    lines=['# 固定五维＋旋转：独立地点确认的全部结果','',
        f'计划20地点/1920对，实际{places}地点/{pairs}对；机械排除{len(data["excluded"])}对。18个既有模型，零训练/选模，两个固定输入条件。','',
        '| 方法 | 基础成功率 % | 基础三种子 % | 重表达成功率 % | 重表达三种子 % |',
        '| --- | ---: | --- | ---: | --- |']
    for role in ROLES:
        values=[[100*records[role+suffix,s]['macro_success_rate'] for s in SEEDS] for suffix in ('','_reframed')]
        lines.append(f'| {LABELS[role]} | {np.mean(values[0]):.3f} | '+ '/'.join(f'{v:.3f}' for v in values[0])+
            f' | {np.mean(values[1]):.3f} | '+ '/'.join(f'{v:.3f}' for v in values[1])+' |')
    lines+=['','| 条件／F对照 | 平均差 pp | 三种子差 pp | 地点×种子95%区间 pp | 预定方向判断 |',
        '| --- | ---: | --- | --- | --- |']
    for row in result['comparisons']:
        condition='重表达' if row['left'].endswith('_reframed') else '基础'
        label=LABELS[row['right'].removesuffix('_reframed')];lo,hi=row['intervals_95_pp']['crossed']
        lines.append(f'| {condition}／{label} | {row["mean_gain_pp"]:+.3f} | '+
            '/'.join(f'{v:+.3f}' for v in row['paired_seed_gains_pp'])+f' | [{lo:.3f},{hi:.3f}] | {row["direction"]} |')
    lines+=['','| 方法/条件/种子 | 区域R@1 % | 全查询正确 % | PCK % | soft EPE m | RMSE中位/p90 m | 无效解 |',
        '| --- | ---: | ---: | ---: | ---: | --- | ---: |']
    def number(x):return 'inf' if x is None else f'{x:.4f}'
    for role in ROLES:
        for condition,suffix in (('base',''),('reframed','_reframed')):
            for seed in SEEDS:
                row=records[role+suffix,seed]
                metrics=read(OUT/'evaluation'/(role+suffix)/str(seed)/'base.json')['metrics']
                macro=metrics['macro'];correct=np.mean([v['valid_and_correct_fraction'] for v in metrics['by_scene'].values()])
                tail=row['all_pair_errors']['rmse_m']
                lines.append(f'| {LABELS[role]}/{condition}/{seed} | {100*macro["region_recall"]:.3f} | {100*correct:.3f} | '+
                    f'{100*macro["center_pck"]:.3f} | {macro["soft_epe"]:.4f} | {number(tail["median"])}/{number(tail["p90"])} | {row["invalid_estimates"]} |')
    lines+=['','| 坐标变化或相对优势变化 | 平均 pp | 地点×种子95%区间 pp |','| --- | ---: | --- |']
    for row in result['method_changes']+result['interactions']:
        label=LABELS[row['role']] if 'role' in row else 'F − '+LABELS[row['right']]
        lo,hi=row['intervals_95_pp']['crossed']
        lines.append(f'| {label} | {row["mean_gain_pp"]:+.3f} | [{lo:.3f},{hi:.3f}] |')
    for role in ('identity','gt_nearest'):
        row=records[role,0]
        lines+=['',f'{role}：成功率{100*row["macro_success_rate"]:.3f}%，无效解{row["invalid_estimates"]}。']
    lines+=['','GT辅助不可部署且不是上界。全部逐地点、旋转/平移/RMSE尾部及38组原始评分见result.json。',
        '两个条件使用相同物理观测，不是两批独立确认；目录不同不证明没有物理别名。',
        '本研究不是完整HRoPE/GeoTransformer系统复现；未验证原生3DGS、RGB或新相机视角。科学审阅待完成。']
    (OUT/'tables.md').write_text('\n'.join(lines)+'\n')
    folder=OUT/'figures';folder.mkdir(exist_ok=False)
    fig,axes=plt.subplots(2,2,figsize=(15,10),layout='constrained')
    for index,(suffix,label,color) in enumerate((('','Base','#1677a3'),('_reframed','Reframed','#8154a5'))):
        left,right=axes[index]
        for i,role in enumerate(ROLES):
            values=[100*records[role+suffix,s]['macro_success_rate'] for s in SEEDS]
            left.bar(i,np.mean(values),color=color)
            left.scatter(i+np.linspace(-.10,.10,3),values,color='white',edgecolors='black',s=25,zorder=3)
        left.set_xticks(range(6),[LABELS[r] for r in ROLES],rotation=25,ha='right')
        left.set_ylim(0,100);left.set_ylabel('Scene-balanced alignment success (%)')
        left.set_title(f'{label}: {places} new places / {pairs} pairs')
        comparisons=result['comparisons'][index*5:(index+1)*5]
        for i,row in enumerate(comparisons):
            lo,hi=row['intervals_95_pp']['crossed']
            right.plot([lo,hi],[i,i],color=color,lw=2);right.plot(row['mean_gain_pp'],i,'o',color=color)
            right.scatter(row['paired_seed_gains_pp'],i+np.linspace(-.08,.08,3),color=color,marker='x')
        right.set_yticks(range(5),['F − '+LABELS[r['right'].removesuffix('_reframed')] for r in comparisons])
        right.invert_yaxis();right.axvline(0,color='.4',ls='--');right.set_xlabel('Difference (pp), 95% seed × place interval')
        right.set_title(label+' paired comparisons')
        left.grid(axis='y',alpha=.2);right.grid(axis='x',alpha=.2)
        left.set_axisbelow(True);right.set_axisbelow(True)
    fig.savefig(folder/'pose_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for axis,rows,labels,title in ((axes[0],result['method_changes'],[LABELS[r['role']] for r in result['method_changes']],'Reframed − base'),
        (axes[1],result['interactions'],['F − '+LABELS[r['right']] for r in result['interactions']],'Change in F-minus-control advantage')):
        for i,row in enumerate(rows):
            lo,hi=row['intervals_95_pp']['crossed'];axis.plot([lo,hi],[i,i],color='#8154a5',lw=2);axis.plot(row['mean_gain_pp'],i,'o',color='#8154a5')
        axis.set_yticks(range(len(rows)),labels);axis.invert_yaxis();axis.axvline(0,color='.4',ls='--')
        axis.set_title(title);axis.set_xlabel('Change (pp), 95% seed × place interval');axis.grid(axis='x',alpha=.2)
    fig.savefig(folder/'changes.png',dpi=160);plt.close(fig)
    write(folder/'manifest.json',dict(created_unix=time.time(),result_sha256=sha(OUT/'result.json'),renderer_sha256=sha(__file__),
        figures={p.name:sha(p) for p in folder.glob('*.png')},visual_review_pending=True))
    print('ROTARY_CONFIRMATION_TABLES_AND_FIGURES_READY',flush=True)


if __name__=='__main__':main()
