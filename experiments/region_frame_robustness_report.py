"""Tables and plots for a frozen, zero-fit coordinate robustness study."""
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.round65_common import SEEDS,read,write,sha
from experiments.region_frame_robustness import OUT,ROLES,owner

LABELS=dict(relative5='Full five + rotary',distance_mlp='Distance + rotary',geotransformer_rpe='Geo PE + rotary',
    fixed_isotropic5='Sphere + rotary',predicted_hrope='Predicted HRoPE',fixed_hrope='Fixed H=K')


def main():
    result=read(OUT/'result.json');check=read(OUT/'independent_checks.json')
    assert check['passed'] and check['result_sha256']==sha(OUT/'result.json')
    assert not (OUT/'tables.md').exists()
    rows={(r['role'],r['seed']):r for r in result['records']}
    comparisons=result['comparisons'];changes=result['method_changes'];interactions=result['interactions']
    lines=['# 固定模型的局部坐标重表达：全部结果','',
        '18个固定模型，零训练；同480旧帧对、同区域、相机信息、标签和求解器。每帧一个固定独立SO(3)旋转和平移，无新视角或新确认。','',
        '| 方法 | 原成功率 % | 重表达成功率 % | 重表达三种子 % | 变化 pp及95%区间 |',
        '| --- | ---: | ---: | --- | --- |']
    for role in ROLES:
        old=[100*rows[role,s]['macro_success_rate'] for s in SEEDS]
        new=[100*rows[role+'_reframed',s]['macro_success_rate'] for s in SEEDS]
        change=next(r for r in changes if r['role']==role);lo,hi=change['intervals_95_pp']['crossed']
        lines.append(f'| {LABELS[role]} | {np.mean(old):.3f} | {np.mean(new):.3f} | '+
            '/'.join(f'{v:.3f}' for v in new)+f" | {change['mean_gain_pp']:+.3f} [{lo:.3f},{hi:.3f}] |")
    lines+=['','| 重表达后F相对对照 | 差 pp | 三种子差 pp | 联合95%区间 pp | 相对差值相对base的变化 pp及区间 |',
        '| --- | ---: | --- | --- | --- |']
    for row,interaction in zip(comparisons,interactions):
        role=row['right'].removesuffix('_reframed');lo,hi=row['intervals_95_pp']['crossed'];a,b=interaction['intervals_95_pp']['crossed']
        lines.append(f"| {LABELS[role]} | {row['mean_gain_pp']:+.3f} | "+'/'.join(f'{v:+.3f}' for v in row['paired_seed_gains_pp'])+
            f" | [{lo:.3f},{hi:.3f}] | {interaction['mean_gain_pp']:+.3f} [{a:.3f},{b:.3f}] |")
    lines+=['','| 方法/种子 | R@1原→新 % | 全查询正确原→新 % | PCK原→新 % | soft EPE原→新 m | 新RMSE中位/p90 m | top1改变 全部/有效 % |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    def f(x):return 'inf' if x is None else f'{x:.4f}'
    for role in ROLES:
        for seed in SEEDS:
            base=read(owner(role)/'evaluation'/role/str(seed)/'base.json')['metrics']
            meta=read(OUT/'evaluation'/(role+'_reframed')/str(seed)/'base.json');new=meta['metrics']
            tail=rows[role+'_reframed',seed]['all_pair_errors']['rmse_m'];sens=meta['sensitivity']
            a=np.mean([r['valid_and_correct_fraction'] for r in base['by_scene'].values()])
            b=np.mean([r['valid_and_correct_fraction'] for r in new['by_scene'].values()])
            lines.append(f'| {LABELS[role]}/{seed} | '+
                f"{100*base['macro']['region_recall']:.3f}→{100*new['macro']['region_recall']:.3f} | "+
                f'{100*a:.3f}→{100*b:.3f} | '+
                f"{100*base['macro']['center_pck']:.3f}→{100*new['macro']['center_pck']:.3f} | "+
                f"{base['macro']['soft_epe']:.4f}→{new['macro']['soft_epe']:.4f} | {f(tail['median'])}/{f(tail['p90'])} | "+
                f"{100*sens['all_query_top1_changed_fraction']:.3f}/{100*sens['valid_query_top1_changed_fraction']:.3f} |")
    lines+=['','全部旋转/平移/RMSE尾部与逐场景数值保留在result.json的38条记录中。无效解不删除；null误差分位数代表无限失败误差。',
        '此固定变换样本不保证任意SO(3)稳健，重复物理帧对不算新确认；没有公平速度比较。']
    (OUT/'tables.md').write_text('\n'.join(lines)+'\n')
    directory=OUT/'figures';directory.mkdir(exist_ok=False)
    fig,axes=plt.subplots(1,2,figsize=(15,5.5),layout='constrained')
    for i,role in enumerate(ROLES):
        for shift,suffix,color,label in ((-.18,'','#8da0cb','Original'),(.18,'_reframed','#1677a3','Reframed')):
            values=[100*rows[role+suffix,s]['macro_success_rate'] for s in SEEDS]
            axes[0].bar(i+shift,np.mean(values),width=.32,color=color,label=label if i==0 else None)
            axes[0].scatter(i+shift+np.linspace(-.06,.06,3),values,color=color,edgecolors='white',s=23,zorder=3)
    axes[0].set_xticks(range(6),[LABELS[r] for r in ROLES],rotation=30,ha='right');axes[0].set_ylim(0,100)
    axes[0].set_ylabel('Scene-balanced alignment success (%)');axes[0].legend();axes[0].set_title('Same 480 physical pairs; frozen models')
    for i,row in enumerate(comparisons):
        lo,hi=row['intervals_95_pp']['crossed'];axes[1].plot([lo,hi],[i,i],color='#1677a3',lw=2)
        axes[1].plot(row['mean_gain_pp'],i,'o',color='#1677a3');axes[1].scatter(row['paired_seed_gains_pp'],i+np.linspace(-.08,.08,3),marker='x',color='#1677a3')
    axes[1].set_yticks(range(5),['F − '+LABELS[r['right'].removesuffix('_reframed')] for r in comparisons]);axes[1].invert_yaxis()
    axes[1].axvline(0,color='.4',ls='--');axes[1].set_xlabel('Difference (pp), 95% seed × scene interval');axes[1].set_title('Comparisons after re-expression')
    for a in axes:a.grid(axis='y' if a is axes[0] else 'x',alpha=.2);a.set_axisbelow(True)
    fig.savefig(directory/'pose_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(15,5),layout='constrained')
    for axis,values,labels,title in ((axes[0],changes,[LABELS[r['role']] for r in changes],'Each method: reframed − original'),
                                   (axes[1],interactions,['F − '+LABELS[r['right']] for r in interactions],'Change in F-minus-control advantage')):
        for i,row in enumerate(values):
            lo,hi=row['intervals_95_pp']['crossed'];axis.plot([lo,hi],[i,i],lw=2,color='#8154a5');axis.plot(row['mean_gain_pp'],i,'o',color='#8154a5')
        axis.set_yticks(range(len(values)),labels);axis.invert_yaxis();axis.axvline(0,color='.4',ls='--');axis.set_title(title)
        axis.set_xlabel('Change (pp), 95% seed × scene interval');axis.grid(axis='x',alpha=.2)
    fig.savefig(directory/'changes.png',dpi=160);plt.close(fig)
    write(directory/'manifest.json',dict(created_unix=time.time(),result_sha256=sha(OUT/'result.json'),renderer_sha256=sha(__file__),
        figures={p.name:sha(p) for p in directory.glob('*.png')},visual_review_pending=True))
    print('FRAME_ROBUSTNESS_TABLES_AND_FIGURES_READY',flush=True)


if __name__=='__main__':main()
