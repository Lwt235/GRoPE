"""Tables and figure for the frozen five-method unused-place confirmation."""
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.round65_common import SEEDS, read, write, sha
from experiments.region_hrope_confirmation import OUT, ROLES
from experiments.region_hrope_report import LABELS, COLORS


def main():
    result=read(OUT/'result.json');checks=read(OUT/'independent_checks.json')
    assert checks['passed'] and checks['result_sha256']==sha(OUT/'result.json')
    manifest=read(OUT/'data_manifest.json')
    assert not (OUT/'tables.md').exists()
    figdir=OUT/'figures';figdir.mkdir(exist_ok=False)
    lines=['# 同相机输入的未用地点确认：完整数据表','',
        f"计划{manifest['planned_pairs']}对，实际{manifest['cache']['pairs']}对；{len(result['scenes'])}个可转换场景；排除{len(manifest['excluded'])}对。15个既有模型，零训练。",'',
        '| 方法 | 对齐成功率均值 % | 三种子成功率 % | 区域R@1 % |',
        '| --- | ---: | --- | ---: |']
    for role in ROLES:
        values=[100*next(r for r in result['records'] if (r['role'],r['seed'])==(role,s))['macro_success_rate'] for s in SEEDS]
        recall=[100*read(OUT/'predictions'/role/str(s)/'base.json')['metrics']['macro']['region_recall'] for s in SEEDS]
        lines.append(f'| {LABELS[role]} | {np.mean(values):.3f} | '+ '/'.join(f'{x:.3f}' for x in values)+f' | {np.mean(recall):.3f} |')
    lines+=['','| 比较 | 平均差 pp | 三种子差 pp | 联合95%区间 pp | 固定方向判断 |',
            '| --- | ---: | --- | --- | --- |']
    for row in result['comparisons']:
        lo,hi=row['intervals_95_pp']['crossed'];values='/'.join(f'{x:+.3f}' for x in row['paired_seed_gains_pp'])
        lines.append(f"| {LABELS[row['left']]} − {LABELS[row['right']]} | {row['mean_gain_pp']:+.3f} | {values} | [{lo:.3f},{hi:.3f}] | {row['direction']} |")
    lines+=['','| 方法/种子 | 成功帧对 | 全查询正确 % | 中心PCK % | Soft EPE m | RMSE 中位/p90 m | 旋转 p90 ° | 中心化平移 p90 m |',
            '| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |']
    for role in ROLES:
        for seed in SEEDS:
            row=next(r for r in result['records'] if (r['role'],r['seed'])==(role,seed))
            metrics=read(OUT/'predictions'/role/str(seed)/'base.json')['metrics']
            correct=np.mean([x['valid_and_correct_fraction'] for x in metrics['by_scene'].values()])
            m,tail=metrics['macro'],row['all_pair_errors']
            def f(x):return 'inf' if x is None else f'{x:.4f}'
            lines.append(f"| {LABELS[role]}/{seed} | {row['successful_pairs']} | {100*correct:.3f} | {100*m['center_pck']:.3f} | {m['soft_epe']:.4f} | {f(tail['rmse_m']['median'])}/{f(tail['rmse_m']['p90'])} | {f(tail['rotation_error_deg']['p90'])} | {f(tail['centered_translation_error_m']['p90'])} |")
    lines+=['','恒等与真值辅助最近邻为诊断；后者不可部署，且不是理论上界。','']
    for role in ('identity','gt_nearest'):
        r=next(r for r in result['records'] if r['role']==role)
        lines.append(f"- {role}: {100*r['macro_success_rate']:.3f}%（{r['successful_pairs']}/{r['pairs']}对）。")
    lines+=['','全部逐场景比较见result.json。左右方法都未满足对称稳定规则时记uncertain，不把它解释成两方法等效。']
    (OUT/'tables.md').write_text('\n'.join(lines)+'\n')
    figure,axes=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
    for i,role in enumerate(ROLES):
        values=[100*r['macro_success_rate'] for r in result['records'] if r['role']==role]
        axes[0].bar(i,np.mean(values),color=COLORS[role],width=.6,alpha=.75)
        axes[0].scatter(i+np.linspace(-.1,.1,3),values,color=COLORS[role],edgecolor='white',s=25,zorder=3)
        axes[0].text(i,max(values)+1.5,f'{np.mean(values):.2f}',ha='center',fontsize=10)
    axes[0].set_xticks(range(5),[LABELS[r] for r in ROLES],rotation=25,ha='right')
    axes[0].set_ylim(0,100);axes[0].set_ylabel('Scene-balanced alignment success (%)')
    axes[0].set_title(f"RMSE < 0.2 m; {manifest['cache']['pairs']} pairs")
    for i,row in enumerate(result['comparisons']):
        lo,hi=row['intervals_95_pp']['crossed'];color='#1769aa' if row['left']=='relative5' else '#c94c64'
        axes[1].plot([lo,hi],[i,i],lw=2,color=color)
        axes[1].plot(row['mean_gain_pp'],i,'o',color=color)
        axes[1].scatter(row['paired_seed_gains_pp'],i+np.linspace(-.09,.09,3),marker='x',color=color,s=25)
    axes[1].set_yticks(range(6),[LABELS[r['left']]+' − '+LABELS[r['right']] for r in result['comparisons']])
    axes[1].set_ylim(5.6,-.6);axes[1].axvline(0,color='.5',ls='--',lw=1)
    axes[1].set_xlabel('Difference (percentage points)');axes[1].set_title('Paired seeds; 95% seed × scene intervals')
    axes[0].grid(axis='y',alpha=.15);axes[1].grid(axis='x',alpha=.15)
    for ax in axes:ax.set_axisbelow(True)
    figure.suptitle(f"Camera-matched regional PE: confirmation on {len(result['scenes'])} previously unused places",fontsize=12)
    path=figdir/'pose_confirmation.png';figure.savefig(path,dpi=170);plt.close(figure)
    write(figdir/'manifest.json',dict(created_unix=time.time(),result_sha256=sha(OUT/'result.json'),
        renderer_sha256=sha(__file__),figures={path.name:sha(path)},visual_review_pending=True,fresh_confirmation=True))
    print('CONFIRMATION_TABLES_AND_FIGURE_READY',str(figdir),flush=True)


if __name__=='__main__':main()
