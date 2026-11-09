"""Descriptive tables and figures; all scientific comparisons are already frozen."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent
result = json.loads((OUT/'result.json').read_text())
ROLES = ('relative5','distance_mlp','geotransformer_rpe','jeffreys_mlp','fixed_isotropic5')
LABELS = ('Five relations','Distance','GeoTransformer PE','Jeffreys MLP','Fixed sphere')
COLORS = ('#1b6ca8','#d47625','#589b61','#956cb4','#6a6a6a')
summary = []
lines = ['# 全查询选择性匹配：自动汇总，待实际审阅', '',
    '主操作点：仅在旧验证集按错误输出 / 全部查询 ≤ 5% 选择阈值，再原样评价。数值为场景等权、三种子均值。', '',
    '| 数据 | 条件 | 方法 | 正确输出 % | 错误输出 % | 保留率 % | 场景平均 precision % | 有对应 recall % | 无对应误接受 % |',
    '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for cohort in ('old5','used9'):
    for condition in ('base','sparse256'):
        for role in ROLES:
            rows = [r for r in result['records'] if (r['cohort'],r['condition'],r['role'])==(cohort,condition,role)]
            values = {k:float(100*np.mean([r['operating_points'][1]['metrics']['macro'][k] for r in rows])) for k in
                ('correct_yield','wrong_yield','coverage','precision','valid_recall','unmatched_acceptance')}
            summary.append(dict(cohort=cohort,condition=condition,role=role,**values))
            lines.append(f'| {cohort} | {condition} | {role} | '+' | '.join(f'{x:.3f}' for x in values.values())+' |')
lines += ['', '| 数据 | 条件 | 五维减控制 | 正确输出差 pp | 联合 95% 区间 | 错误输出差 pp | 联合 95% 区间 |',
    '| --- | --- | --- | ---: | --- | ---: | --- |']
for r in result['comparisons']:
    a,b = r['mean_differences_pp']; ci1,ci2 = r['crossed_seed_scene_95_pp']
    lines.append(f"| {r['cohort']} | {r['condition']} | {r['control']} | {a:+.3f} | [{ci1[0]:+.3f},{ci1[1]:+.3f}] | {b:+.3f} | [{ci2[0]:+.3f},{ci2[1]:+.3f}] |")
lines += ['', '错误输出越低越好；少输出同时可能减少正确与错误，不能只看一列。',
          '全部 1%/5%/10% 操作点、固定保留比例曲线、逐场景计数见 result.json。所有数据均已使用，本次不是新确认。']
(OUT/'actual_review1/automatic_report.md').write_text('\n'.join(lines)+'\n')
(OUT/'actual_review1/summary.json').write_text(json.dumps(summary,indent=2)+'\n')

fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
for ax,(cohort,condition) in zip(axes.flat,((c,v) for c in ('old5','used9') for v in ('base','sparse256'))):
    for role,label,color in zip(ROLES,LABELS,COLORS):
        rows=[r for r in result['records'] if (r['cohort'],r['condition'],r['role'])==(cohort,condition,role)]
        x=[100*r['operating_points'][1]['metrics']['macro']['wrong_yield'] for r in rows]
        y=[100*r['operating_points'][1]['metrics']['macro']['correct_yield'] for r in rows]
        ax.scatter(x,y,s=25,color=color,alpha=.45)
        ax.scatter(np.mean(x),np.mean(y),s=85,color=color,marker='D',label=label,edgecolors='white',linewidths=.5)
    ax.axvline(5,color='gray',linestyle=':',linewidth=1)
    ax.set(title=f'{cohort} / {condition}',xlabel='Wrong outputs / all queries (%)',ylabel='Correct outputs / all queries (%)')
    ax.grid(alpha=.15)
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='outside lower center',ncol=5,fontsize=9)
fig.suptitle('Frozen validation thresholds: smaller points are seeds; diamonds are means')
fig.savefig(OUT/'operating_points.png',dpi=160)
plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained',sharex=True,sharey=True)
for ax,(cohort,condition) in zip(axes.flat,((c,v) for c in ('old5','used9') for v in ('base','sparse256'))):
    for role,label,color in zip(ROLES,LABELS,COLORS):
        rows=[r for r in result['records'] if (r['cohort'],r['condition'],r['role'])==(cohort,condition,role)]
        x=[100*c['requested_coverage'] for c in rows[0]['coverage_curve']]
        y=[100*np.mean([r['coverage_curve'][i]['metrics']['macro']['precision'] for r in rows]) for i in range(len(x))]
        ax.plot(x,y,color=color,marker='o',markersize=3,label=label)
    ax.set(title=f'{cohort} / {condition}',xlabel='Retained queries per scene (%)',ylabel='Precision of retained matches (%)',xscale='log')
    ax.set_xticks([1,2,5,10,20,40,100],labels=['1','2','5','10','20','40','100'])
    ax.grid(alpha=.15)
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='outside lower center',ncol=5,fontsize=9)
fig.suptitle('Fixed-coverage diagnostic; all preset points shown, no test-label threshold selection')
fig.savefig(OUT/'coverage_precision.png',dpi=160)
plt.close(fig)
print('SUMMARY_AND_TWO_FIGURES_WRITTEN')
