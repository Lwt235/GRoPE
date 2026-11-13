"""Final diagnostic plots, tables and accounting; no GPU calls."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/liwt_native_covariance_role_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from experiments.native_covariance_role import OUT,ARMS,CACHE,d,write,diskbytes,PLAN
from experiments.gaussian_scene_data import query_image


def main():
    torch.set_num_threads(4);s=d.read(OUT/'scoring/summary.json');rows=d.read(OUT/'scoring/metrics.json')['rows'];folder=OUT/'figures';folder.mkdir(exist_ok=True)
    write(folder/'contract.json',dict(backend='Python/matplotlib, existing project stack',claim='Conditional geometry/composition and coverage sensitivity at fixed original Render weights; no Transformer PE tested.',exports=['outcomes.png','depth_cases.png'],selection='Four plan-fixed cases plus largest mean-seed development C_match minus F case',image_integrity='Seed20261110; shared GT-valid mask and metric-depth scale within each row; no smoothing',risks=['Original full shape remains in factual coverage and feature binding','C privileged with original trace/minimum axis','Fixed-weight perturbation not optimized alternative','Old data, not freshconfirmation']))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained');methods=['Local','Old_R',*ARMS];colors=['#999999','#9999bb','#3b8068','#d2b175','#6996b1','#b87878']
    for col,(cohort,title) in enumerate([('development12','12 old development places / original queries (primary)'),('train32','32 trained clips / other queries')]):
        key=f'{cohort}/missing/roi/centered_log_mse';ax=axes[0,col]
        for i,(kind,color) in enumerate(zip(methods,colors)):
            v=s['summaries'][key+'/'+kind];ax.bar(i,v['mean'],color=color);ax.scatter(np.full(3,i)+[-.12,0,.12],v['per_seed'],s=14,c='#333333',zorder=3)
        ax.set_xticks(range(6),methods,rotation=15);ax.set_ylabel('Missing ROI spatial log-MSE (lower better)');ax.set_title(title)
        ax=axes[1,col]
        for i,(a,b,label) in enumerate([('C_match','F','Center composition: C_match - F'),('S_match','F','Sphere composition: S_match - F'),('C_raw','C_match','Coverage change: C_raw - C_match'),('C_raw','F','Total: C_raw - F')]):
            v=s['comparisons'][key+'/'+a+'-'+b];lo,hi=v['interval95_percent'];ax.plot([lo,hi],[i,i],c='#597e86',lw=2);ax.scatter(v['relative_error_increase_percent'],i,c='#597e86',s=30);ax.scatter(v['per_seed_relative_percent'],np.full(3,i)+[-.09,0,.09],s=12,c='#999999')
        ax.axvline(0,c='black',lw=.8);ax.set_yticks(range(4),['Center composition','Sphere composition','Coverage change','Total']);ax.invert_yaxis();ax.set_xlabel('Relative error increase %, paired95% intervals');ax.set_title('Fixed contrasts, positive = degradation')
    fig.suptitle('Frozen Render | original content/centers/opacity | 0 training\nCoverage matched groups retain factual full-shape alpha; C retains original trace/minimum axis',fontsize=12)
    fig.savefig(folder/'outcomes.png',dpi=150);plt.close(fig)
    candidates=[]
    for m in d.plan()['catalog']:
        for q in d.QUERIES:
            means={kind:float(np.mean([r['metrics']['roi']['centered_log_mse'] for r in rows if r['id']==m['id'] and r['query']==q and r['condition']=='missing' and r['kind']==kind])) for kind in ['F','C_match']}
            candidates.append(dict(id=m['id'],query=q,difference=means['C_match']-means['F'],means=means))
    worst=max(candidates,key=lambda x:(x['difference'],x['id'],x['query']))
    selected=list(dict.fromkeys([('fit_00',1),('confirm_12',1),('fidelity_00',2),('fidelity_02',11),(worst['id'],worst['query'])]))
    fig,axes=plt.subplots(len(selected),6,figsize=(18,3*len(selected)),layout='constrained',squeeze=False);cm=[]
    for i,(rid,q) in enumerate(selected):
        query=d.load(d.query_path(rid,q));target=query['target'].numpy();valid=query['valid'].numpy();lo,hi=np.percentile(target[valid],[2,98]);axes[i,0].imshow(query_image(query,'missing').permute(1,2,0).numpy())
        values=[target]+[np.load(CACHE/'predictions'/f'{d.SEEDS[0]}_{rid}_{q}_missing_{arm}.npy') for arm in ARMS]
        for j,value in enumerate(values,1):im=axes[i,j].imshow(np.where(valid,value,np.nan),cmap='viridis',vmin=lo,vmax=hi)
        fig.colorbar(im,ax=list(axes[i,1:]),shrink=.6,label='m, shared row scale');axes[i,0].set_ylabel(f'{rid}/q{q}')
        for ax in axes[i]:ax.set_xticks([]);ax.set_yticks([])
        cm.append(dict(id=rid,query=q,display_range_m=[float(lo),float(hi)],all_seed_metrics={kind:[r['metrics']['roi']['centered_log_mse'] for r in rows if r['id']==rid and r['query']==q and r['condition']=='missing' and r['kind']==kind] for kind in ARMS}))
    for ax,label in zip(axes[0],['Missing input','Sensor GT',*ARMS]):ax.set_title(label)
    fig.suptitle('Four fixed cases + largest mean C_match-F development counterexample\nSeed20261110 displayed; selection uses all3seed mean; all cases remain scored',fontsize=12)
    fig.savefig(folder/'depth_cases.png',dpi=150);plt.close(fig)
    write(folder/'selection.json',dict(cases=selected,worst_development_C_match_minus_F=worst,case_metrics=cm,files={str(x):d.sha256(x) for x in [folder/'outcomes.png',folder/'depth_cases.png']},actual_visual_review_pending=True))
    table=['# native_covariance_role1：完整固定数表','','相对百分比正值为误差增加，负值为改善；所有旧数据、固定权重条件依赖，非等效或独立确认。','']
    for cohort in ['development12','train32','train_original','train_added']:
        table += [f'## {cohort}','','|指标/组|均值|三种子|','|---|---:|---|']
        for c,reg,metric in [('missing','roi','centered_log_mse'),('missing','roi','log_mse'),('missing','roi','bias_squared'),('missing','roi_supported','centered_log_mse'),('missing','roi_unsupported','centered_log_mse'),('full','roi','centered_log_mse'),('full','whole','absrel')]:
            key=f'{cohort}/{c}/{reg}/{metric}'
            for kind in methods:
                v=s['summaries'][key+'/'+kind];table.append(f'|{c}/{reg}/{metric}/{kind}|{v["mean"]:.9f}|'+', '.join(f'{x:.9f}' for x in v['per_seed'])+'|')
        table += ['','|主空间比较|绝对差[95%区间]|相对增加%[95%区间]|三种子%|','|---|---|---|---|']
        key=f'{cohort}/missing/roi/centered_log_mse'
        for a,b in [('C_match','F'),('S_match','F'),('C_raw','C_match'),('C_raw','F'),('F','Old_R')]+[(k,'Local') for k in ARMS]:
            v=s['comparisons'][key+'/'+a+'-'+b];table.append(f'|{a}-{b}|{v["mean_absolute"]:.9f} [{v["interval95_absolute"][0]:.9f},{v["interval95_absolute"][1]:.9f}]|{v["relative_error_increase_percent"]:+.4f} [{v["interval95_percent"][0]:+.4f},{v["interval95_percent"][1]:+.4f}]|'+', '.join(f'{x:+.4f}' for x in v['per_seed_relative_percent'])+'|')
        table += ['']
    table += ['全部region/metric/condition的逐地点/逐种子均值和比较见 scoring/summary.json、all_summary_cells.csv、all_comparisons.csv；逐预测和每query/seed/支持分层见 scoring/metrics.json。']
    (OUT/'report_tables.md').write_text('\n'.join(table)+'\n')
    print('Two fixed figures and full result tables written.',flush=True)


if __name__=='__main__':main()
