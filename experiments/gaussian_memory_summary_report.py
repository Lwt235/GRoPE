"""Prospective paired comparisons and preselected visualizations; no selection."""
import json

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_coverage_report import estimate
from experiments.gaussian_memory_summary_data import OUT,OLD,SEEDS,KINDS,path,locked

METRICS=('log_mse','hidden_log_mse','visible_log_mse','rmse_m','absrel','full_log_mse',
         'bias_squared','centered_mse','full_bias_squared','full_centered_mse')
PAIRS=(('global','gaussian'),('center','gaussian'),('large_local','gaussian'),('local','gaussian'),
       ('large_local','global'),('local','global'),('center','global'))


def matrix(rows,kind,metric,scenes,condition='normal'):
    return np.array([[np.mean([r[metric] for r in rows if r['kind']==kind and r['seed']==s and r['scene']==scene and r['condition']==condition])
                      for s in SEEDS] for scene in scenes])


def analysis():
    locked();data=json.loads((OUT/'evaluation/scores.json').read_text());stats={}
    splits=sorted({r['split'] for r in data['rows']})+['old_primary_both_clips']
    for split in splits:
        select=lambda r:r['split'] in ('used_clip40','used_clip80') if split=='old_primary_both_clips' else r['split']==split
        rows=[r for r in data['rows'] if select(r)];full=[r for r in data['full_query'] if select(r)]
        scenes=sorted({r['scene'] for r in rows});summary={k:{} for k in KINDS};comparisons={}
        for metric in METRICS:
            source,key=(full,metric.removeprefix('full_')) if metric.startswith('full_') else (rows,metric)
            values={k:matrix(source,k,key,scenes) for k in KINDS}
            assert all(np.isfinite(v).all() for v in values.values())
            for k,v in values.items():summary[k][metric]=float(v.mean())
            comparisons[metric]={}
            for left,right in PAIRS:
                effect=estimate(values[left]-values[right],scenes)
                effect['relative_reduction_percent']=100*effect['mean_gain']/values[left].mean()
                comparisons[metric][left+'_minus_'+right]=effect
        interventions={}
        for kind in ('gaussian','center','global'):
            normal=matrix(rows,kind,'log_mse',scenes);interventions[kind]={}
            for condition in ('removed','shuffled','wrong_camera'):
                effect=estimate(matrix(rows,kind,'log_mse',scenes,condition)-normal,scenes)
                effect['relative_error_increase_percent']=100*effect['mean_gain']/normal.mean()
                interventions[kind][condition]=effect
        stats[split]={'summary':summary,'comparisons':comparisons,'interventions':interventions,
                      'scenes':scenes,'episodes':len({r['episode'] for r in rows}),
                      'method_unseen_confirmation':split=='confirmation','historically_untouched_claim':False}
    write_json(OUT/'analysis.json',stats);return stats


def figures(stats):
    figures=[];seed=SEEDS[0]
    labels={'gaussian':'Gaussian','center':'Center','global':'Global summary','local':'Matched Local','large_local':'Strong Local'}
    for rid in ('fit_00','transfer_00','fidelity_07','confirm_00','confirm_05','confirm_10','confirm_15'):
        d=torch.load(OUT/'evaluation'/(rid+'_maps.pt'),map_location='cpu',weights_only=False)
        q=int(d['query'][0]);mask=d['mask'][0].reshape(15,20).numpy();valid=d['valid'][0].reshape(15,20).numpy()
        truth=d['target'][0].reshape(15,20).numpy();rgb=d['rgb'].numpy();observed=rgb[q].copy()
        observed[mask.repeat(14,0).repeat(14,1)]=[.485*255,.456*255,.406*255]
        fig,axes=plt.subplots(3,5,figsize=(17,10),constrained_layout=True)
        for ax,pic,title in zip(axes[0,:3],[rgb[0],rgb[12],observed],['Reference 0','Reference 12','Observed query']):
            ax.imshow(pic);ax.set_title(title)
        im=axes[0,3].imshow(np.where(valid,np.exp(truth),np.nan),vmin=.1,vmax=6,cmap='viridis');axes[0,3].set_title('Sensor target (m)')
        axes[0,4].axis('off');axes[0,4].text(.05,.7,'First query/mask\nSeed 20261110\nFixed case selection\nScoring is unclipped',fontsize=11)
        fig.colorbar(im,ax=list(axes[:2].flat),shrink=.6,label='Depth; display saturates at 6 m')
        errors={}
        for ax,kind in zip(axes[1],KINDS):
            pred=d['predictions'][f'{kind}_{seed}_normal'][0].reshape(15,20).numpy();e=(pred-truth)**2
            errors[kind]=np.where(valid,e,np.nan)
            ax.imshow(np.exp(pred),vmin=.1,vmax=6,cmap='viridis');ax.contour(mask.astype(float),[.5],colors='white',linewidths=.5)
            ax.set_title(f'{labels[kind]}: {e[valid].mean():.4f}',fontsize=10)
        vmax=max(float(np.nanquantile(np.stack(list(errors.values())),.95)),.001)
        for ax,kind in zip(axes[2],KINDS):
            im=ax.imshow(errors[kind],vmin=0,vmax=vmax,cmap='magma');ax.set_title(labels[kind]+' error',fontsize=10)
        fig.colorbar(im,ax=list(axes[2]),shrink=.7,label='Squared log-depth error')
        for ax in axes.flat:ax.set_xticks([]);ax.set_yticks([])
        fig.suptitle(d['episode']['scene']+' | '+d['split'])
        dest=OUT/'evaluation'/(rid+'.png');fig.savefig(dest,dpi=110);plt.close(fig);figures.append(str(dest))
    main=stats['confirmation'];old=stats['old_primary_both_clips']
    fig,axes=plt.subplots(2,2,figsize=(15,11),constrained_layout=True)
    for kind,color in [('gaussian','#008a75'),('global','#b46c35')]:
        curves=[]
        for seed in SEEDS:
            log=path(kind,seed).parent/'training.jsonl'
            rows=[json.loads(x) for x in log.read_text().splitlines()]
            curves.append(np.array([r['loss'] for r in rows]).reshape(20,100).mean(-1))
        axes[0,0].plot(np.arange(100,2001,100),np.mean(curves,0),label=labels[kind],color=color)
    axes[0,0].set(title='Matched training budget',xlabel='Step',ylabel='Log-depth MSE');axes[0,0].legend()
    ps=[('global','gaussian'),('center','gaussian'),('large_local','gaussian'),('large_local','global')]
    for cohort,offset,color in [(old,-.17,'#9eafbf'),(main,.17,'#008a75')]:
        for i,(left,right) in enumerate(ps):
            r=cohort['comparisons']['log_mse'][left+'_minus_'+right];den=cohort['summary'][left]['log_mse']
            lo,hi=np.array(r['paired_scene_seed_bootstrap_95'])/den*100;value=r['relative_reduction_percent']
            axes[0,1].barh(i+offset,value,height=.31,color=color)
            axes[0,1].errorbar(value,i+offset,xerr=[[max(0,value-lo)],[max(0,hi-value)]],color='black',capsize=3)
    axes[0,1].set_yticks(range(len(ps)),[labels[r]+' vs '+labels[l] for l,r in ps],fontsize=9)
    axes[0,1].axvline(0,color='black',lw=.7);axes[0,1].set(title='Gray: old 12 places; green: new-to-route 20 places',xlabel='Error reduction (%)')
    scenes=main['scenes'];g=main['comparisons']['log_mse']['global_minus_gaussian']
    axes[1,0].barh(np.arange(len(scenes)),[g['scene_gain'][s] for s in scenes],color='#008a75')
    axes[1,0].set_yticks(np.arange(len(scenes)),[s.split('/')[0] for s in scenes],fontsize=8)
    axes[1,0].axvline(0,color='black',lw=.7);axes[1,0].set(title='All 20 places retained',xlabel='Global minus Gaussian log-MSE')
    order=('large_local','local','global','center','gaussian');bias=[main['summary'][k]['bias_squared'] for k in order]
    spatial=[main['summary'][k]['centered_mse'] for k in order]
    axes[1,1].bar(range(5),bias,label='Query mean bias squared',color='#9eafbf')
    axes[1,1].bar(range(5),spatial,bottom=bias,label='Centered spatial error',color='#b46c35')
    axes[1,1].set_xticks(range(5),[labels[k] for k in order],rotation=25,ha='right',fontsize=9)
    axes[1,1].set(title='Confirmation error decomposition',ylabel='Log-depth MSE');axes[1,1].legend(fontsize=8)
    fig.suptitle('Frozen candidate and one new global control; known cameras; no learned writer')
    dest=OUT/'comparison.png';fig.savefig(dest,dpi=130);plt.close(fig);figures.append(str(dest))
    write_json(OUT/'figure_manifest.json',{'figures':figures})


if __name__=='__main__':
    torch.set_num_threads(4);stats=analysis();figures(stats)
    print(json.dumps({k:{'summary':stats[k]['summary'],'main':stats[k]['comparisons']['log_mse']} for k in ('old_primary_both_clips','confirmation')}))
