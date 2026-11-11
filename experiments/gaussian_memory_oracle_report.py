"""Paired scene-level effects of reference geometry; preserve crossed controls."""
import json

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_oracle_data import OUT, SEEDS, VARIANTS, model_path, load_record, checked
from experiments.gaussian_memory_coverage_report import estimate
from experiments.gaussian_memory_surface_diagnostic import decompose

CASES=['local']+[r+'_'+v for r in ('predicted','sensor') for v in VARIANTS]
PAIRS=[('predicted_fixed','sensor_fixed'),('predicted_gaussian','sensor_gaussian'),('predicted_center','sensor_center'),
       ('local','sensor_fixed'),('local','sensor_gaussian'),('local','sensor_center'),('sensor_fixed','sensor_gaussian'),('sensor_center','sensor_gaussian')]


def select_case(r,case):
    if case=='local':return r['variant']=='local'
    regime,variant=case.split('_',1)
    return r['train_geometry']==r['read_geometry']==regime and r['variant']==variant


def matrix(rows, case, metric, scenes, condition='normal'):
    return np.array([[np.mean([r[metric] for r in rows if select_case(r,case) and r['scene']==scene and r['seed']==seed and r['condition']==condition])
                      for seed in SEEDS] for scene in scenes])


def analysis():
    data=json.loads((OUT/'evaluation/scores.json').read_text());result={};decomposition=[];plan=checked()
    for meta in plan['catalog']:
        d=load_record(meta);maps=torch.load(OUT/'evaluation'/(meta['id']+'_maps.pt'),map_location='cpu',weights_only=False)
        base={'episode':meta['id'],'scene':meta['row']['scene'],'split':meta['eval_split'],'condition':'normal'}
        for case in CASES:
            regime,variant=('none','local') if case=='local' else case.split('_',1)
            for seed in SEEDS:
                name=f'{regime}_{regime}_{variant}_{seed}';pred=maps['predictions'][name+'_normal']
                tags=dict(base,train_geometry=regime,read_geometry=regime,variant=variant,seed=seed)
                decomposition.append(dict(tags,input='masked',**decompose(pred,maps['target'],maps['valid'])))
                q=maps['full_query'];pred=maps['full_predictions'][name]
                decomposition.append(dict(tags,input='full',**decompose(pred,d['target']['log_depth_m'][q],d['target']['valid'][q])))
    for split in sorted({r['split'] for r in data['rows']})+['primary_both_clips']:
        select=lambda r:r['split'] in ('used_clip40','used_clip80') if split=='primary_both_clips' else r['split']==split
        rows=[r for r in data['rows'] if select(r)];full=[r for r in data['full_query'] if select(r)];coh=[r for r in data['consistency'] if select(r)]
        dec=[r for r in decomposition if select(r)];scenes=sorted({r['scene'] for r in rows});summary={c:{} for c in CASES};comparisons={}
        metrics=['log_mse','hidden_log_mse','visible_log_mse','rmse_m','absrel','full_log_mse','consistency_mse','bias_squared','centered_mse','full_centered_mse']
        for metric in metrics:
            source,key=((full,'log_mse') if metric=='full_log_mse' else (coh,'residual_xyz_mse') if metric=='consistency_mse' else
                        ([r for r in dec if r['input']=='full'],'centered_mse') if metric=='full_centered_mse' else
                        ([r for r in dec if r['input']=='masked'],metric) if metric in ('bias_squared','centered_mse') else (rows,metric))
            values={c:matrix(source,c,key,scenes) for c in CASES};comparisons[metric]={}
            for c in CASES:summary[c][metric]=float(values[c].mean())
            for left,right in PAIRS:
                s=estimate(values[left]-values[right],scenes);s['relative_reduction_percent']=100*s['mean_gain']/values[left].mean();comparisons[metric][left+'_minus_'+right]=s
        crossed={}
        for variant in VARIANTS:
            crossed[variant]={}
            for train in ('predicted','sensor'):
                for read in ('predicted','sensor'):
                    selected=[r for r in rows if r['variant']==variant and r['train_geometry']==train and r['read_geometry']==read and r['condition']=='normal']
                    crossed[variant][train+'_to_'+read]=float(np.mean([np.mean([r['log_mse'] for r in selected if r['scene']==scene and r['seed']==seed]) for scene in scenes for seed in SEEDS]))
        interventions={}
        for variant in VARIANTS:
            case='sensor_'+variant;normal=matrix(rows,case,'log_mse',scenes);interventions[variant]={}
            for condition in ('removed','shuffled','wrong_camera','initial_geometry'):
                value=matrix(rows,case,'log_mse',scenes,condition);s=estimate(value-normal,scenes)
                s['relative_error_increase_percent']=100*s['mean_gain']/normal.mean();interventions[variant][condition]=s
        baselines={}
        for geometry in ('predicted','sensor'):
            for variant in ('seed_center','seed_gaussian'):
                values=[np.mean([r['log_mse'] for r in rows if r['scene']==scene and r['read_geometry']==geometry and r['variant']==variant]) for scene in scenes]
                baselines[geometry+'_'+variant]=float(np.mean(values))
        result[split]={'summary':summary,'comparisons':comparisons,'crossed_log_mse':crossed,'interventions':interventions,'baselines':baselines,
                       'scenes':scenes,'episodes':len({r['episode'] for r in rows}),'confirmation_claim':False}
    write_json(OUT/'analysis.json',result);write_json(OUT/'decomposition.json',{'rows':decomposition,'accounting_max_diff':max(r['accounting_max_diff'] for r in decomposition)})
    return result


def figures(stats):
    figures=[];seed=SEEDS[0];cases=['none_none_local','predicted_predicted_gaussian','sensor_sensor_fixed','sensor_sensor_gaussian'];labels=['Local','Predicted Gaussian','Sensor Fixed','Sensor Gaussian']
    for rid in ('fit_00','coverage_04','transfer_00','transfer_08','fidelity_00','fidelity_07','fidelity_11'):
        d=torch.load(OUT/'evaluation'/(rid+'_maps.pt'),map_location='cpu',weights_only=False);q=int(d['query'][0]);mask=d['mask'][0].reshape(15,20).numpy();valid=d['valid'][0].reshape(15,20).numpy()
        target=d['target'][0].exp().reshape(15,20).numpy();rgb=d['rgb'].numpy();observed=rgb[q].copy();observed[mask.repeat(14,0).repeat(14,1)]=[.485*255,.456*255,.406*255]
        fig,axes=plt.subplots(3,4,figsize=(14,10),constrained_layout=True)
        for ax,picture,title in zip(axes[0,:3],[rgb[0],rgb[12],observed],['Reference 0','Reference 12','Observed query']):ax.imshow(picture);ax.set_title(title)
        im=axes[0,3].imshow(np.where(valid,target,np.nan),vmin=.1,vmax=6,cmap='viridis');axes[0,3].set_title('Sensor target (m)');fig.colorbar(im,ax=list(axes[:2].flat),shrink=.6,label='Depth; display saturates above 6 m')
        errors=[]
        for ax,case,label in zip(axes[1],cases,labels):
            pred=d['predictions'][f'{case}_{seed}_normal'][0].reshape(15,20).numpy();error=(pred-np.log(target))**2;errors.append(np.where(valid,error,np.nan))
            ax.imshow(np.exp(pred),vmin=.1,vmax=6,cmap='viridis');ax.contour(mask.astype(float),[.5],colors='white',linewidths=.5);ax.set_title(f'{label}: {error[valid].mean():.4f}',fontsize=10)
        vmax=max(float(np.nanquantile(np.stack(errors),.95)),.001)
        for ax,error,label in zip(axes[2],errors,labels):im=ax.imshow(error,vmin=0,vmax=vmax,cmap='magma');ax.set_title(label+' log-depth error',fontsize=9)
        fig.colorbar(im,ax=list(axes[2]),shrink=.7,label='Squared log-depth error')
        for ax in axes.flat:ax.set_xticks([]);ax.set_yticks([])
        fig.suptitle(d['episode']['scene']+' | '+d['split']+'\nFirst query/mask, seed20261110; sensor is a diagnostic input')
        path=OUT/'evaluation'/(rid+'.png');fig.savefig(path,dpi=110);plt.close(fig);figures.append(str(path))
    main=stats['primary_both_clips'];fig,ax=plt.subplots(1,3,figsize=(16,5.2),constrained_layout=True)
    for index,v in enumerate(VARIANTS):
        for regime,ls in [('predicted','--'),('sensor','-')]:
            curves=[]
            for seed in SEEDS:
                rows=[json.loads(x) for x in (model_path(v,seed,regime).parent/'training.jsonl').read_text().splitlines()];curves.append(np.array([r['loss'] for r in rows]).reshape(20,100).mean(-1))
            ax[0].plot(np.arange(100,2001,100),np.mean(curves,0),color=['#aa7d27','#286ba8','#008a75'][index],ls=ls,label=regime+' '+v)
    ax[0].set(title='Matched fixed-budget training',xlabel='Step',ylabel='Log-depth MSE');ax[0].legend(fontsize=7)
    pairs=PAIRS[:2]+[('local','sensor_fixed'),('local','sensor_gaussian')]
    for i,(left,right) in enumerate(pairs):
        r=main['comparisons']['log_mse'][left+'_minus_'+right];den=main['summary'][left]['log_mse'];lo,hi=np.array(r['paired_scene_seed_bootstrap_95'])/den*100;value=r['relative_reduction_percent']
        ax[1].barh(i,value,color='#008a75');ax[1].errorbar(value,i,xerr=[[value-lo],[hi-value]],color='black',capsize=3)
    ax[1].axvline(0,color='black',lw=.7);ax[1].set_yticks(range(4),['Sensor / predicted: Fixed','Sensor / predicted: Gaussian','Sensor Fixed / Local','Sensor Gaussian / Local'],fontsize=8);ax[1].set(xlabel='Error reduction (%)',title='12 non-training places, both clips')
    changes=main['comparisons']['log_mse']['predicted_gaussian_minus_sensor_gaussian']['scene_gain'];scenes=main['scenes']
    ax[2].barh(np.arange(12),[changes[s] for s in scenes],color=['#008a75' if changes[s]>0 else '#b65e65' for s in scenes]);ax[2].set_yticks(np.arange(12),[s.split('/')[0] for s in scenes],fontsize=8)
    ax[2].axvline(0,color='black',lw=.7);ax[2].set(xlabel='Predicted minus sensor Gaussian error',title='Effect of reference geometry at each place')
    path=OUT/'comparison.png';fig.savefig(path,dpi=140);plt.close(fig);figures.append(str(path));write_json(OUT/'figure_manifest.json',{'figures':figures})


if __name__=='__main__':
    torch.set_num_threads(4);stats=analysis();figures(stats);m=stats['primary_both_clips']
    print(json.dumps({'summary':m['summary'],'main':{k:{z:v[z] for z in ('relative_reduction_percent','paired_scene_seed_bootstrap_95','paired_seed_gain','scenes_better')} for k,v in m['comparisons']['log_mse'].items()}}))
