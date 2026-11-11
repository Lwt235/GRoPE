"""Scene-balanced effects for the fixed factorial readout experiment."""
import json

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_coverage_report import estimate
from experiments.gaussian_memory_surface_diagnostic import decompose
from experiments.gaussian_memory_transport_data import OUT, REGIMES, KERNELS, READERS, SEEDS, checked, load_record, name, model_path
from experiments.gaussian_memory_transport_evaluate import conditions


def case_id(c):return '{regime}_{kernel}_{reader}'.format(**c)


def unique_cases(plan):
    cases={case_id(c):{k:c[k] for k in ('regime','kernel','reader')} for c in plan['cases']}
    cases['none_center_large_local']=dict(regime='none',kernel='center',reader='large_local')
    for regime in REGIMES:
        for kernel in KERNELS:
            c=dict(regime=regime,kernel=kernel,reader='baseline');cases[case_id(c)]=c
    return cases


def matrix(rows, c, metric, scenes, condition='normal'):
    selected=[r for r in rows if all(r[k]==v for k,v in c.items()) and r['condition']==condition]
    result=[]
    for scene in scenes:
        line=[]
        for seed in SEEDS:
            values=[r[metric] for r in selected if r['scene']==scene and (r['seed']==seed or c['reader']=='baseline')]
            assert values,(c,metric,condition,scene,seed)
            line.append(np.mean(values))
        result.append(line)
    return np.array(result)


def analysis():
    plan=checked();cases=unique_cases(plan);data=json.loads((OUT/'evaluation/scores.json').read_text());decomposition=[]
    for meta in plan['catalog']:
        d=load_record(meta);maps=torch.load(OUT/'evaluation'/(meta['id']+'_maps.pt'),map_location='cpu',weights_only=False)
        tag={'episode':meta['id'],'scene':meta['row']['scene'],'split':meta['eval_split'],'condition':'normal'}
        for key,c in cases.items():
            for seed in ((None,) if c['reader']=='baseline' else SEEDS):
                prefix=key if seed is None else key+'_'+str(seed)
                pred=maps['predictions'][prefix+'_normal']
                decomposition.append(dict(tag,**c,seed=seed,input='masked',**decompose(pred,maps['target'],maps['valid'])))
                if seed is not None:
                    uq=maps['full_query'];full=maps['full_predictions'][prefix]
                    decomposition.append(dict(tag,**c,seed=seed,input='full',**decompose(full,d['target']['log_depth_m'][uq],d['target']['valid'][uq])))
    pairs=[]
    for g in REGIMES:
        for k in KERNELS:
            candidate=f'{g}_{k}_transport'
            for control in (f'{g}_{k}_latent',f'{g}_{k}_prior_residual','none_center_local','none_center_large_local',f'{g}_{k}_baseline'):
                pairs.append((control,candidate))
            pairs.append(('none_center_local',f'{g}_{k}_latent'))
        pairs.append((f'{g}_center_transport',f'{g}_gaussian_transport'))
    for reader in READERS:
        for k in KERNELS:pairs.append((f'predicted_{k}_{reader}',f'sensor_{k}_{reader}'))
    stats={}
    for split in sorted({r['split'] for r in data['rows']})+['primary_both_clips']:
        selected=lambda r:r['split'] in ('used_clip40','used_clip80') if split=='primary_both_clips' else r['split']==split
        rows=[r for r in data['rows'] if selected(r)];full=[r for r in data['full_query'] if selected(r)]
        coh=[r for r in data['consistency'] if selected(r)];dec=[r for r in decomposition if selected(r)]
        scenes=sorted({r['scene'] for r in rows});summary={k:{} for k in cases};comparisons={}
        for metric in ('log_mse','hidden_log_mse','visible_log_mse','rmse_m','absrel','full_log_mse','consistency_mse','bias_squared','centered_mse','full_centered_mse'):
            source,key=((full,'log_mse') if metric=='full_log_mse' else (coh,'residual_xyz_mse') if metric=='consistency_mse' else
                        ([r for r in dec if r['input']=='full'],'centered_mse') if metric=='full_centered_mse' else
                        ([r for r in dec if r['input']=='masked'],metric) if metric in ('bias_squared','centered_mse') else (rows,metric))
            values={cid:matrix(source,c,key,scenes) for cid,c in cases.items() if not(c['reader']=='baseline' and metric in ('full_log_mse','full_centered_mse','consistency_mse'))}
            for cid,value in values.items():summary[cid][metric]=float(value.mean())
            comparisons[metric]={}
            for left,right in pairs:
                if left not in values or right not in values:continue
                effect=estimate(values[left]-values[right],scenes)
                effect['relative_reduction_percent']=100*effect['mean_gain']/values[left].mean()
                comparisons[metric][left+'_minus_'+right]=effect
        interventions={}
        for cid,c in cases.items():
            if c['reader'] not in READERS:continue
            normal=matrix(rows,c,'log_mse',scenes);interventions[cid]={}
            for condition in conditions(c['reader'])[1:]:
                effect=estimate(matrix(rows,c,'log_mse',scenes,condition)-normal,scenes)
                effect['relative_error_increase_percent']=100*effect['mean_gain']/normal.mean()
                interventions[cid][condition]=effect
        stats[split]={'summary':summary,'comparisons':comparisons,'interventions':interventions,'scenes':scenes,
                      'episodes':len({r['episode'] for r in rows}),'confirmation_claim':False}
    write_json(OUT/'analysis.json',stats)
    write_json(OUT/'decomposition.json',{'rows':decomposition,'accounting_max_difference':max(r['accounting_max_diff'] for r in decomposition)})
    return stats


def figures(stats):
    figures=[];seed=SEEDS[0]
    displayed=[('none_center_large_local','Large Local'),('none_center_local','Matched Local'),
               ('predicted_center_latent','VGGT Latent C'),('predicted_center_transport','VGGT Transport C'),
               ('predicted_center_prior_residual','VGGT PriorResidual C'),('predicted_gaussian_transport','VGGT Transport G'),
               ('sensor_center_transport','Sensor Transport C'),('sensor_gaussian_transport','Sensor Transport G')]
    for rid in ('fit_00','coverage_04','transfer_00','transfer_08','fidelity_00','fidelity_07','fidelity_11'):
        d=torch.load(OUT/'evaluation'/(rid+'_maps.pt'),map_location='cpu',weights_only=False)
        q=int(d['query'][0]);mask=d['mask'][0].reshape(15,20).numpy();valid=d['valid'][0].reshape(15,20).numpy()
        truth=d['target'][0].reshape(15,20).numpy();rgb=d['rgb'].numpy();observed=rgb[q].copy();observed[mask.repeat(14,0).repeat(14,1)]=[.485*255,.456*255,.406*255]
        fig,axes=plt.subplots(4,4,figsize=(15,13),constrained_layout=True)
        for ax,pic,title in zip(axes[0,:3],[rgb[0],rgb[12],observed],['Reference 0','Reference 12','Observed query']):ax.imshow(pic);ax.set_title(title)
        im=axes[0,3].imshow(np.where(valid,np.exp(truth),np.nan),vmin=.1,vmax=6,cmap='viridis');axes[0,3].set_title('Sensor target (m)')
        fig.colorbar(im,ax=list(axes[:3].flat),shrink=.55,label='Depth; display saturates at 6 m')
        errors={}
        for ax,(key,label) in zip(axes[1:3].flat,displayed):
            pred=d['predictions'][f'{key}_{seed}_normal'][0].reshape(15,20).numpy();error=(pred-truth)**2;errors[key]=np.where(valid,error,np.nan)
            ax.imshow(np.exp(pred),vmin=.1,vmax=6,cmap='viridis');ax.contour(mask.astype(float),[.5],colors='white',linewidths=.5)
            ax.set_title(f'{label}: {error[valid].mean():.4f}',fontsize=9)
        vmax=max(float(np.nanquantile(np.stack(list(errors.values())),.95)),.001)
        for ax,(key,label) in zip(axes[3],[displayed[i] for i in (0,3,6,7)]):
            im=ax.imshow(errors[key],vmin=0,vmax=vmax,cmap='magma');ax.set_title(label+' error',fontsize=9)
        fig.colorbar(im,ax=list(axes[3]),shrink=.7,label='Squared log-depth error')
        for ax in axes.flat:ax.set_xticks([]);ax.set_yticks([])
        fig.suptitle(d['episode']['scene']+' | '+d['split']+'\nFixed first query/mask, seed 20261110; C=center, G=covariance; sensor is diagnostic')
        path=OUT/'evaluation'/(rid+'.png');fig.savefig(path,dpi=110);plt.close(fig);figures.append(str(path))
    main=stats['primary_both_clips'];fig,axes=plt.subplots(2,2,figsize=(15,10),constrained_layout=True)
    for reader,color in [('latent','#aa7d27'),('transport','#008a75'),('prior_residual','#286ba8'),('local','#777777')]:
        curves=[]
        for seed in SEEDS:
            c=dict(regime='none' if reader=='local' else 'predicted',kernel='center',reader=reader,seed=seed)
            rows=[json.loads(s) for s in (model_path(c).parent/'training.jsonl').read_text().splitlines()]
            curves.append(np.array([r['loss'] for r in rows]).reshape(20,100).mean(-1))
        axes[0,0].plot(np.arange(100,2001,100),np.mean(curves,0),label=reader,color=color)
    axes[0,0].set(title='VGGT center: fixed-budget training',xlabel='Step',ylabel='Log-depth MSE');axes[0,0].legend(fontsize=8)
    index=0;labels=[]
    for g,color in [('predicted','#286ba8'),('sensor','#008a75')]:
        for control,label in [(f'{g}_center_latent','Latent'),(f'{g}_center_prior_residual','PriorResidual'),('none_center_local','Matched Local'),('none_center_large_local','Large Local')]:
            r=main['comparisons']['log_mse'][control+'_minus_'+g+'_center_transport'];den=main['summary'][control]['log_mse']
            lo,hi=np.array(r['paired_scene_seed_bootstrap_95'])/den*100;value=r['relative_reduction_percent']
            axes[0,1].barh(index,value,color=color);axes[0,1].errorbar(value,index,xerr=[[value-lo],[hi-value]],color='black',capsize=3)
            labels.append(g+' / '+label);index+=1
    axes[0,1].set_yticks(range(index),labels,fontsize=8);axes[0,1].axvline(0,color='black',lw=.7)
    axes[0,1].set(title='Center Transport vs each control',xlabel='Error reduction (%)')
    scenes=main['scenes']
    for g,offset,color in [('predicted',-.18,'#286ba8'),('sensor',.18,'#008a75')]:
        r=main['comparisons']['log_mse'][g+'_center_prior_residual_minus_'+g+'_center_transport']
        axes[1,0].barh(np.arange(len(scenes))+offset,[r['scene_gain'][s] for s in scenes],height=.35,label=g,color=color)
    axes[1,0].set_yticks(range(len(scenes)),[s.split('/')[0] for s in scenes],fontsize=8);axes[1,0].axvline(0,color='black',lw=.7);axes[1,0].legend(fontsize=8)
    axes[1,0].set(title='Does learned routing beat a fixed prior plus residual?',xlabel='PriorResidual minus Transport error')
    for index,g in enumerate(REGIMES):
        r=main['comparisons']['log_mse'][g+'_center_transport_minus_'+g+'_gaussian_transport'];den=main['summary'][g+'_center_transport']['log_mse']
        lo,hi=np.array(r['paired_scene_seed_bootstrap_95'])/den*100;value=r['relative_reduction_percent']
        axes[1,1].barh(index,value,color='#008a75');axes[1,1].errorbar(value,index,xerr=[[value-lo],[hi-value]],color='black',capsize=3)
    axes[1,1].set_yticks(range(2),REGIMES);axes[1,1].axvline(0,color='black',lw=.7);axes[1,1].set(title='Covariance vs center: Transport',xlabel='Error reduction (%)')
    path=OUT/'comparison.png';fig.savefig(path,dpi=130);plt.close(fig);figures.append(str(path))
    write_json(OUT/'figure_manifest.json',{'figures':figures})


if __name__=='__main__':
    torch.set_num_threads(4);stats=analysis();figures(stats)
    print(json.dumps({'primary_summary':stats['primary_both_clips']['summary']}))
