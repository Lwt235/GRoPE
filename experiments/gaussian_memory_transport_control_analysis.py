"""Post-score description of every control, without changing the main comparisons."""
import json

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_coverage_report import estimate
from experiments.gaussian_memory_transport_data import OUT, REGIMES, KERNELS, READERS, checked
from experiments.gaussian_memory_transport_report import unique_cases, matrix
from experiments.gaussian_memory_setup import sha256


def main():
    plan=checked();cases=unique_cases(plan);data=json.loads((OUT/'evaluation/scores.json').read_text())
    dec=json.loads((OUT/'decomposition.json').read_text())['rows'];result={}
    pairs=[('none_center_large_local',f'{g}_{k}_{r}') for g in REGIMES for k in KERNELS for r in READERS]
    pairs += [(f'{g}_center_{r}',f'{g}_gaussian_{r}') for g in REGIMES for r in READERS]
    for split in sorted({r['split'] for r in data['rows']})+['primary_both_clips']:
        select=lambda r:r['split'] in ('used_clip40','used_clip80') if split=='primary_both_clips' else r['split']==split
        rows=[r for r in data['rows'] if select(r)];full=[r for r in data['full_query'] if select(r)]
        coh=[r for r in data['consistency'] if select(r)];d=[r for r in dec if select(r)]
        scenes=sorted({r['scene'] for r in rows});result[split]={}
        for metric in ('log_mse','hidden_log_mse','visible_log_mse','rmse_m','absrel','full_log_mse','consistency_mse','bias_squared','centered_mse','full_centered_mse'):
            source,key=((full,'log_mse') if metric=='full_log_mse' else (coh,'residual_xyz_mse') if metric=='consistency_mse' else
                        ([r for r in d if r['input']=='full'],'centered_mse') if metric=='full_centered_mse' else
                        ([r for r in d if r['input']=='masked'],metric) if metric in ('bias_squared','centered_mse') else (rows,metric))
            values={cid:matrix(source,c,key,scenes) for cid,c in cases.items() if c['reader']!='baseline'}
            result[split][metric]={}
            for left,right in pairs:
                effect=estimate(values[left]-values[right],scenes);effect['relative_reduction_percent']=100*effect['mean_gain']/values[left].mean()
                result[split][metric][left+'_minus_'+right]=effect
    write_json(OUT/'control_analysis.json',{'complete':True,'new_training_or_predictions':False,
               'scope':'Post-score exhaustive control comparisons with the preserved strong Local and the position-bias counterpart. Primary Transport contrasts are unchanged. All data are used development data, no fresh confirmation or causal decomposition of the changes from surface1.',
               'source_sha256':sha256(__file__),'scores_sha256':sha256(OUT/'evaluation/scores.json'),'comparisons':result})
    summary=json.loads((OUT/'analysis.json').read_text())['primary_both_clips']['summary']
    fig,axes=plt.subplots(1,2,figsize=(15,6),constrained_layout=True)
    plotted=[('none_center_large_local',f'{g}_{k}_latent',g+' Latent '+k) for g in REGIMES for k in KERNELS]
    plotted += [(f'{g}_center_latent',f'{g}_gaussian_latent',g+' Gaussian / Center') for g in REGIMES]
    for i,(left,right,label) in enumerate(plotted):
        row=result['primary_both_clips']['log_mse'][left+'_minus_'+right]
        den=summary[left]['log_mse'];lo,hi=np.array(row['paired_scene_seed_bootstrap_95'])/den*100;value=row['relative_reduction_percent']
        axes[0].barh(i,value,color='#286ba8' if i<4 else '#008a75')
        axes[0].errorbar(value,i,xerr=[[value-lo],[hi-value]],color='black',capsize=3)
    axes[0].set_yticks(range(len(plotted)),[r[2] for r in plotted],fontsize=9);axes[0].axvline(0,color='black',lw=.7)
    axes[0].set(title='Descriptive follow-up of predeclared Latent controls',xlabel='Error reduction (%); first four vs Large Local')
    groups=['none_center_large_local','predicted_center_latent','predicted_gaussian_latent','sensor_center_transport']
    bias=np.array([summary[g]['bias_squared'] for g in groups]);spatial=np.array([summary[g]['centered_mse'] for g in groups]);x=np.arange(len(groups))
    axes[1].bar(x,bias,label='Per-query mean bias squared',color='#9eafbf')
    axes[1].bar(x,spatial,bottom=bias,label='Centered spatial error',color='#b66f49')
    axes[1].set_xticks(x,['Large Local','VGGT Latent C','VGGT Latent G','Sensor Transport C'],rotation=20,ha='right',fontsize=9)
    axes[1].set(title='Lower total error does not ensure better spatial detail',ylabel='Log-depth MSE');axes[1].legend(fontsize=8)
    fig.suptitle('Same frozen predictions, used development places; no new training or confirmation')
    fig.savefig(OUT/'control_summary.png',dpi=130);plt.close(fig)
    keys=('relative_reduction_percent','paired_scene_seed_bootstrap_95','paired_seed_gain','scenes_better')
    print(json.dumps({p:{k:v[k] for k in keys} for p,v in result['primary_both_clips']['log_mse'].items() if p.endswith('latent')}))


if __name__=='__main__':
    torch.set_num_threads(4);main()
