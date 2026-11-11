"""Paired scene statistics and fixed example maps for the richer-memory probe."""
import json

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_direct_data import OUT, SEEDS, VARIANTS, model_path

METRICS=('hidden_mse','visible_mse','all_patch_mse','clean_input_mse')
PAIRS=(('local','center'),('local','wide'),('local','direct'),('wide','direct'),('center','direct'))


def compare(rows,left,right,metric):
    scenes=sorted({r['scene'] for r in rows}); values=[]
    for scene in scenes:
        mean={v:float(np.mean([r[metric] for r in rows if r['scene']==scene and r['variant']==v])) for v in (left,right)}
        values.append({'scene':scene,'left':mean[left],'right':mean[right],'gain':mean[left]-mean[right]})
    delta=np.array([v['gain'] for v in values]); rng=np.random.default_rng(44102)
    boot=delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(-1)
    seed_gain=[float(np.mean([r[metric] for r in rows if r['variant']==left and r['seed']==seed])-
                     np.mean([r[metric] for r in rows if r['variant']==right and r['seed']==seed])) for seed in SEEDS]
    return {'mean_gain':float(delta.mean()),'relative_reduction_percent':float(100*delta.mean()/np.mean([v['left'] for v in values])),
            'scene_bootstrap_95':np.quantile(boot,[.025,.975]).tolist(),'paired_seed_gain':seed_gain,
            'scenes_better':int((delta>0).sum()),'scenes_total':len(scenes),'scenes':values}


def figures():
    paths=[]
    for path in sorted((OUT / 'evaluation').glob('*_maps.pt')):
        d=torch.load(path,map_location='cpu',weights_only=False)
        q=int(d['query'][0]); mask=d['mask'][0].reshape(15,20).numpy(); rgb=d['rgb'].numpy()
        observed=rgb[q].copy(); observed[mask.repeat(14,0).repeat(14,1)]=np.array([.485,.456,.406])*255
        fig,ax=plt.subplots(2,4,figsize=(14,7),constrained_layout=True)
        for a,pic,title in zip(ax[0],[rgb[0],rgb[12],rgb[q],observed],
                             ['Reference 0','Reference 12','Full query: target only','Observed query']):
            a.imshow(pic); a.set_title(title,fontsize=10)
        errors=[d['errors'][v][0].reshape(15,20).numpy() for v in VARIANTS]
        vmax=float(np.quantile(np.stack(errors),.98))
        for a,v,e in zip(ax[1],VARIANTS,errors):
            im=a.imshow(e,vmin=0,vmax=vmax,cmap='magma'); a.contour(mask.astype(float),[.5],colors='cyan',linewidths=.6)
            a.set_title(f'{v}: hidden {e[mask].mean():.4f}\nvisible {e[~mask].mean():.4f}; all {e.mean():.4f}',fontsize=10)
        fig.colorbar(im,ax=list(ax[1]),shrink=.65,label='Normalized token MSE')
        for a in ax.flat:a.set_xticks([]);a.set_yticks([])
        fig.suptitle(d['episode']['scene']+' | '+d['episode']['id']+' | first query/mask; seed 20261110',fontsize=11)
        dest=path.with_name(d['episode']['id']+'.png');fig.savefig(dest,dpi=120);plt.close(fig);paths.append(str(dest))
    return paths


def main():
    result=json.loads((OUT / 'evaluation/scores.json').read_text()); analysis={}
    splits=sorted({r['split'] for r in result['rows']})+['used_both_clips']
    for split in splits:
        def selected(r):return r['split'] in ('used_clip40','used_clip80') if split=='used_both_clips' else r['split']==split
        all_rows=[r for r in result['rows'] if selected(r)]
        rows=[r for r in all_rows if r['condition']=='normal']; clean=[r for r in result['preservation'] if selected(r)]
        summary={v:{m:float(np.mean([r[m] for r in rows if r['variant']==v])) for m in METRICS[:3]} for v in VARIANTS+('backbone',)}
        for v in VARIANTS: summary[v]['clean_input_mse']=float(np.mean([r['clean_input_mse'] for r in clean if r['variant']==v]))
        summary['backbone']['clean_input_mse']=0.
        comparisons={m:{a+'_minus_'+b:compare(clean if m=='clean_input_mse' else rows,a,b,m) for a,b in PAIRS} for m in METRICS}
        interventions={}
        for v in ('center','direct'):
            interventions[v]={c:{m:float(np.mean([r[m] for r in all_rows if r['variant']==v and r['condition']==c]))
                                  for m in METRICS[:3]} for c in ('normal','removed','shuffled','swapped','wrong_camera')}
        analysis[split]={'summary':summary,'comparisons':comparisons,'interventions':interventions,
                         'scenes':len({r['scene'] for r in rows}),'confirmation_claim':False}
    write_json(OUT / 'analysis.json',analysis)
    paths=figures()
    fig,axes=plt.subplots(1,3,figsize=(16,4.5),constrained_layout=True)
    colors={'local':'#85868e','center':'#007d79','wide':'#ab72a3','direct':'#d27616'}
    for v in VARIANTS:
        curves=[]
        for s in SEEDS:
            log=[json.loads(x) for x in (model_path(v,s).parent / 'training.jsonl').read_text().splitlines()]
            curves.append(np.array([r['loss'] for r in log]).reshape(20,100).mean(-1))
        curves=np.array(curves); xs=np.arange(100,2001,100)
        axes[0].plot(xs,curves.mean(0),color=colors[v],label=v)
        axes[0].fill_between(xs,curves.min(0),curves.max(0),color=colors[v],alpha=.1)
    axes[0].set(xlabel='Training step',ylabel='All-patch training MSE');axes[0].legend()
    for ax,split in zip(axes[1:],('used_clip40','used_clip80')):
        s=analysis[split]['summary']
        for i,v in enumerate(('center','wide','direct')):
            delta=[100*(s[v][m]/s['local'][m]-1) for m in METRICS[:3]]
            ax.bar(np.arange(3)+(i-1)*.25,delta,.25,color=colors[v],label=v)
        ax.axhline(0,color='black',lw=.7);ax.set_xticks(range(3),['Hidden','Visible','All'])
        ax.set(title=split+' (already used)',ylabel='Error change vs original Local (%)\nNegative = improvement');ax.legend()
    fig.savefig(OUT / 'comparison.png',dpi=150);plt.close(fig);paths.append(str(OUT / 'comparison.png'))
    write_json(OUT / 'figure_manifest.json',{'figures':paths})
    print(json.dumps({s:v['summary'] for s,v in analysis.items()}),flush=True)


if __name__=='__main__':main()
