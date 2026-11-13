"""Two concise scientific figures from complete fixed-endpoint outputs."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/native_pose_memory_mpl')
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.native_pose_memory_data import *
from experiments.native_pose_memory_score import np_camera

def main():
    import csv
    rows=list(csv.DictReader((OUT/'per_prediction_metrics.csv').open()));dev=[r for r in rows if r['split']=='dev' and r['query_group']=='fit'];summary=read(OUT/'summary.json')['dev_fit'];colors=['#697386','#9b7ab8','#417ca8','#d79a3a','#29977c'];kinds=['Native','L','P','S','G']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
    fig,ax=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
    for i,(k,col) in enumerate(zip(kinds,colors)):
        sub=[r for r in dev if r['kind']==k];a=summary['by_kind'][k];ax[0].bar(i,a['success_percent'],color=col)
        ax[0].text(i,a['success_percent']+.5,f"{a['success_percent']:.1f}",ha='center',fontsize=9)
        for panel,key in [(1,'rotation_deg'),(2,'center_m')]:
            x=np.sort([float(r[key]) for r in sub]);ax[panel].step(x,np.arange(1,len(x)+1)/len(x),where='post',label=k,color=col)
    ax[0].set(xticks=range(5),xticklabels=kinds,ylabel='Success (%)',title='Dev: 12 places × 4 queries × 3 seeds',ylim=(0,100));ax[1].set(xlabel='Rotation error (degrees)',ylabel='Empirical CDF',title='All outcomes retained',xscale='symlog');ax[2].set(xlabel='Camera-center error (m)',title='Map scale fixed by sources',xscale='symlog',xlim=(0,None));ax[1].axvline(5,color='gray',ls=':',lw=1);ax[2].axvline(.2,color='gray',ls=':',lw=1);ax[2].legend(fontsize=8)
    gp=summary['comparisons'][0];fig.suptitle(f"G − P success: {gp['mean']:+.2f} pp  [95% CI {gp['CI95'][0]:+.2f}, {gp['CI95'][1]:+.2f}] — used development data",fontsize=11)
    folder=OUT/'figures';folder.mkdir(exist_ok=True);fig.savefig(folder/'pose_summary.png');fig.savefig(folder/'pose_summary.pdf');plt.close(fig)
    examples=read(OUT/'counterexamples.json');fig,axes=plt.subplots(2,3,figsize=(11,7),layout='constrained')
    for idx,(label,example) in enumerate([('Fixed example',examples['fixed']),('Largest G/P loss',examples['maximum'])]):
        rid,q=example['id'],example['query'];plotseed=20261110
        if idx==1:
            lookup={(r['kind'],int(r['seed'])):r for r in dev if r['id']==rid and int(r['query'])==q}
            plotseed=max(SEEDS,key=lambda seed:(int(lookup[('P',seed)]['success'])-int(lookup[('G',seed)]['success']),float(lookup[('G',seed)]['center_m'])-float(lookup[('P',seed)]['center_m'])))
        payload=load(query(rid,q));sc=load(sceneout(rid));target=payload['c2w'].numpy();c=load(casepath(rid,q));native=np_camera(c['initial_pose'],sc['alignment'])
        axes[idx,0].imshow(payload['rgb'].permute(1,2,0).numpy());axes[idx,0].axis('off');axes[idx,0].set_title(f'{label}\n{rid}, query {q}')
        axes[idx,1].scatter(target[0,3],target[2,3],marker='*',s=140,c='black',label='Truth');centers=[native]
        for kind in KINDS:
            pred=load(CACHE/'predictions'/f'{kind}_{plotseed}'/rid/f'{q}.pt');centers.append(np_camera(pred['pose'],sc['alignment']))
        for k,col,cam in zip(kinds,colors,centers):
            axes[idx,1].scatter(cam[0,3],cam[2,3],label=k,color=col);axes[idx,1].plot([target[0,3],cam[0,3]],[target[2,3],cam[2,3]],color=col,alpha=.4)
        axes[idx,1].set(xlabel='Map x (m)',ylabel='Map z (m)',title=f'Camera centers; seed {plotseed}');axes[idx,1].axis('equal');axes[idx,1].legend(fontsize=7,ncol=2)
        axes[idx,2].axis('off');lines=['All three seeds (rotation°, center m; success)']
        for kind in kinds:
            sr=[r for r in dev if r['kind']==kind and r['id']==rid and int(r['query'])==q];lines.append(kind+': '+ ' | '.join(f"{float(r['rotation_deg']):.2f}, {float(r['center_m']):.3f}; {r['success']}" for r in sr))
        sr=[r for r in dev if r['id']==rid and int(r['query'])==q][0];lines.extend(['',f"Predicted GS content coverage: {float(sr['coverage']):.1%}",'Truth is used only for loss/scoring and this figure.']);axes[idx,2].text(0,1,'\n\n'.join(lines),va='top',fontsize=8)
    fig.savefig(folder/'pose_examples.png');fig.savefig(folder/'pose_examples.pdf');plt.close(fig)
if __name__=='__main__':main()
