"""Display all fixed roles and paired effects; no model/metric selection."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/round50_matplotlib')
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.round50_data import OUT, CONDITIONS, sha, write
from models.round50_local_frame import ROLES


def run():
    summary=json.loads((OUT/'metrics/summary.json').read_text())
    post=json.loads((OUT/'postcheck/result.json').read_text())
    mats=[]
    for regime in ('oracle','noisy'):
        mats.append(np.array([[100*next(t['auc'] for t in summary['tables'] if
            (t['train_regime'],t['input_regime'],t['role'],t['condition'])==(regime,regime,r,c)) for c in CONDITIONS] for r in ROLES]))
    fig,axes=plt.subplots(1,2,figsize=(10,5),layout='constrained')
    lo=min(m.min() for m in mats);hi=max(m.max() for m in mats)
    for ax,mat,regime in zip(axes,mats,('Exact geometry','Corrupted geometry')):
        im=ax.imshow(mat,vmin=lo,vmax=hi,cmap='viridis',aspect='auto')
        ax.set_xticks(range(4),CONDITIONS);ax.set_yticks(range(len(ROLES)),ROLES)
        ax.set_title(regime+' (train and evaluation)')
        for i in range(len(ROLES)):
            for j in range(4):ax.text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center',color='white' if mat[i,j]<(lo+hi)/2 else 'black',fontsize=9)
    fig.colorbar(im,ax=axes,label='Mean PCK-curve AUC [0,8 px], percent',shrink=.8)
    fig.suptitle('R50 controlled development: all roles, three paired seeds')
    folder=OUT/'plots';folder.mkdir(exist_ok=True)
    fig.savefig(folder/'all_roles.png',dpi=160);fig.savefig(folder/'all_roles.svg');plt.close(fig)
    controls=[r for r in ROLES if r!='gs_frame']
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,regime in zip(axes,('oracle','noisy')):
        for j,ctrl in enumerate(controls):
            r=next(r for r in summary['comparisons'] if r['train_regime']==regime and r['input_regime']==regime and r['control']==ctrl and len(r['conditions'])==4)
            ci=np.array(r['scene_bootstrap_ci'])*100;mean=r['auc_delta']*100
            ax.plot(ci,[j,j],color='tab:blue',lw=3,alpha=.5)
            ax.scatter(np.array(r['seed_deltas'])*100,np.full(3,j)+[-.1,0,.1],s=20,c='tab:orange',zorder=3)
            ax.scatter([mean],[j],marker='|',s=120,c='black',zorder=4)
        ax.axvline(0,color='.4',lw=1);ax.set_yticks(range(len(controls)),controls);ax.invert_yaxis()
        ax.set_xlabel('Full GS minus control: AUC percentage points');ax.set_title(regime+' / equal-condition mean')
    fig.suptitle('Paired effects: orange = seeds; blue = scene-bootstrap interval')
    fig.savefig(folder/'paired_effects.png',dpi=160);fig.savefig(folder/'paired_effects.svg');plt.close(fig)
    write(folder/'manifest.json',dict(source_sha256=sha(__file__),summary_sha256=sha(OUT/'metrics/summary.json'),
        postcheck_sha256=sha(OUT/'postcheck/result.json'),grid_resolution_ceiling=post['grid_resolution_ceiling'],
        files={p.name:sha(p) for p in folder.glob('*') if p.suffix in ('.png','.svg')}))
    print('PLOTS',folder,flush=True)


if __name__=='__main__':run()
