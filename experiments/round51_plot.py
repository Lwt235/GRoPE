"""Compact development-result figure with paired seeds and scene differences."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/grope-r51-matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.round51_data import OUT, read, write, sha
from experiments.round51_train import SEEDS, ROLES

LABELS=dict(rope='RoPE',mlp='MLP',hrope='HRoPE adaptation',hrope_paper='HRoPE paper form',
    gs_frame='Full GS frame',gs_iso='Isotropic GS',affine='Free affine',
    frame_identity='Identity frame',gaussian_bias='Gaussian bias',nope='NoPE')
COLORS={r:('#b34c32' if r=='gs_frame' else '#356a9a' if r.startswith('hrope') else '#414141' if r=='mlp' else '#a7afb6') for r in ROLES}


def main():
    summary=read(OUT/'summary.json');selected=read(OUT/'selection.json')
    if len(selected['records'])!=30:raise ValueError('all paired selections required')
    records={(role,seed):read(OUT/'evaluation'/role/f'{seed}_baseline.json') for role in ROLES for seed in SEEDS}
    rows=range(len(ROLES));fig,axes=plt.subplots(2,2,figsize=(13,10),layout='constrained')
    for ax,metric,label,scale in ((axes[0,0],'epe_soft','Soft localization error (px; lower is better)',1),
                                  (axes[0,1],'pck8','PCK8 (%; higher is better)',100)):
        means=[summary['aggregate'][r]['baseline'][metric]*scale for r in ROLES]
        ax.barh(list(rows),means,color=[COLORS[r] for r in ROLES],alpha=.8)
        for j,role in enumerate(ROLES):
            values=[records[role,s]['metrics'][metric]*scale for s in SEEDS]
            ax.scatter(values,[j-.13,j,j+.13],s=17,c='black',zorder=3)
        ax.set_yticks(list(rows),[LABELS[r] for r in ROLES]);ax.invert_yaxis();ax.set_xlabel(label)
        ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
    ax=axes[1,0];scene_names=sorted({x['scene'] for x in records['mlp',11]['rows']})
    for role,offset in (('hrope',-.2),('hrope_paper',0),('gs_frame',.2)):
        values=[]
        for scene in scene_names:
            effects=[]
            for seed in SEEDS:
                group=[[x for x in records[r,seed]['rows'] if x['scene']==scene] for r in (role,'mlp')]
                effect=[sum(x['epe_soft'] for x in g)/sum(x['n_valid'] for x in g) for g in group]
                effects.append(effect[0]-effect[1])
            values.append(np.mean(effects))
        ax.scatter(values,np.arange(len(scene_names))+offset,label=LABELS[role],color=COLORS[role],
            marker='s' if role=='hrope_paper' else 'o',s=43)
    ax.axvline(0,color='black',lw=1);ax.set_yticks(range(len(scene_names)),[n.split('/')[0] for n in scene_names]);ax.invert_yaxis()
    ax.set_xlabel('Soft EPE difference from MLP (px; negative favors method)');ax.legend(fontsize=8);ax.grid(axis='x',alpha=.18)
    ax=axes[1,1]
    drop=[100*(summary['aggregate'][r]['baseline']['pck8']-summary['aggregate'][r]['rgb_permuted']['pck8']) for r in ROLES]
    ax.barh(list(rows),drop,color=[COLORS[r] for r in ROLES]);ax.set_yticks(list(rows),[LABELS[r] for r in ROLES]);ax.invert_yaxis()
    ax.set_xlabel('PCK8 drop after RGB permutation (percentage points)');ax.grid(axis='x',alpha=.18)
    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False)
    axes[0,0].set_title('Localization, with individual training seeds')
    axes[0,1].set_title('Coarse matching, with individual training seeds')
    axes[1,0].set_title('Scene consistency of the requested comparison')
    axes[1,1].set_title('Fixed-checkpoint visual dependence diagnostic')
    fig.suptitle('R51: real SUN3D matching, identical RGB features + positions + cameras\n3 seeds; 5 reused development rooms; 1,000 updates; not fresh confirmation',fontsize=13)
    folder=OUT/'plots';folder.mkdir(exist_ok=True)
    for extension in ('png','svg'):fig.savefig(folder/f'comparison.{extension}',dpi=160)
    plt.close(fig)
    write(folder/'manifest.json',dict(source_sha256=sha(__file__),summary_sha256=sha(OUT/'summary.json'),
        files={str(p):sha(p) for p in folder.glob('comparison.*')}))


if __name__=='__main__':main()
