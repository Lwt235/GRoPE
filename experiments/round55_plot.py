"""Fixed budget comparisons and complete train/validation curves."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/grope-r55-matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.round55_data import OUT,read,write,sha
from experiments.round55_train import ROLES,SEEDS

LABELS=('PRoPE','P + full Gaussian','P + offset','P + plane',
        'Ray point','Ray isotropic','Ray full Gaussian','Ray affine')


def main():
    summary=read(OUT/'summary.json');budget=read(OUT/'budget_comparison.json')
    if len(read(OUT/'selection.json')['records'])!=24:raise ValueError('seal all24 models')
    fig,axes=plt.subplots(1,3,figsize=(18,6),layout='constrained')
    for ax,metric,title,scale in ((axes[0],'epe_soft','Continuous EPE (px; lower is better)',1),
                                  (axes[1],'pck8','PCK8 (%; higher is better)',100)):
        for j,role in enumerate(ROLES):
            d=budget['budget_effect'][role][metric]
            ax.plot([d['original_mean']*scale,d['extended_mean']*scale],[j,j],color='#a0a0a0')
            ax.scatter(d['original_mean']*scale,j,color='#89939e',marker='o',label='1,000 steps' if j==0 else None)
            ax.scatter(d['extended_mean']*scale,j,color='#a8432d',marker='D',label='5,000 steps' if j==0 else None)
        ax.set_yticks(range(8),LABELS);ax.invert_yaxis();ax.set_xlabel(title);ax.legend()
    ax=axes[2];controls=('ray_point','ray_iso','ray_affine','camera_only')
    labels=dict(zip(ROLES,LABELS))
    for j,role in enumerate(controls):
        delta=summary['paired']['ray_gaussian'][role]['epe_soft']
        ax.scatter(delta['per_seed'],[j-.13,j,j+.13],color='#8d969f',s=22,
                   label='5k: individual seeds' if j==0 else None)
        ax.scatter(delta['mean'],j,color='#a8432d',marker='D',s=45,
                   label='5k: seed mean' if j==0 else None)
    ax.axvline(0,color='black',lw=1);ax.set_yticks(range(4),[labels[r] for r in controls]);ax.invert_yaxis()
    ax.set_xlabel('Full Gaussian minus control (px)\n5,000 steps; negative favors Gaussian')
    ax.legend(loc='lower right',fontsize=9)
    for ax in axes:
        ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('R55: matched optimization-budget diagnostic\n24 new fits; 3 paired seeds; 5 reused rooms; no fresh confirmation')
    folder=OUT/'plots';folder.mkdir(exist_ok=True)
    for ext in ('png','svg'):fig.savefig(folder/f'comparison.{ext}',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(14,16),layout='constrained',sharex=True)
    for ax,role,label in zip(axes.flat,ROLES,LABELS):
        for seed,color in zip(SEEDS,('#356a9a','#a8432d','#4b8060')):
            h=read(OUT/'runs'/role/str(seed)/'history.json');x=[d['step'] for d in h]
            ax.plot(x,[d['val_ce'] for d in h],color=color,lw=1.4,label=f'seed{seed} validation')
            ax.plot(x[1:],[d['train_interval_ce'] for d in h[1:]],color=color,lw=1,ls='--',alpha=.65)
            selected=read(OUT/'runs'/role/str(seed)/'run.json')
            ax.scatter(selected['best_step'],selected['val_ce'],marker='D',s=20,color=color)
        ax.axvline(1000,color='black',lw=.8,ls=':');ax.set_title(label);ax.set_ylabel('Cross entropy')
        ax.grid(alpha=.2);ax.legend(fontsize=7);ax.spines[['top','right']].set_visible(False)
    for ax in axes[-1]:ax.set_xlabel('Optimizer updates')
    fig.suptitle('Fixed optimization histories: solid validation CE; dashed sampled training CE\nDiamonds mark validation-selected models; dotted line ends the unchanged original prefix')
    for ext in ('png','svg'):fig.savefig(folder/f'learning_curves.{ext}',dpi=140)
    plt.close(fig)
    write(folder/'manifest.json',dict(source_sha256=sha(__file__),summary_sha256=sha(OUT/'summary.json'),
        budget_comparison_sha256=sha(OUT/'budget_comparison.json'),files={str(p):sha(p) for p in folder.glob('*') if p.suffix in ('.png','.svg')}))


if __name__=='__main__':main()
