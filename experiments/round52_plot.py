"""Display the fixed primary against all controls, including NoPE and simpler shapes."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/grope-r52-matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.round52_data import OUT, read, write, sha
from experiments.round52_train import ROLES, SEEDS

LABELS = dict(nope='No explicit PE', mlp='MLP coordinate warp', hrope='HRoPE adaptation',
    local_gs='Local Gaussian frame', camera_only='Camera only', camera_iso='Camera + isotropic',
    camera_gs='Camera + full Gaussian', camera_affine='Camera + affine')
COLORS = {r:dict(camera_gs='#b34c32', hrope='#356a9a', nope='#30343b', camera_only='#4b8060').get(r, '#a6afb7') for r in ROLES}


def main():
    summary = read(OUT/'summary.json')
    selection = read(OUT/'selection.json')
    if len(selection['records']) != len(ROLES)*len(SEEDS):
        raise ValueError('all selections required')
    records = {(r,s):read(OUT/'evaluation'/r/f'{s}_baseline.json') for r in ROLES for s in SEEDS}
    fig, axes = plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for ax, metric, label, scale in ((axes[0,0], 'epe_soft', 'Soft localization error (px; lower is better)', 1),
                                   (axes[0,1], 'pck8', 'PCK8 (%; higher is better)', 100)):
        means = [summary['aggregate'][r]['baseline'][metric]*scale for r in ROLES]
        ax.barh(range(len(ROLES)), means, color=[COLORS[r] for r in ROLES], alpha=.8)
        for j, role in enumerate(ROLES):
            values = [records[role,s]['metrics'][metric]*scale for s in SEEDS]
            ax.scatter(values, [j-.13,j,j+.13], s=18, color='black', zorder=3)
        ax.set_yticks(range(len(ROLES)), [LABELS[r] for r in ROLES]); ax.invert_yaxis()
        ax.set_xlabel(label); ax.grid(axis='x', alpha=.18); ax.set_axisbelow(True)
    ax = axes[1,0]; controls = [r for r in ROLES if r!='camera_gs']
    for j, role in enumerate(controls):
        delta = summary['paired']['camera_gs'][role]['epe_soft']
        ax.scatter(delta['per_seed'], [j-.13,j,j+.13], s=20, color='#a6afb7')
        ax.scatter([delta['mean']], [j], s=45, marker='D', color='#b34c32')
    ax.axvline(0,color='black',lw=1)
    ax.set_yticks(range(len(controls)), [LABELS[r] for r in controls]); ax.invert_yaxis()
    ax.set_xlabel('Full Gaussian minus control, soft EPE (px)\nNegative favors full Gaussian; diamonds are means')
    ax.grid(axis='x',alpha=.18)
    ax = axes[1,1]
    shown = ('nope','hrope','camera_only','camera_iso','camera_gs','camera_affine')
    colors = {'camera_iso':'#9c7c35','camera_affine':'#8c68a0',**{k:v for k,v in COLORS.items() if k not in ('camera_iso','camera_affine')}}
    for j, role in enumerate(shown):
        values = [summary['frame_gap'][role][str(g)]['epe_soft'] for g in (15,60)]
        ax.plot([0,1],values,marker='o',color=colors[role],label=LABELS[role],lw=1.6)
    ax.set_xticks([0,1],['15-frame gap','60-frame gap']); ax.set_xlim(-.15,1.15)
    ax.set_ylabel('Soft localization error (px)'); ax.legend(fontsize=8); ax.grid(alpha=.18)
    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False)
    axes[0,0].set_title('Continuous localization; individual seeds shown')
    axes[0,1].set_title('Coarse matching; individual seeds shown')
    axes[1,0].set_title('Paired primary comparisons')
    axes[1,1].set_title('Prespecified difficulty proxy')
    fig.suptitle('R52: camera x Gaussian-frame positional encoding\n3 paired seeds; 5 reused development rooms; 1,000 updates; not fresh confirmation',fontsize=13)
    folder=OUT/'plots';folder.mkdir(exist_ok=True)
    for extension in ('png','svg'):
        fig.savefig(folder/f'comparison.{extension}',dpi=160)
    plt.close(fig)
    write(folder/'manifest.json',dict(source_sha256=sha(__file__),summary_sha256=sha(OUT/'summary.json'),
        files={str(p):sha(p) for p in folder.glob('comparison.*')}))


if __name__=='__main__':
    main()
