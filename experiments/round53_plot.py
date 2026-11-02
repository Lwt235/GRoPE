"""Display the fixed primary against all controls, including NoPE and simpler shapes."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/grope-r53-matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.round53_data import OUT, read, write, sha
from experiments.round53_train import ROLES, SEEDS

LABELS = dict(nope='No explicit PE', mlp='MLP coordinate warp', hrope='HRoPE adaptation',
    projection_only='R52 camera projection', pose_only='Camera pose only', ray_point='Ray + point',
    ray_iso='Ray + isotropic', ray_gaussian='Ray + full Gaussian', ray_affine='Ray + affine',
    ray_gaussian_hybrid='Ray Gaussian + rotary')
COLORS = {r:dict(ray_gaussian='#b34c32', hrope='#356a9a', nope='#30343b',
    projection_only='#4b8060', ray_gaussian_hybrid='#8661a0').get(r, '#a6afb7') for r in ROLES}


def main():
    summary = read(OUT/'summary.json')
    selection = read(OUT/'selection.json')
    if len(selection['records']) != len(ROLES)*len(SEEDS):
        raise ValueError('all selections required')
    records = {(r,s):read(OUT/'evaluation'/r/f'{s}_baseline.json') for r in ROLES for s in SEEDS}
    fig, axes = plt.subplots(2,2,figsize=(14,10),layout='constrained')
    for ax, metric, label, scale in ((axes[0,0], 'epe_soft', 'Soft localization error (px; lower is better)', 1),
                                   (axes[0,1], 'pck8', 'PCK8 (%; higher is better)', 100)):
        means = [summary['aggregate'][r]['baseline'][metric]*scale for r in ROLES]
        ax.barh(range(len(ROLES)), means, color=[COLORS[r] for r in ROLES], alpha=.8)
        for j, role in enumerate(ROLES):
            values = [records[role,s]['metrics'][metric]*scale for s in SEEDS]
            ax.scatter(values, [j-.13,j,j+.13], s=18, color='black', zorder=3)
        ax.set_yticks(range(len(ROLES)), [LABELS[r] for r in ROLES]); ax.invert_yaxis()
        ax.set_xlabel(label); ax.grid(axis='x', alpha=.18); ax.set_axisbelow(True)
    ax = axes[1,0]; controls = [r for r in ROLES if r!='ray_gaussian']
    for j, role in enumerate(controls):
        delta = summary['paired']['ray_gaussian'][role]['epe_soft']
        ax.scatter(delta['per_seed'], [j-.13,j,j+.13], s=20, color='#a6afb7')
        ax.scatter([delta['mean']], [j], s=45, marker='D', color='#b34c32')
    ax.axvline(0,color='black',lw=1)
    ax.set_yticks(range(len(controls)), [LABELS[r] for r in controls]); ax.invert_yaxis()
    ax.set_xlabel('Full Gaussian minus control, soft EPE (px)\nNegative favors full Gaussian; diamonds are means')
    ax.grid(axis='x',alpha=.18)
    ax = axes[1,1]
    shown = ('nope','projection_only','ray_point','ray_iso','ray_gaussian','ray_gaussian_hybrid')
    colors = {'ray_iso':'#9c7c35','ray_affine':'#8c68a0',**{k:v for k,v in COLORS.items() if k not in ('ray_iso','ray_affine')}}
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
    fig.suptitle('R53: ray-anchored inverse Gaussian-frame PE\n3 paired seeds; 18 new fits + 12 reused references; 5 old rooms; not fresh confirmation',fontsize=13)
    folder=OUT/'plots';folder.mkdir(exist_ok=True)
    for extension in ('png','svg'):
        fig.savefig(folder/f'comparison.{extension}',dpi=160)
    plt.close(fig)
    write(folder/'manifest.json',dict(source_sha256=sha(__file__),summary_sha256=sha(OUT/'summary.json'),
        files={str(p):sha(p) for p in folder.glob('comparison.*')}))


if __name__=='__main__':
    main()
