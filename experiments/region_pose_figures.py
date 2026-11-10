"""Visual summary of the fixed pose-utility comparison after all estimates."""
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.region_pose_utility import OUT
from experiments.round65_common import read, write, sha


def main():
    result = read(OUT / 'result.json')
    checked = read(OUT / 'independent_checks.json')
    assert checked['passed'] and checked['result_sha256'] == sha(OUT / 'result.json')
    destination = OUT / 'figures'
    destination.mkdir(exist_ok=False)
    labels = {'relative5': 'Full five', 'distance_mlp': 'Distance', 'geotransformer_rpe': 'Geo PE',
              'fixed_isotropic5': 'Fixed sphere', 'identity': 'Identity', 'gt_nearest': 'GT-assisted\nnearest center'}
    colors = ['#1769aa', '#cc7b28', '#7a5aa6', '#4b8e64', '#9b9b9b', '#c3c3c3']
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7), layout='constrained')
    for position, (role, color) in enumerate(zip(labels, colors)):
        values = [100 * row['macro_success_rate'] for row in result['records'] if row['role'] == role]
        axes[0].bar(position, np.mean(values), color=color, width=.65, alpha=.8)
        axes[0].scatter(position + np.linspace(-.09, .09, len(values)), values, color=color,
                        edgecolor='white', linewidth=.5, s=24, zorder=3)
        axes[0].text(position, np.mean(values) + 2, f'{np.mean(values):.2f}', ha='center', fontsize=9)
    axes[0].set_xticks(range(6), list(labels.values()), rotation=25, ha='right')
    axes[0].set_ylim(0, 108)
    axes[0].set_ylabel('Scene-balanced alignment success (%)')
    axes[0].set_title('All 480 pairs; failed estimates count as failures')
    limits = [0.]
    right_labels = []
    for position, row in enumerate(result['comparisons']):
        lo, hi = row['intervals_95_pp']['crossed']
        mean = row['mean_gain_pp']
        values = row['paired_seed_gains_pp']
        axes[1].plot([lo, hi], [position, position], color='#1769aa', lw=2)
        axes[1].plot(mean, position, 'o', color='#1769aa', ms=6)
        axes[1].scatter(values, position + np.linspace(-.09, .09, 3), marker='x', color='#1769aa', s=24)
        axes[1].annotate(f'{mean:+.2f}', (mean, position), xytext=(0, -20),
                         textcoords='offset points', ha='center', fontsize=9)
        right_labels.append('Full five − ' + labels[row['right']])
        limits.extend([lo, hi, *values])
    pad = max(1., .12 * (max(limits) - min(limits)))
    axes[1].set_xlim(min(limits) - pad, max(limits) + pad)
    axes[1].set_ylim(2.6, -.6)
    axes[1].set_yticks(range(3), right_labels)
    axes[1].axvline(0, ls='--', color='.5', lw=1)
    axes[1].set_xlabel('Success-rate difference (percentage points)')
    axes[1].set_title('Paired seeds and 95% seed × scene intervals')
    for axis in axes:
        axis.grid(axis='y' if axis is axes[0] else 'x', alpha=.15)
        axis.set_axisbelow(True)
    fig.suptitle('Fixed regional-PE utility: alignment RMSE < 0.2 m\n'
                 'Previously used five scenes; GT-assisted diagnostic is not deployable', fontsize=12)
    path = destination / 'pose_utility.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)
    write(destination / 'manifest.json', dict(created_unix=time.time(),
        result_sha256=sha(OUT / 'result.json'), independent_checks_sha256=sha(OUT / 'independent_checks.json'),
        renderer_sha256=sha(__file__), figures={path.name: sha(path)},
        visual_review_pending=True, fresh_confirmation=False))
    print('POSE_UTILITY_FIGURE', str(path), flush=True)


if __name__ == '__main__':
    main()
