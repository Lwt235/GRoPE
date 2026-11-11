"""Source-retention contrasts and three descriptive diagnostic figures."""
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from experiments.gaussian_memory_addressability import OUT, CONDITIONS, SEEDS, read, load, write_json, sha256, ROOT
from experiments.gaussian_memory_addressability_analysis import summary, softmax
from experiments.gaussian_memory_visibility_data import CACHE


def retention():
    catalog = sorted(read(OUT / 'plan.json')['catalog'], key=lambda m: m['place'])
    places = [m['place'] for m in catalog]; rows = []
    for meta in catalog:
        rec = read(OUT / 'clips' / f'{meta["id"]}.json'); artifact = load(rec['artifact_path'])
        for condition in ('full', 'persistent'):
            labels = artifact['labels'][condition]
            before = artifact['bindings'][condition + '/recent']; after = artifact['bindings'][condition + '/history']
            for region in ('roi', 'all'):
                valid = labels['query_valid'].numpy().astype(bool)
                if region == 'roi': valid &= artifact['roi'].numpy()[labels['query_patch'].numpy()]
                denom = int(valid.sum())
                row = {'id': meta['id'], 'place': meta['place'], 'condition': condition, 'region': region, 'valid_pixels': denom}
                for stage in ('raw', 'admitted', 'retained'):
                    a = before['coverage'][stage].numpy(); b = after['coverage'][stage].numpy()
                    row[stage + '_gain'] = float(((b & ~a) & valid).sum() / denom)
                    row[stage + '_loss'] = float(((a & ~b) & valid).sum() / denom)
                    row[stage + '_net'] = row[stage + '_gain'] - row[stage + '_loss']
                rows.append(row)
    comparisons = {}
    for condition in ('full', 'persistent'):
        for region in ('roi', 'all'):
            selected = [r for r in rows if r['condition'] == condition and r['region'] == region]
            assert [r['place'] for r in selected] == places
            comparisons[condition + '/' + region] = {key: summary([r[key] for r in selected], places)
                for key in selected[0] if key.endswith(('_gain', '_loss', '_net'))}
    result = {'complete': True, 'rows': rows, 'comparisons': comparisons,
        'scope': 'All retained source-pixel correspondences. Set differences do not establish lost contextual feature information or downstream causality.',
        'source_sha256': sha256(ROOT / 'experiments/gaussian_memory_addressability_figures.py')}
    write_json(OUT / 'retention_comparison.json', result)
    return result


def finish(fig, name):
    fig.savefig(OUT / 'figures' / (name + '.png'), dpi=170, bbox_inches='tight')
    fig.savefig(OUT / 'figures' / (name + '.pdf'), bbox_inches='tight')
    plt.close(fig)


def main():
    torch.set_num_threads(4); analysis = read(OUT / 'analysis.json'); retained = retention()
    (OUT / 'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    colors = ['#A6ACB1', '#E9A34B', '#287F8E']; x = np.arange(12)
    places = analysis['places']; names = [s.replace('harvard_robotics_lab', 'harvard_robotics').replace('mit_w59_conference', 'mit_w59_conf') for s in places]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    prov = analysis['provenance']['persistent/history/roi/early_exclusive']['place_rows']
    for i, (stage, label) in enumerate(zip(('raw', 'admitted', 'retained'), ('Available sources', 'Confidence-admitted', 'Represented by slot members'))):
        axes[0].bar(x + (i - 1) * .25, [r[stage + '_coverage'] * 100 for r in prov], .24, color=colors[i], label=label)
    axes[0].set(ylabel='ROI valid pixels (%)', title='A  History-only corresponding sources: 11.37% available, 4.83% represented')
    axes[0].legend(frameon=False, loc='upper right'); axes[0].grid(axis='y', alpha=.15)
    selected = [r for r in retained['rows'] if r['condition'] == 'persistent' and r['region'] == 'roi']
    axes[1].bar(x, [r['retained_gain'] * 100 for r in selected], color='#287F8E', label='Newly represented')
    axes[1].bar(x, [-r['retained_loss'] * 100 for r in selected], color='#C46252', label='Previously represented, then lost')
    axes[1].plot(x, [r['retained_net'] * 100 for r in selected], 'kD', ms=4, label='Net change')
    axes[1].axhline(0, color='black', linewidth=.7)
    axes[1].set(ylabel='Change in ROI coverage (pp)', title='B  Recent to History, same 256 slots: gains and displaced source correspondences')
    axes[1].set_xticks(x, names, rotation=30, ha='right'); axes[1].legend(frameon=False, ncol=3, fontsize=9)
    fig.suptitle('Persistent masking | 12 previously used places | provenance proxy, not feature reconstruction', fontsize=12)
    fig.tight_layout(); finish(fig, 'retention')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    families = ['prior_only', 'content_static', 'soft_dynamic', 'new_soft']; names = ['Prior only', 'Old C Q/K', 'Old Soft Q/K', 'New Soft Q/K']
    for ax, label, title in zip(axes, ('all', 'early_exclusive'), ('All corresponding sources', 'History-only corresponding sources')):
        ctx = 'persistent/history/roi/' + label
        for i, (kind, name) in enumerate(zip(('content', 'soft', 'soft_center'), ('Uniform prior', 'Gaussian prior', 'Fixed-sphere prior'))):
            stats = [analysis['route'][ctx][fam + '/' + kind] for fam in families]
            means = np.array([s['mean'] * 100 for s in stats]); ci = np.array([s['paired_95'] for s in stats]).T * 100
            ax.bar(np.arange(4) + (i - 1) * .25, means, .24, color=colors[i], label=name)
            ax.errorbar(np.arange(4) + (i - 1) * .25, means, np.stack((means - ci[0], ci[1] - means)), fmt='none', color='#303030', lw=.8, capsize=2)
        ax.set_xticks(np.arange(4), names, rotation=15); ax.set_title(title)
        ax.set_ylabel('Correspondence-weighted routing score (%)'); ax.grid(axis='y', alpha=.15)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle('Same Q/K, different prior | diagnostic score, not task accuracy | paired 95% intervals', fontsize=12)
    fig.tight_layout(); finish(fig, 'routing')

    fig, axes = plt.subplots(3, 5, figsize=(15, 8.5))
    for row, rid in enumerate(('fidelity_00', 'fidelity_01', 'fidelity_02')):
        rec = read(OUT / 'clips' / f'{rid}.json'); artifact = load(rec['artifact_path']); sample = load(CACHE / rid / 'q11_c2.pt')
        lab = artifact['labels']['persistent']; b = artifact['bindings']['persistent/history']
        roi = artifact['roi'].numpy().reshape(15, 20); yy, xx = np.where(roi)
        exclusive = lab['exclusive'].numpy(); covered = b['coverage']['retained'].numpy()
        score = []; difference = []; utility = b['utility']['all'].numpy()
        for item in artifact['qk']:
            if item['family'] != 'soft_dynamic' or item['condition'] != 'persistent' or item['policy'] != 'history': continue
            q = item['q'].float().numpy(); k = item['k'].float().numpy(); logits = q @ k.transpose(0, 2, 1) / 4
            prior = artifact['priors']['persistent/history']; s = {}
            for kind in ('soft', 'soft_center'):
                a = softmax(logits + prior[kind].numpy()); s[kind] = (a * utility[None]).sum(-1).mean(0)
            score.append(s['soft']); difference.append(s['soft'] - s['soft_center'])
        axes[row, 0].imshow(sample['rgb'].numpy())
        axes[row, 1].imshow(exclusive.reshape(210, 280), vmin=0, vmax=1, cmap='Blues')
        axes[row, 2].imshow((exclusive & covered).reshape(210, 280), vmin=0, vmax=1, cmap='Blues')
        image = axes[row, 3].imshow(np.mean(score, 0).reshape(15, 20) * 100, vmin=0, vmax=5, cmap='viridis', extent=(0, 280, 210, 0))
        difference = np.mean(difference, 0).reshape(15, 20) * 100
        diffimage = axes[row, 4].imshow(difference, vmin=-2, vmax=2, cmap='RdBu_r', extent=(0, 280, 210, 0))
        for ax in axes[row]:
            ax.add_patch(Rectangle((xx.min() * 14, yy.min() * 14), (xx.max() - xx.min() + 1) * 14,
                                  (yy.max() - yy.min() + 1) * 14, fill=False, ec='#DF5C48', lw=1.1))
            ax.set_xticks([]); ax.set_yticks([])
        axes[row, 0].set_ylabel(rec['place'])
    for ax, title in zip(axes[0], ('Masked query', 'Available history-only', 'Represented history-only', 'Old Soft routing score (%)', 'Gaussian - sphere (pp)')): ax.set_title(title, fontsize=10)
    fig.suptitle('Fixed first three clips, including a zero-history-support case | red rectangle: scored ROI', fontsize=12)
    fig.subplots_adjust(top=.89, bottom=.12, left=.08, right=.97, wspace=.13, hspace=.1)
    fig.colorbar(image, cax=fig.add_axes([.63, .045, .13, .015]), orientation='horizontal')
    fig.colorbar(diffimage, cax=fig.add_axes([.82, .045, .13, .015]), orientation='horizontal')
    finish(fig, 'fixed_cases')
    write_json(OUT / 'figure_manifest.json', {'complete': True, 'files': {str(p.relative_to(OUT)): sha256(p)
        for p in sorted((OUT / 'figures').iterdir())}, 'source_sha256': sha256(ROOT / 'experiments/gaussian_memory_addressability_figures.py'),
        'fixed_case_ids': ['fidelity_00', 'fidelity_01', 'fidelity_02'], 'images_actually_reviewed': False})


if __name__ == '__main__': main()
