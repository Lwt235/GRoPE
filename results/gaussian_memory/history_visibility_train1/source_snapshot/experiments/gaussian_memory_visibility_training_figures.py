"""Internal research figures: fixed aggregate comparisons and adverse examples."""
import json

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from experiments.gaussian_memory_visibility_training_data import (
    OUT, CACHE, PARENT, EVAL_CACHE, KINDS, SEEDS, CONDITIONS, load_plan, write_json)

NAMES = {'new_content': 'Matched C', 'new_soft': 'Matched Soft', 'new_center': 'Matched Center',
         'content_static': 'Strong old C', 'soft_dynamic': 'Old Soft', 'native': 'Native', 'local': 'Old Local'}
COLORS = {'new_content': '#a65c45', 'new_soft': '#20877d', 'new_center': '#6b83b9',
          'content_static': '#d49c79', 'soft_dynamic': '#80b8a8', 'native': '#888888', 'local': '#aaa766'}


def save(fig, name):
    dest = OUT / 'figures'; dest.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest / f'{name}.png', dpi=140); fig.savefig(dest / f'{name}.pdf'); plt.close(fig)


def aggregates(a):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), layout='constrained')
    for ax, seed in zip(axes, SEEDS):
        for kind, fam in [('content', 'new_content'), ('soft', 'new_soft'), ('soft_center', 'new_center')]:
            ax.plot(np.arange(100, 2001, 100), a['training'][f'{kind}_{seed}']['loss_by_100_steps'],
                    label=NAMES[fam], color=COLORS[fam])
        ax.set(title=f'Seed {seed}', xlabel='Training step', ylabel='Training native log-MSE / 100-step mean')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=9)
    fig.suptitle('Same initialization and same sampled examples; final 2000-step checkpoints only')
    save(fig, 'training_curves')

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout='constrained')
    for fam in ('native', 'content_static', 'new_content', 'new_center', 'new_soft', 'soft_dynamic'):
        key = fam if fam == 'native' else fam + '/history'
        values = [a['conditions'][c]['hidden_log_mse']['means'][key] for c in CONDITIONS]
        axes[0, 0].plot(range(3), values, '-o', label=NAMES[fam], color=COLORS[fam])
    axes[0, 0].set_xticks(range(3), ['Full', 'Query only', 'Persistent'])
    axes[0, 0].set(ylabel='Fixed ROI native log-MSE', title='History; all 12 evaluation places retained')
    axes[0, 0].legend(fontsize=9, ncol=2)
    comparisons = a['conditions']['persistent']['hidden_log_mse']['comparisons']
    keys = ['new_soft:history_vs_recent', 'history:new_soft_vs_new_content',
            'history:new_soft_vs_content_static', 'history:new_soft_vs_new_center',
            'history:new_soft_vs_soft_dynamic', 'history:new_soft_vs_native', 'history:new_soft_vs_local']
    labels = ['History vs Recent', 'Soft vs matched C', 'Soft vs strong old C', 'Soft vs matched Center',
              'Soft vs old Soft', 'Soft vs Native', 'Soft vs old Local']
    for i, key in enumerate(keys):
        r = comparisons[key]; lo, hi = r['relative_gain_percent_95']; mid = r['relative_gain_percent']
        axes[0, 1].hlines(i, lo, hi, color='#277f77'); axes[0, 1].plot(mid, i, 'o', color='#277f77')
    axes[0, 1].axvline(0, color='gray', lw=1); axes[0, 1].set_yticks(range(len(labels)), labels, fontsize=9)
    axes[0, 1].invert_yaxis(); axes[0, 1].set(xlabel='ROI error reduction (%) / paired place + seed 95%', title='Persistent condition: positive favors matched Soft')
    for i, fam in enumerate(('content_static', 'new_content', 'new_center', 'new_soft')):
        bias = a['conditions']['persistent']['bias_squared']['means'][fam + '/history']
        spatial = a['conditions']['persistent']['centered_log_mse']['means'][fam + '/history']
        axes[1, 0].bar(i, bias, color='#c3976c', label='Squared log-depth bias' if i == 0 else None)
        axes[1, 0].bar(i, spatial, bottom=bias, color='#648fa8', label='Centered spatial error' if i == 0 else None)
    axes[1, 0].set_xticks(range(4), ['Strong old C', 'Matched C', 'Center', 'Soft'])
    axes[1, 0].set(ylabel='Whole-image log-MSE', title='Where the whole-image error comes from'); axes[1, 0].legend(fontsize=9)
    families = ['new_content', 'new_center', 'new_soft', 'soft_dynamic']
    for i, fam in enumerate(families):
        vals = [a['conditions'][c]['hidden_log_mse']['comparisons'][fam+':history_vs_recent']['relative_gain_percent'] for c in CONDITIONS]
        axes[1, 1].plot(range(3), vals, '-o', color=COLORS[fam], label=NAMES[fam])
    axes[1, 1].set_xticks(range(3), ['Full', 'Query only', 'Persistent'])
    axes[1, 1].axhline(0, color='gray', lw=1); axes[1, 1].legend(fontsize=9)
    axes[1, 1].set(ylabel='History vs Recent ROI reduction (%)', title='Mean history increment; uncertainty is in the report')
    fig.suptitle('Task-matched training: three seeds, twelve held-out development places\nKnown cameras and full-input calibration; controlled artificial visibility, no fresh confirmation')
    save(fig, 'native_results')

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), layout='constrained')
    places = a['places']; x = np.arange(len(places))
    for key, name, color in [('new_soft:history_vs_recent', 'Soft History vs Recent', '#20877d'),
                              ('history:new_soft_vs_content_static', 'Soft vs strong old C', '#b88056'),
                              ('history:new_soft_vs_new_center', 'Soft vs Center', '#6b83b9')]:
        gain = comparisons[key]['place_gain']; axes[0].plot(x, [gain[p] for p in places], '-o', label=name, color=color)
    axes[0].axhline(0, color='gray', lw=1); axes[0].set_xticks(x, places, rotation=60, ha='right', fontsize=8)
    axes[0].set(ylabel='Baseline minus candidate ROI log-MSE', title='Place-level effects, averaged over seeds'); axes[0].legend(fontsize=9)
    support = a['parent_information_diagnostic']['gain_persistent/roi']['new_consistent_support_fraction']['by_place']
    gains = comparisons['new_soft:history_vs_recent']['place_gain']
    axes[1].scatter([100*support[p] for p in places], [gains[p] for p in places], color='#20877d')
    for p in places:
        axes[1].annotate(p, (100*support[p], gains[p]), xytext=(4, 4), textcoords='offset points', fontsize=7)
    axes[1].axhline(0, color='gray', lw=1)
    axes[1].set(xlabel='New consistent sensor history support / valid ROI (%)', ylabel='Soft Recent minus History ROI log-MSE',
                title='Descriptive information plot; no place selection or causal claim')
    save(fig, 'place_effects')


def examples(a):
    plan = load_plan(); gain = a['conditions']['persistent']['hidden_log_mse']['comparisons']['history:new_soft_vs_content_static']['place_gain']
    worst_place = min(a['places'], key=lambda p: (gain[p], p))
    worst = next(m['id'] for m in plan['catalog'] if m['place'] == worst_place)
    ids = list(dict.fromkeys(plan['fixed_figure_ids'] + [worst]))
    write_json(OUT / 'figure_selection.json', {'fixed_ids': plan['fixed_figure_ids'], 'worst_id': worst,
        'worst_place': worst_place, 'selected_ids': ids, 'seed': SEEDS[0], 'criterion': plan['counterexample']})
    for rid in ids:
        meta = next(m for m in plan['catalog'] if m['id'] == rid)
        sample = torch.load(EVAL_CACHE / rid / 'q11_c2.pt', map_location='cpu', weights_only=False)
        old = torch.load(EVAL_CACHE / rid / 'predictions.pt', map_location='cpu', weights_only=False)
        new = torch.load(CACHE / 'evaluation' / f'{rid}.pt', map_location='cpu', weights_only=False)
        target = sample['target'].numpy(); valid = sample['valid'].numpy(); roi = sample['hidden'].numpy()
        def prediction(export, family, policy):
            index = export['keys'].index([2, family, SEEDS[0], policy])
            return export['prediction_native'][index].numpy() * sample['metric_scale']
        estimates = [prediction(old, 'content_static', 'history'), prediction(new, 'new_content', 'history'),
                     prediction(new, 'new_center', 'history'), prediction(new, 'new_soft', 'history'),
                     prediction(new, 'new_soft', 'recent')]
        names = ['Strong old C / History', 'Matched C / History', 'Matched Center / History',
                 'Matched Soft / History', 'Matched Soft / Recent']
        errors = [np.abs(np.log(p) - np.log(target)) for p in estimates]
        emax = max(.1, float(np.quantile(np.concatenate([e[valid] for e in errors]), .99)))
        dmax = float(np.quantile(target[valid], .99))
        fig, axes = plt.subplots(2, 6, figsize=(18, 6.3), layout='constrained')
        axes[0, 0].imshow(sample['rgb'].numpy()); axes[0, 0].set_title('Persistent query RGB', fontsize=9)
        axes[1, 0].imshow(np.where(valid, target, np.nan), cmap='viridis', vmin=.1, vmax=dmax)
        axes[1, 0].set_title('Sensor depth: scoring only', fontsize=9)
        for i, (pred, error, name) in enumerate(zip(estimates, errors, names), 1):
            axes[0, i].imshow(np.where(valid, pred, np.nan), cmap='viridis', vmin=.1, vmax=dmax)
            axes[0, i].set_title(name, fontsize=9)
            im = axes[1, i].imshow(np.where(valid, error, np.nan), cmap='magma', vmin=0, vmax=emax)
            axes[1, i].set_title(f'ROI log-MSE {np.mean(error[roi & valid]**2):.4f}', fontsize=9)
        yy, xx = np.where(roi)
        for ax in axes.flat:
            ax.axis('off'); ax.add_patch(Rectangle((xx.min()-.5, yy.min()-.5), xx.max()-xx.min()+1,
                yy.max()-yy.min()+1, fill=False, ec='#29d7e2', lw=1.1))
        fig.colorbar(im, ax=axes[1, 1:], fraction=.02, pad=.01, label='Absolute log-depth error')
        qualifier = 'fixed example' if rid in plan['fixed_figure_ids'] else 'prespecified adverse-case selection'
        fig.suptitle(f'{rid} / {meta["place"]} / {qualifier}\nQuery11, persistent condition, seed {SEEDS[0]}; shared depth and error color limits')
        save(fig, f'example_{rid}')


if __name__ == '__main__':
    a = json.loads((OUT / 'analysis.json').read_text()); aggregates(a); examples(a)
