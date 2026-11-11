"""Summaries and diagnostic figures from fixed, already-completed predictions."""
import json

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import OUT, SEEDS, VARIANTS, write_json
from experiments.gaussian_memory_data import REPORT


def mean_rows(data, role, variant, condition='normal'):
    rows = [r for r in data['summary'] if r['role'] == role and r['variant'] == variant and r['condition'] == condition]
    assert rows
    return {key: float(np.mean([r[key] for r in rows])) for key in ('mse', 'cosine')}


def summarize():
    old = json.loads((OUT / 'evaluation/scores.json').read_text())
    corrected = json.loads((OUT / 'projection_matched/scores.json').read_text())
    conf = json.loads((OUT / 'confirmation/scores.json').read_text())
    renderer = json.loads((REPORT / 'evaluation/scores.json').read_text())
    tables = {}
    for role in ('train_query', 'heldout_view', 'heldout_scene'):
        values = {v: mean_rows(old, role, v) for v in ('mlp', 'center')}
        values['gaussian_original'] = mean_rows(old, role, 'gaussian')
        values['gaussian_corrected'] = mean_rows(corrected, role, 'gaussian')
        values['gaussian_corrected_isotropic'] = mean_rows(corrected, role, 'gaussian', 'isotropic')
        values['renderer'] = mean_rows(renderer, role, 'learned_geometry')
        tables[role] = values
    tables['confirmation_scene'] = {v: mean_rows(conf, 'confirmation_scene', v) for v in (*VARIANTS, 'renderer')}
    tables['confirmation_scene']['gaussian_isotropic'] = mean_rows(conf, 'confirmation_scene', 'gaussian', 'isotropic')
    scene_rows = []
    for scene in sorted({r['scene'] for r in conf['per_scene']}):
        item = {'scene': scene}
        for variant in (*VARIANTS, 'renderer'):
            rows = [r for r in conf['per_scene'] if r['scene'] == scene and r['variant'] == variant and r['condition'] == 'normal']
            item[variant] = float(np.mean([r['mse'] for r in rows]))
        item['gaussian_isotropic'] = float(np.mean([r['mse'] for r in conf['per_scene']
                                          if r['scene'] == scene and r['variant'] == 'gaussian' and r['condition'] == 'isotropic']))
        scene_rows.append(item)
    comparisons = []
    for baseline in ('mlp', 'center'):
        delta = np.array([r[baseline] - r['gaussian'] for r in scene_rows])
        rng = np.random.default_rng(44101)
        boot = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(1)
        by_seed = []
        for seed in SEEDS:
            b = next(r['mse'] for r in conf['summary'] if r['seed'] == seed and r['variant'] == baseline and r['condition'] == 'normal')
            g = next(r['mse'] for r in conf['summary'] if r['seed'] == seed and r['variant'] == 'gaussian' and r['condition'] == 'normal')
            by_seed.append({'seed': seed, 'baseline_mse': b, 'gaussian_mse': g, 'improvement': b - g})
        interval = np.quantile(boot, [.025, .975]).tolist()
        comparisons.append({'baseline': baseline, 'mse_reduction': float(delta.mean()),
                            'relative_reduction_percent': float(delta.mean() / np.mean([r[baseline] for r in scene_rows]) * 100),
                            'scene_bootstrap95': interval, 'improved_scenes': int((delta > 0).sum()),
                            'total_scenes': len(delta), 'by_seed': by_seed,
                            'confirmation_rule_met': bool(interval[0] > 0 and (delta > 0).sum() > len(delta) / 2
                                                          and all(x['improvement'] > 0 for x in by_seed))})
    result = {'tables': tables, 'confirmation_per_scene': scene_rows, 'confirmation_comparisons': comparisons,
              'confirmation_rule_met': all(c['confirmation_rule_met'] for c in comparisons),
              'uncertainty_unit': '12 scenes; paired bootstrap after averaging the 3 fixed reader seeds within scene',
              'scope': 'Known cameras, frozen VGGT/writer, 256-Gaussian local memory, compressed teacher feature readout; no VGGT downstream claim'}
    write_json(OUT / 'analysis.json', result)
    print(json.dumps(result), flush=True)
    return result


def plots(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    base = torch.load(OUT / 'evaluation/visual_data.pt', weights_only=False, map_location='cpu')
    fixed = torch.load(OUT / 'projection_matched/visual_data.pt', weights_only=False, map_location='cpu')
    conf = torch.load(OUT / 'confirmation/visual_data.pt', weights_only=False, map_location='cpu')
    exported = []
    folder = OUT / 'final_figures'
    folder.mkdir(exist_ok=True)
    for scope, all_data in [('development', base), ('confirmation', conf)]:
        for rid, data in all_data.items():
            prediction = fixed[rid]['normal'] if scope == 'development' else data['gaussian/normal']
            fields = [('rgb', 'Query RGB (display only)'), ('target', 'Teacher feature'), ('renderer', 'Existing renderer'),
                      ('mlp/normal', 'Attention + MLP'), ('center/normal', '+ Center distance'), ('corrected', '+ Gaussian / gsplat')]
            fig, axes = plt.subplots(3, 6, figsize=(13.8, 6))
            for i, q in enumerate((2, 6, 10)):
                qi = data['qids'].index(q)
                for j, (key, label) in enumerate(fields):
                    a = (prediction if key == 'corrected' else data[key])[qi].numpy()
                    if key != 'rgb':
                        a = np.clip(a[..., :3] / 4 + .5, 0, 1)
                    axes[i, j].imshow(a, interpolation='nearest')
                    axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
                    if i == 0:
                        axes[i, j].set_title(label, fontsize=9)
                    if j == 0:
                        axes[i, j].set_ylabel(f'Query ordinal {q}')
            fig.suptitle(data['scene'] + f' | {scope}\nSame saved memory, 3 query cameras; fixed seed 20261110, common PCA color scale', fontsize=10)
            fig.tight_layout()
            path = folder / (rid + '.png')
            fig.savefig(path, dpi=140); plt.close(fig)
            exported.append(str(path))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    labels = [('heldout_view', 'Held-out views / fit scenes'), ('heldout_scene', '6 development scenes'),
              ('confirmation_scene', '12 confirmation scenes')]
    for ax, (role, label) in zip(axes, labels):
        rows = result['tables'][role]
        g = 'gaussian' if role == 'confirmation_scene' else 'gaussian_corrected'
        keys = ('mlp', 'center', g, 'renderer')
        values = [rows[k]['mse'] for k in keys]
        bars = ax.bar(['MLP', 'Center', 'Gaussian', 'Renderer'], values, color=['#7f8c8d', '#e69f00', '#009e73', '#56b4e9'])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + .007, f'{value:.3f}', ha='center', fontsize=9)
        ax.set(title=label, ylabel='Teacher feature MSE (lower is better)', ylim=(0, max(values) * 1.18))
    fig.suptitle('Fixed memory readout; matched projection for Gaussian; renderer is a separate reference', fontsize=11)
    fig.tight_layout()
    path = folder / 'comparison.png'; fig.savefig(path, dpi=150); plt.close(fig); exported.append(str(path))
    write_json(folder / 'figures.json', exported)
    print(json.dumps({'final_figures': len(exported)}), flush=True)


if __name__ == '__main__':
    plots(summarize())
