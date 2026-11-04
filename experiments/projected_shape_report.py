"""Paired old-data attribution report; never promotes a new utility claim."""
from pathlib import Path

import numpy as np

from experiments.projected_shape_attribution import OUT, ROLES, SEEDS, CONDITIONS, check, parent
from experiments.round65_common import read, write, sha

METRICS = ('psnr', 'ssim', 'lpips', 'raw_mse', 'clipped_fraction')


def role_order(fixed):
    group = fixed.split('_')[0]
    return (group+'_prope', group+'_target_distance', group+'_target_direct',
            group+'_urope', group+'_target_relative5', fixed)


def summarize():
    check()
    selected = read(OUT/'selections.json')
    assert selected['all_six_selected_before_old_eval']
    summary, files, errors, reused = {}, {}, [], {}
    for fixed in ROLES:
        root, candidate = parent(fixed)
        old = read(root/'result.json')
        reused[str(root/'result.json')] = sha(root/'result.json')
        for role in role_order(fixed)[:-1]:
            summary[role] = old['summary'][role]
        summary[fixed] = {}
        for condition in CONDITIONS:
            seeds = []
            for seed in SEEDS:
                path = OUT/'predictions'/fixed/str(seed)/f'{condition}.npz'
                meta = read(path.with_suffix('.json'))
                assert meta['prediction_sha256'] == sha(path)
                assert meta['checkpoint_sha256'] == sha(OUT/'runs'/fixed/str(seed)/'best.pt')
                assert meta['selections_sha256'] == sha(OUT/'selections.json')
                files[str(path)] = sha(path)
                prior = root/'predictions'/candidate/str(seed)/f'{condition}.npz'
                # Result identities were frozen before new fits. Confirm pairing
                # against the saved candidate predictions, without a fresh cohort.
                assert old['prediction_files'][str(prior)] == sha(prior)
                with np.load(path) as z, np.load(prior) as original:
                    assert z['prediction'].shape == (240, 96, 128, 3)
                    assert np.array_equal(z['target_uint8'], original['target_uint8'])
                    assert np.array_equal(z['scene_id'], original['scene_id'])
                    assert np.array_equal(np.bincount(z['scene_id']), [48]*5)
                    p = z['prediction'].astype('float64')
                    # Targets in the training/evaluation API are float32 / 255.
                    target = z['target_uint8'].astype('float32')/255
                    psnr = -10*np.log10(np.maximum(((p.clip(0, 1)-target)**2).mean((1, 2, 3)), 1e-15))
                    mse = ((p-target)**2).mean((1, 2, 3))
                    err = dict(role=fixed, seed=seed, condition=condition,
                               psnr=float(np.max(np.abs(psnr-z['psnr']))),
                               raw_mse=float(np.max(np.abs(mse-z['raw_mse']))))
                    assert err['psnr'] < 2e-5 and err['raw_mse'] < 1e-7
                    errors.append(err)
                    for metric in METRICS:
                        assert z[metric].shape == (240,) and np.isfinite(z[metric]).all()
                    seeds.append({metric: [float(z[metric][z['scene_id'] == scene].mean())
                                           for scene in range(5)] for metric in METRICS})
            summary[fixed][condition] = {}
            for metric in METRICS:
                values = np.array([row[metric] for row in seeds])
                summary[fixed][condition][metric] = dict(seed_scene=values.tolist(),
                    seed_mean=values.mean(1).tolist(), scene_mean=values.mean(0).tolist(), mean=float(values.mean()))

    differences = {}
    for fixed in ROLES:
        _, candidate = parent(fixed)
        differences[candidate] = {}
        for control in role_order(fixed):
            if control == candidate:
                continue
            differences[candidate][control] = {}
            for condition in CONDITIONS:
                differences[candidate][control][condition] = {}
                for metric in METRICS:
                    d = (np.array(summary[candidate][condition][metric]['seed_scene'])
                         - np.array(summary[control][condition][metric]['seed_scene']))
                    differences[candidate][control][condition][metric] = dict(
                        mean=float(d.mean()), seed_scene=d.tolist(),
                        seed_mean=d.mean(1).tolist(), scene_mean=d.mean(0).tolist())

    result = dict(summary=summary, differences=differences,
        difference_direction='existing variable-covariance relative5 minus comparison; PSNR/SSIM higher, LPIPS/MSE lower',
        scope='old-data mechanism diagnostic, observed and RGB groups reported separately',
        new_training_count=6, reused_results_sha256=reused, prediction_files=files,
        independent_primary_metrics=errors, selections=selected['rows'],
        lock_sha256=sha(OUT/'lock.json'), source_sha256=sha(Path(__file__)),
        decision=dict(diagnostic_only=True, fresh_confirmation=False, changes_prior_utility_decisions=False),
        scientific_review_pending=True, visual_review_pending=True, goal_complete=False,
        caveats=[
            'C=Q changes only PE descriptors; the common content still receives complete geometric information.',
            'Same nominal parameters and initial weights; fixing inputs intentionally restricts effective PE dependence.',
            'RGB geometry is jointly relearned, so this is a training intervention, not a fixed-geometry causal effect.',
            'Old five scenes and three paired seeds; previously used confirmation pixels are not accessed.',
            'No fresh RGB promotion and no reversal of the failed full confirmation criterion.',
            'No efficiency claim: upstream covariance arithmetic is still computed before the hook.'])
    write(OUT/'result.json', result)
    plot(result)
    lines = ['# 固定参考形状归因诊断（待实际审阅）', '',
        '保持投影中心、有效性、全部共同几何内容、模型参数和训练预算，只将 PE 中的参考投影协方差替换为固定目标 patch 协方差 Q。', '',
        '旧数据六次新拟合；给定几何与 RGB 预测几何各三个种子。全部六个模型选定后统一导出，原候选与已有控制按冻结身份复用。', '',
        '| 输入组 / 方法 | Base PSNR ↑ | SSIM ↑ | LPIPS ↓ | 外推 PSNR | 变焦 PSNR |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for fixed in ROLES:
        for role in role_order(fixed):
            d = summary[role]
            lines.append(f"| {role} | {d['base']['psnr']['mean']:.4f} | {d['base']['ssim']['mean']:.4f} | {d['base']['lpips']['mean']:.4f} | {d['extrapolate']['psnr']['mean']:.4f} | {d['target_zoom']['psnr']['mean']:.4f} |")
    lines += ['', '主要比较：原五维候选减固定形状对照。PSNR 正值有利于原候选；LPIPS、未裁剪 MSE 负值有利于原候选。', '',
        '| 输入组 / 条件 | ΔPSNR 均值 | 三种子 ΔPSNR | ΔLPIPS | Δ未裁剪 MSE |',
        '| --- | ---: | --- | ---: | ---: |']
    for fixed in ROLES:
        _, candidate = parent(fixed)
        for condition in CONDITIONS:
            d = differences[candidate][fixed][condition]
            seeds = ' / '.join(f'{x:+.4f}' for x in d['psnr']['seed_mean'])
            lines.append(f"| {fixed.split('_')[0]} / {condition} | {d['psnr']['mean']:+.4f} | {seeds} | {d['lpips']['mean']:+.5f} | {d['raw_mse']['mean']:+.7f} |")
    lines += ['', '全部条件、种子、场景、裁剪比例和对已有控制的配对差见 result.json。', '',
        '若固定形状接近或超过原候选，应削弱此图像任务的参考协方差专属收益解释；若原候选稳定更好，则支持变化的参考形状对 PE 的条件性贡献。',
        '这不证明物理几何的唯一因果作用，也不改变 R72 与新场景确认的既有用途判断。RGB 两组会分别学习几何，因此还包含训练适应的影响。',
        '这里只生成诊断结果，实际图表审阅与最终解释另行记录。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print('SHAPE_ATTRIBUTION_SUMMARIZED', flush=True)


def plot(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels = ['PRoPE', 'Distance', 'Direct', 'URoPE', 'Variable shape', 'Fixed C=Q']
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    for row, fixed in enumerate(ROLES):
        for ax, metric in zip(axes[row], ('psnr', 'lpips')):
            for i, role in enumerate(role_order(fixed)):
                values = result['summary'][role]['base'][metric]['seed_mean']
                ax.scatter([i]*3, values)
                ax.plot([i-.2, i+.2], [np.mean(values)]*2, color='black')
            ax.set_xticks(range(len(labels)), labels, rotation=25, ha='right')
            ax.set_title(f"{fixed.split('_')[0]} / base {metric}")
            ax.grid(axis='y', alpha=.2)
    fig.suptitle('Old-data attribution: dots = paired seeds; lines = means')
    fig.savefig(OUT/'comparison.png', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    for row, fixed in enumerate(ROLES):
        _, candidate = parent(fixed)
        for ax, metric in zip(axes[row], ('psnr', 'lpips')):
            for i, condition in enumerate(CONDITIONS):
                values = result['differences'][candidate][fixed][condition][metric]['seed_mean']
                ax.scatter([i]*3, values)
                ax.plot([i-.2, i+.2], [np.mean(values)]*2, color='black')
            ax.axhline(0, color='gray', linewidth=1)
            ax.set_xticks(range(3), CONDITIONS)
            ax.set_title(f"{fixed.split('_')[0]} / delta {metric}")
            ax.set_ylabel('Variable shape minus fixed C=Q')
            ax.grid(axis='y', alpha=.2)
    fig.suptitle('All prespecified conditions; PSNR positive / LPIPS negative favor variable shape')
    fig.savefig(OUT/'paired_conditions.png', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    for row, fixed in enumerate(ROLES):
        root, candidate = parent(fixed)
        for folder, role, label in ((root, candidate, 'Variable shape'), (OUT, fixed, 'Fixed C=Q')):
            histories = [read(folder/'runs'/role/str(seed)/'history.json') for seed in SEEDS]
            for ax, key in zip(axes[row], ('training', 'validation')):
                values = np.array([[entry['mse'] for entry in history[key]] for history in histories])
                epochs = [entry['epoch'] for entry in histories[0][key]]
                ax.plot(epochs, values.mean(0), label=label)
                ax.fill_between(epochs, values.min(0), values.max(0), alpha=.1)
                ax.set_yscale('log')
                ax.set_xlabel('epoch')
                ax.set_title(f"{fixed.split('_')[0]} / {key} MSE")
                ax.legend()
    fig.suptitle('Fixed 60-epoch budget; mean and seed range')
    fig.savefig(OUT/'learning_curves.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    summarize()
