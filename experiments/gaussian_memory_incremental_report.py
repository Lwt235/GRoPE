"""Reproducible research tables, plots, and an evidence-limited Chinese report."""
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.gaussian_memory_setup import sha256
from experiments.gaussian_memory_incremental_data import OUT, CACHE, SEEDS, KINDS, load_plan, write_json
from experiments.gaussian_memory_incremental_states import STATE_OUT


LABELS = {'native': 'Native', 'local': 'L', 'content': 'C', 'geometry': 'G', 'combined': 'CG', 'center': 'CG-center'}
POLICIES = ('static_clean', 'average_clean', 'corrective_clean', 'rebuild_clean')
NAMES = ('Static', 'Average', 'Corrective', 'Prefix rebuild')


def read(name): return json.loads((OUT / name).read_text())
def num(x): return '—' if x is None else f'{x:.5f}'
def pct(x): return f'{x:+.2f}%'
def interval(r): return '[' + ', '.join(num(x) for x in r['paired_place_seed_95']) + ']'


def plots(a, b, state):
    folder = OUT / 'figures'; folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 160})
    g = a['cohorts']['primary51']['modes']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for ax, mode in zip(axes, ('mixed', 'full', 'masked')):
        means = g[mode]['log_mse']['means']; keys = list(means)
        ax.bar([LABELS[k] for k in keys], [means[k] for k in keys], color=['#888888' if k == 'native' else '#3085a1' if k != 'combined' else '#d88435' for k in keys])
        ax.set_title(mode + ' input'); ax.set_ylabel('Native depth log-MSE (lower is better)')
        ax.tick_params(axis='x', rotation=30)
    fig.suptitle('51 previously used evaluation places · three paired seeds')
    fig.savefig(folder / 'static_comparison.png'); fig.savefig(folder / 'static_comparison.pdf'); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True); times = np.arange(1, 5)
    for policy, name in zip(POLICIES, NAMES):
        s = state['cohorts']['primary51']['policies'][policy]
        axes[0, 0].plot(times, s['causal_query']['valid_patch_coverage'], '-o', label=name)
        axes[0, 1].plot(times, s['causal_query']['geometry_log_mse_common_support'], '-o', label=name)
        axes[1, 0].plot(times, s['fixed_early_query']['valid_patch_coverage'], '-o', label=name)
        axes[1, 1].plot(times, s['fixed_early_query']['geometry_log_mse_common_support'], '-o', label=name)
    titles = ('Causal query coverage', 'Causal query geometry error', 'Early query revisited: coverage', 'Early query revisited: geometry error')
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title); ax.set_xlabel('Number of new writable views'); ax.set_xticks(times)
    axes[0, 0].set_ylabel('Valid patch coverage'); axes[1, 0].set_ylabel('Valid patch coverage')
    axes[0, 1].set_ylabel('Log-MSE on common supported patches'); axes[1, 1].set_ylabel('Log-MSE on common supported patches')
    axes[0, 0].legend(fontsize=9)
    fig.suptitle('State diagnostics · same observations and 256 slots; geometry is not native output')
    fig.savefig(folder / 'state_trajectories.png'); fig.savefig(folder / 'state_trajectories.pdf'); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for ax, metric in zip(axes, ('log_mse', 'centered_log_mse')):
        means = b['cohorts']['primary51']['mixed'][metric]['means']
        x = np.arange(2); width = .24
        for j, policy in enumerate(('static', 'corrective_clean', 'rebuild_clean')):
            vals = [means[f'{kind}_{policy}'] for kind in ('content', 'combined')]
            ax.bar(x + (j - 1) * width, vals, width, label=('Static', 'Corrective', 'Prefix rebuild')[j])
        ax.set_xticks(x, ['C', 'CG']); ax.set_title(metric); ax.set_ylabel('Native output error (lower is better)')
    axes[0].legend(); fig.suptitle('Fixed internal readers · three seeds · previously used places')
    fig.savefig(folder / 'native_integration.png'); fig.savefig(folder / 'native_integration.pdf'); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    for ax, kind in zip(axes[:2], ('content', 'combined')):
        for policy, label in [('static', 'Static'), ('corrective_clean', 'Corrective'), ('rebuild_clean', 'Prefix rebuild')]:
            vals = [r['means'][kind + '_' + policy] for r in b['causal_primary51']['mixed']]
            ax.plot(times, vals, '-o', label=label)
        ax.set_title(LABELS[kind] + ': causal native outputs (three seeds)')
        ax.set_xlabel('New writable views; query changes at each time'); ax.set_xticks(times)
    for policy, label in zip(POLICIES, NAMES):
        axes[2].plot(range(5), b['retention_curves_first_seed'][policy], '-o', label=label)
    axes[2].set_title('CG: fixed early query revisited (first seed)')
    axes[2].set_xlabel('New writable views; query 5 always fixed'); axes[2].set_xticks(range(5))
    for ax in axes: ax.set_ylabel('Native depth log-MSE'); ax.legend(fontsize=8)
    fig.savefig(folder / 'native_time_and_retention.png'); fig.savefig(folder / 'native_time_and_retention.pdf'); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for ax, amp in zip(axes.flat, ('-0.2', '0.2', '-0.05', '0.05')):
        for strategy in ('static', 'average', 'corrective'):
            ax.plot(times, b['error_causal_curves_first_seed'][amp][strategy], '-o', label=strategy)
        ax.set_title('Initial local shift ' + amp + ' scene units'); ax.set_xticks(times)
        ax.set_xlabel('New writable views; same current query across strategies'); ax.set_ylabel('Native depth log-MSE')
    axes[0, 0].legend(fontsize=9); fig.suptitle('Controlled-error native outputs · fixed CG first seed · full input')
    fig.savefig(folder / 'native_error_time_curves.png'); fig.savefig(folder / 'native_error_time_curves.pdf'); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for amp, style in [('-0.2', '-'), ('0.2', '--')]:
        suffix = 'minus20' if amp[0] == '-' else 'plus20'
        for strategy, color in [('static', '#888888'), ('average', '#4588af'), ('corrective', '#d88435')]:
            data = state['cohorts']['primary51']['policies'][f'{strategy}_{suffix}']
            axes[0].plot(times, data['causal_query']['paired_clean_geometry_gap_common_support'], style, color=color, marker='o', label=strategy + ' ' + amp)
    errors = b['error_probe_first_seed_full']; x = np.arange(4); width = .25
    for j, strategy in enumerate(('static', 'average', 'corrective')):
        axes[1].bar(x + (j - 1) * width, [errors[k]['means'][strategy] for k in ('-0.2', '0.2', '-0.05', '0.05')], width, label=strategy)
    axes[0].set_title('Geometric deviation from paired clean trajectory'); axes[0].set_xlabel('New writable views'); axes[0].set_ylabel('Common-support squared log-depth gap')
    axes[0].legend(fontsize=8, ncol=2); axes[0].set_xticks(times)
    axes[1].set_xticks(x, ['−20%', '+20%', '−5%', '+5%']); axes[1].set_xlabel('Perturbation in initial scene units')
    axes[1].set_title('Native depth error: fixed CG first seed'); axes[1].set_ylabel('Log-MSE'); axes[1].legend(fontsize=9)
    fig.savefig(folder / 'error_recovery.png'); fig.savefig(folder / 'error_recovery.pdf'); plt.close(fig)


def qualitative(analysis=None):
    chosen = ['transfer_00', 'confirm_00', 'fresh_00']; worst = None
    (OUT / 'figures').mkdir(parents=True, exist_ok=True)
    if analysis is not None:
        primary = analysis['cohorts']['primary51']
        gains = primary['modes']['mixed']['log_mse']['comparisons']['local_minus_combined']['place_gain']
        place = primary['places'][int(np.argmin(gains))]
        worst = sorted(m['id'] for m in load_plan()['catalog'] if m['place'] == place and not m['training'])[0]
        chosen = list(dict.fromkeys(chosen + [worst]))
        write_json(OUT / 'figure_selection.json', {'fixed': ['transfer_00', 'confirm_00', 'fresh_00'],
                   'posthoc_largest_CG_minus_L_place': place, 'first_clip_in_that_place': worst,
                   'selection_uses_main_place_average': True, 'not_confirmation': True})
    for rid in chosen:
        fig, axes = plt.subplots(2, 6, figsize=(15, 5.5), constrained_layout=True)
        for row, condition in enumerate((0, 1)):
            q = 11; sample = torch.load(CACHE / rid / f'q{q}_c{condition}.pt', map_location='cpu', weights_only=False)
            target = sample['target'].numpy(); valid = sample['valid'].numpy()
            vmax = float(np.quantile(target[valid], .98)); scale = sample['metric_scale']
            arrays = [sample['query_depth_native'].numpy() * scale]
            for kind in ('local', 'content', 'combined'):
                record = read(f'evaluation/static/{kind}_{SEEDS[0]}/{rid}.json')
                export = torch.load(record['prediction_path'], map_location='cpu', weights_only=False)
                index = export['sample_keys'].index([q, condition])
                arrays.append(export['prediction_native'][index].numpy() * scale)
            axes[row, 0].imshow(sample['rgb'].numpy()); axes[row, 0].set_title('Full RGB' if condition == 0 else 'Masked RGB')
            axes[row, 1].imshow(np.where(valid, target, np.nan), vmin=0, vmax=vmax, cmap='viridis'); axes[row, 1].set_title('Sensor depth')
            for col, (name, depth) in enumerate(zip(('Native', 'L', 'C', 'CG'), arrays), 2):
                axes[row, col].imshow(depth, vmin=0, vmax=vmax, cmap='viridis'); axes[row, col].set_title(name)
            for ax in axes[row]: ax.axis('off')
        label = 'posthoc adverse place' if rid == worst else 'fixed example'
        fig.suptitle(f'{rid} · {label} · query 11 · seed {SEEDS[0]} · same depth color range per row')
        fig.savefig(OUT / 'figures' / f'example_{rid}.png'); plt.close(fig)


def main():
    assert read('evaluation_pipeline_complete.json')['complete']
    a, b, state = read('static_analysis.json'), read('integrated_analysis.json'), read('state_analysis.json')
    checks, cost, nativecost = read('independent_checks.json'), read('state_costs.json'), read('native_costs.json')
    binding = read('content_binding_check.json')
    calibration = read('calibration_dispersion.json')
    fit_kinds = list(KINDS) + (['center'] if a['center_evaluated'] else [])
    fits = {f'{kind}_{seed}': read(f'{kind}_{seed}/status.json') for kind in fit_kinds for seed in SEEDS}
    assert all(r['complete'] and r['steps'] == 2000 for r in fits.values())
    training_cost = {'fits': len(fits), 'steps': sum(r['steps'] for r in fits.values()),
                     'sum_recorded_fit_seconds': sum(r['seconds_this_invocation'] for r in fits.values()),
                     'peak_allocated_GiB': max(r['peak_GiB'] for r in fits.values()),
                     'timing_scope': 'sum of recorded fit loops including periodic checkpoints; overlapping workers, not elapsed wall time',
                     'cases': fits}
    write_json(OUT / 'training_summary.json', training_cost)
    plots(a, b, state); qualitative(a)
    g = a['cohorts']['primary51']['modes']; primary = g['mixed']['log_mse']
    pe = primary['comparisons']['content_minus_combined']; local = primary['comparisons']['local_minus_combined']
    lines = ['# 增量高斯记忆与 VGGT 内部投影 PE：完整开发验证', '',
             '实验目录：`incremental_vggt1`。这是已用数据上的匹配开发实验，不是新地点确认。', '',
             f"固定记忆时，CG 相对 C 的原生深度 log-MSE 降幅为 **{pct(pe['relative_reduction_percent'])}**，相对 L 为 **{pct(local['relative_reduction_percent'])}**。正数表示误差下降。以下同时报告种子、地点不确定性、完整输入和空间细节代价，不以混合均值单独晋升方案。", '',
             '本轮真正完成的推进是：把高斯几何编码接入冻结 VGGT 的第 12 个 global block，并让它通过后续原生层和 DPT 影响输出；另用同一批因果直接观测检验 256 槽记忆的增量更新。没有学习几何 writer，没有未知相机或长期 SLAM 的结论。', '',
             '## 1. 科学问题与固定比较', '',
             '- A：历史内容与投影几何 PE 是否超过相同容量、相同当前短窗口的适配？竞争解释包括普通二维适配、当前局部内容和整体深度偏差校正。',
             '- B：有限历史权重和局部重建是否吸收新区域、撤销初始错误？竞争解释包括更多观测、遗忘困难区域以及读出器掩盖错误。',
             '- L/C/G/CG 每组 3 个种子、每种子 2,000 步，158,976 个参数；初值和每步样例完全配对。主干和原生 depth head 冻结，后续层保留梯度。',
             '- 原生 VGGT 是额外绝对参照；主对照 L 同样获得 3 个最近可写历史帧和当前帧。L/C 使用相同新增 10 维二维位置描述，G/CG 使用 10 维投影几何。所有组保留原生 RoPE。',
             '- 编码为 `p(u)=[Σ_j w_j(u)·(μ_j^cam, vech(Σ_j^cam)), Σ_j w_j(u)]`；权重采用前到后 alpha 合成。原生 Q/K 归一化与 RoPE 之后使用 `q′=q+A p(u), k′=k+B p(u)`，再接一个 width64 的内容 cross-attention 残差。形状协方差描述区域范围，不代表估计不确定性。',
             '- 85 个已用片段均评价；训练仅原 12 个片段。主结果按 51 个非训练地点等权（63 片段），同时分开旧 12/已用 20/已用 19 地点。不能把片段或 patch 当独立地点。',
             '- 初始化 0–3；写入 4/6/8/10；查询 5/7/9/11 永不写回。遮挡先施加到查询 RGB。尺度来自无记忆分支的相机基线，所有组共享，无真值尺度对齐；本轮 85 片段均可校准。', '',
             f"尺度可计算不代表准确：完整查询窗口的相机基线尺度比率 max/min 中位数为 {calibration['groups']['query_full']['max_min_ratio_quantiles'][0]:.2f}，90% 分位数为 {calibration['groups']['query_full']['max_min_ratio_quantiles'][1]:.2f}。该离散度也会受短基线影响；本轮没有据此筛样本或改尺度。匹配方法共享尺度，端到端方法仍须承担这部分误差；该诊断仅提示，不能把总误差直接归因到更新器或 PE 某一部件。", '',
             '**结构边界：** 几何 PE 与内容 cross-attention 是并行通路。内容槽单独置换不会改变读出，本版没有把每个高斯的位置显式绑定到其内容 K/V。因此即使 CG 有效，也不能宣称已经证明逐槽空间寻址。',
             f"该边界有代数依据及已学习权重的数值验证：非零内容残差平均绝对幅度为 {binding['content_residual_mean_abs']:.5f}，内容置换的最大差为 {binding['content_permutation_max_abs_difference']:.3g}，内容置零的平均差为 {binding['zero_content_mean_abs_difference']:.5f}。这证明分支读取的是无序内容集合，不是本身已经证明任务收益。", '',
             '## 2. 固定记忆：VGGT 原生输出', '',
             '| 方法 | 混合 log-MSE | 完整输入 | 遮挡输入 | 混合去均值空间误差 |',
             '|---|---:|---:|---:|---:|']
    for k, mean in primary['means'].items():
        lines.append(f"| {LABELS[k]} | {num(mean)} | {num(g['full']['log_mse']['means'][k])} | {num(g['masked']['log_mse']['means'][k])} | {num(g['mixed']['centered_log_mse']['means'][k])} |")
    lines += ['', '| 比较（前者误差 − 后者误差） | 相对降幅 | 三种子差 | 地点×种子配对 95% 区间 | 改善地点 |', '|---|---:|---|---|---:|']
    for key, r in primary['comparisons'].items():
        lines.append(f"| {key} | {pct(r['relative_reduction_percent'])} | {', '.join(num(x) for x in r['paired_seed_gain'])} | {interval(r)} | {r['places_better']}/{r['places']} |")
    lines += ['', '区间使用 5,000 次地点与种子配对重采样；三个种子和已用地点仍限制了不确定性的解释。完整输入、隐藏区、可见区、AbsRel、米制 RMSE、均值偏差和训练地点结果完整保存于 `static_analysis.json`。', '',
              '| 地点批次 | CG 相对 C 降幅 | CG 相对 L 降幅 |', '|---|---:|---:|']
    for name in ('old12', 'used_summary20', 'used_writer19', 'training12'):
        comparisons = a['cohorts'][name]['modes']['mixed']['log_mse']['comparisons']
        lines.append(f"| {name} | {pct(comparisons['content_minus_combined']['relative_reduction_percent'])} | {pct(comparisons['local_minus_combined']['relative_reduction_percent'])} |")
    if a['center_evaluated']:
        r = primary['comparisons']['center_minus_combined']
        lines += ['', f"按预定条件追加了三组 CG-center 控制。CG 相对 Center 降幅 {pct(r['relative_reduction_percent'])}，三种子差 {r['paired_seed_gain']}，配对区间 {interval(r)}。此控制同时改变形状描述与投影足迹，不能单独归因于各向异性。"]
    else:
        lines += ['', 'CG 相对 C 未满足“均值改善且至少两个种子改善”的诊断触发条件，因此未追加 CG-center；本轮没有新的高斯形状必要性结论。']
    lines += ['', '## 3. 固定读出：新增观测是否有用', '',
              '| 固定读取器／状态 | 混合 log-MSE | 完整输入 | 遮挡输入 | 混合空间误差 |', '|---|---:|---:|---:|---:|']
    ig = b['cohorts']['primary51']
    for kind in ('content', 'combined'):
        for policy in ('static', 'corrective_clean', 'rebuild_clean'):
            key = kind + '_' + policy
            lines.append(f"| {key} | {num(ig['mixed']['log_mse']['means'][key])} | {num(ig['full']['log_mse']['means'][key])} | {num(ig['masked']['log_mse']['means'][key])} | {num(ig['mixed']['centered_log_mse']['means'][key])} |")
    lines += ['', '| 关键比较 | 相对降幅 | 地点×种子配对区间 |', '|---|---:|---|']
    for key, r in ig['mixed']['log_mse']['comparisons'].items():
        lines.append(f"| {key} | {pct(r['relative_reduction_percent'])} | {interval(r)} |")
    lines += ['', 'Prefix rebuild 使用相同的初始化与新观测，直接重建 256 槽，不重新联合运行全部图像。它是离线参照，并非严格上界。几何与原生输出分别判断，输出改善本身不证明状态纠错。', '',
              '## 4. 状态几何、覆盖与受控错误', '',
              '| 干净状态 | 末次因果查询覆盖 | 末次共同支持几何误差 | 平均局部重建次数／片段 | 容量拒收次数／片段 |', '|---|---:|---:|---:|---:|']
    policies = state['cohorts']['primary51']['policies']
    for k in POLICIES:
        s = policies[k]; e = s['events_mean_per_clip']
        lines.append(f"| {k} | {s['causal_query']['valid_patch_coverage'][-1]:.2%} | {num(s['causal_query']['geometry_log_mse_common_support'][-1])} | {e['local_rebuilds']:.2f} | {e['capacity_dropped']:.2f} |")
    common = policies['static_clean']['causal_query']['common_support_fraction_of_initial']
    counts = policies['static_clean']['valid_place_counts']['causal_query']['geometry_log_mse_common_support']
    lines += ['', f'四次因果查询的共同支持保留比例为 {common}，有可评分共同支持的地点数为 {counts}/51。几何误差在所有对照共同覆盖处计算，并单列覆盖；原先把无覆盖深度钳为近零的评分原样保留作排错记录，不作为几何准确率。该修正不修改状态、模型或原生输出分数。',
              '各时点查询视角不同，不能把时间曲线的升降直接解释为记忆变好或变坏，应比较同一时点的配对策略；固定早期视角的曲线另列。共同支持掩码随状态而变，因此固定早期视角的几何曲线也须配合覆盖解释；原生早期查询的传感器评分掩码则固定不变。容量拒收记录是候选处理次数，不等于独立新区域数量。', '',
              '| 初始局部偏移 | Static 原生误差 | Average 原生误差 | Corrective 原生误差 | Corrective 相对 Average 降幅 | 末次几何恢复差：Static / Average / Corrective |', '|---|---:|---:|---:|---:|---|']
    for amp, suffix in [('-0.2', 'minus20'), ('0.2', 'plus20'), ('-0.05', 'minus05'), ('0.05', 'plus05')]:
        r = b['error_probe_first_seed_full'][amp]; gap = [policies[f'{k}_{suffix}']['causal_query']['paired_clean_geometry_gap_common_support'][-1] for k in ('static', 'average', 'corrective')]
        lines.append(f"| {amp} 初始单位 | {num(r['means']['static'])} | {num(r['means']['average'])} | {num(r['means']['corrective'])} | {pct(r['average_minus_corrective']['relative_reduction_percent'])} | {' / '.join(num(x) for x in gap)} |")
    lines += ['', '| 偏移 | 末次共同支持处几何真值误差：Static / Average / Corrective | Corrective 相对 Average 的误差差区间 |', '|---|---|---|']
    for amp, r in read('state_last_query_accuracy.json').items():
        lines.append(f"| {amp} | {' / '.join(num(r['means'][k]) for k in ('static', 'average', 'corrective'))} | {interval(r['average_minus_corrective'])} |")
    lines += ['', '扰动覆盖固定的 52/256 槽，负号朝向首相机、正号远离；不能挑选有利方向。原生扰动探针仅固定第一个 CG 种子与完整输入，属于机制诊断。几何恢复差是与同策略干净轨迹的偏离，降低不等于更接近真值。',
              '重建/合并来源已从各步事件复原并核对，查询帧未进入来源集合。52 个初始受扰槽的 ancestry 标签都会被保留，局部重建也保留原 ancestry；因此不能把标签计数当作“没有删除或遗忘”的证据。固定受扰区域覆盖和早期查询复测承担这项判断。具体事件与来源见 `perturbed_lineage_diagnostics.json`。',
              '主评价 63 片段中，末次写入后 54 个可能回访受扰区域、9 个没有；回访由初始几何、相机和无记忆直接观测判定，不是真值可见性。两组末次原生输出和固定受扰区域指标保存在 `integrated_analysis.json`，局部覆盖保存于 `revisit_diagnostics.json`。', '',
              '| 早期只读查询的事后回访 | 首次写后误差 | 末次写后误差 |', '|---|---:|---:|']
    for policy, r in b['retrospective_retention'].items():
        lines.append(f"| {policy} | {num(r['after_first_write'])} | {num(r['after_last_write'])} |")
    lines += ['', '这是固定查询 5 在较晚记忆上的回访，不能混入当时的因果预测成绩；查询图像始终不写回。', '',
              '## 5. 资源与实现校验', '',
              f"- 正式训练 {training_cost['fits']} 次、共 {training_cost['steps']:,} 步；各拟合记录的训练循环耗时合计 {training_cost['sum_recorded_fit_seconds'] / 3600:.3f} 小时（包含周期检查点写入，是并行工作时长之和，不是墙钟时长），单进程峰值已分配显存 {training_cost['peak_allocated_GiB']:.3f} GiB，共享 worker 上限 12 GiB。来源见 `training_summary.json`。",
              '- 状态 256 槽：部署字段 53,248 字节，另有 65,536 字节的诊断 lineage；四帧最近观测缓存 232,000 字节。需把辅助缓存计入在线总内存。',
              '- 三个固定片段的 CPU4 四次写入耗时（秒）：' + '; '.join(f"{r['id']} corrective={r['corrective_four_writes_s']:.3f}, prefix={r['rebuild_four_writes_s']:.3f}" for r in cost['rows']) + '。短前缀下批量重建更快，不能声称当前增量融合有速度优势。',
              '- 共享 GPU 上，缓存第 11 层之后到 DPT 的读出中位耗时（秒）：' + '; '.join(f"{r['id']} native={r['timing']['native']['median_seconds']:.4f}, CG={r['timing']['combined']['median_seconds']:.4f}" for r in nativecost['rows']) + '。此测量不含前半网络、RGB 直接观测分支和缓存 I/O，不是完整在线速度或独占性能基准。',
              '- 本版每个查询窗口的统一尺度还依赖无记忆相机预测。因此部署成本需要计入原始相机分支以及插入适配后的后续网络，不能把缓存后半网络时延当作整套在线系统时延。',
              f"- 零适配与原生输出最大差为 {checks['interface_checks']['zero_adapter_max_abs']}，缓存重放最大差为 {checks['interface_checks']['replayed_native_max_abs']}。已独立用 NumPy float64 重算 {checks['exported_predictions_recomputed']} 份导出原生预测评分，并核对三种子的配对初值和每一步采样。",
              '- Alpha 遮挡顺序、零覆盖、坐标/槽置换、二阶矩合并及只读帧拒写等五项有意义的测试通过。适配器梯度、Q/K logits 与输出变化有固定样例记录。',
              '- 普通修复记录：旧 torch 的 argsort 参数签名；原生集成评价前统一了缓存和动态 PE 的截断顺序。其中截断顺序差异仅影响 340 个查询窗口中的 1 个、9 个标量（最大 0.006681），未重训或选择检查点。', '',
              '## 6. 可保留的结论与下一步边界', '',
              '1. 接口和训练可行性已经单独验证；方法用途以第 2 节的强局部对照、历史内容对照和分条件结果判断。不能用训练损失下降替代非训练地点收益。',
              '2. 增量覆盖、几何恢复、原生输出用途分别报告。只有三者相符且旧区域保持合理时，才能保留该条件下的纠偏解释；不要求所有辅助数值全胜，也不将混合结果统一描述为成功。',
              '3. 投影几何通路是否保留，由 CG−C/G−L 判断；高斯形状是否必要只能由本接口的 Center 控制判断。外部读取器的历史正面证据不能迁移为本接口的证据。',
              '4. 本版内容尚未与每槽几何显式绑定，256 槽的状态更新仍可能遗漏新区域。下一轮应围绕实际暴露的主要缺口提出一个可区分解释的修改；本轮不自动扫层、扩大模块或把已用地点重标为新确认。', '',
              '## 7. 原始证据与可视化', '',
              '配置、缓存身份和尺度：`plan.json`、`training_statistics.json`、外部缓存各 `identity.json`；训练日志和最终检查点在各方法／种子目录。',
              '汇总：`static_analysis.json`、`integrated_analysis.json`、`state_analysis.json`、`independent_checks.json`。每帧预测与评分的 SHA256 保存在 `evaluation/`。历史负结果未覆盖。', '',
              '![固定记忆](figures/static_comparison.png)', '', '![增量状态](figures/state_trajectories.png)', '',
              '![原生集成](figures/native_integration.png)', '', '![受控错误](figures/error_recovery.png)', '',
              '![原生因果与固定早期回访](figures/native_time_and_retention.png)', '',
              '![原生受控错误时间曲线](figures/native_error_time_curves.png)', '',
              '固定案例：`figures/example_transfer_00.png`、`example_confirm_00.png`、`example_fresh_00.png`。案例不是整体统计的替代。', '']
    selection = read('figure_selection.json')
    adverse = selection['first_clip_in_that_place']
    lines += [f"另附事后按主地点平均退步选择的反例：{selection['posthoc_largest_CG_minus_L_place']} 的首个片段 {adverse}；它不是独立确认或事先固定案例。", '',
              f'![事后反例](figures/example_{adverse}.png)', '']
    decision_path = OUT / 'method_decision.json'
    if decision_path.exists():
        decision = read('method_decision.json')
        lines[4:4] = [decision['lead'], '']
        start = lines.index('## 6. 可保留的结论与下一步边界') + 1
        end = lines.index('## 7. 原始证据与可视化')
        lines[start:end] = [''] + [line for paragraph in decision['paragraphs'] for line in (paragraph, '')]
    (OUT / 'report.md').write_text('\n'.join(lines))
    sources = list(Path(__file__).parent.glob('gaussian_memory_incremental*.py'))
    sources += [Path(__file__).resolve().parents[1] / 'models' / name for name in ('gaussian_memory_incremental.py', 'gaussian_memory_vggt_adapter.py')]
    write_json(OUT / 'report_manifest.json', {'report_sha256': sha256(OUT / 'report.md'),
        'decision_sha256': sha256(decision_path) if decision_path.exists() else None,
        'source_sha256': {str(p.relative_to(Path(__file__).resolve().parents[1])): sha256(p) for p in sources},
        'figures': {p.name: sha256(p) for p in (OUT / 'figures').glob('*')},
        'figure_review_pending': True, 'goal_complete': False})
    print(json.dumps({'report_written': str(OUT / 'report.md'), 'figures_pending_visual_review': True}), flush=True)


if __name__ == '__main__': main()
