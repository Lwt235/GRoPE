"""Evidence tables and reviewable figures; interpretation remains explicit."""
import json

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.gaussian_memory_setup import ROOT, sha256
from experiments.gaussian_memory_soft_data import OUT, BASE_CACHE, SEEDS, QUERIES, load_plan, write_json
from experiments.gaussian_memory_bound_report import num, pct, interval, comparison_table

LABELS = {'native': 'Native', 'local': 'L', 'content': 'C', 'coverage': 'A', 'bound': 'B',
          'soft': 'Soft', 'soft_center': 'S-center'}
COLORS = {'native': '#888888', 'local': '#517690', 'content': '#589b8c', 'coverage': '#9075b4',
          'bound': '#b0a095', 'soft': '#d88435', 'soft_center': '#b6a85e'}


def read(name): return json.loads((OUT / name).read_text())


def save_figure(fig, name, pdf=True):
    fig.savefig(OUT / 'figures' / f'{name}.png', dpi=160)
    if pdf: fig.savefig(OUT / 'figures' / f'{name}.pdf')
    plt.close(fig)


def plots(static, dynamic, intervention, retention):
    (OUT / 'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    g = static['cohorts']['primary51']['modes']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    for ax, mode in zip(axes[0], ('mixed', 'full', 'masked')):
        means = g[mode]['log_mse']['means']; keys = list(means)
        ax.bar([LABELS[k] for k in keys], [means[k] for k in keys], color=[COLORS[k] for k in keys])
        ax.set_title(mode + ' input'); ax.set_ylabel('Native depth log-MSE (lower better)'); ax.tick_params(axis='x', rotation=20)
    ax = axes[1, 0]; comparisons = g['mixed']['log_mse']['comparisons']
    names = [('content_minus_soft', 'C - Soft'), ('native_minus_soft', 'Native - Soft'),
             ('bound_minus_soft', 'B - Soft'), ('local_minus_soft', 'L - Soft')]
    if static['center_evaluated']: names.append(('soft_center_minus_soft', 'S-center - Soft'))
    for j, (name, label) in enumerate(names):
        r = comparisons[name]; lo, hi = r['paired_place_seed_95']
        ax.plot([lo, hi], [j, j], color='#517690', lw=2); ax.scatter(r['mean_gain'], j, color='#d88435', zorder=3)
    ax.axvline(0, color='#888888', lw=1); ax.set_yticks(range(len(names)), [label for _, label in names])
    ax.set_title('Paired place x seed 95% intervals'); ax.set_xlabel('Error reduction (positive better)')
    ax = axes[1, 1]; means = g['mixed']['centered_log_mse']['means']; keys = list(means)
    ax.bar([LABELS[k] for k in keys], [means[k] for k in keys], color=[COLORS[k] for k in keys])
    ax.set_title('Centered spatial error'); ax.set_ylabel('Mixed centered log-MSE'); ax.tick_params(axis='x', rotation=20)
    ax = axes[1, 2]; means = intervention['cohorts']['primary51']['mixed']['log_mse']['means']
    ax.bar(['Soft intact', 'Content\npermuted', 'Uniform\nprior'], [means[k] for k in ('soft', 'content_permuted', 'uniform_prior')],
           color=['#d88435', '#888888', '#589b8c'])
    ax.set_title('First fixed seed: dependence only'); ax.set_ylabel('Native depth log-MSE')
    fig.suptitle('51 previously used nontraining places; three matched seeds except interventions')
    save_figure(fig, 'static_and_binding')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True); times = np.arange(1, 5)
    for ax, kind in zip(axes[:2], ('content', 'soft')):
        for policy, label in [('static', 'Static'), ('corrective_clean', 'Corrective'), ('rebuild_clean', 'Prefix rebuild')]:
            ax.plot(times, [r['means'][f'{kind}_{policy}'] for r in dynamic['causal_primary51']['mixed']], '-o', label=label)
        ax.set_title(LABELS[kind] + ': fixed reader, three seeds'); ax.set_xlabel('Writes observed (query also changes)')
        ax.set_xticks(times); ax.set_ylabel('Native depth log-MSE'); ax.legend(fontsize=8)
    values = retention['cohorts']['primary51']['log_mse']['means']
    for kind in ('content', 'soft'):
        axes[2].plot([0, 1, 4], [values[f'{kind}_t{t}'] for t in (0, 1, 4)], '-o', label=LABELS[kind], color=COLORS[kind])
    axes[2].set_title('Early query 5 revisited: first seed, full RGB')
    axes[2].set_xlabel('Writes observed; retrospective'); axes[2].set_xticks([0, 1, 4]); axes[2].set_ylabel('Native depth log-MSE'); axes[2].legend()
    fig.suptitle('Compare policies at the same query; retrospective revisit is separate from causal performance')
    save_figure(fig, 'native_integration_and_retention')


def qualitative(static):
    fixed = ['transfer_00', 'confirm_00', 'fresh_00']; primary = static['cohorts']['primary51']
    gains = primary['modes']['mixed']['log_mse']['comparisons']['content_minus_soft']['place_gain']
    place = primary['places'][int(np.argmin(gains))]
    worst = next(m['id'] for m in load_plan()['catalog'] if m['place'] == place and not m['training'])
    chosen = list(dict.fromkeys(fixed + [worst]))
    write_json(OUT / 'figure_selection.json', {'fixed': fixed, 'posthoc_largest_soft_minus_content_place': place,
        'first_catalog_clip_in_that_place': worst, 'used_main_place_average': True, 'not_confirmation': True})
    for rid in chosen:
        fig, axes = plt.subplots(2, 8, figsize=(19, 5.7), constrained_layout=True)
        for row, condition in enumerate((0, 1)):
            q = 11; sample = torch.load(BASE_CACHE / rid / f'q{q}_c{condition}.pt', map_location='cpu', weights_only=False)
            target = sample['target'].numpy(); valid = sample['valid'].numpy(); scale = sample['metric_scale']; assert scale is not None
            vmax = float(np.quantile(target[valid], .98)); arrays = [sample['query_depth_native'].numpy() * scale]
            cases = [('static', f'{k}_{SEEDS[0]}') for k in ('content', 'bound', 'soft')]
            cases += [('integrated', f'{k}_{SEEDS[0]}_corrective_clean') for k in ('content', 'soft')]
            for phase, case in cases:
                record = read(f'evaluation/{phase}/{case}/{rid}.json')
                export = torch.load(record['prediction_path'], map_location='cpu', weights_only=False)
                index = export['sample_keys'].index([q, condition]); arrays.append(export['prediction_native'][index].numpy() * scale)
            axes[row, 0].imshow(sample['rgb'].numpy()); axes[row, 0].set_title('Full RGB' if condition == 0 else 'Masked RGB')
            axes[row, 1].imshow(np.where(valid, target, np.nan), vmin=0, vmax=vmax, cmap='viridis'); axes[row, 1].set_title('Sensor depth')
            for col, (name, depth) in enumerate(zip(('Native', 'C static', 'B static', 'Soft static', 'C Corrective', 'Soft Corrective'), arrays), 2):
                axes[row, col].imshow(depth, vmin=0, vmax=vmax, cmap='viridis'); axes[row, col].set_title(name)
            for ax in axes[row]: ax.axis('off')
        label = ('fixed example; also posthoc adverse place' if rid in fixed else 'posthoc adverse place') if rid == worst else 'fixed example'
        fig.suptitle(f'{rid}: {label}; query 11; seed {SEEDS[0]}; same depth color range per row')
        save_figure(fig, f'example_{rid}', pdf=False)
    return chosen


def main():
    static, dynamic, intervention, retention = (read(name) for name in (
        'static_analysis.json', 'integrated_analysis.json', 'intervention_analysis.json', 'retention_analysis.json'))
    condition, checks, training, costs = (read(name) for name in (
        'conditional_decision.json', 'independent_checks.json', 'training_summary.json', 'native_costs.json'))
    assert checks['complete']
    assert static['center_evaluated'] == dynamic['center_evaluated'] == condition['center_control_triggered']
    plots(static, dynamic, intervention, retention); selected = qualitative(static)
    decision_path = OUT / 'method_decision.json'; decision = read('method_decision.json') if decision_path.exists() else None
    lead = decision['lead'] if decision else '数据与检查已完成；方法取舍仍待结合主比较、动态用途、干预及案例审阅。'
    g = static['cohorts']['primary51']['modes']; primary = g['mixed']['log_mse']; dg = dynamic['cohorts']['primary51']
    lines = ['# 保留全局内容读取的高斯位置偏好', '', lead, '',
        '`soft_read1`，85 个已用片段，12 个训练片段；主评价为 51 非训练地点（63 片段），无新确认。', '',
        '## 1. 改动、假设与比较', '',
        '上一轮的硬投影先验在无支持处屏蔽历史信息，B 相对内容 C 未建立用途。本轮只将逐槽位置项改为 `-log(M)+log1p(M*w_ij)`，M=256；所有真实槽均可读，几何对应提供偏好，null 与 C 一样屏蔽。无覆盖时先验退回 C；均匀覆盖增加的是逐行常数。协方差表示区域范围，不当作位置预测不确定性。', '',
        '沿用冻结 VGGT 的第 12 个 global block、一个 width64/4-head cross-attention 和后续原生层、DPT。所有适配器为 137,472 参数，同初值、同样例序列、同 2,000 步、最终检查点；新增 Soft 三拟合，L/C/A/B 和 Native 按身份复用。条件 Center 若触发再新增三个匹配拟合。', '',
        '主要假设是硬可见性限制不适合作为历史内容相关性的边界。竞争解释包括预测几何误差、内容特征已包含关系、小训练集迁移和静态训练／动态状态不匹配。本轮固定 writer、数据、损失和公式，不扫描温度。初始化帧 0–3，写入 4/6/8/10，查询 5/7/9/11 不写回；已知相机及原有独立尺度规则不变。', '',
        '## 2. 固定记忆的原生输出', '',
        '| 方法 | 混合 log-MSE | 完整输入 | 遮挡输入 | 混合去均值空间误差 |', '|---|---:|---:|---:|---:|']
    for k, mean in primary['means'].items():
        lines.append(f"| {LABELS[k]} | {num(mean)} | {num(g['full']['log_mse']['means'][k])} | {num(g['masked']['log_mse']['means'][k])} | {num(g['mixed']['centered_log_mse']['means'][k])} |")
    lines += ['', '正差表示后者误差更低；地点与种子配对重采样 5,000 次。']
    comparison_table(lines, primary['comparisons'])
    lines += ['| 批次 | Soft 对 C 降幅 | Soft 对 Native 降幅 | Soft 对 B 降幅 |', '|---|---:|---:|---:|']
    for group in ('old12', 'used_summary20', 'used_writer19', 'training12'):
        r = static['cohorts'][group]['modes']['mixed']['log_mse']['comparisons']
        lines.append(f"| {group} | {pct(r['content_minus_soft']['relative_reduction_percent'])} | {pct(r['native_minus_soft']['relative_reduction_percent'])} | {pct(r['bound_minus_soft']['relative_reduction_percent'])} |")
    lines += ['', '完整／遮挡、隐藏／可见区域、均值偏差、空间误差、AbsRel 和米制 RMSE 的全部配对结果见 `static_analysis.json`。整体 log-MSE 改善不自动表示空间细节恢复。', '',
        '## 3. 动态状态与有限回访', '',
        'C 和 Soft 各三种子读取同一批已保存的 Corrective 与 Prefix 状态，此项不以静态获胜为前提。状态和 writer 均保持原样；Prefix 重建 256 槽，没有重新联合运行全部 RGB，因此不是严格上界。', '',
        '| 读取器／状态 | 混合 log-MSE | 完整输入 | 遮挡输入 | 混合空间误差 |', '|---|---:|---:|---:|---:|']
    for k in dg['mixed']['log_mse']['means']:
        lines.append(f"| {k} | {num(dg['mixed']['log_mse']['means'][k])} | {num(dg['full']['log_mse']['means'][k])} | {num(dg['masked']['log_mse']['means'][k])} | {num(dg['mixed']['centered_log_mse']['means'][k])} |")
    comparison_table(lines, dg['mixed']['log_mse']['comparisons'])
    lines += ['各时点的查询不同，只比较同一时点的策略才能隔离更新效果。覆盖增长本身不等于输出改善。', '',
        '有限回访固定首种子、query5、完整输入，使用初始、首次写后和末次写后的 Corrective 状态；这是事后回访，不并入因果成绩。', '']
    comparison_table(lines, retention['cohorts']['primary51']['log_mse']['comparisons'])
    if (OUT / 'information_accounting.json').exists():
        accounting = read('information_accounting.json'); assert accounting['complete']
        lines += ['**增量信息范围：** 对全部 85 片段、1,020 个查询条件的输入身份核对显示，前三个时点的新写入图像都已存在于当前 VGGT 短窗口；末次只有写入帧 4 在窗口之外。它们仍能通过另一种记忆表示影响网络，但新增记忆与原生输入的信息高度重叠。', '',
                  '| 查询 | VGGT 输入帧 | 已可写入的新帧 | 其中不在原生输入窗口的新帧 |', '|---|---|---|---|']
        for q in QUERIES:
            sets = accounting['by_query'][str(q)]['distinct_frame_sets']
            lines.append(f"| {q} | {sets['native_input_frames'][0]} | {sets['new_write_frames_available'][0]} | {sets['new_writes_outside_native_context'][0]} |")
        lines += ['', '这限制了当前动态结果对“积累原生窗口以外的历史信息”的解释范围，不推翻已观察到的 C/Soft 差异，也不证明重复图像无用。记忆更新后的分布变化仍是竞争解释；长期记忆用途需要另行设计。身份核对及事前诊断问题见 `information_accounting*.json`，未修改本轮配置。', '']
    lines += ['## 4. 依赖性与条件形状控制', '',
        '首个固定 Soft 模型在全 85 片段上分别置换内容槽、移除几何位置项。正的差值表示干预误差高于完整 Soft；相对降幅的分母是干预误差。它只检验依赖性，匹配训练的 C/Soft 才检验用途。', '']
    comparison_table(lines, intervention['cohorts']['primary51']['mixed']['log_mse']['comparisons'])
    check = intervention['trained_interface']
    lines += [f"固定训练样例的 BF16 后半网络中，联合置换的最大／平均输出差为 {check['joint_content_and_geometry_prior_max_abs']:.6g}/{check['joint_content_and_geometry_prior_mean_abs']:.6g}；只置换内容为 {check['content_only_max_abs']:.6g}/{check['content_only_mean_abs']:.6g}。CPU FP32 适配器接口测试单独通过；不宣称 BF16 完整网络逐值不变。", '']
    if condition['center_control_triggered']:
        lines += ['静态或 Corrective 满足事前的“主均值改善且至少两个种子改善”条件，已追加三个 Soft-center 匹配训练及 Static/Corrective 评价。球形方差沿用训练数据统计；它同时改变形状与投影足迹，不能单独证明各向异性必要。此触发只是诊断条件，不是方法晋升。', '']
    else:
        lines += ['静态与 Corrective 均未满足事前的均值和种子方向条件，未追加 Center。本轮不产生新的形状必要性结论；详见 `conditional_decision.json`。', '']
    lines += ['## 5. 实现、来源与成本', '',
        '三个 CPU 接口测试及 20 步 GPU 资源／梯度试跑通过。零初始化与原生输出一致，试跑不进入正式比较；新模型逐步核对父轮 C 的全部采样和初值。', '',
        f"从原始标签独立 NumPy float64 复算 {checks['exported_predictions_recomputed']:,} 份新预测，另核对 {checks['reused_parent_rows']:,} 行父轮评分原样复用。所有新模型、原状态、缓存和检查点有来源身份，无样本排除。最大数值差见 `independent_checks.json`。", '',
        f"新增正式训练 {training['fits']} 次、{training['steps']:,} 步，训练循环合计 {training['sum_worker_hours']:.3f} worker 小时，峰值已分配 {training['peak_allocated_GiB']:.3f} GiB。这是共享卡循环之和，不是墙钟或独占性能。", '',
        '固定种子交错计时的共享 GPU 原生/C/Soft 后半网络中位秒数：' + '; '.join(
            f"{r['id']} {r['timing']['native']['median_seconds']:.4f}/{r['timing']['content']['median_seconds']:.4f}/{r['timing']['soft']['median_seconds']:.4f}" for r in costs['rows']) + '。投影单独计时；不含前半网络、相机尺度校准和输入 I/O，不据此宣称完整在线速度提升。', '',
        '## 6. 方法取舍与后续', '']
    lines += [line for paragraph in decision['paragraphs'] for line in (paragraph, '')] if decision else ['方法取舍及图像审阅尚未完成，表格生成不等于方案通过验证。', '']
    lines += ['## 7. 图与证据入口', '', '![静态和干预](figures/static_and_binding.png)', '',
              '![动态和回访](figures/native_integration_and_retention.png)', '']
    for rid in selected: lines += [f'![案例 {rid}](figures/example_{rid}.png)', '']
    lines += ['前三个案例事前固定；最大 Soft−C 主地点的首个片段为明确标注的事后反例，若与固定案例重复则合并。相同一行使用共同深度色阶，案例不代替整体统计。完整来源、配对区间及分条件结果保存在同目录 JSON，旧正负结果仍保留。', '']
    (OUT / 'report.md').write_text('\n'.join(lines))
    sources = list((ROOT / 'experiments').glob('gaussian_memory_soft*.py')) + [ROOT / 'models/gaussian_memory_soft_read.py']
    write_json(OUT / 'report_manifest.json', {'report_sha256': sha256(OUT / 'report.md'),
        'decision_sha256': sha256(decision_path) if decision else None,
        'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        'figures': {p.name: sha256(p) for p in (OUT / 'figures').glob('*')},
        'conclusion_review_pending': decision is None, 'figure_review_pending': True,
        'phase_complete': False, 'continuous_research_goal_active': True})
    print(json.dumps({'report_written': str(OUT / 'report.md'), 'figure_review_pending': True}), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4); main()
