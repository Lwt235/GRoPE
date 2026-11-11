"""Evidence tables and compact scientific figures for the fixed bound-read study."""
import json

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.gaussian_memory_setup import ROOT, sha256
from experiments.gaussian_memory_bound_data import OUT, BASE_OUT, BASE_CACHE, SEEDS, load_plan, write_json

LABELS = {'native': 'Native', 'local': 'L', 'content': 'C', 'coverage': 'A', 'bound': 'B', 'center': 'B-center'}
COLORS = {'native': '#888888', 'local': '#517690', 'content': '#589b8c', 'coverage': '#9075b4', 'bound': '#d88435', 'center': '#b6a85e'}


def read(name): return json.loads((OUT / name).read_text())
def num(x): return '—' if x is None else f'{x:.6f}'
def pct(x): return f'{x:+.2f}%'
def interval(r): return '[' + ', '.join(num(x) for x in r['paired_place_seed_95']) + ']'


def save_figure(fig, name):
    fig.savefig(OUT / 'figures' / f'{name}.png', dpi=160)
    fig.savefig(OUT / 'figures' / f'{name}.pdf')
    plt.close(fig)


def plots(a, b, intervention, error):
    (OUT / 'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    g = a['cohorts']['primary51']['modes']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    for ax, mode in zip(axes[0], ('mixed', 'full', 'masked')):
        means = g[mode]['log_mse']['means']; keys = list(means)
        ax.bar([LABELS[k] for k in keys], [means[k] for k in keys], color=[COLORS[k] for k in keys])
        ax.set_title(mode + ' input'); ax.set_ylabel('Native depth log-MSE (lower better)')
        ax.tick_params(axis='x', rotation=20)
    ax = axes[1, 0]; metrics = g['mixed']['log_mse']['comparisons']
    names = ('coverage_minus_bound', 'local_minus_bound', 'content_minus_bound', 'content_minus_coverage')
    for j, name in enumerate(names):
        r = metrics[name]; lo, hi = r['paired_place_seed_95']
        ax.plot([lo, hi], [j, j], color='#517690', lw=2)
        ax.scatter(r['mean_gain'], j, color='#d88435', zorder=3)
    ax.axvline(0, color='#888888', lw=1); ax.set_yticks(range(len(names)), ['A - B', 'L - B', 'C - B', 'C - A'])
    ax.set_title('Paired place × seed 95% intervals'); ax.set_xlabel('Error reduction (positive better)')
    ax = axes[1, 1]; keys = list(g['mixed']['centered_log_mse']['means'])
    ax.bar([LABELS[k] for k in keys], [g['mixed']['centered_log_mse']['means'][k] for k in keys], color=[COLORS[k] for k in keys])
    ax.set_title('Centered spatial error, mixed inputs'); ax.set_ylabel('Centered log-MSE'); ax.tick_params(axis='x', rotation=20)
    ax = axes[1, 2]; values = intervention['cohorts']['primary51']['mixed']['log_mse']['means']
    ax.bar(['B intact', 'Content\npermuted', 'Coverage\nprior'], [values[k] for k in ('bound', 'content_permuted', 'coverage_prior')],
           color=['#d88435', '#888888', '#9075b4'])
    ax.set_title('Fixed first-seed interventions'); ax.set_ylabel('Log-MSE; dependence only')
    fig.suptitle('51 previously used nontraining places · three matched seeds except interventions')
    save_figure(fig, 'static_and_binding')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    times = np.arange(1, 5)
    for ax, kind in zip(axes[:2], ('coverage', 'bound')):
        for policy, label in [('static', 'Static'), ('corrective_clean', 'Corrective'), ('rebuild_clean', 'Prefix rebuild')]:
            ax.plot(times, [r['means'][f'{kind}_{policy}'] for r in b['causal_primary51']['mixed']], '-o', label=label)
        ax.set_title(LABELS[kind] + ': fixed reader, three seeds')
        ax.set_xlabel('New writable views (query changes)'); ax.set_xticks(times); ax.set_ylabel('Native depth log-MSE'); ax.legend(fontsize=8)
    states = json.loads((BASE_OUT / 'state_analysis.json').read_text())['cohorts']['primary51']['policies']
    for policy, label in [('static_clean', 'Static'), ('corrective_clean', 'Corrective'), ('rebuild_clean', 'Prefix rebuild')]:
        axes[2].plot(times, states[policy]['causal_query']['valid_patch_coverage'], '-o', label=label)
    axes[2].set_title('Same stored states: coverage'); axes[2].set_ylabel('Valid patch coverage'); axes[2].set_xticks(times)
    axes[2].set_xlabel('New writable views (query changes)'); axes[2].legend(fontsize=8)
    fig.suptitle('Paired policy comparisons at each time; time alone does not isolate memory quality')
    save_figure(fig, 'native_integration')

    if error['triggered']:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
        amps = ('-0.2', '0.2', '-0.05', '0.05'); x = np.arange(4); width = .25
        for j, strategy in enumerate(('static', 'average', 'corrective')):
            axes[0].bar(x + (j - 1) * width, [error['amplitudes'][a]['means'][strategy] for a in amps], width, label=strategy)
        axes[0].set_xticks(x, ['−20%', '+20%', '−5%', '+5%']); axes[0].set_title('Controlled initial geometry shift')
        axes[0].set_ylabel('Native log-MSE'); axes[0].legend(fontsize=8)
        for policy, values in error['retention_curves'].items():
            axes[1].plot(range(5), values, '-o', label=policy.replace('_clean', ''))
        axes[1].set_title('Early query 5 revisited retrospectively'); axes[1].set_xlabel('New writable views')
        axes[1].set_ylabel('Native log-MSE'); axes[1].set_xticks(range(5)); axes[1].legend(fontsize=8)
        fig.suptitle('First B seed only · full input · mechanism diagnostics')
        save_figure(fig, 'conditional_error_and_retention')


def qualitative(a):
    fixed = ['transfer_00', 'confirm_00', 'fresh_00']; primary = a['cohorts']['primary51']
    gains = primary['modes']['mixed']['log_mse']['comparisons']['local_minus_bound']['place_gain']
    place = primary['places'][int(np.argmin(gains))]
    # The first catalog clip is fixed by the protocol, not selected by its image.
    worst = next(m['id'] for m in load_plan()['catalog'] if m['place'] == place and not m['training'])
    chosen = list(dict.fromkeys(fixed + [worst]))
    write_json(OUT / 'figure_selection.json', {'fixed': fixed, 'posthoc_largest_B_minus_L_place': place,
               'first_catalog_clip_in_that_place': worst, 'used_main_place_average': True, 'not_confirmation': True})
    for rid in chosen:
        fig, axes = plt.subplots(2, 7, figsize=(17, 5.5), constrained_layout=True)
        for row, condition in enumerate((0, 1)):
            q = 11; sample = torch.load(BASE_CACHE / rid / f'q{q}_c{condition}.pt', map_location='cpu', weights_only=False)
            target = sample['target'].numpy(); valid = sample['valid'].numpy(); scale = sample['metric_scale']
            assert scale is not None
            vmax = float(np.quantile(target[valid], .98))
            arrays = [sample['query_depth_native'].numpy() * scale]
            for kind in ('local', 'content', 'coverage', 'bound'):
                record = read(f'evaluation/static/{kind}_{SEEDS[0]}/{rid}.json')
                export = torch.load(record['prediction_path'], map_location='cpu', weights_only=False)
                index = export['sample_keys'].index([q, condition])
                arrays.append(export['prediction_native'][index].numpy() * scale)
            axes[row, 0].imshow(sample['rgb'].numpy()); axes[row, 0].set_title('Full RGB' if condition == 0 else 'Masked RGB')
            axes[row, 1].imshow(np.where(valid, target, np.nan), vmin=0, vmax=vmax, cmap='viridis'); axes[row, 1].set_title('Sensor depth')
            for col, (name, depth) in enumerate(zip(('Native', 'L', 'C', 'A', 'B'), arrays), 2):
                axes[row, col].imshow(depth, vmin=0, vmax=vmax, cmap='viridis'); axes[row, col].set_title(name)
            for ax in axes[row]: ax.axis('off')
        label = ('fixed example; also posthoc adverse place' if rid in fixed else 'posthoc adverse place') if rid == worst else 'fixed example'
        fig.suptitle(f'{rid} · {label} · query 11 · seed {SEEDS[0]} · same depth color range per row')
        fig.savefig(OUT / 'figures' / f'example_{rid}.png', dpi=160); plt.close(fig)
    return chosen


def comparison_table(lines, comparisons):
    lines += ['', '| 比较：前者误差 − 后者误差 | 相对降幅 | 配对种子差 | 地点×种子 95% 区间 | 改善地点 |',
              '|---|---:|---|---|---:|']
    for key, r in comparisons.items():
        lines.append(f"| {key} | {pct(r['relative_reduction_percent'])} | {', '.join(num(x) for x in r['paired_seed_gain'])} | {interval(r)} | {r['places_better']}/{r['places']} |")
    lines.append('')


def main():
    a, b, intervention, error = (read(name) for name in ('static_analysis.json', 'integrated_analysis.json', 'intervention_analysis.json', 'error_analysis.json'))
    checks, training, costs = (read(name) for name in ('independent_checks.json', 'training_summary.json', 'native_costs.json'))
    assert checks['complete'] and a['center_evaluated'] == a['center_control_triggered']
    assert error['triggered'] == a['error_probe_triggered']
    plots(a, b, intervention, error); selected = qualitative(a)
    g = a['cohorts']['primary51']['modes']; primary = g['mixed']['log_mse']
    decision_path = OUT / 'method_decision.json'; decision = read('method_decision.json') if decision_path.exists() else None
    lead = decision['lead'] if decision else '以下评价已计算；方法取舍仍待结合主比较、干预、动态状态和可视案例完整审阅。'
    lines = ['# 高斯几何与内容逐槽绑定：内部 memory cross-attention 验证', '', lead, '',
             '`bound_read1`，全部 85 片段为已用开发数据，训练 12 片段；主评价 51 非训练地点（63 片段）。这不是新数据确认。', '',
             '## 1. 方法与待验证的解释', '',
             '上一轮几何 Q/K 通路与内容读取相互独立，CG 对仅内容 C 没有明确增量，内容槽的排列也不影响读出。本轮仅替换读取接口：在冻结 VGGT 第 12 个 global block 后，用一个 width64、4-head cross-attention 将每槽高斯的投影支持与该槽 K/V 绑定，后续原生层和 DPT 不变。', '',
             '`score_ij = q_i·k_j / sqrt(16) + log(w_ij)`；`w_ij` 为前到后 alpha 合成权重；额外零 K/V 的 null 槽先验为剩余透射率。形状协方差描述区域范围，不是位置估计不确定性。', '',
             '| 组 | 内容来源 | 各槽先验 | null 先验 |', '|---|---|---|---|',
             '| L | 当前因果短窗口 | 1/256 | 0 |', '| C | 固定历史 M0 | 1/256 | 0 |',
             '| A | 固定历史 M0 | 覆盖 a_i / 256 | 透射率 T_i |', '| B | 固定历史 M0 | 逐槽 w_ij | 同一 T_i |', '',
             'A/B 共享相同的逐 token 覆盖与 null 质量，主比较区分覆盖提示与具体空间对应。L 是容量匹配的实际短窗口对照；它没有历史几何，不能称同几何信息对照。所有组均 137,472 个参数、三配对种子、2,000 步，使用最终检查点。旧结构的适配器不混入匹配主表。', '',
             '竞争解释包括覆盖门控、无序历史内容已足够、预测几何与尺度噪声、小训练集迁移，以及静态训练和动态状态评价差异。绑定是有结构依据的候选解释，不预先作为唯一病因。初始化 0–3；写入 4/6/8/10；查询 5/7/9/11 永不写回。已知相机、共同尺度及数据来源保持原样。', '',
             '## 2. 固定记忆的原生输出', '',
             '| 方法 | 混合 log-MSE | 完整输入 | 遮挡输入 | 混合去均值空间误差 |', '|---|---:|---:|---:|---:|']
    for k, mean in primary['means'].items():
        lines.append(f"| {LABELS[k]} | {num(mean)} | {num(g['full']['log_mse']['means'][k])} | {num(g['masked']['log_mse']['means'][k])} | {num(g['mixed']['centered_log_mse']['means'][k])} |")
    lines += ['', '正的误差差与降幅表示后者改善；区间按地点与种子配对重采样 5,000 次。']
    comparison_table(lines, primary['comparisons'])
    lines += ['| 批次 | B 对 A 降幅 | B 对 L 降幅 | B 对 C 降幅 |', '|---|---:|---:|---:|']
    for group in ('old12', 'used_summary20', 'used_writer19', 'training12'):
        r = a['cohorts'][group]['modes']['mixed']['log_mse']['comparisons']
        lines.append(f"| {group} | {pct(r['coverage_minus_bound']['relative_reduction_percent'])} | {pct(r['local_minus_bound']['relative_reduction_percent'])} | {pct(r['content_minus_bound']['relative_reduction_percent'])} |")
    lines += ['', '完整输入、隐藏/可见区域、均值偏差、AbsRel 和米制 RMSE 的全部分地点及配对结果见 `static_analysis.json`。共同尺度有已知离散度，本轮没有筛样本或真值对齐；其代价仍由方法整体承担。', '']
    if a['center_evaluated']:
        lines += ['满足预定诊断条件后完成三种子 Center 控制。它同时改变协方差描述和投影足迹，只能判断二者的联合贡献，不能单独证明各向异性必要。', '']
    else:
        lines += ['B/A 未满足“主均值改善且至少两个种子改善”的诊断条件，未追加 Center；本轮不能给出新的形状必要性结论。', '']
    lines += ['## 3. 对应关系是否被使用', '',
              '只对首个固定 B 模型做内容置换及替换为 A 先验的零训练干预。表中正值表示干预误差高于完整 B。它们检测依赖性，分布变化仍是竞争解释；匹配训练的 A/B 对照承担用途判断。', '']
    comparison_table(lines, intervention['cohorts']['primary51']['mixed']['log_mse']['comparisons'])
    check = intervention['trained_interface']
    precision_note = []
    if (OUT / 'precision_diagnostic_cpu.json').exists():
        precision = read('precision_diagnostic_cpu.json')
        joint = precision['differences']['joint_content_and_prior_permutation']
        content = precision['differences']['content_only_permutation']
        precision_note = [f"同一样例的 CPU FP32 后半网络复核中，联合置换最大／平均差降至 {joint['max_abs']:.6g}/{joint['mean_abs']:.6g}，只置换内容的平均差仍为 {content['mean_abs']:.6g}。它支持槽对应实现正确；相对 GPU BF16 同时改变了精度和后端，不能将差异全部归因于精度一个因素。未因此改变训练、指标或新增科学门禁。详见 `precision_diagnostic_cpu.json`。", '']
    lines += [f"固定训练样例上，联合置换内容与几何先验的输出最大差为 {check['joint_content_and_geometry_prior_max_abs']:.6g}，只置换内容的最大差为 {check['content_only_max_abs']:.6g}。此处是 BF16 实现与已学接口的诊断，不是泛化收益。", '',
              *precision_note,
              '## 4. 固定读取器能否利用新增观测', '',
              '复用上一轮保存的状态和更新器，A/B 各三种子固定权重，比较 Static、Corrective 和相同观测的 Prefix 重建。Prefix 只重建 256 槽，不重新联合运行全部 RGB；它是参照而非严格上界。', '',
              '| 读取器／状态 | 混合 log-MSE | 完整输入 | 遮挡输入 | 混合空间误差 |', '|---|---:|---:|---:|---:|']
    ig = b['cohorts']['primary51']
    for key in ig['mixed']['log_mse']['means']:
        lines.append(f"| {key} | {num(ig['mixed']['log_mse']['means'][key])} | {num(ig['full']['log_mse']['means'][key])} | {num(ig['masked']['log_mse']['means'][key])} | {num(ig['mixed']['centered_log_mse']['means'][key])} |")
    lines.append(''); comparison_table(lines, ig['mixed']['log_mse']['comparisons'])
    lines += ['各时点的查询不同，时间曲线的升降本身不能隔离记忆质量，需比较同一时点策略。原状态末次覆盖 Static/Corrective/Prefix 为 53.53%/65.58%/91.09%；它们的共同支持几何评分只有 44 个地点可用，和本轮 51 地点的原生输出统计范围不同。覆盖提升不等于读出改善。', '']
    if error['triggered']:
        lines += ['已触发首个 B 种子的受控错误与早期查询回访，以下仅为完整输入、单模型机制诊断。', '',
                  '| 初始偏移 | Static | Average | Corrective | Corrective 对 Average 降幅 | 配对区间 |', '|---|---:|---:|---:|---:|---|']
        for amp, r in error['amplitudes'].items():
            comparison = r['average_minus_corrective']
            lines.append(f"| {amp} 初始单位 | {num(r['means']['static'])} | {num(r['means']['average'])} | {num(r['means']['corrective'])} | {pct(comparison['relative_reduction_percent'])} | {interval(comparison)} |")
        lines += ['', '| 早期查询 5 的事后回访 | 首次写后误差 | 末次写后误差 | 首次−末次区间 |', '|---|---:|---:|---|']
        for policy, r in error['retrospective_retention'].items():
            lines.append(f"| {policy} | {num(r['after_first_write'])} | {num(r['after_last_write'])} | {interval(r['first_minus_last'])} |")
        lines += ['', '固定受扰区域结果和四时点原生输出见 `error_analysis.json`。±20%/±5% 方向全部保留，不挑方向；回访是较晚记忆读取相同早期图像，不混入当时的因果成绩。', '']
    else:
        lines += ['未触发本轮单模型纠偏/遗忘复测；原因是 B/A 与 B/L 的联合主均值及种子条件未满足。这不是新读取器通过了纠偏验证。上一轮受控恢复混合、原生用途未明确及早期回访退步的负结果原样保留，不能自动迁移为本轮相同数值。', '']
    lines += ['## 5. 实现、来源与成本', '',
              f"四项针对接口的测试及 20/200 步 GPU 小试通过；零适配与原生输出差为 {checks['interface_checks']['zero_adapter_max_abs']}，原生缓存重放差为 {checks['interface_checks']['native_replay_max_abs']}。小试模型排除在正式比较之外。", '',
              f"已独立用 NumPy float64 从原始标签重算 {checks['exported_predictions_recomputed']:,} 份导出预测；另核对 {checks['reused_native_rows']:,} 行复用原生参照，以及全部种子的初值和 2,000 步采样配对。来源、检查点、缓存和状态身份保存在每片段记录中。没有新地点确认或数据排除。", '',
              f"正式训练 {training['fits']} 次，共 {training['steps']:,} 步，训练循环合计 {training['sum_worker_hours']:.3f} worker 小时，峰值已分配显存 {training['peak_allocated_GiB']:.3f} GiB。时间是各 worker 循环之和，含周期检查点，不是墙钟时间或独占性能测量。", '',
              '共享 GPU 上缓存后半网络原生/A/B 的中位耗时（秒）：' + '; '.join(
                  f"{r['id']} {r['timing']['native']['median_seconds']:.4f}/{r['timing']['coverage']['median_seconds']:.4f}/{r['timing']['bound']['median_seconds']:.4f}"
                  for r in costs['rows']) + '。投影另外计时，全部样本见 `native_costs.json`；不含前半网络、相机校准和缓存 I/O，不能当作完整在线速度。', '',
              '每次四视图的 float32 log-prior 临时张量为 4×300×257，共 1,233,600 字节；原 256 槽状态与规则 writer 未改。先验缓存只为本次开发加速，不计作部署时常驻的跨查询存储。旧短序列 CPU4 结果仍显示 Prefix 重建可比 Corrective 更快，不能宣称增量更新有速度优势。', '',
              '## 6. 结论与下一步', '']
    lines += [line for paragraph in decision['paragraphs'] for line in (paragraph, '')] if decision else ['方法取舍待完整审阅，不以表格生成自动宣称方案有效。', '']
    lines += ['## 7. 图与原始记录', '',
              '固定配置见 `plan.json`、`evaluation_protocol.json`、源码快照；主分数、独立复算和成本见相应 JSON。旧 `incremental_vggt1` 与此前正负结果均保留。', '',
              '![固定记忆与绑定](figures/static_and_binding.png)', '', '![动态原生输出与覆盖](figures/native_integration.png)', '']
    if error['triggered']: lines += ['![条件纠偏与回访](figures/conditional_error_and_retention.png)', '']
    for rid in selected: lines += [f'![案例 {rid}](figures/example_{rid}.png)', '']
    selection = read('figure_selection.json')
    if selection['first_catalog_clip_in_that_place'] in selection['fixed']:
        lines += [f"三个案例事先固定；其中 {selection['first_catalog_clip_in_that_place']} 同时也是按主地点 B 对 L 的最大退步事后标注的反例，未额外选择不同片段。案例不替代整体统计。", '']
    else:
        lines += ['前三个案例事先固定；额外案例按主地点 B 对 L 的最大退步事后选择，明确标为反例。案例不替代整体统计。', '']
    (OUT / 'report.md').write_text('\n'.join(lines))
    sources = list((ROOT / 'experiments').glob('gaussian_memory_bound*.py')) + [ROOT / 'models/gaussian_memory_bound_read.py']
    write_json(OUT / 'report_manifest.json', {'report_sha256': sha256(OUT / 'report.md'),
        'decision_sha256': sha256(decision_path) if decision else None,
        'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        'figures': {p.name: sha256(p) for p in (OUT / 'figures').glob('*')},
        'conclusion_review_pending': decision is None, 'figure_review_pending': True, 'phase_complete': False,
        'continuous_research_goal_active': True})
    print(json.dumps({'report_written': str(OUT / 'report.md'), 'human_readable_conclusion_and_figure_review_pending': True}), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4); main()
