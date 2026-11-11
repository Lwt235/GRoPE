"""Evidence report and standalone plots for the fixed outside-window test."""
import json

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.gaussian_memory_history_data import (
    ROOT, OUT, BASE_CACHE, FAMILIES, POLICIES, SEEDS, load_plan, sha256, write_json)
from experiments.gaussian_memory_bound_report import num, pct, interval

LABELS = {'content_static': 'C-static', 'soft_static': 'Soft-static',
          'content_dynamic': 'C-dynamic', 'soft_dynamic': 'Soft-dynamic',
          'center_dynamic': 'Center-dynamic', 'native': 'Native', 'local': 'L'}
POLS = {'recent_corrective': 'Recent / Corrective', 'history_corrective': 'History / Corrective',
        'recent_rebuild': 'Recent / Prefix', 'history_rebuild': 'History / Prefix'}
COLORS = ['#387e73', '#e7b083', '#80b2a4', '#c96e2c', '#a8974b']


def read(name): return json.loads((OUT / name).read_text())


def save_figure(fig, name, pdf=True):
    fig.savefig(OUT / 'figures' / f'{name}.png', dpi=160)
    if pdf: fig.savefig(OUT / 'figures' / f'{name}.pdf')
    plt.close(fig)


def ci_panel(ax, entries, title):
    for j, (label, value) in enumerate(entries):
        lo, hi = value['paired_place_seed_95']
        ax.plot([lo, hi], [j, j], color='#517690', lw=2)
        ax.scatter(value['mean_gain'], j, color='#c96e2c', zorder=3)
    ax.axvline(0, color='#888888', lw=1)
    ax.set_yticks(range(len(entries)), [label for label, _ in entries])
    ax.set_title(title); ax.set_xlabel('Error reduction; positive favors candidate')


def plots(main, ranges, retention, diagnostics):
    (OUT / 'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    hist = main['policies']['history_corrective']['modes']['mixed']
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9), constrained_layout=True)
    x = np.arange(4)
    for fam, color in zip(FAMILIES, COLORS):
        means = [main['policies'][p]['modes']['mixed']['log_mse']['means'][fam] for p in POLICIES]
        axes[0, 0].plot(x, means, '-o', label=LABELS[fam], color=color)
    axes[0, 0].set_xticks(x, ['Recent\nCorrective', 'History\nCorrective', 'Recent\nPrefix', 'History\nPrefix'])
    axes[0, 0].set_ylabel('Mixed native log-MSE'); axes[0, 0].set_title('Fixed readers, same late queries')
    axes[0, 0].legend(fontsize=8)
    keys = [('content_dynamic_minus_soft_dynamic', 'C-dyn minus Soft-dyn'),
            ('content_static_minus_soft_dynamic', 'C-static minus Soft-dyn'),
            ('center_dynamic_minus_soft_dynamic', 'Center-dyn minus Soft-dyn')]
    ci_panel(axes[0, 1], [(label, hist['log_mse']['comparisons'][key]) for key, label in keys],
             'History / Corrective: PE and shape utility')
    comparisons = ranges['modes']['mixed']['log_mse']['comparisons']; entries = []
    for policy in ('corrective', 'rebuild'):
        for fam in ('content_static', 'content_dynamic', 'soft_dynamic'):
            entries.append((LABELS[fam] + (' / Corrective' if policy == 'corrective' else ' / Prefix'),
                comparisons[f'{fam}_recent_{policy}_minus_{fam}_history_{policy}']))
    ci_panel(axes[1, 0], entries, 'History gains: Recent minus History')
    ci_panel(axes[1, 1], [(label, hist['centered_log_mse']['comparisons'][key]) for key, label in keys],
             'History / Corrective: centered spatial error')
    fig.suptitle('12 previously used places, 3 paired seeds; 95% place x seed intervals\nAvailability-selected subset; no new confirmation')
    save_figure(fig, 'history_and_position_utility')

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.5), constrained_layout=True)
    ci_panel(axes[0, 0], [(LABELS[f], retention['metrics']['log_mse']['comparisons'][f'{f}_early_minus_{f}_late'])
             for f in FAMILIES], 'Early query 11 revisited: before minus after')
    axes[0, 0].set_xlabel('Error reduction after late writes; retrospective')
    for policy in POLICIES:
        values = diagnostics['policies'][policy]['by_query']
        axes[0, 1].plot([5, 7, 9, 11], [v['query_mean_coverage'] for v in values], '-o', label=POLS[policy])
    axes[0, 1].set_title('Current-view projected coverage'); axes[0, 1].set_ylabel('Mean alpha coverage')
    axes[0, 1].set_xlabel('Late query changes; compare policies at same query'); axes[0, 1].legend(fontsize=8)
    for policy in ('history_corrective', 'history_rebuild'):
        values = diagnostics['policies'][policy]['by_query']
        axes[1, 0].plot([5, 7, 9, 11], [v['slots_with_early_sources'] / 256 for v in values], '-o', label=POLS[policy])
    axes[1, 0].set_title('Early provenance retained in memory slots')
    axes[1, 0].set_ylabel('Fraction with any early source (may be mixed)'); axes[1, 0].set_xlabel('Late query'); axes[1, 0].legend(fontsize=8)
    x = np.arange(4); bottom = np.zeros(4)
    for key, label, color in [('memory_bytes', 'All state tensors', '#517690'),
                              ('recent_buffer_bytes', 'Recent raw observations', '#80b2a4'),
                              ('offline_observation_bytes', 'Offline Prefix observations', '#e7b083')]:
        values = np.array([diagnostics['policies'][p]['by_query'][-1][key] / 1024 for p in POLICIES])
        axes[1, 1].bar(x, values, bottom=bottom, label=label, color=color); bottom += values
    axes[1, 1].set_xticks(x, ['Recent\nCorrective', 'History\nCorrective', 'Recent\nPrefix', 'History\nPrefix'])
    axes[1, 1].set_ylabel('Mean tensor KiB, final state'); axes[1, 1].set_title('State + required observations; Prefix is offline')
    axes[1, 1].legend(fontsize=8)
    fig.suptitle('Retrospective retention, label-free coverage and honest state accounting\nCoverage and provenance are diagnostics, not output-utility evidence')
    save_figure(fig, 'retention_coverage_and_cost')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)
    strategies = ('corrective', 'rebuild')
    for ax, metric, title in [(axes[0], 'log_mse', 'Soft-dynamic: total error history gain'),
                              (axes[1], 'centered_log_mse', 'Soft-dynamic: spatial history gain')]:
        ci_panel(ax, [('Corrective' if p == 'corrective' else 'Prefix',
            ranges['modes']['mixed'][metric]['comparisons'][f'soft_dynamic_recent_{p}_minus_soft_dynamic_history_{p}'])
            for p in strategies], title)
    x = np.arange(2); width = .24
    for j, (metric, label, color) in enumerate([('centered_log_mse', 'Spatial component', '#387e73'),
            ('bias_squared', 'Mean-bias component', '#d88435'), ('log_mse', 'Total', '#517690')]):
        values = [-ranges['modes']['mixed'][metric]['comparisons'][f'soft_dynamic_recent_{p}_minus_soft_dynamic_history_{p}']['mean_gain'] for p in strategies]
        axes[2].bar(x + (j - 1) * width, values, width, label=label, color=color)
    axes[2].axhline(0, color='#888888', lw=1); axes[2].set_xticks(x, ['Corrective', 'Prefix'])
    axes[2].set_ylabel('History minus Recent error (negative better)')
    axes[2].set_title('Total = spatial variance + squared mean bias'); axes[2].legend(fontsize=8)
    fig.suptitle('A prespecified spatial diagnostic reveals a signal that total native error does not establish\nNo ground-truth alignment is applied to predictions; all 12 used places and all 3 seeds retained')
    save_figure(fig, 'history_error_decomposition')


def qualitative(main):
    catalog = load_plan()['catalog']; fixed = ['fidelity_00', 'fidelity_01', 'fidelity_02']
    comp = main['policies']['history_corrective']['modes']['mixed']['log_mse']['comparisons']['content_static_minus_soft_dynamic']
    place = main['places'][int(np.argmin(comp['place_gain']))]
    worst = next(m['id'] for m in catalog if m['place'] == place)
    selected = list(dict.fromkeys(fixed + [worst]))
    write_json(OUT / 'figure_selection.json', {'fixed': fixed, 'posthoc_largest_soft_minus_strong_content_place': place,
        'clip': worst, 'selected': selected, 'query': 11, 'seed': SEEDS[0], 'conditions': [0, 1],
        'scope': 'Fixed cases plus labelled adverse case, no filtering of aggregate evaluation.'})
    cases = [('content_static', 'history_corrective', 'C-static / History'),
             ('soft_dynamic', 'recent_corrective', 'Soft-dyn / Recent'),
             ('soft_dynamic', 'history_corrective', 'Soft-dyn / History'),
             ('soft_dynamic', 'history_rebuild', 'Soft-dyn / Hist Prefix'),
             ('center_dynamic', 'history_corrective', 'Center-dyn / History')]
    for rid in selected:
        fig, axes = plt.subplots(2, 8, figsize=(20, 5.7), constrained_layout=True)
        for row, c in enumerate((0, 1)):
            sample = torch.load(BASE_CACHE / rid / f'q11_c{c}.pt', map_location='cpu', weights_only=False)
            target = sample['target'].numpy(); valid = sample['valid'].numpy(); scale = sample['metric_scale']; assert scale is not None
            vmax = float(np.quantile(target[valid], .98))
            axes[row, 0].imshow(sample['rgb'].numpy()); axes[row, 0].set_title('Full RGB' if c == 0 else 'Masked RGB')
            axes[row, 1].imshow(np.where(valid, target, np.nan), vmin=0, vmax=vmax, cmap='viridis'); axes[row, 1].set_title('Sensor depth')
            arrays = [('Native', sample['query_depth_native'].numpy() * scale)]
            for fam, policy, label in cases:
                record = read(f'evaluation/read/{fam}_{SEEDS[0]}_{policy}/{rid}.json')
                export = torch.load(record['prediction_path'], map_location='cpu', weights_only=False)
                index = export['sample_keys'].index([11, c]); arrays.append((label, export['prediction_native'][index].numpy() * scale))
            for col, (label, depth) in enumerate(arrays, 2):
                axes[row, col].imshow(depth, vmin=0, vmax=vmax, cmap='viridis'); axes[row, col].set_title(label, fontsize=9)
            for ax in axes[row]: ax.axis('off')
        qualifier = ('fixed; also posthoc adverse' if rid in fixed else 'posthoc adverse') if rid == worst else 'fixed example'
        fig.suptitle(f'{rid}: {qualifier}; late query 11; seed {SEEDS[0]}; shared sensor depth scale per row\nRecent / History use Corrective unless Prefix is stated; all current RGB inputs identical')
        save_figure(fig, f'example_{rid}', pdf=False)
    return selected


def table(lines, comparisons, names=None):
    lines += ['| 比较（前者误差减后者） | 相对降低 | 绝对差的95%区间 | 三种子绝对差 | 改善地点 |',
              '|---|---:|---|---|---:|']
    for key, value in comparisons.items():
        label = names.get(key, key) if names else key
        lines.append(f'| {label} | {pct(value["relative_reduction_percent"])} | {interval(value)} | '
                     + ' / '.join(num(v) for v in value['paired_seed_gain'])
                     + f' | {value["places_better"]}/{value["places"]} |')
    lines.append('')


def main():
    a, h, i, r, d, checks, costs = [read(n) for n in ('main_analysis.json', 'history_range_analysis.json',
        'intervention_analysis.json', 'retention_analysis.json', 'state_diagnostics.json',
        'independent_checks.json', 'runtime_costs.json')]
    assert checks['complete'] and checks['exported_predictions_recomputed'] == 9108
    plots(a, h, r, d); selected = qualitative(a)
    decision = read('method_decision.json') if (OUT / 'method_decision.json').exists() else None
    lines = ['# 窗口外历史信息能否提高高斯场景记忆的用途', '',
        decision['lead'] if decision else '固定实验与独立复算已完成；以下数字尚待最终科学解释与图像审阅。', '',
        '## 问题与固定比较', '',
        '此前训练状态匹配并未建立对强纯内容读取器的可靠净收益，而且新增观察多数仍在原生四帧窗口内。本轮不再修改公式：固定五组各三个模型，检验明确位于当前窗口之外的早期观察，以及高斯几何位置项是否帮助读取这些信息。', '',
        '12 个既有非训练地点各含早晚两个不重叠的12帧窗口，共288张已下载RGB。它们按成对可用性确定，且此前相对改善较强，不能代表全部51地点或作为新确认。两窗相隔较大，不是连续长视频。', '',
        '早窗末态接入晚窗，与晚窗重置比较；两者都提供Corrective与累计重建Prefix，共四种256槽状态。所有模型和当前四帧输入相同，几何以早窗世界坐标和单位统一。近期对照共享坐标/尺度元数据，但不读取早期几何与内容槽。B初始四帧联合编码仅在B第3帧到达后整批可用。', '',
        '观察均来自memory-free直接分支；查询帧和传感器标签从不写入。相机已知，主评价使用原固定标定，未重新对齐预测到真值。仅浮点并行累加的逐位检查在预运行修正为合理容差，科学方案与更新运算未改，见 `pre_run_fixes.json`。', '',
        'C也读取同样的高斯聚合内容，因此本轮只检验读取位置项与历史范围，不能证明高斯压缩优于普通token/KV记忆。Center同时改变协方差形状和投影足迹，不是纯方向消融。冻结读法的负结果也不能否定端到端训练。', '',
        '主指标是地点等权的混合原生深度log-MSE，越低越好。差值为前者减后者，正值表示后者改善；区间来自5000次配对地点×种子重采样。12个地点、3个种子均保留；辅助指标不替代主比较。', '',
        '## 四种状态的主误差', '',
        '| 读取器 | Recent-Corrective | History-Corrective | Recent-Prefix | History-Prefix |',
        '|---|---:|---:|---:|---:|']
    for fam in (*FAMILIES, 'native', 'local'):
        lines.append('| ' + LABELS[fam] + ' | ' + ' | '.join(num(a['policies'][p]['modes']['mixed']['log_mse']['means'][fam]) for p in POLICIES) + ' |')
    lines += ['', 'Native与L行在四列中按身份复用，不是四次独立评价。static/dynamic后缀表示先前训练接收的记忆状态，本轮均零训练。', '',
              '## 高斯位置项与强内容读取的增量', '']
    main_comp = a['policies']['history_corrective']['modes']['mixed']['log_mse']['comparisons']
    table(lines, main_comp)
    lines += ['## 同一读取器是否受益于更早历史', '']
    table(lines, h['modes']['mixed']['log_mse']['comparisons'])
    for strategy, values in h['modes']['mixed']['log_mse']['history_gain_interaction'].items():
        for control, value in values.items():
            lines.append(f'- {strategy}：Soft-dynamic历史收益减{LABELS[control]}历史收益 = {num(value["mean_gain"])}，区间 {interval(value)}，种子差 ' + ' / '.join(num(v) for v in value['paired_seed_gain']) + '。')
    lines += ['', '上述交互是两组历史收益之差；正值表示Soft受益更多。Prefix是访问累计原始观察的离线重建参照，不是严格性能上界，也不是在线定长存储。', '',
              '![历史与位置项用途](figures/history_and_position_utility.png)', '', '## 完整、遮挡与空间误差', '']
    lines += ['预定的去均值空间误差用于诊断相对深度结构。总log-MSE严格分解为该空间误差与平均log残差的平方；这里没有用真值重新校正或替换原生预测。下列辅助结果不能替代整体主比较。', '']
    spatial_history = h['modes']['mixed']['centered_log_mse']['comparisons']
    table(lines, {f'{f}_recent_{p}_minus_{f}_history_{p}': spatial_history[f'{f}_recent_{p}_minus_{f}_history_{p}']
        for f in ('content_static', 'content_dynamic', 'soft_dynamic') for p in ('corrective', 'rebuild')})
    lines += ['![历史收益的误差分解](figures/history_error_decomposition.png)', '']
    for policy in POLICIES:
        lines += [f'### {POLS[policy]}', '', '| 指标/输入 | C-static | C-dynamic | Soft-dynamic | Center-dynamic | Native |', '|---|---:|---:|---:|---:|---:|']
        for mode, metric in [('full', 'log_mse'), ('masked', 'log_mse'), ('masked', 'hidden_log_mse'),
                ('masked', 'visible_log_mse'), ('mixed', 'centered_log_mse'), ('mixed', 'bias_squared'),
                ('mixed', 'absrel'), ('mixed', 'rmse_m')]:
            values = a['policies'][policy]['modes'][mode][metric]['means']
            lines.append('| ' + mode + '/' + metric + ' | ' + ' | '.join(num(values[f]) for f in
                ('content_static', 'content_dynamic', 'soft_dynamic', 'center_dynamic', 'native')) + ' |')
        lines.append('')
        if policy == 'history_corrective':
            lines += ['去均值空间误差的配对比较：', '']; table(lines, a['policies'][policy]['modes']['mixed']['centered_log_mse']['comparisons'])
    lines += ['## 早期视角保持与对应依赖', '',
        '回访固定早窗query11完整输入，用早窗末态与晚窗History-Corrective末态读取；三种子全评。前者180行按严格身份复用，后者180份新预测。这是事后回访，不混入因果主成绩。', '']
    table(lines, r['metrics']['log_mse']['comparisons'])
    lines += ['首个Soft-dynamic种子在History-Corrective上做内容槽置换与均匀先验；正差表示干预有损害，只检验依赖。', '']
    table(lines, i['modes']['mixed']['log_mse']['comparisons'])
    lines += ['## 记忆保留、覆盖与资源', '', '| 状态 | 末次查询平均覆盖 | 保留早期来源的槽数 | 完整状态KiB | 近期缓冲KiB | 离线Prefix观察KiB |', '|---|---:|---:|---:|---:|---:|']
    for policy in POLICIES:
        v = d['policies'][policy]['by_query'][-1]
        lines.append(f'| {POLS[policy]} | {v["query_mean_coverage"]:.4f} | {v["slots_with_early_sources"]:.2f} | {v["memory_bytes"]/1024:.2f} | {v["recent_buffer_bytes"]/1024:.2f} | {v["offline_observation_bytes"]/1024:.2f} |')
    lines += ['', '早期来源槽可以同时融合晚期观察，不代表纯早期信息仍完整保存；覆盖不等于输出用途。完整状态包含24列source/conflict与256×256 lineage，不宣称任意长序列恒定字节。Prefix另需累计观察，表中已计入。', '',
        f'新增训练0步、下载0字节。状态生成合计{costs["state_preparation_seconds"]:.2f}秒，预测阶段记录的推理时间合计{sum(v["inference_seconds"] for v in costs["phases"].values()):.2f} worker秒，峰值已分配{costs["peak_allocated_GiB"]:.3f}GiB。使用0/2/4卡，通过两次实时占用检查，每卡一个worker、12GiB上限；这些是运行成本，非独占速度基准。', '',
        '![回访、覆盖与成本](figures/retention_coverage_and_cost.png)', '', '## 固定案例与反例', '',
        '以下均为首种子晚窗query11完整与mask0输入，使用公共传感器色阶；固定三例加按三种子主误差选出的Soft-dynamic相对C-static最差地点，后者明确标为事后反例。图像不能替代整体统计。', '']
    for rid in selected: lines += [f'![{rid}](figures/example_{rid}.png)', '']
    lines += ['## 独立核对与证据边界', '',
        f'{checks["exported_predictions_recomputed"]:,}份新预测已从原始传感器标签以float64 NumPy独立评分，{checks["reused_parent_rows"]}行父轮评分按身份核对。15个检查点、公共尺度、48次旧更新操作、来源时间顺序、只读查询及全量输出身份均检查；没有剔除地点或查询。评分最大绝对差为{max(checks["float64_numpy_max_abs_difference"].values()):.3g}，不影响方法结论。', '',
        '所有模型与数据均已用于开发；此轮回答固定表示与读取器在这12对窗口上的适用边界。不得升级为未用数据确认、准确高斯几何学习、一般PE优势或无限长度记忆成功。', '', '## 方法取舍与下一步', '']
    if decision: lines.extend(p + '\n' for p in decision['paragraphs'])
    else: lines += ['待完成图像实际审阅与结果解释；不会仅凭均值自动选择后续路线。', '']
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n')
    write_json(OUT / 'report_manifest.json', {'report_sha256': sha256(OUT / 'report.md'),
        'decision_sha256': sha256(OUT / 'method_decision.json') if decision else None,
        'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in (
            ROOT / 'experiments/gaussian_memory_history_analysis.py', ROOT / 'experiments/gaussian_memory_history_checks.py',
            ROOT / 'experiments/gaussian_memory_history_report.py')},
        'figures': {p.name: sha256(p) for p in (OUT / 'figures').iterdir()},
        'figure_review_pending': True, 'conclusion_review_pending': decision is None, 'phase_complete': False})
    print(json.dumps({'report_generated': True, 'qualitative_examples': selected, 'decision_present': decision is not None}), flush=True)


if __name__ == '__main__': main()
