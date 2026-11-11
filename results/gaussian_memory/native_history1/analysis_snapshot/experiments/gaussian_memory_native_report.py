"""Fixed result tables and illustrations for native_history1."""
import json

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from experiments.gaussian_memory_native_data import ROOT, OUT, CACHE, ARMS, read, load, sha256, write_json

LABELS = {'recent': 'Anchor + recent', 'history': 'Full history',
          'history_mask': 'Early content masked', 'history_drop': 'Direct early cache removed'}
COMPARISONS = [('history_vs_recent', '完整历史 / 近期'), ('history_vs_history_mask', '完整历史 / 早期遮挡'),
               ('history_vs_history_drop', '完整历史 / 撤掉早期直接缓存')]


def table(stats, metric):
    lines = ['| 比较（候选 / 基准） | 基准误差 | 候选误差 | 相对降低及95%区间 | 改善地点 |', '|---|---:|---:|---:|---:|']
    for key, name in COMPARISONS:
        r = stats[metric]['comparisons'][key]; ci = r['relative_gain_percent_95']
        lines.append(f'| {name} | {r["baseline_mean"]:.6f} | {r["candidate_mean"]:.6f} | '
                     f'{r["relative_gain_percent"]:+.2f}% [{ci[0]:+.2f}, {ci[1]:+.2f}] | {r["positive_places"]}/12 |')
    return '\n'.join(lines)


def figures(a, catalog, ids):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.facecolor': 'white', 'figure.facecolor': 'white'})
    folder = OUT / 'figures'; folder.mkdir(exist_ok=True)
    colors = ['#27848F', '#D99A37', '#BA6257']
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    specs = [(k, n) for (k, _), n in zip(COMPARISONS, ['vs anchor + recent', 'vs early masked', 'vs direct early removed'])]
    for ax, condition, title in [(axes[0, 0], 'persistent', 'A  Persistent masking'), (axes[0, 1], 'full', 'B  Full query input')]:
        comp = a['conditions'][condition]['hidden_log_mse']['comparisons']
        for x, ((key, label), color) in enumerate(zip(specs, colors)):
            r = comp[key]; mean = r['relative_gain_percent']; lo, hi = r['relative_gain_percent_95']
            ax.errorbar(x, mean, yerr=[[mean - lo], [hi - mean]], fmt='o', color=color, capsize=5, markersize=7)
        ax.axhline(0, color='#333', lw=.8); ax.set_xticks(range(3), [n for _, n in specs], rotation=15, ha='right')
        ax.set_ylabel('History ROI error reduction (%)'); ax.set_title(title); ax.grid(axis='y', alpha=.15)
    ax = axes[1, 0]
    for i, condition in enumerate(('full', 'persistent')):
        stats = a['conditions'][condition]
        gains = [stats[m]['comparisons']['history_vs_recent']['gain'] for m in
                 ('hidden_centered_log_mse', 'hidden_bias_squared', 'hidden_log_mse')]
        ax.bar(np.arange(3) + (i - .5) * .32, gains, .31, label=condition.capitalize(), color=colors[i])
    ax.axhline(0, color='#333', lw=.8); ax.set_xticks(range(3), ['ROI spatial', 'ROI bias squared', 'ROI total'])
    ax.set_ylabel('Recent minus history (log-MSE)'); ax.set_title('C  Within-ROI error decomposition'); ax.legend(frameon=False)
    ax = axes[1, 1]; places = a['places']
    support = a['source_support']['persistent']['roi']['new_consistent_fraction_by_place']
    effects = a['conditions']['persistent']['hidden_log_mse']['comparisons']['history_vs_recent']['place_gain']
    for i, place in enumerate(places):
        ax.scatter(100 * support[place], effects[place], color=colors[0], s=35)
        ax.annotate(str(i + 1), (100 * support[place], effects[place]), xytext=(4, 3), textcoords='offset points', fontsize=8)
    ax.axhline(0, color='#333', lw=.8); ax.set_xlabel('New sensor-consistent support / valid ROI (%)')
    ax.set_ylabel('Recent minus history (ROI log-MSE)'); ax.set_title('D  Place effects and added support (descriptive)')
    fig.suptitle('Pretrained StreamVGGT | 12 used places, one checkpoint | positive gain means lower error', fontsize=12)
    fig.tight_layout()
    for ext in ('png', 'pdf'): fig.savefig(folder / ('summary.' + ext), dpi=175, bbox_inches='tight')
    plt.close(fig)
    fig, axes = plt.subplots(len(ids), 6, figsize=(16, 2.7 * len(ids)))
    error_pool = []
    prepared = []
    for rid in ids:
        rec = read(OUT / 'evaluation' / f'{rid}.json'); saved = load(rec['prediction_path']); sample = load(CACHE / rid / 'inputs.pt')
        target = sample['target'].numpy(); valid = sample['valid'].numpy(); hidden = sample['hidden'].numpy()
        errors = {}
        for arm in ('recent', 'history', 'history_mask'):
            idx = saved['keys'].index(['persistent', arm]); pred = saved['depth_metric'][idx].numpy()
            errors[arm] = np.log(np.maximum(pred, 1e-6) / target) ** 2
            error_pool.append(errors[arm][valid])
        prepared.append((rid, rec, sample, target, valid, hidden, errors))
    limit = max(.1, float(np.quantile(np.concatenate(error_pool), .98)))
    diff_limit = max(.01, float(np.quantile(np.concatenate([np.abs(v[-1]['history'] - v[-1]['recent'])[v[4]] for v in prepared]), .98)))
    for row, (rid, rec, sample, target, valid, hidden, errors) in enumerate(prepared):
        image = sample['rgb_uint8'][-1].permute(1, 2, 0).numpy().astype(float) / 255
        image[sample['source_masks'][11].numpy()] = [.485, .456, .406]
        axes[row, 0].imshow(image, extent=(0, 280, 210, 0))
        depth = axes[row, 1].imshow(np.where(valid, target, np.nan), cmap='viridis', vmin=0, vmax=10)
        for col, arm in enumerate(('recent', 'history', 'history_mask'), 2):
            error = axes[row, col].imshow(np.where(valid, errors[arm], np.nan), cmap='magma', vmin=0, vmax=limit)
        difference = axes[row, 5].imshow(np.where(valid, errors['history'] - errors['recent'], np.nan), cmap='RdBu_r', vmin=-diff_limit, vmax=diff_limit)
        yy, xx = np.where(hidden)
        for ax in axes[row]:
            ax.add_patch(Rectangle((xx.min(), yy.min()), xx.max() - xx.min() + 1, yy.max() - yy.min() + 1, fill=False, ec='#E65345', lw=1))
            ax.set_xticks([]); ax.set_yticks([])
        axes[row, 0].set_ylabel(rec['place'])
    titles = ('Masked query', 'Target depth (m)', 'Recent error', 'History error', 'Early-masked error', 'History minus recent')
    for ax, name in zip(axes[0], titles): ax.set_title(name, fontsize=10)
    fig.suptitle('First three fixed clips + worst persistent ROI history change | red difference: history worsens error', fontsize=12)
    fig.subplots_adjust(top=.91, bottom=.1, left=.06, right=.98, hspace=.1, wspace=.08)
    fig.colorbar(depth, cax=fig.add_axes([.24, .04, .12, .012]), orientation='horizontal')
    fig.colorbar(error, cax=fig.add_axes([.54, .04, .15, .012]), orientation='horizontal')
    fig.colorbar(difference, cax=fig.add_axes([.82, .04, .15, .012]), orientation='horizontal')
    for ext in ('png', 'pdf'): fig.savefig(folder / ('cases.' + ext), dpi=175, bbox_inches='tight')
    plt.close(fig)
    write_json(OUT / 'figure_manifest.json', {'complete': True, 'actually_viewed': False,
        'files': {str(p.relative_to(OUT)): sha256(p) for p in sorted(folder.iterdir())},
        'error_display_max': limit, 'difference_display_abs_max': diff_limit,
        'color_scale_rule': 'Shared across all displayed methods and places; 98th percentile display clipping only; scores not clipped.',
        'numbered_scatter_places': dict(enumerate(places, 1)),
        'source_sha256': sha256(ROOT / 'experiments/gaussian_memory_native_report.py')})


def main():
    torch.set_num_threads(4); a = read(OUT / 'analysis.json'); plan = read(OUT / 'plan.json')
    comp = a['conditions']['persistent']['hidden_log_mse']['comparisons']['history_vs_recent']
    worst = min(comp['place_gain'], key=lambda p: (comp['place_gain'][p], p))
    worst_id = next(m['id'] for m in plan['catalog'] if m['place'] == worst)
    ids = list(dict.fromkeys(['fidelity_00', 'fidelity_01', 'fidelity_02', worst_id]))
    write_json(OUT / 'figure_selection.json', {'ids': ids, 'worst_place': worst, 'rule': 'first three fixed clips plus largest persistent ROI history regression; alphabetical tie break'})
    figures(a, plan['catalog'], ids)
    text = ['# native_history1：预训练原生通路的历史用途验证', '', '2026-11-11。12个已用地点、一个官方预训练检查点、完整/持续遮挡、四组共96份原生查询输出；零训练。研究判断见下方解释，不以接口检查替代用途。', '',
        '## 固定比较与边界', '',
        'R从RGB独立重放0/4/6/8/10后读取query11，H多读取1/2/3。首帧锚点0在各组完全相同。H-mask同长度但遮掉1/2/3中与固定查询ROI相关的来源patch，并重编码后续状态。H-drop沿H构建状态，只在query11前移除早期帧的直接缓存；后续帧仍可携带间接历史，因此它不能代替R。', '',
        'R本身已保留首帧0和近期历史，主比较仅检验额外1/2/3帧在共同锚点之上的净增益；阴性结果不能概括成原生记忆整体无用，共同锚点已经提供足够信息也是竞争解释。', '',
        '原生518×392深度双线性映射到既有280×210传感器评分网格。主成绩直接使用模型深度，不使用每方法尺度拟合或旧VGGT尺度。输入遮挡由源传感器几何构造，是人为控制条件；深度和真值对应不作为模型输入。所有地点都已用，不是新确认；多次推理不构成多个种子。', '',
        '## 主结果', '', '正数表示候选完整历史的误差更低；95%区间为5000次配对地点重采样。', '',
        '### 持续遮挡、固定缺失区', '', table(a['conditions']['persistent'], 'hidden_log_mse'), '',
        '### 完整输入、同一物理区域', '', table(a['conditions']['full'], 'hidden_log_mse'), '']
    for metric, title in [('log_mse', '全图'), ('hidden_centered_log_mse', '缺失区内部去均值空间误差'),
                           ('hidden_bias_squared', '缺失区整体偏差平方'), ('hidden_absrel', '缺失区AbsRel'), ('centered_log_mse', '全图去均值空间误差')]:
        text += [f'### 持续遮挡：{title}', '', table(a['conditions']['persistent'], metric), '']
    text += ['## 实际新增信息', '', '| 条件 | 缺失区新增且深度一致支持 | 全图新增且深度一致支持 | 缺失区零新增地点 |', '|---|---:|---:|---:|']
    for c in ('full', 'persistent'):
        s = a['source_support'][c]
        text.append(f'| {c} | {100*s["roi"]["new_consistent_fraction_mean"]:.3f}% | {100*s["all"]["new_consistent_fraction_mean"]:.3f}% | {len(s["roi"]["zero_new_consistent_places"])} / 12 |')
    text += ['', '新增来源仅指1/2/3相对共同锚点0和近期4/6/8/10的补充；不能沿用旧0..3独有覆盖率。该传感器投影是信息条件代理，不保证网络特征保留或利用了这些内容。五个零新增地点保留，分层与相关性仅作诊断。', '',
        '## 研究解释与决定', '']
    interpretation = OUT / 'interpretation.md'
    text += [interpretation.read_text() if interpretation.exists() else '解释待完整统计与核对后写入。', '',
        '## 实现核对与实际成本', '']
    checks = read(OUT / 'independent_checks.json')
    text += [f'96份导出独立NumPy评分，最大评分差{max(checks["metric_max_difference"].values()):.2e}；独立双线性映射最大绝对差{checks["independent_resize_max_abs"]:.2e}。源支持投影另行独立核对，逐地点差异见[independent_checks.json](independent_checks.json)。', '',
        '官方RGB预处理、原生缓存推理回放及短序列因果批处理核对见[preflight.json](preflight.json)。相机头每帧4轮迭代均追加缓存；H-drop按来源移除各层的3个帧条目和相机头的12个条目，保留官方算法。点图和tracking head权重保留，但完整评价只执行深度与相机输出，不能当作全任务性能或速度。', '',
        f'检查点文件{a["cost"]["model_bytes"]/1e9:.3f} GB，全部片段推理循环合计{a["cost"]["clip_inference_seconds"]:.2f}秒，峰值已分配{a["cost"]["peak_GiB"]:.3f}GiB。循环不含模型下载、加载、源支持准备及报告时间。', '',
        '| 查询前缓存 | 聚合器K/V MiB | 相机K/V MiB |', '|---|---:|---:|']
    for arm in ARMS:
        values = [r['before_query_bytes'] for r in a['cache_costs'] if r['arm'] == arm]
        text.append(f'| {LABELS[arm]} | {np.mean([v["kv"] for v in values])/2**20:.3f} | {np.mean([v["cam"] for v in values])/2**20:.3f} |')
    text += ['', '这是未压缩原生通路基线，不是256×32紧凑状态；模型、各层K/V、相机缓存和峰值临时显存分别报告。HF传输停滞后使用作者清华官方备用站，按同一HF固定revision中的检查点SHA256校验，未用代理；网络重试流量未完整计量。', '',
        '## 图与反例', '', '![主结果与来源条件](figures/summary.png)', '', '![固定片段与最大退步](figures/cases.png)', '',
        f'最大退步地点：`{worst}`。色标在所有显示方法与地点间共享，仅显示范围截断；统计不截断。编号地点映射见figure_manifest.json。', '',
        '[固定计划](../../../docs/gaussian_memory_native_route_plan.md) · [身份与配置](plan.json) · [全部统计](analysis.json) · [方法取舍](method_decision.json) · [实际图像审阅](visual_review.json) · [完成记录](phase_completion.json)。']
    (OUT / 'report.md').write_text('\n'.join(text) + '\n')
    print(json.dumps({'report_written': True, 'figures': 2, 'case_ids': ids}), flush=True)


if __name__ == '__main__': main()
