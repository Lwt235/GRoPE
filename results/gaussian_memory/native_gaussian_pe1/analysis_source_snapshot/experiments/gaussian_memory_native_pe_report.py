"""Tables and two figures for the fixed native PE comparison; interpretation separate."""
import csv
import json

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.gaussian_memory_native_pe_data import ROOT, OUT, CACHE, load, read, write_json, sha256, load_plan, input_path

COLORS = {'native': '#667788', 'center': '#1976b9', 'sphere': '#e69521', 'gaussian': '#b33d55'}
LABELS = {'official': 'Official frozen', 'native': 'A Native 2D', 'center': 'B Strong center',
          'sphere': 'C Spherical extent', 'gaussian': 'D Full covariance'}


def figure_summary(a, plan):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax = axes[0, 0]
    for kind in plan['kinds']:
        traces = []
        for seed in plan['seeds']:
            rows = [json.loads(l) for l in (OUT / f'{kind}_{seed}/training.jsonl').read_text().splitlines()]
            y = np.array([r['loss'] for r in rows]); smooth = np.convolve(y, np.ones(100) / 100, 'valid')
            ax.plot(np.arange(len(smooth)) + 100, smooth, color=COLORS[kind], alpha=.2, linewidth=.6); traces.append(smooth)
        ax.plot(np.arange(len(smooth)) + 100, np.mean(traces, axis=0), color=COLORS[kind], label=LABELS[kind], linewidth=1.6)
    ax.set(xlabel='Training step', ylabel='Raw log-MSE + spatial log-MSE', title='Matched training; 100-step moving means'); ax.legend(fontsize=8)
    ax = axes[0, 1]; rows = read(OUT / 'independent_rows.json')
    for kind in plan['kinds']:
        values = [np.mean([r['hidden_log_mse'] for r in rows if r['id'] == rid and r['kind'] == kind and r['condition'] == 'persistent']) for rid in plan['evaluation_ids']]
        ax.plot(np.arange(12), values, marker='o', markersize=3, linewidth=.8, color=COLORS[kind], label=LABELS[kind])
    ax.set_xticks(np.arange(12), [m['place'] for m in plan['catalog']], rotation=65, ha='right', fontsize=7)
    ax.set(ylabel='ROI raw log-MSE', title='Persistent occlusion; mean of 3 seeds'); ax.legend(fontsize=8)
    pairs = ['gaussian_vs_center', 'gaussian_vs_sphere', 'sphere_vs_center', 'center_vs_native', 'gaussian_vs_native']
    names = ['D vs B', 'D vs C', 'C vs B', 'B vs A', 'D vs A']
    for ax, metric, title in [(axes[1, 0], 'hidden_log_mse', 'ROI raw depth error'),
                             (axes[1, 1], 'hidden_centered_log_mse', 'ROI spatial error, mean removed')]:
        records = [a['comparisons']['persistent'][p][metric] for p in pairs]
        mean = np.array([r['relative_gain_percent'] for r in records]); ci = np.array([r['relative_gain_percent_95'] for r in records])
        ax.hlines(np.arange(len(pairs)), ci[:, 0], ci[:, 1], color='#476173', linewidth=1.8)
        ax.scatter(mean, np.arange(len(pairs)), color='#b33d55', s=25, zorder=3)
        ax.axvline(0, color='black', linewidth=.7); ax.set_yticks(np.arange(len(pairs)), names)
        ax.invert_yaxis(); ax.set(xlabel='Relative error reduction (%); paired 95% intervals', title=title)
    fig.suptitle('Native Gaussian PE: fixed contents, matched adaptation, used development places\nPositive reduction means the first method has lower error', fontsize=12)
    for extension in ('png', 'pdf'): fig.savefig(OUT / 'figures' / f'summary.{extension}', dpi=170)
    plt.close(fig)


def figure_cases(a, plan):
    ids = a['fixed_example_ids']; seed = plan['seeds'][0]
    predictions = {}
    for kind in plan['kinds']:
        rec = read(OUT / 'evaluation' / f'{kind}_{seed}.json'); saved = load(rec['prediction_path'])
        predictions[kind] = {k[0]: pred.numpy() for k, pred in zip(saved['keys'], saved['depth_metric']) if k[2] == 'persistent'}
    inputs = {rid: load(input_path(rid, 11, 'persistent')) for rid in ids}
    depth_values = np.concatenate([s['target'].numpy()[s['valid'].numpy()] for s in inputs.values()])
    vmin, vmax = np.percentile(depth_values, [1, 99]); deltas = []
    for rid in ids:
        s = inputs[rid]; t = s['target'].numpy(); valid = s['valid'].numpy()
        delta = np.log(np.maximum(predictions['gaussian'][rid], 1e-6) / t)**2 - np.log(np.maximum(predictions['center'][rid], 1e-6) / t)**2
        deltas.append(np.where(valid, delta, np.nan))
    limit = max(float(np.nanpercentile(np.abs(np.stack(deltas)), 97)), 1e-6)
    fig, axes = plt.subplots(len(ids), 7, figsize=(18, 3.0 * len(ids)), constrained_layout=True, squeeze=False)
    for row, rid in enumerate(ids):
        s = inputs[rid]; valid = s['valid'].numpy(); target = s['target'].numpy()
        rgb = torch.nn.functional.interpolate(s['images'][-1:,...], (210, 280), mode='bilinear', align_corners=False)[0].permute(1, 2, 0).numpy()
        axes[row, 0].imshow(rgb); axes[row, 1].imshow(np.where(valid, target, np.nan), vmin=vmin, vmax=vmax, cmap='viridis')
        for col, kind in enumerate(plan['kinds'], 2):
            im = axes[row, col].imshow(np.where(valid, predictions[kind][rid], np.nan), vmin=vmin, vmax=vmax, cmap='viridis')
        diff = axes[row, 6].imshow(deltas[row], vmin=-limit, vmax=limit, cmap='coolwarm')
        for ax in axes[row]:
            ax.contour(s['hidden'].numpy().astype(float), levels=[.5], colors=['cyan'], linewidths=.5); ax.set_xticks([]); ax.set_yticks([])
        axes[row, 0].set_ylabel(s['place'] + ('\nLargest D-B regression' if rid == a['worst_id'] else ''), fontsize=9)
    for ax, title in zip(axes[0], ['Masked query RGB', 'Sensor target'] + [LABELS[k] for k in plan['kinds']] + ['Squared-log error D minus B\nred: D worse']): ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=axes[:, 1:6].ravel().tolist(), label='Raw depth (m)', shrink=.5, fraction=.015)
    fig.colorbar(diff, ax=axes[:, 6].ravel().tolist(), label='Error difference', shrink=.5, fraction=.04)
    fig.suptitle(f'Fixed first three examples + largest D-B regression; seed {seed}\nShared raw depth scale; cyan marks the fixed scoring ROI. No per-image alignment.', fontsize=12)
    for extension in ('png', 'pdf'): fig.savefig(OUT / 'figures' / f'cases.{extension}', dpi=160)
    plt.close(fig)


def write_report(a, plan):
    checks = read(OUT / 'independent_result_checks.json'); inp = read(OUT / 'independent_input_checks.json')
    statuses = checks['checkpoints']; total_hours = sum(read(OUT / f'{r["kind"]}_{r["seed"]}/status.json')['seconds_this_invocation'] for r in statuses) / 3600
    peak = max(r['peak_GiB'] for r in statuses)
    table = ['| 条件 | 方法 | ROI log-MSE | ROI空间误差 | ROI偏差平方 | 全图log-MSE |', '|---|---|---:|---:|---:|---:|']
    for condition, methods in a['tables'].items():
        for kind, m in methods.items(): table.append(f'| {condition} | {LABELS[kind]} | {m["hidden_log_mse"]:.6f} | {m["hidden_centered_log_mse"]:.6f} | {m["hidden_bias_squared"]:.6f} | {m["log_mse"]:.6f} |')
    effects = ['| 持续遮挡比较 | 原生ROI相对降误差% [95%区间] | 空间相对降误差% [95%区间] | ROI改善地点 | ROI改善种子 |', '|---|---:|---:|---:|---:|']
    for key in ['gaussian_vs_center', 'gaussian_vs_sphere', 'sphere_vs_center', 'center_vs_native', 'gaussian_vs_native', 'native_vs_official']:
        c = a['comparisons']['persistent'][key]; raw, spatial = c['hidden_log_mse'], c['hidden_centered_log_mse']
        def value(r): return f'{r["relative_gain_percent"]:+.3f} [{r["relative_gain_percent_95"][0]:+.3f}, {r["relative_gain_percent_95"][1]:+.3f}]'
        effects.append(f'| {key} | {value(raw)} | {value(spatial)} | {raw["positive_places"]}/12 | {raw["positive_seeds"]}/3 |')
    interpretation = (OUT / 'interpretation.md').read_text() if (OUT / 'interpretation.md').exists() else '结果解释待人工核对，以下为完整固定比较。'
    text = f'''# native_gaussian_pe1：原生读取中的高斯位置编码

{interpretation}

## 设计与范围

四组同一官方StreamVGGT、同一来源/维度/数量的原生历史特征、最后四层rank8 QKV与输出适配，共196,960可训练参数。A保留二维RoPE并配等容量二维关系映射；B叠加Point3R式三维多基频中心RoPE和中心关系；C保留各自范围，D保留完整协方差。中心、球形、完整组初始预测完全相同。形状不进入旋转幅度衰减。

第一遍预测深度/相机，逐patch点集生成均值和协方差；源帧不含未来，查询允许当前RGB。第二遍只训练查询读取，历史writer和原生DPT头冻结。不读取查询真值几何生成PE。数据遮挡构造使用旧传感器几何，训练和评价标签为传感器深度；这些用途与模型输入分开。该表示为高斯参数形式的观测区域，不是完整渲染优化后的3DGS场景。本轮不证明压缩和在线高斯融合已经完成。

12旧训练片段、96训练样本；12地点隔离的已用开发地点、24查询条件。四组各三种子、2000步、有效batch2、相同初始化/采样；AdamW1e-4。损失为原生log-MSE加去均值log-MSE，等权。所有终点固定，无开发选模，零新下载或新确认。

## 主表

{chr(10).join(table)}

{chr(10).join(effects)}

正数表示第一个方法误差更低。区间为5000次地点/种子配对重采样；三个种子和十二旧地点不构成广泛分布保证，区间跨零不表示等效。主要比较D对B/C；其余和输入分层为辅助分析。

![主比较](figures/summary.png)

## 输入条件与反例

阈值仅由训练预测几何中位数决定：{json.dumps(read(OUT / 'training_statistics.json')['thresholds'], ensure_ascii=False)}。各条件的地点、效应及区间见[完整分析](analysis.json)；不按结果排除零支持或失败地点。预测中心近邻覆盖不等于真实新增历史支持，分层结果只用于探索适用范围。

固定前三例及最大退步例为{a['fixed_example_ids']}，最大D-B主误差退步地点{a['worst_place']}。图用固定首种子，选择依据三个种子平均；不逐图对齐尺度。

![固定例与反例](figures/cases.png)

## 正确性、成本与局限

120份缓存的来源和查询边界、训练/开发地点隔离、官方代码与模型身份已核对。NumPy独立复算6份多帧几何，中心最大差{max(r['center_max_abs'] for r in inp['geometry_numpy']):.3g}，协方差最大差{max(r['covariance_max_abs'] for r in inp['geometry_numpy']):.3g}；24个官方历史输出与父轮最大差{inp['old_history_replay_max_abs']:.3g}。288份适配输出独立像素评分最大差{checks['max_metric_abs']:.3g}；最终模型和实际样本日志配对检查见[独立复算](independent_result_checks.json)。

训练循环共{total_hours:.3f} worker小时，峰值已分配显存{peak:.3f}GiB。缓存约{sum(r['bytes'] for r in inp['cache_records']) / 2**30:.3f}GiB，包含历史K/V、训练重放输入、预测几何和审计信息，不是紧凑持久记忆大小。两遍查询推理和实际持久缓存成本另外列于runtime.json（若尚未生成则待测），不能以末段读取时间代替总推理时间。

正式训练前修复了缓存目录和CPU评分设备问题，保留版本记录；原计划/初始预检源码与最终训练源码分开存档。关系MLP采用共同的split-K梯度归约优化：前向和输入梯度不变、权重梯度对照FP32归约通过；不声称与旧BF16梯度训练轨迹逐位相同。见[训练源码锁](training_lock.json)、[运行预检](preflight_final/complete.json)、[运行优化检查](preflight/splitk_gradient_checks.json)。

当前阴性只能约束该预测几何、固定writer与有限末层适配；阳性也只支持本开发条件，需要固定方法后的未用数据确认。已有区域PE证据、原生历史不确定性和旧接法负结果继续保留。

[固定计划](../../../docs/gaussian_memory_native_pe_plan.md) · [全部分析](analysis.json) · [输入核对](independent_input_checks.json) · [方法取舍](method_decision.json)
'''
    (OUT / 'report.md').write_text(text)
    with (OUT / 'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['condition', 'kind'] + list(next(iter(a['tables']['full'].values())).keys())); writer.writeheader()
        for condition, methods in a['tables'].items():
            for kind, m in methods.items(): writer.writerow(dict(condition=condition, kind=kind, **m))


def main():
    plan = load_plan(); a = read(OUT / 'analysis.json'); (OUT / 'figures').mkdir(exist_ok=True)
    figure_summary(a, plan); figure_cases(a, plan); write_report(a, plan)
    write_json(OUT / 'figure_manifest.json', {'files': {str(p.relative_to(OUT)): sha256(p) for p in (OUT / 'figures').glob('*')},
               'source_sha256': sha256(ROOT / 'experiments/gaussian_memory_native_pe_report.py')})


if __name__ == '__main__': main()
