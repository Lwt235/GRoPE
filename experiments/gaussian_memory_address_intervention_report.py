"""Render fixed comparisons, routing diagnostics and examples for this intervention."""
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from experiments.gaussian_memory_address_intervention import (
    OUT, ROOT, INPUT_CACHE, FAMILIES, SEEDS, POLICIES, read, load, sha256, write_json)

NAMES = {'soft_dynamic': '旧 Soft', 'new_soft': '新 Soft'}
ENAMES = ['Old Soft', 'New Soft']


def table(comp, keys):
    rows = ['| 比较 | 基准误差 | 干预误差 | 误差相对降低及95%区间 | 改善种子/地点 |',
            '|---|---:|---:|---:|---:|']
    for key, name in keys:
        r = comp[key]; ci = r.get('relative_gain_percent_95')
        if r['contributing_places'] == 0: rows.append(f'| {name} | — | — | 无有效像素 | 0 |'); continue
        interval = f'[{ci[0]:+.2f}, {ci[1]:+.2f}]' if ci is not None else '不适用'
        rows.append(f'| {name} | {r["baseline_mean"]:.6f} | {r["candidate_mean"]:.6f} | '
                    f'{r["relative_gain_percent"]:+.2f}% {interval} | {r["positive_seeds"]}/3；{r["positive_places"]}/{r["contributing_places"]} |')
    return '\n'.join(rows)


def diagnostics(a):
    result = {}
    for condition in ('full', 'persistent'):
        for policy in ('recent', 'history'):
            for family in FAMILIES:
                for arm in ('correct_all', 'sham_all', 'correct_history', 'sham_history'):
                    rows = [r for r in a['routing_diagnostics'] if r['condition'] == condition and r['policy'] == policy
                            and r['family'] == family and r['arm'] == arm]
                    if not rows: continue
                    summary = {}
                    for metric in ('active_patches', 'active_valid_roi_pixels', 'roi_valid_pixels', 'route_before', 'route_after',
                                   'attention_L1_on_active', 'value_delta_rms_on_active', 'original_value_rms_on_active', 'target_entropy_on_active'):
                        vals = [r[metric] for r in rows if r[metric] is not None]
                        summary[metric] = {'mean': float(np.mean(vals)) if vals else None, 'place_seed_records': len(vals),
                                           'places': len({r['place'] for r in rows if r[metric] is not None})}
                    result['/'.join((condition, policy, family, arm))] = summary
    write_json(OUT / 'routing_summary.json', {'complete': True, 'summaries': result,
        'definition': 'Unconditional route scores include zero-support places; active-only perturbation summaries declare contributing places.'})
    return result


def figures(a, routing, selection):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    folder = OUT / 'figures'; folder.mkdir(exist_ok=True)
    colors = ['#27848F', '#E7A352', '#C46354']
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    comp = a['conditions']['persistent']['hidden_log_mse']['comparisons']
    for ax, target, title in [(axes[0, 0], 'all', 'A  All retained source correspondences'),
                              (axes[0, 1], 'history', 'B  Early-history-only correspondences')]:
        specs = [(f'correct_{target}_vs_original', 'Correct vs original'),
                 (f'correct_{target}_vs_sham_{target}', 'Correct vs shuffled'),
                 (f'sham_{target}_vs_original', 'Shuffled vs original')]
        for i, (key, label) in enumerate(specs):
            stats = [comp[fam + '/history/' + key] for fam in FAMILIES]
            mean = np.array([r['relative_gain_percent'] for r in stats]); ci = np.array([r['relative_gain_percent_95'] for r in stats]).T
            x = np.arange(2) + (i - 1) * .23
            ax.errorbar(x, mean, np.stack((mean - ci[0], ci[1] - mean)), fmt='o', color=colors[i], capsize=4, label=label)
        ax.axhline(0, color='#333', lw=.8); ax.set_xticks(np.arange(2), ENAMES)
        ax.set_title(title); ax.set_ylabel('ROI error reduction (%)'); ax.grid(axis='y', alpha=.15)
    axes[0, 0].legend(frameon=False, fontsize=9, loc='lower left')
    ax = axes[1, 0]
    for i, (key, label) in enumerate([('route_before', 'Original'), ('correct_all', 'Correct addresses'), ('sham_all', 'Shuffled addresses')]):
        values = []
        for fam in FAMILIES:
            r = routing[f'persistent/history/{fam}/' + ('correct_all' if key == 'route_before' else key)]
            values.append(100 * r['route_before' if key == 'route_before' else 'route_after']['mean'])
        ax.bar(np.arange(2) + (i - 1) * .23, values, .22, color=colors[i], label=label)
    ax.set_xticks(np.arange(2), ENAMES); ax.set_ylabel('Correspondence-weighted routing score (%)')
    ax.set_title('C  Routing improves without reliable task gains'); ax.legend(frameon=False, fontsize=9)
    ax = axes[1, 1]; places = a['places']; x = np.arange(len(places))
    for i, fam in enumerate(FAMILIES):
        values = comp[fam + '/history/correct_all_vs_original']['place_gain']
        ax.bar(x + (i - .5) * .35, [values[p] for p in places], .34, color=colors[i], label=ENAMES[i])
    ax.axhline(0, color='#333', lw=.8); ax.set_ylabel('Original minus correct: ROI log-MSE')
    ax.set_xticks(x, [p.replace('harvard_robotics_lab', 'harvard_robotics').replace('mit_w59_conference', 'mit_w59') for p in places], rotation=50, ha='right', fontsize=8)
    ax.set_title('D  Per-place changes, all retained sources'); ax.legend(frameon=False, fontsize=9)
    fig.suptitle('Persistent masking / History | positive task differences mean lower error | 12 places, 3 fixed seeds', fontsize=12)
    fig.tight_layout()
    for ext in ('png', 'pdf'): fig.savefig(folder / ('summary.' + ext), dpi=170, bbox_inches='tight')
    plt.close(fig)

    ids = selection['ids']; fig, axes = plt.subplots(len(ids), 6, figsize=(16, 2.7 * len(ids)))
    for row, rid in enumerate(ids):
        rec = read(OUT / 'evaluation' / f'{rid}.json'); outputs = load(rec['prediction_path'])
        sample = load(INPUT_CACHE / rid / 'q11_c2.pt')
        target = sample['target'].numpy(); valid = sample['valid'].numpy(); hidden = sample['hidden'].numpy()
        pred = {}
        for arm in ('original', 'correct_all', 'sham_all'):
            key = [2, 'soft_dynamic', SEEDS[0], 'history', arm]
            index = outputs['keys'].index(key)
            pred[arm] = outputs['prediction_native'][index].numpy() * sample['metric_scale']
        err = {k: np.log(np.maximum(v, 1e-6) / target) ** 2 for k, v in pred.items()}
        axes[row, 0].imshow(sample['rgb'].numpy())
        depth = axes[row, 1].imshow(np.where(valid, target, np.nan), cmap='viridis', vmin=0, vmax=5)
        for col, arm in enumerate(('original', 'correct_all', 'sham_all'), 2):
            error = axes[row, col].imshow(np.where(valid, err[arm], np.nan), cmap='magma', vmin=0, vmax=1)
        difference = axes[row, 5].imshow(np.where(valid, err['correct_all'] - err['original'], np.nan),
                                       cmap='RdBu_r', vmin=-.1, vmax=.1)
        yy, xx = np.where(hidden)
        for ax in axes[row]:
            ax.add_patch(Rectangle((xx.min(), yy.min()), xx.max() - xx.min() + 1, yy.max() - yy.min() + 1,
                                  fill=False, ec='#E65345', lw=1.))
            ax.set_xticks([]); ax.set_yticks([])
        axes[row, 0].set_ylabel(rec['place'])
    for ax, title in zip(axes[0], ('Masked query', 'Target depth (m)', 'Original error', 'Correct-address error', 'Shuffled-address error', 'Correct minus original')):
        ax.set_title(title, fontsize=10)
    fig.suptitle('First three fixed clips + largest old-Soft ROI regression | seed 20261110 | red: intervention worsens error', fontsize=12)
    fig.subplots_adjust(top=.91, bottom=.1, left=.06, right=.98, hspace=.1, wspace=.08)
    fig.colorbar(depth, cax=fig.add_axes([.24, .04, .12, .012]), orientation='horizontal')
    fig.colorbar(error, cax=fig.add_axes([.54, .04, .15, .012]), orientation='horizontal')
    fig.colorbar(difference, cax=fig.add_axes([.82, .04, .15, .012]), orientation='horizontal')
    for ext in ('png', 'pdf'): fig.savefig(folder / ('cases.' + ext), dpi=170, bbox_inches='tight')
    plt.close(fig)
    write_json(OUT / 'figure_manifest.json', {'complete': True, 'files': {str(p.relative_to(OUT)): sha256(p) for p in sorted(folder.iterdir())},
        'actually_viewed': False, 'source_sha256': sha256(ROOT / 'experiments/gaussian_memory_address_intervention_report.py')})


def report(a, routing, selection):
    primary = a['conditions']['persistent']['hidden_log_mse']['comparisons']; checks = read(OUT / 'independent_checks.json')
    text = ['# address_intervention1：对应读取增强，没有建立原生输出改进', '',
        '2026-11-11。固定六个新旧 Soft 读取器，在12个已用地点完成1152份输出（288份原输出回放、864份干预）和672行背景参照的独立评分。**本轮没有支持“仅让查询更直接读取对应槽，就足以改善当前原生输出”。** 对应地址提高了来源寻址分数，旧模型的打乱地址又确实造成部分损害；这些现象支持有限的地址关系作用，没有形成优于原读法的新方案。', '',
        '不采用本干预，不扫描混合比例、不补种子，不据失败唯一归罪于V、writer或decoder。固定Prefix训练修补仍关闭，原区域PE和独立粗记忆读出任务的有限正面证据继续保留。本轮所有干预使用查询真值辅助的对应标签，不能作为可部署成绩或新确认。', '',
        '## 固定问题与干预', '',
        '主假设是保留内容中还有当前任务可用的信息，改善地址可能使它显现。竞争解释包括：局部表面对应不是任务所需内容；14×14 patch及混合槽使教师对应不完美；固定V/输出映射/后续网络不能利用这种读法；或者输入分布外的干预破坏已有协同。', '',
        '沿用完整输入/持续遮挡、Recent/History同256槽、新旧Soft各三个固定模型及原尺度。只在query11固定ROI且有来源对应的patch，把记忆读值改为 `0.5 O + 0.5 T V`；T由上一轮槽对应U逐行归一化得到。另一半保留原生SDPA结果。三个上下文视图、特殊token和ROI外patch在插入处不变，后续网络仍可能传播变化。没有来源时严格回退原读取。', '',
        '打乱组将相同T按片段ID固定置换到其他槽，同一生效patch、相同权重分布/熵与0.5混合比例；不是严格等特征范数干预。主检验使用全部对应来源，History另检验仅早期独有来源。查询深度用于构造干预，所有阶段均不训练、不调参、不重建状态。', '',
        '## 主结果：持续遮挡、History、固定缺失区', '',
        '下表正数代表误差降低；95%区间由5000次配对地点×种子重采样得到。12地点主指标完整保留，包含无早期独有支持地点。', '',
        table(primary, [(f'{f}/history/{k}', NAMES[f] + '：' + label) for f in FAMILIES for k, label in [
            ('correct_all_vs_original', '原读法 → 对应全部来源'), ('correct_all_vs_sham_all', '打乱全部来源 → 对应全部来源'),
            ('sham_all_vs_original', '原读法 → 打乱全部来源')]]), '',
        '旧/新Soft的对应地址相对原读法均值分别退步0.47%/1.02%，两项区间跨零，只有3/12地点改善。对应组比打乱组平均好约3%，但主区间也跨零。旧Soft打乱组相对原读法退步3.83%，区间支持退步；可以说错误地址有害，不能将其反写为对应地址已经优于原读法。', '',
        '### 仅早期独有来源', '',
        table(primary, [(f'{f}/history/{k}', NAMES[f] + '：' + label) for f in FAMILIES for k, label in [
            ('correct_history_vs_original', '原读法 → 对应历史独有来源'),
            ('correct_history_vs_sham_history', '打乱历史独有来源 → 对应历史独有来源')]]), '',
        '没有形成历史专属的净改进。旧/新Soft对原读取平均退步0.62%/0.83%，区间均跨零。辅助只评分“确有保留早期独有对应”的像素时，只有八个地点贡献，结论同样不确定；这个子集不能替代全12地点主指标。', '',
        '## 干预有没有实际改变寻址', '',
        '| 模型/目标 | 原读取分数% | 对应地址后% | 打乱地址后% | 平均生效patch（ROI共70） |',
        '|---|---:|---:|---:|---:|']
    for fam in FAMILIES:
        for target, label in [('all', '全部来源'), ('history', '历史独有')]:
            c = routing[f'persistent/history/{fam}/correct_{target}']; s = routing[f'persistent/history/{fam}/sham_{target}']
            text.append(f'| {NAMES[fam]} / {label} | {100*c["route_before"]["mean"]:.5f} | {100*c["route_after"]["mean"]:.5f} | {100*s["route_after"]["mean"]:.5f} | {c["active_patches"]["mean"]:.2f} |')
    text += ['', '旧Soft全部来源的读取分数从0.99047%升至3.02276%，历史独有部分从0.03351%升至0.39415%；模型确实收到了改变后的读值。寻址代理明显上升而输出没有可靠改善，排除了“只是干预根本没有进入网络”的实现解释，未证明信息本身缺失或某个模块是唯一病因。', '',
        '| 生效patch上的改变量 | 旧Soft对应 / 打乱 | 新Soft对应 / 打乱 |', '|---|---:|---:|']
    for target, name in [('all', '全部来源'), ('history', '历史独有')]:
        for metric, label in [('attention_L1_on_active', '注意力L1'), ('value_delta_rms_on_active', '读值RMS')]:
            vals = []
            for fam in FAMILIES:
                c = routing[f'persistent/history/{fam}/correct_{target}'][metric]['mean']
                s = routing[f'persistent/history/{fam}/sham_{target}'][metric]['mean']
                vals.append(f'{c:.4f} / {s:.4f}')
            text.append(f'| {name}：{label} | {vals[0]} | {vals[1]} |')
    text += ['', '全部来源的特征改变量量级接近，但不是逐patch严格匹配；历史独有组的条件均值只含八个有支持地点。每片段仅一个固定打乱置换，区间未包含多次置换的不确定性。粗来源标签和输入分布外读取使本干预不是完美oracle；失败不能证明所有正确读法均无效。', '',
        '## 范围、空间误差与强参照', '']
    for condition, title in [('full', '完整输入'), ('persistent', '持续遮挡')]:
        comp = a['conditions'][condition]['hidden_log_mse']['comparisons']
        text += [f'### {title}：全部对应地址相对原读取', '', table(comp, [(f'{fam}/{pol}/correct_all_vs_original', NAMES[fam] + ' / ' + pol)
                    for fam in FAMILIES for pol in POLICIES]), '']
    for metric, title in [('log_mse', '全图'), ('centered_log_mse', '去均值空间误差'), ('bias_squared', '整体偏差平方'),
                           ('supported_roi_log_mse', '有保留来源支持ROI'), ('history_only_roi_log_mse', '有保留历史独有支持ROI')]:
        comp = a['conditions']['persistent'][metric]['comparisons']
        text += [f'### {title}：持续遮挡、History', '', table(comp, [(f'{fam}/history/{arm}_vs_{base}', NAMES[fam] + '：' + arm + ' / ' + base)
                    for fam in FAMILIES for arm, base in [('correct_all', 'original'), ('correct_all', 'sham_all')]]), '']
    text += ['旧Soft的对应组比打乱组去均值空间误差低7.39%，区间[1.03,14.23]，但相对原读取仅平均低1.92%、区间跨零。这个次指标支持打乱会破坏空间信息，仍不能替代主任务净改进。完整输入和Recent对照也没有稳定改善原读取。', '',
        '| 持续遮挡、History背景参照 | ROI log-MSE |', '|---|---:|']
    means = a['conditions']['persistent']['hidden_log_mse']['means']
    for key, name in [('content_static/history/reference', '强旧C'), ('center_dynamic/history/reference', '旧Center'),
                      ('new_content/history/reference', '新C'), ('new_center/history/reference', '新Center'),
                      ('native/-/reference', 'Native'), ('local/-/reference', 'Local')]:
        text.append(f'| {name} | {means[key]:.6f} |')
    text += ['', '这些对照没有查询真值辅助，作为原结果背景完整保留；不得把干预输出对它们的差距宣传为公平方法收益。完整逐地点/逐种子、AbsRel、RMSE及交互见[analysis.json](analysis.json)。', '',
        '## 研究取舍', '',
        '本轮关闭固定0.5对应地址干预，不改主方案，不自动追加温度、强度、位置层或更多种子。当前能说“空间地址参与读取、错误地址可能损害输出”，不能说“地址更准确就已释放未利用的历史收益”。同一固定读取器继续追求寻址分数上升，已不足以推动研究主张。', '',
        '如果继续研究，下一步应先提出并论证一个与既有监督/写入/直接参考实验不同的最小表示适配问题：现有记忆内容与接收它的表示如何形成可学习的任务通路，并让同容量无记忆和中心位置控制共同参与。不能把“改V”或“解冻decoder”当作本轮已证明的答案；具体结构本轮未选择、未训练或排队。已有独立粗读出任务的有限用途保留，与当前原生内部接法分开陈述。', '',
        '## 图像、核对与资源', '', '![配对结果和寻址变化](figures/summary.png)', '', '![固定片段和最差地点](figures/cases.png)', '',
        f'固定前三片段加旧Soft主比较退步最大的地点：`{selection["worst_place"]}`；图示首种子，不能替代三种子统计。', '',
        f'1152份输出与672行原参照全部由NumPy独立评分，288次原生回放最大差{checks["replay_max_abs"]:.1f}；无支持时{checks["empty_support_exact_replays"]}次完全回退。12组保存读值混合独立复算在BF16舍入后差为{checks["mixture_max_abs_after_bf16_rounding"]:.1f}，模型参数及输入身份保持不变。最大评分差低于{max(checks["max_score_difference"].values()):.2e}。', '',
        f'GPU worker循环耗时{a["cost"]["worker_seconds"][0]:.2f}/{a["cost"]["worker_seconds"][1]:.2f}秒，峰值{a["cost"]["peak_GiB"]:.3f}GiB；不含加载/实现/报告时间。GPU2在启动前已有其他用户占用，按既有共享授权通过双次检查并限12GiB；GPU5启动时空闲。没有训练、下载或新确认。', '',
        '[事前计划](../../../docs/gaussian_memory_address_intervention_plan.md) · [固定身份](plan.json) · [独立核对](independent_checks.json) · [路由与幅度](routing_summary.json) · [方法取舍](method_decision.json) · [图像审阅](visual_review.json) · [完成记录](phase_completion.json)。']
    (OUT / 'report.md').write_text('\n'.join(text) + '\n')


def main():
    torch.set_num_threads(4); a = read(OUT / 'analysis.json'); routing = diagnostics(a)
    comp = a['conditions']['persistent']['hidden_log_mse']['comparisons']['soft_dynamic/history/correct_all_vs_original']
    worst = min(comp['place_gain'], key=lambda p: (comp['place_gain'][p], p))
    catalog = read(OUT / 'plan.json')['catalog']; rid = next(m['id'] for m in catalog if m['place'] == worst)
    ids = list(dict.fromkeys(['fidelity_00', 'fidelity_01', 'fidelity_02', rid]))
    selection = {'ids': ids, 'worst_place': worst, 'selection': 'first three fixed clips plus minimum seed-mean original-minus-correct_all ROI log-MSE, old Soft persistent History; alphabetical tie break'}
    write_json(OUT / 'figure_selection.json', selection); figures(a, routing, selection); report(a, routing, selection)
    print(json.dumps({'report_written': True, 'figures': 2, 'case_ids': ids}), flush=True)


if __name__ == '__main__': main()
