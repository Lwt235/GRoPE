"""Reviewable evidence for one training-state change; no automatic success claim."""
import json

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.gaussian_memory_setup import ROOT, sha256
from experiments.gaussian_memory_state_data import OUT, PARENT_OUT, BASE_CACHE, SEEDS, POLICIES, load_plan, write_json
from experiments.gaussian_memory_bound_report import num, pct, interval, comparison_table

LABELS = {'content_dynamictrain': 'C-dynamic', 'soft_dynamictrain': 'Soft-dynamic',
          'content_statictrain': 'C-static', 'soft_statictrain': 'Soft-static',
          'local': 'L', 'native': 'Native', 'soft_center_dynamictrain': 'Center-dynamic'}
COLORS = {'content_dynamictrain': '#287e72', 'soft_dynamictrain': '#d88435',
          'content_statictrain': '#b4d6c9', 'soft_statictrain': '#e6c3a0',
          'local': '#517690', 'native': '#888888', 'soft_center_dynamictrain': '#b6a85e'}


def read(name): return json.loads((OUT / name).read_text())


def save_figure(fig, name, pdf=True):
    fig.savefig(OUT / 'figures' / f'{name}.png', dpi=160)
    if pdf: fig.savefig(OUT / 'figures' / f'{name}.pdf')
    plt.close(fig)


def plots(analyses, intervention, retention):
    (OUT / 'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    primary = analyses['corrective_clean']['cohorts']['primary51']['modes']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    for ax, mode in zip(axes[0], ('mixed', 'full', 'masked')):
        means = primary[mode]['log_mse']['means']; keys = list(means)
        ax.bar([LABELS[k] for k in keys], [means[k] for k in keys], color=[COLORS[k] for k in keys])
        ax.set_title('Corrective memory: ' + mode); ax.set_ylabel('Native depth log-MSE'); ax.tick_params(axis='x', rotation=30)
    comparisons = primary['mixed']['log_mse']['comparisons']; ax = axes[1, 0]
    names = [('content_dynamictrain_minus_soft_dynamictrain', 'C-dyn minus Soft-dyn'),
             ('content_statictrain_minus_soft_dynamictrain', 'C-static minus Soft-dyn*'),
             ('content_statictrain_minus_content_dynamictrain', 'State training effect: C'),
             ('soft_statictrain_minus_soft_dynamictrain', 'State training effect: Soft'),
             ('native_minus_soft_dynamictrain', 'Native minus Soft-dyn')]
    if 'soft_center_dynamictrain_minus_soft_dynamictrain' in comparisons:
        names += [('soft_center_dynamictrain_minus_soft_dynamictrain', 'Center-dyn minus Soft-dyn')]
    for j, (name, label) in enumerate(names):
        r = comparisons[name]; lo, hi = r['paired_place_seed_95']
        ax.plot([lo, hi], [j, j], color='#517690', lw=2); ax.scatter(r['mean_gain'], j, color='#d88435', zorder=3)
    ax.axvline(0, color='#888888', lw=1); ax.set_yticks(range(len(names)), [label for _, label in names])
    ax.set_title('Paired place x seed 95% intervals'); ax.set_xlabel('Error reduction; *posthoc descriptive comparison')
    means = primary['mixed']['centered_log_mse']['means']; keys = list(means); ax = axes[1, 1]
    ax.bar([LABELS[k] for k in keys], [means[k] for k in keys], color=[COLORS[k] for k in keys])
    ax.set_title('Corrective: centered spatial error'); ax.tick_params(axis='x', rotation=30)
    means = intervention['cohorts']['primary51']['mixed']['log_mse']['means']; ax = axes[1, 2]
    ax.bar(['Soft intact', 'Content\npermuted', 'Uniform\nprior'], [means[k] for k in ('soft', 'content_permuted', 'uniform_prior')],
           color=['#d88435', '#888888', '#287e72']); ax.set_title('Corrective dependence: first seed only')
    fig.suptitle('51 previously used nontraining places; static/dynamic suffix denotes TRAINING memory')
    save_figure(fig, 'training_state_and_utility')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)
    for ax, kind in zip(axes[:2], ('content', 'soft')):
        for train in ('statictrain', 'dynamictrain'):
            key = f'{kind}_{train}'
            values = [analyses[policy]['cohorts']['primary51']['modes']['mixed']['log_mse']['means'][key] for policy in POLICIES]
            ax.plot(range(3), values, '-o', label=LABELS[key], color=COLORS[key])
        ax.set_xticks(range(3), ['Static', 'Corrective', 'Prefix']); ax.set_xlabel('Evaluation memory policy; same queries')
        ax.set_ylabel('Native depth log-MSE'); ax.set_title('C' if kind == 'content' else 'Soft'); ax.legend(fontsize=8)
    values = retention['cohorts']['primary51']['log_mse']['means']
    for kind in ('content', 'soft'):
        axes[2].plot([0, 1, 4], [values[f'{kind}_t{t}'] for t in (0, 1, 4)], '-o',
                     label=LABELS[f'{kind}_dynamictrain'], color=COLORS[f'{kind}_dynamictrain'])
    axes[2].set_title('Early query 5 revisited: first seed, full RGB'); axes[2].set_xlabel('Writes observed; retrospective')
    axes[2].set_xticks([0, 1, 4]); axes[2].set_ylabel('Native depth log-MSE'); axes[2].legend(fontsize=8)
    fig.suptitle('Training-state adaptation and evaluation-state range are distinct comparisons')
    save_figure(fig, 'state_range_and_retention')


def qualitative(corrective):
    fixed = load_plan()['fixed_figure_ids']; primary = corrective['cohorts']['primary51']
    gains = primary['modes']['mixed']['log_mse']['comparisons']['content_dynamictrain_minus_soft_dynamictrain']['place_gain']
    place = primary['places'][int(np.argmin(gains))]
    worst = next(m['id'] for m in load_plan()['catalog'] if m['place'] == place and not m['training'])
    chosen = list(dict.fromkeys(fixed + [worst]))
    write_json(OUT / 'figure_selection.json', {'fixed': fixed, 'posthoc_largest_soft_minus_content_place': place,
        'first_catalog_clip_in_that_place': worst, 'used_main_place_average': True, 'not_confirmation': True})
    cases = [('reused', f'{k}_{SEEDS[0]}_corrective_clean', LABELS[f'{k}_statictrain']) for k in ('content', 'soft')]
    cases += [('read', f'{k}_{SEEDS[0]}_corrective_clean', LABELS[f'{k}_dynamictrain']) for k in ('content', 'soft')]
    for rid in chosen:
        fig, axes = plt.subplots(2, 7, figsize=(18, 5.7), constrained_layout=True)
        for row, condition in enumerate((0, 1)):
            q = 11; sample = torch.load(BASE_CACHE / rid / f'q{q}_c{condition}.pt', map_location='cpu', weights_only=False)
            target = sample['target'].numpy(); valid = sample['valid'].numpy(); scale = sample['metric_scale']; assert scale is not None
            vmax = float(np.quantile(target[valid], .98)); arrays = [('Native', sample['query_depth_native'].numpy() * scale)]
            for phase, case, label in cases:
                record = read(f'evaluation/{phase}/{case}/{rid}.json')
                export = torch.load(record['prediction_path'], map_location='cpu', weights_only=False)
                index = export['sample_keys'].index([q, condition]); arrays.append((label, export['prediction_native'][index].numpy() * scale))
            axes[row, 0].imshow(sample['rgb'].numpy()); axes[row, 0].set_title('Full RGB' if condition == 0 else 'Masked RGB')
            axes[row, 1].imshow(np.where(valid, target, np.nan), vmin=0, vmax=vmax, cmap='viridis'); axes[row, 1].set_title('Sensor depth')
            for col, (name, depth) in enumerate(arrays, 2):
                axes[row, col].imshow(depth, vmin=0, vmax=vmax, cmap='viridis'); axes[row, col].set_title(name)
            for ax in axes[row]: ax.axis('off')
        label = ('fixed; also posthoc adverse' if rid in fixed else 'posthoc adverse place') if rid == worst else 'fixed example'
        fig.suptitle(f'{rid}: {label}; query 11; seed {SEEDS[0]}; all readers use Corrective memory; shared depth scale per row')
        save_figure(fig, f'example_{rid}', pdf=False)
    return chosen


def main():
    analyses = {policy: read(f'analysis_{policy}.json') for policy in POLICIES}
    intervention, retention, checks, training, condition = (read(name) for name in (
        'intervention_analysis.json', 'retention_analysis.json', 'independent_checks.json',
        'training_summary.json', 'conditional_decision.json'))
    assert checks['complete'] and analyses['corrective_clean']['center_evaluated'] == condition['center_control_triggered']
    plots(analyses, intervention, retention); selected = qualitative(analyses['corrective_clean'])
    decision_path = OUT / 'method_decision.json'; decision = read('method_decision.json') if decision_path.exists() else None
    primary = analyses['corrective_clean']['cohorts']['primary51']['modes']; main = primary['mixed']['log_mse']
    lines = ['# 训练状态匹配能否改善高斯记忆读取', '',
        decision['lead'] if decision else '固定实验数据与检查已完成；方法取舍与图像审阅待完成。', '',
        '`state_read1`，85 个已用片段，12 个训练地点；主评价为 51 非训练地点（63 片段），没有新确认。', '',
        '## 1. 唯一训练变化', '',
        '保持 137,472 参数读取器、VGGT 内部插入层、后续原生层和 DPT、Soft 公式、规则 writer、相机与尺度规则。只将训练时的初始 M0 替换为每个查询前的 Corrective 状态；C 和 Soft 各三个同初值、同样例序列、同 2,000 步最终模型。记忆本身不参与梯度。旧静态训练模型及原生/L 按身份复用。', '',
        '假设是初始状态训练与增量使用不匹配；竞争解释包括几何或更新误差、训练覆盖、位置偏好冗余，以及新写入图像已存在于原生窗口。主问题仍是匹配训练后 Soft 对 C 的独立价值，而不是只比它自己的旧版本。', '',
        '初始化 0–3，写入 4/6/8/10，查询 5/7/9/11 不写回。前三个查询的新写入帧都已在原生短窗口中，末次只有帧4在窗口外；本实验不能独立确立长历史用途。', '',
        '## 2. Corrective 主比较与训练适应性', '',
        '| 读取器及训练状态 | 混合 log-MSE | 完整 | 遮挡 | 混合空间误差 |', '|---|---:|---:|---:|---:|']
    for key in main['means']:
        lines += [f"| {LABELS[key]} | {num(main['means'][key])} | {num(primary['full']['log_mse']['means'][key])} | {num(primary['masked']['log_mse']['means'][key])} | {num(primary['mixed']['centered_log_mse']['means'][key])} |"]
    lines += ['', 'dynamic/static 后缀表示训练记忆；本节所有方法评价时均读同一 Corrective 状态。正差表示后者误差较低。旧 C 与新 Soft 的跨训练状态比较在首种子结果后明确列为描述性参照，用于避免把新 C 的退步当成方法进步；主比较和条件规则未变。', '']
    comparison_table(lines, main['comparisons'])
    inter = main['training_state_interaction']
    lines += ['', f"训练状态交互量 `(C-dynamic−Soft-dynamic)−(C-static−Soft-static)` 为 {num(inter['mean_gain'])}，地点×种子 95% 区间 {interval(inter)}，配对种子差为 {', '.join(num(x) for x in inter['paired_seed_gain'])}。正值表示训练匹配更有利于 Soft 相对 C，不能单独证明 Soft 已有净优势。", '',
        '| 批次 | Soft-dynamic 对 C-dynamic 降幅 | C 的训练状态收益 | Soft 的训练状态收益 |', '|---|---:|---:|---:|']
    for group in ('old12', 'used_summary20', 'used_writer19', 'training12'):
        c = analyses['corrective_clean']['cohorts'][group]['modes']['mixed']['log_mse']['comparisons']
        lines += [f"| {group} | {pct(c['content_dynamictrain_minus_soft_dynamictrain']['relative_reduction_percent'])} | {pct(c['content_statictrain_minus_content_dynamictrain']['relative_reduction_percent'])} | {pct(c['soft_statictrain_minus_soft_dynamictrain']['relative_reduction_percent'])} |"]
    lines += ['', '全部完整／遮挡、隐藏／可见、均值偏差、空间误差、AbsRel、米制 RMSE 和逐查询配对结果见 `analysis_*.json`。', '',
        '## 3. 状态范围、依赖与回访', '', '| 训练读取器 | Static | Corrective | Prefix |', '|---|---:|---:|---:|']
    for key in ('content_statictrain', 'content_dynamictrain', 'soft_statictrain', 'soft_dynamictrain'):
        values = [num(analyses[policy]['cohorts']['primary51']['modes']['mixed']['log_mse']['means'][key]) for policy in POLICIES]
        lines += [f"| {LABELS[key]} | {' | '.join(values)} |"]
    lines += ['', '三个评价策略使用相同查询；Prefix 是保存观察的固定容量重建，未联合运行全部 RGB，不是严格性能上界。', '',
              '首个 Soft-dynamic 的两个 Corrective 干预只检验依赖。正差表示干预误差更高，相对降幅以干预误差作分母。', '']
    comparison_table(lines, intervention['cohorts']['primary51']['mixed']['log_mse']['comparisons'])
    lines += ['', 'query5 完整输入的初始／首次／末次状态回访仅用首种子，是事后诊断，不并入因果主成绩。', '']
    comparison_table(lines, retention['cohorts']['primary51']['log_mse']['comparisons'])
    lines += ['', '## 4. 条件控制、检查与成本', '',
        ('主比较达到事前至少 1% 且至少两个种子改善的投入条件，追加三个 Soft-center 匹配控制；其结果见主表。这个门槛不是显著性或方法成功定义。' if condition['center_control_triggered'] else
         '主比较没有达到事前至少 1% 且至少两个种子改善的投入条件，未追加 Center；本轮不产生新的形状必要性结论。'), '',
        '原模型和投影证据按源码身份复用；新输入因果检查、两组零残差回放及各两步梯度／资源检查通过。正式训练逐步对齐父轮 C 初值和样本序列。', '',
        f"独立 NumPy 从原始标签复算 {checks['exported_predictions_recomputed']:,} 份新预测，核对 {checks['reused_parent_rows']:,} 行复用评分，无查询或片段排除。", '',
        f"新增 {training['fits']} 次正式训练，{training['steps']:,} 步；训练循环合计 {training['sum_worker_hours']:.3f} worker 小时，峰值已分配 {training['peak_allocated_GiB']:.3f} GiB。共享卡时间不是独占性能。算子未变，上一轮缓存后半网络成本按身份引用于 `baseline_reuse.json`，没有重复短测或声称新速度提升。", '',
        '## 5. 方法取舍', '']
    lines += [line for paragraph in decision['paragraphs'] for line in (paragraph, '')] if decision else ['等待完整结果和图像审阅；计算完成不等于方法通过验证。', '']
    lines += ['## 6. 图与来源', '', '![训练状态与用途](figures/training_state_and_utility.png)', '',
              '![状态范围与回访](figures/state_range_and_retention.png)', '']
    for rid in selected: lines += [f'![案例 {rid}](figures/example_{rid}.png)', '']
    lines += ['前三个案例事前固定，另一个为明确标注的最大主地点退步案例（重复则合并）。各行共同深度色阶；图像不替代统计。历史负结果、源码快照、输入身份与全分条件表均保留。', '']
    (OUT / 'report.md').write_text('\n'.join(lines))
    write_json(OUT / 'report_manifest.json', {'report_sha256': sha256(OUT / 'report.md'),
        'decision_sha256': sha256(decision_path) if decision else None,
        'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in (ROOT / 'experiments').glob('gaussian_memory_state_*.py')},
        'figures': {p.name: sha256(p) for p in (OUT / 'figures').glob('*')},
        'figure_review_pending': True, 'conclusion_review_pending': decision is None,
        'phase_complete': False, 'continuous_research_goal_active': True})
    print(json.dumps({'report_written': str(OUT / 'report.md'), 'review_pending': True}), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4); main()
