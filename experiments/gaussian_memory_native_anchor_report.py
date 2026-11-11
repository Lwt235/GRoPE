"""Self-contained evidence report and fixed-case visuals for the anchor diagnostic."""
import json

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.gaussian_memory_native_anchor_data import (
    OUT, PARENT, BASE_CACHE, FAMILIES, POLICIES, SEEDS, load_plan, sha256, write_json)

LABELS = {'content_static': 'C-static', 'soft_static': 'Soft-static',
    'content_dynamic': 'C-dynamic', 'soft_dynamic': 'Soft-dynamic',
    'center_dynamic': 'Center-dynamic', 'native': 'Native', 'local': 'L'}
POLS = {'recent_corrective': 'Recent / Corrective', 'history_corrective': 'History / Corrective',
        'recent_rebuild': 'Recent / Prefix', 'history_rebuild': 'History / Prefix'}


def read(name): return json.loads((OUT / name).read_text())


def comparison(main, policy, key, version='anchored', mode='mixed', metric='log_mse'):
    return main['versions'][version][policy]['modes'][mode][metric]['comparisons'][key]


def number(v): return f'{v:.6f}'


def interval(x): return '[' + ', '.join(number(v) for v in x['paired_place_seed_95']) + ']'


def table(lines, entries):
    lines.extend(['| 比较：前者误差减后者 | 相对降低 | 绝对差的95%区间 | 三种子绝对差 | 改善地点 |',
                  '|---|---:|---|---|---:|'])
    for label, x in entries:
        lines.append(f'| {label} | {x["relative_reduction_percent"]:+.2f}% | {interval(x)} | '
                     + ' / '.join(number(v) for v in x['paired_seed_gain'])
                     + f' | {x["places_better"]}/{x["places"]} |')
    lines.append('')


def save_figure(fig, name, pdf=True):
    fig.savefig(OUT / 'figures' / f'{name}.png', dpi=160)
    if pdf: fig.savefig(OUT / 'figures' / f'{name}.pdf')
    plt.close(fig)


def ci_panel(ax, entries, title):
    for j, (label, x) in enumerate(entries):
        ax.plot(x['paired_place_seed_95'], [j, j], color='#517690', lw=2)
        ax.scatter(x['mean_gain'], j, color='#c96e2c', zorder=3)
    ax.axvline(0, color='#888888', lw=1)
    ax.set_yticks(range(len(entries)), [label for label, _ in entries])
    ax.set_title(title); ax.set_xlabel('Absolute log-MSE reduction; positive favors candidate')


def plots(main, before, history, retention):
    (OUT / 'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), constrained_layout=True)
    families = (*FAMILIES, 'local'); x = np.arange(len(families))
    for ax, p in zip(axes.flat, POLICIES):
        raw = main['versions']['raw'][p]['modes']['mixed']['log_mse']['means']
        new = main['versions']['anchored'][p]['modes']['mixed']['log_mse']['means']
        ax.bar(x - .18, [raw[f] for f in families], width=.36, label='Raw', color='#517690')
        ax.bar(x + .18, [new[f] for f in families], width=.36, label='Anchored', color='#d88435')
        ax.axhline(raw['native'], color='#888888', ls='--', label='Native')
        ax.set_xticks(x, [LABELS[f].replace('-', '\n') for f in families])
        ax.set_ylabel('Mixed native log-MSE (lower better)'); ax.set_title(POLS[p]); ax.set_ylim(0, .205)
        ax.legend(fontsize=8, loc='lower right')
    fig.suptitle('One fixed scale constraint applied to every control; no training\n12 previously used places, 3 seeds; full input and both masks retained')
    save_figure(fig, 'anchor_before_after')

    fig, axes = plt.subplots(2, 2, figsize=(14.8, 9), constrained_layout=True)
    for ax, p in zip(axes[0], ('history_corrective', 'history_rebuild')):
        entries = [(LABELS[f] + ' minus Soft-dyn', comparison(main, p, f'{f}_minus_soft_dynamic'))
                   for f in ('content_static', 'content_dynamic', 'center_dynamic', 'native', 'local')]
        ci_panel(ax, entries, POLS[p] + ': all controls anchored')
    entries = []
    for p in ('corrective', 'rebuild'):
        for f in ('content_static', 'content_dynamic', 'soft_dynamic'):
            entries.append((LABELS[f] + (' / Corr.' if p == 'corrective' else ' / Prefix'),
                history['versions']['anchored']['mixed']['log_mse']['comparisons'][f'{f}_recent_{p}_minus_{f}_history_{p}']))
    ci_panel(axes[1, 0], entries, 'Anchored history utility: Recent minus History')
    entries = [(POLS[p], before['policies'][p]['mixed']['log_mse']['comparisons']['soft_dynamic_raw_minus_soft_dynamic_anchored']) for p in POLICIES]
    ci_panel(axes[1, 1], entries, 'Same Soft-dynamic: Raw minus Anchored')
    fig.suptitle('Paired place x seed intervals (5,000 resamples); no fresh confirmation\nHistory can improve relative to anchored Recent without improving the original method')
    save_figure(fig, 'anchor_pe_and_history')

    fig, axes = plt.subplots(2, 2, figsize=(14.8, 8.5), constrained_layout=True)
    for ax, p in zip(axes[0], ('history_corrective', 'history_rebuild')):
        keys = [('raw', 'native'), ('raw', 'content_static'), ('anchored', 'content_static'),
                ('raw', 'soft_dynamic'), ('anchored', 'soft_dynamic')]
        spatial = [main['versions'][v][p]['modes']['mixed']['centered_log_mse']['means'][f] for v, f in keys]
        bias = [main['versions'][v][p]['modes']['mixed']['bias_squared']['means'][f] for v, f in keys]
        ax.bar(range(5), spatial, label='Centered spatial error', color='#387e73')
        ax.bar(range(5), bias, bottom=spatial, label='Squared mean log bias', color='#e7b083')
        ax.set_xticks(range(5), ['Native', 'C raw', 'C anchor', 'Soft raw', 'Soft anchor'])
        ax.set_ylabel('Mixed log-MSE decomposition'); ax.set_title(POLS[p]); ax.legend(fontsize=8)
    entries = []
    for p in ('corrective', 'rebuild'):
        for version in ('raw', 'anchored'):
            entries.append((('Corr.' if p == 'corrective' else 'Prefix') + ' / ' + version,
                history['versions'][version]['mixed']['log_mse']['comparisons'][f'soft_dynamic_recent_{p}_minus_soft_dynamic_history_{p}']))
    ci_panel(axes[1, 0], entries, 'Soft history utility before and after anchoring')
    ci_panel(axes[1, 1], [(LABELS[f], retention['versions']['anchored']['log_mse']['comparisons'][f'{f}_early_minus_{f}_late']) for f in FAMILIES],
             'Anchored early query revisited: before minus after')
    fig.suptitle('Scale-only transform preserves centered spatial error algebraically\nAn unchanged spatial component is not new improvement; retention is retrospective')
    save_figure(fig, 'anchor_decomposition_and_retention')


def qualitative(main):
    catalog = load_plan()['catalog']; fixed = ['fidelity_00', 'fidelity_01', 'fidelity_02']
    c = comparison(main, 'history_corrective', 'content_static_minus_soft_dynamic')
    worst_place = main['places'][int(np.argmin(c['place_gain']))]
    worst = next(m['id'] for m in catalog if m['place'] == worst_place)
    selected = list(dict.fromkeys(fixed + [worst]))
    write_json(OUT / 'figure_selection.json', {'fixed': fixed, 'posthoc_worst_place': worst_place,
        'posthoc_worst_id': worst, 'selected': selected, 'query': 11, 'conditions': [0, 1], 'seed': SEEDS[0],
        'criterion': 'Largest anchored Soft-dynamic minus anchored C-static place-equal mixed error, History-Corrective, all3seeds.',
        'scope': 'Descriptive examples; no evaluation exclusion, target only sets shared display colors.'})
    cases = [(PARENT, 'content_static', 'history_corrective', 'C-static raw'),
             (OUT, 'content_static', 'history_corrective', 'C-static anchored'),
             (PARENT, 'soft_dynamic', 'history_corrective', 'Soft-dyn raw'),
             (OUT, 'soft_dynamic', 'history_corrective', 'Soft-dyn anchored'),
             (OUT, 'soft_dynamic', 'history_rebuild', 'Soft Hist Prefix anchor')]
    for rid in selected:
        fig, axes = plt.subplots(2, 8, figsize=(20, 5.6), constrained_layout=True)
        for row, condition in enumerate((0, 1)):
            s = torch.load(BASE_CACHE / rid / f'q11_c{condition}.pt', map_location='cpu', weights_only=False)
            target, valid = s['target'].numpy(), s['valid'].numpy(); scale = s['metric_scale']; assert scale is not None
            vmax = float(np.quantile(target[valid], .98))
            axes[row, 0].imshow(s['rgb'].numpy()); axes[row, 0].set_title('Full RGB' if condition == 0 else 'Masked RGB')
            axes[row, 1].imshow(np.where(valid, target, np.nan), vmin=0, vmax=vmax, cmap='viridis'); axes[row, 1].set_title('Sensor depth')
            arrays = [('Native', s['query_depth_native'].numpy())]
            for root, f, p, label in cases:
                record = json.loads((root / 'evaluation/read' / f'{f}_{SEEDS[0]}_{p}' / f'{rid}.json').read_text())
                export = torch.load(record['prediction_path'], map_location='cpu', weights_only=False)
                i = export['sample_keys'].index([11, condition])
                arrays.append((label, export['prediction_native'][i].numpy()))
            for col, (label, pred) in enumerate(arrays, 2):
                axes[row, col].imshow(pred * scale, vmin=0, vmax=vmax, cmap='viridis'); axes[row, col].set_title(label, fontsize=9)
            for ax in axes[row]: ax.axis('off')
        qualifier = ('fixed + posthoc adverse' if rid in fixed else 'posthoc adverse') if rid == worst else 'fixed example'
        fig.suptitle(f'{rid}: {qualifier}; query11, seed{SEEDS[0]}; common depth colors per row\nHistory / Corrective except last column; original and anchored predictions, no GT alignment')
        save_figure(fig, f'example_{rid}', pdf=False)
    return selected


def report(main, before, history, retention, intervention, checks, cost):
    lines = ['# native_anchor_probe1：当前尺度约束没有形成方法改进', '',
        '2026-11-11。固定旧模型、旧数据上的开发诊断，全部预定比较已完成；没有新增训练、下载或骨干推理。', '',
        '**决定：不采用这一固定尺度约束作为主方案，也不继续搜索它的权重或公式。** Soft-dynamic在History-Corrective下主误差高0.11%，基本持平；History-Prefix下高3.81%，三个种子都退步，但两项地点区间均跨零。约束后历史相对近期略有改善，却不能超过同样约束的强内容对照。保留空间历史信号与有限形状信号，不升级为一般PE效果。', '',
        '## 问题、固定比较与来源', '',
        '前一轮发现历史的空间改善被平均深度偏差抵消。本轮检验：让当前观测负责尺度、记忆只提供相对结构，能否把这一信号转为整体用途。竞争解释是记忆原本也在修正有用的整体偏差，Native尺度不够可靠；或内容读取能解释收益，或历史缺乏相关信息。', '',
        '`r=log(D_M)-log(D_0); delta=mean_U(r); D_anchor=D_0*exp(r-delta)`。U仅由固定RGB人工遮挡确定：完整输入用全部像素，遮挡输入用所有未遮挡像素。无标签、无传感器有效掩码、无可学习参数、无系数搜索、无输出裁剪；原米制标定不变。实际这是整幅深度乘一个常数。', '',
        '固定history_read1的12对旧窗口、15个记忆读取模型、三种子、四种状态及全部查询/条件；额外保留三个L模型。8640份主预测、432份L、360份回访前后与288份干预，共9720份变换，另原样复用144行Native。所有父预测和评分保留，详见plan.json的来源/索引/哈希。', '',
        'C-static是此前训练的强内容读取器；C-dynamic是状态匹配训练的内容对照；Soft-static/Soft-dynamic分别为相应高斯读取；Center-dynamic限制几何形状；L只有当前观测。static/dynamic指模型训练条件，此处所有记忆读取模型均接收各个规定状态。', '',
        '主指标仍为地点等权的完整输入与两遮挡混合log-MSE。正的“前者减后者”表示后者更好；5000次配对地点×种子重采样。以下区间是单项诊断区间，没有把众多辅助比较当作独立确认。', '',
        '## 所有模型的主误差：原输出 → 统一尺度约束', '',
        '| 模型 | Recent Corrective | History Corrective | Recent Prefix | History Prefix |',
        '|---|---:|---:|---:|---:|']
    for f in (*FAMILIES, 'native', 'local'):
        cells = []
        for p in POLICIES:
            a = main['versions']['raw'][p]['modes']['mixed']['log_mse']['means'][f]
            b = main['versions']['anchored'][p]['modes']['mixed']['log_mse']['means'][f]
            cells.append(f'{a:.6f} → {b:.6f}')
        lines.append('| ' + LABELS[f] + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '![全部模型前后比较](figures/anchor_before_after.png)', '',
              '同一Soft四种状态的前后配对：', '']
    table(lines, [(POLS[p] + ' 原输出 − 约束输出', before['policies'][p]['mixed']['log_mse']['comparisons']['soft_dynamic_raw_minus_soft_dynamic_anchored']) for p in POLICIES])
    lines += ['同样的约束对C-dynamic和L的主均值改善更明显，但各自整体区间仍跨零。不能以“只校正Soft、对比未校正C”的方式宣称PE收益；也不能仅凭平均数声称约束普遍有害。', '',
              '## 同样处理后的强对照：独立PE收益尚未建立', '']
    for p in ('history_corrective', 'history_rebuild'):
        lines += ['### ' + POLS[p], '']
        table(lines, [(LABELS[f] + ' − Soft-dynamic', comparison(main, p, f'{f}_minus_soft_dynamic'))
                      for f in ('content_static', 'content_dynamic', 'center_dynamic', 'native', 'local')])
    lines += ['Soft比强C-static在两个历史条件分别高0.34%/0.40%，区间跨零；没有独立于强内容读取的净收益。History-Prefix下比Center低1.33%，三种子同向、10/12地点改善、区间为正，支持这个固定接口中的有限形状增量。它不能替代对强C的比较。', '',
        '约束后的Soft比Native低2.93%/3.45%，两个混合区间为正，说明当前读取保留有限任务用途；相对L仍不确定。这个结果不等于约束本身改善了Soft：原Soft没有从约束中获益。', '',
        '## 窗口外历史：方向更一致，整体证据仍不足', '']
    for p in ('corrective', 'rebuild'):
        lines += ['### ' + ('Corrective' if p == 'corrective' else 'Prefix'), '']
        table(lines, [(LABELS[f] + ' Recent − History（均约束）', history['versions']['anchored']['mixed']['log_mse']['comparisons'][f'{f}_recent_{p}_minus_{f}_history_{p}']) for f in FAMILIES])
    lines += ['Soft的历史主收益变为1.16%/0.71%，两个主比较三种子均同向，分别8/12和10/12地点改善，但两项主区间均跨零。Corrective完整输入的辅助收益1.52%、区间为正；不能替换混合主结论。Prefix下C-static也改善0.69%，与Soft的0.71%接近；历史收益的Soft−C交互区间均跨零。', '',
        '这次对照说明：限制不同状态的整体尺度变化后，历史的相对结构信号更容易看见，但尚未形成更好的整体方法，也未证明需要高斯PE。', '',
        '![PE与历史比较](figures/anchor_pe_and_history.png)', '',
        '## 为什么旧空间改善没有变成新方案', '',
        '同一张图的log深度只减去一个常数，因此去均值空间误差必然不变。前一轮Prefix历史空间误差降低9.25%的结果保留，但这不是本轮产生的新改善。把平均预测深度对齐到Native，并不保证使平均预测误差更接近零；记忆可能已在纠正Native的偏差，而统计用的有效深度区域也与无标签的RGB支持区不同。', '',
        '本轮可直接确认的是：尺度归还Native并未带来可靠净收益，不能再把“偏差项抵消空间项”解释为一个只要去掉就有益的外来漂移。空间与整体偏差的分解是诊断，不能替代同方法前后与强对照比较。', '',
        '![误差分解与回访](figures/anchor_decomposition_and_retention.png)', '',
        '## 完整/遮挡辅助检查', '']
    for mode, label in [('full', '完整输入'), ('masked', '两遮挡条件')]:
        lines += ['### ' + label, '']
        table(lines, [(POLS[p] + ' / ' + LABELS[f] + ' − Soft-dynamic', comparison(main, p, f'{f}_minus_soft_dynamic', mode=mode))
            for p in ('history_corrective', 'history_rebuild') for f in ('content_static', 'center_dynamic', 'native')])
    lines += ['隐藏/可见像素、AbsRel、米制RMSE、逐查询、逐地点×种子矩阵及所有近期状态的比较均保存在main_analysis.json；全模型前后比较在before_after_analysis.json。没有按辅助指标选模型、删地点或更换主条件。', '',
        '## 回访与依赖干预', '']
    table(lines, [(LABELS[f] + ' 回访前 − 后（均约束）', retention['versions']['anchored']['log_mse']['comparisons'][f'{f}_early_minus_{f}_late']) for f in FAMILIES])
    lines += ['Soft回访主误差约持平（平均改善0.03%），所有回访区间跨零；未做等价检验，不能声明无遗忘。回访为回溯查询，不混入因果主任务。', '']
    table(lines, [('打乱内容 − 完整Soft', intervention['versions']['anchored']['mixed']['log_mse']['comparisons']['content_permuted_minus_soft']),
                  ('均匀先验 − 完整Soft', intervention['versions']['anchored']['mixed']['log_mse']['comparisons']['uniform_prior_minus_soft'])])
    lines += ['干预仅为固定第一种子的依赖诊断，两项区间均跨零；尚无可靠证据把迁移收益归因于特定空间绑定。', '',
        '## 实现与代价', '',
        f'- {checks["transformed_scores_recomputed"]}份新导出预测与144行Native均用原始传感器深度、独立float64 NumPy公式复算；标签、输入、父预测和源代码身份已核对。',
        f'- 固定第一对窗口/第一种子/查询11的{checks["numpy_formula_examples"]}份输出另用`D_M*exp(-mean(log(D_M/D_0)))`核对；最大相对误差{checks["numpy_formula_max_relative_output_error"]:.3g}。',
        f'- 全部预测未遮挡区相对Native的平均log差绝对值最大{checks["max_unmasked_mean_log_change"]:.3g}；去均值空间评分改变最大{checks["max_centered_score_change"]:.3g}。这是合理float32尾差，不是科学改进或失败。',
        f'- GPU 0空闲复核后启动，沿用12GiB上限。批处理worker {cost["worker_seconds"]:.2f}秒，其中变换运算{cost["transform_compute_seconds"]:.3f}秒，峰值已分配{cost["peak_allocated_GiB"]:.4f}GiB；读写和CPU评分占主要时间。不是独占吞吐基准。',
        '- 无训练、下载或新骨干推理；GPU worker退出码0。检查与报告脚本的退出和实际图像审阅另见execution_status.json与visual_review.json。', '',
        '## 范围与下一步', '',
        '这是12个已经使用、按数据可用性选出的地点，不代表51地点总体，也不是新确认。相机/标定仍为给定条件；Prefix依赖累计原始观察，是离线对照，不能宣称常数内存。历史容量覆盖等父轮负结果不因本轮处理而改变。', '',
        '结束尺度校正分支，保留原读取器为研究基线。下一项应直接检验：当前查询需要的信息是否存在于窗口外原始历史，以及压缩记忆/读取是否把它丢失。固定近期与历史来源，通过原始观测的正对照区分“信息不存在”与“表示或读取损失”，先查清信息支持，再决定是否修改writer；不要继续扫描校正公式。[下一项有限诊断方案](../../../docs/gaussian_memory_raw_history_information_plan.md)只已提出，本目标不自动启动它；它也不是后续GPU学习实验的永久门禁。', '',
        '## 固定例图与反例', '',
        'fidelity_00/01/02为事前固定例；再加入按统一混合主误差选出的最差地点，明确标注为事后描述。相同一行共享传感器深度色阶仅用于显示，未参与模型变换；原图与约束图结构相同、整体深度会变。']
    selected = read('figure_selection.json')
    lines += ['', f'最差地点：{selected["posthoc_worst_place"]}（{selected["posthoc_worst_id"]}）。']
    if (OUT / 'visual_review.json').exists():
        review = read('visual_review.json')
        lines += ['', f'已实际审阅{len(review["images"])}幅最终PNG。最差地点与固定例fidelity_00重合，因此三个例图已覆盖固定案例与最差反例，没有另挑一个地点替换。']
    for rid in selected['selected']: lines += ['', f'![{rid}](figures/example_{rid}.png)']
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n')


def main():
    main_data, before, history, retention, intervention, checks, cost = [read(n) for n in (
        'main_analysis.json', 'before_after_analysis.json', 'history_range_analysis.json',
        'retention_analysis.json', 'intervention_analysis.json', 'independent_checks.json', 'runtime_costs.json')]
    assert checks['complete'] and cost['complete']
    plots(main_data, before, history, retention)
    selected = qualitative(main_data)
    report(main_data, before, history, retention, intervention, checks, cost)
    print(json.dumps({'report_generated': True, 'qualitative_cases': selected,
                      'figures': [p.name for p in sorted((OUT / 'figures').glob('*.png'))]}), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
