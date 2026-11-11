"""Numerical evidence report; final interpretation is explicitly authored separately."""
import json

from experiments.gaussian_memory_visibility_training_data import OUT, KINDS, SEEDS, CONDITIONS, load_plan

LABELS = {'full': '完整输入', 'query_only': '仅查询遮挡', 'persistent': '持续遮挡'}
NAMES = {'new_soft': '新 Soft', 'new_content': '新 C', 'new_center': '新 Center',
         'content_static': '强旧 C-static', 'content_dynamic': '旧 C-dynamic',
         'soft_dynamic': '旧 Soft-dynamic', 'soft_static': '旧 Soft-static',
         'center_dynamic': '旧 Center-dynamic', 'native': 'Native', 'local': '旧 Local'}


def line(label, r):
    lo, hi = r['relative_gain_percent_95']
    seeds = '/'.join(f'{x:+.2f}' for x in r['seed_relative_gain_percent'])
    return (f'| {label} | {r["baseline_mean"]:.6f} | {r["candidate_mean"]:.6f} | '
            f'{r["relative_gain_percent"]:+.2f}% [{lo:+.2f}, {hi:+.2f}] | {seeds} | {r["positive_places"]}/12 |')


def table(comparisons, keys):
    return '\n'.join(['| 比较（后者为候选） | 基准误差 | 候选误差 | 相对降低及95%区间 | 三种子降低% | 改善地点 |',
                      '|---|---:|---:|---|---|---:|'] + [line(label, comparisons[key]) for key, label in keys])


def main():
    a = json.loads((OUT / 'analysis.json').read_text()); plan = load_plan()
    checks = json.loads((OUT / 'independent_checks.json').read_text())
    selected = json.loads((OUT / 'figure_selection.json').read_text())
    discussion = (OUT / 'interpretation.md').read_text() if (OUT / 'interpretation.md').exists() else '解释尚待完成。'
    cost = a['cost']; input_diag = json.loads((OUT / 'training_input_diagnostic.json').read_text())
    primary = [('new_soft:history_vs_recent', '新 Soft Recent → History'),
               ('history:new_soft_vs_new_content', 'History 新 C → 新 Soft'),
               ('history:new_soft_vs_content_static', 'History 强旧 C → 新 Soft'),
               ('history:new_soft_vs_new_center', 'History 新 Center → 新 Soft')]
    text = ['# history_visibility_train1：一次任务匹配训练检验', '', discussion, '',
        '## 问题与固定方法', '',
        '假设：既有读取器没有见过持续缺失与不同历史范围，任务匹配训练可能恢复历史利用和高斯形状的额外收益。竞争解释：预测或压缩记忆的信息不足，或者普通内容/中心读法已足够。仅改变训练分布，结构、PE公式和骨干不改。', '',
        '当前接法是在冻结 VGGT 第11个全局块处加入一次记忆交叉注意力，256个槽存放预测的高斯中心、协方差与32维内容。Soft 的空间偏好是 `log(1/256 + w_ij)`，其中 `w_ij` 是由高斯投影和前后遮挡累计得到的逐槽权重；在注意力分数中与内容Q/K相加。C的空间偏好为常数；Center用原训练统计的固定球形方差代替每个槽的协方差。三者共享137,472个可训练读取参数，全部几何和VGGT冻结。这里的位置项绑定同一内容槽，Soft始终保留对每个槽的读取能力。', '',
        '## 数据、训练与评价', '',
        '- 原12个训练地点，query5/7/9/11；0–3帧为完整早期历史，查询前的4/6/8/10为近期来源。每个查询独立选定固定mask0区域，用早期来源传感器点投影构造持续遮挡；任何晚期编码中的早期RGB副本也遮挡。查询目标深度不用于选择掩码。',
        '- 三种可见性条件 × Recent/History两范围，共288种训练样本。9次拟合（三方法×三种子），2000步，有效batch2，AdamW 3e-4/weight_decay .01，梯度裁剪5，全图原生log-MSE；同种子完整初始化和每一步样本完全配对。不增加ROI损失或蒸馏，不用评价成绩选模。',
        '- 评价仍为全部12个fidelity地点、query11、三条件和两范围，新增648份预测；原1224份输出及强旧C等对照原样保留。训练与评价地点分离，但评价地点已参与先前研究设计，属于开发验证。',
        f'- 训练输入提取172个窗口，四次旧输入重放完全一致；48个训练查询中{input_diag["zero_selected_queries"]}个没有早期点落入ROI，近期来源平均遮挡{100*input_diag["mean_mask_fraction_of_late_source_patches"]:.2f}%的patch。评价干预和来源支持数据沿用上一轮。',
        '- 所有条件共享完整输入的原尺度标定和已知相机。指标未做目标深度对齐。地点等权，三种子等权，5000次地点与种子联合配对重采样；区间是开发分析，不能替代新数据确认。', '',
        '## 主比较：持续遮挡下固定ROI原生log-MSE', '',
        '正的相对降低表示候选误差更低。History和Recent均为256槽；加入历史同时改变槽内来源与压缩分配。', '',
        table(a['conditions']['persistent']['hidden_log_mse']['comparisons'], primary), '',
        '### 强参照和自身改进', '',
        table(a['conditions']['persistent']['hidden_log_mse']['comparisons'], [
            ('history:new_soft_vs_soft_dynamic', '旧 Soft-dynamic → 新 Soft'),
            ('history:new_content_vs_content_static', '强旧 C-static → 新 C'),
            ('history:new_content_vs_content_dynamic', '旧 C-dynamic → 新 C'),
            ('history:new_soft_vs_native', 'Native → 新 Soft History'),
            ('history:new_soft_vs_local', '旧 Local → 新 Soft History')]), '',
        '## 三可见性条件与历史增量', '']
    for condition in CONDITIONS:
        comp = a['conditions'][condition]['hidden_log_mse']['comparisons']
        text += [f'### {LABELS[condition]}', '', table(comp, [
            ('new_content:history_vs_recent', '新 C Recent → History'),
            ('new_center:history_vs_recent', '新 Center Recent → History'),
            ('new_soft:history_vs_recent', '新 Soft Recent → History'),
            ('history:new_soft_vs_content_static', '强旧 C History → 新 Soft History'),
            ('history:new_soft_vs_new_center', '新 Center History → 新 Soft History')]), '']
    text += ['### 历史增量的交互', '', '| 比较 | 绝对log-MSE增量及95%区间 |', '|---|---|']
    inter = a['history_gain_interactions']['hidden_log_mse']
    for key, label in [('new_soft:persistent_minus_query_only_history_gain', '新Soft：持续减仅查询的历史增益'),
                       ('new_soft:persistent_minus_full_history_gain', '新Soft：持续减完整的历史增益'),
                       ('persistent:new_soft_minus_new_content_history_gain', '持续条件：新Soft减新C的历史增益'),
                       ('persistent:new_soft_minus_new_center_history_gain', '持续条件：新Soft减新Center的历史增益'),
                       ('persistent:new_soft_minus_soft_dynamic_history_gain', '持续条件：新Soft减旧Soft的历史增益')]:
        r = inter[key]; lo, hi = r['paired_95']
        text.append(f'| {label} | {r["mean"]:+.6f} [{lo:+.6f}, {hi:+.6f}] |')
    text += ['', '## 全图、可见区与误差来源', '',
        '全图log-MSE等于去均值空间误差加整体log深度偏差平方。它们共同解释误差来源，不能把偏差减少写成更精细的空间几何恢复。', '']
    for metric, title in [('log_mse', '全图'), ('visible_log_mse', '可见区'),
                          ('centered_log_mse', '去均值空间误差'), ('bias_squared', '整体偏差平方'),
                          ('absrel', 'AbsRel'), ('rmse_m', '米制RMSE')]:
        text += [f'### {title}：持续遮挡', '', table(a['conditions']['persistent'][metric]['comparisons'], primary), '']
    text += ['## 训练、实现与资源', '', '| 方法 | 种子 | 前20步损失 | 后20步损失 | 耗时秒 | 峰值GiB |', '|---|---:|---:|---:|---:|---:|']
    for kind in KINDS:
        for seed in SEEDS:
            r = a['training'][f'{kind}_{seed}']['status']
            text.append(f'| {kind} | {seed} | {r["first20"]:.6f} | {r["last20"]:.6f} | {r["seconds_this_invocation"]:.2f} | {r["peak_GiB"]:.3f} |')
    text += ['', f'正式训练共18,000步、36,000次样本呈现，累计{cost["training_worker_hours"]:.3f} worker小时，峰值{cost["peak_training_GiB"]:.3f} GiB。另有三方法各两步接口/资源预检，未进入正式模型。GPU 2/3/4/5/7每卡一个训练进程，逐次启动前检查占用，12GiB上限；无下载。', '',
        f'全部648份新预测由独立NumPy公式评分，1224行父轮记录与预测身份复用核对；192份训练状态和48组掩码核对，三方法三种子的完整初始化与2000步采样顺序一致。数值评分最大差：`{json.dumps(checks["max_score_difference"], ensure_ascii=False)}`。实现正确性只保证比较可解释，不保证方法有效。', '',
        '## 图像与反例', '',
        '固定展示前三个fidelity片段及新Soft对强旧C区域误差最大的地点（按三种子均值，字母序处理并列）。示例图只取首种子，不替代完整配对统计。', '',
        '![训练曲线](figures/training_curves.png)', '', '![原生任务比较](figures/native_results.png)', '',
        '![地点差异](figures/place_effects.png)', '']
    for rid in selected['selected_ids']: text += [f'![{rid}](figures/example_{rid}.png)', '']
    text += ['## 范围与可复现记录', '',
        '本轮只回答固定Prefix接法在受控可见性分布上，匹配训练能否改善读取。已知相机、完整输入标定、源传感器构造干预、冻结预测几何和旧开发地点都是边界。它不证明严格纯RGB在线系统、真实遮挡、一般位置编码收益、几何写入学习或长期记忆；也不抹去先前区域匹配的正面证据和原生推广的负结果。', '',
        '- [固定计划](plan.json)与[source snapshot](source_snapshot/)记录代码、数据、模型和父轮身份。',
        '- [输入核对](input_checks.json)、[独立评分与配对核对](independent_checks.json)、[完整指标与区间](analysis.json)。',
        '- [方法取舍](method_decision.json)、[实际图像审阅](visual_review.json)与[完成记录](phase_completion.json)。', '']
    (OUT / 'report.md').write_text('\n'.join(text))


if __name__ == '__main__': main()
