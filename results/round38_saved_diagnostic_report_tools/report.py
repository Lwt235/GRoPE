"""Report accepted saved-output diagnosis; JSON/hash only, no new arrays."""
from pathlib import Path
import sys
import time

ROOT = Path('/home/liwt/GRoPE')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments import round38_rms_anchor_contract as C
from experiments import round38_saved_output_diagnostic as P
from experiments import round38_saved_output_diagnostic_audit as A
import numpy as np


def main():
    C.configure_runtime()
    def reject(*args, **kwargs):
        raise RuntimeError('Report cannot decode arrays')
    np.load = reject
    before = C.read(HERE/'before_execution.json')
    assert C.sha(__file__) == before['controller_sha256']
    for name,digest in before['files'].items():
        assert C.sha(ROOT/name)==digest,name
    P.verify_before(P.DEFAULT_OUT)
    for directory in (P.DEFAULT_OUT,P.DEFAULT_AUDIT,ROOT/'results/round38_saved_diagnostic_execution'):
        completed = C.read(directory/'completion.json')
        assert completed['passed'] is True
        C.verify_inventory(directory,completed['files'],exclude=('completion.json',))
    producer = C.read(P.DEFAULT_OUT/'completion.json')
    audit = C.read(P.DEFAULT_AUDIT/'completion.json')
    assert producer['completed_queries']==audit['completed_queries']==2304
    assert audit['producer_completion_sha256']==C.sha(P.DEFAULT_OUT/'completion.json')
    summary = C.read(P.DEFAULT_OUT/'summary.json')
    assert not A.compare(summary,C.read(P.DEFAULT_AUDIT/'summary.json'))
    checks = C.read(P.DEFAULT_AUDIT/'checks.json')
    assert len(checks)==2304 and all(c['passed'] for c in checks)
    all_rows = summary['all']
    eligible = all_rows['categories']['projection.primary_anchor_defined']['True']
    raw_missing = all_rows['stage_counts']['raw']['anchor_set_contains']['False']
    final_missing = all_rows['stage_counts']['probability']['anchor_set_contains']['False']
    transitions = all_rows['categories']['content.set_transition']
    floor_changes = all_rows['arithmetic_events']['raw_to_floor_winner_changed']['True']
    floor_changes_eligible = sum(g['arithmetic_events']['raw_to_floor_winner_changed'].get('True',0)
                                 for g in summary['set_transition'].values())
    assert floor_changes_eligible == 0
    assert sum(transitions[k] for k in ('00','01','10','11')) == eligible
    assert all_rows['stage_counts']['raw']==all_rows['stage_counts']['floor']
    table=['| 场景 | 全部查询 | 可比预测中心 | raw 最大格不含中心 | 最终最大格不含中心 | 内容移出 / 移回 |',
           '|---|---:|---:|---:|---:|---:|']
    for scene,g in summary['by_scene'].items():
        cats=g['categories'];t=cats['content.set_transition']
        table.append(f"| {scene} | {g['population']} | {cats['projection.primary_anchor_defined']['True']} | "
                     f"{g['stage_counts']['raw']['anchor_set_contains']['False']} | "
                     f"{g['stage_counts']['probability']['anchor_set_contains']['False']} | {t['10']} / {t['01']} |")
    report=ROOT/'docs/round38_saved_output_diagnostic_report.md'
    receipt=ROOT/'results/round38_saved_output_diagnostic_root_acceptance_20261101.json'
    assert not report.exists() and not receipt.exists() and not (HERE/'report_completion.json').exists()
    text=f'''# R38 失败后诊断：位移已存在于位置质量，不能简单归咎于内容

全部 2304 个封存查询完成，独立实现逐行与汇总核验通过。结果支持的有限结论是：在能比较预测锚点格的 {eligible} 行中，**{raw_missing} 行（{100*raw_missing/eligible:.4f}%）的原始位置质量最大格已不包含预测中心**。原内容将 {transitions['10']} 行移出、将 {transitions['01']} 行移回中心格，最终不含中心的行数为 {final_missing}。因此“偏移完全由内容引起”不符合这些保存输出；没有证据据此优先调整内容温度。

## 全人口与判定对象

原顺序 48 对、三个场景各 768 查询全部保留。2301 个锚点在相机前方，3 个在后方；2304 个投影都有限。1506 个位于视野内部，其余 798 个在视野外，包括上述 3 个后方锚点。没有恰落网格边界的锚点，全部为 variable-depth 分支，没有结构零。

“可比预测中心”要求前方、有限投影、处于闭 FOV、非结构零；不适用的 798 行仍保留一般 winner 和状态诊断。这 1506 行**不是**原 PCK 的 1096 个有效真值中心，不能据此重定义任务人口或推算新的 PCK。

最大格指封存 native cell 的积分质量；它不是连续密度 mode。距离取预测中心到获胜矩形的最近距离，不把到网格中心的量化误差当作格外位移。记录保留精确并列集合和 first-index 选择；本次主锚点集合与 first-index 计数一致。

{chr(10).join(table)}

## 内容、floor 与物化的分开结果

全人口中，原内容改变了 {all_rows['categories']['content.winner_changed']['True']}/2304 行的 first winner。在可比人口内，完整 2×2 表为：两者都含中心 {transitions['11']}，仅加内容前含中心 {transitions['10']}，仅加内容后含中心 {transitions['01']}，两者都不含中心 {transitions['00']}。这描述相对**预测中心**的移入/移出，不是相对 GT 的收益/损害；也不能从净移回 7 行断言内容改善了真实 PCK。

单次等面积 floor 使全人口 {floor_changes} 行的 first winner 改变、99 行的最大值集合改变；这些变化全部在 798 行主锚点不适用人口中。原因归类为保存数值向量加 floor 后的并列/次序变化，不能误归给内容，也不能把它称为正 raw 质量被证明为零。可比 1506 行的 raw/floor winner 没有变化。logits→log-P 和 log-P→普通 P 均无 winner 改变。

raw 锚点 log-gap 的最大值为 {all_rows['numerics']['stages.raw.anchor_log_gap']['max']:.9f}；到 raw 获胜矩形距离在这 1506 行中的 p90={all_rows['numerics']['stages.raw.rectangle_distance_px']['p90']:.6f}px、p99={all_rows['numerics']['stages.raw.rectangle_distance_px']['p99']:.6f}px、最大值={all_rows['numerics']['stages.raw.rectangle_distance_px']['max']:.6f}px。大部分行的距离为零，不能把所有格外位移都解释成超过 PCK 的 8px 容差。

## 证据和下一步约束

事前冻结的 10 个边界 toy 与禁用真实加载/模型的双 metadata 冻结均通过。诊断耗时 {producer['phase_wall_seconds']:.3f}s，独立复算 {audit['phase_wall_seconds']:.3f}s，均低于各自 120s。两路分别核验并读取 2352 个原封存档案、21024 个白名单字段；没有读取新标签、重跑模型或积分、扫描参数或产生新的任务指标。每条源身份、状态、并列、2×2 计数精确一致，连续统计按冻结容差复核。

该检查不能区分 RMS 横向正则、前方条件、透视非线性和有限网格积分各自的因果贡献；也不能用 raw winner 的反例定位连续 mode。完整独立概率参考仍未通过，因此结论只针对已保存的数值实现。R38 原八门 0/8 及停止决定不变。

接下来优先审查几何表示如何定义 query/key 支撑与可加区域质量，暂不把温度修补或偏斜分布当作已选方案。任何新构造先做数学合法性、与旧失败方案的等价性及同信息对照审查，再另行冻结实验；没有新候选通过全部要求。

协议：[失败后诊断](../ROUND38_POSTFAILURE_DIAGNOSTIC_PROTOCOL.md)。完整 [2304 行 JSON](../results/round38_saved_output_diagnostic/rows.json)、[CSV](../results/round38_saved_output_diagnostic/rows.csv)、[汇总](../results/round38_saved_output_diagnostic/summary.json)、[独立检查](../results/round38_saved_output_diagnostic_audit/checks.json)、[根验收](../results/round38_saved_output_diagnostic_root_acceptance_20261101.json)。
'''
    report.write_text(text)
    files=dict(before['files'])
    for p in (report,Path(__file__),HERE/'before_execution.json'):
        files[str(p.relative_to(ROOT))]=C.sha(p)
    C.write(HERE/'report_completion.json',dict(created_unix=time.time(),files=files,
        new_array_decodes=0,new_model_calls=0,new_task_metrics_computed=False,
        scientific_gate_passed=False,goal_complete=False))
    files['results/round38_saved_diagnostic_report_tools/report_completion.json']=C.sha(HERE/'report_completion.json')
    C.write(receipt,dict(created_unix=time.time(),accepted_saved_output_diagnostic=True,
        completed_queries=2304,independent_statistics_agreed=True,eligible_anchor_queries=eligible,
        raw_winning_cell_excludes_anchor=raw_missing,content_moves_out=transitions['10'],
        content_moves_in=transitions['01'],final_winning_cell_excludes_anchor=final_missing,
        numerical_probability_reference=False,continuous_density_mode_established=False,
        new_task_metrics_computed=False,scientific_gate_passed=False,goal_complete=False,files=files))
    print('SAVED_DIAGNOSTIC_REPORT_ACCEPTED',C.sha(receipt),flush=True)


if __name__=='__main__':
    main()
