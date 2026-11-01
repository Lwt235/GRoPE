"""Describe sealed R38 results without opening NPZ archives or rerunning models."""
from pathlib import Path
import csv
import hashlib
import json
import sys
import time

ROOT=Path('/home/liwt/GRoPE')
sys.path.insert(0,str(ROOT))
from experiments import round38_rms_anchor_contract as C
from experiments import round38_rms_anchor_evaluate as E

def percent(x):
    return 100*float(x['numerator'])/float(x['denominator'])

def main():
    run,audit=C.DEFAULT_OUT,C.DEFAULT_AUDIT
    C.configure_runtime()
    E.require_prelabel(run,audit)
    summary=C.read(run/'evaluation/summary.json')
    screen=C.read(run/'evaluation/screen.json')
    verification=C.read(run/'metrics_audit/verification.json')
    assert verification['passed'] and verification['eligible_for_promotion'] is False
    assert screen['passed']==verification['exploratory_eight_checks_passed']
    root=C.read(ROOT/'results/round38_rms_anchor_prelabel_root_acceptance_20261101.json')
    assert root['accepted_for_exploratory_label_stage'] and root['completed_queries']==2304
    for directory in (run/'evaluation',run/'metrics_audit'):
        completed=C.read(directory/'completion.json')
        C.verify_inventory(directory,completed['files'],exclude=('completion.json',))
        started=C.read(directory/'started.json')
        assert completed['passed'] is True
        assert completed['arithmetic_audit_sha256']==C.sha(audit/'forward_verification.json')
        assert started['prelabel_root_sha256']==C.sha(ROOT/'results/round38_rms_anchor_prelabel_root_acceptance_20261101.json')
    old=C.read(ROOT/'results/round20_anchor_fixed/summary.json')
    f=C.read(run/'forward/completion.json')
    rows=C.read(run/'forward/records.json')
    v=C.read(audit/'forward_verification.json')
    iv=C.read(audit/'input_verification.json')
    out=ROOT/'docs/round38_rms_anchor_triage_report_data'
    report=ROOT/'docs/round38_rms_anchor_triage_report.md'
    assert not out.exists() and not report.exists(), 'preserve prior report'
    out.mkdir()
    source=Path(__file__)
    (out/'report_source.py').write_bytes(source.read_bytes())
    region=summary['region']['macro'];center=summary['center']['macro']
    quad=old['old_by_role']['strong_quad']['by_condition']['g24_g8']['kl']
    old_gauss=old['by_role']['anchor_trace']
    gaussian=old_gauss['by_condition']['g24_g8']['kl']
    gpck=old_gauss['center']['by_condition']['g24_g8']['pck']
    scene_rows=[]
    for scene in C.SCENES:
        a=summary['region']['by_scene'][scene];b=summary['center']['by_scene'][scene]
        scene_rows.append(dict(scene=scene,kl=a['kl'],js=a['js'],direct_kl=a['direct_kl'],direct_js=a['direct_js'],
            pck_percent=percent(b['pck']),pck_count=b['pck_count'],valid_centers=b['n_point'],
            mean_pixel_distance=b['mean'],pixel_rmse=b['rmse']))
    with (out/'per_scene.csv').open('w') as file:
        writer=csv.DictWriter(file,fieldnames=list(scene_rows[0]));writer.writeheader();writer.writerows(scene_rows)
    n_pass=sum(c['passed'] for c in screen['checks'])
    status='暂时满足八项必要条件，仍待全部剩余验证' if screen['passed'] else '未通过八项必要条件，停止当前固定实现'
    gate_table=['| 对照 | 指标 | 本方案 | 门槛 | 结果 |','|---|---|---:|---:|---|']
    for c in screen['checks']:
        val=c['candidate_value'];bound=c['bound']
        if c['metric']=='pck':val=f"{percent(val):.6f}%";bound=f"≥{percent(bound):.6f}%"
        else:val=f'{val:.9f}';bound=f'≤{bound:.9f}'
        gate_table.append(f"| {c['reference']} | {c['metric']} | {val} | {bound} | {'通过' if c['passed'] else '失败'} |")
    scene_table=['| 场景 | KL | PCK@8 | 有效中心 | 距离 RMSE(px) |','|---|---:|---:|---:|---:|']
    for row in scene_rows:
        scene_table.append(f"| {row['scene']} | {row['kl']:.9f} | {row['pck_percent']:.6f}% | {row['valid_centers']} | {row['pixel_rmse']:.6f} |")
    text=f'''# R38 单个中心二阶矩 Gaussian：旧 TRAIN 探索性筛查

结论：**{status}（{n_pass}/8）**。这是固定实现的开发结果，完整独立概率参考尚未通过，不能据此证明一般数值精度、实际任务收益或总体研究目标完成。

## 固定方案与范围

对预测三维支撑的均值 μ、协方差因子 F，取同一预测深度的中心射线位置 a，使用单个 Gaussian：

`U ~ N(a, F Fᵀ + (μ−a)(μ−a)ᵀ)`。

增广因子以 QR 压缩后，复用 R29 的 RMS 正深度透视投影法则。旧 R20 内容偏置、温度、h=.25、面积 floor 与生产求积政策固定；无新网络、混合分布或调参。该构造同时改变深度二阶矩和投影形状，不保证投影后的峰值正好位于 a 的投影。旧二维 Ca 仍参与内容温度，不能称全部内容由新三维参数定义。

全48对旧TRAIN图像、2304个24px查询、每查询432个8px key均保留。先完整封存输入、生产及独立算术审核，再经根验收开放标签；无根据结果重跑或剔除样本。本轮只允许淘汰固定实现，任何阳性都必须继续原验证。

## 效果

区域 KL={region['kl']:.9f}；强 quad 为 {quad:.9f}，相对变化 {(region['kl']/quad-1)*100:+.4f}%。中心 PCK@8={percent(center['pck']):.6f}%。旧 R20 anchor Gaussian 的 KL={gaussian:.9f}、PCK={percent(gpck):.6f}%，仅作描述性比较，不是匹配同一三维中心的归因实验。

{chr(10).join(gate_table)}

两个 quad 角色是同配置别名，不是两份独立对照。PCK 决策按原场景平均的精确 Fraction，显示值只用于阅读。

{chr(10).join(scene_table)}

原 valid-center 总数1096，三场景382/402/312；区域按每场景可见像素质量归一后等权平均。主指标与独立指标路径均完成，核验状态为通过。

## 输入、算术与成本证据

全部2304输入和输出通过独立核验，最大输入归一化误差 {root['maximum_input_normalized_error']:.6g}。输入审核用时 {iv['wall_seconds']:.3f}s；生产阶段 {f['wall_seconds']:.3f}s，累计 {f['charged_cell_node_evaluations']:,} 个 cell-nodes；算术审核 {v['wall_seconds']:.3f}s。实际调用/完成数均2304，未发生固定失败停止。

这些时间包括该阶段规定的加载和输出，不能直接与历史 whole-matrix 对照相除来声称成本通过。输出算术审核不重新积分，不构成独立概率参考。R37参考失败仍保留；新输入的4608次独立32/48参考、raw/P/FOV/tail准确性与原每次250M参考预算均未豁免。

## 后续边界

若八项失败，只停止这一固定实现，不能把未认证概率结果写成数学分布族的不可能性证明。若八项通过，下一步须另行冻结匹配的完整算子成本验证，继而完成独立参考、完整14条件TRAIN、旧VAL尺度和扰动、实际learned Q/K与V任务、真实优化GS，以及最后一次未使用数据联合确认。科学目标仍未完成。

协议：[R38](../ROUND38_RMS_ANCHOR_TRIAGE_PROTOCOL.md)。结果：[独立指标核验](../results/round38_rms_anchor_screen/metrics_audit/verification.json)、[根标签前验收](../results/round38_rms_anchor_prelabel_root_acceptance_20261101.json)。所有旧失败和开发尝试保留。
'''
    report.write_text(text)
    paths=[source,report,out/'report_source.py',out/'per_scene.csv',
        run/'evaluation/summary.json',run/'evaluation/screen.json',run/'evaluation/completion.json',
        run/'metrics_audit/verification.json',run/'metrics_audit/completion.json',
        ROOT/'results/round38_rms_anchor_prelabel_root_acceptance_20261101.json',
        run/'forward_complete.json',run/'forward/completion.json',
        audit/'forward_verification.json',audit/'completion.json',
        ROOT/'results/round20_anchor_fixed/summary.json']
    C.write(out/'completion.json',dict(created_unix=time.time(),files={str(p.relative_to(ROOT)):C.sha(p) for p in paths},
        passed_checks=n_pass,total_checks=8,exploratory_screen_passed=screen['passed'],
        independent_probability_reference=False,numerical_preflight_passed=False,
        scientific_gate_passed=False,goal_complete=False,new_array_decodes=0,new_model_calls=0,
        source_capture_scope='Report-only extraction source copied at invocation; formal model/input/metric freezes are separately retained.'))
    print('R38_REPORT',report,flush=True)

if __name__=='__main__':main()
