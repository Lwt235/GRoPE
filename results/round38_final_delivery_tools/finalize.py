"""Accept the R38 failed development screen and preserve its delivery history."""
from pathlib import Path
import json
import sys
import time

ROOT = Path('/home/liwt/GRoPE')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments import round38_rms_anchor_contract as C
from experiments import round38_rms_anchor_evaluate as E
import numpy as np


def main():
    C.configure_runtime()
    def no_load(*args, **kwargs):
        raise RuntimeError('Final delivery is JSON/hash only')
    np.load = no_load
    pre = C.read(HERE / 'before_execution.json')
    assert C.sha(__file__) == pre['source_sha256']
    for name, digest in pre['files'].items():
        assert C.sha(ROOT / name) == digest, name
    assert C.sha(ROOT / 'README.md') == pre['readme_before_sha256']
    receipt_path = ROOT / 'results/round38_rms_anchor_root_acceptance_20261101.json'
    decision_path = ROOT / 'results/round38_rms_anchor_post_result_decision_20261101.json'
    manifest_path = ROOT / 'results/round38_rms_anchor_delivery_manifest_20261101.json'
    assert not any(p.exists() for p in (receipt_path, decision_path, manifest_path))
    E.require_prelabel(C.DEFAULT_OUT, C.DEFAULT_AUDIT)
    for directory in (C.DEFAULT_OUT / 'prepared', C.DEFAULT_OUT / 'forward',
                      C.DEFAULT_AUDIT / 'input_audit', C.DEFAULT_AUDIT,
                      C.DEFAULT_OUT / 'evaluation', C.DEFAULT_OUT / 'metrics_audit',
                      ROOT / 'results/round38_controllers'):
        completion = C.read(directory / 'completion.json')
        C.verify_inventory(directory, completion['files'], exclude=('completion.json',))
    reconciliation = C.read(ROOT / 'results/round38_delivery_reconciliation_20261101/reconciliation.json')
    assert reconciliation['accepted_controller_log_suffix_reconciliation']
    label_dir = ROOT / 'results/round38_label_stage_execution'
    label_completion = C.read(label_dir / 'completion.json')
    assert label_completion['error'] is None
    assert C.sha(label_dir / 'completion.json') == reconciliation['label_completion_sha256']
    assert C.sha(label_dir / 'execution.log') == reconciliation['final_log_sha256']
    for name, digest in label_completion['files'].items():
        if name != 'execution.log':
            assert C.sha(label_dir / name) == digest, name
    report_completion = C.read(ROOT / 'docs/round38_rms_anchor_triage_report_data/completion.json')
    for name, digest in report_completion['files'].items():
        assert C.sha(ROOT / name) == digest, name
    screen = C.read(C.DEFAULT_OUT / 'evaluation/screen.json')
    verification = C.read(C.DEFAULT_OUT / 'metrics_audit/verification.json')
    summary = C.read(C.DEFAULT_OUT / 'evaluation/summary.json')
    assert len(screen['checks']) == 8 and all(not row['passed'] for row in screen['checks'])
    assert verification['passed'] and verification['decision'] == 'stop_this_fixed_implementation_only'
    inputs = C.read(ROOT / 'results/round38_rms_anchor_prelabel_root_acceptance_20261101.json')
    files = dict(pre['files'])
    files[str((HERE / 'before_execution.json').relative_to(ROOT))] = C.sha(HERE / 'before_execution.json')
    C.write(receipt_path, dict(created_unix=time.time(), accepted_exploratory_failure_delivery=True,
        passed_checks=0, total_checks=8, completed_queries=2304, original_valid_centers=1096,
        regional_kl=summary['region']['macro']['kl'], pck=summary['center']['macro']['pck'],
        production_wall_seconds=inputs['production_wall_seconds'],
        charged_cell_node_evaluations=inputs['charged_cell_node_evaluations'],
        dual_metric_routes_agree=True, all_input_and_output_arithmetic_passed=True,
        label_controller_log_suffix_reconciled=True, report_separate_pre_execution_capture_succeeded=False,
        independent_probability_reference=False, numerical_preflight_passed=False,
        accepted_scope='Stop this fixed implementation only; not a certified impossibility result for its mathematical family.',
        new_array_decodes=0, new_model_calls=0, metric_reexecutions=0,
        files=files, scientific_gate_passed=False, goal_complete=False))
    C.write(decision_path, dict(created_unix=time.time(), root_acceptance_sha256=C.sha(receipt_path),
        decision='stop_this_fixed_implementation_only', selected_next_candidate=False,
        next_action='Prospectively freeze a full-population diagnostic using sealed R38 outputs before any further candidate execution.',
        diagnostic_question='Separate saved positional-mass winner displacement from the effect of the existing content bias; do not infer causality or general numerical accuracy.',
        new_model_calls_authorized_by_this_record=False, new_labels_authorized_by_this_record=False,
        cost_and_full_reference_work_for_this_fixed_implementation_stopped=True,
        no_retuning_or_exclusions=True, original_gates_unchanged=True,
        independent_probability_reference=False, scientific_gate_passed=False, goal_complete=False))
    history = ROOT / 'results/round38_readme_history_20261101'
    assert not history.exists()
    history.mkdir()
    readme = ROOT / 'README.md'
    old = readme.read_bytes()
    (history / 'README.before.md').write_bytes(old)
    first, remainder = old.decode().split('\n', 1)
    addition = '''

**最新：第三十八轮单高斯 3D anchor 方案未通过旧 TRAIN 筛查，八项必要条件 0/8。** 全部 2,304 查询完成并通过输入与独立算术审核；经根验收后，两条指标路径一致确认：区域 KL=0.366603469，较强 quad 的 0.308352829 差 18.8909%；中心 PCK@8=86.336733%，各定位保护均失败。生产阶段 2,406.700 秒。停止这一固定实现，不继续其成本及全概率参考投入；完整概率精度、一般效果和总体研究目标仍未通过。[报告](docs/round38_rms_anchor_triage_report.md) · [根验收](results/round38_rms_anchor_root_acceptance_20261101.json) · [后续决定](results/round38_rms_anchor_post_result_decision_20261101.json)。

报告整理出现一项控制器日志清单滞后问题，以及根代理未在捕获检查失败后停止报告生成的错误；原记录未改，已外置对账。报告没有成功的独立事前提取冻结，不能倒填。[完整说明](docs/round38_delivery_record_note.md)。下一步先以已有封存输出诊断位置质量与内容偏置的影响，不据此选参、剔除样本或声称分布族不可能。
'''
    readme.write_text(first + addition + remainder)
    (history / 'README.after.md').write_bytes(readme.read_bytes())
    C.write(history / 'edit_receipt.json', dict(created_unix=time.time(),
        before_sha256=C.sha(history / 'README.before.md'), after_sha256=C.sha(readme),
        purpose='Prepend accepted R38 negative result and transparent delivery-record correction; preserve prior README bytes.'))
    paths = {ROOT / name for name in files}
    paths.update((receipt_path, decision_path, readme, Path(__file__), HERE / 'source.before.py'))
    for directory in (ROOT / 'results/round38_controllers', ROOT / 'results/round38_prelabel_root_review',
                      label_dir, ROOT / 'results/round38_report_tools',
                      ROOT / 'results/round38_delivery_reconciliation_20261101',
                      ROOT / 'docs/round38_rms_anchor_triage_report_data', history,
                      C.DEFAULT_OUT, C.DEFAULT_AUDIT):
        paths.update(p for p in directory.rglob('*') if p.is_file() and p.suffix != '.npz')
    C.write(manifest_path, dict(created_unix=time.time(),
        files={str(p.relative_to(ROOT)):C.sha(p) for p in sorted(paths)},
        array_inventory_scope='All experiment archive bytes verified by retained stage inventories; this delivery index lists non-NPZ evidence.',
        inherited_log_inventory_exception='Only label-stage execution.log; original prefix and final bytes explicitly reconciled.',
        independent_probability_reference=False, scientific_gate_passed=False, goal_complete=False))
    print('R38_NEGATIVE_DELIVERY_ACCEPTED', C.sha(receipt_path), C.sha(manifest_path), flush=True)


if __name__ == '__main__':
    main()
