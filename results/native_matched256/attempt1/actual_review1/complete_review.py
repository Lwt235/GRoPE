"""Record actual reading of the completed comparison, without changing raw results."""
import hashlib
import json
import statistics
import time
from pathlib import Path

ROOT = Path('/home/liwt/GRoPE')
OUT = ROOT / 'results/native_matched256/attempt1'
HERE = Path(__file__).parent

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')

result, selected, queue = [read(OUT / name) for name in ('result.json', 'selected.json', 'queue_status.json')]
assert sha(OUT / 'result.json') == '80f916a53e2ef386a271c7547dbf229832bfa5e49d33c26566009f6459b977e7'
assert queue['state'] == 'complete' and not queue['active'] and not queue['pending']
assert len(queue['completed']) == 15 and all(r['exit_code'] == 0 for r in queue['completed'])
assert selected['all_nine_selected_before_development_evaluation'] and len(selected['records']) == 9
assert all(r['full_512_stream_matches_old'] for r in selected['pairing'])
assert len(result['records']) == 108 and len(result['comparisons']) == 36 and len(result['interactions']) == 12
assert not result['official_test_access'] and not result['fresh_confirmation']

runs = []
for choice in selected['records']:
    directory = OUT / 'runs' / choice['role'] / str(choice['seed'])
    run = read(directory / 'run.json')
    assert run['epochs'] == 120 and run['steps'] == 46800 and not run['resumed']
    assert sha(directory / 'run.json') == choice['run_sha256']
    assert sha(directory / 'best.pt') == choice['checkpoint_sha256']
    runs.append({key: run[key] for key in ('role', 'seed', 'epochs', 'steps', 'best_epoch', 'seconds', 'peak_allocated', 'resumed')})

lookup = {(r['train_tokens'], r['role'], r['seed'], r['variant']): r for r in result['records']}
matched_improvements = []
for role in ('relative5', 'distance_mlp', 'fixed_isotropic5'):
    gains = [100 * (lookup[256, role, seed, 'sparse256']['metrics']['mca'] -
                    lookup[512, role, seed, 'sparse256']['metrics']['mca']) for seed in (11, 22, 33)]
    matched_improvements.append(dict(role=role, paired_mca_gain_pp=gains, mean_mca_gain_pp=statistics.mean(gains)))

primary = []
for row in result['primary']:
    primary.append(dict(row,
        mean_oa_gain_pp=statistics.mean(row['paired_oa_gain_pp']),
        mean_ce_difference=statistics.mean(row['paired_ce_difference']),
        mean_source_gains_pp={k: statistics.mean(v) for k, v in row['domains'].items()},
        class_sign_counts=dict(positive=sum(x > 1e-9 for x in row['class_mean_gain_pp']),
                               negative=sum(x < -1e-9 for x in row['class_mean_gain_pp']),
                               ties=sum(abs(x) <= 1e-9 for x in row['class_mean_gain_pp']))))

figure = OUT / 'comparison.png'
assert sha(figure) == '65d83e333398432b7eedef08d8bc50e92c3070e8073eb048c34b551590455193'
analysis = dict(completed_runs=runs, primary=primary, matched_workflow_improvements=matched_improvements,
    sparse_interactions=[r for r in result['interactions'] if r['variant'] == 'sparse256'],
    dense_shape_comparison=next(r for r in result['comparisons'] if
        (r['train_tokens'], r['variant'], r['left'], r['right']) == (256, 'dense1024', 'relative5', 'fixed_isotropic5')),
    independent_metric_error_from_frozen_reporter=result['independent_metric_max_error'],
    gpu_validation_replay_max_error=max(r['gpu_logit_error'] for r in selected['records']),
    cpu_validation_prediction_changes=sum(r['cpu_prediction_disagreements'] for r in selected['records']),
    permutation_prediction_changes=sum(r['prediction_changes'] for r in result['permutation']),
    permutation_maximum_logit_error=max(r['maximum_logit_error'] for r in result['permutation']))
write(HERE / 'analysis.json', analysis)

original = (OUT / 'report.md').read_text()
with (HERE / 'automatic_report.md').open('x') as stream:
    stream.write(original)
intro = '''# 原生 256 基元训练对照：完整结果与实际审阅

**匹配 256 个基元的训练和评价后，五维关系没有保留明确的准确率优势。** 原 765 个开发对象、三个配对种子上，relative5 / Distance / 固定球形的 mCA 为 **76.743% / 76.620% / 76.573%**。对两个控制仅高 +0.123 / +0.170 pp；各有反例，种子×类别区间均较宽且跨零。这不是某个尾数未达门槛，也不是执行失败；主要观察是平均准确率增益缩小到接近零。

对 Distance 的三个配对差为 −0.086 / −0.141 / +0.597 pp，区间 [−2.167, 2.383]；对固定球形为 +0.918 / −1.086 / +0.677 pp，区间 [−2.330, 2.973]。没有据此做等效性证明或声称任何实际收益都不存在。

**概率预测上的正面信号保留。** 五维相对 Distance / 固定球形的平均 CE 仍低 0.0280 / 0.0356，三个种子均改善；OA 分别高 +0.305 / +0.436 pp，各为 2/3 正。CE 改善不能替代主准确率，也没有单独证明校准更好。两个来源组及逐类别仍有混合结果，详见实际审阅数据。

同样三个旧种子的 512→256 比较中，五维对 Distance / 固定球形原为 +0.888 / +2.139 pp；改为匹配流程后，优势分别减少 0.765 / 1.970 pp。固定球形在三组中均缩小差距，但差中之差的联合区间分别为 [−4.575, 3.193] / [−6.151, 1.841] pp，仍不足以确证总体交互效应。不能拿官方新五种子的 +2.005 pp 与这里旧三种子的 +0.123 pp 直接作因果差值。

从每种方法看，匹配流程在 256 评价上分别使五维 / Distance / 固定球形提高 +0.441 / +1.206 / +2.411 pp。该模式与“控制方法适配数量后追上”相容，**削弱“少量基元本身使形状关系更有优势”的解释**；尚不能证明全部旧收益只来自数量不匹配。新模型按 val256 CE 选择，旧模型按 val512 CE 选择；训练和选模数量共同改变，且对象/更新预算相同而 FLOPs、基元曝光量不同。

更大数量的评价也有实质反例：256 训练→1024 评价时，五维比固定球形平均低 **2.719 pp**，三个种子均低（−3.737 / −0.777 / −3.644），联合区间 [−5.971, 0.237] pp；对 Distance 为 −0.133 pp、方向混合。这不能再概括为对任意数量变化更稳健。256→512、jitter 与 tilt 的 OA/CE 对 Distance 多有改善，完整数据同时保留。

九项拟合均完成 120 epochs / 46,800 updates、退出码 0，无恢复或重跑；共同初始化及完整数据采样/增强流与各自旧 512 训练一致，全部九模型在开发导出前固定。GPU 验证回放差为零。汇总器对 54 组新预测和 54 组复用预测独立复算，最大指标差 4.44e−16；18 个模型/流程的置换均无分类变化。两面板比较图已实际查看，原始结果、队列及自动报告备份保留。这里的独立算术路径属于同一代理的实现核查，不是外部实验室复现。

**决定：不把匹配 256 流程晋升为改进方案，也不追加种子追求通过。** 保留五维在区域任务及原生 512 流程的既有正面证据，以及官方基础确认不充分和 RGB/迁移负结果。下一步在旧数据上检验更精简的对称关系组织：保留距离、双侧平均位移关系和双侧平均形状关系，检查独立编码两端方向是否确有必要；与完整五维、Distance、固定球形和可学习单散度比较。该假设尚未验证，本报告不将它写成已完成设计或成功原因。

原始 `result.json` 和 `queue_status.json` 的待审阅字段是生成时快照，实际审阅状态见 `review.json`；本次无官方 TEST 访问，也不是未用数据确认。

'''
(OUT / 'report.md').write_text(intro + '## 固定比较的完整自动统计\n\n' + original.split('\n', 1)[1].lstrip())
review = dict(scientific_review_complete=True, visual_review_complete=True, reviewed_unix=time.time(),
    lock_sha256=sha(OUT / 'lock.json'), selection_sha256=sha(OUT / 'selected.json'),
    result_sha256=sha(OUT / 'result.json'), report_sha256=sha(OUT / 'report.md'),
    analysis_sha256=sha(HERE / 'analysis.json'), completed_fits=9, new_evaluation_groups=54,
    reused_evaluation_groups=54, all_fits_full_budget=True, all_jobs_exit_zero=True,
    primary_accuracy_advantage_established=False, primary_equivalence_established=False,
    limited_observation_advantage_hypothesis_supported=False,
    count_shift_only_causal_explanation_established=False,
    all_three_primary_ce_improve_against_both_controls=True,
    fresh_confirmation=False, official_test_access=False, goal_complete=False,
    viewed_figures=[dict(path=str(figure), sha256=sha(figure),
        observation='Actually viewed both panels: matched256 gains cluster around zero; original512 gains larger in mean; new256-to512 covariance advantage mixed and negative in mean. Plot agrees with the fixed comparisons.')],
    implementation_checks=analysis | {'primary': 'See actual_review1/analysis.json for full rows.'},
    next_decision='Keep historical evidence and negatives; do not promote matched256 or add seeds. Test compact symmetric relation organization on used region/native data with strong and scalar-divergence controls, prospectively fixed before any new fits.',
    limitations=result['limitations'] + ['CE improvement is not a standalone demonstration of improved calibration.',
        'Current report joint intervals for reused old comparisons do not rewrite previously frozen conditional intervals or official new-object confirmation.',
        'No statistical equivalence or count-shift-only causal claim.'])
write(OUT / 'review.json', review)
print(json.dumps({name: sha(OUT / name) for name in ('result.json', 'selected.json', 'report.md', 'review.json')}, indent=2))
