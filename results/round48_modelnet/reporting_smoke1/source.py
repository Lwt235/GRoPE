"""Render all prespecified R48 results, without fitting or selecting any model.

Production requires the completed frozen evaluator and sealed training records.
--smoke exercises the renderer with explicitly marked synthetic values only.
"""
import argparse
from collections import Counter
from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from models.round48_point_transformer import ROLES
from experiments.round48_common import ROOT, OUT, SEEDS, VARIANTS, checked, read, write, sha

STRONG = ('rope', 'fourier', 'mlp', 'nope', 'free_quadratic')
LABELS = dict(conditional='Conditional Gaussian', diagonal='Diagonal Gaussian',
              isotropic='Isotropic Gaussian', static='Static Gaussian', fixed='Fixed Gaussian',
              free_quadratic='Free quadratic', rope='RoPE', fourier='Fourier warp + RoPE',
              mlp='MLP warp + RoPE', nope='No explicit PE')


def oa_values(summary, role, variant='base'):
    return 100*np.array([summary['by_role_seed'][role][str(seed)][variant]['oa'] for seed in SEEDS])


def figures(dest, summary, synthetic):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'svg.fonttype': 'none', 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), constrained_layout=True)
    offsets = np.linspace(-.15, .15, len(SEEDS))
    for row, role in enumerate(ROLES):
        values = oa_values(summary, role)
        mean = values.mean()
        color = '#1768ac' if role == 'conditional' else '#777777'
        axes[0].plot([values.min(), values.max()], [row, row], color=color, lw=1.2)
        axes[0].scatter(values, row+offsets, color=color, s=20, alpha=.7)
        axes[0].scatter([mean], [row], marker='D', color=color, s=35)
    axes[0].set(yticks=range(len(ROLES)), yticklabels=[LABELS[r] for r in ROLES],
                xlabel='Development overall accuracy (%)', title='All methods, five training seeds')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=.2)
    for row, control in enumerate(STRONG):
        differences = oa_values(summary, 'conditional')-oa_values(summary, control)
        axes[1].plot([differences.min(), differences.max()], [row, row], color='#777777', lw=1.2)
        axes[1].scatter(differences, row+offsets, color='#1768ac', s=25)
        axes[1].scatter([differences.mean()], [row], marker='D', color='#111111', s=40)
    axes[1].axvline(0, color='#222222', lw=.9)
    axes[1].axvline(1, color='#b16c00', ls='--', lw=.9, label='Prespecified 1 pp margin')
    axes[1].set(yticks=range(len(STRONG)), yticklabels=[LABELS[r] for r in STRONG],
                xlabel='Conditional Gaussian minus control (percentage points)',
                title='Paired differences against strong controls')
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=.2)
    axes[1].legend(loc='best', fontsize=8)
    title = 'SYNTHETIC FIXTURE — NOT EXPERIMENTAL RESULTS' if synthetic else 'R48: official-train development split only'
    fig.suptitle(title+'\nDots: individual seeds; diamonds: means; whiskers: range, not confidence intervals.', fontsize=12)
    for suffix in ('png', 'svg'):
        fig.savefig(dest/('paired_accuracy.'+suffix), dpi=180)
    plt.close(fig)

    variants = tuple(v for v in VARIANTS if v != 'base')
    delta = np.array([[float(oa_values(summary, role, variant).mean()-oa_values(summary, role).mean())
                       for variant in variants] for role in ROLES])
    bound = max(.1, float(abs(delta).max()))
    fig, ax = plt.subplots(figsize=(10.5, 6.3), constrained_layout=True)
    im = ax.imshow(delta, aspect='auto', cmap='RdBu', norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound))
    ax.set(xticks=range(len(variants)), xticklabels=variants,
           yticks=range(len(ROLES)), yticklabels=[LABELS[r] for r in ROLES],
           title=('SYNTHETIC FIXTURE — NOT EXPERIMENTAL RESULTS\n' if synthetic else '')
                 +'Robustness: OA change from each method\'s own base (pp)')
    for i in range(len(ROLES)):
        for j in range(len(variants)):
            ax.text(j, i, f'{delta[i,j]:+.2f}', ha='center', va='center',
                    color='white' if abs(delta[i,j]) > .65*bound else '#111111', fontsize=9)
    fig.colorbar(im, ax=ax, label='Five-seed mean change (percentage points)')
    for suffix in ('png', 'svg'):
        fig.savefig(dest/('robustness.'+suffix), dpi=180)
    plt.close(fig)


def report_text(dest, summary, verification, records, synthetic):
    means = summary['means']
    primary = means['conditional']['base']
    strongest = max(STRONG, key=lambda role: means[role]['base']['oa'])
    delta = 100*(primary['oa']-means[strongest]['base']['oa'])
    heading = '# SYNTHETIC FIXTURE — NOT EXPERIMENTAL RESULTS' if synthetic else '# R48：原生三维 Transformer 位置编码开发实验'
    lines = [heading, '',
        ('此文只验证汇总格式，所有数值为人工合成，未读取任务数据。' if synthetic else
         '本轮已完成 50 次训练及预定开发评估。官方测试集仍未使用，完整研究目标尚未完成。'), '',
        f"固定主方案 conditional 的开发 OA 为 **{100*primary['oa']:.3f}%**；"
        f"强对照中最高均值来自 {strongest}，差 **{delta:+.3f}pp**。", '',
        ('主方案达到本轮预定开发信号条件，仍需独立确认。' if summary['promising_development_only'] else
         '主方案未达到本轮预定开发信号条件；全部连续差值和次要结果保留如下。'), '',
        '这是同一四层全局标量 self-attention 分类骨干内的 PE 比较；'
        'NoPE 同样接收 XYZ 内容。这不是论文 Point Transformer 的完整复现。'
        '本轮结果不消除 R47 匹配任务的负结果，也不能单独证明一般性。', '',
        '官方训练缓存分为 7910 fit / 965 val / 965 dev_eval；所有模型选定后才进行 dev_eval。'
        '各角色使用相同 120 epochs、29,760 次更新和选模规则。五个种子是优化重复，'
        '表中种子标准差和范围不是总体置信区间。', '',
        '## 主开发结果', '',
        '| 方法 | OA (%) | 种子 SD (pp) | MCA (%) | CE ↓ | PE 参数 | 训练及选模分钟 | 峰值 allocated MiB |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for role in ROLES:
        entries = [r for r in records if r['role'] == role]
        metric = means[role]['base']
        lines.append(f"| {role} | {100*metric['oa']:.3f} | {oa_values(summary,role).std(ddof=1):.3f} | "
                     f"{100*metric['mca']:.3f} | {metric['ce']:.6f} | {entries[0]['positional_parameters']} | "
                     f"{np.mean([r['seconds'] for r in entries])/60:.2f} | "
                     f"{max(r['peak_allocated'] for r in entries)/2**20:.1f} |")
    lines += ['', '训练计时包含训练和选模，来自共享主机；不能替代配对推理吞吐测量。', '',
              f'![所有方法及配对差值]({dest / "paired_accuracy.png"})', '',
              '## 主方案相对各对照', '',
              '强对照为 rope/fourier/mlp/nope/free_quadratic。预定信号要求相对每项 OA ≥ +1pp、'
              '至少 4/5 种子正向、MCA 不低于 −0.5pp。组件比较不套用这一阈值。', '',
              '| 对照 | OA 差 (pp) | MCA 差 (pp) | CE 差 | 正向种子 | 五个配对 OA 差 (pp) |',
              '| --- | ---: | ---: | ---: | ---: | --- |']
    for comparison in summary['comparisons']:
        sequence = ', '.join(f'{x:+.3f}' for x in comparison['paired_seed_pp'])
        lines.append(f"| {comparison['control']} | {comparison['oa_difference_pp']:+.3f} | "
                     f"{comparison['mca_difference_pp']:+.3f} | {comparison['ce_difference']:+.6f} | "
                     f"{comparison['winning_seeds']}/5 | {sequence} |")
    lines += ['', '完整高斯与自由二次对照同时改变正定约束和中心—精度耦合，'
              '不能将差值全部归因于正定性。简化版参数量不同。', '',
              '## 预定鲁棒性结果', '',
              '下表为各条件的绝对 OA；图为相对各方法自身 base 的变化。'
              '全部条件的 MCA、CE 和逐类别准确率保存在完整 JSON 中。', '',
              '| 方法 | '+' | '.join(VARIANTS)+' |',
              '| --- | '+' | '.join('---:' for _ in VARIANTS)+' |']
    for role in ROLES:
        lines.append('| '+role+' | '+' | '.join(f"{100*means[role][v]['oa']:.3f}" for v in VARIANTS)+' |')
    lines += ['', f'![鲁棒性变化]({dest / "robustness.png"})', '',
              'Y90 属于训练旋转轴，SO3 为预定的 Rz45°Rx30°；没有训练任意三维旋转不变性。'
              '点数条件同时改变采样及输入归一化，不能称为纯序列长度实验。', '',
              '## 实现与记录核对', '']
    replay = verification['metric_replay']
    permutations = verification['permutation_checks']
    selection = Counter(r['best_epoch'] for r in records)
    lines += [f"- 全量指标回算覆盖 {verification['models']} 个模型、{verification['objects']} 个物体、"
              f"{verification['variants']} 个条件；正确计数不一致记录 "
              f"{sum(r['correct_difference'] != 0 for r in replay)} 条，最大 CE 差 "
              f"{max(r['ce_error'] for r in replay):.3g}。",
              f"- 抽样位置公式告警 {len(verification['sampled_PE_warnings'])} 项；"
              '这些是独立公式路径，不是独立作者或第二套完整网络复现。',
              f"- 反转点序列的最大 logits 差 {max(r['max_logit_error'] for r in permutations):.3g}，"
              f"各模型合计分类翻转 {sum(r['winner_changes'] for r in permutations)} 个。",
              f"- 共享权重发生变化的模型 {sum(r['shared_parameters_changed'] for r in records)}/50。"
              '具体位置张量变化和梯度记录保留在各 run.json。',
              '- 选定 epoch 分布：'+', '.join(f'{k}:{v}' for k,v in sorted(selection.items()))+
              '。固定预算不证明所有模型已收敛。', '',
              f"方案：[冻结协议]({ROOT/'ROUND48_NATIVE_POINT_PE_PROTOCOL.md'})。"
              f"完整结果：[指标 JSON]({OUT/'metrics/summary.json'})、"
              f"[回算记录]({OUT/'metrics/verification.json'})。", '']
    return '\n'.join(lines)


def render(dest, summary, verification, records, synthetic=False):
    if dest.exists():
        raise ValueError('preserve existing report')
    for role in ROLES:
        entries = [r for r in records if r['role'] == role]
        if {r['seed'] for r in entries} != set(SEEDS) or len(entries) != len(SEEDS):
            raise ValueError('all five seed records required for each role')
        for variant in VARIANTS:
            for key in ('oa', 'mca', 'ce'):
                expected = np.mean([summary['by_role_seed'][role][str(s)][variant][key] for s in SEEDS])
                if abs(expected-summary['means'][role][variant][key]) > 1e-12:
                    raise ValueError('summary inconsistency')
    dest.mkdir(parents=True)
    figures(dest, summary, synthetic)
    (dest/'report.md').write_text(report_text(dest, summary, verification, records, synthetic))
    write(dest/'render_record.json', dict(synthetic_fixture=synthetic, source_sha256=sha(__file__),
        unix=time.time(), file_hashes={p.name: sha(p) for p in dest.iterdir() if p.is_file()},
        training_modified=False, checkpoint_selection_modified=False, goal_complete=False))


def production(out):
    checked(out)
    status = read(out/'queue_status.json')
    if status['state'] != 'complete' or status['active']:
        raise ValueError('wait for terminal successful controller state')
    seal = read(out/'audit_training/completion.json')
    summary, verification = read(out/'metrics/summary.json'), read(out/'metrics/verification.json')
    if summary['official_test_access'] or summary['fresh_confirmation'] or len(seal['records']) != 50:
        raise ValueError('unexpected scope or incomplete experiment')
    records, inputs = [], {}
    for relative in ('lock.json', 'audit_training/completion.json', 'metrics/summary.json', 'metrics/verification.json'):
        inputs[str(out/relative)] = sha(out/relative)
    for entry in seal['records']:
        path = Path(entry['path'])/'run.json'
        if sha(path) != entry['run_sha256']:
            raise ValueError('sealed run changed')
        records.append(read(path))
        inputs[str(path)] = entry['run_sha256']
    for path, digest in verification['input_hashes'].items():
        if sha(path) != digest:
            raise ValueError('audited prediction changed')
        inputs[path] = digest
    dest = out/'report1'
    render(dest, summary, verification, records)
    write(dest/'input_hashes.json', inputs)
    print(dest/'report.md')


def smoke():
    table, means, records = {}, {}, []
    for ri, role in enumerate(ROLES):
        table[role] = {}
        for si, seed in enumerate(SEEDS):
            table[role][str(seed)] = {}
            for vi, variant in enumerate(VARIANTS):
                oa = .87+.001*ri+.0007*si-.002*vi
                if variant == 'so3':
                    oa -= .12
                table[role][str(seed)][variant] = dict(oa=oa, mca=oa-.02, ce=1-oa)
            records.append(dict(role=role, seed=seed, positional_parameters=100*ri,
                                seconds=1200+ri, peak_allocated=2**30, best_epoch=100,
                                shared_parameters_changed=True))
        means[role] = {v: {k: float(np.mean([table[role][str(s)][v][k] for s in SEEDS]))
                           for k in ('oa', 'mca', 'ce')} for v in VARIANTS}
    comparisons = []
    for role in tuple(r for r in ROLES if r != 'conditional'):
        delta = [100*(table['conditional'][str(s)]['base']['oa']-table[role][str(s)]['base']['oa']) for s in SEEDS]
        comparisons.append(dict(control=role, oa_difference_pp=float(np.mean(delta)),
            mca_difference_pp=100*(means['conditional']['base']['mca']-means[role]['base']['mca']),
            ce_difference=means['conditional']['base']['ce']-means[role]['base']['ce'],
            paired_seed_pp=delta, winning_seeds=sum(d > 0 for d in delta)))
    summary = dict(means=means, by_role_seed=table, comparisons=comparisons, promising_development_only=False)
    verification = dict(models=50, objects=965, variants=7, metric_replay=[dict(correct_difference=0, ce_error=1e-8)],
                        permutation_checks=[dict(max_logit_error=1e-6, winner_changes=0)], sampled_PE_warnings=[])
    dest = ROOT/'results/round48_modelnet/reporting_smoke1'
    render(dest, summary, verification, records, synthetic=True)
    print('SYNTHETIC RENDER ONLY', dest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    smoke() if args.smoke else production(args.out)
