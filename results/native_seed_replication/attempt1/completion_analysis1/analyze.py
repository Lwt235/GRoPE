"""Read-only analysis of the eight completed fits, using old validation only."""
from datetime import datetime, timezone, timedelta
from pathlib import Path
import time

import numpy as np
import torch

from experiments.round65_common import ROOT, CONFIG, read, sha, write
from experiments.round65_audit import independent_metrics

BASE = ROOT / 'results/native_seed_replication/attempt1'
OUT = Path(__file__).resolve().parent
SEEDS = (44, 55, 66, 77)
ROLES = ('relative5', 'distance_mlp')


def main():
    if (OUT / 'result.json').exists():
        raise ValueError('preserve existing partial analysis')
    manifest = read(BASE / 'data/manifest.json')
    part = next(r for r in manifest['partitions'] if r['split'] == 'val')
    expected_ids = np.asarray(part['ids'])
    labels_path = BASE / 'data/val/labels.npy'
    assert sha(labels_path) == part['hashes']['labels.npy']
    expected_labels = np.load(labels_path, allow_pickle=False)
    records, histories, streams, initializations = [], {}, {}, {}
    worst = 0.
    for seed in SEEDS:
        for role in ROLES:
            folder = BASE / 'runs' / role / str(seed)
            run, history = read(folder / 'run.json'), read(folder / 'history.json')
            assert (run['role'], run['seed'], run['epochs'], run['steps']) == (role, seed, 120, 46800)
            assert run['lock_sha256'] == sha(BASE / 'lock.json')
            assert run['data_sha256'] == sha(BASE / 'data/manifest.json')
            for name, digest in run['hashes'].items():
                assert sha(folder / name) == digest, (role, seed, name)
            assert [r['epoch'] for r in history['training']] == list(range(1, 121))
            assert [r['step'] for r in history['training']] == [390 * i for i in range(1, 121)]
            assert [r['epoch'] for r in history['validation']] == list(range(0, 121, 5))
            assert all(np.isfinite([r['loss'], r['last_gradient_norm']]).all() for r in history['training'])
            assert all(np.isfinite([r['ce'], r['mca'], r['oa']]).all() for r in history['validation'])
            best = min(history['validation'], key=lambda r: r['ce'])
            assert best['epoch'] == run['best_epoch'] and best['ce'] == run['best_val_ce']
            checkpoint = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
            assert (checkpoint['role'], checkpoint['seed'], checkpoint['epoch']) == (role, seed, best['epoch'])
            with np.load(folder / 'validation.npz', allow_pickle=False) as archive:
                assert np.array_equal(archive['ids'], expected_ids)
                assert np.array_equal(archive['labels'], expected_labels)
                metrics = independent_metrics(archive['logits'], archive['labels'])
            error = max(abs(metrics[k] - best[k]) for k in ('ce', 'mca', 'oa'))
            assert error < 1e-8
            worst = max(worst, error)
            sampling = [r['stream_sha256'] for r in history['training']]
            assert streams.setdefault(seed, sampling) == sampling
            assert initializations.setdefault(seed, run['initial_shared_sha256']) == run['initial_shared_sha256']
            histories[role, seed] = history
            records.append(dict(role=role, seed=seed, epochs=run['epochs'], updates=run['steps'],
                best_epoch=best['epoch'], best_validation=metrics,
                first_training_loss=history['training'][0]['loss'], final_training_loss=history['training'][-1]['loss'],
                final_validation={k:history['validation'][-1][k] for k in ('ce', 'mca', 'oa')},
                run_sha256=sha(folder / 'run.json'), artifacts_sha256=run['hashes'],
                completion_marker_modified_unix=(folder / 'run.json').stat().st_mtime,
                independent_validation_metric_error=error))
    assert all(not (BASE / 'runs' / role / '88').exists() for role in ROLES)
    assert not (BASE / 'selected.json').exists() and not (BASE / 'evaluation').exists()
    lookup = {(r['role'], r['seed']):r for r in records}
    paired = []
    for seed in SEEDS:
        a, b = [lookup[role, seed] for role in ROLES]
        x, y = a['best_validation'], b['best_validation']
        paired.append(dict(seed=seed, relative5_epoch=a['best_epoch'], distance_epoch=b['best_epoch'],
            relative5_mca_pct=100*x['mca'], distance_mca_pct=100*y['mca'], mca_gain_pp=100*(x['mca']-y['mca']),
            oa_gain_pp=100*(x['oa']-y['oa']), ce_difference=x['ce']-y['ce']))
    means = {key:float(np.mean([r[key] for r in paired])) for key in (
        'relative5_mca_pct', 'distance_mca_pct', 'mca_gain_pp', 'oa_gain_pp', 'ce_difference')}
    result = dict(analyzed_unix=time.time(), source_sha256=sha(__file__), records=records, paired=paired, means=means,
        completed_fits=8, required_fits=10, missing_seeds=[88], all_completed_budgets_correct=True,
        paired_initializations_match=True, all_120_epoch_sampling_streams_match=True,
        independent_validation_metric_max_error=worst, validation_objects=len(expected_labels),
        formal_selection_complete=False, development_evaluation_exported=False, full_replication_signal=None,
        interpretation='Descriptive old-validation analysis of four completed pairs; validation selected the checkpoints. No new-five development decision or fresh confirmation.',
        new_training=False, new_inference=False, official_test_access=False, goal_complete=False)
    write(OUT / 'result.json', result)
    plot(histories, paired)
    write_report(result)
    print('PARTIAL_VALIDATION_ANALYZED', means, 'metric_error', worst, flush=True)


def plot(histories, paired):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'relative5': 'tab:blue', 'distance_mlp': 'tab:orange'}
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharex=True, layout='constrained')
    for col, seed in enumerate(SEEDS):
        for role in ROLES:
            h = histories[role, seed]
            axes[0, col].plot([r['epoch'] for r in h['training']], [r['loss'] for r in h['training']],
                              color=colors[role], label=role)
            val = [r for r in h['validation'] if r['epoch'] >= 20]
            axes[1, col].plot([r['epoch'] for r in val], [r['ce'] for r in val], color=colors[role])
            best = min(h['validation'], key=lambda r: r['ce'])
            axes[1, col].scatter(best['epoch'], best['ce'], color=colors[role], marker='x', s=55)
        axes[0, col].set_title(f'Seed {seed}')
        axes[1, col].set_xlabel('Epoch')
        for ax in axes[:, col]:
            ax.grid(alpha=.2)
    axes[0, 0].set_ylabel('Training loss (label smoothing)')
    axes[1, 0].set_ylabel('Old validation CE (from epoch 20)')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Eight completed fits; seed 88 not started. Validation curves are not dev_eval results.')
    fig.savefig(OUT / 'training_curves.png', dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')
    values = [r['mca_gain_pp'] for r in paired]
    ax.bar(range(4), values, color=['tab:blue' if x > 0 else 'tab:orange' for x in values])
    ax.axhline(0, color='black', linewidth=.8)
    ax.set(xticks=range(4), xticklabels=SEEDS, xlabel='Completed seed pair',
           ylabel='relative5 - Distance validation mCA (pp)',
           title='Selected-checkpoint validation only; fixed five-pair batch incomplete')
    for i, value in enumerate(values):
        ax.text(i, value, f'{value:+.2f}', ha='center', va='bottom' if value >= 0 else 'top')
    ax.margins(y=.2)
    fig.savefig(OUT / 'validation_pairs.png', dpi=150)
    plt.close(fig)


def write_report(result):
    tz = timezone(timedelta(hours=8))
    stamp = datetime.fromtimestamp(result['analyzed_unix'], tz).strftime('%Y-%m-%d %H:%M:%S %Z')
    lines = ['# 固定原生复现：已完成八次拟合的阶段分析', '', f'分析时间：{stamp}。', '',
        '**已完成 8/10 次拟合，整批复现尚未完成。** 种子 44/55/66/77 的两方法均完成 '
        '120 epochs / 46,800 updates；种子 88 的两个目录尚不存在。原控制器和八个训练进程已经退出，'
        '工具会话 60286 不再存在；旧队列文件仍停留在 running。控制器退出原因无法从现有记录确定，'
        '不能归因为用户暂停、OOM 或模型故障。', '',
        '下表取每个拟合按原规则自动保留的最小验证 CE 检查点。它们来自同一批 765 个旧验证对象，'
        '不是原计划的 dev_eval 结果；统一选模核查、六条件开发导出和五组主判断尚未执行。', '',
        '| 种子 | 五维选中轮次 | Distance 选中轮次 | 五维验证 mCA % | Distance 验证 mCA % | 差 pp |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in result['paired']:
        lines.append(f"| {r['seed']} | {r['relative5_epoch']} | {r['distance_epoch']} | "
            f"{r['relative5_mca_pct']:.3f} | {r['distance_mca_pct']:.3f} | {r['mca_gain_pp']:+.3f} |")
    m = result['means']
    lines += [f"| 四组描述均值 | — | — | {m['relative5_mca_pct']:.3f} | {m['distance_mca_pct']:.3f} | {m['mca_gain_pp']:+.3f} |", '',
        '当前信号是平均收益仍值得关注，但不是每次训练都获益。种子 44/55 的优势较明显；'
        '66 的 mCA 正向，但 CE 只低约 0.00244；77 的 mCA、OA 和 CE 都更差。'
        '这保留了训练随机性敏感的问题，不能用四组平均值宣布稳定超过 Distance，'
        '也不能把一个验证反例直接当成最终 dev_eval 失败。', '',
        '八次训练的损失和验证数值均有限，完整预算、文件身份、原最小 CE 规则、'
        '四组共同初始化与全部 120 轮实际采样流一致性已核对。'
        f"保存验证预测的独立指标复算最大差约 {result['independent_validation_metric_max_error']:.3g}。"
        '未重新推理，也没有按部分结果改选模型、导出 dev_eval 或开放官方 TEST。', '',
        '下一步应保留这八个完整拟合，只补齐原定种子 88 的两项；随后完成十模型统一选模核查、'
        '六条件开发评价与新五组主判断。不要追加额外种子或跳过 88。'
        '本次请求仅分析结果；自主研究目标仍暂停，没有启动补训或测试集下载。', '',
        '[完整数值与校验范围](result.json) · [训练/验证曲线](training_curves.png) · '
        '[四组验证差](validation_pairs.png)']
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
