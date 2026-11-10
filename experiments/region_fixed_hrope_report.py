"""Readable tables and figures; scientific interpretation is reviewed separately."""
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.round65_common import SEEDS, read, write, sha
from experiments.region_fixed_hrope import OUT, DEV
from models.region_hrope_control import ROLES as OLD_ROLES
ROLES = (*OLD_ROLES, "fixed_hrope")

def owner(role):
    return OUT if role == "fixed_hrope" else DEV

LABELS = dict(relative5='Full five', distance_mlp='Distance', geotransformer_rpe='Geo PE',
              fixed_isotropic5='Fixed sphere', predicted_hrope='Predicted HRoPE', fixed_hrope='Fixed H=K')
COLORS = dict(zip(ROLES, ['#1769aa', '#cc7b28', '#7a5aa6', '#4b8e64', '#c94c64', '#444444']))


def main():
    result = read(OUT/'result.json')
    checks = read(OUT/'independent_checks.json')
    assert checks['passed'] and checks['result_sha256'] == sha(OUT/'result.json')
    assert not (OUT/'tables.md').exists()
    figdir = OUT/'figures'
    figdir.mkdir(exist_ok=False)
    lines = ['# 固定H=K控制：完整数据表', '',
        '旧五场景480帧对；新增固定H的三次拟合，复用此前五方法十五个模型。所有方法使用相同图像位置与内参。'
        '对齐采用全部64个top1对应、统一刚体求解器；成功为RMSE<0.2米。', '',
        '## 对齐用途', '', '| 方法 | 成功率均值 % | 三种子成功率 % |', '| --- | ---: | --- |']
    for role in ROLES:
        rows = [next(r for r in result['records'] if (r['role'],r['seed']) == (role,s)) for s in SEEDS]
        values = [100*r['macro_success_rate'] for r in rows]
        lines.append(f'| {LABELS[role]} | {np.mean(values):.3f} | '+ '/'.join(f'{v:.3f}' for v in values)+' |')
    lines += ['', '| 比较 | 平均差 pp | 三种子差 pp | 种子×场景95%区间 pp |', '| --- | ---: | --- | --- |']
    for row in result['comparisons']:
        lo, hi = row['intervals_95_pp']['crossed']
        deltas = '/'.join(f'{v:+.3f}' for v in row['paired_seed_gains_pp'])
        lines.append(f"| {LABELS[row['left']]} − {LABELS[row['right']]} | {row['mean_gain_pp']:+.3f} | {deltas} | [{lo:.3f}, {hi:.3f}] |")
    for role in ('identity','gt_nearest'):
        row = next(r for r in result['records'] if r['role'] == role)
        lines += ['', f"{role}: {100*row['macro_success_rate']:.3f}%。复用同缓存的既有参照；gt_nearest为不可部署诊断，不是理论上界。"]
    lines += ['', '## 逐种子错误与辅助匹配指标', '',
        '| 方法/种子 | 区域R@1 % | 全查询正确 % | 中心PCK % | Soft EPE m | 对齐RMSE中位/p90 m | 旋转p90 ° | 中心化平移p90 m |',
        '| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |']
    metadata = []
    for role in ROLES:
        for seed in SEEDS:
            row = next(r for r in result['records'] if (r['role'],r['seed']) == (role,seed))
            metrics = read(owner(role)/'evaluation'/role/str(seed)/'base.json')['metrics']
            m, tail = metrics['macro'], row['all_pair_errors']
            correct = np.mean([v['valid_and_correct_fraction'] for v in metrics['by_scene'].values()])
            def f(v):
                return 'inf' if v is None else f'{v:.4f}'
            lines.append(f'| {LABELS[role]}/{seed} | {100*m["region_recall"]:.3f} | {100*correct:.3f} | '
                f'{100*m["center_pck"]:.3f} | {m["soft_epe"]:.4f} | {f(tail["rmse_m"]["median"])}/{f(tail["rmse_m"]["p90"])} | '
                f'{f(tail["rotation_error_deg"]["p90"])} | {f(tail["centered_translation_error_m"]["p90"])} |')
            run = read(owner(role)/'runs'/role/str(seed)/'run.json')
            history = read(owner(role)/'runs'/role/str(seed)/'history.json')
            metadata.append(dict(role=role, seed=seed, selected_epoch=run['selected_epoch'],
                train_loss_first=history['training'][0]['loss'], train_loss_last=history['training'][-1]['loss'],
                validation_ce_initial=history['validation'][0]['selection_ce'], validation_ce_selected=run['best_val_ce'],
                parameters=run['parameters'], pe_parameters=run['pe_parameters'],
                epochs=run['epochs'], steps=run['steps'], seconds=run['seconds'], peak_allocated_bytes=run['peak_allocated']))
    lines += ['', '## 训练与代价', '',
        '所有方法60周期/1,080更新，仅按验证CE选模。耗时受卡和并发占用影响，不能据此声称公平速度优势。', '',
        '| 方法/种子 | 选中周期 | train CE 首→末 | val CE 初始→选中 | 总参数/PE参数 | 秒 | 峰值GiB |',
        '| --- | ---: | --- | --- | --- | ---: | ---: |']
    for r in metadata:
        lines.append(f'| {LABELS[r["role"]]}/{r["seed"]} | {r["selected_epoch"]} | {r["train_loss_first"]:.4f}→{r["train_loss_last"]:.4f} | '
            f'{r["validation_ce_initial"]:.4f}→{r["validation_ce_selected"]:.4f} | {r["parameters"]}/{r["pe_parameters"]} | {r["seconds"]:.1f} | {r["peak_allocated_bytes"]/2**30:.3f} |')
    (OUT/'tables.md').write_text('\n'.join(lines)+'\n')
    write(OUT/'training_summary.json', dict(records=metadata, selected_sha256=sha(OUT/'selected.json'),
        complete_fits=len(metadata), new_fits=3, reused_fits=15, same_common_initialization_and_augmented_streams=True,
        isolated_speed_measurement=False, scientific_review_pending=True))

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout='constrained')
    for i, role in enumerate(ROLES):
        values = [100*r['macro_success_rate'] for r in result['records'] if r['role'] == role]
        axes[0].bar(i, np.mean(values), color=COLORS[role], width=.6, alpha=.75)
        axes[0].scatter(i+np.linspace(-.1,.1,3), values, color=COLORS[role], edgecolor='white', s=25, zorder=3)
        axes[0].text(i, max(values)+1.5, f'{np.mean(values):.2f}', ha='center', fontsize=10)
    axes[0].set_xticks(range(len(ROLES)), [LABELS[r] for r in ROLES], rotation=25, ha='right')
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel('Scene-balanced alignment success (%)')
    axes[0].set_title('RMSE < 0.2 m; all 480 pairs')
    for i, row in enumerate(result['comparisons']):
        lo, hi = row['intervals_95_pp']['crossed']
        color = '#1769aa' if row['left'] == 'relative5' else '#c94c64'
        axes[1].plot([lo,hi],[i,i],lw=2,color=color)
        axes[1].plot(row['mean_gain_pp'],i,'o',color=color)
        axes[1].scatter(row['paired_seed_gains_pp'],i+np.linspace(-.09,.09,3),marker='x',color=color,s=25)
    axes[1].set_yticks(range(len(result['comparisons'])), [LABELS[r['left']]+' − '+LABELS[r['right']] for r in result['comparisons']])
    axes[1].set_ylim(len(result['comparisons'])-.4,-.6)
    axes[1].axvline(0,color='.5',ls='--',lw=1)
    axes[1].set_xlabel('Difference (percentage points)')
    axes[1].set_title('Paired seeds; 95% seed × scene intervals')
    axes[0].grid(axis='y',alpha=.15); axes[1].grid(axis='x',alpha=.15)
    for ax in axes:
        ax.set_axisbelow(True)
    figure.suptitle('Camera-information-matched regional PE: old-data development comparison', fontsize=12)
    figure.savefig(figdir/'pose_comparison.png',dpi=170)
    plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(13, 7), layout='constrained')
    for col, seed in enumerate(SEEDS):
        for role in ROLES:
            h = read(owner(role)/'runs'/role/str(seed)/'history.json')
            axes[0,col].plot([r['epoch'] for r in h['training']], [r['loss'] for r in h['training']], label=LABELS[role], color=COLORS[role])
            axes[1,col].plot([r['epoch'] for r in h['validation']], [r['selection_ce'] for r in h['validation']], color=COLORS[role])
        axes[0,col].set_title(f'Seed {seed}')
        axes[1,col].set_xlabel('Epoch')
        for ax in axes[:,col]:
            ax.grid(alpha=.2)
    axes[0,0].set_ylabel('Training CE')
    axes[1,0].set_ylabel('Validation CE (selection)')
    axes[0,2].legend(fontsize=8)
    figure.suptitle('3 new fixed-H fits and 15 reused controls; same budget',fontsize=12)
    figure.savefig(figdir/'learning_curves.png',dpi=170)
    plt.close(figure)
    write(figdir/'manifest.json', dict(created_unix=time.time(), result_sha256=sha(OUT/'result.json'),
        renderer_sha256=sha(__file__), figures={p.name:sha(p) for p in sorted(figdir.glob('*.png'))},
        visual_review_pending=True, fresh_confirmation=False))
    print('HROPE_TABLES_AND_FIGURES_READY', str(figdir), flush=True)


if __name__ == '__main__':
    main()
