"""Readable evidence, fixed examples and explicit limits of the visibility test."""
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import ListedColormap
from PIL import Image

from experiments.gaussian_memory_visibility_data import (
    OUT, CACHE, CONDITIONS, SEEDS, EARLY, LATE, load_plan, write_json)
from experiments.gaussian_memory_visibility_analysis import METRICS
from models.gaussian_memory_view_token import apply_rgb_mask

LABELS = {'full': 'Full input', 'query_only': 'Query masked', 'persistent': 'Context + writes masked'}
NAMES = {'native': 'Native', 'local': 'Local', 'content_static': 'C-static',
         'content_dynamic': 'C-dynamic', 'soft_static': 'Soft-static',
         'soft_dynamic': 'Soft-dynamic', 'center_dynamic': 'Center'}


def read(name): return json.loads((OUT / name).read_text())


def comp(a, condition, metric, key):
    return a['conditions'][condition][metric]['comparisons'][key]


def save(fig, name, pdf=True):
    folder = OUT / 'figures'; folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f'{name}.png', dpi=150, facecolor='white')
    if pdf: fig.savefig(folder / f'{name}.pdf', facecolor='white')
    plt.close(fig)


def figures(a):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    info = a['information']; places = a['places']
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), layout='constrained', gridspec_kw={'width_ratios': [1, 1.65]})
    keys = ['recent_full/roi', 'recent_persistent/roi', 'history_persistent/roi']
    values = [100 * info[k]['consistent_fraction']['mean'] for k in keys]
    axes[0].bar(range(3), values, color=['#566b85', '#e3a657', '#3f8f8b'])
    axes[0].set_xticks(range(3), ['Recent\nfull', 'Recent\nmasked', 'History\nmasked'])
    axes[0].set_ylabel('Sensor-consistent support / valid ROI (%)')
    axes[0].set_title('The intervention removes direct source support')
    axes[0].set_ylim(0, 100)
    for i, v in enumerate(values): axes[0].text(i, v+1, f'{v:.2f}%', ha='center')
    x = np.arange(12)
    for offset, key, label, color in [(-.18, 'gain_full/roi', 'Full source images', '#566b85'),
                                     (.18, 'gain_persistent/roi', 'Masked late source images', '#3f8f8b')]:
        v = info[key]['new_consistent_support_fraction']['by_place']
        axes[1].bar(x+offset, [100*v[p] for p in places], width=.35, label=label, color=color)
    axes[1].set_xticks(x, places, rotation=55, ha='right', fontsize=8)
    axes[1].set_ylabel('New consistent historical support / valid ROI (%)')
    axes[1].set_title('Four places have no early support; all remain included')
    axes[1].legend(fontsize=9)
    fig.suptitle('Input information diagnostic: source sensor geometry only\nFixed ROI, known cameras, no query-depth selection of masks')
    save(fig, 'information_control')

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout='constrained')
    families = ['native', 'content_static', 'content_dynamic', 'center_dynamic', 'soft_dynamic']
    colors = ['#888888', '#a75e46', '#d69b79', '#7695bb', '#287f76']
    for j, (fam, color) in enumerate(zip(families, colors)):
        key = fam if fam == 'native' else fam+'/history'
        values = [a['conditions'][c]['hidden_log_mse']['means'][key] for c in CONDITIONS]
        axes[0, 0].plot(range(3), values, '-o', label=NAMES[fam], color=color)
    axes[0, 0].set_xticks(range(3), [LABELS[c] for c in CONDITIONS], fontsize=9)
    axes[0, 0].set_ylabel('Fixed ROI native log-MSE (lower is better)')
    axes[0, 0].set_title('History states; same 12 places and three frozen seeds')
    axes[0, 0].legend(ncol=2, fontsize=9)
    labels = []; positions = []; j = 0
    for condition in CONDITIONS:
        for key, label in [('history:soft_dynamic_vs_content_static', 'Soft vs C-static'),
                           ('history:soft_dynamic_vs_center_dynamic', 'Soft vs Center'),
                           ('soft_dynamic:history_vs_recent', 'Soft History vs Recent')]:
            r = comp(a, condition, 'hidden_log_mse', key)
            lo, hi = r['relative_gain_percent_95']; mid = r['relative_gain_percent']
            axes[0, 1].errorbar(mid, j, xerr=[[mid-lo], [hi-mid]], fmt='o', color=colors[CONDITIONS.index(condition)+1], capsize=3)
            labels.append(LABELS[condition]+' / '+label); positions.append(j); j += 1
    axes[0, 1].axvline(0, color='gray', lw=1)
    axes[0, 1].set_yticks(positions, labels, fontsize=8); axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlabel('ROI relative error reduction (%)\nPaired place + seed 95% intervals')
    axes[0, 1].set_title('Gain over content supported; history/shape gain uncertain')
    means = a['conditions']['persistent']
    for j, fam in enumerate(['content_static', 'soft_dynamic']):
        key = fam+'/history'; bias = means['bias_squared']['means'][key]; spatial = means['centered_log_mse']['means'][key]
        axes[1, 0].bar(j, bias, color='#bf936a', label='Squared global log-depth bias' if j == 0 else None)
        axes[1, 0].bar(j, spatial, bottom=bias, color='#648da7', label='Centered spatial error' if j == 0 else None)
    axes[1, 0].set_xticks([0, 1], ['C-static / History', 'Soft-dynamic / History'])
    axes[1, 0].set_ylabel('Whole-image log-MSE components')
    axes[1, 0].set_title('Whole-image gain mainly accompanies lower global bias')
    axes[1, 0].legend(fontsize=9)
    support = info['gain_persistent/roi']['new_consistent_support_fraction']['by_place']
    gains = comp(a, 'persistent', 'hidden_log_mse', 'soft_dynamic:history_vs_recent')['place_gain']
    xx = np.array([support[p]*100 for p in places]); yy = np.array([gains[p] for p in places])
    axes[1, 1].scatter(xx, yy, c=np.where(xx > 0, '#287f76', '#999999'), s=45)
    for p, x, y in zip(places, xx, yy):
        if p in ('home_at', 'home_md', 'harvard_robotics_lab', 'brown_bm_1', 'mit_w59_conference'):
                offset = (-5, 22) if p == 'harvard_robotics_lab' else ((-5, 5) if p == 'home_md' else (5, 5))
                axes[1, 1].annotate(p, (x, y), xytext=offset, textcoords='offset points', fontsize=8,
                                    ha='right' if p in ('harvard_robotics_lab', 'home_md') else 'left')
    axes[1, 1].axhline(0, color='gray', lw=1)
    axes[1, 1].set_xlabel('Sensor newly consistent historical support in ROI (%)')
    axes[1, 1].set_ylabel('Soft Recent minus History ROI log-MSE')
    axes[1, 1].set_title('Largest history gain occurs without early ROI support\nDescriptive place plot; no causal attribution or exclusions')
    save(fig, 'native_results')


def examples(a):
    plan = load_plan(); places = a['places']
    gains = comp(a, 'persistent', 'hidden_log_mse', 'history:soft_dynamic_vs_content_static')['place_gain']
    worst_place = min(places, key=lambda p: (gains[p], p))
    worst = next(m['id'] for m in plan['catalog'] if m['place'] == worst_place)
    ids = list(dict.fromkeys(plan['fixed_figure_ids'] + [worst]))
    write_json(OUT / 'figure_selection.json', {'fixed_ids': plan['fixed_figure_ids'], 'worst_id': worst,
        'worst_place': worst_place, 'selected_ids': ids, 'seed_for_visualization': SEEDS[0],
        'criterion': plan['counterexample'], 'metric_selection_uses_all_seed_means': True})
    cmap = ListedColormap(['white', '#dddddd', '#4f987d', '#ce6457', '#8573ab'])
    for rid in ids:
        meta = next(m for m in plan['catalog'] if m['id'] == rid); folder = CACHE / rid
        saved = torch.load(folder / 'masks.pt', map_location='cpu', weights_only=False)
        sample = torch.load(folder / 'q11_c2.pt', map_location='cpu', weights_only=False)
        target, valid, roi = sample['target'].numpy(), sample['valid'].numpy(), sample['hidden'].numpy()
        export = torch.load(folder / 'predictions.pt', map_location='cpu', weights_only=False)
        def pred(fam, policy):
            key = [2, fam, None if fam == 'native' else SEEDS[0], policy]
            index = export['keys'].index(key)
            return export['prediction_native'][index].numpy() * sample['metric_scale']
        estimates = [pred('native', None), pred('soft_dynamic', 'recent'), pred('soft_dynamic', 'history'), pred('content_static', 'history')]
        errors = [np.abs(np.log(p) - np.log(target)) for p in estimates]
        vmax = max(.1, float(np.quantile(np.concatenate([e[valid] for e in errors]), .99)))
        fig, axes = plt.subplots(3, 4, figsize=(15, 9.8), layout='constrained')
        for col, f in enumerate((0, 6, 10, 11)):
            with Image.open(meta['row']['frames'][f]['rgb_path']) as im:
                rgb = torch.from_numpy(np.asarray(im.convert('RGB').resize((280, 210), Image.Resampling.BILINEAR)).copy()).permute(2, 0, 1).float()/255
            if f != 0: rgb = apply_rgb_mask(rgb[None], saved['patch_masks'][f:f+1])[0]
            axes[0, col].imshow(rgb.permute(1, 2, 0).numpy())
            axes[0, col].set_title('Early frame0 (unmasked)' if f == 0 else f'Current frame{f} (masked)', fontsize=9)
        for col, name in enumerate(('early', 'recent_persistent', 'history_persistent')):
            d = saved['sensor_support_maps'][name].numpy(); support = d > 0
            color = np.ones_like(target, dtype=np.uint8); color[~valid] = 0
            good = valid & support & (np.abs(d-target) / target <= .1)
            color[good] = 2; color[valid & support & ~good & (d < target)] = 3
            color[valid & support & ~good & (d >= target)] = 4
            axes[1, col].imshow(color, cmap=cmap, vmin=0, vmax=4, interpolation='nearest')
            axes[1, col].set_title('Sensor '+name.replace('_', ' '), fontsize=9)
        axes[1, 3].imshow(np.where(valid, target, np.nan), cmap='viridis', vmin=.1, vmax=float(np.quantile(target[valid], .99)))
        axes[1, 3].set_title('Query sensor depth (score only)', fontsize=9)
        for col, (error, title) in enumerate(zip(errors, ['Native', 'Soft / Recent', 'Soft / History', 'C-static / History'])):
            im = axes[2, col].imshow(np.where(valid, error, np.nan), cmap='magma', vmin=0, vmax=vmax)
            axes[2, col].set_title(f'{title}: ROI log-MSE {np.mean(error[valid & roi]**2):.3f}', fontsize=9)
        ys, xs = np.where(roi); x0, y0, width, height = xs.min()-.5, ys.min()-.5, xs.max()-xs.min()+1, ys.max()-ys.min()+1
        for row_axes in axes[1:]:
            for ax in row_axes: ax.add_patch(Rectangle((x0, y0), width, height, fill=False, ec='#27d4e2', lw=1.1))
        for ax in axes.flat: ax.axis('off')
        fig.colorbar(im, ax=axes[2, :], fraction=.025, pad=.01, label='Absolute log-depth error')
        qualifier = 'fixed example' if rid in plan['fixed_figure_ids'] else 'prespecified adverse-case selection'
        fig.suptitle(f'{rid} / {meta["place"]} / {qualifier}\nPersistent condition; query11; seed {SEEDS[0]}. Cyan rectangle is the same scoring ROI.')
        legend = [Patch(color=cmap(i), label=n) for i, n in enumerate(['Target invalid', 'No sensor projection', 'Consistent within 10%', 'Too near', 'Too far'])]
        fig.legend(handles=legend, loc='outside lower center', ncol=5, fontsize=9)
        save(fig, f'example_{rid}', pdf=False)


def fmt(r):
    lo, hi = r['relative_gain_percent_95']
    return f'{r["relative_gain_percent"]:+.2f}% [{lo:+.2f}, {hi:+.2f}]'


def text_report(a):
    checks = read('independent_checks.json'); info = a['information']; cost = a['cost']
    lines = ['# history_visibility_control1：强内容对照出现收益，历史补缺和形状增量仍待建立', '',
        '2026-11-11。有限实验完成。**保留 Soft 高斯读取作为开发候选；当前条件下超过强内容的收益值得继续，但尚未建立由窗口外历史补缺、特别是高斯形状带来的可靠增量。**', '',
        '## 最直接的结论', '',
        '在一致遮挡条件下，History-Soft-dynamic 的固定区域原生 log-MSE 比 History-C-static 低16.21%，三个种子同向、9/12地点改善，地点×种子配对95%区间为[1.58%,26.41%]；相对匹配训练的 C-dynamic 低30.10%，区间[4.84%,38.34%]。全图相对强 C-static 也低14.57%，区间[1.70%,23.90%]。这是本轮明确的开发信号，不应因为其他比较不确定而抹掉。', '',
        '但同一 Soft 加入早期历史后，固定区域仅平均改善6.90%，区间[-8.37%,36.06%]；相对 Center 的14.49%均值优势区间为[-10.82%,30.68%]。对 Native/Local 的区域平均优势也未越过零。因而目前没有完成“历史有额外用途→高斯形状比强对照更适合读取这份历史”的完整证据链。', '',
        '## 固定实验与信息边界', '',
        '使用全部12个已用 fidelity 片段，每个片段早期0/1/2/3、晚期写入4/6/8/10、查询11、当前四帧6/8/10/11。Recent只由四个晚期来源构建，History再包含四个早期来源；两者均256槽Prefix重建，不声称常量成本在线更新。', '',
        '三条件为完整输入、仅查询mask0、晚期所有上下文及写入一致遮挡。用早期源传感器点和已知相机确定落入固定query11 mask0矩形的表面，然后遮挡其在晚期输入触及的14×14 patch。早期RGB若被晚期编码重用，也用遮挡副本；早期记忆保持完整。查询深度只在构造完成后评分，没有用于选择掩码。未依照结果换地点、掩码或阈值。', '',
        '源传感器仅用于人工任务构造和信息诊断；候选状态仍使用VGGT预测几何与原置信筛选，也允许网络在被遮挡patch推断内容。所有条件共用原完整输入标定和相机，属于固定标定的特权构造诊断，不是严格纯RGB部署。仅查询条件的原特征/预测可复用，但按本轮共同尺度重新评分。', '',
        '主区域始终是同一个mask0矩形，完整输入也在该区域计分；全图、区域外和误差分解同时报告。冻结五组读取器各三种子及三个Local，Native保留；不训练、不下载。全部数据已用，无独立确认。统计单位是12地点与3个配对模型种子；Native没有三个独立训练种子。', '',
        '## 遮挡是否制造了可补充的信息', '',
        '| 源传感器诊断（固定区域有效像素为分母） | 比例 |', '|---|---:|']
    for label, key, metric in [
        ('早期自身一致支持', 'early/roi', 'consistent_fraction'),
        ('近期完整来源一致支持', 'recent_full/roi', 'consistent_fraction'),
        ('近期一致遮挡后的一致支持', 'recent_persistent/roi', 'consistent_fraction'),
        ('历史一致遮挡后的一致支持', 'history_persistent/roi', 'consistent_fraction'),
        ('完整来源下新增且一致的历史支持', 'gain_full/roi', 'new_consistent_support_fraction'),
        ('一致遮挡下新增且一致的历史支持', 'gain_persistent/roi', 'new_consistent_support_fraction')]:
        lines.append(f'| {label} | {100*info[key][metric]["mean"]:.3f}% |')
    lines += ['', '近期直接一致支持由79.99%降至50.46%，历史新增且一致支持由0.237%升至11.136%，后者地点区间[3.563%,19.921%]。因此在整体均值上，干预确实制造了可补充的直接表面信息，不能把本轮仍不稳定的输出收益全部归于“历史完全没信息”。', '',
        'brown_cogsci_1、brown_cs_3、home_at、mit_32_g725四个地点早期没有覆盖固定区域，晚期源掩码均为空，故一致遮挡与仅查询遮挡相同；全部保留。其余地点也并非移除了区域内全部其他线索，当前直接一致支持仍有50.46%。原始传感器信息不能证明候选的预测高斯状态准确保存了这些表面。', '',
        '![输入信息变化](figures/information_control.png)', '',
        '## 固定原生比较', '',
        '以下为误差降低百分比及地点×种子95%配对区间，正值有利于候选。均使用History状态，历史/近期一行例外。该开发诊断没有多重比较校正，也没有未用数据确认。', '',
        '| 比较 / 固定区域 | 完整输入 | 仅查询遮挡 | 上下文与写入一致遮挡 |', '|---|---|---|---|']
    for label, key in [('Soft对强C-static', 'history:soft_dynamic_vs_content_static'),
                       ('Soft对C-dynamic', 'history:soft_dynamic_vs_content_dynamic'),
                       ('Soft对Center', 'history:soft_dynamic_vs_center_dynamic'),
                       ('Soft对Native', 'history:soft_dynamic_vs_native'),
                       ('Soft对Local', 'history:soft_dynamic_vs_local'),
                       ('Soft History对Recent', 'soft_dynamic:history_vs_recent')]:
        lines.append('| '+label+' | '+' | '.join(fmt(comp(a,c,'hidden_log_mse',key)) for c in CONDITIONS)+' |')
    lines += ['', '| 一致遮挡 / 全图及区域外比较 | 降低比例和区间 |', '|---|---|']
    for label, metric in [('全图：Soft对C-static', 'log_mse'), ('区域外：Soft对C-static', 'visible_log_mse'),
                          ('去均值空间误差：Soft对C-static', 'centered_log_mse'), ('整体偏差平方：Soft对C-static', 'bias_squared')]:
        lines.append(f'| {label} | {fmt(comp(a,"persistent",metric,"history:soft_dynamic_vs_content_static"))} |')
    lines += ['', '全图收益主要伴随整体深度偏差下降；去均值空间误差相对强C只低0.16%，种子混合、区间跨零。这保留“几何偏好帮助稳定粗场景读取”的解释，尚不支持更准确的精细表面恢复。', '',
        '历史增量从仅查询遮挡的6.86%变为一致遮挡的6.90%，不能仅凭两项均值接近宣称等效；配对的原始log-MSE增益变化为+0.001874，区间[-0.017331,+0.022659]，没有建立干预使历史更有用的证据。Soft在只有Recent状态时也已比C-static平均低12.74%，所以超过内容的收益并不专属于窗口外历史。', '',
        '最大的Soft历史增益出现在home_at，而该地点没有早期区域内的直接表面支持；这可能来自区域外语义、整体偏差、记忆组成变化等，不能当成局部表面被补回的证明。这里是事后描述，用于限制因果解释，未删地点或追加显著性筛选。', '',
        '![原生输出与边界](figures/native_results.png)', '',
        '## 实现核对与代价', '',
        f'- 1224份输出全部独立按原始标签评分；最大指标数值差{max(checks["max_score_difference"].values()):.3g}。72份未改变输入的Local预测与父轮最大差{checks["old_local_max_absolute_prediction_difference"]:.3g}。',
        '- 全12地点的源掩码另用NumPy齐次变换独立构造，掩码差0个patch；36张传感器投影图支持像素差0。48份状态均保留256槽、来源符合时间限制，查询不写入。',
        '- 固定fidelity_00的完整/仅查询骨干重放与原缓存前缀、特征和原生深度完全一致。源文件、检查点与输入缓存身份均核对。',
        f'- GPU 2、5各完成6片段，worker耗时{cost["worker_seconds"][0]:.2f}/{cost["worker_seconds"][1]:.2f}秒，峰值已分配{cost["peak_allocated_GiB"]:.3f}GiB；使用12GiB上限并保护其他任务。计时不是独占性能基准。',
        '- 62个新增冻结骨干窗口，1188次读取器原生输出；12份新Native与24份原Native共同重新评分。零训练、零下载。', '',
        '## 方法取舍与下一步', '',
        '保留本轮对强内容读取的正面证据，停止把同一个冻结读取器在更多掩码下的小幅变化当作路线推进。接下来最有辨别价值的是一次固定预算的任务匹配训练：只训练C、Soft高斯、Center三个现有读法，让它们都见过完整/仅查询/一致遮挡及Recent/History状态；相同输入、初值、采样和预算，并保留当前强旧C与冻结模型。模型结构和几何公式保持精炼。', '',
        '训练限于原训练地点，评价地点仍隔离，但本轮12地点已反复使用，只能作为开发验证。若匹配训练后历史增益随缺失信息增加，且高斯稳定超过强C与Center，再考虑未用数据确认；若只有内容级收益，就收窄为共享记忆读取；若历史仍无可靠增量，结束这项固定Prefix读取接法的推广，不继续逐项堆补。该后续只提出，尚未执行。', '',
        '具体建议见[history_visibility_train1](../../../docs/gaussian_memory_visibility_training_plan.md)。训练来源可用性已作只读检查，未启动训练。', '',
        '这轮没有证明一般高斯PE有效，也没有推翻早期解耦架构结论或恢复静态积分核路线。旧实验的所有负结果保留；本轮正面信号对应的是当前内部VGGT读取器、固定相机/标定与人工可见性条件。', '',
        '## 固定例图与反例', '',
        '固定fidelity_00/01/02；另按预定Soft对强C最差地点规则选反例。主统计使用三个种子，例图使用第一固定种子。第一行实际RGB与遮挡；第二行为传感器信息诊断及查询标签；第三行为实际网络输出误差，色阶仅在同一例图内共享。浅灰表示无传感器投影，不是网络拒绝输出。']
    for rid in read('figure_selection.json')['selected_ids']:
        lines += ['', f'![{rid}](figures/example_{rid}.png)']
    if (OUT / 'visual_review.json').exists():
        lines += ['', f'已实际审阅{len(read("visual_review.json")["images"])}幅最终PNG；详见visual_review.json。']
    (OUT / 'report.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    torch.set_num_threads(4); a = read('analysis.json')
    assert read('independent_checks.json')['complete']
    figures(a); examples(a); text_report(a)
    print(json.dumps({'report_generated': True, 'pngs': len(list((OUT / 'figures').glob('*.png')))}), flush=True)
