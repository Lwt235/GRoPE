"""Report the limited historical information and geometry fidelity evidence."""
import json

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from experiments.gaussian_memory_raw_history_data import OUT, CACHE, BASE_CACHE, POLICIES, load_plan, write_json
from experiments.gaussian_memory_incremental_data import sensor_target
from experiments.gaussian_memory_view_data import rectangle

POLS = {'recent_corrective': 'Recent / Corrective', 'history_corrective': 'History / Corrective',
        'recent_rebuild': 'Recent / Prefix', 'history_rebuild': 'History / Prefix'}


def read(n): return json.loads((OUT / n).read_text())


def item(data, kind, key, metric, mode='mixed'): return data['summary'][kind][mode][key][metric]


def pct(x): return '—' if x is None else f'{100*x:.3f}%'


def ci(x): return '—' if x['place_95'] is None else '[' + ', '.join(f'{100*v:.3f}' for v in x['place_95']) + ']'


def save(fig, name, pdf=True):
    fig.savefig(OUT / 'figures' / f'{name}.png', dpi=160)
    if pdf: fig.savefig(OUT / 'figures' / f'{name}.pdf')
    plt.close(fig)


def figures(data):
    (OUT / 'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig = plt.figure(figsize=(14.8, 9), constrained_layout=True); g = fig.add_gridspec(2, 2)
    ax1, ax2, ax3 = fig.add_subplot(g[0, 0]), fig.add_subplot(g[0, 1]), fig.add_subplot(g[1, :])
    keys = ['sensor/recent', 'sensor/history', 'prediction/recent', 'prediction/history']; x = np.arange(4)
    for off, metric, label, color in [(-.18, 'valid_support_fraction', 'Projected support', '#517690'),
        (.18, 'consistent_fraction', 'Within 10% target depth', '#387e73')]:
        ax1.bar(x+off, [100*item(data, 'raw_rows', k, metric)['mean'] for k in keys], width=.36, label=label, color=color)
    ax1.set_xticks(x, ['Sensor\nRecent', 'Sensor\nHistory', 'Predicted\nRecent', 'Predicted\nHistory'])
    ax1.set_ylabel('% of valid query pixels'); ax1.set_title('Raw-source support versus geometric consistency'); ax1.legend(fontsize=8)
    for j, (mode, v) in enumerate([(m, v) for m in ('mixed', 'hidden') for v in ('sensor', 'prediction')]):
        result = item(data, 'history_rows', v, 'consistency_net_gain', mode)
        ax2.plot(np.array(result['place_95'])*100, [j, j], lw=2, color='#517690')
        ax2.scatter(result['mean']*100, j, color='#c96e2c')
    ax2.axvline(0, color='#999999'); ax2.set_yticks(range(4), ['Sensor / all', 'Prediction / all', 'Sensor / hidden', 'Prediction / hidden'])
    ax2.set_title('History minus Recent: net consistent support'); ax2.set_xlabel('Percentage points; 95% paired-place interval')
    places = data['places']; x = np.arange(len(places))
    for offset, v, color in [(-.18, 'sensor', '#387e73'), (.18, 'prediction', '#c96e2c')]:
        values = item(data, 'history_rows', v, 'new_consistent_support_fraction')['by_place']
        ax3.bar(x+offset, [100*values[p] for p in places], width=.36, label=v, color=color)
    ax3.set_xticks(x, places, rotation=22, ha='right'); ax3.set_ylabel('% of valid query pixels')
    ax3.set_title('Newly supported AND geometrically consistent pixels from early history'); ax3.legend()
    fig.suptitle('12 previously used places; fixed sources and all 48 queries\nMasks change scoring regions only; raw projection is not Transformer output utility')
    save(fig, 'raw_source_information')

    fig, axes = plt.subplots(2, 2, figsize=(14.8, 9), constrained_layout=True)
    x = np.arange(12)
    for v, color in [('sensor_matched', '#387e73'), ('prediction_matched', '#c96e2c')]:
        values = item(data, 'raw_rows', v+'/history', 'consistent_fraction')['by_place']
        axes[0, 0].plot(x, [100*values[p] for p in places], 'o-', label=v.replace('_matched', ''), color=color)
    axes[0, 0].set_xticks(x, range(1, 13)); axes[0, 0].set_xlabel('Place index in the same sorted order')
    axes[0, 0].set_ylabel('% of valid query pixels'); axes[0, 0].set_title('Same source pixels: consistent support'); axes[0, 0].legend()
    x = np.arange(4)
    for offset, metric, label, color in [(-.18, 'valid_support_fraction', 'State covers', '#517690'),
        (.18, 'consistent_fraction', 'State depth within 10%', '#387e73')]:
        axes[0, 1].bar(x+offset, [100*item(data, 'state_rows', p, metric)['mean'] for p in POLICIES], width=.36, label=label, color=color)
    labels = ['Recent\nCorrective', 'History\nCorrective', 'Recent\nPrefix', 'History\nPrefix']
    axes[0, 1].set_xticks(x, labels); axes[0, 1].set_ylabel('% of valid query pixels')
    axes[0, 1].set_title('Existing coarse state: coverage and depth fidelity'); axes[0, 1].legend(fontsize=8)
    for offset, metric, label, color in [(-.18, 'state_covers_raw_new_fraction', 'State covers', '#517690'),
        (.18, 'state_consistent_on_raw_new_fraction', 'State depth within 10%', '#387e73')]:
        axes[1, 0].bar(x+offset, [100*item(data, 'retention_rows', 'prediction/'+p, metric)['mean'] for p in POLICIES], width=.36, label=label, color=color)
    axes[1, 0].set_xticks(x, labels); axes[1, 0].set_ylabel('% of new consistent predicted-source support')
    axes[1, 0].set_title('Conditional retention: only 4 places have this support'); axes[1, 0].legend(fontsize=8)
    values = [100*item(data, 'retention_rows', 'prediction/'+p, 'retained_consistent_fraction_all_valid')['mean'] for p in POLICIES]
    axes[1, 1].bar(x, values, color='#517690'); axes[1, 1].set_xticks(x, labels)
    axes[1, 1].axhline(100*item(data, 'history_rows', 'prediction', 'new_consistent_support_fraction')['mean'], color='#c96e2c', ls='--', label='Raw predicted-source addition')
    axes[1, 1].set_ylabel('% of ALL valid query pixels'); axes[1, 1].set_title('Absolute retained support: all 12 places'); axes[1, 1].legend(fontsize=8)
    fig.suptitle('Prediction geometry already differs before compression; broad footprints do not guarantee correct depth\nState projection is a diagnostic of the existing interface, not its learned final depth prediction')
    save(fig, 'geometry_and_state_retention')


def classify(depth, target, valid):
    result = np.zeros(depth.shape, dtype=int)
    present = (depth > 0) & valid
    result[present & (depth >= target*.9) & (depth <= target*1.1)] = 1
    result[present & (depth < target*.9)] = 2
    result[present & (depth > target*1.1)] = 3
    result[~valid] = 4
    return result


def examples(data):
    plan = load_plan(); catalog = {m['id']: m for m in plan['catalog']}
    values = item(data, 'history_rows', 'sensor', 'new_consistent_support_fraction')['by_place']
    worst_place = min(sorted(values), key=lambda p: values[p]); worst = next(p['id'] for p in plan['pairs'] if p['place'] == worst_place)
    fixed = ['fidelity_00', 'fidelity_01', 'fidelity_02']; selected = list(dict.fromkeys(fixed + [worst]))
    write_json(OUT / 'figure_selection.json', {'fixed': fixed, 'selected': selected, 'worst_place': worst_place,
        'worst_id': worst, 'worst_tie_rule': 'Alphabetical place order among equal minima.',
        'criterion': 'Smallest sensor-source newly supported and consistent fraction across all queries, no method-score selection.',
        'query': 11, 'conditions': [0, 1], 'scope': 'Descriptive fixed examples plus adverse case; no queries excluded.'})
    cmap = ListedColormap(['#eeeeee', '#248f80', '#d66446', '#7467a6', '#ffffff', '#cacaca'])
    labels = ['No projection', 'Consistent (within 10%)', 'Too near', 'Too far', 'Invalid target', 'Outside hidden region']
    legend = [Patch(color=c, label=l) for c, l in zip(cmap.colors, labels)]
    for rid in selected:
        maps = torch.load(CACHE / 'maps' / f'{rid}.pt', map_location='cpu', weights_only=False)[11]
        target, valid = sensor_target(catalog[rid]['row']['frames'][11]); target = target.numpy(); valid = valid.numpy()
        arrays = [('Sensor Recent', maps['raw']['sensor']['recent']), ('Sensor History', maps['raw']['sensor']['history']),
                  ('Predicted Recent', maps['raw']['prediction']['recent']), ('Predicted History', maps['raw']['prediction']['history']),
                  ('History Corrective', maps['state']['history_corrective']['depth_m']), ('History Prefix', maps['state']['history_rebuild']['depth_m'])]
        fig, axes = plt.subplots(2, 8, figsize=(21, 6.3), constrained_layout=True)
        for row, condition in enumerate((0, 1)):
            sample = torch.load(BASE_CACHE / rid / f'q11_c{condition}.pt', map_location='cpu', weights_only=False)
            axes[row, 0].imshow(sample['rgb'].numpy()); axes[row, 0].set_title('Full query RGB' if condition == 0 else 'Mask0 query RGB', fontsize=9)
            mask = np.ones_like(valid) if condition == 0 else rectangle(rid, 11, 0).numpy().reshape(15, 20).repeat(14, 0).repeat(14, 1)
            axes[row, 1].imshow(np.where(valid & mask, target, np.nan), cmap='viridis', vmin=0, vmax=np.quantile(target[valid], .98))
            axes[row, 1].set_title('Sensor target depth', fontsize=9)
            for col, (label, depth) in enumerate(arrays, 2):
                color = classify(depth, target, valid); color[~mask] = 5
                axes[row, col].imshow(color, vmin=0, vmax=5, cmap=cmap, interpolation='nearest'); axes[row, col].set_title(label, fontsize=9)
            for ax in axes[row]: ax.axis('off')
        qualifier = 'fixed + lowest added support' if rid == worst and rid in fixed else ('lowest added support' if rid == worst else 'fixed')
        fig.suptitle(f'{rid}: {qualifier}; query11; same source maps in both rows\nTop: all pixels. Bottom: hidden region only. Target labels classify consistency after projection.')
        fig.legend(handles=legend, loc='lower center', ncol=6, fontsize=9)
        fig.set_constrained_layout_pads(h_pad=.3, w_pad=.03)
        save(fig, f'example_{rid}', pdf=False)
    return selected


def text_report(data):
    checks = read('independent_checks.json'); cost = read('projection_complete.json'); source = read('extraction_complete.json')
    lines = ['# raw_history_information1：历史补充量有限，预测几何在压缩前已有明显误差', '',
        '2026-11-11。有限输入信息诊断已完成，全部旧地点保留。**下一步优先改变能够检验历史作用的任务条件；暂不继续修PE公式、扩大记忆或追求更精确的静态投影。**', '',
        '## 这轮回答了什么', '',
        '在现有12对稀疏窗口里，早期历史新增、且与目标深度相差不超过10%的支持，平均仅占查询有效像素的0.858%，六个地点完全没有这种新增支持；预测几何下只有0.249%，八个地点为零。近期观测已经提供大量相关信息。', '',
        '同样的来源像素、相机和投影规则下，传感器来源与预测来源的历史一致支持率为70.768%/23.028%，差47.740个百分点，12个地点均为前者更高，配对95%区间[36.568,59.756]个百分点。这说明误差在进入256槽压缩之前就已明显存在，不能把全部问题归于容量或PE。尚未区分源深度结构、米制标定、时序/位姿误差各自贡献。', '',
        '已有高斯状态的覆盖范围与几何一致性也不同。它是粗区域/内容记忆，投影中心的合成深度不是最终网络预测，不能以它低于稠密传感器投影来宣判PE无效。它支持保留表示损失这个竞争解释，但不足以单独定位writer。', '',
        '## 假设、固定输入与边界', '',
        '假设是：原始历史包含当前查询需要的补充几何，但压缩、更新或attention读取丢失了它。竞争解释包括历史与查询缺乏重叠、近期已覆盖同一区域，以及预测几何/坐标标定不够可靠。', '',
        '沿用history_read1的同12对窗口、4查询、完整/两遮挡条件。Early=A的0/1/2/3/4/6/8/10；Recent=B的0/1/2/3及查询前写入帧；History为并集。Recent是晚期来源集合，不只是Native最近四帧。所有只读查询帧都不进入来源。没有模型选择、场景排除、拟合或下载。', '',
        '192个来源帧传感器深度均已下载。旧缓存只保存区域摘要，因此冻结原VGGT，重放24片段×5原因果四帧窗口，补存192张来源稠密深度；原来的相机基线标定尺度保持不变。源传感器只用于诊断，未注入候选模型；查询深度仅用于投影后的评分。', '',
        '固定210×280网格、像素中心+0.5、源深度(0.1,10)米、最前方正z、最近像素落点、不补洞；一致性容差10%沿用writer原值。主两组分别使用自身有效来源像素；匹配两组使用双方均有效的同一来源像素。没有改变阈值或扫描分辨率。', '',
        '48个相机投影不因人工遮挡改变；遮挡只划分被评分区域。混合、完整和全像素遮挡指标因而重复同一投影，不能当作独立复现。隐藏区域另报。统计单位是12个地点，5000次地点配对重采样，无新增模型种子；以下区间不是多重比较校正后的确认。', '',
        '## 原始支持和新增支持', '',
        '表中前两列均以查询有效深度像素为分母。一致支持要求既有投影，又满足固定10%深度容差；不支持的像素计入分母。条件AbsRel只在有投影且有目标深度的像素计算，各查询先平均再地点等权。', '',
        '| 来源 | 范围 | 投影覆盖 | 一致支持 | 有支持区域AbsRel |', '|---|---|---:|---:|---:|']
    for variant in ('sensor', 'prediction', 'sensor_matched', 'prediction_matched'):
        for rge in ('early', 'recent', 'history'):
            key = variant+'/'+rge
            lines.append(f'| {variant} | {rge} | {pct(item(data,"raw_rows",key,"valid_support_fraction")["mean"])} | {pct(item(data,"raw_rows",key,"consistent_fraction")["mean"])} | {item(data,"raw_rows",key,"supported_absrel")["mean"]:.4f} |')
    lines += ['', '| 来源/区域 | 新增投影覆盖（全部区域像素） | 新增且一致（有效区域像素） | 其95%区间 | 有新增一致的地点 |', '|---|---:|---:|---|---:|']
    for mode, label in [('mixed','全图'),('hidden','人工隐藏区')]:
        for v in ('sensor','prediction'):
            new = item(data,'history_rows',v,'new_consistent_support_fraction',mode)
            lines.append(f'| {v}/{label} | {pct(item(data,"history_rows",v,"new_support_fraction",mode)["mean"])} | {pct(new["mean"])} | {ci(new)} | {new["positive_places"]}/12 |')
    lines += ['', '新增支持并不等于总一致支持净增：早期来源也可能把近期正确表面替换成错误的更近投影。传感器来源全图净增仅0.205个百分点，区间[-0.746,1.436]；预测来源净增0.241个百分点，区间[-0.100,0.632]。人工隐藏区传感器净变为−0.217个百分点，区间同样跨零。没有建立稳定的整体几何净增量。', '',
        '这不证明原始历史毫无用途：传感器新增一致支持在mit_32_g725约6.75%、brown_cogsci_1约1.97%，有明确的局部潜力。所有地点仍计入整体，不把这些地点重新挑成确认集。', '',
        '![原始信息支持](figures/raw_source_information.png)', '',
        '## 进入粗记忆以后保留了什么', '',
        '复用四组已存状态与原project_memory接口，保持3σ、0.1 patch²模糊、coverage>0.1，并把patch覆盖/合成深度展开到对应像素。状态覆盖不能充当真实表面或最终预测。', '',
        '| 状态 | 有效查询像素覆盖 | 一致深度支持 | 条件AbsRel |', '|---|---:|---:|---:|']
    for p in POLICIES:
        x = item(data,'state_rows',p,'supported_absrel')
        lines.append(f'| {POLS[p]} | {pct(item(data,"state_rows",p,"valid_support_fraction")["mean"])} | {pct(item(data,"state_rows",p,"consistent_fraction")["mean"])} | {x["mean"]:.4f}（{x["places"]}/12地点有支持） |')
    lines += ['', 'History-Corrective与History-Prefix在新增且一致的原始预测支持位置上，平均覆盖83.43%/90.96%，但合成深度也一致的比例为40.82%/26.15%。**这两个条件比例只有4个地点有分母**，不能作为12地点整体性能。换算为全部有效查询像素，保留的一致新增支持仅为0.102%/0.075%，原始预测来源新增为0.249%。', '',
        'Recent-Prefix在这些位置也能有很宽的高斯覆盖，说明“覆盖到某位置”不是“保存了该来源的信息”的证明。状态足迹、置信筛选、区域化、合并、更新与原始稠密点不同，本轮不把其差异唯一归因于槽数量，也不称传感器投影是PE上界。原始来源的缺测/标定和状态表示问题可同时存在。', '',
        '![几何和状态保留](figures/geometry_and_state_retention.png)', '',
        '## 核对与计算代价', '',
        f'- 120次冻结骨干重放全部完成，192张来源深度保存；与旧缓存的区域中位深度最大差{source["patch_depth_max_absolute_difference"]}，置信筛选差异{source["total_patch_admission_mismatches"]}个patch。',
        '- 两张空闲A6000分别处理60个窗口，worker耗时20.90/21.66秒，峰值已分配4.636GiB；沿用12GiB上限，两个退出码均0。',
        f'- 全量投影与评分worker {cost["seconds"]:.2f}秒，峰值已分配{cost["peak_allocated_GiB"]:.3f}GiB；零训练/下载。共享资源计时不作独占速度基准。',
        f'- {checks["score_rows_independently_checked"]}行计数/连续误差/保留统计全部独立复算；2个固定地点、2查询、2种来源、3范围的24张图另用NumPy齐次坐标投影复算，支持像素不一致数均为0。',
        '- 原始标签、RGB、缓存、状态和源代码身份保留。旧状态平均覆盖复现最大差约2.65×10⁻⁶，未扩大为新的数值审核。完整结果及条件分母见analysis.json、independent_checks.json。', '',
        '## 分支决定与下一步', '',
        '**当前主要限制首先是任务提供的历史补充量不足，预测几何误差又进一步减少可用信息。** 尚不能依据这一实验要求增加memory容量、重写writer或否定高斯PE。六个地点连传感器来源的新增一致支持也为零；在这批窗口继续补种子或修读取公式，难以建立强证据链。', '',
        '下一项应构造一个有限的受控历史读取正对照：保留早期完整观察，遮挡当前四帧和晚期写入中的同一空间区域，使历史确实可能补充当前缺失；所有方法获得相同输入，仍比较高斯、Center、强内容和Local/Native。先验证这个任务的信息边界，再做GPU原生输出比较，既不把当前覆盖诊断当永久门禁，也不把受控任务包装成真实部署确认。', '',
        '具体方案见[history_visibility_control1](../../../docs/gaussian_memory_history_visibility_plan.md)，目前仅提出，尚未执行或设为新目标。', '',
        '这轮回答的是当前数据中历史支持的数量、来源几何误差和粗状态投影的限制；没有新增Transformer预测或训练。因此history_read1与native_anchor_probe1的整体用途/强C负结果原样保留。', '',
        '## 固定例图与反例', '',
        '固定fidelity_00/01/02、query11，另按最少传感器新增一致支持选择反例；零值并列按地点名排序，结果与fidelity_00重合，不另换地点。上行为全图，下行仅评价mask0隐藏区，源投影相同。颜色由投影完成后的查询标签判定：绿色一致、红色过近、紫色过远；浅灰无投影、白色目标无效。过近/过远表示几何差异，不单独证明真实遮挡或物体运动。']
    for rid in read('figure_selection.json')['selected']: lines += ['', f'![{rid}](figures/example_{rid}.png)']
    if (OUT/'visual_review.json').exists():
        lines += ['', f'已实际查看并审阅{len(read("visual_review.json")["images"])}幅最终PNG，详见visual_review.json。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    torch.set_num_threads(4); data = read('analysis.json')
    assert read('independent_checks.json')['complete']
    figures(data); selected = examples(data); text_report(data)
    print(json.dumps({'report_generated': True, 'examples': selected, 'pngs': len(list((OUT/'figures').glob('*.png')))}), flush=True)
