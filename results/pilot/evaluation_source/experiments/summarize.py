"""Generate the Chinese research record directly from saved measurements."""
import json
from pathlib import Path
import numpy as np


LABELS = {
    'appearance': '仅外观', 'center': '外观 + 中心距离核', 'reprojection': '外观 + 重投影距离核',
    'rope3d_axis': '中心 3D RoPE', 'rope3d_oblique': '中心 3D RoPE（斜向频率）',
    'expected_axis_support': '形状协方差直接用于期望 RoPE',
    'expected_oblique_support': '形状期望 RoPE（斜向频率）',
    'expected_axis_uncertainty_oracle': '位置不确定性期望 RoPE（oracle）',
    'center_normal': '中心距离核 + 法向', 'center_scale': '中心距离核 + 特征值尺度',
    'center_logcov': '中心距离核 + log 协方差',
    'center_projected_logcov': '中心距离核 + 投影协方差',
    'center_bounded_logcov': '中心距离核 + 截断形状距离',
    'center_random_control': '中心距离核 + 随机描述子对照',
    'rope3d_axis_normal': '3D RoPE + 法向', 'rope3d_axis_scale': '3D RoPE + 特征值尺度',
    'rope3d_axis_logcov': '3D RoPE + log 协方差',
    'rope3d_axis_projected_logcov': '3D RoPE + 投影协方差',
    'rope3d_axis_bounded_logcov': '3D RoPE + 截断形状距离',
    'rope3d_axis_moments': '3D RoPE + 平滑二阶矩',
    'rope3d_axis_random_control': '3D RoPE + 随机描述子对照',
    'expected_axis_uncertainty_oracle_logcov': '位置不确定性期望 RoPE（oracle）+ log 协方差',
}


def load_rows(round):
    rows = json.loads((Path('results')/round/'summary.json').read_text())
    return {(r['scenario'], r['method']): r for r in rows}


def table(rows, methods, scenarios):
    names = {'id': '共享支撑', 'depth_noise': '深度噪声', 'shape_noise': '形状噪声',
             'shape_shuffled': '形状打乱', 'independent_support': '独立支撑',
             'view_support': '视角相关支撑'}
    lines = ['| 方法 | ' + ' | '.join(names[s] for s in scenarios) + ' |',
             '|---|' + '---:|'*len(scenarios)]
    for method in methods:
        values = []
        for scenario in scenarios:
            r = rows[(scenario, method)]
            value = f'{r["r1_mean"]*100:.2f}'
            if scenario == 'id':
                value += f' ± {r["r1_seed_std"]*100:.2f}'
            values.append(value)
        lines.append('| ' + LABELS[method] + ' | ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def contrasts():
    raw = np.load('results/fresh_calibrated/per_scene.npz')
    seeds = [11,22,33,44,55]
    pairs = [('rope3d_axis_logcov','rope3d_axis'),
             ('rope3d_axis_logcov','rope3d_axis_normal'),
             ('rope3d_axis_moments','rope3d_axis'),
             ('rope3d_axis_bounded_logcov','rope3d_axis')]
    results = []
    for scenario in ['id','shape_shuffled','view_support']:
        for a,b in pairs:
            delta = np.stack([raw[f'{s}/{a}/{scenario}/r1'] - raw[f'{s}/{b}/{scenario}/r1'] for s in seeds])
            rng = np.random.default_rng(601)
            boots = []
            for _ in range(2000):
                si = rng.integers(0,5,5)
                ci = rng.integers(0,192,(5,192))
                boots.append(delta[si[:,None],ci].mean())
            lo,hi = np.quantile(boots,[.025,.975])
            results.append({'scenario':scenario,'a':a,'b':b,'delta_percentage_points':float(delta.mean()*100),
                            'paired_hierarchical_95_low_pp':float(lo*100), 'paired_hierarchical_95_high_pp':float(hi*100)})
    Path('results/fresh_calibrated/contrasts.json').write_text(json.dumps(results,indent=2)+'\n')
    return results


def main():
    pilot, revision, fresh = [load_rows(r) for r in ['pilot','revision','fresh_calibrated']]
    math = json.loads(Path('results/pilot/math.json').read_text())
    manifests = [json.loads(Path(f'results/{r}/manifest.json').read_text()) for r in ['pilot','revision']]
    run_count = sum(len(m['selection']) for m in manifests)
    peak = max(x['peak_cuda_allocated_mb'] for m in manifests for x in m['selection'])
    elapsed = sum(m['elapsed_seconds'] for m in manifests)
    comparisons = contrasts()
    ci = next(x for x in comparisons if x['scenario']=='id' and x['a']=='rope3d_axis_logcov' and x['b']=='rope3d_axis')
    primary = ['rope3d_axis','rope3d_axis_normal','rope3d_axis_scale','rope3d_axis_logcov',
               'rope3d_axis_projected_logcov','rope3d_axis_bounded_logcov','rope3d_axis_moments','rope3d_axis_random_control']
    report = f'''# 3DGS × RoPE 初步验证记录

日期：2026-10-29（Asia/Shanghai）。状态：数学检查和合成机制验证已完成；真实图像验证尚未开始。

## 结论与当前决定

形状信息在跨视角支撑一致时能够补充位置，但对形状误差、错误对应和支撑变化高度敏感。当前不应把完整高斯相似性直接当作可靠的跨视角对应约束。下一轮优先研究**表面方向、局部支撑范围与位置不确定性的区分及其可信度**。

- 原始路线 A 的直接局部坐标系相对变换暂不作为主要实现：同一协方差的等价主轴符号能改变 logits，必须先给特征定义相容的坐标变换，或改用无歧义描述。
- 路线 B 保留为几何假设，但局部一阶形变本身不能保留切向尺度。本轮没有实现完整的局部单应 Attention；投影协方差相似性只是相关机制消融。
- 新增“位置 RoPE + 独立形状通道”，并比较法向、尺度、完整协方差、平滑矩和截断形状惩罚。它们是候选探针，尚非最终算法。
- 不把任何高斯协方差都解释为位置误差；形状/支撑协方差与预测不确定性分别输入。

## 1. 实际执行了什么

在空闲物理 GPU 3（RTX A6000）运行两轮共 {run_count} 次训练，涉及 22 个不同变体，5 个种子（11、22、33、44、55），每次 400 步。两轮训练和评估合计 {elapsed:.1f} 秒，PyTorch 峰值 allocated 显存 {peak:.1f} MiB（不含 CUDA 上下文）。首次沙箱内 CUDA 不可见，经沙箱外只读检查确认是设备隔离；没有修改驱动或 GLIBC。

每个模型是单头跨注意力匹配器：16 维合成外观 → 48 维隐藏层 → 32 维 Q/K，登记参数均为 2,387。相同初始化、批次顺序、AdamW 配置和训练步数；每 50 步按验证集 Recall@1 选 checkpoint。额外形状通道只有一个有效可学习标量，随机描述子对照具有相同形式。

每个种子：512 个训练场景、96 个验证场景、192 个测试场景；每场景 32 个 token，两视角有已知对应。外观含重复纹理原型，中心含沿相机射线的独立深度误差；候选顺序随机打乱。不同 split 使用独立种子。每个条件合计 960 场景、30,720 个查询，但不同扰动条件使用同一基础场景以便配对。

**这是具有已知潜在形状的合成信息实验。** 形状并非从 RGB 学到，初始设定主动给两视角共享的物理支撑，因此形状可能充当容易匹配的身份信息。没有实际渲染、遮挡、可见性、无匹配 token、真实 3DGS 场景、多层 Transformer 或新视角合成。不能据此判断最终科研创新、真实数据性能或预测器可学习性。

## 2. 数学与数值证据

| 检查 | 实测 | 含义 |
|---|---:|---|
| 相同高斯改变主轴符号后的协方差差异 | {math['affine_gauge']['same_covariance_max_abs']:.1f} | 分布完全不变 |
| 此时原始相对仿射 logits 的绝对变化中位数 | {math['affine_gauge']['fixed_features_logit_gauge_change_median_abs']:.3f} | 固定视觉特征下存在局部坐标歧义 |
| 仿射相对矩阵的全局刚体变换误差 | {math['affine_gauge']['global_rigid_relative_max_abs']:.2e} | 全局坐标抵消不等于局部主轴歧义已解决 |
| 切向尺度在局部跨视角变换中的抵消误差 | {math['tangent']['scale_cancellation_max_relative']:.2e} | 改尺度不会改变同一平面的一阶相对形变 |
| 高斯期望相位与 20 万样本 Monte Carlo 的最大差 | {math['probability']['expected_phase_mc_max_error']:.4f} | 解析期望实现通过数值校验 |
| 同对角、不同非对角协方差，轴向频率响应差 | {math['probability']['axis_frequency_offdiagonal_difference']:.1f} | 轴向频率有信息盲区 |
| 同一反例，加入斜向频率后的响应差 | {math['probability']['oblique_frequency_offdiagonal_difference']:.4f} | 斜向频率可读取相关项，但不保证任务性能更好 |

局部平面接近侧视时，投影切向矩阵的条件数在 89.99° 达到约 5,730。透视投影线性化也有适用范围：sigma/depth=0.1 时条件协方差相对误差约 2.8%，0.2 时约 13.9%。后一检验使用正深度且 z>0.5 的截断样本，完整高斯在透视奇点附近不能假设具有良好的投影矩；原始 JSON 同时记录了拒绝比例。

另发现实现问题：`torch.linalg.eigh` 构造矩阵 log 的普通反向传播在重复特征值处出现 NaN。矩阵 log 本身在 SPD 域内是光滑函数，问题来自特征向量求导实现。本轮描述子预计算，没有通过协方差反传；新增的 trace 归一化二阶矩不经过特征分解，球形协方差处梯度有限。这还不是完整预测器的梯度验证。

## 3. 第一轮：期望 RoPE 的语义不能混用

以下均为 Recall@1（%，共享支撑列为 5 种子的均值 ± 标准差），同一批原始测试场景。

{table(pilot, ['appearance','center','reprojection','rope3d_axis','expected_axis_support','expected_axis_uncertainty_oracle','center_normal','center_scale','center_logcov'], ['id','depth_noise','shape_shuffled'])}

把物理支撑大小直接当作位置误差，会压低旋转块幅度。在此实验固定频率与网络下，这种处理劣于确定性位置 RoPE。使用注入误差的真实协方差是 oracle 诊断，不能作为现实预测器性能。WorldRoPE 原论文还建模有限 patch 范围、多深度假设与层间更新；本实验的直接三维期望算子并非它的复现，更不能用这张表宣称优于或否定 WorldRoPE。

## 4. 第二轮：在更强的相同位置基线上加入形状

{table(revision, primary, ['id','shape_noise','shape_shuffled','view_support'])}

固定 400 步下，完整协方差通道在共享支撑设定很好，形状错误时却有显著下降；简单截断距离能缓解损害。斜向频率的确定性位置 RoPE 为 {revision[('id','rope3d_oblique')]['r1_mean']*100:.2f}%，轴向为 {revision[('id','rope3d_axis')]['r1_mean']*100:.2f}%。因此“能读取更多协方差元素”不等于本任务性能提升。

这里的 `view_support` 把固定图像 footprint 通过各相机投影 Jacobian 提升到同一潜在平面；近侧视使用 ridge 限制支撑大小，仍是局部线性诊断。它改变支撑协方差而保持外观、中心、标签完全相同；不是实际分割、遮挡或渲染过程。

## 5. 验证集校准 + 全新测试场景

发现法向、尺度、log 协方差距离的数值范围不同，400 步的共同学习率可能不能充分调整标量。固定内容网络后，只在验证集选择形状权重倍率：0、0.25、0.5、1、2、4、8、16。每个方法使用同一搜索空间，不使用测试指标选倍率。然后在此前未生成的 `900000 + seed` 场景上统一评估；本表是最终的新场景结果。

{table(fresh, primary, ['id','depth_noise','shape_noise','shape_shuffled','view_support'])}

三个关键观察：

1. 完整 log 协方差相对位置 RoPE 在共享支撑上提高 {ci['delta_percentage_points']:.2f} 个百分点；配对分层 bootstrap 的 95% 区间为 [{ci['paired_hierarchical_95_low_pp']:.2f}, {ci['paired_hierarchical_95_high_pp']:.2f}]。但只用干净验证集调大权重，会明显放大分布变化下的失败。
2. 平滑二阶矩与 log 协方差在共享支撑上接近，说明没有必要为了这个探针强制采用特征分解。正常设定表现接近不代表两种表示在所有扰动下等价。
3. 视角相关支撑变化下，法向通道优于完整协方差通道，而尺度通道出现下降。这支持进一步区分稳定表面方向与视角相关支撑范围；尚未证明额外尺度优于真实的 HRoPE 局部平面编码。

bootstrap 以种子和场景为两层配对抽样，未作多重比较校正，仅供探索。前两轮用过相同测试种子并据此调整设计；最终新场景减少了直接复用问题，但所有任务仍来自同一个人为生成器，不能等价于独立真实数据确认。

## 6. 对候选方法的修订

建议保留一个可解释的分支式原型：中心/投影位置编码负责位置，法向或形状通道提供补充；位置不确定性单独控制频率衰减或权重。形状距离采用无主轴歧义表示，先比较法向、平滑二阶矩与截断协方差距离。

**尚未实现可靠性门控。** 本轮的截断只是限制惩罚上界；干净验证集把其权重调大后仍会受错误形状影响，不能称为已解决鲁棒性。置信度应来自可观测的跨视角一致性、几何预测质量或支撑重叠，而非测试时已知的注入噪声。

下一轮最有判别力的验证：

1. 用真实或渲染 RGB、标定与深度生成对应；加入遮挡、无匹配 token 和不同 patch 支撑。固定同一视觉特征与几何预测来源。
2. 强对照至少包含位置 RoPE、局部平面/HRoPE 与概率支撑/WorldRoPE 的忠实实现；先做 oracle 几何上限，再换成同一预测器输出，区分额外信息与预测误差。
3. 将法向、切向主轴方向、各向异性比和整体尺度分别消融。目前完整协方差与法向之间的差异同时含多个因素，不能全部归因为“尺度”。
4. 在包含几何错误的验证集上选择权重或门控，再用独立场景检验精度与鲁棒性的折中；暂不启动完整 LVSM 训练。

## 7. 文件与复现

- `models/geometry.py`：投影、期望 RoPE、局部仿射与协方差描述算子。
- `experiments/math_probe.py`、`stability_probe.py`：数学反例与数值记录。
- `experiments/synthetic.py`、`matching_probe.py`：数据生成和统一训练。
- `experiments/calibrate_probe.py`：验证集标量校准与全新测试。
- `results/pilot/`、`results/revision/`、`results/fresh_calibrated/`：JSON/CSV、逐场景指标、训练选择及源代码快照；前两轮另有本地 checkpoint。
- `tests/test_geometry.py`：5 项单元检查，涵盖旋转不变的距离、退化谱处的平滑梯度、候选乱序真值以及扰动对照保持不相关输入固定。
- `README.md`：运行命令和已验证环境。依赖已有环境运行，没有安装额外包。

论文定位来自 [WorldRoPE 原文](https://arxiv.org/html/2610.16337v1) 和 [HRoPE 原文](https://arxiv.org/html/2610.28284v1)。这里使用的候选名称不是论文正式算法名；本报告的数值均来自本仓库实验。
'''
    Path('research_report.md').write_text(report)
    print('Wrote research_report.md and fresh_calibrated/contrasts.json')


if __name__ == '__main__':
    main()
