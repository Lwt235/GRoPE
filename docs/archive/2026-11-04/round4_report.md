# 第四轮：进一步校验 Gaussian PE，并检验更精炼的后续设计

2026-10-30。承接第三轮的 48 维 normalized Gaussian PE。本轮完成 **200 组训练**：
两种真实视觉输入各 13 方法 × 5 种子，共 130 组；新程序化场景中 7 方法 × 5 种子 ×
2 种训练条件，共 70 组。另完成冻结权重干预、基元协方差归因、数学及数值审计。

**当前判断：保留原 Gaussian PE 作为干净几何的主要方案；新增缓降频谱作为有理论和数据支持的鲁棒变体，尚不能全面替代原式。**
可以进一步简化固定 query 支撑的实现，但不能把“差异不显著”表述成已经证明移除 query 协方差无损。
研究主张应聚焦“高斯支撑参数化 → 位置相位与频率配比”，尚不应宣称单个 3DGS 基元的全部旋转、尺度属性都贡献了收益。

## 1. 现有方案的关键解释经得住补充对照

原结构是

\[
\Sigma=R\operatorname{diag}(s^2)R^\top,\quad
(m,C)=\text{投影后的支撑均值、协方差},
\]
\[
u_f=e^{-\omega_f^\top(C+\delta^2I)\omega_f/2}
\operatorname{Rot}(\omega_f^\top m)c_f,\qquad
\operatorname{PE}(c)=\|c\|\,u/\|u\|.
\]

均值决定相位，协方差决定各频率的相对权重，最终归一化控制评分幅度。它保持单 token 范数，
但对内容是非线性的，不是正交映射，也不保持所有内容对的内积。

第三轮 fixed/trace/area 主要干预 key。本轮补齐双侧对照、query-only/key-only 因子实验，
并重新训练，而非仅对原权重替换输入。以下为 figurines+temple、原 GS 分区的 CE，越低越好；
两视觉分支共享 1,917 个有效 query 行、91 个有效 query 相机。

| 协方差信息 | 原始 RGB | 冻结 ResNet18 |
|---|---:|---:|
| Q/K 均使用完整支撑 | **1.198967** | **1.172637** |
| Q/K 均固定 | 1.253044 | 1.231050 |
| Q/K 均仅保留 trace | 1.226835 | 1.203838 |
| Q/K 均仅保留 area | 1.219031 | 1.205012 |
| 仅 K 使用完整支撑，Q 协方差固定 | 1.199366 | 1.178046 |
| 仅 Q 使用完整支撑，K 协方差固定 | 1.252299 | 1.229062 |

完整支撑相对双侧固定的 CE 改善为 **4.32% / 4.74%**。配对训练种子/相机簇 bootstrap：

| 完整支撑减去对照 | RGB 差值及 95% CI | 冻结特征差值及 95% CI |
|---|---|---|
| 双侧固定 | −0.05408 [−0.08521, −0.02678] | −0.05841 [−0.09219, −0.03124] |
| 双侧 trace | −0.02787 [−0.03751, −0.01811] | −0.03120 [−0.04311, −0.02128] |
| 双侧 area | −0.02006 [−0.03925, −0.00245] | −0.03237 [−0.05283, −0.01502] |

冻结权重的独立干预也得到同方向结果；各条件仅用验证集重新校准温度后，优势仍保留。
因此，收益不能仅归结为单侧频带失配或评分温度变化。证据支持 key 支撑中的非各向同性信息。

Query 协方差的贡献则较小：key-only 减 full 的 CI 分别为
`+0.000399 [−0.005638, +0.007487]`、`+0.005409 [−0.004545, +0.020400]`。
这些结果没有证明统计非劣效，且所有方法仍使用相同的 **拟合 query 均值**；不能说 query 几何完全不需要。

详见 [重训练对照](results/round4_spectral_real_analysis/README.md)、
[冻结机制与温度审计](results/round4_mechanism_audit/README.md)。

## 2. 新设计：将指数滤波改为尺度混合后的缓降滤波

只替换一个函数，不增加分支、门控或学习参数：

\[
v_f=\omega_f^\top(C+\delta^2I)\omega_f,\qquad
\boxed{a_f=(1+v_f/2)^{-1}},
\]
\[
\operatorname{PE}_{R}(c)=\|c\|\frac{[a_f\operatorname{Rot}(\omega_f^\top m)c_f]_f}
{\|[a_f\operatorname{Rot}(\omega_f^\top m)c_f]_f\|}.
\]

仍是 48→48，整个匹配器仍为 15,171 个学习参数。数学动机不是任意更换激活：
若 `T~Exp(1)`、`Z~N(0,C+δ²I)`，则 `X=m+sqrt(T)Z` 的特征函数为
`exp(iωᵀm)/(1+v/2)`。它保留相同均值和协方差，并缓和不同频率间的指数权重差。
这是已有的 Gaussian scale mixture / 对称多元 Laplace 分布形式，不能把该概率分布或特征函数声称为原创。
该分布及矩公式见 Agostini 等论文的 [Example 4 / Eq.12](https://link.springer.com/article/10.1007/s10208-020-09469-6)。

本轮推导并独立测试了一个直接相关的性质：固定位置和内容，令支撑变成 `sC+C₀`，
则单位范数输出满足

\[
\left\|\frac{\partial\operatorname{PE}_{R}(c)}{\partial\log s}\right\|
\le\frac12\|c\|.
\]

原指数版没有对任意内容成立的同类统一上界。这个界约束表示对整体尺度的敏感性，
不保证匹配正确、概率校准或任意几何误差鲁棒。最后的归一化也使输出不再是原始旋转期望。
当前混合的是包含 `δ²I` 的总有效支撑，不能把 δ 继续解释成严格固定的最小物理 blur。

### 实际效果：存在清晰取舍，未全面胜出

以下是真实数据、RGB、原 GS 分区的受控压力实验；只乘协方差，保持均值、图像、标签和候选 mask 不变。
包括原 Gaussian 的全部四个预定强度，不能按测试结果挑一个较弱对照。

| 输入协方差倍率 | Gaussian×0.25 | Gaussian×0.5 | 原 Gaussian | Gaussian×2 | 新缓降版 |
|---|---:|---:|---:|---:|---:|
| 1，干净 | 1.21727 | 1.20692 | 1.19897 | 1.19274 | 1.20058 |
| 4 | 1.20342 | 1.21539 | 1.25652 | 1.34675 | 1.20576 |
| 16 | 1.26707 | 1.37711 | 1.56084 | 1.80953 | 1.31814 |

缓降版相对原 Gaussian 在 ×4、×16 时的 CE 改善均有相机簇 CI 支持；冻结视觉特征也复现。
相对 Gaussian×0.5，它在这三种条件的 CE 都显著更好。
但 ×16 时，Gaussian×0.25 在 RGB 上显著更好，代价是干净几何表现更弱。
这是一个更平衡的候选，不能宣称所有压力强度都最优。

两种真实视觉分支都由 **旧验证集** 选择了 Gaussian×2，而不是缓降版；此选择没有读取本轮测试结果。
交换独立 GS 子集后，缓降版相对原 Gaussian 的干净 CE 反而显著变差：
RGB `+0.01289 [0.00649, 0.01997]`，冻结特征 `+0.02147 [0.00943, 0.03890]`。
因此，本轮预定的全面替换条件没有满足。

新程序化场景提供了另一个独立数据复核。如下为 visible R1：

| 训练条件 / 测试条件 | 原 Gaussian | 新缓降版 | Gaussian×0.25 | 验证集选定 Gaussian×2 |
|---|---:|---:|---:|---:|
| oracle / 干净 | 97.344% | 96.985% | 96.929% | 97.681% |
| oracle / 尺度错误 | 93.689% | 95.676% | 96.359% | 89.120% |
| 混合几何训练 / 干净 | 97.176% | 97.083% | 96.798% | 97.853% |
| 混合几何训练 / 尺度错误 | 93.766% | 95.713% | 96.153% | 90.255% |

缓降版对原 Gaussian 的尺度错误改善为 **+1.987 pp [1.350, 2.584]**、
**+1.947 pp [1.507, 2.405]**，但不胜过所有简单缩放对照；对 RGB 预测几何也没有稳定 R1 收益。
该程序化 `scale_noise` 是支撑轴尺度扰动，不能直接等同于理论证明中的单一整体倍率。
两训练条件的验证集也都选择了 Gaussian×2；缓降版干净 R1 显著低于这个验证集选定对照。

详见 [完整真实尺度对照](results/round4_scale_tradeoff/README.md)、
[最终点任务统计](results/round4_point_final_analysis/README.md)、
[推导及局限](docs/round4_design_review.md)。实现为 [gaussian_spectral_pe.py](models/gaussian_spectral_pe.py)。

## 3. 更精炼的结构：固定参考支撑可以吸收到内容编码器

若 query 协方差固定为 `C₀=I/12+δ²I`，其频率滤波 `D₀` 为常量。
可以把最后一层内容映射改写成 `W'=D₀W, b'=D₀b`：

\[
q'=\operatorname{RoPE}(\operatorname{normalize}(h'_q),m_q),
\qquad
k'=\operatorname{normalize}\{D(C_k-I/12)\operatorname{RoPE}(h'_k,m_k)\}.
\]

于是 query 只需普通旋转，动态形状滤波只保留在 key。相对协方差可能不正定，
它表示两个正频率滤波的比值，不再是一个概率协方差。拒绝头通过 `D₀⁻¹` 恢复原内容，
否则其 logits 会改变。固定共同滤波可以被末层吸收，也说明它本身不增加普通域内的函数类容量。

这是 **固定 query 协方差版本的重参数化**，并不是把完整双侧模型无损简化的证明。
10 个真实 checkpoint 的转换审计：最大完整 logit 误差 `4.77e−6`，最大概率误差 `1.07e−6`；
最小原始/转换后 encoder 范数为 `0.184`。单线程 CPU、batch16、48×48 匹配：
`5.412 ms → 3.754 ms`，约 **1.44×**。没有 GPU 实测结论。
极小 encoder 输出会落入 `F.normalize` 的 epsilon 分支，不能承诺全域数值等价；独立测试保留了明确反例。
Adam 优化轨迹也不因参数重写而保持一致。

实现及审计：[gaussian_relative_pe.py](models/gaussian_relative_pe.py)、
[转换结果](results/round4_relative_audit/README.md)、[完整推导](docs/round4_relative_support.md)。
当前把它保留为可选简化实现，不自动替换完整方案。

## 4. 关于“3DGS 参数化”的关键归因收缩

区域支撑总 covariance 可分为

\[
C_{\rm total}=C_{\rm centers}+\mathbb E[C_{\rm primitive}].
\]

对实际 48 维 normalized_shape 的五个冻结 checkpoint，本轮分别删除 key、query、双侧
primitive 内部 covariance，保持均值、图像、标签、有效行和温度不变。
四个场景旧相机的双侧干预中，完整模型减 center-only 的 CE 为
**+0.004614，95% CI [0.002226, 0.007322]**；交换分区也为正。
说明在当前协议下，加入基元内部 R/S covariance 总体没有帮助，甚至略有害。
Temple 有极小的反向结果，不能说每个场景都无用。

这项是冻结干预，未重新训练 center-only 模型。它否定的是“当前证据已经证明必须使用每个
3DGS 基元全部 R/S 信息”的强主张，并不否定高斯支撑参数化本身。
当前最有证据的部分是：**用有效区域支撑的均值和非各向同性形状构造 PE**。
详见 [实际 normalized 模型的基元归因](results/round4_primitive_cov_audit/README.md)。

## 5. 验证质量、适用范围与复现

- 真实数据属于第三轮已考察相机上的扩展，不能再次称独立未见真实测试。训练依旧只用 Teatime train，
  用 Teatime validation 选 checkpoint；所有四场景、两 GS 分区、两视觉输入均报告。
  主组 fig+temple 的相机簇 CI 固定这两个场景，不能外推任意场景总体；多个探索性 CI 未作同时覆盖校正。
- 程序化任务使用本轮预定的新命名空间 `12100/12200/12300 + seed`，
  五种子 `11,22,33,44,55`；训练/验证/测试 digest 均重建核对。
- 130 个真实 checkpoint 的验证重放最大误差 `4.44e−16`。
  20 个重复旧方法 checkpoint 与第三轮 state 严格相等，160 组对应新旧预测严格相等。
  真实主分析保存 1,040 份概率档案。
- 70 个点模型完成验证集和全部八条件重放后，最初 bootstrap 报告遇到整数除法输出类型错误。
  调用处已改 float64；最终从 350 份主条件概率档案独立重算 NLL、匹配/拒绝指标和置信区间，
  保留完整重放日志与当时源码。没有重训、重选 checkpoint 或改数据。最终有效结果在
  `results/round4_point_final_analysis`；最终推理重放档案在 `round4_point_verified`。
- 全部 **131 项测试通过**。旧 `models/gaussian_pe.py` 及第三轮冻结权重没有改动。
  独立 float32 安全分流审计虽达到 `3.23e−7` 最大输出误差，完整推理更慢，故不自动采用。
  [数值审计](results/round4_numerical_audit/README.md)
- 现阶段仍是已有 GS/SfM 几何下的条件匹配与程序化机制验证，不是完整 Transformer/NVS 系统。
  重建可能使用评价照片，稀疏轨迹缺失不代表不可见；压力倍率也不是实际估计误差分布。

训练入口（必须换未使用的输出目录，脚本拒绝覆盖冻结实验）：

```bash
python -m experiments.round4_spectral_probe --domain real --kind raw --out results/round4_reproduce_raw
python -m experiments.round4_spectral_probe --domain real --kind frozen --out results/round4_reproduce_frozen
python -m experiments.round4_spectral_probe --domain point --out results/round4_reproduce_point
python -m pytest tests -q
```

完整协议及源码快照分别位于 `results/round4_spectral_real_raw/frozen_protocol.json`、
`results/round4_spectral_real_frozen/frozen_protocol.json`、`results/round4_spectral_point/frozen_protocol.json`。
checkpoint 与大概率数组保留在本机并已被 gitignore 排除。本轮没有推送或提交仓库。
