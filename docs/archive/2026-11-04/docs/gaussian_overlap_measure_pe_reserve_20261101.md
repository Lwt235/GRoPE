# Gaussian overlap 测度 PE：有限理论审查，未选型

2026-11-01。**结论：支持将这个闭式测度机制保留为尚未选型的理论备用；当前描述还缺少冻结一次开发验证所必需的定义。** 它在严格保留 primitive 测度之和时具有 raw overlap 的双侧可加性，确实避开了前一份 Jeffreys 反例的具体机制。没有固定有限维 Q/K、没有预先保证全部效果门，都不是淘汰一个 closed-form 位置 bias 的充分理由。本笔记不证明该方向必然失败，也不把条件数学性质当作最终效果已通过。它是已知 expected-likelihood / probability-product kernel 的应用，不提出新核或新颖性声明；不是 R39，不改变 R38、任何原门槛或失败结论。

仅阅读既有文本、已保存的 JSON 汇总及一手文献；没有打开 NPZ、读取新标签、导入运行模型或进行任何合成/真实数值实验。下述恒等式与反例均为解析推导。

## 1. 必须先固定 alpha 的含义

在同一 d 维坐标空间和 Lebesgue 基准测度下，定义有限非负测度

\[
d\nu_i(x)=\rho_i(x)\,dx,\qquad
\rho_i(x)=m_i\,\mathcal N(x;\mu_i,\Sigma_i),\quad m_i\ge0,\ \Sigma_i\succ0.
\]

本笔记将公式中乘在**归一化** Gaussian 前的系数明确记为总质量 `mᵢ=νᵢ(ℝᵈ)`，不把它默认为 3DGS 的 peak opacity。若直接置 `mᵢ=αᵢ_GS`，这是把原 opacity 数值重新用作质量的建模决定；`mᵢ N` 的中心高度随后依赖 determinant，它并不保留原高斯的峰值。

若想保留未归一化 Gaussian 场的峰值 `aᵢ`，则

\[
\rho_i(x)=a_i\exp[-\tfrac12(x-\mu_i)^T\Sigma_i^{-1}(x-\mu_i)],\qquad
m_i=a_i(2\pi)^{d/2}|\Sigma_i|^{1/2}.
\]

两种约定会给不同尺度的 primitive 不同质量，不能交替使用或按任务挑选。这里没有选择哪一种作为模型。原 3DGS 的 learned opacity 乘在投影到屏幕的 Gaussian 上，再进入有顺序的 alpha blending；它没有提供一个唯一的、坐标无关的三维守恒质量转换。[Kerbl 等原论文，式 3 与第 4 节](https://arxiv.org/html/2308.04079v1) 支持这一渲染定义，**不支持**把上面的 `mᵢ` 解释为已校准的可见面积、置信度或光学厚度。

## 2. 闭式 overlap，以及真正成立的加法

两个 primitive 的 raw overlap 是

\[
K_{ij}=\int\rho_i(x)\rho_j(x)\,dx
=m_i m_j\mathcal N(\mu_i-\mu_j;0,\Sigma_i+\Sigma_j).
\]

正质量时，令 `Sᵢⱼ=Σᵢ+Σⱼ`、`δᵢⱼ=μᵢ−μⱼ`，得到

\[
\ell_{ij}=\log m_i+\log m_j-\tfrac d2\log(2\pi)
-\tfrac12\log|S_{ij}|-\tfrac12\delta_{ij}^TS_{ij}^{-1}\delta_{ij}.
\]

归一化 Gaussian 的这一 overlap 及其 log 已明确出现在 [Vilnis 与 McCallum，ICLR 2015，第 5.1 节](https://arxiv.org/html/1412.6623v4)；更一般的密度乘积核见 [Jebara、Kondor 与 Howard，JMLR 2004](https://www.jmlr.org/papers/v5/jebara04a.html)。这里引入质量与重新分组的讨论，是对既有公式的应用推导；文献中的词嵌入或其他任务结果不能作为本项目效果证据。

让区域 A、B 是**固定 primitive 索引的集合**，保留完整的和：

\[
\rho_A=\sum_{a\in A}\rho_a,\qquad \sigma_B=\sum_{b\in B}\sigma_b,
\qquad K(A,B)=\int\rho_A\sigma_B=\sum_{a\in A}\sum_{b\in B}K(a,b).
\]

如果 A 和 B 分别被不重不漏地划为 `Aᵣ`、`Bₛ`，则严格成立

\[
K(A,B)=\sum_{r,s}K(A_r,B_s).
\]

这里既有 query 可加性，也有 key 可加性。虽然各 Gaussian 在空间上互相重叠，只要索引分组不重不漏，线性恒等式仍然成立；空间重叠并不等于它们组成了不重叠的物理面积分区。

这确实避开 [Jeffreys 备用方向的反例](/home/liwt/GRoPE/docs/compact_gaussian_dual_pe_reserve_20261101.md)：那里的父区间和子区间分别被重新矩拟合，且分数不是两个保留测度的双线性配对。此处换了定义，不能沿用那个反例直接否定 raw 可加性。

成立范围同样严格：

- 合并后若只保留一个 moment-matched Gaussian，通常 `ρ_A` 已经改变，加法证明失效。若每个新网格重新计算 Gaussian primitives，也没有共享的固定测度可用来证明相同分区关系。
- 真正的空间分割是 `ρ_B=1_Bρ`。它可加，但截断 Gaussian 一般不再是完整 Gaussian，上述单对闭式不能直接使用。
- 把一个 primitive 拆成参数相同、质量相加的副本保持测度；普通 3DGS densification 改变中心/尺度/opacity，不能仅因渲染接近就认为测度逐点相同。
- `log K(A,B)=log Σₐᵦ exp(ℓₐᵦ)`，不是对 `ℓₐᵦ` 求和或求平均。把整个 `log K` 乘温度 λ，得到 `(ΣK)^λ`，除 λ=1 等特殊情况外不再等于 `ΣK^λ`。不能以温度重标定保留原加法主张。

## 3. 联合划分不等于最终 row-softmax 的任意合并

先不加 content。定义 `Z_A=Σ_B K(A,B)>0`、`P(B|A)=K(A,B)/Z_A`。保持 query A 不变，只细分 key B 时，分母不变，因此

\[
P(B\mid A)=\sum_sP(B_s\mid A).
\]

但是 query 合并服从的是

\[
P(B\mid A)=\sum_r\frac{Z_{A_r}}{\sum_tZ_{A_t}}P(B\mid A_r).
\]

它按子 query 的 **overlap 行总量 Z** 加权，一般不是按源像素数、源面积或 `m_Aᵣ` 加权。因为

\[
Z_A=\int\rho_A(x)\,\sigma_{\rm all}(x)\,dx,\qquad
\sigma_{\rm all}=\sum_B\sigma_B,
\]

而 Gaussian key 混合通常不满足 `σ_all(x)=常数`。有限 FOV、不同覆盖密度、不同 key 尺度均会使 `Z_A/m_A` 变化。只有额外成立的 partition-of-unity 等条件才可能令两种 query 聚合相同，不能默认获得。先把 query 密度除以自身质量也不会消除这一问题；该行因子在 softmax 中约去后，合并关系仍含各行的覆盖归一化项。

若对全部 A、B 用一个全局总量归一化，得到联合分布 `π(A,B)`，它的双侧粗化确实与求和交换。但这不是当前按 query 给出一个条件分布并按既定人口评价的算子。不得改成联合概率指标来宣称原 query/key 分区门通过。

加上 content 后，实际未归一化权重是

\[
W_{AB}=K(A,B)e^{b_{AB}}.
\]

子 key 共享相同 content bias 时，其几何求和仍可保留。一般重新池化或重新归一化的 Q/K 不满足

\[
K(A,B)e^{b_{AB}}=\sum_{r,s}K(A_r,B_s)e^{b_{A_rB_s}}.
\]

若定义右侧为父权重、始终保留每个 primitive 的 content，则可保留精确求和；代价是父 token 不能只用一个普通 pooled Q/K 代替这些相互作用。由此反推的父 bias 是依赖对方 token 的 `log(weighted sum / K)`，不是独立 token 描述子的无损合并。

V 同样不能被省略：`Σ_b W_Ab v_b` 对保留的子项可加，但一般不等于 `W_AB v_B`，除非子 V 相同或父 V 使用依赖 query A 的权重。保留整个带 V 的密度场可以维持积分加法，却没有证明能压成一个固定的、query 无关的 V。网络中的 LayerNorm、FFN、残差也没有由 raw kernel 的加法自动获得分区等价。

零质量 key 对应 `log K=−∞`；全零 query 行的 softmax 没有定义。任何 fallback 必须事先规定。每 token 加相同 ε 会随 token 数改变总质量；若需要 floor，应明确其可加的基础测度与单位，不能隐藏为数值修复。

## 4. 保留区域形状不解决 native center 语义

`K(A,B)` 描述两个密度场的重合程度，没有记录“源 patch 的哪条射线是中心”。一个明确反例是 query 测度含等质量的 `N(−L,σ²)` 与 `N(L,σ²)`，指定的区域中心为 0；令三个等质量、同 covariance 的 key 中心为 `−L,0,L`。记 `S=σ²+D`。只要 `L²>2S log 2`，侧边 key 中自身对应项 `g_S(0)/2` 已超过中心 key 的 `g_S(L)`，故某个侧边获得更高 raw 分数。这是解析反例，不是新生成标签的实验。

因此保持 mixture 高阶形状可以保住面积分布，却不会普遍使原生第一 argmax 指向指定中心。用另一中心读出、硬指定 winner 或按任务选择不同分布都会改变原合同。这里没有提出这些替代。

[R13](/home/liwt/GRoPE/docs/round13_target_conflict_report.md) 与 [R21](/home/liwt/GRoPE/docs/round21_matched_decomposition_report.md) 已有这种区域/中心权衡的实测证据，不能把“取消 moment fit”直接当作其修复。[R38 保存结果诊断汇总](/home/liwt/GRoPE/results/round38_saved_output_diagnostic/summary.json) 中，1,506 个可比预测 anchor 有 387 个在 raw 阶段就不被最大格包含，content 移出 11 个、移回 18 个；它不支持把当前主要问题简单归因于 content 温度。这些是已保存的描述性诊断，不是本笔记新增实验，也不是传感器中心正确率。

## 5. 坐标、退化 covariance 与真实渲染边界

设共同可逆仿射变换 `y=Tx+t`、`J=|det T|`。若 `m_i` 是守恒总质量，按测度 pushforward 有 `ρ'_i(y)=ρ_i(x)/J`，因此

\[
K'_{ij}=K_{ij}/J,\qquad \ell'_{ij}=\ell_{ij}-\log J.
\]

**raw L2 overlap 不是仿射不变量**；同一坐标变换产生的公共因子会在 row-softmax 中约去，所以在 content、候选集合与其他项也一致变换时，概率不变。单位变化也必须变换全部中心、covariance 和任何有量纲的正则项；不能保持固定数值 `h²I` 却声称任意仿射不变。

若选择“峰值 a 固定”的标量场语义，上节的质量换算会给 `m'_i=Jm_i`，此时 raw overlap 反而变为 `JK_ij`。共同因子仍可约去，但这与守恒测度采用了不同的质量单位约定。两种情况下的 raw 不变量主张都不能混用。R6 的“query 概率密度 × key 面积指示函数近似”又是不同的两侧单位约定。

透视投影是非线性的，不是上述共同仿射变换。一般非线性变换的 Jacobian 随位置变化，两个 pushforward 密度的乘积会出现位置相关的逆 Jacobian，不能约成行常数；严格三维 Gaussian 的透视 pushforward 也不保持 Gaussian。逐 primitive 的 Jacobian 近似、投影后矩拟合与原三维 overlap 都不是同一精确模型。

L2 密度表述要求每个 primitive 有有限平方积分。严格奇异 covariance 的 Gaussian 是低维测度，不是该空间中的普通 L2 密度。某些 singular pair 的 covariance 和仍正定，可通过差分测度密度得到有限的公式延拓；但对自身配对及所有 pairs 没有统一有限保证。例如同一点的两个趋零方差 Gaussian overlap 发散，不能用一个有限 self-kernel 概括。

共同卷积正定 Gaussian `H` 可使每个密度平滑，pair covariance 相应成为 `Σ_i+Σ_j+2H`，且**保留固定卷积下的求和线性性**。这是一个有物理尺度的新正则模型，需要随坐标变换 H，不能称为无影响的 jitter。此处没有选择 H，也不使用它掩盖退化域。

另一个合法但不同的定义是固定共同的正定 base-RBF metric `Γ≻0`，对包括 singular Gaussian/Dirac 在内的概率测度定义核均值内积：

\[
k_\Gamma(x,y)=\exp[-\tfrac12(x-y)^T\Gamma^{-1}(x-y)],\qquad
K^\Gamma_{ij}=m_i m_j\,\mathbb E_{X\sim P_i,Y\sim P_j}k_\Gamma(X,Y).
\]

令 `Σ_i,Σ_j⪰0`，其闭式为

\[
K^\Gamma_{ij}=m_i m_j\frac{|\Gamma|^{1/2}}{|\Gamma+\Sigma_i+\Sigma_j|^{1/2}}
\exp[-\tfrac12\delta_{ij}^T(\Gamma+\Sigma_i+\Sigma_j)^{-1}\delta_{ij}].
\]

该定义对 Dirac 有限，`0<K^Γ_ij≤m_i m_j` 对正质量成立；底层 RBF 是 PSD，核均值内积也 PSD，而且对保留的 primitive 测度仍双线性。概率测度的核均值与 Gaussian-RBF 闭式已有先例：[Muandet 等，NeurIPS 2012，第 2/3 节与 Table 1](https://papers.neurips.cc/paper_files/paper/2012/file/9bf31c7ff062936a96d3c8bd1f8f2ff3-Paper.pdf)。任意正定 Γ 的上述写法也可由白化坐标直接推导，不是新的核。

`K^Γ` 等于两侧分别卷积 `N(0,Γ/2)` 后的 L2 overlap 乘公共因子 `(2π)^(d/2)|Γ|^(1/2)`。所以相同固定 Γ 下，两种**明确的平滑模型**有相同 row-softmax；它们都不是未平滑 L2 模型的无影响数值修补。选择 Γ 是选择基础位置尺度/metric，必须在验证前给定规则与同信息控制。

守恒质量下，共同仿射变换若同时使 `Γ'=TΓTᵀ`，则 determinant 比值与二次型均不变，`K^Γ` 本身也不变；若世界单位缩放而 Γ 数值不变，则该声明不成立。这解决了裸 L2 的退化定义域问题，**不解决**第三节的条件概率/content/V 合并问题、指定中心语义、primitive 数量成本或第六节的有限通道边界。因此 PD 域不能被当作全部 Gaussian 参数化 PE 的硬否定；裸 L2 还是固定 Γ 的核均值模型，应属于最小定义中必须明确的一个选择，本笔记未选其中任一个。

真实 3DGS 按深度顺序使用 `c_i α_i ∏_{j<i}(1−α_j)`。其遮挡、顺序和透射率依赖整条射线；本 overlap 是对称、顺序无关的密度乘积。即使两个同位 splat 的 opacity 相加等于原值，它们的累计 opacity 为 `a_1+a_2−a_1a_2`，一般也不等于原先的线性和。故密度 clone/mass splitting 不等于渲染等价，opacity 不能直接当作独立可见权重。引用原论文只是核对这种区别，不把这个已知 kernel 包装成 alpha-blending 的替代。

## 6. 闭式成本不等于少量通道的标准 attention

单对 Gaussian 可用 `S` 的 Cholesky、solve 与 log determinant 计算，d=2/3 时每对成本小，但任意 full covariance 的开销仍是每 pair `O(d³)`，一般需要 `N_q N_k` 个 pair。区域 A、B 保留 `n_A,n_B` 个不同 primitives 时，精确 overlap 包含 `n_A n_B` 项；合并索引不自动降低 primitive 总配对成本。保存所有组合是二次内存；分块可以降低峰值内存，不能消除计算。

`K` 是 L2 内积核，故同一组密度的 Gram 矩阵 PSD。但这不意味着 `log K` 是 PSD，也不意味着它能精确化成固定数量的普通 Q/K 通道。密度本身是 Hilbert 空间中的函数表示；把它叫 PE 不能替代有限实现与成本证明。

这一点限制的是“统一小通道、标准 Q/K 点积”的实现主张，**不排除紧凑的成对闭式位置 bias**。单 Gaussian 对在二维/三维只有一个小矩阵的代数运算，结构上可能足够精炼，并且无需 R38 的屏幕数值积分；是否达到原成本门仍需未来按同口径测量。闭式表达不能用来宣称成本已通过，无固定维 Q/K 也不能反向用来宣称成本必然失败。

这里有一个明确的有限通道边界。仅考虑一维、零均值、单位质量 Gaussian，query 方差 `u>0`、key 方差 `v>0`，则

\[
\log K(u,v)=\text{constant}-\tfrac12\log(u+v).
\]

对任意固定 `v_0`，消去 query 行常数后，key 函数族含

\[
f_j(u)=\log[(u+v_j)/(u+v_0)].
\]

若不同正数 `v_j` 对应的这些函数有非平凡线性关系，求导后得到 `Σ_j a_j[(u+v_j)^{-1}−(u+v_0)^{-1}]=0`。该有理函数在每个不同的 `−v_j` 极点的留数要求 `a_j=0`，故任意有限多个均线性无关。这证明该**特定 log-overlap 的连续全参数域**没有统一固定有限维、逐 token 可分离的精确 Q/K 点积表示，即使允许删除 query 行常数。它不是所有 Gaussian PE 不可能有限维的证明；固定有限 bank 可用依赖 bank 的高维分解，也不满足所声称的统一小通道数。

有限空间/Fourier 特征可以近似 raw overlap，Gaussian 特征函数使频率积分可写成解析衰减；但有限频率、积分域、精度与维数必须单独验证。将近似 raw 内积再取 log 仍不是一次普通 QK 点积；直接把 raw 内积当 logit 则把权重从 `K` 改为 `exp(K)`，连原来的加法语义也改变。对这些不同算子不能互借结论。

## 7. 与已有方法及实际任务的关系

| 已有路线 | 关系与不能继承的结论 |
| --- | --- |
| R5 高斯期望 RoPE 后归一化 | 内容旋转的特征函数衰减与 `log overlap` 不是同一个算子，有限频率下也不等价。两者都涉及 Gaussian Fourier 量，不构成新颖性或效果证据；R5 实际 V 路径的负结果仍保留。 |
| R6 同 covariance key 的五维式 | 令单 query Gaussian、所有 key 共有 `D` 和质量，`log m_q`、`log m_k` 与 `log|Σ_q+D|` 都是行常数，最终恰好是旧 Mahalanobis 二次位置项；若沿用旧 pair blur，也只是原 `gaussian_measure_log_bias` 在 strength=1 的形式。二维五维展开已经实现。此特例不是新的候选。 |
| R10/R11/R12/R21 区域多分量 | 保留 primitive 和在 raw 层是混合密度，而非一组新均值/协方差。它可能与旧 mixture-to-cell 积分不同，但若仍用原 primitives 与共同 key pixel footprints，差别必须写清；更多分量不是自动摆脱已见中心/区域权衡的机制。 |
| R8 实际 V 深度任务 | 真对应配原预测源值也未过既定收益门，改位置核不能宣称补充了源值中不存在的信息。仍需原同容量单视图、强融合对照与全部查询人口。 |
| R9 实际 GS→图像 RGB 读取 | 若单 GS query 经原 Jacobian 投影、key 为共同 covariance 像素，query opacity/mass 在行归一化中约去，本 overlap 回到同信息椭圆 Gaussian 读取。原 full 相对强 point 仅改善 0.146%、两场景异向的失败不能通过“双侧测度”命名重置。 |

表中依据为 [R5 报告](/home/liwt/GRoPE/round5_report.md)、[R6 报告](/home/liwt/GRoPE/round6_report.md)、[R6 原理论说明](/home/liwt/GRoPE/docs/round6_literature_design.md)、[R8 报告](/home/liwt/GRoPE/round8_report.md)、[R9 报告](/home/liwt/GRoPE/round9_report.md) 与 [R21 报告](/home/liwt/GRoPE/docs/round21_matched_decomposition_report.md)。本笔记只读这些文本及原实现源码，未重放模型。

真正使用双方三维 Gaussian 时，key 的深度/形状来源可能比旧规则图像 key 多了信息。强 point、固定支撑、同迹及原 quad16 控制必须获得同样的中心、质量、相机、key 深度、可见性与内容；同一预算下保留面积/opacity 归因，不能把新增信息或改变候选集合的收益归给 covariance。若增加新的 GS→GS 对应任务，它只能是附加验证，不能替换原 GS→RGB 与实际 V 门。若采用保留所有 primitives 的混合表示，也必须与同 primitives 的既有精确或强近似测度控制比较，报告全部成本。

源码层面，原 [R10 输入构造](/home/liwt/GRoPE/experiments/round10_partition_data.py:187) 已保存 `k_K` 与 `k_predicted_depth`，所以构造双侧三维几何有潜在数据来源。这不等于旧规则 key 基线已经消费了这些字段；未来使用它们属于额外信息路径，仍须提供同信息对照。本次只核对源码字段，没有读取其数组或证明其当前数值适用性。

## 8. 真正的限制，以及进入小验证前所缺的最小定义

唯一得到本次理论支持的正命题是：**对固定 primitive 测度，区域始终保留其完整线性和时，raw Gaussian overlap 对两个索引集合的重分组严格双线性可加。** 它不依赖新标签或结果，且明确避开了“每个区域重新 moment-fit 后打分”的 Jeffreys 反例。

需要区分三种不同性质的限制：

1. **定义域限制。** 严格 L2 式要求正定 covariance 与可归一化的正行质量。它可以由一个事前明确的输入域或平滑规则解决；固定共同 Γ 的核均值定义还可直接容纳 PSD covariance 与 Dirac。后者是一个明确不同的合法模型选择，并非裸 L2 加 jitter 后不变；不得隐藏这项选择、floor 或未定义行。
2. **不能宣称的精确性质。** raw 双侧加法不自动延伸到按源质量合并的条件概率、pooled content/V 或重新构造的网格。若原门是这些操作下的有限效果容忍要求，理论上不精确不等于该门必然失败；应保留原聚合与人口验证，而不能改成 Z 权重来强行获得恒等式。
3. **已经确定的旧方法等价边界。** 单 GS query 到共同 covariance 的像素 key，在相同投影、blur、content 与质量规则下就是原 R6/R9 算子；query opacity 在行 softmax 中消去。这一具体复现不能仅换名称就声称提供了一个克服旧失败的新机制。若未来实际 GS 适配确有不同输入或算子，需要在执行前明确，而且给所有强控制相同信息。

进入一次小规模验证前，最少必须写清以下五项；这里列出缺口，不选择其中分支：

- **token 和固定 primitive 宇宙。** image patch 与实际 GS 各自包含哪些 primitives、均值对应源中心还是区域均值；不同 native 网格是对同一 primitive 集合分组，还是重新生成 primitives。如何让两类任务遵循同一个位置机制，应明确而不按结果换定义。
- **质量与坐标域。** 选择总质量还是 peak-opacity 转换，以及裸 L2 还是固定共同 Γ 的核均值定义；若用 Γ，需要给定不依赖新结果的规则、单位及世界尺度变换方式。声明二维 target frame 或三维 world frame、投影规则、退化域和零行处理；全部控制得到相同质量、位置、相机及深度信息。不能根据两侧尺度或任务结果更换 alpha 语义。
- **唯一的最终输出。** 明确 `log overlap + content → 一次 softmax → 原生 argmax/P@V`；content 是父 token pooled 值还是保留的 primitive 内容，V 怎样聚合。若保留所有 primitive 内容/V，须把这部分存储和成对运算计入，而不是写成一个未实现的无损小 token。
- **原分区与中心合同。** 直接沿用原 query 权重、key 分区、native center、有效人口与门槛；说明哪些量理论上可加、哪些只能经验检验。指定中心信息若没有进入表示，只能保留这一风险，不能另加隐藏读出。
- **一个可区别于旧失败的可检验假设。** 在区域任务上指出它与旧单 Gaussian/mixture/quad16 的具体差别，在实际 GS 路径上说明是否仍落入 R9 等价特例。成本比较仍包括同 primitives/同信息强控制，不能只与 R38 的慢积分相比。

这些定义完成后，一次原范围内的开发验证是否值得运行，可以按其明确假设和预算判断；本审查没有得出“所有这类定义都不值得验证”的结论，也没有替代后续选型。当前明确支持的是继续保留并澄清这一闭式测度备用方向，暂不将一个未补全的描述登记为 R39 或执行实验。用户原来的全部验证目标保持不变。
