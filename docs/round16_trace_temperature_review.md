# Round16 后的设计审查：单 Gaussian 与 trace 内容强度

2026-10-30。本记录写于 R16 完整诊断及独立审计通过之后，是开发结果驱动的
设计建议，不是预注册验证。只读取已获准的 R14、R15 Stage B、R16 旧 TRAIN
汇总及相关冻结源码；没有运行新真实预测、读取 VAL/test、修改冻结算子或选择
新参数。文中的解析 toy 仅用于检验数学边界。

**建议优先研究单 Gaussian full 的一个固定内容系数
`τ=T h²/(h²+tr C)`，暂不推进固定 a=.25 的二次—四次密度。**
这是一条有机制根据、实现简洁、仍可能失败的下一候选；不能声称已经解决
区域质量与中心定位，不能继承任何旧 gate 的通过。

## 证据支持什么

R16 复用了已冻结 Stage A 概率，对 48 个旧 TRAIN pair、5,184 个概率矩阵、
1,016,064 个 query/config 状态完成独立数学审计；没有新增预测。
以下均为先场景内累计、再等场景的中心有效人口结果。

| quartic full | q | T0 top geometry gap | T0 PCK | T5 PCK | T5−T0 | T5 flip |
|---|---|---:|---:|---:|---:|---:|
| h=.25 | g8 | .54686 | 92.68361% | 93.39765% | +.71404pp | 14.083% |
| h=.25 | g16 | .16455 | 91.89359% | 92.17539% | +.28180pp | 25.637% |
| h=.25 | g24 | .04647 | 91.93553% | 85.94499% | −5.99054pp | 42.292% |
| h=.5 | g8 | .22786 | 92.65430% | 92.76784% | +.11354pp | 23.005% |
| h=.5 | g16 | .08652 | 92.10616% | 90.80722% | −1.29894pp | 33.289% |
| h=.5 | g24 | .03127 | 92.01845% | 84.35492% | −7.66353pp | 45.968% |

同 h=.25 的 quartic point 三尺度几何间隔约 1.05，T5 对 g24 仍净改善
.18976pp；Gaussian point 的相应改善为 .29659pp。h=.5 的 quartic full
在 g24/T0 甚至比共享 μ 的最近格高 1.01212pp。这支持研究区域宽度与内容
作用之间的比例，也反对把所有错误都归因于 μ 不准，或直接强制读最近格。
它不证明低 margin 只由四次点势的零 Hessian 造成；原生 cell 积分、支撑
宽度、各向异性和格相位都参与了该 margin。

跨形状推断必须克制：R14 在同 h=.5 的 Gaussian full1 上，g24 的 T5→T0
只改善 **1.3051pp**，同时 direct KL 从 .35719209 降到 .34072754。
不能把 quartic 的 7.66353pp 当作 Gaussian 新方案的预期收益。R15 Stage B
的 Gaussian full1 区域最优 h=.25/T0 得到 J=.351383，而可行 quartic 主选
h1/T0 为 .638013；说明恢复 Gaussian 有现实基线价值，并不证明 trace
系数优于该 T0 基线。完整旧控制及历史失败仍必须保留。

证据：[R16 汇总](../results/round16_mechanism/summary.json)、
[独立审计](../results/round16_mechanism_independent_audit/verification.json)、
[R14 报告](round14_temperature_report.md)、[R15 Stage B 报告](round15_regional_report.md)。
R16 联合 source lock SHA 为
`5f68a27d6988fe4668a8ed666d0ea6beba64c08c92d5e2b1bbc6b7d66a1c9ad0`；
summary SHA 为 `099b35b9d3afaecf6d3c38d3659bc944a02d1b63415a9c59ad6d6f6e18adfb08`。
生产 summary 的 `independent_audit_pending` 是生成时状态；后来独立
verification 已为 passed，科学通过标志仍为 false。

## 建议只保留这一条新公式

令 C 是同一预测 support 的完整 PSD 协方差，坐标与 h 都以 reference8
为单位；S=C+h²I。对 native key rectangle B_j：

\[
 m_j=\int_{B_j}\mathcal N(x;\mu,S)\,dx,
 \quad \beta(C,h)=\frac{h^2}{h^2+\operatorname{tr}C},
 \quad \tau=T\beta,
 \quad P_j=\frac{(m_j+\epsilon |B_j|)\exp(\tau c_j)}
 {\sum_k(m_k+\epsilon |B_k|)\exp(\tau c_k)}.
\]

沿用旧 native 面积 floor、FOV 归一化、内容、key 和 query population；
这里只示意面积单位正确的既有 floor，不能据此改旧 epsilon 或其面积单位。
T 是旧实现中乘在内容上的强度，不能和惯用的 softmax 分母温度混淆。
不缩放几何 logits，不截断 support，不移动 μ，不增添 Gaussian 分量、
中心 anchor 或独立 readout。若内容为 q·k，只需将 query 内容乘 β。

性质直接成立：C=0 时 β=1，回退同 h/T 的原 Gaussian point；T=0 时
完全等于原 Gaussian full1；0<β≤1；trace 光滑，且只需两个已有对角量。
不过 C=0 是数学 point 极限；真实小 patch 的 pixel footprint C 通常仍非零。
trace 及比值对旋转、平移和共同尺度变换 C→s²C、h→s h 不变。
它依赖参考欧氏度量，不是保持 h²I 不变时的任意仿射不变公式。

这不是把“空间支撑范围”自动解释为“预测不可信度”。C 可以是准确的大区域
形状而不是估计噪声；压低内容来自响应稳定性设计，不是一个已证实的 Bayesian
confidence 结论。若未来学习 C，β 也会被 C 的梯度改变；光滑不等于全网络
Lipschitz 或没有协方差操纵的训练问题。

## 连续点密度上的保证，以及 trace 为何胜过 det

对暂时忽略 cell/floor 的连续 Gaussian log-score，加线性内容
c(x)=gᵀ(x−μ)。其唯一 mode 是

\[
 d=x_* -\mu=\tau Sg.
\]

C⪰0 给出 λmax(S)=h²+λmax(C)≤h²+tr C，因此

\[
 \|d\|_2\le\tau\lambda_{\max}(S)\|g\|_2
 \le T h^2\|g\|_2.
\]

右侧正是原 point Gaussian 在相同线性内容下的位移模长。对一般可微 c，
若在全域有 ||∇c||≤G，每个内点驻点同样满足 ||d||≤Th²G；该结论不要求
全局线性。如果另有 Hessian 上界 L 且 Th²L<1，可保证点 log-score 的严格
凹性。若 c 只有有界振幅 R=sup c−inf c，把全局最大值和 μ 比较可得更弱但
无需梯度的点密度界 `||d||≤h sqrt(2TR)`。内容不受任何振幅/梯度限制时，
任意 τ>0 都能让足够大的内容峰压过任意有限的几何代价。

几种纯数学比较如下；**不建议把它们当成新效果搜索网格**。

| 内容系数 β | 连续最坏位移上界 | 结构取舍 |
|---|---|---|
| h²/λmax(S) | Th²||g|| | 最紧的方向无关标量，但重特征值处不可微 |
| h²/(h²+tr C) | Th²||g|| | 光滑、无 eig；推荐固定这个版本 |
| h²/sqrt(det S) | Th² sqrt(κ(S)) ||g|| | 各向异性可任意放大，不满足原 point 界 |
| h²/(h²+tr C/2) | 最多趋近 2Th²||g|| | 维度平均不能给同一个严格界 |

二维 trace 相对谱版本的温度比为
`(h²+λmax C)/(h²+λmax C+λmin C)`，处于 [1/2,1]；rank1 时相同，大且
各向同性的 C 时接近一半。因此 trace 确实偏保守，不能称为无代价的优化。
在 d 维最坏可比谱版本保守 d 倍。

仅作尺度直觉：理想均匀方形、reference8 边长 s=1/2/3 时 C=s²I/12。
h=.25 的 trace β 分别为 .272727/.085714/.04，h=.5 为
.6/.272727/.142857。这些是解析 toy，不是读取真实 C 后调出的阈值。
它可能过早削弱 q8 的有效内容；h→∞ 时 β→1，也不会修复很大 h 本身造成
的位置模糊。

## native 积分后，不能照搬该无余项位移界

取同形矩形 B 以 y 为中心，
`m(y)=∫_B N(y+u;μ,S)du`。条件于落入这个窗口的 u，有精确恒等式

\[
 \nabla_y\log m(y)=-S^{-1}(y-\mu+E[u\mid y]),
 \quad
 \nabla_y^2\log m(y)=-S^{-1}+S^{-1}\operatorname{Cov}(u\mid y)S^{-1}.
\]

积分削弱中心曲率；对光滑内容的连续窗口驻点，只能直接得到
`||y−μ||≤Th²G+r_B`，其中 r_B 是矩形半对角。g8 的 r_B=√2/2
reference8，即约5.657px；这个余项与 μ→sensor-center 误差已经足以说明
该理论界不能直接保证 PCK@8。矩形大小不同、离散 argmax、非光滑内容、
FOV 边界或 floor 主导时更不能用连续驻点等式充当保证。

明确反例：C=0、h=.25、单位 key cell，沿 x 的 Gaussian 窗口质量为
`m(y)=Φ((y+.5)/h)−Φ((y−.5)/h)`。其原点负 log 曲率为

\[
 \kappa_B=\frac{2a\phi(a/h)}{h^3[2\Phi(a/h)-1]},\qquad a=.5.
\]

数值 κ_B=3.62013914，而点密度曲率为16。弱线性内容的连续 cell-mode
位移斜率是 T/κ_B，比 Th² 大 **4.41972 倍**。这是原始 Gaussian point
本身就具有的积分效应，不能靠给 full 加 trace 系数消失。

对实际有限 key 集合，应该采用始终有效的离散条件。记 g_j=log(m_j+floor)，
j0 是原生 first argmax(g)，jτ 是 first argmax(g+τc)。则

\[
 \gamma_{j_\tau}:=g_{j_0}-g_{j_\tau}
 \le\tau(c_{j_\tau}-c_{j_0})\le\tau\operatorname{osc}(c).
\]

若全部竞争 key 的几何间隔都严格大于右侧，赢家不变。相等时必须保留
first-index 规则。对于数学上单位归一化内容的点积，osc(c)≤2，因而
γ_j>2τ 的 key 不可能获胜；生产 float32 的舍入不能拿该理想界作精确断言。
γ=0 的原几何 tie 可被任意正内容差打破。

固定 g,c、精确算术下，降低 τ 不会增加相对 T0 的 flip 指示，也不会增加
所选 key 的几何 penalty γ；这是直线上包络的结构性质，不能当作实验成功。
Euclidean winner 距离、PCK 的 harm/benefit、GT KL 均不具有该单调保证。
新旧赢家可能都不是正确中心，而且内容本来带来的正确修正也会被撤销。

## 区域误差与拆并：风险仍真实存在

对任意目标分布 r 与固定原几何概率 P0，令 Pτ∝P0 exp(τc)，则

\[
 KL(r\Vert P_\tau)-KL(r\Vert P_0)
 =\log E_{P_0}e^{\tau c}-\tau E_r c,
 \quad \frac{d^2}{d\tau^2}KL(r\Vert P_\tau)=\operatorname{Var}_{P_\tau}(c)\ge0.
\]

每个 query 的目标 KL 关于 τ 凸，但其斜率未知；r=P_T 时压低 T 必然损失，
r=P0 时则改善。不同 query 的 β 不同，不能从两个已有 aggregate T0/T5
值推出新方案 aggregate 误差在二者之间。T0 必须作为强实质性控制保留。

Gaussian 的真实 native 质量 m(B) 对不交 key 区域可加，面积 floor 同样可加。
给同一空间内容场乘 exp(τc(x)) 再积分也可加。然而旧路径是先积分、再乘
该 token 的 pooled content；key 拆并后 pooled content 一般改变，完整
attention 不具有无条件可加性。不得改成几何均值 cell-energy 以省成本。

query 拆并还有另一处非闭合：
`C_parent=Σw[C_child+(μ_child−μ_parent)(μ_child−μ_parent)ᵀ]`。
因此 `β_parent≤Σw β_child`。合并 query 会系统性压低内容，这与针对粗尺度
的动机一致，却可能加重旧 common9 分区差异。一个单 Gaussian 的矩闭合
也不能复原多峰/遮挡形状。这两件事都必须用旧分区与尺度 gate 检验。

数值和成本方面，几何部分可保持已有 Gaussian rectangle 积分及24px内部
分块，不必再次引入一个新密度的 Cartesian GL24。新增 trace 代价很小，
但公平吞吐仍需测；不能用理论操作数量或旧 quartic 成本代替效率 gate。
新的非整数 τ 乘 float32 内容的顺序、tiny-FOV/floor 和梯度路径仍需明确锁定。

## 为什么暂缓二次—四次闭合

已有固定 a=.25、b=.03649395645637156 的单密度确实有非零中心曲率；
弱白化内容下 mode 位移约为 Gaussian 的两倍，比纯 quartic 更稳定。
但原坐标下仍约 `d≈2TSg`，没有消除支撑宽度对内容响应的放大。
它还带来新原生积分认证、非 Gaussian point 基线和已知均匀方形的区域
代价。R16 并没有同 h/T 的 Gaussian full 与 quartic full 直接隔离实验，
所以不能把弱曲率认定为唯一病因。详见既有
[纯合成备用推导](round15_quadratic_quartic_reserve.md)。

同时改变径向形状和内容系数会削弱归因；建议此次只改变已有 Gaussian
full1 的内容系数。若这一条不能兼顾原 gates，应保留失败而非立刻给它
叠加中间径向密度、mixture 或 anchor。

## 新颖性与下一步判断标准

Gaussian 参数映射为 PE 已有明确先例：[mip-NeRF](https://arxiv.org/abs/2103.13415)
对 Gaussian support 的 Fourier 特征取期望；
[mip-NeRF 360 补充材料](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Barron_Mip-NeRF_360_Unbounded_CVPR_2022_supplemental.pdf)
还用 off-axis 方向表达完整协方差。它们与这里 native key probability 的读出
不同，但“用均值协方差编码位置”本身不是新贡献。

[Cordonnier 等的 Appendix E](https://arxiv.org/html/1911.03584v2#A5)
已经明确把非各向同性 Gaussian 注意力写成二次相对 PE；
[Gaussian attention bias](https://arxiv.org/abs/2305.04722) 也不是新概念。
[BAM](https://arxiv.org/abs/2505.22842) 已把 positional prior 与内容的乘积
及 generalized Gaussian prior 用于注意力；不能以概率解释本身声称创新。
[LipschitzNorm](https://arxiv.org/abs/2103.04886) 表明无新增学习参数的注意力
归一化也已有工作，但它控制的全网络性质与这里局部空间响应不相同。

本次检索没有证明 trace 这一具体公式是未发表的新方法。可诚实提出的待证
贡献只在于：利用预测空间 support 的总方差，校准同一个 Gaussian PE 与
内容之间的作用比例，并在严格 native 积分、多尺度和扰动条件下验证收益。
仍需真正 optimized 3DGS/其他数据分布的验证，不能从当前预测像素代理任务
声称已经完成通用的 3DGS→PE。

若 root 决定推进，建议只预注册 trace 这一固定公式，保持旧 h/T 网格与强
控制的选参预算，不扫描 trace/λmax/det 或新增系数。最少机制控制是相同
h/T 的原 full1、T0 full1 和 C=0 point；所有三尺度同时报区域 KL、PCK、
全 query/有效中心人口与每 scene，不能只报 flip 减少。机制的可证伪判断
是：相同 full geometry 下，它能否减少粗尺度有害内容，同时保留小尺度
区域收益，并优于原 full1 T0/固定 T 的联合折中。若只回到 T0 或只能通过
削弱强控制得到优势，就没有体现该新系数的研究价值。

这只是推进一条简洁候选的建议；不授权新真实实验，不改变旧科学门槛，
也不承诺该公式能通过它们。
