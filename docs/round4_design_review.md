# Round 4：尺度误差与频率集中下的精炼 PE 设计审查

2026-10-30。已阅读当前 `models/gaussian_pe.py`、
`docs/normalized_pe_review.md`、`round3_report.md`。本审查没有修改模型，也没有运行
训练或匹配器评价。新增纯公式测试位于 `tests/test_round4_sensitivity.py`。

## 结论先行

本轮只优先验证一个改动：将 normalized Gaussian PE 的指数幅度
`exp(-v/2)` 换成 `1/(1+v/2)`，其余均值相位、频率、48 维 head、内容 encoder 和
最后的逐 token 归一化全部保留。它有明确的尺度敏感性动机，不需要 learned gate
或额外模块。但它改变了原始单 Gaussian 的分布假设，不能把数学先例或有界导数
当成实际效果已经提高。

第二个方向只建议保留为之后的机制对照：同 radius 的方向频率组成 shell，分别在
shell 内归一化。它能明确移除 Gaussian filter 的各向同性宽度信息，适合判断收益是
来自尺度还是方向，不建议现在与第一候选一起叠加实现。

直接把 C 减去一个 isotropic 项不是优先候选：某些情况下只是归一化本来就会消掉
的代数常数；跨 radius 时则可能强化高频/方向集中，并不自然解决问题。

## 现有证据实际要求修复什么

第三轮支持 normalized Gaussian PE 在已有重建几何的条件区域匹配中有效，且比
96 维补全简洁。其负对照也很清楚：尺度误差与混合几何误差仍伤害点对应；RGB
预测几何时，完整支撑超出同样归一化固定支撑的收益未获支持。当前结构保持内容
范数，但 shape filter 仍可把能量集中到很少的频率。

因此新实验的目标是：**保住已有 clean-support 收益，同时降低错误 covariance
尺度导致的过激相对频率重排**。不能仅凭“出错时更接近忽略几何的 baseline”宣布
高斯支撑更好；必须保留 full/fixed/trace/area/shuffle 与纯位置对照。

## 候选 A：resolvent 幅度的 normalized spectral PE

第 f 个二维内容块记为 c_f，令

\[
\widetilde C=C+\delta^2I,\quad
v_f=\omega_f^T\widetilde C\omega_f,\quad
\theta_f=\omega_f^Tm,
\]

\[
a_f=(1+v_f/2)^{-1},\qquad
E^R_{m,C}(c)_f=a_fR(\theta_f)c_f,
\]

\[
T^R_{m,C}(c)=\|c\|\frac{E^R_{m,C}(c)}{\|E^R_{m,C}(c)\|},\qquad T^R(0)=0.
\]

和现有方案一样，它是内容相关的非线性映射，只有单 token 范数保持；不是正交或
unitary map，也不保持相同几何下任意 q/k 的内积。对非零有限内容、有限 PSD
covariance，幅度正且分母在实数数学中非零。逐 token 编码后仍做普通 Q/K dot。

### 为什么不是任意换一个激活

小 v 时

\[
e^{-v/2}=1-v/2+v^2/8+O(v^3),\qquad
(1+v/2)^{-1}=1-v/2+v^2/4+O(v^3).
\]

一阶 mean/covariance 行为一致；大 v 时一个按指数衰减，一个按 `2/v` 衰减。
重点不是原始幅度更大，因为最终会归一化，而是不同频率之间的幅度比不再指数放大。

共同平移 m→m+b 仍只左乘共同的 block rotation，Q/K 分数不变。C 的轴符号/
排列 gauge 仍由 covariance 形式消除。固定有限斜频率不产生精确 SO(2) 坐标旋转
不变性。零总 covariance 回到 ordinary RoPE；固定 δ>0 时的零观测 C 则仍有
regularizer filter，不能称严格 point limit。

### 分布解释：同均值/同 covariance 的已知尺度混合

取 `T~Exp(1)` 与 `Z~N(0,C̃)` 独立，定义

\[
X=m+\sqrt{T}Z.
\]

则直接积分给出

\[
\mathbb E e^{i\omega^TX}
=e^{i\omega^Tm}\int_0^\infty e^{-t}e^{-tv/2}\,dt
=\frac{e^{i\omega^Tm}}{1+v/2}.
\]

并且 `E[X]=m`、`Cov[X]=C̃`。这是对称多元 Laplace 的一种经典 Gaussian-scale
mixture，而不是原来那一个 Gaussian；公式比仅用“Laplace”名称更精确，因为文献中
也有其他 multivariate Laplace 家族。Agostini 等 Example 4 / Eq.12 给出了对应 MGF，
并指出其低阶矩与 Gaussian 一致。
[原论文](https://link.springer.com/article/10.1007/s10208-020-09469-6)

这个混合的尺度 T 没有依据观测学习或后验估计。它同时有小尺度质量与较重空间尾部，
例如任意一维投影的中心四阶矩由 Gaussian 的 `3σ⁴` 变成 `6σ⁴`。只能把它视为
对有效 footprint 的固定、较保守频率响应假设；不能称已经校准了定位 uncertainty。

最后的 L2 归一化进一步改变了概率解释：`T^R(c)` 不再等于上述分布下的原始旋转
期望，也不等于 `E_T[normalized GaussianPE(c;TC̃)]`。Q/K 两边各自使用幅度时，
原始配对衰减是两个 resolvent 的乘积，也不是对 q/k 共享一个 T 后的单一 resolvent。

**当前实现约定是混合总有效 support，包括 isotropic regularizer。** δ 也随 T
一起缩放，所以 T→0 时没有严格的最小物理 blur。若使用独立固定 Gaussian blur，
其特征函数应为

\[
e^{-\delta^2\|\omega\|^2/2}(1+\omega^TC\omega/2)^{-1},
\]

与当前约定不同。本轮不增添这条路线；报告应明确是 `C+δ²I` 的总支撑尺度混合。

## 关键可证性质：normalized 输出对 covariance 对数尺度的导数有界

以下是本审查对这个候选的直接推导，不作为新数学理论声称。
保持 mean、camera、content、frequency 与 regularizer 固定，令

\[
\widetilde C(s)=sC+C_0,\quad C,C_0\succeq0,
\quad v_f=\omega_f^TC\omega_f,\quad b_f=\omega_f^TC_0\omega_f,
\]

\[
a_f(s)=[1+(sv_f+b_f)/2]^{-1},\qquad
g_f=\frac{\partial\log a_f}{\partial\log s}
=-\frac{sv_f}{2+sv_f+b_f}\in[-1,0].
\]

令 `y=T^R(c)/||c||`，以及每个 block 的归一化能量 `p_f=||y_f||²`。
归一化的导数在径向分量上相消：

\[
\frac{\partial y_f}{\partial\log s}
=(g_f-\bar g)y_f,\qquad \bar g=\sum_fp_fg_f,
\]

\[
\left\|\frac{\partial T^R(c)}{\partial\log s}\right\|^2
=\|c\|^2\sum_fp_f(g_f-\bar g)^2
\le\tfrac14\|c\|^2.
\]

最后一步使用长度为 1 的区间中变量的方差上界。故导数范数最多为输入范数的 **1/2**。
对于单位内容 Q/K，若只改变某一边的 covariance 尺度，其 dot score 的对数尺度
导数绝对值也至多 1/2；若两边一起变化，可用两边界之和。learned temperature
还会等比例放大 logit 敏感性。

这个结论包括固定正 resolution floor，但不包括 mean/depth 同时漂移、错误 camera、
任意 covariance 旋转/特征值误差，也不包括随 s 重拟合 geometry 的整个数据管线。
它是局部表示敏感性结论，不是匹配 accuracy、拒绝、概率校准或 OOD 风险的保证。

### Gaussian 的明确反例

两个频率的 directional variances 为 `(0,L)`，取内容块半径
`(exp(-L/2),1)` 后作单位归一化。s=1 时，Gaussian filter 后两个 block 能量相等，
而 `g=(0,-L/2)`，所以 normalized 输出的导数范数为 `L/4`，可任意大。
两方向共同的正 isotropic floor 不改变这个结果。

新增独立测试在 L=4、20、40 时分别核对到导数 1、5、10，并验证 resolvent 的
1/2 上界和近紧实例。测试没有导入 matcher/model 实现。

## “较少集中”不等于完全没有 collapse

将非零 observed covariance 统一放大 s，并假设所有 occupied blocks 的 v_f>0，
则大 s 下 normalized resolvent 的相对能量趋于

\[
p_f\ \propto\ \|c_f\|^2/v_f^2.
\]

它不会仅因 s→∞ 就必然变成最小 v 的单一频率；Gaussian 的能量
`p_f∝||c_f||² exp(-sv_f)` 则会集中到 occupied 最小 v。然而前者仍可能因为极端
各向异性、很宽的频率半径范围、稀疏内容或 v_f=0 的方向而高度集中。

实际现有 24 个 logspace(.1,3.981) 频率，若初始各 block 能量相同、C 为极大
isotropic covariance，极限能量为 `p_f∝||ω_f||^-4`：participation
`1/sum(p²)=3.22755`，前三个频率占 **85.37%**。因此只能说它缓和指数式集中，
不能说消除了频率 collapse。

对所有 v_f>0，其 filter 条件数在统一放大时趋于 `v_max/v_min`，比 Gaussian 的
指数条件数温和，但仍可因非常薄的 covariance 变得很大。逐 content 的 Jacobian
也没有一个与全部 geometry 无关的统一有限界；有界 scale 导数不等于所有 content/
geometry 梯度都安全。稳定实现仍需检查真实 forward/loss/gradient 和稀疏 content。

## 第二方向：等 radius shells 的 Gaussian shell-wise normalization

若未来第一候选不能保留 clean-support 收益，可单独考虑：固定 4 个 log-spaced
radius 覆盖 .1–4，每个 radius 6 个二维方向，共 24 个二维内容块、仍为 48 维。
每个 shell 内分别应用原 Gaussian expected rotary，再恢复该 shell 原内容范数。

在同一 radius r 的 shell 中，`C→C+σ²I` 给所有幅度乘相同的
`exp(-σ²r²/2)`，所以在 shell 内归一化后严格消掉。因此这个构造保留各 radius
原内容能量，只改变 shell 内方向分配，明确移除了 isotropic support-width 信号。

它适合作为因果控制，不是免费的鲁棒增强：

- isotropic footprint 再大也不会关闭高频 shell，不能再解释成完整 Gaussian IPE。
- 乘法尺度误差 `C→sC` 同样放大各向异性差异，shell 内仍可能集中到单一方向。
- shell bank 本身改变了频率；必须有“相同 shell bank + 原全 head 归一化”的控制，
  才能分开 bank 更换与归一化位置的贡献。
- 对 resolvent 幅度，上述 isotropic-addition 的严格消除不成立：
  `1/(1+(v+c)/2)` 不是原幅度乘公共常数。因此不要把 Gaussian shell 的推导直接
  套在 resolvent shell 上。

当前主线程只实现候选 A，本审查不建议现在加入 shell 分支或与 A 组合。

## 为什么不优先直接减 isotropic covariance

1. `C−λ_min(C)I` 是 PSD，但丢掉绝对 isotropic 宽度，在球形/重根处还涉及最小
   eigenvalue 的非光滑性。它不是一个保持全部原支撑信息的数值修复。
2. `C−tr(C)I/d` 通常是不定矩阵。Gaussian exponent 可出现大于 1 的方向增益；
   resolvent 甚至可能碰到 denominator=0 或负幅度，不能再当合法概率特征函数。
3. 对单一 radius shell 的 Gaussian filter，减任何 isotropic 项都只是一个公共
   幅度因子，已有归一化完全消掉它，数学上没有新机制。
4. 跨不同 radii 时，这个项是 radius-dependent 相对放大，确实改变模型，但可能
   把能量推到窄方向/高频而加重饱和。必须有机制证据，不能自动叫 scale correction。

## 最少但足以否证的控制

主假设只换 amplitude，不顺带换 frequency bank、resolution、query geometry 或
content backbone。full/fixed/trace/area/shuffle 的 q/k 干预范围必须写清；若本轮
研究双侧完整支撑，两侧使用一致的预定控制。第三轮 key-only 的 fixed 只证明 key
支撑贡献，不与新的 both-side fixed 混称同一个 baseline。

建议保留以下直接对照与审计：

- 原 normalized Gaussian PE 和相同两侧 geometry 控制；pure projected RoPE；固定
  resolvent support。这样才能识别改进是重尾频率响应，还是弱化/忽略了 shape。
- 一个预先固定的更弱 Gaussian attenuation（例如 β=.25 的 `exp(-βv/2)`）作为
  机制对照；不能把“总体滤波变弱”误称 Laplace prior 的独特贡献。若要选 β，
  只用开发数据并保留全部试验，不能在新确认结果后倒推。
- covariance 因子 `s∈{.01,.1,1,10,100}` 的纯尺度扫描，固定 means/content/labels；
  std 扩大 a 对应 covariance 因子 a²，两个量不得混写。另保留单独的方向与中心
  错误条件，避免尺度界外推成全几何鲁棒。
- 输出 block-energy participation、最大 block 能量占比、有效频率半径、embedding
  对 log s 的导数、有限 loss/gradient。诊断必须先于“frequency collapse 已解决”的
  解释；不能仅以输出总 norm=1 代替。
- clean 区域分布 CE 与 Top1 target mass 必须保留，真实 raw RGB 与冻结 ResNet
  输入都报告。尺度噪声改善但 clean 支撑收益消失时，应承认实际 tradeoff。
- reject/AP/ECE 仍须独立评价，较有界的表示不意味着 dustbin 阈值自动校准。

第三轮所有已看过的真实/合成结果，在本轮都只能作为开发或重复验证；不能再次称为
第一次前瞻性确认。新模型若根据这些结果选择，再在同一 203 pairs 上赢，仍是
有价值的迭代证据，但不是独立新发现。真正的确认需要预先固定新的数据/场景或
未使用的随机实验命名空间，不能靠换输出目录制造独立性。

## 先例与贡献边界

Gaussian rotary integration 与概率支撑已有直接先例，WorldRoPE 的闭式积分是明确
背景；它不等于当前规范化滤波，也不能把仅 Q/K score 对照称为完整方法复现。
[WorldRoPE 原论文](https://arxiv.org/html/2610.16337v1)

Head 内 Q/K L2 normalization 加 learned scale 也已有明确先例。
[QK Normalization 原论文](https://arxiv.org/abs/2010.04245)

resolvent/Laplace Gaussian-scale-mixture 与其特征函数是已知概率对象，不能主张
换成 rational amplitude 本身构成新概率数学。有限检索没有证明是否已有完全相同的
3DGS-derived normalized PE 工程组合；“没搜到同名方法”不是 novelty 证明。
合理的候选贡献是一个精简、可解释、有实际误差鲁棒性证据的支撑编码设计，而不是
把已知特征函数改名。

## 本次独立核验

执行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest tests.test_round4_sensitivity -v
```

5 项全部通过：固定正 floor 下的精确导数/加权方差等式、1/2 上界的近紧实例、
Gaussian 无统一上界的反例、每侧 pair score 的尺度导数上界，以及总 support 混合
与固定 blur 卷积不等价。它们验证的是理论公式，未导入模型，不是学习效果证据。
