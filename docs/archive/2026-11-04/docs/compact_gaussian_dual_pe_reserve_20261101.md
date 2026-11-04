# 高斯双参数位置编码：未选型的备用方向及解析拒绝

2026-11-01。本笔记不是 R39，也不修改或替代 R38。研究目标及全部原有门槛保持不变。这里考察一个可精确写成少量 Q/K 通道的数学构造，并给出足以反对其直接用作区域质量定义的解析反例；不是候选晋升、效果实验或对全部 Gaussian PE 的不可能性证明。

R3/R4 已研究 expected rotation、归一化及频谱变体；R6 的五维二次编码依赖同一 key bank 的共同 covariance。R13/R21 显示区域分布与独立中心定位存在实际冲突，R8/R9 的实际 V 与 GS 任务又未获得必要收益。因此，有限维表示本身不能解决这些失败，也不能作为启动新真实实验的充分理由。

## 一个不同的有限维构造

对正定高斯 `G=(μ,C)`，令 `Λ=C⁻¹`。`svec` 保存对称矩阵上三角，并将非对角元素乘 `√2`，使两个 `svec` 的内积等于矩阵 Frobenius 内积。定义自然参数和矩参数：

\[
\theta_G=[\Lambda\mu,-\tfrac12\operatorname{svec}(\Lambda)],\qquad
\eta_G=[\mu,\operatorname{svec}(C+\mu\mu^T)].
\]

高斯 KL 的闭式表达给出

\[
J(G_i,G_j)=\mathrm{KL}(G_i\Vert G_j)+\mathrm{KL}(G_j\Vert G_i)
=(\theta_i-\theta_j)^T(\eta_i-\eta_j).
\]

考虑位置分数 `sᵢⱼ=−J(Gᵢ,Gⱼ)/2`。构造

\[
Q_{g,i}=[\theta_i/\sqrt2,\eta_i/\sqrt2,1],\qquad
K_{g,j}=[\eta_j/\sqrt2,\theta_j/\sqrt2,-\tfrac12\theta_j^T\eta_j].
\]

则 `QᵢᵀKⱼ−θᵢᵀηᵢ/2 = sᵢⱼ`，被减去的量是 query 行常数，故 softmax 等价。若维度为 d，参数向量长 `p=d+d(d+1)/2`，几何通道共 `2p+1`：二维 11、三维 19。任意正定 key covariance 均可使用，不要求同一 bank 的 covariance 相同。面积权重可在 K 最后一维加入 `log |Bⱼ|`。

这与原 overlap 的 `Cᵢ+Cⱼ` 核、期望旋转均不同，但不具有 RoPE 的正交或保范数性质。`C=R diag(s²) Rᵀ` 对轴排列、符号与重根基选择的歧义已经消除。KL 对共同可逆仿射变换不变，因而行常数消去后的上述分数也不变；这不是对不同相机投影误差或内容误差的鲁棒保证。秩亏 covariance 没有在本构造中得到定义；任何正则化都会另改模型。

## 反例一：粗区域宽度被细 key 限制

当所有 key covariance 等于 D，展开高斯 KL，并消去与 key 坐标无关的项：

\[
s_{ij}=-\tfrac14(\mu_i-x_j)^T(C_i^{-1}+D^{-1})(\mu_i-x_j)+\text{row constant}.
\]

连续坐标上的等价 Gaussian covariance 为

\[
V_i=2(C_i^{-1}+D^{-1})^{-1}\preceq2D.
\]

因此扩大 query 支撑并不会使位置质量按同等尺度展开。一维取 `C=L²,D=1`，得到 `V=2L²/(L²+1)<2`；相对作为数学反例的 `N(0,L²)` 区域分布，其 KL 为

\[
\tfrac12\{(L^2+1)/2-1+\log[2/(L^2+1)]\},
\]

随 L 增大而无界。这仅揭示当前分数的宽度限制，不是用 Gaussian 生成标签来证明某种实际方法的优越性。

## 反例二：乘面积仍不能得到分区可加的质量

设一维 query 为 `N(0,1)`，key 用每个区间上的均匀分布前两矩表示。父区间 `B=[−1,1]` 对应 `N(0,1/3)`；子区间 `B₋=[−1,0]`、`B₊=[0,1]` 分别对应 `N(−1/2,1/12)`、`N(1/2,1/12)`。全部 covariance 严格为正，不涉及浮点舍入、奇异矩阵或数值 floor。

一维的精确式是

\[
J=\tfrac12\{D/C+C/D-2+\delta^2(C^{-1}+D^{-1})\}.
\]

由此 `J_parent=2/3`，每个 `J_child=20/3`。对拟议的面积加权质量

\[
M(B)=|B|\exp[-J(G_q,G_B)/2],
\]

得到

\[
M(B)=2e^{-1/3},\qquad M(B_-)+M(B_+)=2e^{-10/3},\qquad
\frac{M(B_-)+M(B_+)}{M(B)}=e^{-3}.
\]

保持另一轴相同即可嵌入二维。给其他不变的 key 一同归一化后，上述质量差也会改变它们与这个区域之间的概率分配，因此 softmax 不能一般消除该问题。此反例否定的是“用当前 Gaussian Jeffreys 相似度乘面积直接定义可加区域质量”的主张；不是所有 Gaussian PE、所有 KL 构造或有限维 Q/K 都不可能可加的证明。

## 文献与本笔记推导的边界

- [Banerjee、Merugu、Dhillon、Ghosh，JMLR 2005，Clustering with Bregman Divergences](https://jmlr.org/papers/v6/banerjee05b.html) 给出 regular exponential families 与相应 Bregman divergences 的关系。这支持自然参数与矩参数之间的经典关系，不是本项目的新理论。
- [Vilnis、McCallum，ICLR 2015，Word Representations via Gaussian Embedding](https://arxiv.org/abs/1412.6623) 已将高斯分布作为表示并采用 KL 等关系；其实际应用与几何 token 任务不同，不能借用其词嵌入实验支持本项目效果。
- [Cordonnier、Loukas、Jaggi，ICLR 2020，Appendix E](https://arxiv.org/html/1911.03584v2) 已讨论广义二次位置编码。少量二次位置通道不是新颖性证据。

这里的 11/19 通道排列、固定 D 时的宽度限制，以及具体区间分割反例，是从这些已知公式对当前设想作出的直接推导，不宣称文献已经验证了这个 PE 的效果或拒绝结论。

## 最小校验与研究处理

校验程序、事前源码哈希和固定输入规则保存在 [纯合成校验目录](/home/liwt/GRoPE/results/compact_gaussian_dual_pe_reserve_20261101)。程序先用 `Fraction` 精确检查 `2/3`、`20/3` 及指数差 `3`，再用 seed `20261101` 的二维和三维各八对严格正定高斯核对直接 KL、双参数恒等式、Q/K 展开和共同可逆仿射变换。有限样本只核对实现与公式一致，不证明全参数域的浮点稳定性。

程序禁止 `np.load` 和模型/训练框架导入，仅使用程序内合成数组，固定 OMP/MKL/OpenBLAS 四线程，不训练、不计算真实标签指标。事前绝对误差界固定为 `1e−10`，不按结果重选样例或阈值。

当前处理是保留这份未选型的否证笔记，不将它直接接替 R38，不读取旧 TRAIN 来试分数。即使上述有限数值校验全部通过，分区反例和宽度限制仍然成立。重新定义积分质量可能产生另一个模型，但那会丢失此处现成的有限维分解，不能作为对当前构造的等价修复；本笔记没有选择这种后续模型。
