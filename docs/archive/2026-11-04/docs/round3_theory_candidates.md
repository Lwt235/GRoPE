# 第三轮：精炼的高斯位置编码候选与否证条件

2026-10-29。此笔记独立阅读了 `models/geometry.py`、`models/rendered_attention.py` 和 `round2_report.md`。下面是待验证假设，不能当成已获得的正结果。

## 研究边界

基本对象是几何支撑高斯 \(G=(\mu,\Sigma)\)，其中 \(\Sigma=R\operatorname{diag}(s^2)R^T\)。\(\mu,\Sigma\) 是位置和空间支撑；颜色/SH 系数属于内容，opacity 也不能自动解释为定位精度。支撑协方差与位置误差协方差必须使用不同变量。高斯的轴方向符号、相等特征值对应的特征向量选择、轴顺序都不是几何信息，因此编码只能依赖 \(\Sigma\)，不能直接消费未作商空间处理的 \(R,s\)。

优先考察两种直接逐 token 变换 Q/K 的构造。二者均不需要 pair bias、可靠性门控或额外法向模块，但代价和语义不同。

## 候选 A：平方根密度的等距提升，优先推荐验证

令 \(p_G(x)=\mathcal N(x;\mu,\Sigma)\)，并定义

\[
u_G(x)=\sqrt{p_G(x)},\qquad \|u_G\|_{L^2}=1.
\]

内容向量逐 token 提升为

\[
\widetilde q_G(x)=q\,u_G(x),\qquad
\widetilde k_G(x)=k\,u_G(x).
\]

它满足严格的等距性质和分数分解：

\[
\|\widetilde q_G\|=\|q\|,\qquad
\langle\widetilde q_{G_i},\widetilde k_{G_j}\rangle
=(q_i^Tk_j)\,\operatorname{BC}(G_i,G_j).
\]

这里 \(\operatorname{BC}=\int\sqrt{p_i p_j}\,dx\)，解析式为

\[
A=(\Sigma_i+\Sigma_j)/2,\quad\delta=\mu_i-\mu_j,
\]

\[
\operatorname{BC}=
\frac{|\Sigma_i|^{1/4}|\Sigma_j|^{1/4}}{|A|^{1/2}}
\exp[-\tfrac18\delta^T A^{-1}\delta].
\]

这是一个结构明确的“空间区域作为位置编码”：把点的位置特征改成单位范数的空间支撑函数，内容与位置通过一个张量积相接。完整协方差的作用是决定该函数的空间形状；协方差不会作为任意跨视角身份码附加到分数中。

### 可证明与不可声称的性质

1. 连续精确形式对共同可逆仿射坐标变换不变。密度的 Jacobian 因子在积分中相消；这比单独输出轴频率的旋转等变性更明确。它不表示模型已经对物体变形或投影近似误差不变。
2. 相同高斯的核为 1，内容范数不随协方差变大而衰减，因此排除了第二轮“候选分数仅因 Expected RoPE 幅值缩小而普遍输给固定 dustbin”的一个来源。它仍可能改变分数分布，不能因此宣称无对应校准已解决。
3. 该构造对轴符号、排列、重根 eigenspace 均无歧义。若实际计算通过 Cholesky/solve/logdet 实现，无需对 eigenvector 求导。极薄高斯需要预先明确物理 thickness/floor；任意 clamp 会改变所比较的区域。
4. 当 \(\Sigma\to0\) 时，它**不退回普通 RoPE**：不同中心的无限窄分布趋于互相正交。这符合“空间支撑重叠”的语义，但不适合把所有点距离都保留为平滑信号的任务。
5. BC 是分布相似度，不是区域 IoU、包含率或可见性。特别是两个同心但尺度不同的高斯也受到惩罚；不能把它叫作“投影可见区域重叠的真值”。

### 实现与复杂度

有限空间 quadrature 给出 \(\phi_m(G)=\sqrt{w_m p_G(x_m)}\)，随后作逐 token L2 归一化。得到 \(\widetilde q=q\otimes\phi(G)\)，head 维度由 \(D\) 增为 \(DM\)。归一化使等距结论在离散情况下依然精确，但只在 quadrature 充分精细时近似真正的 BC。

也可使用 Fourier quadrature。按单位 Fourier 变换约定，

\[
\widehat u_G(\omega)
=(2/\pi)^{d/4}|\Sigma|^{1/4}
e^{-\omega^T\Sigma\omega}e^{-i\omega^T\mu}.
\]

通过 Parseval 得到同一 BC，但有限频率可能带来混叠、负的近似核、过大协方差时有效频率仅剩零频等问题。不能把均值特征的简单 Expected RoPE 直接称作平方根密度特征：其频率衰减常数、determinant 与归一化均不同。

必须报告 DM、实际显存与耗时，并给出同输出维度/同预算的强对照。用精确 BC 直接乘原 qk 分数可先检验该归纳偏置有没有价值，作为 oracle kernel 上限；但若只实现 pairwise 解析核，还不能声称已经实现了小维度、一次性逐 token 的标准 Q/K 编码。

不存在一般性的“零维度代价”变形：若要求方阵 \(P_G\in\mathbb R^{D\times D}\) 严格等距，则 \(P_i^T P_j\) 为正交矩阵，其奇异值全为 1，不可能对任意 pair 等于 \(\operatorname{BC}_{ij}I\)（当 \(0<\operatorname{BC}_{ij}<1\)）。等距、固定原维度、精确纯标量 BC 三个条件一般不可兼得。

### 关键反例与失败标准

- 若内容内积为负，乘一个接近零的核会让该候选分数上升；“几何不相容必然降低 attention 分数”并不成立。应检查内容分数符号/分布，并与同样内容网络的加性核对照比较，不能靠更换温度掩盖此事。
- 当前标签是查询中心点落入哪个目标 patch，而 BC 比较整个支撑。如果几何支撑宽度是无关 nuisance，最优中心匹配器可能应该忽略它；在此任务上失败不能自动否证所有 region PE，但足以否证其直接替换当前位置编码的收益主张。
- 对 RGB-D patch，源/目标 footprint 由不同相机和 patch 网格决定，本来就不必相等。应同时比较固定 footprint 核，否则相机投影贡献会被错误归因于协方差。
- 当前 12px 均匀 patch 的高斯 std 约 3.46px。每 12px 一个 quadrature 点会严重欠采样。必须独立用训练范围生成随机高斯 pairs，测近似核与精确 BC 的误差、极小尺度、边界和共同平移/旋转后的偏差；归一化不能修复分辨率不足。
- 若 key 被投影到 query 相机，则 key 编码依赖 query 视角；可以在每个 view pair 内预计算，但不能声称一套全局 K 能用于所有视角的全部 query。
- 若真实 3DGS 经 splitting/cloning 后几何组成改变而渲染几乎不变，逐 primitive 的编码和 softmax 可能随 token 数量改变。该表示层面的非唯一性与 R 的符号 gauge 是两个问题，后者解决不代表前者解决。

### 新颖性边界

平方根密度与 Bhattacharyya 内积属于已知 Probability Product Kernel，不是本项目的新核。Jebara、Kondor、Howard（2004）已系统给出分布内积框架。[原论文](https://www.jmlr.org/papers/volume5/jebara04a/jebara04a.pdf)

如果本路线成立，可信贡献应落在实际 3DGS/patch 支撑到 PE 的一致构造、可承受的逐 token 近似、与已有位置编码的公平实验，以及特定任务的实证，而不是给 BC 重新起名字。

## 候选 B：Expected RoPE 的等距补全，作为低成本对照

对每个 2 维内容块 \(v\)、频率 \(\omega\)，令

\[
t_G=\omega^T\Sigma\omega,\quad
a_G=e^{-t_G/2},\quad b_G=\sqrt{1-e^{-t_G}},
\quad \theta_G=\omega^T\mu.
\]

定义一个 2→4 维的逐 token 线性映射

\[
P_Gv=\begin{bmatrix}a_G R(\theta_G)v\\b_Gv\end{bmatrix}.
\]

则 \(P_G^T P_G=I\)，且

\[
(P_iq)^T(P_jk)
=a_i a_j q^T R(\theta_j-\theta_i)k+b_i b_j q^Tk.
\]

它只把 Expected RoPE 丢失的能量放入补空间，不需要学习 gate、单独的形状 MLP 或 pair bias。整体 head 维度 D→2D；Σ→0 时补空间为零，严格退回普通 RoPE。其代数属于经典 contraction 的 isometric/unitary dilation 思想，不能主张新算子数学。一个一手作者讲义给出了经典 Halmos block dilation。[讲义](https://cms-math.net.technion.ac.il/files/2016/04/Passer_MVOT_2017-1.pdf)

### 为什么它与候选 A 不同

这是“区域越宽，位置相位越不确定，内容越占主导”的归纳偏置。它没有严格的 BC 重叠语义：当所有 Σ 很大时 a→0，分数退回纯内容，位置和形状差异可能一起消失；一个极小支撑与极大支撑之间则可能被压到零。补空间取同一个身份算子是一种明确但非唯一的选择，因此不能从其保范数性质推出它必然是正确的高斯几何。

轴频率同样看不见 Σ 的非对角元素；必须用固定且覆盖不同方向的斜频率，再给相同频率的普通 3D RoPE 对照。有限频率的坐标旋转等变性也不是精确的，不能从 Σ 的 gauge-safe 性混推出全局 SO(3) 不变性。

数值上宜用 \(b=\sqrt{-\operatorname{expm1}(-t)}\) 避免 t 很小时的消减误差。b 对 t 在零点有平方根导数奇点；几何固定的匹配实验无碍，但端到端预测 Σ 时需明确小正 covariance floor 或不同光滑参数化。添加 floor 后的 point limit 是极限而非精确零点实现。

建议只把它当作低成本诊断：如果它仅修复 norm/dustbin 却没有稳定提高候选排序与无对应 AP，结论应是幅值校准效应，而不是“完整高斯形状有效”。

## 不推荐优先投入的路线

1. 将 \(\mu\)、logΣ 或 Σ 的元素直接拼接进 RoPE 相位。它当然是简洁可用的描述子基线，但“高斯作为位置编码”本身不足以构成新颖性。Mip-NeRF 补充材料 G 节已测试均值 PE 与协方差上三角有符号平方根 PE 的拼接。[官方补充材料](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Barron_Mip-NeRF_A_Multiscale_ICCV_2021_supplemental.pdf)
2. 直接将 \(R\operatorname{diag}(s)\) 当局部 frame 做 relative Q/K。第二轮之前的 gauge 反例已经否证其一般定义。换为唯一对称 \(\Sigma^{1/2}\) 可以解决轴选择歧义，但非正交相对变换仍会放大条件数；它也不自动对全局旋转不变。
3. 把 IPE 的高斯期望作为全新 PE。对 \(e^{i\omega^Tx}\) 求期望得到 \(e^{-\omega^T\Sigma\omega/2}e^{i\omega^T\mu}\) 是已有 Integrated PE 的核心。Mip-NeRF 已用高斯支撑实现多尺度抗混叠。[项目与论文入口](https://jonbarron.info/mipnerf/)

## 最小验证顺序

先冻结一个明确的任务语义，避免用 BC 定义标签再证明 BC 最好。候选 A 先用精确核检验潜在效果，再检验有限 Q/K 特征的近似误差与预算；候选 B 直接在相同内容容量下实现。两者都需要位置、固定/球形支撑、完整支撑、shuffle/区域共享支撑的消融，并同时报告候选排序、拒绝、AP、实际成本。

原独立随机高斯身份任务只用作反例，不用作路线选择。真正的正证据至少需要在未参与设计的新场景、不同 patch 大小/视角与受控支撑误差下仍有增益，并确认额外颜色/相机信息、容量、投影或单纯 norm 变化都不能解释结果。若完整 Σ 相比固定 footprint 无稳定增益，就保留更简单的位置方案，不把失败改写成模块堆叠。
