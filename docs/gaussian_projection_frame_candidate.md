# Gaussian projection frame：数学可行性与否证清单

2026-10-30。只做独立数学审查与一次性 CPU 数值核对，未实现 matcher、未修改
models、未运行候选的学习实验。下列推导是对这个具体构造的分析，效果尚未验证。

## 判断

这个构造是一个真实的、3→4 维逐 token Q/K 几何变换；唯一对称 SPD 平方根确实
消除了直接使用 `R diag(s)` 的轴符号/重根 frame gauge。其全局相似变换不变性也
成立。但它并不保范数，不是高斯重叠核，也不一般满足相对变换的组合律。

尤其是：错误位置可以得到比完全相同位置更高的几何分数；像素坐标 K 会导致很大
条件数；同一个高斯跨视图时，它退化成一个正则化局部平面 homography，薄平面极限
再次丢掉切向椭圆率。因此值得做一个很小的数学/验证集机制 probe，但现在不值得
直接投入完整多 seed 确认训练。成功必须来自实证，不能从 `AA⁺=I` 推出。

## 定义与可证明性质

令世界坐标中的高斯为 `G=(μ,Σ)`，`Σ≻0`，`S=Σ^(1/2)` 为唯一对称 SPD 平方根：

\[
F_G=\begin{bmatrix}S&\mu\\0&1\end{bmatrix},\qquad
P=K[R\mid t],\qquad
A_G=P F_G=K[B\mid x],\quad B=RS,\quad x=R\mu+t.
\]

对三维内容块 `q,k` 定义

\[
q'_i=A_i^Tq_i,\qquad k'_j=A_j^+k_j,\qquad
\langle q'_i,k'_j\rangle=q_i^T M_{ij}k_j,\quad M_{ij}=A_iA_j^+.
\]

由于 B 可逆，A 满行秩，故 `A⁺=Aᵀ(AAᵀ)⁻¹`，以及 `AA⁺=I₃`。
这意味着同一个 token 的 self 变换是恒等，并不意味着 `Aᵀ` 或 `A⁺` 是等距映射。
48 维内容按 16 个三维块使用同一个 A 时变成 64 维，确实紧凑、没有逐 pair
bias/gate。与 48 维 RoPE 比较时仍应报告这个 4/3 输出维度成本。

### 世界坐标 gauge

共同改变世界坐标 `X*=aQX+b`，`a>0,Q∈SO(3)`，且相机同步变为

\[
\mu^*=aQ\mu+b,\quad \Sigma^*=a^2Q\Sigma Q^T,
\quad R^*=RQ^T,\quad t^*=at-RQ^Tb.
\]

唯一 SPD 根满足 `S*=aQSQᵀ`，从而

\[
A^*=aAO,\qquad O=\operatorname{diag}(Q^T,1),\qquad
(A^*)^+=a^{-1}O^TA^+.
\]

因此 `M*ᵢⱼ=Mᵢⱼ` 精确成立。一次性 float64 随机核对误差为 `4.94e-15`。
这是同时更新 Gaussian 和 camera 的坐标 gauge，不是只旋转物体、相机不动时的不变性。

S 在重根 SPD 处是良定义且光滑的矩阵函数，但直接 `eigh` 组装 S 的自动微分仍可能
遇到 repeated-eigenvector backward 问题。固定观测不需要这条几何梯度；若以后预测
Σ，必须单独核验矩阵平方根实现的导数，不能只援引数学唯一性。

## 相对语义到底是什么

显式展开有

\[
M_{ij}=K_i(B_iB_j^T+x_ix_j^T)
             (B_jB_j^T+x_jx_j^T)^{-1}K_j^{-1}.
\]

这不只读取 pixel center：完整 S 确实进入 `BᵢBⱼᵀ` 和 denominator。但它不是
`Gaussian overlap`。可以将其解释为共同潜变量的 homogeneous least-squares 回归：
取 `z~N(0,I₃)`、`\tilde z=(z,1)`，令两个随机齐次投影为 `rᵢ=Aᵢ\tilde z`、
`rⱼ=Aⱼ\tilde z`，则

\[
M_{ij}=\mathbb E[r_ir_j^T]\mathbb E[r_jr_j^T]^{-1}.
\]

同一个世界 Gaussian 在不同视角投影时，这个 latent coupling 有物理意义：两边
来自同一个随机 3D 点。两个独立拟合的不同 region Gaussian 则被隐式绑定为
`Xᵢ=μᵢ+Sᵢz, Xⱼ=μⱼ+Sⱼz`；这是一个规范化坐标 coupling，不是“实际相同表面点”
的保证，也不一般是 Gaussian optimal transport coupling。拟合出相似椭球不意味着
其标准化坐标是正确的 feature 对齐坐标。

### 同一个 Gaussian 跨相机：精确变成 homography

在 key camera 坐标记 `Cⱼ=RⱼΣRⱼᵀ`、`xⱼ=Rⱼμ+tⱼ`，相对相机
`Rᵢⱼ=RᵢRⱼᵀ`、`tᵢⱼ=tᵢ−Rᵢⱼtⱼ`。此时

\[
M_{ij}=K_i(R_{ij}+t_{ij}w_j^T)K_j^{-1},\qquad
w_j=(C_j+x_jx_j^T)^{-1}x_j
=\frac{C_j^{-1}x_j}{1+x_j^TC_j^{-1}x_j}.
\]

最后一个式子只用于解释，不建议数值上先求极薄 C 的逆。
这是 Gaussian 决定的一个平面 homography，平面写作 `wⱼᵀX=1`。
若 `Cⱼ` 有很小的法向方差 `ε²`、正常切向方差且 `nᵀxⱼ≠0`，则

\[
w_j\longrightarrow \frac{n}{n^Tx_j},\qquad
M_{ij}\longrightarrow K_i(R_{ij}+t_{ij}n^T/(n^Tx_j))K_j^{-1}.
\]

薄平面极限仅保留 normal 与 plane distance；切向椭圆率和切向主轴再次消失。
CPU 核对：固定切向 std .2，法向 std 从 .1、.01、.001 到 .0001 时，
相对该平面 H 的 Frobenius 误差依次约 `.005758, 5.802e-5, 5.802e-7, 5.802e-9`。
所以不能把这个候选简单解释成“HRoPE 再自然增加完整形状”：在最典型薄表面情形，
其有物理配对语义的跨相机作用正好又是平面机制。

## 两个直接排名反例

令同相机 `K=R=I,t=0`，所有 covariance 为 `s²I`。则

\[
M_{ij}=I+(\mu_i-\mu_j)\frac{\mu_j^T}{s^2+\|\mu_j\|^2}.
\]

**错误中心胜过 self：** 相同内容 `q=k=e₁`，query `μᵢ=(1,0,2)`。
完全相同 Gaussian 的 key 分数为 1；错误中心 `μⱼ=(.5,0,2)` 分数却为

\[
1+\frac{.25}{s^2+.25+4}>1.
\]

`s=.1` 时直接 NumPy pinv 计算 self `1.0000000000000013`，wrong `1.0586854460`。
这不是数值误差，而是候选本身的非对称双线性结构。共同训练的内容网络可能学会使用
它，但它没有普通正交 PE 在相同内容下的“self 为范数上界”保证。

**位置可以不可见：** 若所有 Gaussian 同深度 `μᵢ,z=μⱼ,z`，且共同内容是 `e₃`，
则所有横向位置的 `e₃ᵀMᵢⱼe₃=1`。若 key 在光轴、内容 k 位于其横向二维子空间，
则 `Mᵢⱼk=k`，任何 query 的横向位移都不改变该块分数。编码并非自动距离感知。

这些反例不证明任意训练任务必败，但足以否证“self 恒等已经带来可靠对应偏好”的
强解释。没有必要以大训练才能发现这点。

## 不满足一般组合律

\[
M_{ij}M_{jk}=A_i(A_j^+A_j)A_k^+,
\]

其中 `Aⱼ⁺Aⱼ` 是 4D 空间中的 rank-3 projector，通常不等于 I₄。
所以一般 `MᵢⱼMⱼₖ≠Mᵢₖ`。同相机、`s=.1`、中心 `(1,0,2),(0,0,2),(-1,0,2)`
的简单实例中，差的 Frobenius norm 是 `.2230493`。

因此它是“由两个 token 因子产生的相对双线性作用”，不是一般的 group representation，
也不是 square invertible frames 所具有的严格 relative closure。薄平面 homography
的特殊极限可恢复真正可组合的平面映射，不能据此推到一般体高斯或任意 token pairs。

## 条件数、深度与 principled 正则

`AAᵀ=K(C+xxᵀ)Kᵀ`。若深度远大于 support 半径，或 support 极细且视线近 grazing，
则最小奇异值很小，key 的 `A⁺k` 会放大。Q/K 各自范数不受界，固定温度并不能提供
统一 logit 上界。

薄 Gaussian **不必然** 奇异：正面平面时 `xxᵀ` 恰好填满原来的薄法向；上面数值实例
法向 std→0 时 `cond(A)` 稳定在约 15.37。真正危险的是视线在切平面内、过小的切向
support、或很大的 depth/support 比，而不是仅仅最小 covariance 特征值很小。

只打开已有 train/validation NPZ，对有效 q/k token observations 做 float64 检查：

| scene | coordinate convention | median cond(A) | p99 | max |
|---|---|---:|---:|---:|
| teatime | calibrated K=I | 34.49 | 122.37 | 563.61 |
| waldo_kitchen | calibrated K=I | 38.21 | 284.78 | 1251.31 |
| teatime | raw pixel K | 2892.13 | 13747.96 | 41457.63 |
| waldo_kitchen | raw pixel K | 3322.97 | 25294.24 | 104899.20 |

共 5,318 / 5,428 个 token observations，重复出现的视图未去重；这里只是输入数值范围
审计，不是统计效果实验。输入根目录为 `results/round3_real_support_data_adjacent`。
直接形成 Gram 的条件数再平方，因此 raw pixel K + float32 Gram solve 不合适。

建议先固定为 calibrated image coordinates `K=I`；这是坐标约定，不是新增网络模块。
若保留 K，需明确图像尺度/主点单位；任意像素换基会相应改变内容块的几何作用，不能
称像素表示不变。求 pinv 时用 A 本身的 QR/SVD/least-squares，不要先显式 inverse Gram
或 inverse covariance。

两种不同正则必须分清，不能混称“数值修复且原算子不变”：

1. **给 support 一个明确小厚度：** `Σ̃=Σ+τ²I`，再由它重新构造 S、A 和精确 pinv。
   若 τ 随世界单位共同缩放，gauge 性质与 exact self 恒等仍成立。τ 可以来自已有
   重建精度或有限像素角分辨率对应的长度；它实际改变了支撑 Gaussian，必须报告。
   仅用 `τ²=η trΣ` 并不能给任意深度/support 比一个统一条件数上界。
2. **Damped dual：** 在 calibrated basis 用
   `A†_λ=Aᵀ(AAᵀ+λI)⁻¹`，其中 λ 必须随 `a²` 缩放，例如 λ 与 `tr(AAᵀ)` 成比例。
   这可解释为 homogeneous regression 的 Tikhonov 正则。self 变成
   `H(H+λI)⁻¹`，不再恒等，且引入方向相关收缩；不能继续援引 `AA⁺=I` 的保证。
   可从预定数值 condition budget 选择最小 λ，而不是靠测试分数调 λ。

若一个候选必须靠每个 test scene 的 clip/gate/后校准才能工作，应拒绝它作为当前
“精炼、统一”的主方案，而不是继续积累修补组件。

## 最小可证伪实验顺序

**先数学/CPU 门槛：** 相似变换 gauge、轴符号与重根、同 token self、上述错误中心
反例、正面薄面与 grazing 扫描、深度/support 比扫描、float32/64 对照、QR/SVD 与
Gram 实现对照。逐 token 导出 Q/K，直接验证其 dot 等于 pairwise 公式，报告编码
范数与 logit 分位数。特别测试两边不同 token 数、token permutation 和 invalid mask。

**再一个小开发集：** 同一内容 encoder/init/batches，比 calibrated-frame full、
保 normal/厚度/切向面积的 plane-isotropic、same-volume sphere、去几何 content、
已有 2D projected-center 强 baseline。保持同一坐标约定和输出预算。不要额外加一个
2D-RoPE 分支来让候选靠已知强基线通过。若想保留 2D 分支，应另开明示 composite
hypothesis，并做有无 Gaussian 分支的同预算比较。

**归因测试：** 同一个 Gaussian 跨视图与独立拟合 region Gaussian 分开；逐步改变
切向椭圆率、法向、中心、相机 baseline；比较 exact plane-limit 公式。检验 Gaussian
形状是否确实改变了正确的候选排序，而不是只放大 logits。若 full 与 plane control
没有稳定区别，结论应回到“Gaussian-derived plane frame”，不是完整形状有效。

**停止标准：** 开发 CE 或可靠的 top1 target mass 仍输给强 2D baseline；改进只来自
温度、clip 或极少数高范数 query；真实不同视角/GS partition 下不稳定；或者 full 与
plane control 没有可解释的差异。此时不打开确认集，也不把符号 gauge 的修复当效果。

## 一个更稳定的近邻，但属于不同候选

如果原 dual frame 因数值增益失败，可以单独研究 row-isometric frame：

\[
U_G=(A_GA_G^T)^{-1/2}A_G,\qquad q'_G=U_G^Tq,\quad k'_G=U_G^Tk.
\]

它同样 3→4 维；`UUᵀ=I₃`，Q/K 保范数，同 token self 为恒等，pair operator norm≤1，
且共同 `A*=aAO` 下 `U*=UO`，保留上述 gauge。它改变了相对语义，不能当成原
`AA⁺` 的数值等价版本，也不恢复组合律。若担心 SPD-root backward，可在固定几何
时先核验前向；严格端到端导数仍需单独处理。这里只提示一个单算子的可选替代，
不建议把它与 ridge、gate、normal loss 等叠加后再声称原构造成立。

## 相关工作边界

GTA 已经把几何相对变换用于 attention 特征对齐，并涉及 V/output 的坐标处理。
这里只分析 Q/K score，不能据此声称同等完整复现了 GTA。
[GTA 原论文](https://arxiv.org/abs/2310.10375)

HRoPE 以局部 slanted plane 的 token-level homography 作用到 attention；上面的薄
Gaussian 极限解释了两者为何接近。此推导不是在重现其 learned geometry head、V/O
或 NVS 训练设置。
[HRoPE 原论文](https://arxiv.org/abs/2610.28284)

MeRoPE 已明确把 homogeneous projective 编码在大 metric translation 下的 logit/
norm 放大列为问题，并采用保范数构造。它不是这里的高斯 frame，但这类稳定性风险
已有直接背景，不应当作新发现。
[MeRoPE 原论文](https://arxiv.org/abs/2609.01252)
