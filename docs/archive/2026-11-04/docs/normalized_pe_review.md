# Normalized Expected-RoPE 独立审查

审查对象是 `models/gaussian_pe.py` 的 `normalized_rotary` 与五种 `normalized_*` 分支。本审查发现原实现的两类范数失败，随后在主代理明确授权下仅修复该函数；没有改模型其他分支、Gaussian covariance、损失或任何旧实验结果。

修复函数 SHA256：`5d3dc893e72545cf56dcfa32e999a4831ad6aaa7487bfc034871853aa2a0c8ce`。当时整个 `models/gaussian_pe.py` SHA256 为 `1fb7bbfbc1631abac28fcff4e304f2eea521f578229e29d603dfd98edd9104d4`。旧开发快照代表旧数值实现，主代理将另跑修复版开发。

## 数学含义与成立的性质

对第 \(f\) 个二维 content block，令

\[
\ell_f=-\tfrac12\omega_f^T\Sigma\omega_f,\qquad
\theta_f=\omega_f^T\mu,\qquad
E_{\mu,\Sigma}(c)=\operatorname{blockdiag}_f[e^{\ell_f}R(\theta_f)]c.
\]

该对照的实数数学定义是

\[
T_{\mu,\Sigma}(c)=\|c\|_2\frac{E_{\mu,\Sigma}(c)}{\|E_{\mu,\Sigma}(c)\|_2},\qquad c\ne0,
\]

并约定 \(T(0)=0\)。有限 covariance 对应的 Gaussian amplitudes 数学上严格为正，因此非零 content 的分母数学上非零。这个定义保留单个向量范数，零 covariance 时回到普通 RoPE；输出仍是 48 维，无额外可学习参数。各 `normalized_*` 与 `projected_rope` 的可学习参数名称、形状完全相同。

common translation \(t\) 只额外施加共享的正交 block rotation \(R(\omega_f^Tt)\)，不改变归一化分母。因此共同平移前后的 Q/K 内积相同。测试覆盖直接 helper 和完整 matcher 的共同像素平移。这不是任意坐标旋转不变性；当前固定频率没有提供该更强保证。

在直接 Expected-RoPE 本身未发生浮点下溢、其范数高于参考 `F.normalize` epsilon 的普通数值区域，修复版的值与 `F.normalize(rotary(...,covariance))*original_norm` 一致。包含严格零 content blocks 时，features、mean、covariance 的梯度也与 float64 直接参考一致，避免了通过屏蔽 inactive blocks 而错误删除其普通梯度。

## 原实现反例与修复

原实现只减去所有频率中最大的 `logamp`，与 content 是否占据该频率无关：

```text
frequency = [[1,0], [2,0]]
covariance = 1000 I
mean = [0,0]
content = [0,0,1,0]
```

max offset 保住的低频 block 恰好为零，唯一非零高频 block 权重为 `exp(-1500)`。原实现 float32/float64 都得到全零输出，输入范数 1 变为 0。即使 covariance 降到 `20 I`，高频权重 `exp(-30)` 尚未下溢，原 `F.normalize(eps=1e-12)` 也只恢复到范数 **0.09357623**。此外，zero covariance、content `[1e-15,0,0,0]` 时，旧 epsilon 会破坏 point limit。

修复没有增加 attenuation floor，也没有改变上述数学定义：

1. 中间值使用 float64，并先按 content 最大绝对值安全缩放，单独保留稳定计算的原范数。
2. 先减去 occupied blocks 的公共 Gaussian log amplitude，再把 block 半径的 log 加入；随后对 occupied weighted log amplitudes 作公共 offset。
3. occupied blocks 使用单位方向乘以稳定的相对权重。至少一个 occupied block 的权重为 1，因此最终归一化不需要任意 epsilon floor。
4. inactive blocks 的普通直接 content 导数保留；仅当其指数系数本身超出 float64 表示范围时作指数上限。该分支的 forward content 严格为零，上限不改变 forward Gaussian 核。
5. 最后施加位置旋转并恢复原范数，cast 回输入 dtype。

先移除公共 Gaussian log amplitude 还避免一个容易忽略的精度问题：若直接计算 `logamp + log(radius)`，当共同 `logamp` 约为 `-1e20` 时，浮点加法会抹去半径差异。修复后，相同频率范数、`covariance=1e20 I` 仍保留正确 content 比例。

上述稀疏反例修复后输出为 `[0,0,1,0]`。测试还覆盖 random SPD 支持至 `1e30`、稀疏 content、float32 content 至 `1e-35`、float64 content 至 `1e-300`，并检查可表示范数恢复。旧有失败实现在测试文件中以 `legacy_normalized_rotary` 独立保留，作为反例证据，不把反例改成宽松容差通过。

## 不能声称严格线性等距、原始 Gaussian 期望或不确定性校准

**它是 content-dependent 非线性映射。** 归一化系数依赖 content 在不同频率上的能量，因此一般

\[
T(c+d)\ne T(c)+T(d),\qquad
\langle T(c),T(d)\rangle\ne\langle c,d\rangle.
\]

例如同一 mean/covariance 下，两个独立频率基向量各自归一化后保持原方向，但它们的和会被不同比例滤波。测试例中非线性残差范数为 **0.789619**，对应同几何内积由 **1** 变为 **1.380271**。所以应称“单 token 范数保持”，不能称为固定的线性正交位置算子或严格距离等距映射。与扩展到 96 维的线性 isometric completion 是不同对照。

**它不等于原始 Gaussian 期望。** `rotary(...,covariance)` 是 \(E[R_Xc]\)，重新归一化一般改变其长度。归一化与期望不能交换。零 covariance 的 point limit 是正确的数学极限，但不能据此声称任意 covariance 下仍是原始 Expected-RoPE 或线性期望算子。

**raw EPE 的 covariance-sum 配对性质也不保留。** raw Q/K 第 \(f\) 个配对幅度仅取决于 \(\Sigma_q+\Sigma_k\)，但重新归一化的两个分母分别依赖各自 covariance 与 content。固定同一 covariance 总和的测试中，raw score 都为 **0.12791752**，两种 covariance 分配后的 normalized scores 却为 **0.77369022** 与 **0.98002570**。

**整体 uncertainty attenuation 被移除。** 对所有 occupied frequencies 共同的 attenuation 因子归一化后完全抵消。只有频率之间的相对滤波仍然存在。相同频率范数、各向同性 covariance 的测试中，从 `0.1 I` 增大到 `1e20 I`，normalized 输出始终与 point-RoPE 相同；raw EPE 则会收缩。这可能避免低范数 Q/K 导致的 logit/dustbin 偏移，但也丢失这种收缩携带的置信度/有效温度信息。一个全局 learned temperature 无法还原每个 token 被删除的不同衰减。它不是无需实验即可宣称的 uncertainty-calibrated 方法。

## 仍然存在的浮点范围与梯度限制

没有声明“所有有限输入都具有有限梯度”。最初的稀疏反例虽然 forward 已稳定，但若 loss 读取空的低频输出，真实 inactive content Jacobian 系数为 \(e^{1500}\)，超过 float64 范围。当前 float64 数值系数最多为 **1.797693e308**，cast 回 float32 的梯度为 **inf**；这是测试保留的极端边界，不能描述为通过所有梯度检查。普通条件下含空 blocks 的真实可表示梯度已单独验证。

若 float64 输入 covariance/mean 本身大到使 \(\omega^T\Sigma\omega\) 或 phase 计算溢出，或者数学输出范数超出输入 dtype 的范围，也无法提供无条件 finite-output 保证。极大 covariance 导致相对权重饱和时，即使梯度有限，也可能为零；“有限梯度”不等于还有足够的几何学习信号。当前训练的 float32 单位 content 和观察到的支持范围须通过实际运行的 finite loss/gradient 检查，不能用数学定义替代。

## 验证命令与结论

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -p test_normalized_pe.py -v
```

独立测试覆盖范数与维度、普通 EPE 值/梯度等价、平移相对性、point limit、重根、极大 SPD、稀疏/极小 content、零 content、旧数值反例、上述非线性与 covariance-sum/uncertainty 反例，以及五种 matcher 的参数/输入控制。修复后可用于新一轮开发，但其真实效果、相对 shape/trace/area/shuffle 控制的优势和泛化仍须由冻结实验确认；本审查没有提供这些效果证据。
