# R38 后续同三维信息对照：仅设计准备

2026-11-01。本文件仅依据既有源码和解析推导准备一个可能需要的 matched control；未运行模型或新增数值实验，未读取真实 NPZ、标签或未使用数据，未进行候选选型。只有 R38 原八门全部阳性后，才可能另行审查、冻结并执行该对照。本文件不是已选 R39，不修改 R38 的设计、冻结源码、人口、门槛或正在执行的实验，也不使缺少独立概率参考的 R38 结果获得晋升资格。

## 唯一拟议对照及旧角色覆盖范围

唯一拟议对照是：**同一 R38 三维输入，经严格前方条件矩后，在条件均值处作 Jacobian 投影的单二维 Gaussian**。其用途是检查 R38 的非线性投影相对于一个使用相同三维信息的解析近似是否仍有收益；原有强对照和八门继续保留，不能由本对照替换。

- [R20 的 prepare_query](../models/round20_anchor_pe.py) 使用旧二维均值、协方差和中心，构造 `Ca=C+(mu-a)(mu-a)^T`；其 `mean_only_matched_tau`、`anchor_trace` 等角色没有对 R38 的同一三维 `(a,Q)` 作 Jacobian 投影。
- [R21 的 prepare/raw_mass](../models/round21_matched_decomposition.py) 匹配 `mu_c/a_c/a_ca/full4` 的二维几何与共同内容。相关二维统计由 [R10 prepare_support](../models/round10_partition_pe.py) 对逐像素已投影支撑归约，通常不等于三维整体矩的 Jacobian 投影。
- [R10 projected_3d_moments/project_finite_support](../experiments/round10_partition_data.py) 确实包含先聚合三维矩、再 Jacobian 投影的路径，但中心是整体 `mu`，不是 R38 的 `a`；它还沿用目标 `Z>=.1`、屏幕坐标和协方差特征值截断。因此不能把它冒称为下面这个无新增截断、同 R38 三维信息的对照。
- [R29 _prepare_validated](../models/round29_log_rms_projective_pe.py) 已给出 RMS 标准化、深度轴 QR 分解及横向条件因子。下面沿用其数学分解，不改变 R29/R38 的生产实现。

直接使用普通 `J(a) Q J(a)^T` 没有自然的全域定义：`a_z=0` 时 Jacobian 奇异；`a_z<0` 且深度方差非零时仍可能存在严格正的前方质量。把这些行删掉、清零，或引入新的深度 clamp，都会削弱或改变对照。本准备稿不采用这些补丁。

## 标准化条件矩的精确定义

以下均在 R38 的 reference-8 齐次坐标中书写。使用完全相同的三维中心 `a` 与因子 `F_a`，其中

`Q = F_a F_a^T = F F^T + (mu-a)(mu-a)^T`。

令 `s = sqrt(a_z^2 + Q_zz)`。当 `s>0` 时，定义 `m=a/s`、`L=F_a/s`。R38 在透视前的标准化三维变量可写为

`V = m + L xi + h [eta_1,eta_2,0]^T`，`h=.25`，

其中 `xi` 和 `eta` 是相互独立的标准 Gaussian。这里只是在标准化坐标中表达原有 RMS 横向正则，不另加屏幕 blur。

对可变深度分支，令

`sigma = ||L_z|| > 0`，`u=L_z/sigma`，`beta=L_xy u`。

因此 `sigma^2=Var(V_z)`，`beta=Cov(V_xy,V_z)/sigma`。取深度轴 `u` 的正交补 `N`，按 R29 的正向 QR 因子构造

`G G^T = (L_xy N)(L_xy N)^T + h^2 I_2`。

`G` 是 R29 所用的横向条件因子；此定义避免从两个接近的 covariance 矩阵相减得到 Schur 补。等价随机表示是 `V_z=m_z+sigma T`、`V_xy=m_xy+beta T+G epsilon`，其中 `T` 与 `epsilon` 相互独立。

记 `t=m_z/sigma`，`p_+=Phi(t)`，并定义

`lambda = phi(t)/Phi(t)`，`v = 1-lambda(lambda+t)`。

严格条件 `V_z>0` 的精确一、二阶矩为

`mu_+ = [m_xy+beta lambda, m_z+sigma lambda]^T`，

`B_+ = H_+ H_+^T`，其中

```text
H_+ = [ G       beta sqrt(v)  ]
      [ 0  0    sigma sqrt(v) ] .
```

`H_+` 是条件 covariance 的正向因子，不表示严格截断后的三维分布本身变成 Gaussian。在精确实数算术中，有限 `t` 有 `p_+>0`、`v>0` 和 `mu_{+,z}>0`。

随后只作一次一阶投影近似。对 `pi(x,y,z)=(x/z,y/z)`，定义

`c=pi(mu_+)`，`S=(J(mu_+) H_+)(J(mu_+) H_+)^T`，

其中

```text
J(mu_+) = [ 1/z  0    -x/z^2 ]
          [ 0    1/z  -y/z^2 ] ,  (x,y,z)=mu_+ .
```

对每个原生 key cell，定义 `r_j=p_+ integral_cell N(w;c,S) dw`，再使用原有的单次面积 floor：

`P = softmax(log(r_j + 1e-30 area_j) + b_j)`。

保留前方概率 `p_+` 很重要：加 floor 后，先把前方质量归一化为 1 通常不再给出同一算子。cell、FOV 与 reference-8 面积语义完全沿用 R38。

## 全人口分支与数值边界

- `s=0`：深度恒为零，raw 全零。
- `sigma=0,m_z<=0`：深度恒在非正半空间，raw 全零。
- `sigma=0,m_z>0`：前方概率为 1，直接对确定深度 Gaussian 投影。此分支在数学上应与 R38 完全相同，是未来必须独立验证的恒等关系。
- `sigma>0`：使用上述严格前方条件矩，无论原中心深度正、零或负，都保留该查询。

raw 全零的行仍经过原 floor 和同一内容 bias；非零内容时不能把结果写成必然均匀。原始三维 covariance 可以秩亏，无需特征值裁剪或 jitter。有效前方分支中，既有 `h>0` 的横向因子使投影后的二维 covariance 在精确算术中正定。

**这些解析性质不是 float64 可用性的证明。** 直接计算 `lambda`、`v=1-lambda(lambda+t)` 以及 `m_z+sigma lambda`，在极端负尾会发生严重抵消；`Phi(t)` 也可能下溢。即使 `H_+` 避免了矩阵 Schur 减法，标量条件矩仍可能算错或失去正性。未来若选择执行，必须先冻结稳定的尾部/条件矩实现及独立验证方法，验证各分支和跨分支一致性，再核验二维矩形积分；不能把数值非正量 clip 为正数、把下溢的正质量伪称结构零，或删除失败查询。资源和精度失败应按固定全人口保留。这里没有选择尾部近似算法、误差阈值或新的积分预算。

## 匹配项、中心语义与可支持的结论

共同项是同一 `(a,Q)`、原 RMS 正则和严格前方条件、原生 keys、完整人口，以及原 R20 `Ca` 产生的 `tau32` 和整矩阵 Torch float32 `Q24 @ K8.T` 内容 bias。不能由新 `S` 重算另一个温度，不能改为逐行点积或 double 内容乘法。这个旧 `Ca` 只是共享内容控制，不能据此声称整个算子已完全由新三维参数生成。

该对照与 R38 使用相同三维输入，但**不是同一投影概率定律**：R38 保留非线性比例变量的投影，拟议对照仅用条件矩的一阶 Gaussian 近似。条件中心 `mu_+` 一般不同于原中心 `a/s`；其屏幕中心也一般不同于 `pi(a)`，更不同于经过旧深度/屏幕截断的 R20 二维中心。不能把结果解释成“同一三维中心不变，只替换积分器”的单因素因果实验。

如果未来另行冻结，成本必须包括双方自己的三维输入归约、中心与因子准备、原内容准备，以及本对照的条件矩与 Jacobian；不能把 R38 已封存输入当作该对照的免费计算。新增对照的用途和任何判定标准必须事先明确，不能在看到其结果后改动旧强对照、原八门、独立概率参考要求或最后的实际任务与未使用数据确认要求。

本文件仅完成设计准备；没有实验结果、性能结论、数值通过结论或下一轮选型。
