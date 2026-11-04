# 方向性内容补偿：未选择的解析备选复核

2026-10-31。仅作解析审查；未读取 R20 新效果或 targets，未运行实验，未修改冻结源。比较显式梯度补偿与固定一次的均值反馈，两者都未选择为候选，没有选定梯度估计器或离散实现合同，也不主张新颖性。背景是已有 [R17 trace 推导](round17_trace_theory.md) 与 [R18 条件归一化推导](round18_conditional_normalization_math_review.md)，这里不把后者的连续结论当成原生离散实验已通过。

## 等式成立，而且可以写得更精炼

令 (C\succeq0, h>0, S=C+h^2I, z=y-\mu)，对一个固定的列向量 (g) 定义无 floor 的连续密度

\[
 p_g(y)\propto \mathcal N(y;\mu,S)
 \exp\{T[c(y)-g^TCS^{-1}z]\}.
\]

若整个概率支撑上 (c(y)=c_0+g^Tz)，利用 (I-CS^{-1}=h^2S^{-1})，指数中的线性项为 (Th^2g^TS^{-1}z)。配方给出

\[
 p_g=\mathcal N(\mu+Th^2g,S).
\]

这是无限域、全局线性内容下的**完整分布恒等式**，不仅是均值近似。它恰好重现 R18 理想连续线性情形的边缘分布，但不等同于 R18 一般非线性或有限 native-cell 算子。

任意内容函数下，线性补偿还可吸收进同一个 Gaussian：

\[
 \mathcal N(y;\mu,S)e^{-Tg^TCS^{-1}z}
 =A\,\mathcal N(y;\mu-TCg,S),\qquad
 \log A=\tfrac12T^2g^TCS^{-1}Cg.
\]

因此，无 floor 时只需把几何中心改成 **\(\mu-TCg\)**，再应用原内容 (Tc(y))；query 共有的 (A) 在归一化中消失。这个形式不用显式计算 (S^{-1})，保留单 Gaussian 与一次内容归一化，结构本身足够精炼。它属于**内容依赖的 Gaussian 中心补偿**；若 PE 要求仅由几何参数计算，这个额外内容依赖必须公开。(C=0) 时退化原 point 内容作用；常量内容且 (g=0)、或 (T=0) 时不改变旧算子。

## 非线性时的限制比“均值一阶近似”更严格

在可归一化、可交换微分积分的光滑无限域条件下，分部积分给出精确关系

\[
 \mathbb E_{p_g}[Y-\mu]
 =T S\mathbb E_{p_g}[\nabla c(Y)]-TCg
 =Th^2g+TS\bigl(\mathbb E_{p_g}[\nabla c(Y)]-g\bigr).
\]

所以取 (g=\nabla c(\mu)) 只是一种**空间线性化**；一般不能声称其关于 (T) 的一阶响应也等于 (h^2g)。一个直接反例是单维有界内容 (c(z)=\sin(kz))、精确局部梯度 (g=k)：

\[
 \left.\frac{d}{dT}\mathbb E_{p_g}[Z]\right|_{T=0}
 =k\,[S e^{-k^2S/2}-C].
\]

当括号为负时，微弱内容的均值响应甚至反向。此例始终可归一化；问题来自整个 Gaussian 支撑上的内容弯曲，而非数值积分或梯度估计误差。一般非线性内容也会改变输出协方差，不能继续宣称它等于 (S)。

即使真实内容是线性的，若真实梯度为 (g_0)、补偿使用估计 \(\hat g\)，均值也是

\[
 \mu+Th^2g_0+TC(g_0-\hat g).
\]

因此，被抵消的协方差放大同时成为**梯度估计误差的放大因子**。减弱内容牵引也可能撤销有用的几何纠偏；这里没有定位、区域质量或扰动收益的保证。

## 离散输入、积分与边界合同尚未闭合

- **g 的来源。** 每个 query 的冻结或 learned QK 只给出 native key 上的标量 (c_j)，没有唯一的连续梯度。固定插值的导数或无标签的线性回归都可能合法，但需要事先指定坐标单位、权重/邻域、边界与视野外位置、退化情况，以及学习时的梯度路径。“不新增网络”并不等于没有新增建模选择。当前未选择任何一种。
- **不能把积分与 key-center 乘法混同。** 用 `log cell_mass + T(c_j - gᵀ C S⁻¹(key_center_j-μ))` 通常不等于对补偿密度在整个 cell 内积分。若定义 (c(y)=c_j) 在每个 cell 内恒定，则无 floor 的精确形式可写为 `exp(T c_j) × mass[N(μ−TCg,S), B_j]`；但此 piecewise-constant 内容已经不满足全局线性前提。即使 (c_j) 来自线性场的中心采样，也不能直接套用连续均值恒等式。
- **有限 FOV 与 native argmax。** 即使连续内容完全线性，有限 FOV 内也得到截断 Gaussian；均值含边界修正，协方差通常不再是 (S)。key-center 概率均值、native winner 与 PCK 不是连续均值。若 (g) 随 native 分区重新估计，也不能据此新增分割/合并一致性保证。
- **floor 不能悄悄忽略。** 配方中的 (A) 只对纯 Gaussian 测度是可约掉的公共因子。例如权重取 ([A m_j+\epsilon |B_j|]e^{Tc_j})，等价 floor 是 (\epsilon/A)，而不是固定 \(\epsilon\)。若先给原密度加均匀 floor 再做线性补偿，floor 本身也会随位置变化。因此直接平移 Gaussian 后沿用固定面积 floor 是另一份必须显式声明的算子合同；tiny-FOV 情形尤其不能靠连续配方跳过审计。

## 有区别的备选：固定一次的内容均值反馈

仍限无限域、无 floor、同一个光滑内容函数，先定义

\[
 P_T\propto\mathcal N(\mu,S)e^{Tc},\quad P_0=\mathcal N(\mu,S),\quad
 r=\mathbb E_{P_T}Y-\mu,\quad
 \mu'=\mu-CS^{-1}r,\quad
 P'_T\propto\mathcal N(\mu',S)e^{Tc}.
\]

只做这一次反馈；(C,h,c) 均保持相同。这里直接使用第一次分布的均值，无需定义 key 场的局部插值梯度。实现中心更新也无需除以 (T)：可用 (CS^{-1}r=r-h^2S^{-1}r)。矩阵 (CS^{-1}) 的特征值为 \(\lambda/(\lambda+h^2)\in[0,1)\)，所以反移中心的范数不超过 \(\|r\|\)；这不是对第二次概率均值或 PCK 的保证。

全局线性 (c(y)=c_0+g^T(y-\mu)) 时，(r=TSg)、\(\mu'=\mu-TCg\)，故仍有完整恒等式 (P'_T=\mathcal N(\mu+Th^2g,S))。对于一般光滑且允许在 (T=0) 附近交换积分与微分的 (c)，记 \(b=\mathbb E_{\mathcal N(\mu,S)}\nabla c\)。Stein 恒等式和一次展开分别给出

\[
 r=TSb+O(T^2),\qquad
 \mu'=\mu-TCb+O(T^2),\qquad
 \mathbb E_{P'_T}Y=\mu+Th^2b+O(T^2).
\]

这次是关于 (T) 的**真实一阶均值响应**，避开了显式局部 (g=\nabla c(\mu)) 的前述反例。R18 的连续条件分布满足

\[
 \left.\partial_T\mathbb E[Y]\right|_0
 =\mathbb E_{X\sim\mathcal N(\mu,C)}
   \operatorname{Cov}_{Y\sim\mathcal N(X,h^2I)}(Y,c(Y))
 =h^2\mathbb E_{\mathcal N(\mu,S)}\nabla c,
\]

所以二者的一阶**均值**确实相同。

但它们的概率分布通常连一阶也不同。取单维 \(\mu=0,c(y)=-\kappa y^2/2,\kappa>0\)。对称性使 (r=0)，反馈版等于原 tilted Gaussian，方差为 \(S/(1+T\kappa S)\)。R18 条件版的方差为

\[
 \frac{C}{(1+T\kappa h^2)^2}+\frac{h^2}{1+T\kappa h^2}.
\]

两者在 (T=0) 的方差导数分别为 \(-\kappa S^2\) 和 \(-\kappa(2Ch^2+h^4)\)，相差 \(-\kappa C^2\)。均值匹配不意味着概率、区域质量、协方差、native mode 或有限 (T=10) 的行为匹配。

**有限 FOV 的一个定义边界。** 可用同一 native 几何/floor 下的 (P_T,P_0) 定义

\[
 r_\Delta=\sum_j(P_{T,j}-P_{0,j})u_j,\qquad
 \mu'=\mu-CS^{-1}r_\Delta,
\]

其中 (u_j) 为 native key 中心。只替换差值中的基线均值，**更新起点仍是 Gaussian 参数 \(\mu\)**。若起点也改成 \(\sum_jP_{0,j}u_j\)，通常会破坏 (T=0) 的原算子恒等式。上述差值定义在 (T=0) 时为零；(C=0) 时反移为零，因此分别恢复原无内容 Gaussian 与原 point 内容算子。常量内容也不产生反移。

这是一份尚未选择的离散定义，不继承无限域定理：key-center 加权均值近似连续均值；FOV/floor 改变基线均值对 \(\mu\) 的导数，native 内容也不是唯一连续场。计算 (P_T,P_0) 可共享第一次几何质量，但 \(\mu'\ne\mu\) 时通常还需要**第二次完整 cell 积分和归一化**。不能把 key-center 指数重权当作精确的新中心 cell 质量。两次内容必须为同一 (c_j)，不能在反馈后偷偷更换 Q/K 或池化规则。

## 判断

**显式局部梯度版暂不推进为候选。** 单 Gaussian 的等价中心补偿确实简洁，但它没有自动解决离散 QK 的梯度定义，且存在上面的明确非线性反例。当前证据不足以选择一个稳健、无标签且跨分区可信的 (g)。

**一次均值反馈值得保留为不同的未选假说。** 它消除了局部梯度估计器这一建模选择，且得到更强的连续一阶均值结论；结构是一次固定的确定性统计反馈，没有新增网络或模块集合。但它仍需第二次几何求值，离散/FOV 合同没有获得上述连续保证，也没有有限内容强度或任务效果证据。两种形式都不能据均值公式推断全部既有门槛会通过。

当实际 GS 使用情形没有可定义的内容场，或 (T=0) 时，两种补偿都消失；本推导不产生额外的纯几何 GS 收益。这里没有提出参数网格、实验或新的通过条件，R20 仍按原冻结协议独立完成。
