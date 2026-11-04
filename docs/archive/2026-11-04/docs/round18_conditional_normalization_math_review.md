# R18 Gaussian 条件归一化：数学与合成否证审查

2026-10-31。结论为**条件通过数学/静态审查，允许按新协议进行最小合成数值否证**；这不是数值运行通过或研究效果通过。本审查没有执行任何数值案例，没有读取真实输入、标签或新数据，没有选择 anchor 替代方案，也没有修改生产源码。审阅对象为 [合成协议](../ROUND18_CONDITIONAL_PREFLIGHT_PROTOCOL.md)、[生产算子](../models/round18_conditional_pe.py)、[runner](../experiments/round18_conditional_preflight.py) 和 [独立参考](../experiments/round18_conditional_reference.py)。下述冻结前小修已由 root 接受，须与最终 reference 一并封存。

## 1. 概率法则成立，有限 FOV 的倾斜不能省略

令 Γ 是 `N(μ,C)` 的概率测度；允许 C 半正定、rank1 或零矩阵，不要求存在二维密度。h>0，B_j 是内部不交、正面积的有限矩形，U 为其并集，content c_j 有限且在潜变量 x 积分时保持不变。定义

\[
 m_j(x)=\int_{B_j}\phi_{h^2I}(y-x)dy,\quad
 M_j(x)=m_j(x)+\epsilon |B_j|,\quad
 A(x)=\sum_jM_j(x),\quad D=E_\Gamma A.
\]

有限 U 下取 ε≥0；当前生产接口限定 ε>0。h>0 且 U 正面积保证 A(x)>0、D>0；不交性给出 `A≤1+ε|U|`。令

\[
 f_j(x)=\frac{M_j(x)e^{Tc_j}}{Z_T(x)},\quad
 Z_T(x)=\sum_kM_k(x)e^{Tc_k},\quad
 \pi_0(dx)=\frac{A(x)}D\Gamma(dx),\quad
 P_j=E_{\pi_0}f_j.
\]

则各 P_j≥0 且和为1。π0 是几何 FOV 条件化的 latent 分布，**不是**一般情况下的原 Γ，也不是 sensor visibility 后验。它与 T 无关，但依赖 key bank 所覆盖的 U、μ、C、h 和 floor。

两个极限严格成立：

- T=0 时 `Z0=A`，故 `P_j=EΓ M_j/EΓ A`。利用 Gaussian 卷积，分子恰为 `∫Bj N(y;μ,C+h²I)dy+ε|Bj|`，因此恢复原 full Gaussian 的同 FOV 条件概率和一次面积 floor。
- C=0 时 Γ=δμ，π0 同为 δμ，故 P_j 是原 point 的 `softmax(log M_j(μ)+T c_j)`。

这是实数运算的恒等式，不意味着新的 float64 GH/CDF 路径逐位重现旧混合 dtype、GL32 生产代码。协议将数值等价与旧字节身份分开是正确的。直接改用 `EΓ f_j` 则一般失去 T0 的 full 卷积恢复，尤其会让 FOV 外 latent 点也以总质量1给 FOV 投票；它只能保留为已声明的诊断控制。

旧“先积分后归一化”有另一个精确表示：

\[
 P_j^{old}=E_{\pi_T}f_j,\quad
 \pi_T(dx)=\frac{Z_T(x)}{E_\Gamma Z_T}\Gamma(dx).
\]

设 `r=Z_T/A`，则

\[
 P_j^{old}-P_j
 =\frac{\operatorname{Cov}_{\pi_0}(r,f_j)}{E_{\pi_0}r}.
\]

新式移除的是内容 evidence 对 latent 分布的额外重权，不是移除全部几何条件化。它可以减少宽 support 的内容牵引，也可能撤销有用的内容修正；没有一般性的 KL、PCK 或语义正确率单调改善定理。

## 2. 全平面线性场的闭式正确，但适用范围窄

仅在 **U=全平面、ε=0、真正连续的 `c(y)=gᵀy`** 下，A=1、π0=Γ，且

\[
 Z_T(x)=e^{Tg^Tx+\frac12T^2h^2\|g\|^2},\qquad
 Y\mid X=x\sim N(x+Th^2g,h^2I).
\]

故新边缘为 `N(μ+Th²g,C+h²I)`；旧式先合并为 `N(μ,C+h²I)` 再指数倾斜，得到 `N(μ+T(C+h²I)g,C+h²I)`。这给出了明确、可否证的连续线性机制区别。

当前实际 content 是有限 native cell 的常值 c_j，不能将 cell 中心值代入连续闭式后声称精确相等。有限 FOV 会改变 π0，并截断每个局部倾斜分布；非线性/重复峰内容不再产生上述单 Gaussian。无界全平面上 ε>0 的均匀 floor 总质量无限，根本没有该归一化概率模型。

runner 的全平面均值/协方差检查直接使用已推导的局部平移和 Gaussian 矩，是有用的代数/矩积分 sanity check；它没有执行实际 finite-cell M/Z 路径，不能替代25个 native 案例的独立数值比较。连续 mode/均值结论也不能推广为 native argmax、PCK 或有限 FOV 稳健性保证。

## 3. Floor、外部质量与拆并

floor 必须只在局部 M_j 内加一次，随后同时进入 A、Z_T 和外积分。T0 时积分常数 ε|Bj| 仍只是一次 floor。非零 T 的 floor 贡献已经受局部归一化影响，一般**不等价**于对最终输出再加原 floor；不得在两处补加。所有坐标共同放大 s 时，μ/h/矩形放大 s、C 放大 s²、floor 密度缩小 s²，才保持同一模型。

若 latent 点极远、几何 FOV 质量相对 floor 可忽略，则新式仍趋于 `P_j∝|Bj|exp(Tc_j)`。因此 floor-dominated 行虽可数值有效，却没有保留有意义的局部定位保证；必须保留并明确报告，不能当作可见行删除。

当前法则始终将可见 key bank 归一为1，没有 outside/unmatched 或真实遮挡概率。添加全平面 exterior cell 时，其 floor 面积无限；不能沿用 ε|B|。若以后定义有界 ambient domain 或另一个 outside 质量/内容，已经是新模型，需要另行声明，不能在本次合成检查中悄悄加入。

固定 U、固定潜变量法则、固定内容场时，key 合并具有以下精确边界：同 content 的子格满足 `M_parent=ΣM_child`，Z/A 不变，故 `P_parent=ΣP_child`。不同子格内容只能保留原细粒度加权质量后再合并；重新 pooling/L2 归一化成一个 c_parent 会改变算子。一个能维持原结果的“等效 content”通常随 x 改变，不是免费的单个 pooled key 描述。

query 合并也不由单一 μ/C 闭合。即便保留精确 Gaussian mixture `Γ=Σw_s Γ_s` 且 content 相同，新条件分布也按 `w_s D_s/ΣwD` 合并各 query 结果，而不一般等于源面积权重 w_s；将 mixture 再矩匹配为一个 Gaussian 又增加了闭合误差。R18 不会恢复丢失的多峰形状或源中心身份。

## 4. Tiny C、数值尾部与复杂度

对固定有限输入、h>0，有界的局部概率及正 D 给出 C→0 的连续 point 极限。无需 C 的逆，也不需要人为 covariance jitter；但不能给所有外部均值/tiny-FOV 输入一个统一的相对误差速率。有限阶积分还可能依赖 covariance 平方根的节点朝向；数学上的 Gaussian 旋转/gauge 等价不自动成为有限 GH 规则的精确等价。

外部标准正态坐标截断到 `[-L,L]²`，令 `t=4Φ(−L)` 是截断外 Γ 质量的并合上界。若总面积 floor 为 F=ε|U|≤1，则 A≤2，且真实截断内 normalizer 满足 `D_in≥F(1−t)`。因此每个概率分量的**截断误差**可保守界为

\[
 \frac{2t}{F(1-t)}.
\]

这不含 quadrature 误差，也不是 L1 上界。只用数值 Dhat 作分母的 `2t/Dhat` 可以作为描述性估计，但 Dhat 不是已证明的下界，不能单独标成严格证书。当前固定 FOV 面积20、ε=1e−30、L=14 下，上述 floor 下界已经足以让尾界远低于1e−8，无须依赖 tiny-visible 行的数值 normalizer。生产 E[A] 与参考 E[A] 的相对差仍应完整报告；概率归一正确不能证明 latent 几何质量积分正确。

GH64 和 GH96 可能一致地漏掉同一区域；GL384/512 的差也只是收敛诊断，不是所有输入上的严格误差定理。若参考自身未收敛，该案例是数值参考不足，不能归因为新概率法则数学错误，更不能据此宣告 production 通过。全部失败和 raw D 必须保留。

二维矩形的局部 isotropic 质量可以用两个一维 CDF 差相乘；但新式失去了非零内容下的一次 Gaussian 卷积闭式。Q 个 query、K 个 key、L_latent 个节点通常需 `O(Q K L_latent)` 的局部积分/归一化，另有内容计算。GH64² 是 **4096 个 latent 节点/query**，不是64；GH96² 是9216。可以分块节省内存，但不能省去相应算力。A/Z/π0 又依赖完整 key bank，因此目前应称条件积分 attention operator，不能直接称为已得到紧凑、独立逐 token 的有限 Q/K PE。

## 5. 冻结前源码检查与最小合成协议建议

生产实现正确计算 `Σw A f / Σw A`，稳定地减去最大内容后求指数，同时保存 old mix-then-normalize 与 bare expectation 诊断。rank0/rank1 不取 C 逆；positive-tail CDF 镜像避免直接减两个已舍入为1的数。当前固定 fixtures 不涉及极端面积/带宽溢出，不能将通过它们外推为“所有有限输入都数值安全”。

最小范围建议沿用 root 已选的固定预算与25案例，不增设参数/径向/anchor 搜索：十一种 Gaussian 几何各 T0/固定非零T 共22行，再加固定 spike、repeated-peaks、content-correction 三行。GH64 为唯一生产预算；GH96 仅诊断，不得看结果后晋升。独立 GL384/512、extent14 与 T0 的独立一维 conditional-Gaussian 积分，足以形成不同节点/积累路径的参考；共用 SciPy 基本函数和 NumPy 线性代数须披露，不称完全无共享依赖。

冻结前两项具体修订建议已获 root 接受：

1. content-correction 的新旧位置诊断应使用**同一 reference512 精度**：新式使用 reference P，旧式由 reference geometry_mass×exp(Tc) 再归一化，避免把 GH64 的积分误差混进两方法差距。该位置是概率加权的 **native key 中心均值**，不是完整连续 Y 均值或真实中心；固定“intended location”只是合成说明，不是新的效用门槛。
2. reference 同时保留以数值 Dhat 得到的尾部描述，使用上节 floor 解析下界作为 certified truncation gate；这只澄清已有 tail 要求，不放宽概率/参考收敛容差。

常 content、C0、T0、排列、平移、共同单位变换及同 content 的 fine/coarse 合并身份应完整保留。point/极限与无效 covariance 拒绝检查不需要新效果数据。25行必须全部完成并保存源/fixture/环境哈希、概率、raw积分、normalizer、时间及失败；概率1e−4、参考收敛1e−8、身份2e−12 等阈值按协议事前固定。

sharp peak、重复峰和纠偏案例是必要的最小压力说明，但即使某个预设位置距离变好，也不构成一般效用或真实定位证据。不得把 correction case 作为选择 h/T、积分阶数或后续候选的隐藏评分函数。此阶段没有科学精度/效率 gate；通过只说明所定数值预算在这些构造中通过，失败则停止该预算进入真实 TRAIN，并保留具体工程或数值原因。

最后，[R10 既有审查](round10_iteration_review.md) 已明确讨论条件行平均与先积分后归一化，以及 key-bank-conditioned query 表示；不能将归一化次序本身宣称为新发现。R18 可否成立，仍取决于以后另行冻结的同信息对照、旧开发门槛、实际 V、优化 GS 和未触碰确认。**本次条件 PASS 不授权这些实验，也不撤销 R17 科学失败。**
