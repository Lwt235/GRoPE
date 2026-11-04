# Round17：单 Gaussian 的 trace 内容系数

2026-10-30。R16 完整旧 TRAIN 机制诊断及独立审计之后，root 选择本文件唯一
候选；不同时比较 λmax、det 或新径向密度。R17 尚未产生真实效果，本文件
不声称通过任何旧科学门槛。完整历史依据和文献核查见
[R16 设计审查](round16_trace_temperature_review.md)。

令 C⪰0 为预测 query support 的协方差，S=C+h²I，native key 区域为 B_j。
唯一改动是内容系数

\[
 \beta=\frac{h^2}{h^2+\operatorname{tr}C},\qquad
 \tau=T\beta,\qquad
 P_j\propto\left(\int_{B_j}\mathcal N(x;\mu,S)dx+\epsilon|B_j|\right)e^{\tau c_j}.
\]

几何、μ、完整 C、native rectangle 积分、面积 floor、FOV、内容表示和
原生 argmax 不变。实际计算顺序锁定为 C/h/T 先转 float64，令
`s=max(h²,C00,C11), r=h²/s, c=C00/s+C11/s, τ=T*(r/(r+c))`，
防止有限大 C 的 trace 中间值溢出；再将 τ
转回 C dtype；内容路径用 tau.float32()*content32，再转 double 加 geometry64。
因此 C=0 回退原 point 的 T*content32，T=0 回退原 full Gaussian softmax。
所有坐标和 h 同用 reference8 单位，C 用对应平方单位。

## 连续性质与其适用范围

对连续点 Gaussian log-density 加线性内容 gᵀ(x−μ)，唯一 mode 位移为
d=τSg。因为 λmax(S)≤h²+tr C，有

\[
 \|d\|_2\le T h^2\|g\|_2.
\]

这是同 h/T 原 point 的线性位移上界。对可微内容若全域 ||∇c||≤G，任意
内点驻点满足同一个以 G 替代 ||g|| 的界。若内容只有有限振幅 R，连续点
全局最大值与 μ 比较给出更弱的 ||d||≤h√(2TR)。任意不受限制的内容峰可
压过任何有限几何代价；公式不能提供无条件 PCK 保证。

trace 对 C 光滑、无需 eig，并对旋转和共同单位缩放不变。二维相对谱版本
h²/λmax(S) 的系数比为
`(h²+λmax C)/(h²+λmax C+λmin C)∈[1/2,1]`。
rank1 时相同，大各向同性 support 时保守近一倍；d 维最坏保守 d 倍。
这是事前结构取舍，不是效果选优。det 版本最坏位移多 √κ(S)，trace/2
版本不能保证同一个原 point 上界，二者不进入 R17 候选搜索。

## 原生积分与离散读取的边界

移动同形窗口时 m(y)=∫_B N(y+u;μ,S)du，有
∇log m=−S⁻¹(y−μ+E[u|y])，Hessian=−S⁻¹+S⁻¹Cov(u|y)S⁻¹。
因此 native 积分的曲率小于点势；连续窗口驻点只能直接保证
||y−μ||≤Th²G+r_B，r_B 为窗口半对角。

明确反例：C=0，h=.25，单位 native 方格。沿一轴移动窗口的中心负
log 曲率为 3.62013914，而点 Gaussian 是16；弱线性内容的窗口 mode
位移斜率比 Th² 大4.41972倍。g8 的 r_B 本身约5.657px，且 μ 不是 sensor
中心。故不能把连续点势上界推广成离散 argmax、PCK@8 或遮挡条件下保证。

实际 native logits g_j=log(m_j+floor)，j0=first argmax(g) 时，任何新赢家 j
满足 `g_j0−g_j≤τ(c_j−c_j0)≤τ osc(c)`。这是含 floor 的精确离散条件。
在精确算术中降低 τ 不会增加相对 T0 的 flip 或所选 key 的几何 penalty；
PCK、Euclidean 距离和 GT KL 则没有单调改善保证。float32 在近 tie 处会
扰动包络边界，因此数值审计保留原 first-index、全部 argmax 差与 gap，
不用“赢家没变”豁免概率误差，也不把 exact tie 差异伪装成新 gate。

## 不具备的性质

trace 是 support 总方差，不自动等于估计不可信度，不能称为已校准的
confidence。它会抑制大而准确的区域内有用内容，也会撤销原内容带来的定位
修正。光滑不等于整个神经网络 Lipschitz；若学习 C，还需检查这条梯度路径。

真实 native key 质量及面积 floor 可加；pooled key 内容改变后的完整注意力
不无条件可加。query 合并的协方差多出 between-scatter，因此
β_parent≤Σw β_child；query 拆并仍不闭合，旧 common9 分区门槛不得豁免。
单 Gaussian 仍不能由一二矩恢复多峰支撑、遮挡或真实中心位置。

区域 KL 关于每个 query 的 τ 凸，但各 query 使用不同 β，不能保证其等场景
aggregate 落在旧 T0/T5 aggregate 之间。必须保留旧 full1 T0、固定 T、强
point/quad 及匹配 h/T 控制。若新公式只是重现 T0，或必须削弱强控制才有
优势，就没有验证其新增研究价值。

## 研究定位

Gaussian support 的位置编码已见
[mip-NeRF](https://arxiv.org/abs/2103.13415)，非各向同性 Gaussian 注意力与
二次 PE 已见 [Cordonnier 等 Appendix E](https://arxiv.org/html/1911.03584v2)，
位置 prior 与内容乘积的概率解释也见 [BAM](https://arxiv.org/abs/2505.22842)。
不能把 μ/C→PE、Gaussian bias 或参数无关归一化本身声称为新颖贡献。
R17 待验证的是支撑总方差与内容响应之间这一固定耦合，是否能在既定分区、
尺度、扰动、效率和后续真实数据关口中带来可信收益。它仍是新的候选，
不能继承旧实现测试之外的任何科学通过结论。
