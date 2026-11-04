# R18：在高斯位置内部归一化，再对区域质量积分

2026-10-31。这是独立的理论候选审查，**不是已验证方案、真实数据协议或新结果**。
本稿读取了已经完整验收的 R17 VAL 科学失败，并阅读 R7/R8/R9 实际 V/GS
报告、R16 机制报告及 R18 两个既有备用方向；没有运行实验、改冻结算子、
读取 fresh 输入或按新效果选参数。

根验收为 [`round17_validation_root_acceptance_20261031.json`](../results/round17_validation_root_acceptance_20261031.json)：
证据链与实现审计通过，科学门槛失败，原固定候选停止。此后至多允许另立
开发假说，不能改写原候选、原门槛或失败人口。

## 1. 推荐只筛查一个不同机制

考虑**几何条件下的局部 attention 期望**：Gaussian 表示区域中质量的位置
分布，每个潜在位置先形成一个归一化对应分布，再按同一个区域测度相加。
它改变归一化与区域混合的顺序，不改变径向核、不扫描另一条 τ(C) 曲线，
也不引入 GT 中心、point winner、额外外观交叉矩或学习 gate。

这与现有 R18 草稿中“固定内容场，先乘再积分”的方向不同：后者仍只在
全局积分后归一化。这里局部 normalizer 留在被积函数内部，因此不是把
一个常数从积分前搬到积分后，也不是现有混合质量的另一种写法。

**设计变化必须保留：**最初口头候选是裸的 `E_N[softmax(local score)]`。
它在有限 FOV 不保持原 T=0 先验，故不推荐。主线程随后只选择了下述
几何 FOV 条件版本用于最小合成否证；这是实验前明确的定义变化，不能说
裸期望本来就与下式等价。本文不选择第二个并行候选。

## 2. 唯一建议定义：先固定几何可见质量，再局部匹配

原生矩形 cells 为 B_j，组成同一个有限视野 F；位置坐标、h、C 使用同一
reference-8 单位。令 `ν_i=N(μ_i,C_i)` 是投影区域的 Gaussian 测度；C 可
退化，C=0 时是 Dirac。h>0 是已有局部位置宽度，不是新的支撑倍率。

每个潜在位置 x 的原生质量为

\[
m_j(x)=\int_{B_j}\mathcal N(y;x,h^2I)\,dy,\qquad
a_j(x)=m_j(x)+\epsilon |B_j|,
\]

\[
A(x)=\sum_j a_j(x),\qquad Z_g=E_{\nu_i}[A(X)].
\]

ε 沿用原来的**同一个 native-area floor 密度**，只需按参考单位换算；
不新增 floor 大小、不在最终 P 上再加 floor。给定当前合法 pooled Q/K，
`c_ij=q_iᵀk_j`，定义

\[
Z_T(x)=\sum_l a_l(x)e^{Tc_{il}},\qquad
\alpha_{ij}(x)=\frac{a_j(x)e^{Tc_{ij}}}{Z_T(x)},
\]

\[
\pi_{i,0}(dx)=\frac{A(x)\,\nu_i(dx)}{Z_g},\qquad
\boxed{P_{ij}=E_{\pi_{i,0}}[\alpha_{ij}(X)]
=\frac{E_{\nu_i}[A(X)a_j(X)e^{Tc_{ij}}/Z_T(X)]}{E_{\nu_i}[A(X)]}.}
\]

这是一套条件测度，不是两个独立 readout。所有任务仍读取这个 P 的原生
argmax 或 `P @ V`。π0 中 A 是由预测几何、原生 key 边界及 h 决定的
FOV 质量，**不是 sensor 可见性、遮挡真值、valid-label 或内容置信度**。
全部原 query 仍进入原评测人口，没有按 A 或预测质量丢弃 query。

π0 也不是 `N(μ,C)` 硬截在 `X∈F`：F 外的潜在位置仍可通过 h 的局部核
把质量送入 F。它相当于先按同一个几何模型条件在“局部目标落入 F”事件上，
之后由内容决定该局部质量在 key 之间如何分配。ε>0 可视为同一个有限域
背景测度，不能扩展成整个无限平面上的均匀概率密度。

### 两项严格回退

当 T=0，`Z_T=A`，所以

\[
P_{ij}=\frac{E[m_j(X)]+\epsilon |B_j|}
{E[\sum_l m_l(X)]+\epsilon |F|}
=\frac{\int_{B_j}\mathcal N(y;\mu_i,C_i+h^2I)dy+\epsilon |B_j|}
{\int_F\mathcal N(y;\mu_i,C_i+h^2I)dy+\epsilon |F|}.
\]

这与现有 full Gaussian 的**连续数学先验**完全相同，包括有限 FOV 与
一次面积 floor。有限 cubature 对这一恒等式仍有近似误差，必须实测，
不能凭解析等式宣布数值逐位相等。

当 C=0，π0 也是同一个点 μ，因而 `P=softmax(log a(μ)+T c)`，精确退回
已有 point 原生 cell 算子。无内容的 Gaussian 支撑没有被换成 point。

## 3. 现有混合后归一化究竟做了什么

现有同一 pooled 内容的 full/quad 混合形式是

\[
P^{old}_{ij}=\frac{e^{Tc_{ij}}E_{\nu_i}[a_j(X)]}
{E_{\nu_i}[Z_T(X)]}
=E_{\pi_{i,T}}[\alpha_{ij}(X)],\quad
\pi_{i,T}(dx)=\frac{Z_T(x)\nu_i(dx)}{E[Z_T(X)]}.
\]

新式固定几何条件 law π0，旧式使用内容证据重加权的 law πT。若 C 表示
“哪一个位置才是真位置”的不确定度，更新潜在位置的证据可能合理；若 C
表示一个区域**同时具有的空间质量**，同一大片区域不应仅因某一位置更
像 pooled 内容就获得额外采样质量。这是待检验的语义取舍，不是普遍概率
正确性的定理，更不是把所有旧方法称为计算有误。

令 `b(x)=Z_T(x)/A(x)`。则

\[
P^{old}_{ij}=\frac{E_{\pi_0}[b(X)\alpha_{ij}(X)]}{E_{\pi_0}[b(X)]},
\quad P^{old}_{ij}-P_{ij}
=\frac{\operatorname{Cov}_{\pi_0}(b(X),\alpha_{ij}(X))}{E_{\pi_0}[b(X)]}.
\]

该恒等式给出直接否证量：若局部内容证据 b 几乎恒定，两种算子应一致；
若两者差异明显，必须能由这个协方差项解释，不能归因于新增几何信息。

### 与已有 quad16 的差别不在节点名称

对固定正权节点 `(x_l,w_l)`，先给每节点加同一 `ε|B_j|`；Σw_l=1，故
混合后仍只有一次 floor。记 `A_l=Σ_j a_lj`，则

\[
\widetilde w_l=\frac{w_lA_l}{\sum_r w_rA_r},\quad
P^{new}_j=\sum_l\widetilde w_l\operatorname{softmax}_j(\log a_{lj}+Tc_j).
\]

已有先混合后 softmax 为

\[
P^{old}_j=\operatorname{softmax}_j\left(\log\sum_l w_la_{lj}+Tc_j\right).
\]

两式一般不等。仅“把 full 换成更多节点”而仍在混合后 softmax，不测试
这个机制。反之，用同一节点/权重对两式做配对比较，能隔离归一化顺序。
真实 quad16 的节点来自实际分块位置与 pixel covariance，并不等于单个
Gaussian 的 quadrature；必须另保留它作为更丰富几何的强控制，不能说新
算子已经拥有其高阶信息。

## 4. 有限 FOV 下必须拒绝的裸期望

裸算子 `E_ν[α(X)]` 在 T=0 等于 `E[m_j(X)/m_F(X)]`（暂忽略 ε），
而旧先验为 `E[m_j]/E[m_F]`。前者把每个潜在位置都强制归一化为一个完整
FOV 分布；远在 FOV 外的位置也可能贡献完整边界质量。二者差异不是实现
误差，而是不同 law。不得默认“区域 Gaussian 在图像内”，再只保留方便样本。

一个仅用于说明归一化顺序的离散反例：两个等权潜在位置的原始 cell 质量
分别为 `a(x1)=(.8,.1)`、`a(x2)=(.0001,.0009)`。T=0 时旧先验及几何条件
新式都给第一格 `0.8001/0.901≈.8880`；裸期望则约
`(.8/.9+.0001/.001)/2≈.4944`。这组数不是声称由某个特定 Gaussian
场景产生的效果数据，只是说明省略 A 会改变先验。真正 Gaussian 的边界、
外部均值与 tiny-FOV 情况必须纳入合成审计。

## 5. 新机制的一条精确推导，以及它没有保证的内容

仅在无限平面、ε=0、连续 key 内容场严格线性
`c(y)=c0+gᵀy` 时，A=1，局部条件分布配方为

\[
\frac{\mathcal N(y;x,h^2I)e^{Tc(y)}}{\int\mathcal N(z;x,h^2I)e^{Tc(z)}dz}
=\mathcal N(y;x+Th^2g,h^2I).
\]

对 X 积分后

\[
P^{new}(dy)=\mathcal N(y;\mu+Th^2g,C+h^2I)dy.
\]

相反，现有 constant-T 乘法得到
`N(μ+T(C+h²I)g,C+h²I)`。新式的线性内容位移不再被区域 C 放大，且
不需要按 trace 猜一个标量补偿。它仍保留整个 C 的形状，不把大区域直接
收缩成中心点。这是本稿唯一推荐筛查的新机制。

这**不是**当前真实 native cell 分布的闭式：实际 cosine 逐 key 常值、有限
FOV、有 floor，并不满足连续线性场条件。一般内容下输出不再是 Gaussian，
可有多个模式；不保证 mode=μ，不保证 region KL 或 PCK，更不保证 native
cell argmax。trace τ 的已有模式位移界也不能直接换名迁移到这里。

此外，单个 μ/C 仍不能辨别相同二阶矩但不同多表面的真分布，也没有补回
源中心像素身份。若 q24→g8 的主要缺失是这些信息，此算子可以失败。若
给定内容就是正确位置的强证据，禁止它更新潜在位置可能反而妨碍纠错。

## 6. 最小可证伪工作，而非新的效果搜索

主线程已选择**仅合成否证**，不启动真实 TRAIN/VAL、实际 V 或 GS；这里
列必须检查的数学内容，不代替随后单独冻结的具体数字/容差/病例清单。

1. **同源先验与退化：**T=0 时与独立解析卷积+native-CDF+一次 floor 对比；
   C=0 时与原 point 对比；常值内容时保持 T0；每行非负、归一及精确面积
   floor 拆并。不得在最后加第二次 floor 或用同一个近似结果自证相等。
2. **法则辨别：**同样 Gaussian 节点、native masses、内容，在混合后归一化
   与固定几何条件 law 的局部归一化之间比较，并核对上面的 covariance
   差分恒等式；证明实际实现测试的是顺序，而非更多位置或不同内容信息。
3. **线性机制：**连续全平面线性场只用来检验推导；native 16 cells 的
   分段内容另行数值计算，不把离散结果强行要求为上述 Gaussian。
4. **非保护反例：**固定的局部尖峰、重复纹理/双峰、边界与 outside mean、
   薄/宽/旋转 C 都报告。至少保留一个内容应合理纠错的例子，不能只构造
   “内容有害”的环境让冻结 π0 获胜。只报告预声明的全部病例，不搜索最优 T。
5. **数值可行性：**若固定 GH 主阶、GH 高阶与独立积分未满足预先误差界，
   保留数值失败，不自动增阶/换病例后称原设计通过；不能跳过 tiny-FOV。
   合成通过也只是证明定义与数值机制可用，不是科学 gate 或真实效果通过。

固定内容场且 splitting 后子 cell 内容相同，a、A、Z 都满足可加性，所以
新式的原生质量拆并一致。若拆并时重新 pooling/normalize key feature，
内容场改变，两种算子均无此保证。沿用当前 pooled 特征的实际流程必须
承认这个边界；不将“同一测度”写成对任意 tokenization 自动不变。

## 7. 简洁结构不等于低成本或标准五通道 PE

inner 核是轴对齐的 h²I，矩形质量可分离为两轴 normal-CDF 差的乘积，
不必再为每个 latent node 调 GL32 二维积分。右尾须用 survival/erfc 差，
否则 `1−1` 消减会把 tiny mass 写成零。FOV 是矩形 cells 的并集时 A 可
按同一 cells 求和；规则完整矩形 F 也可独立用 F 的 CDF 检查。

outer 仍含 `1/Z_T(X)`，一般没有 Gaussian 闭式。如果用 L 个节点，最少
需要 O(L Nq Nk) 个局部质量与归一化，内容矩阵本身可只算一次；值汇聚
可先流式相加 P 再做一次 P@V，或逐节点累加，二者需独立验证。不能说
“一个公式、没有网络”就免费压缩：准备节点、完整 C、FOV 权重、稳定 CDF、
normalizers、outer 积分、Q/K/V 全部计入成本。不得让新方法复用旧 primary
准备，却让控制重算；不能只报缓存后矩阵乘法。

它没有一般的有限五维独立 Q/K 分解：local normalizer 依赖全部 key 内容，
外层再相加也不是一遍普通 QK softmax。可称为 Gaussian 支撑驱动的一个
attention 算子，不能叫“原五通道 PE 的无代价替代”。若用户最终要求严格
一次标准 QK attention，这一点可能足以结束该方向。

### 独立积分范围是实际问题

标准正态 outer 坐标上的 `[-9,9]^2` 截断，遗漏概率约小于
`4 Φ(−9)≈4.6×10⁻¹⁹`。被积 numerator 至多 A≤1+ε|F|，但最后还除以
`Z_g=E[A]`；tiny-FOV/floor-dominated 时 Zg 可能接近 10⁻²⁸，故不能
把这个绝对尾界当作所有 case 的概率误差保证。必须报告 Zg、原始积分及
tail_bound/Zg，或在测试前选择更足够的域/显式尾处理。扩大固定范围会在
同阶 GL 下使节点更稀疏，也不是免费修复。

宽 C、窄 h 时，局部 cell 边界在标准化 outer 坐标里可以很尖；GH32/64
也可能一起欠分辨，且二者恰好接近并不证明正确。独立分片 GL 或基于已知
边界的变换可能更可靠，但节点/子区数量及成本必须先声明，不按分数选择。
支持层面全局旋转/尺度或 eigenbasis 的等价性还须在有限 cubature 层单独
审计；只证明 Gaussian law 无 gauge 歧义不够。

## 8. 与已有下游失败及文献的边界

R7 的实际 V 深度独立确认未过主门槛；R8 真路由配原源值相对 singleview
只改善 1.031%，且源值质量诊断更强。新归一化顺序不会创造更准源深度。
R9 实际 optimized GS/RGB 的既有失败也不能由一个区域分布推导注销。
即使此算子在旧区域开发集通过，仍需要原有实际 V、真实 GS、成本及最后
未使用人口的完整证据。本文不推荐绕过这些问题或更换更容易的任务。

已有 [Expected Attention, 2025, Eq.7–8](https://arxiv.org/html/2510.00636)
在 Gaussian future-query 模型下先求指数分数期望、再归一化；不是本稿
固定几何 law 下的归一化 attention 期望。它说明这个顺序区别应与已有
方法明确比较，不能把“expected attention”当新术语。

[Zheng、Wang、Kong, ICML 2022](https://proceedings.mlr.press/v162/zheng22b.html)
及[作者推导](https://hkunlp.github.io/blog/2022/lara/)已将 attention 写成
输入依赖分布下的 normalized random-feature 期望。其 mixing law 包含
query/key 与注意力权重；它不是本稿的 geometry-only π0，因此不能仅凭
“Gaussian mixture + expectation”认定相同，也不能据此宣称本稿首次。

[Zhang、Feng, EMNLP 2021](https://aclanthology.org/2021.findings-emnlp.121.pdf)
已使用 Gaussian mixture 集中注意力与内容注意力融合；它并不为这里的
finite-FOV 条件 law 或新机制提供效果证据。以上仅为一手来源边界检查，
不是完整 novelty 检索；局部归一化、条件测度与 Gaussian 混合均不是新数学。

**推荐结论：**仅把“固定几何质量 law、先局部归一再积分”当作一个可以
被快速否证的结构选择。其价值在于改变了可精确推导的内容–区域质量耦合，
同时保持 μ/C、相同可用信息和一个原生分布；其主要风险是数值/成本、
内容不能纠正几何、单 Gaussian 信息不足，以及真实 V/GS 瓶颈完全未修复。
在这些检查完成前，不新增径向 family、不叠加 τ/anchor/gate、不宣称有效。
