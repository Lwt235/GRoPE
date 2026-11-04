# R17 的 GS 颜色内容适配：来源与理论可行性审查

2026-10-30。只读审查 R9 协议、训练/读色/评价源码及既有报告；没有加载
context 数组、checkpoint、target RGB 或当前 R17 VAL 产物，没有运行预测、
测试、调参或冻结新协议。本文补充
[downstream 准备](round17_downstream_implementation_plan.md)，不改变任何已冻结方案。

**结论：可以准备一个来源合法、没有新增网络参数的独立 GS 适配协议；
没有证据可据此推荐实际执行或宣称 R17 特征迁移成功。** 它是以既有 GS
颜色为引导的空间读色，有可能只是近似恢复原 GS 颜色。trace 抑制在这个
内容语义下既可能有利，也可能削弱最需要的颜色边界保护。上游完整验证
及实际 V 的要求保持不变；R9 旧分支的失败也不被撤销。

## 1. R9 能提供合法 prior，但不是直接读取 SH 系数

R9 实际使用的是每 primitive 三个 `color_logits`，渲染时取
`sigmoid(color_logits)`，并传入 `sh_degree=None`。这是 **SH0 容量的
view-independent RGB**，没有可再次转换的 SH0 系数。合法的 prior 是
同一 context-only、固定 3,000 步最终 checkpoint 的 sigmoid RGB；导出的
`gaussians.npz.colors` 也是这一数值。不能用 logits 直接比较颜色，不能
把已经解码的 RGB 再套 SH-to-RGB 变换，也不能改用旧全图初始化的 GS。
来源见 [renderer](/home/liwt/GRoPE/experiments/round9_gs_readout.py:61)、
[checkpoint 导出](/home/liwt/GRoPE/experiments/round9_gs_readout.py:200)。

颜色范围及截断政策有明确来源：

- 初始化颜色先裁到 [.001,.999] 再取 logit；训练中 color logits 裁到
  [−10,10]。prior 的 sigmoid 值因此位于 (0,1)，无需另加 RGB 裁断。
- source RGB 是原 context 图像经 `INTER_AREA` 缩到 192×144 后的
  float32 `/255`，保持 sRGB，没有 linear-light、曝光、白平衡或 Lab 变换。
  见 [context 导出](/home/liwt/GRoPE/experiments/round9_context_data.py:184)。
- 新适配应检查有限性和范围；不以静默 clip 修补来源错误。原始 prior、
  original-GS 及无源 fallback 需保持同一数值。旧评价直接计算完整 RGB
  MSE，没有在评分前裁预测或做曝光对齐，见
  [评价函数](/home/liwt/GRoPE/experiments/round9_analysis.py:145)。

24 张 context 同时用于 SIFT/三角化、GS 拟合和 eligible-source 读色。
它们与四张 calibration、四张 gate 图像不重合；但 prior 已拟合了这些
source，不能称 prior 和 source 为独立观测证据。相机仍来自提供的全图
SfM 标定，不能升级为全流程 target 隔离。每个 target 摄像机只接收固定
primitive 颜色的渲染结果，不给 prior 或读色函数提供 target 像素。
划分和范围以 [R9 协议](/home/liwt/GRoPE/ROUND9_PROTOCOL.md) 为准。

## 2. 单次读色的明确数学语义

记 primitive prior 为 rᵢ∈[0,1]³，source s 的观测像素为 vₛⱼ∈[0,1]³，
投影位置/物理支撑为 μᵢₛ、Cᵢₛ，S=C+h²I。对原生像素 Bⱼ 定义

\[
 d_{isj}=\frac{\|r_i-v_{sj}\|_2^2}{3}\in[0,1],\quad
 c_{isj}=-d_{isj},\quad
 \tau_{is}=\frac{Th^2}{h^2+\operatorname{tr}C_{is}},
\]
\[
 m_{isj}=\int_{B_j}\mathcal N(x;\mu_{is},S_{is})dx+\epsilon|B_j|,
 \qquad
 a_{isj}=\frac{m_{isj}e^{-\tau_{is}d_{isj}}}
 {\sum_km_{isk}e^{-\tau_{is}d_{isk}}},\qquad
 y_{is}=\sum_j a_{isj}v_{sj}.
\]

每 source 独立归一化，再按原 eligible-source 集合等权平均 yᵢₛ；无
eligible source 时精确保留 rᵢ。只有这一次 source-RGB 加权读出，不迭代
更新 prior，不另加 rᵢ 与输出的插值门，不改 geometry、opacity 或 renderer。
由于颜色在每个原生像素内按常值定义，乘法和该像素内积分可以交换；这
不等于对粗像素重新 pooled RGB 后还具有相同的内容积分语义。

所有像素仍参与；memory chunking 不成为空间截断。source 资格继续仅由
原 GS 的几何、opacity、alpha/expected-depth 规则确定，不能根据颜色差
删除视图或像素。源视角的 C、τ 和归一化因子可以不同，故保持逐 source
归一化及等权融合。不能把新的归一化常数当成跨视图 confidence。

这属于位置权重与颜色 range 权重相乘的**bilateral 类读出**，不是新的
Gaussian 核原理；经典的域/颜色双权重形式见
[Tomasi–Manduchi 1998 原文，式 5–6](https://users.soe.ucsc.edu/~manduchi/Papers/ICCV98.pdf)。
这里的具体区别是由 projected GS shape 决定空间质量、由 context-fitted
primitive 颜色提供引导，以及事先固定的 trace–temperature 耦合。

它与旧 cosine 内容并不等价：

\[
 c_{isj}=\frac{2r_i^Tv_{sj}-\|v_{sj}\|^2-\|r_i\|^2}{3}.
\]

只有最后一项是当前 source 内的 query 行常数；key RGB 的范数项不能删除。
这里没有 unit normalization，亮度本来就是颜色距离的一部分。旧 cosine
的区间为 [−1,1]，新 c 为 [−1,0]，梯度和语义也不同；使用同数值 T 不等于
同内容强度或同一学习特征的迁移。本审查不选择、扫描或校准新的 h/T。

native pixel mass 也不是 R9 的 `C+I/12+h²I` 点式 EWA：真实像素积分不能
再重复添加 I/12。R9 的 h 以 pixel 为单位，R17 用 reference8；必须同时
转换 μ/8、C/64、h/8。renderer eps2d=.3、旧 readout 的 1e−10 pixel²
数值 floor 不得解释为物理 support confidence。若后续选择哪种固定单位
转移/校准策略，须在独立协议中事先说明，本稿不替其作决定。

## 3. 为什么它可能只是 prior 回拉

固定 source 和空间质量 m，令 `Z(τ)=Σⱼmⱼexp(−τdⱼ)`。精确地有

\[
 \frac{d}{d\tau}E_{a(\tau)}[d]=-\operatorname{Var}_{a(\tau)}(d)\le0.
\]

因此增强内容系数必然降低**被加权的 source 颜色到 prior 的平均平方差**。
这不是到目标颜色或 target RGB 的误差下降保证。分解

`3E[d]=||y−r||²+E[||v−y||²]`

还说明上述单调性不能直接改写为输出 y 到 prior 的距离严格单调。τ 很大
时，权重趋向最接近 prior 的 source 颜色，ties 仍由空间质量分配；prior
未在 source 值中出现时，不保证精确复制 r。τ=0 回退同 native 积分的
纯几何读色；空间与颜色共同常值时，新机制完全不活跃。

只读色输出属于 eligible source RGB 的凸包；它不能发明凸包外的颜色，
但原 prior/fallback 可以位于这个凸包外。原优化颜色可能已经是固定几何、
透明度和黑背景下的有效补偿值，而非单一表面真实颜色。source 像素又可能
混合多层 alpha，故两者即便在同一 sRGB 坐标内也不必是相同物理量。
颜色更接近 prior 可以只是撤销 R9 纯几何读色的损伤。

由此 original-GS 是不可删的系统对照。将新读色与 original-GS 的颜色/
渲染差异、T=0 的增量以及 fallback 贡献保存下来，可以说明机制是否活跃；
但小差异不是独立收益，不能用“更像原 GS”替代完整 target MSE。

## 4. trace 抑制不自动符合这个任务

颜色权重也可写为 `exp[−||r−v||²/(2σ_rgb²)]`，其中

\[
 \sigma_{rgb}^2=\frac{3}{2\tau}
 =\frac{3(h^2+\operatorname{tr}C)}{2Th^2},\quad T>0.
\]

大 support 会放宽颜色接受范围。在一个准确 prior、但空间支撑跨越颜色
边界的 primitive 上，这可能**增加混色**；它与“不相信宽支撑的内容”是
两种不同解释，后者并没有颜色噪声/置信度证据支持。

一个不运行任何数据的解析反例足够说明方向不确定。两个像素具有相等
空间质量、颜色分别为黑和白，prior 为 (.25,.25,.25)。每个输出通道为
`y(τ)=1/[1+exp(τ/2)]`。降低 τ 会增加 y：若希望恢复黑色，误差变大；
若希望恢复白色，误差变小。同一 C 和 prior 下都可能出现这两类真实情况。
这只是单 primitive 的颜色反例，不代替 alpha 渲染任务的完整门槛。

其他无法由 trace 修复的问题包括跨 source 曝光/白平衡差、view-dependent
反射、饱和、阴影、遮挡混色及几何误配。sRGB 的欧氏距离是本任务明确的
数值定义，不是经过校准的线性辐射或感知距离。添加曝光修正、特征编码器、
置信度、视角颜色头或重新训练 GS 都会改变本次最小适配，不能作为默认补丁。

## 5. 可以准备协议的条件与不能省略的对照

来源上没有缺失的合法 query descriptor：R9 final sigmoid RGB 就能提供。
空间/颜色权重有界、一次归一化读色可定义，额外工作只是颜色差和已有
trace 算术，不需要新增训练参数。因此**有条件准备独立协议是可行的**。
但严格的 claimed result 应是“固定 GS 系统的颜色引导空间读出”，而非
“原 learned-content PE 原样迁移”，更不是新 confidence estimator。

最少的角色语义应明确区分：

| 角色 | 空间支撑 | 内容系数/目的 |
|---|---|---|
| trace full | 真实投影 C | 原 trace τ；待评估的耦合 |
| constant-T full | 同一 C、同一 h | 常数 T；隔离 trace 耦合 |
| trace isotropic matched | trace(C)/2 I | 使用原 full C 算出的同一 τ；只隔离方向性 |
| native point | 0 | C=0 的 τ=T；保留强中心对照 |
| native fixed | 预声明固定支撑 | 保留常数 T 的强控制，不自动替它降低内容强度 |
| bilinear / original-GS | 原实现 | 完整保留独立采样参考和既有系统表现 |

所有用到颜色内容的角色都获得同一个 prior/source RGB、同一个打分定义、
相同 source 资格与参数预算。旧已校准强控制和纯几何/T0 参考不能被上述
匹配控制替换；native/full 与旧 EWA 差异也须单列。若采用新的独立校准，
须相同预算和事前规则，不能只调 primary；本稿没有提出新的搜索范围。

全部角色保持相同固定 GS geometry/opacity/renderer、原 fallback 及完整
图像人口，fallback 逐位保留 prior；报告 alpha-composited fallback
贡献，不只报 primitive 数量。原 R9 的两场景等权全像素 MSE、至少 2%、
两场景同向和最强 point/fixed/isotropic/bilinear envelope，以及单独
original-GS 系统对比都保留。只胜过较差的重取色而不能可靠超过原 GS，
最多是限定机制结果。

即便重用旧 context-fitted checkpoint，也须绑定其来源、最终训练状态和
依赖；重用不是新的 seed 或独立确认。目标预测全部生成后才可按新协议
打开 calibration，冻结选择再打开 gate。R9 目标早已作为旧开发评估使用，
新结果不能升级成未见数据确认。上游若失败，不因这个合法内容定义而绕过
既定停止规则。

当前到此为止：可明确合法输入与候选语义，尚不足以推荐收益；不冻结、
不调 h/T、不打开 target RGB、不修改 R17，也不为失败另加模块。
