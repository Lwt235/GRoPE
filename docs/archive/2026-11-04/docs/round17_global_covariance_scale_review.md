# 统一协方差倍率：未选择的备用假设

2026-10-30。写于 R17 固定旧 VAL 前向执行期间；没有读取该轮 VAL 效果，
没有运行新倍率、改变 R17 或启动下一候选。本记录仅整理已完成的代码检索
和代数推导，不能作为效果证据或执行协议。

## 不是首次想到协方差缩放

[R4 spectral 实现](../models/gaussian_spectral_pe.py) 已将两侧物理协方差
统一乘以 .25/.5/1/2，再加原 blur；其校准及取舍保存在
[R4 报告](../round4_report.md)。那是有学习编码器/温度的频域 Gaussian
滤波，不能直接作为当前 native-cell 积分的效果。只读检索未找到 R6–R17
单高斯原生积分或 trace coupling 校准统一倍率的已执行记录。
R13 OU 收缩保持总体协方差，不等价于直接把 C 乘以倍率。

补充文献核查：[GPSToken v1，§3.1–3.2](https://arxiv.org/html/2509.01109v1)
用二维高斯参数表示区域，并学习几何及纹理；其区域初始化标准差为宽/高
各除以6。它是图像 tokenization，不是本仓库条件对应或 R17 温度公式。
**下面是本记录的推断**：与均匀矩形的方差 w²/12 比较，这个初始化方差
w²/36 相当于倍率1/3，说明高斯“覆盖区域”的编码语义未必要求精确保留
区域二阶矩。该数值不成为本项目选定倍率，论文的重建/生成成绩也不能为
当前候选提供效果证据；直接将高斯参数嵌入 Transformer 本身已有先例。

## 最小形式及不同于 h 的地方

若以后考虑这一假设，最一致的单高斯形式是

    S_lambda = lambda C + h² I
    tau_lambda = T h² / (h² + lambda tr C)
    mass_k = integral over native cell k of N(x; mu, S_lambda)
    P = softmax(log(mass + original native area floor) + tau_lambda content)

lambda≥0 是所有场景、所有尺度共用的一个标量。它重新标定几何强度；
分布协方差已是 lambda C+h²I，不能仍称为保留原物理支撑 C 的矩闭合。
这也不是根据 C 估计了置信度。lambda=1 应回退 R17；lambda=0 应回退
同 h/T 原 Gaussian point。若仅改几何而继续用未缩放 C 计算 tau，后一
等价性不成立。这是两种不同候选，不能看结果后选择定义。

对 lambda>0，令 r=h²/lambda，有

    S_lambda = lambda (C+rI)
    tau_lambda = T r / (r+tr C).

r 决定相对形状与内容系数，lambda 还决定整体空间宽度，故与原 h 不完全
重参数化。在点密度近似里，去掉 query-only 常数后，几何能量为
`-(x-mu)^T(C+rI)^(-1)(x-mu)/(2 lambda)`。在 r 固定时，若改变 lambda 并
同时令 `lambda*T` 不变，其连续点分数仅差正的整体因子，argmax 不变，但 softmax
概率仍变。原生矩形积分不满足把 log mass 整体除以 lambda 的同一等价式；
不能以 logit 温度代替真实重新积分，也不能破坏原面积 floor 的一次加法。

连续线性内容 c(x)=g^T(x-mu) 的 mode 位移仍是
`d=tau_lambda S_lambda g`；lambda≥0 给出
`||d|| <= T h² ||g||`。这沿用 R17 的连续点密度范围，不能推广为原生
cell argmax、PCK、真实区域 KL 或下游改善保证。

## 公平比较与否证条件

若未来确有理由推进，必须先冻结唯一设计与 TRAIN 选择预算，而不是在
当前 VAL 上扫描 lambda。最少保留：

- lambda=1 的原固定 trace/full 以及 lambda=0 的充分校准 point。
- fixed、isotropic 的合理同等校准机会；point 的重复 lambda 不算新算子。
- 同 lambda/h/T 的 constant-T full，以隔离几何缩放与内容抑制。
- matched isotropic 使用 primary 原 scaled full C 计算的同一 tau；只改几何。
- 原强 quad4/quad16 和所有旧门槛。是否给 quadrature 同一支撑倍率，须事前
  明确：若给，需令 `mu_s'=mu+sqrt(lambda)*(mu_s-mu)`、
  `C_s'=lambda*C_s`，权重不变；不能只改局部方差便宣称比较同一缩放的支撑。

先审计 lambda=0/1 极限、共同单位变换、质量可加性、有限/PSD 与极窄 tail；
lambda=1 的代数相等不自动意味着执行精度/算术顺序逐字节相同。再做事前
冻结的完整 TRAIN 区域与定位联合筛选。若只在 lambda 接近0时改善，或
fixed/isotropic 获得同样收益，应承认额外 shape 的证据有限；若需要分场景
或分尺度倍率，则当前单一全局假设失败。任何通过仍须旧 VAL、实际 V、
真实 GS 和未触碰确认，不能用这里的代数审查替代。

目前没有选择 lambda 值或搜索网格，也没有认定该方向优于 R17、积分内容
或其他备用设计。下一步取决于完整 R17 结果揭示的具体限制。
