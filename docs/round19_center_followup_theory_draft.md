# R19 之后：中心修正与真实值瓶颈

2026-10-31。**理论准备稿，未选择、未冻结、未执行。** 本文未打开任何 R19
新预测、标签或结果，也不改当前协议/producer。依据仅为既有
[R8 审查](/home/liwt/GRoPE/docs/round8_protocol_review.md)、
[R9 审查](/home/liwt/GRoPE/docs/round9_protocol_review.md)、
[R17 已验收证据](/home/liwt/GRoPE/docs/round17_validation_evidence_review.md)、
[中心语义审查](/home/liwt/GRoPE/docs/round18_post_r17_design_review.md)及
[R12 未验证条件值备选的审查](/home/liwt/GRoPE/docs/round12_value_hypothesis_review.md)。

**判断：即使 R19 通过，也没有充分依据立即投入完整 Ca 几何候选验证。**
它最多支持源中心身份含有旧区域矩丢失的定位信息；从这里到真实下游收益，
还缺少“被读取的合法源值确实提供额外可用信息”这一环。唯一值得优先考虑的
小实验，是固定旧预测路由的合法条件值诊断，见下文；不是另加一个中心方案。
若 R19 失败，停止该中心分支；同一个值诊断仍有独立的旧证据动机，不借失败
改选中心、调 h/T 或恢复已停止的旧候选。

## a、Ca 是否就是 3DGS 参数到 PE

`mean=a, Ca=C+(mu-a)(mu-a)^T` 确实仍是一个 Gaussian。给定任意已固定频率
omega，其位置编码可以写成 Gaussian 的特征函数：

    phi(a,Ca;omega) = exp(-omega^T Ca omega/2)
                     [cos(omega^T a), sin(omega^T a)].

这是“参数到编码”的明确函数；点极限回到普通 Fourier 位置编码，a=mu 时
回到旧 Gaussian 参数。此处只说明接口，没有选择频率、实现或提出新效果。
既有原生格积分乘内容的算子则直接定义位置质量/attention；写成 Gaussian
并不自动证明一个有限维 PE 在 learned Q/K/V 中有效。

Ca 是旧支撑关于指定 a 的二阶矩，不是其原中心协方差，也不是预测置信度。
固定均值 a 时，它有全平面 Gaussian 交叉熵闭合的解释，但不保证有限 FOV、
内容重加权、格 argmax 或真实区域标签更好。Ca-C 的秩一增量沿 mu-a 增宽；
若继续使用 trace 温度，还同时减弱内容系数。不能将两种变化的收益统称为
“中心修好了”，更不能假定增宽不会抵消定位好处。

更关键的是，a 来自旧 mu/C 之外的合法源中心身份。对原生 optimized 3DGS，
primitive 本来已有中心及关于它的协方差；直接使用它们时没有需要补入的
patch 均值偏移。a/Ca 首先是图像区域构造 Gaussian 的语义修正，不能把其
潜在成功包装成新的 GS 编码定理、真实 R/S 证据或一般下游成功。

## 路由改善到下游收益的链条尚未建立

R8 中，GT route 搭配原预测 V 相对同容量 singleview 的 AbsRel 只改善
**1.031%**，没有通过原 2% 开发门槛；GT value 与联合干预改善更大。
这支持源值质量/表示是重要限制，而不是只需更准确中心。GT 路由含可见性、
离散近似、fallback 与特定训练头，因此该数值不是任意合法路由的理论上限，
也不能证明之后所有 PE 无效。

R9 的真实 GS full footprint 相对最强简单 point 读取，gate 宏 MSE 仅改善
**0.1460%**；Tea 恶化 **1.7991%**，Waldo 改善 **1.2207%**，两个原门槛均
失败。它的中心已经是 primitive 中心，故 patch 的 a/Ca 修正没有直接修复链。
RGB alpha 混合、visibility 近似及 fallback 覆盖也限制解释，不能说所有真实
GS 都无效，或默认换 visibility 就会成功。

R17 保留方向性 C 有区域 KL 价值，却有定位和成本代价。这说明几何并非完全
无信息；但它不填补 R8 的值瓶颈，更不能抵消 R9、R17 或 R12 的停止结论。

## 唯一小实验建议：同一路由，保留单 Gaussian 的条件值

这是重新考虑既有 R12 备选的一个**合法路径最小否证**，不是新数学，也不是
已选候选。固定旧 TRAIN 48 对、原生 g8 query/key 和一个已封存的 R17 预测
路由 P；不依 R19 结果选行/中心，不重算或调优 P，不使用 GT 对应、GT 源值
或 GT 可见性。目标传感器深度只在全部新 forward 封存后用于评价。

每个 key 的原 RGB 预测三维点变换到 query 相机，形成 query 图像坐标
r=(u,v) 与 query-camera 深度 z 的**单个联合 Gaussian**。它同时保留位置
块 Crr 与交叉块 Czr，条件深度为

    beta_k = Czr_k (Crr_k + h^2 I)^(-1)
    v_ik = d_k + beta_k (r_i - m_k).

r_i 是已知 query 像素中心，不是 R19 的 a 或传感器投影。h 只借用该固定
路由已经锁定的值并统一坐标单位，不扫描；这项岭收缩是待否证假设，不能
称为已校准噪声。联合矩必须来自原合法像素/三维点，不能从只含二维矩的
缓存猜出 Czr。真正 GS 的三维均值/协方差也能经一阶投影产生这些块，但
透视线性化本身需要验证，不能据此声称已做真实 GS。

同一 P 下比较原常数 patch 深度与条件值；必须保留同信息的平面/逆深度
平面，以及 dense source-depth bilinear 读取。前三者检验同路由的值表示；
bilinear 是不同读取路由的强完整方法参考，不能冒充仅换 V 的消融。所有
方法共享合法输入、物理范围和预先冻结的失败处理；不能在评价时删去外推、
非前向或困难行。仿射条件深度本身就是一种岭平面回归，不因 Gaussian 写法
而有额外信息；透视平面在逆深度中仿射，必须正面比较。

标量线性读取时，每 key 只需三个固定通道
`(d_k-beta_k*m_k, beta_k_x, beta_k_y)`，先用 P 聚合再在 r_i 求值即可。
这不增加网络、混合分量或旁路，但不能与原 log/GELU value MLP 交换顺序；
因此只是线性值信息诊断，不是原 learned head 的等价优化，也不是任意 V 的
通用 PE。非线性 clip/fallback 的先后必须另冻，不能一边依赖逐 key 非线性，
一边声称仍有精确三通道压缩。

评价沿原深度任务的**全部 query sensor-valid 中心**，不能用 R19 的
cross-view center-visible 人口替代；cohort、标签身份和人数需另行绑定。
报告全部三场景 AbsRel/RMSE、配对误差变化、原 query 预测参考和完整成本，
不把确定性重复当训练种子。新数值推进条件须在任何新输出前另冻，本稿
不设置替代旧 gate 的阈值。

若条件值无稳定额外信息，或与更简单的同信息读取相当且无可信成本优势，
应停止这条表示假说；不能退回 Ca 扫描来补收益。若确有价值，也只授权考虑
一个重新声明的最小值表示训练协议，保留原强位置/单视图对照及全部旧 gate。
它既不复活 R12/R17，也不授权 fresh confirmation。当前尚无足够依据将
a/Ca 或该条件值构造称为已经成立的“3DGS 参数化到 PE”最终方案。
