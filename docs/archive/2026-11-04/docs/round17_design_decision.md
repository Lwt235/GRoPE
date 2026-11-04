# Round17：固定一个支撑相关的内容缩放规则

2026-10-30。此设计在完整 R15 训练结果及已审计 R16 诊断之后提出，在任何新
R17 效果之前声明。它是下一项待否证的开发假设，不是已验证方案。执行阶段与
止损条件见 [R17 候选协议](/home/liwt/GRoPE/ROUND17_CANDIDATE_PROTOCOL.md)。

R16 的具体启发是：相同 μ、内容和 h 下，full 核的大区域原生格几何间隔明显
小于 point；h=.5 的 q24 full 在 T0 的定位高于同 μ 点核约 1.012pp，T5 却损失
7.664pp。与此同时，h=.25 的 q8 在三个场景都受益于 T5。因而统一关闭内容或
统一更换中心都缺少充分支持。R16 也没有证明连续四次密度的平顶或各向异性
是唯一原因；这些限制保持在[原报告](/home/liwt/GRoPE/docs/round16_mechanism_report.md)。

本轮回到已有 Gaussian full1 的单高斯矩形质量，只改变一个已存在的内容权重：

    tau_q = T * h^2 / (h^2 + trace(C_q))
    score_qk = old_Gaussian_full1_geometry_qk + tau_q * content_qk.

C 是投影区域的完整支撑协方差，以原 reference-8 坐标计量；h 沿用原四点网格。
trace(C) 与 h² 同单位，规则不依赖坐标旋转，联合缩放 C 与 h 的坐标单位时该
比值不变。C→0 时恢复原点核内容温度；支撑变大时内容权重减小。选用完整
二维 trace 与系数一是本轮固定的简单假设，不是从任务损失推出的最优定律。
不新增 trace/2、特征值、置信度或形状扫描，不学习温度预测器。

**区域大小不是不确定性。** 大 C 可以表示已知且正确的大区域，不能解读成
几何不可靠。此规则可能恰好压制真正有用的语义内容，也可能只把模型推向区域
质量不足的 T0 极限；它不能恢复区域矩丢失的指定中心身份，不能修复 source V
质量，也没有保证抵御错误相机或深度。几何平均化、内容错配及离散 cell 的问题
仍须由相同的原验证逐项检查。

这里也不声称新的高斯分布或 RoPE 旋转恒等式。可将 tau 吸收入 query 的内容
通道，但方法实质是“位置支撑条件化的内容—位置融合”。若它有效，证据首先
支持这个耦合规则，而不是证明原 Gaussian PE 或 3DGS 参数化本身具有全新
理论性质。相关 Gaussian PE、概率位置先验及自适应温度的已有工作见
[文献与新颖性边界](/home/liwt/GRoPE/docs/round17_literature_scope.md)。现在的 C
来自预测像素区域矩，实际 optimized GS 的内部 R/S 仍需独立任务验证。

数值路径固定为 double 计算 tau，并用公共正尺度 s=max(h²,C00,C11)：先算
r=h²/s 与 d=C00/s+C11/s，再算 tau=T*r/(r+d)，避免巨大 trace 求和溢出，
不 clip、不加数学自由参数。具体运算顺序见
[数值前置协议](/home/liwt/GRoPE/ROUND17_PREFLIGHT_PROTOCOL.md)。之后 cast 到
float32，与 float32 content 相乘，最后转 double 加旧 log-mass。
旧 GL32、24px 的 12px 子格积分、原生 key 面积
floor 均不改。生产模块执行的 2×2 对称/PSD 检查及其成本如实保留，不能把
实际开销描述成只有一次 trace。独立合成与三首 TRAIN pair、14 条件的数值
检查均在效果前联合冻结，全部失败例保留；通过数值不等于通过科学门槛。

新增前向只覆盖这一个 `trace_full` 族的原 12 个 h/T。旧 Gaussian/quartic point
的两项全局定位选择与 legacy point_pck 三重保护全部保留，不通过删除较强
参照制造可行集合。旧区域控制复用已审计的 R15 完整八族结果和全来源链；
强 point、quad 及旧 legacy 角色不事后重选。T0 的 48 pair×四 h×三尺度概率
必须逐元素 canonical hash 等于旧 Gaussian full1，这同时检查没有暗改几何。

执行顺序是数值前置检查 → 48 TRAIN 的中心可行集合 → 全 14 条件区域校准。
中心可行集合为空则停止；非空时完整评价全部 12 配置，再在可行集合内按原
J=.5×common9+.5×direct5 选择一次主配置。不能从不满足中心保护的配置中挑
区域最好者，也不能在之后看到止损结果再换另一个可行配置。

本轮在新结果前新增明确的 TRAIN 开发止损筛选：所选配置的 common9/direct5
均须对 strong point 与 legacy point 各改善≥2%；common9 对 strong quad 与
legacy quad16 各恶化≤1%，direct5 各恶化≤2%。这只检查最低宏指标前景，不加
训练三个场景全部同向的新硬门槛。失败停止当前已选候选，不宣称不存在其它
可行方法。这不是 R15 原来已有的规则；R15 的停止仍是原先记录的事后决定。

正式旧 VAL 若获准，保留全部原分区、尺度、六扰动及 zero、三项中心保护、
强控制、逐条件与成本要求。matched constant-T full1 区分温度缩放的作用；
matched isotropic 几何区分方向信息的作用，沿用原 full C 算出的完全相同 tau，
避免 float32 构造等方 C 后的 trace 舍入混入归因。两者都不独立校准、不替换
主选。还报告旧自行优化的 full1；若最终 T=0，则新规则实际未起作用，不能
归功于 trace 缩放。

该方案仍可能在第一道中心检查、区域止损、完整旧 VAL 或真实 V 任务失败。
任何阶段通过都不豁免后续检查；完整区域、下游、实际 GS 和未触碰场景确认
之前，不能交付“已经得到一般有效方案”的结论。
