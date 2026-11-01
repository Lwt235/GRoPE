# R32：初始深度分区修订与独立参考预检报告

报告整理于 2026-11-01；实验、冻结与验收沿用 `20261031` 文件标识。**R32 固定合成数值预检 PASS：23 个样例、46 次参考调用、155 项比较全部通过。** 但原六阶段流程并非全部成功：第六阶段审计因 `numpy.bool_` 无法写入 JSON 而退出，随后在独立目录完成了封存后的纯序列化修复。参考积分只执行了一轮，修复没有重算积分、改变数值门槛或修改冻结源码。

这是合成参考验证，不是一般参数准确性、真实输入概率准确性或效果验证。R31 的失败记录及以前各轮效果结论保持原状；本轮没有真实数组、标签、新生产模型调用、VAL、实际 V 或 GS 优化，也没有晋升科学候选。

本报告只读封存 JSON 和文件字节哈希，没有解码 NPZ 或调用积分器。已核对 [root 验收记录](/home/liwt/GRoPE/results/round32_log_conditional_root_acceptance_20261031.json) 的 46 项绑定、三份完整库存（原生产 98、原失败审计 24、修复审计 34 个文件，均含各自 `completion.json`）、14 项原来源的当前文件及三处快照，以及 3 项修复来源的当前文件和快照。修复锁绑定的原两棵目录及复制证据也逐项相同。root 验收记录 SHA256：`4f61663d9bd6fbe7c445b3bd5aa74b6030e448ea48459e23cba315e1bdedb2e5`。

本轮按 [固定协议](/home/liwt/GRoPE/ROUND32_LOG_CONDITIONAL_REFERENCE_PROTOCOL.md) 和 [设计](/home/liwt/GRoPE/docs/round32_log_conditional_reference_design.md)，继续使用原 R29 的 23 个 base 输入、43 份旧参考证据及原生产输出。23 份输入的描述符和四项数组字节均独立核验，输入审计在参考调用前封存；全部参考封存后才比较旧输出。六项旧 identity 和有限 ε 的近常深度结果只继承，不计为新调用。没有新增“R32 必须与 R31 输出逐位相同”的门槛。

唯一数值参考修订是确定性合并初始深度分区：保留原两端点，按顺序保留能够与前一保留点形成严格可表示中点的内部断点；否则删除该分区边界。若末段与上端点之间无严格中点，则回退删除最后的内部断点。仍使用原 `(left+right)/2` 公式；如果整个原区间也无严格中点，就失败。删除的是分区提示，完整积分区间没有被截去；不引入 ε、概率裁剪或新的尾部豁免。pilot 与外层积分使用同一保留分区，其他全 X 对数条件参考、深度截断、带偏置 floor/P 和预算规则均保留。

| 固定检查 | 数量 | 结果 |
|---|---:|---|
| 输入独立字节审计 | 23 | 全通过 |
| 新参考调用，每例 32/48 阶 | 46 | 全完成 |
| 新 32 对新 48 | 23 | 全通过 |
| 每个新阶数对全部接受过的旧参考 | 86 | 全通过 |
| 原 R29 生产输出对每个新阶数 | 46 | 全通过 |
| 修复后的独立交付与算术检查 | 46 | 全通过 |

155 项比较共享样例、阶数和输出，不能解释为 155 个独立样例。旧 43 份参考由 20 个样例的双阶数值结果与 3 个样例的单解析结果组成；R32 对这些样例统一执行两个新的参考阶数。

**实际分区变化。** 下表给出全部 12 个变深度样例；两阶的分区计数完全一致。其余 11 个常深度或结构零样例不应用此规则，不能把“不适用”写成额外的合并验证。

| 样例 | 原内部断点 | 保留内部断点 | 每阶删除数 | 每阶初始 pilot 数 |
|---|---:|---:|---:|---:|
| `06_radial_central` | 12 | 12 | 0 | 13 |
| `07_radial_noncentral` | 16 | 14 | 2 | 15 |
| `08_full_central` | 17 | 17 | 0 | 18 |
| `09_full_noncentral` | 22 | 22 | 0 | 23 |
| `10_negative_mean` | 22 | 22 | 0 | 23 |
| `11_near_constant` | 0 | 0 | 0 | 1 |
| `12_tiny_front` | 2 | 2 | 0 | 3 |
| `13_native432` | 197 | 197 | 0 | 198 |
| `14_rank2` | 29 | 29 | 0 | 30 |
| `16_radial_central_bias` | 12 | 12 | 0 | 13 |
| `17_full_noncentral_bias` | 22 | 22 | 0 | 23 |
| `18_tiny_front_bias` | 2 | 2 | 0 | 3 |

只有 `07` 实际删除断点：原数组索引 4 的 `−1.6506143785156577` 与索引 16 的 `9.180339887498949`；两阶均记录原因为 `no_strict_midpoint_from_previous_retained`。本次没有发生上端点回退删除。按两阶调用重复累计，内部断点总数从 706 减为 702；这不是四个独立几何缺陷。

独立审计从保存的 original edges 重新执行另一份简单规则，逐字节核对 retained/dropped 数组及索引、值、原因，核对原端点、pilot 中点和工作计数。原始几何 root 的生成仍由冻结来源绑定，没有宣称审计另行重写了 SVD/root 算法。`07` 两阶各有 15 次 pilot、357 次外层求值，最终各 16 个自适应区间；分别使用 `1,595,424 / 2,393,136` 个节点，完整单调用约 `0.862937 / 1.113834 s`。R31 在这个样例上曾于 pilot 前停止，R32 完成的是这项先前明确暴露的缺口，不是未接触过的确认样例。

**数值误差与极小质量。**

| 比较类型 | P 最大绝对误差 | raw-FOV 最大相对误差 | raw-L1 / reference-FOV 最大值 | 原 P／raw 门槛 |
|---|---:|---:|---:|---|
| 新双阶 | `2.220446049250313e-16` | `1.421085471520220e-14` | `5.314185147598529e-15` | `1e-8 / 1e-8` |
| 新参考对每份旧参考 | `1.484923295436147e-15` | `1.136868377216263e-13` | `1.136868377216263e-13` | `1e-8 / 1e-8` |
| 原生产对新参考 | `3.330669073875470e-16` | `1.136868377216263e-13` | `1.136868377216263e-13` | `1e-4 / 1e-6` |

raw 相对误差只统计正质量样例；`04_zero_depth`、`05_negative_constant` 是原合同中的结构零，raw log 为 `−inf`，不参与相对误差除法。新双阶最大 P 差的首个记录是 `07`，最大 raw 差来自 `12_tiny_front`；新旧参考最大 P 差来自 `18_tiny_front_bias`，原生产对新参考最大 P 差来自 `06_radial_central`；后二组 raw 最大差均来自 `28_subfloat_fov`。

全部 46 次调用的最大归一化误差为 `3.9968028886505635e-15`（门槛 `2e-12`），native 总质量与单独 direct-FOV 质量的最大绝对差为 `7.111191715189293e-16`（门槛 `1e-8`）。保存的 log-P 再取 exp 与 P 的差为 0；独立 floor/P 算术最大 P 差为 `2.220446049250313e-16`，log-floored/log-P 最大差均为 `1.4210854715202004e-14`。变深度结果的保存尺度恢复残差均为 0，这只证明保存中间量的恢复一致性，不是独立积分证书。

`12_tiny_front` 的 32 阶 raw-FOV log 为 `−81.00177513658565`，约 `6.6279013206559e-36`，总 floor 约 `1.6e-29`；raw/floor 约 `4.1424383e-7`。`18` 复用该几何但内容偏置不同。`28_subfloat_fov` 两阶均为 `log(raw-FOV)=−765.4648717106797`，对应数学量约 `3.6545181680e-333`，无法作为正 float64 数直接表示；总 floor 约 `2.5e-31`。对数 raw 相对检查仍通过，没有将它替换成零或以 floored P 一致豁免 raw 门槛。新参考未输出旧生产的 packed materialization 清单，不把旧计数冒充本轮测量。

P 接近也不等于 argmax 稳定。新双阶、新旧参考、原生产对新参考分别有 2、4、2 条 argmax 变化，仅涉及 `01_point` 与 `25_wide_constant_axis`；变化均处于双方 `1e-12` 近并列集合 `[5,6,9,10]` 和 `[0,1,2,3]` 内。至少一侧有多元素近并列／精确并列的比较记录分别为 4、14、8 条。这些是重复比较记录，不是独立查询计数。

物理深度尾界的最大 raw 比值为 `1.4696882840695367e-37`，最大带偏置 P 界为 `2.4352372297342188e-43`，均低于原 `1e-10 / 1e-8` 门槛。它们只约束固定深度范围外的质量，raw 分母仍是计算出的保留 FOV 质量，不是区间包围值。全 X 积分没有 X 截断，但仍存在求积与浮点误差。

外层 exp 在 18 次调用中记录合计 428,288 个相对下溢分量事件；LSE 的 `rectangle_panels / native_total / floor` 分别记录 178,188 / 232 / 196 个丢失项，其余五类为 0。对应最大局部 log-relative omitted bound 分别为 `−742.3201002023112 / −766.4849498197723 / −746.0814938246217`。这些是不同阶段重复求值的局部诊断，不能相加当作独立质量损失；pilot 尺度不是全域上界，scaled 外积分误差估计、局部 LSE 界和双阶一致性也不能提供一般参数或全流程的严格误差证书。

**积分成本。** 参考只执行一次，完整阶段耗时 `111.23214978352189 s`，charged nodes 为 `412,301,376`，低于 `1800 s / 600,000,000` 的上限。最大单调用为 `13_native432` 的 48 阶：`218,875,568` 节点、约 `49.888642 s`（单调用节点上限 `250,000,000`）。最大深度求值 `4,398`、最大 outer intervals `199`，均在原 `32,768 / 512` 上限内。全量 pilot / quadrature 深度求值为 `726 / 17,010`，合计 `17,736`。

全部单调用时间之和 `110.94552688859403 s`，参考函数自报时间之和 `110.75155018828809 s`，都不替代完整阶段时间。最大 RSS `122,324 KiB` 是累计进程高水位，不是单调用独占内存。修复只重新检查保存数组，其时间不混入这轮积分成本；[修复执行记录](/home/liwt/GRoPE/results/round32_log_conditional_audit_repair_execution_20261031.json) 单独保存了冻结和审计阶段时序。

**原失败和修复边界。** [原执行记录](/home/liwt/GRoPE/results/round32_log_conditional_execution_20261031.json) 的前五阶段 exit code 均为 0，第六阶段 `independent_output_audit` 为 1。原因是审计新诊断 `strict_midpoints_exist` 保留了 `numpy.bool_`，写 `checks.json` 时抛出 `TypeError: Object of type bool_ is not JSON serializable`。原 [failure.json](/home/liwt/GRoPE/results/round32_log_conditional_independent_audit/failure.json)、部分 checks、日志和完成库存均保留。前期 84 项联合测试通过没有覆盖到这个真实输出序列化路径，不能据此淡化工程失败。

依据另冻的 [修复协议](/home/liwt/GRoPE/ROUND32_LOG_CONDITIONAL_AUDIT_REPAIR_PROTOCOL.md)，新目录只逐字节复制原锁、来源快照及前置输入审计证据，明确标注它们是继承证据，不是新的前向前冻结。新的 repair lock 绑定原两棵目录、失败记录、旧控制器历史与修复源码。修复临时替换的仅是 JSON writer：保持原格式和 `allow_nan=False`，只接受 exact `numpy.bool_→bool`，其他不支持类型仍报错，写入限于新目录，退出时恢复 writer。冻结的数学函数、门槛、46 个检查和 155 项比较函数均未修改。前后均复核原目录不变，修复与 root 验收的积分次数都是 0。[修复审计结果](/home/liwt/GRoPE/results/round32_log_conditional_independent_audit_serialization_repair/verification.json) 与原失败目录分开保存。

原 root 验收脚本还曾提前假定存在 `all_stages_completed` 字段，产生 `KeyError`；这不是另一次参考失败。原脚本、回执及明确标注为终端转录的错误见 [原控制器开发历史](/home/liwt/GRoPE/results/round32_root_controller_development_history_20261031.json)，没有被覆盖。后续控制器分别核验五个成功阶段、失败的第六阶段和新修复结果，其 [执行前源码快照](/home/liwt/GRoPE/results/round32_repaired_controllers/before_execution.json) 独立保存。

开发验证方面，参考 28、runner 27、审计 29 项 toy 均首轮通过，联合为 **84 项通过、14 项来源冻结**；参考保留原 15 项测试，只新增分区逻辑测试。修复 wrapper 另有 **13 项 toy 首轮通过**，覆盖真假 NumPy 布尔值、原失败复现、非法类型/NaN 拒绝、写入边界、异常恢复及数值失败不被改成通过。两组验证均保留测试前源码快照和日志，见 [原联合预检](/home/liwt/GRoPE/results/round32_log_joint_preflight_20261031.json)、[参考开发历史](/home/liwt/GRoPE/results/round32_reference_development_history_20261031.json)、[审计开发历史](/home/liwt/GRoPE/results/round32_audit_development_history_20261031.json) 与 [修复开发历史](/home/liwt/GRoPE/results/round32_audit_repair_development_history_20261031.json)。

完整交付为 [23 个样例汇总](/home/liwt/GRoPE/docs/round32_log_conditional_reference_report_data/cases23.csv)、[46 次参考调用](/home/liwt/GRoPE/docs/round32_log_conditional_reference_report_data/reference_calls46.csv)、[155 项比较](/home/liwt/GRoPE/docs/round32_log_conditional_reference_report_data/comparisons155.csv)。每例和每次调用包含实际 coalescing 记录；空值表示不适用，不表示零误差。继承的有限 ε 近常深度结果及六项 identity 未被包装为本轮新验证。

本轮支持的结论是：固定分区修订后，这 23 个合成输入上的新独立参考完成了全部原数值检查；封存后的输出序列化问题已通过独立修复处理。它没有建立真实输入的一般概率精度或任何任务效果。后续真实参考实验仍需单独协议、来源冻结、预算和完整审计，不能从本轮 PASS 自动推导科学成功。

| 交付 CSV | SHA256 |
|---|---|
| `cases23.csv` | `0f45fcd7981068642ea91e41c51dc43083ec5b24de3629d6f9c8f95fb747ba22` |
| `reference_calls46.csv` | `b78e735fc68d013ef6d1ed3a47c1ed4d58c060bb7ba39a697263f22c098590eb` |
| `comparisons155.csv` | `cbc98ec73fb0ce0d5e888f96d244d2e1f582a448b442903c1346e9bebc14eef0` |
