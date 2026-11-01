# R35：GL8 独立参考在固定真实输入上的预检

2026-11-01。**R35 数值预检 FAIL，完整失败证据已通过 root 验收。** 72 项输入全部通过身份及字节审计。144 个预定调用位置中，22 次实际尝试、21 次完成；第 22 次调用因外层求积报告舍入误差而未收敛，后续 122 次按冻结的 fail-fast 规则跳过。254 项比较完整保留，其中 69 项具备完整数值结果并通过，185 项因缺少完整参考结果而失败。

正式六阶段及 root 验收成功退出，表示交付、失败留存和审计合同执行完毕，不表示数值全通过。`delivery_integrity_passed=true`，`numerical_preflight_passed=false`；收据中的 `accepted_for_numerical_interpretation=true` 是允许解释已保存证据，不能替代后一字段。本轮没有新生产模型前向、标签读取、效果评估或候选晋升，一般准确性、科学门槛与研究目标完成标记仍为 false。

本报告只读封存 JSON 和文件字节哈希，没有解码 NPZ 或执行积分。已核对 [root 验收记录](/home/liwt/GRoPE/results/round35_log_real_root_acceptance_20261101.json) 的 **34 项文件绑定**、**16 项当前来源及生产/审计双快照**、生产 **247** 与审计 **31** 个文件的完整库存，以及 prepared、forward 和 input-audit 子库存；文件数均包括各自 completion。收据 SHA256：`cd6766fc0b0f5501a8b9e85e4fb84252fd6ddc0d5be7fbaebe616d8a98f46b70`。

**固定范围。** 按 [设计](/home/liwt/GRoPE/docs/round35_log_real_reference_design.md) 与 [协议](/home/liwt/GRoPE/ROUND35_LOG_REAL_REFERENCE_PROTOCOL.md)，使用原 R30 接受的全部 72 项 prepared 输入及原 R33 顺序，包括曾失败和未尝试的项目：c3/c5/c6 各第一对 `[30,45]`，query 尺度 8/16/24 各 8 个固定索引，每项都有 432 个原生 key。

只逐字节复制四项 float64 数组：`mu[3]`、`factor[3,3]`、`bounds[432,4]`、`content_bias[432]`，保留七项 query 身份字段。bias 仍全零，bounds 仍为 `(pixel_boundary−.5)/8`，没有重新构造三维矩、相机映射、anchor 或内容法则。输入来源是已接受的 R30 prepared 档案，没有沿历史路径重新读取 R23 原始数据。独立输入审计完成于 `1793465036.4029086`，早于参考阶段开始 `1793465037.1307747`；全量调用结果封存后才读取旧生产和旧参考输出。

新参考直接使用 R34 冻结文件，SHA256 为 `d42d7c7fa4e26cb4357635ac8ac0df524a3ae0b3d8f6f1d2314f5202b9cc8ff5`。它只在 R32 参考基础上将窄正态 GL16 改为 GL8，分支跨度仍为 `.25`；full-X、SVD、断点合并、固定 pilot 尺度、深度截断及所有其他数学保持。R35 没有继续修改参考。每项依次执行 32/48 阶，阶段预算保持 R33 的 `36,000,000,000` 节点与 `10,800 s`；单调用保持 `250,000,000` 节点、`32,768` 次深度求值及 `512` 个外层区间。

原 216 项比较全部保留：72 项新双阶及 144 项原 R30 生产对新参考。另固定加入 38 项严格回归：将 R33 已完成并经独立审计接受的 **19 份输出**分别与两个新阶数比较。这 19 份来自原前 9 个输入的双阶及第 10 个输入的旧 32 阶，不含旧失败 partial 或 skipped 档案。它们是已知开发回归材料，不是未接触过的新确认人口。六项旧 identity 和有限 ε 的近常深度证据只继承，没有增加新 identity 调用。

**全部调用位置。** 每个 scene×size 固定 8 项输入、16 个调用位置。

| 场景 | query size | 实际尝试 | 完成 | 实际失败 | fail-fast 跳过 |
|---|---:|---:|---:|---:|---:|
| c3 | 8 | 16 | 16 | 0 | 0 |
| c3 | 16 | 6 | 5 | 1 | 10 |
| c3 | 24 | 0 | 0 | 0 | 16 |
| c5 | 8 | 0 | 0 | 0 | 16 |
| c5 | 16 | 0 | 0 | 0 | 16 |
| c5 | 24 | 0 | 0 | 0 | 16 |
| c6 | 8 | 0 | 0 | 0 | 16 |
| c6 | 16 | 0 | 0 | 0 | 16 |
| c6 | 24 | 0 | 0 | 0 | 16 |
| 合计 | — | **22** | **21** | **1** | **122** |

按输入计，10 项双阶完成、1 项只有 32 阶完成、61 项的两个阶数均未尝试。后 61 项的参考可完成性和精度未知，不能说它们各自失败，更不能据此比较 c5/c6 或 q24 的难度。跳过行保存空档案、零实际工作、明确的 `attempted=false` 和触发原因，没有新的该行输入解码或积分调用；全 72 项此前已经过 prepare/input-audit，逐调用未读不等于整个流程未读。

**实际失败及与 R33 的关系。** 唯一实际失败为 zero-based index 21，即第 22 次调用：

`harvard_c3__hv_c3_1__g16__q0030__order48`

对应像素 bounds `[96,32,112,48]`，原错误全文为：

`ReferenceFailure: RuntimeError: independent scaled outer quadrature failed: Target precision could not be reached due to rounding error.`

它用了 **159,392,448** 个节点、约 **41.007551 s**，没有耗尽单调用或阶段预算。115 次 pilot 加 2,499 次 quadrature 深度求值，共 2,614 次；外层返回 117 个区间。原始与保留内部断点均为 114，没有实际合并删除。本次所有 22 次尝试均没有删除断点，因而不是 R31 的不可表示中点问题复发。

这次失败与 R33 的预算中断不同：外积分已经返回，保存了 `outer_scaled_integral`、partial raw logs、computed mask、pilot 和局部丢失诊断。依据 [冻结控制流](/home/liwt/GRoPE/experiments/round34_log_narrow_reference.py:537)，报出该错误前已检查 433 个 scaled 分量有限且为正，并完成有限 raw-log 恢复；随后因 `info.success=false` 拒绝接受。**不能把它描述为 partial 全为 `−inf`、没有算出任何 raw，或输出零概率。** 同时，它没有完整的 `log_floored_mass`、`log_probability` 或 `probability`，尚未执行正常化与尾部门槛，不能把 partial 充当合格概率结果。

失败记录的 `scaled_outer_error_estimate=7.356992937981754e−13` 只是 scaled vector 的外层估计。记录没有给出足以定位舍入来源的独立误差分解，不能根据这一数值小于某个绝对数，就推翻原收敛判定或称 raw/P 已达标。同一输入的 32 阶完成，估计为 `7.164325856846042e−13`，使用 104,234,560 节点、116 个外层区间、约 28.125175 s；原规则的外层 `epsabs=0`，32/48 阶 `epsrel` 分别为 `1e−11 / 1e−12`。两个估计相近不表示两个阶数应有相同判定。

R33 原先在 `c3 g16 q0015` 的 48 阶因节点预算失败，R35 该调用已经完成：225,449,616 节点、约 44.593266 s，低于原 250M 上限；同项 32 阶为 150,301,456 节点。其新双阶和生产比较通过，旧 32 阶也参与了严格回归。原 R33 的失败档案没有被替换，且旧失败 partial 没被当作参考。此次只是把可完成前缀向后推进，随后暴露不同的收敛失败，不是完整真实人口已通过。

**比较覆盖及成功子集。**

| 比较组 | 固定数量 | 有完整数值并通过 | 缺失导致失败 | P 最大绝对误差 | raw-FOV 最大相对误差 | raw-L1 / reference-FOV 最大值 |
|---|---:|---:|---:|---:|---:|---:|
| 新 32 对新 48 | 72 | 10 | 62 | `2.220446049250313e−16` | `4.4408920985006143e−16` | `4.457411859515692e−16` |
| 原 R30 生产对新参考 | 144 | 21 | 123 | `1.4432899320127035e−15` | `1.1990408665951659e−14` | `1.1578930636955128e−14` |
| 新参考对 19 份旧 R33 结果 | 38 | 38 | 0 | `2.220446049250313e−16` | `4.4408920985006143e−16` | `4.457411859515692e−16` |
| 合计 | **254** | **69** | **185** | — | — | — |

参考之间的 P/raw 门槛仍为 `1e−8 / 1e−8`，生产对参考仍为 `1e−4 / 1e−6`。最大值只覆盖表中的有结果子集；新增 38 项全部通过不能抵消原 216 项未通过。185 条失败比较均保存实际缺失原因 `KeyError: 'log_raw_mass'`，其中 2 条依赖本次实际失败调用，183 条依赖 skipped 结果。它们不是 185 次独立精度超限；缺失误差在 CSV 留空，没有补成零。

21 次完成参考的保存输出算术、尺度恢复、direct-FOV 和深度尾界均通过。其最大归一化误差为 `8.881784197001252e−16`，最大 native sum 对单独 direct-FOV 的绝对差为 `9.991959429269933e−16`，分别低于 `2e−12 / 1e−8`。独立 floor/P 算术最大 P 差为 `2.220446049250313e−16`，log-floored/log-P 最大差均为 `1.4210854715202004e−14`，保存尺度恢复残差为 0。其余 123 个调用位置没有可通过的完整算术结果。

全部 69 条有数值比较记录均无 argmax 变化、无 exact 或 `1e−12` 近并列。这只描述当前成功前缀及重复回归记录，不保证剩余输入的排序稳定。成功参考的 native raw 总质量 log 范围为 `[-2.581914726936636, 5.551115123125783e−16]`；总面积 floor 的 log 为 `−63.00912720157726`。微小正 log 原样保留，没有事后裁剪。这个子集没有整个 FOV 极小质量案例，但不能据此认为每个远端 cell 都可普通物化，更不能覆盖未完成项的 tiny-tail 精度。

成功子集的最大深度截断 raw-L1/FOV 界为 `1.2879902770760522e−71`，最大带偏置 P 界为 `9.019397147163708e−45`，低于原 `1e−10 / 1e−8` 门槛。这些只约束深度域外贡献，分母是计算出的保留质量，不是区间包围值。GL8 的 `1.88e−13` 也只约束局部精确算术余项；它没有防止本次外层舍入未收敛，不能当作全流程误差证书。

22 次实际尝试（含失败一次）累计记录 1,526,916 个 outer-exp 相对下溢分量事件。LSE 的 rectangle-panels、native-total、floor 分别记录 485,689 / 1,112 / 1,031 个丢失项；最大局部 log-relative omitted bound 分别为 `−741.763509787297 / −742.1785395258281 / −745.7194124065979`。它们是重复求值的局部诊断，不能相加当作整体损失或独立样本数。pilot 尺度不是全域上界；成功参考最大 scaled 外积分估计 `4.799684017264721e−8` 也不能直接同 unscaled raw/P 门槛比较。

**成本和停止原因。** 完整参考阶段为 **449.4680993780494 s**，总 charged nodes **1,930,008,008**，均低于固定阶段限制。最大单调用节点为已成功的 c3 g16 q0015、48 阶的 225,449,616；本次触发停止的是另一个调用的收敛失败，不是耗尽全局时间或节点。所有行时间合计 `448.2327400557697 s`，21 次成功参考自报时间合计 `406.8170326128602 s`，两者都不替代完整阶段时间。最大 RSS **116,320 KiB** 是累计进程高水位。

22 次尝试的 pilot／quadrature 求值总计 1,794 / 38,976，合计 40,770 次；最大单调用深度求值 3,672、外层区间 166，未超过原限制。不能用该成功前缀的平均时间估计剩余 122 次的成本，也不能把 R33 与 R35 的不同停止位置和总时间直接当作完整任务的速度比较。

**开发与时序。** [合同开发历史](/home/liwt/GRoPE/results/round35_contract_development_history_20261101.json) 首次 metadata-only 检查通过：1,103 项绑定、72 个 specs 和 19 个唯一 oracle，在禁止数组解码条件下完成。[runner 历史](/home/liwt/GRoPE/results/round35_preflight_development_history_20261101.json) 42 项 mock 首轮通过，[审计历史](/home/liwt/GRoPE/results/round35_audit_development_history_20261101.json) 61 项首轮通过，[联合预检](/home/liwt/GRoPE/results/round35_joint_preflight_development_history_20261101.json) **103 项通过、16 项来源冻结**。联合测试是原集合重验，不是新增独立实验。

各测试之前保留了源码、命令和日志；最终三份控制器也都有 [执行前快照](/home/liwt/GRoPE/results/round35_controllers/before_execution.json)。[正式执行记录](/home/liwt/GRoPE/results/round35_log_real_execution_20261101.json) 的六阶段均 exit 0，失败调用及后续 skipped outcomes 被正常封存。[控制器完成记录](/home/liwt/GRoPE/results/round35_controllers/completion.json) 的完整五文件库存另行核对，SHA256 为 `efe25ef5f4b589789437e8f93199b738235da1fc08ce0c8c85a32a0dfc9dfe53`，确认三控制器成功退出且源码与执行前快照相同。上述正式流程与 root 验收只重核保存证据，没有重跑积分；没有改动冻结参考、放宽收敛规则、在正式人口内重试该调用或删除后续位置。此处“没有重试”限定正式 R35 执行，后续单独诊断见下方补记。

完整附表为 [72 个输入](/home/liwt/GRoPE/docs/round35_log_real_reference_report_data/cases72.csv)、[144 个调用位置](/home/liwt/GRoPE/docs/round35_log_real_reference_report_data/reference_calls144.csv)、[254 项比较](/home/liwt/GRoPE/docs/round35_log_real_reference_report_data/comparisons254.csv) 和 [19 份旧参考索引](/home/liwt/GRoPE/docs/round35_log_real_reference_report_data/oracles19.csv)。另有 [统计 JSON](/home/liwt/GRoPE/docs/round35_log_real_reference_report_data/statistics.json) 与 [提取完成记录](/home/liwt/GRoPE/docs/round35_log_real_reference_report_data/completion.json)。纯标准库提取器先保存源码与命令，首轮执行成功，实际输出日志保留；只读 JSON/字节，没有 NPZ 解码或积分。

| 交付 CSV | 行数 | SHA256 |
|---|---:|---|
| `cases72.csv` | 72 | `4806ebe6f1fcf0e52632eda49d017ba969e445ee414d2a451a9f996f82aec293` |
| `reference_calls144.csv` | 144 | `b627b3ca00458f52d5f7542a5fb691ed65ac029bdefc28a19f78909abf49fa82` |
| `comparisons254.csv` | 254 | `ea53d98f0d9b4e5d4bdebfe3358b5e9dd052a9a3f61f8018ffc381716cef1e8b` |
| `oracles19.csv` | 19 | `10174611c7cf8bf06772494c8a6070aa046daf5610bc0f2503a9a16bc98229f2` |

R34 的合成 PASS、R30 的 72/72 生产可完成性、R35 的真实参考失败是不同证据。本轮没有建立完整真实人口的独立概率准确性，更没有提供任何新效果结果；已有科学失败和后续效果、扰动、实际 V、GS、未触碰数据确认要求都保持有效。

**2026-11-01 补记：独立的失败后诊断。** 报告初版交付后，另行冻结并执行了**一次**已知失败调用的观测性重放，详见 [舍入诊断](/home/liwt/GRoPE/docs/round35_reference_roundoff_diagnosis.md) 和 [执行收据](/home/liwt/GRoPE/results/round35_roundoff_postfailure_diagnostic_20261101/execution_receipt.json)（SHA256 `6463eb9f6bb2d3782f12eccd508fb48d3af7db9dd3f04eb82914a4b2ad133d4f`）。该调用保持原输入、参考、容差和停止判据，约 41.985 s、159,392,448 节点，复现了原失败原因、全部工作计数和 partial 数组字节；新增的是 SciPy 区间观测。原返回分区的当前叶舍入下限约为原收敛目标的 1.0843 倍，这解释了该分区的停止问题，仍不是 raw/P 误差证书。它未执行 floor/P 验收，也未把失败 partial 提升为 oracle。

这次诊断属于正式结果之后的单独调用，未纳入上表 144 个位置或成本，未改写 R35 的 21 完成／1 失败／122 跳过及完整预检 FAIL。初始未执行草稿和目录碰撞审计尝试也保留，均没有新增积分。本补记前的报告字节及 SHA256 已存入 [draft history](/home/liwt/GRoPE/docs/round35_log_real_reference_report_data/draft_history/20261101_postfailure_diagnostic_supplement/before_edit.json)；原统计、CSV、提取记录和正式封存输出保持不变。
