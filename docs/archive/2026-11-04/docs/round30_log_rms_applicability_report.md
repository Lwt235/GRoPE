# R30：对数 RMS 投影模型在固定 72 个真实输入上的适用性

2026-10-31。**本轮适用性检查通过：72/72 输入精确复用审计通过，72/72 模型调用实际执行并完成，72/72 保存输出算术检查通过；总前向阶段 56.966884 秒，未超原预算。** 同一组输入在 R28 中为 40/72 完成。本轮说明所冻结的对数表示实现能够处理这组输入，不能解释为定位、区域分布或其他效果提升；没有独立真实概率参考，也没有读取标签。

报告只读取根验收后的封存 JSON、源码和文件哈希，没有解码真实 NPZ、重新运行模型或参考积分。已复核 [root acceptance](/home/liwt/GRoPE/results/round30_log_rms_root_acceptance_20261031.json) 的 29 项文件绑定、13 份当前源码及两侧快照、生产目录 171 个文件和审计目录 24 个文件，并核对 prepared/forward/input-audit 子清单。根验收 SHA256：`158613a99a83328bd902ff1c537b782bdf6de18b70e1990d86011d7fb97373a5`。文件数包含各自 completion。

## 固定输入、模型与执行边界

人口保持 c3/c5/c6 三个旧 TRAIN 场景、各自 pair `[30,45]`、query 尺寸 8/16/24、每尺度 8 个预定索引，共 72 个查询。每个查询对 432 个原生 key 格积分，共 31,104 个 query–cell 出现次数；这不是完整训练集或未触及确认集。

来源是已接受的 R28 `prepared/arrays`。只解码并逐字节复制四个 float64 字段：`input__mu[3]`、`input__factor[3,3]`、`input__bounds[432,4]`、`input__content_bias[432]`。其余九个历史字段只保留哈希元数据，未重新解码；不回到 R23 原观察、不再映射相机或重建协方差。新的 prepared 档案恰含四字段。432 个格的边界为 `(pixel_boundary-.5)/8`，每格面积 1、总面积 432，bias 精确为零。

报告独立比对了全部新旧 prepared 记录的四类 canonical 输入 SHA，全部相同。独立输入审计还逐字节检查了 dtype、shape、数据、原查询身份、边界和零 bias，全部通过；这里没有用容差允许输入改变。R28 映射审计的 `2.591716307e-15` 属于继承证据，不能当作本轮复制产生的误差。

模型是 R29 已冻结的 `models/round29_log_rms_projective_pe.py`，SHA256 `fcb1b6af25a1beca982c6d9911f04b1cbeb922d3cead5cc6e11ebbd177c3ef30`。保持同一 RMS 正则投影概率律、严格 `Z>0`、`h=.25`、floor `1e-30`、GL24 和最大边长 `2h` 的积分 tile；不改准入、深度秩、内容规则或模型结构。源冻结在数组解码前；全 72 个复制档案封存后完成独立输入审计，再开始任何模型调用。输入审计 completion 时间为 `1793459070.2692595`，最早逐调用开始晚于它 `0.4038589 s`；全局前向 seal 完成后才开始最终算术审计。详见[协议](/home/liwt/GRoPE/ROUND30_LOG_RMS_APPLICABILITY_PROTOCOL.md)和[输入验收](/home/liwt/GRoPE/results/round30_log_rms_applicability_independent_audit/input_verification.json)。

## 全部 72 个结果与 R28 对照

| 场景 | query 尺寸 | R28 完成 / 8 | R30 完成 / 8 | 不能普通物化的正质量格数 | subnormal 格数 |
|---|---:|---:|---:|---:|---:|
| c3 | 8 | 4 | 8 | 524 | 19 |
| c3 | 16 | 6 | 8 | 191 | 11 |
| c3 | 24 | 4 | 8 | 332 | 22 |
| c5 | 8 | 1 | 8 | 1,101 | 47 |
| c5 | 16 | 3 | 8 | 303 | 25 |
| c5 | 24 | 4 | 8 | 134 | 41 |
| c6 | 8 | 6 | 8 | 480 | 22 |
| c6 | 16 | 6 | 8 | 106 | 14 |
| c6 | 24 | 6 | 8 | 56 | 6 |
| **合计** | — | **40 / 72** | **72 / 72** | **3,227** | **207** |

每组的新输入检查和输出算术检查也都是 8/8。没有未调用、超时、失败或遗漏案例。全部 72 个都是非结构零的 variable-depth 分支，保存的正 raw log 有限；`all_raw_mass_zero=false`、`zero_raw_cells=0`、`area_prior_used=false`。精确最大值和 `1e-12` 近最大值均只有一个，这仅描述本次保存输出；没有独立真实参考，不能检查或声称跨参考的 argmax 稳定。

R28 的 32 个失败输入在新冻结的 R30 全部完成，原 40 个完成输入也全部完成。R28 的联合错误串是 `positive density underflow/overflow; not a structural zero`，旧日志没有保留触发值的极值；不能将旧 32 例都改称“已测出的 underflow”。本次 30 个含未物化质量的查询都来自旧 32 个失败输入，但节点级失败与整格物化失败并非同一事件。这是表示与可完成性的证据，不是新概率律或科学质量改进。

完整逐查询身份、旧状态、新状态、工作量、质量诊断和算术检查见 [72 行 CSV](/home/liwt/GRoPE/docs/round30_log_rms_applicability_report_data/outcomes_72_rows.csv)；分组见 [9 行 CSV](/home/liwt/GRoPE/docs/round30_log_rms_applicability_report_data/scene_size_9_rows.csv)。R28 的原失败记录保持不变。

## 原始质量与局部 log-sum-exp 丢项

30/72 个查询中共有 3,227/31,104 个原生格的正质量不能物化成严格正的普通 float64；另外 30/72 个查询中共有 207 个可物化但属于 subnormal 的格。这两类格计数不重叠；两类查询集合可以重叠。遗漏的普通索引不代表零质量，subnormal 有值也不保证相对精度。规范质量仍由有限 raw log 保留。

原生格 raw 总质量最小出现在 `c5 / g8 / q431`：log 为 `-63.519821863777715`，由该 log 物化约 `2.592339475e-28`。总 floor 为 `4.32e-28`，raw/floor 约 `0.6000785821`；这是唯一总 raw 低于 floor 的查询。它有 350 个未物化格、3 个 subnormal 格，仍为非结构零，未触发 floor-only 替代。这个输出可以完成和归一化，不等于原始几何分布已经获得独立准确性验证。

全部 raw 总质量 log 的范围为 `[-63.519821863777715, 6.661338147750939e-16]`；上端是总质量约 1 的浮点末位偏差，未裁剪到 1。前方质量 `p+` 在 `[0.9998320421535702,1]`。已记录节点 log density 最低 `-3795.2490795285976`，最高 `0.2162014027317824`；有限 log 的存在不构成误差证书。

| 局部求和范围 | 发生相对 exp 丢项的查询数 | 丢项总次数 | 最大局部绝对遗漏界的 log | 最大局部相对遗漏界的 log |
|---|---:|---:|---:|---:|
| floor | 27 | 2,888 | `-814.7186038410241` | `-745.6410510512028` |
| direct FOV tiles | 23 | 9,041 | `-745.1451229871402` | `-745.1431009139014` |
| native total | 30 | 3,203 | `-741.6952612431492` | `-740.2069145938649` |
| 其余五类范围 | 0 | 0 | 无记录 | 无记录 |

其余五类是 tile nodes、cell tiles、floor total、floored total、content normalization；完整八类分母和最大值对应查询在 [LSE 汇总 CSV](/home/liwt/GRoPE/docs/round30_log_rms_applicability_report_data/lse_scope_8_rows.csv)。表中两类最大界可能来自不同查询。丢项是局部相对 exp 的操作计数，不能跨范围合计为独立格数或失败数；native total 的 3,203 次与普通未物化的 3,227 格也不是同一定义。

这些界相对于已计算的 log 输入，只覆盖相对 exp 变成零的项，是各次局部正和界的最大值；不能拼成整条计算的误差界。它们不覆盖密度、求积、加法吸收或其他舍入。审计独立重算了保存输入可恢复的 native/floor/归一化相关诊断；节点和 tile 内部诊断没有重新积分复核。其他范围无丢项记录也不等于无数值误差。

## 算术检查与成本

| 保存输出检查 | 最大误差 | 原门槛 |
|---|---:|---:|
| P 归一化 | `1.665334537e-15` | `2e-12` |
| 同 raw log 独立重新归一化后的 P | `2.021456070e-30` | `2e-12` |
| 已存 log floor / log P 算术差 | 各 `1.421085472e-14` | `2e-12` |
| 已存 `exp(log P)` 与 P | `0` | `2e-12` |
| 原生格 raw 总和与另算 direct FOV 的绝对差 | `6.493068560e-15` | `1e-8` |

直接 FOV 可加性在等价 log/expm1 形式中按原绝对门槛检查。它比较同一生产积分的两条分区计算路径，不是独立概率参考。R30 **没有**真实 raw 相对误差、独立 P 误差或参考双阶收敛结果；R29 合成通过也不能填补这些缺项。

每个完整调用使用 `1,990,656` 个密度节点（native/direct FOV 各 1,728 tiles，每 tile `24²` 节点），低于每调用 4,000,000 上限；72 个调用共 `143,327,232` 节点。全前向 phase 为 `56.966884283 s`，低于共享 600 秒上限。逐调用加输入加载之和为 `56.357907070 s`，单次范围 `0.698465308–0.872397667 s`；它不含所有逐行记录工作，不能代替 phase 时间。

准备阶段 `0.204474185 s`；输入审计的记录时间跨度约 `0.475320101 s`，最终算术审计汇总时间跨度约 `0.989430666 s`，不与前向 phase 混算。生产峰值 RSS `91,984 KiB` 是进程累计高水位，不是每查询独占内存。R28 前向 phase 为 `23.468733233 s`，但其中 32 个调用提前失败；不能把两个 phase 比值当作等工作量速度比较，也不能由这 72 例外推完整训练成本。

## 记录完整性与未完成的验证

[联合预检](/home/liwt/GRoPE/results/round30_log_joint_preflight_20261031.log)为 68 项纯合成测试通过，并在 metadata-only sentinel 下验证来源冻结；正式管线和根验收均首次通过。13 份来源及不可变 R29 模型都保留双快照。开发记录为[生产端首轮 20 项通过](/home/liwt/GRoPE/results/round30_producer_development_history_20261031.json)与[独立审计首轮 24 项通过](/home/liwt/GRoPE/results/round30_audit_development_history_20261031.json)。

生产端原日志、收据及测试/生产源码已从 `/tmp` 逐字节归档到 [development logs](/home/liwt/GRoPE/results/round30_producer_development_logs/attempt1.json)。归档 helper 首次错读了收据字段名 `returncode`（实际为 `exit_code`），在复制完成后停止；随后只核对已有副本并写历史，没有覆盖原证据或重跑模型。[归档错误记录](/home/liwt/GRoPE/results/round30_producer_archival_helper_development/receipt.json)明确区分执行命令重建的脚本、终端转录错误与同时保存的原日志。这不属于测试失败或正式管线失败。

本轮真实输入已按授权读取，因此 `real_inputs_read=true`；冻结阶段仍为 false。`independent_probability_reference / general_numerical_accuracy_established / labels_read / efficacy_tested / candidate_selected / scientific_gate_passed / goal_complete` 均为 false。**适用性通过只解决这组输入能否在原预算内完成并满足输入身份和保存输出算术合同。** 一般真实数值准确性、区域/中心效果、鲁棒性、实际 V、优化 GS、未触及数据上的联合确认仍未完成，原门槛不得因本轮通过而放宽。

## 附表 SHA256

CSV 只由封存 JSON 提取；LSE 的 null/空白表示没有记录到对应相对 exp 丢项，不表示零误差证书。72 行表的 `lse_diagnostics_json` 保留逐查询全部局部诊断；新旧状态仅用于可完成性对照。

| 文件 | 行数 | SHA256 |
|---|---:|---|
| [outcomes_72_rows.csv](/home/liwt/GRoPE/docs/round30_log_rms_applicability_report_data/outcomes_72_rows.csv) | 72 | `5e21242f2dbd400f7e11b87c5fc80df534bd3e583fd4aa6f86b06b1441e18fc8` |
| [scene_size_9_rows.csv](/home/liwt/GRoPE/docs/round30_log_rms_applicability_report_data/scene_size_9_rows.csv) | 9 | `53297f757694a56d367c73ffaedb338fcd97a4a0069c007241d889ac504bbcbe` |
| [lse_scope_8_rows.csv](/home/liwt/GRoPE/docs/round30_log_rms_applicability_report_data/lse_scope_8_rows.csv) | 8 | `ef9b9b9fc15aa11b854036b8fd74a174b2c9d493a7b26a694f4954297c44f4f7` |

主证据：[全 72 前向记录](/home/liwt/GRoPE/results/round30_log_rms_applicability/forward/records.json)、[全局前向封存](/home/liwt/GRoPE/results/round30_log_rms_applicability/forward_complete.json)、[逐输出独立检查](/home/liwt/GRoPE/results/round30_log_rms_applicability_independent_audit/checks.json)、[审计汇总](/home/liwt/GRoPE/results/round30_log_rms_applicability_independent_audit/summary.json)。
