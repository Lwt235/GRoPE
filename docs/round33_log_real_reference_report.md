# R33：固定真实输入的独立对数参考预检

2026-11-01。**R33 数值预检 FAIL，完整失败证据已通过 root 验收。** 72 个输入全部通过字节身份审计；144 个预定参考调用位置中，20 次实际尝试、19 次完成、1 次因单调用节点预算停止，其余 124 次按预先冻结的 fail-fast 规则跳过。216 项比较完整保留，28 项有完整数值结果并通过，188 项因缺少完整参考结果而失败。成功前缀不能替代全人口通过。

本轮六个管线阶段的 exit code 都为 0，表示准备、调用、失败保存和最终审计按合同完成；它不表示 144 次参考全部成功。`delivery_integrity_passed=true`，但 `numerical_preflight_passed=false`。没有新生产模型前向、标签读取、效果评估或候选晋升；一般数值准确性、科学门槛和研究目标完成标记仍为 false。

本报告只读封存 JSON 和文件字节哈希，没有解码 NPZ、执行积分或读取标签。已独立复核 [root 验收记录](/home/liwt/GRoPE/results/round33_log_real_root_acceptance_20261101.json) 的 32 项文件绑定，生产目录 244 个文件、审计目录 28 个文件的完整库存（均含各自 completion），以及 14 项源码的当前文件和生产／审计双快照。prepared、forward、input-audit 子库存也逐一核对。root 收据 SHA256 为 `d6fd1ba1387d7f20061cb4585de6e4c78b04ac24825c294116d1ab8461bee9cf`。

**固定范围与来源。** 按[协议](/home/liwt/GRoPE/ROUND33_LOG_REAL_REFERENCE_PROTOCOL.md)和[设计](/home/liwt/GRoPE/docs/round33_log_real_reference_design.md)，原样复制 R30 已接受的 prepared 输入，而不是沿其历史 metadata 再去读取 R23 原始档案。人口为 c3/c5/c6 各第一对 `[30,45]`，query 尺度 8/16/24 各 8 个固定索引，共 72 项；每项均有 432 个原生 key。四项 float64 数组为 `mu[3]`、`factor[3,3]`、`bounds[432,4]`、`content_bias[432]`；本轮 bias 全为零，原生 bounds 使用 `(pixel_boundary−.5)/8`。

完整 72 项先复制封存，再独立核验四项 canonical hashes、七项身份字段、原生 bounds 和零 bias。输入审计结束于 `1793463152.9137433`，早于首个调用行开始 `1793463153.43422`；全部 144 个调用位置封存后才读取旧 R30 输出作比较。相机映射、预测深度三维统计的正确性继承 R23/R28/R30 已接受证据，本轮没有重新估计相机、构造 anchor 或重算生产概率。

使用未改动的 R32 独立参考，SHA256 `7ff87df14a54cfb2aaa96fe48b4935b18deedab3a1188fc3bda4c3f2fbd76836`。每项先 32 阶、后 48 阶；保留 full-X 积分、初始深度断点合并、固定 pilot 尺度、log mass、一次面积 floor、原精度和单调用限制。只事先扩大阶段总量至 `36,000,000,000` 节点／`10,800 s`；单调用仍为 `250,000,000` 节点、`32,768` 次深度求值和 `512` 个外层区间。六项旧 identity 与有限 ε 近常深度证据仅继承，不算本轮新增真实输入验证。

**全人口结果。** 下表统计调用位置，每个 scene×size 固定 8 项输入、16 个调用位置。

| 场景 | query size | 实际尝试 | 完成 | 实际失败 | fail-fast 跳过 |
|---|---:|---:|---:|---:|---:|
| c3 | 8 | 16 | 16 | 0 | 0 |
| c3 | 16 | 4 | 3 | 1 | 12 |
| c3 | 24 | 0 | 0 | 0 | 16 |
| c5 | 8 | 0 | 0 | 0 | 16 |
| c5 | 16 | 0 | 0 | 0 | 16 |
| c5 | 24 | 0 | 0 | 0 | 16 |
| c6 | 8 | 0 | 0 | 0 | 16 |
| c6 | 16 | 0 | 0 | 0 | 16 |
| c6 | 24 | 0 | 0 | 0 | 16 |
| 合计 | — | 20 | 19 | 1 | 124 |

按独立输入计，9 项双阶完成、1 项仅 32 阶完成、62 项均未尝试。后 62 项的参考可完成性与误差未知，不能说 c5/c6 或 q24 本身已被测出数值失败。跳过行保存空输出档案、零实际工作和明确的 `attempted=false`，不再读取该调用输入，也不产生 reference-start 事件。此前全 72 项已被 prepare/input-audit 读取，不能把逐调用跳过误写为整个流程从未读取这些输入。

唯一实际失败为第 20 次调用（zero-based index 19）：`harvard_c3__hv_c3_1__g16__q0015__order48`，query 像素 bounds `[48,16,64,32]`。原错误为：

`ReferenceFailure: RuntimeError: full-X partition exceeds remaining fixed inner-node budget before allocation`

已消耗 `249,938,112` 个节点；下一批完整节点分配将超过 `250,000,000`，所以在分配前停止。实际已记账节点没有超过上限。该调用经历 165 次 pilot、3,049 次 quadrature 深度求值，共 3,214 次；其中 inner-X 节点 `121,716,384`，narrow-Y 节点 `128,221,728`。保留原始断点、pilot、固定尺度、最后一次 conditional log mass、partial raw 与 computed mask 等实际证据，没有完整 `log_raw_mass` 或 P。`outer_intervals=0` 表示外层积分未正常返回，不能解释为没有外层求值；`scaled_outer_error_estimate=null` 不能当作零误差。

同一输入的 32 阶已完成，使用 `193,043,744` 节点、3,672 次深度求值和 166 个最终外层区间，约 `32.064461 s`；48 阶失败调用约 `37.284762 s`。一次较低阶完成不能豁免冻结的双阶门槛。已尝试的 20 次调用均未实际删除深度断点；失败输入原／保留内部断点都是 164。这次失败不是 R31 的不可表示 midpoint 问题复发。

**比较覆盖与成功子集误差。**

| 比较类型 | 预定 | 有完整数值并通过 | 缺失导致失败 | P 最大绝对误差 | raw-FOV 最大相对误差 | raw-L1 / reference-FOV 最大值 |
|---|---:|---:|---:|---:|---:|---:|
| 新 32 阶对新 48 阶 | 72 | 9 | 63 | `2.220446049250313e-16` | `4.4408920985006143e-16` | `4.457411859515692e-16` |
| 原 R30 生产对每个新阶数 | 144 | 19 | 125 | `1.4432899320127035e-15` | `1.1990408665951659e-14` | `1.1578930636955128e-14` |

表中最大值仅来自有结果的 9／19 项比较。双阶 P/raw 门槛仍为 `1e-8 / 1e-8`，生产对参考仍为 `1e-4 / 1e-6`。生产最大 P 差来自 c3 g8 q307 的 48 阶比较，raw 两项最大差来自 c3 g8 q431 的 48 阶比较。全部 188 项失败记录的实际原因为 `KeyError: 'log_raw_mass'`：其中 2 项依赖唯一实际失败的参考结果，另 186 项依赖 fail-fast 未执行结果。它们不是 188 次独立精度超限，CSV 的缺失误差留空，不补零。

已完成的 19 次参考全部通过保存输出算术、尺度恢复、深度尾界和 direct-FOV 检查；另外 125 个位置没有可通过的完整结果。成功子集的最大归一化误差为 `8.881784197001252e-16`，最大 native sum 对单独 direct-FOV 绝对差为 `8.881784197001244e-16`，原门槛分别为 `2e-12 / 1e-8`；保存尺度恢复残差为 0。28 项有结果的比较均没有 argmax 变化，也没有 exact 或 `1e-12` 近并列；该观察仅覆盖当前成功前缀，不能外推剩余人口。

成功参考的 native raw 总质量 log 范围为 `[-2.581914726936636, 5.551115123125783e-16]`，没有整个 FOV 极小质量样例；最大值的微小正数是保留的浮点结果，没有事后裁为零。总面积 floor 的 log 为 `−63.00912720157726`。这不能说明每个远端 cell 都可普通物化，也不能覆盖失败或跳过项的 tiny-tail 精度。CSV 保留 log 值，报告没有从 partial 构造完整 raw/P。

成功子集的最大深度截断 raw 比值为 `1.2879902770760522e-71`，最大带偏置 P 界为 `9.019397147163708e-45`，低于原 `1e-10 / 1e-8` 门槛。这些只约束固定深度范围外的遗漏，raw 分母为计算出的保留 FOV 质量，并非区间包围值。最大 scaled 外积分误差估计为 `4.799684017264721e-8`；它是 scaled vector 的估计，不能直接与 unscaled raw/P 门槛相比较，也不是总误差证书。

20 次实际尝试（含失败一次）累计记录 `1,186,309` 个 outer-exp 相对下溢分量事件；LSE 的 rectangle-panels、native-total、floor 分别记录 `376,012 / 1,050 / 974` 个丢失项，最大局部 log-relative omitted bound 分别为 `−741.763509787297 / −742.1785395258281 / −745.7194124065979`。这些是重复求值的局部诊断，不能当作独立查询计数、完整积分质量损失或全流程证书。pilot 尺度不是全域上界；参考比较成功也不构成一般参数准确性证明。

**资源与失败边界。** 完整参考阶段耗时 `376.8084569256753 s`，总 charged nodes `1,903,770,272`，阶段时间和节点预算都尚有余量。停止原因是单调用预分配预算，不是共享截止时间。全部保存行时间之和为 `375.6894925162196 s`；19 次成功参考函数自报时间之和为 `338.0540315862745 s`，均不能代替阶段总时间或冒充 20 次实际积分时间。最大进程 RSS 为 `125,404 KiB`，是累计高水位。20 次实际尝试的 pilot／quadrature 深度求值累计 `1,564 / 33,562`；不同调用和阶数共享输入，不能当作独立样本。不能按前缀平均耗时推算未执行 124 次调用的成本。

**开发与封存记录。** [contract 开发历史](/home/liwt/GRoPE/results/round33_contract_development_history_20261101.json) 保留两次 metadata probe：首次错误假设旧 input-check 含全部七个身份字段；修正后仅检查其实际四字段，同时 prepared/spec/production 仍完整互核七字段。第二次在禁止 `np.load` 下验证 667 项绑定、72 个 specs，通过且没有数组解码或积分。

[runner 历史](/home/liwt/GRoPE/results/round33_preflight_development_history_20261101.json) 的 36 项 mock 首轮通过；[auditor 历史](/home/liwt/GRoPE/results/round33_audit_development_history_20261101_v2.json) 保留首轮 43 项通过及补强后的 48 项通过。补强只使未尝试的 `input_decode`／IO 失败明确归为交付失败，保留完整人口，不修改参考数学或精度门槛。本轮实际 input-delivery failure 为 0。[联合历史](/home/liwt/GRoPE/results/round33_joint_preflight_development_history_20261101.json) 保留首次 79 项通过及最终 **84 项通过、14 项来源冻结**；正式执行绑定最终 v2，没有在正式结果后修补源码或重跑积分。

测试来源快照在对应测试前保存；控制器快照时序则须分别陈述：[capture.json](/home/liwt/GRoPE/results/round33_controllers/capture.json) 明确记录 joint 控制器在预检完成后复制、execution 控制器在正式运行启动后复制（记录为启动以来未编辑）、root 控制器在首次验收前复制。不能将这三份统一称为执行前快照。[正式执行记录](/home/liwt/GRoPE/results/round33_log_real_execution_20261101.json) 和原日志均保留，六阶段均 exit 0；root 只重核保存证据，积分重放次数为 0。

完整附表为 [72 项输入](/home/liwt/GRoPE/docs/round33_log_real_reference_report_data/cases72.csv)、[144 个调用位置](/home/liwt/GRoPE/docs/round33_log_real_reference_report_data/reference_calls144.csv)、[216 项比较](/home/liwt/GRoPE/docs/round33_log_real_reference_report_data/comparisons216.csv)。另有[汇总统计](/home/liwt/GRoPE/docs/round33_log_real_reference_report_data/statistics.json)和[提取完成记录](/home/liwt/GRoPE/docs/round33_log_real_reference_report_data/completion.json)；纯标准库提取器在执行前已保存[源码快照](/home/liwt/GRoPE/docs/round33_log_real_reference_report_data/source_snapshot/grope_round33_report_extract.py)。该提取只读取 JSON 和文件字节，首轮完成，未修改原结果。

| CSV | 行数 | SHA256 |
|---|---:|---|
| `cases72.csv` | 72 | `d84b5f79a8b8280426178e55b205a150cb1e7818ff5184db76b0962c234bb968` |
| `reference_calls144.csv` | 144 | `2da7078d8784d86218e73192748137771158f5ad2facbf6018b755c2ddd2838f` |
| `comparisons216.csv` | 216 | `35921c2e7c96f48729d2384f822e02e7f5839ea5fc6477a666c8ec1a94598198` |

本轮保留的结论是：固定参考在成功前缀上与旧生产输出一致，但未在原单调用资源限制内完成全部真实输入，因此不能进入效果解释。R32 合成通过、R30 的 72/72 生产可完成性及 R33 的不完整独立参考是不同证据；R23/R26/R28/R31 失败与既有科学效果失败均不被覆盖。任何后续修订须另行说明、冻结与验证，原门槛继续有效。
