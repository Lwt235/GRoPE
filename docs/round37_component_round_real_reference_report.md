# R37：分量舍入参考在全部固定真实输入上的复验

2026-11-01。**R37 数值预检 FAIL，失败交付已通过 root 验收。** 72 项输入全部通过字节身份审计；144 个预定参考调用位置中，56 次实际尝试、55 次完成，1 次在初始外积分尚未完成时触及单调用节点预算，随后 88 次按冻结的 fail-fast 规则跳过。296 项比较全部保留：162 项有完整数值结果并通过，134 项因参考缺失而未通过。没有已完成比较的数值误差超限，但完整人口仍未通过。

正式六阶段及 root 验收成功退出，含义是交付与审计正常完成；`delivery_integrity_passed=true`、`numerical_preflight_passed=false`。收据的 `accepted_for_numerical_interpretation=true` 允许解释这些证据，不代表数值通过。本轮未重新运行生产模型，未读取标签，未选择 anchor、内容温度或效果候选，未开展科学效果验证。一般数值准确性、科学门槛与研究目标完成标记均为 false。

本报告只读已接受的 JSON 与文件字节哈希，没有解码 NPZ、重放积分或执行新测试。已核对 [root 收据](/home/liwt/GRoPE/results/round37_component_round_real_root_acceptance_20261101.json) 的 34 项文件绑定、19 项当前来源与生产/审计双快照，以及生产 250、审计 34 个文件的完整库存，另核 prepared 75、forward 149、input-audit 3 和控制器 7 个文件的子库存；这些数字均包括相应 completion。root 收据 SHA256：`0f7552a673de1422b215972c105eccddeb0fc5119b8716fa1168aded5570caf5`。

**固定输入与唯一方法差别。** [设计](/home/liwt/GRoPE/docs/round37_component_round_real_reference_design.md) 和 [协议](/home/liwt/GRoPE/ROUND37_COMPONENT_ROUND_REAL_REFERENCE_PROTOCOL.md) 保留原 R33/R35 的全部 72 项、原顺序与每项 432 个 key：c3/c5/c6 各第一对 `[30,45]`，query 尺度 8/16/24 各 8 个固定索引。仍只复制 R30 已接受的四项 float64 输入 `mu[3]`、`factor[3,3]`、`bounds[432,4]`、`content_bias[432]`，保留七项身份字段。bias 为零，没有重建三维矩、相机映射或 anchor。72 项准备输入全部封存，再经独立输入审计，才执行参考；输入审计完成于 `1793468179.4686203`，参考阶段开始于 `1793468180.660653`。旧生产与旧参考输出仅在全 144 个结果位置封存后由独立审计解码。

R37 直接导入已经通过 R36 合成检验的参考与辅助积分器，没有再次修改它们：

- [REF36](/home/liwt/GRoPE/experiments/round36_component_round_reference.py)，SHA256 `6e09a0d0cfb4e142ab6390819af91e30b538b6874856a296fa6b130f76049725`。
- [分量舍入积分器](/home/liwt/GRoPE/experiments/round36_component_round_quadrature.py)，SHA256 `7af6ba236e9e17fdc64d5798da17112b2f4cc25d945f0b407d6fe1ebb80ffa37`。

相对 R35，概率法则、SVD 条件化、full-X、GL8 窄正态区间、pilot、GK21 节点、深度域与外部误差门槛保持。外层积分改用当前活动叶的逐分量 `fsum`，误差估计为

\[
E=\sum_{\ell\in\mathrm{active}}e_\ell+
\max_j\sum_{\ell\in\mathrm{all\ completed\ history}}r_{\ell j},
\qquad
\mathrm{target}=\max(\mathrm{epsabs},\mathrm{epsrel}\|I\|_\infty)/8.
\]

严格要求完整当前分区且 `E < target`；目标中已经含 `/8`。被替换的父区间和失败时已完成的孤立子区间仍计入历史舍入项。全部初始区间完成前不能检查收敛或提前接受。该估计不是连续积分或浮点全流程误差证书。

每项按 32、48 阶依次执行；单调用仍限 250,000,000 节点、32,768 次深度求值与 512 个活动外层区间，整个阶段仍限 36,000,000,000 节点及 10,800 秒。原 216 项比较全部保留，另加入 80 项严格历史回归：R33 的 19 份已完成输出与两个新阶数比较，共 38 项；R35 的 21 份已完成输出共 42 项。共 40 份 oracle 以 `(id, oracle_id)` 区分来源，重叠的 query/order 不覆盖，也不是额外独立样本。旧失败 partial 与 skipped 结果不作 oracle。六项 identity 与有限 ε 的近常深度证据仅继承，没有新 identity 调用。

**全人口结果。** 每个 scene×size 固定 8 项输入、16 个调用位置。

| 场景 | query size | 实际尝试 | 完成 | 实际失败 | fail-fast 跳过 |
|---|---:|---:|---:|---:|---:|
| c3 | 8 | 16 | 16 | 0 | 0 |
| c3 | 16 | 16 | 16 | 0 | 0 |
| c3 | 24 | 16 | 16 | 0 | 0 |
| c5 | 8 | 8 | 7 | 1 | 8 |
| c5 | 16 | 0 | 0 | 0 | 16 |
| c5 | 24 | 0 | 0 | 0 | 16 |
| c6 | 8 | 0 | 0 | 0 | 16 |
| c6 | 16 | 0 | 0 | 0 | 16 |
| c6 | 24 | 0 | 0 | 0 | 16 |
| 合计 | — | **56** | **55** | **1** | **88** |

按输入计，27 项双阶完成，1 项仅 32 阶完成，44 项两个阶数均未尝试。后 44 项的参考可完成性和精度未知，不能称为它们各自失败，也不能据此比较场景难度。跳过行保存空输出档案、零实际工作及明确 stop trigger，没有调用阶段的新输入读取或积分；这不否认全部 72 项早已在 prepare/input-audit 中读取。

**实际失败：初始分区的成本，非已测数值不一致。** 唯一失败是 zero-based index 55，即第 56 次调用：

`harvard_c5__hv_c5_1__g8__q0184__order48`

像素 query bounds 为 `[128,56,136,64]`，原错误为：

`ReferenceFailure: RuntimeError: full-X partition exceeds remaining fixed inner-node budget before allocation`

已计费 **249,980,432** 节点，约 **20.464116 s**。下一个分配所需的固定工作超过剩余单调用预算，因此在超过上限前停止；不能因为计费数略小于 250M 就说预算尚足。工作分解为 32,230,512 个 inner-X 节点和 217,749,920 个窄 Y 正态节点，后者约占 87.1%。这些是实际已计费工作，尚不是把完整该调用跑完的成本。

该调用完成 133 次 pilot；初始外积分应有 133 个区间，实际只完成其中 56 个 GK21 区间，尚未完成第 57 个。保存的 quadrature callbacks 为 1,184，等于 `56×21+8`；失败发生在初始区间遍历中，`split_count=0`、`initial_checked_before_split=false`、`complete=false`。待完成区间是 `[-6.970043193627319, -6.959142391112373]`。

保存 prefix 的 `E=6.043334846392533e−15`、`target=6.799559666088495e−14`，比值约 0.088878。**这个比值只来自不完整初始域，不能当作已收敛，更不能提前截断积分域接受。** 独立 trace 审计通过仅表示已完成片段、历史舍入、工作计数和失败位置自洽；`strict_convergence=false`，没有放宽原要求。

失败档案保留四项输入、pilot、尺度、最后一次条件质量、五项外层 trace、partial raw 与 computed mask。它没有合格的完整 `log_raw_mass`、floor/P 或尾界结果；保留 partial 不等于补造零概率，也不能作为下一轮 oracle。本报告不重新解码这些数组，依据 accepted audit 与其保存的字段/哈希记录解释。

同一输入的 32 阶完成：231,684,400 节点、约 25.910356 秒，133 个初始区间、零次细分；完整 `E=1.7301116311163592e−13 < target=3.3814615984329404e−12`。单阶完成不能替代冻结要求中的双阶一致性检查。

原 R33 曾失败的 c3 g16 q0015、48 阶在本轮完成，用 222,383,856 节点；原 R35 的 c3 g16 q0030、48 阶也完成，用 153,104,016 节点，`E≈3.44267e−14 < target≈3.27372e−13`，115 个初始区间即停止。两项新双阶和生产比较通过，旧失败档案保持原样。R36 的改变修复了已知的 R35 外层舍入停止问题，但没有保证所有真实输入都能在原成本内完成。

**全部比较与误差范围。**

| 比较组 | 固定数量 | 有完整数值并通过 | 缺失导致未通过 | 最大 P 绝对误差 | 最大 raw-FOV 相对误差 | 最大 raw-L1 / reference-FOV |
|---|---:|---:|---:|---:|---:|---:|
| 新 32 对新 48 | 72 | 27 | 45 | `6.66134e−16` | `4.44089e−16` | `1.15868e−15` |
| 原 R30 生产对新参考 | 144 | 55 | 89 | `1.47105e−15` | `1.11022e−14` | `1.06899e−14` |
| 新参考对旧 R33 | 38 | 38 | 0 | `3.33067e−16` | `1.33227e−15` | `1.33480e−15` |
| 新参考对旧 R35 | 42 | 42 | 0 | `3.33067e−16` | `1.33227e−15` | `1.33480e−15` |
| 合计 | **296** | **162** | **134** | — | — | — |

参考间 P/raw 门槛仍为 `1e−8 / 1e−8`，生产对参考仍为 `1e−4 / 1e−6`。162 条可比记录全部通过，且无 argmax 变化、exact tie 或 `1e−12` 近并列；这是已完成前缀及重复历史回归的证据，不覆盖未完成输入。134 条缺失比较均明确记录 `KeyError: 'log_raw_mass'`，不是 134 次数值误差超限，误差列留空。80 项历史回归通过不能抵消原 216 项中的缺失。

55 次完成参考的保存输出算术、direct-FOV、尺度恢复和深度尾门槛均通过。最大归一化误差 `8.881784197001252e−16`，native sum 对独立 direct-FOV 最大绝对差 `9.98478362648844e−16`。独立重算 log-floored/log-P 最大差均为 `1.4210854715202004e−14`，P 最大差 `2.498001805406602e−16`，尺度恢复残差为 0。成功参考 native raw 总质量的 log 范围为 `[-2.581914726936637, 5.551115123125783e−16]`，微小正值未裁剪。

成功子集最大深度尾 raw-L1/FOV 界 `1.2879902770760522e−71`，最大带 bias 的 P 界 `9.019397147163708e−45`。这些只界定深度截断贡献；分母来自计算出的保留质量，不是严格区间包围。GL8 的 `1.88e−13` 是局部精确算术余项，component error 也是数值估计，两者都不是整个概率算子的总误差证书。

56 次尝试均保留 component trace：55 个完整并收敛、1 个不完整；完整调用中 26 个无需初始分区之外的细分。所有 trace 共 4,685 个已完成区间评估、4,624 个当前活动区间、61 次已提交细分和 98,393 次 callbacks；失败区间的部分工作已计费，但不伪造已完成 trace 行。最大 `E/target≈0.941205`，最大历史舍入项/target≈0.355160。历史区间与当前活动区间不能混同，也不能把失败 prefix 的比值混入“完整收敛”的判定。

56 次尝试累计 outer-exp 局部下溢事件 3,929,219。LSE rectangle-panels/native-total/floor 分别记录 881,635 / 2,736 / 2,458 个局部丢失项，最大 log-relative omitted bounds 分别约 `−741.763510 / −741.698295 / −745.719412`。这些是重复求值的局部诊断，不能相加当成整个积分的误差，也不是独立样本数。全部实际尝试的断点合并均未删除点，本轮没有重现不可表示初始中点问题。

**成本与先验下界的关系。** 参考阶段总耗时 **1,080.166745 s**，计费 **4,931,562,096** 节点，未耗尽阶段预算。停止原因是某一调用的 250M 节点限制。所有行 wall 时间合计 1,077.866144 秒；55 次完整参考自报时间合计 1,055.071694 秒，失败调用没有该自报时间字段，缺失不当作实际耗时为零。三种时间口径分别保留。最大 RSS 107,568 KiB 是进程累计高水位，最大单行时间约 44.752763 秒。

实际工作共 4,640 次 pilot 与 98,393 次 quadrature 深度求值。节点分解为 3,957,924,704 个 inner-X 和 973,637,392 个窄 Y 节点。完成区间数、活动区间数、callback 数和收费节点是不同计数，不能互相替代。

执行前的 [最低 X 准入成本诊断](/home/liwt/GRoPE/docs/round36_real_initial_cost_lowerbound_report.md) 对全部 72 项没有找到必然超预算的反例，最高 48 阶下界为 154,790,736；其结论明确是 **unknown**，未包含 Y 断点、窄 CDF 或自适应工作。本轮实际失败不与该下界矛盾，不能把低于上限的必要下界当成可完成性保证。

同 query/order 的成本对照另在 [40 项历史配对表](/home/liwt/GRoPE/docs/round37_component_round_real_reference_report_data/matched_historical_costs40.csv) 保留。19 份 R33 对照的旧/新节点合计为 1,653,832,160 / 1,416,287,344；21 份 R35 为 1,770,615,560 / 1,740,740,608。两组互有重叠，不能相加成为 40 个独立新样本；它们都是历史成功子集，不能将其节省比例外推到全部 72 项，更不能拿不同停止位置的阶段总时间作完整任务加速比。

**开发、提取与可追溯交付。** [runner 开发索引](/home/liwt/GRoPE/results/round37_preflight_development_history_20261101.json) 记录首轮 46 个 mock 测试通过、2.92 秒，19 项来源测前快照及 before/after 哈希一致；[审计开发索引](/home/liwt/GRoPE/results/round37_audit_development_history_20261101.json) 首轮 87 项通过；[联合预检索引](/home/liwt/GRoPE/results/round37_joint_preflight_development_history_20261101.json) 记录 133 项通过、19 项来源与 metadata 冻结。联合测试是同一集合重验，不是新增独立科学证据。来源与命令均在相应测试前留档，正式过程中没有修改来源、重试失败调用、删除后续位置或放宽门槛。

三控制器执行前源码与正式完成情况由 [控制器库存](/home/liwt/GRoPE/results/round37_controllers/completion.json) 绑定，SHA256 `16680dde4419213e9fbdfcf543d52269004c6441c56f3fdd7f86c8b773c55122`。本报告提取也先封存源码、命令及已验收输入哈希。首轮因沿用 R35 控制器索引字段而在生成 CSV 前退出；只修正该后处理字段，第二轮完成。两轮源码、日志、命令和退出码均保留，没有改动正式证据、解码 NPZ、执行积分或新增测试。

完整附表：[72 项输入](/home/liwt/GRoPE/docs/round37_component_round_real_reference_report_data/cases72.csv)、[144 个调用位置](/home/liwt/GRoPE/docs/round37_component_round_real_reference_report_data/reference_calls144.csv)、[296 项比较](/home/liwt/GRoPE/docs/round37_component_round_real_reference_report_data/comparisons296.csv)、[40 个来源区分的 oracle](/home/liwt/GRoPE/docs/round37_component_round_real_reference_report_data/oracles40.csv)、[56 份外层 trace 汇总](/home/liwt/GRoPE/docs/round37_component_round_real_reference_report_data/outer_trace_summaries.csv)、[统计 JSON](/home/liwt/GRoPE/docs/round37_component_round_real_reference_report_data/statistics.json) 和 [提取完成记录](/home/liwt/GRoPE/docs/round37_component_round_real_reference_report_data/completion.json)。提取器的 `real_inputs_read=false` 只表示本次报告没有解码数组，不覆盖原实验真实读取标记。

本报告截止于正式 R37 root 验收，不纳入任何后续失败诊断。R36 合成通过、R30 的 72 项生产可完成性和 R37 的部分真实参考一致性，各自作用范围不同。完整独立真实参考仍未通过；本轮没有提供区域质量、中心定位、扰动、实际 V、实际 GS 或未触碰数据上的新效果证据，也没有改变那些原有验证要求。
