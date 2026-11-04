# R36：逐分量累计外层舍入估计的合成参考预检

2026-11-01。**固定合成预检 PASS：23 个原输入、46 次新参考和全部 339 项比较通过。** 新外层驱动保留 GK21 标量截断估计，将历史舍入向量逐分量累计后取最大值，并从当前叶区间重新求和。结果支持该修订在这组固定合成人口上的数值可用性；不代表全部真实输入已通过，更不代表位置编码效果通过。

R35 的真实预检仍是 **21 次完成、1 次舍入收敛失败、122 次未尝试**。其正式失败、单独的一次失败后诊断及 partial 均原样保留，没有被本轮改写为通过或升级为 oracle。本轮未重跑生产模型、读取真实输入或标签；报告本身只读封存 JSON 和文件字节，没有 NPZ 解码或积分调用。

已核对 [root 验收记录](/home/liwt/GRoPE/results/round36_component_round_root_acceptance_20261101.json) 的 **33 项文件绑定、19 项当前来源及生产/审计双快照**，以及生产 **103**、审计 **33** 个文件的完整库存；prepared、forward、input-audit 和控制器子库存也通过，文件数包含各自完成记录。root receipt SHA256：`e12d0e38dbbc9bf4381239eb319d9da0b7ffbfc9bc931e6deeeb21dbfed80fa8`。

**算法修订。** 按 [设计](/home/liwt/GRoPE/docs/round36_component_round_reference_design.md) 与 [协议](/home/liwt/GRoPE/ROUND36_COMPONENT_ROUND_REFERENCE_PROTOCOL.md)，保留原 RMS 概率 law、SVD、完整有限 X 域、GL32/48、窄区间 GL8 和跨度 0.25、深度域与中点合并、固定 pilot 尺度及所有 raw/P、尾界和资源门槛。GK21 节点、权重、标量 Gauss–Kronrod 差与 resasc 非线性修正保持；外层停止和调度的改动则明确如下。

每个完成区间保留积分向量、施加旧标量舍入 floor 前的截断估计 `e_l`，以及舍入向量 `r_l=50 ε h_l Σ_n w_n |f_n|`。每轮以 `math.fsum` 重算当前叶积分、当前叶标量截断项之和及**全部历史区间**的逐分量舍入和，包含已经被细分的父区间。新的停止量为

`E = Σ_active e_l + max_j Σ_history r_lj < max(epsabs, epsrel × ||I||∞) / 8`。

`epsabs=0`，32/48 阶 `epsrel=1e−11 / 1e−12` 不变。初始分区全部求完后先检查停止条件；需要细分时，每次选原每叶优先级 `max(e_l,max_j r_lj)` 最大的一项，平局取最左区间。两个子区间都完成后才替换父区间。它同时改变了原批量细分和增量 `children−parent` 累计，不能描述为与原 SciPy 控制流逐条一致。

这些舍入及截断量仍是**估计**。保存 trace 可检验调度、计费和累计是否遵循约定，不把启发式误差估计变成真实积分误差、全浮点误差或全流程严格证书。

**完整人口与回归。** 四项输入及描述符逐字节复用 R29；独立输入审计先于新参考，全部 46 个结果全局封存后才解码旧输出。135 份旧 oracle 包括 R29 的 40 份双阶数值结果与 3 份解析结果、R32 的 46 份、R34 的 46 份。索引包含来源与阶数，未合并不同来源的同阶记录。

| 固定检查或比较 | 数量 | 结果 |
|---|---:|---|
| 输入及描述符独立审计 | 23 | 全通过 |
| 新 32/48 阶调用、交付及算术检查 | 46 | 全完成、全通过 |
| 新双阶互比 | 23 | 全通过 |
| 新对 R29 旧参考 | 86 | 全通过 |
| 新对 R32 旧参考 | 92 | 全通过 |
| 新对 R34 旧参考 | 92 | 全通过 |
| 原 R29 生产对新参考 | 46 | 全通过 |
| 比较合计 | **339** | **全通过** |

339 项比较共享输入与输出，不是 339 个独立样例；改变偏置的 base 也不等于独立几何。六项旧 identity 和有限 ε 的近常深度结果只作为继承证据，没有新增 identity 或 limit 调用。完整人口见 [23 例汇总](/home/liwt/GRoPE/docs/round36_component_round_reference_report_data/cases23.csv)、[46 次调用](/home/liwt/GRoPE/docs/round36_component_round_reference_report_data/reference_calls46.csv)、[339 项比较](/home/liwt/GRoPE/docs/round36_component_round_reference_report_data/comparisons339.csv) 与 [135 份旧参考](/home/liwt/GRoPE/docs/round36_component_round_reference_report_data/oracles135.csv)。

**数值误差、极小质量与并列。** 下表是完整规定比较人口的最大值；结构零不适用 raw 相对除法。

| 比较 | P 最大绝对差 | raw-FOV 最大相对差 | raw-L1 / reference-FOV 最大值 | P / raw 门槛 |
|---|---:|---:|---:|---|
| 新双阶 | `2.220446049250313e−16` | `1.421085471520220e−14` | `5.314185147598529e−15` | `1e−8 / 1e−8` |
| 新对 R29 | `1.484923295436147e−15` | `1.136868377216263e−13` | `1.136868377216263e−13` | `1e−8 / 1e−8` |
| 新对 R32 | `2.220446049250313e−16` | `1.421085471520220e−14` | `5.314185147598529e−15` | `1e−8 / 1e−8` |
| 新对 R34 | `2.220446049250313e−16` | `1.421085471520220e−14` | `5.314185147598529e−15` | `1e−8 / 1e−8` |
| 原生产对新参考 | `3.330669073875470e−16` | `1.136868377216263e−13` | `1.136868377216263e−13` | `1e−4 / 1e−6` |

新双阶及新对 R32/R34 的最大 P 差首见于 `07_radial_noncentral`，raw 最大差来自 `12_tiny_front`；新对 R29 最大 P 差来自 `18_tiny_front_bias`，原生产对新参考最大 P 差来自 `06_radial_central`。后两组 raw 最大差来自 `28_subfloat_fov`。这些严格回归通过不等于输出逐位相同，也没有加入逐位输出门槛。

46 次结果的最大归一化误差为 `3.9968028886505635e−15`，native raw 和单独 direct-FOV 的最大绝对差为 `5.551115123125763e−16`，低于 `2e−12 / 1e−8`。独立 floor/P 算术最大 P 差为 `2.220446049250313e−16`，log-floored 与 log-P 最大差均为 `1.4210854715202004e−14`。保存尺度恢复残差为 0，只证明中间量恢复一致，不能替代独立积分比较。

`04_zero_depth` 和 `05_negative_constant` 是解析结构零，raw 为零且积分工作为零；其余正质量例没有因 floor 主导而免除 raw 门槛。`12_tiny_front` 的 32 阶 native log raw 为 `−81.00177513658565`，约 `6.6279013206559e−36`；总 floor 约 `1.6e−29`，raw/floor 约 `4.1424383e−7`，前方质量约 `1.776482112077654e−33`。`18` 复用该几何但改变偏置。`28_subfloat_fov` 两阶 native log raw 均为 `−765.4648717106797`，数学质量约 `3.6545181680e−333`，低于正 float64 可表示范围；总 floor 约 `2.5e−31`。它仍通过对数 raw 相对检查，没有以普通零或 floored P 接近代替原门槛。

| 比较组 | argmax 变化记录 | 至少一侧精确并列 | 至少一侧 `1e−12` 近并列 |
|---|---:|---:|---:|
| 新双阶 | 2 | 4 | 4 |
| 新对 R29 | 4 | 14 | 14 |
| 新对 R32 | 4 | 16 | 16 |
| 新对 R34 | 4 | 16 | 16 |
| 原生产对新参考 | 2 | 8 | 8 |

全部 **16** 条 argmax 变化只涉及 `01_point` 和 `25_wide_constant_axis`；变化索引均在双方近并列集合 `[5,6,9,10]` 或 `[0,1,2,3]` 内。这是重复比较记录数，不是独立查询数。小 P 误差不保证 argmax 稳定，相关变化没有被隐藏。

**外层 trace 与局部界。** 12 个变深度 base 的双阶共 24 次调用有完整五项 trace；其余常深度或结构零分支不生成外层 trace。保存数组的独立审计全部通过，覆盖分区、确定的调度顺序、非负性、当前叶 `fsum`、包含父区间的历史累计、严格停止条件和实际回调计费。报告仅提取这些已接受的 JSON 核验结果，没有再次解码 trace 或重算积分。

| trace 汇总 | 实际值 |
|---|---:|
| 变深度调用数 | 24 |
| 初始区间总数 | 726 |
| 完成区间求值总数（含历史父区间） | 788 |
| 最终当前叶区间总数 | 757 |
| 已提交二分总数 | 31 |
| 初始分区完成即通过、无需细分的调用 | 11 |
| GK21 深度回调总数 | 16,548 = 21 × 788 |
| 最大 `E / target` | `0.9512483481542431`，12、32 阶 |
| 最大历史舍入项 / target | `0.45578537156824006`，11、48 阶 |

其余 13 次变深度调用进行了细分；最大单调用细分 7 次。最大 scaled 估计为 `2.9347188189692586e−11`，对应 target `3.085123695261755e−11`；这些带 pilot 尺度的量不能直接与 raw/P 门槛比较。`13_native432` 两阶均在 198 个初始区间完成后停止，没有细分。逐调用详情见 [外层 trace 汇总](/home/liwt/GRoPE/docs/round36_component_round_reference_report_data/outer_trace_summaries.csv)。

中点合并沿用原规则，仅 `07` 两阶各将内部断点从 16 减为 14，均因前向严格中点不可表示；其余没有删除，上端点回退也未发生。全调用保留内部断点总数为 702，没有删掉积分域。

深度截断遗漏质量最大为 `1.9481897837873628e−72`，raw-L1/FOV 尾界最大 `1.4696882840695367e−37`，带偏置 P 尾界最大 `2.4352372297342188e−43`，满足原 `1e−10 / 1e−8` 相对 raw/P 截断门槛。它们只约束深度域外贡献，分母为计算出的保留 FOV 质量；不包含内外层求积或浮点误差。GL8 的 `1.88e−13` 仍只是局部精确算术余项界。

外层 exp 共记录 **405,026** 个相对下溢分量事件。LSE 的 rectangle-panels、native-total、floor 分别记录 **176,322 / 232 / 196** 个丢失项，最大局部 log-relative omitted bound 分别为 `−742.3201002023112 / −766.4849498197723 / −746.0814938246217`。这些重复局部事件不能相加当作全流程质量损失；pilot 不是全域上界。全有限 X 域、双阶一致、trace 一致和局部界共同通过，仍不是一般参数域正确性的证书。

**资源与旧工作量。** 本轮参考正式执行一次，完整阶段 **103.4571019988507 s**，charged nodes **374,023,888**，低于原合成阶段 `1800 s / 600,000,000` 限额。

| 工作项 | R34 保存值 | R36 保存值 |
|---|---:|---:|
| 总 charged nodes | 379,773,536 | 374,023,888 |
| X 节点 | 347,245,696 | 341,496,048 |
| 窄正态 PDF 节点 | 32,527,840 | 32,527,840 |
| pilot 深度求值 | 726 | 726 |
| quadrature 深度求值 | 17,010 | 16,548 |
| 最终叶区间总数 | 768 | 757 |

11 次调用的计费减少，其余相同；总节点少 **5,749,648，约 1.514%**。这里汇总的是保存账本，不是重跑旧算法。最大单调用 `13_native432`、48 阶为 **202,674,880** 节点、约 **47.286943 s**，低于 `250M`；最大深度求值 4,356、叶区间 198，低于 `32,768 / 512`。最大 RSS **105,988 KiB** 为累计进程高水位。行时间合计 `103.16842853277922 s`、参考函数自报时间合计 `102.86297670006752 s`，均不代替完整阶段时间；最后 records 写入之后的来源/库存收尾另计。

R34 阶段为约 108.441426 s，两次观察相差约 4.98 s；这不是重复计时或受控性能基准，不能据此承诺真实输入更快或全部成功。本轮未调用 R35 已知失败的真实输入，也未证明该失败已修好。

**开发、封存与交付。** [参考开发历史](/home/liwt/GRoPE/results/round36_reference_development_history_20261101.json) 首轮 53 项通过、2 项解析 ULP 断言失败；原快照和日志保留，随后只修正这两项测试的显式 ULP 容差，第二轮 55 项全部通过，参考实现与正式阈值未改。[runner](/home/liwt/GRoPE/results/round36_preflight_development_history_20261101.json) 32 项 mock 首轮通过，[审计](/home/liwt/GRoPE/results/round36_audit_development_history_20261101.json) 67 项首轮通过；[联合预检](/home/liwt/GRoPE/results/round36_joint_preflight_development_history_20261101.json) **154 项通过、19 项来源冻结**，是同一集合的重验，不是额外 154 项独立实验。

测试前保存了当时全部可用源码、测试和命令，尚未完成的并行文件在历史中明确列出；正式联合冻结包含完整来源。[合同开发检查](/home/liwt/GRoPE/results/round36_contract_development_history_20261101.json) 首轮 metadata-only 通过 1,411 项绑定、23 个输入和 135 个旧 oracle；禁止数组读取和积分的冻结检查也通过。AST 检查确认原 R34 顶层函数/类除 `_check_policy`、`integrate` 外保持，新增外层 helper 单独进入完整来源闭包。

[三份控制器](/home/liwt/GRoPE/results/round36_controllers/before_execution.json) 均有执行前源码快照，[完成记录](/home/liwt/GRoPE/results/round36_controllers/completion.json) 确认源码未变、联合预检/正式流程/root 均 exit 0，正式六阶段均 exit 0。没有封存后的数值修补或正式重试。报告提取器同样先保存 [源码与命令](/home/liwt/GRoPE/docs/round36_component_round_reference_report_data/before_execution.json)，首轮执行成功；[完成记录](/home/liwt/GRoPE/docs/round36_component_round_reference_report_data/completion.json) 和 [统计 JSON](/home/liwt/GRoPE/docs/round36_component_round_reference_report_data/statistics.json) 绑定各附表。

| 交付 CSV | 行数 | SHA256 |
|---|---:|---|
| `cases23.csv` | 23 | `eaad22868b8762062ba2a7d18adf4fda9b5534b832fac47ab9c8910b67ae2d4b` |
| `reference_calls46.csv` | 46 | `cd7a22f2df68555e019c6c87bf71349830b89d59c697a62d816ec1ff6b893909` |
| `comparisons339.csv` | 339 | `da7d7debce3a55171b6483c260da775fe864683d466c412d69f08a61637b30ec` |
| `oracles135.csv` | 135 | `6f184b8352ca21d2c4e6c684c49bb8ce8bf5e2b6954b02271b314eb3ef94e4ef` |
| `outer_trace_summaries.csv` | 24 | `a1bef1d07f4f1c3582f305f7c0579b2b8074b71d18c206693dbaf654253d6e06` |

若继续，仍须单独冻结并完成全部原真实输入的双阶参考、生产对照和已接受前缀回归，再讨论效果实验。旧 TRAIN、区域/中心、扰动、实际 V、GS 和未触碰数据确认均不能由本轮合成 PASS 替代；内容尺度规则和科学候选未在本轮选择。
