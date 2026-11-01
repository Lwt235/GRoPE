# R28：固定真实几何输入的可完成性与成本检查

2026-10-31。**结论：适用性检查 FAIL。** 固定 72 项全部实际调用，40 项完成、32 项失败，无跳过或截止时间短路。72 项输入审计全部通过，40 项成功输出的算术审计也全部通过；但协议要求全部 72 项完成，因此 `applicability_calls_completed=false`。根验收接受完整失败证据，不代表概率精度或效果通过；`independent_probability_reference=false`、`general_numerical_accuracy_established=false`、`efficacy_tested=false`、`scientific_gate_passed=false`、`goal_complete=false`。

## 来源、人口与时序

已重核[根验收记录](/home/liwt/GRoPE/results/round28_rms_root_acceptance_20261031.json)全部 29 项文件绑定、11 个源码当前文件及双侧快照、生产 169 文件和审计 22 文件的完整库存（含 completion），均通过。根记录 SHA256：`419f393e42b8aad4c16cf4ad4b585b8960584c1e13ba18e845f860088a6cf489`。本报告只解析封存 JSON/metadata，文件哈希检查不解码 NPZ；没有模型重放或新增实验。

按[冻结协议](/home/liwt/GRoPE/ROUND28_RMS_APPLICABILITY_PROTOCOL.md)，原样复用 R23 已审计的 72 个预测几何输入：旧 TRAIN 的 c3/c5/c6，每场景 pair `[30,45]`，query 尺度 8/16/24，各 8 个事先固定索引。只读取原 prepared 档案中 8 个授权字段，没有重建 patch、解码原始 R10 archive、读取标签或接触 fresh 数据。`target__covariance` 指预测点统计变换到目标相机后的协方差，不是传感器 GT。

将原 H、offset 组为齐次矩阵 \(A=[[H_{00},H_{01},c_0],[H_{10},H_{11},c_1],[0,0,1]]\)，再计算 \(\mu_h=A\mu,F_h=AF\)。保留完整 H（含可能的 skew）、原生 432-cell bounds 的 `(pixel_boundary−0.5)/8` 单位和精确零 bias；不额外平移主点、不修补协方差或深度。沿用冻结 R26 RMS 投影模型、\(h=.25\)、一次面积 floor、GL24 与 \(2h\) tile，没有选择新 anchor、Q/K/V 或温度规则。

联合 **76 个纯合成测试通过**，metadata-only 冻结预检通过后才访问授权数组。封存时序已核：双侧冻结 → 全量准备封存 → 独立输入审计封存 → 72 次前向 → 全量前向封存 → 最终审计。输入最大归一化误差 **`2.5917163069442953e−15`**，低于本轮 `1e−12`。本轮门槛比 R23 的 `1e−10` 严格；旧设计稿“与之前相同”的措辞不准确，实际协议与代码均为 `1e−12`。

## 全人口结果与失败边界

| 场景 | query 尺度 | 尝试 | 完成 | 失败 |
|---|---:|---:|---:|---:|
| c3 | 8 | 8 | 4 | 4 |
| c3 | 16 | 8 | 6 | 2 |
| c3 | 24 | 8 | 4 | 4 |
| c5 | 8 | 8 | 1 | 7 |
| c5 | 16 | 8 | 3 | 5 |
| c5 | 24 | 8 | 4 | 4 |
| c6 | 8 | 8 | 6 | 2 |
| c6 | 16 | 8 | 6 | 2 |
| c6 | 24 | 8 | 6 | 2 |
| **合计** | — | **72** | **40** | **32** |

32 项封存的错误串完全相同：`NumericalFailure: positive density underflow/overflow; not a structural zero`。它来自正密度求值的合并保护，包含 log-density 非有限、指数结果非有限或非正等检查。**日志没有保存触发节点的 log-density 极值及具体子条件，因此不能将 32 项精确诊断为“已测得下溢”，也不能断言溢出已被排除。** 目前可确定的是这些调用在该保护处失败；全部 72 项均记录为 variable-depth、非结构零。

所有失败都保留 partial raw、partial direct-FOV 和实际工作量，均在完成 native 阶段之前退出，direct-FOV 完成 tile 数为 0。其中 24 项尚未完成任何 native tile；其余 8 项保留已累加部分。未完成位置的初始化 0 不是已证实的零质量，失败项没有伪造 P、面积先验或重试。完整记录见[72 项结果表](/home/liwt/GRoPE/docs/round28_rms_applicability_report_data/outcome_72_rows.csv)及[9 个场景/尺度汇总](/home/liwt/GRoPE/docs/round28_rms_applicability_report_data/scene_size_9_rows.csv)。这些固定查询不是场景总体的随机抽样估计。

## 成功项提供了什么证据

40 项均通过保存输入/输出身份、有限非负 raw/P、一次 floor、零 bias 归一化、first-index argmax、tile/node 工作量及独立输出算术复算。最大归一化误差和同 raw 重算 P 差均为 **`2.220446049250313e−16`**；native raw 和与单独 direct-FOV 积分的最大差为 **`1.3322676295501878e−15`**，低于 `1e−8`。

按审计累加的成功项 FOV raw 范围为 `[0.07562905630817585, 1.0]`，总 floor 固定 `4.32e−28`；没有 raw 零 cell、总 raw 低于 floor 或面积先验分支。40 项均无超过 1 个的精确/`1e−12` 近似最大值；这是本次已完成子集的排序诊断，不是对失败项或一般输入的保证。逐项数值见[40 项算术检查表](/home/liwt/GRoPE/docs/round28_rms_applicability_report_data/success_40_arithmetic.csv)。

这里没有真实输入上的独立概率积分，因此没有检验“生产对独立参考”的 raw 相对误差或 P 精度。direct-FOV 与 native 使用同一生产密度；算术一致性不能独立证明该密度/求积正确。`6.822702098808657e−11` 的 GL 绝对界仅适用于精确算术，排除密度求值、分解与舍入误差，也不是普遍 raw 相对/P 证书。

## 成本与下一步边界

前向阶段 **`23.468733232468367 s`**，低于共享 600 s；累计记账 **83,109,888 个密度节点**，其中成功项 79,626,240、失败项 3,483,648。每个成功调用为 1,990,656 节点，低于单次 4M；没有预算导致的未尝试项。调用及输入加载耗时和为 `23.171234794 s`，不同于包含逐例记录写入的整阶段耗时。输入准备阶段 `0.322648982 s`；输入审计元数据起止间隔约 `0.578 s`，最终审计约 `0.745 s`，均单独记录。峰 RSS `90,132 KiB` 是进程累计高水位，不能解释为单查询独占内存。失败提前退出，因此阶段耗时不能外推为“全部 72 项成功”的成本。

[审计开发记录](/home/liwt/GRoPE/results/round28_audit_development_history_20261031.json)保留一次正式运行前的静态修正：将 CLI main guard 移至函数定义之后，并加 AST 顺序检查；修正时未读取正式数组或运行模型。独立审计首次 toy 测试 26 通过，无失败测试尝试；随后联合 76 测试通过。该记录不应被写成正式实验失败或正式重跑。

R23 为 0/72 完成，R28 为 40/72 完成；两轮的数学 law 和实现不同，且 R23 部分行因共享截止时间未进入积分。这只能比较固定人口下的**实现可完成性**，不能解释为定位、区域质量或科学效果提升。R23、R26 及已有科学失败均保留；R27 的合成数值通过也不覆盖本轮真实适用性失败。

后续可考虑[尚未验证的 log 表示修订](/home/liwt/GRoPE/docs/round28_underflow_remedy_unselected.md)，但合并错误串尚不能证明它能消除全部失败。任何修订必须另建源码/协议并先冻结；正质量不可表示、结构零、未完成 partial 必须区分，原 raw/P、direct-FOV、独立参考及资源门槛不能放宽。真实效果、稳健性、actual V、优化 GS 与未接触确认集仍未通过，本报告不晋升方案或宣布研究目标完成。

附录 SHA256：outcome72 `736979bd73789c22327233b13ce7250af2fa53a2349d57a89ecc340c3287838a`；scene/size9 `3221b88846590df2d552e4aef2708d1edca78c0d6684a4128f67203be87bc33f`；success40 `563e669c3f2f3a6dd90521accc899e07ad064eec169edc247a5e9a67ed7fb7cf`。
