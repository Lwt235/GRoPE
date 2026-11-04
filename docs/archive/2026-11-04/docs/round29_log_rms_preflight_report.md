# R29：RMS 投影高斯的对数质量表示——固定合成数值预检

2026-10-31。**本轮数值预检通过：29/29 生产调用、40/40 旧数值参考调用、3/3 新解析参考、23/23 基础比较、6/6 恒等关系及近常深度检查全部通过。** 这支持在已冻结合成人口中使用对数质量表示；不构成一般输入准确性、真实数据可完成性或效果通过。R28 的 40/72 完成、32/72 失败记录不变，R26 及此前科学门槛失败也不被覆盖。

本报告只读取已封存 JSON、文件字节及三个新增合成案例的保存数组；未重新调用生产模型或参考积分，未读取真实输入、标签。根验收为 [root acceptance](/home/liwt/GRoPE/results/round29_log_rms_root_acceptance_20261031.json)，SHA256：`e05e005945abe19d5d83a59c28e3a17d24bb884ed10425adff9c450ca574d811`。报告生成前已逐项核对其 26 个文件绑定、20 个当前源码及两侧快照、生产目录 58 个文件和审计目录 56 个文件；后两数包含各自 completion，清单内分别列 57、55 个文件。

## 改了什么，保留什么

R29 保留 R26 的同一个概率模型：单个齐次三维高斯，按深度 RMS 添加 XY 高斯正则，再在严格 `Z>0` 上投影；它仍不同于 R23 的固定屏幕高斯卷积。`h=.25`、面积 floor `1e-30`、GL24、最大边长 `2h` 的积分小矩形、负尾分支及原数值准入条件均保留。没有新增网络、混合分量、协方差修补或科学候选。

改变的是表示与求和顺序：节点先保留有限 log density，以正权重 log-sum-exp 累积 tile、原生格和独立计算的 direct FOV；只加一次面积 floor，再加固定 bias、归一化。原始质量的规范输出是 `log_raw_mass`。普通 float64 的可选输出只保存可以物化的索引及数值，不能把遗漏索引解释为零质量。原先的精确结构零仍单列为 `-inf`；未完成积分由独立 mask 和 tile 计数标记。非有限 log、深度秩丢失或原正前方质量准入失败仍须失败。

旧 26 个 R27 描述保持不变。报告独立比对了其 canonical JSON，以及保存的 `mu/factor/bounds/content_bias` 四类输入 canonical SHA，全部相同；输出字节不要求相同。顺序为旧 20 个基础案例、新 3 个基础案例、旧 6 个恒等关系。23 个基础案例包括 4 个只改变 bias 的重复几何，共 19 种输入几何；不能视为 23 份独立确认数据。固定 bias 也不是已选择的真实内容温度规则。

依据：[冻结协议](/home/liwt/GRoPE/ROUND29_LOG_RMS_PREFLIGHT_PROTOCOL.md)、[对数表示与误差定义](/home/liwt/GRoPE/docs/round29_log_rms_representation.md)、[冻结描述](/home/liwt/GRoPE/results/round29_log_rms_preflight/fixtures.json)。

## 独立参考与数值门槛

旧 20 个基础案例使用未改动的 R27 SVD/条件高斯参考，分别运行 GL32、GL48，保留原截断、节点和失败规则。新增三例只使用一次独立的常正深度、对角屏幕协方差高斯 log-CDF 矩形公式；它们没有伪造双阶收敛检查。所有 direct FOV 都另行计算。

下表的 raw-FOV 指原生格原始质量之和，raw-L1 的分母是参考的这一总和；direct-FOV 可加性另列。相对误差在 log/expm1 表示中与原阈值比较，普通诊断值下溢不参与放行。04、05 的原始质量为解析结构零，只有这两例的 raw 相对指标不适用，仍核验精确零及 floor 后的输出。

| 检查 | 本轮最大值 | 原阈值 / 范围 |
|---|---:|---:|
| 保存的生产 P 对独立参考，最大绝对误差 | `2.220446049e-16` | `1e-4`；23 基础案例 |
| 生产原生格总 raw 相对误差 | `2.273736754e-13` | `1e-6`；28 |
| 生产 raw-L1 / 参考总 raw | `2.273736754e-13` | `1e-6`；28 |
| 旧参考 GL32/48 的 P 误差 | `2.220446049e-16` | `1e-8`；20 对 |
| 旧参考双阶总 raw 相对误差 | `1.421085472e-14` | `1e-8`；12、18 同几何 |
| 旧参考双阶 raw-L1 / 参考总 raw | `3.522631984e-15` | `1e-8` |
| 生产原生格和 / direct FOV 绝对差 | `1.168267210e-15` | `1e-8`；13 |
| 旧数值参考 direct FOV 绝对差 | `3.330669074e-16` | `1e-8` |
| 新解析参考 direct FOV 绝对差 | `2.220446049e-16` | `1e-8`；27 |
| 生产归一化误差 | `3.996802889e-15` | `2e-12` |
| 参考归一化误差 | `6.661338148e-15` | `2e-12` |
| 生产已存 log floor / log P 的独立算术误差 | 各 `1.421085472e-14` | `2e-12` |

全部 29 份保存输出的算术检查通过；同 raw 重新算 P 的最大差为 `4.334635419e-38`，保存的 `exp(log P)` 与 P 差为零。这里是已存输出的算术复核，不是另一遍独立几何积分。完整逐例结果见 [23 基础比较 CSV](/home/liwt/GRoPE/docs/round29_log_rms_preflight_report_data/base_23_rows.csv) 和 [43 参考调用 CSV](/home/liwt/GRoPE/docs/round29_log_rms_preflight_report_data/reference_43_rows.csv)。

## 三个新增案例检验了什么

| 案例 | 固定输入与目的 | 观察结果 |
|---|---|---|
| 27 `native432_far_cells` | `mu=(12,9,1), F=0`；432 个原生格，边界为 `(pixel_boundary-.5)/8`，考察远格 | 432 个 raw log 全部有限，最小 `-1508.1652197202468`；116 个正质量格不能普通物化，316 个可物化格中有 10 个 subnormal。原生格 raw 总和的 log 为 `0`，P 最大误差 `1.110223025e-16`，raw-L1 比值 `2.667151919e-16`。 |
| 28 `subfloat_fov` | `mu=(0,0,1), F=0`；x 边界 `[9.75,10,10.25]`、y 边界 `[-.25,0,.25]`，考察整个 FOV 都低于普通浮点范围 | 四个格均有有限 raw log，普通物化列表为空；原生格总 raw log 为 `-765.4648717106796`，解析参考为 `-765.4648717106799`。raw 相对误差与 raw-L1 比值均 `2.273736754e-13`，仍通过原门槛。 |
| 29 `visible_far_tiles` | `mu=(0,0,1), F=0`；单格 `[0,-.25,12,.25]`，可见质量与远端小质量共存 | raw 质量 `0.34134474606854326`；独立 raw 相对差 `6.661338148e-16`。cell 和 direct FOV 各 24 tiles，局部 LSE 均发生 4 次相对 exp 丢项并记录界；单格 P=`1` 本身没有区分能力，必须看 raw。 |

28 的面积 floor 总量为 `2.5e-31`，raw/floor 约 `1.461807267e-302`。四个保存 P 都约为 `0.25`，P 比较误差为零；**仅凭这一 P 无法证明原始几何质量正确，保留 raw 相对检查是必要的。** 其独立计算的生产 direct FOV log 为 `-765.4648717106896`，与原生格和的 log 不相等；绝对差的 log 为 `-790.7928636602841`，普通诊断显示 `0` 只因不可表示，绝非精确可加或结构零。raw-FOV 相对指标并不是这项 direct-FOV 绝对指标。

旧 tiny-front 12、18 也保留：`p+=1.776482112e-33`，由 log 物化的原生格 raw 总和约 `6.627901321e-36`，floor 约 `1.6e-29`；两例都不是结构零，仍检查 raw。04 的零深度 RMS 与 05 的负常深度才是精确结构零，raw 为零、积分工作量为零；04 的非零 bias 仍参与 floor 后的归一化。

## 相对 exp 丢项、并列与有限极限

log-sum-exp 本身仍会丢掉相对极小项。本次只有 27、29 记录到此类事件：

| 案例 / 局部求和范围 | 丢项次数 | 最大局部绝对遗漏界的 log | 最大局部相对遗漏界的 log |
|---|---:|---:|---:|
| 27 / floor | 98 | `-815.1590466144644` | `-746.0814938246431` |
| 27 / direct FOV tiles | 401 | `-745.428507969238` | `-745.4284195480437` |
| 27 / native total | 116 | `-766.4849498197734` | `-766.4849498197734` |
| 29 / cell tiles | 4 | `-804.9901571600741` | `-803.915294833212` |
| 29 / direct FOV tiles | 4 | `-804.9901571600741` | `-803.915294833212` |

这些是各次局部正和的界的最大值，不能相加成全管线误差界，也不是独立失败人数。它们仅覆盖已计算 log 输入在相对 exp 时变为零的项，不覆盖密度、求积、加法吸收或其他舍入。审计从保存输入独立重算 native/floor/归一化相关诊断；节点与 tile 内部诊断未另行积分重放。普通 subnormal 有值也不保证相对精度。完整八类 LSE 记录保留在 [29 生产调用 CSV](/home/liwt/GRoPE/docs/round29_log_rms_preflight_report_data/production_29_rows.csv) 的 JSON 列。

微小 P 误差不意味着 argmax 稳定。生产对高阶参考有两例第一最大值索引变化：01 为 `5→6`，25 为 `1→0`；双阶参考比较也在这两例变化。对应 `1e-12` 近并列集合分别为 `[5,6,9,10]` 和 `[0,1,2,3]`。生产侧 01 有 3 个精确最大值、25 有 2 个；12、28 各有 4 个精确最大值。它们都在原 P 门槛内，不能据此写“无并列”或“排序完全稳定”。

六项恒等关系全部通过：正世界缩放、屏幕平移、屏幕单位连同 h/floor 变换、坐标轴旋转、key 重排、不等面积格合并。前四项以 17 为 parent，重排以 13 为 parent，合并以 09 为 parent；它们是生产关系检查，不是额外独立参考。最大 raw 绝对差 `1.661372088e-16`，最大 P 差 `1.665334537e-16`，均小于 `1e-8`。详见 [6 恒等关系 CSV](/home/liwt/GRoPE/docs/round29_log_rms_preflight_report_data/identity_6_rows.csv)。

近常深度 11 使用固定非零深度因子 `2^-36`，与 03 的常深度案例比较；保留非零分支，P 差 `1.016439537e-19`，raw-L1/常深度 FOV 为 `1.987603217e-19`，通过原 `1e-8` 门槛。这只检查一个固定有限 ε，不证明任意小深度方差下的一致收敛。

旧参考截断界的最大 raw 相对量为 `1.469688284e-37`、最大 P 界为 `2.435237230e-43`，分别低于 `1e-10`、`1e-8`；其分母是计算的保留 FOV 质量，不是区间包围。它们只约束截断。双阶差、外层误差估计、生产精确算术 GL 界也都不能成为包含密度与浮点误差的总证书；三个解析 CDF 参考没有截断，但也没有严格的浮点 CDF 误差包围。

## 成本、开发历史和结论范围

| 阶段 | 实际人口 | phase 时间 | 工作量 | 预算 |
|---|---:|---:|---:|---:|
| 生产 | 29 全完成 | `2.817918479 s` | `8,085,888` 节点评估 | phase `600 s`；每调用 `4,000,000` |
| 独立审计 / 参考 | 40 数值 + 3 解析，43 全实际调用 | `50.789532749 s` | `332,794,192` 数值节点；另 `1,760` 次 CDF 评估 | phase `1,800 s`、`600,000,000` 节点；每数值调用 `250,000,000` |

单次生产最大 `1,990,656` 节点，单次参考最大 `174,918,640`（13 / GL48），均未超预算。生产每条记录的调用及输出时间之和为 `2.728668045 s`，参考逐调用时间之和为 `50.026260916 s`，它们不等于含记录等工作在内的 phase 时间；最终 provenance/completion 的收尾另记。生产 RSS 最大 `88,732 KiB` 是进程累计高水位，不是每例独占内存。参考 CDF 计数为 `1732+20+8=1760`，不折算或伪造求积节点。这里的合成小规模成本不能外推真实总体成本。

联合预检为 [73 项纯合成测试通过](/home/liwt/GRoPE/results/round29_log_joint_preflight_20261031.log)，并用抛异常 sentinel 验证冻结不解码 NPZ、不调用正式积分；20 份来源、运行时及双快照均锁定。保留的开发历史包括：

- [模型历史](/home/liwt/GRoPE/results/round29_model_development_history_20261031.json)：首轮通过。
- [生产端历史](/home/liwt/GRoPE/results/round29_producer_development_history_20261031.json)：两轮各 19 项通过；第二轮补齐审计锁七项 false 范围检查，未改数值门槛。第一轮仅有记录的源码 hash，不伪称保存了其全部原始源码。
- [参考与审计历史](/home/liwt/GRoPE/results/round29_audit_development_history_20261031.json)：首轮 28 通过、1 失败，因合成 mock 缺少最终收据所需的 audit lock；补 mock 后 29 通过，再加结构零算术测试后 30 通过。静态审查中的工作量保留和范围接线修正在正式冻结前完成。
- [根验收脚本历史](/home/liwt/GRoPE/results/round29_root_controller_development_history_20261031.json)：首次 expected dict 未包含 `A.SCOPE`，完整字典断言失败；v2 补范围字段及检查后通过。原错误文本明确标为终端输出转录。两次只复核保存结果，没有重跑生产或参考积分，不能把它记作正式数值管线失败或第二次正式尝试。

本轮可以支持的结论是：**在这 29 个预先固定的合成调用中，规范对数质量表示保留了正的极小质量，且全部原 raw/P、direct FOV、关系与预算门槛通过。** 它没有证明 R28 的 32 个失败输入已修复，没有验证一般非零均值相关高斯域，也没有给出真实效果证据。`real_inputs_read / labels_read / general_numerical_accuracy_established / efficacy_tested / candidate_selected / scientific_gate_passed / goal_complete` 在本轮均为 false。

根的[结果后决策](/home/liwt/GRoPE/results/round29_log_rms_post_result_decision_20261031.json)已单独选择 [R30 准备设计](/home/liwt/GRoPE/docs/round30_log_rms_applicability_design.md)：新冻结后仅复用固定 72 个已接受旧 TRAIN 输入，检查适用性和成本；不读标签、不放宽门槛。本报告不把该下一步写成已完成，也不晋升效果候选。区域、中心、扰动、实际 V、优化 GS 和未触及确认集等原科学验证要求仍保留。

## 附表与 SHA256

CSV 由封存 JSON 提取，不重新积分。空白表示原 JSON 的 null 或不适用；请结合 `structural_zero`、`raw_relative_applicable`、参考类型和 log 字段解释，不能一律读为零。解析三例没有双阶 reference 列。`maximum_per_cell_relative_error_descriptive_only` 只是描述量，不新增门槛。身份关系、scope 和完整逐项证据以封存文件为准。

| 文件 | 行数 | SHA256 |
|---|---:|---|
| [base_23_rows.csv](/home/liwt/GRoPE/docs/round29_log_rms_preflight_report_data/base_23_rows.csv) | 23 | `b5296cf7134f4a0b3fe294e631c26f43345181ff22e3ad30c589f7995643a2a5` |
| [reference_43_rows.csv](/home/liwt/GRoPE/docs/round29_log_rms_preflight_report_data/reference_43_rows.csv) | 43 | `29bcad150b6bd10ca10ce21e9973444d99ecc83e0241ba977f0efdf0a1788f15` |
| [identity_6_rows.csv](/home/liwt/GRoPE/docs/round29_log_rms_preflight_report_data/identity_6_rows.csv) | 6 | `b59df74ff043736d93eb6aff8b3bacb85204c062a8570c04166ddb5e1e56bfa6` |
| [production_29_rows.csv](/home/liwt/GRoPE/docs/round29_log_rms_preflight_report_data/production_29_rows.csv) | 29 | `3ec6008047cb10b6655afcc758406601e24bb7b4a79d968e7dfaec64ea5dbbac` |

主证据：[生产封存](/home/liwt/GRoPE/results/round29_log_rms_preflight/forward_complete.json)、[独立审计汇总](/home/liwt/GRoPE/results/round29_log_rms_independent_audit/summary.json)、[逐基础案例比较](/home/liwt/GRoPE/results/round29_log_rms_independent_audit/comparisons.json)、[近常深度检查](/home/liwt/GRoPE/results/round29_log_rms_independent_audit/near_constant_limit.json)。
