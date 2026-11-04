# R17 尺寸温度归因：外部验收与报告清单

本清单只依据 producer / auditor 源码的输出契约编写，未读取本次归因结果或
主 VAL 效果。它不修改冻结协议、不新增科学门槛。下列检查须在归因生产端和
独立审计均结束后执行；结果写在两个冻结目录之外。

## 先验收证据链，再提取效果

- 核对两个 `lock.json` 与各自 `lock.sha256`；producer 当前联合来源为84项，
  auditor 为37项，原文件及 `source_snapshot` 哈希均须一致。核对 auditor
  `run`、`producer_lock_sha256`、固定 `AUDIT_POLICY`、源码清单和冻结先后。
  不可用 `/tmp` 接口预检锁替代正式运行锁。
- producer `completion.json.files` 应精确覆盖其目录内除自身以外的所有文件，
  核验全部字节哈希；同时核对其 `lock_sha256`、`summary_sha256`。
  auditor completion 也按同样规则验收完整目录，不仅检查文件是否存在。
- auditor `completion.passed` 与 `verification.passed` 必须均为 `true`，
  不得出现异常 `failure.json`；数值失败也必须保留并报告。绑定 verification
  中 `producer_{lock,forward_complete,summary,completion}_sha256` 四项到实际文件，
  并绑定 `independent_audit_lock_sha256`。审计 completion 没有单独的 producer
  引用，必须通过已校验的 verification 和 audit lock 完成连接。
- 核对主清单、三个 forward completion、全局 seal、三个 evaluation completion
  和两端开始/结束记录。应满足：producer freeze → audit freeze → 各场景 forward
  → 全局 seal → 每个 evaluation → producer summary/completion → audit。
  验证 `started.json` 的锁引用；`verification.started_unix` 应与审计 started
  记录一致，verification 结束时间不晚于审计 completion。
- 固定配置只能是 `h=.25, temperature=10`，角色只能是 `trace_full` 与
  `size_temperature`。本研究沿用已接受 TRAIN 的 primary、来源链及透明的路径
  修补记录，不重新选择配置。R10提供 dense/mixed 标签；g8/g16/g24中心和可见性
  使用精确 R13 Stage A 标签，不能替换为近似相等的 R10 坐标。

## 完整人口与数值记录

| 检查项 | 固定人口或要求 |
|---|---|
| TRAIN | c3 / c5 / c6 三场景，各16对，共48对 |
| 条件与角色 | 原9条件 + 直接尺度5条件，2角色；1,344个矩阵/指标行 |
| 全量证据 | 672个条件档各保留2个float64概率矩阵及int32 argmax；原始概率512,434,944字节 |
| 原primary | `primary_old_hash_matches=672`，全部匹配已接受 R17B 的原概率哈希 |
| 配对与分层 | 672条paired行；每对23个条件×原生尺寸分层，共1,104条stratum行 |
| 输入描述 | 48个support档、四query分区，39,888个唯一query；21个scene×query×width×height组 |
| 确定性 | 每场景首pair的14条件×2角色共28个重复矩阵；三场景84个 |
| 独立数值样本 | 每场景首pair，14条件×2角色×8固定query；84矩阵、672 query行、全部native keys |
| 指标重建 | 全部672个配对条件的native/common target counts，全部1,344个角色指标行 |

每张表都按 `(scene,pair,condition,role)` 或明确的配对/尺寸键检查无缺失、
无重复、无新增；整数、形状、哈希、来源及精确分数不得以浮点容差代替。
核对 verification 的所有人口计数，并逐项读取 `numerical_checks.json`：
所有 `passed=true`、`numerical_failure_count=0`；same-raw误差≤1e−12，
production/reference与hybrid误差≤1e−4，GL64/96收敛误差≤1e−8，
metric sum误差≤1e−8，summary误差≤1e−10。原始归一化/有限性检查继续有效。
保留tiny/zero-FOV、参考argmax差异及近平局，不因误差限内就省略这些记录。

## 报告必须保留的信息

- 两角色全部scene×condition的KL/JS、direct KL/JS、中心正确数/可见数及PCK，
  common9/direct5宏平均和仅作描述的J。区域指标按原mass在scene内聚合，再
  等权平均scene/condition；不要重算成全数据池加权平均。
- 配对差异统一注明 `size − observed`：PCK用精确Fraction，显示百分点；
  KL/JS差异及相对差异注明基准。保留全query/可见query的argmax翻转、benefit/harm
  计数。原生尺寸分层只报告direct区域指标，不能把非线性common KL分摊进去。
  空可见分层保留 `None`，不能当零准确率或悄悄剔除。
- 输入描述按唯一query统计tau/trace及tau比值的线性min/p05/median/p95/max，
  不随key条件重复计权。mixed的16×8与8×16必须分别保留。
- 明确区分两类argmax变化：两角色之间的归因翻转，与production/reference的
  数值差异。明确672个旧primary匹配指概率数组，不是新增的Stage A PCK重放证明。
- 报告审计边界：全量重放共享旧准备/高斯算子，不能称为全量独立forward；
  独立NumPy moments/integration只覆盖预声明84矩阵样本。tau、指标及汇总有
  独立核查；无新增真实任务训练或计时。

## 外部输出链的处理边界

当前没有 `accept` 子命令或独立 `decision.json`。producer summary/completion
中的 `independent_audit_pending=true` 是冻结时状态；审计通过后不要回写。
需要发布时，在外部新目录写一个验收回执，绑定上述两锁、两个completion、
verification、numerical_checks、producer summary和independent_summary哈希，
再从校验过的summary提取表格。这是报告验收，不是新增实验审计或科学通过。

禁止向已完成的producer/auditor目录添加报告、日志、回执或修改状态字段：
其严格inventory会把任何新增文件判为改变证据人口。报告提取器也必须保留
`scientific_gate_passed=false`、`scientific_gate_defined=false`、
`descriptive_attribution_only=true`，以及selection/current-validation未改变的声明。

本研究只能说明在固定h/T和原full-C几何下，观测trace相对已知query尺寸的
条件性TRAIN归因。它不构成等价性/显著性结论，不替换primary，不提供成本优势，
也不能据此宣布通过主VAL几何/鲁棒性、实际V、真实GS或未使用数据确认门槛。
这些门槛的状态须由各自独立证据决定，不能从本次归因的数值审计通过外推。
