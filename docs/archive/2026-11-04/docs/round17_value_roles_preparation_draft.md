# R17 actual-V 角色与别名：条件准备草稿

2026-10-31。**未冻结、未训练、未选择新的科学候选。** 本文仅依据已接受的
TRAIN 选择、冻结协议及源码整理角色；没有读取主 R17 VAL 的效果、指标、
summary、decision 或正在生成的审计结果。主 VAL 的完整审计与科学门槛若未
通过，不得启动这里的真实数据导出、训练或计时。合成接口通过不能解除该条件。

此前 [下游准备](round17_downstream_implementation_plan.md) 和
[条件分支](round17_post_validation_branches_draft.md) 已列必须保留的类别，
但没有列完整算子去重表。本文补这一缺口；机器可读对应为
[draft JSON](round17_value_roles_preparation_draft.json)。JSON 是文档，不是
可执行角色配置或授权锁。

## 1. 来源和共用精度合同

- R17 主选、七 regional 控制、两个 localizer、七 legacy 控制及两个 matched
  控制，直接来自 [冻结 VAL 协议](../ROUND17_VALIDATION_PROTOCOL.md) 的
  19 个语义别名和 15 个算子；配置由已接受
  [TRAIN selection](../results/round17_regional/selection.json) 追溯。
- R12 的八个 fusion 角色及 singleview 来自
  [R12 深度协议](../ROUND12_DEPTH_PROTOCOL.md)、
  [几何配置](../experiments/round12_depth_geometry.py) 与
  [原比较顺序](../experiments/round12_depth_analysis.py)。其中只有
  `iso4_matched` 和 `singleview` 在 R17 的 15 个算子中没有对应项。
- 全部新的融合角色使用已预检的
  [R17 实际 V 接口](../models/round17_depth_attention.py)：纯几何 float64，
  学习参数、Q/K、预测深度、V/head 为 float32；`tau32*content32` 为 float32，
  加几何和 softmax 为 float64，汇聚前仅 `attention.float() @ V`。
  constant 角色的 tau 是全局 T，不能借用 trace 角色的最终概率。
- 这保留 R12 控制的物理定义和已选 h/T，却**不是旧 R12 float32 几何缓存／
  softmax 的逐位重用**。全部 fusion 在新精度合同下重新训练；历史结果、
  checkpoints 或 float32 cache 不能冒充这次结果。本文的 alias 是未来
  共同新合同内的同一算子，不宣称不同历史实现的数组天然相同。
- singleview 独立训练原 R7 同容量头，固定 h=.5/T=5，使用原 float32
  quadratic 自几何、query 自己的 feature/V，完全不读跨源 geometry 或 C。
  原始冻结 query depth 另报一次，无参数、无训练，不制造五份 seed 重复。

## 2. 17 个独立训练算子，28 个语义别名

`v00`—`v14` 顺序对应 R17 原 `o00`—`o14`；新 ID 仅供此草稿引用。
表中 G 是纯 geometry 身份，P32/64 是上述新融合精度政策。所有 h 均以
reference-8 为单位。只有 trace 两角色使用 `support_covariance` 计算 tau；
其他融合角色不读取该字段。各角色仍使用同一原 41,345 参数头。

| ID | family / interface mode | h | T | tau_cov | G | 语义别名 |
|---|---|---:|---:|---|---|---|
| v00 | trace_full / trace | .25 | 10 | 原 full C32 | g00 | primary |
| v01 | quartic_point / constant | 1 | 5 | 不读取 | g01 | regional_quartic_point |
| v02 | gaussian_point / constant | 1 | 5 | 不读取 | g02 | regional_gaussian_point, legacy_point, r12_point |
| v03 | gaussian_fixed / constant | .5 | 5 | 不读取 | g03 | regional_gaussian_fixed, legacy_fixed, r12_fixed |
| v04 | gaussian_full1 / constant | .25 | 0 | 不读取 | g00 | regional_gaussian_full1, legacy_full1, r12_full1 |
| v05 | gaussian_full4 / constant | .25 | 0 | 不读取 | g04 | regional_gaussian_full4 |
| v06 | gaussian_quad4 / constant | .5 | 0 | 不读取 | g05 | regional_gaussian_quad4 |
| v07 | gaussian_quad16 / constant | .5 | 5 | 不读取 | g06 | regional_gaussian_quad16, legacy_quad16, r12_quad16 |
| v08 | gaussian_point / constant | .25 | 5 | 不读取 | g07 | localizer_gaussian_point |
| v09 | quartic_point / constant | .25 | 5 | 不读取 | g08 | localizer_quartic_point |
| v10 | gaussian_point / constant | .5 | 5 | 不读取 | g09 | legacy_point_pck, r12_point_pck |
| v11 | gaussian_full4 / constant | .5 | 5 | 不读取 | g10 | legacy_full4, r12_full4 |
| v12 | gaussian_quad4 / constant | .5 | 5 | 不读取 | g05 | legacy_quad4, r12_quad4 |
| v13 | gaussian_full1 / constant | .25 | 10 | 不读取 | g00 | matched_constant_temperature |
| v14 | trace_isotropic_matched / trace | .25 | 10 | 原 full C32，非 iso C | g11 | matched_isotropic |
| v15 | gaussian_iso4_matched / constant | .5 | 5 | 不读取 | g12 | r12_iso4_matched |
| v16 | original_quadratic_self / singleview | .5 | 5 | 不读取 | self | r12_singleview |

`gaussian_iso4_matched` 是本文区分四分量 iso 与单高斯 iso 的名称；原 R12
配置名称为 `iso4_matched`、method=`iso4`。它没有加入冻结的 R17 VAL family
集合，后续 exporter 必须显式支持并独立复核，不能直接传给不支持它的
`round17_validation_pe.geometry`。

全部 R12 别名也保留，故 19+9=28 个语义名称；它们对应 16 个 fusion 训练
算子加 1 个独立 singleview。若未来冻结五配对 seeds，就是 85 个独立
role/seed 训练运行，不能按28个别名重复制造独立证据。每个语义比较仍保留。

## 3. 几何身份与训练身份分开

16 个 fusion 算子只需要 **13 种纯 geometry**。单视图还有第14种自身几何，
不属于跨源缓存。这里仅盘点数学和数值身份，不选择存储方案。

| G | 不含内容的几何定义与源码 |
|---|---|
| g00 | 单 Gaussian，预测全像素 mean 和 full C32，h=.25；R17 prepare_query/geometry 委托 R15 gaussian_full1，GL32 native-cell mass |
| g01 / g08 | quartic point，C=0，h=1/.25；R15 quartic tiled GL24 native-cell 积分 |
| g02 / g07 / g09 | Gaussian point，C=0，h=1/.25/.5；R15 Gaussian native-cell 路径 |
| g03 | Gaussian fixed，h=.5，C=area/(64*12) I；本任务 g8 为 I/12，保持历史定义 |
| g04 / g10 | 四个等面积 stratum mean，保留各 stratum 散布+平均 pixel C，h=.25/.5，按原权重混合 GL32 mass |
| g05 / g06 | quad4/quad16，h=.5；stratum mean +平均 pixel C，不添加 stratum 散布；权重、所有像素来源保持原定义 |
| g11 | 原 full C32 先转 double，再取 tr(C)/2 I 仅用于几何，h=.25；不是先在float32中重新求 trace 再圆整；tau仍保留原 full C32 |
| g12 | 四 strata 各自的 full node C32 转 double 后取同 trace isotropic C，h=.5；原 `gaussian_stratum_logits(...,'iso4',...)` |
| self | 原 R7 `controlled_features(...,'full',.5)`；自 mean/cov、原 quadratic score，与native-cell积分缓存不同 |

所有 fusion native key 为同一原 g8 432-cell 分区。Gaussian 路径使用原 GL32
实现及其 diagonal-C 快路径；quartic 使用原 GL24 tiled 实现。都只加一次
`1e-30 * native_key_area/64` floor，不另加 key Gaussian covariance。像素
预测坐标和 covariance 统一除以8/64，support accumulation 顺序保持原float32。
具体实现来源为 [R17 PE](../models/round17_validation_pe.py)、
[R15 PE](../models/round15_validation_pe.py)、
[strata](../models/round11_stratum_pe.py) 和
[support](../models/round10_partition_pe.py)。实际深度 feature/V 仍取原R7合法
输入；不能把区域 pooled frozen feature 相似度当作在线 learned content。

可以共享 g00 的纯 float64 geometry，但 v00/v04/v13 的 trace/constant
规则或 T 不同，必须独立训练。g05 也仅让 v06/v12 共享纯几何，不能共享学习
轨迹。v05 与 v11 的 h/T 都不同，连 geometry 也不相同。singleview 即使
h/T 与部分 fusion 相同，也不能 alias。恰好某样本 C=0 或 T=0、概率巧合
相同，都不能消除 family、校验、数据来源或算子成本的差别。

真正训练 alias 必须冻结同一 family、h/T、几何生成/积分/floor/精度、tau
及 C 来源、在线 Q/K/V 定义、字段白名单、模型初始化、数据/批次、优化预算、
checkpoint选择和执行校验。所有角色的完整配对成本都重算自身几何；训练
期间共享缓存不能变成计时收益。这份纸面清单不构成任何缓存身份实测。

## 4. 比较集合与精确平局：保留项和待冻结建议

R12 已要求的最小 position 顺序原样保留：
`r12_point → r12_point_pck → r12_fixed`（v02,v10,v03）。
最小 compression 顺序原样保留：
`r12_full1 → r12_quad4 → r12_quad16`（v04,v12,v07）。
依据五 seed 的等 scene/seed validation AbsRel 取最小；只在原始数值精确
相等时按声明顺序取先者，不设置近似平局epsilon，不按时间挑较弱或较慢对照。
checkpoint仍按 steps0,100,...,1000 的最早严格最小 equal-scene AbsRel选择。

**下列扩展仅为未冻结建议，不是新增已生效门槛或已选赢家：**

- position 在旧三项之后，按 R17 语义先出现顺序追加
  `regional_quartic_point → localizer_gaussian_point → localizer_quartic_point`
  （v01,v08,v09）。R17 strong Gaussian point 与旧 point 已是同一完整角色，
  不重复训练；两个新 localizer 不因其区域校准目标不同而被删去。
- compression 在旧三项之后追加
  `regional_gaussian_full4 → regional_gaussian_quad4 → r12_full4`
  （v05,v06,v11）。这样保留旧 full4 以及新 TRAIN 分别校准的 full4/quad4，
  而不默默用 legacy 配置替代它们。
- matched constant（v13）、matched trace-isotropic（v14）和原
  R12 iso4_matched（v15）保留为归因报告；沿既有合同，它们不替换主选，
  也不自动加入上述强比较 gate。是否为新实际任务另加比较规则，必须在
  训练前明确冻结，不能看任务分数后纳入或移出。
- singleview（v16）保留独立的严格效用比较，不并入可被其他角色替换的
  strong-position/compression选择集合。原始 query depth 独立报告。

原门槛仍为对最强 position 至少2% AbsRel收益，对singleview严格改善，两
场景及至少4/5 seeds改善；compression保留原精度/成本二选一与RMSE保护。
旧R12失败锁不能被这份准备绕过。最终全部角色、集合、tie顺序、数据来源、
五seed训练合同、数值/实际任务审计及完整成本必须另行冻结；上游不通过时
不启动本清单对应的实验。

## 5. 缓存资源：待冻结的单组内存方案

本节是 root 的静态容量估算与条件建议，不是缓存实现、导出或性能测试。
704 pairs、每对432 queries/keys，以及原模型的128维输入给出：

| 对象 | 原始字节数 |
|---|---:|
| 一组 float64 geometry | 1,051,066,368 |
| 13组 geometry 全部持久化 | 13,663,862,784（12.725 GiB） |
| 所有 pair 的 full C32 | 4,866,048 |
| 所有 pair 的 q+k 两份128维 float32 feature | 311,427,072 |
| 一组 geometry + full C + q/k 小计 | 1,367,359,488（1.273 GiB） |

feature维度来自原 `round7_depth_attention.py` 的 `Linear(128,128)`，
不是外部backbone的其他层宽度。上述小计未含原始观察缓存、标签、临时张量、
训练梯度、优化器、设备复制或库开销，不能当作RAM/GPU峰值。
2026-10-31只读资源检查时，磁盘约4.8 GiB可用，宿主报告约427 GiB可用内存；
所读cgroup v1 memory limit为近int64上限。这些只代表检查时状态，实际执行
前仍须重新确认进程内存限制、可用空间和所选训练设备，不能据此保证分配成功。

候选存储方式是先在CPU RAM中逐组生成全部13组完整float64几何，逐组保留
规范hash并释放缓存，在任何训练前封存完整manifest。随后训练时逐组再生、
核对hash，供该组内声明的独立role/seed训练复用；原始输入、模型权重、
全部历史和预测仍持久保留。
缓存仅含纯几何及合法支撑，Q/K、内容分数、tau算术、softmax、V/head在线执行。
singleview沿原实现独立处理。按几何分组执行的次序也须在训练前冻结；各run
仍重置配对初始化与batch序列，不因前一角色效果提前跳过后续角色。

该方式改变的是**派生缓存的保存策略**，所以不能直接沿用要求`.npy`文件
存在的R12来源锁。若以后采用，须在任何真实导出/训练前另冻下列合同：

1. 完整绑定原输入文件、相机、pair顺序、分区、算子源码、精度与运行时；
   逐pair记录mean、实际C及几何数组的dtype/shape/规范字节SHA-256。当前
   纸面g00/g05等价性须在新exporter中核验；hash相同不替代独立数学检查。
2. 全部角色、全部pair的几何来源和规范hash先生成并汇入完整manifest，
   由训练锁一次性绑定，继承R12“所有角色geometry hashes先于任何训练”
   的时序。不能第一组封存后就训练，再边看训练结果边补其他组。训练时
   每组再生并精确核对原hash，随后只取不可变缓存的batch，seed前后继续
   检查身份和hash。trace两角色的tau都来自同一原full C32，不能从logits
   重建C或借用各向同性几何C。
3. 派生RAM缓存可释放，但其来源、hash与运行记录保留；独立审计按冻结
   来源完整再生所有pair并与原hash比较，再完成原合同要求的独立数值复算。
   再生不一致时保留失败，不能修改hash、容差或仅报告有利子集。此方案
   尚无实际缓存再生证据，不能把合成adapter检查当作通过。
4. 若中断，先核实原任务终态，再按冻结恢复规则再生；没有原hash可核对
   的未封存组不能冒充已完成。空间不足不能触发降低精度、删除历史证据或
   缩小人口的隐式回退。
5. 另报缓存生成时间及内存；最终配对成本仍逐角色重算自身准备、积分、
   tau校验和完整learned attention/V/head。RAM缓存不计为算子加速证据。

完整预封存、训练前再生和独立审计再生是三项实际工作，不能隐藏由内存
策略增加的再生时间。独立草稿审查指出了原逐组立即训练建议与R12时序的
冲突，已在真实执行前改为上述完整预封存方式。

以上使后续实现有明确资源路径，但不解除上游审计和科学门槛。当前未分配
几何缓存、未打开真实训练数据、未导出或启动任何role/seed训练。
