# R20 固定中心构造：实施准备稿

2026-10-31。**仅准备，尚未实现、冻结或执行 R20。** 本稿只读既有源码、锁、
已接受的选择/完成元数据和磁盘余量；没有运行新 forward、读取新数组或打开
目标标签。R19 探索 screen 已由 root 验收，但不构成 Gaussian 候选或下游
科学通过。本稿按 root 最新选择收缩为固定 `h=.25,T=10` 的旧 TRAIN 否证；
不运行 12-grid，不训练网络，固定构造失败后不在同轮搜索参数救援。

设计依据是 [固定方案审查](round20_anchor_fixed_design_review.md) 与
[既有证据复用边界](round18_baseline_reuse_boundaries.md)。本文提供实现和
资源清单，最终机器 gate、数值预检、角色清单及执行授权须以另冻协议为准。

## 1. 固定主构造及最小代码改动

令旧支撑均值、协方差为 `mu32,C32`，单位已经是 reference-8；令 a 来自
原 R13 `predicted_centers(inputs, query_bounds)` 的 float64 像素输出。root
已选择如下精度，不将新参数重新舍入为 float32：

```text
a64    = a_px64 / 8
delta  = mu32.to(float64) - a64
Ca64   = C32.to(float64) + delta[:, :, None] * delta[:, None, :]
tau    = T * h² / (h² + trace(Ca64))
P      = softmax64(log(native_gaussian_mass64(a64,Ca64,h) + area_floor64)
                   + (tau.to(float32) * content32).to(float64))
```

`Ca` 是旧支撑关于指定 a 的二阶矩，包含沿 `mu-a` 的秩一增宽；它不是旧
中心协方差或预测置信度。保留旧 C 中已有 pixel covariance，不再添加一次
pixel footprint。a 引入了旧区域矩以外的合法源中心身份，故必须保留 same-a
point/fixed。原生 optimized 3DGS 已有 primitive 中心，本次图像区域修正不
自动证明其编码新颖性、真实 GS 成功或有限维 RoPE/QK 表示已成立。

最小新模块仅负责 a/Ca 准备、固定角色分派和严格 allowlist；直接调用冻结
数学，不重写积分或引入网络/缓存框架：

- `models/round10_cell_integral_fast.py::gaussian_cell_mass_fast`：原 GL32
  native Gaussian 积分，依赖 `round10_cell_integral` 的 Legendre/CDF。
- `models/round15_validation_pe.py::{prepare_key,geometry}`：原 key 格、24px
  children 和一次 floor。新 Gaussian 角色可显式映射为 full1 积分调用，
  同时在外层保留新角色身份；不得把新 a/Ca 的结果标记成旧 full1。
- `models/round17_trace_pe.py::{trace_temperature,content_bias,probabilities}`：
  已支持 float64 C；温度先转 float32 再乘内容，geometry/softmax 均 float64。
- `models/round10_partition_pe.py::{pool_content,prepare_support,group_mean}`：
  原 float32 支撑和内容池化顺序。已有缓存可用于本轮准确率准备；独立审计
  仍须核对缓存、输入和其原执行语义，不能以 allclose 代替来源身份。

模型只接收对应角色所需的已批准输入；文件、目标、选择和线程管理留在
producer。a 仅依赖 `q_predicted_depth,q_K,k_K,q_to_k_transform,bounds`；
相机 float64 guard、双线性取深度、前向阈值、safe-z 和投影截断均沿原函数。
完整 R13 脚本有其它功能，宜沿 R19 的 AST 提取仅该函数并绑定源码/AST hash。
原 `a_front/a_fov` 只作描述，不筛掉任何中心或区域评价行。

## 2. 新计算角色、几何共用与不得混淆的 alias

下表是 root 已指定的新角色。所有角色保存全 14 条件结果；定位参考
在其中 q8/q16/q24→g8 提取，避免另存一套重复 Stage A 概率。表中 `T*content`
沿旧 float32 乘法，`tau(Ca)*content` 沿 TRACE 的 float32 乘法。

| 完整算子 | mean / 积分 covariance | h / T | 内容系数 | 来源/用途 |
|---|---|---|---|---|
| anchor_trace | a / Ca | .25 / 10 | tau(Ca) | 唯一固定 primary |
| anchor_mean_only | a / C | .25 / 10 | 同 tau(Ca) | 只归因 Ca 的增宽，不晋升 |
| anchor_constant | a / Ca | .25 / 10 | T | 同几何温度归因，不晋升 |
| anchor_iso_matched | a / tr(Ca)/2 I | .25 / 10 | 同 tau(Ca) | 完整形状归因（各向异性及方向），不能仅归因 rotation；不能改用 iso 舍入后的 tau |
| a_gaussian_point_regional | a / 0 | 1 / 5 | T | 旧 regional point/strong point/legacy point 的同 a 配置 |
| a_gaussian_point_localizer | a / 0 | .25 / 5 | T | 旧 Gaussian PCK 选择 |
| a_gaussian_point_legacy_pck | a / 0 | .5 / 5 | T | 旧 legacy point_pck |
| a_gaussian_point_matched | a / 0 | .25 / 10 | T | 同 primary h/T；不得只用此一个 point 对照 |
| a_gaussian_fixed_regional | a / area_px/(64·12) I | .5 / 5 | T | 旧 regional fixed 与 legacy fixed 两角色 alias |
| a_gaussian_fixed_matched | a / area_px/(64·12) I | .25 / 10 | T | 同 primary h/T 的固定范围对照 |
| a_quartic_point_regional | a / 0，quartic 核 | 1 / 5 | T | 旧 quartic regional 配置 |
| a_quartic_point_localizer | a / 0，quartic 核 | .25 / 5 | T | 旧 quartic PCK 选择 |

这是 **12 个完整新算子、10 组独立 geometry**（8 Gaussian、2 quartic）：
primary/constant 共用一组积分；Gaussian point 的 `.25,T5` 与 `.25,T10`
共用一组积分。其他相同角色名不
自动构成 alias。共用 geometry 只节约开发 forward；完整 P、角色记录和
科学比较仍独立保留，未来角色计时各自承担自身准备和执行成本。

旧选择中**没有单独 fixed_pck**。不能从已见旧 calibration_summaries 临时
挑一个然后称为“旧 PCK 预选”。fixed regional 与 legacy 都是 `.5,T5`。
这些新 a 版本只是转移旧配置，并非新 a 的最优 point/fixed。固定阶段若通过，
同信息基线的等预算校准可另冻；本阶段失败不立即转为新网格搜索。

root 已决定同 a 的两个 quartic point 角色均全 14 计算，以免新中心改变
点核家族优劣后漏检。它们不能 alias 为 Gaussian point；旧 mu quartic 两
角色也完整继承。`anchor_mean_only` 固定同主选 tau(Ca)，只隔离秩一几何
增宽，不是一个温度又换为 tau(C) 的完整 `(a,C)` 新候选。

## 3. 精确旧选择及继承结果路径

主选择源为 `results/round17_regional/selection.json`，SHA256：
`eadfd5071378c7239676ce58a0364ec2cb44c80ea3f198be800819691a700054`。
以下只是读取既有选择，未按新 a 重选。grid index 使用原 `[.25,.5,1,2] ×
[0,5,10]` 顺序，原精确平局次序保持。R20 主选固定，无新 winner tie-break。

| 旧选角色 | family | h / T | grid | 继承/别名说明 |
|---|---|---|---|---|
| R17 primary | trace_full | .25 / 10 | 2 | 旧 mu/C 的原主选，完整保留 |
| regional quartic point | quartic_point | 1 / 5 | 7 | 旧强点候选族 |
| regional Gaussian point | gaussian_point | 1 / 5 | 7 | strong point；同 legacy point |
| regional fixed | gaussian_fixed | .5 / 5 | 4 | 同 legacy fixed |
| regional full1 | gaussian_full1 | .25 / 0 | 0 | 同 legacy full1 |
| regional full4 | gaussian_full4 | .25 / 0 | 0 | 与 legacy full4 不同 |
| regional quad4 | gaussian_quad4 | .5 / 0 | 3 | 与 legacy quad4 不同 |
| regional quad16 | gaussian_quad16 | .5 / 5 | 4 | strong quad；同 legacy quad16 |
| Gaussian point localizer | gaussian_point | .25 / 5 | 1 | 原 R15/R17 全局 PCK 选择 |
| quartic point localizer | quartic_point | .25 / 5 | 1 | 原 R15/R17 全局 PCK 选择 |
| legacy point_pck | gaussian_point | .5 / 5 | 4 | 仍是单独保护角色 |
| legacy full4 | gaussian_full4 | .5 / 5 | 4 | 不并入 regional full4 |
| legacy quad4 | gaussian_quad4 | .5 / 5 | 4 | 不并入 regional quad4 |
| 历史 quartic primary | quartic_full | 1 / 0 | 6 | 历史失败参照，非新候选 |

旧 regional 行来自 `results/round15_regional/evaluation/<scene_slug>/rows.json`，
旧精确中心计数来自 `results/round15_feasibility/evaluation/<scene_slug>/rows.json`；
R17 原主选分别来自 `results/round17_regional/evaluation/<scene_slug>/rows.json`
及 `results/round17_feasibility/evaluation/<scene_slug>/rows.json`。三 scene slug
为 `harvard_c3__hv_c3_1`、`harvard_c5__hv_c5_1`、`harvard_c6__hv_c6_1`。

复用须绑定 R17 selection 中的 regional lock/forward/summary/completion 与
`results/round17_regional_audit_path_repair/audit/verification.json`；Stage A
decision/lock/forward/completion 和
`results/round17_feasibility_independent_audit/verification.json`；以及 Stage A
lock 的完整 `baseline_bindings`。其内已绑定 R15 的 source/锁、逐行表、
completion、接受决定和两个独立审计，包括修正后的 feasibility audit 路径。
`results/round10_partition_b/selection.json` 提供 legacy 选择源，不能用其
旧中心近似计数代替后来 R13 精确中心人口。

新联合汇总按 scene/pair/scenario/condition/operator/config 严格 join，核对
旧 input/target/probability hash、shape、人口和加性统计后重汇总。清楚标记
`inherited_evidence`，不声称本轮重跑了旧基线。只有完整算子与全部身份不变
才能引用；新 a/Ca 的任何角色不能引用旧 mu 的 P 或 KL/PCK 充当新计算。

## 4. cohort、合法输入、cache 与封存

严格沿 `results/round17_feasibility/lock.json` 中 48 个 TRAIN entries：三场景
各 16 对。该锁 SHA 为
`57e5a498b38ca2cfd761ff58d1519e1036f96ff69b97c778fb4debebf5980cf0`。
保留全部原九条件 `{g8,g16,mixed}²`，加 `g8→g24,g24→g8,g16→g24,g24→g16,
g24→g24`。四 native token 数为 **432、108、243、48**。

| 内容 | 精确来源模板/要求 |
|---|---|
| 原合法预测与特征 | `results/round10_partition_data/train/<scene>/<pair>/inputs.npz`，绑定原 manifest/input hash，不重新生成 backbone 预测 |
| 旧 mu32/C32/bounds | `results/round17_size_temperature_attribution/forward/<scene>/arrays/<pair>/supports.npz`，`g8/g16/mixed/g24__mean,__cov,__bounds` |
| 支撑来源接受链 | `results/round17_size_root_acceptance_20261031.json`，其绑定的 producer/audit 锁、source snapshots、forward/completion/verification |
| R19 已接受 a 与中心身份 | `results/round19_center_semantics/forward/arrays/<scene>/<pair>.npz`，`g8/g16/g24__a_px` 与原 bounds；只在新前向时按锁读取 |
| R19 接受链 | `results/round19_center_root_acceptance_20261031.json` 及 producer/audit 锁、forward seal、双 audit verification、completion |
| 精确中心目标，封存前禁止打开 | `results/round13_target_conflict/targets/<scene>__<pair>.npz`，沿原 target_contract 的 expected hash 与 `*_point_xy,*_point_visible` |
| 原区域目标，封存前禁止打开 | 原 pair 的 `labels.npz`，沿原 TRAIN manifest expected hash |

R19 只有三 query 尺度，**没有 mixed a**。R20 必须按 mixed 自身 bounds
执行同一个 R13 函数。最简单的合同是四 query 都重算 a，并要求三重叠尺度
与已接受 R19 exact array identity；mixed 沿独立投影参考检查。a 原函数的
file/source/AST SHA 已写在 R19 lock 的 `source_function`，不得只绑定名字。
R19 lock SHA 为
`81a4cb07b2c2ee934c43049a1fb78982727057a195950040b397d1adb591506a`。

key 仍为原 native key；24px Gaussian 格沿四个 12px 积分 children，先相加
再加一次 `1e-30 * area_px / 64`。children 不是新 token。共同区域投影仍是
area-only；内容仍是旧 float32 pooled/normalized 特征及其 float32 GEMM。
不得改 token、owner、front/FOV mask、key density 或 pool 顺序修复效果。

冻结前只读取 expected target hash 元数据。联合 producer/auditor 源、完整
递归 import 闭包、runtime、角色表、输入/支撑/R19/基线接受链先冻结；全部
48 对全部新 P、argmax、support/a/Ca/tau/diagnostics 与 canonical hashes
落盘后生成 global forward seal，独立前向审计通过才读取实际 target bytes
并验证 hash、评价。每阶段有不可覆盖 guard、started/failure/completion 和
完整文件清单。旧 TRAIN 已用于开发，这个顺序不恢复统计盲法。

## 5. 数值预检、评价和停止边界

新 mean/Ca 改变了积分域输入，旧 R17 数值通过不自动继承。另冻预检须覆盖
Ca 构造恒等式/PSD/symmetry、`tr(Ca)=tr(C)+||delta||²`、单位一致性、离场景
中心和极小 FOV、24px child 求和/一次 floor、同内容浮点运算，以及新 query
完整 dtype/device/形状/finite guard；独立参考不导入新生产实现。

`a=mu`、`delta=0` 时应回到旧 R17 的对应数学与精度路径，具体 exact-vs-
tolerance 检查须由协议预声明；`T=0` 回到 **anchored** Gaussian full1，
不要求与旧 mu full1 完全相同。`C=0` 但 `delta!=0` 仍有秩一 Ca，不能把它
当 point 极限；完整 point 极限是 Ca=0。旧独立积分误差与收敛要求不放宽。

全量报告原 mass-weighted KL/JS、common9/direct5、等场景等条件
`J=.5*common9+.5*direct5`，并保留 q8/q16/q24→g8 的原 center-valid 人口、
native argmax/first-index ties、整数 correct/valid 和精确 Fraction。
不得用 nearest-a 代替实际 point attention，不删 front/FOV 外或错误行。

root 已选的固定阶段 screen 必须在正式协议/实现中完整冻结：

- 原九项中心保护与八项 TRAIN 区域止损比较全部保留。
- 全 14 条件的每一条件 KL 相对旧 quad16 不得恶化超过 2%。
- q8/q16/q24 分别对 same-a Gaussian localizer、quartic localizer、legacy
  point_pck 保留最多 .5pp 损失，共新增九项精确中心保护。
- common9 与 direct5 分别相对 same-a regional Gaussian point、quartic
  point 和 fixed 至少改善 2% KL，共六项新区域要求。

它们是固定 TRAIN 的资源推进规则，尚非已执行的科学门槛。原 `1/200`
中心损失边界无 epsilon；分母、等场景规则和阈值方向须机器合同明确。
固定主选不换角色、不按效果换 T/h、Ca 系数或中心插值。通过这里只能授权
协议指定的下一步，不能称完成 VAL/六扰动/成本/actual-V/真实 GS/fresh。

## 6. 磁盘、内存、时间与依赖预算

纯形状算术：一个固定完整算子，每对全部 14 条件含 667,233 个概率元素，
即 5,337,864 B；48 对 **256,217,472 B = 0.238621 GiB**，672 个矩阵。

| 新完整算子组成 | 全量矩阵数 | float64 P 原始字节 | GiB |
|---|---:|---:|---:|
| 10 个 Gaussian 角色小计 | 6,720 | 2,562,174,720 | 2.386211 |
| 2 个 quartic point 角色小计 | 1,344 | 512,434,944 | .477242 |
| 12 个固定新算子合计 | 8,064 | 3,074,609,664 | 2.863453 |

这不是 12-grid：最后一行是 12 个**事前固定控制/归因算子**。支持/a/Ca/tau
只有 39,888 个 query 行，原数组为数 MB；argmax/元数据/源快照额外计入。
完整 P 每个只存一份，角色 alias 保存映射；auditor 可读取同一保存数组并
独立重算指标/预声明数值样本，不复制第二份全量 P。临时写入留足最大单
archive 空间，不能把压缩比当可用空间保证。

2026-10-31 此次只读 `df -B1` 快照可用 **12,611,764,224 B（约 11.75 GiB）**。
故固定规模优先保存全部 P，执行前再检查余量，并为 audit、failure 和后续
写入预留空间。旧 12-grid 多族方案的空间估算不再是当前方案。若实际余量
不足，须在冻结前明确逐矩 canonical commitment、全局封存后确定性重放且
每矩 hash 完全一致的替代合同；不可看到效果后删除数组或临时降低审计。

内存按 pair/condition 流式：最大 `432×432 float64` 矩阵仅 1,492,992 B；
无须把全部新 P 或全部 geometry 留在 RAM。保持旧积分 chunk 上限与来源
绑定，最多缓存当前 pair 的固定 geometry/content；future latency 不能把
这些缓存算成免费共享成本。a 的投影和 Ca 构造是新增准备成本，必须计入
以后每个角色自己的完整耗时。

旧完成时间元数据可作调度参考，不能当新速度证据：R17 三场景 CPU4 并行
Stage B 每场景 2,688 矩阵 forward 为 50.5–63.3 秒，重放评价为 73.7–77.2
秒，完整独立 audit 约 182 秒；旧 size study 每场景两固定角色、448 矩阵
forward 约 24.2–24.6 秒。它们含不同 I/O/复用，不能按矩阵数线性承诺 ETA。
root 提醒旧 quartic point 已测完整每矩约 .514/.647 秒，不能误套 Gaussian
point 的 .012 秒。两项新 quartic 控制的一次全量 pass 粗预算额外约十分钟，
具体随矩阵形状/固定 runtime 变化，不因较慢删除控制。本轮 Gaussian 部分
按分钟级工作预算，预检、完整绑定/审计与 quartic 工作量合计保守安排
数十分钟；真正预算由冻结后的固定
runtime 验证，不构成性能结论。旧 CPU4/Torch2.6.0+cu124/NumPy1.26.4 路径
与 R19 CPU1/NumPy 投影路径须显式统一并记录线程池，不能默默混用 profile。

最小新增实现预计为一个纯算子模块、一个 producer、一个独立 auditor 和
对应合成测试/协议；不改任何 R17/R19 冻结源。闭包至少绑定上述 Gaussian
与 TRACE、支撑/内容、原分格及 R13 AST 源；同时绑定
`models/round15_quartic_tiled.py`、其纯数学依赖与独立参考。metrics 可复用
既有已冻结定义，但新独立审计必须自行重汇总、核对全量人口和 gate。

本稿到此仅完成实施准备；没有新模型实现、测试执行、真实 forward、标签
读取或新的科学结论。
