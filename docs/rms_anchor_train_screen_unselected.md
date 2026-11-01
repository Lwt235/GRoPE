# 未选择：RMS 三维 anchor 的旧 TRAIN 最小筛查准备

2026-11-01。仅据旧源码、R20/R21/R25 已接受报告及静态推导。**未选择、非正式协议、不授权实验。** R37 仍在运行，未观察中间结果；本稿未读取数组、标签或执行积分。

## 待检验的一个问题

待检验：**三维 Gaussian 关于合法预测中心的二阶矩，经 RMS 投影后，能否同时满足原区域与中心必要门槛？** R37 未校正、零内容的72例即使通过，也不覆盖新 anchor、内容或完整2304人口。

唯一建议提交后续选择的主臂是 **3D second-moment anchor + 冻结旧 b32**。未校正 Gaussian 和 zero-content 只作诊断，不事后挑赢家。R21 的形状臂改善 KL 却损害中心，R25 重尾替换也未改善区域；此处先检验取舍，不同时搜索温度。

## 输入与算子边界

patch 支持保留点间散布和 within-pixel covariance，沿 [R23 QR 构造](../experiments/round23_ray_applicability.py)。a₃ 取 [R13 `predicted_centers`](../experiments/round13_mixture_mechanism.py) 的 `target`，在 target-z clamp/screen clip **之前**：

- 仅使用合法 `q_predicted_depth`、q/k 相机及 q→k 变换。源预测深度沿原规则 `nan→10, +inf→20, −inf→.1`，裁至 `[.1,20]`，在原中心射线双线性取样并反投影。
- 不从旧 a₂ 反造 a₃；旧 a₂ 已有 target-depth `.1` 下界及 `±1024` 裁剪。新路径不修补 target-depth、不按 front/FOV 排除行、不读传感器深度或 GT。
- 原均值、因子及 a₃ 用同一 [R28 线性映射](../experiments/round28_rms_applicability.py)进入 homogeneous reference-8 screen/depth 坐标，记 m、F、a；不先作透视除法。

待选主几何为

```text
Q = F Fᵀ + (m−a)(m−a)ᵀ
U ~ N(a,Q)
```

用 `[F, m−a]` 的 QR 得因子，不裁特征值；沿原 RMS XY 正则 h=.25、严格 Z>0 投影、原生 cell 及一次面积 floor。新统计/舍入须冻结，不宣称逐位等于旧二维矩。

内容直接继承接受的 R20 support：q24 `[48,128]`、k8 `[432,128]` 和 `τ32[48]`。用原 Torch **整矩阵** float32 路径计算

```text
content32 = Q24_32 @ K8_32.T
b32 = tau32[:,None] * content32
tau = 10 × .25² / (.25² + tr(old_Ca))
```

复核旧稳定 float64 τ 公式转 float32 的精确一致性，沿用封存 τ32；不重归一化、不换逐行/NumPy GEMM。b32 转 double 后合入 log-floored mass，沿 RMS log 归一化与 first-index argmax，见 [R25 输入接口](../experiments/round25_student_screen.py)。zero 可复用 raw；未校正几何需另算，其工作/审计覆盖单列，不能冒充主臂4608参考的覆盖。

## 固定人口、指标与八门

原48对旧 TRAIN（每scene16对）及原顺序，q24→g8，每对48 query/432 key，共2304 query。key 为144×192图像的8px cell，边界 `(pixel_boundary−.5)/8`。失败、负深度、视野外、低质量行全保留。

区域用原 `gt_xy/gt_visible`；中心用 R13 `g24_point_xy/g24_point_visible`，有效数382/402/312，共1096。区域按scene内可见质量汇总后等scene平均；PCK为8px、first-index winner及精确 Fraction 等scene平均。完整报告 KL/JS/direct、计数、mean/RMSE及scene反例，可复用[独立指标原语](../experiments/round25_student_metrics.py)。

| 必要检查 | 原门槛 |
| --- | --- |
| strong_quad 区域保护 | KL ≤ 1.02 × 已接受参考 KL |
| legacy_quad16 区域保护 | 同上；两角色为同一完整配置，不是两个独立基线 |
| 三个旧中心控制 | PCK ≥ 各参考 PCK − 1/200：point_localizer、quartic_localizer、legacy_point_localizer |
| 三个 R20 anchor 中心控制 | 同界限：anchor_point_localizer、anchor_quartic_localizer、anchor_point_legacy |

八门不变，不加容差、不以 JS 替代 KL。旧参考未经新 a₃ 优化；后三者使用旧 a₂，不能声称严格同中心。R20 anchor Gaussian 仅描述性对照，不是第九门。

## 两种必须前瞻选择的执行顺序

**A．原顺序。** 联合 metadata/source/runtime 冻结 → 全2304输入独立审计 → 完整主臂生产、全量算术及全局封存 → 同一2304输入的4608次双阶参考 → 原数值门槛及根验收 → 标签、双路指标和八门。数值失败即停止标签，保持 [R25 原门禁](../experiments/round25_student_evaluate.py)。

**B．另行前瞻选择的单向 exploratory triage。** 新协议须明确“完整参考先于标签”的例外，不改旧冻结门禁。仍先冻结，审计**全2304输入**、完成**全部主臂生产/算术并全局封存**；前置项失败不读标签。之后才读旧 TRAIN 标签、独立重算同八门：

- 任一门失败仅允许 **stop implementation**。没有全量独立数值验收，不能冒称科学否证或证明 RMS 不可能有效。
- 暂时全满足也不得晋升。仍须对**同一冻结2304输入/封存预测**完成4608参考与原数值门槛；不因标签改 anchor、τ、h/T、人口、顺序或挑诊断臂。
- 不是抽样替代参考。另列数值哨兵仅可增加检查，不抵扣4608参考或 positive 的后续义务。旧 TRAIN 不是未触碰确认集。

两路径均不开 fresh、VAL、实际 V/GS；其余13条件、强控制及原后续验证仍未执行。

## v2 补充：八门之外仍有成本门（未选择）

[R17 成本合同](../ROUND17_VALIDATION_PROTOCOL.md)的 **zero 指零扰动，不是 zero-content/T0**。每个 strong/legacy quad 均须满足二选一：`KL≤.98×参考且时间≤1.25×`，或 `KL≤1.01×参考且时间≤.5×`。八个 TRAIN 门不包含此要求。R30 已记每query约.70–.87s、1,990,656节点，提示当前实现成本风险；但它与R25的不同算子、不同人口计时，以及R17的完整矩阵中位数，均不能直接作性能判定或外推总耗时。

可另冻**无标签成本必要条件筛查**，不优化当前 R37。候选和对照必须使用同一合法预测输入、相机、分区、机器与线程设置，执行完整算子。每次重算各自统计、anchor/因子、原生积分、floor、检查、内容和归一化/common投影；主臂还需支付旧 Ca/τ/pooled QK 的实际构造成本，不能把封存 b、几何或训练缓存当免费加速。公共预测/投影、输入输出与独立参考审计另记，不能把审计阶段总时间混作生产增量。

原计时为五个旧场景首对×common9共45格，每格3次warmup、15次测量取中位数，再等格平均。若只测旧TRAIN子集/q24，最多得到预先声明的实现停止诊断，不能判原45格门失败；q24不在common9。完整45格涉及两个old VAL场景的输入，仍需另行前瞻授权，并禁止读标签；这不由上面的TRAIN准备自动开放。

一种仅可拒绝的前瞻方案：先得某必需quad的完整45格中位数之和 B，再按固定顺序得候选完整已测格之和 S。未测格仅取0下界；若 **S>1.25B**，最终均值必超过该对照两条路线的时间上限，可停止当前冻结实现。等于不能拒绝；每个入和格必须完成原15次测量，不能从少量样本或单query外推。保留45格schedule与明确未测状态，只报下界，不能报告完整候选均值或PASS；≤1.25也仍缺KL条件。

这种前置顺序需新协议例外：原代码在效果评估完成后才计时，且采用角色交错轮转；“全基线先测”还改变了交错顺序，存在负载/热状态漂移，不能冒称原计时合同原样执行。须预声明运行控制、完整比较对象、停止规则及保守下界算术，避免移项/舍入在边界误拒。无新数据或计时，当前未证成本FAIL。

## 冻结前尚须决定的事项与一般性限制

按2304人口、4608参考及诊断几何，**读新输入前**声明 phase 时间/节点、单调用限额、失败/跳过留存及重试政策；不照搬72例预算、不外推耗时。精度、raw相对误差、P、direct-FOV、尾部及失败人口要求不因 triage 降低。新 a₃ 来源、因子和输入审计须补齐；72旧档案不是完整新人口缓存。

单 Gaussian 仍紧凑，但旧 b32 使整体尚非仅由三维参数定义。rank-one 项也改变深度、front mass 和 RMS 尺度，不是纯平移；常深度极限不保证一般 anchor 模态，非零深度方差下全平面普通矩可能不存在。实际 GS 的 a=m 时校正恒等，不可直接外推 patch 收益。见[anchor 反例](rms_anchor_small_depth_limit_unselected.md)、[内容代理审查](rms_content_spread_options_unselected.md)。未证明新颖性、有限精确 QK/RoPE、实际 V 效果或一般性。
