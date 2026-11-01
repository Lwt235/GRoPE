# 3DGS 参数化位置编码：仍需完成的验证

2026-11-01，在 R38 固定生产运行期间根据既有协议整理。本文是索引，不修改任何门槛，不选择新主方案，不允许由子集通过推断总体成功。当前尚无方案通过全部验证。

## 1. R38 是固定实现的必要条件筛查

唯一主臂为 `N(a, FFᵀ+(μ−a)(μ−a)ᵀ)`，采用既有 RMS 正深度投影；旧内容、τ、h 和生产求积政策固定。48 对 TRAIN、2304 查询、432 key；完整输入和输出算术审核及根验收后，才开放旧标签。两个 KL 比较及六个 PCK 比较只覆盖 24→8。

若 8/8 通过，仍必须在完全相同输入上完成 4608 次独立 32/48 参考及所有原 raw/P/FOV/tail/资源门。每次参考 250M 节点上限不变；R37 的失败及旧72例不能替代新人口。完整概率参考之前，只允许淘汰当前实现，不能正向晋升。[R38 协议](../ROUND38_RMS_ANCHOR_TRIAGE_PROTOCOL.md)

## 2. 完整旧 TRAIN

48 对、14 条件、60 项要求的交集：8 项宏区域规则、9 项旧中心保护、28 项逐条件对 quad 保护、6 项 same-a 区域控制、9 项 same-a 中心保护。common9 为 g8/g16/mixed 的 3×3；direct5 为涉及 g24 的原五条件。

common9/direct5 对旧强/legacy point 均须至少2% KL 改善；对 quad 的宏保护分别为最多1%/2%劣化，全部14条件各最多2%劣化；对 same-a point/quartic/fixed 的两种宏 KL 均须至少2%改善；三尺度 PCK 对全部指定 localizer 的损失最多0.5pp。还须增加合法的同三维信息强对照，不能借此替换旧控制。R38 单条件成功不能继承其他52项。[R20 协议](../ROUND20_ANCHOR_FIXED_PROTOCOL.md)

## 3. 旧 VAL、鲁棒性和完整算子成本

原 c8/bm4 人口和全部标签口径保留。zero common9/direct5 对各强/legacy point 至少2% KL 改善且两场景改善；14条件各对 quad 最多2%劣化；q8/q16/q24→g8 的 PCK 损失最多0.5pp。六个扰动为深度×0.9/1.1、相机 y 轴±1°、目标 x 平移±0.01m；每个状态保留同扰动 quad 区域保护及 localizer 中心保护。

对强/legacy quad 各自满足 `(KL≤0.98× 且 time≤1.25×) 或 (KL≤1.01× 且 time≤0.5×)`。原计时为五旧场景首 metadata pair × common9，共45单元；4 CPU threads，每格每角色3 warmups/15 repeats、固定循环交错角色顺序、先格内 median 再宏平均。各方法必须计入自身 support/anchor/QR、pooling/content、几何、积分、floor、τ、softmax 和公共投影。共享预测器成本单列。历史单查询时长不能除以历史完整矩阵时长。

若 R38 暂时通过，另行冻结扩展 common9 的合法构造与匹配成本验证，再投入完整参考。非负时间下界可以设计成仅允许停止的工程筛查，但若改变原交错顺序，不能据此宣称原正式成本门失败或通过；任何阳性仍须完整原门验证。[R17 VAL 协议](../ROUND17_VALIDATION_PROTOCOL.md)

## 4. 实际 learned Q/K 和 attention @ V

沿用41,345参数、576 TRAIN/128 old-VAL、配对 seeds 11/22/33/44/55、1000 steps、batch8、AdamW(.001,.0001)、clip1、log-depth L1、每100步评估并选最早最佳 checkpoint。保留所有 query-sensor-valid 位置及在线内容、梯度和 V 干预。

对最强 position 的 AbsRel 至少改善2%；对独立重训 singleview 严格改善；每个比较须两场景及至少4/5 seeds 改善。最强 compression 的原精度/成本二选一及对 position/compression 的 RMSE≤2%劣化仍必需。计时含完整 PE+V/head。废止草案中的“singleview也必须改善2%”不重新套用。R8 oracle 路由属于旧失败诊断，不是要求重跑的强制 oracle 门。[正式深度合同](../ROUND12_DEPTH_PROTOCOL.md)、[下游接口计划](round17_downstream_implementation_plan.md)

## 5. 真实 optimized GS / RGB

输入必须是真实 `R diag(s²) Rᵀ` 及审计后的投影。Teatime/Waldo 的24 context/4 calibration/4 gate、重建、renderer、geometry、opacity、SH0容量和 source 集保留；context-only 内容定义先冻结，各角色信息预算、visibility/fallback 相同，全目标像素计分。先封存预测，再校准，最后开 gate。

相对校准后的 point/fixed/isotropic/bilinear 最强 gate envelope，MSE 至少改善2%且两场景改善；另外比较 original-GS，若输给它，只能支持有限机制结论。实际 GS 的 a=μ，R38 rank-one 校正本身为恒等，patch 改善不能外推为 optimized R/S 有效。[R9 协议](../ROUND9_PROTOCOL.md)、[下游接口计划](round17_downstream_implementation_plan.md)

## 6. 最后的未使用数据联合确认

所有适用旧开发门通过后，排除全部旧/预览21 places，选5个真正未用顶层 places、固定480 pairs。先冻结最终模型、配置、checkpoint、角色、预算和统计，完整前向封存后再开标签，不依误差或可见率选样本。当前 home_ac/hotel_barcelona/hotel_graz/mit_1_242/harvard_computer_lab 仅为元数据候选。

20,000 draws、95% CI、固定 RNG20261030：区域为 paired-scene，深度为 paired scene×seed。区域对 point 至少2%改善且差值CI上界<0、至少4/5 scenes改善，PCK差值CI下界>−0.005，保留强quad精度/成本及逐尺度/扰动保护；深度对锁定强对照至少2%改善且CI上界<0、至少4/5 seeds改善，对singleview差值CI上界<0，RMSE相对差CI上界<+2%。失败后该数据成为开发数据，不能调参后再次称其为确认。

必须为最终方案另冻完整联合合同；旧 full4 草案不是 R38 的现成确认授权。[继承库存与草案](round11_validation_inventory.md)、[原深度确认合同](../ROUND7_PROTOCOL.md)

数学恒等式的检查须匹配实际模型：旧五维二次 Q/K 恒等式有共同 key covariance 的前提，R38不能冒认该恒等式，也不能因此免除实际任务、成本、GS和确认要求。结构精炼、输入审计或数值通过均不等于效果成功。
