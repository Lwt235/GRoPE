# R17 旧验证之后：条件分支草稿

状态：旧 VAL 正在按已冻结合同执行；本文没有读取其输出，也不预判通过。
仅依据既有协议与 [下游接口计划](round17_downstream_implementation_plan.md)
作准备，不授权训练、GS 实验或新数据访问，不改变任何旧门槛。

规则来源以 [R17 候选合同](../ROUND17_CANDIDATE_PROTOCOL.md)、
[R12 实际 V 合同](../ROUND12_DEPTH_PROTOCOL.md)、[R7](../ROUND7_PROTOCOL.md)、
[R8](../ROUND8_PROTOCOL.md)、[R9](../ROUND9_PROTOCOL.md) 为准。
[继承清单](round11_validation_inventory.md) 内已标为废止的深度草案不能覆盖 R12。

## 共同入口：先接受审计，再解释效果

必须等待两场景完整前向、评价、计时和独立审计。实现/来源审计通过与科学
门槛通过分开记录；不能把完成运行、数值一致或部分条件改善当作整体通过。
本轮 primary、h/T、19 个语义角色及全部旧门槛保持冻结，不按 VAL 结果重选。

若来源、数值或重放检查失败，本轮效果暂不可作为证据。保留失败，另立只修
工程问题的来源锁和复查范围；明确修复发生在效果产生前还是后，不改统计、
人口或模型来修复审计。若审计通过而科学门槛失败，则保留真实负结果，进入
下方失败分支。两种情况不能互换标签。

## 若旧 VAL 全部通过：先实际 V，再解决真实 GS 接口

旧 VAL 通过仅支持这一个固定候选在反复使用的给定相机开发人口上成立。
它不能修复 R7 的独立深度负结果、R9 的 GS/RGB 负结果，也不能单独完成
“一般、精炼、实际有效”的目标。

### 实际 V：保留任务，在线接入 tau

新建下游来源锁，不绕过旧 R12 因其自身 stress 失败而禁止启动的锁。最小
改变仍是一次 attention 内的内容系数：

```text
geometry = log(native Gaussian mass + one native area floor)
content = normalize(Wqk*q_rgb) @ normalize(Wqk*k_rgb).T
attention = softmax(geometry + tau(C,h,T)[:,None] * content)
value = learned_V(source_rgb_feature, predicted_source_depth_in_query_camera)
depth = original_query_residual_head(query, attention @ value)
```

使用冻结 R17 的 double 比值计算、float32 tau 与内容乘法；C 是原完整区域
支撑，单位为 reference-8，不是置信度。几何可以缓存，Q/K、内容、softmax、
V/head 必须每次在线运行；区域实验保存的最终概率不能替代 learned attention。
独立预检下游精度接口，尤其 float64 attention 到原 float32 V/head 的转换，
在看新深度分数前锁定；不得悄悄沿用 R12 float32 logits 以规避该检查。

| 必须保留 | 冻结时具体要求 |
|---|---|
| 容量与训练 | 原 41,345 参数单块头；五配对 seeds 11/22/33/44/55；每角色 1,000 steps、batch8、AdamW(.001,.0001)、clip1、valid log-depth L1；每100步含0评价，最早最低 equal-scene AbsRel 选 checkpoint。初始化与 batch 序列配对并哈希 |
| 数据与标签边界 | 原 c3/c5/c6 各192 train、c8/bm4 各64 old-val；全部 query-sensor-valid 中心，包括跨视图不可见、出界、无对应位置；GT/visibility/boundary 仅用于监督或统计，不能进入前向或删减损失人口；不作 GT 深度对齐 |
| 定位与压缩控制 | 至少保留 R12 point/point_pck/fixed、full1/quad4/quad16 的锁定角色；同时保留 R17 已锁定强点/定位/强 quadrature 角色，严格相同完整算子可 alias。训练前明确比较集合及固定 tie 顺序，不能看深度结果后挑慢的或弱的对照 |
| 两个归因控制 | constant-T full1 与 primary 同 h/T；trace-isotropic 几何用 trace(C)/2 I，但 tau 使用原 full C，逐位相同；两者不单独校准、不替换 primary。另保留各自旧校准 full1 |
| 自身信息控制 | 独立重训同容量 singleview，原 R7 query 自注意力、native geometry、h=.5/T=5，不借跨源缓存；另报原始冻结 query depth，不制造其五份独立种子 |
| 原始证据 | 完整历史、所有角色/种子选中权重和原始深度、每 pair 指标和相同计数；独立复算。记录实际 Q/K、V/head 梯度和在线内容路径；zero-V/reverse-V 是保存预测的诊断，不是参赛方法或选 checkpoint 依据 |

沿 R12/R17 的**旧开发**判定，不采用曾废止的“对 singleview/quad 均需2%”草案：

- 对预声明集合中最强 position 的 AbsRel 至少改善 **2%**，对 independently
  trained singleview **严格改善**；每项比较都须两场景种子均值改善，且至少
  **4/5** 个 seed 的场景宏均值改善。
- 最强 compression 按五种子验证 AbsRel 选，保留原 full1→quad4→quad16
  的精确平局顺序及最小集合。对它仍是二选一：AbsRel 至少改善 **2%** 且
  增量 PE+实际 V/head 时间 **≤1.25×**，或 AbsRel 恶化 **≤1%** 且时间
  **≤0.5×**；RMSE 对强 position 与 compression 的恶化均 **≤2%**。
- 配对计时重算每个角色自己的支撑、积分、tau 校验/算术及完整 Q/K/V/head，
  五旧场景首 pair ×五 selected checkpoints，3 warmup/15 repeats；逐格中位数
  后等 scene/seed 平均。缓存训练吞吐不是此成本，公共预测器/投影/传输单列。

如果扩展比较集合，新角色归属和 tie 顺序须在训练前另行明确，不能删除上述
旧最小集合或降低阈值。旧两个场景不提供总体置信区间。T=0 或极小 tau 的
有效 Q/K 学习情况必须如实披露；仅有非零梯度不证明任务受益。

R8 的“真路由＋原预测源值对 singleview ≥2%、≥4/5 seeds、两场景改善”是
已失败的诊断继续门槛，不能重新解释成 R17 已过的效用门槛，也不要求为了
宣布 R17 有效而让 privileged oracle 先变成正结果。真路由、传感器源值不能
成为正式前向输入，干预亦非严格性能上界。其失败削弱继续只修路由的依据；
本轮必须靠新的在线内容耦合实际任务证据回答，而非假定源值瓶颈已解决。
源值质量/多表面 patch/遮挡限制依然可能使本轮失败。实际 V 若失败，不用
更多区域成功、换更容易的任务或先开新确认把失败注销。

### 实际 optimized GS/RGB：先补语义合同，再实验

旧 R9 没有 content Q/K；直接放入 tau 不会改变其读色结果，乘 geometry
或 opacity 又改变了 R17 定义。必须先给出合法的 context-only primitive
query 与 image-pixel key 描述、相似度、归一化和训练/冻结时点；目前准备文档
没有选定该内容定义。若没有合理且简洁的定义，保留 GS 适用性未证明，不把
区域 C 的形式相似当作 optimized R/S 效果。

新合同须保留：真实 optimized R/s→世界协方差→相机投影的来源审计；R9
Teatime/Waldo kitchen 的 **24 context/4 calibration/4 gate** 固定图像角色；
同一 context-only 重建、renderer、geometry、opacity、SH0 容量与源视图集合。
原重建可完整绑定 provenance 后作为共享固定输入，不能冒充新 seed；若重建
则保留原 context-only 初始化及预算，不用 target RGB、旧全图点/track 选优。
相机仍来自既有全图 SfM，不能宣称全流程未见视角隔离。

- 保留同信息预算的 native point/fixed/isotropic、constant-T full、trace full，
  以及 bilinear、original-GS；旧 EWA/full 另作算子变化参考。source RGB 是 V。
  共享可见性与无源 fallback，各源分别归一化后等权融合；全目标像素计分，
  报 fallback 的渲染 alpha 贡献，不删难像素。
- R9 带宽单位是 pixel，R17 是 reference-8；mu/C/h 必须联合转换。先选固定
  物理转移或同预算旧任务校准并冻结；不按结果择优。native pixel 积分不可
  再叠加表示同一像素面积的 key I/12，旧五维恒等式也不证明新积分数值正确。
- 保留“全部预声明预测完成后才开校准 RGB，选择封存后才开 gate RGB”的
  顺序。新角色/内容合同需新预测清单，不能沿用旧18配置数量冒称新覆盖完成。
- 原验收门槛不变：对已校准 point/fixed/isotropic/bilinear 中 gate 宏 MSE
  最强 envelope 至少改善 **2%**，并对同一 comparator 两场景均改善；报告
  calibration-selected control。另与同 geometry 的 original-GS 比较，若只
  胜简化读色而输 original-GS，最多保留机制证据，不能称系统改善。

两个旧 GS 场景、每场景单次共享重建和确定性重跑不构成独立场景/种子证据。
未定义 content、实际 V 未过或 GS 门槛失败，都不能报告通用方案已完成。

## 若旧 VAL 科学门槛失败：停止这个固定候选，重新做开发

保留完整失败、各尺度/扰动/成本和归因结果，按原 gate 给结论。不得在本轮
改 h/T、换为 matched constant/iso、缩小人口或改阈值来称通过。通过部分
门槛可以界定有限用途，但不能偷偷替换用户的一般性目标。

下一轮先针对失败提出**一个**可证伪机制，优先合成和旧 TRAIN 诊断，再冻结
唯一候选、同预算强控制、中心可行性及区域止损条件。若改模型、数值路径或
合法输入信息，另建来源锁；需要重新覆盖受影响的先前检查。当前 VAL 只有
开发回归用途，不能再次称独立确认。没有新机制或可用信息时，应保留有界
结论和未完成项，而非只为寻找正分数增加模块/扫描配置。

反复使用同一数据的主要风险不会因 hash、独立审计或 bootstrap 消失：

1. **如实登记适应过程。** 保留所有候选、配置预算、失败和停止决定。当前
   c8/bm4、旧 GS gate 及此前已看过/预览的 place 都列为已使用。重新划分同一
   place 的近邻帧、换 sequence、重跑随机种子不恢复独立性。
2. **开发与确认隔离。** 新候选仅在旧 TRAIN/开发集合选设计、h/T、角色、
   checkpoint 和预算；旧 VAL 通过不能证明已经克服对这两个场景的适应。
   不在本轮失败后立刻打开新场景寻找有利分数。
3. **只确认冻结方案。** 只有适用区域、尺度、扰动、成本、实际 V 和 GS
   开发证据可信后，另冻完整联合确认合同。先按元数据与预声明无分数 QA
   规则选真正未用 place，排除所有旧/预览身份；所有替换及原因留痕。不要
   依据可见率、误差或预测质量挑场景/样本。
4. **一次确认，不回调再称确认。** 先封存模型、配置、checkpoint、分析、
   主要终点和失败规则，再完整前向/哈希，再读标签；全部角色使用相同群体。
   不在新数据重选 checkpoint、角色或指标。若确认失败，保留负结果；这些
   数据此后成为开发数据，后续修改不能在同一集合重新宣称独立成功。
5. **保留统计单位和尝试史。** 新协议继续继承 R7 的 paired scene×seed
   分析和各项主要阈值，以及区域/中心/跨尺度/噪声/成本保护；不能用像素或
   相关 pair 充当独立场景。旧 R7 确认使用20,000次 crossed bootstrap、95%
   区间及其预声明种子规则；新合同需明确继承和新增角色范围。若计划多个
   新候选/多次独立确认，必须在打开结果前声明总预算、停止与多重选择处理，
   完整披露全部尝试，不能只展示最后一次成功及其未校正区间。

五个未用 place 的确认仍只支持声明域、给定相机和任务，不证明跨任意数据域
的一般性。协议审计解决可追溯和实现一致性；实际 V/GS 与独立新人口的证据
解决不同问题，彼此不能代替。上述路径均为待上游结果触发的准备，当前没有
运行新下游或新场景实验。
