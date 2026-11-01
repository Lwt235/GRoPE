# Round 11：继承验证清单与后续确认草案

2026-10-30。依据 R6–R11 协议、报告及场景选择元数据整理；没有读取历史封存测试输入、标签或预测。本文不修改已冻结的 R11 门槛。R11 的旧开发结果即使通过，也只允许进入以下验证，不能据此声称“所有此前验证通过”。下列后续数值是**尚待执行前锁定的草案**，不是已完成成绩，也不用于改变 R11 的判定。

**后续修订记录：** 关口 C 的下游比较合同已在任何 Round12 训练或深度分数产生前，由 [ROUND12_DEPTH_PROTOCOL](../ROUND12_DEPTH_PROTOCOL.md) 取代。正式草案保留 R7 对最强 point/point_pck/fixed 的 ≥2% 效用门槛和对 singleview 的严格改善；full1/quad4/quad16 中按验证 AbsRel 选最强者，另按预声明的精度/增量 PE+实际 V/head 成本二选一门槛比较。下文原先“对含 quad16 的强融合和 singleview 都需 ≥2%”是已废止草案，保留仅为审查记录。此修订不改变 R11 或任何历史已运行门槛。

## 1. 哪些证据需要重新建立

| 已有检查 | 历史证据与限制 | 对 full4 的要求 |
|---|---|---|
| R6 独立传感器区域标签、四种尺度、强已知尺寸对照 | 五个新场景区域 KL 改善 6.08%，当时通过；仅规则网格与旧单高斯算子 | 重新确认。新混合算子、先投影后取矩以及矩形积分均改变了方法；不能继承旧效果数值。保留真实像素计数、公共群体和可见质量加权 |
| R6 中心定位保护 | 对专门按 PCK 校准的 point 有 −0.5pp 保护，旧方案通过 | full4 必须独立通过；区域 KL 不替代点定位。新场景要求配对场景区间下界大于 −0.005 |
| R6 五维精确 Q/K 与低开销 | 单高斯、共同 key 协方差时成立 | **此结构结论不适用 full4**。log 混合矩形质量不是同一五维二次式；只能声称四分量位置 bias。需要报告实际准备、积分、内容与 V 聚合成本 |
| R7 未参与校准的 24px 尺度 | 旧场景诊断区域改善，24→8 中心 PCK 有明显代价 | 重做未校准尺度检查并保留点保护；不能只报平均区域改善 |
| R7 真实 attention @ V 深度预测 | 新场景 AbsRel 仅改善 0.335%，区间跨零；未胜过同容量单视图 | **失败未被修复**。必须在同一全查询、单次 attention、同容量头下重新开发并确认；区域匹配通过没有替代效力 |
| R8 真对应/传感器源值干预 | 真路由配原源值相对单视图仅改善 1.031%；传感器源值较有帮助 | 诊断，不是方法成绩或严格上界。提醒 full4 可能无法解决源值问题；不能将传感器辅助 V 加入正式方法以过关 |
| R7 实际 optimized GS 内部 R/S | 内部形状未胜过强位置控制；不等 key 协方差破坏五维精确式 | **尚未建立**。full4 使用 RGB 预测像素的区域矩，未直接验证 optimized primitive R/S。不能将参数化形式相同说成此项已过 |
| R9 实际 RGB 读取与 GS 新视角预测 | 两个旧开发场景 MSE 仅改善 0.146%，方向相反，分支停止 | **失败保留**。full4 没有直接对应的 primitive 输入接口；若要声称真实 GS/NVS 通用收益，需另有物理含义清楚的适配、相同控制及新证据，不能用区域通过注销此项 |
| R10 A/B 混合分区、强 quad16、成本、PCK | A/B 均失败；B 修复 key 矩形质量但未修复整体效果 | full4 重新通过全部冻结条件及独立审计才算旧开发通过。仍不声称 query 拆并精确不变 |
| R11 无标签稠密预测分布保真 | 四个局部完整矩优于单高斯/稀疏分层中心；极小 FOV 尾部影响大 | 仅说明几何压缩误差较低；不能成为传感器真值或 downstream 的正结果 |
| 几何扰动/噪声稳健性 | R10/11 列为后续要求，R6–R9 协议中未发现一个可继承的完整噪声验收表 | 标为**未完成**，不能虚构历史通过。先声明具体扰动与失败条件，再执行 |

“此前所有验证”应逐项有证据与适用范围。数学上不再适用的五维恒等式不能强加给另一函数；但深度任务失败、实际 GS/RGB 失败也不能通过写成“不适用”而消失。若最终只有区域匹配成立，交付的是有限范围的区域 PE，原来更一般的研究目标仍未完成。

## 2. 立即顺序：全部在旧开发数据上完成后才开新确认

**关口 A：完成 R11 独立审计。** 核对冻结链、全部保存概率、各角色共同群体、校准选择、强基线、逐条件指标和配对计时。固定 full4 为候选；不能因为 iso4 较快或局部较好就在本轮事后替换。

**关口 B：冻结尺度与扰动回归。** 沿用 R10 的 48 训练/32 旧验证 pair、已选择 h/T、所有原角色与主归一化约定，不重校准。先锁协议与实现，后打开这些新增条件的分数。

- 尺度条件固定为 8→24、24→8、16→24、24→16、24→24。全部原始像素仍参与，24px 内容由对应 3×3 base8 特征均值后归一化，full4 保持 2×2 的四个 12px 分层；quad16 为 4×4 的 6px 分层。按各条件自身真实硬 cell 计数评价，方法间目标完全相同；不把不同 key 类别数的 KL 直接解释为尺度优劣。
- **24px 实现前置问题：** B 的 GL32 依据覆盖的最大原生 key 边长为 16px。24px key 必须确定性拆成四个 12px 子矩形后求积分并相加，再取 log；这是积分分块，不产生额外 token、内容或自由参数。先对独立 SciPy/CDF 参考验证，再运行效果评价。24px 与 common16 不能直接按父子矩阵聚合；新增条件用 direct KL，原九条件 common16→8 主指标保持不变。
- 尺度开发通过要求：五条件等权 KL 相对已锁定 point 至少降低 2%，两个旧验证场景方向均改善；各条件相对已锁定 quad16 恶化均不超过 2%；24→8 的 PCK@8 对已锁定 point_pck 下降不超过 0.5pp。报告 fixed/full1/iso4、各场景和各条件，不能以目标 stride24 的量化上限掩盖 24→8 退步。
- 扰动固定为六个独立状态：前向预测深度整体乘 0.9 或 1.1；相对相机变换在目标相机坐标系左乘绕 y 轴 +1° 或 −1°；相对变换在目标相机 x 方向加 +0.01m 或 −0.01m。分别改变**前向**几何，传感器标签及评价相机保持原始值；不组合扰动、不对各方法使用不同误差。深度缩放对双方 RGB 预测缓存一致施加，重新投影像素/footprint，不能只乘最终 covariance。
- 在原九分区上每种扰动分别计算原 common-grid 指标。每个状态相对受同一扰动的 quad16，KL 恶化不超过 2%，g8/g8 PCK@8 对受同扰动 point_pck 下降不超过 0.5pp；所有输出有限。保留绝对退化曲线及零扰动结果。这些是受控偏差敏感性检查，不称现实传感器噪声分布或完整抗噪证明。

**关口 C：原深度任务的真实 V 验证。** 这是阻止“区域效果替代实际任务”的最低必要检查。保持 R7 的全查询人口、41,345 参数的单次 attention/V/残差头、损失、训练预算与五个配对种子；只把几何 logits 换成候选或相应控制。几何来自同一冻结 RGB 预测和给定相机，V 仍是实际可获得的 RGB 特征与预测深度，没有传感器源值、GT 可见性或置信度附加模块。

具体旧开发合同建议如下，在训练前另行锁定：

1. 训练 c3/c5/c6 各 192 pair；验证 c8/Brown bm4 各 64 pair。所有 query sensor-valid stride8 中心参与，包括另一视图遮挡、出界与无有效对应的位置。沿用 R7 半像素、米制变换、边界定义、无 GT scale alignment。
2. 角色为 full4、iso4_matched、full1、quad4、quad16、point、point_pck、fixed、同容量 singleview，另报冻结单目基线；保留各角色已锁定区域 h/T，不新增调参网格。相同配置的重复角色仅标为等价，不制造重复证据。基线与候选均重新训练，不能直接拿旧 R7 checkpoint 作唯一基线，因为几何生产与算子已改变。
3. 五种子 11/22/33/44/55；1,000 steps、batch8、AdamW lr .001/weight decay .0001、gradient clip1，log-depth L1。每 100 step 包括 0 评价，选最早最低 equal-scene AbsRel。必须保存完整历史；若预算不足、未收敛，结论限定该预算，不能按方法单独加时。
4. strong fusion 为 point/point_pck/fixed/full1/quad4/quad16 中五种子验证 AbsRel 最低者。full4 的验证 AbsRel 对 strong fusion 与 singleview **都至少降低 2%**，每个比较均需两个场景改善及至少 4/5 seed 宏平均改善；对 strong fusion 的 RMSE 恶化不超过 2%。这是进入新数据的开发筛选，不能用两个反复使用的场景给总体 CI。
5. 保存深度数组、原始误差和有效数，独立复算；验证 attention @ V 和 V/head 梯度实际参与、置零/置换 V 的冻结干预会改变预测。参数相同但梯度路径不同的控制要注明。几何可预计算以节约训练，所有角色相同处理，完整系统成本另计。

这项实验可能再次失败，R8 已给出合理风险证据。失败后不得先开新区域确认来补充“成功数量”，或仅用更容易的任务替代全查询深度验收；需要新的信息假设，仍在旧开发集先验证。

## 3. 若需要实际 RGB 作为 V：一个合法但不替代原验收的备选

最小可研究的观察读取任务是**固定遮挡 query 区域的跨视图 RGB 补全**：遮挡位置在任何标签分数前由帧身份固定；query 的 RGB 特征、预测深度及所有衍生几何都必须在遮挡后的图像重新计算，source 为完整且不同的 RGB 帧。V 可直接为 source cell 的真实观测 RGB；监督是遮挡 query cell 的原始 RGB。所有遮挡区域进入主评价，不能按 GT 可见性筛选。

需要同预算 singleview/query-only、强 point、quad16 与 full4，共同浅头或共同直接读取方式；输入 RGB 拷贝、几何和内容的未遮挡缓存、同帧 source、同位置 query skip 都必须排除。单视图可以使用遮挡后的全部合法上下文，不能人为削弱。分层区域均值颜色与像素恢复应分开声明。预先锁定至少 2% MSE 改善且场景一致的开发标准，再考虑独立确认。

这能够测试实际观测 V，仍是**新任务**；既不自动修复 R7 深度，也没有重现 R9 的真实 optimized GS/NVS 接口。当前不建议为它跳过关口 C，更不直接下载新 RGB 选择有利场景。若返回 R9，必须先定义四个分量由哪些实际观测组成，避免任意把一个 rendering Gaussian 切成四个同义参数来制造接口。

## 4. 条件触发的新场景联合确认草案

只有 A/B/C 全部通过后才启用；在任何新 RGB 预览、预测或标签读取之前，锁定下列规则、所有旧开发选择/训练 checkpoint、生产与分析源码、数值例子、角色和完整指标。元数据资格与固定 RGB 去重 QA 可以在无分数条件下进行；每次替换及理由留痕。质量、可见率、几何误差和模型表现均不得用于换场景。

- 五个此前未用或预览的顶层 place，各取固定 48 个 query ordinal（30..N−91 均匀取整）及 +15/+60 source，共 480 pair。同一场景其他 sequence 不算独立 place。完整 RGB↔pose 序号和最近 sensor timestamp 先审计；缺档按预先顺序拒绝整个候选并记录，不挑 pair。所有成功入组 pair 保留。
- 候选 full4 与 B 的 point/center、point_pck、fixed、isotropic、full1、quad4/16，以及 R11 iso4、iso4_matched 均保留锁定配置。严格角色等价可共享计算，但保留等价标注。深度使用关口 C 的所有已训练种子与已选 checkpoint，不在新数据选择 step 或方法。
- 新区域主指标继续原 common16→8 九分区、同一传感器群体、可见质量加权后 condition/scene 等权。新增五种 24px 条件单独用同条件 direct KL；六种几何扰动另报。全像素预测几何与真实标签导出分离，先完成并哈希所有前向观察，再读取标签。这是给定 RGB-D 重建相机的条件评估。
- 区域确认要求保留全部 R11 效果/成本交集：对锁定 point KL ≥2% 改善，且 paired-scene 差值 95% CI 上界 <0；至少 4/5 场景改善；对强 quadrature 通过原二选一准确率/成本条件，各原条件 KL 恶化均 ≤2%；PCK@8 对 point_pck 的差值 CI 下界 >−0.005。准确率分支另外要求差值 CI 上界 <0；效率分支要求相对 KL 恶化 CI 上界 <+1%。同设备重新配对计时，不沿用 R11 的机器瞬时钟值。
- 为继承 R6 的已知尺寸保护，额外保留旧开发校准选出的 point/fixed 中较强者，8→16 与 16→8 各自要求 ≥2% KL 改善且差值 CI 上界 <0；8→8、16→16 相对恶化 CI 上界 <+2%。这是新确认额外预注册要求，不回改 R11。
- 24px 与扰动沿用关口 B 的阈值，并增加各主要非劣比较的 scene-bootstrap 上界条件。24→8 定位下界 >−0.005。不同 key 网格上的 PCK@8 不跨网格直接比较，提供同量化的对照。
- 深度继承 R7：full4 相对锁定 strong fusion 的 AbsRel ≥2% 改善，差值 CI 上界 <0，至少 4/5 seed 宏平均改善；相对 singleview 差值 CI 上界 <0；对 strong fusion 的 RMSE 相对差 CI 上界 <+2%。报告每 scene/seed、全查询/边界、raw singleview 与 V 干预。任何一项失败都不能报告“此前下游验证已通过”。
- 区域为 20,000 次 paired scene bootstrap；深度为相同 20,000 次 paired scene×seed crossed bootstrap，RNG 20261030。共享重采样索引，报告各 scene、leave-one-scene-out、场景来源组敏感性；不把像素/pair、确定性复跑或近邻帧当独立样本。五个 place 的区间仍不能证明跨任意数据域一般性。
- 独立审计须核对原始 sensor 取样与标签、全部角色 pair/condition/count、保存概率/深度与指标重算、完整来源锁、选择与统计。数值审计使用固定 metadata-first 子集的独立积分参考，检查 tiny-FOV、finite/normalization、FOV 外行为；全部失败案例保留。共享预测器、额外 covariance 准备、积分、V/head、缓存/传输、峰值内存与增量耗时分项报告。

完成后可支持的最高结论仍是“此冻结四高斯区域表示在声明的新 SUN3D place、尺度、受控扰动与实际深度任务上成立”。若实际 GS 内部 R/S 或 RGB 新视角任务尚无新证据，仍须明确列为未通过/未证明；不能改称通用 3DGS PE 已完整解决。

## 5. 元数据排除清单与待资格审查候选

仅读取四份旧 selection JSON 的场景身份，得到以下 **21 个必须排除的顶层 place**，包含预览后拒绝的 Brown bm1：

```
brown_bm_1, brown_bm_4, brown_cogsci_1, brown_cs_3,
harvard_c11, harvard_c3, harvard_c5, harvard_c6, harvard_c8,
harvard_robotics_lab, home_at, home_bksh, home_md, hotel_umd,
mit_32_d507, mit_32_g725, mit_46_6conf, mit_76_studyroom,
mit_dorm_next_sj, mit_lab_hj, mit_w59_conference
```

来源哈希：

| selection | SHA256 |
|---|---|
| `data/round5_sun3d/selection.json` | `cec3e618efefb6cb76373bc477cae0f527d56ba7f0dbce325c381af3c7af3ab8` |
| `data/round6_sun3d/selection.json` | `c7297b2cd040c271c49a90cf85e1d1e7893c4c57b2f77119a16580b72bad9631` |
| `data/round6_sun3d/selection_v2.json` | `4925a965e311908463b9ba0ebc5f932494cb3872b160777eda434613b8d17d1d` |
| `data/round7_sun3d/selection.json` | `be816055c073529fc450e3e210698ca26d5178fb2d1d70103c76b5a89d43598f` |

2026-10-30 只访问官方目录 HTML，保存于 [metadata inventory](../results/round11_confirmation_metadata/inventory.json)。以下是可供下一阶段资格审查的命名顺序，**尚未入组，也尚未验证相机完整性或物理场景独立性**：

| 顺序 | place / 优先按目录字典序审查的 sequence | 状态 |
|---|---|---|
| 1 | `home_ac/home_ac_scan1_2012_aug_22` | 仅确认目录存在 |
| 2 | `hotel_barcelona/scan1_2012_july_23` | 同上 |
| 3 | `hotel_graz/scan1_2012_aug_29` | 同上 |
| 4 | `mit_1_242/classroom_1242_oct_30_2012_scan1_erika` | 同上 |
| 5 | `harvard_computer_lab/hv_c1_1` | 同上；Harvard 物理重复风险需固定预览审查 |
| 6–10 备用 | `home_ag/apartment_ag_nov_7_2012_scan1_erika`；`hotel_beijing/beijing_hotel_1`；`mit_3_133/classroom_3133_nov_6_2012_scan1_erika`；`home_han/apartment_han_oct_31_2012_scan1_erika`；`hotel_grenoble/scan1_2012_july_11` | 同上 |

同 place 若第一 sequence 元数据不合格，可按目录序检查下一 sequence，仍只占一个 place；一旦入组，不因观测效果改选 sequence。候选表不是已经可用的数据清单，不能预写 480 对完成。此次没有下载新 RGB、sensor depth、pose 文件，没有跑新场景预测或读取测试指标。官方目录来源为 [SUN3D data](https://sun3d.cs.princeton.edu/data/)，每份子目录 HTML 的 URL/内容 SHA 已保存。

依据：[R6 协议](../ROUND6_PROTOCOL.md)、[R6 报告](../round6_report.md)、[R7 协议](../ROUND7_PROTOCOL.md)、[R7 报告](../round7_report.md)、[R8 协议](../ROUND8_PROTOCOL.md)、[R8 报告](../round8_report.md)、[R9 协议](../ROUND9_PROTOCOL.md)、[R9 报告](../round9_report.md)、[R10](../ROUND10_PROTOCOL.md)、[R10B](../ROUND10B_PROTOCOL.md)、[R11](../ROUND11_PROTOCOL.md)。
