# 第七轮独立协议审查：真实跨视图 V 融合预测 query 深度

2026-10-30。设计建议，尚无第七轮新测试标签或模型结果。本轮要检验第六轮位置编码能否带来**真实 attention/V 聚合的下游深度收益**，不是继续选择最有利的对应 KL。

## 首要泄漏与归因风险

1. **监督群体由 query sensor 决定。** 所有 query sensor 正值/有限的固定评价位置都参加，包括没有跨视图可见对应、被遮挡或目标深度缺测的位置。禁止用 target GT depth、一致性阈值或跨视图 overlap 筛 primary、attention key、训练权重或置信度。GT 标签有效 mask 只控制 loss/指标，不成为 forward 字段。
2. **要胜过容量匹配的单视图校正。** 已有 RGB 预测深度本就可用。训练一个小 head 可能只是纠正其尺度/偏差；不能把这种收益归因于多视图融合。原始冻结预测器与同预算 query-only head 都是必需对照。
3. **V 中的物理量要处在同一相机坐标。** key 原生深度 z 不是 query 深度 z。若 V 直接含预测几何，应仅由 key 的 RGB 预测点云与共同相机变换到 query 坐标，再聚合；不得用 target sensor 深度或 GT warp。若 V 只用视觉特征，应写清相机信息如何进入预测，避免单线性 head 被要求从未提供的姿态完成变换。
4. **冻结输入提供相同信息。** 四个位置控制共享 RGB features、预测深度、K/C2W、query 输入/残差、V、解码头和参数预算。full/fixed/isotropic/point 的唯一差异应是位置关系。不能让 full 额外获得预测深度/V，而 point 只看 RGB。

## 推荐最小任务合同

先固定 query 网格、预测量（例如米制 z 或 log z）、GT 的原生最近邻/区域聚合约定及输出分辨率。不得在看到验证/测试表现后改成更容易的区域平均或只报低分辨率平滑误差。若用 stride8 token 做 coarse depth，明确它是粗深度预测；插值到原图不自动证明薄结构或边界恢复。

建议结构保持小：query 预测深度与冻结特征作为共同输入，一次实际 cross-attention 把 key V 搬到 query，接共享小 head 预测 query 深度或其残差。若保留 query depth residual，所有 learned 控制也保留。五维 full 的支持来自 RGB 预测区域矩，不是优化 3DGS primitive R/S。

主线程当前实现计划为共享128->128 Q/K投影（identity初始化后L2）、冻结第六轮各角色h/T；V由128D RGB与变换到query相机的预测log z经过129->64 GELU生成，query分支同维度，拼接后128->64 GELU->1预测log-depth residual。各角色约41k参数，单视图self-context独立训练同预算。query原始深度参考采用同一个中心四像素bilinear平均，不能用另一读法削弱raw-predictor基线。该结构含实际AV及共用query残差，符合最小下游检验；冻结area-calibrated h/T的结论应称该算子的下游迁移，不能声称每个depth控制已各自达到最优。若保留point_pck h=.5/T=10的强中心控制，应在新test前登记。

提供的相机来自 RGB-D 重建，因此系统继续叫“RGB 预测深度/特征 + 提供相机”，非严格 RGB-only。sensor GT 只构造监督，禁止用 GT scale fit、GT pose refinement 或 GT normal/visibility 形成输入支持。输出正值约束、有限 fallback、训练深度范围与异常处理应在开发阶段统一冻结；不为预测坏的行丢弃评价。

## 最小公平对照与 V 审计

| 对照 | 目的 |
|---|---|
| 原始冻结 query 深度 | 区分整体系统改进与预测器原性能 |
| 同容量/预算 query-only 深度 head | 区分多视图帮助与单视图学习校准；实际额外容量如实记录 |
| point + 相同真实 V 融合 | 最强均值位置归因对照 |
| size-aware fixed + 相同 V 融合 | 已知 token 范围是否已足够 |
| same-trace isotropic/full | 投影支撑范围及方向/轴比的额外价值 |

query-only 应独立训练同预算，不能只把 full 模型测试时的 V 置零当作它。zero-V、打乱 key V 或阻断 V 梯度的诊断应证明实现确实走了 V；这些是冻结干预，可能受分布变化影响，不能单独证明必要性。对每个 V 投影、聚合/输出层检查非零梯度与改变 V 能改变输出；“调用 attention API”不等于有有效融合。

GT 缓冲置零/修改后，全部前向输入须逐元素不变；删除 target sensor 文件不应改变准备好的 observation 输入。输入不得含可暗示 query GT 缺测的 mask，除非该 mask 完全来自 RGB 预测且给所有方法相同使用；最干净条件是完整网格和固定 finite fallback。

## 数据、采样与独立性

训练沿用旧 Harvard c3/c5/c6，各192对；validation 为旧c8与Brown bm4，各64对。训练采样、归一化拟合、head/预算/PE超参数均只用这些训练/验证数据。若第六轮已固定的 h/T 沿用，应直接锁定；若重新训练/调节它们，所有位置控制相同策略与验证预算，明确这是新的下游适配条件。

新测试至少五个此前完全未使用的顶层 place，排除旧16组：`harvard_c3,c5,c6,c8,c11`、`hotel_umd`、`mit_lab_hj`、`mit_76_studyroom`、`mit_dorm_next_sj`、`brown_bm_1,brown_bm_4,brown_cogsci_1,brown_cs_3`、`mit_32_d507`、`home_at,home_bksh`。BM1虽未参与上轮正式测试，已看过其RGB，也继续排除。

按原生 SUN3D 官方目录选择候选，完整 RGB ordinals 与 camera rows 一一核对；固定48个query ordinals、offset15/60得到每scan96对，不按 RGB内容、GT有效率、估计几何质量或方法表现选 pair。所有固定 pair 保留。完整性失败或固定 RGB QA 显示可能同室重扫时，可以在测试前按非性能理由替换，保留原候选、目录/相机哈希、拒绝理由和新版本清单。

本代理完成的候选为 `mit_46_6conf/bcs_floor6_conf_1`、`mit_32_g725/g725_1`、`mit_w59_conference/conference_w59_nov_6_2012_scan1_erika`、`harvard_robotics_lab/hv_s1_1`、`home_md/home_md_scan9_2012_sep_30`。五个native端点/完整RGB-pose布局均通过，25,931个相机旋转的最大正交误差约1.08e-7；固定480对、1,362文件下载完成并校验哈希。selection SHA=`be816055c073529fc450e3e210698ca26d5178fb2d1d70103c76b5a89d43598f`。W59最近深度最大time gap29.725ms，其他<=16.63ms，保留全部pair并披露同步近似；没有按此筛选。

固定首/中/末 selected RGB 检查全部候选，疑似共享地点再用事先固定的12帧预览。不同 place-ID 与预览只能提供隔离证据，不能保证完全独立建筑；长轨迹可能跨房间。提供每scan及采集组敏感性，不将多个同楼房间包装成多个独立机构。下载RGB/depth允许；确认协议封存前不生成新测试标签、运行新测试模型或查看模型分数。

## 指标、五种子与确认建议

主指标建议预先选 **AbsRel**，对全部 query GT 有效位置汇总；另报米制 MAE/RMSE、log RMSE、delta1、原始预测深度对照及 GT 有效比例。不要通过测试 GT 做尺度对齐后计算主结果。先在每个 scan/seed 内按有效 query 汇总，再对scan和seed等权，不将像素/pair 当独立场景。

边界误差、预测几何误差分层、跨视图近似可见性可作固定诊断，但不能决定主评价人群。边界只能由 query GT 的预定有效邻域深度梯度规则生成，GT洞边缘与真实表面边缘分开说明；GT normal 则需避免无效深度插值制造虚假法向。选择某个方法最有利的边界阈值属于测试调参。

本轮若有随机训练，采用五个配对训练seed，相同初始化、batch序列、优化器、预算和 validation checkpoint规则。按 scan-macro validation AbsRel 选择最早最佳 checkpoint，并冻结强 point/fixed baseline；单视图模型也同预算选择。第六轮确定性方法没有训练seed，不能把其复制结果纳入本轮seed统计。

建议成功条件在正式协议中一次性冻结：full 对 validation选择的强点/fixed融合基线 AbsRel 相对降低至少2%、paired seed/place 95%差值CI上界<0、至少4/5训练seed改善；同时 full 对同容量 query-only 的误差也须有CI支持，才能说“多视图V带来收益”。RMSE不能出现预定实际容忍幅度以上的退步；幅度与是经验guardrail或统计非劣效应在测试前说清。full 对 trace 的独立归因区间作探索性结果，不由主胜利自动推断各向异性必要。

固定20,000次paired crossed seed/place bootstrap及随机数种子；所有方法共享抽样，place内保留全部相关帧。报告逐seed、逐scan、leave-one-scan和采集组敏感性；五个新scan仅支持有限总体范围。若收益仅来自 query-only 校正、只在GT跨视图可见子集成立或V路径没有贡献，应保留负结果，不回到同任务追KL或叠新模块。

## 封存与最低验收

- 开发集完成单位/相机闭环、GT输入隔离、全query评价群体、V梯度/值干预、所有对照学习与吞吐验收。
- 冻结数据分割、固定pair、源代码/协议/分析器、RGB与预测缓存、GT标签各自哈希；记录所有验证历史与最优checkpoint。
- 测试导出必须由匹配selection/exporter/protocol哈希的锁授权；执行前重放验证，核验完整5scan×96pair identities；测试不得静默重评覆盖。
- 独立用固定概率/深度预测子集复算误差，另对完整逐pair计数和seed/place CI做算术核验。范围明确为粗深度任务；不自动声称NVS、任意GS内部R/S或所有Transformer收益。
