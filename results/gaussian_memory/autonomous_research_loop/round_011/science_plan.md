# round_011：来源特征与形状共同形成，能否改变原生 PE 增量？

2026-11-13。**支持一次有限状态形成比较 native_feature_field_pe1，尚无有效方法结论。** 唯一问题是：逐场景用实际保存的完整 VGGT 特征训练 Gaussian 渲染支撑，是否改变参数 PE 相对同信息强中心的原生用途？固定已有读出网络／PE 作为测量接口；新增的是可被对照推翻的 C1/C2 形成条件。原相机细化、直接算子扫描与固定场 reader 自动变体仍关闭。

**依据及反证。** [gaussian_scene_memory1](../../gaussian_scene_memory1/report.md)／[source_geometry_control1](../../source_geometry_control1/report.md) 先拟合 RGB／深度／覆盖，再绑定 1024D 特征；改善来源几何曾帮助中心读取，却未解锁 PE。[prototype1](../../prototype1/report.md) 已让共享 writer 的形状接受 PCA32 特征重构梯度，六旧场景特征 MSE 改善 5.69%；其后 [attention1](../../attention1/report.md) 在另十二个已用场景仍未稳定超过中心。因此“特征监督形状”不是新发现或充分条件。[surface1](../../surface1/report.md)／[joint_write1](../../joint_write1/report.md) 是查询粗深度监督的共享 32D writer，无逐场景优化；[geometry_transport1](../../geometry_transport1/report.md) 冻结整个状态；[native_memory_context_pe1](../../native_memory_context_pe1/report.md) 只改内容。最近 [camera_only_feature_pe1](../../camera_only_feature_pe1/report.md) 的完整场固定，PE 与 Native 净用途均未通过。**新可辨识条件是 source-only、逐场景、实际完整特征渲染目标下，“仅值拟合”与“值＋协方差拟合”的配对形成，再测原生 PE 及两因素交互。** 这补上 [round008](../round_008/science_plan.md) 缺少的形成机制，不把 Σ 改称定位不确定性。

**唯一形成干预。** 使用同一实际 gsplat 有效内容贡献 \(w_{vij}\)，沿用读出端的归一化特征场
\[
b_{vi}=\frac{\sum_j w_{vij}f_j}{\max(\sum_jw_{vij},10^{-8})},\qquad
L_{\rm src}=\operatorname{mean}_{v,i,c}(b_{vic}-F_{vic})^2 .
\]
覆盖不超过 \(10^{-8}\) 时 b=0；全 patch／通道计损失，零覆盖不筛除。\(F_v\) 是四来源冻结图像编码器的实际 32×32×1024 输出，不降维。旧后绑定没有这条损失到 Σ 的梯度，J 有而 C 没有：

| 来源形成 | 强中心 P | 同迹球形 S | 完整形状 G |
| --- | --- | --- | --- |
| C：仅 f 拟合，Σ 固定 | C/P | C/S | C/G |
| J：f 与 Σ 共同拟合 | J/P | J/S | J/G |

两行从相同旧场开始，**μ、opacity、DC、槽集合／有效性、相机和单位固定**。只优化原有内容槽的 f；J 再优化其 rotation／log-scale。同来源循环顺序、各 300 次 Adam 更新；f LR=0.01，J rotation／log-scale 沿原构建器 0.001／0.02，scale 沿用 \(u[10^{-5},2]\) 范围；末步保存，不选模、不加损失／增删槽。优化 FP32，保存几何 FP32／f FP16。每行的一份场由全部 PE 组及种子共享，形成后冻结；不是各 PE 组分别学习形状。

**匹配原生比较。** 严格复用 [camera-only 合同 §§2–5](../../../../docs/camera_only_feature_pe_execution_plan.md) 的 128 宽四头单层 reader、实际渲染 anchor、29 维位置 MLP、初始化／学习条件及原生接口，不改变公式或接收器。P/S/G 普通输入均含完整 f／Σ／opacity、中心邻域 B／多尺度密度；共同中心位置基底外四维分别为 density／球形 footprint／完整 footprint。这是同信息、同容量的编码分配，非 nested 加维或全系统去形状。固定中心与有效槽也使 C/J 的纯中心 B／density 一致。形成导致内容和覆盖改变属处理的一部分，交互不单独识别“纯 Σ 信息”的因果效应。

原 44 场、32 训练／12 开发、四来源 [0,3,6,9] 及全部数据身份不变。**形成器不接 query RGB／教师／标签／评价真值**；每个开发场也只以自己的来源构建，不用开发 query 选择步数。原 sensor／joint-validity 初始化仍是特权条件。保存后按原三个种子，各组做 2000 次 batch2 reader 特征学习，共 **18 fits**；query 梯度不回传到场。完整 VGGT／DPT 冻结，reader 只接场与相机，预测特征填入原 missing ROI，再运行全部原生层。保留不填补的 Native、每行不学习 reader 的 RasterFill；旧未形成场只作历史参照。原四拟合 query、全部八 query 评价／原 valid 与地点权重不变，原零覆盖与新零覆盖全部保留；保存的每模型 K/V 可重载复用于八 query。

**主判据。** 仍用十二开发地点原四 query 的 missing-ROI 去均值 log-depth MSE。首要 \( \Delta_J=E_{J,P}-E_{J,G}\)，另预定交互
\(I=(E_{J,P}-E_{J,G})-(E_{C,P}-E_{C,G})\)；报告绝对／相对效应与原配对地点／种子区间，不能从“两行显著性不同”推断交互。P/S/G 全比较及各组对 Native／同场 RasterFill 全报。PE 和 Native 净用途沿用实用开发筛：平均降误差 ≥1%、95% 下界 >0、三个种子同向；G/S 决定非球形额外用途。来源／查询特征 raw、spatial、bias、覆盖／形状变化仅作解释证据；保留原生 raw／bias／全图 AbsRel、训练拟合与其他 query、开发其他 query、逐地点及最大反例。竞争解释是视角依赖、非唯一分解／覆盖与幅度变化、仅共同内容变好或强中心足够；源重构下降不排除它们，也不替代 O6。

**结果决定。** PE 与 Native 净用途同时有实用支持，才保留至多一个开发候选；J 优势伴随正交互的支持，才支持“联合形成改变 PE 作用”。若仅 C 已有效或两行相近，优先简化为仅内容拟合；交互不确定时，有主证据的候选仍可保留，但该机制主张未立。G 无明确额外用途而 S 有效时优先球形。若仅共同场／feature／训练内改善，或 PE 无 Native 净用途，结束本形成条件下的自动 PE 变体，不补 seed／步数／数据／损失／层数。相对同场 RasterFill 的主量或 missing 全图 AbsRel 增误差 ≥1% 且区间支持时，保留材料代价供 Astra 权衡；小而不确定的辅助负均值不硬否决，跨零也非非劣性。所有数据均已开发；候选后另审全研究使用史并锁真正未用数据，当前无 verified fresh reserve。

**投入与范围。** 88 份场×300 来源更新、18×2000 reader 更新；含必要预检／恢复／全评价和成本，**GPU ≤8 worker 小时，新增 DATA 峰值 ≤36 GiB、项目 ≤1 GiB**。不追加训练或机制扫描。成本分列旧场形成、此次逐场景拟合、K/V 构建、每 query 读出、完整 native 与摊销；不能称前馈构建。Sol 按实测 GPU/CUDA／剩余产物峰值准入并保护旧成果和他人任务，普通计数／恢复按[现行协议](../protocol.md)具体化，硬预算不足保存交回。O2/O3 公共粗 GS、O4/O5 实际多视图读取、O6 同信息 PE 与原生用途均保持；O6 位于服务 VGGT 的 memory-reader Transformer，非原生 24 层新增 PE。实质改变 C1/C2；C3/C4/C5 固定为测量选择，C6 加来源拟合而沿用 reader 学习。给定相机、sensor、静态四源、冻结原生网络、深度／人工缺失与旧地点均是临时范围。分析阶段 0 GPU／训练／新预测；无外部阻塞，总体目标未达，历史阴性和关闭决定不改。

