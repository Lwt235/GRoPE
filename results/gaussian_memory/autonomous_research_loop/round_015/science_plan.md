# round_015：原生任务梯度形成公共场，是否改变参数 PE 的用途？

2026-11-13。**支持一次有限比较 `native_task_writer_pe1`；尚未实施，更未证明有效。** 唯一新条件是让共享 source-only writer 的物理几何和内容，经实际渲染与原生 VGGT 深度输出共同接受查询任务梯度。假设是任务所需的空间支撑能在成场时被学习，从而改变参数 PE 超过强中心的用途；竞争解释是只有内容／覆盖或共同适配变好、几何承载非物理捷径、强中心已经足够，或只记住训练片段。原先各具体家族的关闭决定保持，不把接收器解冻当新贡献。

**实际覆盖核对。** [prototype1](../../prototype1/report.md)／[joint_write1](../../joint_write1/report.md) 已有共享几何＋内容 writer，但分别用特征重构／独立粗深度头，有限载体及粗读收益不能迁移为原生 PE。[native_feature_field_pe1](../../native_feature_field_pe1/report.md) 的 Σ 确受完整特征渲染梯度、源重构改善，但属逐场景来源优化，后续查询梯度不回场，PE／Native 均未成立。[native_memory_context_pe1](../../native_memory_context_pe1/report.md) 已有原生任务训练的共享 writer，然而[实际 forward](../../../../models/gaussian_scene_context_writer.py)只改 f，使用固定渲染贡献矩阵；其关系 PE 未成立。[native_scene_exchange_pe1](../../native_scene_exchange_pe1/report.md) 只更新查询内隐藏副本，持久物理场不变，主 PE 与 Native 用途未立。**未见上述同一原生损失到共享物理成场的闭环已被覆盖。** [native_render_parameter_pe1](../../native_render_parameter_pe1/report.md) 等早已训练全部 frame/global/DPT；本轮保持这一条件，复用其渲染及 Q/K 接口作测量，不以另一注入层立项。

**唯一形成干预与必要对照。** 原 44 份、2048 槽／1024D 来源场为同一初始化；只增一个共享点式残差 MLP（两层 128 宽隐藏层、输出层零初始化），输入完整 f、第一来源坐标系中心／完整 logΣ／opacity。无新 attention、分组、head、门控或辅助损失。

| 共享 writer 训练范围 | 强中心 P | 同迹球形 S | 完整形状 G |
| --- | --- | --- | --- |
| C：学习 f、opacity；μ／Σ 固定 | C/P | C/S | C/G |
| J：同上，另学习 μ／Σ | J/P | J/S | J/G |

几何输出头六组同样存在，C 不启用；J 用有界残差：中心各轴不超过原 unit 的 0.25、log-scale 改变量不超过 0.7、旋转角不超过 π/6，保持正 scale／有效 rotation。opacity 用 logit 残差，内容用加法残差；有效槽／绑定身份、unit 不变，不增删基元。**source-only 指构建输入**：writer 不接 query RGB／相机／ID／标签，也无逐场景可训练参数；训练 query 标签仅经共享权重的任务梯度起作用。开发场只前馈构建、保存重载一次，全部八 query 与 full/missing 复用同一物理状态，不再运行 writer 或写回 query。学习改变几何不等于学得更准确的物理几何。

**同信息 PE 控制。** 沿用 [原渲染 Q/K 合同](../../../../docs/native_render_parameter_pe_execution_plan.md) 的实际 alpha 合成 1024D 内容、单次输入融合、52→64→1024 PE 和全部 24 个原生 global 块；不新增读出器。共同 46 维含中心多尺度编码、中心混合散布及原 ray 几何；最后六维 P=0、S=同迹球 logΣ、G=完整 logΣ，S 只改变 PE，实际场／渲染仍为完整形状。**所有组还把同一完整 logΣ 六维按实际贡献作未归一化渲染，接入既有 fusion 的普通输入**（首层仅扩六列，零初始化），并同样提供给 writer；中心散布不是 G 的专属信息。这样比较编码分配，不隐去 P 的普通形状信息。渲染／位置量由当前场计算，J 的 native 损失经渲染、普通内容与 PE 回传几何，不能沿用冻结缓存或 detach 这些路径。

各组同源输入、容量、初始化、监督和样本顺序；**训练后的场与接收器允许不同**。G/P 识别的是 PE 选择沿整条训练链的总效应，含其对成场的间接作用，不能声称固定同场的直接 PE 因果效应。形成×PE 交互检验几何可学习性是否改变这一总效应，不单独识别某个 Σ 参数、覆盖或唯一病因。

**关键学习与数据。** 沿原 32 训练／12 已用开发、来源 [0,3,6,9]、训练 query [2,5,8,11]、其他 query [1,4,7,10]、原 full/missing 与 valid/ROI 身份；原零覆盖病例均保留，即使新场改变支持。三配对种子 20261110/11/12，每组从官方权重重置，**18 fits×2000 更新、batch2，仅末步**。原 frame/global/DPT 全部学习，encoder／初始化来源依赖冻结；native LR=1e−6，writer／fusion／PE LR=1e−4，AdamW wd=.01、clip5，原完整有效图 raw 与去均值 log-depth MSE 等权损失。FP32 可训练参数／Adam与几何，native BF16；写入／保存内容的同一 FP16 cast 保留梯度。无来源重构目标、开发适配，不将 query 真值作为构建输入。旧 Native、32 地点 Local／Render 用同身份输出作净用途与成本参照；后两者不是新增 writer 的严格容量消融，形成与 PE 因果对照是本轮六组。

**主终点和取舍。** 仍为十二开发地点原四 query 的 missing-ROI 去均值 log-depth MSE，首要 ΔJ=E(J/P)−E(J/G)；另预定 I=ΔJ−[E(C/P)−E(C/G)]、J/G 对 C/G、G/S、S/P 及各组对 Native／Local／Render。沿原地点／query 等权、配对地点×种子区间；不能由两行显著性不同推断交互。实用开发筛沿用平均降误差 ≥1%、95% 下界 >0、三种子正；PE 与相对 Native／Local 的主量净用途须分别成立才保留候选。正 I 及 J/G 对 C/G 的支持才支持额外几何形成理由；若仅 C 或 S 在同一预定标准下有效而复杂方案无材料优势，保留较简者，**J/G 主假设仍照原终点报告**，不把简化候选冒称 J 或各向异性成功。筛过也只是开发候选，须再审全研究使用史并锁真正未用数据确认。

若只有共同 writer、raw／偏差或训练内收益，或 PE 仍无原生净用途，结束本原生任务成场条件下的自动 PE 变体，不补 seed／步数／数据或扫描残差界限、损失、层数。保留训练拟合／其他 query、开发其他 query、raw／bias／全图 AbsRel、原与新支持分层、逐地点／种子及最大反例；相对参照全图 AbsRel 增误差 ≥1% 且区间支持作为材料代价交回权衡，小而不确定的辅助负均值不硬否决，跨零不称非劣性。成本须分一次成场、逐 query 全原生链及复用摊销，不能由 G/P 小差掩盖对 Native／Render 的额外成本。

**有限投入与大纲。** 正式 18×2000，必要预检／恢复、完整评价与成本全部计入 **18 GPU worker 小时、80 GiB 新 DATA 峰值、1 GiB 项目新增**；无新数据／权重下载、无额外拟合或扫描。分析时只读快照 DATA 约 241 GiB／项目约 14.7 GiB 可用，并非启动许可：Sol 按实时 GPU/CUDA、CPU/RAM 和含恢复暂存的实际峰值准入，动态利用空卡，保留旧／他人成果和永久 final；不足则保存交回，不能删控制或暗扩预算。普通工程、计数及恢复按[现行协议](../protocol.md)具体化。

O2/O3 仍是可缓存公共粗 GS，O4/O5 是实际多视图渲染读取，O6 直接在原生 Transformer Q/K 比较同信息参数 PE；O1 的 patch／基元区分不变。实质改变 C1/C2 的共享物理成场与 C6 梯度终点；C3/C4 复用测量接口并补齐共同普通 Σ，C5 保持。sensor/joint-validity、给定相机、静态四源、2048／1024、深度／人工缺失及旧地点都是临时范围。分析实际 0 GPU／训练／新预测／新文献，整体目标未达、未暂停、无外部阻塞或已核验 fresh reserve。
