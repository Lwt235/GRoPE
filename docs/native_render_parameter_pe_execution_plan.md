# native_render_parameter_pe1：固定 Render 内容上的参数 Q/K PE

**最新完整结果（2026-11-13，round_002）：native_render_parameter_pe1固定实施完成，交回Astra取舍。** 保留原Render的P/S/G各三种子2000步、6336新预测与4224继承评分、独立复算、200完整暖查询及两图实际审阅均完成。开发原查询G/P主空间误差高2.935%（降误差95%区间[−10.158,+1.284]%），三个种子均未改善；S/P高3.837%，G/S低0.868%但区间跨零，预定主筛均不通过。G对原R高2.592%、对Local低6.394%，区间均跨零，净用途仍不确定；不把跨零称等效、非劣性或用途保持。

G/P原始ROI误差低9.409%[2.580,19.123]%，bias²同步下降；完整AbsRel低6.887%但区间跨零。训练片段其他查询G对原R低2.609%[0.812,4.902]%、对Local低43.460%[36.676,49.234]%，保留原Render的条件用途；G/P的0.769%均值增量仍不确定。主空间、原始/标定与片段内用途分别陈述，不改主终点或唯一归因。暖完整query R/P/S/G中位110.30/140.66/139.44/139.50ms，G/P成本条件通过，但相对原R公共PE开销约26.47%。

实际保留18000/尝试18009/旧丢弃9更新，pilot6另列；预检实际17/计费18（训练pilot6、R回放4、零PE4、恢复3/4）清楚分项。GPU总5.202248 worker小时，本轮DATA22.522GiB、项目约0.053GiB，均在原8h/26/2GiB内；资源余量按独立附录运行，完整状态安全让出、容量恢复后唯一替代队列续接，全部GPU/守护正常退出，RAM冗余退休，未删除旧成果。实现/恢复成立、用途与O6增量、独立确认分开；O6本接法主筛未达，raw及训练正证据保留。全部旧数据、sensor/joint-validity/给定相机/静态来源范围继续，总体目标未达；方法决定与下一方案交回同一Astra，实现阶段不自行续训。[完整报告](../results/gaussian_memory/native_render_parameter_pe1/report.md) · [全数表](../results/gaussian_memory/native_render_parameter_pe1/report_tables.md) · [实测规则标记](../results/gaussian_memory/native_render_parameter_pe1/method_decision.json)。

<details>
<summary>形成时计划状态与实施资源历史</summary>

**资源重新准入等待（2026-11-13）：** round_002首次附录yield后P/S第三种子已完整保存1975/1993步，G未启动；保留15968/实际尝试15977/旧丢弃9/新增丢弃0更新，累计4.533755 GPU worker小时。六份完整拟合及4224新预测保留，原队列34546保持T，无活动GPUworker；相隔41.4秒快照H=6.279407GiB，重新准入缺口0.220593GiB。未降低阈值、重启或改变科学锁，主任务协调稳定空间。CPU监视退出竞态修复已留证据、尚未启动；未从不齐结果作方法取舍。[恢复交接](../results/gaussian_memory/native_render_parameter_pe1/resource_yield_v1.json) · [精确恢复说明](../results/gaussian_memory/native_render_parameter_pe1/execution_resume.md)。以下进度按形成时保留。


**资源安全等待（2026-11-13，round_002）：** `native_render_parameter_pe1`已完成前两种子P/S/G共6/9拟合及4224新预测；第三种子P19/S22步完整状态已保存，G尚未启动。数据盘实时余量14.976GiB、固定剩余产物约7.481GiB，预计完成后7.495GiB低于原10GiB工程余量；仅本人worker已退出，原队列PID34546明确SIGSTOP，禁止重复队列或未经新交接恢复。实际保留12041/尝试12050/旧丢弃9更新，累计3.489576 GPU worker小时；科学公式、样本、精度及硬预算均未改变，也未以不齐结果作方法取舍。交回原Astra仅审查工程资源余量；当前不是方法阴性、完整结项或总体研究阻塞。[精确停态](../results/gaussian_memory/native_render_parameter_pe1/disk_resource_pause.json) · [恢复说明](../results/gaussian_memory/native_render_parameter_pe1/execution_resume.md) · [资源交接](../results/gaussian_memory/native_render_parameter_pe1/resource_pause_handoff.json)。下方旧进度按形成时保留。


2026-11-13，round_002有限开发计划，方案形成时状态：分析完成、待主任务顺序交接Sol；现已顺序交接并开始实施，见下方进度。依据[结果分析](../results/gaussian_memory/autonomous_research_loop/round_002/analysis.md)与[机器计划](../results/gaussian_memory/autonomous_research_loop/round_002/plan.json)。不修改native_input_pe1、native_render_factorization1既有完成报告/身份。

**实施进展（2026-11-13，round_002）：** 352份位置字段已完成并锁定，原r/patch/scene只读复用；两训练输入独立NumPy抽样字段差0、零PE原R一致、六个丢弃更新和完整状态恢复预检通过。P/S首种子20261110在空闲GPU0/2正式训练，固定九拟合各2000步；尚无完整效果或方法采用。预检Adam恢复内存重叠及跳过阶段清理两项普通修复保留，无额外pilot更新或科学方案变更。[执行状态](../results/gaussian_memory/native_render_parameter_pe1/queue_status.json) · [预检](../results/gaussian_memory/native_render_parameter_pe1/preflight.json)。

</details>

## 1. 唯一问题与比较

假设：在保留已有视图对齐 Render 内容和覆盖的条件下，将内在基元参数直接用于已有 Transformer Q/K，能在强中心编码之外改善原生深度。竞争解释为中心集合几何已足够、内在参数不可靠/冗余、PE只适合已训练片段、共同适配或幅度变化产生表面收益，以及额外编码扰乱已有用途。

新拟合只有P/S/G各三种子（20261110/11/12）。三组同官方初始化、原Render融合初值、场景/内容/采样、优化器、完整原生接收训练和2000步，不从已训练Render继续加步数。继承同32地点同预算的R和Local最终预测作为用途参考，不重训它们。

| 组 | 共同46维中心字段 | 六维新增字段 | 原Render内容/权重 |
| --- | --- | --- | --- |
| P | 全部 | 全零 | 完全保留 |
| S | 全部 | 每基元同迹球形 intrinsic log-cov 的加权均值 | 完全保留，包括完整形状原权重 |
| G | 全部 | 每基元完整 intrinsic log-cov 的加权均值 | 完全保留 |

强中心共享由中心产生的多尺度信息及中心间离散二阶矩；不将这部分当高斯协方差独有收益。P仍经高斯来源绑定/渲染、有效覆盖访问内容，不是从整个系统删除shape/opacity。三组形式参数相同，P六个零输入导致384个第一层权重不活跃；不以伪特征填槽来掩盖。此比较不主张同信息下某种新理论编码最优。

## 2. 固定位置字段

沿用 `experiments/native_coverage_data.py` 的44场景和查询定位函数；原32×32缓存 `render` 不重写、不重新归一化。将原渲染实际前向贡献记为 `w_ij = renderer_weight_ij * has_content_j`，只用有效内容槽；`a_i=Σ_j w_ij`。渲染宽高32、K/14、近远面、eps2d及WXYZ四元数约定沿用 `models/gaussian_scene_memory.py`。没有新visibility阈值、top-k或温度。

每查询由给定 `c2w=(R_q,t_q)` 与保存的正 `field['unit']=u` 计算：

`p_j = R_q^T (μ_j-t_q)/u`，`B_j=R_q^T R_j`，

`L_j = B_j diag(2 log(s_j/u)) B_j^T`。

这里 `R_j,s_j` 是保存3DGS的旋转和正scale；L是相机坐标中的内在协方差矩阵对数。直接用旋转与log-scale构造，无须逆协方差、数值积分或把真实支撑当位置误差。S用 `L_j^S = log((s_jx²+s_jy²+s_jz²)/(3u²)) I`。不把Σ跨基元混合成一个物理高斯，也不把中心间离散项并入独占shape字段。

令有效权重平均 `<x>_i=Σ_jw_ij x_j/a_i`（a>0）。共同46维中心字段依序为：

1. `m_i=<p>_i`，3维。
2. `<sin(π2^k p)>_i`和`<cos(π2^k p)>_i`，k=0…4、每轴，30维；固定顺序k、sin/cos、xyz。
3. `asinh(svec(<(p-m_i)(p-m_i)^T>_i))`，6维。这是**中心间**离散矩，仅用中心和共同w计算，P/S/G一致。
4. `center_ray_descriptor(...,unit=u)`的7维原量按同w平均：五个尺度(.02,.05,.1,.2,.5)的径向log1p、asinh沿射线距离、log1p中心距离。射线像素中心和K/14完全沿用旧实现。

`svec(A)=(Axx,Ayy,Azz,√2 Axy,√2 Axz,√2 Ayz)`。末六槽为P零、S `<svec(L^S)>`、G `<svec(L)>`。全部加权累加/除法及中心离散矩准备在FP64完成后存FP32；w来自原FP32渲染。仅a==0时把所有字段和位置增量置零，不用epsilon/覆盖阈值。所有零覆盖/小覆盖查询仍进入评价。单位/scale非正或非有限是输入/实现错误，版本留档修复或返回，不能静默更换几何算子。

缓存只保存 `a, center46, sphere6, full6` 及身份，不持久化全1024×2048权重或复制1024维内容。共352查询，full/missing共用源/相机位置字段。查询RGB/深度、目标valid/ROI、训练标签不参与构造；监督仍只用于既有训练/评分。字段规范化不改变原r，最终还乘原a，没有将本轮变成新覆盖或归一化实验。

## 3. 原生接法和训练

唯一新增模块 `PE(z)=Linear(52,64,bias=True) → GELU → Linear(64,1024,bias=False)`，共68,928参数。末层权重全零；同seed三组所有初始参数相同。原Render fusion初值必须匹配既有同seed初值；新模块初始化用独立局部RNG上下文，不能改变共同训练采样或已有融合初始化。

每查询只计算一次 `e_i=a_i*PE(z_i)`。对全部24个既有global块：原qkv投影之后，将e按16heads×64dim reshape后分别加到patch Q和K，随后原q_norm/k_norm、原2D RoPE、相同全零native mask、SDPA、output照旧。五个special token的e为零，V保持原投影；frame块保持原结构。PE只在已有Q/K中使用，未加新reader/attention或输出头。保留先前 `patches + RenderedFeatureFusion(patches,r)`，保留所有DPT取层、depth输出和camera/register tokens。零PE应回到原Render路径；禁止额外learnable gate、layer embedding、3D rotary或独立深度残差。

训练沿用native_training_coverage1：冻结图像encoder、来源scene/features与first-view特殊token，训练全部frame/global及原DPT（637,383,906参数）、原fusion（787,712）和新增PE（68,928）；每组总638,240,546可训练参数。每fit从共同官方预训练重置，不从旧Render权重继续。batch2，2000更新/4000样本呈现；实际采样顺序逐seed与原覆盖计划一致。loss为原full-valid raw/centered log-MSE等权和，native LR1e−6，fusion/PE LR1e−4，AdamW weight decay0.01，gradient clip5；FP32参数/Adam，原BF16算术和checkpoint规则，eval模式但指定参数有梯度。中间开发结果只描述，不能选模型/改终点；一律step2000。

最新完整结果只继承对照的匹配用途，不假称R/Local在本轮新拟合。若新wrapper改变零PE原生算术路径，先做有限重放并说明差异；不能悄悄把不同内核效应算PE收益。常规精度/索引/序列化错误按版本修复，不另开科学轮；若必须更换字段、插入位置、训练范围/损失/主数据或预算，先经主任务返回Astra。

## 4. 数据、指标和计数

原coverage计划固定32训练/12开发的地点、所有raw/source/cache SHA、官方预训练SHA及三seed schedule为权威身份，锁定于实施配置。4source、所有旧scene与encoder/r缓存只读复用。训练仅四查询[2,5,8,11]及full/missing；所有44场景八查询[2,5,8,11,1,4,7,10]、两条件全量评估。

- 新正式预测 `3组×3seed×44场景×8查询×2条件=6336`。
- 继承R/Local两组同44×8×2×3，合计4224评分；不重导全量对照。
- 原生R/Local有限接入回放最多24前向；零PE/必要预检最多12前向（不重复已有完整审计）。
- 成本固定seed20261110、fit_00/q1 missing与fidelity_00/q2 full、P/S/G/R每项5 warm+20 measured，共200前向。
- 非训练原生前向计划最多6572，上限6600；训练18,000更新、36,000样本呈现单列。必要预检优化更新≤6，权重丢弃，不计为正式fit。没有来源重建、额外fit或新下载。

主终点是旧12开发地点原四查询missing ROI centered log-MSE；首要G−P，完整形状G−S及简化S−P预先指定。报告绝对误差差与相对降误差，seed/地点/查询配对、地点与查询等权；5000次地点与配对seed重采样，固定seed20261113，95%区间。种子数仅3、地点仅12，区间条件于当前有限协议，不是一般性证明；不将辅助选择当新的验证。所有主/辅助比较及种子/地点差公开。

辅助：32训练片段其他查询[1,4,7,10]（与拟合查询分开，原12/新增20亦分开）；开发其他四/八合并；原始ROI log-MSE及其bias²分解；full整图AbsRel及完整空间误差；继承R/Local净用途；与已有支持分组的描述（不以分组替换主终点）。沿用原评分valid/ROI，零覆盖和反例全部保留。

两图：完整终点/配对增量与成本图；固定fit_00/q1、confirm_12/q1、fidelity_00/q2、fidelity_02/q11的Local/R/P/S/G深度例和事前规则选择的最大开发G−P反例（所有查询均已评分）。原metric depth一致尺度显示，不平滑、不只给最佳例；使用既有Python绘图链并留下实际打开审阅记录。

## 5. 决定规则与停止

有限开发筛选 `screen(X,P)`：主相对降误差≥1%，95%区间下界>0，三个配对seed均改善。此实际幅度线在运行前固定，不是等效/无效定理。比较全部公开；它不是经多重检验校正的确认性结论。

若screen(G,P)成立且G−S有支持（降误差区间下界>0），完整形状才有继续理由；否则若screen(S,P)成立，优先保留球形候选，完整形状增量仍未立。进入下一独立确认候选时，R/Local净用途与完整输出仍要审阅，但均值差仅作软排序与权衡描述，不设“必须不更差”的零容差门槛。开发原四查询中，主误差对R/Local、原始ROI及full整图AbsRel对P/R/Local，只有相对误差增加≥1%且配对95%区间支持增加，才作为有支持的材料性损伤，阻止按净用途候选晋升。1%沿用同一事前实际幅度线，避免把微小但有统计支持的变化夸大；不是正式非劣性证明。训练片段其他查询对R的变化只界定适用范围与训练内/跨地点权衡，不自动否决跨地点改进。所有负均值及区间仍报告；跨零且幅度较大的退步属于未决，不能称用途保留，后续确认必须同时预定PE增量、R/Local净用途和完整输出质量。暖完整查询中位时延比P增加≤10%的成本条件保持。

此为实施前设计评议v1对最初严格均值规则的细化：不新增组、试验、预算或选模机会，无新结果参与修改；[记录](../results/gaussian_memory/autonomous_research_loop/round_002/design_review_v1.json)。

若只P改善R，保留中心用途，关闭本接口的形状新增价值；若只训练片段其他查询改善，记录条件范围，结束跨地点主张；若形状增量有支持但原用途/全图损失明显，保留PE信号，候选不晋升。所有主方向不成立则结束此固定参数Q/K接法。任何结果都不触发本计划内的温度、幅度、宽度、频率、层数、LR、数据、种子或步数扫描，也不追加机制轮次。

全部9fit与固定评价完成后返回同一Astra作取舍。达到资源上限、无法保留严格匹配身份、不可恢复非有限训练或实质方案变更时保存状态，经主任务交回；不绕过匹配条件，不以跑完部分组宣称成功。已有完成标记/模型/导出先核验复用，不能重复拟合。

## 6. 资源与交付

总GPU hard cap8 worker小时，含字段准备、预检、9fit、评价/成本和普通恢复；预计约5–6.5小时，仅作估算，不是运行承诺。最多2并发worker，最多1张有外部任务的卡；优先空闲GPU，启动/恢复前核验占用和CUDA。初始进程cap12GiB、可用显存至少18GiB并保留6GiB外部余量；检查后可在空闲卡把cap调至14GiB、准入20GiB，算法/精度不得改变。保护其他用户，必要时完整状态让出并恢复，不终止外部进程。

CPU工作集目标≤32GiB；专用tmpfs恢复状态（含模型/optimizer/step/配置/schedule及RNG）最多2份运行中任务，预留≥20GiB可用并记录实际峰值。checkpoint每100步及让出时保存；最终权重永久保存并核SHA后才退役本轮冗余。不得删除/压缩旧权重或旧完成预测腾空间。

使用既有授权数据根下独立 `native_render_parameter_pe_v1`，9最终权重约21.4GiB，6336 FP32 metric-depth预测约1.04GiB，位置字段<0.2GiB，新数据盘hard cap26GiB；项目报告/图/小元数据hard cap2GiB。预检观测项目盘约17GiB、数据盘52GiB可用；Sol实时重查，要求计划占用后项目剩≥8GiB、数据剩≥10GiB。不复制旧完整模型/内容/图像缓存，不持久化全权重矩阵。权限或实际容量不满足时先保存现场回报，不静默删除旧成果或变更权重精度。

成本需分别记录字段构建/缓存bytes、训练worker wall及checkpoint I/O、原生接收查询、含冻结图像encoder+原内容渲染+PE字段构建的暖完整查询、来源构建的继承成本。200个成本前向以完整query执行，可同时分段计时；不重复第二套成本实验。加载/等待分开列明，不把缓存查询等同端到端，不用继承source成本冒充新测。

必要核验限定为：零PE回到原R、三组共同输入/初始化/采样、独立NumPy抽样重算中心离散与log-cov字段、Q/K插入及V不直接改动、PE/native有效梯度与可恢复状态、主表独立复算、完整seed/地点反例和两图实际审阅。无需重做旧完整报告的数值审核。所有普通预检修复留version记录；报告、固定数表、逐query评分、成本、method_decision、manifest、implementation_handoff及living docs同步后交回。

## 7. 大纲与确认范围

核心O2/O3公共粗3DGS、O4/O5每视图相关读取、O6参数PE增量均保持；此次直接测O6并评价O4/O5，复用O2/O3。C1/C2/C5/C6保持，C3采用既有原Render，新增变更C4。2048基元、52输入/64隐藏、全24global、32训练、三seed等是当前可推翻的实现选择，不是大纲要求。sensor/joint-validity、给定相机/尺度、静态四source、冻结来源与encoder、depth单任务是未解除的临时边界，不能说已实现纯RGB、未知相机、在线融合或所有VGGT下游。

本轮全部已用数据，没有独立确认。只有发展出明确候选后，后继计划才先查全研究使用历史并锁定真正未开发地点/序列、样本SHA、训练/选择/评估角色及控制；无明确未用身份不得运行或声称确认。旧目录名、额外视角、coverage未选地点不满足fresh判定。总体目标尚未达成，完成本有限计划不自动代表方法成功。
