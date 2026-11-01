# GRoPE：高斯几何与多视角 Attention 的初步验证

**最新：第三十八轮单高斯 3D anchor 方案未通过旧 TRAIN 筛查，八项必要条件 0/8。** 全部 2,304 查询完成并通过输入与独立算术审核；经根验收后，两条指标路径一致确认：区域 KL=0.366603469，较强 quad 的 0.308352829 差 18.8909%；中心 PCK@8=86.336733%，各定位保护均失败。生产阶段 2,406.700 秒。停止这一固定实现，不继续其成本及全概率参考投入；完整概率精度、一般效果和总体研究目标仍未通过。[报告](docs/round38_rms_anchor_triage_report.md) · [根验收](results/round38_rms_anchor_root_acceptance_20261101.json) · [后续决定](results/round38_rms_anchor_post_result_decision_20261101.json)。

报告整理出现一项控制器日志清单滞后问题，以及根代理未在捕获检查失败后停止报告生成的错误；原记录未改，已外置对账。报告没有成功的独立事前提取冻结，不能倒填。[完整说明](docs/round38_delivery_record_note.md)。下一步先以已有封存输出诊断位置质量与内容偏置的影响，不据此选参、剔除样本或声称分布族不可能。

**最新：第三十七轮真实输入独立参考预检失败，科学目标仍未完成。** 原72项输入全部通过身份与字节审计；144个预定调用位置中56次尝试、55次完成，1次触及250M单次节点预算，后88次按冻结规则未尝试。296项比较中162项有完整数值并通过，134项缺少完整参考；没有把成功前缀当作整轮通过。阶段1080.17秒、4,931,562,096节点，完整失败交付及独立复算已验收。[报告](docs/round37_component_round_real_reference_report.md) · [验收](results/round37_component_round_real_root_acceptance_20261101.json)。

失败发生于第二个场景的g8/query184/48阶：已收费249,980,432节点，其中窄Y分支占87.11%；仅完成56/133个初始外层区间，尚未自适应细分。旧R33预算失败位置和R35舍入失败位置在本轮均已完成，但新的失败仍阻止晋升。该位置的完整初始几何计数已确认：原48阶参考至少需要347,526,680节点，有局部解析误差界的保守变阶方案仍需294,537,897，均超过250M单次预算，因此不实现这项变阶优化。诊断没有计算新概率或改变R37失败。[诊断](docs/round37_reference_budget_diagnosis.md) · [局部解析界](docs/round37_narrow_order_bound_analysis.md)。

下一轮已选择结构精简的单高斯3D anchor构造：以预测源中心作均值，将相对该中心的二阶矩送入原RMS投影法则，保持旧内容偏置。R38先做固定旧TRAIN的失败筛查：全部2,304查询通过输入审计、前向封存和独立算术审计后，才读取标签检查原八项必要条件；阳性结果只允许保留该固定实现继续验证，不代表晋升或通过科学门槛。截至本次R37交付整理，R38尚无真实实验结果，完整独立概率参考、效果与成本、扰动、实际V、GS和新场景要求仍保留。[设计](docs/round38_rms_anchor_triage_design.md) · [协议](ROUND38_RMS_ANCHOR_TRIAGE_PROTOCOL.md) · [后续决定](results/round37_component_round_real_post_result_decision_20261101.json)。

**第三十六轮固定合成参考预检通过，科学目标仍未完成。** 23个输入的46次参考全部完成，339项新旧比较通过；保留标量截断误差估计，历史舍入项改为逐分量累计后取最大值，精度、资源预算和概率模型保留。阶段约103.46秒，374,023,888节点；这些结果不能替代真实输入或效果验证。[报告](docs/round36_component_round_reference_report.md) · [验收](results/round36_component_round_root_acceptance_20261101.json)。

随后完成的纯几何成本诊断覆盖全部72个真实输入；48阶初始节点下界最大154,790,736，未超过原250M单次预算。结果仍为未知：没有计算概率，也未证明实际全部节点、收敛或精度可过。据此冻结的第三十七轮复用原输入和生产结果，验证144个参考位置及296项比较；执行后的失败见上方，原门槛和失败后停止规则均保留。[成本诊断](docs/round36_real_initial_cost_lowerbound_report.md) · [R37设计](docs/round37_component_round_real_reference_design.md) · [选型记录](results/round36_component_round_post_result_decision_20261101.json)。

**第三十五轮真实输入独立参考预检仍为失败。** 72项输入全部通过身份与字节审计；144个预定调用位置中22次尝试、21次完成，1次因外层求积的舍入误差报告未收敛，后122次按预设规则跳过。254项比较完整保留，69项有完整结果并通过，185项缺少完整参考。原R33超预算位置已完成，但不能用成功前缀替代整轮通过。[报告](docs/round35_log_real_reference_report.md) · [验收](results/round35_log_real_root_acceptance_20261101.json)。

单次原样复现及独立审查确认，R35失败分区的跨分量舍入下限比收敛目标高8.43%；随后实施的R36修订与完整合成结果见上方。R37中原失败位置已经完成，但全真实人口仍因新的预算失败而未通过。[诊断](docs/round35_reference_roundoff_diagnosis.md) · [R36设计](docs/round36_component_round_reference_design.md) · [选型记录](results/round35_log_real_post_result_decision_20261101.json)。

**第三十四轮固定合成数值预检通过。** 23个输入的46次参考全部完成，247项比较通过；窄正态区间改为8节点，其余模型、阈值与预算保留。节点数从412,301,376降至379,773,536；局部数学界不代表一般浮点准确性或真实效果。[报告](docs/round34_log_narrow_reference_report.md) · [验收](results/round34_log_narrow_root_acceptance_20261101.json)。

**最新：第三十三轮真实输入独立参考预检失败，科学目标仍未完成。** 72项输入全部通过字节核验；144个调用位置中20次尝试、19次完成，1次耗尽单次节点预算，后124次按预设规则未尝试。216项比较保留完整人口：28项有完整结果并通过，188项缺少结果；不能由成功前缀推断其余项。失败交付和独立复算已验收。[报告](docs/round33_log_real_reference_report.md) · [验收](results/round33_log_real_root_acceptance_20261101.json)。

[几何计数诊断](docs/round33_reference_budget_diagnosis.md)确认失败发生在初始外层求积，尚未进入自适应细分。下一轮只将参考的窄正态区间求积由16节点改为8节点，保留分支范围、全部预算和精度门槛；局部数学误差界不代表全流程精度。先完成23个合成输入的46次参考、247项新旧比较，再考虑全部72项真实输入。模型、anchor/content和全部科学验证要求保留。[R34设计](docs/round34_log_narrow_reference_design.md) · [后续决定](results/round33_log_real_post_result_decision_20261101.json)。

**最新：第三十二轮固定合成独立参考数值预检通过，科学目标仍未完成。** 23个固定输入的46次参考全部完成，155项比较通过；只合并无法表示严格中点的分割边界，完整积分区间和原误差门槛不变。原第六阶段审计因NumPy布尔值序列化失败，已单独冻结输出修复并重新审计封存结果，未重积分；原失败及提前根验收错误均保留。[报告](docs/round32_log_conditional_reference_report.md) · [验收](results/round32_log_conditional_root_acceptance_20261031.json)。

下一轮按原顺序验证全部72个R30真实输入的双阶独立参考及216项比较，保持每次调用和精度门槛。总预算按144次调用预先扩大到360亿节点/3小时；首个必要条件失败后停止实际调用，全部计划和未尝试结果仍留档。此阶段不读标签、不选择anchor或内容温度，也不宣称效果通过。[R33设计](docs/round33_log_real_reference_design.md) · [后续决定](results/round32_log_conditional_post_result_decision_20261101.json)。

**最新：第三十一轮独立对数参考预检失败，科学目标仍未完成。** 46次调用完成44次，155项比较通过148项；失败来自同一样例两个阶数的初始深度分区无法表示严格中点。已完成子集的误差满足原门槛，但不豁免缺失结果。[报告](docs/round31_log_conditional_reference_report.md) · [验收](results/round31_log_conditional_root_acceptance_20261031.json)。

下一轮仅合并无法取严格中点的相邻分割边界，保留完整积分区间与全部原门槛，重新冻结并验证全部23个合成样例的46次参考及155项比较。旧失败、生产模型和科学验证要求均保留。[R32设计](docs/round32_log_conditional_reference_design.md) · [后续决定](results/round31_log_conditional_post_result_decision_20261031.json)。

**最新：第三十轮固定真实输入适用性检查通过，尚无方案通过全部科学验证。** 完全复用R28的72个输入，R29对数实现72/72调用完成、输入身份及输出算术全部通过；前向阶段56.967秒。3,227个正质量格不能普通float64物化但保留有限log；一个查询的raw FOV约2.59e−28，低于总floor，因此不能用P归一化替代raw精度。此轮没有独立真实概率参考或效果标签。[报告](docs/round30_log_rms_applicability_report.md) · [验收](results/round30_log_rms_root_acceptance_20261031.json) · [后续决定](results/round30_log_rms_post_result_decision_20261031.json)。

下一项先验证新的独立对数条件积分参考：完整X区间、固定pilot缩放及原深度尾界，保持原双阶/raw/P门槛和预算。只复用R29的23个合成基础输入、已封存生产和参考输出，做46次新参考；通过后另行冻结全部72真实结果的独立概率核验。模型、anchor/content和科学门槛不变。[设计](docs/round31_log_conditional_reference_design.md)。


**第二十九轮对数质量实现通过固定合成数值预检，尚无方案通过全部科学验证。** 保留同一RMS投影高斯模型，29次生产、40次旧数值参考及3次新解析参考全部通过；原26个输入描述和字节不变。生产对参考最大P误差2.22e−16、原始质量相对误差2.27e−13，整个FOV低于普通浮点表示范围的例子也通过原raw门槛。生产/参考阶段2.818s/50.790s；不据此宣称真实适用性或效果。[报告](docs/round29_log_rms_preflight_report.md) · [验收](results/round29_log_rms_root_acceptance_20261031.json) · [后续决定](results/round29_log_rms_post_result_decision_20261031.json)。

随后第三十轮重新冻结并复用R28已审计的72个输入，四类模型输入逐字节复制，先做独立身份核验，再检验新实现的全部调用、成本和输出算术。不读取标签，不重做几何映射或选择anchor/content，也不将这项检查当作一般真实概率精度验证。[设计](docs/round30_log_rms_applicability_design.md)。

**第二十八轮真实输入适用性检查失败。** 72项输入全部通过独立几何核验，72次实际调用中40项完成、32项在正密度下溢/溢出合并保护处失败。前向阶段23.469秒，40项输出算术通过；没有真实概率独立积分，合并错误串也未区分具体下溢/溢出条件。失败及部分工作完整保留。[报告](docs/round28_rms_applicability_report.md) · [验收](results/round28_rms_root_acceptance_20261031.json)。

**第二十七轮通过固定合成数值预检。** 保留R26生产模型和原24个输入，仅修正独立参考的标准X分段，再加入两个宽视野例；26次生产、40次参考、全部raw/P/FOV和恒等性检查通过。最大参考FOV可加性差3.33e−16，生产对参考最大P差1.89e−15；生产/参考阶段1.511s/50.886s。该结论不覆盖一般参数域或真实效果，R26失败仍保留。[报告](docs/round27_rms_preflight_report.md) · [验收](results/round27_rms_root_acceptance_20261031.json)。

**第二十六轮 RMS 深度正则投影未通过合成数值预检。** 24次生产调用全部完成，原生格上的双参考/生产比较、六项恒等性及固定 ε 近常深度检查通过；但36次独立参考中7次违反直接FOV可加性门槛，涉及4条目、3种几何。最大差1.024e−4，超过1e−8，不能用其他通过项覆盖。生产/参考阶段分别1.471s/35.147s，预算通过；未读取真实输入或标签。[报告](docs/round26_rms_preflight_report.md) · [失败验收](results/round26_rms_root_acceptance_20261031.json) · [后续决定](results/round26_rms_post_result_decision_20261031.json)。

随后第二十七轮保留同一生产模型，仅在独立参考的标准X积分中加入固定宽度2的分段，补上对宽区间Gaussian密度峰的覆盖；保持误差门槛、阶数和预算，保留原24例并增加2个宽视野常深度例。新冻结运行通过全部数值检查，仍未晋升真实效果候选。[修订设计](docs/round27_reference_revision_design.md)。

**第二十五轮固定 Student-t4 必要条件筛查失败。** 48对旧TRAIN全部2,304个24→8查询及4,608次独立参考计算通过数值门槛，但区域KL为0.468172，强quad16为0.308353，恶化51.83%，超过0.314520的界限。六项宏中心保护通过；三个场景的KL/JS均比匹配Gaussian及quad16更差。停止此固定构造，不继续调ν/h/T或开启其后续确认。[报告](docs/round25_student_screen_report.md) · [验收](results/round25_student_root_acceptance_20261031.json) · [停止和后续决定](results/round25_student_post_result_decision_20261031.json)。

随后第二十六轮检验了单个三维Gaussian按深度二阶矩规范化、在XY加入固定正则后做正深度透视投影的构造，结果见上方。它保留深度与横向关系，仍无额外网络或混合组件；但改变了固定屏幕Gaussian平滑的含义，沿射线深度变化也可能产生额外重尾。[数学准备稿](docs/round26_rms_projective_design_draft.md)保留此反例与全部原门槛，目前没有新真实效果、已选择的3D anchor/content规则或效果候选晋升。

**第二十四轮固定 Student-t4 合成数值预检通过，尚无方案通过全部科学验证。** 18种几何×2种固定偏置、6项恒等检查及72次独立参考调用全部通过。生产对最高阶参考最大P误差1.04e−12，raw质量相对检查亦通过；42次生产阶段约0.328秒。该结论仅限冻结合成输入，不能当作真实效果或一般参数域保证。[报告](docs/round24_student_preflight_report.md) · [验收](results/round24_student_root_acceptance_20261031.json) · [下一步决定](results/round24_student_post_result_decision_20261031.json)。

随后第二十五轮完成该固定构造的必要条件筛查，数值通过而区域效果失败，见上方。

**第二十三轮真实输入适用性检查失败，尚无方案通过全部科学验证。** 72 个固定旧 TRAIN 查询均通过独立三维输入核验，但原射线积分实现没有生成任何完整概率结果：22 次内部下溢、25 次单次积分预算耗尽、25 次共享时限失败。整轮耗时600.338秒，超出600秒；最后24条没有实际积分，不能据此判定这些输入自身数值失败。独立审计和root验收确认了完整失败记录。[报告](docs/round23_ray_applicability_report.md) · [72条状态](docs/round23_ray_applicability_report_data/all_72_statuses.csv) · [验收](results/round23_ray_root_acceptance_20261031.json) · [后续决定](results/round23_ray_post_result_decision_20261031.json)。

停止推进这一冻结射线实现的效果实验。随后完成的固定 Student-t4 合成数值检查见上方；已有同协方差圆环反例，尖中心与重尾不保证区域收益。

**第二十二轮单3D Gaussian投影测度通过固定合成数值预检，尚无方案通过全部科学验证。** 直接将严格正深度的三维Gaussian投影到图像，沿用h=.25平滑和一次面积floor，T=0。15个固定基础案例及8项变换/极限检查全部通过，独立64/96阶参考与完整库存验收通过。最大P误差1.92e−15；低于floor的原始FOV质量约1.78e−33，原始质量相对误差4.37e−12。覆盖包括变深度相关covariance，但不构成任意非零均值输入的误差保证。[数值报告](docs/round22_ray_preflight_report.md) · [接受记录](results/round22_ray_root_acceptance_20261031.json) · [后续决定](results/round22_ray_post_result_decision_20261031.json)。

这不是效果候选晋升：相关三维例仅16格就耗时约0.485秒，真实432格成本和一般数值适用性仍需检查。下一项仅对72个预先固定的旧TRAIN几何输入做无标签适用性/成本失败筛查，并先独立核验三维输入；即使所有调用完成，也不解锁效果标签或替代一般数值参考。原区域、中心、扰动、实际V、GS和未触碰联合确认要求全部保留。

**第二十一轮匹配机制诊断完成，尚无方案通过全部验证。** 48对旧TRAIN、三种query尺度、八个固定角色的1,152个概率矩阵与576个原始质量矩阵已封存，独立数值、指标审计和root库存验收通过。共同内容下，24→8的full4区域KL由单Gaussian的0.325720降至0.258441，但中心PCK从91.9355%降至46.1252%；零内容也有同类冲突。换源中心改善宏定位却损失粗尺度区域精度，补二阶矩只能部分补回，不能据区域收益宣称整体成功。全部结果属于重复旧TRAIN的机制诊断，不是新的候选通过。[完整报告](docs/round21_matched_decomposition_report.md) · [接受记录](results/round21_matched_root_acceptance_20261031.json) · [后续决定](results/round21_matched_post_result_decision_20261031.json)。

随后第二十二轮完成了这个精简投影测度的固定合成数值预检；结果见上方，尚未检验真实数据效果。

**第二十轮固定中心 Gaussian 未通过完整开发筛查，尚无方案通过全部验证。** 48 对旧 TRAIN、14 条件、12 角色的8,064个概率矩阵已完整封存，独立数值与指标审计及root核验通过；固定主方案只通过58/60项要求，不能近似算作通过。失败两项是同一24→8条件对同配置quad16的两个角色：区域KL为0.332541，对照0.308353，差7.844%，超过2%限制，三个场景均退化。中心保护全部通过，24→8等场景PCK相对旧trace提高1.4743pp，但c3下降0.5236pp。区域与定位的取舍仍未解决；停止这一固定构造，不启动其旧VAL、实际V、GS或fresh，也不事后提升归因角色。[完整报告](docs/round20_anchor_fixed_report.md) · [接受记录](results/round20_anchor_root_acceptance_20261031.json) · [停止决定](results/round20_anchor_post_result_decision_20261031.json)。

[第二十轮协议](ROUND20_ANCHOR_FIXED_PROTOCOL.md)固定h=.25/T=10，单高斯均值用源中心a，二阶矩用`C_a=C+(mu-a)(mu-a)^T`；点/固定形状对照获得相同a，旧强控制全部保留。matched归因表明rank-one增宽与各向异性对区域KL有作用，但没有额外q24中心PCK收益；JS反例、极小原始质量行及旧full4优势均在完整报告中保留。随后第二十一轮以共同内容强度完成中心、二阶矩与高阶形状的固定机制诊断，未晋升任何角色。实际V、优化GS和未触碰确认仍是后续必需验证。

**第十九轮中心信息诊断通过探索筛选。** 48 对旧 TRAIN、28,224 个查询中，合法预测深度投影的源中心相对区域均值，在8/16/24px尺度的最近格定位分别改善0.5137/0.6257/2.4038pp，24px三个场景均改善。全部前向先封存，独立投影审计后才读标签；两项审计通过，root核验完整库存及来源。c3的16px定位下降0.3525pp、8px距离RMSE恶化等反例均保留。此结果只说明中心身份值得继续检查，最近格诊断并非强point attention，更不是Gaussian/实际V/GS/新场景效果。[完整报告](docs/round19_center_semantics_report.md) · [接受记录](results/round19_center_root_acceptance_20261031.json)。

**第十八轮条件归一化实现未通过合成数值预算。** 25个预设案例中，GH64²生产积分有8例至少一项失败，含5例参考积分也未充分收敛；另3例已明确证明生产误差超限。未把GH96提升为主选，未进入真实TRAIN。连续概率法则未因此被否定，但目前没有可靠高效实现。[数值报告](docs/round18_conditional_preflight_report.md)保留所有失败和数值边界。

**第十七轮完整旧验证与独立审计已完成，固定候选未通过科学门槛。** 单高斯位置分布配合 `τ=T h²/(h²+tr C)`，固定 h=.25/T=10。32 对旧验证图像、15 个实际算子的 32,640 个矩阵已完整重放；独立数值/指标审计通过，原始文件及来源链另经核验。常规九类条件 KL 相对强 point 降25.61%，相对quad16降1.93%；增量算子时间为quad16的0.06734倍，故效率路径通过。但24→8、24→24的区域KL分别比quad16差16.08%、2.65%；24→8中心PCK比两个选定定位对照低0.8876pp。相机旋转±1°的区域KL分别差7.37%、9.64%，负旋转和负平移的中心保护也失败。**停止这一固定候选进入实际V、优化GS或新场景确认，保留全部失败，研究目标仍未完成。** [完整门槛](results/round17_validation_report_tables/all_gate_conditions.json) · [接受记录](results/round17_validation_root_acceptance_20261031.json) · [后续决定](results/round17_validation_post_result_decision_20261031.json)。旧训练选择、历史路径修复、审计中断和运行时失败记录均保留；旧VAL只能继续作为开发回归，不能再次称独立确认。

旧TRAIN的温度归因也已完成1,344个矩阵及独立审计：仅用尺寸的温度使common9/direct5 KL略降0.1175%/0.0878%，但24→8中心PCK低0.3599pp；不能据此替换主选或删除几何C。[归因报告](docs/round17_size_temperature_report.md)保留完整人口与边界。主VAL的同τ等方控制说明完整各向异性在zero/scale区域KL上分别改善3.98%/7.68%，同时24→8中心PCK低0.9262pp；这支持保留形状信息的价值，也表明当前区域与定位取舍未解决。归因对照不是通过验证的新候选。

**实际 V 接口准备完成，真实训练尚未启动。** 新适配保留原41,345参数，在线 learned Q/K 与 V/head；73项单元测试、16项纯合成数值检查通过。双精度参考的深度最大相对误差1.64×10⁻⁷、参数梯度最大相对误差1.22×10⁻⁶；初次MKL/OpenMP导入失败及单独冻结的一行导入顺序修复均保留。此项只证明声明精度接口在合成输入上正确，真实任务仍以主VAL全部通过为前提。[预检报告](docs/round17_value_interface_preflight_report.md)。

后续[角色与缓存草稿](docs/round17_value_roles_preparation_draft.md)列明完整对照去重和资源需求；内存缓存建议保留“全部几何哈希先于任何训练”的封存时序。该草稿尚未冻结或执行。

[近期六项方法的原文核查](docs/round17_recent_prior_synthesis_20261031.md)区分了 Gaussian 参数拼接、中心/相机位置编码和完整协方差控制位置关系；保留版本及访问限制，不据有限检索宣称新颖性，也不改变实验门槛。

**第十六轮全量机制诊断及独立审计完成，尚无新方案通过科学门槛。** [诊断报告](docs/round16_mechanism_report.md)复核了全部5,184个旧训练概率矩阵、1,016,064个query/configuration状态：在h=.25/.5时，四次full核加入T5内容使q8定位变化+0.714/+0.114pp，却使q24下降5.991/7.664pp；大尺度原生格几何间隔更小，相同均值的点核没有同等退化。结果支持检查支撑与内容权重的耦合，不能只归咎共享均值误差，也不能推出全局关闭内容。此次没有新预测或新数据，54项测试及独立审计通过仅证明诊断链正确。[交付清单](results/round16_delivery_manifest.json)。这项诊断随后提出第十七轮固定候选：保留Gaussian full1几何，只令内容温度`τ=T h²/(h²+tr C)`；C是空间支撑，非置信度。[候选协议](ROUND17_CANDIDATE_PROTOCOL.md)与[设计决定](docs/round17_design_decision.md)保留原12格、三项定位保护、全部后续验证，并在新效果前声明区域止损规则；当前结果见上方第十七轮条目。

**第十五轮完整训练校准已审计，停止固定四次核候选，旧验证未启动。** [区域结果报告](docs/round15_regional_report.md)：48个旧训练pair、8类×12配置×14条件共64,512条结果完整重放与审计。满足三尺度定位保护的主选h=1/T=0，原九条件KL为0.8990，高于强point的0.6432与quad16的0.5226；五尺度KL也更差。无约束区域最优h=.5/T=5则违反中心定位保护。停止属于看到完整训练结果后的新增开发决定，不是预注册Stage B门槛或未运行VAL的失败。原选参、GL16数值失败、GL24修复及全部结果保留；一般有效性目标尚未完成。[定位报告](docs/round15_feasibility_report.md) · [停止决定](results/round15_regional_post_result_decision.json)。

**第十四轮确认关闭内容不足以修复粗尺度定位。** [第十四轮报告](round14_report.md)：48 个旧训练 pair、1,008 个概率矩阵及独立审计表明，关闭 full4 内容项只补回训练定位缺口的 10.84%，却使 8/16px 区域 KL 恶化 24.69%/12.18%。同带宽与温度下单高斯仍有较好定位、较差区域精度；下一候选先检查只保留均值和协方差的四次密度闭合。尚无通过全部验证的新方案，原失败与门槛保留。

**第十三轮完成两项旧训练诊断，停止固定 OU 收缩候选。** [第十三轮报告](round13_report.md)：24px 真实区域峰值与独立中心的严格冲突约29.72%，但满足中心峰值约束的最小KL代价很小，不能据此豁免定位保护。无标签前向诊断中，full4 与完整预测像素分布的T5 argmax已有23/24一致；固定OU收缩在全部24查询、两个温度下均损失分布精度，故不继续扫描参数。两项独立审计通过，原门槛不变。

**第十二轮尺度检查未通过。** [第十二轮报告](round12_report.md)：固定 full4 在五种24px组合上区域 KL 相对 point 降41.89%、相对quad16降1.02%，六项几何扰动均通过；但24→8中心PCK损失11.20pp，独立审计确认，故没有启动该版深度重训或新场景确认。

**第十一轮四个局部高斯通过全部旧开发区域门槛及独立审计。** [第十一轮报告](round11_report.md)：在已有 quad4 的固定四分层中补回局部 covariance，区域 KL 相对 point 降低24.21%；与 quad16 精度基本相同，增量算子约快3.52倍。九种分区均通过非劣检查，点定位损失0.157pp，满足0.5pp保护。范围仍限两个旧验证扫描；实际深度、GS/RGB 和新场景有效性尚未成立。接下来按冻结协议检验24px、几何扰动和真实V任务。

**第十轮两种区域编码均未通过完整门槛。** [第十轮报告](round10_report.md)：固定80对旧开发图像，在真实拆分、合并与混合尺度上比较单高斯和4/16点。Gaussian overlap相对点基线KL降19.18%，但差于16点1.25%、PCK降0.613pp；实际矩形积分相对点降23.57%，差于16点0.762%、算子约快11倍，但粗query条件与PCK保护仍失败。两轮独立审计通过，未打开新的确认集。下一项受控验证只为已有四个子区域补回内部covariance，见[第十一轮协议](ROUND11_PROTOCOL.md)。

**第九轮完成真实GS参数读取观测RGB、固定几何渲染的最小验证，未过继续门槛。** [第九轮报告](round9_report.md)：两旧开发场景新做context-only特征匹配/三角化和GS拟合，各24输入、4校准、4验收图；相机沿用全图SfM，结论限给定标定。18读取配置中，full相对最强point的验收RGB MSE仅降0.146%，两场景一好一坏；相对本轮同容量原GS RGB参考仅降0.0477%。五维读取与相应EWA数值等价，但内部R/S的额外实用价值未获支持。按预设规则停止该分支；215项测试及独立复算通过。[协议](ROUND9_PROTOCOL.md) · [统计](results/round9_analysis/README.md) · [实景算子审计](results/round9_operator_audit/summary.json)。

**第八轮完成对应／源值四组诊断及单视图控制，共25次重训。** [第八轮报告](round8_report.md)：旧开发集上，真对应＋原预测源值相对单视图仅改善1.031%，未过2%门槛；传感器辅助源值改善5.992%，两者联合改善6.584%。这些额外真值输入仅用于定位限制，不能作为可部署方法成绩。42.90%的有效查询有可核验对应，源值辅助在其余区域反而比原融合基线退步；结果支持优先检查源值质量，暂停当前深度任务中的协方差单独调优。149项相关测试通过，独立复算及基线逐位重现通过，一般下游有效性目标仍未完成。[协议](ROUND8_PROTOCOL.md) · [统计](results/round8_oracle_analysis/README.md) · [独立审查](docs/round8_protocol_review.md)。

**第七轮完成尺度、真实 V 深度预测和优化后 GS 参数验证。** [第七轮报告](round7_report.md)：24px五种新尺度在旧五场景上区域KL降低11.09%，但24→8点PCK8相对fixed下降0.46个百分点。五个新场景、六对照×五种子的粗深度任务中，full相对验证预选point仅改善0.335%，95%区间跨零，同容量singleview均值更好，未通过主门槛。真实GS审计也未支持内部R/S收益。当前保留区域支撑PE的精炼正结果，一般下游有效性仍未获证实。[协议](ROUND7_PROTOCOL.md) · [确认统计](results/round7_depth_analysis/README.md) · [原始预测审计](results/round7_depth_test_audit/README.md)。

**第六轮五维 Gaussian 位置编码通过预设区域匹配确认门槛。** [第六轮报告](round6_report.md) 将支撑协方差转成五维非对称 Q/K 几何通道，与冻结视觉特征拼接；零新增学习网络、两个全局校准标量。在五个新扫描、480 对、四种尺度组合上，区域 KL 相对尺寸感知 fixed 对照降低6.08%，两跨尺度方向降低3.20%/9.51%。点PCK@8为95.076%，比专门调优的点对照低0.089pp，满足预设保护门槛。范围限已提供相机的条件区域匹配；五维精确性依赖规则目标网格，且二次Gaussian attention数学已有先例。[模型](models/gaussian_quadratic_pe.py) · [协议](ROUND6_PROTOCOL.md) · [封存统计](results/round6_confirmation_analysis/README.md)。

**第五轮的负结果保持不变。** [第五轮报告](round5_report.md) 包含六方法×五种子的真实SUN3D两层注意力实验。normalized Gaussian PE相对投影点RoPE的PCK8差值为−0.070pp，95%区间[−0.400,+0.166]pp，未过主门槛；旧GS的双侧center-only重训练也未支持内部R/S的独立增益。它与第六轮的区域匹配任务、最小结构分别解释，不能由第六轮改写。[第五轮协议](ROUND5_PROTOCOL.md)。

[第四轮报告](round4_report.md) 包含200组训练，检查双侧形状对照、尺度鲁棒性与结构简化。其条件区域匹配结果与第五轮的新任务分开解释，不能直接视为两层网络或NVS收益。

[第三轮报告](round3_report.md) 保留原48维、零新增学习参数的“高斯期望旋转后归一化”及其前瞻真实相机验证。该轮证据限已有重建几何下的条件区域匹配。第三轮协议见 [ROUND3_PROTOCOL.md](ROUND3_PROTOCOL.md)。

本仓库验证把高斯参数化转成位置编码的方法。第六轮验证极简Gaussian支撑编码与独立真实像素面积标签；第五轮加入真实两层注意力与跨场景点对应；第三轮使用已训练GS和独立SfM轨迹标签；前两轮以数学检查与合成机制实验为主。所有轮次均不是WorldRoPE / HRoPE完整复现，目前没有完整新视角合成网络结果。

第三轮验证入口：`python -m pytest tests -q`。模型核心见 [gaussian_pe.py](models/gaussian_pe.py) 的 `normalized_rotary`，完整复现和本地输入要求见第三轮报告。第二轮报告仍保留，第一轮接近 99% 的结果应按后续身份捷径对照重新解释。

实验结论与方案调整见 [research_report.md](research_report.md)。原始逐场景结果、验证集选择、训练曲线与运行环境保存在 `results/`；每轮包含当时源代码快照及 SHA-256。

## 第二轮复现

```bash
python -m unittest discover -s tests -v
python -m experiments.launch_gpu --module experiments.identity_probe --out results/reproduced_identity
python -m experiments.launch_gpu --module experiments.rendered_probe --out results/reproduced_rendered
python -m experiments.launch_gpu --module experiments.rgb_geometry_probe --train-only --out results/reproduced_rgb_geometry
python -m experiments.rgb_geometry_probe --eval-only --out results/reproduced_rgb_geometry --matching results/reproduced_rendered
```

GPU 启动器会重新检查空闲设备；CPU 运行可直接调用各模块并传 `--device cpu`。完整默认配置为五种子、每匹配器 600 步，门控按验证集规则有条件运行。第一阶段身份探针沿用每模型 400 步。

现有结果的报告与审计命令：

```bash
python -m experiments.uncertainty_calibration
python -m experiments.round2_audit
python -m experiments.round2_summary
```

校准、审计及报告模块默认读取交付时的 `results/round2_*` 目录。重放与重新生成审计需要本地 checkpoint；这些 checkpoint 仍保留在本机并由 `.gitignore` 排除。2026-11-01按用户要求清理了R2–R4十个预测目录中的4,140个过时NPZ，释放约2.33 GiB；旧逐预测审计须先重新生成相应输出。[删除清单与依赖检查](results/early_prediction_cleanup_20261101/plan.json)保留文件哈希。原始数据、逐场景指标、CSV、manifest、源代码快照及预览图仍保留。

本轮训练了一个**小型合成域 RGB 几何预测器**，不是预训练几何模型。HRoPE/WorldRoPE 只实现了注明范围的 Q/K 分数算子，未复现 V/输出处理、完整预测头或 NVS 训练。

## 环境与运行

当前已验证环境：Python 3.11.7、PyTorch 2.6.0+cu124、NumPy 1.26.4。前两轮合成实验无需下载模型或数据；第三轮使用已有本地重建数据，冻结视觉对照另需已有 ResNet18 权重，来源与哈希均记录在 manifest。不安装 CUDA 扩展，也不修改系统 GLIBC。

```bash
python -m unittest discover -s tests -v
python -m experiments.math_probe
python -m experiments.stability_probe
```

第一轮：13 个变体、5 个种子、每个模型 400 步。可以直接使用 CPU：

```bash
python -m experiments.matching_probe --device cpu --out results/reproduced_pilot
```

也可以由启动器检查 GPU 占用，按 7 → 0 顺序选择无计算任务且显存占用低于 128 MiB 的卡。没有空闲卡时停止，不抢占其他进程。Codex 沙箱可能隐藏设备，此时 GPU 命令需要经执行环境授权后运行。

```bash
python -m experiments.launch_gpu --out results/reproduced_pilot
```

第二轮按首轮发现加入强基线消融、平滑矩描述和支撑区域变化：

```bash
python -m experiments.launch_gpu --out results/reproduced_revision \
  --methods center,rope3d_axis,rope3d_oblique,expected_oblique_support,rope3d_axis_normal,rope3d_axis_scale,rope3d_axis_logcov,rope3d_axis_projected_logcov,rope3d_axis_bounded_logcov,rope3d_axis_moments,rope3d_axis_random_control,expected_axis_uncertainty_oracle_logcov
```

对已训练模型仅在验证集校准几何权重，再用全新测试场景评估：

```bash
python -m experiments.calibrate_probe
python -m experiments.summarize
```

`calibrate_probe` 默认读取 `results/revision/checkpoints/`。权重保留在本地，但 `.gitignore` 排除这些可重新生成的 checkpoint。若重新训练到其他目录，使用 `--checkpoints` 指定。

严格重现某轮历史代码时，从对应的 `results/<轮次>/source/` 中运行其模块；最终代码新增了 `view_support` 对照，首轮源代码快照保留原始 8 种场景。

结果审计修复了极端噪声下尺度特征取 log 的正数下限。训练过程未改动；前两轮指标由已有 checkpoint 重新评估，修复前指标保存在 `pre_numerical_audit/`，修复后评估代码在 `evaluation_source/`，记录见 `results/numerical_audit.json`。需要重建此审计时运行 `python -m experiments.audit_results`。

## 第一轮实现的实验边界（第三轮以最新报告为准）

- 每个模型只有 2,387 个登记参数；外观网络和初始化相同，几何距离只增加有效标量权重。所有模型登记同样数量的参数，部分参数在基线中不参与计算。
- 两视角共享潜在支撑的合成设定有利于形状匹配；协方差没有从真实 RGB 中预测。它用于验证信息是否可能有用，不能证明可学习性或实际性能。
- `expected_*` 是直接三维期望 RoPE 消融；`reprojection` 是重投影距离核。二者都不应标记为正式 WorldRoPE 复现。`*_uncertainty_oracle*` 使用已知注入误差协方差，只是诊断参照。
- `center_projected_logcov` / `rope3d_axis_projected_logcov` 比较共同查询视角中的投影协方差，不包含完整的局部单应 Attention。
- 特征分解描述子在本轮预计算，未通过协方差预测器反向传播。端到端接入前需要修复重复特征值处的梯度实现；`covariance_moments` 提供平滑对照。
