# round_005：结束固定场形状诊断，核查下一条参数 PE 路线的事实依据

2026-11-13；分析角色 `/root/research_analysis_astra`。本阶段只评价和制定有限计划，没有训练、GPU调用、重算历史效果或改动已完成报告。实施完成身份沿用主任务一次核验的[18项输入](input_reports.json)和[交接索引](analysis_input_handoff.json)。

## 1. 本轮取舍

`native_covariance_role1`进入事前 `limited_uncertain_or_mixed` 分支，**结束这次四组诊断及同一固定场上的协方差 PE 自动接口修补**。训练片段内形状条件依赖成立，开发上的实用依赖未建立；这两条同时保留。结果不证明形状无用、纯中心已足够、重训中心方法不劣，也没有新增 O6 或独立确认。下一有限阶段为 `gaussian_pe_route_review1`：固定四项原文与作者代码的事实核查，直接回答是否存在能支持强中心匹配比较、且与已关闭尝试有实质条件差异的参数 PE 问题。当前没有足够依据再启动一轮拟合。

### 完整结果的决定作用

下表正数表示干预相对 F **误差增加**；与此前 PE 报告的“降误差”符号相反。区间是原计划的地点配对区间，种子不作为额外独立地点。

| 固定比较 | 12个开发地点原查询 | 32个训练片段其他查询 | 本次判断 |
| --- | --- | --- | --- |
| C_match − F | +0.5957% [−0.8526,+1.7919] | +4.0803% [1.7890,6.4606] | 开发未满足≥1%、下界>0的原实用规则；片段内条件依赖保留 |
| S_match − F | +1.3999% [−0.2811,+3.0246] | +5.3332% [2.6680,8.8185] | 开发幅度均值达到1%但区间不支持；片段内保留 |
| C_raw − F | +1.3708% [−0.5072,+2.9550] | +4.2003% [2.0170,6.5345] | 未匹配覆盖的开发响应同样不确定，不选择它替代主比较 |

三种子在上述两个主干预中均同向，开发 C_match 的地点为7正5负、S_match为8正4负；同向性不足以绕过已定实用规则。训练原12片段的 C/S 代价6.5364%/9.4546%，新增20片段为2.8170%/3.2132%，区间均为正，条件正证据不限于原12片段。没有预定的训练×开发交互估计，不能仅由两边显著性不同宣布差异已被统计证明。[完整报告](../../native_covariance_role1/report.md)、[全部数表](../../native_covariance_role1/report_tables.md)、[事前分支](../round_004/plan.json)。

四组对Local仍保留约38.85–41.95%的训练片段空间用途；开发F相对Local误差低8.7369%，区间[−21.8790,+8.9719]%仍跨零。训练内用途与跨地点净用途分开。开发 C_match 的原始ROI误差变化−0.1627%、bias² −0.4595%、完整AbsRel −0.2028%均不确定；S_match对应+0.2980%、−0.1334%、+0.2079%也不确定。训练 C_match 完整AbsRel约+0.2368%的小代价有支持，不能因空间收益而省略。开发未支持区域 C_raw 的约+1.4310%误差有支持；支持/未支持两层全部保留，不筛除困难像素取得阳性。原始、偏差、完整图、原12/新增20、逐地点/种子/query与最大平均反例 `fidelity_01/q11` 均以原报告为准，无新主终点。

F新路径与旧缓存R的开发主差+0.0238% [−0.0381,+0.0830]%单列；它没有被当作输出相等门槛或唯一病因。开发零支持像素F/C/S约20.8700%/21.3643%/19.4722%；C/S回退位置原a_F质量仅0.149413%/0.014557%，但质量小不证明局部误差可忽略。`fidelity_02/q8`、`fidelity_03/q5,q8,q11`四个原a_F全零query保留评分。相对内容范数 C_match 约0.9871–1.0220、S_match约0.9798–1.0027，是描述，不追加范数控制或推出单一因果。[输入统计](../../native_covariance_role1/input_statistics_summary.json)、[回退清单](../../native_covariance_role1/fallback_cases.csv)、[既有两图审阅](../../native_covariance_role1/figures/visual_review.json)。

本次实现成立：4224正式新预测+2112继承，4232 native前向、528实际渲染、0训练，独立复算和原mask身份完整；GPU 0.165128worker小时、DATA 2.809215GiB均在原限内，全部进程退出。CPU支持身份查表修复已版本记录，无GPU恢复或科学变更。接受主任务完成核验，不再hash大权重或重复GPU审计。[实施交接](../../native_covariance_role1/implementation_handoff.json)、[成本](../../native_covariance_role1/cost.json)、[完成检查](../round_004/coordinator_completion_check.json)。

## 2. 保留与收窄的解释

C从中心邻域得到方向和轴比例，且额外共享原trace/最薄轴；绑定内容和覆盖a_F仍携带原形状。因此这里的中心替代受到优待，但也没有移除系统全部形状信息。固定权重输入扰动只能说明这个已训练Render对输入分布的条件响应，不能代表经训练的中心系统上限。片段内正响应支持“该载体在部分已适配场景中使用了固有支撑”；不支持“同样的信息放入Transformer PE必然提供中心之外的增益”。

仍相容的竞争解释包括：形状对已适配片段有作用而跨地点作用弱/不稳定；形状已由共同渲染和内容表达，显式PE多为冗余；中心几何对当前深度读出已表达主要可用位置信息；固定场的几何/内容与任务所需关系不匹配；现有参数PE的学习方式没有取得可迁移优势。覆盖空洞、尺度或浮点差异都不能被选为唯一原因。这些是范围解释，不是下一串自动消融任务。

既有公共粗场及可重载writer状态的正确性、训练片段Render约42%的用途、writer开发原query的共同均值线索、query参数PE的raw正线索均保留；分别属于载体/适配/标定证据。最近单层writer主G/P降误差−0.0577% [−0.3739,+0.1347]%，渲染后Q/K主G/P −2.935%且三种子均未改善，不能被这些辅助线索抵销。未检出损害不称非劣性。总体公共3DGS记忆＋可靠参数PE及fresh确认目标仍未达到。

## 3. 为什么不再直接换一个接口

| 相关历史 | 已经改变过的条件和结果 | 排除的重复提案 |
| --- | --- | --- |
| [bound_read1](../../bound_read1/report.md)、[soft_read1](../../soft_read1/report.md)、[learned_projection_pe1](../../learned_projection_pe1/report.md) | 硬/软真实合成偏好、可学习投影标量均未建立可靠中心增量 | 再换温度、log变换或投影强度 |
| [source_geometry_control1](../../source_geometry_control1/report.md) | sensor来源改善中心组，但G/P在两种来源均无可靠主增量 | 将高质量/传感器场当作已证充分条件 |
| [native_gaussian_pe1](../../native_gaussian_pe1/report.md)、[coarse_memory_pe1](../../coarse_memory_pe1/report.md)、[gaussian_scene_memory1](../../gaussian_scene_memory1/report.md) | 原生适配、粗K/V、真实可渲染场已有受控阴性；共同收益常伴偏差变化 | 把接入原生、增加槽位或实际GS本身称新问题 |
| [native_input_pe1](../../native_input_pe1/report.md)、[native_render_parameter_pe1](../../native_render_parameter_pe1/report.md) | 输入attention、保留Render的原生Q/K参数PE均完整训练；后一raw正而空间主阴性 | 再换PE注入层或用raw替换主终点 |
| [native_memory_context_pe1](../../native_memory_context_pe1/report.md) | 真正query-independent单层关系writer，同完整cov普通内容和强中心，主增量未建立 | 只改writer邻域/层数/几何关系公式 |
| [native_render_factorization1](../../native_render_factorization1/report.md)、本次诊断 | 局部内容/覆盖/范数及形状支撑已有固定权重分解 | 连续做载体扫描而偏离O6 |
| [joint_write1](../../joint_write1/report.md)、[surface1](../../surface1/report.md)、[geometry_transport1](../../geometry_transport1/report.md) | 逐槽联合状态、显式几何传递和latent读取已有原始正线索与空间代价 | 把“联合学习几何和内容”或深度传递本身当新路线 |
| [更早区域/原生历史](../../../research_related_work/gaussian_parameter_encoding_20261106/notes.md)、[living历史](../../../../docs/research_current.md) | 区域关系PE有有限正证据，进入原生任务后强中心比较不稳定；期望RoPE、频率/局部坐标/区域重构均已有尝试 | 将物理支撑协方差自动解释为定位不确定性，或重命名旧频率/区域方案 |

这张表是与当前决定相关的索引，不是声称已审尽所有研究使用史。旧12/32训练、12开发、51地点、早期区域“确认”等全部已用；后续不能从中重命名fresh。

## 4. 下一有限阶段：参数 PE 原文与作者实现事实核查

**假设：** 某项外部工作可能具有本研究尚未受控检验的状态形成/位置组织条件，能构造同信息强中心比较，值得后来形成一个精炼的O6实验。竞争解释是：其形状只进入普通内容/渲染，系统成绩混合了状态学习与额外输入；或者真实参数PE虽存在，但与本地已关闭接口同构，不能提供新的投入依据。

只查四项。初步原文阅读给出核查入口，**不是完成的源码审阅，也不是方法采用**：

| 固定工作与原始来源 | 为什么本次值得核查 | 必须解决的事实缺口 |
| --- | --- | --- |
| [GaussianRenderFormer，2611.05593v1](https://arxiv.org/html/2611.05593v1)，[作者项目](https://svlwoof.github.io/gaussianrenderformer) | 原文有投影footprint加入cross-attention key及去除该分支的条件结果 | 投影形状/中心/普通属性如何分别进入K与V，和本地投影/QK接口究竟相同在哪里、训练对象或状态形成哪里不同；论文系统PSNR不等于O6 |
| [GPSToken，2509.01109v1](https://arxiv.org/html/2509.01109v1)，[仓库入口](https://github.com/xtudbxk/GPSToken) | 高斯区域与token内容共同形成，提供状态语义的比较对象 | 它是2D区域系统；参数查询、位置编码和普通内容的准确边界，何者可进入3D公共状态，何者只是旧区域/joint_write的重述 |
| [SplatFormer，2411.06390v1](https://arxiv.org/html/2411.06390v1)，[ICLR官方版本](https://openreview.net/pdf?id=9NfHbWKqMF)，[作者项目](https://sergeyprokudin.github.io/splatformer/) | 实际3DGS的共享feed-forward属性细化，便于核查状态学习与PE的区别 | 作者实现中中心与其他属性分别走什么路径；不能因用了Point Transformer就推定Σ是PE |
| [JSRE，2601.02339v1](https://arxiv.org/html/2601.02339v1)，[ICCV官方版本](https://openaccess.thecvf.com/content/ICCV2025/papers/He_Joint_Semantic_and_Rendering_Enhancements_in_3D_Gaussian_Modeling_with_ICCV_2025_paper.pdf) | 原文将形状图描述与中心PE组合，是“参数被使用”与“参数作为PE”的重要对照 | 原文z=f(G)+PE(c)及图构造、训练状态的准确源码路径；形状描述作为内容不能重标为已证PE增量 |

GPSToken/JSRE已有[2026-11-06原文笔记](../../../research_related_work/gaussian_parameter_encoding_20261106/notes.md)；本阶段的新交付是锁定作者源码的参数数据流、原始消融混杂和可构造的匹配对照，不重复abstract综述。M3/StreamVGGT/CUT3R/XStreamVGGT已有[11-11调研](../../../../docs/gaussian_memory_related_work_20261111.md)，不继续扩展名单。

Sol仅核验书目/版本/仓库身份，提取μ/Σ/opacity/f→Q/K/V/logits/状态/renderer的事实链、可复用状态与训练输入、消融和最小工程边界，保留未知项；**机制解释、候选取舍及下一科学方案由原Astra在完整交回后决定**。不要求论文预先证明G>P；没有隔离消融是待研究问题，不是自动否决。但需要区分可追溯事实与假说，不能将系统成绩当PE证据。

四项核查结束后，Astra最多形成一个有实质条件差异的候选；若只是普通内容、载体几何、已测接口或无法构造有意义对照，则不安排它们的训练。全部未找到候选是这次有限核查的有效结果，不等于总体目标成功、权限/资源阻塞，也不自动开启下一串论文检索。原Astra仍须据完整结果决定下一个有价值的问题或诚实收窄研究主张。

## 5. 比较合同、预算与大纲

后续可行性表必须让中心对照获得相同公共状态/内容、所有仅由中心可得的距离、方向、邻域和离散二阶/多尺度结构，以及共同普通形状内容；G额外量应能定位到参数的**位置组织**作用。若状态联合更新导致内容变化，须明确同起点、容量、训练范围和形状普通输入，允许由PE造成的后续状态差异，但不得把额外数据/分支/监督混成PE。S可用于分离尺度与非球形关系，是否需要由Astra后续具体方案决定，本轮不扩组。该合同是防止概念混淆的设计问题，不是对每个历史实验追加检查。

固定预算：4项工作、作者源码/配置文件最多24份、网络检索/取回最多48次、从首次取证起60分钟；0训练、GPU、模型推理、渲染、权重或数据下载。新增项目文件与暂存合计≤128MiB，DATA新增0。只做原文/源码静态追踪，必要时用现有CPU解析器检查符号/维度，不导入执行作者模型或安装依赖。到达预算即以已核实事实及unknown项交回，不补项、不另开综述。详见[执行方案](../../../../docs/gaussian_pe_route_review_execution_plan.md)与[机器计划](plan.json)。

本次直接服务O6的问题定义和强对照可实现性，并核查O2/O3的公共粗状态、O4/O5的读出路径；O1不变。C1–C6没有本轮实验性变更，仅记录外部选择差异；2048基元/1024维、给定相机、sensor/joint-validity、静态四源、depth和旧数据均仍是临时范围，不升为要求。没有新数据实验、独立确认或fresh reserve；若将来出现合格候选，须在确认前审查全研究使用史并锁真正未用身份，本次不把未完成fresh审查当零GPU文献核查的阻塞。

分析请求配置为创建时显式 `gpt-6-astra/max`，本次复用无覆盖字段，后台生效配置不可独立核验。下一实施只复用现有CLI UUID `01a1d350-d1ec-70f2-9e37-8960b2dfe7a8`，每次请求 `gpt-6.1-sol/high/service_tier=fast`；主已见client-selected Sol/high，后台Fast没有独立字段。主控制派发、goal/state/events；本分析完成交接后释放科学写租约。
