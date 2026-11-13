# Gaussian 参数 PE 路线核查：固定执行方案

2026-11-13，`gaussian_pe_route_review1`，round_005。只读核查原文与作者源码，不是训练方案。科学依据与前轮取舍见[分析](../results/gaussian_memory/autonomous_research_loop/round_005/analysis.md)，机器边界见[plan.json](../results/gaussian_memory/autonomous_research_loop/round_005/plan.json)。输出根目录固定为 `results/gaussian_memory/gaussian_pe_route_review1/`。

## 1. 唯一问题与分工

核查四项工作中Gaussian参数的真实数据流及比较条件，为Astra判断“是否有一个与已关闭本地接口有实质差异、可相对强中心检验的PE问题”提供事实。竞争解释包括：形状只是普通内容/渲染；系统收益来自额外输入、状态学习或容量；有参数PE但与本地既有接口相同；确有不同的状态/位置组织条件。外部正结果和消融均不能直接替代VGGT公共记忆中的O6证据。

**Sol只负责事实提取和必要工程可行性核查。** 不负责选择科学假说、解释唯一失败原因、采用/否决方法、设计下一轮训练，或将系统指标归因为PE。原Astra在结果交回后统一解释、对照历史并作候选取舍。允许明确unknown，不以填满所有格子为验收门槛。

## 2. 固定四项与来源身份

| 工作 | 固定原文入口 | 官方版本/作者入口种子 | 优先取证位置 |
| --- | --- | --- | --- |
| GaussianRenderFormer | [原文v1](https://arxiv.org/html/2611.05593v1) | [作者项目](https://svlwoof.github.io/gaussianrenderformer) ，仓库候选 [仓库入口](https://github.com/SVLwoof/gaussianformer) | Gaussian属性编码、中心RoPE、投影footprint到K/V的路径；去footprint消融 |
| GPSToken | [原文v1](https://arxiv.org/html/2509.01109v1) | [仓库入口](https://github.com/xtudbxk/GPSToken) | Gaussian参数查询/更新、2D区域内容形成、位置编码与渲染/重构 |
| SplatFormer | [原文v1](https://arxiv.org/html/2411.06390v1) | [ICLR官方版](https://openreview.net/pdf?id=9NfHbWKqMF) ，[作者项目](https://sergeyprokudin.github.io/splatformer/) ，仓库候选 [仓库入口](https://github.com/ChenYutongTHU/SplatFormer) | 原始GS到共享细化器、中心位置编码、其他属性内容/残差及训练损失 |
| JSRE | [原文v1](https://arxiv.org/html/2601.02339v1) | [ICCV官方版](https://openaccess.thecvf.com/content/ICCV2025/papers/He_Joint_Semantic_and_Rendering_Enhancements_in_3D_Gaussian_Modeling_with_ICCV_2025_paper.pdf) ，仓库候选 [仓库入口](https://github.com/fly519/JSRE) | 协方差图描述、z=f(G)+PE(c)、Q/K/V、传播和状态更新 |

仓库URL是待按论文/作者项目相互链接核实的入口，不把名称相似当官方身份。记录真实标题、作者、发布日期、会议/版本、所读公式/节/表编号；预印本与会议版本若不同，只记录与本题有关的差异，不做全版本追踪。确认作者链接后读取默认分支commit，保存commit SHA、文件路径/行号/符号和小文件SHA；固定已取到的版本，不跟踪上游变化。无法确认官方仓库或代码缺失须直报；一份来源不可用时最多换一个官方入口，不去社区复刻补齐未知事实。

GPSToken/JSRE已有[旧笔记](../results/research_related_work/gaussian_parameter_encoding_20261106/notes.md)，不再产出abstract级综述；此次新增价值是源码路径、比较条件与同信息对照可实现性。M3等[已有综述](gaussian_memory_related_work_20261111.md)只作本地历史，不扩展第五项。

## 3. 每项统一取证表

同一schema记录四项，缺失处填`unknown`及已尝试入口，不推测：

1. **状态和输入：** Gaussian是2D区域、真实3DGS、潜在查询还是不确定性变量；μ、Σ/scale/rotation、opacity、视觉/语义f分别从何而来；是否sensor/给定camera/预拟合场；哪些是query-independent、可保存复用，哪些需每个query或scene重新优化。
2. **准确路径：** 从上述参数至embedding、Q/K/V、logits/bias/RoPE、MLP/状态更新、renderer的符号或短公式。注明是共同内容通道还是显式位置通道，作者命名与代码运算分列；若路径同时进入多处全部标出。对Σ的支撑/置信/定位不确定性语义只摘录作者定义，不替作者推定。
3. **比较事实：** 相关原始消融的组别、共同/不同输入、容量和训练范围、数据、指标与数值、统计或重复次数是否给出。没有真正中心/球形匹配消融就写缺失；不从整系统PSNR/mIoU或全模块删除推定孤立PE收益。
4. **工程事实：** 能否用公开代码定位参数PE最小单元、已有参数维度、图/邻域/位置构造和状态构建/每query调用；需要哪些外部依赖/权重/数据。记录一次构建、query读取、N次摊销的原文成本或可由调用图得出的计数，缺少实测就未知，不估造VGGT耗时。
5. **本地对应：** 按下方固定历史逐项写“事实相同/事实不同/未知”及证据定位；不要用新论文名称替代实质差异。给出可构造强中心对照涉及的具体张量/分支，不作科学优先级排名。

同信息强中心可行性必须说明：仅由中心可计算的邻域、距离、方向、离散二阶矩/多尺度结构不能归为Σ独有；P应共享G已有的普通形状内容、来源/相机/覆盖及共同适配。若论文连同状态形成或内容一起改动，列明怎么在共同架构/初始状态/参数容量/训练范围下隔离PE，或为什么当前事实还不够。允许由位置编码造成下游内容更新差异，但不将额外监督、输入或分支造成的变化写成PE。

**最小工程验证仅在事实路径需要时做：** 使用已有CPU文本/AST/配置解析检查维度、分支或调用位置，最多两个小帮助脚本；不导入执行作者代码、不装依赖、不跑模型/合成数值实验、不克隆全仓库。没有必要就零脚本。原文与代码冲突保留双方定位，不为追求一致无限调试。

## 4. 固定本地历史

按[round005分析](../results/gaussian_memory/autonomous_research_loop/round_005/analysis.md)的历史表读相关部分即可，不重复完整数值审计。最低对应包括：

- `bound_read1/soft_read1/learned_projection_pe1/native_input_pe1/native_render_parameter_pe1`：投影与query PE，不能只因新公式或新层重开。
- `native_memory_context_pe1`：公共query-independent关系writer，已给P完整cov普通内容和中心邻域；与新的共同状态学习须实质区分。
- `joint_write1/surface1/geometry_transport1/gaussian_scene_memory1`及早期区域PE：联合状态、显式/latent几何、真实GS和区域重构均有历史。
- `source_geometry_control1/native_gaussian_pe1/coarse_memory_pe1`：来源质量、原生适配与粗记忆已测。
- `native_render_factorization1/native_covariance_role1`：载体内容/覆盖/形状作用不能充作PE成绩，本次不再延伸干预。

不以“过去所有变体都失败”宣告所有可能性不存在；也不以某个维度没逐项测过便称新问题。Sol给可追溯事实，Astra判定是否实质不同。

## 5. 数据、预算与实际资源准入

所有本地效果数据均已用于开发。本次只读既有报告和身份索引，**不打开新场景/查询作评分、筛选或调参，不读取/复制大权重或预测数组**。44场身份、32训练/12开发及来源/查询列表保留在[round004计划](../results/gaussian_memory/autonomous_research_loop/round_004/plan.json)；更早51地点、区域确认和旧`confirm_*`命名同样已用。未声称完成全研究fresh审查；当前没有verified fresh reserve，候选成立后才能另行审全历史并先锁真正未用确认数据。论文自己的测试集不能被认作本项目fresh。

| 项目 | 本阶段固定上限 |
| --- | --- |
| 原始工作 | 4，固定上述名单；无第五项替补 |
| 作者源码/README/配置文件 | 共24份（可不均分）；API目录/commit元数据不计源码文件，但计取回次数 |
| 外部搜索/打开/API/HTTP取回 | 共48次URL请求或检索项；多URL调用逐项计，失败也计；本方案给出的已有事实不需重复取回 |
| 主动执行时间 | 首次取证至事实交付最多60分钟，含取回等待；到时立即写已取得内容/未知项，不以预算未满为完成要求 |
| 模型/算力 | 0训练、0更新、0模型推理、0rasterization、0 GPU小时；无GPU占用探测或CUDA初始化需要 |
| 新权重/数据 | 0；允许小型官方论文/文本源码证据快照，不下载数据集、checkpoint或整仓库 |
| 本轮所有新增文件含暂存 | 项目盘总计≤128MiB；DATA新增0；CPU帮助脚本≤2，不要求生成 |

开始时用`statvfs`核实项目盘有足够128MiB产物空间及1GiB剩余余量；当前[资源快照](../results/gaussian_memory/autonomous_research_loop/round_005/resource_snapshot.json)项目盘约15.7GiB可用，仅为计划时证据。每完成一项或写大于1MiB的来源快照前重查剩余空间和本轮字节；预计最终余量<1GiB或会超128MiB即停止取回、保存小报告并交回。这是一轮小型CPU文档工作的临时工程余量，不沿用GPU训练的8/10GiB或高频下降门禁。不删除旧/他人成果，不以易失内存冒充永久交付。只保存必要来源，普通URL/解析问题简洁留版本和未知项，不新增科学轮次。

预算计时只针对本次Sol取证，开始即记录UTC和计数；协调等待不触发擅自重试。最多24文件/48请求/60分钟任一达到即停止新增取证，产出边界诚实的报告。一次访问失败不是整体外部阻塞；可以在原预算内用一处官方替代入口，随后用公式级事实和unknown完成。

## 6. 交付与后续决定

交付以下少量文件即可，无额外形式审计：

- `report.md`：四项事实表及横向对应，引用靠近主张，原文与代码可区分；无独立方法采用结论。
- `sources.json`：真实书目、官方身份链、paper版本、repo commit、已读小文件SHA/定位和可用性。
- `mechanism_facts.json`：四项统一schema的路径/共同变量/消融/成本/控制可实现性事实，unknown显式。
- `retrieval_log.jsonl`、`cost.json`：最多48条取回记录、开始结束时间/实际文件数/字节/CPU帮助脚本情况，0GPU/model调用。
- `implementation_handoff.json`：状态、实际Sol CLI UUID/请求与可见配置、输入计划身份、产物SHA、未解决事实、已退出；科学取舍明确交原Astra。必要来源快照及一份普通问题记录可附，不扩报告数量。

四项逐一记录已核实/代码不可用/预算内未定位等状态后，Sol结束turn交主。该阶段的主要产出是**可复核的PE数据流和同信息比较边界**，辅助是状态复用、依赖与成本事实，不计算新的效应量、显著性或论文排名。

下一次Astra依下列分支决定，Sol不提前执行：

| 取证结果 | Astra随后可改变的决定 |
| --- | --- |
| 具有可定位参数位置路径，并有实质不同的状态/任务条件、可构造强中心控制 | 最多选择一个精炼候选，另写有限开发方案；不是自动批准训练，也不要求外部论文已经证明G>P |
| 只有普通形状内容/渲染/共同学习，或实际路径与已关闭接口相同 | 不采用这些作为新PE实验理由，保留原工作系统价值；结束本固定四项核查，不以换名重开 |
| 源码/消融不足或有冲突，但公式事实可见 | 以有限证据判断可否在本地明确构造反事实；不因缺代码机械否决，也不臆测细节、不无限取证 |
| 四项均未形成有依据的候选 | 如实交回“本有限范围无候选”；总体目标未达，不称成功/资源权限阻塞，不自动扩综述。原Astra仍负责下一实质问题或范围收窄 |

## 7. 大纲与角色边界

O6是直接问题：Gaussian参数怎样成为Transformer的位置组织信息及如何与强中心比较；O2/O3限定要服务VGGT公共粗3DGS状态，O4/O5要求明确视图读取/渲染路径，O1不变。本次不改C1–C6、不跑实验。静态四源、sensor/joint-validity、给定相机尺度、2048/1024、depth、冻结/全适配等都是已有实现的临时边界，外部条件差异须事实记录，不能因当前方案未覆盖便假定用户放弃。

原Astra仍为创建请求`gpt-6-astra/max`的collaboration任务，followup不能重传配置，后台有效值不可独立核验。主顺序复用**已有**Sol CLI UUID `01a1d350-d1ec-70f2-9e37-8960b2dfe7a8`，每次实际请求`gpt-6.1-sol/high/service_tier=fast`；client-selected Sol/high已由主见到，后台actual Fast未暴露。Sol不创建替代角色、不改变goal/state/events；完成本有限事实交接后释放写租约，等待主复用原Astra。
