# 研究大纲与实现对应：持续更新记录

**最新完成（2026-11-12）：native_receiver_adaptation1有限验证完成。** Local／Global／Render各三个配对种子、各2000步，全部24组frame/global交互块与原DPT同等训练，2880份新预测和864份继承开发结果、配对交互、完整训练集评价、成本及两图实际审阅完成。Render主空间误差0.022975，Local 0.022665、Global 0.029797；R对Local平均高1.37%，降误差区间[−29.36,+16.57]%，未建立可靠净用途，也不证明普遍有害。对Global低22.89% [4.32,39.45]%，正确绑定对打乱低22.16% [8.53,37.70]%，保留空间组织与绑定的有限正证据。

R−Local适配交互−0.000492，区间[−0.002967,+0.002147]跨零；R−Global交互−0.005174，区间[−0.012570,−0.000675]。训练集R对Local低41.42%而开发优势未建立，旧冻结行已有53.08%训练优势；共同适配不是该现象的唯一来源。原始缺失与完整全图AbsRel相对Local平均高7.29%/4.61%，区间跨零；支持ROI对Global低26.14%有支持、对Local低8.19%仍不确定。

结束此版固定保存内容＋单次融合＋小数据完整原生接收适配候选，不采用Render或Global，也不把Local称为高斯方法。O4整体可靠净用途未建立；本轮服务O4/O5、主要改C6，复用O2/O3，O6新增PE价值未检验。全部为旧数据、sensor辅助来源/共同有效性、给定相机和静态来源，非纯RGB、新确认或全writer/编码器联合训练；不外推为所有3DGS memory不可行，也不设为新PE研究的通用门槛。总体精炼有效方法仍未达成，无自动后继。

[完整报告](../results/gaussian_memory/native_receiver_adaptation1/report.md) · [固定数表](../results/gaussian_memory/native_receiver_adaptation1/report_tables.md) · [方法取舍](../results/gaussian_memory/native_receiver_adaptation1/method_decision.json)

以下旧“当前／进行中／仅设计”措辞均按形成时点保留；本段为最新状态。

**两配对种子中间描述（非最终取舍）：** 共同适配Local/Global/Render主空间误差0.022671/0.029323/0.022905；R对Local平均高1.04%（两种子分别低0.055%／高2.12%），对Global低21.89%，正确绑定对打乱低23.11%。训练集同指标R/Local为0.004338/0.007747，开发相对Local优势未建立；原始缺失误差R/Local为0.112456/0.107239，完整AbsRel为0.274628/0.268602。1920份新预测独立复算最大差1.92e−15，不给部分置信区间或方法采用，第三种子及2000步预算保持。[两种子描述](../results/gaussian_memory/native_receiver_adaptation1/two_seed_description.json)。

**首配对种子中间描述（非最终取舍）：** 20261110的共同适配Local/Global/Render缺失ROI空间误差为0.022575/0.028688/0.022562；R对Local仅低0.055%，对Global低21.35%，正确绑定对打乱低21.35%。训练集同指标R/Local为0.004369/0.008265，开发优势未相应保留。原始ROI误差R/Local为0.106994/0.107604；完整全图AbsRel为0.261883/0.266064。13份导出共1248新预测独立复算最大差1.92e−15，无部分置信区间、新确认或方法采用；剩余配置与预算保持。[首种子描述](../results/gaussian_memory/native_receiver_adaptation1/first_seed_description.json)。


**进度更新（2026-11-12）：native_receiver_adaptation1已完成2/9次2000步。** 首种子Local/Global的开发、完整训练集及冻结行训练集评价共576份新预测已导出，独立复算最大差1.92e−15；两份最终原生权重已SHA核对持久化到数据盘。Render和后续种子继续训练，尚无完整配对方法结论。临时续训转专用内存文件系统后，检查点写入约8s，原慢盘130–226s记录保留；科学源码/配置不变。[运行状态](../results/gaussian_memory/native_receiver_adaptation1/execution_status.json)。


**正式训练启动（2026-11-12）：** native_receiver_adaptation1六个原训练条件预检完成，零融合/原前向一致、48交互块及DPT有梯度，完整模型与双优化器恢复预测差0，峰值10.429GiB。科学配置及来源身份已锁定，Local/Global首种子已在真实GPU2/1启动，实际约6.37亿原生参数与原融合共同训练；正式输出尚未齐备，无效果结论。预检顺序优化器引用修复及资源让出记录保留。[运行状态](../results/gaussian_memory/native_receiver_adaptation1/execution_status.json)。


**当前授权执行（2026-11-12）：native_receiver_adaptation1已设为有限目标。** 用户接受原生接收网络共同适配建议，见[固定执行方案](native_receiver_adaptation_execution_plan.md)。复用保存场景及原融合，三组各三种子2000步，同等训练全部原生frame/global块和原DPT；主要改C6、服务O4/O5，O6仍未解决。已开始资源与旧训练片段预检准备；首个18GiB准入检查因显存余量不足而未启动CUDA子进程，按实测资源继续安排。当前无正式训练或新效果结论；下方“仅设计”为授权前记录。


**最新分析与建议（2026-11-12，仅设计）：** [直接融合结果的解释与原生接收网络适配建议](rendered_feature_results_analysis_20261112.md)。已有预测分解显示：R对Local原始误差增加的95.22%伴随ROI均值偏差平方增加；支持区内空间改善被无支持区内退步抵消。该描述不定位唯一病因，也不产生GT可部署选择器。建议复用同一场景与融合结构，让Local/Global/Render同等微调全部原生frame/global块及DPT，与已完成冻结行比较接收训练条件的交互。主要变更C6、服务O4/O5，O6仍待明确检验；已有末四层LoRA、联合writer和各接口正负结果保留。新增九次拟合只是提案，尚未设新目标或执行，旧有限目标保持完成。


**最新完成（2026-11-12）：rendered_feature_native1有限验证完成。** Local／Global／Render各三个配对种子、各2000步，共九次训练；864份适配与288份绑定打乱预测、全量独立复算、成本测量和两图实际审阅完成。Render相对Local的缺失ROI空间误差高3.20%，三个种子均退步，地点/种子区间仍宽且跨零；原始缺失误差与完整全图AbsRel分别高18.99%和17.15%，区间支持增加。不采用本版Render或Global，也不把Local晋升为高斯方法，结束固定保存内容＋单次输入融合候选。

保留有限正证据：正确绑定比打乱的主空间误差低25.77% [15.86,37.59]%；来源一致支持ROI内Render比Global低13.60% [1.81,25.75]%，但相对Local的9.53%均值收益区间跨零。依赖与条件线索不等于整体净用途。三组是同容量的附加上下文选择，不是完整Local再追加来源的严格嵌套比较，不能据此宣布来源条件信息为零。

本轮实现O4/O5的一种读取接法，复用O2/O3；O4可靠净用途未建立，O6新增PE价值未在本轮检验，原生2D RoPE保持。全部为旧数据、来源传感器/共同有效性、给定相机、静态来源和冻结骨干/DPT条件，非纯RGB或新确认。总体精炼有效方案目标仍未达成，本有限目标不自动启动后继研究；所有历史正负结果保留。

[完整报告](../results/gaussian_memory/rendered_feature_native1/report.md) · [固定数表](../results/gaussian_memory/rendered_feature_native1/report_tables.md) · [方法取舍](../results/gaussian_memory/rendered_feature_native1/method_decision.json)

以下开头的旧授权和提案记录保留为历史；当前映射及尝试登记已更新为最终结果。

**当前执行（2026-11-12）：rendered_feature_native1已获授权并设为有限目标。** [执行方案](rendered_feature_native_execution_plan.md)主要变更C3，为同一保存场景的Local／Global／Render三组建立一次编码器输出融合，以原生深度检验O4/O5；O2/O3复用，O6仍未达成。24份场景缓存和六项原训练预检已完成，已完成8/9拟合和960份新预测导出，暖查询成本完成，仅最后Render训练中，尚无最终方法效果。下方“仅设计”段落保留为授权前历史。

**最新提案（2026-11-12，仅设计，未设目标或执行）：** [rendered_feature_native1建议](learned_projection_results_analysis_20261112.md)复用当前场景/内容，在图像编码器输出处一次融合渲染特征，匹配Local／场景平均Global／空间Render。检验O4/O5的原生用途，主要修改C3；O2/O3保留，O6仍是未达成的核心目标。此提案没有取代下方已完成实现和阴性结论。

**当前完成（2026-11-12）：learned_projection_pe1有限验证已完成。** 三组各三种子2000步、864份新预测独立复算、成本及两图实际审阅完成。保留中心关系并学习投影输入后，G相对P/S主空间误差仍平均高3.02%/0.96%，三个种子均退步，区间跨零；S对P也无增量。各组相对同容量L的净用途未确立。旧固定G的完整输入空间代价修复17.08% [4.68,28.09]%，属于接口修复，未转为超过本轮强中心的收益。

结束当前投影标量输入在此接口上的映射/强度修补，不采用G/S，也不晋升P为可靠memory方法。来源传感器/共同有效性、静态给定相机、冻结骨干和旧数据是边界，非纯RGB或新确认；旧来源质量正证据和区域任务成果保留，无自动后继。下方按轮次保留历史，以本段为当前状态。

[完整报告](../results/gaussian_memory/learned_projection_pe1/report.md) · [固定数表](../results/gaussian_memory/learned_projection_pe1/report_tables.md) · [方法取舍](../results/gaussian_memory/learned_projection_pe1/method_decision.json)

**上一轮状态（2026-11-12）：source_geometry_control1有限验证已完成。** 48份来源状态、四组各三种子2000步（12次训练）、1152份适配预测及1728份干预全部完成；2880份新预测独立复算、三张最终图已实际审阅。来源质量改善有有限正证据，但投影PE增量和相对L的可靠净用途未建立，结束此固定诊断，不晋升G或自动开展后继任务。

P-sensor对P-pred的缺失ROI空间误差低4.27% [0.71,11.91]%，三个种子同向，说明中心PE读取器能利用来源几何及绑定的质量改善。G对P在pred/sensor条件主误差平均高6.62%/5.02%，区间跨零；主交互+0.000453，区间[−0.001950,+0.003349]，未建立质量解锁G优势。G-sensor对L平均高0.63%；P-sensor对L低4.18%但区间跨零。

完整输入ROI中，G对P在pred/sensor条件的空间误差高17.64%/19.25%，区间均支持增加。传感器源场景直接深度AbsRel平均低44.25%，但空间质量和特征重观察未全面改善，不能称精确oracle。两行共同有效性也使用传感器信息，均是额外来源信息的旧数据诊断，不是纯RGB方法成绩或新确认。

[完整报告](../results/gaussian_memory/source_geometry_control1/report.md) · [固定数表](../results/gaussian_memory/source_geometry_control1/report_tables.md) · [方法取舍](../results/gaussian_memory/source_geometry_control1/method_decision.json)

**上一轮状态（2026-11-12）：gaussian_scene_memory1的C1–C6有限实现与验证已完成。** 24份来源场景、L/T/C/P/G各三种子2000步（15次训练）、1440份适配预测、288份参照和864份固定干预全部完成；2592份预测独立复算，三张最终图已实际审阅。当前配置未建立可靠场景净用途或投影PE增量，结束此版接法，不晋升G，也不将负结果外推为所有3DGS memory不可行。全部为旧数据，没有新确认或自动后继任务。

[完整报告](../results/gaussian_memory/gaussian_scene_memory1/report.md) · [固定结果表](../results/gaussian_memory/gaussian_scene_memory1/report_tables.md) · [方法取舍](../results/gaussian_memory/gaussian_scene_memory1/method_decision.json)

**实施修复记录（已完成）：** 正式Gaussian拟合前修正AnySplat XYZW→gsplat WXYZ，归档旧派生状态并重建全部24份；抽样协方差最大相对差约6.16e−7。首T拟合前修复patch中心深度采样。十五次最终训练、保存重读和配对身份均在正确状态下核对；配置、假设与预算未按开发成绩调整。[修复记录](../results/gaussian_memory/gaussian_scene_memory1/failures/quaternion_order_v1/source_fix.json)。

更新：2026-11-12（Asia/Shanghai）。依据：用户提供的导师设计大纲，以及用户要求建立持续更新记录、防止偏移和重复实验的指示。

本文件是**制定或调整研究计划前必读的对应与尝试记录**。随计划、实现选择、实验结果和方法取舍更新；原始配置、完整指标和反例仍以各轮计划及报告为准。这里的“直接对应”表示设计关系，不表示已经实现或验证成功。尚未执行的方案必须明确标注。

## 1. 大纲与当前已检验方案逐项对应

核心目标：**为VGGT建立一份比较粗的、以3DGS为形式的公共场景记忆；当前视图从中提取相关信息，读取利用渲染和高斯参数，参数参与位置编码。**

最近完成的是[native_receiver_adaptation1](native_receiver_adaptation_execution_plan.md)，主要改C6；复用同一24份sensor保存场景及输入融合，三组各三种子共同训练完整原生接收/DPT。2880新预测与864继承开发结果及审阅已完成：相对Global和打乱有有限空间正证据，相对Local的可靠净用途仍未建立，该固定候选结束。此前来源质量、固定融合和投影PE的正负结果均保留；O6尚未由本轮检验。

公共场景实现来源为[粗3DGS场景记忆方案](gaussian_scene_memory_plan_20261111.md)，用户授权的C1–C6已完成实现、全部固定训练及审阅，结果未建立可靠净用途或PE增量。[实施记录](gaussian_scene_memory_execution_plan.md)。该实现不晋升为有效方法；`coarse_memory_pe1`是此前已完成的另一项粗K/V比较，不作为本方案前置门槛。

| 大纲编号 | 大纲中的意思 | 当前拟议方案的对应 | 对应关系与待验证部分 |
| --- | --- | --- | --- |
| O1 | VGGT是per-token，每个token是一个场景点 | 当前图像token作为查询，访问场景记忆 | 思想对应；“一个token就是一个三维场景点”是直观描述。实际主要图像token来自patch，另有相机等特殊token，不能要求每个token与独立世界点一一对应。 |
| O2 | 3DGS建立一个公共的场景表达 | 所有来源视图共同构建一份高斯场景，多个查询共享它 | **直接对应且已实现**；24份保存场景可供四查询共享，部分视角存在错位和覆盖损失，净用途未建立。 |
| O3 | 为VGGT建立memory，形式为比较coarse的3DGS | 保存中心、形状、透明度和绑定的视觉特征 | **直接对应**；2048个基元、1024维内容、FP16等是资源和实现选择，不是大纲要求。 |
| O4 | 每个view从3DGS场景表达提取当前view的相关信息 | 保存内容渲染到查询视角、编码器输出一次融合；冻结与完整原生接收/DPT适配已比较 | 两轮各九次拟合完成；完整适配R对Local主误差平均高1.37%、区间宽且跨零，整体可靠净用途未建立。对Global低22.89%和正确绑定依赖保留；训练优势未迁移为开发强对照优势。 |
| O5 | 读取过程用到3DGS的渲染和参数 | 实际alpha合成渲染1024维特征成为融合输入，Global共用覆盖 | 对应一种实现，非唯一要求；实现与绑定依赖已验证，完整适配R对Global空间误差低22.89% [4.32,39.45]%，但原生整体用途与O6仍未建立。 |
| O6 | 参数表示为PE | 中心、形状、透明度结合查询相机生成相对位置偏置 | 属于广义几何PE的**一种尝试**，不是已经确立的新RoPE，也不是将整份场景替换为一个PE向量。固定比较已完成，对同内容中心PE未建立可靠增量；这不否定其他PE实现。 |

O1的技术澄清依据[VGGT原文§3.2–3.3](https://arxiv.org/html/2503.11651v1)。O2–O6保留导师原始目的；不能因为某一种实现失败，自动修改这些目标。大纲并未规定在线更新、未知相机或端到端训练，它们目前是能力边界和后续开放问题，不能擅自说成已完成，也不能擅自当成用户要求放弃的目标。

## 2. 当前实现选择与临时简化

下列选择可以被有证据的后续计划调整；它们不是新增的大纲约束。任何调整须保留原选择的结果和变更理由。

| 选择编号／关联大纲 | 当前选择及理由 | 其他解释或实现 | 当前证据和边界 |
| --- | --- | --- | --- |
| C1／O2、O3 | 本轮复用source_geometry_control1/sensor的24份保存场景，沿用给定来源相机反投影和300步来源拟合 | AnySplat原始高斯、共享writer、纯前馈或增量构建 | **24份保存状态身份不变。** 前轮48份状态/12训练及来源质量改善证据保留；本轮不重建来源，也未获得可靠原生净用途。传感器和共同有效性特权仍非RGB成绩。 |
| C2／O3、O4 | 原1024维alpha贡献绑定内容，保存后多查询共用；本轮不重新写入 | 学习压缩、其他原生层内容、直接渲染特征 | **本轮三读法共用相同内容与有效性。** 每场景4,319,652字节张量；前轮特征重观察局限保留，不把维数当作准确对应，也不唯一归罪V。 |
| C3／O4、O5 | 单个2048→256→1024残差MLP在编码器输出处融合一次，Local/Global/Render同容量 | 原四层共享attention、其他接收接口 | 冻结行九次/1152新预测、共同适配行九次/2880新预测均完成；R对Local主误差分别高3.20%/1.37%，区间宽，当前两版均不采用。保留支持条件、绑定和Global对照的正证据；本轮不检验O6。 |
| C4／O6 | 同一8→16→4 MLP保留7维中心关系，第8维为零/球形/完整投影贡献 | 参数向量PE、Q/K变换、RoPE等 | **G对P/S主误差高3.02%/0.96%，三种子均退步、区间跨零。** 完整输入对旧固定G修复17.08%但没有中心增量；收束这类投影输入，不外推全部协方差/opacity无价值或所有强PE已比较。 |
| C5／O4、O5 | 给定来源/查询相机和共同尺度，在来源反投影中控制相机输入 | 预测相机后读取、联合定位 | **已完成的临时条件。** 传感器/共同有效性为额外诊断信息，保存场景仍非所有视角都准确；给定相机不能解释成完全消除了几何误差，不支持未知相机用途。 |
| C6／O2–O6 | 同一融合下共同训练全部24组frame/global块及原DPT，编码器及保存场景冻结；静态四查询共享 | 在线更新、共同学习构建/内容、其他接收训练范围 | **native_receiver_adaptation1已完成并收束。** 原生637,383,906参数、九次2000步；R−Local适配交互−0.000492且区间跨零，R−Global交互−0.005174且区间为负，未解锁强Local净用途。训练/开发差异、旧冻结/末四层LoRA和全部资源记录保留；旧小训练集及O6缺口仍在。 |

本轮已检验结构：来源图像→粗高斯特征场；当前视图token＋查询相机→几何引导的共享读取→VGGT原生几何输出。RGB、渲染特征和覆盖质量用于检查载体；主要取舍仍由记忆用途、PE增量、空间误差、整体偏差及成本共同决定。

**C3变更的形成记录（后已授权完成，非待执行提案）：** 来源图像编码器内容按查询视角直接渲染，在当前同一编码器输出处加入单个残差融合层，再运行全部原生层与DPT。C1/C2保存状态不变，C5/C6给定相机、传感器特权、静态和冻结条件仍为临时边界；C4投影标量修补保持收束，O6未来比较需要另行明确。固定Local／Global／Render三组同容量训练，检验空间内容是否提供超出共同适配和场景整体信息的原生用途。与prototype1、末层教师恢复和joint_write1独立粗头的实质区别，以及不同结果下的取舍，见[分析与建议](learned_projection_results_analysis_20261112.md)。

**本轮方案与最终状态：** [来源质量对照的形成分析](gaussian_scene_memory_analysis_20261112.md)随后获用户授权，已按[实施方案](source_geometry_control_plan_20261112.md)完成。改变C1的受控诊断未建立当前投影规则的可靠增量；原C1–C6和父轮正负结果保留。

**设计形成与实际完成：** [结果分析](source_geometry_results_analysis_20261112.md)形成时仅建议，随后用户授权[固定执行](learned_projection_pe_execution_plan.md)，现已完成。保留中心并给予投影学习机会的受控对照仍无可靠增量；相关旧阴性不改写，也不把这次实现说成首次可学习PE。

**C6变更形成记录（后已授权并完成）：** 保留保存场景、原内容和单次融合，三组同等共同微调全部原生frame/global交互层及原DPT；与当前仅训练融合的三组比较，检验接收训练条件是否改变场景净用途。主要服务O4/O5，O2/O3复用；不新增writer/门控，不将共同适配收益写成O6。区别于native_gaussian_pe1的末四层LoRA与joint_write1的独立粗头；12训练地点、给定相机/传感器/静态条件仍限制泛化。假设、竞争解释、九次新增拟合预算与结果分支见[分析及建议](rendered_feature_results_analysis_20261112.md)。 完整结果见本文件首段；不再是待执行计划。

## 3. 已有尝试与结果登记

以下是与当前选择直接相关的初始登记，**不是完整历史实验清单**。更早轮次及未列实验仍须检索[研究入口](research_current.md)、[证据边界](gaussian_memory_evidence_scope.md)和[历史索引](archive/2026-11-04/INDEX.md)；“本表没有”不等于“没有做过”。已有数据的后续重用不能改称新确认。

| 尝试／证据来源 | 对应大纲／选择 | 已做的尝试与结果 | 当前取舍与防重复提示 |
| --- | --- | --- | --- |
| [prototype1](../results/gaussian_memory/prototype1/report.md)，已完成 | O2–O5；C1、C2、C3 | 256槽、32维特征、共享writer、gsplat直接渲染特征；保存状态可独立多视角读取。六个未参与当轮优化的旧场景，学习几何组对固定几何组特征MSE低5.69%。 | 保留可重观察特征载体的有限正证据；单种子、小规模、已知相机，不是原生VGGT输出或准确几何证据。“加renderer”不是新工作。 |
| [joint_write1](../results/gaussian_memory/joint_write1/report.md)，已完成 | O2–O4、O6；C1、C4 | 共享可学习写入接回粗读出，已有相对强对照的有限用途；几何更新相对仅内容／冻结状态的额外用途未确立，空间细节有代价。 | 保留正负结果；后续不能把“可学习writer存在”与“学会准确几何”混为一谈。 |
| [content_center1](../results/gaussian_memory/content_center1/report.md)，已完成 | O3、O4、O6；C2、C4 | 固定几何只学习内容，51已用主地点Gaussian相对匹配Center整体log-MSE低5.17%；收益主要来自均值偏差，精细空间优势未建立。 | 是独立粗读出接口的有条件正证据；不能迁移为内部VGGT接法已成功，也不据此认定形状更新必需。 |
| [incremental_vggt1](../results/gaussian_memory/incremental_vggt1/report.md)，已完成 | O2–O4；C6 | 增量状态与原生VGGT平台已实现，覆盖增加未形成明确原生纠偏用途，保留方向差异与回访退步。 | 在线更新不是完全未探索；新更新方案须说明与该轮状态、输入及训练条件的实质区别。 |
| [bound_read1](../results/gaussian_memory/bound_read1/report.md)，已完成 | O4–O6；C3、C4 | 标准投影合成权重逐槽进入内部attention，硬支持读取；51非训练地点整体误差对内容C平均高6.28%，未建立可靠增量。 | 保留负结果，不晋升为方法；对应实现成立不等于用途成立。 |
| [soft_read1](../results/gaussian_memory/soft_read1/report.md)，已完成 | O4–O6；C3、C4 | 改为保留全局内容读取的软偏好；Soft对C整体误差平均高0.32%、区间跨零；缓解大部分硬读取损失，但未建立可靠新增用途。 | `log1p(M*w)`不是新公式；后续复用须说明改变了什么实质条件。 |
| [history_visibility_train1](../results/gaussian_memory/history_visibility_train1/report.md)，已完成 | O4–O6；C3、C6 | C/Soft/Center任务匹配训练，没有建立稳定历史净收益或形状增量，完整输入还存在代价。 | 固定Prefix接法的训练修补已收束；不因重命名或单次波动自动重开。 |
| [native_gaussian_pe1](../results/gaussian_memory/native_gaussian_pe1/report.md)，已完成 | O4、O6；C4、C6 | 未压缩原生内容、末四层匹配适配；共同ROI误差低13.87%，约99.95%下降来自均值偏差；完整高斯与中心／球形没有实用差异。 | 结束该固定接法的形状扩展；不将此负结果外推为所有3DGS memory无效或新场景方案的前置门禁。 |
| [coarse_memory_pe1报告](../results/gaussian_memory/coarse_memory_pe1/report.md)，已完成 | O3、O4、O6；C4 | 全层粗K/V、三组各三种子2000步、432份输出。256/1024槽高斯对中心空间降误差−0.204%／−0.114%，各三种子均未改善、区间跨零；球形亦无实用空间增量。共同原始收益主要伴随偏差减少，约1%完整输入AbsRel辅助线索保留。 | 结束此固定接法，不补种子、扫容量或修公式。共同压缩约17.94/5.16倍不属于形状收益，两遍查询未加速。不是拟议的可渲染统一特征场，不否定其他高斯PE；全部旧开发数据。 |
| [gaussian_scene_memory1报告](../results/gaussian_memory/gaussian_scene_memory1/report.md)／[实施记录](gaussian_scene_memory_execution_plan.md)，已完成 | O2–O6；C1–C6 | 24份真实粗3DGS与1024维绑定；L/T/C/P/G各三种子2000步，2592份预测独立复算，三图审阅。G对P/L/T主误差平均高6.97%/5.50%/5.37%，各区间跨零；C/P对L约1%的均值不稳定。正确绑定/相机优于破坏；全图AbsRel辅助均值收益保留，第三种子不稳定。 | 按原定取舍结束该固定组合，不采用G或自动晋升C/P，不补种子、改公式或自动启动新任务；已用数据、给定相机、静态来源、冻结骨干。不能唯一定位失败原因，不否定原始公共场景构想；直接特征渲染融合及其他PE未在该配置下比较。 |
| [source_geometry_control1报告](../results/gaussian_memory/source_geometry_control1/report.md)，已完成 | O2–O6；C1–C6 | 来源质量×P/G四组各三种子2000步，48场景、2880新预测独立复算及三图实际审阅。P-sensor对P-pred主误差低4.27% [0.71,11.91]%；G对P在pred/sensor条件均值高6.62%/5.02%，交互区间跨零。完整输入G对P空间代价有支持。 | 结束此固定诊断；保留来源质量可被利用的有限正证据，不晋升G或P-sensor为可靠RGB方法。共同传感器有效性、静态给定相机、冻结接收和旧数据是边界；非精确oracle，不唯一定位病因，无自动后继。 |
| [learned_projection_pe1报告](../results/gaussian_memory/learned_projection_pe1/report.md)，已完成 | O4–O6；改C4，复用O2/O3 | 三组各三种子2000步、864新预测独立复算和两图实际审阅。G对P/S主误差高3.02%/0.96%，三个种子均退步；相对L无可靠净用途。旧G完整输入空间代价修复17.08%，未形成中心增量。 | 结束当前投影标量映射/强度修补，不采用G/S。传感器特权、静态给定相机、冻结接收及旧数据边界保留；CPU搬运验证与梯度小差异透明记录，无自动后继或扫描。 |
| [rendered_feature_native1报告](../results/gaussian_memory/rendered_feature_native1/report.md)，已完成 | O4/O5；改C3，复用O2/O3 | 同一保存场景直接渲染1024维特征、编码器输出一次融合；三组各三种子2000步，1152新预测独立复算及两图实际审阅。R对Local主误差高3.20%，原始缺失/完整AbsRel高18.99%/17.15%；支持ROI对Global低13.60%，绑定依赖25.77%保留。 | 结束此固定内容＋融合候选，不采用Render/Global，不晋升Local；三组不是严格嵌套来源增量。区别于教师恢复和独立粗头，未验证O6、非RGB/新确认；静态给定相机/传感器/冻结边界保留，无自动后继。 |
| [native_receiver_adaptation1报告](../results/gaussian_memory/native_receiver_adaptation1/report.md)，已完成 | O4/O5；改C6，复用O2/O3 | 同一保存内容/融合，九次完整原生接收/DPT适配、2880新预测及864继承开发结果；R对Global低22.89%、对Local高1.37%且区间跨零，训练对Local低41.42%但未迁移。绑定和支持条件正证据保留。 | 结束本固定小数据候选，不采用R/Global、不晋升Local。R−Local交互不确定，R−Global交互有利；非O6、纯RGB或新确认，不成为所有PE研究的前置门禁，无自动后继。 |

各行只摘录影响取舍的结果；完整数值、不确定性、数据身份、成本和反例以链接报告为准。新结果出现时保留原结果和旧决定，增加后续取舍，不能覆盖成一直有效或一直无效。

建议形成时仅设计的[来源质量×P/G对照](gaussian_scene_memory_analysis_20261112.md)已随后授权并完成，见新增结果行；当时状态保留在按日期记录中。

## 4. 每次新计划必须做的核对

制定、修订或恢复研究计划时，先读本文件和`research_current.md`，再查与本次选择相关的旧计划／报告。在新计划中写一段简短的“**大纲对应核对**”，至少说明：

1. **服务哪个核心问题：** 引用O编号，区分直接要求、实现尝试和临时能力边界；说明本次改变的C选择。
2. **此前做过什么：** 引用相关尝试及结果；若复用旧做法，写清新的实质条件与信息价值。“换名称、补种子、换代理指标”本身不构成新思路。
3. **本轮怎样改变决定：** 给出假设、竞争解释、固定匹配比较，以及不同结果会保留、收窄或结束什么。实现正确、机制依赖、原生用途、独立确认分开陈述。

计划确定时同步更新第1、2节，并在第3节登记为“仅设计／待执行”；实际启动后改为“进行中”。出现会改变设计或结论的事实、完成、失败或主动收束时，更新状态、结果来源、适用范围和取舍。运行中的改动保留原因与版本，不能覆盖原计划。

此核对是短记录和阅读要求，**不是新的数值门禁、用户审批流程或每次修bug都要新增科学轮次的要求**。常规工程修复沿用现有版本化记录；不因所有辅助指标未赢而阻止有意义的主任务实验。

可复用的计划段落：

> **大纲对应核对（日期）：** 已核对[对应记录](research_outline_alignment.md)。本轮服务O…，保留直接要求…，尝试C…，临时条件为…。相关旧尝试为…，结果…；本次实质变化是…。固定比较…检验假设…；若结果…则…，若结果…则…。对应记录已同步；结果尚未产生／结果来源见…。

## 5. 防偏移与结果更新原则

- **主问题保持可见：** 场景记忆是否帮助VGGT，及高斯几何PE是否比匹配中心编码多提供价值。渲染、特征重建、寻址分数只提供对应层次证据，不能独自替代原生用途。
- **允许否定具体选择：** 可以得到“内容有用但PE无增量”“只在给定相机下有用”等有限结论；不靠堆模块或无限调公式维持某个假设。
- **区分复用和重复：** 有新条件和明确决策价值的复测可以执行；先说明为何旧结果不足以回答本轮问题。失败接口不成为永久禁令，也不因改名获得新的有效性。
- **变更可追溯：** 大纲若由用户澄清或修改，保留旧解释和变更原因；实现选择、临时简化不能反过来变成导师原始要求。

## 6. 维护记录

| 日期 | 更新与依据 |
| --- | --- |
| 2026-11-11 | 按用户要求建档：保存六项对应表、实施选择与临时条件、相关历史尝试及结果；将计划前核对和结果后更新写入仓库AGENTS与研究入口。既有实验配置未变，新场景方案未执行。 |
| 2026-11-11 | coarse_memory_pe1按固定配置完成，独立评分、成本复测及图表审阅后收束该粗K/V接法；登记主空间阴性、辅助正线索、偏差与存储/耗时边界。保留拟议场景方案仅设计状态，无自动后继。 |
| 2026-11-11 | 用户另行明确授权C1–C6实施，建立gaussian_scene_memory1目标，开始官方依赖与代码准备；无正式成绩。保留上一轮结束时“仅设计”的历史状态，当前状态更新为进行中。 |

2026-11-11实施进展：两个旧训练预检的五组零初始化深度与官方当前图像结果完全一致，四插入层均有有效梯度；峰值约5.6–5.8GiB。预检权重丢弃。高斯状态4,319,652字节；同预算T保存2095个token、4,318,215字节。固定配置见[plan.json](../results/gaussian_memory/gaussian_scene_memory1/plan.json)，原生检查见[native_integration_checks.json](../results/gaussian_memory/gaussian_scene_memory1/native_integration_checks.json)。这些是实现证据，不是用途或新确认。

实施修复登记：参照导出的目录创建问题已修正；T点中心使用nearest-exact源深度采样，首T拟合前修复旧nearest的patch角落偏差，原状态保留。Gaussian场景未改，L训练未受影响，未按开发成绩调整。属于常规实现修复，不是新路线或新确认。

固定场景预览已实际查看：前两个旧开发地点的同场景四查询渲染呈明显质量差异；fidelity_00部分视角严重模糊/几何失真，fidelity_01保留可辨认布局但仍有瑕疵。此为载体诊断，不是PE胜负；保留全部地点和固定比较，待原始/粗化前后及原生用途共同解释，不按预览改配置。[预览与实际审阅](../results/gaussian_memory/gaussian_scene_memory1/scene_preview/review.json)。

预定GT来源支持统计已完成：12旧开发地点、4查询、地点及查询等权下，固定缺失ROI约77.56%有效表面在至少一个完整来源视图中有传感器一致支持，全图约70.03%。这是输入机会，不能等同于高斯场景已正确保存了相同比例的内容；预览中的高alpha也不能替代保真。只按原计划作支持/无支持条件评价，未增加阈值或排除样本。[完整来源条件](../results/gaussian_memory/gaussian_scene_memory1/source_conditions.json)。

单种子中间控制记录（非最终结论）：20261110的T/L缺失ROI空间log-MSE为0.024817/0.024095，T平均高约2.99%，7/12地点改善；原始ROI误差也未改善。仅一个种子、旧开发数据，不给稳健性判断，不改变后续C/P/G或预算；完成三个种子后统一分析。[中间描述](../results/gaussian_memory/gaussian_scene_memory1/first_seed_L_T_description.json)。

2026-11-12首种子L/T/C中间记录：缺失ROI空间log-MSE为0.024095/0.024817/0.025921；C对L/T平均高7.58%/4.45%，两项均仅4/12地点改善；完整输入ROI的C也未改善。不是三种子稳定结论，继续固定P/G和全部种子；P/G即使优于C，仍须考察L/T净用途。[单种子描述](../results/gaussian_memory/gaussian_scene_memory1/first_seed_L_T_C_description.json)。

2026-11-12首种子G中间记录：缺失ROI空间log-MSE为0.026664，仍高于L/T/C的0.024095/0.024817/0.025921；原始G误差0.122552低于C但高于L。打乱绑定/错误相机的空间误差为0.033383/0.031579，表明固定模型依赖对应信息，不证明相对匹配强对照有净收益。训练后移除记忆的96份预测与原生当前图像输出逐像素完全一致，符合零残差接口；该移除同时移除了已训练适配器，不能替代L。仅单种子，P待完成，后续比较和预算不变。[描述及固定干预](../results/gaussian_memory/gaussian_scene_memory1/first_seed_L_T_C_G_description.json)。

2026-11-12首个五组配对种子已齐备并核对：缺失ROI空间L/T/C/P/G为0.024095/0.024817/0.025921/0.025872/0.026664。G对P主空间误差高3.06%，原始误差却低9.08%；未出现相对L/T的净用途。完整输入ROI的G对P空间误差低2.34%，但仍高于L/T，作为单种子辅助迹象保留。五组公共初值逐参数相同、样本顺序相同、均2000步/96导出；累计1056预测含固定参照和G干预已独立复算。仍缺其余两个种子，不给稳定结论，不改后续方案。[首组完整描述](../results/gaussian_memory/gaussian_scene_memory1/first_seed_five_arm_description.json)。


2026-11-12载体与成本检查完成（训练仍进行中）：24份保存状态及保真图的SHA已复核，前两个固定旧开发地点的原始GS/DC→2048粗化→300步来源拟合图已实际查看。12开发地点等权下，来源RGB MSE由原始0.107875／粗化初始0.247031降至拟合后0.042281；独立查询由0.106314／0.236901降至0.070632。原始参照是共同相机坐标下的DC渲染，不是官方完整SH质量。查询全patch特征MSE为0.022785，覆盖部分cos为0.918945；48个查询中5个整视角无覆盖，条件指标不含其未定义值，全patch及原生评价均保留全部地点。图像显示来源初始对齐/重建误差、粗化损失及拟合后的模糊并存，不能唯一归因或据此改配置。两旧训练片段的完整来源→保存状态约6.36/5.92秒，另有构建器/原生模型冷加载15.06/13.23秒；完整未缓存暖查询L/T/C/P/G中位约142/146/142/146/163ms。测量处于共享GPU且有波动，不作独占加速或精细速度排序主张。此为O2–O5/C1–C3载体与成本证据，不替代O4/O6原生用途和PE增量；15次固定训练继续。[保真汇总](../results/gaussian_memory/gaussian_scene_memory1/carrier/aggregate.json) · [实际图像审阅](../results/gaussian_memory/gaussian_scene_memory1/carrier/preview_review.json) · [完整成本](../results/gaussian_memory/gaussian_scene_memory1/carrier_and_cost_description.json)。


2026-11-12直接深度参照补充：288份固定参照的独立评分已汇总，其中直接渲染保留48个独立查询和全部12地点。有效传感器像素的alpha覆盖地点等权为86.69%，但5/48查询整视角无覆盖；覆盖不代表准确几何。直接深度全图空间log-MSE为0.233382，全覆盖条件部分为0.129165（43/48可估视角），缺失ROI为0.182354；完整输入原生当前图像全图/ROI为0.011272/0.013730。直接渲染原始全图误差3.435847含预定的未覆盖来源单位填值，覆盖部分为0.951333；AbsRel也完整保留，不据单一指标下结论。覆盖条件指标只提供载体诊断，与全图指标取样不同，不能宣称匹配提升；所有地点和查询仍进入正式L/T/C/P/G比较。该记录服务O2/O3/O5的几何载体，未改C1–C6、训练配置或预算。[完整参照与覆盖分母](../results/gaussian_memory/gaussian_scene_memory1/references/summary.json)。


2026-11-12训练与资源接续：第二种子T已完成2000步及96预测，累计7/15拟合、672适配预测；含参照及首G固定干预的1248预测已独立复算，最大指标差8.58e−13。GPU0外部任务增加显存后，本任务C主动退让并在GPU1从检查点续训。依据已完成拟合最大allocated8.95GiB、实际T进程约9.7GiB，启动空闲量由保守24,000MiB改为22GiB（22,528MiB）；仍最多两张外部共享卡、每进程12GiB上限，低于12GiB余量主动退让。仅替换CPU排队器，原C进程被接管而未重启；P在GPU5启动后余量约13.4GiB。科学配置、匹配样本和15次预算未改。此前阈值是资源实现选择，不是大纲或数值门禁。[变更及原代码](../results/gaussian_memory/gaussian_scene_memory1/runtime_measured_headroom_v1/reason.json)。完整三种子结论仍待完成。


2026-11-12中途验收：第二种子C完成2000步及全部96预测，累计8/15拟合、768适配预测；含固定参照及首G干预的1344预测全部独立复算，最大指标差8.58e−13。已完成8次拟合的日志均有连续1–2000步且损失/梯度有限；C资源退让后有两段保留时钟，976步附近停止并由900检查点恢复，额外76步/约67.86秒已单列，不计为已保留优化。P在GPU5的临时外部分配出现后也自动退让、从100检查点恢复，原中断日志保留；G已在GPU1实际接续运行。仍有7次拟合未完成，不作三种子方法结论。[独立复算进度](../results/gaussian_memory/gaussian_scene_memory1/partial_scoring/independent_recomputation.json) · [优化时长与中断工作](../results/gaussian_memory/gaussian_scene_memory1/training_summary.json)。


2026-11-12第二种子G收尾：2000步、96份原生评价及288份固定干预完整导出，第三种子L已实际接续；第二种子P仍在GPU5训练。累计9/15拟合、864适配预测、576固定干预，连同288参照共1728预测已独立复算，最大指标差8.58e−13；9次完成拟合的保留日志均有连续2000步且损失/梯度有限。固定报告表格生成逻辑已准备，但须等完整2592预测评分后才生成，不自动决定方法或关闭目标。还缺6次拟合及全部三种子结论，保留现有预算和配置。[执行状态](../results/gaussian_memory/gaussian_scene_memory1/execution_status.json) · [独立复算进度](../results/gaussian_memory/gaussian_scene_memory1/partial_scoring/independent_recomputation.json)。


2026-11-12第三种子L收尾：L三种子均完成2000步及各96预测；累计10/15拟合、960适配预测、576固定干预、288参照，共1824预测已独立复算，最大指标差8.58e−13。10次完成拟合均有连续1–2000步及有限损失/梯度。T第三种子已在GPU1实际启动，P第二种子继续在GPU5；其后仍有C/P/G第三种子，剩余5次拟合，目标尚未完成。不用已齐备L组与未齐备候选构成不匹配的最终比较。[当前执行状态](../results/gaussian_memory/gaussian_scene_memory1/execution_status.json) · [优化日志汇总](../results/gaussian_memory/gaussian_scene_memory1/training_summary.json)。


2026-11-12前两配对种子齐备：P第二种子完成2000步及96预测，前两种子L/T/C/P/G全部完成；第三种子L亦完成，T/C已在GPU1/5执行。累计11/15拟合、1056适配预测、576固定干预及288参照，共1920预测已独立复算，最大指标差8.58e−13。11次完成拟合的保留日志均连续2000步且损失/梯度有限；循环时长与被丢弃中断工作分开记录。仍缺第三种子T/C/P/G的4次拟合，不输出完整三种子结论或改配置。[执行状态](../results/gaussian_memory/gaussian_scene_memory1/execution_status.json) · [独立复算进度](../results/gaussian_memory/gaussian_scene_memory1/partial_scoring/independent_recomputation.json)。


2026-11-12第三种子T收尾：T三种子完成2000步及各96预测，L/T两个强匹配控制均已齐备；累计12/15拟合、1152适配预测、576固定干预和288参照，共2016预测已独立复算，最大指标差8.58e−13。12次完成拟合的日志均连续2000步、损失/梯度有限；第三种子C在GPU5继续，P已在GPU1实际接续，G待执行。仍缺C/P/G第三种子的3次拟合及完整三种子结果，固定配置不变。[执行状态](../results/gaussian_memory/gaussian_scene_memory1/execution_status.json) · [独立复算进度](../results/gaussian_memory/gaussian_scene_memory1/partial_scoring/independent_recomputation.json)。


2026-11-12两完整配对种子的中间描述（非最终结论）：严格只比较前两种子和全部12旧开发地点，缺失ROI空间log-MSE的L/T/C/P/G为0.024950/0.024961/0.025096/0.025004/0.026496。G对P空间误差高5.97%（两个种子分别高3.06%/9.08%），对L/T高6.19%/6.15%；G对P的原始误差却低8.87%，偏差平方低12.28%。完整输入ROI的单种子空间辅助正迹象未保持：G对P先低2.34%、后高2.72%，两种子合并高0.115%。完整输入全图AbsRel对L低1.17%（两个种子同向、仅5/12地点改善），对T/P低12.42%/11.67%，辅助正迹象保留；不以主指标失败抹去它，也不将其改成主终点。该记录服务O4/O6用途与PE增量的原定区分，无新区间、调参、样本排除或预算变化；第三种子C/P/G未齐备，不能称稳定结论或新确认。[两种子严格配对描述](../results/gaussian_memory/gaussian_scene_memory1/two_seed_five_arm_description.json)。


2026-11-12第三种子P/C完成：P先完成2000步和96预测，随后C也完整导出。L/T/C/P四组各三种子均已齐备，累计14/15拟合、1344适配预测、576固定干预及288参照，共2208预测已独立复算，最大指标差8.58e−13；14次完成拟合日志均有连续2000步且损失/梯度有限。最后G种子已在GPU1实际运行，队列无其他待启动任务；仍需G的2000步、96评价和288干预，以及全2592评分/配对区间、三张图实际审阅、报告/取舍和大纲同步，目标尚未完成。[执行状态](../results/gaussian_memory/gaussian_scene_memory1/execution_status.json) · [优化与中断工作](../results/gaussian_memory/gaussian_scene_memory1/training_summary.json)。


2026-11-12完整初始化与采样核对：15份实际保存的初始化均已读取，三种子各自五组的公共参数逐张量完全相同，T/P额外192参数相同；固定采样顺序重新生成后，与14次完成拟合的逐步实际样本记录全部一致，最后G仅核对其已写入前缀。96份锁定输入清单SHA保持一致。训练源码相对修复后锁定版本的差异已用原文重建哈希确认：仅增加中断日志文件名冲突处理，模型、优化器、损失、采样及检查点逻辑未改。这是O4/O6匹配比较的身份依据，不是效果或全实验完成证明；G结束后仍须重跑补齐其2000步。[配对身份核对](../results/gaussian_memory/gaussian_scene_memory1/paired_controls_identity.json) · [训练源码差异](../results/gaussian_memory/gaussian_scene_memory1/training_source_change_review.json)。


2026-11-12工作报告整理：已将本轮方法、固定比较、临时范围、场景保真和成本写入[工作报告](../results/gaussian_memory/gaussian_scene_memory1/report.md)。报告首段明确“尚未完成”，最终三种子结果、三图实际审阅与方法取舍仍待最后G。成本边界已明确为预处理RGB张量输入的构建/模型计算，图像JPEG解码、预处理及冷加载不混入暖计算，不宣称完整原始照片I/O加速。这是已有结果的文档整理，不改方案或增加实验。

## 2026-11-12最终结果、对应与取舍

**当前状态（2026-11-12）：gaussian_scene_memory1的C1–C6有限实现与验证已完成。** 24份来源场景、L/T/C/P/G各三种子2000步（15次训练）、1440份适配预测、288份参照和864份固定干预全部完成；2592份预测独立复算，三张最终图已实际审阅。当前配置未建立可靠场景净用途或投影PE增量，结束此版接法，不晋升G，也不将负结果外推为所有3DGS memory不可行。全部为旧数据，没有新确认或自动后继任务。

主终点缺失ROI空间log-MSE的L/T/C/P/G为0.025027/0.025058/0.024764/0.024685/0.026404。G相对P/L/T误差平均高6.97%/5.50%/5.37%，三个种子相对P均退步，但地点重采样区间跨零，不能断言总体必然有害。P/C对L平均低1.37%/1.05%，区间跨零且种子方向不一，也不晋升为可靠方法。正确G对打乱绑定/错误相机的主误差低21.77%/16.38%，区间为正，支持对应关系被使用；这不等于超过强匹配对照。

前两种子G对P原始误差低8.87%的迹象在第三种子翻转，三种子合并反而高2.08%。完整输入全图AbsRel对P/T仍平均低6.04%/6.46%，但区间跨零且第三种子退步；对L仅低0.032%。保留辅助正结果，不改主终点。粗场景部分视角有错位、模糊与覆盖缺失，原始初始化也有损失，不能唯一归因writer、特征绑定、PE或接收网络。

[完整报告](../results/gaussian_memory/gaussian_scene_memory1/report.md) · [固定结果表](../results/gaussian_memory/gaussian_scene_memory1/report_tables.md) · [方法取舍](../results/gaussian_memory/gaussian_scene_memory1/method_decision.json)

本轮服务O2–O6：C1的AnySplat初始化与300步来源拟合、C2的实际alpha写入1024维内容、C3的共享内部软读取、C4的投影PE均完成实现与固定比较；C5给定相机、C6静态来源和冻结骨干保持临时范围，不成为大纲永久要求。未比较直接渲染特征融合，未完成未知相机、在线融合、全骨干联合训练或未用地点确认。依据原定取舍结束本固定组合，不自动重启旧接法或新增扫描。

状态预算为4,319,652字节，T为4,318,215字节/2095 tokens；相对官方历史状态的压缩是共同存储收益，不是PE收益。两旧训练片段从预处理来源RGB到保存GS及T控制约6.36/5.92秒；暖模型查询L/T/C/P/G中位142/146/142/146/163ms，共享GPU不支持独占速度结论。十五次保留优化循环共8.226643 worker小时，另有0.137577小时中断未保存工作；峰值8.951GiB。全部成本边界与失败记录见报告。

此前第6节带日期的中间进度和两种子数值保留为历史记录；不能覆盖本节完整三种子结论。


## 2026-11-12来源几何对照获得授权

用户明确“设为目标并执行”，已建立source_geometry_control1有限目标，见[实施方案](source_geometry_control_plan_20261112.md)。对应O2–O6，暂改C1，保留C2–C6主体及临时能力边界。先前仅建议状态保留为历史；当前状态是进行中（两个旧训练片段构建预检），尚无新原生效果。实质变化和不同结果决定见方案；不改写任何已完成轮次。


2026-11-12固定训练启动：source_geometry_control1两个原训练片段的8个条件通过，唯一配置已锁定；48份来源场景、原维度绑定与查询缓存完成身份核对。四组首种子已在GPU0/1/4/5实际优化，初始公共参数与原L逐张量相同、样本轨迹配对，峰值约6.5–8.9GiB。12次拟合及同步质量/成本/交互评价继续，尚无最终方法结论。[运行状态](../results/gaussian_memory/source_geometry_control1/execution_status.json)。


首个完整配对种子20261110的中间描述（不作最终取舍）：四组均完成2000步、96评价，两个G各288干预，合计960份新预测独立复算最大差2.50e−15。缺失ROI空间log-MSE的P-pred/G-pred/P-sensor/G-sensor为0.024935/0.026492/0.024355/0.024069；G对P从预测条件高6.25%变为传感器条件低1.17%，主交互绝对值+0.00184265。但G-sensor相对同容量无历史L仅低约0.11%，尚无净用途结论。没有置信区间或新确认，另外两种子与全部预定比较保持不变。[单种子描述](../results/gaussian_memory/source_geometry_control1/first_seed_four_arm_description.json)。


两完整配对种子的中间结果（20261110/11，非最终结论）：8次2000步、768适配与1152干预输出完成，1920份新预测独立复算最大差2.50e−15。主空间误差P-pred/G-pred/P-sensor/G-sensor为0.025563/0.026526/0.024386/0.024880；G对P在pred条件平均高3.76%，sensor条件平均高2.03%。传感器条件逐种子降误差为+1.17%/−5.22%，主交互绝对值逐种子+0.00184265/−0.000906827，均值+0.000467910。G-sensor对L均值仅低0.28%，仅5/12地点改善。不以首种子正迹象晋升，不因第二种子翻转改变第三种子或加入新公式；无新区间、无新确认。[两种子描述](../results/gaussian_memory/source_geometry_control1/two_seed_four_arm_description.json)。


## 2026-11-12完整三种子与方法取舍

**当前状态（2026-11-12）：source_geometry_control1有限验证已完成。** 48份来源状态、四组各三种子2000步（12次训练）、1152份适配预测及1728份干预全部完成；2880份新预测独立复算、三张最终图已实际审阅。来源质量改善有有限正证据，但投影PE增量和相对L的可靠净用途未建立，结束此固定诊断，不晋升G或自动开展后继任务。

P-sensor对P-pred的缺失ROI空间误差低4.27% [0.71,11.91]%，三个种子同向，说明中心PE读取器能利用来源几何及绑定的质量改善。G对P在pred/sensor条件主误差平均高6.62%/5.02%，区间跨零；主交互+0.000453，区间[−0.001950,+0.003349]，未建立质量解锁G优势。G-sensor对L平均高0.63%；P-sensor对L低4.18%但区间跨零。

完整输入ROI中，G对P在pred/sensor条件的空间误差高17.64%/19.25%，区间均支持增加。传感器源场景直接深度AbsRel平均低44.25%，但空间质量和特征重观察未全面改善，不能称精确oracle。两行共同有效性也使用传感器信息，均是额外来源信息的旧数据诊断，不是纯RGB方法成绩或新确认。

[完整报告](../results/gaussian_memory/source_geometry_control1/report.md) · [固定数表](../results/gaussian_memory/source_geometry_control1/report_tables.md) · [方法取舍](../results/gaussian_memory/source_geometry_control1/method_decision.json)

本轮服务O2–O6、控制C1并保留C2–C6主体；原实施选择和临时条件没有变成导师永久要求。已有raw_history_information1、address_intervention1、native_gaussian_pe1、coarse_memory_pe1和gaussian_scene_memory1的正负结果均保留。没有补种子、扫温度或新增任务；有限目标结束不等于总体有效方法目标达成。


2026-11-12 learned_projection_pe1首配对种子中间描述：主空间误差P/S/G=0.02435180/0.02459225/0.02509068，G对P高3.03%，没有新区间或最终效果结论。新增投影列已更新，但仅有一个完整配对种子，其余任务继续，配置不变。[中间记录](../results/gaussian_memory/learned_projection_pe1/first_seed_description.json)。

2026-11-12 learned_projection_pe1资源实现更新：全卡不足旧余量阈值时，原两个训练片段上验证精确CPU缓存逐样本搬运，四条件输入和前向相同、梯度最大差6.10e−5，预检峰值5.796GiB。使用7GiB上限恢复最后两项，模型/内容/精度/样本/预算不变；这不改变C4比较或O2–O6对应，不新增模型模块或新科学轮次，不把原22GiB工程余量当作研究前提。[实现记录](../results/gaussian_memory/learned_projection_pe1/memory_staging/implementation_record.json)。

## learned_projection_pe1最终对应与取舍（2026-11-12）

**当前完成（2026-11-12）：learned_projection_pe1有限验证已完成。** 三组各三种子2000步、864份新预测独立复算、成本及两图实际审阅完成。保留中心关系并学习投影输入后，G相对P/S主空间误差仍平均高3.02%/0.96%，三个种子均退步，区间跨零；S对P也无增量。各组相对同容量L的净用途未确立。旧固定G的完整输入空间代价修复17.08% [4.68,28.09]%，属于接口修复，未转为超过本轮强中心的收益。

结束当前投影标量输入在此接口上的映射/强度修补，不采用G/S，也不晋升P为可靠memory方法。来源传感器/共同有效性、静态给定相机、冻结骨干和旧数据是边界，非纯RGB或新确认；旧来源质量正证据和区域任务成果保留，无自动后继。下方按轮次保留历史，以本段为当前状态。

[完整报告](../results/gaussian_memory/learned_projection_pe1/report.md) · [固定数表](../results/gaussian_memory/learned_projection_pe1/report_tables.md) · [方法取舍](../results/gaussian_memory/learned_projection_pe1/method_decision.json)

O2/O3保存场景和内容身份成立，O4读取实现及O5渲染贡献输入成立；O6的新增用途和相对L的净收益没有建立。只收束本次C4与当前接收接口的投影标量路线，不修改导师核心目标。旧source_geometry_control1、bound/soft、native/coarse及区域正成果均保留。


## rendered_feature_native1首种子描述（2026-11-12，非最终结论）

四次2000步、384适配及96固定绑定打乱预测完成，480份独立复算最大差1.32e−15。首种子Local/Global/Render主空间误差0.025018/0.027106/0.025587，R对Local高2.27%、对Global低5.60%；正确绑定对打乱低25.93%。缺失原始误差和完整全图AbsRel对Local分别高19.52%/15.48%。依赖性不替代O4净用途，也不证明O6；没有区间或新确认，余下五拟合保持不变，第二Global/Render在GPU1/7执行。[单种子来源](../results/gaussian_memory/rendered_feature_native1/first_seed_description.json)。


## rendered_feature_native1两种子描述（2026-11-12，非最终结论）

六次2000步、576适配及192固定绑定打乱预测完成，768份独立复算最大差1.32e−15。前两种子Local/Global/Render主误差0.024986/0.027384/0.025646，R对Local高2.64%、逐种子均退步，对Global低6.35%、逐种子均改善；正确绑定对打乱低26.09%。原始缺失误差及完整全图AbsRel对Local高16.86%/15.98%。Local的附加输入是当前全局均值，R是渲染来源内容，属于同容量不同上下文选择，不能单独证明来源信息的条件边际价值为零。没有新区间/新确认或最终采用决定；第三种子固定执行。[描述](../results/gaussian_memory/rendered_feature_native1/two_seed_description.json)。

## rendered_feature_native1最终对应与取舍（2026-11-12）

**最新完成（2026-11-12）：rendered_feature_native1有限验证完成。** Local／Global／Render各三个配对种子、各2000步，共九次训练；864份适配与288份绑定打乱预测、全量独立复算、成本测量和两图实际审阅完成。Render相对Local的缺失ROI空间误差高3.20%，三个种子均退步，地点/种子区间仍宽且跨零；原始缺失误差与完整全图AbsRel分别高18.99%和17.15%，区间支持增加。不采用本版Render或Global，也不把Local晋升为高斯方法，结束固定保存内容＋单次输入融合候选。

保留有限正证据：正确绑定比打乱的主空间误差低25.77% [15.86,37.59]%；来源一致支持ROI内Render比Global低13.60% [1.81,25.75]%，但相对Local的9.53%均值收益区间跨零。依赖与条件线索不等于整体净用途。三组是同容量的附加上下文选择，不是完整Local再追加来源的严格嵌套比较，不能据此宣布来源条件信息为零。

本轮实现O4/O5的一种读取接法，复用O2/O3；O4可靠净用途未建立，O6新增PE价值未在本轮检验，原生2D RoPE保持。全部为旧数据、来源传感器/共同有效性、给定相机、静态来源和冻结骨干/DPT条件，非纯RGB或新确认。总体精炼有效方案目标仍未达成，本有限目标不自动启动后继研究；所有历史正负结果保留。

[完整报告](../results/gaussian_memory/rendered_feature_native1/report.md) · [固定数表](../results/gaussian_memory/rendered_feature_native1/report_tables.md) · [方法取舍](../results/gaussian_memory/rendered_feature_native1/method_decision.json)

本轮的实质变化是同层一次特征融合及完整原生深度损失，而非新PE；负结果不唯一定位writer、内容、冻结接收或训练规模。几何仅经渲染权重影响输入，没有独立米制信息通道，是待研究的表示边界，未被证明是唯一病因。surface1、geometry_transport1已有深度V/独立输出尝试，不能将同一思路重命名为新方案。后续如调整路线，应明确相对于这些尝试的新条件与可改变的决定；本次不自动启动。

## native_receiver_adaptation1授权与对应（2026-11-12）

用户明确“设为目标并执行”，已建立有限目标，按[固定执行方案](native_receiver_adaptation_execution_plan.md)落实C6的原生接收网络共同学习。核心仍为O2/O3共享粗3DGS、O4/O5视图读取用途及未解决O6；本次不新增writer/门控或替代头。C1/C2/C3保存内容及融合保持，C4投影修补关闭；静态给定相机/传感器/旧小训练集是临时范围。此前输入融合、末四层LoRA、来源质量及共享writer的正负结果保留；新的信息价值是比较完整原生接收与DPT学习后，场景对匹配Local/Global的相对净用途是否改变。


### 正式资源调度更新

首正式Global在12步资源让出时保存了完整模型与双优化器并恢复，实际采样前缀与原冻结行一致、首步损失一致。随后观测到共享卡约2–3GiB短时波动导致频繁迁移，已保留11GiB进程上限与17GiB启动准入，将运行时让出阈值调整为4GiB实际空闲，最多两张共享卡；低于阈值仅让出自身进程。该阈值属于资源实现选择，非用户科学约束；原6GiB记录和变更依据保留，科学源码、精度、实际样本及步数不变。[资源变更记录](../results/gaussian_memory/native_receiver_adaptation1/resource_margin_revision/record.json)。
