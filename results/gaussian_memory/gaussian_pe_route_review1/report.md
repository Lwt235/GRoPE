# gaussian_pe_route_review1：四项原文与作者源码事实交接

2026-11-13。有限静态核查完成，待原 Astra `/root/research_analysis_astra` 评价机制、实质区别及至多一个候选／无候选。本文不给方法排序、不选后继。四项均交出已核实事实及明确 unknown；外部请求达到48次后停止联网。未运行 GPU、CUDA 探测、模型推理、训练或渲染，未下载权重／数据，未安装依赖或 clone 仓库。

目标仍是为 VGGT 建立精炼公共粗3DGS记忆，查清参数PE超过强中心几何的可靠用途及范围。此轮只补 O6 问题定义与 O2–O5 状态／读取事实，没有新效果分数、独立确认或总体目标达成。历史 writer／查询PE阴性、Render片段内正证据及跨地点不确定性全部保留。

完整逐项数据流、消融条件、工程分支和未知项见 [mechanism_facts.json](mechanism_facts.json)；[sources.json](sources.json) 给出每次URL、固定commit、快照SHA256、文件行数及119个静态符号位置。所有公式和表号均对应所读固定 arXiv v1；不将未读正式版当成已核验全文。[逐请求账](retrieval_log.jsonl) 包括失败、重定向和浏览器四项读取；浏览器显示结果与完整HTTP快照分别保存。

## 书目及作者实现身份

|工作|已核实身份|固定代码身份／边界|
|---|---|---|
|[GaussianRenderFormer](https://arxiv.org/html/2611.05593v1)|Shahaf Valencia Levy、Sagie Benaim，2026-11-03预印本；论文→[作者项目](https://svlwoof.github.io/gaussianrenderformer/)→仓库，README反向链接一致|[SVLwoof/gaussianformer](https://github.com/SVLwoof/gaussianformer/tree/fb8cd4daf5b1fc0ea4ce9b72071701ecd4a0e670)，`fb8cd4d…`；仓库名不同但链接链明确|
|[GPSToken](https://arxiv.org/html/2509.01109v1)|Zhengqiang Zhang、Rongyuan Wu、Lingchen Sun、Lei Zhang，2025-09-01v1；[NeurIPS2025官方条目](https://neurips.cc/virtual/2025/loc/san-diego/poster/119331)与论文／仓库身份吻合|[xtudbxk/GPSToken](https://github.com/xtudbxk/GPSToken/tree/2f34eb0914556659a0775a99d36fb56c6c07ce07)，`2f34eb0…`；是包含后来更新的代码，不能自动称v1原训练源码|
|[SplatFormer](https://arxiv.org/html/2411.06390v1)|Yutong Chen、Marko Mihajlovic、Xiyi Chen、Yiming Wang、Sergey Prokudin、Siyu Tang，2024-11-10v1；论文→[作者项目](https://sergeyprokudin.github.io/splatformer/)→仓库互链；作者称ICLR2025|[ChenYutongTHU/SplatFormer](https://github.com/ChenYutongTHU/SplatFormer/tree/446ffb5dd1c35b4b8f94953a22046bde5714a094)，`446ffb5…`；作者Pointcept子模块锁`c4aa232…`。OpenReview官方forum/PDF均转挑战页，正式版内容及会议归属没有独立官方页面核验，作者归属证据与该unknown分列|
|[JSRE](https://arxiv.org/html/2601.02339v1)|Jingming He、Chongyi Li、Shiqi Wang、Sam Kwong；[CVF官方记录](https://openaccess.thecvf.com/content/ICCV2025/html/He_Joint_Semantic_and_Rendering_Enhancements_in_3D_Gaussian_Modeling_with_ICCV_2025_paper.html)确认ICCV2025、28354–28363页；arXiv在2026-01-05上线|[fly519/JSRE](https://github.com/fly519/JSRE/tree/40a5bb4c61a09ae4ef9267f54db715ceeb423e9b)，`40a5bb4…`，仅README／图片。README自称官方、题名作者一致，但未找到作者原文／官方页面反向仓库链接；作者归属仍部分未核实。无模型代码，代码链接为占位地址|

## GaussianRenderFormer：普通属性、中心RoPE与footprint key分支可分定位

每基元14维为中心3、正scale3、WXYZ quaternion4、RGB3、opacity1，全部经14→768映射进入普通token。12层场景encoder使用中心3D RoPE；六层view decoder的query来自相机ray patch，Gaussian key还用相机坐标3D中心和投影中心2D RoPE。scale／rotation也已随普通token进入K/V。footprint另算相机投影协方差，提取两轴log标准差与双角方向四量，经每层零初始化Linear4→768加到K，发生在K归一化和RoPE之前；keys-only不直接改V。[模型入口](https://github.com/SVLwoof/gaussianformer/blob/fb8cd4daf5b1fc0ea4ce9b72071701ecd4a0e670/gaussianformer/models/gaussianformer.py#L43)、[footprint](https://github.com/SVLwoof/gaussianformer/blob/fb8cd4daf5b1fc0ea4ce9b72071701ecd4a0e670/gaussianformer/models/view_transformer.py#L53)、[注入顺序](https://github.com/SVLwoof/gaussianformer/blob/fb8cd4daf5b1fc0ea4ce9b72071701ecd4a0e670/gaussianformer/layers/attention.py#L57)。

输出是DPT生成RGB，输出路径没有GS rasterizer。普通属性已含形状；footprint是额外显式投影形状key条件，不应把它改称协方差不确定性或形状RoPE。论文称场景编码每对象一次；源码只证明一次forward内多视角共享encoder结果，公开pipeline每次调用仍重新forward，没有所读文件可证的永久保存／重载cache接口。

原文§6.2/Table4在1000对象harness上报告完整34.42dB、无footprint32.71dB，差−1.71±0.07dB为单次训练的对象bootstrap；只对完整recipe另跑一个seed。无footprint保留全属性token及中心RoPE，但少了23,040个分支参数；没有检出等容量强中心邻域替代。它支持该作者条件下的分支用途事实，不是本项目O6收益或新确认。正式配方、harness、K+V、2D-RoPE与patch大小变体分别保留，未混成一个消融。[原文§4、§6.2](https://arxiv.org/html/2611.05593v1)。

工程上可定位的最小单元是`footprint`→`key_embed`→`k_add`。同信息中心对照须共享全14维普通内容、原3D/2D中心、相机、V和decoder，提供中心可得的邻域／多尺度几何并匹配位置分支容量及初始化；单独关分支不足以构成该强对照。源码默认关闭这些开关且patch8，论文最终是patch4加2D／footprint；未下载checkpoint确认其实际配置。Python3.12/PyTorch2.9等作者依赖与旧GLIBC本机兼容性未测试。编码和逐view投影／decoder成本可按调用图分列，未推算本地VGGT时延。[配置](https://github.com/SVLwoof/gaussianformer/blob/fb8cd4daf5b1fc0ea4ce9b72071701ecd4a0e670/gaussianformer/models/config.py#L6)。

## GPSToken：二维区域参数与内容共同形成token

输入是单张图像、熵划分区域和五维`[sigma_x,sigma_y,rho,x,y]`。参数经MLP加到可学习token；同一初始区域的RoIAlign内容在每block加入，标准self-attention用归一化后的同一状态作Q/K/V。没有所读源码中的独立协方差bias／RoPE。独立输出头预测五维参数残差与texture，随后二维有界splatting成feature map再解码图像；没有3D相机、独立opacity或世界坐标GS场。[编码](https://github.com/xtudbxk/GPSToken/blob/2f34eb0914556659a0775a99d36fb56c6c07ce07/models/gpstoken.py#L58)、[attention](https://github.com/xtudbxk/GPSToken/blob/2f34eb0914556659a0775a99d36fb56c6c07ce07/models/attnblock.py#L21)。

论文AppendixE/Table6比较无自适应、仅refinement、完整init+refinement；PSNR23.52／24.00／24.06与recFID1.02／0.81／0.65是完整tokenizer成绩，变化同时涉及分配／初值／状态学习，不是形状PE独立增量。未识别该表的配对初始化、参数量相等、训练重复或区间。当前代码的inverse-sigmoid参数更新与论文Eq5直接残差写法不完全同文；spread初始化／支持截断常量也有版本或单位解释缺口，原定位保留，未替作者修公式。[原文§3.2、AppendixB/E](https://arxiv.org/html/2509.01109v1)。

M128配置给128槽、16内容维、宽512、30块、ROI8×8、64×64渲染。可将单图latent保存后decode；这没有建立共享多视角3D场状态。若要问PE增量，需先明确共同完整五维普通内容和ROI／renderer，再划出独立位置分支；直接删sigma/rho会改普通内容、Q/K/V、输出几何及渲染，不能称同信息PE对照。导入其CUDA wrapper会编译内核，本轮只读文本，未import。[配置](https://github.com/xtudbxk/GPSToken/blob/2f34eb0914556659a0775a99d36fb56c6c07ce07/configs/gpstoken_m128.yaml)、[wrapper](https://github.com/xtudbxk/GPSToken/blob/2f34eb0914556659a0775a99d36fb56c6c07ce07/gscuda/gswrapper.py#L12)。

## SplatFormer：真实GS属性细化，位置组织来自中心网格

`FeaturePredictor`将means、log-scale、opacity参数、quaternion、SH拼成普通feature；中心另成为`coord/grid_coord`，用于Z／Hilbert序列化、池化与稀疏卷积CPE。作者锁定Pointcept子模块证明QKV由`point.feat`线性映射，当前wrapper硬置`enable_rpe=False`，不应因使用Point Transformer便推定Σ是PE。层级输出与原属性concat，经零初始化属性head预测residual，再加回GS；refiner forward没有query相机输入，训练／渲染另传相机。[属性路径](https://github.com/ChenYutongTHU/SplatFormer/blob/446ffb5dd1c35b4b8f94953a22046bde5714a094/models/feature_predictor.py#L125)、[wrapper](https://github.com/ChenYutongTHU/SplatFormer/blob/446ffb5dd1c35b4b8f94953a22046bde5714a094/models/pointtransformer_v3.py#L120)、[实际attention/CPE](https://github.com/ChenYutongTHU/Pointcept/blob/c4aa232bfbc24dc0f3d6699dff03632f9d0ff595/pointcept/models/point_transformer_v3/point_transformer_v3m1_base.py#L172)。

初始GS拟合、共享refiner和多query rasterization是不同阶段；源码允许一次refine结果渲染多view，未验证独立永久保存协议。原文Table4的PTv3直接输出21.36dB、Mink残差22.67、PTv3残差23.06比较backbone／预测目标；Appendix对比2D／3D监督。没有隔离Σ-PE的强中心或球形消融；不能把整个系统成绩归给PE。[原文§4、Table4、AppendixB/C](https://arxiv.org/html/2411.06390v1)。

同信息PE对照必须保留全属性普通输入、原属性skip、相同residual heads、中心网格层级、renderer与监督，再另定位参数位置分支；去掉shape普通输入或禁用几何细化会改实验问题。代码有可选pseudo-target预训练，实际各run的配置覆盖未全读；不声称所有作者run只受2D监督。所读gin与论文MLP文字有层数边界，正式checkpoint架构未知。gsplat／Pointcept／spconv等依赖未安装或验证本机兼容性。

## JSRE：形状图描述、中心PE、传播与状态管理须分列

固定arXiv§3.1公式给出`M=(I+beta*Sigma)^-1`、双端平均度量边权、Laplacian与多方向谱／Chebyshev描述；然后`z=f(G)+PE(c)`映射到Q/K/V。作者显式称PE的是中心项；Σ描述进入普通token内容及图构造，另在Eq8–10的Mahalanobis传播／门控中使用。opacity在GS渲染语义中，但没有已核实的直接opacity→QKV代码路径。[原文Eq1–10](https://arxiv.org/html/2601.02339v1)。

关键状态边界是原文先按当前camera frustum选择局部点，再FPS／半径邻域编码；不能自动把这部分称query-independent永久writer。GS场与跨场景projection-pattern basis也不是同一状态，后者保存／重载用于新scene。语义引导增删GS与SH调整会改状态和renderer，不是单一PE。[原文§3.1、§3.2、§3.3](https://arxiv.org/html/2601.02339v1)。

Table5的noAGCD以MLP替代shape描述、noTLE以MLP替代Transformer；完整/noAGCD/noTLE的mIoU50.6/48.3/49.2及PSNR29.86/29.63/29.78是模块消融，混合内容／容量／传播／学习条件。FPS比较还关闭semantic branch，不能记为全图编码成本。缺少配对强中心PE隔离及重复／区间信息不机械否决该工作；它们是未获证据。[原文§4.3](https://arxiv.org/html/2601.02339v1)。

完整非截断仓库tree没有实现代码；README的July2026预期发布时间不等于已发布。公式可描述的工程单元为图构造、每方向谱分解、`Q=J(D+1)`描述、同宽中心PE、局部Transformer和传播；实际J/D/K、超参、solver、gradient、cache和调用频率均unknown。CVF正式PDF响应超过本轮单文件4MiB保护上限，未保存／阅读、不重试；官方HTML仅验证书目，正式版公式差异未知。同信息中心反事实需把普通shape信息、传播、门控、拓扑／SH与监督共同保留，分别定位图内容和位置组织，不能直接把删descriptor称PE效果。

## 与锁定本地历史的事实对应

|已完成尝试|本轮可核实的同／异与限制|
|---|---|
|[bound_read1](../bound_read1/report.md)、[soft_read1](../soft_read1/report.md)、[learned_projection_pe1](../learned_projection_pe1/report.md)|已有硬／软合成bias及可学习投影控制，未建立可靠中心增量。GRF的key向量注入与这些逐query标量bias在代码形式上不同；是否值得新问题由Astra判断|
|[native_input_pe1](../native_input_pe1/report.md)、[native_render_parameter_pe1](../native_render_parameter_pe1/report.md)|已有单次输入及保留Render的原生Q/K参数PE；空间主阴性和raw辅助线索均保留。GRF不能只因“注入K”被称本地未试过参数attention|
|[native_memory_context_pe1](../native_memory_context_pe1/report.md)|已有query-independent关系writer，普通内容包含完整cov且P有强中心邻域；没有可靠G/P增量。GRF场景encoder与SplatFormer共享refiner不能因“一次写入”本身被称首次；层级／RGB监督／输出状态的不同是事实，科学意义待评|
|[joint_write1](../joint_write1/report.md)、[surface1](../surface1/report.md)、[geometry_transport1](../geometry_transport1/report.md)|已试共享几何／内容学习、显式／latent读取；joint_write的粗raw用途正线索和空间代价保留。GPSToken的联合形成与SplatFormer属性残差不是自动新问题|
|[gaussian_scene_memory1](../gaussian_scene_memory1/report.md)、[source_geometry_control1](../source_geometry_control1/report.md)、[native_gaussian_pe1](../native_gaussian_pe1/report.md)、[coarse_memory_pe1](../coarse_memory_pe1/report.md)|已有真实GS、来源质量、原生适配与粗记忆控制；共同标定／来源用途不等于PE增量，不能以“真实GS”或“更好初始化”重命名旧路线|
|[native_render_factorization1](../native_render_factorization1/report.md)、[native_covariance_role1](../native_covariance_role1/report.md)|局部内容／覆盖／形状固定权重依赖诊断已有结果；片段内正证据和开发不确定性同时保留。这不能证明PE成功、纯中心充分或重训中心方法劣势|
|[11-06原文笔记](../../research_related_work/gaussian_parameter_encoding_20261106/notes.md)与[早期区域／RoPE记录](../../../docs/research_current.md)|GPSToken／JSRE的原文先例已记录；本轮新增是固定作者代码数据流、frustum／reuse边界与消融事实。没有进行第五项外部工作检索或全史fresh审计|

以上来自本轮实际读取的本地报告和注明来源的原Astra历史索引；原Astra的关闭／保留决定不是Sol新提出的科学结论。所有同信息控制都只是张量／分支要求，未实施、未选择样本或预算。缺作者代码或没有现成强消融不是候选自动否决规则；具体解释与至多一个后继问题由原Astra负责。

## 预算、工程事件与交接边界

首次取证UTC `2026-11-13T06:08:57.486906+00:00`；最后外部响应 `2026-11-13T06:12:44.232735+00:00`。请求48/48，含首个沙箱HTTP失败、四项浏览器打开、失败的OpenReview入口、项目及会议规范重定向、超单文件保护的PDF；作者文件22/24，四固定工作，无第五项。网络预算达到即停止，未知项照实交付。保存文本／元数据约数MiB，完整实际成本和交付时刻见[cost.json](cost.json)，DATA新增0。两份小CPU辅助脚本仅做HTTP/HTML提取和AST静态索引；未import／执行作者代码，无数值测试或造图。

初始statvfs可用16,871,620,608字节，超过128MiB阶段额度+1GiB余量；HTTP逐请求保护同时覆盖单项完成／大写入前检查，四项完成再次记录。全部阶段文件与living-doc正增长、暂存上界计入128MiB。本轮无自有GPU/model进程，所有有限HTTP／索引进程已正常退出，无他人进程操作。普通URL／guard／日志ID问题见[engineering_issues.json](engineering_issues.json)：并行只读批次导致临时请求ID冲突，完整原日志保留后按开始时刻规范化48条唯一ID，没有漏计或重复请求。科学合同未变。

CLI UUID复用`01a1d350-d1ec-70f2-9e37-8960b2dfe7a8`；本次实际请求配置由用户／主派发记录为`gpt-6.1-sol / high / service_tier=fast`。主先前rollout观察到client-selected Sol/high和auto_review；本角色没有独立后台effective model/effort/tier证明，后台tier记录`unknown`，未切换或改全局配置／认证。三份输入SHA与派发值逐一匹配，原分析plan/handoff不改；未调用goal、修改协调state/events、创建角色或发跨线程消息。

C1–C6无实验变动；给定相机尺度、sensor/joint-validity、四静态source、2048／1024、depth及旧数据仍是临时范围，不升为核心要求。当前无verified fresh reserve。交回完整事实后结束本CLI turn，主核验并复用原Astra；此输出不授权下一实验。
