# round_006：四项事实取舍与直接视图特征读取

2026-11-13，Astra科学分析完成。**四项核查没有给出可直接移植并称为O6已证的方案；结束该固定文献核查及原固定场/深度联合适配接口的自动PE修补。下一有限工作为`camera_only_feature_pe1`：直接监督公共粗GS＋相机的图像特征读取，用冻结VGGT的缺失区域输出判断位置编码收益。** 不是总体成功、停止或外部阻塞；完整目标、原正负证据及未来真独立确认保持。

实施前已读AGENTS、current/outline、授权/协议、round005原计划/分析和完整`gaussian_pe_route_review1`报告/四项机制事实/来源/交接；主已完成71小产物与48请求/22文件/688.91秒/零GPU预算检查，未重复网络、旧数值或大权重审计。精确输入为[analysis_input_handoff.json](analysis_input_handoff.json)和[input_reports.json](input_reports.json)。本轮仅CPU读代码/现有数据元信息和写文档，无GPU、模型调用、训练或数据下载；不操作主goal/state/events。

## 1. 四项分别证明了什么

| 工作及可核验来源 | 可以保留的事实/机制线索 | 不能据此得出的结论与本轮取舍 |
| --- | --- | --- |
| GaussianRenderFormer，arXiv2611.05593v1；作者互链仓库SVLwoof/gaussianformer，fb8cd4d… | 普通14维含μ/scale/rotation/RGB/opacity；中心3D/2D RoPE外，投影footprint两log轴＋双角经零初始化分支加K，V不直接改。RGB视图生成由learned DPT输出。Table4完整34.42/无footprint32.71dB，单次训练对象bootstrap差−1.71±0.07dB。 | 无footprint少23040参数，无等容量强中心邻域替代；系统/该分支分数不是本研究O6。共享只证一个forward内多view，不能推永久cache。作者12层场景＋6层view的大模型不是精炼方法现成答案。保留“直接视图生成监督下位置分支可能有用”的问题线索，不照搬模型或默认代码配置。 |
| GPSToken，arXiv2509.01109v1、NeurIPS2025核实；作者repo 2f34eb0… | 二维五参数与ROI特征共同形成token，普通Q/K/V；参数及texture联合预测、二维splatting。 | Table6的23.52/24.00/24.06PSNR同时改变分配/初值/参数与内容；不是3D公共状态或独立Σ-PE。当前代码inverse-sigmoid更新与论文残差文字有差异，版本未知保留。不能以“Gaussian token”直接继承O6，也不因缺强消融机械否决。 |
| SplatFormer，arXiv2411.06390v1；作者repo446ffb5…/Pointcept子模块c4aa232… | 全GS属性作普通feature，中心网格/Hilbert-Z序列化/稀疏CPE；已读wrapper独立RPE关闭。query-independent refiner输出属性残差，后由相机渲染。 | Table4 PTv3直接21.36/Mink残差22.67/PTv3残差23.06改变backbone/目标，不隔离参数PE。内容/几何细化已与本地joint_write/surface有实质重叠，直接再做writer并无新决定价值。作者声称ICLR2025，官方OpenReview挑战未独立确认；实际pseudo-target路径、checkpoint结构仍未知。 |
| JSRE，arXiv2601.02339v1；CVF确认ICCV2025书目，fly519/JSRE候选40a5bb4… | Σ图度量/谱描述进入普通内容f(G)，显式PE为中心项；另有协方差传播、语义拓扑/SH更新。相机frustum先选局部点。 | Table5模块50.6/48.3/49.2mIoU、29.86/29.63/29.78PSNR混合内容/传播/容量，非Σ-PE。frustum编码不能默认为query-independent writer；跨场景basis非同一GS状态。repo只有README/图片，反向作者身份未实证；正式CVF PDF未获取，公式只据arXiv，solver/梯度/cache未知。无须再追取、修作者谱公式或把缺源码当普适否决。 |

以上均由已保存的[事实报告](../../gaussian_pe_route_review1/report.md)、[mechanism_facts](../../gaussian_pe_route_review1/mechanism_facts.json)和[sources](../../gaussian_pe_route_review1/sources.json)逐项定位到原文/commit。GRF作者实现[footprint与视图块](https://github.com/SVLwoof/gaussianformer/blob/fb8cd4daf5b1fc0ea4ce9b72071701ecd4a0e670/gaussianformer/models/view_transformer.py)提供可描述的位置路径；原文[§4/§6.2](https://arxiv.org/html/2611.05593v1)的条件证据不能补出未做强中心对照。正式版本/代码默认与最终配方差异原样保留，无新增网络核查。

科学取舍：不选择GPSToken联合token、SplatFormer残差refiner或JSRE图传播作为新实验；它们的主要增益路径是普通内容/状态/系统变化，没有一个仅凭当前证据值得复制为新的参数PE接法。也不将GRF四维K分支换名安装回已失败的深度共同适配接口。**论文新颖性、现成代码或作者强消融不是实验价值的硬门槛；否决的是此次直接移植的额外信息价值。** 可测试的条件变化来自学习任务，而非文献名称。

## 2. 本地证据的合并边界

`native_covariance_role1`按原limited/uncertain/mixed分支结束：开发C_match−F +0.5957%[−0.8526,+1.7919]、S_match−F +1.3999%[−0.2811,+3.0246]的退步未达到原规则；32训练片段其他query +4.0803%[1.7890,6.4606]/+5.3332%[2.6680,8.8185]为条件形状依赖。C仍获原trace/最薄轴优待，原绑定内容/a_F含形状；固定权重扰动不证明纯中心足够、重训中心受限或PE成功。四个全零a_F query、开发零支持约20.87%、反例及完整raw/bias/full均保留，不据此归为唯一bug。

最近三项位置学习已比“没有训练到”更有约束：`native_input_pe1`的G/P +0.323%不确定，训练其他query G/L退步6.70%；保留renderer的`native_render_parameter_pe1`主G/P −2.935%且三seed不正，raw +9.409%有正线索但共同成本G/R +26.47%；source-only `native_memory_context_pe1`主G/P −0.0577%[−0.3739,+0.1347]，G/S、S/P也未有实用增量。共同writer对旧R开发+8.5578%仍不确定、训练其他query对Local约43.10%可靠，不能算O6。这些结果关闭各自自动变体，不支持不断换邻域/温度/位置注入处。

`source_geometry_control1`的sensor强中心有所改善而G/P仍无，`learned_projection_pe1`的G/P约−3.02%三seed阴性，bound/soft、native/coarse、末层教师重构和逐槽joint_write等历史保留。`geometry_transport1`latent raw改善主要含bias，空间/原生并未成立；旧`prototype1`已重构教师特征，旧`context_supervision1`已学末层上下文教师且跨地点不优，不能把任何“feature loss”包装为全新。与此并存的是早期区域PE在9个新地点的确认：R@1约41.389%，较Distance +2.988pp/Geo +3.362pp/球形+1.782pp、三seed正；它的任务/数据范围应保留，不能推出当前depth接口也有效，也不能从近轮阴性普遍否定Gaussian位置参数。

这些证据支持收窄**固定粗场＋当前RGB读取旁路＋深度联合适配**的自动PE修补，而不是限制总体目标只能做depth、固定sensor、固定状态或固定训练方式。O2/O3公共coarseGS、O4视图读取、O5渲染及参数、O6强中心位置增量是核心；监督目标和梯度路径是C项选择。本轮主动改后者，同时保留实际GS与原生用途评价。

## 3. 唯一后继问题与为什么值得一次完整比较

选`camera_only_feature_pe1`：来源GS＋相机经真实渲染取得特征anchor，单个128维四头reader预测完整query图像编码器特征残差；完整query缓存只作监督。训练完全不接当前RGB，VGGT全冻结。评价在已知缺失ROI替换encoder patches并跑原生24层/DPT。**主量仍是原生缺失ROI空间深度G/P，特征正线索是分层证据。** 这避免把完整目标换成容易通过的重构任务；同时直接检查原native失败是否仅适用于原学习条件。由于多个条件共同改变，任何成功不能反推某一旧失败唯一病因。

主假设：在直接学习视图特征的条件下，Gaussians的相机投影支持作为Transformer位置条件，较同信息中心组织更容易保留可被冻结VGGT使用的空间细节。竞争解释至少包括：中心邻域/相机已经足够；形状只经普通内容或共同renderer有用；feature重构主要降低均值而不改善空间/下游；训练128目标过拟合、未见地点状态误差占主导；无RGB reader依赖相机先验而非绑定内容；原生对合成encoder特征敏感。它们分别由P/S/G、Native/RasterFill、raw/spatial/bias分解、完整train/dev/query统计及唯一预定绑定置换暴露，不追加新组或扫描。

强中心公平性：全部arm共同f/完整logΣ/opacity/中心B与density普通输入，同实际render/coverage、源数量、有效mask、相机、主中心多尺度位置、架构/参数/初值/优化/训练。共同PE含中心B投影footprint，因此跨基元中心散布不是G独享。额外4维P用中心density、S用trace球、G用完整Σ投影；等预算编码分配而非nested新增维度。P可由普通内容利用Σ，G/P只评价显式位置组织的增量，不是“额外知道形状”。

公共性是真实可重用接口：396来源K/V状态只依赖源field和最终小reader权重，重载后同一状态供8query；原GS基底作为不可变依赖引用，查询不重新编码来源或用目标构建状态。导出state不是每query写一个伪公共memory。实际GS renderer共同anchor满足O5，未用普通attention完全代替真实高斯载体。

既有API/缓存足以实施：`native_coverage_data`已给44场/8query/full/missing的1024维encoder缓存；`patch_tokens`与`forward_read(reader=None)`是原生编码/接收入口；原有效alpha、源坐标和gsplat算子已有。没有必须单立科学合同核查的未知，张量布局、版本化普通修复及有限pilot在同一实施内完成。没有外部权限/资源/论文源码缺口。精确公式、九拟合、预算及评价见[执行方案](../../../../docs/camera_only_feature_pe_execution_plan.md)。

## 4. 固定预算与结果改变的决定

9×2000、batch2，六discarded pilot；正式feature3312、native4368（含144个G绑定置换），396正常＋36辅助公共状态。native≤4650、非训练reader≤3500、独立source编码≤600、实际gsplat≤400；6 GPU worker小时、最多2worker，新增DATA18GiB/项目1GiB。全部原生冻结，只保存小模型和FP32预测/KV，不复制大权重。真实资源/可用CUDA由Sol启动前检查；当前CPU空间快照另列，不把旧轮工程余量升级为科学条件。无大规模数值重审；新评分独立复算及两图实际审阅足够。

主开发G/P≥1%、95%下界>0、三seed均正为PE实用筛；相对Native同样有净用途才形成整体候选，G/S决定非球形额外解释，S/P可按同筛选择更简单球形方案。只有一个进入后继，仍为旧数据开发结果；真正确认要先审全部使用历史，再锁未用身份。主端点、原始/均值/完整图代价和计时不可选择性替换。微小跨零退步不是硬否决；≥1%且区间支持的RasterFill主量或missing全图AbsRel伤害是须权衡的实质代价，不自动晋升。特征指标不另增通过门禁。

若共同reader有效而PE无实用增量，保留内容读取成果并关闭此camera-only单层特征监督PE家族自动变体；若仅特征PE阳性，下游未达、不得用系统重构分数宣告完整目标成功；若全部阴性，结束此固定组合而非所有任务/状态。无任何分支自动补seed/步数/层/数据/损失或继续论文串。原Astra接完整结果后承担下一实质选择，当前不是仅留下等待状态的负报告。

## 5. 交接与配置事实

分析角色仍`/root/research_analysis_astra`，创建实际请求gpt-6-astra/max，本次followup无模型/effort覆盖参数，后台生效配置不独立暴露；不把请求当证实。本轮未调用goal工具；历史goal协调异常已有主记录，本次不推测原因。下一实施仅复用已完成standby的CLI UUID `01a1d350-d1ec-70f2-9e37-8960b2dfe7a8`，主每turn明确请求gpt-6.1-sol/high/service_tier=fast；主见client-selected Sol/high，后台actual Fast仍unknown。Sol采事实/实施，Astra作机制解释及科学取舍，主独占协调。最终handoff后本turn结束并释放科学写租约，由主顺序派发。
