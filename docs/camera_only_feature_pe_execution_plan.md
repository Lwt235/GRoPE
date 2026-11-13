# camera_only_feature_pe1：直接监督公共记忆的相机条件特征读取

2026-11-13，round_006，实施前固定方案。科学决定见[本轮分析](../results/gaussian_memory/autonomous_research_loop/round_006/analysis.md)，机器计划见[plan.json](../results/gaussian_memory/autonomous_research_loop/round_006/plan.json)。只实施此一项；普通预检与正式实验合在同一有限阶段，不另设合同审计轮。

## 1. 问题、范围与历史差异

检验：**公共粗GS状态与查询相机直接预测VGGT图像编码器特征时，显式Gaussian位置组织是否较同信息强中心组织产生可靠的原生用途？** reader不接当前query RGB/patch、教师目标、深度、查询编号或缺失mask。完整query特征只作训练监督；整个官方VGGT冻结。实际GS渲染保留为共同anchor，只有一个小cross-attention reader学习其残差。无当前RGB旁路不保证形状有用；P/S/G全部保留完整形状普通内容，检验的是位置组织的归纳偏置。

这是学习任务与梯度路径的改变：从“当前RGB query＋深度损失驱动的融合/原生共同适配”转为“相机＋公共状态→完整视图特征，随后冻结原生网络填补评价”。不是仅换PE注入层，也不把渲染用途叫PE。`native_input_pe1`曾用当前图像Q并替换旧renderer；`native_render_parameter_pe1`是在保留renderer后的原生Q/K上加PE；`native_memory_context_pe1`是在来源关系writer内共同训练深度；三者的阴性均保留。`prototype1`已做固定渲染的查询特征重构及来源/几何学习，`context_supervision1`已做当前图像Q＋原始4源token的末层上下文教师重构；因此**特征监督本身不是首次**。本轮区别为真实公共粗GS、无query图像的可缓存K/V读取、同信息强中心PE对照，以及不学习原生接收器的原生端点。

GRF原文/作者代码只提供“视图生成任务中完整属性内容之外有footprint key分支”的条件线索，不提供本项目效果或同容量强中心结论。本轮不移植其12层encoder/6层decoder/DPT RGB系统，不继续四项综述，不下载安装作者依赖。source_geometry、learned_projection、joint_write/surface/geometry_transport的旧结果不能由换论文名抹去，也不禁止有明确不同学习任务的比较。

O2/O3核心由44份真实公共GS及可重载衍生K/V实现；O4用同一状态读取8个相机；O5保留实际alpha合成及参数；O6在服务VGGT的**记忆reader Transformer**内比较位置编码，非VGGT原生24层新增PE。O1保持patch≠世界点。C1来源形成不改；C2增加共享小K/V编码但不改原f/几何；C3为渲染anchor＋相机特征读出；C4为下述P/S/G；C5仍给定相机；C6只学小reader、特征监督、冻结全部VGGT。sensor/joint-validity、静态4源、给定尺度、2048槽/1024维、depth端点和旧数据均是临时范围，非大纲要求或已解决能力。

## 2. 数据与共同状态

复用`native_training_coverage1/plan.json`的32训练/12开发地点和全部44场，身份沿用原manifest，源帧固定[0,3,6,9]。拟合查询[2,5,8,11]，另评[1,4,7,10]；主开发仍12地点的原4查询。每训练epoch单元为32×4=128个完整特征目标，**不把full/missing重复为两份相同训练样本**。旧12/新增20训练分别保留统计。全部已用于开发，没有fresh reserve；本轮不扩大数据、源、seed或目标。

读取`experiments/native_coverage_data.py`的`scene_path/query_path/patch_path/feature_path`，重用完整/遮挡448×448编码器缓存，各1024个1024维patch（32×32）。原缺失方块[112:336,112:336]严格对应patch[8:24,8:24]；目标深度和ROI仍按旧210×210评分协议。完整特征为已有冻结`patch_tokens`输出，不称世界不变教师。真实相机坐标/单位使用现有field及source_c2w约定。

场景原μ、Σ、opacity、has_content及原绑定f全程固定。定义源首相机坐标下p=(μ−o0)R0/u，S=R0ᵀΣR0/u²。中心邻域只由有效中心算：每槽取31个最近其他有效中心（稳定slot编号破同距），δ为归一化中心差，B=mean(δδᵀ)+1e−6I；不足31用全部，无邻居用1e−6I。density4为第1/8/16/31近邻距离的log1p，缺秩取最后可用，均无则0。P/S/G都得到B、density4及完整S，不把中心集合散布当Gaussian特有形状。svec对角按原值、非对角乘sqrt(2)，matrix-log在FP64预处理。

每个fit/scene导出一次FP32 K/V及几何字段，396份正常公共衍生状态；同一重载状态读取8query，评价不再调用source encoder或源RGB。原场f/μ/Σ/opacity采用不可变manifest引用，不复制396份1024维原内容；衍生状态与44份原场共同构成逻辑公共memory。报告基底与衍生状态真实bytes、依赖路径/SHA、一次构建与加载成本，不称单文件自足或原场构建免费。

## 3. 唯一小模型与匹配PE

**共同实际渲染anchor。** 用原gsplat算子/原几何/opacity、32×32格，颜色[f×has_content,has_content]得到r与有效覆盖a_F。无内容slot的opacity仍保留原遮挡作用。令b=r/max(a_F,1e−8)，a_F≤1e−8处b=0。r/a_F缓存可复用历史实际合成，所有方法一致；成本必须真实重算。归一化服务当前特征重构目标，不能称旧R归一化修复或由旧固定权重证明有效；round001全图幅度代价仍保留。

**普通内容共同。** 源输入[LayerNorm(f)1024, asinh(p)3, svec(log S)6, opacity1, svec(log B)6, density4]共1044维，经Linear→128、GELU、LayerNorm，分别Linear128得到K/V。每组输入、参数个数、初始化和训练范围相同；P也可从普通内容使用完整Σ。

**Q只来自公共状态＋相机。** [LayerNorm(b)1024, camera20, log1p(a_F)1]共1045维，经Linear128、GELU、LayerNorm、Linear128得到Q。camera20为源首相机系中的asinh(o/u)3、单位ray方向3、归一化patch坐标x/y2（2×patch中心坐标/32−1），以及对(x,y)在π×{1,2,4}上的sin/cos12。o使用相对源首相机的平移；ray来自相机内参与旋转。无当前RGB、query feature、mask或深度输入。

**位置字段。** 基元在query相机内的归一化中心t；共同25维为asinh(t)3、归一化投影uv2、log1p(norm(t))1、单位方向3、uv在π×{1,2,4}上的sin/cos12、中心B的投影footprint4。投影uv使用32×32格的2×投影坐标/32−1，不作frustum裁剪。仅有效内容且原相机物理z>0.01者进入attention；任何非法后方投影先按统一有限占位算、再mask，不泄漏NaN。全部无有效key时仍有一个固定零K/V背景token。

给相机坐标协方差A，以32×32内参的投影Jacobian J算C=JAJᵀ+0.3I；footprint φ(A)=[log sqrt(λmax), log sqrt(λmin), cos2θ, sin2θ]，双角取[(Cxx−Cyy),2Cxy]的单位向量，分母<1e−12时固定(1,0)。B先还原物理单位并旋转到query相机；S/G也用物理Σ。四额外位置维固定如下：

| arm | 共同25维以外的4维 | 解释 |
| --- | --- | --- |
| P | density4 | 纯中心多尺度密度，活跃四维；中心B投影已在共同字段 |
| S | φ((trΣ/3)I) | 原trace球形的投影支持；普通内容仍为完整Σ |
| G | φ(Σ) | 原完整Gaussian投影支持 |

位置MLP为29→64 GELU→128（末层无bias且零初始化），加到K；四头各32维，scores=Q·(K+PE)/sqrt(32)，标准softmax，无温度/额外距离门控。P/S/G以同等4维位置预算比较编码分配，**不是严格nested加维比较**；所有中心信息仍可由共同普通输入使用，B投影/多尺度uv是共同位置输入。V不加位置分支。单次attention输出与Q拼为256维，经256→256 GELU→1024残差decoder，末层weight/bias零初始化；最终f_hat=b+residual。没有writer self-attention、native PE、额外fusion/gate或几何学习。

## 4. 固定学习与原生评价

P/S/G×seed20261110/11/12，共9fit，每fit2000更新，batch2，同seed的128目标固定成对随机排列。AdamW lr3e−4、weight_decay0.01、gradient_clip5；唯一loss=全1024patch×1024channel原始feature MSE，固定最后一步，不选checkpoint/改loss或扫描超参。reader权重、Adam及训练计算FP32；官方VGGT全部冻结eval，训练不前向native/DPT，不传任何深度梯度。六个丢弃pilot更新（每arm2次、仅首seed）计费后重置，正常预检即可，不加新科学门禁。

每fit保存完整最终小模型及可恢复状态，导出396重载memory。正常feature预测9×44×8=3168；全部FP32保存。原生missing评价：从原遮挡query的缓存encoder patches出发，仅替换中心256patch为f_hat（转换为原patch dtype）；其他patch不变，经`models/gaussian_scene_native.py:forward_read(reader=None)`的官方24层/2D RoPE/DPT，无额外训练或融合。每次调用是fresh单帧native sequence。保存210×210 FP32 depth及原评分身份。

共同固定参照各352：Native-missing不填补；RasterFill用b作同样填补；Native-full用完整缓存patch、不填补。总正常native=3168+3×352=4224。全图full的同一Native-full结果按构造共享，不能宣称学习后全图保真/非劣性；missing条件全输出仍可能变化，必须完整评分。RasterFill不是历史训练R，旧R分数仅历史语境，不能冒充同权重基线。

唯一预定条件诊断：G三seed、12开发原4query，共144例，将f在有效slot间按固定seed20261113/scene排序置换；μ/Σ/opacity/has_content、相机均保持，**K/V和实际渲染anchor都用同一置换f重建**。导出36额外状态。它只检验该固定模型对内容绑定的条件依赖，不是新增PE对照或唯一因果；不以其正负追加诊断。正式reader feature=3312，native=4368。四个原a_F全零query及所有低覆盖样本均保留。

## 5. 指标与决定

主量不改变为容易通过的feature分数：**12开发原4query，missing ROI centered log-depth MSE，G相对P降误差**。地点等权、同seed配对，复用5000次seed/place配对bootstrap（固定20261113）与全部逐地点/seed/query表。实用PE筛为均值≥1%、95%下界>0、三seed均正；只作旧数据开发筛选。G/S同量为非球形增量解释；若G/S也明确为正才称非球形额外价值；S/P可按同主规则筛更简单的球形候选，至多选择一个。多比较完整列出，任何后继仍需真独立确认。

同时报告G/P、G/S、S/P及各arm对Native/RasterFill的主量、ROI raw/bias²、missing全图AbsRel/空间/raw、全支持与低/零支持分层；训练原12/新增20、原拟合query/其他query及开发其他query分开。相对Native的主量须有实用净用途（同≥1%、下界>0、三seed正）才进入整体候选；共同P/S/G受益不能算PE。对RasterFill主量或missing全图AbsRel若退步≥1%且95%支持伤害，视为材料性用途代价，需Astra权衡、不可自动晋升；微小跨零负均值只作排序/范围材料，不作硬零容差门槛。成本完整报告，不继承旧轮任意通用时延门槛。

特征证据为次级：full-view与ROI的raw MSE、逐channel去patch均值后的spatial MSE、bias²（三者一致分解）、cosine及G/P/S/RasterFill、预定binding诊断。特征改善不是原生成功；不另设feature必须通过的机械门槛。原生主增量而feature不显著时如实记录指标差异；反过来仅feature正而native无实用增量，只能是组件证据。

结果分支固定：①主PE与Native净用途均过筛且无未处理材料性损伤，留下一个开发候选，Astra先全历史数据审查并锁真未用确认；本轮不称采用。②共同reader/特征有益但PE无实用增量，保留普通读取用途，关闭此单层camera-only特征监督Gaussian-PE家族的自动变体。③仅特征PE正而native未过筛，保留分层机制证据、原生用途未达，不自动微调接收器/换监督延长假设。④均阴性或退步，关闭此固定组合，不外推所有任务/状态/Gaussian参数无效。所有分支均终止本实验，不补seed/步数/层/邻域/频率/温度/损失/数据。总体研究仍由Astra据完整证据选择实质下一问题，不能把本轮结项当成功、暂停或外部阻塞。

## 6. 预算、成本与安全恢复

硬预算：9×2000正式更新，预检6更新；正式训练样本reader/source调用36000，pilot12另列。非训练native≤4650（4368正式＋200cost＋≤24预检＋≤32普通恢复=4624）；非训练reader≤3500（3312正式＋完整query成本内共用的120次reader＋≤12预检＋≤32恢复=3476）。独立source编码≤600（396正常＋36置换＋150cost＋≤12预检=594，余6恢复）。真实gsplat≤400，允许160完整query成本＋144置换＋≤16预检＋≤32恢复=352；已有352个正常r/a_F缓存只读复用，不额外生成贡献矩阵。计费按样本调用，batch不能隐去次数，恢复/丢弃全部入账。

GPU≤6 worker小时，最多2worker、最多共享1张有其他任务的卡；每worker上限10GiB，开跑实测空闲≥16GiB并保留其他用户6GiB余量。不占他人进程，优先空闲卡；实时CUDA/占用核验由Sol在启动前执行，分析阶段无GPU探测。达到预算即完整保存并交回，不悄悄增预算或用不齐结果下方法结论。

新增DATA≤18GiB、项目≤1GiB，含临时/恢复/原子替换峰值。3312×1024² FP32 feature约12.938GiB；4368×210² FP32 depth约0.718GiB；432×2048×256 FP32 K/V约0.844GiB；其余几何/小权重/完整恢复/图表及单次归档暂存合计预留≤3.5GiB。按fit分片原子发布，不保留全量重复archive；旧原场/教师/cache/官方VGGT权重只引用不复制，不删旧或他人成果，不以tmpfs当永久final。

工程准入按实时剩余产物估算：H=可用DATA空间−尚未落盘的固定最终产物−当前所需临时峰值（已写入产物不得重复扣）。开始/恢复H≥8GiB，项目扣除剩余产物后≥2GiB；运行H<6GiB或实际free<5GiB即停止接新任务、完整持久化model/Adam/RNG/索引/计数并退出。30秒无解释下降≥0.5GiB只在H<8GiB紧张带额外触发；宽裕时记账。检查点每100步及资源退出保存，小模型无需易失性大模型副本；验证最终状态后只可退休本轮重复resume，保留失败版本与累计丢弃。此是工程预留，不是科学假设或普适门禁。

成本固定两个旧训练scene的同一原query（沿既有前两场景顺序），不挑效果：Native、RasterFill、P/S/G首seed各5warm＋15timed，共200native；每次完整成本含真实query图像encoder、真实GS feature raster（Native无）、位置计算/读取、patch替换、native/DPT，不能把缓存IO当实际读取。这200次中P/S/G各两scene、5warm＋15timed，合计120次实际reader；用同一次调用内部CUDA events及外层同步分别提取reader、完整memory-read分段成本，**不另跑独立120次reader**。分段结果标明GPU event时间，完整query同时保留CPU wall与同步GPU时间，不能把事件分段时间冒充完整延迟。source K/V编码3arm×2scene×(5warm+20timed)=150调用。source构建（包括共同中心B/density准备与K/V投影，公共部分可另分列但不记零）、冷加载、暖memory读取、完整native成本以及N=1/4/8/32总摊销分开；基底GS既有构建只继承已测事实并标明，不重跑或记零。时序轮换/同步并报告共享GPU条件，不从细小负时差推加速。

## 7. 交付与职责

常规预检只确认一批数据/字段有限性、reader API不读query图像/教师、同参数与成对初值、零残差对应RasterFill特征、mask精确对齐、source-state重载、六pilot梯度和一次完整恢复。用原两训练场景作必要核对，无数值微差门槛或旧大权重重审；ordinary bug版本化在原轮修复。若需改监督/源/位置字段/端点等科学条件，保存现场交Astra。

输出`results/gaussian_memory/camera_only_feature_pe1/`的锁定config/输入与原始数据身份、budget/cost、9最终小权重与396＋36状态身份、完整3312 feature/4368 native预测、全部指标/表/固定bootstrap、独立NumPy float64新评分复算（不重审旧评分）、两张图及实际审阅、counterexamples、report/method_decision/implementation_handoff；两图覆盖主与辅助/完整代价，以及固定两scene和最大开发平均G−P退步例的feature/深度/支持。可重用现有评分与绘图工具，不增加第三审计系统。

Sol负责实现事实、正常工程选择与完整固定结果；科学解释/取舍由原Astra回收。实施只复用CLI UUID `01a1d350-d1ec-70f2-9e37-8960b2dfe7a8`，由主每turn显式请求gpt-6.1-sol/high/service_tier=fast；client选项可见不等于后台Fast核验。Astra创建请求gpt-6-astra/max，本次collaboration复用无覆盖字段，后台effective未知。主独占goal/state/events并顺序交接；本分析不启动训练或实施角色。
