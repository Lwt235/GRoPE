# camera_only_feature_pe1：相机条件视图特征监督的固定PE比较

2026-11-13T07:50:48.098794+00:00。九次各2000更新、396正常及36置换公共状态、3312 FP32 feature预测、4368 FP32 native预测、全部独立NumPy复算、固定bootstrap和两图实际审阅完成。此处只报告完整实测事实及预定flag；原Astra负责科学解释、取舍和后继，本轮不自行选择候选或宣布总体目标达成。

主开发原四query的missing ROI空间深度G/P降误差 **+0.1020% [-4.0429, +3.7452]%**；G/S **+0.6866% [-4.1856, +5.2527]%**；S/P **-0.5887% [-4.1450, +1.9493]%**。G/P三个seed为[-2.2968837096453587, 3.128304376812052, -0.4331685415094544]%，12地点中6地点G更低、6更高。预定PE筛flag：G/P=False，G/S=False，S/P=False。未通过不称等价／非劣性；通过也只是旧数据开发筛选。

## 固定问题与实现范围

按[锁定执行方案](../../../docs/camera_only_feature_pe_execution_plan.md)和[原Astra机器计划](../autonomous_research_loop/round_006/plan.json)，只训练一个128维4头cross-attention reader预测完整1024×1024 encoder feature残差。原真实GS renderer提供共同归一化anchor b，Q来自b、覆盖及相机20维，不接当前query RGB／queryfeature／mask／depth／编号。完整query encoder feature仅监督；整个官方VGGT被冻结，无native/DPT训练forward或深度梯度。主端点保留开发原query原生空间深度，不以feature分数替换。

三组共享完整Σ普通内容、原f/opacity、中心31邻域B和density4及相机／中心多尺度位置、原渲染／coverage、容量、初值、优化器及采样。每组精确656,768参数。共同25位置维加四槽，P为density4、S为同trace球投影footprint、G为原Σ footprint；同29维分配，不是nested新增独占信息。只有K加位置，V不加；背景零K/V始终存在。普通内容源LN(f)＋source20，经1044→128 GELU/LN及独立K/V；query1045→128 GELU/LN→128；位置29→64 GELU→128；attention与Q拼接256→256 GELU→1024，输出末层与位置末层零初始。没有增加writer/nativePE/fusion/gate或几何学习。

源帧0/3/6/9；原32train／12dev，拟合query2/5/8/11，其他1/4/7/10；每epoch128完整目标，full/missing不重复作两个训练样本。三配对seed20261110/11/12，AdamW3e-4/wd.01/clip5，reader/Adam FP32，全特征rawMSE，batch2，固定2000末步；六discarded pilot后正式重置。初值和4000样本顺序按seed跨arm一致，9份final含model／Adam／RNG／sampler／budget完整恢复状态。

每fit/scene一次source编码为FP32 K/V后永久保存重载，同一SHA读取8相机。原GS以不可变路径／权威SHA引用，衍生状态不复制原1024维f；原场与衍生K/V共同构成逻辑公共memory，不称单文件自足或原场构建免费。正常评价重用既有实际r/a缓存；置换及成本真实重算。原native评价从missing encoder cache出发，仅替换[8:24,8:24]中心256patch，经原24层、2DRoPE、DPT，无额外学习。Native-full共同352结果按构造共享，不能当学习后full保真或非劣性。

## 全部主、净用途与辅助证据

|比较|开发原query空间降误差95%区间|对Native实用净用途flag|
|---|---|---|
|P−Native|-7.1097% [-42.8606, +13.9854]%|False|
|S−Native|-7.7403% [-42.5608, +13.5934]%|False|
|G−Native|-7.0005% [-43.7305, +14.2951]%|False|
|RasterFill−Native|-26.9703% [-80.9713, +8.0937]%|仅共同anchor参照，不是历史训练R|

开发feature full-view raw MSE的G/P降误差+0.2049% [-1.7724, +1.6519]%，是次级组件证据；共同G相对RasterFill的feature raw降误差为+63.6564% [+49.1601, +70.3744]%，不能归为PE。训练其他query的G/RasterFill空间改善+6.1072% [+1.8251, +11.1130]%，原12／新增20分别+1.6453% [-1.2288, +5.5438]%／+8.9367% [+3.1120, +15.7635]%，仍均为旧数据条件用途。raw／去channel均值空间／bias²／cosine与ROI／全视图／覆盖分层均完整报告。特征改善不证明原生用途，也不要求feature必须显著才报告native实测。 missing全图AbsRel的G/RasterFill降误差为-0.7061% [-1.5778, -0.0973]%，即误差均值增加约0.706%，区间支持小幅代价；它低于事前1%材料性均值阈值，但不是零代价或保真证据。native raw／bias²／空间量、missing全图AbsRel／空间／raw、原始几何支持与无支持／零有效覆盖分别保留。材料性伤害flags见[method_decision.json](method_decision.json)：相对RasterFill的主量或missing全图AbsRel退步≥1%且95%支持才标记；微小不确定负均值没有硬零容差否决。

训练原12／新增20、拟合与其他query、开发其他query、所有G/P、G/S、S/P及Native/RasterFill比较、全部seed/place/query均见[全数表](report_tables.md)、[全部比较CSV](scoring/all_comparisons.csv)、[逐预测native](scoring/per_prediction_native.csv)／[feature](scoring/per_prediction_feature.csv)。5000配对seed/place bootstrap固定20261113，地点/query等权；三seed和12dev有限，不是通用门禁、等价检验或独立确认。

唯一预定G内容置换144例将同一置换f同时用于重建K/V及实际anchor；开发原query空间量置换相对正常G的降误差为-45.5716% [-96.4155, -20.1531]%（负值表示置换更差）。它只说明固定模型的条件绑定响应，不是PE独立比较或唯一机制证明，不触发额外扫描。原a_F全零query共10个，完整清单在[覆盖统计](coverage_statistics.json)，全部保留预测和评分，无困难样本筛选。

最大三seed平均G−P开发退步例为fidelity_02/q11（brown_cogsci_1），空间误差差+0.00406275；其余全部反例在[counterexamples.csv](counterexamples.csv)。[汇总图](figures/outcomes.png)与[固定两旧训练scene＋最大反例](figures/cases.png)均实际view_image审阅，[审阅记录](figures/visual_review.json)保留可见事实；个例图只展示首seed，不替代三seed机制结论。

## 正确性、普通修复与完整预算

必要预检确认字段有限、API无query图像输入、同参数／成对初值、零残差等于RasterFill、ROI对齐且其他patch逐值不变、source K/V保存重读、六pilot梯度及一次完整恢复。补充一个CPU全key无效fixture确认零背景使输出有限，实际reader调用也计费。所有新3312 feature和4368depth独立NumPyfloat64复算，保留原mask及每文件SHA；未重审旧模型／历史数值，无大权重重复hash。

两项普通修复在同轮版本记录：首次Ninja／扩展缓存PATH缺失，1次失败raster尝试计费、0pilot；后恢复断言比较CPU/CUDA step张量失败，保留Pstep1完整快照、仅执行剩余5pilot，未重复已有调用或增加正式更新。旧锁／旧源码快照／精确diff和[runtime_source_revision_v2.json](runtime_source_revision_v2.json)均保存，监督／字段／端点／组／预算不变。失败预检的已通过assert与未落盘诊断值分开记录，不捏造丢失数值。 评分展示v3将cosine统一作为绝对相似度差／区间而非降误差百分比；尤其zero-coverage RasterFill的零向量cosine=0，百分比不可定义。原cosine值、绝对bootstrap和全部error指标不变，原summary／CSV版本保留，无重跑模型／评分或新门禁。

实际counts：{'formal_updates': 18000, 'pilot_updates': 6, 'training_reader': 36000, 'training_source': 36000, 'pilot_reader': 12, 'pilot_source': 12, 'native': 4570, 'reader': 3441, 'source': 590, 'raster': 307}。正常保留18000更新、训练reader/source各36000样本；pilot6及各12样本另计，0native训练。GPU含加载／准备／失败／恢复／I/O共0.386715worker小时≤6；最多两worker，均选空闲GPU0/1，未占GPU5其他任务。所有GPU子进程已实际退出；最终CLI完成由主在本turn结束后观察。本轮DATA永久14.938503GiB≤18，atomic峰值与headroom账见[资源记录](resources/ledger.jsonl)及[cost.json](cost.json)；项目≤1GiB。恢复每100步持久化，验证final后仅退休本轮重复resume，无旧／他人资产删除。

## 实际成本与一次写入复用

成本固定前两旧训练scene/query2、五方法轮换、各5warm＋15timed，共200完整native query；其中P/S/G共120次reader就在相同调用内部取CUDA event段，未另跑120次reader。source3arm×2scene×25=150，均计费；实际160次GS cost raster。完整cost含query图像encoder、实际原GS raster、实际FP64相机位置、reader、patch替换、原native/DPT与210px resize；没有拿缓存I/O代替真实计算。event段含dispatch／记账间隙，CPUwall含同步与检查，不称纯kernel时延。共享服务器测量，无独占／微小时差加速结论。 实际共同view_fields helper一次计算B/S/G各字段，再按arm选择额外四维；P未读未用的S/G字段仍有共同准备开销，因此这些是实际实现时延，不是各arm最小化后的kernel成本。Native/RasterFill不存在reader的空事件间隙标为不适用，不计虚构reader调用。

|方法|完整query CPUwall中位ms|GPUevent总ms|memory read event ms|reader event ms|
|---|---:|---:|---:|
|Native|104.149|104.101|—|—|
|RasterFill|113.053|113.008|7.708|—|
|P|118.422|118.375|12.901|1.277|
|S|117.294|117.244|12.829|1.279|
|G|118.104|118.057|12.762|1.271|

source中心B/density准备、K/V投影、实际state冷加载、序列化、基底／衍生bytes和N=1/4/8/32完整摊销见[cost_amortized_totals.json](cost_amortized_totals.json)。原source构建沿用既有权威实测参考及其边界，sensor获取／AnySplat预计算／raw预处理／模型冷载不能藏为0；不同20场pipeline范围单列。N32只是示意，实际同状态读取8query。 [逻辑加载记录](load_cost_measured.json)分开CPU文件读取和CPU→GPU/权重装入，[共同输入准备](source_input_cost_measured.json)另测固定两场的FP32源LayerNorm/拼接；均无额外网络forward。OS page cache未清空，冷载指本次显式逻辑物化，不声称物理冷盘。摊销分别给不含／含实际native冷载参考的总量。

## 证据边界及交回

本轮直接服务O2/O3的公共粗GS及可重载K/V、O4同状态多view读取、O5真实渲染与参数、O6 memory reader Transformer的参数位置组织；不是原生24层新增PE。C1/C5保持，C2/C3/C4/C6按计划改变。sensor/joint-validity、给定相机／尺度、静态四源、2048×1024、固定geometry/content、冻结VGGT、已知missing ROI与depth／旧数据仍为临时边界。

原prototype/context_supervision已有特征监督，来源writer、输入attention及保留Render的原生Q/K阴性保持；无query RGB的直接监督任务是本轮条件变化，不改写这些历史。共同renderer／reader收益不冒充PE；P也有完整Σ普通内容，不能叫删除全部shape；P/S/G等29维预算分配不是nested信息增加。全部已用，无verified fresh reserve，任何合格开发候选以后仍需全使用史审查与真独立身份锁定。

实现复用CLI UUID `01a1d350-d1ec-70f2-9e37-8960b2dfe7a8`，主实际派发请求Sol/high/fast；主报告client-selected Sol/high，本角色没有独立后台effective model/effort/tier证明，实际tier为unknown。未改全局配置／认证、未创建角色／任务、未发跨线程消息、未调用goal或修改主state/events。原科学输入身份不改；完整结果交主复用原Astra，结束本turn释放实施租约，不自行继续下一轮。
