# native_feature_field_pe1：逐场景来源特征形成与原生PE用途

按[唯一科学合同](/home/liwt/GRoPE/results/gaussian_memory/autonomous_research_loop/round_011/science_plan.md)（SHA256 `8ef5ad0ee10ff96b261710a0b015cc21aa6748d9c3df954c57a41c3197598901`）完成88份场×300来源更新、18×2000 reader拟合、6624份FP32特征和7328份新FP32原生深度。704个唯一Native-missing/full结果只读继承，各行评分引用同一基线。本文报告实测事实与固定flags，科学取舍交原Astra；无fresh或总体成功声明。

## 固定问题与历史范围

新因素是source-only、逐场景、实际完整1024D渲染目标下的C（仅f）与J（f＋Σ）配对形成；不是首次特征监督形状。[prototype1](../prototype1/report.md)的PCA32 writer线索、[attention1](../attention1/report.md)强中心确认阴性、[source_geometry_control1](../source_geometry_control1/report.md)来源改善未解锁PE、[camera-only](../camera_only_feature_pe1/report.md)固定场阴性均保留。round007/010路线关闭和round008无新设计不变，本轮不重开相机细化或直接算子扫描。

来源只有0/3/6/9；24份完整特征目标缓存继承、20份补80次冻结编码。每场循环v=step%4，实际gsplat颜色[f×has_content,has_content]得到r/a，a≤1e−8设零，全patch/通道raw MSE。μ/opacity/DC/槽集合与有效性/相机/单位/mass固定，仅有内容槽的f可优化；J额外优化rotation/log-scale。Adam的f/rotation/log-scale LR=.01/.001/.02，300末步，几何FP32/f FP16；没有增删槽、附加损失或查询输入。

每行一份场由所有P/S/G和seed共享，形成后冻结。普通内容含完整f/Σ/opacity、中心B与density，三组相同128宽四头camera-only reader与29维位置MLP；末四维分配density/球形/full footprint，只有K加位置，V不加。查询真特征只监督reader，原生网络全冻结、无native/DPT训练forward。2000步batch2、FP32 reader/AdamW3e-4 wd.01 clip5、每100步恢复，最后一步而非选模；每行六pilot丢弃后重置。

与原接口相同，只在missing patch[8:24,8:24]填预测特征，随后真实官方24层/2DRoPE/DPT；Native-full按构造共享，不是学习后保真/非劣性证据。新场每行352份RasterFill重算，原场旧结果只作历史。每行396正常＋36置换KV保存重载，多query复用；置换只用于原G条件绑定诊断（连同实际anchor重建），不证明唯一因果。

O2/O3公共粗GS、O4/O5读取/实际渲染和O6 memory-reader PE服务VGGT保持；C1/C2形成条件、C6来源拟合改变，C3/C4/C5固定为测量选择。给定相机、sensor/joint-validity、静态四源、冻结原生、人工missing与旧数据都是临时范围。不是前馈/免费构建、纯RGB/未知相机能力或原生24层内新增PE。

## 主终点、交互与净用途

开发12地点原四query missing-ROI centered log-depth MSE，地点/seed配对等权，固定5000次bootstrap seed20261113。正降误差有利；主量为J行G对P，交互直接用两行配对增量差，不从显著性不同推断交互。

|行|G/P降误差及95%|G/S|S/P|G/Native|G/RasterFill|
|---|---|---|---|---|
|C|+0.1106% [-2.0879, +3.4735]%|+0.3765% [-1.3723, +2.5659]%|-0.2668% [-3.0218, +2.2630]%|-11.4959% [-50.0006, +11.7585]%|+12.6025% [-1.5776, +23.4742]%|
|J|+0.0857% [-2.8703, +3.1167]%|-0.5030% [-5.4413, +4.0998]%|+0.5857% [-1.7143, +3.1810]%|-20.1340% [-65.5636, +4.8828]%|+10.4539% [-1.7872, +19.5109]%|

Δ_J=E_JP−E_JG=+2.4181623e-05，95% [-0.00083524891, +0.00096468676]；I=(JP−JG)−(CP−CG)=-4.812049e-06，95% [-0.0018540334, +0.0013021458]。相对增量差为-0.0250个百分点 [-6.2119, +4.4483]，见[factorial_summary.json](factorial_summary.json)。

实用PE及Native净用途筛为均值≥1%、95%下界>0、三seed均正；只是本固定配置开发筛，不是通用门禁、等价或独立确认。每行完整P/S/G互比及对Native/同场RasterFill、逐seed/place/query、训练原12/新增20与其他query、支持/零覆盖/全图raw/bias/AbsRel与特征分解全部在[C全表](C/report_tables.md)、[J全表](J/report_tables.md)及各行scoring CSV/JSON。

C：G/P三seed=[-0.656208972055472, 2.7352313018935814, -1.7441633499059916]%，PE筛=False；G/Native筛=False。辅助feature raw G/P=-0.0362% [-1.4078, +1.1437]%；固定G置换相对正常G主量=-46.3109% [-97.4120, -15.2676]%。不将feature/绑定响应当Native净用途或PE因果证明。
J：G/P三seed=[1.3828242385991119, -2.4592668426757447, 1.3530895246013586]%，PE筛=False；G/Native筛=False。辅助feature raw G/P=+0.2232% [-1.6398, +1.8603]%；固定G置换相对正常G主量=-54.2350% [-114.4176, -14.5219]%。不将feature/绑定响应当Native净用途或PE因果证明。

C共同reader的完整query feature raw G/RasterFill降误差+60.7358% [+46.9591, +68.0053]%；这项组件收益不替代主PE与Native净用途。

J共同reader的完整query feature raw G/RasterFill降误差+57.5251% [+43.8011, +65.1823]%；这项组件收益不替代主PE与Native净用途。

## 来源、形状、覆盖与反例

C保存精度下来源raw MSE均值 0.0070756871→0.0054306681（相对低23.249%）；平均逐场形成耗时11.551s。完整source raw/spatial/bias、f幅度、Σ/trace/最薄轴/各向异性变化见[formation_statistics.json](formation_statistics.json)，固定中心B/density/有效性两行逐tensor相同。
C原零覆盖query 10，新零覆盖10，全部保留。最大三seed平均G−P退步例 `fidelity_03` query8，绝对误差差=+0.0021600559；全排名见各行counterexamples.csv，覆盖逐query记录见preparation。
J保存精度下来源raw MSE均值 0.0070756871→0.0047193408（相对低33.302%）；平均逐场形成耗时9.879s。完整source raw/spatial/bias、f幅度、Σ/trace/最薄轴/各向异性变化见[formation_statistics.json](formation_statistics.json)，固定中心B/density/有效性两行逐tensor相同。
J原零覆盖query 10，新零覆盖10，全部保留。最大三seed平均G−P退步例 `fidelity_02` query8，绝对误差差=+0.0065468571；全排名见各行counterexamples.csv，覆盖逐query记录见preparation。

来源重构下降不排除视角依赖、非唯一f/Σ分解、覆盖和幅度改变、普通内容改善或强中心已足够。形成改变内容/覆盖是处理的一部分，I不能单独识别“纯Σ信息”因果。材料性RasterFill主量或全图AbsRel伤害（≥1%且区间支持）按各行method_decision完整标记交Astra权衡；小且不确定的辅助逆均值不硬否决，跨零不称非劣性。

## 成本、恢复与实际预算

实际GPU累计0.793737 worker小时≤8，包含来源、预检/pilot、让出/恢复、reader、全评价、真实成本及加载/I/O。来源26400更新无重放，reader36000更新/72000训练样本，pilot12更新/24样本另计；实际native=7732、非训练reader=6880、K/V编码=1192、真实gsplat=28596。完整逐样本与阶段账见[budget.json](budget.json)。

DATA逻辑35.137GiB，实际分配35.194GiB，含两worker原子暂存保守峰值35.241GiB≤36。完整永久路径/bytes/身份见[DATA_manifest.json](DATA_manifest.json)。旧输入/权重只引用，不复制或重hash大权重。来源最终checkpoint保留完整FP32参数/Adam/RNG；reader最终状态保留model/Adam/RNG/样本流/预算。

GPU7初始空闲后出现他人新进程，只向自有C发送可恢复退出；confirm_03第293步保存后恢复仅剩7步，前15场继承。J继续，随后C复用单张有余量共享卡，全部时间计费；没有模型失败后择优或丢弃科学结果。资源与源码小修见[engineering_notes.md](engineering_notes.md)，所有实际OS退出码见runtime。

|行/方法|完整query CPU wall中位 ms|encoder GPU ms|memory-read GPU ms|native/DPT GPU ms|
|---|---:|---:|---:|---:|
|C/Native|108.038|15.560|—|91.926|
|C/RasterFill|117.771|15.558|9.181|91.746|
|C/P|122.003|15.577|14.567|91.547|
|C/S|121.160|15.632|13.204|91.407|
|C/G|121.599|15.662|13.166|91.504|
|J/Native|108.848|15.655|—|92.802|
|J/RasterFill|117.615|15.634|8.841|92.705|
|J/P|122.237|15.556|14.358|92.067|
|J/S|122.121|15.643|14.061|92.027|
|J/G|123.146|15.597|13.015|93.291|

每行两固定训练scene各5warm＋15timed，Native/RasterFill/P/S/G共200真实native；120 reader就在这些完整query内，不另跑120reader。每次fresh图像encoder、真实GS raster、实际位置/reader、patch替换和原native/DPT，CUDA event分段与同步CPU wall分列，非独占速度主张。源几何/B/density/logΣ、ordinary与KV完整构建额外各行6次实测，无额外query benchmark。共同初始场另一次176个来源渲染补齐raw/spatial/bias的NumPy参考分解；这是固定源评价，不新增优化/条件或query输入，见source_initial_decomposition.json。来源fit分段计时从初始场/目标物化之后开始，完整GPU会话另含所有加载与间隙；不把分段和摊销已知项之和称采集端到端总时延。一次300步场形成、KV构建/序列化、冷逻辑加载/暖重载、模型加载及N=1/4/8/32摊销见[cost.json](cost.json)，32只是摊销示例。

旧prepared-source→GS构建成本和扩展来源已有测量只读继承并标明范围；sensor采集/原图预处理/部分预测器冷加载未知，未记为零。本次逐场拟合不是前馈构建；共享来源目标准备的80编码与24已有缓存依赖单列。GPU事件包括dispatch/计数间隙，不冒充孤立kernel时间，预处理RGB之后计query，不含JPEG解码或相机估计。

## 核对、图与交接

两行preflight包括同容量/同初值、零残差=RasterFill、中心mask之外不改、源KV重载、六pilot有限梯度和完整Adam/RNG恢复；独立评分为NumPy float64对全部新feature/depth及新覆盖mask，另有Torch float64比较记录，不重复旧模型推理或恢复旧数值门禁。Native在新覆盖分层下复算，身份和推理结果只读继承。两行形成与reader不新增数据/seed/损失/层数。

最终两图：[主与辅助/交互/代价](figures/outcomes.png)、[两行固定两例和各自最大反例](figures/cases.png)。实际view_image审阅记录[figure_review.json](figure_review.json)，完整指标而非图像决定证据。

请求gpt-6.1-sol/high/service_tier=fast，实际复用CLI UUID 01a1d350-d1ec-70f2-9e37-8960b2dfe7a8。请求及主client观察不等于本角色后台attestation；后台有效model/effort/tier未独立暴露，记unknown。未改全局配置、认证、系统GLIBC、主goal/state/events或创建/派发其他角色。完整结果交原Astra；不自行选后继或宣称总体目标已达，结束本CLI turn后释放写租约。
