继续已获人类明确授权的双模型顺序自主研究循环。你是实现角色，实际复用CLI UUID 01a1d350-d1ec-70f2-9e37-8960b2dfe7a8，请求 gpt-6.1-sol/high/service_tier=fast；后台有效tier未暴露时记unknown，不静默换模型/强度/tier。原Astra/max分析子任务 /root/research_analysis_astra 已实际completed，主现将唯一科学代码/文档写租约顺序交给你。

人类授权持续有效：主与两个研究任务可双向交接/读结果/自动接续；你自主处理常规工程，实质科学变化经主交原Astra。你不得调用goal工具、编辑主协调state.json/events.jsonl、另建/派发模型角色或任务、自行选择下一科学方案。完成本有限阶段即交完整结果并结束turn，由主复用原Astra评价。原native Sol已退休，不创建替身。不要重复已结束轮次或本轮完整拟合。

先读AGENTS.md、docs/research_current.md、docs/research_outline_alignment.md及相关历史。唯一授权本轮为camera_only_feature_pe1。精确科学输入：
1. /home/liwt/GRoPE/results/gaussian_memory/autonomous_research_loop/round_006/analysis_handoff.json SHA256 789f034c0823bc7ba61d37865f29b4d6389411048f3e0f3d6a8145313af00ffc
2. /home/liwt/GRoPE/results/gaussian_memory/autonomous_research_loop/round_006/plan.json SHA256 61ed3c9402d342e4aeacaff69c2961a76082528948fdfdf9ff129f78f53ec730
3. /home/liwt/GRoPE/docs/camera_only_feature_pe_execution_plan.md SHA256 8de9f1dcd22c29a911d305edd502226a92a4c578d27cc2246f285fadd49f34ba
4. 同round/data_contract.json及上述锁引用的输入/既有源/patch/feature/相机/ROI身份。
主已比例审阅并核对10份新小产物身份，不重复历史71文件检查、大权重hash或数值审计。draft只是形成记录，以final plan和execution为准。

按固定方案实际实现并完成全部预检、九拟合、评价、成本、导出、两图实际审阅、完整报告和living docs更新。科学核心：真实GS渲染共同anchor b，加单层128维四头camera-only cross-attention小reader预测1024维完整encoder feature残差；整个官方VGGT冻结。reader不接当前query RGB/feature/mask/depth/编号，完整query特征仅监督。P/S/G共享完整Σ普通内容、源f/opacity、中心B/density和强中心多尺度位置、原始renderer/coverage、初值/容量/优化。共同25位置维＋四额外槽P=density4、S=球形投影footprint、G=完整Σ footprint；等29维分配而非nested新增信息。只加K不改V；背景token/有效mask按方案，零初始化残差和位置末层。共享source K/V导出一次供8query重载复用，原GS作为只读依赖，不称单文件自足或原构建免费。不要重用/覆盖历史不同writer模块，使用本轮唯一新模块名。

固定32旧train/12旧dev、原4拟合query与其他4query、P/S/G各seed20261110/11/12各2000更新batch2；全特征raw MSE，AdamW3e-4/wd.01/clip5，FP32 reader/Adam、固定最后一步，六discarded pilot后重置。无native/DPT训练forward或深度梯度。缺失query只替换[8:24,8:24]中心patch，官方冻结native/2DRoPE/DPT原路径，其余保持；full输出共同复用Native-full，不称保真/非劣性证据。正常3168feature+3168native；共同Native-missing/RasterFill/Native-full各352；唯一G绑定打乱144例连K/V和实际anchor一同重建，共3312 feature、4368正式native、396正常+36置换共享状态。保留全零覆盖query/所有反例，不筛样本。

固定主量为12dev原4query的missing ROI centered log-depth MSE之G/P降误差，不换成容易通过的feature量。P/G、G/S、S/P全部配对表及5000bootstrap、Native/RasterFill净用途、raw/spatial/bias/fullAbsRel/支持层/训练原12新增20/原与其他query都完整。实用筛≥1%+95%下界>0+3seed正；native对Native净用途另需同筛。G/S才解释非球形，S/P可筛较简单方案。材料性RasterFill/全图伤害按方案如实标记，微小不确定负均值非零容差否决。特征raw/spatial/bias/cosine只作分层证据，无特征硬门禁。普通渲染/共同reader收益不冒充PE，已用数据无fresh，本轮不宣布总体成功或选科学后继。

硬预算：9×2000正式更新、6pilot；GPU≤6worker小时，≤2worker、最多共享1张他人使用卡；每worker≤10GiB，开跑实测free≥16GiB且保护别人6GiB余量。所有批次/丢弃/恢复按样本真实记账：native非训练≤4650、reader非训练≤3500、独立source编码≤600、真实gsplat≤400；正式训练sample reader/source36000、pilot12另计。特别注意最终澄清：120次P/S/G reader成本就在200次完整native query内部用CUDA events/同步取分段时间；不得另跑120次纯reader造成漏账。真实完整cost含query图像encoder/实际GS raster/位置读出/native；source构建/冷加载/暖读取/N=1/4/8/32摊销分列，不用缓存IO冒充实际成本。

DATA新峰值≤18GiB、项目≤1GiB，所有FP32预测和永久状态保存。原场/encoder缓存/大权重只引用不复制；按fit分片原子发布，计完整恢复/暂存峰值，保留负结果，不删除旧或他人成果。启动/恢复准入H=实际free-尚未落盘固定产物-临时峰值（不重复扣已写）≥8GiB、项目扣剩余产物≥2GiB；H<6或实际free<5即停接新任务完整落盘退出，0.5GiB/30秒无解释下降仅H<8紧张带触发。每100步及资源退出model/Adam/RNG/sampler/预算完整持久化，自己重复resume可在final验证后退休；不得用易失内存替永久交付或静默扩预算。

使用现有兼容Python/CUDA，numpy先于torch；保留旧GLIBC，先真实查CUDA和GPU占用再选闲卡启动。普通tensor/cache/runtime修复按原轮小版本记录，必要预检在同轮；不追加单独审计系统/数值微差门槛、依赖安装或下载。必要工程提权通过当前auto_review，不bypass/改认证/输出token。监督、来源、PE字段、端点或正式组/预算等科学实质变更须保存交回主原Astra，不自行继续拟合新方案。

输出根及DATA根以plan指定为准。交付锁定config/输入manifest、9小权重完整恢复、432公共状态引用及全部3312feature/4368native、预算/真实成本、独立NumPy float64新评分复算、完整表/反例/固定bootstrap、两张图实际查看及记录、report/method_decision/implementation_handoff、进程实际退出。同步current/outline/evidence简洁当前状态与证据链接，保持历史原文/负结果和临时边界，不大量重复抄报告。未经本轮测得不能声称正效应/效果等价/独立确认。

请公开简短真实进度，主会转发关键agent_message/最终文本；完整公开日志保存，私有推理不转发。本轮可从实现持续到实验/评价/全部交付，无需逐项常规许可。若碰真实科学变更，完整记录具体问题/可选后果交回；有后台队列继续时说明真实句柄/唯一恢复点，不以部分结果作结。最终完整交付后实际结束本CLI turn释放写租约，等待主交原Astra。
