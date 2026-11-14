# native_source_resolved_pe1：固定预算的来源分量与原生对应

2026-11-14，round028实现交付。唯一[科学合同](../autonomous_research_loop/round_028/science_plan.md)，SHA256 `19235eca9476843f1ac4965928dbbfe9c9c5c4068f11ae3ddbfaaf89f9e3dec4`。15 fits×1000更新、21120新末步预测完成；旧完整均值P/G、当前图像L共12672预测及Native2/4共2816预测／评分原身份继承，没有重训、重跑或新数据。全部数据均used development。

假设是同预算保留逐来源残差比提前均值更适合原生对应，并可能使footprint PE有强中心以外的增量；竞争解释是残差主要为view上下文／遮挡／偏移，损失通道分辨率不值得，或共同普通Σ与中心已足够。固定F/P、F/S、F/G、M/P、M/G及继承强参照，不调秩、权重、来源数、位置层、seed、步数、阈值或数据。

## 主要事实及范围

主PE F/G−F/P为0.156 [-0.718, 0.986]pp，未通过预定筛；F/P−M/P表示效应与PE交互的区间均跨零。相对Native4的有条件共同适配收益保留，但对旧完整均值P和L的净用途未建立；全部数据used，无独立确认候选。

| Used-dev original contrast | PCK8 difference pp [paired95%] |
|---|---|
| F_G-F_P | 0.156 [-0.718, 0.986] |
| F_P-M_P | 0.307 [-0.956, 1.706] |
| M_G-M_P | -0.164 [-1.631, 1.288] |
| F_G-M_G | 0.626 [-0.671, 1.778] |
| F_G-F_S | 0.553 [-0.494, 1.978] |
| F_S-F_P | -0.398 [-1.596, 0.795] |
| F_G-P | -2.499 [-7.423, 1.199] |
| F_G-G | -2.418 [-8.610, 1.689] |
| F_G-L | -1.176 [-5.068, 1.206] |
| F_G-Native2 | 13.872 [11.804, 16.006] |
| F_G-Native4 | 16.449 [13.457, 20.064] |
| F_S-P | -3.052 [-8.391, 0.873] |
| F_S-L | -1.729 [-5.876, 1.110] |
| F_S-Native4 | 15.896 [12.909, 19.321] |
| F_P-P | -2.654 [-7.845, 1.121] |
| F_P-L | -1.331 [-5.323, 1.226] |
| F_P-Native4 | 16.293 [13.394, 19.720] |
| M_P-P | -2.961 [-7.701, 0.786] |
| M_G-P | -3.125 [-8.202, 0.725] |
| M_G-L | -1.802 [-5.724, 0.797] |
| M_G-Native4 | 15.823 [12.597, 19.660] |
| (F_G−F_P)−(M_G−M_P) | 0.320 [-1.342, 2.114] |

| Method | PCK8 % [paired95%] | EPE px | Pair-tail95 px |
|---|---|---|---|
| F_P | 24.237 [21.192, 27.536] | 18.157 | 27.465 |
| F_S | 23.840 [20.539, 27.253] | 18.158 | 27.582 |
| F_G | 24.393 [21.088, 27.881] | 18.147 | 27.467 |
| M_P | 23.931 [20.254, 27.443] | 18.000 | 27.175 |
| M_G | 23.767 [20.429, 27.422] | 18.062 | 27.414 |
| P | 26.892 [21.430, 33.297] | 18.861 | 28.334 |
| G | 26.811 [20.926, 33.804] | 19.085 | 28.496 |
| L | 25.569 [21.490, 30.283] | 20.637 | 30.830 |
| Native2 | 10.521 [7.330, 14.117] | 21.709 | 30.701 |
| Native4 | 7.944 [5.828, 10.411] | 22.519 | 31.601 |

主PE F_G−F_P为0.156 [-0.718, 0.986]pp；表示F_P−M_P为0.307 [-0.956, 1.706]pp；表示×PE交互为0.320 [-1.342, 2.114]pp。原合同开发筛：F/G PE=False、对旧P/L/Native4净用途={'P': False, 'L': False, 'Native4': True}、候选=False；F/S次要候选=False。预定规则是PE均值≥1pp且95%下界>0，并对旧完整P、L、Native4净差区间均为正才形成开发候选；F/S同规则仅属次要，不能挽救G主比较。仅表示或连续误差改善保留内容用途，不替代O6；候选未立时结束此固定预算接法，后续科学取舍仍交原Astra，不追加扫描。标记不替代材料性尾部／成本权衡；没有fresh确认或总体目标达成。

共同内容表示、同表示PE及原生净用途分列。F/M是共同均值／质量匹配、F额外保留来源残差的学习链总效果，不是固定同一接收器的直接因果干预，也不能唯一归为同物理表面关联；F/G与M/G各自学习。旧P/G/L使用完整1024维内容而非新PCA初值，是保留较强完整信息的研究参照，不冒称严格新初始化配对。Native4没有GS来源sensor/pose特权，净用途只限该条件。

![主结果](/home/liwt/GRoPE/results/gaussian_memory/native_source_resolved_pe1/figures/primary.png)

主估计每锚/query对有效点先算PCK@8（448px），再锚、query、地点等权；无标签端点undefined，模型/初始化失败保留PCK0、EPE∞。5000次配对交叉重采样地点与种子，seed20261114，共同索引；不把点或view对当独立地点。主连续EPE差-0.010 [-0.181, 0.188]px、每view对95%尾部差0.002 [-0.258, 0.295]px，负值有利F/G，不能替代主PCK。所有dev/train、原/other/all8、原train12/added20、逐place/seed/query/anchor和尾部见[完整数表](report_tables.md)、[summary](scoring/summary.json)、[viewpairs](scoring/viewpairs.json)、[query](scoring/query.json)、[source](scoring/source.json)、[pooled tails](scoring/pooled_point_tails.json)。other/added不是fresh。

## 实际表示与原生链

原44份2048基元alpha绑定场只读，绝非C/J writer场；来源顺序0/3/6/9。只用32训练地点131072个原始source DINO token计算共同μ与正交PCA255，无whitening/query/标签/秩扫描；保留训练来源方差94.5007%。按原4view alpha权重定义总质量m_i、p_iv=m_iv/m_i、z_iv=Σ_u w_ivu Uᵀ(x_vu−μ)/m_i。F=(z_iv,p_iv)四块，M=(p_ivΣ_vz_iv,p_iv)四块，同4×256=1024；原has_content保持，零质量不补来源。88份FP16内容状态均source-only保存、重载和自身小文件身份校验；全局PCA共享常量也计依赖。

1045→1024映射共同初始化四个255列块各为U并相加、bias=μ；四质量列及其余几何列初始0。FP32学习adapter与Adam，原生BF16；数学初值相同，FP16保存／BF16运算可造成有限精度差。固定预检FP32映射F−M max/RMS={'max_abs': 0.000899195671081543, 'RMS': 1.8926125392653724e-05}，BF16={'max_abs': 0.03125, 'RMS': 0.0003173324854801604}；四次输出坐标差max/RMS=4.654602/1.050624px，loss差0.00375080。全部44场FP16解码差最大0.00167114、场级RMS最大0.00002235，逐场在[binding](binding)。实数匹配不等于位级输出相等；未另设数值门禁、未调初值或挑样本。该边界妨碍把F/M差异唯一解释成纯来源残差机制。

每组1,236,864参数：同映射、29→64→1024零输出位置MLP、query首frame/global rank8 QKV/proj LoRA。所有原native24对块、DPT、TrackHead、camera/depth冻结；first existing global读取只读GS，无新增层／reader／反馈。源锚prefix冻结，所有八query／四锚复用相同内容状态及全局adapter。Σ表示原基元支撑而非定位不确定性。普通完整logΣ六通道、opacity、强中心25维、原预测query相机、原2DRoPE／GS连续中心RoPE共享；仅四个K位置槽P强中心density／S同迹球／G完整footprint不同，不给G独占形状，V无位置项，零K/V背景保持。

官方两帧TrackHead真实四次坐标监督，等权SmoothL1 beta8；AdamW1e−4/wd.01/clip5，batch1，三个原seed20261110–12与配对1000sampler，固定step1000。2850个固定batch无有效GT，loss及梯度范数0，仍沿原AdamW动量／decay计step，不重采样、不称零跟踪误差；有效GT而loss0有0个，[全部记录](unsupervised_training_batches.json)。每seed五组初始化及sampler摘要完全相同，正式15000唯一更新无重做／丢弃；预检一次F/P Adam更新明确丢弃。

每个连续评价进程对模型/place内容仅加载一次并供四锚八query复用；恢复时必要的只读重载另计，不重建来源状态，查询不输入形成器、不写场；660份[依赖索引](public_states)引用88共同状态、15最终adapter、原几何和独立native源锚混合状态，而非660份独立场或逐scene参数。训练原128锚缓存与评价8锚LRU只保留不可变原生前缀，全部真实cache miss和加载进worker时间，未将cache命中冒称完整query成本。原相机、标签、GS普通字段与PE只读复用，不重新预测或渲染已有1408case。源sensor/pose标定／query预测相机、静态四源、有限标签与冻结范围都是本次实验选择，非用户核心要求。

## 覆盖、零支持和全部反例

原8×8网格64点不因conf/vis删选；source原448最近邻到210、原query210 sensor/pose与共同valid；双向深度容差max(.02m,.02depth)、两cycle≤1px，沿原合同不调整。1408view对共90112点，GT有效26003；零GT对354、raster零支持38、attention零键0；来源标定无效56、非有限预测相机/内参0。原所有零覆盖与fidelity_03稀疏地点权重保持；无GT/零支持不能宣称准确或无用。覆盖／每阶段拒绝和每点残差仍有原case身份，[coverage](scoring/coverage.json)和[条件支持分层](scoring/support_strata.json)不替代全体端点。

| Cohort | Queries | Defined/all pairs | Labelled/candidate points | Zero-GT pairs |
|---|---|---|---|---|
| dev12 | original | 112/192 | 2414/12288 | 80 |
| dev12 | other | 110/192 | 2432/12288 | 82 |
| train32 | original | 415/512 | 10011/32768 | 97 |
| train32 | other | 417/512 | 11146/32768 | 95 |


该覆盖表只连接不变的原label清单，不重复旧像素指标计算。

![覆盖](/home/liwt/GRoPE/results/gaussian_memory/native_source_resolved_pe1/figures/coverage.png)

| Contrast | Place | Seed | Anchor→query | PCK difference pp | GT points |
|---|---|---|---|---|---|
| F_G−F_P | fidelity_00 | 20261110 | 3→5 | -33.333 | 3 |
| F_G−F_P | fidelity_09 | 20261111 | 6→5 | -15.789 | 38 |
| F_G−F_P | fidelity_11 | 20261111 | 6→11 | -12.500 | 8 |
| F_P−M_P | fidelity_11 | 20261112 | 9→2 | -50.000 | 2 |
| F_P−M_P | fidelity_04 | 20261110 | 0→8 | -30.000 | 10 |
| F_P−M_P | fidelity_11 | 20261112 | 9→5 | -27.273 | 11 |
| F_G−L | fidelity_00 | 20261111 | 3→5 | -33.333 | 3 |
| F_G−L | fidelity_03 | 20261112 | 0→2 | -23.333 | 30 |
| F_G−L | fidelity_07 | 20261111 | 9→5 | -20.000 | 5 |
| F_G−P | fidelity_00 | 20261110 | 3→5 | -33.333 | 3 |
| F_G−P | fidelity_00 | 20261111 | 3→5 | -33.333 | 3 |
| F_G−P | fidelity_00 | 20261111 | 9→11 | -33.333 | 3 |
| F_G−Native4 | fidelity_04 | 20261110 | 0→11 | -25.000 | 4 |
| F_G−Native4 | fidelity_04 | 20261111 | 0→11 | -25.000 | 4 |
| F_G−Native4 | fidelity_04 | 20261112 | 0→11 | -25.000 | 4 |

全部方向的view对差与最大反例未截断地保留于[清单](scoring/all_pairwise_counterexamples.json)。图包括固定首开发例及最大开发F/G−F/P、F/P−M/P、F/G−L反例；无图像排除或视觉调阈值。sensor投影、遮挡与场景边界不等于独立真实物理轨迹。

![坐标与反例](/home/liwt/GRoPE/results/gaussian_memory/native_source_resolved_pe1/figures/tracking_cases.png)
![全部训练轨迹](/home/liwt/GRoPE/results/gaussian_memory/native_source_resolved_pe1/figures/training.png)

图已实际查看，具体观察见[figure_review](figure_review.json)。新21120预测由独立NumPy计算、Torch FP64复核PCK/EPE/median/tail／失败／无标签，最大绝对差5.68434188608e-14，覆盖21120记录；旧12672训练参照与2816Native评分直接连接原独立审计结果，没有重算旧像素指标。

## 来源、完整query与摊销成本

| Arm | Once/source binding s | Native anchor+calib s | Complete query s | At8queries s/query |
|---|---|---|---|---|
| F_P | 1.1576 | 0.5624 | 2.3978 | 2.6130 |
| F_S | 1.1576 | 0.5624 | 2.0061 | 2.2213 |
| F_G | 1.1576 | 0.5624 | 1.9172 | 2.1324 |
| M_P | 1.1576 | 0.5624 | 1.9494 | 2.1645 |
| M_G | 1.1576 | 0.5624 | 1.9706 | 2.1857 |

全局source-only PCA实际1.345178s，44场绑定合计16.093465s，均已计本轮GPUworker。逐场绑定定时含原场和来源特征读取、真实4view alpha raster、投影、F/M两份写盘／重载及数值披露；保守将两表示共同形成时间各计一次适用部署。绑定从既有四来源DINO token起点；其历史编码依赖由旧prepared-source formation中的fresh frozen encoder/lifting范围保留，本轮没有为44场重复编码。若部署没有token cache，必须支付四来源编码；表中binding秒数不能代替原始RGB端到端形成。不是反复benchmark。原GS physical formation没有重跑，也不免费：其旧来源预测／拟合／绑定warm计时范围与未知采集、预处理、冷加载在[cost_inherited](cost_inherited.json)原样保留，不能用小warm数充作全端到端形成。

唯一成本案例fit00/a0/q2、首seed，从既有448 RGB张量重新DINO编码，真实source native前缀＋DPT＋camera/depth/sensor标定；query重新编码、初始完整native query＋camera、真实GS raster和字段、第二完整带记忆native query＋DPT＋TrackHead。2次DINO、1source frame、1初始化query、5适配query／5四迭代TrackHead、1raster；无新增Native2/4/P/G/L成本调用。共享阶段执行一次并向各适用pipeline计账；F/M状态各一次加载、PCA／binding／源prefix／query／模型与adapter冷加载单列。1/4/8摊销包括每场一次binding及载入、native anchor形成，训练共同PCA全局开销单列，原GS形成另列，绝不当作免费。因此表中摊销是已存在原场／来源token条件下的新形成加混合锚与完整query成本，不是完整raw RGB→状态→query的总延迟。

模型／输入的cold标签仅指新进程载入，OS页缓存和驱动／kernel缓存未清空，不是uncached磁盘／驱动启动benchmark。固定顺序F/P,F/S,F/G,M/P,M/G，首次TrackHead/kernel初始化与warm缓存不能单独辨识；只一个案例，不推断稳定态速度排序或独立cold-first-GS延迟。成本卡准入独占、其他卡现场见[cost_dispatch](cost_dispatch.json)；不是服务器隔离benchmark。原raw图像decode/resize、sensor获取不在448张量起点内；这里的重新编码不是fresh研究数据。旧Native2/4/P/G/L成本只作为带原范围的历史参考，不是与新五组严格同时的延迟比较。

单锚依赖逻辑KV 96.469MiB、实际底层QKV backing 144.703MiB，DPT 24.500MiB，source camera KV backing 0.250000MiB与公制对齐常量52bytes也必须保留；F/M内容每份约4.002MiB，全局PCA 2.008MiB，原几何／源中心依赖0.614MiB。原完整均值f在新成本read中去除，不作模型输入；存量原场文件仍含该未使用均值，不能把其几何张量字节当作共享文件物理体积。以上计数是一锚形成后的运行态张量，不是新落盘的纯GS历史：native prefix/DPT实际由只读source-DINO cache重建，未写新的prefix文件；660依赖索引补全该源cache路径。[actual_cost](actual_cost.json)另列原source-DINO/RGB/sensor/场文件的实际路径和字节。已有重建输入的存储／形成不免费。全局adapter参数约4.718MiB，完整Adam是训练恢复依赖而非部署必要项；全局native权重和query临时激活另列，不能由4MiB内容文件称整个混合历史已经压缩成功。

## 实现、恢复、实际资源

五个真实预检query/backward的source/位置输出／两LoRA梯度有限非零，native参数与源prefix只读；一次F/P完整参数＋Adam＋CPU/CUDA RNG＋1000sampler精确恢复。[preflight](preflight.json)记录所有实际梯度和有限精度差，未重复旧原生禁用记忆／空支持检查，引用其不变接口的[原验证](../native_track_memory_pe1/preflight.json)。最初CPU在受限文件系统写DATA失败exit1、0GPU/模型调用；随后使用已授权host环境正常构建，记录[startup001](engineering_revisions/startup001.json)。GPU4随后出现外部高利用率任务，F/P评价保存34marker、M/P训练保存完整step8后actualwait75。旧FIFO同卡补位逻辑被队首待恢复评价阻住；版本化修复为按每卡选择兼容pending项，以及同一训一评容量的双向补位。仅SIGTERM自有manager114384，旧owner22661 actualexit75、所有child实际wait；F/S已133评价、F/G204评价、M/Gstep139一并保存。新唯一manager89284/owner24252按完整状态只补缺失；恢复时GPU6也已有外部任务，未当空卡使用。M/P从9继续、F/P从第35个未提交case继续，不重做有效训练／已提交预测；所有加载、恢复与争用均进worker账。[head_of_line002](engineering_revisions/head_of_line002.json)保留代码前版本／身份和实际退出。GPU4两次短空隙又使恢复任务仅前进到M/G140、141就礼让，所有冷载计入预算。随后不再刷新正常6/7任务，增加CPU-only guard持续观测GPU4，外部活动后稳定空闲120秒才在未来worker导入torch／CUDA／大模型前放行；冷却内CPU快速exit75、0模型调用记录保留，pool墙钟仍保守计入预算，详见[admission_guard003](engineering_revisions/admission_guard003.json)。已知自有正常阶段切换不机械重置等待；新外部活动才重置该卡。guard没有作业派发权，fits完成自动终止。后来实际发现短外部PID可在5秒观察间隔内出现后消失，未来worker的pre-CUDA准入遂同时读取同卡external_PID_arrived的actualwait收据，以最近退出为120秒冷却下界，覆盖GPU7同类事件，并持续观测全部物理卡的外部活动；未重启manager89284或中断正常worker，[foreign_receipt007](engineering_revisions/foreign_receipt007.json)保留代码版本。一次16.7秒只读利用率观察中，GPU6连续5–84%、均值26.25%，两worker各用约1.54／1.68个CPU核、物理磁盘读取0、总写入约200KiB；短时低util包含主机发射／同步与小算子突发，未用额外模型或benchmark细分其占比，也未发现足以安全改变live数值路径的明显优化，见[utilization006](engineering_revisions/utilization006_live_observation.json)。其余普通失败与礼让以[worker](workers)、[工程记录](engineering_revisions)实际发生为准，不把失败预检记成功。

唯一动态队列按实时CUDA/PID、显存／RAM／空间准入所有安全空卡，无默认双卡上限；foreign PID到达只礼让本轮进程，在当前有效更新／case原子提交后保存完整checkpoint并actualwait，永久final/已提交marker不重复。一个已final的约5.7GB evaluator可与独立待训约29.4GB fit在有余量的自有安全卡共存，两份训练缓存不共卡；不以余显存冒称他人高利用率卡空闲。运行中GPU4曾暂为两个自有evaluator，静态一训二评估计40813/48671MiB、余7858MiB，比当前6144MiB底线多1714MiB；但已加载manager没有安全动态三进程接口，未为改并发而重启／打断有效工作，也未额外benchmark。此为本次在线调度选择，不是硬件装不下或默认双卡限制，[concurrency008](engineering_revisions/concurrency008_live_decision.json)保留判断。后来0–3卡实际释放并连续安全空闲超过200秒，原cost单例先在GPU4完成实际27.651857worker秒、worker130826与owner70765均actualexit0；随后仅对已训练完成的两个evaluator在case边界SIGTERM，旧69822／118942实际wait75，同一89284改派GPU0／9403与GPU1／9405，从各自682／474已提交case之后续接，原GPU6／111661和GPU7／119327两次训练未中断。没有新增拟合、评价重算或benchmark；余下空卡没有独立既定任务便不造工作，[eval_redistribution009](engineering_revisions/eval_redistribution009.json)保留完整实际收据。其后6/7又出现新外部任务，末两M模型评价分别在738／568已提交case处安全退出75，15份step1000永久final不变；当时全8卡均为外部高负载，原唯一manager等待安全资源，GPUworker计时保持不变。等待中先在CPU导出并实际查看完整训练图，后续仍只补缺失，详见[foreign_all_cards011](engineering_revisions/foreign_all_cards011.json)及实际worker收据。所有外部工作和旧结果保持，旧GLIBC不动、无系统安装／下载。

总GPUworker **12.884227h /18h**，含加载、PCA／绑定、失败、预检、恢复、缓存重建、完整训练／评价／成本，所有524子worker实际wait返回。新DATA最终0.6532GiB、含并发原子暂存保守峰0.9032GiB /8GiB；报告写出时项目含源码0.1335GiB、保守峰0.1960GiB /1GiB。完整Adam checkpoint最大14.185MiB，15永久final保留，无自有RAM文件系统暂存或隐藏副本。失败worker未采全峰值明记unknown。[完整调用／资源账](runtime_budget_calls.json)含PID/卡/实际wait、checkpoint反传重计算、恢复和唯一更新；CPU评分／图／交付实际资源另列（[评分](CPU_scoring_runtime.json)、[图末次](CPU_figure_runtime.json)，先前排版导出时间另存pass1／pass2与early记录）；其PID为sandbox进程空间，评分host PID35827另行观测，不能混作host PID2。封存后的项目总字节含manifest／handoff自身，见完整资源账。476次准入延后在torch/CUDA导入前快速退出、0模型调用，其pool墙钟仍保守计入上述预算。最终终端退出确认见[actual_terminal_confirmation](actual_terminal_confirmation.json)。

## 大纲与交回边界

O2/O3公共粗GS、O4/O5真实多query读取和O6同信息强中心比较保持；仅C2逐来源表示实质改变，C1/C3/C4/C5/C6匹配复用。PCA255、2048×1024、静态四源、混合原生锚前缀、来源sensor/pose特权及旧有限标签是实施选择。round027旧完整均值家族关闭、历史区域内容K证据／R替换未决、depth/pose/writer/address等正负与反例不改；round026旧manager停止触发源unknown及旧成本单例固定顺序边界沿原报告保留，没有追认原因或额外复测。本轮不证明准确物理Σ、纯GS替代历史、一般PE有效／无效、fresh或总体成功。只有本合同事实交主→原Astra作集中科学取舍，不自选后继或追加训练。
