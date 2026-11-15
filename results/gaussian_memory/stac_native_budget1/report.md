# stac_native_budget1：真实作者政策、强内容与原生窗口

2026-11-15，round049完整实施[唯一科学合同](/home/liwt/GRoPE/results/gaussian_memory/autonomous_research_loop/round_048/science_plan.md)，SHA `09a6c033f2442a917f50b3c12791b3500687a9b993d0265d4ca9d89fc83f12ec`。**主S/H重访旋转改善 E_H−E_S=-1.6314 [-3.6151, -0.2662]°**；S/W净用途和深度代价独立列出。12地点×96帧×三政策全部完成，0训练／数据或权重下载；全部used development，不称fresh、采用、总体成功或自动后继。

| 比较（正值误差下降） | 重访旋转° [95%配对区间] | 全帧AbsRel改善 [区间] | 预定旋转flag | 明显AbsRel代价flag |
| --- | --- | --- | --- | --- |
| S_vs_H | -1.6314 [-3.6151, -0.2662] | 0.0089 [0.0036, 0.0152] | False | False |
| S_vs_W | -1.6314 [-3.6151, -0.2662] | 0.0089 [0.0036, 0.0152] | False | False |
| H_vs_W | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | False | False |

旋转均值≥1°且配对下界>0、AbsRel增加≥.01且区间支持退步是合同的开发投入标记，不是普适数值门槛、等价或非劣性证明。S/H为主，S/W为原生净用途；仅胜窗口不能建立空间政策的额外价值。政策自身反馈、源位置和相机共同变化，本比较不唯一归因于几何或Gaussian／PE。科学处置交原Astra；[主控制总体目标](../autonomous_research_loop/goal_revision_after_round042/revised_goal.md)及全部历史关闭保持。

S相对H/W的全帧AbsRel改善+0.008912 [0.003580,0.015169]；raw log-MSE改善+0.096687 [0.025322,0.186421]主要来自偏差平方改善+0.097143，去均值结构改善为−0.000456 [−0.000812,−0.000168]。全帧锚旋转／中心及绝对GT几何退步，重访平移／绝对几何区间跨零；这些正负证据均保留，深度辅助不替代主旋转。

## 固定作者、输入和真实计算路径

固定 Rainzor/STAC commit `fd7e718597cf9963de85c8fffae32a698e8619f5`，原CUDA attention及segment merger；没有换Python参考融合。源码归档15,096,985bytes，仅同commit源码，原918Gitblob逐一匹配；仅C++ wrapper header/bindings增加完整状态观察／恢复接口，算法kernel不改。实际现有官方checkpoint加载missing=[]、额外仅未用track_head；1,190,596,120参数／4,762,384,480权重内存bytes。继承权重SHA `3d3d914d09609fdb485e16947c4885cca0e5ea4e5557598a3b3ffb1cfe0a2f59`，没有重新哈希大权重。[源码身份](source_identity.json)、[真实加载与首帧无压缩对应](load_validation.json)记录兼容边界，不把静态推断冒称加载。

| 首固定帧：作者vs本地官方 | meanAbs | maxAbs |
| --- | --- | --- |
| depth | 0.000240012 | 0.060706019 |
| pose_enc | 0.000031106 | 0.000170946 |
| world_points | 0.000125365 | 0.122002244 |

同原始共享权重和BF16路径仍有上述有限精度差；对应检查不是逐位等价保证，也未建立跨后端精度排名。没有把简化体素称完整STAC。本轮已实际运行作者原CUDA merger与默认激活读取；原round047静态未知及定位边界见[源事实报告](../stac_route_review1/report.md)，本轮运行证据按新范围补充，不覆写旧报告。

| 组 | 政策 | 共同读取／相机 |
| --- | --- | --- |
| S | 作者recent4、HH2、首帧；retrieval2、体素.05、buf8／pivot4、阈值.8/.8/.2、segment原合并 | 全部24global，chunk4，同sourceRoPE、原有效BF16 KV；作者point必算 |
| H | 首帧+recent4+HH91，原作者attention colsum／更新，最大合法容量 | 无空间库，point省去；其节省可增加容量 |
| W | 首帧+recent95完整帧，最大合法窗口 | 同原attention输出kernel，无HH／空间库，point省去 |

传入recent_size=4；作者STACVoxelKV第109行将self.recent_size日志元数据覆盖成retrieval_size=2，但派生recent_token_size仍为4×1041。按真实作者执行，未为日志数字修政策。原1041=5特殊+1036patch，来源位置与特殊token未删；S使用真实代表count-bias、原相似度排序与几何读取次序，不视为空间均值替代。

所有组kf_every=1，原12地点每处96独立稀疏RGB，不再stride10。每4帧联合处理，最多看到chunk内后3帧；相机对全已到达pose-token前缀重新四轮精化。对目标t和来源s取包含t的chunk提交时同一前缀姿态，深度取自身chunk输出；不用最终96姿态回溯评分。跨时点几何及首帧刚体gauge也取该提交前缀，无GT尺度拟合。此输入权限不同于旧逐帧实验，旧W分数既不复用也不重算。GT从未写入模型／政策／容量。

## 先S成本、再容量、最后GT

先完成12条S，仅记录预测／成本。B在2026-11-15T00:17:52.002979+00:00、任何GT评分前锁定：实际S最大峰15,362,249,910bytes按64MiB向上取整，**B=15,367,929,856bytes (14.312500GiB)≤16GiB**。原旧B不被替换；这是固定96范围的预算点，不是任意长流全局有界部署。

| 组 | HH／recent容量 | hot帧 | 最大reachable reserve | 合法生命周期界GiB | 实际历史峰GiB | B未用GiB |
| --- | --- | --- | --- | --- | --- | --- |
| H | 91 | 96 | 96 | 11.796095 | 11.754534 | 2.557966 |
| W | 95 | 96 | 96 | 11.510176 | 11.446300 | 2.866200 |

S实际峰14.307210GiB。H/W完整历史条件=True；若两者保留全部96，HH没有淘汰选择，不能以假想最后gather压小普通对照。即使同输出，H仍维护attention scores，额外成本保留；本轮两种原政策分别实际执行，未把别名虚报为额外独立证据。

生命周期三组共同固定首地点prefix8完整恢复。H/W物理buffer4→8→有限96可达最大reserve，不改变逻辑容量、HH打分／选择、KV精度或来源。一次扩展逐tensor释放旧buffer，真实旧／新副本重叠计入；全部可达整数与hot96不触发prune的非单调边界前GT计算。旧未使用的v1全96假想恢复上界仅压控制不公平，已在控制启动前撤销并保留版本。不设无用horizon填充，不把恢复假设当科学条件。后续外部资源礼让若发生仍独立遵守真实B，不能免费恢复或暗减容量。

历史计费含GPU／CPU完整untyped backing、原全RGB可读文件、原作者CPU images/depth/conf及S point archive、pose-token view背后的完整聚合输出、前缀相机快照、索引／score／count／位置／体素／overflow／retrieval、CUDA原生workspace、Python对象、checkpoint与原子输出／chunk数值备份重叠。真实别名计一次；权重和当前推理激活另报VRAM，历史专属workspace不免费。作者所有CPU archive在流结束前保留且计费；最早9个永久文件保留全archive，后续v2只保留科学depth／prefixposes／costs，均不反馈未来预测。

## 主端点、NA与逐地点

原固定63远端目标t=33…95，s≠0且t−s≥32，原GT min-z共视配对及5%重访身份不变。地点先平均，再对有定义地点等权；5000次原12地点ID配对bootstrap、seed20261114，NA不填0，保留所有留一地点结果。原selected-pair独占和all-far-union独占均扣原首帧+recent7参考支持；仅分层描述，不偷换为本轮有利窗口。

| 地点 | 重访/63 | S° | H° | W° | H−S° |
| --- | --- | --- | --- | --- | --- |
| brown_bm_1 | 3 | 0.7450 | 0.3789 | 0.3789 | -0.3661 |
| brown_bm_4 | 0 | NA | NA | NA | NA |
| brown_cogsci_1 | 14 | 15.7182 | 15.8166 | 15.8166 | 0.0984 |
| brown_cs_3 | 0 | NA | NA | NA | NA |
| harvard_robotics_lab | 59 | 3.3873 | 3.4518 | 3.4518 | 0.0645 |
| home_at | 0 | NA | NA | NA | NA |
| home_bksh | 4 | 5.6675 | 4.7776 | 4.7776 | -0.8899 |
| home_md | 8 | 13.4834 | 5.6751 | 5.6751 | -7.8082 |
| mit_32_d507 | 42 | 4.3533 | 3.4811 | 3.4811 | -0.8721 |
| mit_32_g725 | 44 | 3.6336 | 3.4125 | 3.4125 | -0.2211 |
| mit_46_6conf | 9 | 18.9554 | 14.5740 | 14.5740 | -4.3814 |
| mit_w59_conference | 6 | 28.7067 | 28.4003 | 28.4003 | -0.3063 |

全1152帧、756远端目标，189重访／9有定义地点；3地点主NA、276零共视、291独占零覆盖均保留。主留一地点均值范围-1.8476…-0.8593°；GT近零平移仅方向辅助NA，0/756，不删旋转。真实远端时距12.33…641.76秒。

![固定比较及逐地点](/home/liwt/GRoPE/results/gaussian_memory/stac_native_budget1/figures/primary.png)

## 全部原生辅助、深度与支持

| 全96帧地点等权 | S | H | W | H−S [CI] | W−S [CI] | W−H [CI] |
| --- | --- | --- | --- | --- | --- | --- |
| trajectory_rotation_anchor_deg | 36.8785 | 32.6150 | 32.6150 | -4.2635 [-8.9691, -0.5428] | -4.2635 [-8.9691, -0.5428] | 0.0000 [0.0000, 0.0000] |
| trajectory_center_error_raw_m | 4.3800 | 4.3052 | 4.3052 | -0.0749 [-0.1178, -0.0353] | -0.0749 [-0.1178, -0.0353] | 0.0000 [0.0000, 0.0000] |
| absolute_worldpoint_GT_error_raw_m | 5.0215 | 4.9458 | 4.9458 | -0.0757 [-0.1154, -0.0356] | -0.0757 [-0.1154, -0.0356] | 0.0000 [0.0000, 0.0000] |
| absrel | 0.7409 | 0.7498 | 0.7498 | 0.0089 [0.0036, 0.0152] | 0.0089 [0.0036, 0.0152] | 0.0000 [0.0000, 0.0000] |
| log_mse | 2.0054 | 2.1021 | 2.1021 | 0.0967 [0.0253, 0.1864] | 0.0967 [0.0253, 0.1864] | 0.0000 [0.0000, 0.0000] |
| centered_log_mse | 0.0089 | 0.0084 | 0.0084 | -0.0005 [-0.0008, -0.0002] | -0.0005 [-0.0008, -0.0002] | 0.0000 [0.0000, 0.0000] |
| bias_squared | 1.9965 | 2.0936 | 2.0936 | 0.0971 [0.0257, 0.1870] | 0.0971 [0.0257, 0.1870] | 0.0000 [0.0000, 0.0000] |
| RMSE_m | 2.0687 | 2.0875 | 2.0875 | 0.0188 [0.0073, 0.0318] | 0.0188 [0.0073, 0.0318] | 0.0000 [0.0000, 0.0000] |
| failure | 0.0000 | 0.0000 | 0.0000 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |

raw log-MSE=去均值结构+偏差平方；AbsRel、RMSE、坏像素及真实失败并列。无GT尺度或仿射对齐。跨时一致性与绝对GT几何、GT采样discrepancy同时保留，避免塌缩伪装一致性。辅助正例不替换主旋转，物理尺度准确性／PE／各向异性未由本轮证明。

| 目标集合 | 辅助指标 | S/H | S/W | H/W |
| --- | --- | --- | --- | --- |
| all63far_targets | relative_rotation_deg | -5.7660 [-12.6099, -0.4617] | -5.7660 [-12.6099, -0.4617] | 0.0000 [0.0000, 0.0000] |
| all63far_targets | relative_translation_direction_deg | -2.9827 [-4.9513, -1.3699] | -2.9827 [-4.9513, -1.3699] | 0.0000 [0.0000, 0.0000] |
| all63far_targets | relative_translation_vector_error_raw_m | -0.0542 [-0.0923, -0.0183] | -0.0542 [-0.0923, -0.0183] | 0.0000 [0.0000, 0.0000] |
| all63far_targets | cross_time_point_consistency_raw_m | -0.0173 [-0.0680, 0.0313] | -0.0173 [-0.0680, 0.0313] | 0.0000 [0.0000, 0.0000] |
| all63far_targets | paired_absolute_GT_geometry_error_raw_m | -0.0397 [-0.1122, 0.0390] | -0.0397 [-0.1122, 0.0390] | 0.0000 [0.0000, 0.0000] |
| all63far_targets | failure | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| revisit_coverage_ge5pct | relative_rotation_deg | -1.6314 [-3.6151, -0.2662] | -1.6314 [-3.6151, -0.2662] | 0.0000 [0.0000, 0.0000] |
| revisit_coverage_ge5pct | relative_translation_direction_deg | -2.4444 [-11.0314, 4.6285] | -2.4444 [-11.0314, 4.6285] | 0.0000 [0.0000, 0.0000] |
| revisit_coverage_ge5pct | relative_translation_vector_error_raw_m | -0.0141 [-0.0460, 0.0169] | -0.0141 [-0.0460, 0.0169] | 0.0000 [0.0000, 0.0000] |
| revisit_coverage_ge5pct | cross_time_point_consistency_raw_m | 0.0016 [-0.0636, 0.0669] | 0.0016 [-0.0636, 0.0669] | 0.0000 [0.0000, 0.0000] |
| revisit_coverage_ge5pct | paired_absolute_GT_geometry_error_raw_m | 0.0035 [-0.0255, 0.0395] | 0.0035 [-0.0255, 0.0395] | 0.0000 [0.0000, 0.0000] |
| revisit_coverage_ge5pct | failure | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| nonrevisit_lt5pct | relative_rotation_deg | -5.9559 [-12.8623, -0.4383] | -5.9559 [-12.8623, -0.4383] | 0.0000 [0.0000, 0.0000] |
| nonrevisit_lt5pct | relative_translation_direction_deg | -2.5322 [-4.8227, -0.6119] | -2.5322 [-4.8227, -0.6119] | 0.0000 [0.0000, 0.0000] |
| nonrevisit_lt5pct | relative_translation_vector_error_raw_m | -0.0652 [-0.1063, -0.0288] | -0.0652 [-0.1063, -0.0288] | 0.0000 [0.0000, 0.0000] |
| nonrevisit_lt5pct | cross_time_point_consistency_raw_m | -0.0241 [-0.0792, 0.0295] | -0.0241 [-0.0792, 0.0295] | 0.0000 [0.0000, 0.0000] |
| nonrevisit_lt5pct | paired_absolute_GT_geometry_error_raw_m | -0.0489 [-0.1219, 0.0340] | -0.0489 [-0.1219, 0.0340] | 0.0000 [0.0000, 0.0000] |
| nonrevisit_lt5pct | failure | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |

![raw、去均值与偏差](/home/liwt/GRoPE/results/gaussian_memory/stac_native_budget1/figures/depth.png)

| 原支持定义（重访） | 层 | 记录 | 地点 | S/H° | S/W° | H/W° |
| --- | --- | --- | --- | --- | --- | --- |
| selected_pair_exclusive | 0 | 6 | 1 | 0.4108 [0.4108, 0.4108] | 0.4108 [0.4108, 0.4108] | 0.0000 [0.0000, 0.0000] |
| selected_pair_exclusive | (0,5%] | 136 | 9 | -1.6035 [-3.4096, -0.3783] | -1.6035 [-3.4096, -0.3783] | 0.0000 [0.0000, 0.0000] |
| selected_pair_exclusive | (5,20%] | 44 | 9 | -1.6674 [-3.8158, -0.1549] | -1.6674 [-3.8158, -0.1549] | 0.0000 [0.0000, 0.0000] |
| selected_pair_exclusive | >20% | 3 | 2 | -0.0763 [-0.6151, 0.4624] | -0.0763 [-0.6151, 0.4624] | 0.0000 [0.0000, 0.0000] |
| selected_pair_exclusive | undefined | 0 | 0 | NA | NA | NA |
| all_far_union_exclusive | 0 | 5 | 1 | 0.4445 [0.4445, 0.4445] | 0.4445 [0.4445, 0.4445] | 0.0000 [0.0000, 0.0000] |
| all_far_union_exclusive | (0,5%] | 84 | 7 | -0.8544 [-1.5271, -0.2204] | -0.8544 [-1.5271, -0.2204] | 0.0000 [0.0000, 0.0000] |
| all_far_union_exclusive | (5,20%] | 78 | 8 | -3.0084 [-6.9519, -0.1028] | -3.0084 [-6.9519, -0.1028] | 0.0000 [0.0000, 0.0000] |
| all_far_union_exclusive | >20% | 22 | 8 | -0.8246 [-1.5638, -0.2909] | -0.8246 [-1.5638, -0.2909] | 0.0000 [0.0000, 0.0000] |
| all_far_union_exclusive | undefined | 0 | 0 | NA | NA | NA |

全部756与重访子集每层全部指标／配对区间在support_strata_paired.json，0／undefined不删。不据层内相关调整阈值或作因果结论。

| S/H最不利重访行 | t/s | 覆盖 | 独占 | 秒 | 改善° |
| --- | --- | --- | --- | --- | --- |
| fidelity_07 | 86/52 | 0.096 | 0.086 | 160.24 | -19.3457 |
| fidelity_07 | 87/52 | 0.097 | 0.042 | 162.25 | -14.5617 |
| fidelity_10 | 79/45 | 0.064 | 0.064 | 38.14 | -13.3150 |
| fidelity_07 | 89/52 | 0.140 | 0.072 | 171.70 | -12.2247 |
| fidelity_10 | 92/58 | 0.175 | 0.040 | 38.11 | -11.6166 |

| S/W最不利重访行 | t/s | 覆盖 | 独占 | 秒 | 改善° |
| --- | --- | --- | --- | --- | --- |
| fidelity_07 | 86/52 | 0.096 | 0.086 | 160.24 | -19.3457 |
| fidelity_07 | 87/52 | 0.097 | 0.042 | 162.25 | -14.5617 |
| fidelity_10 | 79/45 | 0.064 | 0.064 | 38.14 | -13.3150 |
| fidelity_07 | 89/52 | 0.140 | 0.072 | 171.70 | -12.2247 |
| fidelity_10 | 92/58 | 0.175 | 0.040 | 38.11 | -11.6166 |

| H/W最不利重访行 | t/s | 覆盖 | 独占 | 秒 | 改善° |
| --- | --- | --- | --- | --- | --- |
| fidelity_00 | 93/1 | 0.455 | 0.057 | 211.02 | 0.0000 |
| fidelity_00 | 94/2 | 0.458 | 0.100 | 211.05 | 0.0000 |
| fidelity_00 | 95/2 | 0.399 | 0.007 | 213.06 | 0.0000 |
| fidelity_02 | 79/40 | 0.052 | 0.044 | 39.88 | 0.0000 |
| fidelity_02 | 80/40 | 0.064 | 0.017 | 40.39 | 0.0000 |

全部正负／零覆盖／非重访原行保留all_counterexamples.csv。反例三张depth共用0至三图pooled99%分位色阶，仅展示相同目标自己的chunk深度；旋转值来自完整提交前缀姿态，不由颜色推断。![主比较最大反例](/home/liwt/GRoPE/results/gaussian_memory/stac_native_budget1/figures/counterexample.png)

## 完整形成／读取／恢复成本与失败

| 组 | 冷worker数 | load至wait秒 | backbone+heads | 全前缀camera | 几何／位置 | retrieval | prune／merge | 完整chunk秒 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S | 16 | 748.7205 | 378.7278 | 3.5131 | 14.2524 | 6.5913 | 16.0568 | 427.7510 |
| H | 13 | 474.9215 | 222.5295 | 3.2098 | 0.0000 | 0.0000 | 0.9365 | 230.0857 |
| W | 13 | 435.6072 | 183.7119 | 3.2102 | 0.0000 | 0.0000 | 0.9017 | 191.2652 |

完整query是4张当前RGB到原生backbone、depth／S-point、全前缀四轮camera、实际几何／检索／淘汰、CPU作者输出及chunk数值保存。缓存attention计时不能代替完整query；camera前缀增长、首形成、RGB读取、CPU-GPU-DATA传输和恢复均非免费。组件为同步envelope，嵌套方法计时不可相加；完整Popen→actualwait包含import／加载／失败／I-O。实际共享调度／外部保护由原worker准入收据保存，不把非独占混合耗时作方法速度因果排名。单独PCIe精确字节／kernel-only时间未额外profile，未知不伪造0。

| 实际调用／状态 | 数量 |
| --- | --- |
| complete | True |
| successful_formal_streams | 36 |
| unique_input_images | 1152 |
| policy_submitted_frames | 3456 |
| formal_successful_chunk_calls | 864 |
| all_successful_chunk_calls_including_discarded | 890 |
| discarded_successful_chunks | 26 |
| discarded_successful_frames | 104 |
| failed_incomplete_chunk_attempts | 1 |
| total_attempted_chunk_calls | 891 |
| warmup_author_calls | 37 |
| author_loadcheck_calls | 1 |
| official_correspondence_calls | 1 |
| total_author_model_calls | 928 |
| total_native_aggregator_calls | 929 |
| total_successful_prefix_camera_calls | 890 |
| prefix_camera_refinement_iterations | 3560 |
| all_warmup_camera_calls | 38 |
| formal_depth_head_calls | 864 |
| formal_point_head_calls | 288 |
| formal_source_geometry_updates | 288 |
| old_scores_recomputed | 0 |
| training_updates | 0 |
| dataset_or_weight_downloads | 0 |

工程失败保留：①初次S首8帧恢复缺layer_idx，后状态不可用，2个chunk／8帧discard；失败新chunk尝试1次，完整修复后重建首流。②S/fidelity09已到96，末尾可选overflow getter空Tensor转标量失败，数值未提交且无可恢复状态，24个chunk／96帧discard；9条既完整S保持原样。两者所有调用／worker费用计入，不当科学阴性。修复002改可选count为unknown／NA，最小数值chunk备份和永久数值先提交。成功首流恢复保留全部tensor固定样本／RNG／24层index，完成chunk不重复；H/W同prefix8实测恢复。原作者recent元数据覆盖未修。最早9S永久文件为含作者archive的旧格式，其余为科学数值最小格式；实时推理archive均相同保留并计B，实际不同归档I/O成本留账，不将混合版本wall作纯算法速度因果排名。

所有GPU计费上界 **0.727569worker小时≤8**；其中preflight含三组首地点真实恢复及辅助上界 **0.331481小时≤1**。管理worker实际Popen→wait1659.249143s；全部manager阶段wall 2132.338s，首操作至成本汇总混合wall 3907.343s（含准备／工程／等待），实际兼容阶段 1248.181s；辅助960s为缺完整进程wall时的保守预算预约，不能称实际消耗，loadcheck body35.254485s另记。42GPU子进程全部actualwait，exit码{'0': 36, '1': 2, '75': 4}。原源码／兼容／构建90混合分钟内完成真实加载、CUDA及首完整S，未扩deadline。

单worker allocated峰18.231099GiB≤40，RSS10.565899GiB。DATA实际采样峰8.499426GiB、保守完整原子checkpoint并发界44.499426GiB≤48，永久最终8.499426GiB；成本汇总时项目源码／构建与已产物0.082245GiB≤2，最终以terminal_record当前字节为准。只用实时安全CUDA卡，不以余显存当空闲算力；保护他人进程、旧数据和GLIBC。

![完整成本与历史峰](/home/liwt/GRoPE/results/gaussian_memory/stac_native_budget1/figures/costs.png)

## 独立复算、范围与交接

新三组全部3456提交帧保留，其中有有效GT且预测有限的3456depth帧NumPy/Torch独立指标最大差2.66e-15、FP64resize差8.37e-14；NumPy／官方Torch相机解码差1.33e-15、NumPy／SciPy旋转差1.21e-05°。原标签身份继承，0旧分数重计算、0评分模型调用。预测坏值保留显式failure，GT零支持NA不变。

O2/O3真实作者空间状态与实际总byte、O4原生精度—成本为核心；不是完整3DGS、全局永久有界或O5/O6成功。C1共同chunk4／前缀camera权限、C2/C3三种原政策、C4 sourceRoPE、C5原旧稀疏输入、C6冻结均为临时配置。旧C/I/G、U/C、XYZ／PE关闭及正负保持。全部used，未来采用仍需真正未用确认；本报告只交事实，不自选后继。

完整证据：[唯一handoff](implementation_handoff.json)、[manifest](manifest.json)、[全部指标／flags](scoring/summary.json)、[每帧](scoring/frames.csv)、[逐对](scoring/pairs.csv)、[逐地点](scoring/by_place.csv)、[支持配对](scoring/support_strata_paired.json)、[所有反例](scoring/all_counterexamples.csv)、[字节](history_bytes.json)、[成本](costs.json)、[调用](call_accounting.json)、[终态](terminal_record.json)、[实际图审](figure_review.json)。

