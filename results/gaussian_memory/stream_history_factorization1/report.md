# stream_history_factorization1：历史形成 × 当前读取

2026-11-15，round036完整实施[round035唯一合同](/home/liwt/GRoPE/results/gaussian_memory/autonomous_research_loop/round_035/science_plan.md)，SHA `937e4d5badb255ada4d8b85bc3d1182f47112438d356e9d6d42c11c0241ac4a5`。**主I_read重访旋转改善 3.4547 [-0.1270, 7.8404] 度**，正值表示W形成的旧条目在固定实际W相机前缀后可降低当前读取误差。原12地点×96完整RGB、固定189重访／756远端目标；四格输出全部完成，0训练／下载。全部数据used development；不声明独立确认、同byte部署优胜、高斯用途或总体成功，后续科学决定归原Astra。

| 固定效应 | 旋转度 [95%配对区间] | AbsRel误差对比 [区间] | 旋转材料flag | AbsRel材料代价flag |
| --- | --- | --- | --- | --- |
| I_read（主）=E_WW−E_WF | 3.4547 [-0.1270, 7.8404] | -0.0108 [-0.0201, -0.0016] | False | True |
| I_write=E_WW−E_FW | 3.0551 [-3.1883, 9.5292] | -0.0227 [-0.0353, -0.0106] | False | True |
| joint=E_WW−E_FF | 6.7670 [0.8223, 14.1369] | -0.0227 [-0.0371, -0.0084] | True | True |
| direct A/FF=E_FF−E_A | -0.0291 [-0.0965, 0.0340] | 0.0000 [0.0000, 0.0000] | False | False |
| interaction=(E_FW−E_FF)−(E_WW−E_WF) | 0.2572 [-4.5194, 5.7864] | 0.0108 [0.0050, 0.0179] | 预定辅助 | 直接配对，不由显著性差推断 |

误差差的正值均按表中明确公式解释。interaction正值表示F形成下的直接长读取改善比W形成下更大；它不是独立新候选。direct A/FF正值表示旧A误差比新FF低，才直接支持固定W相机隐藏值后旧A收益下降的解释；不能仅凭FF未显著就归因相机。旋转≥1°且下界>0、AbsRel增加≥.01且区间支持退步是本轮开发投入标记，不是通用阈值、等价证明或自动后继。辅助正向、负向、宽区间、地点差异和原零支持均不删。

当前总体／近期目标以[主控制的权威目标](../autonomous_research_loop/goal_revision_20261115/revised_goal.md)为准：总体保持revision31，先完整交付本轮再据结果决定至多一个后继；不重跑已完P0/P1、不自动P2。本报告不改主goal/state/events或目标状态。

## 唯一干预与因果范围

假设是W形成的旧K/V在固定接收端后仍有直接旋转用途；竞争解释为完整历史主要改变随后写入、读取／形成必须耦合，或旧收益大部分经相机隐藏值传播。相对round032只匹配相机保留规则、round034同时改变未来写入的固定配额保存，本次实质变化是交叉来源／读取并固定实际相机tensor。I_read、I_write、joint／interaction和直接A/FF各自回答不同分支；完整结果交原Astra按唯一合同决定，不在这里自动采用／扩数据或重开旧T/D／PE家族。

两条参考聚合器流独立按原生规则写入：F依赖自己的完整历史；W仅依赖自己的首帧＋最近7个完成帧。在W淘汰前仅把本帧24层raw K/V记录到完整W影子库；影子库只供WF诊断读取，绝不回流W参考。当前RGB在WW、FF、WF、FW各自全24层重新计算，保持原位置、5个特殊token、当前自身K/V、原SDPA和精度；WW聚合器同时是W参考当前形成，FF聚合器同时是F参考当前形成，符合合同的等价复用。

| 四格 | 形成来源／过去读取 | 未来提交 |
| --- | --- | --- |
| WW | W／首帧＋最近7 | 独立W聚合器状态＋唯一共同W相机状态 |
| WF | W影子／全部到达帧 | 本次交叉输出全部丢弃 |
| FW | F／首帧＋最近7 | 本次交叉输出全部丢弃 |
| FF | F／全部到达帧 | F参考仅提交自己的聚合器写入；其相机输出丢弃 |

同一帧所有camera worklist均由同一个尚未提交的WW前缀list浅复制，元素直接引用逐tensor相同的不可变原生K/V；每个分支完成原4轮，native cat形成新工作状态而不改旧tensor。前缀shape/dtype/固定样本在每分支前后核对；只有WW返回状态在整帧所有分支结束后提交，再统一淘汰至首帧＋最近7。不是仅规则相同或只固定相机窗口长度。

**这是逐帧条件反事实集合，不是四条独立在线系统轨迹。** F/W参考只接受自己的聚合器输出，WF/FW/FF相机或交叉聚合器输出不进入未来。跨时旋转／平移／点几何由这类条件输出构成，应据此解读，可能存在状态分布失配。FF使用实际W相机隐藏值，所以不是旧A或旧F；旧A/FF仅作预定接收端辅助。即使形成收益存在，也不能把完整F教师生产的紧凑读取当有界在线实现。

前0…8帧读取前缀尚完全相同：每帧只算一次WW，F参考写入同一数学raw K/V，四格camera/depth也共享同一次。第9帧起固定四次原生当前图像计算。该预定工程等价只减少重复调用，没有按结果筛选；原完整96身份和每个逻辑输出均保留。

## 身份、实现与恢复验证

同一冻结官方提交 `05682bcb05683c4c71522e9cb618fc8ca349063c`、5,026,743,569-byte权重，继承SHA `3d3d914d09609fdb485e16947c4885cca0e5ea4e5557598a3b3ffb1cfe0a2f59`，未重新哈希大权重。BF16原生聚合器、FP32 camera/depth、518×392完整RGB、4次camera精化，eval且所有权重requires_grad=False。输入、原pose、顺序及标签沿用旧配置SHA `767f1101b15248b218afdd682b9c322b97d3dff50c02285851f8b6f47e7c1cd7`；新配置SHA `8d2dc3458890df659426bffffff8eb4410a6d0c0d7e73692015a5f6755ec8709`。GT只离线评分，不进入保留／前向，0下载／训练／新标签投影／旧像素评分。

首地点96帧作为固定成本预检且复用正式输出；WW与旧W全部depth/pose最大绝对差分别为0、0；新FF depth与旧F depth最大差0，该head不依赖camera。这是原生staging／参考形成身份验证，不要求新FF pose等于旧A/F，不把数值身份当效果证据。

固定首流在完整frame55主动安全保存，原worker actualwait75；后继从56恢复，无已完成帧重复。完整F/W源库是同一组CPU共享映射文件，fsync后保持原字节，不重新构造教师；独立W bounded K/V、共同camera4轮缓存、CPU/CUDA RNG、ID、已完成输出和next_frame保存／恢复。恢复核对全部bounded/camera shape/dtype/固定样本与配置身份并删除消费checkpoint；F/W库完整原精度映射继续，只在当前帧所在列写入。突发未保存错误将停止，不悄悄回滚重跑。一次工程validation001仅补全最终全96回放检查，模型数值路径未变；实际外部礼让若有均见workers／恢复记录。

## 端点、全地点与反例

固定s≥1、t−s≥32，t=33…95；原210×280深度前向min-z共视容差max(.05m,.05目标深度)，每目标最大覆盖者／平局最早s、覆盖≥5%重访。标签不重算，不改变20%分层，不按预测筛支持。先每地点重访均值再已定义地点等权；5000次配对bootstrap重采样全部12地点ID，seed20261114，NA不填0，区间条件于有定义重采样分母。每个对比及交互都直接用配对每地点差构造，不相减两个区间。

| 地点 | 重访/63 | WW | WF | FW | FF | 旧A | I_read | interaction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| brown_bm_1 | 3 | 0.3142 | 0.4848 | 0.4527 | 0.3671 | 0.3963 | -0.1706 | 0.2562 |
| brown_bm_4 | 0 | NA | NA | NA | NA | NA | NA | NA |
| brown_cogsci_1 | 14 | 17.1729 | 15.6323 | 17.3260 | 16.0951 | 15.9688 | 1.5406 | -0.3097 |
| brown_cs_3 | 0 | NA | NA | NA | NA | NA | NA | NA |
| harvard_robotics_lab | 59 | 3.8205 | 3.1102 | 4.1153 | 3.6470 | 3.6921 | 0.7103 | -0.2420 |
| home_at | 0 | NA | NA | NA | NA | NA | NA | NA |
| home_bksh | 4 | 28.0826 | 31.9214 | 41.2344 | 26.8577 | 27.0590 | -3.8388 | 18.2155 |
| home_md | 8 | 32.5030 | 26.7176 | 13.2332 | 5.5863 | 5.7450 | 5.7854 | 1.8615 |
| mit_32_d507 | 42 | 5.0181 | 3.9005 | 6.8646 | 3.7531 | 3.7606 | 1.1176 | 1.9940 |
| mit_32_g725 | 44 | 11.5448 | 3.8267 | 4.3210 | 3.5375 | 3.5116 | 7.7180 | -6.9346 |
| mit_46_6conf | 9 | 38.7212 | 20.9592 | 22.3812 | 16.6724 | 16.5883 | 17.7620 | -12.0532 |
| mit_w59_conference | 6 | 28.5263 | 28.0581 | 28.2795 | 28.2840 | 28.3404 | 0.4682 | -0.4726 |

原756远端目标、189重访来自9/12地点；三处NA不删。共视零覆盖276、选定独占零支持291完整保留。I_read正向地点7/9，逐一留出均值范围1.6663…4.3664°；真实远端时距12.33…641.76s。

![全部固定效应与地点](/home/liwt/GRoPE/results/gaussian_memory/stream_history_factorization1/figures/primary.png)

## Raw深度、偏差、平移与条件几何

| 全96帧地点等权 | WW | WF | FW | FF | I_read | I_write | joint |
| --- | --- | --- | --- | --- | --- | --- | --- |
| trajectory_rotation_anchor_deg | 42.1740 | 40.6130 | 36.4846 | 33.1411 | 1.5610 [-0.2927, 3.6558] | 5.6894 [2.3002, 9.5960] | 9.0329 [4.0493, 14.9362] |
| trajectory_center_error_raw_m | 4.5459 | 4.4292 | 4.4519 | 4.3263 | 0.1167 [0.0827, 0.1526] | 0.0940 [0.0365, 0.1533] | 0.2196 [0.1561, 0.2885] |
| absolute_worldpoint_GT_error_raw_m | 5.2002 | 5.0667 | 5.0920 | 4.9518 | 0.1336 [0.0932, 0.1837] | 0.1083 [0.0376, 0.1848] | 0.2484 [0.1690, 0.3421] |
| absrel | 0.7253 | 0.7361 | 0.7481 | 0.7481 | -0.0108 [-0.0201, -0.0016] | -0.0227 [-0.0353, -0.0106] | -0.0227 [-0.0371, -0.0084] |
| log_mse | 1.8933 | 1.9059 | 2.1382 | 2.0791 | -0.0127 [-0.0974, 0.0739] | -0.2450 [-0.3819, -0.1152] | -0.1859 [-0.3689, -0.0314] |
| centered_log_mse | 0.0091 | 0.0085 | 0.0092 | 0.0084 | 0.0006 [0.0002, 0.0011] | -0.0001 [-0.0003, 0.0001] | 0.0006 [0.0002, 0.0011] |
| bias_squared | 1.8842 | 1.8975 | 2.1291 | 2.0707 | -0.0133 [-0.0980, 0.0730] | -0.2449 [-0.3818, -0.1152] | -0.1865 [-0.3693, -0.0323] |
| RMSE_m | 2.0392 | 2.0567 | 2.0838 | 2.0820 | -0.0175 [-0.0390, 0.0045] | -0.0446 [-0.0669, -0.0221] | -0.0428 [-0.0734, -0.0125] |
| failure | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |

raw log-MSE=去均值结构项＋bias平方逐项保留，完整AbsRel、RMSE和无效像素／相机失败原数不删。全帧与跨时几何只做首帧刚体gauge，不拟合GT scale或仿射；不能把尺度塌缩的一致性当绝对几何准确。所有交叉输出的轨迹统计是条件诊断，不能当独立在线闭环。

| 固定集合 | 指标 | I_read | I_write | joint | direct A/FF | interaction |
| --- | --- | --- | --- | --- | --- | --- |
| all63far_targets | relative_rotation_deg | 2.9162 [-0.3340, 6.1787] | 8.6402 [3.4354, 14.5383] | 13.9429 [6.6237, 22.7899] | 0.0295 [-0.0799, 0.1273] | 2.3866 [-0.8355, 5.6313] |
| all63far_targets | relative_translation_direction_deg | 7.0335 [0.8368, 12.5441] | 8.3880 [3.0439, 13.6129] | 13.2073 [5.6057, 20.0627] | 0.0844 [-0.0704, 0.2054] | -2.2142 [-7.2147, 2.9053] |
| all63far_targets | relative_translation_vector_error_raw_m | 0.1074 [0.0325, 0.1719] | 0.1258 [0.0474, 0.2078] | 0.1949 [0.0966, 0.2905] | 0.0017 [0.0007, 0.0028] | -0.0383 [-0.1011, 0.0298] |
| all63far_targets | cross_time_point_consistency_raw_m | 0.0540 [-0.0991, 0.1779] | 0.0918 [0.0095, 0.1852] | 0.1449 [0.0165, 0.2593] | -0.0005 [-0.0025, 0.0015] | -0.0009 [-0.0989, 0.0993] |
| all63far_targets | paired_absolute_GT_geometry_error_raw_m | 0.1053 [0.0626, 0.1526] | 0.0508 [-0.0087, 0.1102] | 0.1905 [0.1235, 0.2616] | 0.0014 [0.0000, 0.0030] | 0.0344 [-0.0283, 0.1211] |
| all63far_targets | failure | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| revisit_coverage_ge5pct | relative_rotation_deg | 3.4547 [-0.1270, 7.8404] | 3.0551 [-3.1883, 9.5292] | 6.7670 [0.8223, 14.1369] | -0.0291 [-0.0965, 0.0340] | 0.2572 [-4.5194, 5.7864] |
| revisit_coverage_ge5pct | relative_translation_direction_deg | -0.5162 [-9.4434, 7.1742] | 2.7623 [-1.3137, 7.3119] | 8.6957 [-1.7324, 21.4584] | 0.0307 [-0.2197, 0.2697] | 6.4496 [-1.2847, 16.6814] |
| revisit_coverage_ge5pct | relative_translation_vector_error_raw_m | -0.0075 [-0.1363, 0.0798] | 0.0418 [-0.0395, 0.1450] | 0.0611 [-0.0346, 0.1841] | 0.0013 [0.0002, 0.0025] | 0.0268 [-0.0438, 0.1476] |
| revisit_coverage_ge5pct | cross_time_point_consistency_raw_m | 0.0293 [-0.0865, 0.1823] | 0.1162 [-0.0550, 0.3687] | 0.1506 [-0.0954, 0.4896] | -0.0015 [-0.0029, 0.0000] | 0.0051 [-0.0706, 0.0686] |
| revisit_coverage_ge5pct | paired_absolute_GT_geometry_error_raw_m | 0.0296 [-0.0386, 0.0978] | 0.0614 [-0.0320, 0.1867] | 0.0803 [-0.0281, 0.2112] | -0.0010 [-0.0018, -0.0002] | -0.0108 [-0.0639, 0.0357] |
| revisit_coverage_ge5pct | failure | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| nonrevisit_lt5pct | relative_rotation_deg | 2.5738 [-0.9421, 6.6360] | 8.8397 [2.9566, 15.5076] | 14.0353 [5.7874, 23.7213] | 0.0321 [-0.0823, 0.1358] | 2.6218 [-0.8028, 5.9017] |
| nonrevisit_lt5pct | relative_translation_direction_deg | 6.5972 [-0.1367, 13.0484] | 8.6255 [2.7313, 14.5923] | 12.3654 [4.2628, 20.0471] | 0.0964 [-0.0639, 0.2171] | -2.8573 [-8.5832, 2.8323] |
| nonrevisit_lt5pct | relative_translation_vector_error_raw_m | 0.1369 [0.0530, 0.2075] | 0.1403 [0.0550, 0.2269] | 0.2300 [0.1241, 0.3323] | 0.0019 [0.0009, 0.0029] | -0.0473 [-0.1114, 0.0233] |
| nonrevisit_lt5pct | cross_time_point_consistency_raw_m | 0.0740 [-0.0910, 0.2143] | 0.1036 [0.0142, 0.2016] | 0.1591 [0.0198, 0.2880] | -0.0002 [-0.0023, 0.0018] | -0.0184 [-0.1297, 0.0958] |
| nonrevisit_lt5pct | paired_absolute_GT_geometry_error_raw_m | 0.1264 [0.0797, 0.1759] | 0.0523 [-0.0097, 0.1149] | 0.2122 [0.1455, 0.2798] | 0.0017 [0.0002, 0.0035] | 0.0336 [-0.0349, 0.1221] |
| nonrevisit_lt5pct | failure | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |

所有目标的GT采样点差、平移近零NA及模型失败显式保留在逐对表；没有删除无支持目标来改善几何分数。![全部raw深度代价](/home/liwt/GRoPE/results/gaussian_memory/stream_history_factorization1/figures/depth.png)

## 原支持层与全部反例

| 原独占定义／重访子集 | 原支持层 | 记录数 | 地点数 | I_read | I_write | joint | interaction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| selected_pair_exclusive | 0 | 6 | 1 | -0.0924 [-0.0924, -0.0924] | 0.0685 [0.0685, 0.0685] | -0.8545 [-0.8545, -0.8545] | -0.8306 [-0.8306, -0.8306] |
| selected_pair_exclusive | (0,5%] | 136 | 9 | 2.6190 [-0.3871, 5.5166] | 3.1763 [-4.1821, 10.5790] | 6.6824 [1.3009, 13.7728] | 0.8872 [-4.0752, 7.3379] |
| selected_pair_exclusive | (5,20%] | 44 | 9 | 5.3098 [0.6241, 13.4710] | 3.4190 [-1.2675, 9.0463] | 7.2688 [0.5419, 16.1090] | -1.4600 [-6.6132, 2.2138] |
| selected_pair_exclusive | >20% | 3 | 2 | 3.4796 [2.6375, 4.3218] | -0.3363 [-1.8645, 1.1919] | 3.2799 [1.1420, 5.4179] | 0.1366 [-0.0958, 0.3690] |
| selected_pair_exclusive | undefined | 0 | 0 | NA | NA | NA | NA |
| all_far_union_exclusive | 0 | 5 | 1 | -0.0998 [-0.0998, -0.0998] | 0.0928 [0.0928, 0.0928] | -0.8782 [-0.8782, -0.8782] | -0.8712 [-0.8712, -0.8712] |
| all_far_union_exclusive | (0,5%] | 84 | 7 | 2.1964 [-1.5234, 6.2937] | 0.3974 [-6.7445, 6.4315] | 3.8455 [0.2451, 8.2020] | 1.2516 [-5.2524, 9.7438] |
| all_far_union_exclusive | (5,20%] | 78 | 8 | 6.0215 [1.0472, 13.5594] | 7.9244 [-0.2635, 18.7765] | 10.9987 [0.9187, 23.7766] | -2.9472 [-8.9489, 0.1233] |
| all_far_union_exclusive | >20% | 22 | 8 | 5.3623 [0.8278, 13.6342] | 0.3876 [-5.3781, 7.1906] | 5.5690 [0.8727, 13.6450] | -0.1809 [-7.1413, 5.7873] |
| all_far_union_exclusive | undefined | 0 | 0 | NA | NA | NA | NA |

两种独占定义仍扣除原首帧＋最近7参考mask：选定best-pair与all-far-union分列。层内结果是关联，不当因果、单调趋势或调阈值依据；原全部756目标和重访子集的每层所有指标、地点分母与配对区间在support_strata_paired／support_strata_interaction，零支持与undefined全保留。

| I_read（主）=E_WW−E_WF最大重访反例 | t/s | 覆盖 | 独占 | 秒数 | 旋转改善° |
| --- | --- | --- | --- | --- | --- |
| fidelity_06 | 38/1 | 0.066 | 0.045 | 259.82 | -13.0871 |
| fidelity_10 | 95/61 | 0.292 | 0.008 | 38.11 | -2.8808 |
| fidelity_08 | 51/4 | 0.064 | 0.031 | 89.32 | -2.3762 |
| fidelity_07 | 88/56 | 0.056 | 0.030 | 150.86 | -2.2593 |
| fidelity_06 | 39/1 | 0.065 | 0.009 | 262.30 | -1.9395 |

| I_write=E_WW−E_FW最大重访反例 | t/s | 覆盖 | 独占 | 秒数 | 旋转改善° |
| --- | --- | --- | --- | --- | --- |
| fidelity_08 | 93/46 | 0.129 | 0.090 | 89.35 | -76.9282 |
| fidelity_06 | 38/1 | 0.066 | 0.045 | 259.82 | -50.6628 |
| fidelity_07 | 91/44 | 0.081 | 0.059 | 218.83 | -7.9971 |
| fidelity_08 | 56/20 | 0.315 | 0.057 | 68.37 | -4.9938 |
| fidelity_08 | 77/45 | 0.143 | 0.012 | 60.76 | -4.9699 |

| joint=E_WW−E_FF最大重访反例 | t/s | 覆盖 | 独占 | 秒数 | 旋转改善° |
| --- | --- | --- | --- | --- | --- |
| fidelity_08 | 51/4 | 0.064 | 0.031 | 89.32 | -3.3064 |
| fidelity_08 | 81/48 | 0.072 | 0.026 | 62.78 | -2.9034 |
| fidelity_08 | 72/27 | 0.085 | 0.018 | 85.33 | -1.6056 |
| fidelity_09 | 60/27 | 0.078 | 0.001 | 42.20 | -1.4713 |
| fidelity_09 | 59/27 | 0.090 | 0.015 | 41.63 | -1.4635 |

| direct A/FF=E_FF−E_A最大重访反例 | t/s | 覆盖 | 独占 | 秒数 | 旋转改善° |
| --- | --- | --- | --- | --- | --- |
| fidelity_06 | 38/1 | 0.066 | 0.045 | 259.82 | -1.0069 |
| fidelity_07 | 87/52 | 0.097 | 0.042 | 162.25 | -0.5005 |
| fidelity_07 | 89/52 | 0.140 | 0.072 | 171.70 | -0.3436 |
| fidelity_07 | 88/56 | 0.056 | 0.030 | 150.86 | -0.2880 |
| fidelity_08 | 92/47 | 0.064 | 0.033 | 85.33 | -0.1975 |

全部比较756×5反例和756项直接交互原行均导出，不只保存最好／最坏子集。![最大主比较反例及绝对几何](/home/liwt/GRoPE/results/gaussian_memory/stream_history_factorization1/figures/counterexample.png)

## 真实历史库、教师与转移字节

每来源96帧bank容量9,824,108,544bytes，两份F/W合计19,648,217,088bytes=18.298828GiB，从分配时即按完整逻辑文件容量计账，包括尚未到达的未写列；未写列从不读入forward。F当前rawKV只按列附加到自身full库，W当前rawKV只写影子库；W参考另有最多818,675,712bytes的独立有界CPU状态，唯一共同W相机GPU 2,097,152bytes。当前native窗口追加至9后再淘汰，瞬态和恢复另计。

CPU staging不是免费：两组来源共96个rawKV文件以shared mmap在CPU寻址，CPU驻留页和磁盘backing均披露，是同一映射逻辑storage的两种硬件占用，不另伪造免费独立历史库。RSS包含页驻留，磁盘按完整18.3GiB文件而非du稀疏已分配块收费。各bank仅供本流进行期间使用，永久四格输出与完成SHA原子提交后删除本轮自有bank；未来重读新query必须重新形成并付费。共享映射写入经过OS页缓存，未声称每个正常结束流都物理落盘全库；唯一次安全恢复实际fsync计时与kernelIO记录明确分列。00–06每个global attention prehook把本层所选raw past复制到GPU；07–11以同一rawKV的18.30GiB GPU常驻镜像避免重复大H2D，CPU／disk镜像仍保留并分别收费，GPU镜像在段完成后、checkpoint／最终输出I-O前释放。WW始终用独立有界参考缓存，不回读W影子。原native qkv/cat、qk_norm／原RoPE／SDPA及当前自身K/V照常执行；只改变缓存硬件位置，GPU镜像模式每帧释放allocator未用保留区。引用F的FW/FF同时依赖完整F形成；WF依赖W形成和完整影子库；所有分支还依赖WW实际相机形成。共用数学工作只在联合实际账计一次，各单格依赖仍列出，不能把“共享”说成独立可部署的便宜方法。

| 保守硬件历史账（非DATA文件cap） | GiB |
| --- | --- |
| CPU_file_mapped_source_bank_resident_capacity_bound_bytes | 18.298828 |
| disk_source_bank_backing_bytes | 18.298828 |
| CPU_W_old8_new9_pruned8_raw_copy_bound_bytes | 2.382660 |
| CPU_complete_exit_serialization_bound_bytes | 0.764404 |
| disk_bounded_checkpoint_atomic_bound_bytes | 1.528809 |
| GPU_source_bank_mirror_capacity_bytes | 18.298828 |
| GPU_F_bounded_index_work_copy_bound_bytes | 0.762451 |
| GPU_one_layer_raw_past_and_native_cat_bound_bytes | 0.762451 |
| GPU_W_camera_prefix_and_native_branch_work_copy_bound_bytes | 0.015625 |
| shared_native_grid_index_metadata_allowance_bytes | 0.015625 |
| whole_selected_readable_RGB_backing_bytes | 0.244747 |
| total_bytes | 61.373257 |

| 单格依赖 | 真实形成／保存范围 |
| --- | --- |
| WW | Wreference causalwrites+itsboundedstate+commonWWcamera; fullWshadowisjointdiagnosticwriteoverhead not neededbyWWalone |
| WF | AllWreference formation andfullWshadow+currentfullread+commonWWcamera |
| FW | AllFreference fullformation+currentboundedread+commonWWcamera formation fromWW |
| FF | AllFreference fullformation+currentfullread+commonWWcamera formation fromWW |

原完整12流可读RGB backing 262,795,460bytes保守列为每格共同依赖；实际forward只单调读取当前RGB，没有旧图重编码暗库。必要teacher／影子／CPU-GPU-DATA传输、原生位置与索引、恢复副本、原子写均收费。旧GT与预测归档不回流forward；物理新增输出仍计DATA。本轮不是旧8帧B、不是等byte候选，不给有界部署优越性结论。

## 成本、精确调用与实际终态

固定首流成本准入在2026-11-14T17:09:08.448427+00:00登记、科学评分前；实测GPU worker 482.7087s，含两次加载和完整恢复／原子输出，另11流同案例×1.25余量投影总7119.9526s≤14400。该首流预测复用，没有额外benchmark或为比速度重跑；冷载指新进程加载，不保证物理文件uncached。不能从分段预检开销宣称单格独立部署加速。

尚未开始的07–11启用GPU镜像，普通工程006验证复用了完整07：WW depth/pose与旧W最大差0/0，FF depth与旧F差0，frame9两交叉对亦为0。07全进程178.790963s另计入预检／成本，连原profile合计661.499614s≤1800，所有输出计正式，0额外query。CPU模式00的分段成本准入与GPU镜像07的资源校验均在科学评分前登记；没有中断或重做已完成流。

| 实际新调用 | 数量 |
| --- | --- |
| logical_cell_frame_outputs | 4608 |
| unique_current_RGB_native_aggregator_calls | 4284 |
| actual_camera_head_calls | 4284 |
| actual_depth_head_calls | 4284 |
| actual_camera_refinement_rounds | 17136 |
| reference_W_aggregator_calls | 1152 |
| reference_F_aggregator_actual_calls | 1044 |
| reference_F_first9_shared_with_W | 108 |
| counterfactual_WF_FW_aggregator_calls | 2088 |
| logical_FF_equals_F_reference_aggregator | True |
| repeated_completed_stream_frames | 0 |
| model_loads | 13 |
| training | 0 |
| downloads | 0 |
| new_label_projection_calls | 0 |
| old_pixel_metrics_recomputed | 0 |

每流实际9＋87×4=357次完整聚合器／camera／depth，12流4284；每次camera4轮共17136。逻辑四格输出4608，不把复用视为漏帧；W参考1152次，F参考1044次＋108次共享前缀，WF/FW共2088次。每次完整当前query包括RGB encoder、全部24层、原生camera4轮／depth及真实选择／转移，缓存读取延迟未假装完整query。单格表是模块计时，history索引选择及Python bookkeeping残差保留在四格完整envelope内，不冒称单格独立完整query。

| 条件实际独立调用 | 次数 | aggregator含transfer秒 | camera4轮＋depth秒 | 共享WW前缀逻辑帧 |
| --- | --- | --- | --- | --- |
| WW | 1152 | 481.4790 | 31.5178 | 0 |
| WF | 1044 | 1412.3216 | 25.9241 | 108 |
| FW | 1044 | 177.5134 | 25.6574 | 108 |
| FF | 1044 | 1460.3267 | 26.5679 | 108 |

| 所有流实测CPU暂存环节 | 实际合计 |
| --- | --- |
| D2D_bank_write_bytes | 91.49414 GiB |
| D2D_bank_write_seconds | 1.69221 s |
| D2H_bytes | 1156.25720 GiB |
| D2H_seconds | 368.78659 s |
| H2D_bytes | 7337.83008 GiB |
| H2D_seconds | 2151.32085 s |
| bank_write_seconds | 173.16266 s |

联合两参考形成的aggregator共1941.8057s，同时产出WW/FF当前结果，不重复向联合账收费；WF/FW额外交叉当前读取共1589.8350s。四格完整帧envelope摊销3.320452s/到达帧；全部进程含冷载／I-O／恢复摊销3.587285s/到达帧。两者都含实际转移；前者不含帧间进程冷载，后者含。本轮无独立单格冷载测速或纯attention微基准。

H2D hook计时含同步、真实CPU→CUDA复制和有关调度，不冒称纯PCIe带宽；native norm／RoPE／attention在aggregator剩余时间内，native raw复制瞬态／allocator保留也未隐藏。CPU bank write与D2H分列；联合完整query envelope及逐帧成本可重建。前七流较低瞬时GPU利用率主要来自逐层同步暂存；后五流采用同raw字段GPU镜像，全部容量／D2D写入亦计费，没有增加训练／扫描来填卡。不同固定地点与硬件模式的实际时耗分列，不当成单格方法速度的匹配试验。![完整成本及转移随历史增长](/home/liwt/GRoPE/results/gaussian_memory/stream_history_factorization1/figures/costs.png)

GPU所有Popen→actualwait合计 **1.147931worker小时≤4**；预检／成本0.183750小时≤.5，13实际进程均已wait。模型加载body合计199.6263s，完整保存IO11.1899s；失败／礼让／恢复从未扣掉。单worker NVML峰32.011719GiB≤40，allocated24.844017、reserved31.693359GiB，RSS峰23.885372GiB。DATA采样峰21.785231GiB、含全部永久输出＋两源库＋完整有界保存／原子暂存的保守峰24.201279GiB≤32；最终3.486403GiB，project运行期采样峰与封装前实数在terminal_record，最终封装实数另见sealer回执。

宿主CUDA可访问，restricted sandbox不可见设备不被误诊驱动坏。全8卡持续观察外部PID与稳定空闲，只使用当时安全资源；两个完整source bank在32GiB DATA内只容一条活跃／恢复流，是本轮可核算存储约束，不是默认两卡上限。外部任务到达只礼让自有worker，完整保存后missing-only续跑；纯CPU manager等待不计GPUworker，但自身wall及所有GPU加载／IO实耗均记录。不修改系统libc，不动其他用户文件／进程，不删旧结果。

GPU没有意外失败；一次预定完整保存exit75已从56恢复并复用其输出。CPU版本回执helper曾因/tmp缺PYTHONPATH在任何模型／数据工作前退出，补路径后复用同一helper；figure007只修正实际看图发现的颜色图例，不改指标。常规修复、前后源版本与CPU阶段用时均保留。

## 独立复算、边界与交付索引

新WF/FW/FF全部3456份depth用NumPy FP64／Torch独立指标核对，最大差2.66e-15；FP64双线性resize最大差8.39e-14；NumPy／官方Torch pose解码差9.99e-16；NumPy／SciPy相对SO3角差1.23e-05度。WW和旧A的2304逐帧／1512逐对指标直接继承，0旧像素评分／标签重投影／新模型调用。交互与直接A/FF从相同地点／相同对原行配对，线性恒等式和独立bootstrap核对见interface_validation。

本次只辨别值得研究的是旧K/V直接读取、状态形成或它们耦合；实际固定W相机隐藏值的比较仍是条件干预，不唯一解释所有在线失败。完整F/W影子库和全部形成费用不是有界方法，旧T/D固定配额及全部PE家族关闭保持；旧round032旋转阳性／raw深度代价也不抹去。服务O3有界实现机制与O4原生用途；O2高斯结构不新增，O5渲染／O6 PE仍可选；C1两条自己写入、C2未压缩K/V、C3当前交叉＋同一相机、C4无新PE、C5旧12×96、C6冻结都是临时实验选择。没有fresh、总体完成、采用方法或自选后继。

完整证据：[manifest](manifest.json)、[唯一实现交接](implementation_handoff.json)、[评分全数](scoring/summary.json)、[每帧](scoring/frames.csv)、[每对](scoring/pairs.csv)、[地点](scoring/by_place.csv)、[支持层](scoring/support_strata_paired.json)、[支持交互](scoring/support_strata_interaction.json)、[全部反例](scoring/all_counterexamples.csv)、[直接交互原行](scoring/interaction_rows.csv)、[真实字节](history_bytes.json)、[成本](costs.json)、[调用](call_accounting.json)、[实际看图](figure_review.json)、[终态](terminal_record.json)。所有新预测路径／SHA由completed索引，原配置／标签／权重保持旧位置。

