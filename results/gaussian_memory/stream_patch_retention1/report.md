# stream_patch_retention1：同实际历史字节的原生patch保存

2026-11-15，round038完整实施[round037唯一合同](/home/liwt/GRoPE/results/gaussian_memory/autonomous_research_loop/round_037/science_plan.md)，SHA `60655230da4a3bcb49666faddf0cdf8bea75aa59767528f7ae1af6f0fd6d5ecc`。**主I_space=E_U−E_C重访旋转改善 6.2937 [-5.2777, 26.2323] 度**。U/C各12×96完整原生预测；W旧12流输出／评分继承，仅固定首地点一次96帧匹配成本重放。0训练／下载，所有数据used development。完整结果交原Astra作取舍，不自动高斯／P2，不宣称fresh或总体成功。

| 固定比较（正值误差降低） | 重访旋转度 [95%配对区间] | 全帧AbsRel改善 [区间] | 旋转材料flag | 明显AbsRel代价flag |
| --- | --- | --- | --- | --- |
| C_vs_U | 6.2937 [-5.2777, 26.2323] | -0.0174 [-0.0259, -0.0098] | False | True |
| C_vs_W | 5.2756 [-3.5320, 13.8501] | -0.0228 [-0.0305, -0.0150] | False | True |
| U_vs_W | -1.0181 [-16.0604, 10.4361] | -0.0054 [-0.0129, 0.0013] | False | False |

≥1°且地点配对下界>0、AbsRel增加≥.01且区间支持退步是预定开发投入标记，不是普适门槛／等价／非劣性证明。C/U始终是主比较，C/W和U/W净用途分别保留；不以较差对照上的增量替代原生净用途，也不把深度辅助换成旋转主成功。当前总体目标由[主控制的canonical目标](../autonomous_research_loop/goal_revision_20261115/revised_goal.md)与[round037科学取舍](../autonomous_research_loop/round_037/science_plan.md)限定，本实现不改goal/state/events。

## 固定假设、政策和实际源位置

检验独立原生patch束能否在有界、自身反馈的流中保留长间隔旋转用途，并且预测3D中心覆盖是否比普通分散保存更有用。竞争解释包括普通样本已足够、预测几何／离群点误导覆盖、原生上下文不能这样稀疏保留或近期容量削减代价仍更大。相对已关闭T/D整帧政策，此次改变不可分割保存单位，不换描述子或扫描配额；相对round036无完整F教师／W影子／条件交叉。旧长历史收益及raw深度代价、旧FPS静态池化正负和PE家族关闭均保留。

| 政策 | patch历史 | 特殊token／相机 |
| --- | --- | --- |
| W | 首帧＋最近7完整原生帧，旧输出／评分继承 | 原首帧＋最近7；每次camera4轮 |
| U | 首帧＋最近3完整patch，远端M个固定64b哈希最小者 | 5特殊token另保首帧＋最近7；自身相机同规则 |
| C | 与U同容量／候选；近期中心锚定的欧氏FPS远端 | 同U规则，实际隐藏值各组自己形成 |

共同M在效果前一次固定为 **4117**。来源token ID=1041×frame＋localtoken（特殊0…4、patch5…1040）；全24层同一patch的raw K/V作为束保留，未融合、质量为1。U以splitmix64(ID xor20261115)无符号升序取最小M，哈希碰撞按ID。C每次仅从自身已保留远端＋刚退出近期patch选择，以首帧／最近3完整patch的有效中心为锚，依次取到已覆盖集合最近欧氏距离最大者，平局按源ID；距离用FP64平方欧氏，输入中心FP32。未满M全留；无效候选不进距离，若有效不足M则用同U优先级补足，无有效锚时初距无穷且按ID选首个。没有额外confidence、尺度或阈值筛选。

每组当前原生depth在14×14 patch内排序，取中间两值均值作为中位深度；射线经过原图patch像素中心(x=14c+7,y=14r+7)，用该组当前FP32预测相机／内参反投影到预测世界系。非有限／非正中位深度或非有限相机使中心无效，不删输入或指标。中心在写入时固定，U同样得到并付费；当前RGB及未来状态均由各自保留历史形成，另一组、GT、Full与淘汰库从不进入forward。C/U是在线政策总效应，内容与未来几何可随之不同，不当固定内容的纯几何因果或准确物理中心证据。

全24层保留未归一化raw K/V；每个保留token以自身原source二维位置做原qk_norm／RoPE，再原SDPA。特殊位置仍(0,0)，patch为原(row+1,col+1)；按来源ID排序只改变过去集合，不重投影，不加log n／新PE／融合。当前1041token、全部24层及4轮camera均原生完整计算。聚合器全局attention的packed接口使past具有单个packed维，非首帧特殊embedding与原生相同；不规则key positions逐条保留，不把整帧pos重复到错误patch。相机和特殊token各自首帧＋最近7；patch首帧＋最近3＋远端M，各类统一在整帧结束后淘汰。

官方提交 `05682bcb05683c4c71522e9cb618fc8ca349063c`，5,026,743,569-byte权重继承SHA `3d3d914d09609fdb485e16947c4885cca0e5ea4e5557598a3b3ffb1cfe0a2f59`，未重新大哈希。完整518×392 RGB，BF16聚合器、FP32 camera/depth、原SDPA、冻结eval。旧配置SHA `767f1101b15248b218afdd682b9c322b97d3dff50c02285851f8b6f47e7c1cd7`，新配置SHA `581c4b05c3794ce03483daf4f1097e1b101c4520776dcdfdb445016010f546a8`；原12×96身份、时间、c2w/K/标签全继承，GT仅离线。

## 必要验证、恢复与调用身份

固定首地点先作唯一W96完整成本重放，再完成U96/C96并复用正式输出。W96与旧W depth/pose最大差0；U/C容量尚未淘汰的前7帧，原生输出各差0。没有用旧W像素指标重复验收。C的单次5点GPU FPS toy核对精确欧氏／对称平局选取；正式首帧自身位置与完整过去同序时可回放原接口。恢复在U00整帧47后完整保存并actualexit75，同一fit由48恢复，已完成query无重复。恢复核对全部K/V和camera shape/dtype/固定样本、源ID／中心样本和CPU/CUDA RNG，消费checkpoint在下一forward前删除。全部逐帧来源保留／哈希／不复活淘汰ID的CPU独立核对见interface_validation。

正常运行不留周期旧checkpoint；cooperative退出才保存当前精确状态、RNG及输出。突发未保存错误停止并保留失败，不悄悄回滚重跑。profile的局部分段回放摘要不是最终验证来源，CPU使用已完成归档前缀一次核对；普通metadata002保留profile源码并给未开始formal增加begin/complete调用记录，没有改模型路径。一次CPU配置helper在模型／数据工作前遇到旧字段缺失，按同一catalog求实际RGB总bytes修复，记录CPU_configuration001。

## 主指标、地点、NA与全部原反例

原t=33…95，s≥1、t−s≥32；原210×280 GT前向min-z共视max(.05m,.05目标深度)、最大覆盖／平局最早source、≥5%重访和20%层完全继承，0新投影／标签重选。先各地点重访误差均值再已定义地点等权。配对bootstrap5000次重采样全部12地点ID，seed20261114；NA不填0，区间条件于有定义抽样分母，所有留出地点均保留。

| 地点 | 重访/63 | W° | U° | C° | U−C° |
| --- | --- | --- | --- | --- | --- |
| brown_bm_1 | 3 | 0.3142 | 0.3638 | 0.4381 | -0.0743 |
| brown_bm_4 | 0 | NA | NA | NA | NA |
| brown_cogsci_1 | 14 | 17.1729 | 25.2738 | 38.6289 | -13.3551 |
| brown_cs_3 | 0 | NA | NA | NA | NA |
| harvard_robotics_lab | 59 | 3.8205 | 3.5982 | 2.8750 | 0.7233 |
| home_at | 0 | NA | NA | NA | NA |
| home_bksh | 4 | 28.0826 | 7.3809 | 4.2615 | 3.1194 |
| home_md | 8 | 32.5030 | 83.9422 | 7.4403 | 76.5019 |
| mit_32_d507 | 42 | 5.0181 | 4.2171 | 3.6634 | 0.5536 |
| mit_32_g725 | 44 | 11.5448 | 3.7799 | 5.0135 | -1.2336 |
| mit_46_6conf | 9 | 38.7212 | 18.0627 | 27.3511 | -9.2884 |
| mit_w59_conference | 6 | 28.5263 | 28.2479 | 28.5516 | -0.3036 |

756远端目标，189重访来自9/12地点；三处NA、276零共视、291选定独占零支持均保留。主正向4/9，逐一留出均值范围-2.4823…8.7498°；远端真实时距12.33…641.76s。GT平移近零仅方向辅助记NA，目标数0/756，不删主旋转。

![固定比较和全部地点](/home/liwt/GRoPE/results/gaussian_memory/stream_patch_retention1/figures/primary.png)

## 全帧raw深度、平移和几何

| 全96帧地点等权 | W | U | C | U−C [区间] | W−C [区间] | W−U [区间] |
| --- | --- | --- | --- | --- | --- | --- |
| trajectory_rotation_anchor_deg | 42.1740 | 39.8673 | 39.1985 | 0.6688 [-3.8482, 4.3546] | 2.9755 [0.2069, 6.0915] | 2.3067 [-0.4915, 5.3668] |
| trajectory_center_error_raw_m | 4.5459 | 4.4910 | 4.4953 | -0.0043 [-0.0538, 0.0461] | 0.0506 [0.0046, 0.1019] | 0.0549 [0.0230, 0.0883] |
| absolute_worldpoint_GT_error_raw_m | 5.2002 | 5.1360 | 5.1430 | -0.0070 [-0.0701, 0.0497] | 0.0573 [0.0025, 0.1243] | 0.0642 [0.0221, 0.1139] |
| absrel | 0.7253 | 0.7308 | 0.7482 | -0.0174 [-0.0259, -0.0098] | -0.0228 [-0.0305, -0.0150] | -0.0054 [-0.0129, 0.0013] |
| log_mse | 1.8933 | 1.8637 | 2.1027 | -0.2389 [-0.3815, -0.1198] | -0.2094 [-0.3226, -0.1098] | 0.0295 [-0.0330, 0.0962] |
| centered_log_mse | 0.0091 | 0.0090 | 0.0086 | 0.0003 [0.0001, 0.0006] | 0.0004 [0.0001, 0.0007] | 0.0001 [-0.0002, 0.0003] |
| bias_squared | 1.8842 | 1.8548 | 2.0940 | -0.2393 [-0.3819, -0.1201] | -0.2098 [-0.3230, -0.1102] | 0.0294 [-0.0330, 0.0960] |
| RMSE_m | 2.0392 | 2.0465 | 2.0916 | -0.0451 [-0.0671, -0.0249] | -0.0525 [-0.0715, -0.0343] | -0.0074 [-0.0231, 0.0088] |
| failure | 0.0000 | 0.0000 | 0.0000 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |

raw log-MSE=去均值结构＋偏差平方，各项／完整AbsRel／RMSE／坏像素和失败均保留。只作首帧刚体gauge对齐，无GT尺度／仿射拟合。跨时点一致性并列绝对GT几何和GT投影采样差，避免塌缩伪装一致；这些是自身因果在线轨迹，仍非准确物理地图或连续千帧确认。

| 目标集合 | 指标 | I_space | C/W | U/W |
| --- | --- | --- | --- | --- |
| all63far_targets | relative_rotation_deg | 0.1633 [-7.6608, 6.1347] | 3.3193 [-2.3748, 8.4240] | 3.1560 [-0.0893, 6.4733] |
| all63far_targets | relative_translation_direction_deg | 1.4596 [-0.7798, 3.6536] | 5.8351 [0.0215, 10.8955] | 4.3755 [-0.3337, 8.5710] |
| all63far_targets | relative_translation_vector_error_raw_m | 0.0054 [-0.0217, 0.0349] | 0.0558 [-0.0133, 0.1245] | 0.0504 [-0.0047, 0.1037] |
| all63far_targets | cross_time_point_consistency_raw_m | 0.0675 [-0.0108, 0.1572] | 0.0938 [0.0140, 0.1829] | 0.0263 [-0.0848, 0.1149] |
| all63far_targets | paired_absolute_GT_geometry_error_raw_m | 0.0169 [-0.0380, 0.0714] | 0.0355 [-0.0184, 0.0999] | 0.0187 [-0.0356, 0.0762] |
| all63far_targets | failure | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| revisit_coverage_ge5pct | relative_rotation_deg | 6.2937 [-5.2777, 26.2323] | 5.2756 [-3.5320, 13.8501] | -1.0181 [-16.0604, 10.4361] |
| revisit_coverage_ge5pct | relative_translation_direction_deg | -2.1151 [-4.5349, -0.0539] | 2.0030 [-2.1047, 6.3670] | 4.1180 [-0.1504, 9.0560] |
| revisit_coverage_ge5pct | relative_translation_vector_error_raw_m | -0.0140 [-0.0530, 0.0444] | -0.0007 [-0.0643, 0.0656] | 0.0133 [-0.0303, 0.0531] |
| revisit_coverage_ge5pct | cross_time_point_consistency_raw_m | 0.0850 [-0.0725, 0.3503] | 0.0860 [-0.0832, 0.3271] | 0.0010 [-0.0645, 0.0683] |
| revisit_coverage_ge5pct | paired_absolute_GT_geometry_error_raw_m | 0.0156 [-0.0817, 0.1982] | -0.0086 [-0.0987, 0.1199] | -0.0242 [-0.0867, 0.0244] |
| revisit_coverage_ge5pct | failure | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| nonrevisit_lt5pct | relative_rotation_deg | -1.0446 [-9.7731, 5.1293] | 2.4213 [-3.2186, 7.5864] | 3.4659 [0.3626, 7.5174] |
| nonrevisit_lt5pct | relative_translation_direction_deg | 1.9168 [-0.3649, 4.1426] | 5.4372 [-0.6565, 10.9061] | 3.5204 [-1.2502, 7.8724] |
| nonrevisit_lt5pct | relative_translation_vector_error_raw_m | 0.0016 [-0.0271, 0.0315] | 0.0614 [-0.0104, 0.1318] | 0.0598 [0.0025, 0.1158] |
| nonrevisit_lt5pct | cross_time_point_consistency_raw_m | 0.0680 [-0.0237, 0.1714] | 0.1080 [0.0216, 0.2059] | 0.0400 [-0.0777, 0.1383] |
| nonrevisit_lt5pct | paired_absolute_GT_geometry_error_raw_m | 0.0128 [-0.0451, 0.0721] | 0.0419 [-0.0214, 0.1167] | 0.0291 [-0.0272, 0.0873] |
| nonrevisit_lt5pct | failure | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |

![全帧深度代价](/home/liwt/GRoPE/results/gaussian_memory/stream_patch_retention1/figures/depth.png)

## 原支持层和完整反例

| 原定义（重访子集） | 支持层 | 记录数 | 地点数 | C/U | C/W | U/W |
| --- | --- | --- | --- | --- | --- | --- |
| selected_pair_exclusive | 0 | 6 | 1 | -0.2155 [-0.2155, -0.2155] | -0.3161 [-0.3161, -0.3161] | -0.1005 [-0.1005, -0.1005] |
| selected_pair_exclusive | (0,5%] | 136 | 9 | 7.6270 [-9.2685, 33.0454] | 3.7044 [-12.0663, 16.8978] | -3.9226 [-23.5407, 11.0400] |
| selected_pair_exclusive | (5,20%] | 44 | 9 | 4.3190 [-7.7273, 21.3459] | 3.9854 [0.5378, 9.6018] | -0.3336 [-12.8877, 10.5842] |
| selected_pair_exclusive | >20% | 3 | 2 | -0.0593 [-0.9182, 0.7997] | 1.9955 [1.3489, 2.6421] | 2.0548 [0.5492, 3.5604] |
| selected_pair_exclusive | undefined | 0 | 0 | NA | NA | NA |
| all_far_union_exclusive | 0 | 5 | 1 | -0.2082 [-0.2082, -0.2082] | -0.3506 [-0.3506, -0.3506] | -0.1423 [-0.1423, -0.1423] |
| all_far_union_exclusive | (0,5%] | 84 | 7 | 0.0918 [-1.2493, 1.7436] | 7.8958 [0.7871, 17.3312] | 7.8040 [0.3913, 16.1527] |
| all_far_union_exclusive | (5,20%] | 78 | 8 | 4.3314 [-25.7917, 44.2700] | -6.0283 [-36.2134, 15.2622] | -10.3598 [-35.3611, 10.3739] |
| all_far_union_exclusive | >20% | 22 | 8 | -0.9666 [-3.6817, 0.6661] | 2.7339 [0.5665, 5.9134] | 3.7005 [0.4100, 9.5417] |
| all_far_union_exclusive | undefined | 0 | 0 | NA | NA | NA |

两独占定义仍扣除原W首帧＋最近7的参考mask，不换为有利的U/C覆盖；selected最佳source与all-far-union分列。全部756目标及重访子集、每层全部指标和配对地点区间在support_strata_paired.json，0／undefined不删。层内相关不是因果／单调性结论，不调阈值。

| C/U最大重访反例 | t/s | 覆盖 | 独占 | 秒数 | 旋转改善° |
| --- | --- | --- | --- | --- | --- |
| fidelity_02 | 79/40 | 0.052 | 0.044 | 39.88 | -92.2154 |
| fidelity_02 | 80/40 | 0.064 | 0.017 | 40.39 | -89.8589 |
| fidelity_10 | 79/45 | 0.064 | 0.064 | 38.14 | -54.9008 |
| fidelity_09 | 84/37 | 0.130 | 0.005 | 60.43 | -12.7543 |
| fidelity_09 | 85/10 | 0.167 | 0.006 | 98.27 | -11.9863 |

| C/W最大重访反例 | t/s | 覆盖 | 独占 | 秒数 | 旋转改善° |
| --- | --- | --- | --- | --- | --- |
| fidelity_02 | 80/40 | 0.064 | 0.017 | 40.39 | -158.6214 |
| fidelity_02 | 79/40 | 0.052 | 0.044 | 39.88 | -136.5444 |
| fidelity_10 | 79/45 | 0.064 | 0.064 | 38.14 | -5.7566 |
| fidelity_10 | 95/61 | 0.292 | 0.008 | 38.11 | -2.3589 |
| fidelity_02 | 86/37 | 0.062 | 0.004 | 48.97 | -2.2068 |

| U/W最大重访反例 | t/s | 覆盖 | 独占 | 秒数 | 旋转改善° |
| --- | --- | --- | --- | --- | --- |
| fidelity_07 | 89/52 | 0.140 | 0.072 | 171.70 | -109.5448 |
| fidelity_07 | 88/56 | 0.056 | 0.030 | 150.86 | -98.2114 |
| fidelity_07 | 86/52 | 0.096 | 0.086 | 160.24 | -96.3212 |
| fidelity_07 | 87/52 | 0.097 | 0.042 | 162.25 | -91.7451 |
| fidelity_02 | 80/40 | 0.064 | 0.017 | 40.39 | -68.7625 |

全部756×3比较的原行、正负个例／零覆盖和非重访均保存all_counterexamples.csv，不只挑最佳／最差。![最大C/U反例](/home/liwt/GRoPE/results/gaussian_memory/stream_patch_retention1/figures/counterexample.png)

## 实际同byte上限、所有副本与淘汰

固定B=**2,726,555,844bytes**。M=4117时单份最大状态payload上界818,424,049bytes；每个保留token包含24层raw K/V 98,304bytes＋int64源ID8＋FP32中心12＋valid1，来源位置从ID无损派生；最大patch为4×1036＋M，特殊为8×5，相机2,097,152bytes，另同128KiB arena。三份GPU／CPU／单个原子disk状态＋全部12可读RGB 262,795,460＋8MiB grid／RoPE／Python／索引／FPS距离／RNG／ZIP余量，总 **2,726,456,215≤B**；M+1已超限。U/C同一副本／恢复策略，不以无效中心减少账或暗增容量，未满容量时仍按最大容量比较。

普通forward每层原生追加后，旧GPU K/V在完整aggregator返回后立即释放，再构造新的保留state；old＋appended与appended＋selected各峰都低于三份最大state，不额外保存旧库。camera旧引用同样在淘汰前释放。save时GPU当前state＋CPU精确copy＋唯一新disk原子文件；继续前消费并删除checkpoint，淘汰raw不在旧可读库。全部序列的源ID轨迹只作输出provenance，不被forward读取。native归一化／RoPE中间量、当前激活／模型／allocator reserve另报，原raw历史副本不藏为免费activation。8MiB共同余量覆盖FPS当前距离／索引及GPU／CPU metadata临时对象；原始完整RGB库存保守计给每组，实际只按当前顺序读取一次。

预测depth／pose、原GT与输出provenance／partial归档均无forward回流，不作预测历史输入；物理新增仍全部计DATA。无教师／影子／他组／已淘汰raw内容暗库。源码、实际state-file bytes与恢复、原生grid／实测forward总bytes见history_bytes与execution_identity。W原完整8帧自己的3份／128KiB／1MiB旧预算B保持，新增W只作同机成本回放。

![真实patch来源保留轨迹](/home/liwt/GRoPE/results/gaussian_memory/stream_patch_retention1/figures/retention.png)

## 完整形成、更新、读取、传输和资源成本

科学评分前2026-11-14T18:52:40.365735+00:00，固定首地点W/U/C含4次实际冷进程（U完整恢复分段）共187.223604s；U 52.828097s，C 101.208531s。含25%余量的全轮投影2305.227235s≤14400。C预检包含CUDA扩展首次编译和唯一必要FPS toy检查，U含人为恢复开销，均实际收费，输出复用正式；不以不等验证／编译开销声称方法速度排名，冷进程也不保证物理uncached权重。

| 实际完整流 | 环节 | 总秒 | 均值ms/到达帧 | 帧数 |
| --- | --- | --- | --- | --- |
| W | aggregator_seconds | 11.74235 | 122.3162 | 96 |
| W | camera_depth_seconds | 2.38335 | 24.8265 | 96 |
| W | center_selection_eviction_seconds | 0.38623 | 4.0233 | 96 |
| W | RGB_read_seconds | 1.15945 | 12.0776 | 96 |
| W | frame_wall_seconds | 16.99671 | 177.0491 | 96 |
| U | aggregator_seconds | 141.10422 | 122.4863 | 1152 |
| U | camera_depth_seconds | 28.55445 | 24.7868 | 1152 |
| U | center_selection_eviction_seconds | 24.51688 | 21.2820 | 1152 |
| U | RGB_read_seconds | 14.37921 | 12.4820 | 1152 |
| U | frame_wall_seconds | 225.83382 | 196.0363 | 1152 |
| C | aggregator_seconds | 140.29256 | 121.7817 | 1152 |
| C | camera_depth_seconds | 28.50914 | 24.7475 | 1152 |
| C | center_selection_eviction_seconds | 219.47107 | 190.5131 | 1152 |
| C | RGB_read_seconds | 14.47177 | 12.5623 | 1152 |
| C | frame_wall_seconds | 415.58465 | 360.7506 | 1152 |

形成包含当前完整encoder／24层raw写入与自身相机深度；更新包含同组中心构建、候选／GPU FPS或哈希／淘汰和实际copy；读取使用原norm／source RoPE／SDPA。完整query是当前RGB到aggregator、camera4轮、depth及真实选择／传输和输出，不把缓存attention当完整query。模块计时残差、CPU/GPU同步和archive写入保留于完整帧／进程wall，选择envelope不能冒充孤立纯GPU距离kernel时间。CPU-GPU-DATA真实转移、RNG／源ID小copy、编译／保存／恢复全部进程计费，没有免费cache或教师。转移字节及其可观测范围在costs.transfer_ledger；未另加profiler／benchmark重放。

| 固定首地点实际全部进程wall | 秒 |
| --- | --- |
| W | 33.186977 |
| U | 52.828097 |
| C | 101.208531 |

| 实际新调用 | 数量 |
| --- | --- |
| formal_U_C_native_frames | 2304 |
| W_fixed_complete_cost_replay | 96 |
| actual_aggregator_calls | 2400 |
| actual_camera_head_calls | 2400 |
| actual_depth_head_calls | 2400 |
| camera_refinement_rounds | 9600 |
| model_loads | 26 |
| repeated_completed_policy_stream_frames | 0 |
| FPS_interface_arithmetic_calls | 1 |
| training | 0 |
| downloads | 0 |
| new_label_projection_calls | 0 |
| old_W_pixel_metrics_recomputed | 0 |
| profile_complete_native_frames_without_separate_hooks | 288 |
| actual_policy_FPS_calls | 1068 |
| actual_policy_hash_calls | 1068 |
| finite_native_frames | 2400 |

GPU所有Popen→actualwait合计 **0.321226worker小时≤4**，预检／成本0.052007小时≤.5；26进程均已wait，失败／恢复／加载／I-O未扣除。模型加载body 374.207932s，完整保存IO 1.002830s。单worker NVML峰7.851562GiB≤40，allocated 6.643433、reserved 7.498047GiB，RSS峰9.939709GiB。DATA采样峰1.888404GiB；全部25永久输出＋保守8worker同时完整exit/输出原子暂存＋扩展编译backing的峰界9.806344GiB≤12，最终1.888404GiB。project≤1GiB与实际终态见terminal_record。

宿主CUDA／NVML实际准入，restricted exec看不到设备不被误诊驱动故障；全8卡稳定空闲观察，独立U/C剩余流按CPU/RAM／DATA峰动态用所有安全卡，无继承单worker／两卡上限。忙卡不因余显存当空算力，外部任务保护只礼让自有worker；唯一manager和actualwait收据保留，等待不重算。0下载，不改GLIBC／系统库，不删旧结果。

## 独立评价、O/C范围及交接

新U/C全部2304份depth独立NumPy／Torch复算最大差2.66e-15，resize最大差7.82e-14，NumPy／官方Torch pose差1.33e-15，NumPy／SciPy旋转差1.17e-05°。原W1152逐帧＋756逐对指标逐行继承，0旧像素重评分／新标签投影／评分模型调用。直接原行配对bootstrap、1152帧U哈希独立选择和2304帧无淘汰ID复活核对见interface_validation。

本轮仅选择保存粒度与空间强基线，不是高斯／体素／真实STAC胜负，不证明各向异性或原生patch等于物理点。O3有界状态和O4原生精度—成本为核心，O2仅预测中心空间参照，O5渲染／O6 PE无新增；C1自身因果形成、C2全层未融合束、C3固定U/C淘汰、C4原二维位置、C5旧稀疏SUN3D、C6冻结均为临时选择。旧整帧T/D及所有PE家族关闭、旧条件旋转正证据及raw深度代价继续保留。材料flag仅决定科学后续值得判断的对象，结果完整交原Astra，不自选后继／扩容量／改锚或哈希，全部used，无独立确认或总体完成。

完整交付：[唯一handoff](implementation_handoff.json)、[manifest](manifest.json)、[评分及flags](scoring/summary.json)、[每帧](scoring/frames.csv)、[每对](scoring/pairs.csv)、[地点](scoring/by_place.csv)、[支持配对](scoring/support_strata_paired.json)、[全反例](scoring/all_counterexamples.csv)、[byte与恢复](history_bytes.json)、[完整成本](costs.json)、[精确调用](call_accounting.json)、[接口](interface_validation.json)、[图实看](figure_review.json)、[实际终态](terminal_record.json)。永久预测SHA及路径由25份completed索引，原W/GT与旧结果保持原路径，模型大权重不重哈希。

