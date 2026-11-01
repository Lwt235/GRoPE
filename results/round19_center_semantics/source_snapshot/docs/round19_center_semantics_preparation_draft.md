# R19 条件准备：合法预测中心是否比区域均值更有用

2026-10-31。**未冻结、未执行、未选择新候选。** 本稿只读旧源码及来源元数据，
没有打开新的预测/标签数组或计算反事实效果。R18 首次 GH64 合成实现失败仍待
完成审计；它不否定其连续公式，也不自动授权本分支。root 之后决定是否执行。

最小第一步只比较旧区域均值 μ 与合法中心射线投影 a；暂不生成新 Gaussian
概率、不扫 h/T、不读 VAL/fresh。沿原中心检查，query 为原生 g8/g16/g24，
key 固定为原生 g8 的432个格，不把 query 尺度误作 key 尺度。

## 人口与可复用来源

固定旧 TRAIN c3/c5/c6 各16对、共48对，直接继承 R17 Stage A 的
`target_contract.entries`，按 `(scene,pair,query,query_index)` 严格连接。
每对432/108/48个query，共28,224行；原中心有效人口为9,521/2,463/1,096。
全部行保留，评价只使用原中心有效标志，不再与区域有效性、预测前向/FOV
标志相交。上述人数是旧合同核对值，不是本稿重新计算的结果。

| 对象 | 实际来源及绑定方式 |
| --- | --- |
| cohort、输入、精确中心GT | [R17 feasibility lock](/home/liwt/GRoPE/results/round17_feasibility/lock.json) 的 `training_manifest_records`、`inputs`、`target_contract.entries`；目标链同时绑定 R13 lock/completion/audit |
| 合法原输入 | `/home/liwt/GRoPE/results/round10_partition_data/train/<scene_slug>/<pair_dir>/inputs.npz`；逐文件 SHA 及对应 manifest、相机和预测缓存来源继承原锁 |
| 中心预测 a | [R13 predicted_centers](/home/liwt/GRoPE/experiments/round13_mixture_mechanism.py:39)，仅需 `q_predicted_depth,q_K,k_K,q_to_k_transform` 与原分区bounds |
| 旧 μ、C与bounds | `/home/liwt/GRoPE/results/round17_size_temperature_attribution/forward/<scene_slug>/arrays/<pair_dir>/supports.npz` 的 `g8/g16/g24__mean,__cov,__bounds`；由 [size根验收](/home/liwt/GRoPE/results/round17_size_root_acceptance_20261031.json) 及 producer/audit completion 绑定 |
| 已接受R17主选winner | 同目录 `g8_g8.npz,g16_g8.npz,g24_g8.npz` 中 `trace_full__argmax`；对应 `records.json` 绑定 input SHA、原生概率SHA、argmax SHA及R17B原概率SHA |
| 精确中心标签 | `/home/liwt/GRoPE/results/round13_target_conflict/targets/<scene_slug>__<pair_dir>.npz` 的 `g8/g16/g24_point_xy`、`*_point_visible`；每文件期望SHA在上述target contract，审计为 [R13 verification](/home/liwt/GRoPE/results/round13_target_independent_audit/verification.json) |

例如首个 c3 pair 为 `[30,45]`，pair_dir 为 `000_0000030_0000045`，
scene_slug 为 `harvard_c3__hv_c3_1`。必须取锁内完整48条，不能靠目录glob
纳入其它轮次/场景。g8/g16也统一读取R13精确目标；其坐标与旧R10只作过
allclose核对，不能混用后声称边界PCK逐位相同。

复用不是重新生成实验：新锁绑定既有审计接受链、完整文件SHA和逐数组身份；
只解码所需mean/bounds和144个已保存winner数组，不需重跑1,728个R17前向或
GL积分。若有来源/人口不一致则停止，不能改用“接近”的数组。旧R17 winner
只作固定参照，不能依诊断效果切换旧12格配置。第一步不需要区域标签、
source sensor图像、size-temperature角色分数或完整旧概率矩阵。

## a、μ与先预测后标签的合同

R13函数在原矩形的半像素几何中心对RGB预测深度双线性取值，再沿该射线
反投影并变换。它已经规定 `nan→10,+inf→20,−inf→.1`、深度clip[.1,20]、
投影分母z下限.1及输出坐标clip[−1024,1024]。这些是既有有限输入约定，
不能因旧PCK结果换插值或选区域内更好的像素；`in_front/in_fov`只报告，
不掩蔽评价。a返回像素单位，旧μ是reference-8单位，比较时明确仅转换一次。

第一步输出完整 a、旧μ、两者差、有限/前向/FOV标志，以及各自到固定g8格
中心的平方距离最小索引。精确平局取第一个原生索引；出视野也保留相同
最近格规则。不能改成“包含a的格”、借旧point winner、使用GT allowed set，
或把 nearest-μ 叫作带内容的正式point模型。

新producer/独立auditor、代码快照、输入白名单、48对人口、分区、算术/单位、
输出字段、平局及错误处理先联合冻结。全48对的全部新预测与固定参照身份
封存后，才允许核对目标文件实际SHA或解码目标；冻结时只读已有期望哈希。
新审计独立重建中心射线及最近格，核验旧μ来源，并独立复算计数/汇总。
此处没有实际执行这些检查，不能借R13旧的24-query审计冒充全量新审计。

## 能否证什么，以及推进前仍需冻结什么

评价沿原 `squared_distance≤64` 的PCK8、原有效中心和等场景口径，保存
整数correct/benefit/harm及精确Fraction。同时报告 a/μ 的连续中心误差、
最近格分歧，以及在固定R17正确/错误行上的完整转移；分组只能解释，不能
参与前向或选人口。这能否证“源中心身份提供了可用定位信息”的开发假说，
不能证明新Gaussian区域质量、鲁棒性或actual-V效用。

执行前需明确：用于决定是否继续的主尺度/指标、所需最小改善、其它尺度
允许损失、逐场景一致性要求，以及没有改善时是否停止。数值与规则必须在
新效果前写进独立锁，不能看结果后改成只需某个场景/分位数获益。本稿没有
选择这些阈值，亦没有新增已生效科学门槛。三旧场景不是独立泛化确认。

只有定位信息诊断提供事前规定的支持，才考虑另冻一个单Gaussian否证：
`mean=a, C_a=C+(μ−a)(μ−a)ᵀ`，并给point/fixed同样a，保留旧强参照。
固定均值交叉熵解释及其限制见 [设计审查](/home/liwt/GRoPE/docs/round18_post_r17_design_review.md)。
第一步不生成 C_a 概率，也不将“最近a较好”视作该模型通过。后续仍需原
14区域条件、中心保护、六扰动与完整成本；当前仅zero中心诊断无法回答
R17的旋转失败。之后才是同容量actual-V、真实GS和最后未触碰确认，
所有旧失败及启动限制保留。
