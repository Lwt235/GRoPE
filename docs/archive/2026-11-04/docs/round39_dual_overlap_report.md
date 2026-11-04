# R39：双侧三维 Gaussian overlap 未通过旧 TRAIN 必要筛查

2026-11-01。固定主方案 **0/8 项通过，停止该固定设计**。2304 个查询、全部三臂预测及独立数值验证完整，两个标签指标实现一致。没有触及 VAL、实际 V、真实 GS 或新的确认集；没有修改原门槛，也没有将消融臂替换成新主方案。总体研究目标仍未完成。

## 检验的唯一设计

每个原生 patch 只保留一个中心 a 和三维二阶矩 Q；中心取预测中心射线，Q=mean(pixel covariance)+mean[(x−a)(x−a)ᵀ]。双方都从原 RGB 预测深度构建，query 变换到 key 相机坐标。采用已知 Gaussian kernel-mean overlap：

    S = γI + Qq + Qk
    log K = log(mq mk) + (3 log γ − log det S − δᵀ S⁻¹δ)/2
    P = softmax(log(K + mq mk × 1e−30) + 原内容偏置)

γ 固定为两完整图像典型像素世界长度平方的 4 倍，mq=9、mk=1。没有训练新模块、扫描带宽、改温度或增加另一个中心读出。full、point、isotropic 获得完全相同的两侧深度、中心、γ、内容及候选集合；后两者分别移除支撑、保留同迹但移除各向异性。

额外的 key 深度是本次明示的信息变化，不是协方差自己的收益。这个核并不新颖，已有 [核均值闭式来源](https://papers.neurips.cc/paper_files/paper/2012/file/9bf31c7ff062936a96d3c8bd1f8f2ff3-Paper.pdf)；本仓库也已有二维 overlap 和另一种三维 Bhattacharyya 核。按新网格重拟合单 Gaussian 不保持精确划分可加性，本轮没有借用 R38 的积分-law 数值证明。

## 结果

全部 48 对、三场景各16对；每对48个q24查询、432个g8 key。原1096个真值有效中心、原像素可见质量和等场景宏平均均保留。区域 KL 越低越好；PCK@8 越高越好。

| 固定角色 | 宏 KL | 宏 PCK@8 | 宏 RMSE（px） |
|---|---:|---:|---:|
| full | 2.640316460 | 70.712261% | 11.295596 |
| isotropic | 2.207739311 | 66.638974% | 11.594061 |
| point | 16.134258159 | 72.068390% | 9.692203 |

主方案 KL=2.640316460，原 strong_quad/legacy_quad16 的共同参考为 0.308352829；劣化 756.265%，超过原允许2%的上限。两个名字对应同一个配置，不能算成两份独立证据。

同信息 full 相对 point 的区域分布更好，但 full 的 PCK 仍较低；isotropic 的 KL 比 full 更好、PCK 更差。这个结果不支持在本设置下同时取得区域与定位优势，更不支持越过旧二维强对照。不能根据这些消融差异直接认定深度不一致、determinant 偏好或内容强度就是原因。

| 主方案场景 | KL | PCK@8（原计数） | RMSE（px） |
|---|---:|---:|---:|
| harvard_c3/hv_c3_1 | 2.323489806 | 252/382 = 65.968586% | 11.128943 |
| harvard_c5/hv_c5_1 | 2.767997274 | 308/402 = 76.616915% | 11.213907 |
| harvard_c6/hv_c6_1 | 2.829462301 | 217/312 = 69.551282% | 11.543938 |

| 原参考 | 指标 | 原通过边界 | 判定 |
|---|---|---:|---|
| strong_quad | kl | 0.314519886 | 失败 |
| legacy_quad16 | kl | 0.314519886 | 失败 |
| point_localizer | pck | 0.908029162 | 失败 |
| quartic_localizer | pck | 0.906960786 | 失败 |
| legacy_point_localizer | pck | 0.905650550 | 失败 |
| anchor_point_localizer | pck | 0.931237560 | 失败 |
| anchor_quartic_localizer | pck | 0.930169184 | 失败 |
| anchor_point_legacy | pck | 0.928576346 | 失败 |

## 数值与证据边界

事前快照的10项合成测试全部通过，包括直接 Gaussian 期望的20/32阶三维 Gauss–Hermite 交叉验证、常深度平面、Dirac/PSD、共同刚体与尺度变换、depth交换及极端floor。真实数据的144项独立矩阵检查覆盖48对×3臂，全部通过；这里复算的是新闭式算子，未重跑旧R38参考。

最大几何归一化误差 4.3965e-14，最大log-P归一化误差 2.1246e-12，最大P绝对误差 7.5973e-13；所有原生first-index winner精确一致。原q像素几何重建误差为零。生产阶段 9.683s，独立数值阶段 10.202s；这包含来源核验与保存，不是原正式完整算子速度门，也不能与R38单查询耗时直接计算加速比。三臂全部数组共 76,373,268 bytes。

源代码/协议在合成测试前及正式执行前封存，独立数值核验和根验收先于标签读取。双标签路径最大逐行差 3.638e-12、汇总差 7.276e-12，八项判定完全相同。所有阶段exit0，未重试、未修改冻结源、未省略失败查询。

## 后续约束

停止当前γ和双侧支撑定义，保留其失败。下一步先提出可证伪的机制诊断，区分双视图深度对齐、log determinant偏好及支撑形状各自的影响；目前尚无证据选定其中一个解释，也未选择新的修补参数。原TRAIN全条件、VAL/扰动/完整成本、实际V、真实GS和未用数据联合确认门全部仍在。一个结构简洁、实现正确的公式不等于有效方案。

[固定协议](../ROUND39_DUAL_OVERLAP_PROTOCOL.md) · [三臂汇总](../results/round39_dual_overlap/evaluation/summary.json) · [原八门](../results/round39_dual_overlap/evaluation/screen.json) · [独立数值检查](../results/round39_dual_overlap/audit/checks.json) · [独立指标核验](../results/round39_dual_overlap/metrics_audit/verification.json) · [根验收](../results/round39_dual_overlap_root_acceptance_20261101.json)。
