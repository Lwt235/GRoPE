# 第五轮数据独立性与研究主张审查

2026-10-30。只审查数据来源、分割和可支持的结论；未修改现有协议、训练或模型文件。本次网页工具可读取部分官方页面或其搜索索引，但若干原始服务器直取返回 403/404；**目录列名不等于已经验证帧、深度和 pose 可下载且配对正确**。实际访问与文件核验仍以数据代理的盘点为准。

## 1. UW RGB-D Scenes v2：不适合直接当作 14 个独立房间

[官方页面](https://rgbd-dataset.cs.washington.edu/dataset/rgbd-scenes-v2/)和 [README](https://rgbd-dataset.cs.washington.edu/dataset/rgbd-scenes-v2/README.txt)确认的是 14 个视频重建场景，并未提供 14 个独立物理房间的声明或 room-ID 映射。README 还明确 pose 来自 RGB-D Mapping，每段视频分别以首帧为坐标原点；这不是外部高精度轨迹真值。

[ICRA 2014 作者论文](https://rse-lab.cs.washington.edu/papers/hmp3d-icra-2014.pdf)的数据段描述这些新场景来自 lounge、coffee room、meeting area 和 office desk 等采集环境，视频为绕桌观察。该原 URL 当前直接读取失败，但搜索索引保留原文数据段。不能由这些文字推出恰好四个独立房间，也不能把 14 个 capture/setup 升格成 14 个独立房间。

因此不采用 UW 的数字 scene-ID 随机切分作为本轮“物理场景隔离”主证据。若以后使用，先根据固定背景/布局元数据把疑似同室录像整体归组，并冻结映射；只有 capture-ID 隔离而无法确认物理分组时，结论应限定为已知采集环境内的跨设置泛化，CI 也不能声称覆盖新房间总体。pose 进入模型时还应明示 RGB-D 重建相机这一信息条件。

## 2. SUN3D：按 place 分组，而不是按 sequence 分组

[SUN3D 原论文](https://openaccess.thecvf.com/content_iccv_2013/papers/Xiao_SUN3D_A_Database_2013_ICCV_paper.pdf)区分 415 条序列与 254 个空间，并说明部分地点重复采集。[官方目录/序列清单](https://sun3d.cs.princeton.edu/)也将 `hv_c3_1`、`hv_c3_2` 放在相同 `harvard_c3` 下，而 c5、c6、c8、c11 各有自己的序列组。这支持将顶层 place-ID 作为保守的分割单位，禁止同组的另一条录像进入其他 split。

候选分割如下；此表是计划，尚非帧文件完整性验收：

| Split | 顶层 place-ID | 候选序列 / 约束 |
|---|---|---|
| train | `harvard_c3` | `hv_c3_1`；其余 c3 录像不可进 val/test |
| train | `harvard_c5` | `hv_c5_1`；同组锁定 train |
| train | `harvard_c6` | `hv_c6_1`；同组锁定 train |
| validation | `harvard_c8` | `hv_c8_3`；同组锁定 validation |
| test | `harvard_c11` | `hv_c11_2`；同组锁定 test |
| test | `hotel_umd` | `maryland_hotel3`；整个 hotel_umd 排除出 train/val |
| test | `mit_lab_hj` | `lab_hj_tea_nov_2_2012_scan1_erika`；整个 lab_hj 排除出 train/val |
| test | `mit_76_studyroom` | `76-1studyroom2`；同组锁定 test |
| test | `mit_dorm_next_sj` | `dorm_next_sj_oct_30_2012_scan1_erika`；同组锁定 test |

该方案没有顶层 place-ID 交叉，比 UW 无物理映射的 capture 切分更可辩护。Harvard 的不同 c-ID 支持不同 place 分组，但官方列表本身不提供房间平面图或逐房间的物理隔离证明；应在固定预览中排查同房间重扫、门口/走廊重叠和错误路径，而不使用模型结果决定是否合并。

这是 **place/room-heldout 候选**，不是 building-heldout：Harvard 组可能共享建筑，多个 MIT 测试空间也共享机构/采集风格。不要将不同 place-ID 说成不同建筑，或据 5 个 place 推断任意建筑总体。

## 3. CI 与数据验收

主汇总按 5 个测试 place 等权 scene-macro。五个训练种子共享相同测试地点，不能当作五份独立数据。建议配对 seed/place 重采样，place 内按 query 帧簇或连续帧块处理相关性；方法间共享抽样。必须附 5 个 place 的独立结果及留一 place 敏感性。仅五个地点，bootstrap 区间仍是有限样本证据。

另按 Harvard / 酒店 / MIT 采集组做敏感性报告，识别结果是否完全由一组驱动；只有约三个机构/环境簇，不给“跨建筑总体”高置信度结论。若帧盘点或几何注册失败，不按模型表现挑替代场景；先依照预先固定的完整性规则处理，并在确认前重新冻结场景清单。

## 4. 提供的相机与“RGB-only”边界

[SUN3D 官方 SfM 代码](https://github.com/PrincetonVision/SUN3Dsfm)说明其位姿由 RGB-D 配准和后续 bundle/object adjustment 得到。若使用这些 pose 作为前向输入，即使深度网络本身只读取 RGB，也应称：

> RGB 估计局部深度/支撑，在共同提供的 RGB-D 重建相机条件下做匹配。

所有位置和几何基线共享这些相机；不能称完整系统严格 RGB-only，也不能暗示 pose 与 GT depth 没有关联。GT depth 不得直接用于输入局部几何、预测深度尺度校准或 attention mask。标签由传感器深度和提供 pose 近似生成，不称无误差真值光流。

官方代码还要求将 SUN3D 的 16-bit 深度 PNG 撤销循环移位后再按毫米转换；不能把原 PNG 数值直接除 1000。RGB/depth 注册、K 的 resize 半像素约定、pose 方向和 frame 对齐均需数值闭环验收。

## 5. 本轮最小诚实研究范围

由 RGB 预测深度邻域拟合 `(mu,Sigma)`，再写成 `Sigma=R diag(s²)Rᵀ`，提供的是 **Gaussian 支撑参数化**。它没有产生经过 3DGS 光栅渲染优化的 primitive，也没有验证 opacity、SH 颜色或 densification 的作用。因此本轮最多支持“Gaussian 参数化支撑 PE 可用于预测几何下的两层匹配网络”，不能把结果写成“优化后的 3DGS R/S 比普通点云几何更有价值”。

这仍然符合把 Gaussian 参数化转成 PE 的基础主题，但需要将“参数化形式的贡献”与“3DGS 优化所得参数的特有贡献”明确分开。

若要进一步检验后者，最小补充是对已有真实 GS 区域输入进行匹配训练：

1. 完整区域 covariance：中心散布加权内部 primitive covariance；
2. center-only covariance：保留同一高斯中心、权重、区域划分和均值，仅删除内部 covariance；
3. 两端同时处理，全部模型共用内容、标签、掩码、初始化、批次与优化预算；每种子按自身验证损失选择 checkpoint，五种子完整报告。

可以复用旧 full checkpoint 作为已训练的一侧，只要其训练过程与 center-only 新训练严格匹配并重放核对；旧数据上的比较属于新的机制消融，不算未见数据确认。若做强度选择，两侧须有相同验证预算；不得只替 full 保留最优强度。

这个重训练比较可区分“冻结模型不适应删掉内部 covariance”与“内部 R/S 确有独立任务收益”。即便 center-only 不输，也只能收缩当前流水线中的内部 R/S 主张，不否定 Gaussian 支撑 PE；即便 full 更好，也还需排除共享 primitive 身份、额外几何质量及普通点云拟合等解释，才可提出 3DGS 特有优势。
