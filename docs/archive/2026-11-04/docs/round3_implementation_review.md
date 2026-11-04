# 第三轮实现独立审查

日期：2026-10-29。审查范围：`models/gaussian_pe.py`、`experiments/region_pe_probe.py`、`experiments/gaussian_pe_probe.py` 与真实区域数据加载约定。本文件不报告方法性能。

## 已验证的正确性

新增 `tests/test_region_pe.py` 验证软交叉熵正确屏蔽无效 key、无效 query 与未使用的 dustbin；评价器使用同一候选归一化；完美软分布的 KL/JS 为零；真实数据加载器按几何有效性和保留质量规则筛选并重归一化标签；模型不读取标签、track count、可见性或筛选字段。isometric 补全分支的自匹配 Gram 对角线验证范数保持，零协方差时回到同频率 projected RoPE。

## 发现的问题与解释边界

1. **密度特征的协方差反传下溢。** 原 `softmax(2*logamp).sqrt()` 在 float32 下使部分幅度精确为零，随后 `sqrt(0)` 的导数污染协方差梯度。以 256 特征、Σ=I 和非平凡线性特征损失即可触发 NaN；Σ=10I 和 100I 同样触发。现有匹配训练冻结几何，内容网络梯度不经过该分支，因此这不意味着已经跑出的内容网络训练结果错误。要将该编码用于可学习的 3DGS 参数，必须修复。数学等价的 `exp(logamp - .5*logsumexp(2*logamp))` 可避免这个导数奇点。已加入相应回归测试，并通知主线程修复模型。
2. **无有效 query 的场景不是零误差样本，已修复。** 审查时评价器对这种场景把逐场景指标写为 0，并同时写 `valid_rows=0`；主线程已改为 NaN，并在整个评价集没有有效行时抛出明确错误。整体指标按有效 query 筛选，计算正确；训练也只抽取至少一个有效 query 的 pair。后续按场景统计和 bootstrap 仍须剔除 `valid_rows=0`。现有 adjacent 真实集：teatime 的 train/validation/test 各 36 pairs 中空 pair 为 1/2/5；Waldo test 空 pair 为 3。新增测试覆盖 NaN 行与全局空监督错误。
3. **isotropic 控制保持 trace，不保持面积。** 当前 `tr(Σ)/2 I` 的 trace 等于原协方差，保存的是均方半径；其 determinant 通常不同。报告应写“trace 匹配各向同性”。如果声称同面积，则实现应为 `sqrt(det Σ) I`，但不能看到测试结果后再更换主对照定义。
4. **真实数据评价有覆盖条件。** 评估行需同时满足至少 4 个共同 SfM tracks、query 几何有效、至少 95% 标签质量留在几何有效 key 上。保留的标签在有效 key 上重归一化。该规则对各方法相同，并且几何有效性来自可观测 Gaussian 数量，不输入真实匹配标签作为特征；但 95% 筛选本身使用真值对应质量，属于有条件的评价群体，应报告有效行/有效 pair 数量与 retained mass，而非宣称覆盖所有真实图像区域。
5. **真实世界单位与跨场景对照。** `rope3d` 直接使用原 SfM 世界坐标与固定频率；其物理单位在不同重建场景之间可以任意变化。其他投影方法使用 patch 单位。因此主要形状贡献应由 shape 对 fixed、trace-isotropic 和同信息 projected/absolute 控制判断。相对未统一世界尺度的 3D RoPE 的优势，不能全部解释为 Gaussian 形状贡献。
6. **区域损失有条件于可见性/共同 tracks。** synthetic 无可见面积的行和 real 无足够共同 tracks 的行不构成 dustbin；区域训练直接去掉最后的 dustbin logit。它检验条件区域分布传递，不能据此说明开放集遮挡拒绝问题已经解决。
7. **真实 pairs 共享 query 相机。** adjacent pair 选择使同一 query 视角配多个 key 视角。置信区间至少应按 `q_id` 聚类重采样，不能把所有 pair 当作独立重复；跨两个真实场景的结果应同时报告逐场景值。两场景不足以证明一般真实场景总体的稳健收益。

## 没有发现的泄漏

在审查的前向计算中，Q/K 特征只读取 RGB、中心及几何协方差。真实加载器的 key validity 作为所有方法共享的候选 mask，在 loss/evaluation 外部使用；监督标签与 retained mass 没有输入位置编码。bridge 实验的 RGB 几何预测器由固定旧 checkpoint 提供，各方法共用，测试场景使用新 namespace。上述隔离不能消除真实重建本身可能已使用测试照片的限制：当前真实分割验证 matcher 的跨相机泛化，不是未见几何场景的重建泛化。
