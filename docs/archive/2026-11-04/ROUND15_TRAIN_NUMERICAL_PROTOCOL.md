# Round15 固定旧 TRAIN 前向数值预检

2026-10-30。只检验数值实现，不计算任何真实任务效果。父线程批准在新 feasibility 读取标签前完成本预检；旧纯合成协议与冻结来源不改写。

固定旧 R10 数据的 Harvard c3/c5/c6 各 metadata 排序第一条 train pair `[30,45]`。每个场景取 q8/q16/q24 的 `np.linspace(0,N-1,8,dtype=int)` 八个原生 query，key 均为完整 432 个 g8 原生矩形。全部前向 query 保留，不用可见性、in-front 或传感器选择。输入只解码 `q_to_k_xy`、`q_to_k_pixel_cov`、`q_features`、`k_features`，不解码任何标签或传感器文件。

固定方法 `quartic_full` 与 `quartic_point`，h 为 `.25,.5,1,2`，内容 T 为 `0,5,10`。真实 production 路径沿用冻结 float32 像素分组均值、完整协方差和父区域内容池化，使用冻结 tiled GL16。point 仅将同一全区域均值对应的输入协方差设为零。总协方差仍为 `C+h²I`。原 key 质量加一次 `1e-30×area`，随后加入相同父内容分数并 softmax；g8 的 reference-unit 面积为 1。

独立参考不导入实际积分算法：用 NumPy 字面矩形切片重算 float64 均值、完整协方差和池化内容；用 Cholesky 白化坐标直接计算 quartic 密度，采用独立张量积 Cartesian GL64 与 GL96。两阶参考先经同一 native-area floor 和全部 T 内容项，再要求所有归一化概率的最大绝对差 ≤`1e-8`。GL96 是预先指定的参考；若未收敛就报告失败，不把不确定参考解释成模型失败或成功。

最终 production 概率相对独立 GL96 完整链的最大绝对差须 ≤`1e-4`，每个 query/方法/h/T 都检查。另将 NumPy 支持与内容送入原 GL16，分离 float32 支持/内容误差与积分误差：GL16(NumPy 支持) 相对 GL96 也须 ≤`1e-4`。均值/协方差与独立切片的 `abs(delta)/(1+abs(reference))` 最大值须 ≤`1e-4`；内容绝对误差须 ≤`1e-5`。这些仅为数值容差，不是科学效果门槛。

计数：3 场景×3 query 尺度×8 query =72 个 support；再乘4个h×2方法，得到576个 geometry rows（72个8×432矩阵）；乘3个T，得到1728个 probability rows（216个8×432矩阵）。保存全部 raw mass 和实际/混合/GL64/GL96概率。逐行报告 raw zero/tiny-FOV（tiny阈值1e-20）、零质量单元数及 floor 占比；不能丢弃尾部或数值退化行。

argmax 是否改变单独透明报告，不以 argmax 相同作为额外筛选。保存 reference top1−top2 gap、实际与参考 winner 索引、两者 winner 的双方概率与差值。这些不是 GT 正误，不能解释为定位改善或劣化。

新源为 `experiments/round15_train_numerical.py` 和对应测试。`freeze` 先绑定协议/代码/测试、完整冻结模型依赖、tiled 数值审计锁与报告、三个 input/manifest 和旧数据 completion；保存来源快照。只有 `run` 才读取 allowlist 数组。`run` 必须 CPU、禁止覆盖或静默续跑，先写 started，完成全部病例后记录通过或失败。预检失败仍保存全部诊断与原始结果；不改变 shape、h/T、样本或有效群体来使其通过。

默认产物 `results/round15_train_numerical`。`summary.json` 包含 passed、lock_sha256、source_hashes、reference_converged、max_probability_error、argmax_change_count 与无标签声明；`completion.json` 包含 lock/summary SHA 和所有产物哈希。只有完整预检通过，后续 feasibility 才可继续读取旧 TRAIN 标签。本预检本身不证明候选效果，也不通过任何验证/确认门槛。
