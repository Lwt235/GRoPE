# R15 接入实际 V 深度任务：最小实现审查

2026-10-30。这里只读了冻结的 R7/R12/R15 源码、协议及既有数据身份和运行环境元数据；没有启动训练、解码新数据或读取 Stage B 区域效果。候选及其最终 h/T 尚未在本文确定。

**原来的深度头和数据可以继续使用。真正需要新增的是候选几何缓存、来源绑定和角色/分析适配。当前不能直接启动旧 R12 trainer：其科学前置关口已经失败；新候选尚未完成后续推广关口，而且当前磁盘余量与 CUDA 状态也需先解决。**

## 可以直接继承的部分

| 部分 | 来源与必须保留的行为 |
|---|---|
| 深度头 | [R7 模型](/home/liwt/GRoPE/models/round7_depth_attention.py)、[R12 外部几何适配](/home/liwt/GRoPE/models/round12_depth_attention.py)：41,345 个注册参数，同一初始化顺序；共享无 bias 的 128→128 Q/K，129→64 query/value 分支，128→64→1 head |
| 数据 | [R7 数据生产](/home/liwt/GRoPE/experiments/round7_depth_data.py)：576 train（c3/c5/c6 各 192）、128 old-val（c8/bm4 各 64），192×144 图像、stride8、432 tokens；保持原 pair 顺序及 input/label hashes |
| 优化与评价 | [R7 训练](/home/liwt/GRoPE/experiments/round7_depth_train.py)、[R12 训练](/home/liwt/GRoPE/experiments/round12_depth_train.py)：五配对 seeds 11/22/33/44/55，1,000 steps、batch8、AdamW lr=.001/wd=.0001、梯度裁剪 1、有效 query 的 log-depth L1 |
| 配对证据 | 同 seed 的初始化和 replacement-sampling batch 序列一致；每 100 steps 评价（含 step0），按等场景 val AbsRel 的首次严格最小值选 checkpoint；保留 raw depth、zero/reverse-V 干预、梯度和逐 pair 加性指标 |

实际 attention 是 `softmax(T * normalize(Wq) @ normalize(Wk).T + geometry_logits)`，没有额外的 `sqrt(d)`。它聚合学到的 64 维 source V，再与 query 分支拼接预测 log-depth residual。residual 限制在 [-3,3]，最终深度限制在 [.1,20]。这是对真实 `attention @ V` 的任务验证，不能由区域 KL 或中心 PCK 代替。

原始单视角对照必须独立训练：用 query 自身作 key/value，沿用原 native geometry（h=.5、T=5），保持同容量；它不读取 cross-view geometry。zero-V/reverse-V 是训练后干预，不能替代这个对照。

数据中的 `q_depth` 是 RGB 预测深度的原生中心采样；`q_value_depth` 是 query 预测 patch 三维均值的 z；`k_value_depth` 是 key 预测 patch 均值变换到 **query camera** 后的 z。三者不能混用。depth loss 的 target/valid/boundary 必须原样复用 R7：有效 query sensor depth 不要求跨视角可见。不能改用 R13/Stage A 的 `point_visible` 作 loss mask，也不能用 source sensor depth 构造 V。现有位姿是数据集提供的重建元数据，所以结论仍是给定位姿条件下的任务。

## T=0 的实际影响

最终候选若是 T=0，Q/K 的 **16,384** 个参数没有数据损失梯度；余下 **24,961** 个 query/value/head 参数仍可学习。RGB 内容仍通过 query 和 V 参与预测。因此必须同时报告注册参数数、T 和有效梯度路径，不能把“参数数相同”写成“有效容量完全相同”。AdamW 对零梯度参数仍可能做权重衰减，不应把它解释为学习到了内容路由。

最小适配保留原头的 Q/K 计算，再乘 T=0；现有梯度审查允许 T=0 的 Q/K 梯度为有限零值。只在计时时跳过这段计算会使成本不真实。若以后要优化该路径，须在新源码冻结前统一适用于所有 T=0 角色，并同步处理 `grad=None` 规则及数值等价检查。

强位置控制和 singleview 保留各自冻结配置，不能为了激活候选 Q/K 把 T 改成 5，也不能为了名义容量对称削弱控制。若需要把核形状贡献与内容路由分开，matched-T0 point 可在训练前声明为归因对照；它不能替换选定的强 point 或 legacy 控制，更不能在 depth val 上重新搜索 h/T。

## 几何缓存哪些可用，哪些必须重做

[R12 几何生产](/home/liwt/GRoPE/experiments/round12_depth_geometry.py) 已按原 R7 pair 顺序导出八角色纯几何 float32 `[N,432,432]`，其中**没有 T×content**。它的 RGB 缓存、相机、特征逐元素 join 和 predicted-pixel 投影可以作为新生产的冻结来源；所有像素参与，不按 GT 或预测可见性筛选。

- 已有 Gaussian 缓存只有在 family、h、原生 key、积分/floor、dtype、输入及 pair 顺序一致时才可复用。因为缓存不含内容，单独改变在线 T 不改变它；新 manifest 仍须明确记录新的 T 和被引用缓存的 hash，不能把旧角色温度一起继承。
- 所有 quartic 几何都需要新导出；其他 Gaussian 若选中的 h 不同也必须重算。新 inventory 可以引用不可变的旧 NPY，避免复制约 4GB 数据，但旧 R12 inventory 硬编码八角色和相对文件名，因此需新适配器。
- Stage A/B 的概率不能作为可训练 attention：它们只覆盖 48 train pairs，且已加入冻结 RGB 内容并 softmax，不能反映训练中的 Q/K。即使 T=0，其概率也只适合作数值交叉检查，不能替代完整 704-pair 几何契约和现场成本。
- 旧 R7 checkpoint 或旧候选分数不能当成新候选的配对训练。R12 深度训练没有启动；其缓存完成不等于模型效果通过。

新导出应保持 GL24、原生半像素矩形、面积 floor 只加一次，以及实际训练的 float32 geometry cast。首 metadata pair 需要核对新 geometry 与独立 reference/冻结 forward 的概率；原 R12 的 full4 preflight 不能替 quartic 通过这一检查。前向生产本身无需读取 labels。

## 最少新增改动与不能绕过的关口

不修改冻结文件，新增候选专属 protocol、geometry adapter、model role wrapper、trainer/analysis adapter。原头的层、初始化、loss 和 batch/evaluation helpers 可以复用；新 wrapper 只扩充外部角色身份，不能把 quartic 伪装成旧 `full4`。分析器需泛化 primary/strong position/strong compression/point-localizer 等角色，同时保留既有干预、梯度、原始预测独立重放和 paired hashes。

R12 trainer 的 freeze/verify/analysis 硬绑定 `R12 stress gates.passed` 及其独立审计链，而这个关口科学上失败。禁止改 flag、绕过 guard 或复用其命名空间。新 R15 depth protocol 应绑定将来完整通过的候选选择、旧验证/尺度/扰动及独立审计链；Stage A 可行集合或 Stage B 校准本身不是启动 depth 的依据。

按 [R15 候选协议](/home/liwt/GRoPE/ROUND15_CANDIDATE_PROTOCOL.md) 和 [R12 深度协议](/home/liwt/GRoPE/ROUND12_DEPTH_PROTOCOL.md)，继承的最低要求包括：

- 对强 position 的等场景/种子 AbsRel 至少改善 2%，严格优于同容量 singleview；这两类比较均要求两个场景改善、至少 4/5 配对 seed 改善。
- 对预先定义并按五 seed val AbsRel 选出的最强 compression，满足“AbsRel 改善至少 2%且增量时间≤1.25×”或“AbsRel 恶化≤1%且时间≤.5×”；不能只选择慢基线。
- RMSE 对强 position/compression 恶化≤2%，并通过人口、参数、来源、初始化/batch、公平预算、真实 V/梯度及独立预测核验。

新训练角色集合、选择顺序和 tie 规则须在训练前确定，包含新 quartic point、强 Gaussian 及 legacy 保护，避免把旧控制静默替换。上述两个场景/4-seed 规则在旧实现针对 position/singleview，不应错误声称旧协议也把它施加到了 compression。

成本必须现场重新执行角色需要的 support preparation、原生积分、floor、float32 cast、learned Q/K、softmax、V/head；不能用缓存加载或纯积分耗时。沿用五 scene 首原 pair、五 seed 选定 checkpoint、3 warmups+15 repetitions、轮换角色顺序。共享 predictor、dense projection、transfer 可以排除但另报。反复使用的两个 old-val 场景不支持总体置信区间，也不能抹去 R7/R9 的历史失败。

## 现场资源约束

以下是本次只读审查的环境快照，不是运行保证：

- 2026-10-30 10:14:21 UTC，文件系统可用 **10,259,005,440 bytes**。若保留 root 原要求的 10GB，只剩约 259MB 余量；一个新 704-pair float32 几何角色需要 **525,533,184 bytes**（约 .489GiB），已经超过该余量。须先规划存储位置/额度；不得删除冻结失败证据腾空间。
- 现有八角色 payload 是 **4,204,265,472 bytes**，含 NPY headers 是 **4,204,270,592 bytes**；原协议笔误已有[单独勘误](/home/liwt/GRoPE/docs/round12_depth_data_protocol_erratum.md)。尽量引用旧匹配角色，仅生产缺失的 kernel/h。
- 主机约 503GiB RAM、407GiB available、128 logical CPUs，但 CPU 配额仍由并行任务协调。R12 loader 虽用 mmap 读 NPY，随后 `np.concatenate` 并转 tensor，实际会把当前角色的全部 train/val geometry 放入内存/设备，不是 minibatch streaming。仅 q/k 特征另有约 311MB，还未计入其余输入、临时副本和 activations。
- `nvidia-smi` 当前无法与驱动通信，GPU 可用性未确认。不能沿用历史 A6000 的速度承诺当前 ETA，也不应为本任务修改系统驱动。新 GL24 的纯 synthetic CPU operator 成本同样不能代替实际 V 总成本。

因此代码适配可以先完成，完整缓存导出与训练仍需在科学前置通过后，重新确认空间、执行设备和并行资源预算。若九个核心角色、五 seeds、1,000 steps，则是 45,000 updates；最终角色数应以新冻结协议为准。

## 新颖性边界

“让几何作用于 V/O”已有直接先例：POD-RoPE v2 把同一 expected-rotation 统计用于 Q/K 以及依赖 coherence 的 V/O 变换；WorldRoPE A.7 明确将 value 变换到公共坐标框架再变回 query 框架，并报告对应消融。见 [POD-RoPE §3.4，2026-10-28 版本](https://arxiv.org/html/2610.33745v2)、[WorldRoPE A.7](https://arxiv.org/html/2610.16337v1#A7)。不能把这项一般做法作为本项目的新颖性，也不能把论文效果计入我们的证据。

本次最小适配只是以几何确定路由权重来聚合 learned V，并未实现 V/O 通道旋转。现在额外引入该变换会改变原来的单一候选比较；它也不能证明来自实际优化 3DGS primitive 的 R/S 参数具有额外收益。
