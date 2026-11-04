# R17 downstream implementation preparation — conditional only

本文只读核对 R17 协议、R7/R8/R9 报告及相关实现，准备后续接口。
没有打开未触碰数据、执行预测、读取 R17 新 Stage A/B 分数或修改冻结源。
**所有下述真实执行都 pending 上游全部通过**：接受的 R17 数值审计、完整
Stage A/B 与 TRAIN 停止检查、随后另行冻结的旧 VAL 几何/尺度/扰动/成本
回归及独立审计。本文不是执行协议，不选择新参数，也不宣称候选有效。

依据：[R17 正式合同](../ROUND17_CANDIDATE_PROTOCOL.md)、
[继承验证清单](round11_validation_inventory.md)、
[当前深度合同](../ROUND12_DEPTH_PROTOCOL.md)。清单内已废止的深度草案
不能覆盖 R12 修订。新 R17 下游协议必须外置，不能修改或直接绕过 R12
因旧 full4 鲁棒性失败而禁止训练的锁。

## 已有证据要求回答什么

| 旧证据 | 对 R17 的约束 |
|---|---|
| R7 实际 V 深度独立确认：AbsRel 对锁定 point 仅改善 0.335%，区间跨零；同容量 singleview 均值更好 | 必须重新训练并通过实际任务，不能用区域 KL 或梯度非零代替任务效益 |
| R8 旧开发诊断：真对应＋原源值相对 singleview 仅改善 1.031%；传感器辅助源值帮助更大 | 路由改进未必解决 RGB 预测源深度质量、patch 混合及遮挡；传感器源值和 oracle 路由均不得成为正式前向输入 |
| R7 实际 primitive 内部 R/S 没有胜过强中心/点控制；任意 key 协方差时五维展开不再精确 | 图像区域支撑 C 与 optimized primitive C 不是同一证据；不宣称任意 GS-to-GS 的精确五维 PE |
| R9 context-only GS/RGB：full 对强 point MSE 仅改善 0.146%，两场景方向相反 | 实际 RGB V 绕开了预测深度限制，仍未通过；新的读出需保留相同强控制、原 GS、完整图像和失败范围 |

这些数值来自已公开的 [R7](../round7_report.md)、[R8](../round8_report.md)、
[R9](../round9_report.md) 报告，而非重新打开其测试数据。
不能把 R8 干预叫作严格性能上界，也不能据此断言 R17 必然无效。

## 1. 实际 V 深度：最小可实施变化

保留 [R7 单块模型](../models/round7_depth_attention.py) 的全部 41,345 个
参数、初始化顺序、共享恒等初始化 Q/K 投影、64 维 query/V 投影及残差头。
新实现可参照 [R12 模型适配](../models/round12_depth_attention.py)，但使用
新文件和新来源锁。改变仅发生在下面的注意力分数：

```text
q = normalize(Wqk * query_rgb_feature)
k = normalize(Wqk * source_rgb_feature)
content = q @ k.T                         # 每次训练在线计算，有梯度
geometry = log(native_Gaussian_mass + one_native_area_floor)
tau = T * h² / (h² + trace(C))             # 原始完整区域支撑，reference-8单位
attention = softmax(geometry + tau[:,None] * content)
aggregate = attention @ learned_V(source_rgb_feature, source_predicted_depth)
depth = original_query_depth_residual_head(query_feature, aggregate)
```

调用冻结的 R17 trace 运算序，而不是另写近似式。tau 不乘 V，不变成 query
loss 权重，不改变源可见性，也不替代 softmax。严禁把区域实验保存的最终
概率或冻结特征相似度塞入训练充当 learned attention。几何可以缓存；内容、
Q/K、softmax、V 和输出头必须在线。C 来自冻结预测几何，因此本实验不训练
预测器或 covariance 网络；存在对 C 的解析梯度不等于训练了这些部分。

**数据适配。** 原始 576 train / 128 old-val pair 的 R7 `inputs.npz` 提供
`q,k,q_depth,k_value_depth` 等合法字段；V 中的深度是 source RGB 预测点变换
到 query 相机后的 z，保持 [原导出定义](../experiments/round7_depth_data.py)。
query 的全部 sensor-valid stride8 中心参与，包括 source 遮挡、出界和无对应
位置。`target,valid,boundary` 只进入损失/统计，不进入模型字段白名单。
给定相机仍来自 RGB-D 重建，不声称完全 RGB-only。

**新增缓存最少为纯几何和支撑。**
[R12 geometry 导出](../experiments/round12_depth_geometry.py) 已建立逐 pair
身份、RGB feature 与预测 cache 的严格 join，可复用其只读数学/元数据逻辑。
但其缓存只有 float32 `geometry_logits`，没有可恢复 R17 tau 的完整 C；
也不能从高斯 logits 反推一个可信 C。新导出应保存物理定义一致的 full C
`[N,432,2,2]` 和纯几何 `[N,432,432]`，并绑定 mean、分区、输入及相机来源。
C 内包含所有预测像素的散布和像素 footprint，不按源可见性裁掉像素。
已经保存的预测观察可在来源完全相同时复用，不能把旧 full4 的缓存冒充 full1。

为最直接继承 R17 数值合同，推荐新 cache 保留 float64 geometry，内容及
tau×content 保持 float32，softmax 保持 float64。若 V/head 保持原 float32，
明确在 `attention @ V` 前仅将 attention 转 float32，并独立检查输出与梯度。
这个最终 cast 是新下游接口，须预检，不能声称其等同旧 R12 的 float32
logit/softmax。各融合控制使用相同精度政策。若另选 float32 geometry 压缩，
必须事前明确并通过数值检查，不能因为磁盘或分数事后静默换精度。

按 704 pairs ×432² 计算，每个独立 float64 geometry operator 约
1,051,066,368 字节；full C float32 约 4,866,048 字节。一次只加载一个角色；
参数不同但纯 geometry 相同的角色可共享缓存，tau 在训练时各自计算。
**共享 geometry 不代表完整角色等价**：常数 T 与 trace tau 不得合并。
这只是容量估算，没有重新读取或测量数据。

**角色与归因。** 新 manifest 冻结 primary 的已选 h/T，以及同 h/T 的
constant-T Gaussian full1、trace-isotropic matched（使用原 full C 的同一
tau，几何才取 trace(C)/2 I）。保留旧单独校准的 full1、full4、quad4/16、
point/point_pck/fixed、当前锁定的强 point/quad 和定位角色；严格相同的
完整算子/configuration 可以 alias，语义比较仍保留。旧 R12 强 position
最小集合 point/point_pck/fixed、强 compression 最小集合 full1/quad4/quad16
不能被较弱角色替换。新角色属于哪个预声明比较集合须在训练前写明；不要
看深度结果后改集合。constant-T full1 是 tau 归因对照，trace-isotropic
是方向性归因对照，两者不能互相替代。

同容量 singleview 保留原 R7 自注意力 geometry、h=.5/T=5 和 query 自身 V，
不用 cross-source cache，也不为了让 primary 获胜而施加新 tau。
另报原始冻结 query depth。若 primary T=0，新 coupling 未生效，Q/K 梯度
应为零；即便任务通过，也不能归因为 trace 温度的学习内容收益。

**保持的验收与执行。** 五配对 seeds 11/22/33/44/55，每角色 1,000 steps、
batch8、AdamW(.001,.0001)、clip1、log-depth L1；每100步含0评价，最早最低
equal-scene AbsRel 选 checkpoint。初始化和 batch 序列须哈希相同。按
有效位置在 scene/seed 内累计，再等 scene/seed；保留全部历史和原始预测。
沿 R12/R17：对强 position AbsRel 至少改善2%，对独立重训 singleview 严格
改善；各比较两场景和至少4/5 seeds改善；对最强 compression 保留原精度/
成本二选一；RMSE 对强 position 和 compression 恶化均不超过2%。两旧场景
没有独立泛化区间。失败后不得靠换任务或先开新区域确认注销这项失败。

最少附加审计：checkpoint 参数数/初始化/梯度；在线 Q/K 参数更新确实改变
content/attention；zero-V 和 reverse-V 的完整保存预测；GT 字段注入不影响
前向；point C=0 与 T=0 极限；单位变换；缓存身份与独立数值复算。V 干预是
诊断，不用于选择 checkpoint 或替换 independently trained singleview。
小 tau 使梯度弱属于真实模型行为，不能强行要求其达到某个有利幅度。

完整计时按五旧场景首 pair、五 selected checkpoints、3 warmup/15 repeats
重算支撑、native integral、trace 验证/算术、learned Q/K、softmax、V/head；
保持 R17/R12 的配对与角色顺序。训练缓存吞吐不能代替此成本，公共预测器、
像素投影和传输单列。旧 R7 的训练耗时只说明小头任务规模，不能当 R17 估时。

## 2. 实际 optimized GS / RGB：有一个必须先解决的接口缺口

[R9 RGB 算子](../models/round9_gaussian_readout.py) **没有 content Q/K**，
只有 geometry softmax @ observed RGB。直接在旧 R9 里计算 tau 而不引入合法
content，输出不变；把 tau 乘 geometry 或 opacity 则改变了 R17 定义。不能
为重跑 R9 而掩盖这个缺口。深度任务通过也不自动给 primitive 一个128维描述。

最小物理适配应保留 R9 的单次颜色读出和原 renderer：

```text
Sigma_world = R * diag(s²) * R.T              # 真正优化后的 primitive R/S
(mu_image,C_image) = camera_project(mu_world,Sigma_world)
mass_j = integral_over_native_pixel_j Normal(mu_image, C_image + h² I)
alpha_j = softmax(log(mass_j + one_area_floor) + tau(C_image,h,T)*content_j)
primitive_color_per_source = sum_j alpha_j * observed_source_RGB_j
primitive_color = equal_average_over_fixed_eligible_source_views(...)
NVS = original_fixed_GS_renderer(means,scales,rotation,opacity,primitive_color)
```

这只是接口模板，**content_j 尚未由 R17 区域/深度合同定义，因此当前不是
可执行新 GS 候选**。后续若决定尝试，必须事前给出 context-only primitive
query descriptor 和 source-pixel key descriptor 的物理来源、相似度、归一化、
尺度、训练/冻结时点及所有控制的相同信息预算。已有 SH0 优化颜色与观测
颜色可提供一个不增加网络的内容来源，但颜色兼容性打分仍是新的任务假设，
不等同于 R17 的128维学习特征；可能只是把读色拉回原 GS 颜色。本文不选择
这个打分，也不扫描它。新网络、跨视图置信度或第二路 anchor 不是此准备的
默认补丁。若无法给出不牵强的内容定义，宁可暂留 GS 适用性未证明。

在内容合同有明确理由后，最少工程要求如下：

- **真实参数与隔离。** 直接验证 optimized R/s 重建 covariance、wxyz
  quaternion 和相机 Jacobian；沿 R9 context-only 24/4/4 图像角色与固定
  renderer/geometry/opacity/SH0 容量。旧 context-fitted 状态可在哈希及
  provenance 完整时作为共享固定输入；不是新的训练 seed 或新独立场景。
  若重建则仅用 context SIFT/三角化/颜色和固定训练预算，不读全图旧 points、
  tracks 或 target RGB 初始化。先预测后校准、冻结选择后验收保持不变。
- **单位和积分。** R9 h 是 pixel，R17 h 是 reference-8；C_px/64、mu_px/8、
  h_px/8 必须同时转换。不能把数值相同的两个 h 当作相同物理带宽。新 GS
  协议须事前决定固定物理转移还是独立的旧任务等预算校准，不能事后择优。
  若采用 R17 native integral，不再额外把 key I/12 协方差加进 S，否则重复
  处理像素面积。Renderer eps2d、opacity 和数值 PSD floor 也不得被叫作
  tau 的物理 support/confidence。逐像素积分与旧 EWA/五维点式不同，需独立
  数值检查，不能继承旧五维等价审计当新积分已过。
- **公平归因。** native point、fixed、isotropic、constant-T full 和 trace
  full 具有同一 content、source 集、归一化、V 和预算；另保留旧 EWA/full
  作为实现变化参考、bilinear 和 original-GS 渲染。不拿弱 isotropic 单独
  证明内部 R/S 有用。若共同内容是学习的，其参数量和训练预算必须等同。
- **可见性及失败范围。** 继续只用共享固定 GS geometry/opacity 的既有
  源可见性；每源视图各自归一化后等权融合。不可跨视图删除原来不能消去
  的行常数。无 eligible source 时统一保留原 context-trained 颜色；报告
  fallback 的渲染 alpha 贡献，保留全部目标像素，不只计可重着色部分。
- **验收不降级。** 原 full-pixel MSE、两场景等权与至少2%/两场景同向条件、
  最强 point/fixed/isotropic/bilinear envelope 及单独 original-GS 比较都须
  保留。仅胜过简化读取但输给原 GS，最多支持机制范围，不能称改善了系统。
  只在这个旧开发与实际 V 都有可信证据后讨论未触碰场景确认。

这条 primitive→image-cell 路径避开了任意 variable-key GS-to-GS kernel 的
五维不等价问题，但不解决该问题。也不能把规范化密度解释成渲染 alpha
或真实 surface visibility probability。C 仍是形状，不是估计误差的校准置信度。

## 3. 依赖、执行边界与最可能失败点

| 工作 | 可复用的只读依赖 | 必须新增/重新冻结 |
|---|---|---|
| actual-V 数据/头/统计 | R7 allowlist、V 定义、loss/metrics、同容量头；R12 pair-cache join 和训练/计时结构 | R17 角色 manifest、full C＋纯 geometry 缓存、在线 tau learned-content 适配、梯度/精度审计、来源锁、训练/验收脚本 |
| GS/RGB | R9 context-only 数据协议、真实 R/s 投影、renderer wrapper、visibility/fallback、全像素评价和独立执行审计 | 合法 content 定义；native pixel integral/单位适配；匹配控制；数值/成本检查和独立校准-验收锁 |
| 后续确认 | 旧 place 排除清单及先元数据后输入的流程 | 仅上游适用关口通过后，新的不重调参数/模型的完整联合确认协议 |

Python/Torch/NumPy/SciPy 与原冻结预测器使用现有环境。GS 另外依赖
MiniColmap 和既有 gsplat1.5.3 CUDA 扩展/renderer binary 来源锁；不为准备
任务安装新依赖、不改系统 GLIBC。GPU 和磁盘在正式启动前重新核验，不复用
旧瞬时空闲情况作为授权或性能证据。

主要失败风险是：宽支撑的 tau 过度削弱有用 learned content；T=0/极小 tau
造成机制不活跃；native cell 边界和非线性内容不满足连续点密度位移界；
R8 暴露的源值质量与 patch 多表面混合没有改善；singleview 能从相同 query
信息完成大部分校正；真实 GS 的渲染形状不对应应读取的颜色支撑，且 R9
fallback/覆盖误差会限制输出变化。大 C 只意味着此表示覆盖范围大，不能据此
判该 token 不可靠、自动降低它的 loss 或删除它。

这份准备的最小落点是**实际 V 先改一处在线内容乘法并保留原头**；GS 先解决
内容语义与物理单位的缺口。两者都保留单个位置分布和一次 V 读出，不靠额外
模块回避已有失败；上游或适用下游关口失败，就保留失败并停止该固定候选。
