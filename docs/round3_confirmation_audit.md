# 第三轮真实数据确认的独立审计

2026-10-30。审计范围：`experiments/real_support_geometry_probe.py`、两份
confirmation 的 source snapshot、冻结设置、逐行输出、数据 manifest 与协议。
没有修改实验源码、重新选择带宽/温度或训练模型。

## 结论

完整 3D 协方差相比保留法向/厚度/切向面积的 plane-isotropic 控制，确实
改善了当前真实区域的稀疏 SfM 轨迹分布预测；这一差异经过独立复算成立。
但是，**它尚未战胜按开发验证集选择的强 2D projected-center RBF**。
外部场景合并结果反而显著较差，并且存在明显的场景异质性。
这些结果支持“完整区域支撑含有有效信息”的局部机制结论，不能作为
“已得到效果可靠的简洁有限维 PE”的完成证据。

## 统计实现与归档一致性

主协议规定 row-averaged soft CE。当前 `cluster_difference` 对每个
`(scene,q_id)` 保留完整的多个 key 视图和 query-region 行，再对 camera
cluster 有放回抽样，统计量是：

`sum(sampled camera difference sums) / sum(sampled camera row counts)`。

它和逐行汇总具有相同 estimand。它**不是**先计算每个 camera 均值再对
camera 等权平均。若将后者的 CI 配给前者的汇总值，会产生统计量错配；
当前确认归档没有这个错误。例如 held-out primary 的 full-minus-plane
逐行差值为 -0.13990738，camera 等权均值则为 -0.16572395，不能互换。

独立复算结果：

| 检查 | held-out camera confirmation | external-scene confirmation |
|---|---:|---:|
| 逐行 CSV 重算 summary 的配置/场景数 | 456 | 456 |
| 6 项 summary 指标最大绝对误差 | 0 | 0 |
| full-vs-plane 差值/CI 复算数 | 96 | 96 |
| bootstrap CI 最大绝对误差 | 0 | 0 |
| 检查的数据/manifest SHA256 数 | 74 | 218 |
| SHA256 不一致数 | 0 | 0 |
| 同一配置内重复 row identifier | 0 | 0 |

复算 CSV 时把 `q_id/k_id/query_region` 转回整数，以保持归档代码的
数值排序及固定 bootstrap 抽样顺序。将它们当字符串排序会改变有限次
Monte Carlo 重采样的顺序，虽不改变统计量定义，也会使区间端点略有差异。

held-out source snapshot SHA256 为
`88bc42045ea48fd16572641745823e57ffd41275cdad0e3ea4eb12aeee665ff1`；
external snapshot 为
`97493d49d818bcaf3bea454935d434d4bf6618656e30ea3253212124ede8c1a8`。
两者均匹配各自 `source_sha256.json`。当前脚本与 external snapshot 一致。
后续脚本变动不应覆盖这两份历史快照。

## 强 2D 对照必须允许各自的已冻结 score mode

原有差值表主要在相同 score mode 内比较。对于机制消融这有用，但不足以
判断是否胜过完整的强基线。开发阶段已经独立为所有 mode 选择了带宽和
温度；开发验证 CE 对 mode 的偏好如下：

| 方法 | BC/RBF 直接归一化验证 CE | softmax(kernel) 验证 CE | 验证集偏好的 mode |
|---|---:|---:|---|
| world_full_bc | 1.21077 | 1.52986 | normalise_kernel |
| world_plane_isotropic | 1.35796 | 1.61561 | normalise_kernel |
| projected_center_rbf | 1.23533 | 1.16190 | softmax_kernel |
| pixel_center_rbf | 1.23354 | 1.18025 | softmax_kernel |
| projected_query_bc | 1.19923 | 1.32946 | normalise_kernel |

因此追加审计固定使用 `world_full_bc/normalise_kernel` 与
`projected_center_rbf/softmax_kernel`。后者已冻结 width=1 patch、temperature=8；
这里没有根据确认结果再选参数。以下为 primary partition 的**full 减基线**
CE；负值有利于 full。CI 仍为同一个 seed=219782、4,000 次 query-camera
cluster ratio bootstrap。这是补充比较，不重新定义原有 primary 检验。

| 数据 | rows / query cameras | full 3D CE | 强 2D CE | 差值 | 95% CI |
|---|---:|---:|---:|---:|---|
| held-out 合并 | 604 / 35 | 1.12805 | 1.09547 | +0.03257 | [-0.09880, +0.12372] |
| teatime | 299 / 17 | 1.22784 | 1.22564 | +0.00219 | [-0.27437, +0.17299] |
| waldo_kitchen | 305 / 18 | 1.03022 | 0.96786 | +0.06235 | [+0.01474, +0.11261] |
| external 合并 | 2448 / 107 | 1.33146 | 1.20782 | +0.12364 | [+0.06928, +0.17761] |
| figurines | 1450 / 54 | 1.59050 | 1.33426 | +0.25625 | [+0.18481, +0.32178] |
| temple | 998 / 53 | 0.95510 | 1.02413 | -0.06903 | [-0.10902, -0.03500] |

这一结论在 GS parity swap 和相同 validity mask 下仍成立：

| 数据 | partition/mask | full 减强 2D CE | 95% CI |
|---|---|---:|---|
| held-out | alt | +0.02635 | [-0.10474, +0.11586] |
| held-out | joint_primary | +0.03167 | [-0.10044, +0.12407] |
| held-out | joint_alt | +0.02742 | [-0.10611, +0.11840] |
| external | alt | +0.11113 | [+0.05870, +0.16572] |
| external | joint_primary | +0.12578 | [+0.07284, +0.17922] |
| external | joint_alt | +0.11366 | [+0.06162, +0.16712] |

不能只引用 `normalise_kernel` 下 full3D-minus-projected-center 的负值，
因为它忽略了验证集已显示更优的 2D baseline mode。也不能只展示 temple，
隐藏 figurines 的相反结果。以上 CI 描述这些场景内 camera 的变化，
没有对新的场景总体进行推断，也没有进行多比较校正。

## full3D 与 plane 控制的含义

令协方差特征值为 `lambda0 <= lambda1 <= lambda2`。控制保留中心、
最小特征值/法向，把两个切向特征值均改成 `sqrt(lambda1*lambda2)`。
它因此保留法向、厚度和切向椭圆面积（也保留三维 determinant），移除
切向椭率与方向。full/control 使用相同 row、key mask、标签和评分方式；
没有发现控制被额外改变中心或失配单位的问题。

这里“法向”只是协方差的最小特征轴，不是独立测得的真实表面法向。
primary 有效 q/k region 实例中，`lambda1/lambda0 < 2` 的比例分别为
teatime 37.8%、waldo_kitchen 35.0%、figurines 50.3%、temple 35.3%；因此
不少聚合区域不是清晰的薄平面。近重根 `<1.05` 只占约 0.2–0.4%，
没有发现大量完全退化的法向选轴，但不能将该控制叫作“真实 normal 上限”。

这一消融可以识别“完整的**聚合区域**协方差含有额外信息”，不能仅凭它
区分信息来自 3DGS primitive 内部协方差，还是来自区域中 primitive
中心分布的二阶矩。二者在导出时按 total covariance 相加；需要另行分解。
此外，保留同一 native covariance 而没有再调控制的独立温度，识别的是
当前参数化消融结果，不是证明所有 normal/plane PE 都不如完整 PE。

## 数据泄漏、掩码与适用范围

源码将 `q_features/k_features` 排除在本零训练 kernel probe 的加载字段中。
核只使用几何、相机和既定网格；SfM IDs 没有作为输入。`soft_labels`、
`label_valid` 和 track counts 用于选择/评价有效监督行，没有进入核公式。
新增的 label-mutation 回归测试验证：在 row mask 不变时，修改目标分布不
改变任何 kernel。因而没有发现“用正在评价的 BC 公式构造标签”的循环。

每张图的 GS 拟合来自本视图投影后的可见中心；q 与 k 使用互斥偶/奇
primitive 子集，alt 互换。joint 模式先求两边 q/k validity 的交集，才给
所有方法以同一个 key mask 条件化标签，避免比较不同监督集合。新增测试
对这一行为作了非平凡掩码验证。

四个场景的 train/validation/test camera sets 各含 54 张图，三组两两无
交集。held-out probe 只打开 test NPZ；external probe 在事前声明未参与
开发的两个新场景打开全部三组 camera blocks。后者不能因文件名包含
`train` 就叫 matcher training leakage，但必须保留 external 场景从未用于
带宽/温度选择的条件。

同时，场景的优化 3DGS 和 SfM 重建可使用这些照片。因此这是“已有场景
重建可用时的条件匹配”，并非从完全留出的照片重新构建几何的泛化实验。
轨迹转移标签是稀疏纹理显著点的条件分布，不是密集面积覆盖或遮挡真值。
保留质量门槛为 `>0`，可能舍弃任意多原标签质量后继续条件化；不能把
结果等同于原始未过滤任务，也不能混用主训练 runner 的 `>=.95` 协议。

## 从 exact kernel 到 PE 的缺口

`normalise_kernel` 产生 `BC/sum(BC)`；等价 softmax logits 为 `logBC`。
它与标准 softmax QK 的 `softmax(beta*BC)` 不同。后者才直接对应
内容点积固定时的平方根密度 tensor-lift PE。当前主 exact-kernel 正结果
不能自动转移到另一种 attention 归一化，更不能证明有限特征近似和训练后
任务表现同样有效。相关有限 PE、内容编码、强位置基线与开销必须继续验证。

## 新增回归测试

`tests/test_geometry_confirmation.py` 的 6 项测试全部通过：不等长 camera
cluster 的 row-weighted estimand 与 bootstrap、summary/差值一致性、BC 的
共同可逆仿射坐标变换不变性、两种 score mode 的区别、joint parity 公共
掩码/条件标签、以及标签不进入 kernel。运行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest tests.test_geometry_confirmation -v
```

这些测试是正确性证据，不是效果、创新性或研究任务已完成的证据。
