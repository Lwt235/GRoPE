# 真实 GS 区域协方差的归因实验

预先固定的问题：区域拟合协方差

\[
C_{\rm total}=\sum_a w_a\Sigma_a+\sum_a w_a(\mu_a-\bar\mu)(\mu_a-\bar\mu)^T
\]

中，真实 GS 内部协方差 \(\Sigma_a=R_aS_a^2R_a^T\) 是否在区域点云形状之外独立贡献性能？`centeronly` 保留第二项，因此它已经包含完整区域形状，不能称作仅中心坐标基线。`internalonly` 仅保留第一项；`flooronly` 仅保留原数据的固定极小数值 floor，不是经过宽度优化的强中心基线。

所有方法使用相同区域均值、相同 opacity 权重、相同 CPU opaque center z-buffer、相同偶/奇 GS 分区、相同有效候选及查询行。分别对反事实协方差作相同固定 eigenvalue floor `(1e-4*scene_scale)^2`。未 floor 的两个分量严格相加为总协方差；独立 floor 后无需相加相等。每次拟合都与原 `fit_regions` 的 float64 实际输出逐项核验；已有 NPZ 的 float32 量化误差另外记录。

主评分固定为 exact world-space Bhattacharyya coefficient 的逐行归一化 `BC/sum(BC)`，在 log 域计算，零拟合参数。主要指标为 full 减去 centeronly 的 soft CE 差。保留 KL、top1 target mass 和 max-bin accuracy。相同原始 >=4 common-track 标签行，过滤 q invalid，屏蔽 invalid keys，条件化归一化剩余目标质量。主 mask 保留严格大于零质量；另外完整报告 >=95% retained mass。primary 为 q-even/k-odd，alt 交换；joint 两组在两套相同有效 mask 下比较。统计以 query camera 为 cluster，4,000 paired bootstrap；这只描述当前场景内的相机差异。

开发阶段只打开 teatime、waldo_kitchen 的 train/validation pair NPZ。所有方法、floor、mask、指标在运行前写入 `results/round3_cov_decomposition/frozen_protocol.json`，不调任何超参数。随后独立确认可打开这两个场景 test，以及 figurines、temple 全部固定相机块；确认不使用开发结果改变方法或参数。原始 GS 几何、相机和 SfM 模型可含全部原始照片，因此这是匹配归因，不是未见照片的重建泛化。

所有归因都针对既定 exporter：内部协方差不参与 opaque-center visibility，删除内部协方差并不重新 rasterize。其贡献若显著，表示在该固定区域模型下提供额外支持信息，不能推出 GS 参数化独有、也不能排除点云的其他估计器复现同类形状。

复现：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -p test_gs_cov_decomposition.py -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.gs_cov_decomposition --phase development
# 开发输出审核后，以完全相同的预先冻结协议确认：
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.gs_cov_decomposition --phase confirmation
```

结果保存在阶段子目录的 `summary.json`、`per_row.csv`、`README.md`、相机统计及逐 pair 分量 NPZ。

## 实测结果：GS 内部协方差没有带来独立收益

开发集 pooled validation 的 full CE 为 **1.210772**，centeronly 为 **1.192461**；差值 **+0.018312**，paired camera 95% CI **[+0.007414,+0.027590]**。CE 越低越好，因而删除内部 GS 协方差反而改善结果。这个发现没有用于重新选参数；确认阶段继续使用运行前冻结的全部四个零参数方法。

| 独立确认数据 | 有效 query rows | full CE | centeronly CE | full−centeronly CE | camera-cluster 95% CI |
|---|---:|---:|---:|---:|---|
| teatime test | 299 | 1.227837 | 1.207240 | +0.020597 | [+0.012186,+0.032876] |
| waldo_kitchen test | 305 | 1.030219 | 1.009957 | +0.020262 | [+0.014251,+0.026823] |
| figurines external | 1,450 | 1.590505 | 1.463362 | +0.127143 | [+0.106942,+0.144945] |
| temple external | 998 | 0.955100 | 0.947947 | +0.007153 | [+0.004417,+0.009582] |
| 合并（相机混合，非场景总体估计） | 3,052 | 1.291206 | 1.224419 | +0.066787 | [+0.051488,+0.081727] |

四个场景、两种 primitive 分区、joint masks，以及 >=95% retained-mass 过滤的 CE 差均同方向，且上述每种场景/条件的 95% camera interval 均大于零。合并 top1 target mass 也由 full 的 0.588039 提高至 centeronly 的 0.601625。temple 的 top1 target mass 几乎不变（0.687652 与 0.687532），所以不能把每个指标都表述为改善。

内部协方差不是数值零。按各相机/parity 的有效区域均值再平均，确认阶段 internal / total trace 比例分别约为 teatime **1.93%**、waldo **5.94%**、figurines **4.68%**、temple **2.19%**；在总协方差最小特征轴方向的方差比例约 **13.68% / 17.19% / 29.34% / 10.22%**。即使它对法向厚度有可测贡献，这种贡献也没有转成当前 exact normalized-BC 任务的性能收益。

`internalonly` 的合并 CE 为 121.945，`flooronly` 为 438,188.847。二者的协方差相对跨区域中心偏移过小，导致对错误概率极其自信；这些数值只能说明它们作为当前未经温度/宽度标定的核不适合此任务，不能用来宣称该方案战胜了所有中心方法。

因此，此前 full regional covariance 相对 isotropic covariance 的收益，可以由 **trained GS centers 的区域点云形状** 解释；没有证据把该收益归因给 3DGS 内部 \(R,S\)。这不证明内部协方差在别的任务、token 定义、可见性模型或学习型 PE 中无用，也不把 trained GS centers 等同于未经训练的原始 SfM 点云。它直接否定的是“当前区域 BC 实验已证明 GS 内部 \(R,S\) 独立有效”的更强说法。

## 已完成的验证

四项专门单元测试全部通过，覆盖总协方差分解、原 exporter 重放、covariance 不改变固定可见性/均值/权重、固定 floor 及 invalid-region 行为。开发只读取 **144** 个 teatime/waldo train/validation pair NPZ，测试/外部 pair 为零；确认再读取两个场景各 36 个 test pairs 和两个外部场景各 108 个 pairs。原始几何/模型/相机及所有读入 pair 的 SHA256、逐相机统计、逐 query-region 指标均保留。`verification.json` 另外核验开发 summary、协议及实现 SHA 在确认前后不变。开发主/alt 的 >=95% mask 与 float32 主训练 loader 比较，没有边界精度差异。
