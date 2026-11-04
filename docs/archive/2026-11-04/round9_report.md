# 第九轮：真实3DGS参数读取观测颜色的最小验证

2026-10-30。**实验完成，未通过预设继续门槛；停止这一具体分支。** 将真实优化后高斯的内部旋转/尺度转成五维位置特征，从输入图像读取RGB，再用固定GS渲染器预测留出视图。完整协方差相对最强点位置对照的验收RGB MSE只下降 **0.146%**，两个场景一好一坏；预设要求至少2%且两场景同向。相对本轮原GS颜色渲染仅下降0.0477%，也由相反场景结果抵消而来。

这轮确实使用真实优化的各向异性高斯、实际源RGB和alpha合成渲染，不再依赖预测源深度作为V。但它是两个旧开发场景上的小规模、给定相机实验，不能证明内部R/S在所有任务无用，也没有得到满足一般性目标的新PE方案。

## 从原idea到本轮实现

每个真实高斯携带中心和内部形状：

\[
G=(\mu,\Sigma),\qquad \Sigma=R\operatorname{diag}(s^2)R^T.
\]

将其投影到某张输入图像得到均值m、协方差C。图像像素是规则key网格，公共像素协方差为I/12。令

\[
A=(C+I/12+h^2I)^{-1},\qquad
w(x)\propto\exp[-\tfrac12(m-x)^TA(m-x)].
\]

沿用第六轮五维非对称Q/K二次展开，softmax后读取实际源RGB。每张源图像分别归一化，再对同一个高斯的合格源视图等权平均；没有跨视图误删不同的行常数。最终只替换高斯颜色，中心、旋转、尺度、透明度和渲染器完全相同。

纯几何五维式与相应归一化椭圆高斯加权采样数学等价。这一轮检验内部形状是否有实际信息价值，不把展开式或与EWA相同的结果当作新机制。当前精确性仍依赖规则、共同协方差的图像key；没有扩展成任意GS-to-GS五维精确编码。

## 输入隔离与重建

旧GS的稀疏初始化来自全图SfM，不能直接作为本轮context-only拟合结果。因此新生成了两套重建：

- 场景为旧开发Teatime和Waldo kitchen，按相机文件名排序只用历史前三分块中的前两块；各均匀选择32帧，每第四帧为target，得到24张输入、4张校准、4张验收图像。历史第三块照片与历轮封存测试产物未打开。
- 只从24张输入照片重新提取SIFT、匹配、三角化、取色，未使用旧3D点、旧keypoints/tracks或Gaussian checkpoint。初始799/1728点，经预先固定的四次增密到每场景12,000个高斯。
- 相机K/R/t沿用既有全图SfM标定。因此严格表述是**给定相机条件下，目标RGB不参与本轮几何初始化、GS拟合和颜色读取**；相机来源并未实现全流程未见视角隔离。
- 每场景一个seed11重建，固定3,000步、全图MSE；采用每高斯一个RGB的视角无关表示（SH0容量），固定最后一步，不用目标选择checkpoint。模型和每种读取结果不是独立训练种子。

输入分辨率192×144，RGB保持sRGB，统一INTER_AREA缩放。输入视图平均逐图PSNR为 **23.69/22.17 dB**，平均alpha为0.952/0.944。重建正常工作，但留出视图仍有明显几何/覆盖误差。这是小规模SH0验证，不是原有百万高斯SH3模型或完整预算3DGS复现。

## 公平对照和预先选择

四种位置编码各有相同的四个h候选：0.25、0.5、1、2像素。

| 方法 | 查询支撑协方差 |
|---|---|
| full | 真实高斯投影C |
| isotropic | trace(C)/2 × I |
| fixed | 固定I，即每轴1像素平方 |
| point | 0，仍保留公共I/12+h²I |

另加无调参的双线性中心采样，以及本轮GS原来优化得到的RGB渲染。全部颜色表示、几何、透明度、源图像集合、可见性与fallback相同。没有增加颜色网络、置信度网络或学习预测头。全像素softmax采用float64，只按内存分块，没有裁掉高斯远端像素。

先生成全部 **18配置×8目标视图×2场景=288张预测**，完成输入与产物哈希检查，再打开校准RGB。每角色按两场景校准宏MSE选h，四者均选0.25；写出选择锁后才打开验收RGB。主对照预先定义为校准后四个简单方法在验收上的最强者，实际为point；同时保留校准阶段选出的bilinear，避免依赖单一选择说法。

## 验收结果

主指标包含每张验收图的全部像素和RGB通道，先均匀平均场景内四图，再等权平均两场景；没有GT可见性筛选或颜色/曝光对齐。共 **8张图、221,184个像素**。

| 方法 | 校准MSE | 验收MSE↓ | 从宏MSE换算的PSNR↑ |
|---|---:|---:|---:|
| full | 0.020094805 | 0.022208574 | 16.5348 dB |
| isotropic | 0.021589113 | 0.022909748 | 16.3998 dB |
| fixed | 0.019902658 | 0.022581662 | 16.4624 dB |
| point | 0.019492259 | 0.022241047 | 16.5284 dB |
| bilinear | 0.019456183 | 0.022259686 | 16.5248 dB |
| 原GS RGB | 0.016328053 | 0.022219161 | 16.5327 dB |

full比最强point的MSE仅降 **0.1460%**，比校准锁定bilinear降0.2296%，两种判定都失败。full比isotropic降3.06%、比fixed降1.65%，但这些弱于point的对照不足以支持内部形状具有不可替代的价值。

| 场景 | full MSE | point MSE | full相对point |
|---|---:|---:|---:|
| Teatime | 0.016114925 | 0.015830127 | 恶化1.799% |
| Waldo kitchen | 0.028302222 | 0.028651967 | 改善1.221% |

原GS参考也没有被可靠超越：full的宏MSE改善只有 **0.04765%**，Teatime恶化8.53%、Waldo改善4.35%。上表PSNR是从宏MSE换算；另存的逐图PSNR均值中，full为16.99465、point为16.99548、原GS为17.00084 dB。聚合顺序不同产生极小方向差异，均不改变主MSE门槛失败。

只有两个重复使用的开发场景、每场景四张验收图，不能估计场景总体显著性。完整带宽校准、逐图与逐场景统计见[独立分析](results/round9_analysis/README.md)及[summary.json](results/round9_analysis/summary.json)。

## 可见性与失败范围

源可见性完全由固定GS几何/透明度估计：中心在有效像素中心范围，alpha≥0.5，深度与渲染expected depth相符。它不是精确表面可见性，透明/混合层可能误判。各方法共享这个资格，不根据预测效果调整阈值。

| 场景 | 至少一个源视图可读取的高斯 | 无可读源、保留原色的高斯 | fallback在验收alpha总贡献中占比 |
|---|---:|---:|---:|
| Teatime | 9,334/12,000 | 2,666/12,000 | **27.98%** |
| Waldo kitchen | 8,797/12,000 | 3,203/12,000 | **53.10%** |

primitive数量比例不能代替图像影响范围：Waldo超过一半的验收alpha贡献仍来自fallback。该限制可能削弱改色影响，但它不能证明修正可见性后一定有效，也不授权看结果后删掉这些像素。

以下固定展示每场景在预定顺序中的第一张验收图，没有按分数挑选。可见简单重建在未见视图仍有模糊和几何误差；full与point的差异很小。图中MSE是单帧指标，不代替上面的全体结果。

![每场景第一张验收图，实际目标与三种固定几何渲染](results/round9_figures/first_gate_views.png)

## 独立核验与研究决策

- [数据重放](results/round9_data_audit/verification.json)：独立核验24/4/4划分、相机归一化、48张输入RGB，RGB最大差0，相机误差≤5.55×10⁻¹⁷；不读取目标RGB或封存第三块照片。
- [评价前完整性检查](results/round9_pre_evaluation_integrity.json)和[独立产物审计](results/round9_artifact_audit/verification.json)：源码、数据、checkpoint、原高斯、颜色、预测及算子样本哈希链一致。18角色在fallback高斯处使用相同原色。
- [实景算子审计](results/round9_operator_audit/summary.json)：48源视图、371个实际高斯—视图样本，独立NumPy全像素Mahalanobis与保存五维读取结果最大RGB差 **3.04×10⁻¹³**；独立R/s与投影协方差差≤9.09×10⁻¹³。等价性成立，不能当作额外性能收益。
- [独立执行重放](results/round9_execution_audit/verification_after_evaluation.json)：训练批次哈希与四次增密符合协议；所有源视图可见性/读取计数一致，完整双线性读色及固定primitive的直接Mahalanobis融合误差≤2.99×10⁻⁸；**288张目标预测、alpha及fallback贡献重放误差均为0**。不读取目标RGB，不导入本仓库训练器或模型。
- [独立指标复算](results/round9_metric_audit/verification.json)：192个frame×method统计及带宽选择、宏平均、门槛最大差3.47×10⁻¹⁸，独立确认失败。
- 相关测试 **215项全部通过**（8.11秒），覆盖第六至九轮保留算子、输入合同、梯度、像素约定及分析；实现正确不等于研究效果通过。

本轮按[预先协议](ROUND9_PROTOCOL.md)停止“直接用真实高斯内部形状读取RGB颜色”这一分支，不以新增网络或重选验收人群挽救分数，也不为该失败候选启动新场景确认。仍保留第六轮区域支撑PE的限定正结果；当前实验不能把它推广成通用3DGS PE。

结果收窄了可行方向：内部渲染高斯的范围，不自动等于应读取观测颜色的范围；高斯形状用于位置关系需要和token所代表的信息一致。这个解释与实验相容，但并非已独立证明的唯一失败机制。一般性、精炼且实际有效的原始研究目标尚未完成。

## 资源与复现

训练协议锁SHA256：`7d7f53de51c4c179efad6b571f172bb966acf991342d24ebe0dc476dbdc4ffb2`。本机保存源码快照、两套GS、18配置原始预测、颜色及独立统计。先前封存源码不修改。

新数据准备约8.78秒CPU；两个GS训练/输入评价合计约23.18秒GPU，两张空闲RTX A6000；源颜色读取及目标预测合计约23.83秒。另有实现、输入/预测I/O和审计成本。此计时只代表192×144、12,000高斯的小规模实验，不是高质量3DGS系统训练成本。

```bash
/home/liwt/miniColmap/.venv/bin/python -m pytest -q tests/test_round9*.py
/home/liwt/miniColmap/.venv/bin/python -m experiments.round9_gs_readout freeze --out results/round9_gs_replay
CUDA_VISIBLE_DEVICES=<空闲GPU> /home/liwt/miniColmap/.venv/bin/python -m experiments.round9_gs_readout train --scene teatime --out results/round9_gs_replay
CUDA_VISIBLE_DEVICES=<空闲GPU> /home/liwt/miniColmap/.venv/bin/python -m experiments.round9_gs_readout train --scene waldo_kitchen --out results/round9_gs_replay
CUDA_VISIBLE_DEVICES=<空闲GPU> /home/liwt/miniColmap/.venv/bin/python -m experiments.round9_gs_readout readout --scene teatime --out results/round9_gs_replay
CUDA_VISIBLE_DEVICES=<空闲GPU> /home/liwt/miniColmap/.venv/bin/python -m experiments.round9_gs_readout readout --scene waldo_kitchen --out results/round9_gs_replay
/home/liwt/miniColmap/.venv/bin/python -m experiments.round9_execution_audit --phase before-evaluation --runs results/round9_gs_replay --analysis results/round9_analysis_replay --out results/round9_execution_replay
/home/liwt/miniColmap/.venv/bin/python -m experiments.round9_analysis --runs results/round9_gs_replay --data results/round9_context_data --out results/round9_analysis_replay
```

先以实际空闲GPU编号替换占位符；本机环境依赖既有MiniColmap venv、固定gsplat1.5.3 CUDA扩展和带来源哈希的输入数据，不安装依赖或修改系统GLIBC。分析器先校准锁定再开验收，存在的输出目录拒绝覆盖。
