# 第五轮：真实两层注意力与跨场景确认

2026-10-30。**当前“高斯期望 RoPE 后归一化”没有通过本轮实用收益门槛。**
30 组正式训练、5 个保留测试房间、独立统计与概率核算均已完成。
该结果缩小了前几轮的结论：条件区域匹配中的收益，尚未转化为本轮真实两层
cross-attention 的跨场景点对应收益。不能据此称已有一个效果稳定的新 PE 方案。

## 1. 实际验证的方案

从输入 RGB 预测公制深度，把每个图像 patch 的点云压缩成一个三维 Gaussian
$(\mu,\Sigma)$；$\Sigma$ 可分解为 3DGS 使用的旋转与轴尺度形式。跨相机投影得到
二维均值与协方差 $C=J\Sigma J^\top$。原视角 patch 的中心和范围已知，使用精确网格
及其离散均匀协方差，避免三维矩近似把原生像素中心移偏。

对已归一化的内容向量 $h$，第 $j$ 个二维通道块为

$$u_j=\exp(-\tfrac12\omega_j^\top C\omega_j)R(\omega_j^\top\mu)h_j,
\qquad \Phi(h;\mu,C)=u/\|u\|_2.$$

它就是**高斯期望 RoPE 加后置 L2 归一化**。独立复数参考计算验证了值与梯度的等价性。
不能把同公式再命名为一个新的 WorldRoPE 对照；本轮也没有复现 WorldRoPE 的完整
几何预测、V/输出变换或 NVS 网络。[算子审计](docs/round5_operator_audit.md) 记录了官方
论文、代码状态、确定性/概率 query 和混合分布边界。

网络保持简单：冻结 128D 视觉特征 → 96D 投影 → 两层双向 cross-attention →
共享 48D 内容匹配头。PE 只作用于注意力 Q/K；包含真实的 V 聚合、残差、LayerNorm
和 FFN，最终匹配头不另加几何项。六个方法均为 **166,611 个学习参数**。
开发集干预和梯度检查证明两层 V 路径实际参与计算；它仅证明实现有效，不是 PE 收益证据。

这里的 Gaussian 来自预测深度区域的矩，**不是优化后的 3DGS 基元内部 R/S**。
两者的贡献用独立的旧数据重训练实验区分，见第 5 节。

## 2. 信息条件与确认协议

- SUN3D 按顶层 place-ID 分成 3 个训练房间、1 个验证房间、5 个测试房间。
  训练/验证/测试分别 576/64/480 对；固定帧序号选样，不依赖模型分数。
- 冻结 ResNet18 和 Depth Anything V2 metric Hypersim Small；预测深度不通过
  验证/测试 GT 深度校准尺度。**所有方法共享数据集提供的 RGB-D 重建相机位姿**，
  所以准确名称是“RGB 预测深度＋已知重建相机”，不是严格 RGB-only 系统。
- 192×144 图像、24×18 token 网格，主指标 PCK@8 缩放后像素。原生传感器深度只
  构造对应与共同有效 mask，模型始终处理全部 432 个候选。没有预测质量筛选。
- 113,228/10,556/110,189 个训练/验证/测试有效 query。测试有效比例 **53.14%**；
  任务是条件于 GT 可见对应的匹配，不包含遮挡拒绝。最近网格 oracle PCK8 全为100%，
  各测试房间 PCK4 oracle 为77.97%–79.80%，细精度指标需结合量化上限解释。
- 每方法 5 个种子，均训练 1,000 步；以验证 CE 选最早的最优 checkpoint。
  按五种子验证 PCK8，预定强基线选为 `point`：94.5169%，`fixed` 为94.3482%。
- 全30个模型及选择结果封存后，先逐个重放验证集，再打开测试评价。所有输入、标签、
  场景身份、代码、检查点和选择记录均有哈希。
- 主门槛：normalized 相比所选基线 PCK8 至少+1pp、配对 seed/place 95% CI 下界>0、
  至少4/5种子正向、CE相对恶化≤2%。20,000次交叉 bootstrap；没有把 token 当独立样本。

完整信息条件、数值回退、分割、预算与门槛见 [冻结协议](ROUND5_PROTOCOL.md)。
五个 place 的区间仅支持有限的跨房间结论；这是 place-heldout，非 building-heldout，
冻结基础模型的预训练重叠也未知。

## 3. 主要结果

下表先在每个房间按有效 query 汇总，再对5个房间与5个种子等权平均。
EPE单位也是缩放后像素；PCK使用 argmax 网格预测。

| 方法 | PCK8 ↑ | PCK4 ↑ | CE ↓ | argmax EPE ↓ |
|---|---:|---:|---:|---:|
| 投影点 RoPE：预定主基线 | **92.351%** | 57.550% | 1.583625 | **4.6846** |
| 固定支撑＋后归一化 | 92.235% | 57.268% | 1.589427 | 4.7108 |
| 高斯期望，无后归一化 | 92.288% | **57.871%** | 1.585652 | 4.7155 |
| **当前 normalized Gaussian** | **92.281%** | **57.409%** | **1.584159** | **4.7137** |
| 同 trace 各向同性支撑 | 92.269% | 57.352% | 1.590871 | 4.7269 |
| Resolvent：预登记次候选 | 92.315% | 57.427% | **1.581133** | 4.6987 |

**主差值 normalized−point = −0.06955pp，95% CI [−0.39973,+0.16576]pp；3/5种子正向。**
CE相对变化+0.03377%，仅通过CE容忍条件，其余三项主门槛均失败。不能把 tiny 差异改写成
两个方法已证明等价；本轮区间支持的结论是，没有出现预定的+1pp实际收益。

| 测试 place | point PCK8 | normalized PCK8 | 差值 pp |
|---|---:|---:|---:|
| Harvard c11 | 78.573% | 78.803% | +0.230 |
| Hotel UMD | 92.397% | 92.042% | −0.355 |
| MIT studyroom | 97.382% | 97.287% | −0.095 |
| MIT dorm | 96.566% | 96.507% | −0.059 |
| MIT lab | 96.837% | 96.768% | −0.069 |

一个保留的小信号：相对同trace各向同性，完整形状 CE 下降0.006712，约0.42%，
区间[−0.013192,−0.001832]；但PCK8仅+0.01245pp，区间跨零。这是未做多重比较校正的
探索性概率分配差异，且完整形状未优于更强的 point 基线，不能据此推翻主失败结论。
Resolvent 的CE点估计略低，也不构成替代方案的确认。

所有方法、逐种子/场景、leave-one-place、机构组敏感性见
[统计结果](results/round5_analysis/summary.json)。

## 4. 值得保留的现象

训练/验证开发后、测试前另外登记了一个零学习参数诊断：在相同预测几何下，为每个
query 选择投影均值最近的 key。它不使用GT mask作预测，不参与主基线或门槛选择。
其测试 scene-macro PCK8为 **91.704%**，已经接近学习模型；PCK4为 **61.477%**，
argmax EPE为 **4.6161**，还优于上表的学习模型。

因此，学习模型在PCK8上的小幅改善伴随着细阈值精度损失。这个现象提示下一步应先
检查模型是否保留了已有几何精度。它不能单独定位为PE错误，也不能证明注意力网络
没有价值；最终头、训练目标和表示容量都属于本轮具体设置。

另外，训练/验证输入的有效支撑轴比中位数约1.07–1.12，多数区域接近各向同性。
这种数据分布为可变形状提供的额外信息有限。区域范围与中心预测误差也是不同量，
不能把局部patch的协方差直接称为估计不确定性。

诊断记录：[几何基线](results/round5_inputs/geometric_test.json)、
[训练/验证支撑统计](results/round5_inputs/train_val_support_diagnostics.json)、
[真实V路径](results/round5_dev_path_audit/README.md)。

## 5. 补齐真实 GS 内部协方差的重训练对照

对旧数据另训练 full 与双侧 center-only 各5种子，完全匹配原验证选中的尺度2、
训练预算与批次。仅移除基元内部协方差，保留中心分布、均值、特征与标签；该实验
明确是旧场景机制审计，不计作新的SUN3D跨场景确认。

五个 full 模型逐张量复现旧权重和验证选择。四场景 pooled CE：full1.170130，
center1.161619；full−center=+0.008511，种子/相机区间[+0.002257,+0.013687]。
但场景等权差值仅+0.005175，场景/种子/相机区间[−0.006149,+0.018029]；去掉Figurines后
差值约−0.000098，说明聚合优势主要由该场景驱动。备用分区也呈现同样限制。

结论是：在该旧设置中，充分适应后的重训练仍未支持内部R/S的独立增益；
不能推广成“R/S普遍无用”。详见 [重训练解释](results/round5_primitive_retrain_raw/analysis.md)。

## 6. 验收与复现

- 全部 **199项测试通过**，包括独立算子恒等式、投影/单位、GT泄漏隔离、真实V梯度、
  数据格式、配对统计和概率核算的回归测试。旧 `gaussian_pe.py` 的SHA保持不变。
- 30个正式训练全部完成，训练计时之和967.50秒，约0.269 GPU小时（不含开发、
  特征缓存、测试、下载）；使用重新确认空闲的物理GPU0/1/2，进程内均映射为cuda:0。
- [完整审计](results/round5_completion_audit/audit.json)：2,357项检查通过，涵盖
  30组配置/历史/最优步、源文件、实际缓存、封存、30次验证重放、完整pair集合和统计。
- [独立概率核算](results/round5_probability_audit/summary.json)：30模型×每场景前3对，
  共450个归档、15个不同pair。独立NumPy重算PCK计数完全相等；每有效query的CE最大
  差1.92e−7、soft-EPE最大差9.74e−6。它是固定子集核算，**不是全部480对概率的独立重放**。
- 另从完整测试PCK计数独立复算主差值和20,000次CI，与正式结果逐位一致。
- 旧GS实验：10个检查点、40概率归档、725项数据哈希通过验证，80个预测数组重放完全一致。

```bash
python -m pytest tests -q
OPENBLAS_NUM_THREADS=1 python -m experiments.round5_completion_audit \
  --out results/round5_completion_reaudit
OPENBLAS_NUM_THREADS=1 python -m experiments.round5_probability_audit \
  --runs results/round5_confirmation --out results/round5_probability_reaudit
OPENBLAS_NUM_THREADS=1 python -m experiments.round5_analysis \
  results/round5_confirmation/*/val.json results/round5_confirmation/*/test.json \
  --out results/round5_reanalysis
```

封存选择见 [confirmation_lock.json](results/round5_confirmation/confirmation_lock.json)，
源代码快照见 `results/round5_confirmation/source/`。大缓存、权重和概率数组在本机保留，
由.gitignore排除；元数据和分析结果可随仓库保存。

## 7. 当前研究决策

把当前算子保留为**经过审计的研究基线**，暂不将它升级为“已确认有效的新方法”，
也不据本轮结果直接扩大到完整NVS。它的数学形式已有对应方法，新增实验未体现
相对强点位置基线的实际收益，优化GS基元内部协方差的价值也仍缺乏场景级支持。

若继续该主题，下一步应先检验更具体的假设：**高斯支撑在跨分辨率、真实区域覆盖的
对应任务中是否有必要性**。保持简洁模型和同信息点/固定支撑基线，使用新的保留场景，
先验证支撑范围的独立价值，再决定是否设计新算子。这只是下一项可证伪假设，
尚不是一个已通过实验的新方案；本轮五个测试房间不能再被当作未见确认集。

## 数据与方法来源

[SUN3D官方数据与论文](https://sun3d.cs.princeton.edu/)；
[SUN3D官方格式说明](https://github.com/PrincetonVision/SUN3Dsfm#our-depth-file-format)；
[Depth Anything V2官方公制深度说明](https://github.com/DepthAnything/Depth-Anything-V2/tree/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf/metric_depth)。
输入与下载哈希见 `results/round5_inputs/selection.json`、`download_manifest.json` 和
`pretrained_sources.json`；物理场景划分理由见 [数据范围审查](docs/round5_dataset_scope_review.md)。
