# 第九轮独立协议审查：真实 3DGS 形状读取真实 RGB

2026-10-30。本文在查看第九轮结果前提出约束。研究问题是：**保持真实优化高斯的渲染几何不变，将它的投影形状用于读取源图像颜色，能否比点位置及简单尺度控制得到更好的新视角预测？** 本轮是两个旧开发场景上的投入筛选，不能替代全新场景确认。

## 可行性与结论边界

这个实验有可执行的最小结构：源图像 → 在源图像坐标下投影的高斯 → 每视图五维几何 attention 读取 RGB → 固定规则融合多视图颜色 → 同一个冻结 GS renderer。规则像素网格具有共同 key 支撑，符合当前五维 quadratic lift 的适用条件。无需增加深度预测网络或新的内容特征模块。

它检验优化 primitive 的内部旋转/尺度是否有助于真实观测读取，与前几轮“由许多 primitive 中心汇总得到区域协方差”的问题不同。所有对照的投影中心相同，只有用于读取颜色的支持矩阵改变。若改动高斯本体的尺度、opacity、排序或目标 renderer，则已经超出单独位置编码的归因。

纯几何五维 attention 与同一离散椭圆高斯核的归一化加权采样等价。此等价关系来自已有 [quadratic 实现](../models/gaussian_quadratic_pe.py)，不能把它当作独立性能创新。数值等价只说明把采样写成 attention 没有引入错误；后续贡献仍需说明真实形状相对于简单位置/尺度控制的作用、计算接口优势或新任务效果。两处开发场景通过也不能宣称通用 PE 已成立。

## 首要前提：几何训练的输入合同

已有 Tea/Waldo 重建不能直接冒充“只用源图像拟合”的模型。[前轮范围审计](round7_gs_scope.md)已说明这些共享重建可能使用过评价照片。新的 GS 拟合至少需要固定并保存以下清单及哈希：

- context 图像身份、目标图像身份、原始图像路径/哈希、相机参数和缩放规则；两个身份集合不相交。
- 初始化位置、尺度、旋转、opacity、颜色的来源，GS 优化实际打开的 RGB 清单、所有优化损失和训练步数。
- 相机估计/SfM、初始稀疏点云及归一化的来源；拟合结束后的 geometry、opacity、颜色、optimizer/checkpoint 和源码哈希。

只过滤旧点云中“在 context 被观察到”的点，不能消除其坐标曾由目标观测参与优化的影响。应使用 context 观测重新三角化、context-only 重建或其他不接触目标 RGB 的明确初始化，再进行仅含 context 图像的 GS 训练。旧 GS 参数不应作为新几何的隐藏初始化。

给定已知相机的新视角任务可以使用目标相机来渲染；相机来自全图 SfM 时，必须写明“给定共享 SfM 相机条件下，GS 参数拟合和颜色读取未使用目标 RGB”，不能升级为目标图像从未参与任何几何估计。若使用全图 SfM 匹配关系筛出 context 轨迹，也应披露共同匹配与相机来源。不能在缺少新 SfM 的情况下悄悄把范围声明扩大。

开发目标 RGB 只进入冻结预测后的指标计算及预先允许的超参数选择，不能进入 GS 几何拟合、source visibility、颜色读取、分支选择、fallback 或早停。目标图像用于开发选择后，只能称为“未参与拟合的开发目标视图”，不是独立确认集。

主线程采用的具体划分是：每个场景仅取旧 train/validation 图像块，即按名称排序的前两个三分块；均匀选择 32 帧，每第四帧作为目标，共 24 context / 8 target。八个目标再固定拆为四个 calibration、四个 gate。新提取 context-only SIFT、匹配并三角化，不读取旧 point XYZ、RGB 或 track 拓扑。GS 训练预算约 3,000 步，所有源视图使用三通道常量 RGB；最终以执行前封存的机器可读 split 与训练配置为准。

## 颜色容量、可见性与多视图融合

主比较应统一为每个 Gaussian 三通道常量 RGB（SH0 等价容量）。如果原始 GS 使用更高阶 SH，需另列完整 SH 原模型作为系统参考，并保留相同 SH0 容量的拟合颜色对照。所有读取角色使用相同颜色表示、数值范围、曝光/白平衡、背景和 renderer。若不进行高阶 SH 训练，本轮只能验证 view-independent 的简化真实 GS；真实指参数由渲染目标优化，而非复现官方大规模训练质量。

源图像 RGB 是前后多个 splat 和背景的 alpha 混合，并不等于某个 Gaussian 的隐含颜色。直接在投影 footprint 内平均 RGB 不是对 alpha 混合的逆运算，因此读取可能因遮挡、透明度、背景和纹理边界偏移而失败。失败必须保留，不能在看到目标误差后只选平坦区域。

source visibility 与 source-view 融合规则应只由同一个冻结完整 GS 及源相机确定，跨 full、isotropic、fixed、point、bilinear 完全一致。不能按每个读取控制自己的协方差重新生成 eligibility，也不能使用目标 RGB/深度筛选。若用近似中心 z-buffer 而不是完整透射率，应说明它忽略了部分遮挡，报告差异的适用范围。固定共享规则本身可能依赖完整 GS 形状，因此主差分是“在相同形状可见性条件下，读取 footprint 的增益”，不是全部 R/S 信息的有无。

可解释的优先实现是：固定完整 GS 的源视图 compositing 中，读取 Gaussian 投影中心处、其深度之前的累计透射率，按结果前固定阈值判定遮挡；或按同一 renderer 给出的可见 alpha 积分/未遮挡 alpha 积分比值判定。二者都无需颜色和目标信息。若接口只能提供归一化 expected depth，可采用预先固定的中心深度容差与 alpha 门槛，但需披露这是透明/多层表面的近似。本轮不应再按目标分数搜索 visibility 规则。所有角色还须共享 bilinear 可定义的像素中心范围 `[0,W−1]×[0,H−1]`，避免边界角色之间出现额外 fallback 差异。

没有合格源观测的 Gaussian 必须使用预先固定、对所有控制相同的 fallback，例如该同一 context-only GS 的拟合 RGB，不能删去其目标像素。记录 Gaussian 数量、可见视图数、fallback 数量，以及 fallback primitives 在目标渲染中的 alpha contribution；仅计 primitive 数量不足以反映对主分数的影响。

多视图实现需特别检查五维展开中省略的常数。每个源视图的投影均值/精度不同，省略的 query 常数只是**该视图内部**的常数。将所有视图的 keys 直接拼接后 softmax，会改变不同视图的相对权重。最简合同是每视图独立 softmax 读 RGB，再用所有角色一致的几何规则融合；若采用全局 softmax，则必须恢复每视图常数，并与完整密度核核验。跨视图 determinant 因子也不再自动抵消。

## 对照的精确定义与同预算选择

所有角色共享同一份拟合 GS 几何/opacity、源 RGB、相机、可见性、融合、fallback、目标 renderer 和完整目标像素集合。

| 角色 | 只允许改变的读取算子 | 能回答的问题 |
|---|---|---|
| full | 投影真实三维 R/S 得到的二维协方差，加入所有角色共享的像素支撑/数值 floor | 优化后形状是否有用 |
| isotropic | 在二维读取空间使用同 trace 的球形矩阵 | 方向/各向异性是否超越同总尺度 |
| fixed | 每个 Gaussian 使用同一预定或开发选择的二维范围 | 自适应尺度是否超越固定尺度 |
| point | 明确定义的最近点或零 Gaussian footprint 版本；仍保留共同 pixel 支撑时应注明 | 高斯支撑相对中心位置的作用 |
| bilinear | 在相同中心读取四邻域实际 RGB | 与标准连续点采样比较 |
| fitted RGB GS | 同一 context-only GS 原本拟合的 SH0/RGB，保持几何和 renderer 完全相同 | 直接读值是否优于已拟合颜色 |
| full EWA reference | 使用与 full 完全相同的离散范围、归一化、边界、visibility 的显式核 | 数值等价核验，不作为独立竞争方法 |

“same trace”应明确在投影后的二维空间定义；将三维 covariance 球化后再投影通常不会保留二维 trace，会同时改变尺度与方向因素。如果另做三维球化，须单独命名。

固定读取窗口、裁剪方式、带宽/温度、covariance 乘数和数值 floor 必须在看目标结果前写入机器可读配置。full 不得独享更大的搜索预算。当前方案是所有有尺度参数的角色各用相同四个 h 候选，按两场景 calibration 宏 MSE 选择，所有尝试、tie-break 和最终配置都保留；bilinear 是无调参控制。各 role 配置锁定后才打开 gate RGB 并运行一次 gate 评价。若所有方法仅用一组物理默认值，则结论限定该共同 operating point，不能解释为最优方法比较。

高斯截断、窗口边界和 source mask 会改变实际支持；EWA 与五维实现应在同一支持上比较。不同坐标原点会影响 float32 展开的消减误差，要求显式二维 Mahalanobis 的 float64 参考，并报告最大权重/颜色差，不只检查平均差。若窗口是基于 full 设定后用于所有角色，需披露该共同先验并保证最宽控制没有被额外截断。

## 指标、预设门槛及停止规则

主指标是所有预定目标图像的**全部 RGB 像素** MSE，颜色范围固定为 [0,1]。先在每图累计，再按预定 scene 内权重、最后两场景等权宏平均；如分辨率一致，仍应记录每图实际像素数。固定的背景、无覆盖像素、fallback 与边界都纳入主分数。PSNR、误差图、可见性层和纹理层只能补充解释，不能在失败后替换主指标。

继续门槛沿用用户已接受的计划：full 相对最强**非等价的简单位置/尺度采样控制**，gate scene-macro MSE 至少下降 2%，并且两个开发场景都同方向。最强对照应是固定评估集合中 gate scene-macro 最低者，不能每图为 full 选择较弱对照；这是比先在 calibration 锁定一个对照更保守的检查，所有 role 的 h 仍只能由 calibration 选择。另列 calibration 锁定最强对照的 gate 结果。为更谨慎，可另报各场景最强简单控制的 envelope，但不能事后把它替代已登记的主比较。拟合 RGB GS 单独提供系统能力参考。

门槛失败则停止该分支，保留所有负结果。若只优于 point/bilinear、却不优于 stronger fixed/isotropic，则不支持真实 R/S 的必要性。即使通过采样控制门槛，若仍明显差于同容量拟合 RGB GS，只能证明一个读取机制，不能宣布得到更好的实际 GS 方法。下一轮全新场景确认只能在源代码、配置、主对照和选择规则封存后启动，不能以不断试旧目标直到通过替代该规则。

确定性颜色读取对一个冻结 GS 重复运行五次，不是五个研究种子。若拟合 GS 本身使用独立随机初始化/训练顺序且重新拟合，才是有效的配对种子；各读取方法需复用同一 seed 的几何。只有两个场景时不把相关目标像素或图像当成大量独立场景，不提供虚假的总体显著性。首轮一次固定 GS 拟合的结果应明确称为小规模开发筛选。

## 最终实现与独立核验

结果前审查完成后，已只读核对封存的 [Round 9 协议](../ROUND9_PROTOCOL.md)、数据导出器、训练/读取 runner 及分析器。输入划分、3,000 步、单 RGB 容量、共同 source visibility、每视图归一化、四候选 h、原 RGB fallback 与 gate 规则一致。训练从 799/1,728 个新三角化点开始，按固定四次 growth 计划扩至每场景 12,000 个 Gaussian；增密只看 context 拟合的 opacity 与尺度，没有根据目标表现增加预算。

实际 visibility 使用固定 GS expected depth 的近似：中心位于闭像素中心范围、源 alpha≥0.5、`|z−D|≤0.02+0.03D`，深度单位为 context-camera 归一化单位。这符合结果前固定的简单可见性合同，但不等于真实遮挡标签。读取各角色共享完整几何提供的同一 eligibility。该限制不能因为没有胜过强控制而在本轮事后调参。

训练锁 SHA256 为 `7d7f53de51c4c179efad6b571f172bb966acf991342d24ebe0dc476dbdc4ffb2`。分析器在封存前完成 14 项测试，包括 gate 图像全部不存在时仍能完成 calibration、输入/源码/选择变更必须先于 gate RGB 读取被拒绝、完整像素指标及两场景方向门槛。实际执行使用包含 OpenCV 的 MiniColmap 环境；默认 Python 缺少 OpenCV 时相关集成测试跳过，不能将跳过视为已验证。

另用未导入模型或训练器的新代码完成三项独立审计：

- [数据重放](../results/round9_data_audit/verification.json)：从原 dataset metadata 独立重算选帧、K 缩放与 context 相机归一化，重放全部 48 张 context RGB；RGB 最大差为 0，相机最大差 5.55×10⁻¹⁷。没有打开 target RGB、旧 geometry/tracks 或封存第三块图像。
- [产物链与 fallback 审计](../results/round9_artifact_audit/verification.json)：核对 lock、run、training、state、Gaussian 导出、colors、predictions 与 operator sample 哈希链；独立 R/s 重建 covariance 最大相对差 4.95×10⁻⁷。18 个角色在无有效源视图的 primitive 上颜色逐位相同。该审计不打开 target RGB，补足主分析器不递归检查 state/training 产物的范围。
- [指标二次复算](../results/round9_metric_audit/verification.json)：主分析完成后，独立读取 16 张开发目标并复算 192 个 frame×method 的 MSE/PSNR、calibration 选择与 gate；不导入主分析器。最大指标差 3.47×10⁻¹⁸。仅复算已经锁定的六个 gate 方法，不为未选 h 追加 gate 搜索。

没有修改前轮或本轮封存源码，没有读取历轮封存测试输入、标签或预测。相机仍来自共享全图 SfM；以上通过是实现合同与算术正确性的结论，不是任务性能或因果识别通过。

## 对结果的独立判断

四个几何角色都由 calibration 选择 h=0.25；calibration 最强简单对照为 bilinear，gate 最强简单对照为 point。full 的 gate 宏 MSE 为 0.022208574，point 为 0.022241047，只降低 **0.1460%**，低于预设 2%。Tea 的误差反而增加 **1.7991%**，Waldo 降低 **1.2207%**，两场景方向门槛也失败。采用 calibration 锁定的 bilinear 作参考，宏下降只有 0.2296%，方向同样不一致。

full 相对同 trace isotropic 的宏误差降低 3.0606%，相对 fixed 降低 1.6522%。这说明各向异性读取优于某些更强平滑的控制，但不能推翻 point/bilinear 更强对照：现有结果不支持完整 R/S 支撑是读取真实 RGB 所必需的。不能只摘取 isotropic 比较作为成功证据。

与相同几何的原拟合 RGB GS 相比，full 宏 MSE 仅降低 0.04765%，而 Tea 恶化 8.5291%、Waldo 改善 4.3515%，总体数字主要体现相反方向的抵消，不能描述为稳定系统改善。

还须披露覆盖与几何质量：context 平均逐帧 PSNR 为 Tea 23.69 dB、Waldo 22.17 dB，是有限预算的小模型。gate 中 fallback primitives 占 alpha 总贡献的 27.976% / 53.097%，大于仅按 primitive 个数看到的 22.22% / 26.69%。因此本轮不证明真实形状在所有重建和可见性条件下无效，也不能假设改进 visibility 就一定过门槛。

按预定规则，本分支应停止，不进入新场景确认，不追加颜色/置信度网络或在已打开 gate 上继续寻找带宽。可保留“真实 GS → 共同像素网格的五维读取可以精确实现”这一算子结果；本轮没有建立“一般性、结构精炼且效果可靠”的新方法。
