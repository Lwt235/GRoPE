# 第六轮独立协议审查：真实区域对应与单一 Gaussian 相对位置核

2026-10-30。以下是**确认实验前的设计建议，不是已完成结果**。第六轮尚未冻结正式确认结果；新房间的原生端点已由主线程确认可访问，帧/深度/相机完整性与物理独立仍须数据盘点验收。不得改写第五轮封存的失败结论。

## 最早应固定的四件事

1. **同位置的强点对照。** 若 Gaussian 核加在最终 logits，point-distance/fixed/isotropic 核也必须加在同一位置。只胜第五轮无末端几何的 head，不能归因于 Gaussian 支撑。
2. **标签由真实传感器像素转移生成。** 禁止用 Gaussian、CDF、Mahalanobis 距离或候选 PE 核生成监督分布。面积分布不天然属于 Gaussian；真实边缘、遮挡和多峰都应保留。
3. **坐标统一。** 8/16 两种 token 的均值和 covariance 必须处在相同像素坐标，再比较或相加。不能各除自己的 stride 后直接做相对距离。
4. **新测试房间与双重验收。** 至少五个此前未使用的物理 place；区域概率分配有实际收益，同时不明显破坏原来的点对应精度。区域标签任务不能替代点任务的历史失败。

## 1. 具体问题与简洁实现边界

建议检验：在共享 RGB 预测几何与相机条件下，**Gaussian 支撑的相对位置核**是否比投影中心的距离核，在同尺度及跨尺度真实区域对应上更有效。信息条件继续是 RGB 预测深度加数据集提供的 RGB-D 重建相机，非完整 RGB-only。区域 covariance 是支撑，不自动等于中心估计不确定性；也不是优化后的 3DGS primitive R/S。

**最新主方案应优先选择确定性最小模型：** 冻结128D RGB特征的点积乘标量温度 `T`，直接加同一个几何项 `log K_ij`；不额外训练两层网络，**核系数固定1，只校准 bandwidth `h` 与内容温度 `T` 两个全局scalar**。所有point/fixed/isotropic/full使用相同预登记网格和validation汇总，不为full增加调参机会。训练两层attention的300步结果只作辅助开发，不能声称最小模型胜过已充分收敛的注意力网络。若后续改用attention，则所有对照采用相同`head`/`all`放置并在确认前冻结，核bandwidth跨层/方向共享。核在 **target/key图像坐标** 中比较 `q_to_k` 的 source Gaussian 与 target 的已知 native Gaussian：

\[
S_{ij}=C_i+C_j+\tau^2I,\qquad
\log K_{ij}=-\tfrac12(\mu_i-\mu_j)^\top S_{ij}^{-1}(\mu_i-\mu_j)
-\tfrac12\log\det S_{ij}.
\]

常数可省，但须固定是否保留 determinant 项；不能看到测试结果后切换。它是 Gaussian 密度的重叠核，不是 Wasserstein 距离。作为 logits 的 relative-position bias 不要求正定；若声称逐 token PE 的内积等价，则需要另给可分解特征映射/近似误差，不能把任意 pairwise 函数直接称为独立 token 编码。此路线属于相对位置编码的变体，不能冒称新的 Gaussian 期望恒等式。

最小主模型四种几何控制均为零个额外网络权重、两个同预算校准scalar；不得再给full单独增加强度/MLP/gate。内容-only控制可用于机制诊断，但其未使用的位置scalar不能计作同等有效自由度。最小性与实际收益是两个独立判断，不能用结构简洁代替新测试结果。

**当前均匀 target 网格中的简化必须明确：** native `C_k` 对所有key相同，故对固定query的 `S_i=C_{q->k,i}+C_k+h²I` 不随key变；`-logdet(S_i)/2` 是softmax行常数。已知 target 面积的 `log A_k` 也为行常数。两者严格抵消，不能宣称本实验验证了 mass correction 或 determinant normalization 的收益。实际可检验的主要机制是**query-dependent Mahalanobis位置形状/支撑宽度与跨尺度泛化**，它是已知机制，是否有实践收益仍需实验。只有另设不等面积key集合时才可单独验证质量修正；本轮不为此堆新任务。

这种简化给出精确5D方向性lift。令 `A_i=S_i^-1`、`a_i=A_i mu_i`，则 `Qpos_i=[a_x,a_y,-A_xx/2,-A_xy,-A_yy/2]`、`Kpos_j=[x_j,y_j,x_j²,x_j*y_j,y_j²]` 的点积，等于Mahalanobis logit省略query行常数。与 `[sqrt(T)*RGB, Qpos]` / `[sqrt(T)*RGB, Kpos]` 拼接即可；**不能再对整个拼接向量做L2归一化，否则不再等价**。必须独立验证概率、梯度及单位变换等价性。它是两端不同映射、条件于目标相机/已知native尺寸的逐token位置通道，不是全视角共享同一向量，也不支持任意逐key变化covariance的同一5D恒等式；不要过度扩展数学主张。

## 2. 第五轮 bottleneck 与必需的归因对照

第五轮两层 Q/K 的确使用了几何，V 也确实参与；但末端 48D 内容 head 没有几何路径直达 logits。固定点几何诊断在细阈值上较好，提示末端内容压缩可能损失定位信息，**尚未证明它就是失败原因**。第六轮先在旧 train/val 比较下列匹配结构：

| 对照 | 末端分数 | 能回答什么 |
|---|---|---|
| 内容基线 | 相同 `s_ij` | 末端重新引入几何的总体价值 |
| **point-distance** | `s_ij` 加同一单-bandwidth中心距离核，系数1 | 相同位置/相同信息的最强中心控制 |
| size-aware fixed | 同核；两端支撑替换为已知 token 尺寸的固定各向同性 | 区域大小本身是否足够 |
| same-trace isotropic | 同核；两端 covariance 各自替换为 `(trace C)/2 I` | 形状方向/轴比的独立价值 |
| full | 同核；完整两端 covariance | 主候选 |

固定对照须保留 8/16 的**已知尺寸差**，不能所有尺度都错误地置为相同像素 covariance。trace 对照应两端替换；原生 query covariance 本就由网格已知，需如实说明其与 fixed 的关系。可以留一个纯几何最近中心/同核无内容诊断，但不通过额外的 GT pruning 加强任何方法。

主归因顺序为：full 是否胜 validation 选出的强末端点/fixed 控制；再看 full 对 trace/fixed 是否有支持。如果只是所有末端核都胜旧 head，应结论为“保留几何直达定位改善”，不能声称 Gaussian 形状必要。支持足迹效应需要 full 对 fixed 的对比；支持各向异性需要 full 对 trace 的对比。不要仅凭冻结模型替换 covariance 的下降作因果证明，应重训练同预算控制。

## 3. 真实面积标签：测量合同

建议同一分辨率图像上构造 `8->8, 16->16, 8->16, 16->8` 四种条件。主线程批准的可实施采样为：对每个 pair，一次性投影 **192x144 均匀源像素中心网格**，把每个中心映回原生分辨率以最近邻读取真实 sensor depth；然后按 query/target cell 聚合，四尺度重用同一真实转移图。这是固定分辨率的真实面积数值求积，并非枚举全部640x480原生测量像素；边缘/薄结构的采样误差要披露，可仅在旧 train/val固定子集对更密采样做收敛核对。

对 query cell `i`：

1. `M_i` 为其图像矩形包含的源求积网格点数，即 stride8时64点、stride16时256点；按固定半像素 resize 约定映射传感器。不按深度/预测质量挑点。
2. 仅对正值 query sensor depth 反投影，用共享相机变换，检查正深度、目标画幅与目标 sensor-depth 一致性。继续沿用已审计的最近邻采样和事先固定的米制容差。
3. 每个观测可见像素投到 target 的**真实硬矩形 cell**，累计 `A_ij`。同表面、中心可见、单峰和边界纯净度都不是额外条件。
4. `V_i=sum_j A_ij`；`V_i>0` 时 `y_ij=A_ij/V_i`。`V_i=0` 的行没有监督，仍保留在 forward 与覆盖率统计中，不当作“全负 key”或 dustbin。

这里得到的是**条件于被观测为可见的源图像面积分布的192x144网格近似**，不是物理三维表面积，也不是 target-image 面积。透视压缩、遮挡、多表面 query 会形成非 Gaussian 甚至多峰标签；这些是任务要保留的现象。如果目标改为 target-image area，必须改变测度/权重并预先说明，不能把两者混称真实面积。

`f_i=V_i/M_i` 是观测可见覆盖比例，包含源深度缺测、目标缺测、画外与不一致的影响，并非无误差的真实可见率。主分析建议保留所有 `V_i>0`，在 place 内按 `V_i` 加权，即每个观测可见源像素等权；另报 query-row 等权敏感性。这样低可见行不被静默删除，也不与充分观测行享有相同测量权重。训练是否采用同一加权要在开发阶段写明。

**不建议 primary 使用 `f>=.95` 或“标签保留质量>=95%”**，这会选择少遮挡、较单峰的友好人群。若需要可见率分析，预先固定 `[0,.25), [.25,.5), [.5,.75), [.75,1]` 四档，全部报告，每档共同 GT mask；不得测试后改变主阈值。`A,V,f`、真实 depth、无效原因均不得进入输入几何/视觉特征、attention mask、置信度或预测质量筛选。

应提供独立解析小例子：同帧身份转移、纯平移跨 cell、遮挡边界、两个分离表面、缺测、16->8 的四块覆盖。每例直接数读取真实 sensor depth 的求积点，验证 label 归一/面积守恒；GT 深度扰动应改变标签，但前向输入逐元素不变。

## 4. 跨尺度单位与概率测度

最简单是全部几何在 192x144 的公共像素坐标：mean 用像素，covariance 用像素平方。由该分辨率的离散像素中心定义支撑时，native 均匀 cell covariance 为 `(s^2-1)/12 I`；连续矩形面积测度则为 `s^2/12 I`。**当前region exporter选择后者：3D像素中心scatter加回1x1像素footprint，native covariance用`s²/12`，全部输入共除固定reference stride8（cov除64），包括16stride图像。** 该处理须冻结，不能混称旧第五轮的离散精确footprint。反向匹配同样使用统一reference单位。

query/target 均来自各自完整独立网格，不通过 GT 对应对齐区域。投影 covariance 的 Jacobian、已知原生 footprint、有限 fallback 与原先的 label 隔离继续适用。

Gaussian 密度重叠与“落入 target cell 的概率质量”并不等价。若同一 key 集含不同面积，normalized density 通常还需要已知 target-cell 测度因子才能表达 mass；必须在开发阶段做“一个 key 分成四个”的概率守恒检查。如果每个方向的 target 网格内 stride 固定，则该面积因子是行内常数，softmax 可抵消；不能由此宣称任意自适应混合网格也自动守恒。

8/16 的类别数和目标熵不同，不能直接用 pooled CE 选最友好条件。建议 area 主指标用 `KL(y||p)=CE-H(y)`，另报有界 JS 和 CE；在每个 condition 内先聚合，再对四 condition 等权。KL 是分配误差而非密度/面积估计误差，零目标项要严格按 `0 log 0=0` 处理。真实标签不能为迎合 Gaussian 而拟合成单一矩分布。

## 5. 开发、独立场景与确认门槛

先在旧三间 train 与旧 c8 validation 做结构/尺度/损失可行性开发；新 `brown_bm_4/brown_bm_4` 明确划作 validation（64对），可用于正式checkpoint/基线选择，不能事后改称测试。旧五间测试已经打开，不能再作新确认。新 test place 不得属于先前九个顶层组：`harvard_c3,c5,c6,c8,c11`、`hotel_umd`、`mit_lab_hj`、`mit_76_studyroom`、`mit_dorm_next_sj`。同 place 的其他 sequence 也不算新房间。

**v2分割按模型表现读取前的物理独立性QA修订：** `brown_bm_1/brown_bm_1` 的固定12帧与validation `brown_bm_4` 都显示办公室/走廊及相似装修，无法排除共享物理区域。它被排除出确认测试、保留为aux；不是声称两者已证明是相同房间。v1清单、12帧身份/哈希和拒绝原因单独保留。第五测试place替换为官方native列表有效的 `home_bksh/home_bksh_oct_30_2012_scan2_erika`，保留其余四个 `brown_cogsci_1/brown_cogsci_1`、`brown_cs_3/brown_cs3`、`mit_32_d507/d507_2`、`home_at/home_at_scan1_2013_jan_1`；每place固定96对。新清单为独立`selection_v2.json`，不覆盖v1。

[SUN3D官方清单](https://sun3d.cs.princeton.edu/)与[3DMatch作者数据页](https://3dmatch.cs.princeton.edu/)将这些扫描分别列组，但不提供完整物理房间平面图；不能仅由字符串确认独立。固定首/中/末RGB已显示cogsci会议室、cs休息/咖啡区、d507会议/学习室的不同布局，未发现与旧9组的明显相同房间，但少量预览不能证明绝无共享走廊。继续限制为place/scan-heldout，不声称building-heldout；Brown/MIT/home采集组的留组敏感性仅作描述，home不虚称独立机构，五组不足以证明任意机构泛化。

确定性主模型没有随机训练重复，**不应复制五遍或随意初始化两个scalar来制造“五种子”**。在validation上预登记相同的`h,T`候选网格、按place及四condition等权选择，锁定每方法唯一校准结果，再对五个新place确认；不把calibration folds当独立测试场景。原“五训练seed”要求是针对随机学习网络的设计建议，现因主模型移除学习网络，在新测试打开前明确修订统计单位。辅助attention训练若保留，才使用五个配对种子，并单独报告，不能混入确定性主CI。

四个condition均用于validation校准时，只能称跨尺度匹配，**不能称未见尺度组合的零样本泛化**；后者需要另设明确未参与校准/训练的组合。

建议确认门槛如下，具体选择在新测试打开前写入正式协议，不可事后挑一项过关：

| 项目 | 推荐固定标准 |
|---|---|
| 主基线选择 | 每方法同预算validation校准后，仅按place及四-condition等权area KL，在point-distance/size-aware fixed中选较强者；平局用预定顺序 |
| area 主收益 | 四-condition等权KL相对下降至少2%；candidate−baseline的paired place 95% CI上界<0；至少4/5测试place改善，作为场景稳定性条件而非训练seed |
| 跨尺度泛化 | `8->16` 与 `16->8` 分别都满足至少2% KL下降、CI上界<0；不得只挑有利方向 |
| 同尺度保护 | `8->8` 与 `16->16` 分别 KL 相对恶化不超过2%；报告各自区间，若称非劣效应要求区间上界也低于预定margin |
| 原点任务保护 | 同一area-calibrated模型的同一`8->8` logits，另用独立真实中心点标签计算PCK@8；同时对area主基线与validation PCK-optimal point控制分别要求paired place95%差值CI下界>−0.5pp；后者也只可在相同预登记h,T网格按validation PCK选择，不能看test再挑；完整报告PCK4/EPE/point CE |
| 形状解释 | full vs trace、full vs fixed 的独立同预算结果支持相应解释；不靠主胜利自动推断各向异性必要 |

这是一个交集式验收：主结论需要所有预定主/保护条件都成立，不从多个尺度择优。其余探索性控制区间不宣称家族同时覆盖。若 KL 基线接近零，不能用不稳定百分比夸大收益，应在 validation 阶段预定绝对差或明确未达到实际收益标准。

点任务继续用真实 query 中心的既有 sensor-depth 对应，**不得从 area 分布的期望/argmax重建所谓点真值**。16-stride target 的点 PCK 另报原始像素阈值、每条件 nearest-grid oracle 与 EPE，避免粗网格上限混成算法效应。

确定性主统计为固定20,000次、RNG20261030的paired **place-only** bootstrap：先在place/condition内按预定查询测度聚合，再对place与condition等权；同一replicate对所有条件/方法共享place抽样。重叠pair/token不当独立数据，不复制虚假seed。至少五个新place，完整逐place/scale结果、留一place/采集组敏感性；只有五组，区间总体外推仍有限。辅助随机attention网络才另用paired crossed seed/place bootstrap，不能混淆两种估计目标。

## 6. 最小验收清单

- 旧 train/val 小样本证明所有匹配控制能学习、核强度/温度有限、单位变换不改变预测、GT 变更只影响标签。均值最近中心与点-bias是必须的开发比较。
- 像素计数标签通过解析例子；同一 pair 的四尺度转移图一致，所有 `V=0` 行/低覆盖行/无效原因完整记录；没有 Gaussian 标签或质量筛行。
- 新 place 元数据、pair选择规则、输入/标签分离缓存、共享冻结权重/预处理与完整性哈希锁定；不得复用旧测试作未见确认。
- 冻结主指标、所有保护条件、baseline选择、两scalar相同校准网格、唯一方法选择和place-only统计，再运行确认。辅助网络才另冻结五种子/统一训练步数与<=4 GPU小时预算；先做<=300步开发，不对某方法单独延长。
- 封存协议/源码/分析器/selection/实际input-label-manifest SHA、全部验证历史与最佳checkpoint；测试入口先核验封存并重放验证，再核验完整 test identities，最后打开测试评价。另保存固定概率子集供独立 NumPy 核算。

若 area 改善而点保护失败，应报告区域分配与定位的权衡；若只在同尺度或高可见档改善，应限制结论。若仅点-bias就解释全部收益，优先选择更简单的点方案，Gaussian 支撑假设保持未确认。
