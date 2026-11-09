# GRoPE：高斯参数化与 Transformer 位置编码

研究方向是将高斯参数化用于结构精炼、具有可信实验支持的 Transformer 位置编码。2026-11-07 用户明确近期优先级：固定完整五维，查清相对强位置编码的可靠适用范围，接受有证据支持的有限范围成果；一般性仍是待检验的问题。

**当前统一入口：[研究主文档](docs/research_current.md)。** 文档包含研究动机、五维相对关系 `relative5` 的公式、完整证据链、关键对照、负结果、已完成的新场景确认及最新预测分析。

**最新区域重构新场景确认已完成并实际审阅：主用途为混合结果。** 九个可用新地点、864 帧对、12 个既有模型、36 组预测，零训练。重构条件 F−Distance 召回 +1.919 pp、三种子均正，区间 [0.019,3.841] pp；F−GeoTransformer / 固定球形为 +1.704/+0.928 pp，各有一个负种子、区间跨零，稳定强对照优势和额外协方差贡献未完整确认。基础与固定成员稀疏条件对三项控制仍为三种子和联合区间正。保持五维，结束本次重构推广确认；覆盖从 35.479% 降到 19.170%，不称总体更耐稀疏。预定十地点中 home_ph_cz/ph_cz_1 缺外参，保留缺失。[固定确认方案](docs/region_retokenization_confirmation_plan.md) · [完整结果与实际审阅](results/region_retokenization_confirmation/attempt1/report.md)

**此前旧数据边界检验保留为开发证据。** 重构后 F 对 Distance/GeoTransformer 条件召回 +2.463/+2.107 pp，支持进入上述新场景确认；新结果显示旧数据正面信号没有完整迁移。完整五维不变，保留所有正面与混合结论。[旧开发结果](results/region_retokenization/attempt1/report.md)

**原生区域 PE 的四组比较已完成并实际审阅，未建立可靠用途。** 12次完整训练、72组评价全部正常结束。完整五维基础mCA为78.457%，对Distance/Geo分别+0.446/+1.066 pp，均有一个种子退步且联合区间跨零；对中心散布控制仅+0.043 pp。改变输入数量时四组都明显退化。按事前决定结束这条固定区域构造接法，不自动补种子或扫描分组；保留完整五维及既有观测区域匹配证据，总体目标尚未达成。[固定方案](docs/native_region_pe_plan.md) · [完整结果与实际审阅](results/native_region_pe/attempt1/report.md) · [独立复算](results/native_region_pe/attempt1/execution_checks.json)

**2026-11-09：平均精度归一化的六次训练、评价及实际审阅完成，机制收尾结束。** 归一化 N 的区域召回为 39.750%，对 Distance 高 3.081 pp、三个种子均正；完整五维 F 仍高于 N 0.878 pp。原生 N 的 mCA 为 79.344%，对 Distance 高 0.917 pp，但种子方向混合、联合区间跨零，未解决强对照缺口。N 在 ModelNet 基础子组有改善、MACGS 有退步，来源与类别混杂，只保留为线索。固定完整五维，不自动追加消融或晋升 N。[固定方案](docs/normalized_precision_plan.md) · [结果与实际审阅](results/normalized_precision/attempt1/report.md)

**精简版：[固定方法、核心证据与适用边界](docs/research_core.md)。** 用户恢复自主目标后，全查询区域匹配诊断已经完成并审阅：复用 75 份旧预测，无新训练。旧验证阈值下，五维少输出正确、也少输出错误；最高置信少量匹配没有领先，保留较多候选时的均值仍有排序收益。原训练未直接监督无对应拒识，因此结果限定于原模型加最大 softmax 规则。区域对应排序仍是最可靠的用途，可靠输出优势和一般性目标均未确立，不自动新增拒识模块或搜索置信度。[完整诊断与审阅](results/region_selective_matching/attempt1/report.md)。当前目标为 active。

**2026-11-08：同等数量增强的六次原生训练及审阅完成，按事前约定收束本次推广尝试。** 混合训练后，256/512/1024 基元评价的五维−Distance mCA 为 **+0.307/+0.989/+1.599 pp**，每个数量都有一个种子退步，联合区间较宽；没有确立稳定优势。256 条件下 Distance 的改善更大；512 差距缩小却来自两者准确率下降且五维下降更多，1024 差距扩大也来自 Distance 下降更多，不能据此证明旧收益全部来自数量不匹配。五维 CE 在九组新比较中均更低，正面信号保留。完整五维继续为主候选，核心已确认用途以观测区域匹配为主；原生保留形状归因与有限条件证据，不晋升混合训练，不自动增加消融、RGB 或新下载。原生审阅时目标暂停，后已恢复并完成上述区域诊断。[范围与决定](docs/research_current.md#101-最新收敛安排先决定原生任务的适用边界) · [本轮固定方案](docs/native_count_augmentation_plan.md) · [完整结果与实际审阅](results/native_count_augmentation/attempt1/report.md)

**方向无关平均精度控制已全部完成并实际审阅，未保住主要用途。** 六次完整训练及 168 组新旧预测比较结束。Q 的区域召回/原生 mCA 为 **37.641%/77.380%**，完整五维分别高 **2.988/2.649 pp**，两个任务三个种子均正；区域中心定位也共同改善。区域差的联合区间 [2.122,4.181] pp，原生 [−0.663,5.883] pp 仍宽。Q 的原生基础准确率比 Distance 低 1.046 pp、三个种子均低。不推广 Q，保留完整方向关系；平均精度在方向存在时的条件性作用仍待检验。[固定方案](docs/mean_precision_plan.md) · [完整结果与实际审阅](results/mean_precision/attempt1/report.md)

**两组各向异性关系的四组检验已全部完成并实际审阅。** 12 次完整预算训练、统一选模和 144 组新旧预测比较正常结束。仅位移项保留各向异性的 M，区域召回为 **41.489%**，比完整五维高 **0.861 pp**，中心定位平均也改善，但该差的联合区间仍跨零；原生分类 M 为 **78.229%**，低于完整五维的 **80.029%**，也未稳定超过 Distance。仅协方差项保留各向异性的 S 两任务均未保住主要用途。保留完整五维，M 作为区域研究线索；随后完成上述平均精度控制。[固定计划](docs/anisotropy_components_plan.md) · [完整结果与实际审阅](results/anisotropy_components/attempt1/report.md)

**“各自大小球形”归因已完成并审阅，保留完整五维。** 六次拟合和 96 组新旧预测完成；完整五维对局部球形的区域基础/稀疏召回高 +2.804/+3.820 pp，三个种子均正，中心定位也改善。原生基础 mCA 高 +2.558 pp，配对差 +3.560/−0.048/+4.163 pp，联合种子×类别区间仍跨零；局部球形则比 Distance 低 0.955 pp、三个种子均低。局部大小没有保住主要用途，支持进一步研究方向与非球形的联合关系，但本轮不能单独归因到朝向或宣称一般有效。[固定计划](docs/trace_isotropic_plan.md) · [完整结果与实际审阅](results/trace_isotropic/attempt1/report.md)

**现有预测的成功与失败已整理。** 对 Distance，已用九场景纠正/引入错误为 15.321%/12.333%，原生主确认对象组为 6.999%/5.877%，匹配 256 为 6.489%/6.365%。区域收益保留，原生样本交换与种子依赖明显；尚未找到跨任务一致的几何成功条件。该次整理只分析已有预测，随后启动上述球形归因；跨来源测试仍保留为可选草案。[完整报告与案例](results/prediction_success_failure/attempt1/report.md)

目前，观测区域匹配与 R71 旧场景图像任务支持候选的条件性用途。但五个新场景确认中，五维对 Distance 的 PSNR 差为 +0.224/+0.505/−0.471 dB，未通过完整稳定收益标准；对 Direct、URoPE、PRoPE 的正面比较保留。R72 RGB 也为混合结果。固定形状诊断已完成并审阅：大部分图像收益在固定参考协方差后仍保留，原五维在给定几何组仅高 +0.024 dB、种子方向混合，在 RGB 基础条件有 +0.055 dB 的小幅一致收益。变化协方差的专属贡献需要收窄解释。

R73 原生 3DGS 验证已完成并审阅：五维 mCA 为 80.029%，对 Direct / 旧十一维平均高 +3.254 / +2.938 pp，三个种子均正；对 Distance 平均高 +1.603 pp，但配对差为 +3.394/−0.951/+2.367 pp，稳定用途标准未通过。区域匹配的 GeoTransformer/角度描述符补充对照也已完成：五维基础召回分别高 +4.103/+4.161 pp，三个种子均正、中心 PCK 共同改善。三个区域固定球形对照也已完成：变化协方差比同容量径向控制再高 +3.340/+3.363 pp（基础/稀疏），中心指标共同改善。同一对照在原生 3DGS 也已完成：五维比固定球形高 +2.731 pp，三个种子均正；固定球形弱于 Distance，原生性能稳定性仍需验证。历史负结果、确认失败与结论边界同时保留；尚不能宣称一般有效。

**原生五组新种子的固定复现已全部完成并审阅。** 十个模型均完成原定 120 轮，统一选模后完成六条件评价。新五组基础 mCA 为 79.645% 对 Distance 的 77.763%，平均 **+1.882 pp、5/5 正**，配对种子区间 [1.190, 2.714] pp，达到原定开发信号。六条件均值均正，但有个别扰动反例；基础种子×类别区间 [−0.322, 4.143] pp 仍跨零，来源及 OA/CE 也有混合表现。旧三组与合并八组单列，不改写 R73 旧结论。[完整结果与审阅](results/native_seed_replication/attempt1/report.md)

**完整官方 TEST 确认已完成并审阅。** 31 个已选模型、六条件、186 组预测全部完成，零新增拟合。主确认使用 3,845 个无已知历史 ID 重叠对象：新五组的五维 / Distance mCA 为 **79.601% / 78.479%，差 +1.122 pp**，联合种子×对象区间 **[−0.326,2.693] pp**，预定结论仍为证据尚不充分。变化协方差相对同容量固定球形在新对象上再高 **+2.244 pp**、旧三组均正、联合区间 [1.001,3.446]；预定 sparse256 条件对 Distance 为 **+2.005 pp**、五组均正、区间 [0.520,3.500]。形状归因和稀疏证据值得保留，不能替代基础主确认或覆盖 RGB/迁移负结果。另 2,245 个历史复用对象单独报告，整个官方 TEST 今后均按已用数据处理。[完整结果与分析](results/native_confirmation/attempt1/report.md) · [实际审阅](results/native_confirmation/attempt1/review.json)

本次完整缓存约 1.208 GiB，确认控制器已结束，31 个模型退出码均为 0。[共享调度补充](results/native_confirmation/attempt1/gpu_resources1/report.md)本次未启用。

**旧数据的匹配 256 基元对照也已全部完成并审阅：准确率优势缩小到接近零。** 九次训练均完成 120 轮、46,800 步，统一选模及六条件评价正常结束。256→256 的五维 / Distance / 固定球形 mCA 为 **76.743% / 76.620% / 76.573%**，差 +0.123 / +0.170 pp，种子方向混合且区间宽；CE 对两个控制仍为三种子均改善。这削弱“少量基元本身使形状更有优势”的解释，但训练与验证数量共同改变，不能证明所有旧收益只来自数量不匹配。256→1024 对固定球形还低 2.719 pp、三种子均负。后续对称关系检验也已完成，未保留核心用途，结果见下一段。[固定方案](docs/native_matched256_plan.md) · [完整结果与分析](results/native_matched256/attempt1/report.md) · [实际审阅](results/native_matched256/attempt1/review.json)

**对称三维检验已全部完成并审阅，保留五维候选。** 15 次拟合完整结束；三维对五维的区域召回低 2.817 pp、原生 512 mCA 低 2.206 pp，三个种子均退步，区域中心定位也变差。匹配 256 mCA 有 +0.900 pp 局部改善，但区间宽、密集条件仍输固定球形。可学习 Jeffreys 也未形成更强的一般替代。后续优先级已调整为整理现有预测的成功与失败。[固定方案](docs/symmetric_relation_plan.md) · [完整结果与实际审阅](results/symmetric_relation/attempt1/report.md)

**区域新场景确认已完成并审阅：** 24 个既有模型、两条件、48 组预测，无新训练。在九个可用的新 place 上，五维基础 R@1 为 41.389%，对 Distance / GeoTransformer / 固定球形分别 +2.988 / +3.362 / +1.782 pp，三个配对种子均改善，中心定位共同改善。预定十个 place 中一个缺官方内参，保留缺失且不替换；结果只覆盖九个可用场景，不覆盖原生/RGB/投影任务的既有负证据。[完整确认与审阅](results/region_confirmation/attempt1/report.md)

原生固定球形的三个完整拟合、18 组新预测及审阅均完成，全部输入流配对、GPU 选模回放一致。基础 mCA：五维 80.029%、固定球形 77.298%、Distance 78.426%；稀疏类别区间跨零及局部来源组反例保留。[完整结果与审阅](results/native_shape_attribution/attempt1/report.md)

利用额外空闲卡完成了可学习高斯散度的三个区域对照：散度比 Distance 更好，五维在旧五场景及已用九个扩展场景仍平均高约 1.4–1.6 pp；中心误差存在反例，跨种子/场景区间仍有不确定性。后一个比较不重称为全新数据确认。[散度对照与解释](results/region_jeffreys_control/attempt1/report.md) · [已用九场景补充](results/region_jeffreys_control/attempt1/used_scene_extension1/report.md)

空闲 A6000 上的实测显示：保持参数与公式不变、重排关系投影顺序后，原生五维前向＋反向由 188.60 降至 123.81 ms，峰值显存由 10.986 降至 7.888 GiB。对照也同等优化后，五维仍比 Distance 慢约 16.34%；区域小输入没有加速。测量不含数据加载、优化器和常驻完整训练缓存，当前训练未采用该关系投影重排。[并行核对](results/native_shape_attribution/attempt1/parallel_resume_check_20261105.json) · [原实现成本](results/pe_runtime_diagnostic/attempt1/report.md) · [投影重排实测与范围](results/factored_relation_runtime/attempt1/report.md)

九个旧模型在 120 个旧 fit 对象上的只读注意力诊断已完成：五维的位置/内容分数幅度比中位数为 2.78–3.00，低于 Distance 的 3.57–4.73，注意力通常更分散。未发现能把旧弱种子 22 唯一归因于整体位置项过强的证据；局部尾部仍保留，不由此添加温度、归一化或门控，也不把内部统计当成效果确认。[诊断结果与边界](results/native_attention_balance_diagnostic/attempt1/report.md)

| 位置 | 内容 |
| --- | --- |
| [docs/research_current.md](docs/research_current.md) | 当前设计、证据、分析、结论和下一步的唯一主入口 |
| [docs/research_core.md](docs/research_core.md) | 固定五维方法、核心证据链、代价和适用边界的精简说明 |
| [全查询区域可靠性](results/region_selective_matching/attempt1/report.md) | 75 份已有预测的选择性匹配诊断；可靠输出优势未确立，保留候选排序收益 |
| [原生同等数量增强与范围收束](results/native_count_augmentation/attempt1/report.md) | 六次训练及实际审阅完成；数量增强未确立稳定强对照优势，保留正面信号与限制，结束本次原生推广尝试 |
| [docs/archive/2026-11-04/INDEX.md](docs/archive/2026-11-04/INDEX.md) | 历史轮次原文、协议、失败记录和未执行草案 |
| [docs/archive/2026-11-04/manifest.json](docs/archive/2026-11-04/manifest.json) | 归档前后路径与逐文件 SHA-256 |
| [docs/round70_nvs_plan.md](docs/round70_nvs_plan.md) | 已完成实验的冻结计划，保留依赖路径 |
| [docs/round70_geometry_diagnostic_plan.md](docs/round70_geometry_diagnostic_plan.md) | 已完成的解释性诊断计划，保留依赖路径 |
| [docs/round71_target_projection_plan.md](docs/round71_target_projection_plan.md) | 目标查询读取参考区域的投影 PE，已完成并通过开发用途标准 |
| [docs/round72_rgb_projection_plan.md](docs/round72_rgb_projection_plan.md) | RGB 预测几何的投影 PE，已完成并保留混合结果 |
| [docs/projection_confirmation_plan.md](docs/projection_confirmation_plan.md) | 五个新场景确认已完成；完整用途标准未通过，正面子比较保留 |
| [R73 原生五维验证](results/round73_native_relative5/attempt1/report.md) | 已完成并审阅；相对 Direct/旧十一维稳定改善，Distance 下仍有种子不稳定 |
| [固定形状归因诊断](results/projected_shape_attribution/attempt1/report.md) | 已完成并审阅；保持共同内容和参数，只替换 PE 的参考形状输入，多数图像收益仍保留 |
| [几何编码补充对照](results/region_geometry_controls/preparation/plan.json) | 当前固定查询与完整共同内容下，新增 GeoTransformer 位置算子及角度描述符 MLP；六次拟合及评价完成，五维保留约 4 pp 的开发优势 |
| [区域固定球形归因](results/region_shape_attribution/preparation/plan.json) | 已完成三个配对控制与审阅，变化协方差提供额外区域收益 |
| [区域新场景确认](results/region_confirmation/attempt1/report.md) | 九个可用新场景完成并审阅；用途优势与额外协方差作用保留，预定第十个场景缺标定 |
| [可学习散度对照](results/region_jeffreys_control/attempt1/report.md) | 当前预算和完整内容下的三个配对控制已完成；五维在经典散度加 MLP 之上仍有额外区域收益，局部反例保留 |
| [整模型 GPU 成本](results/pe_runtime_diagnostic/attempt1/report.md) | 原实现比旧 full11 省计算，但仍有高于 Distance 的开销；范围和全部测量保留 |
| [关系投影重排](results/factored_relation_runtime/attempt1/report.md) | 参数与公式不变，原生省时省显存；全部兼容对照同等优化，区域没有速度收益 |
| [原生确认的历史对象核对](results/native_confirmation_metadata/attempt1/historical_overlap1/report.md) | 官方 TEST 的 2,245 个底层对象曾用于早期 ModelNet 开发，另 3,845 个无已知历史 ID 重叠且覆盖 60 类；现已按冻结计划开放基元，不能称全量全新 |
| [逐对象缓存验证](results/native_stream_cache_validation/attempt1/report.md) | 六个已用对象完整复现原缓存，两个对象通过镜像网络验证；完整测试缓存估算约 1.208 GiB，可避免常驻保留全部原始文件 |
| [缓存到推理的执行准备](results/native_confirmation_runtime_validation/attempt1/report.md) | 旧六对象、两模型、六条件与原预测一致，旧完整验证回放一致；支持来源复用分组；该执行检查只使用旧数据，正式确认现已另行启动 |
| [原生固定球形归因](results/native_shape_attribution/attempt1/report.md) | 已完成并审阅；原五维额外 +2.731 pp，三种子均正；固定球形未超过 Distance |
| [原生未用对象确认与审阅](results/native_confirmation/attempt1/report.md) | 31 个模型、186 组预测完成；基础主比较 +1.122 pp、联合区间跨零，形状额外收益与稀疏比较为正，未宣称一般有效 |
| [原生未用对象确认计划](results/native_confirmation_protocol/attempt1/report.md) | 事前规则已完整执行，旧三组与合并八组不覆盖新五组主判断 |
| [原生确认执行衔接](results/native_confirmation_workflow_validation/attempt1/report.md) | 原缓存、最多八卡推理与汇总已完成旧数据检查；正式下载、最多六卡并行推理及完整输出核对均已完成 |
| [原生种子完整复现](results/native_seed_replication/attempt1/report.md) | 10/10 拟合、60 组新预测及实际审阅完成；新五组基础 +1.882 pp、5/5 正，扰动和来源反例保留；不改写 R73 旧结论 |
| [匹配 256 基元训练诊断](results/native_matched256/attempt1/report.md) | 九次训练和完整审阅完成；匹配后对 Distance / 固定球形的准确率增益仅 +0.123 / +0.170 pp，CE 改善保留；未证明数量不匹配是唯一原因 |
| [三个对称关系量的开发检验](results/symmetric_relation/attempt1/report.md) | 15 次拟合与实际审阅完成；三维损失区域与原生 512 核心用途，匹配 256 局部改善，不推广 |
| [现有预测的成功与失败](results/prediction_success_failure/attempt1/report.md) | 已有区域/原生预测的逐样本纠正、退步、种子一致性和几何关联；保留全部主比较案例，不作因果或新确认声明 |
| [局部大小与完整椭球的归因](results/trace_isotropic/attempt1/report.md) | 六次配对拟合与完整审阅完成；局部球形未保住主要用途，完整形状关系的区域额外收益较明确，原生仍有种子/类别不确定性 |
| [两组各向异性关系的四组检验](results/anisotropy_components/attempt1/report.md) | 12 次完整训练及实际审阅完成；M 保住区域用途但未保住原生收益，S 未支持一般替代，保留完整五维 |
| [方向无关平均精度控制](results/mean_precision/attempt1/report.md) | 六次完整训练及实际审阅完成；Q 未保住两任务主要收益，保留方向关系与完整五维 |
| [旧模型注意力诊断](results/native_attention_balance_diagnostic/attempt1/report.md) | 九模型、120 旧 fit 对象，未支持全局位置项过强解释弱种子；无新训练或确认 |
| [models/](models/) / [experiments/](experiments/) / [tests/](tests/) | 方法实现、实验程序与实现检查 |
| [results/](results/) | 各轮原始结果、配置身份、报告与审阅记录 |
| [AGENTS.md](AGENTS.md) | 继续研究前应遵循的项目说明 |

2026-11-04 整理：归档 323 份历史 Markdown 及 112 个报告附件，另保留旧 README 和旧状态长文快照。原始数据、模型、实验实现和结果目录未搬移。历史文档里的“最新状态”和旧门禁不覆盖现行主文档；历史重跑所需路径恢复方式见归档索引。
