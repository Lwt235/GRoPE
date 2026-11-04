# GRoPE：高斯参数化与 Transformer 位置编码

研究目标是得到一个具有一般性、结构精炼且有可信实验支持的位置编码方案。

**当前统一入口：[研究主文档](docs/research_current.md)。** 文档包含研究动机、五维相对关系 `relative5` 的公式、完整证据链、关键对照、负结果、已完成的新场景确认及正在执行的验证。

目前，观测区域匹配与 R71 旧场景图像任务支持候选的条件性用途。但五个新场景确认中，五维对 Distance 的 PSNR 差为 +0.224/+0.505/−0.471 dB，未通过完整稳定收益标准；对 Direct、URoPE、PRoPE 的正面比较保留。R72 RGB 也为混合结果。固定形状诊断已完成并审阅：大部分图像收益在固定参考协方差后仍保留，原五维在给定几何组仅高 +0.024 dB、种子方向混合，在 RGB 基础条件有 +0.055 dB 的小幅一致收益。变化协方差的专属贡献需要收窄解释。

R73 原生 3DGS 验证已完成并审阅：五维 mCA 为 80.029%，对 Direct / 旧十一维平均高 +3.254 / +2.938 pp，三个种子均正；对 Distance 平均高 +1.603 pp，但配对差为 +3.394/−0.951/+2.367 pp，稳定用途标准未通过。区域匹配的 GeoTransformer/角度描述符补充对照也已完成：五维基础召回分别高 +4.103/+4.161 pp，三个种子均正、中心 PCK 共同改善。三个区域固定球形对照也已完成：变化协方差比同容量径向控制再高 +3.340/+3.363 pp（基础/稀疏），中心指标共同改善。当前把同一对照用于原生 3DGS，训练队列已启动。历史负结果、确认失败与结论边界同时保留；尚不能宣称一般有效。

原生稳定性复现已固定五组新种子、共十次配对拟合，模型与预算保持不变；旧三组、新五组及合并结果分别报告。准备和检查已完成，尚未启动训练；先完成原生固定球形归因，再确定后续强对照范围。

**区域新场景确认已完成并审阅：** 24 个既有模型、两条件、48 组预测，无新训练。在九个可用的新 place 上，五维基础 R@1 为 41.389%，对 Distance / GeoTransformer / 固定球形分别 +2.988 / +3.362 / +1.782 pp，三个配对种子均改善，中心定位共同改善。预定十个 place 中一个缺官方内参，保留缺失且不替换；结果只覆盖九个可用场景，不覆盖原生/RGB/投影任务的既有负证据。[完整确认与审阅](results/region_confirmation/attempt1/report.md)

原生固定球形训练继续在 GPU1/2/3 三卡并行。空闲 A6000 上的实测显示：保持参数与公式不变、重排关系投影顺序后，原生五维前向＋反向由 188.60 降至 123.81 ms，峰值显存由 10.986 降至 7.888 GiB。对照也同等优化后，五维仍比 Distance 慢约 16.34%；区域小输入没有加速。测量不含数据加载、优化器和常驻完整训练缓存，当前训练实现未改动。[并行核对](results/native_shape_attribution/attempt1/parallel_resume_check_20261105.json) · [原实现成本](results/pe_runtime_diagnostic/attempt1/report.md) · [投影重排实测与范围](results/factored_relation_runtime/attempt1/report.md)

| 位置 | 内容 |
| --- | --- |
| [docs/research_current.md](docs/research_current.md) | 当前设计、证据、分析、结论和下一步的唯一主入口 |
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
| [整模型 GPU 成本](results/pe_runtime_diagnostic/attempt1/report.md) | 原实现比旧 full11 省计算，但仍有高于 Distance 的开销；范围和全部测量保留 |
| [关系投影重排](results/factored_relation_runtime/attempt1/report.md) | 参数与公式不变，原生省时省显存；全部兼容对照同等优化，区域没有速度收益 |
| [原生独立确认的数据准备](results/native_confirmation_metadata/attempt1/report.md) | 官方 TEST 6,090 个对象的来源、交集与体量已核对；基元内容与评分未开放 |
| [逐对象缓存验证](results/native_stream_cache_validation/attempt1/report.md) | 六个已用对象完整复现原缓存，两个对象通过镜像网络验证；完整测试缓存估算约 1.208 GiB，可避免常驻保留全部原始文件 |
| [原生固定球形归因](results/native_shape_attribution/preparation/plan.json) | 同一个径向控制迁入原生基元；三次完整预算拟合，CUDA 检查通过，训练进行中 |
| [原生种子复现](results/native_seed_replication/preparation/plan.json) | 已冻结五组新种子配对比较，仅新增固定随机重复，尚未启动；不改写 R73 旧结论 |
| [models/](models/) / [experiments/](experiments/) / [tests/](tests/) | 方法实现、实验程序与实现检查 |
| [results/](results/) | 各轮原始结果、配置身份、报告与审阅记录 |
| [AGENTS.md](AGENTS.md) | 继续研究前应遵循的项目说明 |

2026-11-04 整理：归档 323 份历史 Markdown 及 112 个报告附件，另保留旧 README 和旧状态长文快照。原始数据、模型、实验实现和结果目录未搬移。历史文档里的“最新状态”和旧门禁不覆盖现行主文档；历史重跑所需路径恢复方式见归档索引。
