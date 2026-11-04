# GRoPE：高斯参数化与 Transformer 位置编码

研究目标是得到一个具有一般性、结构精炼且有可信实验支持的位置编码方案。

**当前统一入口：[研究主文档](docs/research_current.md)。** 文档包含研究动机、五维相对关系 `relative5` 的公式、从早期机制到 R72 的完整证据链、关键对照、负结果、结论边界及新场景确认计划。

目前，观测区域匹配与 R71 旧场景图像任务支持候选的条件性用途。但五个新场景确认中，五维对 Distance 的 PSNR 差为 +0.224/+0.505/−0.471 dB，未通过完整稳定收益标准；对 Direct、URoPE、PRoPE 的正面比较保留。R72 RGB 也为混合结果。R73 已开始当前五维在真实优化 3DGS 上的配对训练，另有旧数据固定形状诊断计划用于区分形状与中心度量的贡献。历史负结果、确认失败与结论边界同时保留；尚不能宣称一般有效。

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
| [docs/round73_native_relative5_plan.md](docs/round73_native_relative5_plan.md) | 当前五维方案直接用于原生优化 3DGS，复用 R65 数据、骨干和训练预算 |
| [models/](models/) / [experiments/](experiments/) / [tests/](tests/) | 方法实现、实验程序与实现检查 |
| [results/](results/) | 各轮原始结果、配置身份、报告与审阅记录 |
| [AGENTS.md](AGENTS.md) | 继续研究前应遵循的项目说明 |

2026-11-04 整理：归档 323 份历史 Markdown 及 112 个报告附件，另保留旧 README 和旧状态长文快照。原始数据、模型、实验实现和结果目录未搬移。历史文档里的“最新状态”和旧门禁不覆盖现行主文档；历史重跑所需路径恢复方式见归档索引。
