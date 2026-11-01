# Gaussian 参数进入位置编码：近期原文核查

2026-10-31。主 R17 旧 VAL 独立审计尚在执行，本次未读取其效果。使用
`nature-academic-search` 的原文与版本核验流程；专用 arXiv/CrossRef MCP
不可用，回退官方网页和可公开读取的论文。此为聚焦核查，不是穷尽性检索，
不改变已冻结候选、门槛或数据用途。

**USR-Drive。** 已读 [arXiv v1](https://arxiv.org/html/2608.19036v1#S3.SS3)，
2026-08-19。§3.3 Eq.1–4 将空间锚点和时间的 Fourier 特征送入 MLP。
GS 锚点是冻结几何先验中心的 opacity 加权质心，转换到首帧 ego 坐标；
box 使用固定 BEV 锚点。两者在去噪中保持固定，不取自干净目标。
该 UPE 公式没有输入完整协方差、Gaussian 旋转或尺度；不能把整个系统
输出 GS 解读为其位置编码使用了完整 GS 参数。本次未核实正式发表状态。

**PE-Field。** 已读 [arXiv v1](https://arxiv.org/html/2510.20385v1#S3)，
2025-10-23。§3.2–3.3 为不同 attention heads 分配不同 sub-patch 层级，
并在互不重叠的向量段编码投影 x/y 与相机光轴深度 z。方法包含多位置层级，
没有单 Gaussian 矩闭合公式。§3.4 丢弃投影出网格的 source tokens，空位由
noise tokens 补全。指定 OpenReview PDF 遇浏览器验证，故此处只据可核对的
arXiv 版本，不以缓存摘要断言录用或版本一致性。

六项工作的公式角色对比如下；判断仅覆盖所读版本，不将方法名等同于输入字段。

| 工作 | 位置机制实际使用的几何 | 本次核查的区别 |
| --- | --- | --- |
| [GPSToken v2](https://arxiv.org/html/2509.01109v2#S3) | 区域初始化 Gaussian 参数投影为 query；初始相关系数为零 | 学到的完整二维协方差用于 splatting，不是文中给出的旋转 attention bias |
| [NPNet v1](https://arxiv.org/html/2602.00542v1#S3) | 输入标准差汇总为标量，调节 RBF 带宽与 cosine 混合 | 没有每 token 完整协方差；原型温度未与协方差耦合 |
| [MLLMSplat 官方正文](https://openaccess.thecvf.com/content/CVPR2026/papers/Xiu_MLLMSplat_A_2D_MLLM-Powered_Framework_for_3D_Gaussian_Splatting_Understanding_CVPR_2026_paper.pdf) | 相机 K/R/t 构成跨视图 attention 变换 | 这里的 R 是相机旋转；GS 协方差用于 tokenizer 等其他环节 |
| [GST-VLA arXiv v1](https://arxiv.org/html/2603.09079v1#S3) | 质心的 sin/cos PE | log-scale/opacity 为普通拼接字段；Gaussian 为对角协方差 |
| [USR-Drive v1](https://arxiv.org/html/2608.19036v1#S3.SS3) | 先验中心的加权锚点与时间 | 所述 UPE 未使用完整协方差 |
| [PE-Field v1](https://arxiv.org/html/2510.20385v1#S3) | 分层 sub-patch 投影坐标与深度 RoPE | 使用多位置层级，而非单 Gaussian 支撑积分 |

前四项的版本、公式和访问限制分别见
[GPSToken/NPNet 笔记](round17_prior_gpstoken_npnet_20261031.md) 与
[MLLMSplat/GST-VLA 笔记](round17_prior_mllmsplat_gstvla_20261031.md)。
GST-VLA 的指定 OpenReview 版本未完成正文核验，不能与 arXiv v1 混用。
既有 Gaussian PE、概率位置先验、温度调整及期望 RoPE 的先例仍保留在
[原 R17 文献边界](round17_literature_scope.md) 和
[R6 文献记录](round6_literature_design.md)，本次六篇差异不证明新颖性。

**对后续设计的推断。** 应检验完整支撑形状直接决定位置关系的实际价值，
而不以“Gaussian token”或新增名称代替证据。当前原生区域积分使用完整 C；
trace 只控制内容强度。这两项作用须分别解释：固定尺寸温度归因只能检查
观测 trace 的额外作用；matched isotropic 检查同 trace 下的各向异性与方向
结构，同时消除了特征值差异，不能把结果单独归因为旋转效应。
同配置消融也不能代替充分校准的强对照。已有
[尺寸温度结果](round17_size_temperature_report.md) 尚未支持观测 trace 对区域
KL 的一致优势，故不能预先把这项温度写成主要贡献。

PE-Field 对 sub-patch 表达的处理提示需要留意压缩丢失的空间结构；这是研究
线索，不能证明应增加层级、增加模块或放宽中心定位保护。其 token 丢弃规则
也不能直接移入当前全查询人口验证。R17 主审计接受后先按原门槛判定；若失败，
根据具体失败与已审计归因选择下一项单一假设并重新冻结。若通过，仍须实际 V、
真实优化 GS 和未触碰确认；本次文献核查不计作任何一项性能验证。
