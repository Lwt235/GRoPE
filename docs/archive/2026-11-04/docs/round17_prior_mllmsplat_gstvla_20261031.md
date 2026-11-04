# 两项近邻工作的原文核查（2026-10-31）

范围仅限以下两项；未读取 R17 VAL 效果、未做实验。依 `nature-academic-search`，专用 arXiv/CrossRef MCP 不可用时回退官方原文。网页抓取受阻后，只读下载了 CVF 正文/补充和 GST-VLA arXiv v1。

**MLLMSplat。** 全题为 *MLLMSplat: A 2D MLLM-Powered Framework for 3D Gaussian Splatting Understanding, Generation, and Editing*；作者 Jingqiao Xiu、Can Wang、Dong Xu。已核对 [CVF 官方条目](https://openaccess.thecvf.com/content/CVPR2026/html/Xiu_MLLMSplat_A_2D_MLLM-Powered_Framework_for_3D_Gaussian_Splatting_Understanding_CVPR_2026_paper.html)：CVPR 2026，33301–33311。

- [正文](https://openaccess.thecvf.com/content/CVPR2026/papers/Xiu_MLLMSplat_A_2D_MLLM-Powered_Framework_for_3D_Gaussian_Splatting_Understanding_CVPR_2026_paper.pdf) §3.1、Eq.1–3：GS 的 μ/Σ/opacity 参与投影渲染贡献和特征聚合；Σ 还用于 FPS 的 Mahalanobis 距离。特征最终按 Z-order 序列化。这里不能把“用于 tokenizer”写成“协方差进入 RoPE”。
- §3.2、Eq.4–5：GaPE 以相机内外参构造 \(P_i=[K_iR_i,K_it_i;0,1]\)。已直接查看 PDF 第 4 页（页码 33304）的 Eq.5：\(\langle q_i^{\mathrm{GaPE}},k_j^{\mathrm{GaPE}}\rangle=\langle P_i^Tq_i,P_j^{-1}k_j\rangle=q_i^TP_iP_j^{-1}k_j\)。query 的变换是 \(P_i^Tq_i\)，展开内积后为 \(q_i^TP_i\)，最终乘积中的 \(P_i\) **没有转置**。**这里的 R 是相机旋转**；GS 的 μ/Σ/尺度没有进入该式。跨视图使用 GaPE，其余保留 RoPE；解码器结合 GaPE 与二维 RoPE 输出 splatter images。[补充 §D](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Xiu_MLLMSplat_A_2D_CVPR_2026_supplemental.pdf) 解释同视图时两者相容。

**GSTVLA / GST-VLA。** 指定 [OpenReview 记录](https://openreview.net/forum?id=J57nR3hyAd) 的正文、论坛/API 本次分别遭遇 403/浏览器验证，**未完成该版本全文及决定元数据核验**。[作者主页](https://s-elim.github.io/publications/) 将它列为 Md Selim Sarowar、Sungho Kim 的 MusIML@ICML 2026 workshop 论文；[主办方](https://www.musiml.org/events/2026-ICML/index.html) 说明这是默认非归档的 workshop。不能据 PDF 模板页脚认定 ICML 主会录用。

可独立核实的 [arXiv:2603.09079v1](https://arxiv.org/html/2603.09079v1) 日期为 2026-03-10，作者为 Md Selim Sarowar、Omer Tariq、Sungho Kim，**与上述作者列表不同，不能混用版本**。

- v1 §III-A.2–3、Eq.2–4：中心 \(c=p+\mu\)；协方差为轴对齐 \(\Sigma=\mathrm{diag}(\exp(2\sigma))\)。PE 仅为中心的多频 sin/cos。log-scale σ 与 opacity α 和语义特征、PE 拼接后线性投影；它们不是 Fourier 频率或衰减因子，也没有 GS 旋转参数 R。
- v1 Eq.5 为常规缩放点积池化；Eq.6 的 Σ 参与辅助深度渲染。对角 Σ 无法表达任意方向法线，这是数学限制；不照抄原文更强的“orientation”叙述。arXiv 元数据注明结果初步、实验仍在进行；此处不采用其数值作为可靠性能证据。

**与当前设计的边界。** 已读 MLLMSplat 原文及 GST-VLA v1 未见“单个完整 C 的原生区域积分质量 + trace 温度”形式。此结论只覆盖上述版本和公式；不替代 GSTVLA 指定版本的缺失核验，更不证明新颖性。可确认的近邻分别是相机相对变换进入 attention，以及质心 PE 加 Gaussian 参数字段。

下载正文 SHA-256：MLLMSplat `81c921f25d855eef2fcf21fb674ece7062982a943b9130f674c298eac89ab749`；GST-VLA v1 `d3f5b9f647a44564c2bfe47cb62e78e173f948bfcb9b00af5319ea1cd2969f62`。
