# R70 运行期间的定向文献补查

2026-11-04。只补充方法边界，不改变 R70 的30次拟合、选模、指标或判据。
R70 的旧评估尚未导出，以下判断不依赖其结果。目标仍包括实际用途、适用的
鲁棒/泛化验证和未使用数据确认，不以一项匹配成功缩小整体目标。

采用 nature-academic-search 技能。当前没有专用 arXiv/CrossRef MCP；发现的
Hugging Face paper_search 调用返回 UNAVAILABLE。实际检索使用网页检索，
随后核实 arXiv 正文、作者项目页及 GitHub 源码。按 arXiv ID/题名合并同篇
记录；这是定向方法检查，未完成穷尽新颖性检索。未下载数据或读取保留场景。

| 论文/来源 | 核实到的相关内容 | 对本研究的作用 |
| --- | --- | --- |
| Xie et al., *URoPE: Universal Relative Position Embedding across Geometric Spaces*, 2026, arXiv:2604.18747 | 把沿 key 相机射线的固定深度点投影到 query 图像，再使用2D RoPE；按 query view 整理 attention | 新找到的相关外部对照与交互坐标设计参考 |
| Cordonnier, Loukas, Jaggi, *On the Relationship between Self-Attention and Convolutional Layers*, ICLR2020, arXiv:1911.03584，Appendix E | 非各向同性高斯的二次位置 attention 已有先例 | 本仓库 R6/R47 已讨论，不能再次算新发现或把五维数目当创新 |
| 本仓库 R47、R53、R62/R64 的已完成实验 | 动态高斯、Gaussian affine frame、RGB参数预测已有负结果 | 不能把重新命名的相同算子当新方案；保留强对照失败 |

出处：[URoPE 正文](https://arxiv.org/html/2604.18747v1)、
[作者项目页](https://urope-pe.github.io/)、
[官方实现](https://github.com/Applied-Intuition-Open-Source/URoPE/tree/5d71a04778df7a0a9ed1deca82d9bf5d7de05b09)、
[二次位置编码先例](https://arxiv.org/abs/1911.03584)。R47详见
[旧报告](round47_conditional_gaussian_report.md)。不把论文作者报告的收益写成我们的复现结论。

## 当前方法与这份证据的关系

URoPE 编码的是投影后的位置关系；它不编码每个输入区域的完整协方差。
R69 的五维量表达位移相对于两端区域形状、以及两形状间的关系；它属于
learned relative PE，不是新旋转编码或首次使用高斯注意力。现有同信息收益
支持有限预算下的组织方式，而非证明有形状内容的普通 Transformer 无法表示它。

R70在标准相机PE上仅对参考—参考交互加高斯项，目标—参考仍由相机PE处理。
因此下面两个问题必须分开：参考区域之间的高斯关系是否有实际用途；目标
查询读取参考图像时是否需要区域支持信息。目前实现没有直接检验后一个问题。
这来自模型接口的事实，不是用未来负结果替候选找借口；若本轮失败，原失败
照常保留。不能由此预先认定改变放置位置就会成功。

一个后续可检验的推论是：在参考—目标交互中，以目标相机为共同2D坐标，
使用已知目标patch范围和投影后的参考高斯矩，从而不读取目标RGB或深度。
它需要和同样投影坐标/几何信息的Distance、Direct及URoPE配对，才能区分
投影位置优势和高斯范围优势。该推论尚未冻结、未实现、未训练；先完成R70。
它也不是恢复静态积分核或要求高精度求积先过关。

## 官方 URoPE 接入前需要明确的具体事实

已读取 commit `5d71a04778df7a0a9ed1deca82d9bf5d7de05b09` 的实现与NVS入口，
源码身份记录在 `results/round70_nvs/attempt1/related_work_sources.json`。

- 默认16头、group_size=4对应4深度；若接入本轮4头骨干，不能无意变成只有
  一个深度锚。保留4锚需要明确适配group_size，而非直接沿用默认参数。
- 实现对输入view matrix左乘diag(1,-1,-1,1)，注释将其解释为OpenGL到OpenCV。
  SUN3D接口已是OpenCV。未来接入要用已知相机的投影例子核实约定；不以函数
  能运行代替坐标正确，也不凭一行注释声称作者完整实现错误。
- 实际深度离散采用min+(max-min)*index/depth_num，不含max端点；z使用abs(z)
  加epsilon并裁剪投影patch坐标。范围与本轮米制/3的尺度须清楚说明。
- 当前发布实现同时旋转Q/K/V并逆变换输出；不能仅照摘要写成一个Q/K替换。
- `nvs/dataset.py` 中名为identity_unit_distance的函数当前注释掉了距离缩放。
  不能根据函数名假定实际深度单位。论文附录写8k迭代，而发布脚本为80k，
  也不能把任一数字当成本轮2160步的小模型能复现论文结果的依据。

这些是移植范围与可复现性事项，不增加本轮门禁或补跑。后续是否补URoPE对照、
是否检验交互位置，需要结合本轮完整结果决定；不能跳过现有验证转入另一轮追分。
