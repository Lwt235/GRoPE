核查日期：2026-10-31。仅核查两篇原始文献的方法与版本；未读取 R17 主 VAL 效果，未开展实验或作可行性判断。沿用 nature-academic-search 流程，专用 arXiv/CrossRef MCP 不可用，使用官方网页。

**GPSToken。** [版本历史](https://arxiv.org/abs/2509.01109)：v1 为 2025-09-01，最新可验证 arXiv 为 v2（2025-09-19）；[正式论文集](https://proceedings.neurips.cc/paper_files/paper/2025/hash/10831838f1cdd8fcba90188e88e7a121-Abstract-Conference.html)确认 NeurIPS 2025 主会发表。逐项核对 [v1](https://arxiv.org/html/2509.01109v1#S3) 与 [v2](https://arxiv.org/html/2509.01109v2#S3) §3.1–3.3；下述方法一致，未声称两版全文相同。

Eq.1 的五参数对应完整二维协方差，非仅各向同性尺度：

\[
\Sigma=\begin{bmatrix}\sigma_x^2&\rho\sigma_x\sigma_y\\\rho\sigma_x\sigma_y&\sigma_y^2\end{bmatrix}.
\]

§3.2 将区域初始化的 \(g^{init}=(w/6,h/6,0,x,y)\) 投影为 query embedding，逐块加入 RoIAlign 特征，以自注意力预测残差 \(\Delta g\)（Eq.5）。学习后的非零 \(\rho\) 确实使渲染核旋转；Eq.2–3 是截断、去归一化高斯的点值 splatting。**初始 query 的 \(\rho=0\)**；文中没有给出将学习后的旋转直接变成 attention 距离偏置的公式。§3.3 的布局条件也来自校准后的初始参数。所核方法未给出协方差耦合的内容温度或 key 区域质量积分。[方法原文](https://arxiv.org/html/2509.01109v2#S3)

**NPNet。** [版本页](https://arxiv.org/abs/2602.00542)仅列 v1（2026-01-31），备注为 IV 2026 已接收；本次未核实正式 IEEE 论文集条目。[§III-B](https://arxiv.org/html/2602.00542v1#S3.SS2)的未编号公式为

\[
\sigma_g=\tfrac13\sum_{a=1}^3\mathrm{Std}(X_{:,a}),\quad
\sigma_a=\sigma_0(1+\sigma_g),\quad
\lambda=\operatorname{sigmoid}((\sigma_g-\tau)\kappa).
\]

固定 anchors 上逐坐标计算 RBF 与 cosine，按 \(\lambda\) 混合；分割另拼接固定 Fourier 特征。Eq.1 用 \((H+H_{pos})\odot H_{pos}\) 调制邻域特征，再池化；无学习权重。这里是输入统计控制的**标量带宽**，没有每个 token 的完整 \(\Sigma\)、旋转矩阵或区域积分。§III-E 的 \(\operatorname{softmax}(\gamma s)\) 用于原型匹配，未给出 \(\gamma\) 与几何协方差耦合；上式 \(\tau\) 是混合阈值，不能误认成我们的内容温度。[原始方法](https://arxiv.org/html/2602.00542v1#S3)

**与当前实现的边界。** 本地 [R17 算子](/home/liwt/GRoPE/models/round17_trace_pe.py)将已有完整 \(C\) 的原生 key 区域积分对数，与 \(Th^2/(h^2+\operatorname{tr}C)\) 乘内容分数组合。上述两文没有给出这一具体组合；这只是本次两篇范围内的差别，不能证明首创。不得宣称首次用 Gaussian 几何构造 token/query，或首次让输入尺度自适应调节位置编码；也不能借两文结果证明本方案有效。
