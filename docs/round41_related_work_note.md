# R41 后续设计的相关工作边界

2026-11-01，训练等待期间核对的原始文献。未据此修改正在运行的
R41 方法、参数、选择规则或测试集。下面是局部检索，不是完整新颖性审查。

- [Neural Gaussian Scale-Space Fields](https://arxiv.org/abs/2405.20980)
  用 Fourier 特征调制与 Lipschitz 约束学习连续、各向异性的高斯尺度空间。
  它支持“协方差调制频谱”的既有技术背景；过滤信号的效果不能直接证明
  将同样的幅度调制用于对应注意力也有效。
- [Localized Gaussians as Self-Attention Weights for Point Clouds Correspondence](https://arxiv.org/abs/2409.13291)
  已研究以局部高斯构造注意力，并比较固定与可学习方差。
  因而“高斯局部性＋可学习带宽”本身不能作为我们的新颖性主张。
- [NPNet](https://arxiv.org/html/2602.00542v1)
  的编码按输入全局离散程度调整标量带宽及 Gaussian/cosine 混合，
  分割时另加 Fourier 特征。这个机制与逐个真实 GS 的完整 R/S
  协方差不同，但提醒我们必须控制普通尺度适应带来的收益。

我的推断：后续若改成“以 GS 形状定义局部距离”，必须与固定/可学习
各向同性距离、trace-only 和真实 R/S 消融比较；不能仅凭公式更像
高斯、参数更少，或一项损失下降宣称有效。另一方面，滤波区域、匹配
中心和实际渲染是不同目标，应该用各自的直接证据判断，不能互相代替。

R41 的训练损失是上下文基元中心 RGB，主评估则是完整目标图像。
如果二者给出不同排序，下一项有价值的控制变量实验是固定编码结构和
数据，改用上下文渲染图像损失训练，检查监督目标是否影响结论。
这是待检验的推断，不是预设 R41 失败的解释。

## R42 训练期间补充核对

2026-11-01。未修改已冻结的 R42。以下限制下一步的新颖性主张，不是新的实验结论。

- [3D Gaussian Point Encoders，WACV 2026](https://openaccess.thecvf.com/content/WACV2026/html/James_3D_Gaussian_Point_Encoders_WACV_2026_paper.html)
  及[方法全文 §3.1–3.4](https://arxiv.org/html/2511.04797v1)：
  用可学习均值与 Cholesky 精度构成高斯基函数，再线性混合为逐点特征；
  另研究自然梯度与 PointNet 蒸馏。因而“把一组可学习高斯当空间编码基”
  已有直接先例。其基函数并非直接从图像渲染拟合得到的 GS；这个差别本身
  也不足以成为创新或保证迁移有效。
- [Elliptical Attention，NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c63908a3e946af0e7978c23737229137-Abstract-Conference.html)：
  已用 Mahalanobis 度量定义注意力方向。不能把“欧氏距离换成椭圆距离”
  当作新贡献；物理 R/S 是否提供有效方向仍要由相同容量的实验证明。
- [Fast and Accurate Point Cloud Registration using Trees of Gaussian Mixtures，ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/papers/Benjamin_Eckart_Fast_and_Accurate_ECCV_2018_paper.pdf)：
  已讨论各向异性协方差的点到局部几何关系。薄高斯对法向与切向施加不同
  约束的解释也不是新理论。

还应防止重复本仓库已有工作：`models/gaussian_quadratic_pe.py` 已有二维五维
展开，R9 已检验真实投影 R/S 读取 RGB。若考虑三维局部度量，必须说明改变了
什么科学假设；仅把五维升成九维不是新方法。新设计目前没有冻结或开始训练。

## 可学习频率路线的直接先例

R42 完成前后补充检索，仍未据此修改 R42：

- [Learnable Fourier Features for Multi-dimensional Spatial Positional Encoding，NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/84c2d4860a0fc27bcf854c444fb8b400-Abstract.html)
  已学习 Fourier 投影并配合 MLP 形成多维空间编码。只把频率设为可训练参数
  不是新贡献；无 MLP 的结构化频率探针也不能冒称该整套方法的复现。
- [Rotary Position Embedding for Vision Transformer](https://arxiv.org/abs/2403.13298)
  是多维视觉 RoPE 的直接比较入口；需核对具体混合频率实现后再称忠实基线。
- [Head-wise Adaptive Rotary Positional Encoding for Fine-Grained Image Generation，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Head-wise_Adaptive_Rotary_Positional_Encoding_for_Fine-Grained_Image_Generation_CVPR_2026_paper.html)
  官方摘要描述在 rotary 之前加入按 SVD 参数化的可学习线性变换。
  因而用旋转和缩放调整频率，必须比较普通可学习坐标变换，不能仅凭 R/S
  写法声称独有的表示能力。此处只核对官方摘要，尚未完成公式/代码复现审查。

推导上的注意点：若频率写为 w_m=mu+B*epsilon_m，则
w_m^T*x=mu^T*x+epsilon_m^T*B^T*x。这明确包含共享载频与线性坐标变换，
收益可能来自参数约束或优化，而非新的函数类。有限固定 epsilon 也不能直接
当作完整 Gaussian 频率积分。若声称编码仅依赖协方差，还需避免不同等价
R/S 因子给有限频率样本带来歧义。这里是候选审查，尚无新方法有效性证据。

2026-11-01，R45 冻结前的公式核对补充：
[HARoPE v1 §3.3](https://arxiv.org/html/2510.10489v1) 的 A_h 是 head-dimension
方阵，作用于 Q/K 内容：q'=R_x A_h q，k'=R_y A_h k。它保留原坐标的相对
旋转项，并不是二维位置坐标的仿射变换。上面的摘要级线索不能当作两者等价
或复现关系的证据。R45 的仿射/自由频率控制来自明确代数关系，另有完整
Gaussian/RBF/Fourier/MLP 对照；不冒称已经复现 HARoPE。
