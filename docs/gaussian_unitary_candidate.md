# 固定宽度 Gaussian orthogonal completion

2026-10-30。实现为 `models/gaussian_unitary_pe.py`；这里只说明数学含义与验证边界，
尚无训练效果结论。它属于经典 orthogonal/unitary dilation 的具体应用，不声称新的
算子数学，也不是一个新的 Gaussian density kernel。

## 单个算子

每个四维内容块写成 `q=(u,v)`，两部分各二维。对固定二维频率 ω，令

\[
\theta_G=\omega^T\mu,\quad t_G=\omega^TC\omega,\quad
a_G=e^{-t_G/2},\quad b_G=\sqrt{1-e^{-t_G}}.
\]

\[
U_G\begin{bmatrix}u\\v\end{bmatrix}
=\begin{bmatrix}R(\theta_G)(a_Gu-b_Gv)\\b_Gu+a_Gv\end{bmatrix}
=\operatorname{diag}(R(\theta_G),I)
\begin{bmatrix}a_GI&-b_GI\\b_GI&a_GI\end{bmatrix}
\begin{bmatrix}u\\v\end{bmatrix}.
\]

两项都是正交矩阵，故 `UᵀU=I₄`。全部 48 维内容分成 12 个四维块，编码后仍是
48 维；比旧的 48→96 等距 embedding 少一倍 head 宽度，并让以前作为补空间的新
通道 v 也携带独立学习的内容。无额外 bias、gate、shape MLP 或 pairwise kernel。

共同位置平移 δ 只在输出第一子空间左乘共同 `R(ωᵀδ)`，Q/K 内积精确不变。
相同 G 下任意 q/k 的内积不变。归一化原内容后 candidate score 在 [-1,1]，所以
temperature cap 也给出 logit 上界。以上保证都不意味着正确匹配必然排名第一。

## 相同宽度能不能表达形状

可以。令 `α_G=acos(exp(-t_G/2))∈[0,π/2]`，中间矩阵就是在 u/v 两个子空间之间
混合角 α 的正交旋转。各方向的 t 是 covariance 的线性测量；固定的斜频率看得到
非对角元素，至少三个独立方向可约束二维 covariance 的三个自由度。

若两个 G 的中心相同，则每块相对算子为 mixing-angle 差 `αⱼ−αᵢ` 的旋转。
对相同内容 q，几何相似性是 `cos(αⱼ−αᵢ)||q||²`，因此 shape 改变可确实影响排序，
且不依赖降低内容范数。不同中心时出现

\[
U_i^TU_j=W(\alpha_i)^T\operatorname{diag}(R(\theta_j-\theta_i),I)
W(\alpha_j),
\]

它编码相对相位与 shape mixing 的共同作用。这不是完整 BC 或包含率相似性。
有限固定频率也不具备精确共同 SO(2) 旋转不变性；covariance gauge 安全不能代替它。

## 从 point 到 wide support 的边界

- `t→0`：`a→1,b→0`，得到 `[Rθ u;v]`。只有一半内容通道带中心相位，另一半为
  identity。公平对照是本模块的 `unitary_point`；只与全部通道 RoPE 比较会混入通道
  分配差异。
- 小正 t：`a≈1−t/2,b≈sqrt(t)`，shape 的一阶 mixing 很强，但 covariance 对零点的
  导数有平方根奇点。严格正 SPD 梯度已通过测试；不声称 Σ=0 的 covariance 梯度稳定。
- `t→∞`：得到 `[-Rθ v;u]`。位置作用转移到 v，并不会从全体内容中消失。这与旧
  expanded isometry 的“大 support 退回纯内容”不同。
- 所有足够大的 directional variance 都让 α 数值饱和到 π/2，无法继续区分形状。
  小 covariance 在低频下同样只产生很小 mixing，因此实际可识别范围受固定频率和
  support 单位影响。

因为它保留全部通道信息、wide support 仍有相位，不能解释成 Expected PE 的抗混叠
滤波，也不能解释成定位不确定性积分。将 `[aRθu;bu]` 补全成方阵改变了其他内容
子空间的行为，是一个新待测 inductive bias，不能继承旧 embedding 的实验效果。

## 实现约定与 controls

RGB encoder、temperature、dustbin 与原 `GaussianPEMatcher` 相同，共 15,171 个参数；
所有六种 unitary 方法共享完全相同的 12 个 golden-angle 斜方向，频率 norm 为 .1–4。
这里没有新学习参数。

输入使用同样的 `pixq`、`k_to_q`、`projected_qcov`、`projected_kcov`。除 point 外，
query covariance 始终是已知 query rectangle；所有 support 方法在 covariance control
之后统一加 `resolution² I`，默认 resolution=.25 patch units。它是明确的物理 blur，
不是纯数值 epsilon。point 强制 t=0，不添加这项 blur。

| 方法 | key covariance（加共同 blur 之前） |
|---|---|
| `unitary_point` | 0；query 也为 0 |
| `unitary_fixed` | I/12 |
| `unitary_shape` | 观测 covariance |
| `unitary_trace` | 同 trace 球形 covariance |
| `unitary_area` | 同 determinant 球形 covariance |
| `unitary_shuffled` | 固定 roll(13)，不读取标签或匹配身份 |

area/trace 名称指 blur 前的 support。area control 加 blur 后，其有效 determinant
不保证仍与 full 一致；报告不能悄悄把“blur 前同面积”称成“最终有效核同面积”。
roll(13) 对当前 48-token grids 是 derangement，但不保证对任意 token 数都无 fixed
point；它也会包含无效几何占位条目，需要和数据 validity protocol 一起解释。

模型不消费任何 labels、track ID、visibility 或 validity mask；候选 mask 与 query 行
筛选由现有 regional loss/evaluator 处理。这与原 projected matcher 的 contract 一致。
区域实验仍然不能给出无对应/dustbin 结论。

## 验证

新增 `tests/test_gaussian_unitary_pe.py` 的 8 项测试涵盖 4×4 完整正交性、48 维宽度
与范数、same-G 内积、所有 methods 的共同平移、covariance sign gauge、球形 SPD
梯度、zero-variance point limit、wide-support 子空间交换、controls、频率/容量公平、
不读 labels，以及 off-diagonal covariance 可观测性。与既有 Gaussian/world/region
相关 tests 合计 36 项 CPU 全通过。没有运行训练；这些 tests 只证明实现满足约定。
