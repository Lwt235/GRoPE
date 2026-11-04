# 射线 pushforward PE：未选择的可行性草稿

2026-10-31。仅推导、只读既有源码/记录并复核一手文献；未实现、未运行合成或真实前向、未读标签。本文件不选择 R21 之后的候选，不改变旧失败和门槛。参见 [原测度笔记](round20_unselected_ray_measure_note.md)、[数学复核](round20_ray_measure_math_review.md)。

**可以零新增可调量定义一致的几何算子**：把一个合法三维 Gaussian 的正深度子概率测度投影到图像，再沿用已有 screen-space h 平滑，最后积分 native cell。它不要求额外网络、分量数、jitter 或温度参数。必须明确的建模合同仍包括三维输入来源、正深度、相机、坐标单位和面积 floor；“没有新可调量”不等于无需这些选择，也不等于已有廉价、稳定实现或完整内容耦合方案。

## 1. 三维信息必须真实存在

合法输入可来自实际 GS 的 `mu_w` 与 `Sigma_w=R_s diag(s²) R_sᵀ`，或由 RGB 预测深度、给定相机与源区域像素构造的三维矩。世界到目标相机的刚体变换 `(R,t)` 给出 `mu=R mu_w+t`、`Sigma=R Sigma_w Rᵀ`。若保留因子 `L_w`，可直接用 `L=R L_w`，不需要求逆或改变秩。这里的 Gaussian 是归一化空间概率；GS opacity 不能未经合同就充当该概率质量或可见性。

本仓库已有合法上游来源：`experiments/round10_partition_data.py:dense_observations` 由 RGB 深度和相机产生 `q_points3`、`q_pixel_cov3`；每像素 within covariance 为 `z² A Aᵀ/12`，其中 `A=Kq^-1[:,:2]`，没有加入任意法向厚度。区域可用这些三维像素的全部一、二阶矩，再刚体变换。此处仅核对源码，未打开档案。它仍是三维区域的 Gaussian closure，不会凭空恢复原区域的全部形状。旧 `projected_3d_moments` 随后使用 Jacobian、z floor 和 clipping；那条二维近似不是本草稿的精确 pushforward。

仅给旧二维 `(mu2,C2)` 不足以重建三维 Gaussian。透视丢失深度尺度与相关结构；多个三维分布可给出相同二维近似矩。不能反推一个“默认深度/厚度”再声称同信息比较。任何实际对照必须共享同一个合法三维输入，并把新增来源信息与投影算法收益分开。区域三维质心的投影也不等于 R19 的独立指定中心 a；改用该三维区域 Gaussian 不会自动继承 R20 已得到的中心保护。

## 2. 全秩的 ray density：先说清测度

先用无畸变归一化像平面 `u=(u,v)`，`r=(u,v,1)`。设 `rho=N3(mu,Sigma)`、`Sigma` 正定，`X=z r,z>0`。坐标体积 Jacobian 为 `z²`：

`p_+(u,v)=∫_0^∞ rho(zr) z² dz`，`n_+(u,v)=∫_0^∞ rho(zr) z³ dz`。

令 `Lambda=Sigma^-1`、`A=rᵀLambda r`、`B=rᵀLambda mu`、`D=muᵀLambda mu`、`m=B/A`、`s²=1/A`，以及

`g=(2π)^(-3/2)|Sigma|^(-1/2) exp[-(D−B²/A)/2]`，
`F_n=∫_0^∞ z^n exp[-(z−m)²/(2s²)] dz`。

则 `p_+=g F2`，`n_+=g F3`；`F0=s sqrt(2π) Φ(m/s)`，`F1=mF0+s² exp[-m²/(2s²)]`，`Fn=mF(n−1)+(n−1)s²F(n−2)`（n≥2）。特别地，`F2=(m²+s²)F0+m s² exp[-m²/(2s²)]`。闭式是实数恒等式，负 m 的消减使直接递推可能失效。用 Cholesky `Sigma=LLᵀ` 可将 `D−B²/A` 写成 `||L^-1(mu−m r)||²`，但这并未解决负尾截断矩、underflow 或 tiny-FOV 归一化。

`∫p_+ du dv=P(Z>0)=Φ(mu_z/sqrt(Sigma_zz))`，不是1。物理弧长线积分是 `||r|| gF0`；相机 z 参数的线积分为 `gF0`；ray maximum 又是另一量。它们都不能代替 `gF2`。标准 pinhole 内参从像素到 `(u,v)` 的面积 Jacobian 为 `1/|det K_2x2|`，reference-8 坐标还须一致换单位。`mu=0,Sigma=sigma²I` 时 `p_+=1/[4π(1+u²+v²)^(3/2)]`，总质量1/2；此例直接排除将它当二维 Gaussian 或省略 z² 的解释。

完整无限图像一般有重尾，不能默认投影均值/协方差存在。因此不能把“精确投影 covariance”直接塞进原 trace 温度公式。初步理论合同只定义几何；后续若保留内容，必须另行明确相同的固定内容规则，不从此处自动生成新温度。

## 3. PSD 的统一定义与退化支撑

对 rank r∈{0,1,2,3}，定义 `X=mu+L xi`，`xi~N(0,I_r)`、`LLᵀ=Sigma`。对图像 Borel 集 B：

`nu(B)=E[1{X_z>0} 1{pi_K(X)∈B}]`。

这是 pushforward 子概率测度，与因子的正交换基无关；没有三维密度时也成立。不能将伪逆和伪行列式塞回三维 `gF2` 公式。显式添加 epsilon I 是新三维模型，不是正确处理奇异分布。

| 秩 | 无平滑投影的数学对象 |
| --- | --- |
| 2 | Gaussian 位于仿射平面。若单位法向 n、平面 `nᵀX=d≠0`，一条 ray 的交点深度 `z=d/(nᵀr)>0`；面积密度 rho_plane 的图像密度为 `rho_plane(zr) d²/|nᵀr|³`。平面穿过相机（d=0）时投影可集中于像平面的一条直线，须按测度处理。 |
| 1 | Gaussian 位于三维仿射直线，投影一般是图像直线上的一维测度；若直线穿过相机且有正深度支撑，则前向部分可能全部投影成一个点。 |
| 0 | `mu_z>0` 时为 `delta_pi(mu)`；`mu_z≤0` 时正深度子测度为零。不能把后方点夹到某个正深度冒充该 pushforward。 |

在相机平面上有概率原子时，投影本身未定义；上式通过严格正深度事件排除该原子。相机平面边界处不保证对秩退化连续。例如以相机原点为中心的缩小 Gaussian 不必趋于 rank0 的前向零测度。此边界不能被静默 jitter 掩盖。输入可保留明确的秩/因子；若数值矩阵明显非 PSD，应失败并记录，不能效果驱动地 clip 特征值。

## 4. 原 h、有限 FOV 与更低维积分

在原 reference-8 图像坐标令 `epsilon_h~N2(0,h²I)` 独立于 X，h沿用原固定 .25，不引入三维长度噪声。对矩形 B，定义

`W_B,h(y)=Pr(y+epsilon_h∈B)=Π_d[Φ((b_hi,d−y_d)/h)−Φ((b_lo,d−y_d)/h)]`，
`M_h(B)=E[1{Z>0} W_B,h(pi_K(X))]`。

这一式统一涵盖所有秩，并把平滑与落格联合处理。等价地，全秩时 `M_h(B)=∫_R² p_+(u) W_B,h(Hu+c) du`，而不是对 ray 位置与模糊位移做朴素四维积分。不能先裁掉原始投影的 FOV 外质量再平滑；FOV 外点可能被 h 平滑送入边缘格。

还有更有用的独立等价式。以 `U=(X_x,X_y)`、`v=Sigma_zz>0`、`b=Sigma_Uz/v`、`D=Sigma_UU−Sigma_Uz Sigma_zU/v`，有

`U|Z=z ~ N(mu_U+b(z−mu_z),D)`。

写 pinhole/reference-8 投影为 `y=H U/z+c`，则

`Y_h|Z=z ~ N2(H[mu_U+b(z−mu_z)]/z+c, H D Hᵀ/z²+h²I)`，
`M_h(B)=∫_0^∞ N1(z;mu_z,v) GaussianCellMass(B;mean(z),cov(z)) dz`。

这只需要一维 depth 外积分和二维 Gaussian rectangle primitive；解析消去了独立图像 blur，适用于 PSD，因为条件 D 可奇异而 h>0 使图像 covariance 正定。若 v=0，PSD 保证 cross covariance 为零；Z是常数，mu_z>0 时直接计算一次二维 Gaussian cell mass，mu_z≤0 时质量为零。这是精确的独立条件分支，不是用阈值把小 v 擅自设零。实际计算 D 的消减、近零 z 的巨大尺度及负深度尾仍需独立审计。

设有限 FOV Ω 为所有 native cells 的并。未加 floor 时 `Σ_B M_h(B)=M_h(Ω)≤P(Z>0)`。若沿用旧 attention fallback，则明确使用**未条件归一化**的前向子质量：

`P_B=[M_h(B)+1e−30 area_ref8(B)] / [M_h(Ω)+1e−30 area_ref8(Ω)]`。

先除以 P(Z>0) 再加相同 floor 会得到另一个算子，尤其影响后方极小尾；不得混用。正深度质量为零时，上式给出面积先验并保留该 query，需标记为 fallback，不能称存在真实可见概率。无 floor 的质量对 cell 合并严格可加；固定面积 floor 也可加。加上不同 native key 内容后，不再自动拥有跨分区一致性。

固定 h>0、`mu_z>0` 时，Sigma→0 得到 `N2(pi(mu),h²I)` 的 native-cell 质量，连同同一 floor 精确恢复前向 point 核。若再取 h→0，仅有通常的弱收敛；落在 cell 边界的原子不能保证与某个 half-open hard assignment 逐项连续。近/后方夹深度的旧 point 不是这里的恒等极限。

## 5. 精炼性、实际 GS 与 V 的边界

本构造只使用一个三维 Gaussian 和相机来定义一个图像测度，比“换中心再加分支”更直接对应 GS 参数化；它也没有自动的定长精确 Q/K 特征图或 RoPE 群性质。表示参数少不等于 pairwise native-cell 求值便宜：一般需每个 query/cell 的深度外积分，含近零 z/负尾的难例；反向梯度、tiling/culling 和 rank 边界的总成本均未知。固定 GL32 的现有区间保证不能直接搬给这个新外积分。若只能靠不断增加节点、经验截断/jitter 或分场景修补才能过数值门槛，应停止该实现假设。

即使优化 GS 的指定中心本来就是 mu，且 T=0，非线性 pushforward 仍可能不同于 `N2(pi(mu),J Sigma Jᵀ)`：这里改变的是投影律，不是 anchor 或内容。这使它拥有 R20 修正所没有的潜在作用范围，但没有推出显著收益。实际 GS 可能已用很多小原语补偿了仿射误差；原空间 Gaussian 自身也可能是错误的区域近似。普通 pinhole 的本合同不能直接声称覆盖所有畸变/rolling-shutter 相机；一般测度定义可推广，以上条件 Gaussian 简化未必可用。

同一联合测度确实可以给出 cell 深度 `E[Z 1{Z>0}W_B,h(pi X)]/M_h(B)`；它和 PE 共用积分，不要求新增 V 模块。若只验证 PE，就不必实现此读出。反之，不能把 ratio 的格中心值替代 ratio of integrals，也不能交换 `E[f(Z)]` 与 `f(E[Z])`。面积 floor 没有自带深度/颜色分布；若扩展 V，须单独定义其值先验或明示分开的 fallback。射线概率仍不提供遮挡、透射率、RGB/SH 或上游几何修复。

## 6. 最小无标签 synthetic preflight 合同草案

这是待另行冻结的合同草案，不是实施授权。只测试几何（T=0）、既有 h=.25、一个固定 pinhole/reference-8 相机、固定有限矩形 FOV 与原生/合并/不等面积 cell。无真实档案、标签、温度/带宽/厚度搜索。建议固定下列少量解析类型，并在任何执行前列出精确的 mu、因子 L、相机、cells 和数值预算：

1. 前向 point、后方 point、相机平面 point：核对原 h 点核/零前向质量及明示面积 fallback。
2. 定深 rank2 平面：一次二维 Gaussian 解析落格；倾斜 rank2 平面：surface homography 积分与统一定义互检；另含过相机平面。
3. 一条偏置 rank1 直线及过相机直线：独立一维 latent 积分，核对退化图像线/点与正深度权重。
4. fullrank 零均值等方：核对 `gF2` 的解析密度及前向质量1/2；fullrank 倾斜/大深度方差/均值在后方的固定难例：核对正尾和 finite-FOV 质量。
5. 固定远场窄支撑例与固定 rank2/1/0 极限例：只检验预声明极限，不扫描选择效果更好的模型。

生产提案可用条件 Z 的一维外积分；独立参考须用另一表达式（fullrank 的 gF2+cell-CDF 权重，rank2 的面积测度、rank1 的 latent 测度、rank0 解析式），并另核对全空间正深度质量。所有例必须报告原始 mass、FOV mass、最终 P、zero/tiny/fallback、argmax 差、时间/内存和固定求值次数，不能只报告 floored P 的绝对误差。合并可加性、刚体换坐标等价、共同正尺度变化 `(mu,Sigma)→(a mu,a²Sigma)` 的图像质量不变性、point 极限和屏幕单位变换均必须通过；深度读出若仅作为数学核对，应按正尺度同步缩放。

参考收敛标准和 tail/归一化误差、生产预算须在代码和合成集合一起冻结，沿用已有数值门槛的含义而非看到失败后改 order/tolerance。尤其应要求 raw mass 的可信误差界足以控制对应 FOV 归一化；underflow 到零不能仅凭最终 P 很接近面积 prior 就判正确。任何参考不收敛、尾部无界误差、奇异分支不一致或成本超过预声明预算均停止，保留失败证据；通过只证明这一固定数值域可计算，不证明旧区域/中心/压力/V/GS/未见数据门槛通过。

## 7. 相关工作限制

方向 Gaussian 的 projected-normal 密度早已存在；这里的像平面表达由球面方向密度换测度得到，不能将它宣称为新的 Gaussian 家族或仅凭闭式获得新颖性。[Figueras 等，§2.3/Eq.2.3](https://vmsta.org/journal/VMSTA/article/329/read)。

3DGUT 已用 unscented projection 处理非线性相机，并在3D ray上取最大粒子响应；其 §4.2/Eq.11 不是 z² 加权概率 pushforward。UT 估计的2D conic还用于加速筛选；不能将这些机制混成这里的精确概率积分，也不能把“保留非线性投影/射线几何”本身当创新。附录 C 的固定 bicycle 重建中，静态 pinhole 的 UT/EWA 投影 KL 分布接近，fisheye/rolling-shutter 差异更明显；这限制了把复杂相机收益直接外推到当前静态 pinhole 任务的理由。[3DGUT 原论文](https://arxiv.org/html/2412.12507v1)。

Celarek 等区分 opacity、extinction 和体渲染，§6.4 已利用 ray 上一维 Gaussian 的解析分段积分；其结果显示更准确的渲染对许多小原语的收益有限。该证据限制“更精确必然改善 R9/GS”的推断，不能反过来裁定本 PE 一定无用。[原论文](https://arxiv.org/html/2502.19318v1)。此前笔记中的 Exact-GS 官方页面本次仍返回403，未据未读正文新增技术或效果断言。

**当前结论：数学定义可保持单一且无新增调参量；合理的下一步上限是另行选择并冻结一个无标签数值否证合同。来源信息公平性、固定预算的稳定实现和实际效益仍未建立，现阶段不选择该候选。**
