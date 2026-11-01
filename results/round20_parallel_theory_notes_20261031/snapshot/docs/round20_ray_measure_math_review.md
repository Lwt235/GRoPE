# 未选择射线备选的独立数学复核

2026-10-31。只复核
[root 备选笔记](/home/liwt/GRoPE/docs/round20_unselected_ray_measure_note.md)及下述
一手 projected-normal 论文；没有实现、数值实验或新数据。此备选不进入 R20，
不改任何已冻结源、当前主选或门槛。**主公式正确，但物理线积分需补射线长度
因子；目前不建议据这些恒等式另开效果实验。**

## 测度与闭式

设正定三维 Gaussian 密度为 rho，r=(u,v,1)，X=z r，z>0。坐标变换的
Jacobian 绝对值为 z²，因此 `p_+(u,v,z)=rho(zr)z²` 是未归一化的前半空间
联合密度。root 的平方配方及 F0/F1/递推正确：定义
`A=r^T Lambda r, B=r^T Lambda mu, m=B/A, s²=1/A` 后，
`p_+(u,v)=g F2`，`E[Z|u,v,Z>0]=F3/F2`。这里是 camera-z，不是欧氏 range。
前向归一化常数为 `Pr(Z>0)=Phi(mu_z/sqrt(Sigma_zz))`；单 primitive 的固定
FOV 内再归一化可以消去它，但额外 floor、多 primitive 混合或幅值/opacity
比较时不能默默丢弃其相对权重。

**一处测度修正：** `g F0` 是 `integral rho(zr) dz`。若“line integral”指物理
弧长，令 ell=||r||、t=ell*z、n=r/ell，则

    integral_0^infinity rho(t n) dt = ell * g * F0.

ell 随图像位置变化，不能视为跨 ray 的共同常数。该固定 ray 下的 z 均值仍
为 F1/F0，range 均值则为 ell*F1/F0；投影概率的 range 均值是 ell*F3/F2。
ray maximum 又是另一量，其未加权密度最大处为 z=max(m,0)（m<=0时为边界
上确界）。它不是上述任一积分或条件均值。

传统 projected normal 定义为 `N(mu,Sigma)` 的方向 `X/||X||`，不是一种新
Gaussian 分布；一般三维密度在
[Figueras 等的原论文 §2.3, Eq.(2.3)](https://vmsta.org/journal/VMSTA/article/329/read)
已有明确公式。以下坐标等价为本次独立推导：若球面方向密度为 q(n)，则
`dOmega=du dv/ell³`，故 `gF2=q(r/ell)/ell³`，再按前半球归一化。这也说明
球面密度不能直接当像素密度。普通无畸变相机另需乘
`|det d(u,v)/d(pixel_x,pixel_y)|`，标准无skew内参下为 `1/(fx*fy)`。

一个无需执行的解析核对是 mu=0、Sigma=sigma²I：未归一化前向图像密度为
`1/[4*pi*(1+u²+v²)^(3/2)]`，总质量1/2；条件前向密度分母变成2*pi。
它不是二维 Gaussian。正定 Gaussian 在 Z=0 附近有非零密度，完整无限图像
上的投影一般有重尾，不能默认全局图像均值/协方差存在；有限 FOV 不是一个
可省略的实现细节。窄远场可作 delta-method Gaussian 近似，不能把近似矩
说成无限图像下的精确投影矩。

## native cell、h 与退化支撑

令 `p=gF2`、深度一阶矩密度 `n=gF3`。native cell B 的质量是
`M_B=integral_B p`，其条件深度是 `integral_B n / M_B`，不是中心射线的
F3/F2，也不是对各射线条件均值的等权平均。有限 FOV、native 半像素边界、
极小 M_B 的数值参考与统一失败处理仍须单独定义。

若 h 表示图像上独立 Gaussian 定位噪声，正确联合处理为
`p_h=K_h*p, n_h=K_h*n`，然后对 cell 积分并取比值。
`K_h*(n/p)` 通常不等于 `n_h/p_h`。透视投影前给三维 Sigma 加某个常数矩阵，
也通常不等于投影后的二维 h²I 卷积；两者有不同深度依赖。只对概率加面积
floor 并没有定义该额外质量的深度值：若声称一个统一联合概率模型，还需
合法的 floor 一阶矩/值先验，否则应明确这是分开的 attention fallback 合同。

Sigma 奇异时，三维 Lebesgue 密度、逆矩阵和上述 g 均失效。离相机原点的
rank-2 平面可另外用平面面积测度 pushforward；单 ray 深度退化为
`z=b/(normal^T r)`，需要前向/平行处理。平面穿过相机可使图像支撑降维。
不能直接用 pseudoinverse 套三维公式；任意 jitter 改变模型。一个正确平面
极限主要复现已有 ray-plane 读取，不能自动证明完整 GS 形状有额外价值。

## 仍未建立的价值链

这套法则不提供遮挡、透射率、alpha 反混合、真实 RGB 值或 upstream 预测
修复。即使深度均值精确，`E[f(Z)]` 通常不等于 `f(E[Z])`，所以不能与
log/GELU、learned V 或 RGB/SH 非线性交换；有限格后也没有自动的固定三通道
表达或精确有限维 Q/K map。负尾递推消减、极小 FOV 归一化与二维 cell/blur
积分的总成本均未验证。R8/R9 的限制并未被概率恒等式解除。

保留其为澄清测度的理论记录即可。当前缺乏它比同信息 plane/bilinear 或既有
近似在真实任务中更有用的证据，不推荐在 R20 之外立即追加实施或将它预置为
R20 失败后的赢家。
