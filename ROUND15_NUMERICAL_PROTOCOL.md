# Round15：四次径向闭合的纯数值与成本预检

本协议只允许合成几何。不得读取真实 observation、传感器标签、旧开发效果或新 test；本轮不做候选调参或科学效果判定。用户研究目标与 R10–R12 的任务门槛保持原样。下一轮真实 TRAIN 效果必须另立协议并经主任务明确启动。

## 固定数学对象

二维总协方差 `S=C+h²I`，`q=(x−μ)ᵀS⁻¹(x−μ)`：
`p(x)=exp(−q²/(4π))/(π² sqrt(det S))`。
形状系数完全由 `Cov(p)=S` 确定，无新学习参数。输入 `C=0` 时是 quartic point blur，与旧 Gaussian point 不同；`C+h²I` 不是 Gaussian convolution。它是二维矩闭合，不能称为三维非高斯 primitive 的精确投影，也不保证 query 重新分区闭合。

角度 GL 原型按白化顶点与平行方向分段。该路线已因格边/格角的梯度错误及边界附近非均匀收敛被否决为训练默认实现。原 source、参数、数值和梯度反例永久保留在 `results/round15_quartic_angular_failure`；不采用自定义 backward 掩盖问题。

实际默认采用正的固定 Cartesian tensor-product GL 积分，并将原 key 按物理宽度等分成每边不大于1的积分子矩形。子矩形质量求和回原 key，不改变 attention tokens、分布或读出；floor 只在原 key 上加一次。native8/16/24 的完整视野均对应432个基础积分格。公开有限域是 `h>=.25`、原矩形每边 `<=3`（参考8像素单位），float64；不以库函数接口声称任意极端条件数都可靠。独立合成审计覆盖 PSD（包括0）、旋转细长支撑、协方差大特征值16384、点在边/角/远处、native8/16/24、tiny FOV 与全 underflow。对真实域的额外验证必须在任何标签之前完成。

## 数值选择与固定边界

原不细分实现比较 GL24/32/48：48通过，24/32在大格未通过原始质量与拆分1e−8；失败记录与48模型保持冻结。误差按格宽分解说明小格无需大格的阶数，因此在效果前新建独立 wrapper，先核验每边≤1的固定 GL16，只有数值不通过才升到24。选择依据仍只为独立精度，不能读任务效果选阶数。每次独立审计有独立输出目录、source snapshot 和失败记录，不覆盖。通过要求保持：120单矩形与240拆分的绝对误差均不超过1e−8；12完整FOV归一化概率最大误差不超过1e−4；4类均值/协方差/带宽梯度对独立 adaptive finite difference 最大误差不超过2e−5。归一化、均值、二/四阶矩、仿射变换、尾部、batch/chunk、边角及 bounds 梯度也由本地数学测试检查。

FOV 数值审计明确使用 `mass + 1e−30 * area` 再归一化，area在参考8单位中等于 native key mass/64；同时记录无 floor 的质量，禁止把 fallback 下的归一化稳定误称尾积分精确。独立 SciPy 参考采用带最小 Mahalanobis 指数平移的 Cartesian adaptive integration，不调用模型积分或角度原函数。

## 无标签成本合同

先完成并绑定通过的独立数值报告，再冻结 model/tests、成本 runner、本协议及 Gaussian baseline 积分源 hashes。CPU/GPU 输出均拒绝覆盖；每次启动和结束重验 frozen hashes。固定输入是144×192画面、q8/q16/q24完整规则网格、g8全部432 keys。固定仿射 `[[1.05,.10],[-.05,.95]]` 和平移 `[.10,-.15]` 作用于 query 几何，h=.5，float64。输入数组随结果保存 SHA，零真实数据。

四个算子角色是 tiled quartic单矩闭合、Gaussian full1、Gaussian full4、Gaussian quad16；另保留原未细分 quartic GL48 的成本参考。full1与两种quartic实现共享矩；full4每个2×2子域保留完整域矩；quad16使用4×4子域均值及原像素协方差。全部角色在同一进程、同一CPU/GPU执行，积分后同权重求和。这里只计 positional integration+节点求和，不计 pooling、content、softmax或模型训练；不得将比例当作旧正式 gate 时间。

CPU固定4线程、1次 warmup、3次测量；GPU固定3次 warmup、15次测量，计时前后同步、报告peak allocated memory。GPU繁忙时延后，不抢占其他任务。结果保留逐次时间、中位数、相对quad16/full4/full1比例、输入/source/hardware身份。默认chunk是quartic16query×32key，Gaussian256query×432key；chunk只是公开执行细节，不是数学参数。本轮不扫chunk以追逐有利速度。

数值或成本预检失败都原样汇报。CPU通过不意味着GPU速度通过；本轮任何结果都不构成区域KL、PCK、噪声鲁棒性或一般性方案的证据。
