# Round15 四次径向闭合：数值与 CPU 成本预检

**结论：固定四次密度的数值实现已过独立合成检查，值得进入另行冻结的 TRAIN 机制实验；尚无真实任务效果或 GPU 成本证据。** 本轮没有读取真实 observation、传感器标签、旧 val 或新 test，不构成旧门槛通过。

## 数学对象与实现

只保留区域均值 μ 与协方差 C，令 S=C+h²I、q=(x−μ)ᵀS⁻¹(x−μ)：

`p(x)=exp[−q²/(4π)]/(π² sqrt(det S))`。

它归一化为1、均值μ、协方差S，形状系数没有学习或搜索。它是已知广义 Gaussian 分布族的固定形状特例与二维矩闭合，不能宣称分布本身有新颖性。C=0 时给出 quartic point blur，并非旧 Gaussian point；h²I 的含义为匹配总协方差，并非 Gaussian convolution。原 Gaussian point 强对照必须保留。

最终实现先把每个 native key 矩形等分成边长≤1（参考8像素单位）的积分小格，以固定 Cartesian GL16×16 正权重求质量，再合回原 key。native8/16/24 的全视野都对应432个积分小格；小格不成为 attention token，也不进入内容分数或独立 softmax。函数返回原 key 的 raw mass，不加 floor。若后续沿用既有 floor，应在原 key 上仅加一次 `1e−30*area`。

实现：`models/round15_quartic_tiled.py`；基础密度与 Cartesian 积分：`models/round15_quartic_pe.py`。两份均已冻结，未修改旧 R10–R14 源。

## 拒绝的角积分与保留的数值失败

按白化顶点、平行方向分段的一维角积分虽在多数值上准确，但中心恰在格边/角时直接 autograd 错误；近边界的有限差分也非均匀收敛。一个固定边界反例中，正确的 x 均值梯度约+0.0516584，角积分自动微分却为−0.123616；把阶数升到128仍未修复。另一个细长协方差案例 GL32 原始质量误差1.0834e−5。没有采用自定义 backward 掩盖这一问题。

完整原型、测试与反例保存在 `results/round15_quartic_angular_failure`。基础 Cartesian 原格积分的 GL24/32 也分别未过预声明 raw/split 1e−8：最大误差2.443e−5、4.995e−7；GL48通过，但计算昂贵。根据误差的格宽依赖，在任何效果读取之前增加确定性的积分细分，复用同一个密度与同一误差门槛。旧失败和原GL48快照均保留。

## 独立数值结果

独立参考为带尾部指数平移的 Cartesian adaptive SciPy integration，未复用模型角积分或固定节点。覆盖120个矩形、240个拆分、12组完整视野及4组均值/协方差/带宽梯度；包含 C=0、旋转细长协方差、大特征值16384、中心在边/角/格外、native8/16/24、tiny FOV与完全 underflow。

| 实现 | raw / split 最大绝对误差 | FOV 归一化最大误差 | 判断 |
|---|---:|---:|---|
| 原格 GL24 | 2.443e−5 | 1.581e−5 | raw / split 失败 |
| 原格 GL32 | 4.995e−7 | 1.305e−8 | raw / split 失败 |
| 原格 GL48 | 3.203e−11 | 6.273e−14 | 通过，保留为参考 |
| ≤1 小格 GL16，合回原格 | 3.1733e−11 | 3.3053e−5 | 通过，固定实现 |

最后一项归一化最大误差来自总质量约6.89e−21的尾部视野，低于预声明1e−4上限。全 underflow raw mass严格为0，证明内部没有偷偷加子格 floor。混合格宽核查保留原 key 数量并满足面积守恒。本地24项测试还覆盖归一化、均值、二/四阶矩、总协方差的仿射变换、可表示尾质量、边角与原 bounds 梯度、batch/chunk及合成四分量总矩恒等式。有限合成域通过不等于对任意病态协方差或极小视野都保证精度。

独立报告：`results/round15_quartic_tiled_independent_order16_v1/verification.json`。
独立最终锁：`results/round15_quartic_tiled_numerical_audit_frozen/lock.json`，SHA256 `d4a7c4d376f2845c74ddeeea99f8e9ba5adb60346ec974a2dd333f6e2f6ec8a8`。

## CPU 合成成本

按 `ROUND15_NUMERICAL_PROTOCOL.md`，同一进程、float64、4线程、1次warmup后3次测量中位数。输入为完整 q8/q16/q24 网格和432个g8 keys，经固定仿射变换；h=.5。计时仅包括位置积分及节点加权求和，未含 pooling、content、softmax或训练，不能替代旧正式 gate 计时。

| query 尺度 | tiled quartic | Gaussian full1 | Gaussian full4 | Gaussian quad16 | 原格 quartic GL48 |
|---|---:|---:|---:|---:|---:|
| 8 px | 0.555 s | 0.198 s | 0.724 s | 2.443 s | 3.807 s |
| 16 px | 0.125 s | 0.0293 s | 0.176 s | 0.687 s | 0.886 s |
| 24 px | 0.0497 s | 0.0122 s | 0.0589 s | 0.371 s | 0.352 s |

新实现比 quad16 快4.40/5.51/7.45倍，比full4快1.30/1.41/1.19倍，比原格GL48快约6.9–7.1倍；仍比full1慢2.80/4.26/4.08倍。CPU结果支持计算可行性，但不是 GPU 性能证据。检查时8张GPU均98–100%忙，未抢占作业或运行GPU计时。

成本源和合成输入锁：`results/round15_quartic_preflight/lock.json`，SHA256 `c3a78daa35266ef10850f46e6e5bf3850f6a2f64fe14d2e18c541ff2f3714fab`。
完整时间：`results/round15_quartic_preflight/cost_cpu.json`。
该锁在独立最终freeze发出前已绑定相同数值report/completion；后发最终审计锁通过仅追加的 `numerical_audit_link.json` 连接，未更改任何源、数值配置、合成输入或时间结果。

## 后续限度

可以考虑一个固定旧TRAIN cohort、同 μ/C/h/T 的 Gaussian/full4/quad16/旧point 强对照实验，先完成全部 forward 再读标签，保留既有区域KL、native argmax/PCK与尺度/扰动门槛；这需要主任务另行启动，当前未执行。

候选的单一中心和平坦内部并不保证PCK：区域均值不是独立传感器中心投影，内容分数仍可移动峰，轻尾还可能增加几何误差下的区域KL。若这些机制在真实 TRAIN 失败，应原样否决或重新讨论基本假设，不能把数值通过称作有效方案。
