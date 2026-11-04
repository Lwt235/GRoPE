# Round15 固定 GL24 数值修复与独立 CPU 成本

本协议仅是数值实现修复。真实 TRAIN 无标签预检发现固定 tiled GL16 在 Harvard c6 第一 pair 的 q8 index0、quartic_point、h=.25 的 T0/5/10 三行归一化概率超出1e−4容差：实际最大约0.0030897，独立支持上的积分误差约0.0030937；GL64→96参考差小于6.7e−14。其余1725行通过、argmax差0都不能豁免失败。原始结果、旧协议、源码和锁全部保留，真实效果标签继续暂停。

固定将每个≤1参考单位积分子矩形的 Cartesian Gauss–Legendre 阶数从16升至24。唯一变化是每维固定数值节点数；原密度、同一μ/C、h/T范围、支持/内容dtype、区域分块、query/key身份、原key floor、归一化、有效群体和科学门槛都不变。不引入效果相关精度、尾部条件分支、重新采样或局部调参。

冻结的 `models/round15_quartic_tiled.py` 已支持显式 `quadrature_order=24`，无需修改任何模型文件。独立审计必须在新目录以同一个冻结 SciPy 参考完整重做120矩形、240拆分、12完整FOV以及梯度检查，所有误差门槛不变。真实 TRAIN 预检也必须在新namespace对完全相同的576 geometry /1728 probability行全部重跑，保持参考GL64/96和原来容差。记录每项通过或失败，不能因只剩少量失败缩小验证域。

新成本入口 `experiments/round15_quartic_preflight24.py` 复用冻结的 `experiments/round15_quartic_preflight.py` 函数；只在创建锁时暂时扩展其 source名单，增加本协议、新入口及新测试。这是进程内的 provenance清单扩展，不改写旧源码，不改变旧输入生成器或计时循环。新入口要求独立审计明确通过order24，随后锁定相同新阶数；任何order16/其他阶数的报告或成本锁都被拒绝。

成本其余合同完整沿用 `ROUND15_NUMERICAL_PROTOCOL.md`：纯合成144×192，q8/q16/q24→g8，h=.5，float64；固定仿射和4个主要角色及旧GL48参考；CPU4线程，1 warmup+3次重复。记录每次原始时间和中位数，不与旧进程时间直接相除作为主要相对结果。所有比较角色在新进程中重测。此计时只包含位置积分和节点求和，不是正式任务gate时间，不证明GPU速度或真实效果。

默认输出 `results/round15_quartic_preflight24`，拒绝覆盖；启动和结束验证全部源、旧源快照、独立审计报告与合成输入的hash。新的成本锁绑定order24独立report/completion；独立最终audit锁如后发，可采用仅追加的明确来源链接，但不修改既有锁或结果。

只有完整合成与真实无标签数值检查通过、成本结果可审查，主任务才可决定是否新冻结候选效果协议。本协议不授权任何真实标签读取、旧val/test评估或候选效果选择。
