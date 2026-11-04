# Round10 key partition mechanism diagnostic

这是无标签的算子诊断，不是任务效果验证。固定每个 g16 query Gaussian，只改变 key 的真实矩形分割；所有结果求和回同一 g16 key 网格。内容温度固定为 0。

使用三个旧 calibration 场景各自清单中的第一个相机对（ordinal 0），并额外构造恒定深度、同相机的解析平面。未打开任何 labels.npz、sensor 图像或测试数据。

| Operator | Real mean L1 | Real max L1 | Real mean JS | Max raw split defect |
|---|---:|---:|---:|---:|
| overlap | 0.016625349 | 0.99809167 | 0.00045346645 | 0.08457237 |
| scipy_cell | 8.9515012e-17 | 1.0712491e-15 | -1.8373589e-18 | 3.3306691e-16 |
| actual_cell | 1.1533093e-16 | 2.6376554e-15 | 1.2733225e-18 | 2.220446e-16 |

独立 NumPy 实现 area-scaled Gaussian overlap；SciPy 的二维矩形积分提供独立参考；候选 B 的正值积分算子在 CPU float64 下单独对照参考。上述差异描述将矩形 key 近似为高斯后产生的分割依赖性，不能证明该缺陷是实际任务误差的主因，也不能证明 B 会通过已设定的精度、定位和效率门槛。

这个结果只能支持执行已经预声明的 B 对照实验；B 仍需与同样使用真实矩形积分的 point/quad4/quad16 对照公平比较。

归一化比较只在两个 FOV 原始质量都大于 1e-12 的行上定义；原始质量误差保留所有 query。每行是否可归一化、原始/归一化概率、L1 和 JS 均保存为 NPZ/JSON，避免尾部精度造成虚假的巨大相对误差。

数据与源码 hash、精确 pair 身份在 `results/round10_key_partition_diagnostic/lock.json`；原始分数在 `summary.json` 与四个 NPZ。最大 B-vs-SciPy 绝对误差为 4.441e-16。
