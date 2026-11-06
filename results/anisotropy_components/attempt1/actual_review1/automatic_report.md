# 位移与协方差各向异性：四组配对检验（自动汇总，待实际审阅）

十二次新拟合按原验证 CE 选模并统一固定，随后导出 48 组旧开发预测，与 96 组既有预测比较。

| 任务 | 方法 | 三种子指标 % | 均值 % |
| --- | --- | --- | ---: |
| region | relative5 | 41.403 / 38.685 / 41.797 | 40.628 |
| region | distance_mlp | 39.139 / 36.583 / 34.287 | 36.670 |
| region | fixed_isotropic5 | 38.689 / 36.747 / 36.428 | 37.288 |
| region | trace_isotropic5 | 39.116 / 37.640 / 36.716 | 37.824 |
| region | metric_anisotropy5 | 42.446 / 39.639 / 42.382 | 41.489 |
| region | shape_anisotropy5 | 39.477 / 36.506 / 36.214 | 37.399 |
| native512 | relative5 | 81.225 / 78.590 / 80.273 | 80.029 |
| native512 | distance_mlp | 77.831 / 79.542 / 77.907 | 78.426 |
| native512 | fixed_isotropic5 | 79.192 / 77.503 / 75.199 | 77.298 |
| native512 | trace_isotropic5 | 77.664 / 78.639 / 76.111 | 77.471 |
| native512 | metric_anisotropy5 | 77.966 / 78.875 / 77.846 | 78.229 |
| native512 | shape_anisotropy5 | 77.669 / 75.751 / 79.531 | 77.650 |

| 任务 | 左减右 | 三种子差 pp | 均值 pp | 种子×场景/类别区间 pp |
| --- | --- | --- | ---: | --- |
| region | relative5 − shape_anisotropy5 | +1.926 / +2.179 / +5.582 | +3.229 | [+1.606, +5.425] |
| region | relative5 − metric_anisotropy5 | -1.043 / -0.954 / -0.585 | -0.861 | [-1.772, +0.138] |
| region | metric_anisotropy5 − trace_isotropic5 | +3.330 / +2.000 / +5.666 | +3.665 | [+1.679, +5.858] |
| region | shape_anisotropy5 − trace_isotropic5 | +0.361 / -1.133 / -0.502 | -0.425 | [-1.764, +0.496] |
| region | metric_anisotropy5 − distance_mlp | +3.306 / +3.056 / +8.095 | +4.819 | [+2.301, +8.066] |
| region | shape_anisotropy5 − distance_mlp | +0.338 / -0.077 / +1.927 | +0.729 | [-0.517, +2.065] |
| region | metric_anisotropy5 − fixed_isotropic5 | +3.757 / +2.893 / +5.954 | +4.201 | [+2.441, +5.965] |
| region | shape_anisotropy5 − fixed_isotropic5 | +0.788 / -0.240 / -0.213 | +0.111 | [-0.797, +1.064] |
| region | metric_anisotropy5 − shape_anisotropy5 | +2.968 / +3.133 / +6.168 | +4.090 | [+2.179, +6.045] |
| region | relative5 − distance_mlp | +2.264 / +2.102 / +7.509 | +3.958 | [+1.737, +7.342] |
| region | relative5 − trace_isotropic5 | +2.287 / +1.046 / +5.080 | +2.804 | [+1.016, +5.223] |
| region | trace_isotropic5 − distance_mlp | -0.023 / +1.056 / +2.429 | +1.154 | [-0.042, +3.001] |
| native512 | relative5 − shape_anisotropy5 | +3.556 / +2.840 / +0.743 | +2.379 | [-0.428, +5.121] |
| native512 | relative5 − metric_anisotropy5 | +3.258 / -0.284 / +2.428 | +1.801 | [-0.949, +4.731] |
| native512 | metric_anisotropy5 − trace_isotropic5 | +0.302 / +0.236 / +1.735 | +0.758 | [-1.744, +3.502] |
| native512 | shape_anisotropy5 − trace_isotropic5 | +0.004 / -2.888 / +3.420 | +0.179 | [-3.843, +3.890] |
| native512 | metric_anisotropy5 − distance_mlp | +0.136 / -0.667 / -0.061 | -0.197 | [-2.594, +2.218] |
| native512 | shape_anisotropy5 − distance_mlp | -0.162 / -3.791 / +1.624 | -0.776 | [-4.249, +2.580] |
| native512 | metric_anisotropy5 − fixed_isotropic5 | -1.226 / +1.372 / +2.647 | +0.931 | [-2.193, +3.919] |
| native512 | shape_anisotropy5 − fixed_isotropic5 | -1.523 / -1.753 / +4.332 | +0.352 | [-3.146, +4.591] |
| native512 | metric_anisotropy5 − shape_anisotropy5 | +0.298 / +3.124 / -1.685 | +0.579 | [-2.715, +4.201] |
| native512 | relative5 − distance_mlp | +3.394 / -0.951 / +2.367 | +1.603 | [-1.589, +4.995] |
| native512 | relative5 − trace_isotropic5 | +3.560 / -0.048 / +4.163 | +2.558 | [-0.635, +6.447] |
| native512 | trace_isotropic5 − distance_mlp | -0.166 / -0.903 / -1.796 | -0.955 | [-3.613, +1.730] |

| 任务 | 左减右 | 纠正率 % | 新增错误率 % | 全种子纠正/退步个数 |
| --- | --- | ---: | ---: | --- |
| region | relative5 − shape_anisotropy5 | 15.214 | 11.985 | 250 / 140 |
| region | relative5 − metric_anisotropy5 | 9.578 | 10.438 | 67 / 64 |
| region | metric_anisotropy5 − trace_isotropic5 | 15.268 | 11.603 | 315 / 172 |
| region | shape_anisotropy5 − trace_isotropic5 | 7.915 | 8.339 | 29 / 47 |
| native512 | relative5 − shape_anisotropy5 | 7.546 | 5.167 | 3 / 2 |
| native512 | relative5 − metric_anisotropy5 | 6.946 | 5.145 | 3 / 3 |
| native512 | metric_anisotropy5 − trace_isotropic5 | 6.459 | 5.702 | 3 / 1 |
| native512 | shape_anisotropy5 − trace_isotropic5 | 6.398 | 6.219 | 4 / 0 |

| 任务/条件 | I=F−M−S+T pp | 三种子 pp | 种子×场景/类别区间 pp |
| --- | ---: | --- | --- |
| region/base | -0.436 | -1.404 / +0.180 / -0.084 | [-1.883, +1.121] |
| region/sparse256 | -0.331 | -1.647 / +0.211 / +0.443 | [-1.757, +1.130] |
| native512/base | +1.622 | +3.254 / +2.604 / -0.992 | [-2.025, +5.141] |
| native512/sparse256 | +1.882 | +4.767 / +1.222 / -0.343 | [-2.262, +6.662] |
| native512/dense1024 | +1.884 | +2.330 / +1.247 / +2.075 | [-1.324, +5.114] |
| native512/jitter02 | +2.680 | +4.192 / +2.590 / +1.258 | [-0.743, +5.983] |
| native512/tilt | +1.377 | +1.948 / +1.774 / +0.407 | [-1.508, +4.357] |
| native512/permuted | +1.622 | +3.254 / +2.604 / -0.992 | [-2.113, +5.195] |

F=完整五维，M=仅位移项保留各向异性，S=仅协方差项保留各向异性，T=各自大小球形。
I 是匹配重训模型在给定指标上的成绩交互，不直接等同真实几何的因果交互，不独自决定推广。
完整条件、四分组、高/低中心歧义组、逐场景/类别/来源及中心定位、OA/CE 见 result.json；
纠正/退步、不同种子反转、全部逐查询/对象事件和固定 ID 案例见 error_analysis.json 与 events/。
本轮是旧数据上的机制归因。两组控制保留平均平方尺度与有序两端，分别改变 PE 位移和协方差关系的各向异性。
共同内容一直保留完整协方差。区间跨零不等于等效，本轮也不能改写已有确认的不确定性或 RGB/跨域负结果。
