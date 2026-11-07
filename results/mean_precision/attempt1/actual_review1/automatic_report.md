# 方向无关平均精度控制（自动汇总，待实际审阅）

六次新拟合按原验证指标选模并统一固定，再导出 24 组旧开发预测，与 144 组既有预测比较。

| 任务 | 方法 | 三种子指标 % | 均值 % |
| --- | --- | --- | ---: |
| region | relative5 | 41.403 / 38.685 / 41.797 | 40.628 |
| region | distance_mlp | 39.139 / 36.583 / 34.287 | 36.670 |
| region | fixed_isotropic5 | 38.689 / 36.747 / 36.428 | 37.288 |
| region | trace_isotropic5 | 39.116 / 37.640 / 36.716 | 37.824 |
| region | metric_anisotropy5 | 42.446 / 39.639 / 42.382 | 41.489 |
| region | shape_anisotropy5 | 39.477 / 36.506 / 36.214 | 37.399 |
| region | mean_precision5 | 38.806 / 36.290 / 37.827 | 37.641 |
| native512 | relative5 | 81.225 / 78.590 / 80.273 | 80.029 |
| native512 | distance_mlp | 77.831 / 79.542 / 77.907 | 78.426 |
| native512 | fixed_isotropic5 | 79.192 / 77.503 / 75.199 | 77.298 |
| native512 | trace_isotropic5 | 77.664 / 78.639 / 76.111 | 77.471 |
| native512 | metric_anisotropy5 | 77.966 / 78.875 / 77.846 | 78.229 |
| native512 | shape_anisotropy5 | 77.669 / 75.751 / 79.531 | 77.650 |
| native512 | mean_precision5 | 76.387 / 78.084 / 77.670 | 77.380 |

| 任务 | 左减右 | 三种子差 pp | 均值 pp | 种子×场景/类别区间 pp |
| --- | --- | --- | ---: | --- |
| region | relative5 − mean_precision5 | +2.597 / +2.395 / +3.970 | +2.988 | [+2.122, +4.181] |
| region | mean_precision5 − trace_isotropic5 | -0.310 / -1.350 / +1.110 | -0.183 | [-1.574, +1.635] |
| region | mean_precision5 − distance_mlp | -0.334 / -0.293 / +3.540 | +0.971 | [-0.819, +3.410] |
| region | mean_precision5 − fixed_isotropic5 | +0.117 / -0.457 / +1.399 | +0.353 | [-0.687, +1.559] |
| region | mean_precision5 − metric_anisotropy5 | -3.640 / -3.349 / -4.555 | -3.848 | [-4.721, -2.736] |
| region | mean_precision5 − shape_anisotropy5 | -0.672 / -0.216 / +1.612 | +0.241 | [-0.839, +1.612] |
| region | relative5 − distance_mlp | +2.264 / +2.102 / +7.509 | +3.958 | [+1.737, +7.436] |
| native512 | relative5 − mean_precision5 | +4.838 / +0.506 / +2.604 | +2.649 | [-0.663, +5.883] |
| native512 | mean_precision5 − trace_isotropic5 | -1.278 / -0.555 / +1.559 | -0.091 | [-3.366, +2.942] |
| native512 | mean_precision5 − distance_mlp | -1.444 / -1.458 / -0.237 | -1.046 | [-3.504, +1.320] |
| native512 | mean_precision5 − fixed_isotropic5 | -2.805 / +0.581 / +2.470 | +0.082 | [-3.655, +3.691] |
| native512 | mean_precision5 − metric_anisotropy5 | -1.579 / -0.791 / -0.176 | -0.849 | [-3.569, +1.697] |
| native512 | mean_precision5 − shape_anisotropy5 | -1.282 / +2.334 / -1.861 | -0.270 | [-3.082, +2.955] |
| native512 | relative5 − distance_mlp | +3.394 / -0.951 / +2.367 | +1.603 | [-1.584, +4.916] |

| 任务 | 左减右 | 纠正率 % | 新增错误率 % | 全种子纠正/退步个数 |
| --- | --- | ---: | ---: | --- |
| region | relative5 − mean_precision5 | 14.980 | 11.992 | 268 / 156 |
| region | mean_precision5 − trace_isotropic5 | 8.948 | 9.131 | 31 / 38 |
| region | mean_precision5 − distance_mlp | 11.963 | 10.992 | 85 / 87 |
| region | mean_precision5 − fixed_isotropic5 | 10.068 | 9.715 | 49 / 54 |
| native512 | relative5 − mean_precision5 | 7.532 | 4.882 | 4 / 2 |
| native512 | mean_precision5 − trace_isotropic5 | 6.404 | 6.495 | 0 / 3 |
| native512 | mean_precision5 − distance_mlp | 5.774 | 6.820 | 1 / 3 |
| native512 | mean_precision5 − fixed_isotropic5 | 7.041 | 6.958 | 4 / 2 |

F=完整五维，Q=方向无关平均精度，T=各自大小球形；M/S 是上一轮的两组单独保留版本。
Q 的非对角关系保留平均大小与平均精度，自关系与 F 一致；所有方法共同内容仍保留完整方向信息。
完整条件、中心定位、OA/CE、场景/类别/来源及固定几何分组见 result.json；
全部逐查询/对象错误交换、种子反转和固定 ID 案例见 error_analysis.json 与 events/。
本轮只提供旧开发数据上的机制与用途证据。未作自动晋升，不改写既有确认不确定性与负结果。
