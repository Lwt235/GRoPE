# 对称高斯关系：固定开发实验（自动汇总，待实际审阅）

15 次新拟合均已按旧验证集选模并统一固定，之后导出开发预测；复用 138 组已绑定身份的控制预测。

| 主读数 | 方法 | 三种子指标 % | 均值 % |
| --- | --- | --- | ---: |
| region/base | relative5 | 41.403 / 38.685 / 41.797 | 40.628 |
| region/base | distance_mlp | 39.139 / 36.583 / 34.287 | 36.670 |
| region/base | fixed_isotropic5 | 38.689 / 36.747 / 36.428 | 37.288 |
| region/base | jeffreys_mlp | 40.594 / 37.986 / 38.643 | 39.074 |
| region/base | geotransformer_rpe | 38.979 / 36.296 / 34.301 | 36.525 |
| region/base | symmetric3 | 38.457 / 36.314 / 38.662 | 37.811 |
| native512/base | relative5 | 81.225 / 78.590 / 80.273 | 80.029 |
| native512/base | distance_mlp | 77.831 / 79.542 / 77.907 | 78.426 |
| native512/base | fixed_isotropic5 | 79.192 / 77.503 / 75.199 | 77.298 |
| native512/base | symmetric3 | 78.682 / 76.946 / 77.843 | 77.824 |
| native512/base | jeffreys_mlp | 79.934 / 72.949 / 79.171 | 77.351 |
| native256/sparse256 | relative5 | 78.139 / 75.954 / 76.136 | 76.743 |
| native256/sparse256 | distance_mlp | 78.225 / 76.094 / 75.539 | 76.620 |
| native256/sparse256 | fixed_isotropic5 | 77.221 / 77.040 / 75.459 | 76.573 |
| native256/sparse256 | symmetric3 | 78.139 / 76.494 / 78.294 | 77.642 |
| native256/sparse256 | jeffreys_mlp | 77.438 / 75.624 / 78.193 | 77.085 |

| 主读数 | 左减右 | 三种子差 pp | 均值 pp | 种子×场景/类别区间 pp |
| --- | --- | --- | ---: | --- |
| region/base | symmetric3 − relative5 | -2.946 / -2.372 / -3.134 | -2.817 | [-3.728, -2.119] |
| region/base | symmetric3 − distance_mlp | -0.682 / -0.269 / +4.375 | +1.141 | [-0.956, +4.241] |
| region/base | symmetric3 − fixed_isotropic5 | -0.232 / -0.433 / +2.235 | +0.523 | [-0.949, +2.122] |
| region/base | symmetric3 − jeffreys_mlp | -2.137 / -1.672 / +0.020 | -1.263 | [-2.393, +0.315] |
| region/base | symmetric3 − geotransformer_rpe | -0.522 / +0.018 / +4.362 | +1.286 | [-1.019, +4.044] |
| native512/base | symmetric3 − relative5 | -2.542 / -1.644 / -2.431 | -2.206 | [-4.699, +0.116] |
| native512/base | symmetric3 − distance_mlp | +0.852 / -2.595 / -0.064 | -0.603 | [-3.380, +2.119] |
| native512/base | symmetric3 − fixed_isotropic5 | -0.510 / -0.557 / +2.644 | +0.526 | [-2.319, +3.524] |
| native512/base | symmetric3 − jeffreys_mlp | -1.252 / +3.997 / -1.328 | +0.472 | [-3.202, +4.422] |
| native512/base | jeffreys_mlp − relative5 | -1.291 / -5.642 / -1.102 | -2.678 | [-6.266, +0.653] |
| native512/base | jeffreys_mlp − distance_mlp | +2.103 / -6.593 / +1.265 | -1.075 | [-6.477, +3.639] |
| native512/base | jeffreys_mlp − fixed_isotropic5 | +0.742 / -4.554 / +3.972 | +0.053 | [-4.723, +4.667] |
| native256/sparse256 | symmetric3 − relative5 | +0.000 / +0.540 / +2.158 | +0.900 | [-1.671, +3.710] |
| native256/sparse256 | symmetric3 − distance_mlp | -0.086 / +0.399 / +2.755 | +1.023 | [-1.771, +4.152] |
| native256/sparse256 | symmetric3 − fixed_isotropic5 | +0.919 / -0.546 / +2.835 | +1.069 | [-1.875, +4.210] |
| native256/sparse256 | symmetric3 − jeffreys_mlp | +0.701 / +0.869 / +0.101 | +0.557 | [-1.913, +3.004] |
| native256/sparse256 | jeffreys_mlp − relative5 | -0.701 / -0.329 / +2.057 | +0.342 | [-2.448, +3.252] |
| native256/sparse256 | jeffreys_mlp − distance_mlp | -0.787 / -0.470 / +2.654 | +0.466 | [-2.595, +3.462] |
| native256/sparse256 | jeffreys_mlp − fixed_isotropic5 | +0.217 / -1.415 / +2.734 | +0.512 | [-2.753, +3.684] |

所有条件、逐种子/类别/来源/场景/四分组、中心定位及 OA/CE 见 result.json。区域 hard EPE 从所有旧/新保存预测按同一规则补算，未重新训练对照。
三个独立量在配对训练中使用五个重复槽；没有额外模块，不声称本轮获得速度收益。
均值接近不表示等效，区间跨零不表示没有作用。图与完整结果须实际审阅后再作设计决定；既有确认与 RGB/跨域负结果不改写。
