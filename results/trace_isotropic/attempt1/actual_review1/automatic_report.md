# 各自大小球形：局部尺度与椭球关系（自动汇总，待实际审阅）

六次新拟合按原验证 CE 选模并统一固定，随后导出 24 组旧开发预测，与 72 组既有预测比较。

| 任务 | 方法 | 三种子指标 % | 均值 % |
| --- | --- | --- | ---: |
| region | relative5 | 41.403 / 38.685 / 41.797 | 40.628 |
| region | distance_mlp | 39.139 / 36.583 / 34.287 | 36.670 |
| region | fixed_isotropic5 | 38.689 / 36.747 / 36.428 | 37.288 |
| region | trace_isotropic5 | 39.116 / 37.640 / 36.716 | 37.824 |
| native512 | relative5 | 81.225 / 78.590 / 80.273 | 80.029 |
| native512 | distance_mlp | 77.831 / 79.542 / 77.907 | 78.426 |
| native512 | fixed_isotropic5 | 79.192 / 77.503 / 75.199 | 77.298 |
| native512 | trace_isotropic5 | 77.664 / 78.639 / 76.111 | 77.471 |

| 任务 | 左减右 | 三种子差 pp | 均值 pp | 种子×场景/类别区间 pp |
| --- | --- | --- | ---: | --- |
| region | relative5 − trace_isotropic5 | +2.287 / +1.046 / +5.080 | +2.804 | [+1.011, +5.223] |
| region | trace_isotropic5 − fixed_isotropic5 | +0.427 / +0.893 / +0.289 | +0.536 | [-0.400, +1.313] |
| region | trace_isotropic5 − distance_mlp | -0.023 / +1.056 / +2.429 | +1.154 | [-0.012, +3.001] |
| region | relative5 − distance_mlp | +2.264 / +2.102 / +7.509 | +3.958 | [+1.735, +7.241] |
| region | relative5 − fixed_isotropic5 | +2.714 / +1.939 / +5.369 | +3.340 | [+1.870, +5.442] |
| native512 | relative5 − trace_isotropic5 | +3.560 / -0.048 / +4.163 | +2.558 | [-0.666, +6.456] |
| native512 | trace_isotropic5 − fixed_isotropic5 | -1.528 / +1.136 / +0.912 | +0.173 | [-2.588, +3.041] |
| native512 | trace_isotropic5 − distance_mlp | -0.166 / -0.903 / -1.796 | -0.955 | [-3.540, +1.753] |
| native512 | relative5 − distance_mlp | +3.394 / -0.951 / +2.367 | +1.603 | [-1.604, +5.054] |
| native512 | relative5 − fixed_isotropic5 | +2.033 / +1.087 / +5.074 | +2.731 | [-0.236, +6.097] |

| 任务 | 左减右 | 纠正率 % | 新增错误率 % | 全种子纠正/退步个数 |
| --- | --- | ---: | ---: | --- |
| region | relative5 − trace_isotropic5 | 15.040 | 12.236 | 258 / 187 |
| region | trace_isotropic5 − fixed_isotropic5 | 9.957 | 9.421 | 64 / 62 |
| region | trace_isotropic5 − distance_mlp | 11.955 | 10.800 | 100 / 63 |
| native512 | relative5 − trace_isotropic5 | 7.809 | 5.251 | 6 / 0 |
| native512 | trace_isotropic5 − fixed_isotropic5 | 6.377 | 6.204 | 1 / 2 |
| native512 | trace_isotropic5 − distance_mlp | 5.574 | 6.529 | 4 / 3 |

完整条件、四分组、高/低中心歧义组、逐场景/类别/来源及中心定位、OA/CE 见 result.json；
纠正/退步、不同种子反转、全部逐查询/对象事件和固定 ID 案例见 error_analysis.json 与 events/。
本轮是旧数据上的机制归因。各自大小球形保留平均平方尺度；去掉方向与非球形的联合信息，不代表只去掉朝向。
共同内容一直保留完整协方差。区间跨零不等于等效，本轮也不能改写已有确认的不确定性或 RGB/跨域负结果。
