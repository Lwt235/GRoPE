# 平均精度归一化收尾控制（自动汇总，待实际审阅）

六次新拟合统一按原验证指标选模，24 组新预测与 96 组旧预测比较。

| 任务 | 方法 | 三种子指标 % | 均值 % |
| --- | --- | --- | ---: |
| region | relative5 | 41.403 / 38.685 / 41.797 | 40.628 |
| region | distance_mlp | 39.139 / 36.583 / 34.287 | 36.670 |
| region | trace_isotropic5 | 39.116 / 37.640 / 36.716 | 37.824 |
| region | mean_precision5 | 38.806 / 36.290 / 37.827 | 37.641 |
| region | normalized_precision5 | 40.921 / 37.208 / 41.122 | 39.750 |
| native512 | relative5 | 81.225 / 78.590 / 80.273 | 80.029 |
| native512 | distance_mlp | 77.831 / 79.542 / 77.907 | 78.426 |
| native512 | trace_isotropic5 | 77.664 / 78.639 / 76.111 | 77.471 |
| native512 | mean_precision5 | 76.387 / 78.084 / 77.670 | 77.380 |
| native512 | normalized_precision5 | 80.568 / 79.080 / 78.383 | 79.344 |

| 任务 | 左减右 | 三种子差 pp | 均值 pp | 种子×场景/类别区间 pp |
| --- | --- | --- | ---: | --- |
| region | relative5 − normalized_precision5 | +0.482 / +1.477 / +0.674 | +0.878 | [-0.047, +1.776] |
| region | normalized_precision5 − distance_mlp | +1.782 / +0.625 / +6.835 | +3.081 | [+0.380, +6.504] |
| region | normalized_precision5 − trace_isotropic5 | +1.805 / -0.431 / +4.406 | +1.927 | [-0.307, +4.333] |
| region | normalized_precision5 − mean_precision5 | +2.115 / +0.918 / +3.296 | +2.110 | [+0.646, +3.602] |
| region | relative5 − distance_mlp | +2.264 / +2.102 / +7.509 | +3.958 | [+1.724, +7.346] |
| native512 | relative5 − normalized_precision5 | +0.657 / -0.489 / +1.890 | +0.686 | [-1.679, +3.645] |
| native512 | normalized_precision5 − distance_mlp | +2.737 / -0.462 / +0.477 | +0.917 | [-2.004, +3.846] |
| native512 | normalized_precision5 − trace_isotropic5 | +2.904 / +0.441 / +2.273 | +1.872 | [-0.783, +4.855] |
| native512 | normalized_precision5 − mean_precision5 | +4.181 / +0.996 / +0.714 | +1.964 | [-1.225, +5.476] |
| native512 | relative5 − distance_mlp | +3.394 / -0.951 / +2.367 | +1.603 | [-1.663, +4.976] |

F=完整五维，N=平均精度归一化，T=各自大小球形，Q=方向无关平均精度。
N 保留方向与归一化谱权重，自关系沿用 F；共同内容仍含完整协方差。
完整扰动、分组、中心/OA/CE、来源和错误交换结果保存在 result.json。
本轮属于已用开发数据收尾检验，主候选仍为 F，不自动推广或启动新确认。
原生强对照不确定性、RGB 和全查询可靠性负结果保持有效。
