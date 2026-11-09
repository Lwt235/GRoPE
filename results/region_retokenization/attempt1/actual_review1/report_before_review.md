# 稀疏区域重构：自动汇总，待实际审阅

同一旧五场景、480 帧对，12 个冻结模型，无新增拟合。全部查询比例并不表示已训练拒识。

| 条件 | 方法 | 条件 R@1 % | 全查询正确 % | 有效对应 % | 中心 PCK % | hard EPE | soft EPE | CE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | relative5 | 40.628 | 13.669 | 33.239 | 14.221 | 0.41351 | 0.40159 | 2.97482 |
| base | distance_mlp | 36.670 | 12.386 | 33.239 | 12.528 | 0.44443 | 0.42757 | 3.08114 |
| base | geotransformer_rpe | 36.525 | 12.335 | 33.239 | 12.660 | 0.45118 | 0.42666 | 3.08667 |
| base | fixed_isotropic5 | 37.288 | 12.569 | 33.239 | 12.808 | 0.44017 | 0.42008 | 3.07406 |
| fixed_sparse256 | relative5 | 37.576 | 12.595 | 33.239 | 12.828 | 0.43323 | 0.41342 | 3.03523 |
| fixed_sparse256 | distance_mlp | 32.873 | 11.076 | 33.239 | 10.846 | 0.47878 | 0.44666 | 3.17707 |
| fixed_sparse256 | geotransformer_rpe | 33.055 | 11.127 | 33.239 | 11.083 | 0.48342 | 0.44718 | 3.19013 |
| fixed_sparse256 | fixed_isotropic5 | 34.214 | 11.523 | 33.239 | 11.300 | 0.46923 | 0.43833 | 3.17543 |
| retokenized256 | relative5 | 43.381 | 6.488 | 14.341 | 15.422 | 0.35221 | 0.35489 | 3.15429 |
| retokenized256 | distance_mlp | 40.918 | 6.121 | 14.341 | 14.679 | 0.37281 | 0.37467 | 3.20963 |
| retokenized256 | geotransformer_rpe | 41.274 | 6.182 | 14.341 | 14.126 | 0.37688 | 0.36526 | 3.20498 |
| retokenized256 | fixed_isotropic5 | 41.966 | 6.302 | 14.341 | 14.824 | 0.37407 | 0.36808 | 3.21280 |

| 条件 | F 减对照 | 指标 | 三种子差 pp | 均值 pp | 联合区间 pp |
| --- | --- | --- | --- | ---: | --- |
| base | distance_mlp | region_recall | +2.264 / +2.102 / +7.509 | +3.958 | [+1.724,+7.315] |
| base | distance_mlp | valid_and_correct_fraction | +0.703 / +0.721 / +2.425 | +1.283 | [+0.565,+2.347] |
| base | geotransformer_rpe | region_recall | +2.424 / +2.389 / +7.496 | +4.103 | [+1.657,+7.088] |
| base | geotransformer_rpe | valid_and_correct_fraction | +0.661 / +0.794 / +2.549 | +1.335 | [+0.521,+2.495] |
| base | fixed_isotropic5 | region_recall | +2.714 / +1.939 / +5.369 | +3.340 | [+1.865,+5.380] |
| base | fixed_isotropic5 | valid_and_correct_fraction | +0.881 / +0.672 / +1.748 | +1.100 | [+0.592,+1.764] |
| fixed_sparse256 | distance_mlp | region_recall | +2.802 / +3.407 / +7.900 | +4.703 | [+2.399,+7.722] |
| fixed_sparse256 | distance_mlp | valid_and_correct_fraction | +0.851 / +1.104 / +2.603 | +1.519 | [+0.781,+2.542] |
| fixed_sparse256 | geotransformer_rpe | region_recall | +2.764 / +3.073 / +7.726 | +4.521 | [+1.839,+7.249] |
| fixed_sparse256 | geotransformer_rpe | valid_and_correct_fraction | +0.754 / +0.960 / +2.690 | +1.468 | [+0.586,+2.684] |
| fixed_sparse256 | fixed_isotropic5 | region_recall | +3.078 / +1.939 / +5.071 | +3.363 | [+1.557,+5.527] |
| fixed_sparse256 | fixed_isotropic5 | valid_and_correct_fraction | +0.988 / +0.573 / +1.654 | +1.072 | [+0.491,+1.751] |
| retokenized256 | distance_mlp | region_recall | +0.462 / +3.161 / +3.767 | +2.463 | [+0.565,+4.405] |
| retokenized256 | distance_mlp | valid_and_correct_fraction | +0.072 / +0.415 / +0.612 | +0.366 | [+0.083,+0.725] |
| retokenized256 | geotransformer_rpe | region_recall | +1.814 / +1.047 / +3.460 | +2.107 | [+0.602,+3.696] |
| retokenized256 | geotransformer_rpe | valid_and_correct_fraction | +0.275 / +0.101 / +0.540 | +0.305 | [+0.061,+0.605] |
| retokenized256 | fixed_isotropic5 | region_recall | +0.230 / +2.305 / +1.711 | +1.415 | [-0.348,+3.420] |
| retokenized256 | fixed_isotropic5 | valid_and_correct_fraction | -0.010 / +0.277 / +0.291 | +0.186 | [-0.063,+0.494] |

EPE 是配对归一化场景单位。三条件的查询身份与标签不相同；同条件方法间才共享评价人群。
重构条件沿用抽样前视图均值及尺度，不是完全稀疏传感器流程。原生、RGB 和选择性输出负结果保持有效。
