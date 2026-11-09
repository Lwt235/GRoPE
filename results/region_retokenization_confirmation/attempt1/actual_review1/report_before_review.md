# 区域重构未用场景确认：自动汇总，待实际审阅

预定十个地点，九个元数据可用；缺失地点不替换。12 个旧模型，零训练。
指标为可评价场景宏平均；新数据现已使用，不能再次当作 fresh。

| 条件 | 方法 | 条件 R@1 % | 全查询正确 % | 有效覆盖 % | PCK % | hard EPE | soft EPE | CE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | relative5 | 39.577 | 14.403 | 35.479 | 14.748 | 0.404 | 0.390 | 3.066 |
| base | distance_mlp | 37.203 | 13.592 | 35.479 | 13.672 | 0.426 | 0.406 | 3.119 |
| base | geotransformer_rpe | 36.690 | 13.385 | 35.479 | 13.492 | 0.431 | 0.408 | 3.159 |
| base | fixed_isotropic5 | 37.702 | 13.795 | 35.479 | 13.822 | 0.424 | 0.403 | 3.151 |
| fixed_sparse256 | relative5 | 37.252 | 13.534 | 35.479 | 13.623 | 0.421 | 0.401 | 3.138 |
| fixed_sparse256 | distance_mlp | 34.159 | 12.480 | 35.479 | 12.175 | 0.455 | 0.424 | 3.220 |
| fixed_sparse256 | geotransformer_rpe | 33.635 | 12.269 | 35.479 | 12.010 | 0.461 | 0.427 | 3.260 |
| fixed_sparse256 | fixed_isotropic5 | 34.610 | 12.651 | 35.479 | 12.268 | 0.453 | 0.421 | 3.269 |
| retokenized256 | relative5 | 42.593 | 8.586 | 19.170 | 16.509 | 0.353 | 0.354 | 3.268 |
| retokenized256 | distance_mlp | 40.675 | 8.186 | 19.170 | 15.479 | 0.368 | 0.365 | 3.297 |
| retokenized256 | geotransformer_rpe | 40.890 | 8.263 | 19.170 | 15.602 | 0.372 | 0.361 | 3.318 |
| retokenized256 | fixed_isotropic5 | 41.666 | 8.385 | 19.170 | 16.037 | 0.365 | 0.358 | 3.327 |

| 条件 | F 减对照 | 指标 | 三种子差 pp | 均值 pp | 联合区间 pp |
| --- | --- | --- | --- | ---: | --- |
| base | distance_mlp | region_recall | 1.672 / 1.332 / 4.120 | 2.375 | [0.831, 4.494] |
| base | distance_mlp | valid_and_correct_fraction | 0.579 / 0.421 / 1.434 | 0.811 | [0.232, 1.578] |
| base | geotransformer_rpe | region_recall | 1.903 / 1.641 / 5.117 | 2.887 | [0.796, 5.351] |
| base | geotransformer_rpe | valid_and_correct_fraction | 0.600 / 0.533 / 1.921 | 1.018 | [0.209, 2.048] |
| base | fixed_isotropic5 | region_recall | 0.731 / 1.408 / 3.488 | 1.876 | [0.272, 3.750] |
| base | fixed_isotropic5 | valid_and_correct_fraction | 0.184 / 0.423 / 1.218 | 0.609 | [0.025, 1.329] |
| fixed_sparse256 | distance_mlp | region_recall | 2.600 / 2.235 / 4.445 | 3.093 | [1.640, 5.054] |
| fixed_sparse256 | distance_mlp | valid_and_correct_fraction | 0.921 / 0.700 / 1.540 | 1.053 | [0.512, 1.796] |
| fixed_sparse256 | geotransformer_rpe | region_recall | 2.691 / 2.427 / 5.734 | 3.617 | [1.584, 5.762] |
| fixed_sparse256 | geotransformer_rpe | valid_and_correct_fraction | 0.894 / 0.807 / 2.092 | 1.265 | [0.503, 2.176] |
| fixed_sparse256 | fixed_isotropic5 | region_recall | 2.169 / 2.362 / 3.395 | 2.642 | [1.280, 4.014] |
| fixed_sparse256 | fixed_isotropic5 | valid_and_correct_fraction | 0.712 / 0.751 / 1.187 | 0.883 | [0.378, 1.433] |
| retokenized256 | distance_mlp | region_recall | 1.789 / 0.911 / 3.055 | 1.919 | [0.019, 3.841] |
| retokenized256 | distance_mlp | valid_and_correct_fraction | 0.372 / 0.208 / 0.620 | 0.400 | [0.055, 0.754] |
| retokenized256 | geotransformer_rpe | region_recall | 1.021 / -0.142 / 4.232 | 1.704 | [-0.722, 4.422] |
| retokenized256 | geotransformer_rpe | valid_and_correct_fraction | 0.195 / -0.065 / 0.840 | 0.323 | [-0.182, 0.835] |
| retokenized256 | fixed_isotropic5 | region_recall | 0.347 / -0.544 / 2.979 | 0.928 | [-1.231, 3.567] |
| retokenized256 | fixed_isotropic5 | valid_and_correct_fraction | 0.064 / -0.090 / 0.627 | 0.200 | [-0.206, 0.697] |

自动数值模式（尚需定位/绝对输出及范围审阅）：{'utility_numeric_pattern': {'distance_mlp': True, 'geotransformer_rpe': False}, 'covariance_numeric_pattern': False, 'complete_planned_cohort_evaluated': False, 'center_and_all_query_results_require_joint_review': True, 'scope': 'Only available members of the predeclared cohort; missing scenes are not replaced.'}
独立总体/逐场景指标最大差：1.33e-15。
完整缺失、覆盖、技术排除、逐场景和种子差保留在 result.json。
EPE 是配对归一化场景单位；更高条件召回不自动代表总体更耐稀疏。
