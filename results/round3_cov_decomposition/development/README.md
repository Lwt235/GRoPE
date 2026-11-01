# GS covariance decomposition: development

All methods use identical regional centers, opacity weights, primitive-center visibility, key/row masks and numerical covariance floor. No kernel parameter is fitted. The targets are sparse SfM-track transfer, not dense overlap.

| Scene | split | covariance | rows | CE | KL | top1 target mass | max-bin accuracy |
|---|---|---|---:|---:|---:|---:|---:|
| all | all | full | 2121 | 1.1706 | 0.5855 | 0.58575 | 0.69967 |
| all | all | centeronly | 2121 | 1.1588 | 0.57364 | 0.59009 | 0.70486 |
| all | all | internalonly | 2121 | 226.71 | 226.13 | 0.50115 | 0.58369 |
| all | all | flooronly | 2121 | 9.9788e+05 | 9.9788e+05 | 0.55752 | 0.66054 |
| all | train | full | 1242 | 1.1422 | 0.53168 | 0.5924 | 0.7182 |
| all | train | centeronly | 1242 | 1.1349 | 0.52438 | 0.59504 | 0.72222 |
| all | train | internalonly | 1242 | 221.85 | 221.23 | 0.49445 | 0.58132 |
| all | train | flooronly | 1242 | 1.1493e+06 | 1.1493e+06 | 0.55578 | 0.66184 |
| all | validation | full | 879 | 1.2108 | 0.66155 | 0.57635 | 0.67349 |
| all | validation | centeronly | 879 | 1.1925 | 0.64324 | 0.58309 | 0.68032 |
| all | validation | internalonly | 879 | 233.59 | 233.04 | 0.5106 | 0.58703 |
| all | validation | flooronly | 879 | 7.8391e+05 | 7.8391e+05 | 0.55997 | 0.6587 |
| teatime | all | full | 1105 | 1.2905 | 0.73548 | 0.56325 | 0.65068 |
| teatime | all | centeronly | 1105 | 1.2762 | 0.72118 | 0.57047 | 0.65882 |
| teatime | all | internalonly | 1105 | 391.64 | 391.08 | 0.45518 | 0.51041 |
| teatime | all | flooronly | 1105 | 1.8634e+06 | 1.8634e+06 | 0.52073 | 0.59729 |
| teatime | train | full | 635 | 1.2359 | 0.66152 | 0.58335 | 0.68976 |
| teatime | train | centeronly | 635 | 1.2253 | 0.65091 | 0.58781 | 0.69606 |
| teatime | train | internalonly | 635 | 379.77 | 379.2 | 0.46295 | 0.52441 |
| teatime | train | flooronly | 635 | 2.1736e+06 | 2.1736e+06 | 0.53151 | 0.6126 |
| teatime | validation | full | 470 | 1.3642 | 0.83541 | 0.53609 | 0.59787 |
| teatime | validation | centeronly | 470 | 1.3449 | 0.81612 | 0.54704 | 0.60851 |
| teatime | validation | internalonly | 470 | 407.67 | 407.14 | 0.44467 | 0.49149 |
| teatime | validation | flooronly | 470 | 1.4444e+06 | 1.4444e+06 | 0.50617 | 0.5766 |
| waldo_kitchen | all | full | 1016 | 1.0402 | 0.42239 | 0.61022 | 0.75295 |
| waldo_kitchen | all | centeronly | 1016 | 1.031 | 0.41317 | 0.61143 | 0.75492 |
| waldo_kitchen | all | internalonly | 1016 | 47.341 | 46.723 | 0.55114 | 0.66339 |
| waldo_kitchen | all | flooronly | 1016 | 56507 | 56506 | 0.59753 | 0.72933 |
| waldo_kitchen | train | full | 607 | 1.0441 | 0.39585 | 0.60186 | 0.74794 |
| waldo_kitchen | train | centeronly | 607 | 1.0403 | 0.39201 | 0.6026 | 0.74959 |
| waldo_kitchen | train | internalonly | 607 | 56.635 | 55.987 | 0.52741 | 0.64086 |
| waldo_kitchen | train | flooronly | 607 | 77819 | 77819 | 0.58118 | 0.71334 |
| waldo_kitchen | validation | full | 409 | 1.0344 | 0.46177 | 0.62263 | 0.76039 |
| waldo_kitchen | validation | centeronly | 409 | 1.0172 | 0.44458 | 0.62452 | 0.76284 |
| waldo_kitchen | validation | internalonly | 409 | 33.548 | 32.975 | 0.58637 | 0.69682 |
| waldo_kitchen | validation | flooronly | 409 | 24877 | 24876 | 0.6218 | 0.75306 |

Full minus center-only paired camera differences (negative CE favors internal GS covariance):

| partition | mask | scene | split | CE difference | camera-cluster 95% CI |
|---|---|---|---|---:|---|
| primary | conditional | all | all | 0.011862 | [0.0061915, 0.017651] |
| primary | conditional | all | train | 0.0072975 | [0.00013839, 0.012956] |
| primary | conditional | all | validation | 0.018312 | [0.007414, 0.02759] |
| primary | conditional | teatime | all | 0.014299 | [0.0036644, 0.023603] |
| primary | conditional | teatime | train | 0.010606 | [-0.0015983, 0.018668] |
| primary | conditional | teatime | validation | 0.019289 | [-0.00086531, 0.036862] |
| primary | conditional | waldo_kitchen | all | 0.0092114 | [0.0039733, 0.013931] |
| primary | conditional | waldo_kitchen | train | 0.003836 | [-0.0037435, 0.0089353] |
| primary | conditional | waldo_kitchen | validation | 0.017189 | [0.011856, 0.022873] |
| primary | strict95 | all | all | 0.014317 | [0.009248, 0.01932] |
| primary | strict95 | all | train | 0.0094144 | [0.0025849, 0.015458] |
| primary | strict95 | all | validation | 0.021436 | [0.013951, 0.029681] |
| primary | strict95 | teatime | all | 0.017678 | [0.0078992, 0.026597] |
| primary | strict95 | teatime | train | 0.011161 | [-0.0025997, 0.021555] |
| primary | strict95 | teatime | validation | 0.026986 | [0.014272, 0.041214] |
| primary | strict95 | waldo_kitchen | all | 0.010857 | [0.0077463, 0.014328] |
| primary | strict95 | waldo_kitchen | train | 0.0076416 | [0.005117, 0.010529] |
| primary | strict95 | waldo_kitchen | validation | 0.015607 | [0.0094305, 0.022062] |
| alt | conditional | all | all | 0.012626 | [0.0063269, 0.018442] |
| alt | conditional | all | train | 0.0090378 | [0.0047849, 0.012947] |
| alt | conditional | all | validation | 0.017756 | [0.0026713, 0.029219] |
| alt | conditional | teatime | all | 0.015717 | [0.0033293, 0.024988] |
| alt | conditional | teatime | train | 0.012803 | [0.0070179, 0.017575] |
| alt | conditional | teatime | validation | 0.019723 | [-0.0079932, 0.040484] |
| alt | conditional | waldo_kitchen | all | 0.0092708 | [0.0048339, 0.013631] |
| alt | conditional | waldo_kitchen | train | 0.0050875 | [-0.00062945, 0.0096993] |
| alt | conditional | waldo_kitchen | validation | 0.015515 | [0.0088689, 0.022145] |
| alt | strict95 | all | all | 0.015483 | [0.011529, 0.020196] |
| alt | strict95 | all | train | 0.010216 | [0.0064275, 0.014323] |
| alt | strict95 | all | validation | 0.023129 | [0.015472, 0.031964] |
| alt | strict95 | teatime | all | 0.019968 | [0.012858, 0.027855] |
| alt | strict95 | teatime | train | 0.012237 | [0.0052383, 0.018903] |
| alt | strict95 | teatime | validation | 0.030905 | [0.01845, 0.045887] |
| alt | strict95 | waldo_kitchen | all | 0.010862 | [0.0075031, 0.014629] |
| alt | strict95 | waldo_kitchen | train | 0.008179 | [0.0051705, 0.011771] |
| alt | strict95 | waldo_kitchen | validation | 0.014864 | [0.0079602, 0.021781] |
| joint_primary | conditional | all | all | 0.011924 | [0.006176, 0.017658] |
| joint_primary | conditional | all | train | 0.0070315 | [-4.3007e-05, 0.01249] |
| joint_primary | conditional | all | validation | 0.018907 | [0.0080166, 0.028568] |
| joint_primary | conditional | teatime | all | 0.014666 | [0.0038031, 0.024119] |
| joint_primary | conditional | teatime | train | 0.010184 | [-0.0015725, 0.017855] |
| joint_primary | conditional | teatime | validation | 0.020827 | [0.0004598, 0.038406] |
| joint_primary | conditional | waldo_kitchen | all | 0.0089594 | [0.003706, 0.013561] |
| joint_primary | conditional | waldo_kitchen | train | 0.0037338 | [-0.004048, 0.0089184] |
| joint_primary | conditional | waldo_kitchen | validation | 0.016734 | [0.011467, 0.022499] |
| joint_primary | strict95 | all | all | 0.014374 | [0.0095738, 0.019156] |
| joint_primary | strict95 | all | train | 0.0091076 | [0.0025332, 0.014792] |
| joint_primary | strict95 | all | validation | 0.022071 | [0.015132, 0.029799] |
| joint_primary | strict95 | teatime | all | 0.017961 | [0.0086951, 0.026529] |
| joint_primary | strict95 | teatime | train | 0.010638 | [-0.0024445, 0.020382] |
| joint_primary | strict95 | teatime | validation | 0.028444 | [0.017108, 0.041592] |
| joint_primary | strict95 | waldo_kitchen | all | 0.01068 | [0.0076662, 0.014128] |
| joint_primary | strict95 | waldo_kitchen | train | 0.0075591 | [0.0051171, 0.010323] |
| joint_primary | strict95 | waldo_kitchen | validation | 0.015341 | [0.009135, 0.021941] |
| joint_alt | conditional | all | all | 0.012001 | [0.0056462, 0.017806] |
| joint_alt | conditional | all | train | 0.0086414 | [0.0033977, 0.013011] |
| joint_alt | conditional | all | validation | 0.016798 | [0.001927, 0.027715] |
| joint_alt | conditional | teatime | all | 0.014782 | [0.002578, 0.023909] |
| joint_alt | conditional | teatime | train | 0.01317 | [0.0075601, 0.018026] |
| joint_alt | conditional | teatime | validation | 0.016998 | [-0.010589, 0.037273] |
| joint_alt | conditional | waldo_kitchen | all | 0.0089957 | [0.0035002, 0.013754] |
| joint_alt | conditional | waldo_kitchen | train | 0.0039036 | [-0.0042524, 0.0096882] |
| joint_alt | conditional | waldo_kitchen | validation | 0.016571 | [0.011531, 0.022202] |
| joint_alt | strict95 | all | all | 0.014843 | [0.011156, 0.01924] |
| joint_alt | strict95 | all | train | 0.0098181 | [0.006054, 0.014004] |
| joint_alt | strict95 | all | validation | 0.022188 | [0.015394, 0.030067] |
| joint_alt | strict95 | teatime | all | 0.018806 | [0.012113, 0.02597] |
| joint_alt | strict95 | teatime | train | 0.011908 | [0.0046901, 0.018655] |
| joint_alt | strict95 | teatime | validation | 0.02868 | [0.017178, 0.042583] |
| joint_alt | strict95 | waldo_kitchen | all | 0.010763 | [0.0075927, 0.014305] |
| joint_alt | strict95 | waldo_kitchen | train | 0.0077034 | [0.0050086, 0.010918] |
| joint_alt | strict95 | waldo_kitchen | validation | 0.015333 | [0.0091651, 0.021773] |

`centeronly` keeps the full regional covariance of primitive means; it is a shape-rich point-cloud control. `internalonly` keeps opacity-weighted individual GS covariances while retaining the same regional mean. `flooronly` uses the fixed tiny numerical floor; it is not a tuned center RBF competitor. Raw center + internal moments sum exactly before independent eigenvalue flooring. All fits are replayed against the original float64 exporter, and serialized original float32 arrays are checked separately. Camera bootstraps do not establish generalization to a population of scenes.
