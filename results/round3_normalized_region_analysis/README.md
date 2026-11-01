# Dense synthetic support: frozen normalized_shape confirmation

Five completed training seeds; 45 archived checkpoints replayed on independently sampled test scenes. The task predicts dense pixel-area transfer conditional on visibility. It has no dustbin/occlusion-rejection supervision and is not real-photograph matching or NVS.

Training/validation/test namespaces are 4100/4200/4300 plus the training seed. Fixed settings: 192/48/96 scenes per seed, 600 optimizer steps, validation every 50 steps including the untrained state, resolution sigma=.25 patch units, nine predeclared methods. No new fitting, selection or tuning was performed.

| Condition | Control | Full-minus-control KL (= CE difference) | 95% seed-scene CI | Five seed differences | JS difference | JS 95% CI |
|---|---|---:|---|---|---:|---|
| oracle | projected_rope | -0.007393 | [-0.008519, -0.006289] | -0.00596, -0.00764, -0.00860, -0.00782, -0.00695 | -0.001040 | [-0.001553, -0.000631] |
| oracle | expected_fixed | -0.007928 | [-0.008851, -0.007024] | -0.00754, -0.00715, -0.00870, -0.00792, -0.00833 | -0.001182 | [-0.001560, -0.000802] |
| oracle | expected_shape | -0.027843 | [-0.029495, -0.026055] | -0.02813, -0.02658, -0.02952, -0.02688, -0.02810 | -0.005963 | [-0.006578, -0.005358] |
| oracle | isometric_shape | +0.003535 | [+0.002989, +0.004052] | +0.00318, +0.00350, +0.00399, +0.00401, +0.00299 | +0.000943 | [+0.000585, +0.001288] |
| oracle | normalized_fixed | -0.007244 | [-0.008637, -0.005952] | -0.00597, -0.00724, -0.00923, -0.00801, -0.00577 | -0.000779 | [-0.001362, -0.000234] |
| oracle | normalized_trace | -0.004067 | [-0.004654, -0.003481] | -0.00421, -0.00391, -0.00480, -0.00394, -0.00347 | -0.000506 | [-0.001128, +0.000004] |
| oracle | normalized_area | -0.004967 | [-0.005754, -0.004161] | -0.00457, -0.00554, -0.00590, -0.00488, -0.00394 | -0.000468 | [-0.001178, +0.000065] |
| oracle | normalized_shuffled | -0.006964 | [-0.008131, -0.005880] | -0.00596, -0.00729, -0.00855, -0.00696, -0.00607 | -0.001006 | [-0.001472, -0.000529] |
| grid_ood | projected_rope | -0.024811 | [-0.027293, -0.022550] | -0.02334, -0.02872, -0.02353, -0.02293, -0.02554 | -0.004294 | [-0.004768, -0.003768] |
| grid_ood | expected_fixed | -0.021571 | [-0.023536, -0.019491] | -0.02246, -0.02395, -0.01934, -0.01968, -0.02242 | -0.003848 | [-0.004307, -0.003222] |
| grid_ood | expected_shape | -0.071728 | [-0.075336, -0.067883] | -0.07146, -0.07364, -0.07246, -0.06721, -0.07388 | -0.015249 | [-0.016125, -0.014359] |
| grid_ood | isometric_shape | +0.008395 | [+0.006644, +0.009958] | +0.00947, +0.00532, +0.00912, +0.00790, +0.01018 | +0.001552 | [+0.001172, +0.001909] |
| grid_ood | normalized_fixed | -0.024028 | [-0.026365, -0.021762] | -0.02214, -0.02743, -0.02512, -0.02390, -0.02155 | -0.003923 | [-0.004626, -0.003111] |
| grid_ood | normalized_trace | -0.012287 | [-0.013867, -0.010811] | -0.01426, -0.01391, -0.01118, -0.01161, -0.01047 | -0.002160 | [-0.002822, -0.001530] |
| grid_ood | normalized_area | -0.013534 | [-0.015671, -0.011742] | -0.01424, -0.01713, -0.01177, -0.01274, -0.01180 | -0.002159 | [-0.002777, -0.001552] |
| grid_ood | normalized_shuffled | -0.023416 | [-0.026372, -0.020879] | -0.02228, -0.02850, -0.02260, -0.02040, -0.02330 | -0.004146 | [-0.004493, -0.003788] |
| wide | projected_rope | -0.032482 | [-0.036365, -0.028657] | -0.03215, -0.03027, -0.03515, -0.03436, -0.03047 | -0.001861 | [-0.002523, -0.001260] |
| wide | expected_fixed | -0.034530 | [-0.038360, -0.030641] | -0.03575, -0.02995, -0.03681, -0.03614, -0.03400 | -0.002845 | [-0.003309, -0.002409] |
| wide | expected_shape | -0.050015 | [-0.053676, -0.046424] | -0.05413, -0.05147, -0.04930, -0.04766, -0.04752 | -0.010053 | [-0.010848, -0.009276] |
| wide | isometric_shape | +0.002089 | [-0.000558, +0.004493] | +0.00036, +0.00269, +0.00488, +0.00474, -0.00222 | -0.000696 | [-0.001250, -0.000155] |
| wide | normalized_fixed | -0.032790 | [-0.037024, -0.028880] | -0.03241, -0.03017, -0.03402, -0.03739, -0.02996 | -0.001772 | [-0.002327, -0.001281] |
| wide | normalized_trace | -0.015410 | [-0.017863, -0.013160] | -0.01328, -0.01744, -0.01407, -0.01796, -0.01430 | -0.001356 | [-0.001896, -0.000918] |
| wide | normalized_area | -0.021661 | [-0.024362, -0.019186] | -0.02086, -0.02131, -0.02045, -0.02463, -0.02106 | -0.001422 | [-0.002075, -0.000909] |
| wide | normalized_shuffled | -0.032224 | [-0.036823, -0.027887] | -0.03349, -0.03223, -0.03734, -0.02710, -0.03096 | -0.002044 | [-0.002656, -0.001467] |

Negative differences favor full covariance. The same-family support controls have 48D output: fixed covariance; trace-preserving isotropic covariance; determinant/area-preserving isotropic covariance; and cyclically shuffled covariance. The latter is an incorrect-support intervention, not a same-information competitor.

## Absolute means

| Condition | Method | CE | KL | JS | Top1 target mass |
|---|---|---:|---:|---:|---:|
| oracle | projected_rope | 0.933525 | 0.156230 | 0.041160 | 0.654226 |
| oracle | expected_fixed | 0.934060 | 0.156765 | 0.041302 | 0.654000 |
| oracle | expected_shape | 0.953974 | 0.176679 | 0.046083 | 0.650633 |
| oracle | isometric_shape | 0.922597 | 0.145302 | 0.039177 | 0.654657 |
| oracle | normalized_fixed | 0.933375 | 0.156081 | 0.040900 | 0.654243 |
| oracle | normalized_shape | 0.926131 | 0.148837 | 0.040120 | 0.654912 |
| oracle | normalized_trace | 0.930198 | 0.152904 | 0.040626 | 0.654580 |
| oracle | normalized_area | 0.931098 | 0.153803 | 0.040588 | 0.654545 |
| oracle | normalized_shuffled | 0.933095 | 0.155801 | 0.041126 | 0.654240 |
| grid_ood | projected_rope | 1.071923 | 0.250994 | 0.061908 | 0.601938 |
| grid_ood | expected_fixed | 1.068683 | 0.247755 | 0.061463 | 0.601882 |
| grid_ood | expected_shape | 1.118841 | 0.297912 | 0.072863 | 0.589619 |
| grid_ood | isometric_shape | 1.038717 | 0.217789 | 0.056062 | 0.604282 |
| grid_ood | normalized_fixed | 1.071141 | 0.250212 | 0.061537 | 0.602170 |
| grid_ood | normalized_shape | 1.047112 | 0.226184 | 0.057614 | 0.604872 |
| grid_ood | normalized_trace | 1.059400 | 0.238471 | 0.059775 | 0.603328 |
| grid_ood | normalized_area | 1.060647 | 0.239718 | 0.059774 | 0.603247 |
| grid_ood | normalized_shuffled | 1.070528 | 0.249600 | 0.061760 | 0.602086 |
| wide | projected_rope | 1.045281 | 0.257974 | 0.055192 | 0.621590 |
| wide | expected_fixed | 1.047329 | 0.260022 | 0.056176 | 0.621037 |
| wide | expected_shape | 1.062814 | 0.275508 | 0.063384 | 0.611995 |
| wide | isometric_shape | 1.010710 | 0.223404 | 0.054028 | 0.621683 |
| wide | normalized_fixed | 1.045589 | 0.258282 | 0.055103 | 0.621557 |
| wide | normalized_shape | 1.012799 | 0.225492 | 0.053331 | 0.623681 |
| wide | normalized_trace | 1.028209 | 0.240902 | 0.054687 | 0.622207 |
| wide | normalized_area | 1.034460 | 0.247154 | 0.054753 | 0.622430 |
| wide | normalized_shuffled | 1.045023 | 0.257716 | 0.055375 | 0.621457 |

## Scope and verification

Query/key rectangles are generated independently; covariance is fitted from each view's own surface samples. Labels count dense reprojected surface pixels rather than a Gaussian formula or shared random support identity. Targets measure image-pixel mass, not physical surface-area IoU or Gaussian overlap.

Each paired bootstrap first resamples training seeds, then independent synthetic scenes within the selected seed. It preserves all valid query rows in a scene and computes a ratio of sums within each seed before averaging seeds. Rows are not independent replicates. CE and KL differences coincide because every method uses identical labels and masks; target entropy is checked explicitly.

Every source snapshot, checkpoint and test-dataset digest is verified. Archived model code is executed directly; subsequent current-model candidates are excluded. Replay metrics are compared with all saved test records and per-scene caches, and checkpoint selection is checked against the complete validation history. See summary.json for hashes, raw five-seed differences and checks; predictions contains float32 probability archives.

These results can support or reject a dense synthetic support mechanism. Even consistent synthetic gains cannot replace real-data replication on new cameras, robust predicted geometry, strong geometric/learned competitors, and rejection/calibration checks. No overall research-success claim follows from this report alone.
