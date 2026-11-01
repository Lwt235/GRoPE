# Dense synthetic support: frozen isometric confirmation

Five completed training seeds; 45 archived checkpoints replayed on independently sampled test scenes. The task predicts dense pixel-area transfer conditional on visibility. It has no dustbin/occlusion-rejection supervision and is not real-photograph matching or NVS.

Training/validation/test namespaces are 4100/4200/4300 plus the training seed. Fixed settings: 192/48/96 scenes per seed, 600 optimizer steps, validation every 50 steps including the untrained state, resolution sigma=.25 patch units, nine predeclared methods. No new fitting, selection or tuning was performed.

| Condition | Control | Full-minus-control KL (= CE difference) | 95% seed-scene CI | Five seed differences | JS difference | JS 95% CI |
|---|---|---:|---|---|---:|---|
| oracle | isometric_fixed | -0.011305 | [-0.012436, -0.010267] | -0.01058, -0.01012, -0.01268, -0.01185, -0.01130 | -0.002201 | [-0.002558, -0.001891] |
| oracle | isometric_isotropic | -0.004313 | [-0.004874, -0.003747] | -0.00416, -0.00351, -0.00495, -0.00478, -0.00416 | -0.000709 | [-0.000867, -0.000521] |
| oracle | isometric_area | -0.004633 | [-0.005289, -0.004022] | -0.00431, -0.00391, -0.00546, -0.00509, -0.00439 | -0.000691 | [-0.000817, -0.000531] |
| oracle | isometric_shuffled | -0.012277 | [-0.013612, -0.011081] | -0.01163, -0.01079, -0.01377, -0.01336, -0.01184 | -0.002369 | [-0.002654, -0.002090] |
| oracle | projected_rope | -0.010928 | [-0.012355, -0.009550] | -0.00914, -0.01114, -0.01259, -0.01183, -0.00993 | -0.001983 | [-0.002326, -0.001617] |
| oracle | expected_shape | -0.031377 | [-0.033109, -0.029653] | -0.03132, -0.03009, -0.03351, -0.03088, -0.03108 | -0.006906 | [-0.007316, -0.006458] |
| oracle | absolute_geometry | -0.018064 | [-0.023939, -0.012770] | -0.01287, -0.02815, -0.01098, -0.02004, -0.01828 | -0.006755 | [-0.008318, -0.005344] |
| grid_ood | isometric_fixed | -0.029618 | [-0.031627, -0.027730] | -0.03254, -0.02927, -0.02885, -0.02815, -0.02928 | -0.005472 | [-0.006039, -0.004892] |
| grid_ood | isometric_isotropic | -0.011541 | [-0.012523, -0.010479] | -0.01277, -0.01170, -0.01168, -0.00989, -0.01167 | -0.002305 | [-0.002558, -0.002088] |
| grid_ood | isometric_area | -0.010702 | [-0.011605, -0.009689] | -0.01126, -0.01134, -0.01073, -0.00924, -0.01094 | -0.001893 | [-0.002156, -0.001642] |
| grid_ood | isometric_shuffled | -0.032497 | [-0.034643, -0.030379] | -0.03459, -0.03113, -0.03171, -0.03068, -0.03437 | -0.006169 | [-0.006762, -0.005612] |
| grid_ood | projected_rope | -0.033206 | [-0.035443, -0.030924] | -0.03281, -0.03403, -0.03265, -0.03083, -0.03572 | -0.005846 | [-0.006290, -0.005433] |
| grid_ood | expected_shape | -0.080123 | [-0.084092, -0.075849] | -0.08093, -0.07895, -0.08157, -0.07511, -0.08405 | -0.016801 | [-0.017585, -0.016015] |
| grid_ood | absolute_geometry | -0.052987 | [-0.059283, -0.046447] | -0.05385, -0.05731, -0.04287, -0.05219, -0.05873 | -0.013896 | [-0.015173, -0.012507] |
| wide | isometric_fixed | -0.036404 | [-0.041700, -0.031708] | -0.03448, -0.03168, -0.04131, -0.04130, -0.03325 | -0.002218 | [-0.002564, -0.001858] |
| wide | isometric_isotropic | -0.015575 | [-0.017243, -0.013896] | -0.01451, -0.01418, -0.01697, -0.01638, -0.01583 | -0.001965 | [-0.002285, -0.001647] |
| wide | isometric_area | -0.019566 | [-0.022224, -0.017034] | -0.01798, -0.01678, -0.02109, -0.02196, -0.02002 | -0.001227 | [-0.001583, -0.000877] |
| wide | isometric_shuffled | -0.038727 | [-0.044324, -0.033464] | -0.03719, -0.03584, -0.04458, -0.04395, -0.03207 | -0.003133 | [-0.003853, -0.002326] |
| wide | projected_rope | -0.034570 | [-0.040058, -0.029285] | -0.03252, -0.03296, -0.04003, -0.03909, -0.02825 | -0.001164 | [-0.001987, -0.000249] |
| wide | expected_shape | -0.052104 | [-0.055786, -0.047706] | -0.05449, -0.05415, -0.05418, -0.05240, -0.04530 | -0.009357 | [-0.010102, -0.008517] |
| wide | absolute_geometry | -0.038168 | [-0.048023, -0.029794] | -0.04683, -0.04667, -0.02670, -0.03284, -0.03780 | -0.007690 | [-0.008854, -0.006581] |

Negative differences favor full covariance. The main controls have the same 96D output: fixed covariance; trace-preserving isotropic covariance; determinant/area-preserving isotropic covariance; and cyclically shuffled covariance. The latter is an incorrect-support intervention, not a same-information competitor.

## Absolute means

| Condition | Method | CE | KL | JS | Top1 target mass |
|---|---|---:|---:|---:|---:|
| oracle | projected_rope | 0.933525 | 0.156230 | 0.041160 | 0.654226 |
| oracle | expected_fixed | 0.934060 | 0.156765 | 0.041302 | 0.654000 |
| oracle | expected_shape | 0.953974 | 0.176679 | 0.046083 | 0.650633 |
| oracle | isometric_fixed | 0.933901 | 0.156607 | 0.041378 | 0.653994 |
| oracle | isometric_shape | 0.922597 | 0.145302 | 0.039177 | 0.654657 |
| oracle | isometric_isotropic | 0.926909 | 0.149615 | 0.039886 | 0.654103 |
| oracle | isometric_area | 0.927229 | 0.149935 | 0.039868 | 0.654287 |
| oracle | isometric_shuffled | 0.934874 | 0.157579 | 0.041545 | 0.653776 |
| oracle | absolute_geometry | 0.940660 | 0.163366 | 0.045932 | 0.651439 |
| grid_ood | projected_rope | 1.071923 | 0.250994 | 0.061908 | 0.601938 |
| grid_ood | expected_fixed | 1.068683 | 0.247755 | 0.061463 | 0.601882 |
| grid_ood | expected_shape | 1.118841 | 0.297912 | 0.072863 | 0.589619 |
| grid_ood | isometric_fixed | 1.068335 | 0.247407 | 0.061534 | 0.601564 |
| grid_ood | isometric_shape | 1.038717 | 0.217789 | 0.056062 | 0.604282 |
| grid_ood | isometric_isotropic | 1.050258 | 0.229329 | 0.058368 | 0.602627 |
| grid_ood | isometric_area | 1.049419 | 0.228490 | 0.057955 | 0.602730 |
| grid_ood | isometric_shuffled | 1.071214 | 0.250286 | 0.062231 | 0.601404 |
| grid_ood | absolute_geometry | 1.091704 | 0.270776 | 0.069959 | 0.593174 |
| wide | projected_rope | 1.045281 | 0.257974 | 0.055192 | 0.621590 |
| wide | expected_fixed | 1.047329 | 0.260022 | 0.056176 | 0.621037 |
| wide | expected_shape | 1.062814 | 0.275508 | 0.063384 | 0.611995 |
| wide | isometric_fixed | 1.047114 | 0.259807 | 0.056246 | 0.620955 |
| wide | isometric_shape | 1.010710 | 0.223404 | 0.054028 | 0.621683 |
| wide | isometric_isotropic | 1.026285 | 0.238979 | 0.055993 | 0.620170 |
| wide | isometric_area | 1.030277 | 0.242970 | 0.055255 | 0.621149 |
| wide | isometric_shuffled | 1.049438 | 0.262131 | 0.057161 | 0.620004 |
| wide | absolute_geometry | 1.048878 | 0.261571 | 0.061717 | 0.617347 |

## Scope and verification

Query/key rectangles are generated independently; covariance is fitted from each view's own surface samples. Labels count dense reprojected surface pixels rather than a Gaussian formula or shared random support identity. Targets measure image-pixel mass, not physical surface-area IoU or Gaussian overlap.

Each paired bootstrap first resamples training seeds, then independent synthetic scenes within the selected seed. It preserves all valid query rows in a scene and computes a ratio of sums within each seed before averaging seeds. Rows are not independent replicates. CE and KL differences coincide because every method uses identical labels and masks; target entropy is checked explicitly.

Every source snapshot, checkpoint and test-dataset digest is verified. Archived model code is executed directly; subsequent current-model candidates are excluded. Replay metrics are compared with all saved test records and per-scene caches, and checkpoint selection is checked against the complete validation history. See summary.json for hashes, raw five-seed differences and checks; predictions contains float32 probability archives.

These results can support or reject a dense synthetic support mechanism. Even consistent synthetic gains cannot replace real-data replication on new cameras, robust predicted geometry, strong geometric/learned competitors, and rejection/calibration checks. No overall research-success claim follows from this report alone.
