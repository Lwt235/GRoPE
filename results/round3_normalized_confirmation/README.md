# Frozen normalized Gaussian PE: fresh-camera replication

Five seeds fit Teatime train only. Validation CE selects each checkpoint before fresh prediction. Both raw RGB and the predeclared frozen ResNet18 representation are reported. All methods receive the same fitted query geometry. Fixed/trace/area/shuffle change key covariance only.

Primary group: previously unused figurines + temple camera pairs. Differences below are normalized_shape minus control; negative CE favors full support. The 95% intervals use paired seed/query-camera bootstrap, stratified by scene. The fixed geometric baseline is repeated across seeds only to form paired differences.

## raw, GS parity 0

| Control | External CE | Full-minus-control CE | 95% CI | Four-scene macro CE |
|---|---:|---:|---|---:|
| projected_rope | 1.245438 | -0.046470 | [-0.078685, -0.019829] | 1.351431 |
| expected_fixed | 1.252948 | -0.053980 | [-0.082722, -0.030107] | 1.323763 |
| expected_shape | 1.313018 | -0.114050 | [-0.140115, -0.090744] | 1.336546 |
| isometric_shape | 1.200779 | -0.001811 | [-0.019311, 0.016686] | 1.283601 |
| normalized_fixed | 1.252299 | -0.053331 | [-0.083099, -0.027006] | 1.359418 |
| normalized_shape | 1.198967 | — | — | 1.290956 |
| normalized_trace | 1.227520 | -0.028552 | [-0.038186, -0.018782] | 1.328757 |
| normalized_area | 1.220350 | -0.021383 | [-0.040511, -0.003864] | 1.323918 |
| normalized_shuffled | 1.246299 | -0.047332 | [-0.072769, -0.025675] | 1.349603 |
| absolute_geometry | 1.885751 | -0.686783 | [-0.803513, -0.57762] | 1.804765 |
| projected_center_rbf | 1.250572 | -0.051605 | [-0.097336, -0.014707] | 1.314211 |
| pixel_center_rbf | 1.242654 | -0.043687 | [-0.088662, -0.006023] | 1.312130 |

Frozen gates: {"support": true, "anisotropy": true, "versus_projected_rope": true, "versus_strong_projected_rbf": true}

## raw, GS parity 1

| Control | External CE | Full-minus-control CE | 95% CI | Four-scene macro CE |
|---|---:|---:|---|---:|
| projected_rope | 1.240999 | -0.060292 | [-0.08685, -0.038075] | 1.316844 |
| expected_fixed | 1.243090 | -0.062383 | [-0.085387, -0.043331] | 1.289667 |
| expected_shape | 1.291667 | -0.110960 | [-0.138136, -0.085016] | 1.302814 |
| isometric_shape | 1.185527 | -0.004820 | [-0.021285, 0.012017] | 1.253699 |
| normalized_fixed | 1.245192 | -0.064486 | [-0.089373, -0.042331] | 1.322427 |
| normalized_shape | 1.180707 | — | — | 1.258822 |
| normalized_trace | 1.208990 | -0.028283 | [-0.038409, -0.019547] | 1.289968 |
| normalized_area | 1.211794 | -0.031087 | [-0.045957, -0.016014] | 1.292535 |
| normalized_shuffled | 1.240478 | -0.059771 | [-0.082796, -0.039951] | 1.312578 |
| absolute_geometry | 1.802503 | -0.621797 | [-0.729666, -0.526566] | 1.746315 |
| projected_center_rbf | 1.244652 | -0.063945 | [-0.103639, -0.03052] | 1.282803 |
| pixel_center_rbf | 1.240937 | -0.060230 | [-0.098031, -0.026336] | 1.285050 |

Frozen gates: {"support": true, "anisotropy": true, "versus_projected_rope": true, "versus_strong_projected_rbf": true}

## frozen, GS parity 0

| Control | External CE | Full-minus-control CE | 95% CI | Four-scene macro CE |
|---|---:|---:|---|---:|
| projected_rope | 1.237413 | -0.064776 | [-0.098885, -0.038082] | 1.382428 |
| expected_fixed | 1.202080 | -0.029443 | [-0.056583, -0.005799] | 1.323578 |
| expected_shape | 1.251451 | -0.078814 | [-0.103487, -0.055578] | 1.340942 |
| isometric_shape | 1.182047 | -0.009410 | [-0.026677, 0.007838] | 1.318983 |
| normalized_fixed | 1.229062 | -0.056425 | [-0.087987, -0.030646] | 1.354102 |
| normalized_shape | 1.172637 | — | — | 1.302594 |
| normalized_trace | 1.204650 | -0.032013 | [-0.043918, -0.022162] | 1.347980 |
| normalized_area | 1.204640 | -0.032003 | [-0.052064, -0.014864] | 1.349894 |
| normalized_shuffled | 1.216288 | -0.043651 | [-0.071626, -0.020958] | 1.353779 |
| absolute_geometry | 1.781538 | -0.608901 | [-0.684512, -0.536668] | 2.049205 |
| projected_center_rbf | 1.250572 | -0.077935 | [-0.122388, -0.041522] | 1.314211 |
| pixel_center_rbf | 1.242654 | -0.070017 | [-0.114118, -0.03109] | 1.312130 |

Frozen gates: {"support": true, "anisotropy": true, "versus_projected_rope": true, "versus_strong_projected_rbf": true}

## frozen, GS parity 1

| Control | External CE | Full-minus-control CE | 95% CI | Four-scene macro CE |
|---|---:|---:|---|---:|
| projected_rope | 1.233574 | -0.079016 | [-0.109951, -0.05452] | 1.351244 |
| expected_fixed | 1.194826 | -0.040268 | [-0.062351, -0.019465] | 1.294166 |
| expected_shape | 1.234036 | -0.079478 | [-0.106013, -0.054052] | 1.314347 |
| isometric_shape | 1.166369 | -0.011811 | [-0.028955, 0.006539] | 1.293693 |
| normalized_fixed | 1.224612 | -0.070054 | [-0.09864, -0.046781] | 1.322042 |
| normalized_shape | 1.154558 | — | — | 1.272516 |
| normalized_trace | 1.185233 | -0.030676 | [-0.040459, -0.022027] | 1.310829 |
| normalized_area | 1.197361 | -0.042804 | [-0.062188, -0.027244] | 1.320083 |
| normalized_shuffled | 1.213650 | -0.059092 | [-0.085611, -0.037796] | 1.322581 |
| absolute_geometry | 1.732870 | -0.578312 | [-0.649843, -0.51024] | 2.005175 |
| projected_center_rbf | 1.244652 | -0.090094 | [-0.129603, -0.0562] | 1.282803 |
| pixel_center_rbf | 1.240937 | -0.086380 | [-0.12535, -0.050976] | 1.285050 |

Frozen gates: {"support": true, "anisotropy": true, "versus_projected_rope": true, "versus_strong_projected_rbf": true}

## Scope and verification

203 pairs were selected before matcher evaluation by excluding all previously exported camera IDs. Tea/kitchen have few remaining query cameras; per-scene results and scene-macro means are included in JSON. The known scenes, GS reconstruction and SfM may already use these source photographs. This is conditional regional transfer against sparse observed tracks, not unseen-scene reconstruction, dense visibility or rejection truth. Five training seeds are not five independent datasets. No test-based checkpoint, feature, partition or bandwidth selection is allowed.

Verification executes archived model code, checks every frozen input/source hash, validates all 100 checkpoints against complete validation selection histories and replays validation metrics. Every evaluated NPZ is checked against its manifest. Probabilities and their hashes are archived. See fresh_summary.json for all scene/partition/seed results, proper-score and target-mass comparisons.
