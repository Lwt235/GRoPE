# GS covariance decomposition: confirmation

All methods use identical regional centers, opacity weights, primitive-center visibility, key/row masks and numerical covariance floor. No kernel parameter is fitted. The targets are sparse SfM-track transfer, not dense overlap.

| Scene | split | covariance | rows | CE | KL | top1 target mass | max-bin accuracy |
|---|---|---|---:|---:|---:|---:|---:|
| all | all | full | 3052 | 1.2912 | 0.79115 | 0.58804 | 0.67333 |
| all | all | centeronly | 3052 | 1.2244 | 0.72436 | 0.60163 | 0.69135 |
| all | all | internalonly | 3052 | 121.95 | 121.45 | 0.49387 | 0.56127 |
| all | all | flooronly | 3052 | 4.3819e+05 | 4.3819e+05 | 0.54249 | 0.62156 |
| all | test | full | 1200 | 1.1436 | 0.63453 | 0.61537 | 0.70917 |
| all | test | centeronly | 1200 | 1.115 | 0.60589 | 0.6219 | 0.72 |
| all | test | internalonly | 1200 | 193.45 | 192.94 | 0.51329 | 0.58 |
| all | test | flooronly | 1200 | 1.0901e+06 | 1.0901e+06 | 0.57295 | 0.6625 |
| all | train | full | 1047 | 1.2779 | 0.81456 | 0.61806 | 0.69914 |
| all | train | centeronly | 1047 | 1.1914 | 0.72801 | 0.63665 | 0.72302 |
| all | train | internalonly | 1047 | 69.853 | 69.39 | 0.52638 | 0.59121 |
| all | train | flooronly | 1047 | 5601.8 | 5601.3 | 0.57915 | 0.64661 |
| all | validation | full | 805 | 1.5285 | 0.99417 | 0.50825 | 0.58634 |
| all | validation | centeronly | 805 | 1.4306 | 0.89621 | 0.52585 | 0.60745 |
| all | validation | internalonly | 805 | 83.112 | 82.577 | 0.42263 | 0.49441 |
| all | validation | flooronly | 805 | 29068 | 29067 | 0.44941 | 0.52795 |
| figurines | all | full | 1450 | 1.5905 | 1.0648 | 0.51149 | 0.57931 |
| figurines | all | centeronly | 1450 | 1.4634 | 0.93762 | 0.53913 | 0.61517 |
| figurines | all | internalonly | 1450 | 99.788 | 99.263 | 0.38929 | 0.43655 |
| figurines | all | flooronly | 1450 | 19051 | 19051 | 0.44988 | 0.5069 |
| figurines | test | full | 288 | 1.3802 | 0.86112 | 0.56139 | 0.63889 |
| figurines | test | centeronly | 288 | 1.3134 | 0.79435 | 0.58335 | 0.67361 |
| figurines | test | internalonly | 288 | 91.458 | 90.939 | 0.41329 | 0.44444 |
| figurines | test | flooronly | 288 | 35230 | 35229 | 0.48437 | 0.53472 |
| figurines | train | full | 633 | 1.5845 | 1.0444 | 0.51029 | 0.57978 |
| figurines | train | centeronly | 633 | 1.4488 | 0.90874 | 0.54103 | 0.61927 |
| figurines | train | internalonly | 633 | 100.81 | 100.27 | 0.39366 | 0.44234 |
| figurines | train | flooronly | 633 | 2932.9 | 2932.4 | 0.45987 | 0.51185 |
| figurines | validation | full | 529 | 1.7122 | 1.2 | 0.48576 | 0.54631 |
| figurines | validation | centeronly | 529 | 1.5624 | 1.0502 | 0.51278 | 0.57845 |
| figurines | validation | internalonly | 529 | 103.11 | 102.59 | 0.37098 | 0.42533 |
| figurines | validation | flooronly | 529 | 29531 | 29530 | 0.41915 | 0.48582 |
| teatime | all | full | 299 | 1.2278 | 0.65978 | 0.59223 | 0.68896 |
| teatime | all | centeronly | 299 | 1.2072 | 0.63919 | 0.59537 | 0.69565 |
| teatime | all | internalonly | 299 | 642.58 | 642.01 | 0.43086 | 0.49498 |
| teatime | all | flooronly | 299 | 4.3109e+06 | 4.3109e+06 | 0.53128 | 0.63211 |
| teatime | test | full | 299 | 1.2278 | 0.65978 | 0.59223 | 0.68896 |
| teatime | test | centeronly | 299 | 1.2072 | 0.63919 | 0.59537 | 0.69565 |
| teatime | test | internalonly | 299 | 642.58 | 642.01 | 0.43086 | 0.49498 |
| teatime | test | flooronly | 299 | 4.3109e+06 | 4.3109e+06 | 0.53128 | 0.63211 |
| temple | all | full | 998 | 0.9551 | 0.51856 | 0.68765 | 0.78858 |
| temple | all | centeronly | 998 | 0.94795 | 0.51141 | 0.68753 | 0.78858 |
| temple | all | internalonly | 998 | 30.726 | 30.289 | 0.64262 | 0.73547 |
| temple | all | flooronly | 998 | 17008 | 17007 | 0.65942 | 0.75451 |
| temple | test | full | 308 | 0.95284 | 0.52037 | 0.68183 | 0.77597 |
| temple | test | centeronly | 308 | 0.94379 | 0.51132 | 0.68183 | 0.77597 |
| temple | test | internalonly | 308 | 29.145 | 28.713 | 0.63454 | 0.72078 |
| temple | test | flooronly | 308 | 16842 | 16841 | 0.65839 | 0.75325 |
| temple | train | full | 414 | 0.80921 | 0.46318 | 0.78285 | 0.88164 |
| temple | train | centeronly | 414 | 0.79771 | 0.45168 | 0.78285 | 0.88164 |
| temple | train | internalonly | 414 | 22.525 | 22.179 | 0.72931 | 0.81884 |
| temple | train | flooronly | 414 | 9682.4 | 9682 | 0.76153 | 0.85266 |
| temple | validation | full | 276 | 1.1765 | 0.59964 | 0.55135 | 0.66304 |
| temple | validation | centeronly | 276 | 1.1779 | 0.60111 | 0.55091 | 0.66304 |
| temple | validation | internalonly | 276 | 44.791 | 44.214 | 0.52161 | 0.62681 |
| temple | validation | flooronly | 276 | 28181 | 28180 | 0.5074 | 0.6087 |
| waldo_kitchen | all | full | 305 | 1.0302 | 0.51108 | 0.62191 | 0.72787 |
| waldo_kitchen | all | centeronly | 305 | 1.01 | 0.49082 | 0.62378 | 0.73115 |
| waldo_kitchen | all | internalonly | 305 | 15.369 | 14.85 | 0.56607 | 0.64918 |
| waldo_kitchen | all | flooronly | 305 | 12409 | 12408 | 0.61114 | 0.72131 |
| waldo_kitchen | test | full | 305 | 1.0302 | 0.51108 | 0.62191 | 0.72787 |
| waldo_kitchen | test | centeronly | 305 | 1.01 | 0.49082 | 0.62378 | 0.73115 |
| waldo_kitchen | test | internalonly | 305 | 15.369 | 14.85 | 0.56607 | 0.64918 |
| waldo_kitchen | test | flooronly | 305 | 12409 | 12408 | 0.61114 | 0.72131 |

Full minus center-only paired camera differences (negative CE favors internal GS covariance):

| partition | mask | scene | split | CE difference | camera-cluster 95% CI |
|---|---|---|---|---:|---|
| primary | conditional | all | all | 0.066787 | [0.051488, 0.081727] |
| primary | conditional | all | test | 0.028632 | [0.013963, 0.043663] |
| primary | conditional | all | train | 0.08655 | [0.061403, 0.11066] |
| primary | conditional | all | validation | 0.097959 | [0.065453, 0.13026] |
| primary | conditional | figurines | all | 0.12714 | [0.10694, 0.14494] |
| primary | conditional | figurines | test | 0.066779 | [0.0083104, 0.11645] |
| primary | conditional | figurines | train | 0.13564 | [0.11444, 0.15518] |
| primary | conditional | figurines | validation | 0.14984 | [0.1203, 0.18287] |
| primary | conditional | teatime | all | 0.020597 | [0.012186, 0.032876] |
| primary | conditional | teatime | test | 0.020597 | [0.012186, 0.032876] |
| primary | conditional | temple | all | 0.0071531 | [0.0044173, 0.0095819] |
| primary | conditional | temple | test | 0.0090499 | [0.0046799, 0.012411] |
| primary | conditional | temple | train | 0.011495 | [0.0089855, 0.013815] |
| primary | conditional | temple | validation | -0.0014761 | [-0.0064101, 0.0029904] |
| primary | conditional | waldo_kitchen | all | 0.020262 | [0.014251, 0.026823] |
| primary | conditional | waldo_kitchen | test | 0.020262 | [0.014251, 0.026823] |
| primary | strict95 | all | all | 0.068242 | [0.05412, 0.082474] |
| primary | strict95 | all | test | 0.030836 | [0.022341, 0.041022] |
| primary | strict95 | all | train | 0.085526 | [0.060115, 0.11006] |
| primary | strict95 | all | validation | 0.10257 | [0.070423, 0.13304] |
| primary | strict95 | figurines | all | 0.13213 | [0.11553, 0.14811] |
| primary | strict95 | figurines | test | 0.08051 | [0.055773, 0.10752] |
| primary | strict95 | figurines | train | 0.1364 | [0.11624, 0.15556] |
| primary | strict95 | figurines | validation | 0.15205 | [0.12333, 0.18479] |
| primary | strict95 | teatime | all | 0.022054 | [0.012467, 0.036483] |
| primary | strict95 | teatime | test | 0.022054 | [0.012467, 0.036483] |
| primary | strict95 | temple | all | 0.0072427 | [0.0044926, 0.009556] |
| primary | strict95 | temple | test | 0.0090476 | [0.0047164, 0.01248] |
| primary | strict95 | temple | train | 0.011091 | [0.0089799, 0.012813] |
| primary | strict95 | temple | validation | -0.0015165 | [-0.0070571, 0.0036053] |
| primary | strict95 | waldo_kitchen | all | 0.020976 | [0.014527, 0.028005] |
| primary | strict95 | waldo_kitchen | test | 0.020976 | [0.014527, 0.028005] |
| alt | conditional | all | all | 0.067715 | [0.052518, 0.082275] |
| alt | conditional | all | test | 0.02941 | [0.019343, 0.040835] |
| alt | conditional | all | train | 0.089603 | [0.061375, 0.11745] |
| alt | conditional | all | validation | 0.096787 | [0.065837, 0.12592] |
| alt | conditional | figurines | all | 0.12933 | [0.11079, 0.147] |
| alt | conditional | figurines | test | 0.071116 | [0.03742, 0.10381] |
| alt | conditional | figurines | train | 0.14036 | [0.11242, 0.16634] |
| alt | conditional | figurines | validation | 0.14828 | [0.12091, 0.17592] |
| alt | conditional | teatime | all | 0.018463 | [0.012546, 0.026929] |
| alt | conditional | teatime | test | 0.018463 | [0.012546, 0.026929] |
| alt | conditional | temple | all | 0.0071986 | [0.0046736, 0.0093837] |
| alt | conditional | temple | test | 0.0082687 | [0.003782, 0.01189] |
| alt | conditional | temple | train | 0.011612 | [0.0098932, 0.013169] |
| alt | conditional | temple | validation | -0.00047163 | [-0.0049177, 0.0045837] |
| alt | conditional | waldo_kitchen | all | 0.021807 | [0.0153, 0.028127] |
| alt | conditional | waldo_kitchen | test | 0.021807 | [0.0153, 0.028127] |
| alt | strict95 | all | all | 0.069367 | [0.054337, 0.083583] |
| alt | strict95 | all | test | 0.03135 | [0.022156, 0.041831] |
| alt | strict95 | all | train | 0.087307 | [0.059745, 0.11414] |
| alt | strict95 | all | validation | 0.10374 | [0.072702, 0.13217] |
| alt | strict95 | figurines | all | 0.13432 | [0.11691, 0.15071] |
| alt | strict95 | figurines | test | 0.083372 | [0.054482, 0.11234] |
| alt | strict95 | figurines | train | 0.13902 | [0.11183, 0.16352] |
| alt | strict95 | figurines | validation | 0.15361 | [0.12928, 0.17937] |
| alt | strict95 | teatime | all | 0.019895 | [0.012728, 0.03078] |
| alt | strict95 | teatime | test | 0.019895 | [0.012728, 0.03078] |
| alt | strict95 | temple | all | 0.0078786 | [0.0054793, 0.010092] |
| alt | strict95 | temple | test | 0.0086308 | [0.0042446, 0.012154] |
| alt | strict95 | temple | train | 0.011457 | [0.009736, 0.012933] |
| alt | strict95 | temple | validation | 0.0010268 | [-0.0038419, 0.0069217] |
| alt | strict95 | waldo_kitchen | all | 0.022812 | [0.015385, 0.030314] |
| alt | strict95 | waldo_kitchen | test | 0.022812 | [0.015385, 0.030314] |
| joint_primary | conditional | all | all | 0.065286 | [0.050463, 0.079816] |
| joint_primary | conditional | all | test | 0.027431 | [0.013147, 0.041881] |
| joint_primary | conditional | all | train | 0.085711 | [0.060541, 0.10962] |
| joint_primary | conditional | all | validation | 0.095897 | [0.064339, 0.12669] |
| joint_primary | conditional | figurines | all | 0.12456 | [0.10437, 0.1422] |
| joint_primary | conditional | figurines | test | 0.061627 | [0.0039729, 0.10992] |
| joint_primary | conditional | figurines | train | 0.13482 | [0.11388, 0.15427] |
| joint_primary | conditional | figurines | validation | 0.14682 | [0.11869, 0.17915] |
| joint_primary | conditional | teatime | all | 0.021221 | [0.012047, 0.034012] |
| joint_primary | conditional | teatime | test | 0.021221 | [0.012047, 0.034012] |
| joint_primary | conditional | temple | all | 0.0069922 | [0.0043472, 0.0092662] |
| joint_primary | conditional | temple | test | 0.008967 | [0.0048802, 0.012278] |
| joint_primary | conditional | temple | train | 0.011073 | [0.0088229, 0.012913] |
| joint_primary | conditional | temple | validation | -0.00129 | [-0.0064011, 0.0032434] |
| joint_primary | conditional | waldo_kitchen | all | 0.020068 | [0.013974, 0.026759] |
| joint_primary | conditional | waldo_kitchen | test | 0.020068 | [0.013974, 0.026759] |
| joint_primary | strict95 | all | all | 0.06677 | [0.05311, 0.080274] |
| joint_primary | strict95 | all | test | 0.030228 | [0.021977, 0.040133] |
| joint_primary | strict95 | all | train | 0.08471 | [0.059611, 0.10889] |
| joint_primary | strict95 | all | validation | 0.099224 | [0.069526, 0.12609] |
| joint_primary | strict95 | figurines | all | 0.12955 | [0.11442, 0.14354] |
| joint_primary | strict95 | figurines | test | 0.079448 | [0.055355, 0.10583] |
| joint_primary | strict95 | figurines | train | 0.13576 | [0.11645, 0.15358] |
| joint_primary | strict95 | figurines | validation | 0.14616 | [0.1211, 0.17413] |
| joint_primary | strict95 | teatime | all | 0.020893 | [0.01209, 0.034095] |
| joint_primary | strict95 | teatime | test | 0.020893 | [0.01209, 0.034095] |
| joint_primary | strict95 | temple | all | 0.0075166 | [0.0048489, 0.0097568] |
| joint_primary | strict95 | temple | test | 0.0095526 | [0.005984, 0.01237] |
| joint_primary | strict95 | temple | train | 0.010996 | [0.0089073, 0.012637] |
| joint_primary | strict95 | temple | validation | -0.00094035 | [-0.0067644, 0.0043394] |
| joint_primary | strict95 | waldo_kitchen | all | 0.020716 | [0.01411, 0.027877] |
| joint_primary | strict95 | waldo_kitchen | test | 0.020716 | [0.01411, 0.027877] |
| joint_alt | conditional | all | all | 0.064394 | [0.050168, 0.078075] |
| joint_alt | conditional | all | test | 0.027855 | [0.018571, 0.038257] |
| joint_alt | conditional | all | train | 0.084313 | [0.057817, 0.11061] |
| joint_alt | conditional | all | validation | 0.093677 | [0.063587, 0.12201] |
| joint_alt | conditional | figurines | all | 0.12336 | [0.10547, 0.1404] |
| joint_alt | conditional | figurines | test | 0.066763 | [0.035607, 0.095969] |
| joint_alt | conditional | figurines | train | 0.13225 | [0.10496, 0.15731] |
| joint_alt | conditional | figurines | validation | 0.14381 | [0.11738, 0.17052] |
| joint_alt | conditional | teatime | all | 0.017525 | [0.012173, 0.025047] |
| joint_alt | conditional | teatime | test | 0.017525 | [0.012173, 0.025047] |
| joint_alt | conditional | temple | all | 0.0064738 | [0.0039115, 0.0086892] |
| joint_alt | conditional | temple | test | 0.0074001 | [0.0024795, 0.011315] |
| joint_alt | conditional | temple | train | 0.011467 | [0.010038, 0.01279] |
| joint_alt | conditional | temple | validation | -0.0019908 | [-0.0061996, 0.0022102] |
| joint_alt | conditional | waldo_kitchen | all | 0.022114 | [0.015275, 0.028982] |
| joint_alt | conditional | waldo_kitchen | test | 0.022114 | [0.015275, 0.028982] |
| joint_alt | strict95 | all | all | 0.06682 | [0.052416, 0.080754] |
| joint_alt | strict95 | all | test | 0.030403 | [0.02156, 0.040678] |
| joint_alt | strict95 | all | train | 0.083649 | [0.057049, 0.10974] |
| joint_alt | strict95 | all | validation | 0.10056 | [0.070211, 0.12833] |
| joint_alt | strict95 | figurines | all | 0.12968 | [0.11222, 0.14573] |
| joint_alt | strict95 | figurines | test | 0.081733 | [0.053763, 0.10935] |
| joint_alt | strict95 | figurines | train | 0.13355 | [0.10633, 0.15784] |
| joint_alt | strict95 | figurines | validation | 0.14796 | [0.12375, 0.17357] |
| joint_alt | strict95 | teatime | all | 0.018655 | [0.012236, 0.028764] |
| joint_alt | strict95 | teatime | test | 0.018655 | [0.012236, 0.028764] |
| joint_alt | strict95 | temple | all | 0.0075788 | [0.0052747, 0.0095777] |
| joint_alt | strict95 | temple | test | 0.0086194 | [0.0045532, 0.011946] |
| joint_alt | strict95 | temple | train | 0.0116 | [0.01021, 0.012848] |
| joint_alt | strict95 | temple | validation | -0.00057831 | [-0.0047987, 0.0038605] |
| joint_alt | strict95 | waldo_kitchen | all | 0.022661 | [0.015076, 0.030865] |
| joint_alt | strict95 | waldo_kitchen | test | 0.022661 | [0.015076, 0.030865] |

`centeronly` keeps the full regional covariance of primitive means; it is a shape-rich point-cloud control. `internalonly` keeps opacity-weighted individual GS covariances while retaining the same regional mean. `flooronly` uses the fixed tiny numerical floor; it is not a tuned center RBF competitor. Raw center + internal moments sum exactly before independent eigenvalue flooring. All fits are replayed against the original float64 exporter, and serialized original float32 arrays are checked separately. Camera bootstraps do not establish generalization to a population of scenes.
