# Isometric PE frozen confirmation replay

Only completed five-seed runs enter this report. All differences are candidate minus control; lower CE/KL is better. Camera intervals concern the observed reconstructed scenes, not a scene population.

## Real sparse-track transfer

All methods use the runner's >=95% retained-mass mask. The strong projected RBF uses its OLD validation-preferred score mode and frozen width/temperature. That old sweep included Teatime and Waldo validation, so Waldo is held out from neural fitting, not from this baseline's hyperparameter development. Projected-center RBF uses the fitted query center; pixel-center RBF and the proposed PE use the native image query center. Fixed baselines are not independent seed replicates.

| Condition | Control | CE difference | seed-camera 95% CI | Five seed differences |
|---|---|---:|---|---|
| teatime | isometric_fixed | -0.06960 | [-0.20387, -0.00266] | -0.0621, -0.0662, -0.0844, -0.0639, -0.0713 |
| teatime | isometric_isotropic | -0.02409 | [-0.06456, -0.00062] | -0.0278, -0.0224, -0.0243, -0.0199, -0.0261 |
| teatime | isometric_area | -0.02430 | [-0.07302, +0.00281] | -0.0221, -0.0260, -0.0294, -0.0210, -0.0230 |
| teatime | isometric_shuffled | -0.10507 | [-0.22346, -0.03429] | -0.1084, -0.0905, -0.1083, -0.1115, -0.1067 |
| teatime | expected_shape | -0.04131 | [-0.12822, +0.01232] | -0.0328, -0.0440, -0.0443, -0.0560, -0.0295 |
| teatime | projected_rope | -0.08385 | [-0.29761, +0.02268] | -0.0632, -0.0763, -0.0992, -0.0985, -0.0821 |
| teatime | absolute_geometry | -0.63975 | [-0.95832, -0.46034] | -0.6771, -0.5988, -0.6360, -0.6094, -0.6775 |
| teatime | projected_center_rbf | -0.00382 | [-0.15183, +0.08884] | +0.0140, -0.0150, -0.0032, -0.0169, +0.0019 |
| waldo_kitchen | isometric_fixed | -0.01818 | [-0.02694, -0.00953] | -0.0139, -0.0167, -0.0098, -0.0208, -0.0297 |
| waldo_kitchen | isometric_isotropic | -0.00356 | [-0.00966, +0.00757] | -0.0098, -0.0050, -0.0049, +0.0108, -0.0089 |
| waldo_kitchen | isometric_area | -0.00386 | [-0.00836, -0.00010] | -0.0032, -0.0022, -0.0010, -0.0025, -0.0104 |
| waldo_kitchen | isometric_shuffled | -0.03038 | [-0.04779, -0.01413] | -0.0217, -0.0366, -0.0285, -0.0251, -0.0400 |
| waldo_kitchen | expected_shape | -0.00913 | [-0.02738, +0.01255] | -0.0021, -0.0161, -0.0086, -0.0129, -0.0060 |
| waldo_kitchen | projected_rope | -0.00714 | [-0.02570, +0.01014] | +0.0131, -0.0181, -0.0004, -0.0098, -0.0205 |
| waldo_kitchen | absolute_geometry | -0.48689 | [-0.57147, -0.40892] | -0.5362, -0.4413, -0.5396, -0.5060, -0.4113 |
| waldo_kitchen | projected_center_rbf | +0.08950 | [+0.03875, +0.14534] | +0.0831, +0.0860, +0.0839, +0.0984, +0.0961 |
| pooled | isometric_fixed | -0.04353 | [-0.11088, -0.00939] | -0.0377, -0.0411, -0.0466, -0.0421, -0.0502 |
| pooled | isometric_isotropic | -0.01368 | [-0.03365, +0.00073] | -0.0187, -0.0136, -0.0144, -0.0043, -0.0174 |
| pooled | isometric_area | -0.01394 | [-0.03882, -0.00011] | -0.0125, -0.0139, -0.0150, -0.0116, -0.0166 |
| pooled | isometric_shuffled | -0.06721 | [-0.12649, -0.03011] | -0.0644, -0.0632, -0.0679, -0.0677, -0.0729 |
| pooled | expected_shape | -0.02500 | [-0.06774, +0.00484] | -0.0173, -0.0298, -0.0262, -0.0341, -0.0176 |
| pooled | projected_rope | -0.04497 | [-0.14967, +0.01056] | -0.0245, -0.0468, -0.0492, -0.0536, -0.0508 |
| pooled | absolute_geometry | -0.56227 | [-0.71854, -0.45545] | -0.6057, -0.5190, -0.5871, -0.5570, -0.5426 |
| pooled | projected_center_rbf | +0.04348 | [-0.03322, +0.09919] | +0.0490, +0.0362, +0.0409, +0.0415, +0.0497 |
| teatime_alt | isometric_fixed | -0.07720 | [-0.22341, -0.00478] | -0.0694, -0.0746, -0.0922, -0.0701, -0.0797 |
| teatime_alt | isometric_isotropic | -0.02291 | [-0.06200, +0.00037] | -0.0274, -0.0214, -0.0230, -0.0172, -0.0256 |
| teatime_alt | isometric_area | -0.02765 | [-0.07875, +0.00029] | -0.0258, -0.0294, -0.0326, -0.0240, -0.0266 |
| teatime_alt | isometric_shuffled | -0.11821 | [-0.25390, -0.03906] | -0.1218, -0.1041, -0.1225, -0.1229, -0.1198 |
| teatime_alt | expected_shape | -0.03939 | [-0.13079, +0.01644] | -0.0312, -0.0422, -0.0419, -0.0528, -0.0288 |
| teatime_alt | projected_rope | -0.08997 | [-0.31211, +0.02190] | -0.0696, -0.0840, -0.1061, -0.1007, -0.0895 |
| teatime_alt | absolute_geometry | -0.65596 | [-1.00225, -0.46503] | -0.6879, -0.6188, -0.6546, -0.6255, -0.6929 |
| teatime_alt | projected_center_rbf | -0.01075 | [-0.17159, +0.09480] | +0.0066, -0.0210, -0.0103, -0.0225, -0.0065 |
| waldo_kitchen_alt | isometric_fixed | -0.01766 | [-0.02658, -0.00877] | -0.0130, -0.0168, -0.0096, -0.0200, -0.0289 |
| waldo_kitchen_alt | isometric_isotropic | -0.00385 | [-0.01006, +0.00731] | -0.0103, -0.0054, -0.0052, +0.0107, -0.0090 |
| waldo_kitchen_alt | isometric_area | -0.00430 | [-0.00876, -0.00050] | -0.0035, -0.0030, -0.0016, -0.0029, -0.0106 |
| waldo_kitchen_alt | isometric_shuffled | -0.02826 | [-0.04647, -0.01201] | -0.0182, -0.0353, -0.0272, -0.0221, -0.0385 |
| waldo_kitchen_alt | expected_shape | -0.00844 | [-0.02583, +0.01365] | +0.0009, -0.0184, -0.0082, -0.0105, -0.0061 |
| waldo_kitchen_alt | projected_rope | -0.00540 | [-0.02272, +0.01123] | +0.0148, -0.0161, +0.0012, -0.0071, -0.0198 |
| waldo_kitchen_alt | absolute_geometry | -0.49176 | [-0.57810, -0.41118] | -0.5377, -0.4491, -0.5430, -0.5139, -0.4150 |
| waldo_kitchen_alt | projected_center_rbf | +0.08250 | [+0.03166, +0.13814] | +0.0777, +0.0776, +0.0776, +0.0913, +0.0883 |
| pooled_alt | isometric_fixed | -0.04697 | [-0.12017, -0.01019] | -0.0408, -0.0453, -0.0503, -0.0447, -0.0539 |
| pooled_alt | isometric_isotropic | -0.01324 | [-0.03255, +0.00116] | -0.0187, -0.0133, -0.0140, -0.0030, -0.0172 |
| pooled_alt | isometric_area | -0.01580 | [-0.04226, -0.00159] | -0.0144, -0.0160, -0.0168, -0.0133, -0.0185 |
| pooled_alt | isometric_shuffled | -0.07254 | [-0.14058, -0.03161] | -0.0692, -0.0692, -0.0741, -0.0717, -0.0785 |
| pooled_alt | expected_shape | -0.02368 | [-0.06894, +0.00704] | -0.0149, -0.0301, -0.0248, -0.0313, -0.0173 |
| pooled_alt | projected_rope | -0.04704 | [-0.15643, +0.01089] | -0.0267, -0.0495, -0.0517, -0.0532, -0.0541 |
| pooled_alt | absolute_geometry | -0.57260 | [-0.74394, -0.45834] | -0.6117, -0.5327, -0.5980, -0.5688, -0.5519 |
| pooled_alt | projected_center_rbf | +0.03659 | [-0.04664, +0.09702] | +0.0426, +0.0290, +0.0344, +0.0353, +0.0416 |

Sparse-track transfer is a proxy; the supplied reconstruction may have used the evaluation photographs. GS-subset swaps can change eligible rows, so primary/alt are separate robustness evaluations rather than identical-mask paired geometry effects.

## Original point task

Visible Recall@1 includes rejecting a visible match as an error. Closed-candidate ranking, unmatched AP, false acceptance and calibration remain separate in JSON and probability archives.

| Regime | Condition | Control | Visible R1 difference | seed-scene 95% CI |
|---|---|---|---:|---|
| oracle_train | oracle | projected_rope | +0.00491 | [+0.00095, +0.00940] |
| oracle_train | oracle | expected_shape | +0.00792 | [-0.00691, +0.02608] |
| oracle_train | oracle | isometric_fixed | +0.00037 | [-0.00075, +0.00154] |
| oracle_train | oracle | isometric_shuffled | +0.00164 | [-0.00036, +0.00380] |
| oracle_train | depth_noise | projected_rope | +0.00944 | [+0.00554, +0.01351] |
| oracle_train | depth_noise | expected_shape | +0.01128 | [-0.00410, +0.02883] |
| oracle_train | depth_noise | isometric_fixed | -0.00011 | [-0.00150, +0.00130] |
| oracle_train | depth_noise | isometric_shuffled | +0.00153 | [-0.00043, +0.00352] |
| oracle_train | normal_noise | projected_rope | +0.00085 | [-0.00296, +0.00517] |
| oracle_train | normal_noise | expected_shape | +0.01513 | [+0.00069, +0.03285] |
| oracle_train | normal_noise | isometric_fixed | -0.00369 | [-0.00536, -0.00201] |
| oracle_train | normal_noise | isometric_shuffled | +0.00269 | [+0.00098, +0.00441] |
| oracle_train | scale_noise | projected_rope | -0.02935 | [-0.03509, -0.02391] |
| oracle_train | scale_noise | expected_shape | +0.17170 | [+0.15096, +0.19331] |
| oracle_train | scale_noise | isometric_fixed | -0.03388 | [-0.03849, -0.02961] |
| oracle_train | scale_noise | isometric_shuffled | +0.00604 | [+0.00119, +0.01081] |
| oracle_train | mixed | projected_rope | -0.00374 | [-0.00791, +0.00085] |
| oracle_train | mixed | expected_shape | +0.05147 | [+0.03488, +0.06960] |
| oracle_train | mixed | isometric_fixed | -0.00980 | [-0.01183, -0.00793] |
| oracle_train | mixed | isometric_shuffled | +0.00415 | [+0.00030, +0.00865] |
| oracle_train | depth_fit | projected_rope | +0.00707 | [+0.00321, +0.01124] |
| oracle_train | depth_fit | expected_shape | +0.02467 | [+0.01070, +0.04240] |
| oracle_train | depth_fit | isometric_fixed | +0.00048 | [-0.00057, +0.00153] |
| oracle_train | depth_fit | isometric_shuffled | +0.00521 | [+0.00312, +0.00729] |
| oracle_train | wide | projected_rope | +0.00196 | [-0.00506, +0.00978] |
| oracle_train | wide | expected_shape | +0.00799 | [-0.00461, +0.02059] |
| oracle_train | wide | isometric_fixed | +0.00274 | [+0.00053, +0.00505] |
| oracle_train | wide | isometric_shuffled | +0.00838 | [+0.00562, +0.01124] |
| oracle_train | rgb_pred | projected_rope | +0.01025 | [+0.00613, +0.01468] |
| oracle_train | rgb_pred | expected_shape | +0.01452 | [+0.00203, +0.02988] |
| oracle_train | rgb_pred | isometric_fixed | -0.00074 | [-0.00154, +0.00005] |
| oracle_train | rgb_pred | isometric_shuffled | -0.00058 | [-0.00138, +0.00021] |
| robust_train | oracle | projected_rope | +0.00769 | [+0.00303, +0.01242] |
| robust_train | oracle | expected_shape | -0.00477 | [-0.01396, +0.00597] |
| robust_train | oracle | isometric_fixed | -0.00048 | [-0.00221, +0.00130] |
| robust_train | oracle | isometric_shuffled | +0.00211 | [+0.00053, +0.00403] |
| robust_train | depth_noise | projected_rope | +0.01282 | [+0.00793, +0.01822] |
| robust_train | depth_noise | expected_shape | +0.00003 | [-0.00921, +0.01039] |
| robust_train | depth_noise | isometric_fixed | -0.00043 | [-0.00200, +0.00129] |
| robust_train | depth_noise | isometric_shuffled | +0.00264 | [+0.00044, +0.00538] |
| robust_train | normal_noise | projected_rope | +0.00321 | [-0.00106, +0.00779] |
| robust_train | normal_noise | expected_shape | +0.00214 | [-0.00744, +0.01299] |
| robust_train | normal_noise | isometric_fixed | -0.00496 | [-0.00728, -0.00259] |
| robust_train | normal_noise | isometric_shuffled | +0.00285 | [+0.00097, +0.00475] |
| robust_train | scale_noise | projected_rope | -0.02692 | [-0.03240, -0.02181] |
| robust_train | scale_noise | expected_shape | +0.14828 | [+0.13006, +0.16654] |
| robust_train | scale_noise | isometric_fixed | -0.03510 | [-0.03973, -0.03093] |
| robust_train | scale_noise | isometric_shuffled | +0.00784 | [+0.00353, +0.01205] |
| robust_train | mixed | projected_rope | -0.00068 | [-0.00519, +0.00391] |
| robust_train | mixed | expected_shape | +0.03564 | [+0.02352, +0.04872] |
| robust_train | mixed | isometric_fixed | -0.01097 | [-0.01362, -0.00851] |
| robust_train | mixed | isometric_shuffled | +0.00462 | [+0.00026, +0.00969] |
| robust_train | depth_fit | projected_rope | +0.01021 | [+0.00626, +0.01419] |
| robust_train | depth_fit | expected_shape | +0.01013 | [-0.00020, +0.02246] |
| robust_train | depth_fit | isometric_fixed | -0.00059 | [-0.00249, +0.00126] |
| robust_train | depth_fit | isometric_shuffled | +0.00478 | [+0.00256, +0.00704] |
| robust_train | wide | projected_rope | +0.00609 | [-0.00079, +0.01358] |
| robust_train | wide | expected_shape | -0.00267 | [-0.01211, +0.00674] |
| robust_train | wide | isometric_fixed | +0.00178 | [-0.00159, +0.00602] |
| robust_train | wide | isometric_shuffled | +0.00923 | [+0.00557, +0.01268] |
| robust_train | rgb_pred | projected_rope | +0.01489 | [+0.01035, +0.02068] |
| robust_train | rgb_pred | expected_shape | +0.00443 | [-0.00317, +0.01235] |
| robust_train | rgb_pred | isometric_fixed | +0.00026 | [-0.00133, +0.00193] |
| robust_train | rgb_pred | isometric_shuffled | -0.00005 | [-0.00143, +0.00126] |

Rejection and calibration (five-seed means; all numbers below are percentages). Higher unmatched AP is better; lower false acceptance/ECE is better. These threshold and calibration measures must accompany Recall@1.

| Regime | Condition | Method | Unmatched AP | False acceptance | ECE |
|---|---|---|---:|---:|---:|
| oracle_train | oracle | projected_rope | 56.13 | 77.98 | 5.21 |
| oracle_train | oracle | expected_shape | 45.18 | 85.83 | 7.01 |
| oracle_train | oracle | isometric_fixed | 59.62 | 88.32 | 17.11 |
| oracle_train | oracle | isometric_shape | 61.30 | 88.74 | 16.81 |
| oracle_train | rgb_pred | projected_rope | 47.02 | 82.27 | 5.08 |
| oracle_train | rgb_pred | expected_shape | 35.54 | 91.51 | 7.73 |
| oracle_train | rgb_pred | isometric_fixed | 53.67 | 88.74 | 13.38 |
| oracle_train | rgb_pred | isometric_shape | 55.04 | 89.11 | 12.95 |
| robust_train | oracle | projected_rope | 56.71 | 76.05 | 6.60 |
| robust_train | oracle | expected_shape | 46.32 | 88.42 | 7.06 |
| robust_train | oracle | isometric_fixed | 59.42 | 88.62 | 17.08 |
| robust_train | oracle | isometric_shape | 61.20 | 88.76 | 16.48 |
| robust_train | rgb_pred | projected_rope | 47.95 | 80.25 | 5.02 |
| robust_train | rgb_pred | expected_shape | 36.77 | 94.41 | 7.93 |
| robust_train | rgb_pred | isometric_fixed | 53.64 | 89.09 | 13.33 |
| robust_train | rgb_pred | isometric_shape | 55.24 | 89.44 | 12.59 |

This report records evidence and failures; a positive isolated comparison does not establish the complete research objective. Probability files are local replay artifacts; source, input and checkpoint hashes are in the JSON summaries.
