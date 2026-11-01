# External-scene zero-tuning exact-kernel confirmation

All widths and temperatures were frozen from development before opening evaluation files. No fitting or selection uses test. The labels are sparse SfM-track transfer proxies; BC is not coverage truth.

| Method | score | partition | test CE | test KL | top1 target mass | max-bin accuracy |
|---|---|---|---:|---:|---:|---:|
| full_bc | normalise_kernel | primary | 1.2436 | 0.7542 | 0.6014 | 0.6846 |
| full_bc | normalise_kernel | alt | 1.2455 | 0.7579 | 0.6034 | 0.6864 |
| full_bc | softmax_kernel | primary | 1.3043 | 0.8149 | 0.6014 | 0.6846 |
| full_bc | softmax_kernel | alt | 1.2957 | 0.8082 | 0.6034 | 0.6864 |
| fixed_key_cov | normalise_kernel | primary | 1.3008 | 0.8114 | 0.5978 | 0.6806 |
| fixed_key_cov | normalise_kernel | alt | 1.3052 | 0.8177 | 0.5994 | 0.6803 |
| fixed_key_cov | softmax_kernel | primary | 1.5475 | 1.0581 | 0.5978 | 0.6806 |
| fixed_key_cov | softmax_kernel | alt | 1.5529 | 1.0654 | 0.5994 | 0.6803 |
| isotropic_same_area | normalise_kernel | primary | 1.2668 | 0.7775 | 0.5964 | 0.6810 |
| isotropic_same_area | normalise_kernel | alt | 1.2716 | 0.7841 | 0.6003 | 0.6799 |
| isotropic_same_area | softmax_kernel | primary | 1.3927 | 0.9033 | 0.5964 | 0.6810 |
| isotropic_same_area | softmax_kernel | alt | 1.3933 | 0.9058 | 0.6003 | 0.6799 |
| isotropic_same_trace | normalise_kernel | primary | 1.3359 | 0.8465 | 0.5918 | 0.6748 |
| isotropic_same_trace | normalise_kernel | alt | 1.3356 | 0.8481 | 0.5948 | 0.6733 |
| isotropic_same_trace | softmax_kernel | primary | 1.3775 | 0.8881 | 0.5918 | 0.6748 |
| isotropic_same_trace | softmax_kernel | alt | 1.3671 | 0.8796 | 0.5948 | 0.6733 |
| projected_query_bc | normalise_kernel | primary | 1.2799 | 0.7905 | 0.5992 | 0.6838 |
| projected_query_bc | normalise_kernel | alt | 1.2794 | 0.7919 | 0.6049 | 0.6877 |
| projected_query_bc | softmax_kernel | primary | 1.3245 | 0.8351 | 0.5992 | 0.6838 |
| projected_query_bc | softmax_kernel | alt | 1.3077 | 0.8201 | 0.6049 | 0.6877 |
| world_full_bc | normalise_kernel | primary | 1.3315 | 0.8421 | 0.5833 | 0.6646 |
| world_full_bc | normalise_kernel | alt | 1.3229 | 0.8354 | 0.5887 | 0.6705 |
| world_full_bc | softmax_kernel | primary | 1.5195 | 1.0301 | 0.5833 | 0.6646 |
| world_full_bc | softmax_kernel | alt | 1.4950 | 1.0075 | 0.5887 | 0.6705 |
| projected_query_isotropic_same_area | normalise_kernel | primary | 1.2906 | 0.8012 | 0.5928 | 0.6757 |
| projected_query_isotropic_same_area | normalise_kernel | alt | 1.2884 | 0.8009 | 0.6037 | 0.6893 |
| projected_query_isotropic_same_area | softmax_kernel | primary | 1.4226 | 0.9333 | 0.5928 | 0.6757 |
| projected_query_isotropic_same_area | softmax_kernel | alt | 1.4107 | 0.9232 | 0.6037 | 0.6893 |
| world_isotropic_same_volume | normalise_kernel | primary | 1.9829 | 1.4935 | 0.5229 | 0.5899 |
| world_isotropic_same_volume | normalise_kernel | alt | 1.9596 | 1.4721 | 0.5342 | 0.6010 |
| world_isotropic_same_volume | softmax_kernel | primary | 1.9205 | 1.4311 | 0.5229 | 0.5899 |
| world_isotropic_same_volume | softmax_kernel | alt | 1.9045 | 1.4170 | 0.5342 | 0.6010 |
| world_isotropic_same_trace | normalise_kernel | primary | 2.0607 | 1.5713 | 0.4729 | 0.5294 |
| world_isotropic_same_trace | normalise_kernel | alt | 2.0400 | 1.5525 | 0.4837 | 0.5425 |
| world_isotropic_same_trace | softmax_kernel | primary | 2.1464 | 1.6570 | 0.4729 | 0.5294 |
| world_isotropic_same_trace | softmax_kernel | alt | 2.1070 | 1.6195 | 0.4837 | 0.5425 |
| world_plane_isotropic | normalise_kernel | primary | 1.5951 | 1.1058 | 0.5339 | 0.6091 |
| world_plane_isotropic | normalise_kernel | alt | 1.5871 | 1.0996 | 0.5426 | 0.6145 |
| world_plane_isotropic | softmax_kernel | primary | 1.7053 | 1.2159 | 0.5339 | 0.6091 |
| world_plane_isotropic | softmax_kernel | alt | 1.6900 | 1.2025 | 0.5426 | 0.6145 |
| pixel_center_rbf | normalise_kernel | primary | 1.2737 | 0.7843 | 0.5978 | 0.6806 |
| pixel_center_rbf | normalise_kernel | alt | 1.2803 | 0.7928 | 0.5994 | 0.6803 |
| pixel_center_rbf | softmax_kernel | primary | 1.2080 | 0.7186 | 0.5978 | 0.6806 |
| pixel_center_rbf | softmax_kernel | alt | 1.2123 | 0.7248 | 0.5994 | 0.6803 |
| projected_center_rbf | normalise_kernel | primary | 1.2847 | 0.7953 | 0.5997 | 0.6806 |
| projected_center_rbf | normalise_kernel | alt | 1.2937 | 0.8062 | 0.6026 | 0.6848 |
| projected_center_rbf | softmax_kernel | primary | 1.2078 | 0.7184 | 0.5997 | 0.6806 |
| projected_center_rbf | softmax_kernel | alt | 1.2118 | 0.7243 | 0.6026 | 0.6848 |
| world_center_rbf | normalise_kernel | primary | 3.1173 | 2.6279 | 0.5353 | 0.6078 |
| world_center_rbf | normalise_kernel | alt | 3.1211 | 2.6336 | 0.5407 | 0.6108 |
| world_center_rbf | softmax_kernel | primary | 2.3160 | 1.8266 | 0.5353 | 0.6078 |
| world_center_rbf | softmax_kernel | alt | 2.3154 | 1.8279 | 0.5407 | 0.6108 |
| covshuffle_mean | normalise_kernel | primary | 1.3659 | 0.8766 | 0.5775 | 0.6532 |
| covshuffle_mean | normalise_kernel | alt | 1.3825 | 0.8950 | 0.5745 | 0.6488 |
| covshuffle_mean | softmax_kernel | primary | 1.4875 | 0.9981 | 0.5775 | 0.6532 |
| covshuffle_mean | softmax_kernel | alt | 1.4927 | 1.0052 | 0.5745 | 0.6488 |

Primary full-3D minus plane-isotropic CE differences (negative favors full covariance):

| Partition | Scene | CE difference | paired camera 95% CI |
|---|---|---:|---|
| primary | all | -0.2637 | [-0.3365, -0.2064] |
| primary | figurines | -0.3549 | [-0.4910, -0.2583] |
| primary | temple | -0.1311 | [-0.1608, -0.1061] |
| alt | all | -0.2642 | [-0.3424, -0.2065] |
| alt | figurines | -0.3587 | [-0.5010, -0.2657] |
| alt | temple | -0.1271 | [-0.1524, -0.1051] |
| joint_primary | all | -0.2653 | [-0.3357, -0.2082] |
| joint_primary | figurines | -0.3567 | [-0.4881, -0.2616] |
| joint_primary | temple | -0.1333 | [-0.1632, -0.1080] |
| joint_alt | all | -0.2616 | [-0.3413, -0.2033] |
| joint_alt | figurines | -0.3549 | [-0.5011, -0.2618] |
| joint_alt | temple | -0.1268 | [-0.1512, -0.1047] |

`primary` and `alt` each reproduce the development mask: at least four original shared tracks, valid query geometry, strictly positive remaining target mass after masking invalid keys. `joint_*` uses identical query/key masks and conditional labels for a fair GS-subset swap comparison. Lost target mass and every exclusion are retained in `summary.json`. The main runner's >=0.95 retained-mass filter is a separate protocol.

The 3D controls preserve center and respectively normal/thickness/tangential area, volume, or trace. The stronger projected-center RBF baseline remains present; improvement over isotropic 3D controls does not establish superiority over it. `summary.json` contains per-scene summaries and every query-camera cluster's paired differences. Confidence intervals describe variation among observed cameras in these two scenes; no wider scene-population claim is warranted.
