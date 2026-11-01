# Independent exact-kernel confirmation

All widths and temperatures were frozen from development before opening test files. No fitting or selection uses test. The labels are sparse SfM-track transfer proxies; BC is not coverage truth.

| Method | score | partition | test CE | test KL | top1 target mass | max-bin accuracy |
|---|---|---|---:|---:|---:|---:|
| full_bc | normalise_kernel | primary | 1.1620 | 0.6186 | 0.5962 | 0.7119 |
| full_bc | normalise_kernel | alt | 1.1623 | 0.6210 | 0.5903 | 0.6964 |
| full_bc | softmax_kernel | primary | 1.3544 | 0.8110 | 0.5962 | 0.7119 |
| full_bc | softmax_kernel | alt | 1.3620 | 0.8207 | 0.5903 | 0.6964 |
| fixed_key_cov | normalise_kernel | primary | 1.3374 | 0.7940 | 0.5786 | 0.6821 |
| fixed_key_cov | normalise_kernel | alt | 1.3298 | 0.7884 | 0.5861 | 0.6898 |
| fixed_key_cov | softmax_kernel | primary | 1.4145 | 0.8712 | 0.5786 | 0.6821 |
| fixed_key_cov | softmax_kernel | alt | 1.4137 | 0.8724 | 0.5861 | 0.6898 |
| isotropic_same_area | normalise_kernel | primary | 1.1773 | 0.6339 | 0.5881 | 0.7036 |
| isotropic_same_area | normalise_kernel | alt | 1.1770 | 0.6356 | 0.5868 | 0.6947 |
| isotropic_same_area | softmax_kernel | primary | 1.4220 | 0.8786 | 0.5881 | 0.7036 |
| isotropic_same_area | softmax_kernel | alt | 1.4307 | 0.8893 | 0.5868 | 0.6947 |
| isotropic_same_trace | normalise_kernel | primary | 1.2173 | 0.6739 | 0.5815 | 0.6954 |
| isotropic_same_trace | normalise_kernel | alt | 1.2147 | 0.6734 | 0.5842 | 0.6947 |
| isotropic_same_trace | softmax_kernel | primary | 1.4025 | 0.8592 | 0.5815 | 0.6954 |
| isotropic_same_trace | softmax_kernel | alt | 1.4049 | 0.8636 | 0.5842 | 0.6947 |
| projected_query_bc | normalise_kernel | primary | 1.1215 | 0.5781 | 0.6179 | 0.7285 |
| projected_query_bc | normalise_kernel | alt | 1.1195 | 0.5781 | 0.6175 | 0.7310 |
| projected_query_bc | softmax_kernel | primary | 1.2987 | 0.7553 | 0.6179 | 0.7285 |
| projected_query_bc | softmax_kernel | alt | 1.3113 | 0.7699 | 0.6175 | 0.7310 |
| world_full_bc | normalise_kernel | primary | 1.1280 | 0.5847 | 0.6072 | 0.7086 |
| world_full_bc | normalise_kernel | alt | 1.1253 | 0.5840 | 0.6041 | 0.7063 |
| world_full_bc | softmax_kernel | primary | 1.4136 | 0.8703 | 0.6072 | 0.7086 |
| world_full_bc | softmax_kernel | alt | 1.4246 | 0.8832 | 0.6041 | 0.7063 |
| projected_query_isotropic_same_area | normalise_kernel | primary | 1.1285 | 0.5852 | 0.6034 | 0.7185 |
| projected_query_isotropic_same_area | normalise_kernel | alt | 1.1274 | 0.5860 | 0.6054 | 0.7178 |
| projected_query_isotropic_same_area | softmax_kernel | primary | 1.4047 | 0.8613 | 0.6034 | 0.7185 |
| projected_query_isotropic_same_area | softmax_kernel | alt | 1.4187 | 0.8774 | 0.6054 | 0.7178 |
| world_isotropic_same_volume | normalise_kernel | primary | 1.4477 | 0.9043 | 0.5687 | 0.6722 |
| world_isotropic_same_volume | normalise_kernel | alt | 1.4768 | 0.9355 | 0.5661 | 0.6634 |
| world_isotropic_same_volume | softmax_kernel | primary | 1.7091 | 1.1658 | 0.5687 | 0.6722 |
| world_isotropic_same_volume | softmax_kernel | alt | 1.7266 | 1.1852 | 0.5661 | 0.6634 |
| world_isotropic_same_trace | normalise_kernel | primary | 1.6310 | 1.0877 | 0.5435 | 0.6308 |
| world_isotropic_same_trace | normalise_kernel | alt | 1.6360 | 1.0946 | 0.5392 | 0.6254 |
| world_isotropic_same_trace | softmax_kernel | primary | 1.6317 | 1.0884 | 0.5435 | 0.6308 |
| world_isotropic_same_trace | softmax_kernel | alt | 1.6571 | 1.1157 | 0.5392 | 0.6254 |
| world_plane_isotropic | normalise_kernel | primary | 1.2680 | 0.7246 | 0.5937 | 0.6921 |
| world_plane_isotropic | normalise_kernel | alt | 1.2716 | 0.7303 | 0.5765 | 0.6683 |
| world_plane_isotropic | softmax_kernel | primary | 1.5211 | 0.9777 | 0.5937 | 0.6921 |
| world_plane_isotropic | softmax_kernel | alt | 1.5505 | 1.0092 | 0.5765 | 0.6683 |
| pixel_center_rbf | normalise_kernel | primary | 1.3122 | 0.7688 | 0.5786 | 0.6821 |
| pixel_center_rbf | normalise_kernel | alt | 1.3021 | 0.7608 | 0.5861 | 0.6898 |
| pixel_center_rbf | softmax_kernel | primary | 1.1263 | 0.5829 | 0.5786 | 0.6821 |
| pixel_center_rbf | softmax_kernel | alt | 1.1273 | 0.5859 | 0.5861 | 0.6898 |
| projected_center_rbf | normalise_kernel | primary | 1.2940 | 0.7507 | 0.6007 | 0.7136 |
| projected_center_rbf | normalise_kernel | alt | 1.2822 | 0.7408 | 0.6001 | 0.7096 |
| projected_center_rbf | softmax_kernel | primary | 1.0955 | 0.5521 | 0.6007 | 0.7136 |
| projected_center_rbf | softmax_kernel | alt | 1.0990 | 0.5576 | 0.6001 | 0.7096 |
| world_center_rbf | normalise_kernel | primary | 3.3325 | 2.7891 | 0.5716 | 0.6772 |
| world_center_rbf | normalise_kernel | alt | 3.3931 | 2.8517 | 0.5666 | 0.6634 |
| world_center_rbf | softmax_kernel | primary | 1.8655 | 1.3222 | 0.5686 | 0.6722 |
| world_center_rbf | softmax_kernel | alt | 1.8621 | 1.3208 | 0.5628 | 0.6584 |
| covshuffle_mean | normalise_kernel | primary | 1.3063 | 0.7629 | 0.5654 | 0.6619 |
| covshuffle_mean | normalise_kernel | alt | 1.3031 | 0.7617 | 0.5706 | 0.6644 |
| covshuffle_mean | softmax_kernel | primary | 1.4838 | 0.9404 | 0.5654 | 0.6619 |
| covshuffle_mean | softmax_kernel | alt | 1.4721 | 0.9308 | 0.5706 | 0.6644 |

Primary full-3D minus plane-isotropic CE differences (negative favors full covariance):

| Partition | Scene | CE difference | paired camera 95% CI |
|---|---|---:|---|
| primary | all | -0.1399 | [-0.1863, -0.0976] |
| primary | teatime | -0.2201 | [-0.2845, -0.1558] |
| primary | waldo_kitchen | -0.0613 | [-0.0890, -0.0335] |
| alt | all | -0.1463 | [-0.1912, -0.1032] |
| alt | teatime | -0.2163 | [-0.2838, -0.1475] |
| alt | waldo_kitchen | -0.0786 | [-0.1141, -0.0458] |
| joint_primary | all | -0.1366 | [-0.1826, -0.0938] |
| joint_primary | teatime | -0.2135 | [-0.2800, -0.1476] |
| joint_primary | waldo_kitchen | -0.0614 | [-0.0892, -0.0338] |
| joint_alt | all | -0.1460 | [-0.1917, -0.1025] |
| joint_alt | teatime | -0.2170 | [-0.2850, -0.1485] |
| joint_alt | waldo_kitchen | -0.0766 | [-0.1131, -0.0428] |

`primary` and `alt` each reproduce the development mask: at least four original shared tracks, valid query geometry, strictly positive remaining target mass after masking invalid keys. `joint_*` uses identical query/key masks and conditional labels for a fair GS-subset swap comparison. Lost target mass and every exclusion are retained in `summary.json`. The main runner's >=0.95 retained-mass filter is a separate protocol.

The 3D controls preserve center and respectively normal/thickness/tangential area, volume, or trace. The stronger projected-center RBF baseline remains present; improvement over isotropic 3D controls does not establish superiority over it. `summary.json` contains per-scene summaries and every query-camera cluster's paired differences. Confidence intervals describe variation among observed cameras in these two scenes; no wider scene-population claim is warranted.
