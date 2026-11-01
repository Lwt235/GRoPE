# Real regional geometry development probe

Only train and validation data were opened. Targets are sparse observed SfM-track transfer, not coverage truth. Validation-selected results are diagnostic and require held-out confirmation.

| Method | score | val CE | val KL | top1 target mass | max-bin accuracy |
|---|---|---:|---:|---:|---:|
| full_bc | normalise_kernel | 1.2383 | 0.6891 | 0.5587 | 0.6519 |
| full_bc | softmax_kernel | 1.4206 | 0.8714 | 0.5587 | 0.6519 |
| fixed_key_cov | normalise_kernel | 1.2717 | 0.7225 | 0.5702 | 0.6621 |
| fixed_key_cov | softmax_kernel | 1.5001 | 0.9509 | 0.5702 | 0.6621 |
| isotropic_same_area | normalise_kernel | 1.2414 | 0.6922 | 0.5575 | 0.6462 |
| isotropic_same_area | softmax_kernel | 1.4673 | 0.9181 | 0.5575 | 0.6462 |
| isotropic_same_trace | normalise_kernel | 1.3101 | 0.7609 | 0.5463 | 0.6314 |
| isotropic_same_trace | softmax_kernel | 1.4662 | 0.9170 | 0.5463 | 0.6314 |
| projected_query_bc | normalise_kernel | 1.1992 | 0.6500 | 0.5911 | 0.6974 |
| projected_query_bc | softmax_kernel | 1.3295 | 0.7802 | 0.5911 | 0.6974 |
| world_full_bc | normalise_kernel | 1.2108 | 0.6616 | 0.5764 | 0.6735 |
| world_full_bc | softmax_kernel | 1.5299 | 0.9806 | 0.5764 | 0.6735 |
| projected_query_isotropic_same_area | normalise_kernel | 1.2073 | 0.6581 | 0.5865 | 0.6871 |
| projected_query_isotropic_same_area | softmax_kernel | 1.4242 | 0.8750 | 0.5865 | 0.6871 |
| world_isotropic_same_volume | normalise_kernel | 1.6812 | 1.1320 | 0.5551 | 0.6542 |
| world_isotropic_same_volume | softmax_kernel | 1.8550 | 1.3057 | 0.5551 | 0.6542 |
| world_isotropic_same_trace | normalise_kernel | 1.6488 | 1.0996 | 0.5161 | 0.6098 |
| world_isotropic_same_trace | softmax_kernel | 1.9017 | 1.3525 | 0.5161 | 0.6098 |
| world_plane_isotropic | normalise_kernel | 1.3580 | 0.8087 | 0.5618 | 0.6576 |
| world_plane_isotropic | softmax_kernel | 1.6156 | 1.0664 | 0.5618 | 0.6576 |
| pixel_center_rbf | normalise_kernel | 1.2335 | 0.6843 | 0.5702 | 0.6621 |
| pixel_center_rbf | softmax_kernel | 1.1803 | 0.6310 | 0.5702 | 0.6621 |
| projected_center_rbf | normalise_kernel | 1.2353 | 0.6861 | 0.5904 | 0.6906 |
| projected_center_rbf | softmax_kernel | 1.1619 | 0.6127 | 0.5904 | 0.6906 |
| world_center_rbf | normalise_kernel | 3.3055 | 2.7563 | 0.5600 | 0.6587 |
| world_center_rbf | softmax_kernel | 1.7269 | 1.1777 | 0.5580 | 0.6576 |
| covshuffle_mean | normalise_kernel | 1.3404 | 0.7911 | 0.5392 | 0.6255 |
| covshuffle_mean | softmax_kernel | 1.5745 | 1.0253 | 0.5392 | 0.6255 |

All methods use identical row masks. Invalid key candidates are masked, retained label mass is recorded and targets are renormalized. `summary.json` records every camera's paired differences and cluster confidence intervals; `per_row.csv` preserves individual observations. Covariance shuffles average five metrics, not predictions.

Image means/covariances are normalized consistently by patch width 24 / 24^2. Fixed key covariance is 24^2/12 I before normalization. Same-area isotropic covariance preserves sqrt(det C), leaving scalar scale while removing direction/anisotropy. Query support is the fixed known image region unless the method explicitly says projected/world. No RGB training is performed.
