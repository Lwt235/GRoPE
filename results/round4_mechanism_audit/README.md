# Frozen normalized Gaussian PE: covariance mechanism audit

Descriptive follow-up on previously evaluated camera pairs, not new confirmation. Ten frozen normalized_shape checkpoints (two visual inputs, five seeds); no training.

All means/RGB/labels/masks are fixed. Covariances are changed before the existing .25² I blur. Temperature multipliers use old Teatime validation only. The separate-calibration column gives each intervention its own scalar; it cannot change candidate rankings.

| Input | Intervention | Frozen CE | Shared-temp CE | Separate-temp CE | Frozen top1 target mass |
|---|---|---:|---:|---:|---:|
| raw | original | 1.198967 | 1.196604 | 1.196604 | 0.603211 |
| raw | query_fixed | 1.198037 | 1.195485 | 1.195485 | 0.602599 |
| raw | key_fixed | 1.251084 | 1.240875 | 1.240442 | 0.596865 |
| raw | both_fixed | 1.252508 | 1.241455 | 1.240747 | 0.597182 |
| raw | query_trace | 1.198181 | 1.195917 | 1.195917 | 0.603885 |
| raw | key_trace | 1.228896 | 1.224278 | 1.224278 | 0.598423 |
| raw | both_trace | 1.228419 | 1.223795 | 1.223795 | 0.598881 |
| raw | query_area | 1.197639 | 1.195095 | 1.195095 | 0.604326 |
| raw | key_area | 1.226198 | 1.220000 | 1.220079 | 0.599188 |
| raw | both_area | 1.225597 | 1.219067 | 1.219087 | 0.599409 |
| raw | query_shuffled | 1.199189 | 1.194683 | 1.194683 | 0.604201 |
| raw | key_shuffled | 1.249417 | 1.238904 | 1.238029 | 0.596888 |
| raw | both_shuffled | 1.255426 | 1.241053 | 1.239236 | 0.598606 |
| frozen | original | 1.172637 | 1.164815 | 1.164815 | 0.609037 |
| frozen | query_fixed | 1.171555 | 1.163750 | 1.163750 | 0.609088 |
| frozen | key_fixed | 1.223843 | 1.206785 | 1.206785 | 0.606099 |
| frozen | both_fixed | 1.226678 | 1.208598 | 1.208598 | 0.605990 |
| frozen | query_trace | 1.171644 | 1.163981 | 1.163981 | 0.609337 |
| frozen | key_trace | 1.206429 | 1.195486 | 1.195486 | 0.603820 |
| frozen | both_trace | 1.205924 | 1.194986 | 1.194986 | 0.604465 |
| frozen | query_area | 1.172211 | 1.164093 | 1.164093 | 0.608956 |
| frozen | key_area | 1.202231 | 1.189602 | 1.189602 | 0.607479 |
| frozen | both_area | 1.202864 | 1.189710 | 1.189710 | 0.608192 |
| frozen | query_shuffled | 1.177476 | 1.166784 | 1.166784 | 0.608359 |
| frozen | key_shuffled | 1.219352 | 1.201930 | 1.201930 | 0.605629 |
| frozen | both_shuffled | 1.232450 | 1.209746 | 1.209746 | 0.604245 |

Interpretation must distinguish geometric information from matching of learned spectral filters. Removing only key shape can misalign its frequency allocation with the still-fitted query. If both-side replacement restores performance, the key-only ablation overstates a uniquely key-shape interpretation. Conversely, frozen intervention degradation alone cannot prove shape necessity: the content encoder was trained with full covariance, and retrained controls are required to separate information loss from input-distribution shift.

See summary.json for per-scene/seed metrics, selected validation temperatures, paired camera/seed intervals, exact data/implementation/checkpoint hashes, and both-minus-key-only contrasts. Inference fixes scenes and resamples query cameras; it is not a population-of-scenes confidence claim.
