# Real covariance scaling stress

post-hoc frozen input stress, not prospective confirmation or a realistic prediction-error guarantee

Both covariance inputs are multiplied before the unchanged blur floor. No retraining, temperature fitting, test-scale choice, or mean perturbation. These corruptions are controlled probes and do not model the full error of predicted geometry.

| Visual input | Physical covariance multiplier | Method | CE | JS | Top1 target mass |
|---|---:|---|---:|---:|---:|
| raw | 0.25 | gaussian_full | 1.225823 | 0.156485 | 0.602469 |
| raw | 0.25 | resolvent_full | 1.225401 | 0.159177 | 0.600956 |
| raw | 0.25 | gaussian200_full | 1.218895 | 0.151661 | 0.605621 |
| raw | 1.0 | gaussian_full | 1.198967 | 0.160006 | 0.603211 |
| raw | 1.0 | resolvent_full | 1.200583 | 0.160792 | 0.602808 |
| raw | 1.0 | gaussian200_full | 1.192738 | 0.160798 | 0.601023 |
| raw | 4.0 | gaussian_full | 1.256516 | 0.181793 | 0.586322 |
| raw | 4.0 | resolvent_full | 1.205759 | 0.171217 | 0.601596 |
| raw | 4.0 | gaussian200_full | 1.346753 | 0.201709 | 0.575743 |
| raw | 16.0 | gaussian_full | 1.560841 | 0.242559 | 0.541161 |
| raw | 16.0 | resolvent_full | 1.318136 | 0.201927 | 0.580129 |
| raw | 16.0 | gaussian200_full | 1.809534 | 0.281668 | 0.497771 |
| frozen | 0.25 | gaussian_full | 1.207048 | 0.147547 | 0.608456 |
| frozen | 0.25 | resolvent_full | 1.212793 | 0.148549 | 0.609237 |
| frozen | 0.25 | gaussian200_full | 1.207465 | 0.145081 | 0.609436 |
| frozen | 1.0 | gaussian_full | 1.172637 | 0.149540 | 0.609037 |
| frozen | 1.0 | resolvent_full | 1.183657 | 0.149035 | 0.610057 |
| frozen | 1.0 | gaussian200_full | 1.168928 | 0.150998 | 0.605579 |
| frozen | 4.0 | gaussian_full | 1.217028 | 0.167784 | 0.598101 |
| frozen | 4.0 | resolvent_full | 1.180609 | 0.156529 | 0.606429 |
| frozen | 4.0 | gaussian200_full | 1.296663 | 0.184742 | 0.586304 |
| frozen | 16.0 | gaussian_full | 1.478966 | 0.219169 | 0.562554 |
| frozen | 16.0 | resolvent_full | 1.279507 | 0.180942 | 0.593682 |
| frozen | 16.0 | gaussian200_full | 1.682266 | 0.251932 | 0.529162 |

See summary.json for paired camera/seed differences and all per-scene/seed values. The validation-selected Gaussian is fixed across stress factors. This descriptive probe alone cannot establish general robustness or a new deployment improvement.
