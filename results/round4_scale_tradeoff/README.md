# Covariance-filter scale tradeoff: all Gaussian controls

post-hoc all-scale fairness extension on previously evaluated real pairs

All four physical Gaussian scales are reported, with the same frozen model seeds, visual inputs, query/key means, labels and candidate masks. No evaluation result selects a preferred Gaussian scale. Both covariance inputs are scaled before the unchanged resolution floor. This is a controlled corruption, not a measured real estimator-error distribution.

| Visual input | Input covariance multiplier | Gaussian×0.25 CE | Gaussian×0.5 CE | Gaussian×1 CE | Gaussian×2 CE | Resolvent CE |
|---|---:|---:|---:|---:|---:|---:|
| raw | 0.25 | 1.236407 | 1.230335 | 1.225823 | 1.218895 | 1.225401 |
| raw | 1.0 | 1.217268 | 1.206922 | 1.198967 | 1.192738 | 1.200583 |
| raw | 4.0 | 1.203422 | 1.215386 | 1.256516 | 1.346753 | 1.205759 |
| raw | 16.0 | 1.267071 | 1.377109 | 1.560841 | 1.809534 | 1.318136 |
| frozen | 0.25 | 1.228473 | 1.218082 | 1.207048 | 1.207465 | 1.212793 |
| frozen | 1.0 | 1.208651 | 1.190805 | 1.172637 | 1.168928 | 1.183657 |
| frozen | 4.0 | 1.190525 | 1.189980 | 1.217028 | 1.296663 | 1.180609 |
| frozen | 16.0 | 1.249651 | 1.335736 | 1.478966 | 1.682266 | 1.279507 |

summary.json contains CE/JS/Top1 target mass means, all resolvent-minus-Gaussian paired seed/query-camera confidence intervals, clean-to-stress changes and every scene/seed value. Negative CE differences favor Resolvent. Prior completed stress results and their source hashes are preserved; overlapping means replay exactly. Tradeoff interpretations must consider clean performance and all stress factors rather than selecting whichever control loses on one condition.
