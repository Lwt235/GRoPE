# R67: late introduction of Gaussian shape

Matched terminal120 distance parents; two roles each get the same20 additional epochs. Used full60-class development data. This does not replace the original R66 result.

| Method | Base mCA seeds11/22/33 % | Mean % | CE | Class-average CE | Selected added epochs |
|---|---|---:|---:|---:|---|
| distance_continue | 79.203 / 77.894 / 78.643 | 78.580 | 0.73785 | 0.89155 | [20, 5, 15] |
| shape_after_distance | 78.981 / 78.579 / 78.365 | 78.642 | 0.73778 | 0.89126 | [20, 5, 15] |

| Condition | Paired mCA changes pp | Mean pp | Paired-class95% pp | Lead |
|---|---|---:|---|---|
| base | -0.222 / +0.685 / -0.278 | +0.062 | [-0.241, +0.370] | False |
| sparse256 | +0.163 / -0.722 / +0.060 | -0.167 | [-0.500, +0.120] | False |
| dense1024 | -0.079 / +0.000 / +0.098 | +0.006 | [-0.119, +0.118] | False |
| jitter02 | +0.064 / +0.343 / +0.083 | +0.164 | [-0.105, +0.469] | False |
| tilt | +0.139 / +0.889 / -0.306 | +0.241 | [-0.083, +0.620] | False |
| permuted | -0.222 / +0.685 / -0.278 | +0.062 | [-0.241, +0.370] | False |

All classes, sources and secondary scores are retained in result.json. Bootstrap is over classes after seed averaging, not a seed-population interval. Source categories are disjoint. Stage0 selection means no accepted adaptation. Scientific and actual visual review pending.

![Comparison](comparison.png)
