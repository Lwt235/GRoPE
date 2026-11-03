# R62: RGB to Gaussian relations to PE

18 new fits;12 exact R55 reference models.36 new+24 reused predictions;60 independent metric recomputations.
RGB features remain frozen;parameter head,PE and matching network are trained jointly.
All new selections precede new evaluation. Old development data;no fresh confirmation.
Observed full/center are PRIVILEGED diagnostics,not fair RGB competitors.
Relation placement is calibrated cross-attention;the original R58 placement was within-view self-attention.
Scientific interpretation and actual image review pending at generation.

## baseline

| Method | Soft EPE px | PCK8 % | CE | Soft >16px % |
| --- | ---: | ---: | ---: | ---: |
| PRoPE reference | 3.86360 | 87.336 | 1.83359 | 2.803 |
| Calibrated plane | 3.87887 | 87.337 | 1.81385 | 2.832 |
| Ray point | 3.70651 | 88.062 | 1.81465 | 2.777 |
| Ray affine | 3.66693 | 88.312 | 1.79342 | 2.717 |
| Camera + MLP phase | 3.92554 | 87.070 | 1.85893 | 3.033 |
| Direct learned relation | 3.86734 | 87.374 | 1.81853 | 2.800 |
| RGB Gaussian relations | 3.99521 | 86.935 | 1.86781 | 3.176 |
| RGB center only | 4.09188 | 86.452 | 1.86411 | 3.431 |

Privileged diagnostic groups:

| Method | Soft EPE px | PCK8 % | CE | Soft >16px % |
| --- | ---: | ---: | ---: | ---: |
| Observed Gaussian (privileged) | 3.12407 | 91.230 | 1.65266 | 1.455 |
| Observed center (privileged) | 3.38778 | 90.140 | 1.70447 | 2.127 |

## rgb_permuted

| Method | Soft EPE px | PCK8 % | CE | Soft >16px % |
| --- | ---: | ---: | ---: | ---: |
| PRoPE reference | 5.58606 | 75.605 | 2.44540 | 5.427 |
| Calibrated plane | 5.97097 | 72.700 | 2.52505 | 5.816 |
| Ray point | 5.48755 | 74.399 | 2.47529 | 5.056 |
| Ray affine | 5.45368 | 75.149 | 2.43722 | 4.947 |
| Camera + MLP phase | 5.62606 | 75.972 | 2.46366 | 5.553 |
| Direct learned relation | 5.72967 | 74.530 | 2.47651 | 5.586 |
| RGB Gaussian relations | 5.42612 | 77.953 | 2.36107 | 5.251 |
| RGB center only | 5.73226 | 76.497 | 2.39424 | 5.690 |

Privileged diagnostic groups:

| Method | Soft EPE px | PCK8 % | CE | Soft >16px % |
| --- | ---: | ---: | ---: | ---: |
| Observed Gaussian (privileged) | 4.10927 | 84.962 | 1.97437 | 2.630 |
| Observed center (privileged) | 4.63735 | 82.292 | 2.11936 | 3.521 |

## Fixed paired contrasts

EPE deltas are candidate minus control;negative is favorable.

Observed-input contrasts use extra depth information and are diagnostic, not fair RGB comparisons.

| Comparison | EPE mean px | EPE seeds11/22/33 | Favorable seeds/scenes | PCK8 mean pp |
| --- | ---: | --- | --- | ---: |
| RGB Gaussian relations minus PRoPE reference | +0.13161 | +0.04225/+0.16684/+0.18575 | 0/3;2/5 | -0.401 |
| RGB Gaussian relations minus Calibrated plane | +0.11634 | +0.07567/+0.13684/+0.13652 | 0/3;2/5 | -0.402 |
| RGB Gaussian relations minus Ray point | +0.28871 | +0.23710/+0.39126/+0.23777 | 0/3;0/5 | -1.127 |
| RGB Gaussian relations minus Ray affine | +0.32829 | +0.29152/+0.42262/+0.27072 | 0/3;0/5 | -1.377 |
| RGB Gaussian relations minus Camera + MLP phase | +0.06968 | +0.02703/+0.02543/+0.15658 | 0/3;2/5 | -0.135 |
| RGB Gaussian relations minus Direct learned relation | +0.12787 | -0.04376/+0.24258/+0.18480 | 1/3;2/5 | -0.439 |
| RGB Gaussian relations minus RGB center only | -0.09667 | -0.13454/-0.04077/-0.11469 | 3/3;5/5 | +0.483 |
| Observed Gaussian (privileged) minus Observed center (privileged) | -0.26372 | -0.08951/-0.44275/-0.25889 | 3/3;5/5 | +1.089 |
| Observed Gaussian (privileged) minus RGB Gaussian relations | -0.87115 | -0.77546/-0.90419/-0.93379 | 3/3;5/5 | +4.295 |
| Observed Gaussian (privileged) minus PRoPE reference | -0.73953 | -0.73321/-0.73736/-0.74804 | 3/3;5/5 | +3.894 |
| Observed center (privileged) minus PRoPE reference | -0.47582 | -0.64370/-0.29461/-0.48915 | 3/3;5/5 | +2.805 |
| Observed Gaussian (privileged) minus Direct learned relation | -0.74327 | -0.81921/-0.66162/-0.74899 | 3/3;5/5 | +3.856 |

## Frame gap

| Method | 15frame EPE | 60frame EPE |
| --- | ---: | ---: |
| PRoPE reference | 2.46311 | 6.66836 |
| Calibrated plane | 2.49410 | 6.66487 |
| Ray point | 2.36165 | 6.35709 |
| Ray affine | 2.33141 | 6.30378 |
| Camera + MLP phase | 2.46200 | 6.89200 |
| Direct learned relation | 2.46914 | 6.66513 |
| RGB Gaussian relations | 2.52095 | 6.97174 |
| RGB center only | 2.55574 | 7.18440 |
| Observed Gaussian (privileged) | 2.06176 | 5.26754 |
| Observed center (privileged) | 2.19957 | 5.80806 |

Frame spacing is not an isolated zoom/surface-tilt intervention.

## Observed-input coverage

Counts below are unique-frame tokens,not the pair-query evaluation population.

| Split | Frames | Tokens | Missing center depth | Median fallback | Entire patch missing |
| --- | ---: | ---: | ---: | ---: | ---: |
| fit | 790 | 341280 | 61805 | 30754 | 31051 |
| val | 96 | 41472 | 5485 | 2978 | 2507 |
| old_eval | 719 | 310608 | 35182 | 20411 | 14771 |

## New fits

| Method | PE params | Selected steps | Fit/selection cumulative seconds |
| --- | ---: | --- | ---: |
| Camera + MLP phase | 1586 | 3700/5000/3400 | 958.28 |
| Direct learned relation | 11719 | 3700/4400/3400 | 2259.17 |
| RGB Gaussian relations | 11047 | 4100/5000/3500 | 4256.25 |
| RGB center only | 10465 | 3700/3400/2800 | 3714.34 |
| Observed Gaussian (privileged) | 10368 | 3900/5000/2400 | 2639.75 |
| Observed center (privileged) | 10368 | 4300/2900/2600 | 2610.96 |

Independent metric max difference:1.98e-09.
Selection CE max difference:3.99e-08;CPU replay max:4.98e-05.
Shared-host timing is not a controlled speed benchmark. No global goal completion.

![Paired EPE](/home/liwt/GRoPE/results/round62_rgb_relations/attempt1/report1/comparison.png)

![Validation curves](/home/liwt/GRoPE/results/round62_rgb_relations/attempt1/report1/learning_curves.png)
