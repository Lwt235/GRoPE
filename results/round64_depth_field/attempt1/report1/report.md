# R64: depth-field Gaussian relations in RGB matching

9 new fits;24 exact R55/R62 RGB reference models.18 new+48 reused predictions;66 independent metric recomputations.
RGB features remain frozen;parameter head,PE and matching network are trained jointly.
All nine new selections precede new evaluation. Old development data;no fresh confirmation.
Primary:depth_surface. Depth_footprint isolates the derivative term;depth_local_mlp receives the same local field information.
Gaussian coupling and reachable range both change from R62;no uniquely isolated degrees-of-freedom claim.
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
| Depth surface Gaussian | 3.94646 | 87.325 | 1.80895 | 2.970 |
| Depth footprint Gaussian | 3.96773 | 86.891 | 1.86893 | 3.063 |
| Same-information local MLP | 3.91565 | 87.102 | 1.86994 | 2.919 |

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
| Depth surface Gaussian | 6.00109 | 75.909 | 2.40108 | 6.204 |
| Depth footprint Gaussian | 5.35487 | 78.185 | 2.36432 | 5.225 |
| Same-information local MLP | 5.64301 | 75.579 | 2.50042 | 5.502 |

## Fixed paired contrasts

EPE deltas are candidate minus control;negative is favorable.

All listed methods use RGB/cameras only; reused references do not count as independent new evidence.

| Comparison | EPE mean px | EPE seeds11/22/33 | Favorable seeds/scenes | PCK8 mean pp |
| --- | ---: | --- | --- | ---: |
| Depth surface Gaussian minus PRoPE reference | +0.08286 | +0.06627/+0.00835/+0.17396 | 0/3;1/5 | -0.011 |
| Depth surface Gaussian minus Calibrated plane | +0.06759 | +0.09969/-0.02164/+0.12472 | 1/3;2/5 | -0.013 |
| Depth surface Gaussian minus Ray point | +0.23995 | +0.26111/+0.23277/+0.22597 | 0/3;0/5 | -0.737 |
| Depth surface Gaussian minus Ray affine | +0.27953 | +0.31554/+0.26414/+0.25892 | 0/3;0/5 | -0.987 |
| Depth surface Gaussian minus Camera + MLP phase | +0.02093 | +0.05104/-0.13305/+0.14478 | 1/3;2/5 | +0.254 |
| Depth surface Gaussian minus Direct learned relation | +0.07912 | -0.01974/+0.08409/+0.17301 | 1/3;2/5 | -0.049 |
| Depth surface Gaussian minus RGB Gaussian relations | -0.04875 | +0.02402/-0.15848/-0.01180 | 2/3;3/5 | +0.390 |
| Depth surface Gaussian minus RGB center only | -0.14542 | -0.11052/-0.19925/-0.12649 | 3/3;4/5 | +0.873 |
| Depth surface Gaussian minus Depth footprint Gaussian | -0.02127 | -0.01086/-0.09145/+0.03849 | 2/3;3/5 | +0.434 |
| Depth surface Gaussian minus Same-information local MLP | +0.03081 | +0.09575/-0.02848/+0.02517 | 1/3;1/5 | +0.222 |
| Depth footprint Gaussian minus Same-information local MLP | +0.05208 | +0.10660/+0.06297/-0.01332 | 1/3;3/5 | -0.212 |
| Depth footprint Gaussian minus PRoPE reference | +0.10413 | +0.07712/+0.09980/+0.13547 | 0/3;2/5 | -0.445 |
| Depth footprint Gaussian minus Calibrated plane | +0.08886 | +0.11055/+0.06981/+0.08623 | 0/3;2/5 | -0.446 |
| Depth footprint Gaussian minus Ray point | +0.26123 | +0.27197/+0.32423/+0.18748 | 0/3;0/5 | -1.171 |
| Depth footprint Gaussian minus Ray affine | +0.30081 | +0.32640/+0.35559/+0.22043 | 0/3;0/5 | -1.421 |
| Depth footprint Gaussian minus RGB center only | -0.12415 | -0.09967/-0.10780/-0.16498 | 3/3;4/5 | +0.439 |

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
| Depth surface Gaussian | 2.51319 | 6.81562 |
| Depth footprint Gaussian | 2.49847 | 6.91767 |
| Same-information local MLP | 2.49017 | 6.76310 |

Frame spacing is not an isolated zoom/surface-tilt intervention.


## New fits

| Method | PE params | Selected steps | Fit/selection cumulative seconds |
| --- | ---: | --- | ---: |
| Depth surface Gaussian | 10465 | 3300/4000/2400 | 4863.89 |
| Depth footprint Gaussian | 10465 | 4700/4000/3900 | 4444.74 |
| Same-information local MLP | 10369 | 3700/4400/4800 | 2624.92 |

Independent metric max difference:2.12e-09.
Selection CE max difference:4.78e-08;CPU replay max:5.86e-05.
Shared-host timing is not a controlled speed benchmark. No global goal completion.

![Paired EPE](/home/liwt/GRoPE/results/round64_depth_field/attempt1/report1/comparison.png)

![Validation curves](/home/liwt/GRoPE/results/round64_depth_field/attempt1/report1/learning_curves.png)
