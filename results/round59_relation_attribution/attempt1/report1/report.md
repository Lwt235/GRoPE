# R59 learned-relation attribution results

12 new fits;18 EXACT R58 references;30 selected models;120 main condition groups and30 base-reframe diagnostics.
The candidate was chosen after R58 development results. Reference fits/predictions are reused, not new independent evidence.
All data remain previously used official-training ModelNet development data. Full research goal is unachieved.
Scientific interpretation and visual review remain pending when this report is generated.

## base

| Method | Region R@1 % | Center PCK % | Soft EPE | CE |
| --- | ---: | ---: | ---: | ---: |
| Gaussian invariant MLP (candidate) | 38.499 | 22.504 | 0.316488 | 3.352861 |
| Distance MLP | 35.078 | 19.996 | 0.330808 | 3.443439 |
| Local descriptor MLP | 37.108 | 21.437 | 0.323537 | 3.387721 |
| Angle + descriptor MLP | 37.247 | 21.358 | 0.323636 | 3.380668 |
| Full log-matrix MLP | 34.278 | 19.368 | 0.332969 | 3.464639 |
| R58 scalar Gaussian | 34.458 | 19.374 | 0.331958 | 3.465510 |
| R58 Gaussian metric | 34.274 | 19.221 | 0.331812 | 3.470909 |
| R58 sinusoidal distance | 35.135 | 19.970 | 0.330620 | 3.450592 |
| R58 distance + angle | 34.951 | 19.808 | 0.332029 | 3.455356 |
| No explicit PE | 17.683 | 8.789 | 0.416250 | 3.940711 |

| Candidate minus control | Region paired pp | Mean region pp [object-conditional95%] | Center paired pp | Soft EPE paired |
| --- | --- | --- | --- | --- |
| Distance MLP | +3.6505/+3.9361/+2.6763 | +3.4210 [+2.9103,+3.9569] | +2.5157/+2.7592/+2.2492 | -0.0153/-0.0152/-0.0125 |
| Local descriptor MLP | +1.4740/+2.0528/+0.6452 | +1.3907 [+0.8698,+1.9444] | +1.2853/+1.3222/+0.5929 | -0.0074/-0.0068/-0.0069 |
| Angle + descriptor MLP | +2.7669/+0.5853/+0.4042 | +1.2521 [+0.7207,+1.8126] | +2.1931/+0.5534/+0.6924 | -0.0134/-0.0049/-0.0032 |
| Full log-matrix MLP | +5.1091/+3.6862/+3.8672 | +4.2209 [+3.6427,+4.7973] | +3.5498/+2.5884/+3.2705 | -0.0231/-0.0113/-0.0151 |

Eligible counts [39245, 39183]; fraction 0.634942. Compare methods within the same condition; population changes across sampling conditions.

## noise02

| Method | Region R@1 % | Center PCK % | Soft EPE | CE |
| --- | ---: | ---: | ---: | ---: |
| Gaussian invariant MLP (candidate) | 36.546 | 19.813 | 0.328397 | 3.425747 |
| Distance MLP | 33.104 | 17.160 | 0.342193 | 3.517932 |
| Local descriptor MLP | 35.588 | 18.993 | 0.332992 | 3.448550 |
| Angle + descriptor MLP | 35.416 | 18.858 | 0.333762 | 3.444383 |
| Full log-matrix MLP | 32.250 | 16.380 | 0.346041 | 3.530054 |
| R58 scalar Gaussian | 32.343 | 16.643 | 0.344109 | 3.545268 |
| R58 Gaussian metric | 32.178 | 16.508 | 0.343282 | 3.547612 |
| R58 sinusoidal distance | 33.023 | 17.146 | 0.342582 | 3.525380 |
| R58 distance + angle | 32.967 | 17.061 | 0.343603 | 3.525427 |
| No explicit PE | 16.018 | 7.291 | 0.425815 | 3.999441 |

| Candidate minus control | Region paired pp | Mean region pp [object-conditional95%] | Center paired pp | Soft EPE paired |
| --- | --- | --- | --- | --- |
| Distance MLP | +4.0267/+4.3442/+1.9575 | +3.4428 [+2.8572,+4.0496] | +2.9111/+3.1013/+1.9459 | -0.0162/-0.0157/-0.0095 |
| Local descriptor MLP | +2.1310/+1.3392/-0.5951 | +0.9584 [+0.3739,+1.5526] | +1.6323/+0.8110/+0.0154 | -0.0095/-0.0053/+0.0009 |
| Angle + descriptor MLP | +2.8032/+0.6118/-0.0244 | +1.1302 [+0.6099,+1.6564] | +2.1502/+0.3830/+0.3316 | -0.0137/-0.0041/+0.0017 |
| Full log-matrix MLP | +5.6757/+4.1424/+3.0705 | +4.2962 [+3.6534,+4.9235] | +4.0666/+3.2556/+2.9754 | -0.0266/-0.0161/-0.0102 |

Eligible counts [38897, 38908]; fraction 0.629898. Compare methods within the same condition; population changes across sampling conditions.

## sparse256

| Method | Region R@1 % | Center PCK % | Soft EPE | CE |
| --- | ---: | ---: | ---: | ---: |
| Gaussian invariant MLP (candidate) | 39.140 | 22.538 | 0.294675 | 3.714421 |
| Distance MLP | 37.309 | 21.262 | 0.301270 | 3.757858 |
| Local descriptor MLP | 38.774 | 22.631 | 0.297048 | 3.753251 |
| Angle + descriptor MLP | 38.941 | 22.550 | 0.296454 | 3.732117 |
| Full log-matrix MLP | 37.300 | 21.068 | 0.303155 | 3.737693 |
| R58 scalar Gaussian | 35.065 | 19.211 | 0.307840 | 3.805903 |
| R58 Gaussian metric | 34.936 | 19.240 | 0.308064 | 3.816034 |
| R58 sinusoidal distance | 37.226 | 21.266 | 0.299212 | 3.783111 |
| R58 distance + angle | 37.605 | 21.372 | 0.299600 | 3.788902 |
| No explicit PE | 23.507 | 12.062 | 0.368418 | 4.021318 |

| Candidate minus control | Region paired pp | Mean region pp [object-conditional95%] | Center paired pp | Soft EPE paired |
| --- | --- | --- | --- | --- |
| Distance MLP | +2.7579/+2.0334/+0.6997 | +1.8303 [+1.0408,+2.5844] | +2.0866/+1.1885/+0.5526 | -0.0105/-0.0047/-0.0046 |
| Local descriptor MLP | +1.0061/+1.2204/-1.1283 | +0.3661 [-0.3194,+1.0862] | +0.4056/+0.4623/-1.1478 | -0.0062/-0.0032/+0.0023 |
| Angle + descriptor MLP | +1.6083/-0.0301/-0.9831 | +0.1984 [-0.4313,+0.8398] | +1.1531/-0.4180/-0.7705 | -0.0087/+0.0010/+0.0024 |
| Full log-matrix MLP | +2.2584/+2.1238/+1.1354 | +1.8392 [+1.1408,+2.5309] | +1.4507/+1.8386/+1.1212 | -0.0124/-0.0095/-0.0035 |

Eligible counts [28250, 28206]; fraction 0.457060. Compare methods within the same condition; population changes across sampling conditions.

## low_overlap

| Method | Region R@1 % | Center PCK % | Soft EPE | CE |
| --- | ---: | ---: | ---: | ---: |
| Gaussian invariant MLP (candidate) | 35.692 | 19.957 | 0.315678 | 3.588736 |
| Distance MLP | 32.600 | 17.823 | 0.324232 | 3.642973 |
| Local descriptor MLP | 33.011 | 18.094 | 0.323310 | 3.651759 |
| Angle + descriptor MLP | 33.318 | 18.213 | 0.323027 | 3.639564 |
| Full log-matrix MLP | 31.557 | 16.825 | 0.329580 | 3.680483 |
| R58 scalar Gaussian | 33.052 | 18.138 | 0.325583 | 3.643536 |
| R58 Gaussian metric | 33.295 | 18.393 | 0.323302 | 3.629579 |
| R58 sinusoidal distance | 32.688 | 18.063 | 0.325584 | 3.655510 |
| R58 distance + angle | 32.489 | 17.786 | 0.326371 | 3.657201 |
| No explicit PE | 18.961 | 9.408 | 0.394321 | 3.962435 |

| Candidate minus control | Region paired pp | Mean region pp [object-conditional95%] | Center paired pp | Soft EPE paired |
| --- | --- | --- | --- | --- |
| Distance MLP | +4.3203/+2.0153/+2.9396 | +3.0917 [+2.3921,+3.7746] | +2.9163/+1.2425/+2.2434 | -0.0112/-0.0051/-0.0094 |
| Local descriptor MLP | +3.6474/+2.4200/+1.9736 | +2.6803 [+2.0222,+3.3445] | +2.5449/+1.6988/+1.3441 | -0.0085/-0.0047/-0.0097 |
| Angle + descriptor MLP | +4.0705/+1.7404/+1.3107 | +2.3739 [+1.7608,+2.9895] | +2.9646/+1.2291/+1.0359 | -0.0124/-0.0050/-0.0046 |
| Full log-matrix MLP | +5.2563/+3.2444/+3.9023 | +4.1343 [+3.3607,+4.9139] | +3.9572/+2.2551/+3.1811 | -0.0183/-0.0102/-0.0132 |

Eligible counts [30131, 29911]; fraction 0.486091. Compare methods within the same condition; population changes across sampling conditions.

## Fixed frame-change diagnostic

Same physical base pairs and labels. Rotate view0 Z90,view1 X90,translate both(.7,-.3,.2),transform covariance;keep invariant local features.
All30 models were actually evaluated. Matrix MLP is not invariant by construction; its sensitivity is model behavior,not a failed implementation.

| Method | Region change pp | Center change pp | CE change | Max row-probability change | Changed A-to-B winners (all queries,3seeds) |
| --- | ---: | ---: | ---: | ---: | ---: |
| R58 scalar Gaussian | +0.0000 | +0.0000 | -0.000000 | 1.35601e-06 | 407 |
| R58 Gaussian metric | +0.0000 | +0.0000 | -0.000000 | 1.46031e-06 | 413 |
| R58 sinusoidal distance | +0.0000 | +0.0000 | -0.000000 | 1.35601e-06 | 481 |
| R58 distance + angle | +0.0017 | -0.0047 | +0.000040 | 0.0569094 | 575 |
| Gaussian invariant MLP (candidate) | +0.0000 | +0.0000 | +0.000000 | 1.3411e-06 | 446 |
| No explicit PE | +0.0000 | +0.0000 | +0.000000 | 0 | 0 |
| Distance MLP | +0.0000 | +0.0000 | -0.000000 | 9.83477e-07 | 465 |
| Local descriptor MLP | +0.0000 | +0.0000 | -0.000000 | 1.04308e-06 | 453 |
| Angle + descriptor MLP | -0.0064 | -0.0034 | +0.000057 | 0.113347 | 704 |
| Full log-matrix MLP | +0.0429 | -0.1122 | +0.001096 | 0.199961 | 62306 |

The probability/winner diagnostic is A-to-B and includes ineligible queries; task metrics remain bidirectional on eligible queries.
One fixed reframing does not estimate an entire distribution of pose robustness. Mathematical invariance and measured task utility are distinct.

## Fitting and checks

| Method | PE parameters | Selected epochs | Fit seconds | Reused fit |
| --- | ---: | --- | ---: | --- |
| R58 scalar Gaussian | 9216 | 45/39/33 | 1454.40 | True |
| R58 Gaussian metric | 9216 | 45/36/39 | 1426.61 | True |
| R58 sinusoidal distance | 9216 | 42/39/39 | 1393.30 | True |
| R58 distance + angle | 18432 | 39/45/33 | 2933.21 | True |
| Gaussian invariant MLP (candidate) | 10368 | 45/57/45 | 1501.00 | True |
| No explicit PE | 0 | 48/54/51 | 1369.48 | True |
| Distance MLP | 9408 | 39/39/42 | 1430.87 | False |
| Local descriptor MLP | 11520 | 45/48/45 | 1430.61 | False |
| Angle + descriptor MLP | 11616 | 39/45/45 | 3248.40 | False |
| Full log-matrix MLP | 10752 | 39/45/48 | 1481.00 | False |

NEW fitting/selection cumulative time 7590.89s. Shared-host timing is not a controlled speed benchmark.
Independent metric maximum error 1.33e-15. Source/cache/model/prediction identities and actual sampling/init pairing checked.
Independent arithmetic is a separate NumPy path by the same author,not a second-lab replication.
Intervals are object bootstraps conditional on these three fitted models,not training-seed uncertainty or fresh confirmation.

## Decision pending scientific review

Evaluate directional-shape attribution against ALL fixed controls and conditions. Do not reinterpret R58 scalar-Gaussian failure as a success.
The complete goal still requires appropriate cross-task,real-scene,generalization and untouched confirmation evidence.

![Comparisons](/home/liwt/GRoPE/results/round59_relation_attribution/attempt1/report1/comparison.png)

![Learning curves](/home/liwt/GRoPE/results/round59_relation_attribution/attempt1/report1/learning_curves.png)
