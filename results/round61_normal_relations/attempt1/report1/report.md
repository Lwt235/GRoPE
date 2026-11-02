# R61: normal/spectrum attribution

6 NEW synthetic fits + 6 NEW real adaptations. 15 source and 15 target reference models reused exactly.
48 NEW independently recomputed prediction groups;120 old groups reused by identity. Total168.
Both new controls use the same covariance inputs,backbone,actual data stream and fixed optimization as references.
Target roles start from their OWN source models;this is a complete source-plus-target pipeline comparison.
Synthetic primary: original query-weighted region recall. Real primary: scene-macro region recall.
Development data only. Sparse populations change. No fresh confirmation or general-effectiveness claim.
Normal PPF control is not a complete RoITr reproduction. Scientific/visual review pending at generation.

## synthetic / base

| Method | Region R@1 % | Center PCK % | Soft EPE | CE |
| --- | ---: | ---: | ---: | ---: |
| Full Gaussian relations (candidate) | 38.499 | 22.504 | 0.316488 | 3.352861 |
| Normal PPF + spectra | 38.046 | 21.993 | 0.319263 | 3.352896 |
| Axial relations + original spectra | 38.323 | 22.242 | 0.318436 | 3.349411 |
| Distance MLP | 35.078 | 19.996 | 0.330808 | 3.443439 |
| Local descriptor MLP | 37.108 | 21.437 | 0.323537 | 3.387721 |
| Angle + descriptor MLP | 37.247 | 21.358 | 0.323636 | 3.380668 |
| Full log-matrix MLP | 34.278 | 19.368 | 0.332969 | 3.464639 |

| Candidate minus control | Region mean pp | Region paired seeds pp | Center mean pp |
| --- | ---: | --- | ---: |
| Distance MLP | +3.421 | +3.650/+3.936/+2.676 | +2.508 |
| Local descriptor MLP | +1.391 | +1.474/+2.053/+0.645 | +1.067 |
| Angle + descriptor MLP | +1.252 | +2.767/+0.585/+0.404 | +1.146 |
| Full log-matrix MLP | +4.221 | +5.109/+3.686/+3.867 | +3.136 |
| Normal PPF + spectra | +0.453 | +0.852/+0.008/+0.499 | +0.511 |
| Axial relations + original spectra | +0.176 | +0.278/+0.760/-0.510 | +0.261 |

Eligible query fraction: 0.634942; valid counts: [39245, 39183].
Minimum-eigenspace multiplicities k1/k2/k3: [123520, 0, 0]; repeated fraction 0.

## synthetic / noise02

| Method | Region R@1 % | Center PCK % | Soft EPE | CE |
| --- | ---: | ---: | ---: | ---: |
| Full Gaussian relations (candidate) | 36.546 | 19.813 | 0.328397 | 3.425747 |
| Normal PPF + spectra | 36.289 | 19.339 | 0.330459 | 3.425875 |
| Axial relations + original spectra | 36.281 | 19.313 | 0.330363 | 3.421645 |
| Distance MLP | 33.104 | 17.160 | 0.342193 | 3.517932 |
| Local descriptor MLP | 35.588 | 18.993 | 0.332992 | 3.448550 |
| Angle + descriptor MLP | 35.416 | 18.858 | 0.333762 | 3.444383 |
| Full log-matrix MLP | 32.250 | 16.380 | 0.346041 | 3.530054 |

| Candidate minus control | Region mean pp | Region paired seeds pp | Center mean pp |
| --- | ---: | --- | ---: |
| Distance MLP | +3.443 | +4.027/+4.344/+1.957 | +2.653 |
| Local descriptor MLP | +0.958 | +2.131/+1.339/-0.595 | +0.820 |
| Angle + descriptor MLP | +1.130 | +2.803/+0.612/-0.024 | +0.955 |
| Full log-matrix MLP | +4.296 | +5.676/+4.142/+3.070 | +3.433 |
| Normal PPF + spectra | +0.258 | +0.942/+0.067/-0.235 | +0.473 |
| Axial relations + original spectra | +0.265 | +0.603/+0.861/-0.668 | +0.500 |

Eligible query fraction: 0.629898; valid counts: [38897, 38908].
Minimum-eigenspace multiplicities k1/k2/k3: [123520, 0, 0]; repeated fraction 0.

## synthetic / sparse256

| Method | Region R@1 % | Center PCK % | Soft EPE | CE |
| --- | ---: | ---: | ---: | ---: |
| Full Gaussian relations (candidate) | 39.140 | 22.538 | 0.294675 | 3.714421 |
| Normal PPF + spectra | 38.749 | 22.282 | 0.294489 | 3.750072 |
| Axial relations + original spectra | 39.558 | 23.004 | 0.294216 | 3.700750 |
| Distance MLP | 37.309 | 21.262 | 0.301270 | 3.757858 |
| Local descriptor MLP | 38.774 | 22.631 | 0.297048 | 3.753251 |
| Angle + descriptor MLP | 38.941 | 22.550 | 0.296454 | 3.732117 |
| Full log-matrix MLP | 37.300 | 21.068 | 0.303155 | 3.737693 |

| Candidate minus control | Region mean pp | Region paired seeds pp | Center mean pp |
| --- | ---: | --- | ---: |
| Distance MLP | +1.830 | +2.758/+2.033/+0.700 | +1.276 |
| Local descriptor MLP | +0.366 | +1.006/+1.220/-1.128 | -0.093 |
| Angle + descriptor MLP | +0.198 | +1.608/-0.030/-0.983 | -0.012 |
| Full log-matrix MLP | +1.839 | +2.258/+2.124/+1.135 | +1.470 |
| Normal PPF + spectra | +0.391 | +1.513/+0.468/-0.808 | +0.256 |
| Axial relations + original spectra | -0.419 | +0.106/+0.097/-1.460 | -0.466 |

Eligible query fraction: 0.457060; valid counts: [28250, 28206].
Minimum-eigenspace multiplicities k1/k2/k3: [123520, 0, 0]; repeated fraction 0.

## synthetic / low_overlap

| Method | Region R@1 % | Center PCK % | Soft EPE | CE |
| --- | ---: | ---: | ---: | ---: |
| Full Gaussian relations (candidate) | 35.692 | 19.957 | 0.315678 | 3.588736 |
| Normal PPF + spectra | 34.235 | 18.920 | 0.319412 | 3.613499 |
| Axial relations + original spectra | 34.745 | 19.281 | 0.316585 | 3.587147 |
| Distance MLP | 32.600 | 17.823 | 0.324232 | 3.642973 |
| Local descriptor MLP | 33.011 | 18.094 | 0.323310 | 3.651759 |
| Angle + descriptor MLP | 33.318 | 18.213 | 0.323027 | 3.639564 |
| Full log-matrix MLP | 31.557 | 16.825 | 0.329580 | 3.680483 |

| Candidate minus control | Region mean pp | Region paired seeds pp | Center mean pp |
| --- | ---: | --- | ---: |
| Distance MLP | +3.092 | +4.320/+2.015/+2.940 | +2.134 |
| Local descriptor MLP | +2.680 | +3.647/+2.420/+1.974 | +1.863 |
| Angle + descriptor MLP | +2.374 | +4.070/+1.740/+1.311 | +1.743 |
| Full log-matrix MLP | +4.134 | +5.256/+3.244/+3.902 | +3.131 |
| Normal PPF + spectra | +1.457 | +1.992/+0.911/+1.467 | +1.036 |
| Axial relations + original spectra | +0.947 | +1.572/+0.671/+0.598 | +0.676 |

Eligible query fraction: 0.486091; valid counts: [30131, 29911].
Minimum-eigenspace multiplicities k1/k2/k3: [123520, 0, 0]; repeated fraction 0.

## real_source / base

| Method | Region R@1 % | Center PCK % | Soft EPE | CE |
| --- | ---: | ---: | ---: | ---: |
| Full Gaussian relations (candidate) | 35.629 | 12.902 | 0.587169 | 3.330644 |
| Normal PPF + spectra | 35.825 | 12.989 | 0.588350 | 3.351445 |
| Axial relations + original spectra | 35.673 | 13.059 | 0.607121 | 3.371718 |
| Distance MLP | 36.742 | 13.452 | 0.535265 | 3.234470 |
| Local descriptor MLP | 33.384 | 11.936 | 0.564963 | 3.333618 |
| Angle + descriptor MLP | 33.511 | 11.867 | 0.574844 | 3.338078 |
| Full log-matrix MLP | 33.215 | 11.720 | 0.544288 | 3.300616 |

| Candidate minus control | Region mean pp | Region paired seeds pp | Center mean pp |
| --- | ---: | --- | ---: |
| Distance MLP | -1.113 | +3.144/-7.857/+1.372 | -0.550 |
| Local descriptor MLP | +2.245 | +3.802/-5.227/+8.160 | +0.966 |
| Angle + descriptor MLP | +2.118 | +4.714/-2.291/+3.930 | +1.036 |
| Full log-matrix MLP | +2.413 | +3.295/-2.801/+6.747 | +1.182 |
| Normal PPF + spectra | -0.196 | -1.536/-0.961/+1.909 | -0.087 |
| Axial relations + original spectra | -0.044 | -1.452/-1.588/+2.907 | -0.157 |

Eligible query fraction: 0.332389; valid counts: [10264, 10158].
Evaluable scenes: 5/5;scene-order/differences are retained in comparisons.json.
Minimum-eigenspace multiplicities k1/k2/k3: [61440, 0, 0]; repeated fraction 0.

## real_source / sparse256

| Method | Region R@1 % | Center PCK % | Soft EPE | CE |
| --- | ---: | ---: | ---: | ---: |
| Full Gaussian relations (candidate) | 40.852 | 15.247 | 0.509131 | 3.384141 |
| Normal PPF + spectra | 40.700 | 14.756 | 0.531418 | 3.455646 |
| Axial relations + original spectra | 40.964 | 15.101 | 0.552786 | 3.451467 |
| Distance MLP | 42.545 | 15.939 | 0.473315 | 3.327587 |
| Local descriptor MLP | 39.290 | 13.659 | 0.477731 | 3.383141 |
| Angle + descriptor MLP | 39.348 | 14.287 | 0.496033 | 3.401636 |
| Full log-matrix MLP | 39.799 | 14.799 | 0.478484 | 3.349342 |

| Candidate minus control | Region mean pp | Region paired seeds pp | Center mean pp |
| --- | ---: | --- | ---: |
| Distance MLP | -1.693 | -0.169/-2.576/-2.333 | -0.692 |
| Local descriptor MLP | +1.562 | +2.113/-1.584/+4.159 | +1.587 |
| Angle + descriptor MLP | +1.504 | +3.493/+0.287/+0.733 | +0.959 |
| Full log-matrix MLP | +1.053 | -0.052/+1.813/+1.399 | +0.448 |
| Normal PPF + spectra | +0.152 | -0.144/+1.651/-1.050 | +0.491 |
| Axial relations + original spectra | -0.112 | -1.623/+2.414/-1.127 | +0.146 |

Eligible query fraction: 0.143408; valid counts: [4543, 4268].
Evaluable scenes: 5/5;scene-order/differences are retained in comparisons.json.
Minimum-eigenspace multiplicities k1/k2/k3: [61440, 0, 0]; repeated fraction 0.

## real_adapted / base

| Method | Region R@1 % | Center PCK % | Soft EPE | CE |
| --- | ---: | ---: | ---: | ---: |
| Full Gaussian relations (candidate) | 46.280 | 17.168 | 0.361792 | 2.863933 |
| Normal PPF + spectra | 44.945 | 16.450 | 0.374074 | 2.905325 |
| Axial relations + original spectra | 45.488 | 16.546 | 0.368621 | 2.880864 |
| Distance MLP | 44.469 | 16.033 | 0.382496 | 2.932316 |
| Local descriptor MLP | 42.583 | 15.337 | 0.390848 | 2.966711 |
| Angle + descriptor MLP | 42.737 | 15.261 | 0.388958 | 2.939580 |
| Full log-matrix MLP | 40.329 | 14.527 | 0.403255 | 3.002970 |

| Candidate minus control | Region mean pp | Region paired seeds pp | Center mean pp |
| --- | ---: | --- | ---: |
| Distance MLP | +1.810 | +4.172/-0.254/+1.512 | +1.135 |
| Local descriptor MLP | +3.696 | +3.651/+2.230/+5.209 | +1.831 |
| Angle + descriptor MLP | +3.542 | +4.559/+2.182/+3.886 | +1.906 |
| Full log-matrix MLP | +5.951 | +6.676/+3.735/+7.441 | +2.640 |
| Normal PPF + spectra | +1.334 | +0.717/+0.465/+2.822 | +0.718 |
| Axial relations + original spectra | +0.791 | -0.501/+0.017/+2.857 | +0.622 |

Eligible query fraction: 0.332389; valid counts: [10264, 10158].
Evaluable scenes: 5/5;scene-order/differences are retained in comparisons.json.
Minimum-eigenspace multiplicities k1/k2/k3: [61440, 0, 0]; repeated fraction 0.

## real_adapted / sparse256

| Method | Region R@1 % | Center PCK % | Soft EPE | CE |
| --- | ---: | ---: | ---: | ---: |
| Full Gaussian relations (candidate) | 48.782 | 17.765 | 0.323693 | 3.057277 |
| Normal PPF + spectra | 46.210 | 17.090 | 0.335972 | 3.126642 |
| Axial relations + original spectra | 47.608 | 17.541 | 0.331187 | 3.073548 |
| Distance MLP | 46.700 | 17.314 | 0.335776 | 3.124103 |
| Local descriptor MLP | 45.853 | 16.931 | 0.336647 | 3.113926 |
| Angle + descriptor MLP | 46.022 | 16.658 | 0.341889 | 3.122524 |
| Full log-matrix MLP | 45.002 | 17.007 | 0.360738 | 3.140882 |

| Candidate minus control | Region mean pp | Region paired seeds pp | Center mean pp |
| --- | ---: | --- | ---: |
| Distance MLP | +2.081 | +2.359/+0.907/+2.978 | +0.451 |
| Local descriptor MLP | +2.929 | +3.034/+1.903/+3.851 | +0.833 |
| Angle + descriptor MLP | +2.760 | +3.642/+2.554/+2.083 | +1.107 |
| Full log-matrix MLP | +3.780 | +3.344/+2.865/+5.130 | +0.757 |
| Normal PPF + spectra | +2.572 | +0.990/+3.194/+3.531 | +0.674 |
| Axial relations + original spectra | +1.174 | +0.550/-0.461/+3.433 | +0.223 |

Eligible query fraction: 0.143408; valid counts: [4543, 4268].
Evaluable scenes: 5/5;scene-order/differences are retained in comparisons.json.
Minimum-eigenspace multiplicities k1/k2/k3: [61440, 0, 0]; repeated fraction 0.

## New fitting and checks

| Domain | Role | PE parameters | Selected epochs | Updates per seed | Cumulative fit seconds |
| --- | --- | ---: | --- | --- | ---: |
| synthetic | Normal PPF + spectra | 10272 | 45/48/42 | 14880/14880/14880 | 1528.63 |
| synthetic | Axial relations + original spectra | 10368 | 45/39/45 | 14880/14880/14880 | 1500.39 |
| real | Normal PPF + spectra | 10272 | 18/42/24 | 1080/1080/1080 | 111.44 |
| real | Axial relations + original spectra | 10368 | 27/27/21 | 1080/1080/1080 | 108.68 |

NEW fitting/selection cumulative 3249.13s;shared-host timing,not a controlled speed benchmark.
Independent new metric maximum difference 1.78e-15;same-author separate arithmetic.
Old scalar/classification/sparse/source-transfer negatives remain. No global goal completion.

![Paired region differences](/home/liwt/GRoPE/results/round61_normal_relations/attempt1/report1/comparison.png)

![Validation curves](/home/liwt/GRoPE/results/round61_normal_relations/attempt1/report1/learning_curves.png)
