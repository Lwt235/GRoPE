# R57: native center RoPE and Gaussian-pair similarity — development results

Nine new fits and39 reused R56/R48 references, seeds11/22/33. Same data/backbone120epoch budget.
Official test and matching confirmation scenes remain untouched. No generality/novelty claim.

## Base utility

| Role | OA % | mCA % | CE | PE parameters |
| --- | ---: | ---: | ---: | ---: |
| RoPE + full Gaussian-pair | 89.292 | 84.486 | 0.434493 | 1746 |
| RoPE + adaptive pair distance | 89.223 | 83.683 | 0.445731 | 1746 |
| RoPE + learned-center distance | 88.877 | 84.019 | 0.433895 | 582 |
| RoPE + full Gaussian | 89.223 | 83.749 | 0.431164 | 1746 |
| RoPE + fixed distance | 88.808 | 83.584 | 0.435727 | 0 |
| RoPE + free quadratic | 89.154 | 84.192 | 0.434653 | 1746 |
| Full Gaussian alone | 88.325 | 83.436 | 0.448035 | 1746 |
| Diagonal Gaussian alone | 89.223 | 83.776 | 0.439629 | 1164 |
| Isotropic Gaussian alone | 88.187 | 82.581 | 0.471134 | 776 |
| Static Gaussian alone | 88.048 | 83.374 | 0.461933 | 18 |
| Fixed distance alone | 87.945 | 82.977 | 0.453262 | 0 |
| Free quadratic alone | 88.532 | 83.423 | 0.440329 | 1746 |
| Center RoPE | 88.774 | 83.695 | 0.457481 | 0 |
| Fourier + RoPE | 88.532 | 83.792 | 0.443398 | 320 |
| MLP + RoPE | 89.154 | 84.457 | 0.437119 | 448 |
| No explicit PE | 87.841 | 82.517 | 0.489798 | 0 |

## Paired primary differences

| Control | OA delta pp | Seed11/22/33 deltas pp | Wins / ties | mCA delta pp | CE delta |
| --- | ---: | --- | --- | ---: | ---: |
| RoPE + adaptive pair distance | +0.069 | -1.658, +1.036, +0.829 | 2/3 ; 0 ties | +0.803 | -0.011239 |
| RoPE + learned-center distance | +0.415 | +0.104, +1.036, +0.104 | 3/3 ; 0 ties | +0.467 | +0.000598 |
| RoPE + full Gaussian | +0.069 | -1.244, +1.347, +0.104 | 2/3 ; 0 ties | +0.737 | +0.003329 |
| RoPE + fixed distance | +0.484 | +0.311, +0.518, +0.622 | 3/3 ; 0 ties | +0.902 | -0.001234 |
| RoPE + free quadratic | +0.138 | -0.415, +0.933, -0.104 | 1/3 ; 0 ties | +0.294 | -0.000160 |
| Full Gaussian alone | +0.967 | +0.000, +2.073, +0.829 | 2/3 ; 1 ties | +1.050 | -0.013542 |
| Diagonal Gaussian alone | +0.069 | -0.207, +0.829, -0.415 | 1/3 ; 0 ties | +0.710 | -0.005136 |
| Isotropic Gaussian alone | +1.105 | -0.311, +2.280, +1.347 | 2/3 ; 0 ties | +1.906 | -0.036641 |
| Static Gaussian alone | +1.244 | -0.933, +2.591, +2.073 | 2/3 ; 0 ties | +1.112 | -0.027440 |
| Fixed distance alone | +1.347 | -0.207, +2.694, +1.554 | 2/3 ; 0 ties | +1.509 | -0.018769 |
| Free quadratic alone | +0.760 | -0.311, +2.073, +0.518 | 2/3 ; 0 ties | +1.063 | -0.005836 |
| Center RoPE | +0.518 | +0.104, +1.658, -0.207 | 2/3 ; 0 ties | +0.791 | -0.022988 |
| Fourier + RoPE | +0.760 | +0.000, +1.451, +0.829 | 2/3 ; 1 ties | +0.694 | -0.008906 |
| MLP + RoPE | +0.138 | -0.415, +1.554, -0.725 | 1/3 ; 0 ties | +0.029 | -0.002626 |
| No explicit PE | +1.451 | +0.104, +3.316, +0.933 | 3/3 ; 0 ties | +1.969 | -0.055305 |

## All fixed conditions (absolute OA %)

| Role | base | y90 | so3 | jitter02 | points256 | points1024 | permuted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RoPE + full Gaussian-pair | 89.292 | 89.154 | 6.425 | 88.981 | 87.392 | 89.672 | 89.292 |
| RoPE + adaptive pair distance | 89.223 | 88.843 | 6.736 | 88.532 | 87.254 | 88.877 | 89.223 |
| RoPE + learned-center distance | 88.877 | 88.601 | 6.978 | 88.221 | 87.876 | 89.430 | 88.877 |
| RoPE + full Gaussian | 89.223 | 88.221 | 5.803 | 88.394 | 87.737 | 89.223 | 89.223 |
| RoPE + fixed distance | 88.808 | 88.428 | 6.149 | 88.636 | 87.599 | 89.603 | 88.808 |
| RoPE + free quadratic | 89.154 | 89.119 | 6.218 | 89.154 | 87.772 | 89.430 | 89.154 |
| Full Gaussian alone | 88.325 | 88.497 | 5.941 | 88.187 | 87.392 | 88.739 | 88.325 |
| Diagonal Gaussian alone | 89.223 | 88.877 | 6.218 | 88.497 | 87.807 | 89.568 | 89.223 |
| Isotropic Gaussian alone | 88.187 | 88.014 | 7.737 | 87.737 | 86.908 | 88.290 | 88.187 |
| Static Gaussian alone | 88.048 | 88.428 | 5.976 | 88.083 | 86.839 | 88.152 | 88.048 |
| Fixed distance alone | 87.945 | 87.910 | 5.769 | 87.427 | 87.254 | 88.601 | 87.945 |
| Free quadratic alone | 88.532 | 88.428 | 6.356 | 88.566 | 87.737 | 88.808 | 88.532 |
| Center RoPE | 88.774 | 88.152 | 4.525 | 88.601 | 86.874 | 89.050 | 88.774 |
| Fourier + RoPE | 88.532 | 87.668 | 4.801 | 88.083 | 87.185 | 88.670 | 88.532 |
| MLP + RoPE | 89.154 | 88.670 | 4.629 | 88.428 | 87.668 | 89.637 | 89.154 |
| No explicit PE | 87.841 | 87.945 | 5.907 | 87.358 | 86.390 | 87.703 | 87.841 |

## Fixed-checkpoint shape interventions

Remove only the covariance-compatibility term at fixed checkpoints. Earlier layer interventions can alter later features and predicted centers.
This does not isolate a physical covariance causal effect; compare with the separately trained equal-capacity distance control.
Distribution shift also changes predictions; sensitivity alone does not establish useful physical geometry.

| Role / intervention | OA delta pp by seed | Mean OA delta pp | Mean CE delta |
| --- | --- | ---: | ---: |
| RoPE + full Gaussian-pair / shape_off | -0.311, +0.207, -0.622 | -0.242 | +0.009674 |

## Cost and bounded checks

New training/selection cumulative seconds: 24645.52. Includes checkpoint overhead; shared-host cost is not a formal throughput benchmark.
48 independent formula samples: maxP error 6.9255335e-06; winner changes 0; warnings 0.
336 saved-prediction metric replays: maximum absolute correct-count difference 0; maximum CE difference 1.1321677e-07.
Same-author independent arithmetic, not an independent laboratory or full-network rewrite.
The primary comparison needs substantive interpretation; there is no automatic1pp promotion threshold.

![Base comparisons](/home/liwt/GRoPE/results/round57_gaussian_pair/attempt1/report1/comparison.png)

![Learning curves](/home/liwt/GRoPE/results/round57_gaussian_pair/attempt1/report1/learning_curves.png)
