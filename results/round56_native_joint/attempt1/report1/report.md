# R56: native center RoPE and regional preference — development results

Nine new fits and30 reused R48 references, seeds11/22/33. Same data/backbone120epoch budget.
Official test and matching confirmation scenes remain untouched. No generality/novelty claim.

## Base utility

| Role | OA % | mCA % | CE | PE parameters |
| --- | ---: | ---: | ---: | ---: |
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
| RoPE + fixed distance | +0.415 | +1.554, -0.829, +0.518 | 2/3 ; 0 ties | +0.164 | -0.004563 |
| RoPE + free quadratic | +0.069 | +0.829, -0.415, -0.207 | 1/3 ; 0 ties | -0.444 | -0.003489 |
| Full Gaussian alone | +0.898 | +1.244, +0.725, +0.725 | 3/3 ; 0 ties | +0.313 | -0.016871 |
| Diagonal Gaussian alone | +0.000 | +1.036, -0.518, -0.518 | 1/3 ; 0 ties | -0.028 | -0.008465 |
| Isotropic Gaussian alone | +1.036 | +0.933, +0.933, +1.244 | 3/3 ; 0 ties | +1.168 | -0.039970 |
| Static Gaussian alone | +1.174 | +0.311, +1.244, +1.969 | 3/3 ; 0 ties | +0.374 | -0.030769 |
| Fixed distance alone | +1.278 | +1.036, +1.347, +1.451 | 3/3 ; 0 ties | +0.772 | -0.022098 |
| Free quadratic alone | +0.691 | +0.933, +0.725, +0.415 | 3/3 ; 0 ties | +0.326 | -0.009165 |
| Center RoPE | +0.449 | +1.347, +0.311, -0.311 | 2/3 ; 0 ties | +0.054 | -0.026317 |
| Fourier + RoPE | +0.691 | +1.244, +0.104, +0.725 | 3/3 ; 0 ties | -0.043 | -0.012235 |
| MLP + RoPE | +0.069 | +0.829, +0.207, -0.829 | 2/3 ; 0 ties | -0.709 | -0.005955 |
| No explicit PE | +1.382 | +1.347, +1.969, +0.829 | 3/3 ; 0 ties | +1.231 | -0.058634 |

## All fixed conditions (absolute OA %)

| Role | base | y90 | so3 | jitter02 | points256 | points1024 | permuted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
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

Input anchors stay fixed. Each intervention preserves the mean predicted from its current layer input; earlier interventions may change later features and predicted means.
Sphere preserves covariance trace; Z rotation preserves covariance eigenvalues.
Distribution shift also changes predictions; sensitivity alone does not establish useful physical geometry.

| Role / intervention | OA delta pp by seed | Mean OA delta pp | Mean CE delta |
| --- | --- | ---: | ---: |
| RoPE + full Gaussian / isotropic_trace | -32.642, -14.611, -15.959 | -21.071 | +0.880821 |
| RoPE + full Gaussian / rotate_z90 | -68.912, -47.772, -44.145 | -53.610 | +2.296073 |
| Full Gaussian alone / isotropic_trace | -35.026, -42.694, -38.964 | -38.895 | +1.567793 |
| Full Gaussian alone / rotate_z90 | -60.311, -72.332, -55.751 | -62.798 | +2.642302 |

## Cost and bounded checks

New training/selection cumulative seconds: 16492.57. Includes checkpoint overhead; shared-host cost is not a formal throughput benchmark.
39 independent formula samples: maxP error 6.9255335e-06; winner changes 0; warnings 0.
273 saved-prediction metric replays: maximum absolute correct-count difference 0; maximum CE difference 1.1321677e-07.
Same-author independent arithmetic, not an independent laboratory or full-network rewrite.
The primary comparison needs substantive interpretation; there is no automatic1pp promotion threshold.

![Base comparisons](/home/liwt/GRoPE/results/round56_native_joint/attempt1/report1/comparison.png)

![Learning curves](/home/liwt/GRoPE/results/round56_native_joint/attempt1/report1/learning_curves.png)
