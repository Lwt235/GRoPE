# Gaussian PE Fourier approximation audit

Development namespace 1191 only, four scenes per condition. No training or correspondence labels are used. Query covariance is I/12; key covariance is projected_kcov. All arithmetic is float64 on CPU.

| Condition | M | MAE | 95th percentile absolute error | Maximum absolute error | Geometric top-1 agreement | Negative kernel fraction |
|---|---:|---:|---:|---:|---:|---:|
| oracle | 64 | 0.06149 | 0.16308 | 0.33368 | 98.96% | 47.65% |
| oracle | 128 | 0.06240 | 0.16598 | 0.33866 | 98.96% | 47.28% |
| oracle | 256 | 0.03422 | 0.10983 | 0.28386 | 98.96% | 43.07% |
| oracle | 512 | 0.01847 | 0.07232 | 0.13544 | 99.48% | 39.21% |
| oracle | 1024 | 0.01081 | 0.04592 | 0.08516 | 99.48% | 39.33% |
| oracle | 4096 | 0.00132 | 0.00515 | 0.02510 | 100.00% | 38.47% |
| mixed | 64 | 0.06034 | 0.15867 | 0.33385 | 99.48% | 47.48% |
| mixed | 128 | 0.06106 | 0.16177 | 0.33902 | 99.48% | 47.18% |
| mixed | 256 | 0.03307 | 0.10700 | 0.28362 | 99.48% | 43.17% |
| mixed | 512 | 0.01777 | 0.07005 | 0.13536 | 100.00% | 39.14% |
| mixed | 1024 | 0.01043 | 0.04399 | 0.09278 | 100.00% | 38.81% |
| mixed | 4096 | 0.00131 | 0.00517 | 0.02521 | 100.00% | 38.43% |
| normal_noise | 64 | 0.06145 | 0.16125 | 0.33508 | 98.96% | 47.58% |
| normal_noise | 128 | 0.06235 | 0.16563 | 0.33810 | 98.44% | 47.04% |
| normal_noise | 256 | 0.03413 | 0.10897 | 0.28377 | 98.96% | 43.22% |
| normal_noise | 512 | 0.01838 | 0.07187 | 0.13447 | 100.00% | 39.19% |
| normal_noise | 1024 | 0.01081 | 0.04535 | 0.08691 | 100.00% | 39.06% |
| normal_noise | 4096 | 0.00132 | 0.00515 | 0.02544 | 99.48% | 38.75% |
| scale_noise | 64 | 0.05628 | 0.14690 | 0.31635 | 98.44% | 46.76% |
| scale_noise | 128 | 0.05670 | 0.15032 | 0.32125 | 98.44% | 46.55% |
| scale_noise | 256 | 0.02914 | 0.09559 | 0.27366 | 98.96% | 43.86% |
| scale_noise | 512 | 0.01550 | 0.05808 | 0.14058 | 99.48% | 38.83% |
| scale_noise | 1024 | 0.00909 | 0.03801 | 0.09687 | 99.48% | 36.79% |
| scale_noise | 4096 | 0.00126 | 0.00514 | 0.04000 | 100.00% | 36.47% |
| wide | 64 | 0.06166 | 0.16450 | 0.33443 | 98.96% | 48.04% |
| wide | 128 | 0.06205 | 0.16470 | 0.33910 | 98.96% | 47.76% |
| wide | 256 | 0.03470 | 0.10637 | 0.28405 | 99.48% | 43.52% |
| wide | 512 | 0.01786 | 0.06893 | 0.29308 | 100.00% | 39.53% |
| wide | 1024 | 0.01037 | 0.04204 | 0.27739 | 100.00% | 39.84% |
| wide | 4096 | 0.00145 | 0.00536 | 0.23982 | 100.00% | 38.47% |
| depth_fit | 64 | 0.06023 | 0.15756 | 0.35155 | 98.44% | 47.22% |
| depth_fit | 128 | 0.06094 | 0.16132 | 0.35620 | 98.96% | 46.81% |
| depth_fit | 256 | 0.03359 | 0.10863 | 0.28379 | 98.96% | 43.09% |
| depth_fit | 512 | 0.01838 | 0.07129 | 0.27560 | 98.96% | 39.72% |
| depth_fit | 1024 | 0.01096 | 0.04684 | 0.26346 | 100.00% | 39.54% |
| depth_fit | 4096 | 0.00150 | 0.00534 | 0.21807 | 100.00% | 37.25% |

Selection: none of the audited M values passes. The numerical rule requires, in every development condition, MAE ≤ 0.01, 95th percentile ≤ 0.03, maximum error ≤ 0.1, and geometric top-1 agreement ≥ 98%. This checks approximation quality, not task performance.

Geometric top-1 compares the maxima of the exact and approximate kernels; it does not read correspondence ground truth. The JSON also records margins, overlapping-pair error, covariance eigenvalue ranges, exact unit-norm checks, source hashes, and stress tests.

Stress support variances range from 1e-6 to 100 and include rotated anisotropic ellipses. These artificial cases diagnose where the finite proposal fails; they are not benchmark test scenes and are not used to choose task hyperparameters. Unit normalization alone does not guarantee an accurate BC kernel.

| Stress support | M | MAE | Maximum absolute error |
|---|---:|---:|---:|
| isotropic_variance_1e-06/same_support | 256 | 0.11146 | 0.41712 |
| isotropic_variance_1e-06/same_support | 4096 | 0.07467 | 0.25201 |
| isotropic_variance_1e-06/fixed_query_support | 256 | 0.02292 | 0.39023 |
| isotropic_variance_1e-06/fixed_query_support | 4096 | 0.00297 | 0.30313 |
| isotropic_variance_0.0001/same_support | 256 | 0.11079 | 0.41444 |
| isotropic_variance_0.0001/same_support | 4096 | 0.07342 | 0.24802 |
| isotropic_variance_0.0001/fixed_query_support | 256 | 0.02271 | 0.33031 |
| isotropic_variance_0.0001/fixed_query_support | 4096 | 0.00266 | 0.24428 |
| isotropic_variance_0.001/same_support | 256 | 0.10492 | 0.38992 |
| isotropic_variance_0.001/same_support | 4096 | 0.06255 | 0.21339 |
| isotropic_variance_0.001/fixed_query_support | 256 | 0.02311 | 0.20450 |
| isotropic_variance_0.001/fixed_query_support | 4096 | 0.00213 | 0.12833 |
| isotropic_variance_0.01/same_support | 256 | 0.06723 | 0.21309 |
| isotropic_variance_0.01/same_support | 4096 | 0.01264 | 0.04259 |
| isotropic_variance_0.01/fixed_query_support | 256 | 0.03243 | 0.13677 |
| isotropic_variance_0.01/fixed_query_support | 4096 | 0.00248 | 0.01305 |
| isotropic_variance_0.0833333/same_support | 256 | 0.04712 | 0.19922 |
| isotropic_variance_0.0833333/same_support | 4096 | 0.00323 | 0.01620 |
| isotropic_variance_0.0833333/fixed_query_support | 256 | 0.04712 | 0.19922 |
| isotropic_variance_0.0833333/fixed_query_support | 4096 | 0.00323 | 0.01620 |
| isotropic_variance_1/same_support | 256 | 0.03194 | 0.12787 |
| isotropic_variance_1/same_support | 4096 | 0.00152 | 0.00626 |
| isotropic_variance_1/fixed_query_support | 256 | 0.01684 | 0.06065 |
| isotropic_variance_1/fixed_query_support | 4096 | 0.00068 | 0.00410 |
| isotropic_variance_10/same_support | 256 | 0.20028 | 0.60811 |
| isotropic_variance_10/same_support | 4096 | 0.00348 | 0.01226 |
| isotropic_variance_10/fixed_query_support | 256 | 0.02089 | 0.09125 |
| isotropic_variance_10/fixed_query_support | 4096 | 0.00029 | 0.00075 |
| isotropic_variance_100/same_support | 256 | 0.27476 | 0.95889 |
| isotropic_variance_100/same_support | 4096 | 0.00458 | 0.01613 |
| isotropic_variance_100/fixed_query_support | 256 | 0.01488 | 0.05077 |
| isotropic_variance_100/fixed_query_support | 4096 | 0.00093 | 0.00155 |
| anisotropic_eigenvalues_1e-06_0.0833333/same_support | 256 | 0.16649 | 0.46989 |
| anisotropic_eigenvalues_1e-06_0.0833333/same_support | 4096 | 0.06476 | 0.24443 |
| anisotropic_eigenvalues_1e-06_0.0833333/fixed_query_support | 256 | 0.03465 | 0.51111 |
| anisotropic_eigenvalues_1e-06_0.0833333/fixed_query_support | 4096 | 0.00524 | 0.45768 |
| anisotropic_eigenvalues_0.001_1/same_support | 256 | 0.28005 | 0.72614 |
| anisotropic_eigenvalues_0.001_1/same_support | 4096 | 0.09883 | 0.27605 |
| anisotropic_eigenvalues_0.001_1/fixed_query_support | 256 | 0.02520 | 0.09079 |
| anisotropic_eigenvalues_0.001_1/fixed_query_support | 4096 | 0.00381 | 0.07351 |
| anisotropic_eigenvalues_0.001_100/same_support | 256 | 0.36679 | 0.85205 |
| anisotropic_eigenvalues_0.001_100/same_support | 4096 | 0.10664 | 0.51931 |
| anisotropic_eigenvalues_0.001_100/fixed_query_support | 256 | 0.03675 | 0.08527 |
| anisotropic_eigenvalues_0.001_100/fixed_query_support | 4096 | 0.00886 | 0.04539 |

Interpretation: M=256 preserves most geometric top-1 choices but leaves substantial signed oscillation on distant pairs. Near-overlap error is much smaller than global error. Increasing M to 4096 resolves most common supports, while extremely thin projected supports in wide/depth_fit still fail the maximum-error criterion. The JSON records their covariance ranges.

For thin supports, true Fourier energy extends past the fixed proposal's well-sampled range. Finite-map normalization can then inflate the sampled low-frequency energy and overestimate overlap with wider supports. Stress cases show that unit norm alone is not a sufficient approximation guarantee. Investigate exact BC first, then improve quadrature or state a physically justified resolution floor and audit it again.

This is an approximation diagnostic. Poor kernel error cannot be interpreted as a failure of the continuous BC hypothesis; a downstream gain from a poor approximation must be described as a different finite positional kernel, not automatically attributed to true Gaussian overlap.

## Complete component nets and legacy parity

Legacy index parity confines one mixture component to positive y and the other to negative y. It is not a valid quadrature for arbitrary integrands. However, the BC kernel and feature normalizers only use integrands even under omega -> -omega; opposite half-space errors can cancel for this special case. Therefore failure of an odd test integral does not establish asymptotic bias in the BC kernel. The replacement gives each component a complete Sobol net and removes this unnecessary dependency.

| M | Rule | Odd integral, exact 0 | Even integral absolute error |
|---:|---|---:|---:|
| 256 | legacy_parity | -1.311935 | 0.031735 |
| 256 | complete_component_nets | -0.037870 | 0.039022 |
| 4096 | legacy_parity | -1.308053 | 0.000803 |
| 4096 | complete_component_nets | -0.003309 | 0.000044 |
| 65536 | legacy_parity | -1.311391 | 0.000012 |
| 65536 | complete_component_nets | 0.000419 | 0.000011 |

## Resolution floor changes the target kernel

Both query and key covariance receive floor*I before computing either exact BC or its feature approximation. This is a deliberately different, blurred support kernel, not a numerical equivalence. Selection still uses only development geometry.

| Added variance floor | Smallest passing M |
|---:|---:|
| 0.01 | 4096 |
| 0.0225 | 4096 |
| 0.0625 | 1024 |

Detailed error metrics, covariance ranges, norm checks and each passing-M list are in audit.json. Reproduce with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.gaussian_pe_approximation`. The earlier legacy-rule audit remains in results/round3_approximation.
