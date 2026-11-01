# Gaussian PE Fourier approximation audit

Development namespace 1191 only, four scenes per condition. No training or correspondence labels are used. Query covariance is I/12; key covariance is projected_kcov. All arithmetic is float64 on CPU.

| Condition | M | MAE | 95th percentile absolute error | Maximum absolute error | Geometric top-1 agreement | Negative kernel fraction |
|---|---:|---:|---:|---:|---:|---:|
| oracle | 64 | 0.06369 | 0.15920 | 0.36893 | 98.44% | 48.65% |
| oracle | 128 | 0.03490 | 0.11231 | 0.29567 | 99.48% | 44.06% |
| oracle | 256 | 0.01946 | 0.07143 | 0.14723 | 100.00% | 40.64% |
| oracle | 512 | 0.01201 | 0.04741 | 0.09393 | 98.96% | 38.24% |
| oracle | 1024 | 0.00465 | 0.01608 | 0.03405 | 98.96% | 39.24% |
| oracle | 4096 | 0.00086 | 0.00313 | 0.00780 | 100.00% | 35.16% |
| mixed | 64 | 0.06205 | 0.15740 | 0.36925 | 98.96% | 48.64% |
| mixed | 128 | 0.03363 | 0.10936 | 0.29558 | 99.48% | 43.74% |
| mixed | 256 | 0.01880 | 0.06869 | 0.14956 | 100.00% | 40.48% |
| mixed | 512 | 0.01162 | 0.04604 | 0.09494 | 100.00% | 37.81% |
| mixed | 1024 | 0.00457 | 0.01534 | 0.03413 | 99.48% | 38.44% |
| mixed | 4096 | 0.00086 | 0.00312 | 0.01615 | 100.00% | 35.09% |
| normal_noise | 64 | 0.06361 | 0.15865 | 0.36968 | 98.96% | 48.48% |
| normal_noise | 128 | 0.03477 | 0.11068 | 0.29526 | 98.96% | 43.78% |
| normal_noise | 256 | 0.01941 | 0.07103 | 0.14502 | 100.00% | 40.85% |
| normal_noise | 512 | 0.01199 | 0.04754 | 0.09357 | 100.00% | 38.02% |
| normal_noise | 1024 | 0.00464 | 0.01606 | 0.03413 | 100.00% | 39.30% |
| normal_noise | 4096 | 0.00086 | 0.00312 | 0.01771 | 100.00% | 34.73% |
| scale_noise | 64 | 0.05718 | 0.14612 | 0.35337 | 97.92% | 47.35% |
| scale_noise | 128 | 0.02937 | 0.09741 | 0.28752 | 98.96% | 43.86% |
| scale_noise | 256 | 0.01666 | 0.05762 | 0.15049 | 99.48% | 39.72% |
| scale_noise | 512 | 0.01017 | 0.04076 | 0.10083 | 99.48% | 36.56% |
| scale_noise | 1024 | 0.00432 | 0.01397 | 0.06213 | 98.96% | 35.68% |
| scale_noise | 4096 | 0.00081 | 0.00304 | 0.05595 | 100.00% | 33.82% |
| wide | 64 | 0.06315 | 0.16193 | 0.36940 | 98.96% | 48.44% |
| wide | 128 | 0.03531 | 0.11079 | 0.29504 | 98.96% | 44.16% |
| wide | 256 | 0.01879 | 0.06737 | 0.29076 | 100.00% | 40.69% |
| wide | 512 | 0.01142 | 0.04369 | 0.27662 | 100.00% | 39.67% |
| wide | 1024 | 0.00482 | 0.01587 | 0.27199 | 100.00% | 39.78% |
| wide | 4096 | 0.00093 | 0.00305 | 0.24056 | 100.00% | 35.37% |
| depth_fit | 64 | 0.06231 | 0.15611 | 0.36873 | 98.44% | 48.12% |
| depth_fit | 128 | 0.03420 | 0.11034 | 0.29561 | 98.96% | 44.12% |
| depth_fit | 256 | 0.01939 | 0.06998 | 0.27989 | 98.96% | 40.98% |
| depth_fit | 512 | 0.01201 | 0.04763 | 0.26833 | 100.00% | 38.79% |
| depth_fit | 1024 | 0.00482 | 0.01562 | 0.22978 | 100.00% | 39.49% |
| depth_fit | 4096 | 0.00103 | 0.00314 | 0.23568 | 100.00% | 34.64% |

Selection: none of the audited M values passes. The numerical rule requires, in every development condition, MAE ≤ 0.01, 95th percentile ≤ 0.03, maximum error ≤ 0.1, and geometric top-1 agreement ≥ 98%. This checks approximation quality, not task performance.

Geometric top-1 compares the maxima of the exact and approximate kernels; it does not read correspondence ground truth. The JSON also records margins, overlapping-pair error, covariance eigenvalue ranges, exact unit-norm checks, source hashes, and stress tests.

Stress support variances range from 1e-6 to 100 and include rotated anisotropic ellipses. These artificial cases diagnose where the finite proposal fails; they are not benchmark test scenes and are not used to choose task hyperparameters. Unit normalization alone does not guarantee an accurate BC kernel.

| Stress support | M | MAE | Maximum absolute error |
|---|---:|---:|---:|
| isotropic_variance_1e-06/same_support | 256 | 0.14911 | 0.44504 |
| isotropic_variance_1e-06/same_support | 4096 | 0.43115 | 0.72004 |
| isotropic_variance_1e-06/fixed_query_support | 256 | 0.01272 | 0.36568 |
| isotropic_variance_1e-06/fixed_query_support | 4096 | 0.00144 | 0.17568 |
| isotropic_variance_0.0001/same_support | 256 | 0.14736 | 0.44132 |
| isotropic_variance_0.0001/same_support | 4096 | 0.42513 | 0.71125 |
| isotropic_variance_0.0001/fixed_query_support | 256 | 0.01247 | 0.30614 |
| isotropic_variance_0.0001/fixed_query_support | 4096 | 0.00113 | 0.11795 |
| isotropic_variance_0.001/same_support | 256 | 0.13217 | 0.40787 |
| isotropic_variance_0.001/same_support | 4096 | 0.36582 | 0.62308 |
| isotropic_variance_0.001/fixed_query_support | 256 | 0.01247 | 0.18376 |
| isotropic_variance_0.001/fixed_query_support | 4096 | 0.00066 | 0.01480 |
| isotropic_variance_0.01/same_support | 256 | 0.05088 | 0.23715 |
| isotropic_variance_0.01/same_support | 4096 | 0.01225 | 0.03962 |
| isotropic_variance_0.01/fixed_query_support | 256 | 0.01755 | 0.06824 |
| isotropic_variance_0.01/fixed_query_support | 4096 | 0.00152 | 0.01017 |
| isotropic_variance_0.0833333/same_support | 256 | 0.02388 | 0.10099 |
| isotropic_variance_0.0833333/same_support | 4096 | 0.00199 | 0.01437 |
| isotropic_variance_0.0833333/fixed_query_support | 256 | 0.02388 | 0.10099 |
| isotropic_variance_0.0833333/fixed_query_support | 4096 | 0.00199 | 0.01437 |
| isotropic_variance_1/same_support | 256 | 0.01459 | 0.08692 |
| isotropic_variance_1/same_support | 4096 | 0.00068 | 0.00429 |
| isotropic_variance_1/fixed_query_support | 256 | 0.00608 | 0.02901 |
| isotropic_variance_1/fixed_query_support | 4096 | 0.00031 | 0.00248 |
| isotropic_variance_10/same_support | 256 | 0.08320 | 0.19800 |
| isotropic_variance_10/same_support | 4096 | 0.00170 | 0.00713 |
| isotropic_variance_10/fixed_query_support | 256 | 0.00972 | 0.03681 |
| isotropic_variance_10/fixed_query_support | 4096 | 0.00014 | 0.00049 |
| isotropic_variance_100/same_support | 256 | 0.14036 | 0.66146 |
| isotropic_variance_100/same_support | 4096 | 0.00357 | 0.01881 |
| isotropic_variance_100/fixed_query_support | 256 | 0.00602 | 0.02611 |
| isotropic_variance_100/fixed_query_support | 4096 | 0.00053 | 0.00078 |
| anisotropic_eigenvalues_1e-06_0.0833333/same_support | 256 | 0.11047 | 0.34088 |
| anisotropic_eigenvalues_1e-06_0.0833333/same_support | 4096 | 0.60112 | 0.92911 |
| anisotropic_eigenvalues_1e-06_0.0833333/fixed_query_support | 256 | 0.02069 | 0.53129 |
| anisotropic_eigenvalues_1e-06_0.0833333/fixed_query_support | 4096 | 0.00085 | 0.07354 |
| anisotropic_eigenvalues_0.001_1/same_support | 256 | 0.23009 | 0.60085 |
| anisotropic_eigenvalues_0.001_1/same_support | 4096 | 0.60463 | 0.93900 |
| anisotropic_eigenvalues_0.001_1/fixed_query_support | 256 | 0.01946 | 0.09225 |
| anisotropic_eigenvalues_0.001_1/fixed_query_support | 4096 | 0.00401 | 0.22371 |
| anisotropic_eigenvalues_0.001_100/same_support | 256 | 0.34839 | 0.87380 |
| anisotropic_eigenvalues_0.001_100/same_support | 4096 | 0.09775 | 0.54231 |
| anisotropic_eigenvalues_0.001_100/fixed_query_support | 256 | 0.04384 | 0.09035 |
| anisotropic_eigenvalues_0.001_100/fixed_query_support | 4096 | 0.00652 | 0.05407 |

Interpretation: M=256 preserves most geometric top-1 choices but leaves substantial signed oscillation on distant pairs. Near-overlap error is much smaller than global error. Increasing M to 4096 resolves most common supports, while extremely thin projected supports in wide/depth_fit still fail the maximum-error criterion. The JSON records their covariance ranges.

For thin supports, true Fourier energy extends past the fixed proposal's well-sampled range. Finite-map normalization can then inflate the sampled low-frequency energy and overestimate overlap with wider supports. Stress cases show that unit norm alone is not a sufficient approximation guarantee. Investigate exact BC first, then improve quadrature or state a physically justified resolution floor and audit it again.

This is an approximation diagnostic. Poor kernel error cannot be interpreted as a failure of the continuous BC hypothesis; a downstream gain from a poor approximation must be described as a different finite positional kernel, not automatically attributed to true Gaussian overlap.
