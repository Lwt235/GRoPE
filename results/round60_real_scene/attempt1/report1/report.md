# R60 real-scene correspondence results

30 exact source models;30 NEW target adaptations;120 prediction groups. Same architecture and old scene pairs.
Both source transfer and fixed target adaptation were planned before target outcomes. All target selections precede all old_eval exports.
Native sensor depth is observed INPUT;relative pose is supervision/scoring only. No RGB-only,optimized-3DGS or full-registration claim.
Metrics use pair-normalized scene units,not metres or pixels. Primary aggregate is scene-macro region recall.
Old development scenes and reused source models are not fresh confirmation. The full research goal is unachieved.
Scientific interpretation and visual review are pending when this report is generated.

## source / base

Evaluable scenes 5/5; below uses scene macro and three-seed means.

| Method | Region R@1 % | Center PCK % | Soft EPE | CE | Query-weighted R@1 % |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gaussian invariant MLP (candidate) | 35.629 | 12.902 | 0.587169 | 3.330644 | 36.466 |
| R58 scalar Gaussian | 35.384 | 12.848 | 0.557711 | 3.294948 | 36.028 |
| R58 Gaussian metric | 36.858 | 13.684 | 0.535356 | 3.214176 | 37.512 |
| R58 sinusoidal distance | 36.799 | 13.491 | 0.556443 | 3.266367 | 37.504 |
| R58 distance + angle | 37.274 | 13.772 | 0.545741 | 3.247536 | 37.946 |
| No explicit PE | 20.423 | 7.337 | 0.689288 | 3.731789 | 20.515 |
| Distance MLP | 36.742 | 13.452 | 0.535265 | 3.234470 | 37.447 |
| Local descriptor MLP | 33.384 | 11.936 | 0.564963 | 3.333618 | 34.402 |
| Angle + descriptor MLP | 33.511 | 11.867 | 0.574844 | 3.338078 | 34.339 |
| Full log-matrix MLP | 33.215 | 11.720 | 0.544288 | 3.300616 | 33.963 |

| Candidate minus control | Region mean pp | Region paired seeds pp | Center mean pp | Region scene means pp |
| --- | ---: | --- | ---: | --- |
| R58 scalar Gaussian | +0.245 | +1.986/-5.200/+3.947 | +0.054 | +0.129/-1.136/-0.542/+0.581/+2.190 |
| R58 Gaussian metric | -1.229 | +2.078/-9.455/+3.691 | -0.781 | -2.280/-2.426/-1.254/-0.254/+0.069 |
| R58 sinusoidal distance | -1.170 | -0.202/-4.558/+1.250 | -0.588 | -1.583/-2.470/-0.552/+0.390/-1.635 |
| R58 distance + angle | -1.646 | -0.793/-7.052/+2.908 | -0.869 | -2.246/-3.440/-0.351/+0.983/-3.174 |
| No explicit PE | +15.206 | +16.416/+9.880/+19.321 | +5.565 | +13.483/+8.678/+15.132/+18.333/+20.402 |
| Distance MLP | -1.113 | +3.144/-7.857/+1.372 | -0.550 | -1.859/-2.514/-0.120/+0.575/-1.649 |
| Local descriptor MLP | +2.245 | +3.802/-5.227/+8.160 | +0.966 | +1.308/+2.194/+4.653/+0.359/+2.710 |
| Angle + descriptor MLP | +2.118 | +4.714/-2.291/+3.930 | +1.036 | +2.444/+1.577/+2.046/+1.119/+3.403 |
| Full log-matrix MLP | +2.413 | +3.295/-2.801/+6.747 | +1.182 | +0.869/+1.081/+3.881/+2.702/+3.534 |

Scene order: harvard_c11/hv_c11_2, hotel_umd/maryland_hotel3, mit_76_studyroom/76-1studyroom2, mit_dorm_next_sj/dorm_next_sj_oct_30_2012_scan1_erika, mit_lab_hj/lab_hj_tea_nov_2_2012_scan1_erika.
Seed and scene differences are descriptive;nearby frames are correlated. No query-level confidence interval is presented.

| Scene | Pairs | Valid queries | Eligible fraction |
| --- | ---: | ---: | ---: |
| harvard_c11/hv_c11_2 | 96 | 3874 | 0.315267 |
| hotel_umd/maryland_hotel3 | 96 | 3023 | 0.246012 |
| mit_76_studyroom/76-1studyroom2 | 96 | 3324 | 0.270508 |
| mit_dorm_next_sj/dorm_next_sj_oct_30_2012_scan1_erika | 96 | 5391 | 0.438721 |
| mit_lab_hj/lab_hj_tea_nov_2_2012_scan1_erika | 96 | 4810 | 0.391439 |

Excluded by native input availability: 0; zero-valid pairs: 25.
Sparse supports and eligible populations differ from base;compare matched methods within each condition.

## source / sparse256

Evaluable scenes 5/5; below uses scene macro and three-seed means.

| Method | Region R@1 % | Center PCK % | Soft EPE | CE | Query-weighted R@1 % |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gaussian invariant MLP (candidate) | 40.852 | 15.247 | 0.509131 | 3.384141 | 42.549 |
| R58 scalar Gaussian | 39.392 | 14.926 | 0.516734 | 3.454605 | 40.465 |
| R58 Gaussian metric | 40.687 | 15.096 | 0.505761 | 3.417239 | 42.197 |
| R58 sinusoidal distance | 41.243 | 15.570 | 0.528859 | 3.434475 | 42.674 |
| R58 distance + angle | 42.419 | 16.291 | 0.504745 | 3.415532 | 44.127 |
| No explicit PE | 31.199 | 11.540 | 0.625116 | 3.720864 | 31.884 |
| Distance MLP | 42.545 | 15.939 | 0.473315 | 3.327587 | 44.236 |
| Local descriptor MLP | 39.290 | 13.659 | 0.477731 | 3.383141 | 41.032 |
| Angle + descriptor MLP | 39.348 | 14.287 | 0.496033 | 3.401636 | 41.093 |
| Full log-matrix MLP | 39.799 | 14.799 | 0.478484 | 3.349342 | 41.161 |

| Candidate minus control | Region mean pp | Region paired seeds pp | Center mean pp | Region scene means pp |
| --- | ---: | --- | ---: | --- |
| R58 scalar Gaussian | +1.460 | +2.239/-0.708/+2.850 | +0.320 | +0.622/-3.271/+0.751/+4.531/+4.667 |
| R58 Gaussian metric | +0.165 | +2.606/-3.018/+0.907 | +0.150 | -1.304/-2.577/+1.914/+0.995/+1.797 |
| R58 sinusoidal distance | -0.391 | -1.702/+0.360/+0.170 | -0.324 | +2.000/-2.279/+0.024/-1.350/-0.348 |
| R58 distance + angle | -1.567 | -1.825/-3.380/+0.505 | -1.044 | -1.289/-2.114/-1.332/-0.213/-2.884 |
| No explicit PE | +9.653 | +9.824/+8.295/+10.839 | +3.707 | +8.474/+1.751/+10.441/+11.105/+16.493 |
| Distance MLP | -1.693 | -0.169/-2.576/-2.333 | -0.692 | -4.059/-3.766/+0.363/+0.302/-1.304 |
| Local descriptor MLP | +1.562 | +2.113/-1.584/+4.159 | +1.587 | +2.119/+0.694/+2.689/+2.630/-0.319 |
| Angle + descriptor MLP | +1.504 | +3.493/+0.287/+0.733 | +0.959 | +2.593/+0.264/+3.828/+1.084/-0.246 |
| Full log-matrix MLP | +1.053 | -0.052/+1.813/+1.399 | +0.448 | +1.496/-2.213/+2.204/+1.475/+2.304 |

Scene order: harvard_c11/hv_c11_2, hotel_umd/maryland_hotel3, mit_76_studyroom/76-1studyroom2, mit_dorm_next_sj/dorm_next_sj_oct_30_2012_scan1_erika, mit_lab_hj/lab_hj_tea_nov_2_2012_scan1_erika.
Seed and scene differences are descriptive;nearby frames are correlated. No query-level confidence interval is presented.

| Scene | Pairs | Valid queries | Eligible fraction |
| --- | ---: | ---: | ---: |
| harvard_c11/hv_c11_2 | 96 | 2250 | 0.183105 |
| hotel_umd/maryland_hotel3 | 96 | 1009 | 0.082113 |
| mit_76_studyroom/76-1studyroom2 | 96 | 1376 | 0.111979 |
| mit_dorm_next_sj/dorm_next_sj_oct_30_2012_scan1_erika | 96 | 1876 | 0.152669 |
| mit_lab_hj/lab_hj_tea_nov_2_2012_scan1_erika | 96 | 2300 | 0.187174 |

Excluded by native input availability: 0; zero-valid pairs: 117.
Sparse supports and eligible populations differ from base;compare matched methods within each condition.

## adapted / base

Evaluable scenes 5/5; below uses scene macro and three-seed means.

| Method | Region R@1 % | Center PCK % | Soft EPE | CE | Query-weighted R@1 % |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gaussian invariant MLP (candidate) | 46.280 | 17.168 | 0.361792 | 2.863933 | 47.150 |
| R58 scalar Gaussian | 41.497 | 14.940 | 0.393369 | 2.988513 | 42.038 |
| R58 Gaussian metric | 42.346 | 15.525 | 0.385767 | 2.935963 | 42.919 |
| R58 sinusoidal distance | 43.697 | 16.192 | 0.391852 | 2.944508 | 44.539 |
| R58 distance + angle | 43.747 | 16.083 | 0.391592 | 2.944978 | 44.527 |
| No explicit PE | 22.090 | 7.808 | 0.630790 | 3.612902 | 22.198 |
| Distance MLP | 44.469 | 16.033 | 0.382496 | 2.932316 | 45.208 |
| Local descriptor MLP | 42.583 | 15.337 | 0.390848 | 2.966711 | 43.654 |
| Angle + descriptor MLP | 42.737 | 15.261 | 0.388958 | 2.939580 | 43.817 |
| Full log-matrix MLP | 40.329 | 14.527 | 0.403255 | 3.002970 | 41.046 |

| Candidate minus control | Region mean pp | Region paired seeds pp | Center mean pp | Region scene means pp |
| --- | ---: | --- | ---: | --- |
| R58 scalar Gaussian | +4.782 | +5.494/+3.431/+5.422 | +2.227 | +4.121/+3.231/+3.560/+7.420/+5.579 |
| R58 Gaussian metric | +3.933 | +4.745/+1.804/+5.250 | +1.643 | +3.098/+2.393/+3.329/+6.764/+4.082 |
| R58 sinusoidal distance | +2.582 | +2.860/+1.485/+3.402 | +0.975 | +2.891/+2.283/+3.028/+4.322/+0.388 |
| R58 distance + angle | +2.533 | +2.673/+1.006/+3.919 | +1.085 | +2.229/+1.279/+3.851/+4.736/+0.568 |
| No explicit PE | +24.190 | +24.807/+22.331/+25.431 | +9.360 | +23.412/+16.518/+25.010/+28.232/+27.775 |
| Distance MLP | +1.810 | +4.172/-0.254/+1.512 | +1.135 | +1.790/+0.342/+2.447/+3.531/+0.942 |
| Local descriptor MLP | +3.696 | +3.651/+2.230/+5.209 | +1.831 | +3.760/+4.510/+4.753/+1.966/+3.493 |
| Angle + descriptor MLP | +3.542 | +4.559/+2.182/+3.886 | +1.906 | +3.872/+4.808/+4.101/+2.047/+2.883 |
| Full log-matrix MLP | +5.951 | +6.676/+3.735/+7.441 | +2.640 | +5.791/+3.959/+6.428/+6.189/+7.387 |

Scene order: harvard_c11/hv_c11_2, hotel_umd/maryland_hotel3, mit_76_studyroom/76-1studyroom2, mit_dorm_next_sj/dorm_next_sj_oct_30_2012_scan1_erika, mit_lab_hj/lab_hj_tea_nov_2_2012_scan1_erika.
Seed and scene differences are descriptive;nearby frames are correlated. No query-level confidence interval is presented.

| Scene | Pairs | Valid queries | Eligible fraction |
| --- | ---: | ---: | ---: |
| harvard_c11/hv_c11_2 | 96 | 3874 | 0.315267 |
| hotel_umd/maryland_hotel3 | 96 | 3023 | 0.246012 |
| mit_76_studyroom/76-1studyroom2 | 96 | 3324 | 0.270508 |
| mit_dorm_next_sj/dorm_next_sj_oct_30_2012_scan1_erika | 96 | 5391 | 0.438721 |
| mit_lab_hj/lab_hj_tea_nov_2_2012_scan1_erika | 96 | 4810 | 0.391439 |

Excluded by native input availability: 0; zero-valid pairs: 25.
Sparse supports and eligible populations differ from base;compare matched methods within each condition.

## adapted / sparse256

Evaluable scenes 5/5; below uses scene macro and three-seed means.

| Method | Region R@1 % | Center PCK % | Soft EPE | CE | Query-weighted R@1 % |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gaussian invariant MLP (candidate) | 48.782 | 17.765 | 0.323693 | 3.057277 | 50.584 |
| R58 scalar Gaussian | 44.458 | 16.217 | 0.346825 | 3.192883 | 45.867 |
| R58 Gaussian metric | 46.068 | 16.747 | 0.338172 | 3.154247 | 47.584 |
| R58 sinusoidal distance | 46.663 | 17.526 | 0.351376 | 3.123208 | 48.576 |
| R58 distance + angle | 45.936 | 16.943 | 0.349557 | 3.151894 | 47.637 |
| No explicit PE | 33.941 | 12.483 | 0.520460 | 3.549034 | 34.922 |
| Distance MLP | 46.700 | 17.314 | 0.335776 | 3.124103 | 48.651 |
| Local descriptor MLP | 45.853 | 16.931 | 0.336647 | 3.113926 | 47.914 |
| Angle + descriptor MLP | 46.022 | 16.658 | 0.341889 | 3.122524 | 47.985 |
| Full log-matrix MLP | 45.002 | 17.007 | 0.360738 | 3.140882 | 46.661 |

| Candidate minus control | Region mean pp | Region paired seeds pp | Center mean pp | Region scene means pp |
| --- | ---: | --- | ---: | --- |
| R58 scalar Gaussian | +4.324 | +4.054/+4.378/+4.539 | +1.548 | +3.763/+0.760/+4.700/+6.237/+6.159 |
| R58 Gaussian metric | +2.714 | +3.272/+1.248/+3.622 | +1.017 | +1.378/-0.529/+3.731/+4.584/+4.406 |
| R58 sinusoidal distance | +2.119 | +2.482/+0.411/+3.464 | +0.238 | +1.970/+2.180/+3.052/+2.203/+1.188 |
| R58 distance + angle | +2.846 | +3.239/+0.256/+5.042 | +0.821 | +2.356/+0.925/+4.360/+3.340/+3.246 |
| No explicit PE | +14.841 | +14.883/+13.932/+15.707 | +5.281 | +11.970/+7.763/+15.843/+17.466/+21.159 |
| Distance MLP | +2.081 | +2.359/+0.907/+2.978 | +0.451 | +0.874/+0.991/+4.603/+3.127/+0.812 |
| Local descriptor MLP | +2.929 | +3.034/+1.903/+3.851 | +0.833 | +2.222/+2.213/+6.468/+2.843/+0.899 |
| Angle + descriptor MLP | +2.760 | +3.642/+2.554/+2.083 | +1.107 | +2.430/+2.478/+4.724/+2.719/+1.449 |
| Full log-matrix MLP | +3.780 | +3.344/+2.865/+5.130 | +0.757 | +0.252/+0.826/+5.814/+5.775/+6.232 |

Scene order: harvard_c11/hv_c11_2, hotel_umd/maryland_hotel3, mit_76_studyroom/76-1studyroom2, mit_dorm_next_sj/dorm_next_sj_oct_30_2012_scan1_erika, mit_lab_hj/lab_hj_tea_nov_2_2012_scan1_erika.
Seed and scene differences are descriptive;nearby frames are correlated. No query-level confidence interval is presented.

| Scene | Pairs | Valid queries | Eligible fraction |
| --- | ---: | ---: | ---: |
| harvard_c11/hv_c11_2 | 96 | 2250 | 0.183105 |
| hotel_umd/maryland_hotel3 | 96 | 1009 | 0.082113 |
| mit_76_studyroom/76-1studyroom2 | 96 | 1376 | 0.111979 |
| mit_dorm_next_sj/dorm_next_sj_oct_30_2012_scan1_erika | 96 | 1876 | 0.152669 |
| mit_lab_hj/lab_hj_tea_nov_2_2012_scan1_erika | 96 | 2300 | 0.187174 |

Excluded by native input availability: 0; zero-valid pairs: 117.
Sparse supports and eligible populations differ from base;compare matched methods within each condition.

## Target adaptation and checks

Each role starts from its OWN selected source checkpoint. Actual target initial backbone weights are not shared across roles.
Same target data/order/budget/optimizer/selection;this compares complete source-plus-target pipelines,not an isolated reset of target PE.

| Method | PE parameters | Selected target epochs | Target updates | Target fit seconds |
| --- | ---: | --- | --- | ---: |
| R58 scalar Gaussian | 9216 | 42/27/30 | 1080/1080/1080 | 123.78 |
| R58 Gaussian metric | 9216 | 15/27/18 | 1080/1080/1080 | 107.97 |
| R58 sinusoidal distance | 9216 | 33/18/36 | 1080/1080/1080 | 105.15 |
| R58 distance + angle | 18432 | 33/27/48 | 1080/1080/1080 | 218.12 |
| Gaussian invariant MLP (candidate) | 10368 | 27/27/21 | 1080/1080/1080 | 112.18 |
| No explicit PE | 0 | 12/30/18 | 1080/1080/1080 | 97.63 |
| Distance MLP | 9408 | 30/30/21 | 1080/1080/1080 | 107.57 |
| Local descriptor MLP | 11520 | 27/33/24 | 1080/1080/1080 | 106.99 |
| Angle + descriptor MLP | 11616 | 27/24/21 | 1080/1080/1080 | 236.27 |
| Full log-matrix MLP | 10752 | 18/39/21 | 1080/1080/1080 | 107.76 |

NEW target fitting/selection cumulative 1323.42s. Shared-host timing is not a controlled speed benchmark.
Independent NumPy metric maximum difference 1.33e-15. Separate arithmetic by the same author,not another laboratory.
R59 sparse uncertainty and older classification/scalar-form failures remain evidence. No global goal or fresh-confirmation claim.

![Region comparisons](/home/liwt/GRoPE/results/round60_real_scene/attempt1/report1/comparison.png)

![Target validation curves](/home/liwt/GRoPE/results/round60_real_scene/attempt1/report1/learning_curves.png)
