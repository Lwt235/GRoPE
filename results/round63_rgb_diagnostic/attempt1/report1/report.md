# R63: fixed-weight RGB geometry diagnostic

48 old fit pairs / 64 old val pairs;3seeds;54prediction groups;no new fitting.
All original branches replay R62 selected weights. No old_eval or confirmation inputs.
Observed substitutions add privileged information and cause distribution shifts.
These are development diagnostics,not fair trained RGB competitors or causal proof.
Scientific interpretation and actual image review pending at generation.

## fit

| Model / intervention | Soft EPE px | PCK8 % | CE |
| --- | ---: | ---: | ---: |
| Predicted Gaussian baseline | 1.41782 | 98.841 | 1.15960 |
| Replace center | 20.25401 | 35.871 | 5.89203 |
| Replace shape | 8.02259 | 61.849 | 3.43277 |
| Replace both | 19.93356 | 34.801 | 6.03467 |
| Initial fixed shape | 2.60789 | 93.948 | 1.42394 |
| Fixed sphere | 2.53002 | 94.762 | 1.41346 |
| Zero added relation | 18.99084 | 34.449 | 4.88417 |
| Predicted center baseline | 1.57575 | 98.570 | 1.18891 |
| Observed full baseline (privileged) | 1.25153 | 99.455 | 1.13045 |

Paired intervention minus unmodified predicted-full;negative EPE is favorable.

| Intervention | EPE mean px | Seeds11/22/33 | Favorable seeds/scenes | PCK8 delta pp |
| --- | ---: | --- | --- | ---: |
| Replace center | +18.83618 | +18.82592/+19.12403/+18.55861 | 0/3;0/3 | -62.970 |
| Replace shape | +6.60477 | +4.81403/+6.52973/+8.47055 | 0/3;0/3 | -36.992 |
| Replace both | +18.51573 | +18.44149/+19.22873/+17.87698 | 0/3;0/3 | -64.040 |
| Initial fixed shape | +1.19006 | +1.47215/+0.26255/+1.83548 | 0/3;0/3 | -4.893 |
| Fixed sphere | +1.11219 | +1.12754/+1.36686/+0.84218 | 0/3;0/3 | -4.079 |
| Zero added relation | +17.57301 | +19.75435/+17.87134/+15.09335 | 0/3;0/3 | -64.392 |

## val

| Model / intervention | Soft EPE px | PCK8 % | CE |
| --- | ---: | ---: | ---: |
| Predicted Gaussian baseline | 3.08519 | 90.988 | 1.54851 |
| Replace center | 26.44074 | 23.847 | 7.58879 |
| Replace shape | 10.37488 | 49.176 | 4.27928 |
| Replace both | 25.23301 | 24.163 | 7.28214 |
| Initial fixed shape | 3.98137 | 85.433 | 1.78608 |
| Fixed sphere | 4.24490 | 83.867 | 1.94337 |
| Zero added relation | 28.43777 | 18.959 | 6.86148 |
| Predicted center baseline | 3.13941 | 90.811 | 1.55259 |
| Observed full baseline (privileged) | 2.67133 | 93.312 | 1.49110 |

Paired intervention minus unmodified predicted-full;negative EPE is favorable.

| Intervention | EPE mean px | Seeds11/22/33 | Favorable seeds/scenes | PCK8 delta pp |
| --- | ---: | --- | --- | ---: |
| Replace center | +23.35555 | +23.51698/+21.97904/+24.57063 | 0/3;0/1 | -67.140 |
| Replace shape | +7.28969 | +5.11902/+6.91346/+9.83659 | 0/3;0/1 | -41.812 |
| Replace both | +22.14782 | +22.31925/+21.95286/+22.17137 | 0/3;0/1 | -66.825 |
| Initial fixed shape | +0.89618 | +0.72227/+0.24004/+1.72624 | 0/3;0/1 | -5.555 |
| Fixed sphere | +1.15971 | +0.86031/+1.96001/+0.65883 | 0/3;0/1 | -7.121 |
| Zero added relation | +25.35259 | +28.27403/+26.38025/+21.40349 | 0/3;0/1 | -72.029 |

## Geometry against sensor-derived support

Entries below average the three seed-specific medians;not a pooled median.
Center errors use known native centers;shape errors use patches with>=16/64samples.
Sensor support is not a ground-truth learned GS parameterization.

| Split | Model | Layer | Center median m | Shape log median | Predicted depth median m | Observed depth median m |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| fit | predicted_full | 1 | 0.82068 | 4.49606 | 2.29205 | 2.50300 |
| fit | predicted_full | 2 | 0.71939 | 3.97459 | 2.43216 | 2.50300 |
| fit | predicted_point | 1 | 0.81531 | 4.11668 | 2.28896 | 2.50300 |
| fit | predicted_point | 2 | 0.71250 | 4.11668 | 2.32485 | 2.50300 |
| val | predicted_full | 1 | 1.07465 | 3.89996 | 2.25982 | 3.17800 |
| val | predicted_full | 2 | 0.85293 | 3.39413 | 2.49103 | 3.17800 |
| val | predicted_point | 1 | 1.07115 | 3.49128 | 2.25180 | 3.17800 |
| val | predicted_point | 2 | 0.91395 | 3.49128 | 2.38252 | 3.17800 |

## Observed covariance range

Cholesky coordinate clipping is not an optimal covariance projection. Tiny near-floor changes are not a material bottleneck.

| Split / stratum | Token exposures | Any coordinate outside % | Relative Frobenius median / p90 | Log change median |
| --- | ---: | ---: | --- | ---: |
| fit/all | 41472 | 42.388 | 8.61142e-09 / 0.832647 | 7.666e-08 |
| fit/native_center | 33893 | 47.700 | 1.36124e-08 / 0.857092 | 8.58943e-08 |
| fit/patch16 | 36713 | 46.730 | 1.26823e-08 / 0.856593 | 8.45738e-08 |
| fit/empty_patch | 3870 | 0.000 | 1.57205e-09 / 1.57205e-09 | 2.72287e-09 |
| val/all | 55296 | 32.547 | 5.01098e-09 / 0.722306 | 3.97859e-08 |
| val/native_center | 47951 | 35.342 | 5.61533e-09 / 0.76394 | 4.75966e-08 |
| val/patch16 | 51150 | 34.616 | 5.44785e-09 / 0.75746 | 4.52112e-08 |
| val/empty_patch | 3355 | 0.000 | 1.57205e-09 / 1.57205e-09 | 2.72287e-09 |

Raw coordinate statistics,coverage strata,saturation and scene details are retained in analysis/summary.json.
Independent metric max difference:3.42e-08.
Original val logit max difference:0;CE max difference:0.
All model weights unchanged. No global goal completion.

![Interventions](/home/liwt/GRoPE/results/round63_rgb_diagnostic/attempt1/report1/interventions.png)

![Geometry](/home/liwt/GRoPE/results/round63_rgb_diagnostic/attempt1/report1/geometry.png)
