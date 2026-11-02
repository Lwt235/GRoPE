# R50 local-frame development results

Controlled procedural matching; three paired seeds. No real-task confirmation.

All methods receive the same RGB, center/shape/camera information.

## Train/evaluate with oracle geometry

| role | base AUC % | slant | zoom | wide | mean AUC % | base PCK8 % | base EPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rope | 77.529 | 69.687 | 29.211 | 59.501 | 58.982 | 100.000 | 1.798 |
| mlp | 77.510 | 69.668 | 31.385 | 59.680 | 59.561 | 100.000 | 1.799 |
| hrope | 77.824 | 70.532 | 25.020 | 63.998 | 59.343 | 100.000 | 1.774 |
| gs_frame | 77.764 | 69.985 | 18.662 | 62.832 | 57.311 | 100.000 | 1.779 |
| gs_iso | 77.755 | 69.708 | 18.821 | 62.950 | 57.309 | 100.000 | 1.780 |
| affine | 77.773 | 70.004 | 19.347 | 62.853 | 57.494 | 100.000 | 1.778 |
| frame_identity | 77.729 | 69.526 | 17.748 | 62.242 | 56.811 | 100.000 | 1.782 |
| gaussian_bias | 77.257 | 69.293 | 9.484 | 59.031 | 53.766 | 99.993 | 1.819 |

Fixed development guide versus MLP and affine: False.

## Train/evaluate with noisy geometry

| role | base AUC % | slant | zoom | wide | mean AUC % | base PCK8 % | base EPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rope | 77.529 | 70.100 | 30.695 | 59.872 | 59.549 | 100.000 | 1.798 |
| mlp | 77.538 | 70.110 | 31.851 | 60.023 | 59.880 | 100.000 | 1.797 |
| hrope | 77.792 | 71.053 | 24.855 | 63.876 | 59.394 | 100.000 | 1.777 |
| gs_frame | 77.731 | 70.549 | 17.342 | 62.566 | 57.047 | 100.000 | 1.782 |
| gs_iso | 77.722 | 70.345 | 18.307 | 62.607 | 57.245 | 100.000 | 1.782 |
| affine | 77.732 | 70.751 | 17.855 | 62.748 | 57.272 | 100.000 | 1.781 |
| frame_identity | 77.726 | 70.485 | 18.724 | 62.156 | 57.273 | 100.000 | 1.782 |
| gaussian_bias | 77.256 | 69.559 | 9.336 | 58.858 | 53.752 | 100.000 | 1.819 |

Fixed development guide versus MLP and affine: False.

## Full-GS paired AUC differences

| geometry | control | scope | delta pp | positive seeds | scene-bootstrap interval pp |
| --- | --- | --- | ---: | ---: | --- |
| oracle | rope | base | 0.234 | 3/3 | [0.159, 0.312] |
| oracle | rope | four conditions | -1.671 | 0/3 | [-1.984, -1.383] |
| oracle | mlp | base | 0.253 | 3/3 | [0.173, 0.337] |
| oracle | mlp | four conditions | -2.250 | 0/3 | [-2.583, -1.923] |
| oracle | hrope | base | -0.060 | 0/3 | [-0.112, -0.017] |
| oracle | hrope | four conditions | -2.033 | 0/3 | [-2.201, -1.859] |
| oracle | gs_iso | base | 0.009 | 2/3 | [-0.005, 0.023] |
| oracle | gs_iso | four conditions | 0.002 | 1/3 | [-0.101, 0.101] |
| oracle | affine | base | -0.009 | 1/3 | [-0.030, 0.014] |
| oracle | affine | four conditions | -0.184 | 0/3 | [-0.303, -0.064] |
| oracle | frame_identity | base | 0.034 | 2/3 | [0.004, 0.069] |
| oracle | frame_identity | four conditions | 0.500 | 2/3 | [0.366, 0.637] |
| oracle | gaussian_bias | base | 0.507 | 3/3 | [0.349, 0.688] |
| oracle | gaussian_bias | four conditions | 3.545 | 3/3 | [3.299, 3.787] |
| noisy | rope | base | 0.202 | 3/3 | [0.124, 0.282] |
| noisy | rope | four conditions | -2.502 | 0/3 | [-2.836, -2.177] |
| noisy | mlp | base | 0.193 | 3/3 | [0.120, 0.272] |
| noisy | mlp | four conditions | -2.833 | 0/3 | [-3.176, -2.508] |
| noisy | hrope | base | -0.061 | 0/3 | [-0.115, -0.011] |
| noisy | hrope | four conditions | -2.347 | 0/3 | [-2.526, -2.169] |
| noisy | gs_iso | base | 0.009 | 3/3 | [-0.011, 0.032] |
| noisy | gs_iso | four conditions | -0.198 | 1/3 | [-0.289, -0.109] |
| noisy | affine | base | -0.001 | 2/3 | [-0.019, 0.017] |
| noisy | affine | four conditions | -0.224 | 0/3 | [-0.324, -0.128] |
| noisy | frame_identity | base | 0.005 | 1/3 | [-0.020, 0.030] |
| noisy | frame_identity | four conditions | -0.226 | 1/3 | [-0.352, -0.103] |
| noisy | gaussian_bias | base | 0.475 | 3/3 | [0.324, 0.645] |
| noisy | gaussian_bias | four conditions | 3.295 | 3/3 | [3.085, 3.500] |

Continuous results and all cross-regime/shape-stratum metrics remain in summary.json.
A failed guide is not a theorem of infeasibility. A synthetic pass is not evidence of general real-task utility.
