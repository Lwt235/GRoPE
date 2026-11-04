# R65: native optimized Gaussian parameters as Transformer PE

Scope: **INTERNAL DEVELOPMENT**. Five methods, three seeds, six conditions; all90 groups reported.
All roles receive the same center/log-covariance/opacity/DC content. No RGB parameter predictor.
Internal development split of official TRAIN; no published TEST reproduction or fresh confirmation.
Scientific interpretation and actual image review remain pending.

The fixed base development lead criterion is NOT MET.
This flag does not establish the full research goal or override earlier negative results.

## Base condition

| Method | Mean-class accuracy % | Overall accuracy % | CE | MACGS mCA % | ModelNet mCA % |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gaussian relation | 77.092 | 82.832 | 0.70246 | 74.019 | 80.165 |
| Direct MLP | 76.776 | 82.222 | 0.75136 | 73.176 | 80.375 |
| Distance MLP | 78.426 | 83.529 | 0.73671 | 75.867 | 80.986 |
| Center RoPE | 71.163 | 76.906 | 0.86051 | 66.022 | 76.304 |
| NoPE | 69.829 | 74.771 | 1.03220 | 64.961 | 74.696 |

## All fixed contrasts

Gaussian minus control, mean-class accuracy in percentage points; positive favors Gaussian.
Intervals resample classes after averaging paired seeds (10,000 samples); they are not independent-seed confidence intervals.

| Condition | Control | Mean gain pp | Seeds11/22/33 pp | Class bootstrap95% pp | Positive class means | Fixed criterion |
| --- | --- | ---: | --- | --- | ---: | --- |
| base | Direct MLP | +0.316 | +3.412/-1.187/-1.277 | [-2.417, +2.639] | 27/60 | False |
| base | Distance MLP | -1.334 | +0.044/-3.528/-0.519 | [-3.519, +0.663] | 19/60 | False |
| base | Center RoPE | +5.929 | +5.843/+7.029/+4.916 | [+3.617, +8.328] | 41/60 | True |
| base | NoPE | +7.263 | +7.972/+5.451/+8.367 | [+4.612, +10.067] | 40/60 | True |
| sparse256 | Direct MLP | -0.409 | +2.051/-0.043/-3.236 | [-2.952, +2.081] | 29/60 | False |
| sparse256 | Distance MLP | -1.917 | -0.055/-4.174/-1.523 | [-4.043, +0.208] | 21/60 | False |
| sparse256 | Center RoPE | +4.998 | +7.419/+5.311/+2.264 | [+2.647, +7.465] | 40/60 | True |
| sparse256 | NoPE | +7.762 | +8.837/+6.650/+7.798 | [+4.653, +10.951] | 41/60 | True |
| dense1024 | Direct MLP | -0.086 | -0.002/+1.357/-1.615 | [-2.041, +1.515] | 27/60 | False |
| dense1024 | Distance MLP | -1.038 | -0.699/-0.672/-1.743 | [-2.734, +0.459] | 23/60 | False |
| dense1024 | Center RoPE | +4.923 | +3.216/+8.756/+2.796 | [+2.506, +7.347] | 39/60 | True |
| dense1024 | NoPE | +7.235 | +7.921/+7.212/+6.572 | [+4.563, +10.018] | 39/60 | True |
| jitter02 | Direct MLP | +0.014 | +3.412/-1.467/-1.903 | [-2.452, +2.138] | 25/60 | False |
| jitter02 | Distance MLP | -1.542 | +0.234/-3.638/-1.222 | [-3.720, +0.583] | 20/60 | False |
| jitter02 | Center RoPE | +5.301 | +5.988/+6.330/+3.583 | [+3.009, +7.658] | 39/60 | True |
| jitter02 | NoPE | +7.161 | +8.434/+4.939/+8.110 | [+4.541, +9.924] | 41/60 | True |
| tilt | Direct MLP | +0.742 | +4.174/-0.622/-1.328 | [-1.730, +2.908] | 30/60 | False |
| tilt | Distance MLP | -1.473 | -0.657/-2.684/-1.078 | [-3.586, +0.491] | 19/60 | False |
| tilt | Center RoPE | +5.577 | +5.164/+6.461/+5.106 | [+3.205, +7.954] | 39/60 | True |
| tilt | NoPE | +8.007 | +9.446/+6.552/+8.023 | [+5.392, +10.746] | 44/60 | True |
| permuted | Direct MLP | +0.316 | +3.412/-1.187/-1.277 | [-2.417, +2.639] | 27/60 | False |
| permuted | Distance MLP | -1.334 | +0.044/-3.528/-0.519 | [-3.519, +0.663] | 19/60 | False |
| permuted | Center RoPE | +5.929 | +5.843/+7.029/+4.916 | [+3.617, +8.328] | 41/60 | True |
| permuted | NoPE | +7.263 | +7.972/+5.451/+8.367 | [+4.612, +10.067] | 40/60 | True |

## Per-seed results and training cost

| Method | Seed | mCA % | OA % | CE | Selected epoch | Params / PE params | Peak GiB | Elapsed h |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Gaussian relation | 11 | 77.875 | 83.399 | 0.69746 | 100 | 352796 / 10368 | 12.259 | 2.857 |
| Direct MLP | 11 | 74.463 | 80.654 | 0.78569 | 95 | 353276 / 10848 | 12.337 | 2.553 |
| Distance MLP | 11 | 77.831 | 82.484 | 0.75504 | 100 | 351836 / 9408 | 12.104 | 2.249 |
| Center RoPE | 11 | 72.032 | 77.516 | 0.84241 | 115 | 342428 / 0 | 1.629 | 0.452 |
| NoPE | 11 | 69.903 | 74.641 | 1.06392 | 115 | 342428 / 0 | 1.604 | 0.396 |
| Gaussian relation | 22 | 76.013 | 82.353 | 0.70361 | 90 | 352796 / 10368 | 12.259 | 2.797 |
| Direct MLP | 22 | 77.200 | 82.876 | 0.73223 | 110 | 353276 / 10848 | 12.337 | 2.554 |
| Distance MLP | 22 | 79.542 | 84.314 | 0.72904 | 105 | 351836 / 9408 | 12.104 | 2.252 |
| Center RoPE | 22 | 68.985 | 75.033 | 0.90370 | 120 | 342428 / 0 | 1.629 | 0.444 |
| NoPE | 22 | 70.562 | 75.556 | 0.98291 | 110 | 342428 / 0 | 1.604 | 0.397 |
| Gaussian relation | 33 | 77.387 | 82.745 | 0.70632 | 90 | 352796 / 10368 | 12.259 | 2.797 |
| Direct MLP | 33 | 78.664 | 83.137 | 0.73616 | 105 | 353276 / 10848 | 12.337 | 2.499 |
| Distance MLP | 33 | 77.907 | 83.791 | 0.72604 | 120 | 351836 / 9408 | 12.104 | 2.306 |
| Center RoPE | 33 | 72.471 | 78.170 | 0.83541 | 120 | 342428 / 0 | 1.629 | 0.469 |
| NoPE | 33 | 69.021 | 74.118 | 1.04977 | 110 | 342428 / 0 | 1.604 | 0.397 |

Peak allocated memory and elapsed time are from shared-host fits, not a controlled speed comparison.
Complete per-class/seed/condition accuracies are in class_accuracy.csv; every aggregate and domain metric is in tables.json.

## Data conversion and interpretation

The covariance floor is1e-8 I. The table describes its trace contribution; it is not a thin-axis distortion bound.

| Split | Objects | Sampled with replacement | Median of object median floor fractions | Largest object maximum floor fraction |
| --- | ---: | ---: | ---: | ---: |
| fit | 6227 | 0 | 1.69493e-05 | 0.966579 |
| val | 765 | 0 | 1.70107e-05 | 0.614297 |
| dev_eval | 765 | 0 | 1.76434e-05 | 0.611633 |

A gain over center-only PE controls alone does not establish a Gaussian advantage over the direct MLP.
A source-specific or density-specific gain retains that scope. Permutation is an implementation diagnostic.
Tilt follows full SO(3) training and therefore is not unseen-rotation-distribution transfer.
ModelNet objects may overlap earlier development objects. Nonidentical MACGS physical instances are not independently grouped.
R62/R64 RGB failures and earlier direct-transfer/sparse failures remain unresolved; this experiment cannot erase them.
Any fresh confirmation requires a prospective unused-data plan after scientific review; no such data is read here.

Independent metric maximum discrepancy: 4.441e-16.
Selected GPU replay maximum logit discrepancy: 0; CPU replay: 7.629e-06.

![primary_contrasts.png](/home/liwt/GRoPE/results/round65_actual_gs/attempt1/report1/primary_contrasts.png)

![condition_gains.png](/home/liwt/GRoPE/results/round65_actual_gs/attempt1/report1/condition_gains.png)

![validation_curves.png](/home/liwt/GRoPE/results/round65_actual_gs/attempt1/report1/validation_curves.png)

![class_gains.png](/home/liwt/GRoPE/results/round65_actual_gs/attempt1/report1/class_gains.png)

