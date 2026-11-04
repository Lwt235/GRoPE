# R69 fixed relation-column decomposition

Nine new paired fits, thirty identity-verified reused R68 evaluation groups. Used development only.
Descriptive compact candidate: relative5. This is not a formal noninferiority conclusion.

| Condition | Method | Region R@1 % | Center PCK % | Soft EPE | CE |
|---|---|---:|---:|---:|---:|
| base | distance_mlp | 36.6698 | 12.5278 | 0.4276 | 3.0811 |
| base | direct_mlp | 35.6419 | 12.1154 | 0.4318 | 3.0998 |
| base | normal_scale | 36.8162 | 12.6303 | 0.4293 | 3.0444 |
| base | gaussian_relation | 40.4364 | 13.9754 | 0.3956 | 2.9652 |
| base | shuffled_gaussian | 35.1379 | 11.8571 | 0.4341 | 3.0912 |
| base | relative5 | 40.6283 | 14.2208 | 0.4016 | 2.9748 |
| base | metric3 | 38.0663 | 13.1434 | 0.4144 | 3.0491 |
| base | spectrum7 | 37.3627 | 12.7094 | 0.4206 | 3.0535 |
| sparse256 | distance_mlp | 32.8734 | 10.8464 | 0.4467 | 3.1771 |
| sparse256 | direct_mlp | 32.7698 | 10.9473 | 0.4505 | 3.2050 |
| sparse256 | normal_scale | 33.9761 | 11.3655 | 0.4433 | 3.1196 |
| sparse256 | gaussian_relation | 37.3505 | 12.6045 | 0.4083 | 3.0360 |
| sparse256 | shuffled_gaussian | 32.1921 | 10.6188 | 0.4522 | 3.1837 |
| sparse256 | relative5 | 37.5762 | 12.8276 | 0.4134 | 3.0352 |
| sparse256 | metric3 | 35.4110 | 11.9610 | 0.4257 | 3.1145 |
| sparse256 | spectrum7 | 34.0371 | 11.1905 | 0.4410 | 3.1613 |

| Condition | Candidate minus control | Paired region gains pp | Mean pp | Scene95% pp | Center pp |
|---|---|---|---:|---|---:|
| base | relative5 - gaussian_relation | -0.657 / -0.368 / +1.601 | +0.192 | [-0.436, +0.820] | +0.245 |
| base | relative5 - distance_mlp | +2.264 / +2.102 / +7.509 | +3.958 | [+3.219, +4.907] | +1.693 |
| base | relative5 - direct_mlp | +5.917 / +2.274 / +6.769 | +4.986 | [+4.225, +5.748] | +2.105 |
| sparse256 | relative5 - gaussian_relation | -1.256 / -0.130 / +2.064 | +0.226 | [-0.074, +0.522] | +0.223 |
| sparse256 | relative5 - distance_mlp | +2.802 / +3.407 / +7.900 | +4.703 | [+3.869, +5.572] | +1.981 |
| sparse256 | relative5 - direct_mlp | +5.298 / +2.445 / +6.676 | +4.806 | [+4.086, +5.386] | +1.880 |
| base | metric3 - gaussian_relation | -0.979 / -3.933 / -2.198 | -2.370 | [-3.408, -1.179] | -0.832 |
| base | metric3 - distance_mlp | +1.941 / -1.463 / +3.711 | +1.396 | [+0.469, +2.347] | +0.616 |
| base | metric3 - direct_mlp | +5.594 / -1.291 / +2.970 | +2.424 | [+1.714, +3.135] | +1.028 |
| sparse256 | metric3 - gaussian_relation | -1.080 / -3.179 / -1.559 | -1.939 | [-2.815, -0.756] | -0.644 |
| sparse256 | metric3 - distance_mlp | +2.978 / +0.358 / +4.277 | +2.538 | [+1.399, +3.641] | +1.115 |
| sparse256 | metric3 - direct_mlp | +5.475 / -0.603 / +3.052 | +2.641 | [+2.063, +3.238] | +1.014 |
| base | spectrum7 - gaussian_relation | -4.719 / -1.846 / -2.656 | -3.074 | [-3.657, -2.359] | -1.266 |
| base | spectrum7 - distance_mlp | -1.798 / +0.625 / +3.253 | +0.693 | [+0.263, +1.172] | +0.182 |
| base | spectrum7 - direct_mlp | +1.854 / +0.796 / +2.512 | +1.721 | [+1.135, +2.492] | +0.594 |
| sparse256 | spectrum7 - gaussian_relation | -5.478 / -1.738 / -2.724 | -3.313 | [-3.785, -2.785] | -1.414 |
| sparse256 | spectrum7 - distance_mlp | -1.420 / +1.799 / +3.111 | +1.164 | [+0.957, +1.428] | +0.344 |
| sparse256 | spectrum7 - direct_mlp | +1.077 / +0.838 / +1.887 | +1.267 | [+0.602, +2.297] | +0.243 |

No search or additional fitting is authorized by a failed simplification. Retain full11 if required. Known original region membership under thinning; unchanged20422 valid out of61440 queries. Scene intervals condition on the selected three seeds. Scientific and visual review pending.

![Comparison](comparison.png)
