# Full R65 source-error diagnostic

Output-only decomposition of the original90 groups. Primary results remain the original60-way classification. Source-known scoring uses the true source to restrict labels and is diagnostic only. Different source/category sets prevent causal identification of a source shortcut.

| Method | Base mCA % | Cross-source error % | Within-source error % | Source-known mCA % |
|---|---:|---:|---:|---:|
| gaussian_relation | 77.092 | 0.617 | 22.291 | 77.138 |
| direct_mlp | 76.776 | 0.648 | 22.576 | 76.776 |
| distance_mlp | 78.426 | 0.556 | 21.018 | 78.658 |
| center_rope | 71.163 | 0.139 | 28.698 | 71.163 |
| nope | 69.829 | 1.065 | 29.107 | 70.106 |

Gaussian minus control: gain = cross-source error reduction + within-source error reduction. Report signed percentage points, not a fraction of a possibly small net gain.

| Control | Seed | Gain pp | Cross-source contribution pp | Within-source contribution pp | Source-known gain pp |
|---|---:|---:|---:|---:|---:|
| direct_mlp | 11 | +3.412 | +0.000 | +3.412 | +3.551 |
| direct_mlp | 22 | -1.187 | +0.093 | -1.280 | -1.187 |
| direct_mlp | 33 | -1.277 | +0.000 | -1.277 | -1.277 |
| distance_mlp | 11 | +0.044 | -0.833 | +0.878 | +0.183 |
| distance_mlp | 22 | -3.528 | +0.509 | -4.038 | -3.945 |
| distance_mlp | 33 | -0.519 | +0.139 | -0.658 | -0.797 |
| center_rope | 11 | +5.843 | -0.833 | +6.676 | +5.981 |
| center_rope | 22 | +7.029 | -0.185 | +7.214 | +7.029 |
| center_rope | 33 | +4.916 | -0.417 | +5.333 | +4.916 |
| nope | 11 | +7.972 | +0.278 | +7.694 | +8.111 |
| nope | 22 | +5.451 | +1.204 | +4.247 | +4.896 |
| nope | 33 | +8.367 | -0.139 | +8.506 | +8.089 |

All source/class/seed/condition records are in result.json. Scientific interpretation is pending. This diagnostic cannot establish effectiveness when the original strong-control comparison fails.
