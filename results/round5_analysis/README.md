# Round 5 paired seed/place analysis

Validation-selected position baseline: **point**. Primary practical gate passed: **False**.

Each place is weighted equally after aggregating its valid queries. Intervals resample model seeds and physical places as crossed factors; tokens and overlapping pairs are not treated as independent datasets. Only five held-out places support the default interval, so population generalization remains limited.

| Method | CE | PCK8 | PCK4 | Argmax EPE | Soft EPE |
|---|---:|---:|---:|---:|---:|
| normalized | 1.584159 | 0.922812 | 0.574095 | 4.713740 | 3.757378 |
| expected | 1.585652 | 0.922882 | 0.578712 | 4.715452 | 3.739873 |
| fixed | 1.589427 | 0.922345 | 0.572679 | 4.710780 | 3.721716 |
| isotropic | 1.590871 | 0.922688 | 0.573519 | 4.726941 | 3.826074 |
| point | 1.583625 | 0.923508 | 0.575500 | 4.684555 | 3.695484 |
| resolvent | 1.581133 | 0.923151 | 0.574273 | 4.698740 | 3.774216 |

All seed/place metrics, candidate-minus-control differences, 95% crossed intervals, leave-place/group-out effects and coverage counts are in summary.json. Leave-out analyses are descriptive sensitivity checks, not additional confirmatory tests. CE worsening is an empirical 2% guardrail, not a formal noninferiority test.
