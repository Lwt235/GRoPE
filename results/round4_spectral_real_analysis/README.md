# Round 4: retrained covariance controls and spectral family

Descriptive extension on already examined real cameras. Five matched training seeds; both visual inputs; all four scenes and both GS partitions are retained. Primary group is figurines + temple.

All 130 checkpoints replayed old validation before extension evaluation. Source snapshots, complete checkpoint-selection histories, training/validation data hashes and evaluation NPZ hashes are verified. Gaussian covariance scale is selected only by five-seed mean Teatime validation CE.

| Visual input | GS partition | Method | CE | JS | Top1 target mass |
|---|---|---|---:|---:|---:|
| raw | 0 | gaussian_full | 1.198967 | 0.160006 | 0.603211 |
| raw | 0 | gaussian_qonly | 1.252299 | 0.165956 | 0.595125 |
| raw | 0 | gaussian_konly | 1.199366 | 0.160217 | 0.603130 |
| raw | 0 | gaussian_none | 1.253044 | 0.165205 | 0.595719 |
| raw | 0 | gaussian_trace | 1.226835 | 0.165275 | 0.600264 |
| raw | 0 | gaussian_area | 1.219031 | 0.163929 | 0.599334 |
| raw | 0 | gaussian25_full | 1.217268 | 0.162618 | 0.599867 |
| raw | 0 | gaussian50_full | 1.206922 | 0.161322 | 0.601829 |
| raw | 0 | gaussian200_full | 1.192738 | 0.160798 | 0.601023 |
| raw | 0 | resolvent_full | 1.200583 | 0.160792 | 0.602808 |
| raw | 0 | resolvent_none | 1.254604 | 0.165514 | 0.596327 |
| raw | 0 | resolvent_trace | 1.220172 | 0.164017 | 0.599584 |
| raw | 0 | resolvent_area | 1.225236 | 0.164167 | 0.599705 |
| raw | 0 | gaussian_selected | 1.192738 | 0.160798 | 0.601023 |
| raw | 1 | gaussian_full | 1.180707 | 0.158400 | 0.604137 |
| raw | 1 | gaussian_qonly | 1.245192 | 0.166036 | 0.599258 |
| raw | 1 | gaussian_konly | 1.183833 | 0.159121 | 0.603582 |
| raw | 1 | gaussian_none | 1.247953 | 0.165726 | 0.599163 |
| raw | 1 | gaussian_trace | 1.208307 | 0.163894 | 0.601356 |
| raw | 1 | gaussian_area | 1.210657 | 0.163618 | 0.601536 |
| raw | 1 | gaussian25_full | 1.211497 | 0.162468 | 0.600977 |
| raw | 1 | gaussian50_full | 1.196288 | 0.160566 | 0.602828 |
| raw | 1 | gaussian200_full | 1.169419 | 0.158637 | 0.604833 |
| raw | 1 | resolvent_full | 1.193598 | 0.160398 | 0.602665 |
| raw | 1 | resolvent_none | 1.249840 | 0.166085 | 0.598904 |
| raw | 1 | resolvent_trace | 1.213861 | 0.163970 | 0.601943 |
| raw | 1 | resolvent_area | 1.219809 | 0.164364 | 0.601412 |
| raw | 1 | gaussian_selected | 1.169419 | 0.158637 | 0.604833 |
| frozen | 0 | gaussian_full | 1.172637 | 0.149540 | 0.609037 |
| frozen | 0 | gaussian_qonly | 1.229062 | 0.152891 | 0.606890 |
| frozen | 0 | gaussian_konly | 1.178046 | 0.149302 | 0.609717 |
| frozen | 0 | gaussian_none | 1.231050 | 0.152450 | 0.607661 |
| frozen | 0 | gaussian_trace | 1.203838 | 0.153953 | 0.604633 |
| frozen | 0 | gaussian_area | 1.205012 | 0.151673 | 0.609715 |
| frozen | 0 | gaussian25_full | 1.208651 | 0.150316 | 0.610210 |
| frozen | 0 | gaussian50_full | 1.190805 | 0.149636 | 0.609922 |
| frozen | 0 | gaussian200_full | 1.168928 | 0.150998 | 0.605579 |
| frozen | 0 | resolvent_full | 1.183657 | 0.149035 | 0.610057 |
| frozen | 0 | resolvent_none | 1.232313 | 0.152240 | 0.607922 |
| frozen | 0 | resolvent_trace | 1.204765 | 0.151371 | 0.609200 |
| frozen | 0 | resolvent_area | 1.209644 | 0.151001 | 0.608411 |
| frozen | 0 | gaussian_selected | 1.168928 | 0.150998 | 0.605579 |
| frozen | 1 | gaussian_full | 1.154558 | 0.148310 | 0.610193 |
| frozen | 1 | gaussian_qonly | 1.224612 | 0.153673 | 0.608472 |
| frozen | 1 | gaussian_konly | 1.163385 | 0.148650 | 0.610692 |
| frozen | 1 | gaussian_none | 1.228924 | 0.153584 | 0.608937 |
| frozen | 1 | gaussian_trace | 1.184686 | 0.152944 | 0.605105 |
| frozen | 1 | gaussian_area | 1.197876 | 0.152097 | 0.608854 |
| frozen | 1 | gaussian25_full | 1.203556 | 0.150705 | 0.612170 |
| frozen | 1 | gaussian50_full | 1.179682 | 0.149301 | 0.611621 |
| frozen | 1 | gaussian200_full | 1.146441 | 0.149146 | 0.608843 |
| frozen | 1 | resolvent_full | 1.176025 | 0.149050 | 0.611578 |
| frozen | 1 | resolvent_none | 1.230022 | 0.153342 | 0.608845 |
| frozen | 1 | resolvent_trace | 1.198339 | 0.151814 | 0.608580 |
| frozen | 1 | resolvent_area | 1.205301 | 0.151795 | 0.609151 |
| frozen | 1 | gaussian_selected | 1.146441 | 0.149146 | 0.608843 |

Differences and 95% paired seed/query-camera intervals are in summary.json. Negative candidate-minus-control CE favors the candidate. This camera bootstrap treats scenes as fixed strata, and five model seeds are not five independent datasets. Known GS/SfM may use evaluation photos; labels are conditional sparse-track transfer, not dense occlusion or NVS truth. The joint replacement decision additionally requires the separately analyzed point/noise guardrails.
