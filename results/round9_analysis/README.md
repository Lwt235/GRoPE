# Round 9 independent staged RGB analysis

Two reused development scenes; given shared SfM cameras; no generalization confidence interval

Calibration selected each role's bandwidth before gate RGB was opened. All gate pixels are scored.

| Role | Selected method | Calibration MSE | Gate MSE | Gate PSNR from macro MSE |
|---|---|---:|---:|---:|
| full | full_h0.25 | 0.020094805 | 0.022208574 | 16.5348 |
| isotropic | isotropic_h0.25 | 0.021589113 | 0.022909748 | 16.3998 |
| fixed | fixed_h0.25 | 0.019902658 | 0.022581662 | 16.4624 |
| point | point_h0.25 | 0.019492259 | 0.022241047 | 16.5284 |
| bilinear | bilinear | 0.019456183 | 0.022259686 | 16.5248 |
| original_gs | original_gs | 0.016328053 | 0.022219161 | 16.5327 |

Primary control: `point`. Relative full MSE reduction: 0.1460%; scenes improved: 1/2; gate passed: **False**.

Limits:

- One independently fitted GS per scene; deterministic readouts are not seed replicates
- Shared full-image SfM cameras; GS initialization and fitting use context RGB only
- Four calibration and four gate views in each of two reused development scenes
- Gate best simple control is conservative comparison, not extra method tuning
- Failure stops this branch; passing sampling gate does not establish a novel or general GS method
