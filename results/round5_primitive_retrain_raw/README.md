# Legacy GS covariance: paired full versus center-only retraining

Both arms train on identical old Teatime RGB/features/means/cameras/labels and batches, five seeds, 500 steps, validation every20 including0. The old validation-selected Gaussian scale2 and common resolution .25 are fixed for both arms. The complete full arm reproduces every old checkpoint tensor, selected step and validation metric exactly. Center-only removes internal primitive covariance on both query and key before the common scale2; each arm selects its own checkpoint by its corresponding validation CE.

This is a descriptive mechanism audit on previously inspected scenes/conditions. It addresses the frozen-input-swap distribution-mismatch limitation. It is not a new independent confirmation or a globally scale-retuned center-only comparison.

| parity | group | full CE | center CE | full-center | paired seed/camera 95% CI |
|---|---|---:|---:|---:|---|
| 0 | teatime | 1.192879 | 1.187023 | +0.005856 | [-0.003734,+0.025336] |
| 0 | waldo_kitchen | 0.985829 | 0.986562 | -0.000733 | [-0.008746,+0.003165] |
| 0 | figurines | 1.321868 | 1.300873 | +0.020995 | [+0.012286,+0.031689] |
| 0 | temple | 1.006817 | 1.012232 | -0.005415 | [-0.017939,+0.000966] |
| 0 | external | 1.191831 | 1.181737 | +0.010094 | [+0.002679,+0.016301] |
| 0 | all | 1.170130 | 1.161619 | +0.008511 | [+0.002257,+0.013687] |
| 1 | teatime | 1.188795 | 1.186242 | +0.002553 | [-0.006282,+0.017685] |
| 1 | waldo_kitchen | 0.997324 | 0.999327 | -0.002003 | [-0.012464,+0.003020] |
| 1 | figurines | 1.295087 | 1.270863 | +0.024224 | [+0.012693,+0.038516] |
| 1 | temple | 1.006009 | 1.010620 | -0.004612 | [-0.015777,+0.001085] |
| 1 | external | 1.175717 | 1.163400 | +0.012317 | [+0.003934,+0.020870] |
| 1 | all | 1.158068 | 1.148284 | +0.009783 | [+0.002548,+0.016682] |

Positive CE differences favor retrained center-only. Both parity results, per-seed effects, per-scene effects and leave-one-place sensitivity are retained. The additional scene-macro seed/place/camera intervals use only four reused reconstructed scenes (two for external); they are descriptive and cannot establish broad scene-population generalization. Sparse-track targets are not dense visibility truth. Center-only preserves optimized GS means, fixed visibility and center dispersion: a result here cannot establish that all optimized primitive R/S are universally useless.

Source/data hashes, all10 checkpoints, all40 probability archives, sparse-track labels/masks, row metrics/identities, complete validation histories and finite-gradient diagnostics are saved. No GPU or SUN3D evaluation data were used.
