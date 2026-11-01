# Refit audit interpretation

All ten arms completed on one CPU thread in 107 seconds. Five full-covariance arms reproduce the old `gaussian200_full` checkpoint tensors, selected steps and validation metrics exactly. Both arms use the old validation-selected covariance multiplier 2, the same seeds/batches/optimizer/budget, and separate earliest-minimum validation selection. Seed33 selects step60 for full and step80 for center; the other seeds select the same step in both arms. All gradients are finite.

This removes the earlier audit's limitation that a full-trained encoder was evaluated with an unfamiliar center-only covariance distribution. The center model is now trained and selected with center-only inputs on both sides. The comparison still fixes the old common physical scale and does not independently retune the center-only covariance scale.

Differences below are full minus retrained center, so positive CE favors center. The first interval samples paired training seeds and query cameras within the fixed scene set. The second weights reconstructed scenes equally and additionally resamples the tiny scene set; it is descriptive, not a reliable population claim.

| partition / group | full CE | center CE | row-weighted difference [seed/camera 95% CI] | scene-macro difference [seed/place/camera 95% CI] |
|---|---:|---:|---|---|
| primary / external | 1.191831 | 1.181737 | +.010094 [+.002679,+.016301] | +.007790 [−.013241,+.027492] |
| alternate / external | 1.175717 | 1.163400 | +.012317 [+.003934,+.020870] | +.009806 [−.011531,+.032765] |
| primary / all four | 1.170130 | 1.161619 | +.008511 [+.002257,+.013687] | +.005175 [−.006149,+.018029] |
| alternate / all four | 1.158068 | 1.148284 | +.009783 [+.002548,+.016682] | +.005041 [−.007143,+.019468] |

All five seeds favor center in each pooled external/all comparison and both partitions. This alone does not establish stability across scenes: Figurines favors center by +.020995/+ .024224 CE, while Temple favors full by −.005415/−.004612 with camera/seed intervals crossing zero. Removing Figurines changes the all-scene macro difference to −.000098/−.001354. The aggregate center advantage is therefore driven principally by Figurines. Per-scene values, per-seed values and every leave-one-scene result are in `summary.json`.

Pooled all-scene JS also favors center (+.002873/+ .002994 full-minus-center). Top1 target-mass and dominant-region-accuracy differences are small and their seed/camera intervals cross zero. The evidence concerns probability allocation on conditional sparse-track correspondences; it is not proof of improved dense correspondence accuracy or rejection.

The result provides no evidence that retaining weighted internal primitive R/S covariance independently improves this selected legacy PE setup after both arms adapt through training. It does **not** show that optimized primitive R/S is universally useless: four previously inspected reconstructed scenes, fixed opaque-center visibility, sparse labels, the old scale2 operating point and optimized GS centers remain material limits. Center-only preserves regional point-cloud shape; regional Gaussian support gains cannot automatically be attributed to 3DGS-internal covariance.

`verification.json` records an independent post-exit replay: all ten checkpoint hashes, nine source hashes and 725 original/component/manifest hashes pass; all 80 checkpoint prediction arrays reproduce the 40 saved probability archives bitwise, and shared labels/masks and metrics match exactly. Float32 probability row-sum error is at most 3.58×10⁻⁷. Three focused regression tests pass. No SUN3D data or GPU was used in this experiment.

Replay:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m results.round5_primitive_retrain_raw.replay_verification
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests/test_round5_primitive_retrain.py
```
