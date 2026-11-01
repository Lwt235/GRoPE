# Round 7: scale transfer, actual V aggregation, and primitive scope

Development protocol, 2026-10-30. Round 5/6 sources, data and conclusions remain sealed.

## Questions and order

1. Apply the exact Round 6 five positional channels and locked h/T to previously unused 24-pixel cells, including 8↔24, 16↔24 and 24→24. The old validation and already opened Round 6 test scans are **development diagnostics**, not another independent confirmation.
2. Test whether this positional representation improves a trained **coarse query depth prediction task with actual attention/value aggregation**. This task can fail despite the established area-distribution KL benefit. All query sensor-valid token centers participate, including cross-view occlusions and out-of-frame correspondences.
3. Audit actual optimized Gaussian primitive R/S separately. A common native target-cell covariance admits an exact five-dimensional lift; arbitrary unequal projected primitive covariances require the full pair kernel. Numerical approximation is not evidence for useful primitive correspondence.

## Depth task: fixed observations and target

RGB features and metric predictions are the unchanged official frozen ResNet18 / DepthAnythingV2 Hypersim models used in Rounds 5/6. Shared supplied SUN3D K/C2W come from RGB-D reconstruction; this is not a strictly RGB-only system. The forward input contains no sensor depth, GT visibility or sensor validity mask.

Output is one metric camera-z depth at each fixed stride-8 center of the 192×144 image (24×18 = 432 positions). GT is native sensor depth nearest-sampled at the half-pixel-rescaled center, using the existing fixed nearest-tie rule. Only positive finite query GT selects the loss/evaluation population; source/key GT is never required. Query baseline prediction is the bilinear value at the center (average of four central pixels), clamped .1–20 m. Key V depth is the mean RGB-predicted 3D patch point transformed into the query camera, z clamped .1–20 m, then logged. No GT scale alignment.

Boundary diagnostic: a valid query center is a boundary if any valid horizontal/vertical adjacent center has depth difference > .1 m + .1 × min(depths). Missing sensor neighbors do not create boundaries. This coarse-grid metric does not establish native-resolution edge recovery.

## Minimal learned model and controls

One attention operation. A shared learnable 128×128 identity-initialized projection and L2 normalization form visual Q/K. The existing five positional channels are concatenated; their dot product is added to T times visual similarity. There is no extra sqrt(d) scaling. The Round 6 h/T values remain fixed: full/fixed/isotropic (.5,5), point (1,10), point_pck (.5,10). V is GELU(Linear([source feature, log source depth in query camera],64)). Query is GELU(Linear([query feature, log baseline depth],64)). Concatenated query and attention-weighted V pass through Linear(128,64), GELU, Linear(64,1) to predict a log-depth residual. Residual is clamped [-3,3], final prediction [.1,20] m. No visibility gate, covariance network, confidence module or multi-block stack.

All six learned roles share parameters, initialization, batches, optimization and checkpoint selection: full, fixed, isotropic, point, point_pck, and a separately trained **singleview** self-context control. Singleview uses the query itself as keys/values with native query cell means/covariances and (.5,5), so its full parameter capacity remains active. Also report the untrained frozen query depth. Zero-V and reversed-key-V are frozen interventions, not substitutes for retraining singleview.

Training scenes: old Harvard c3/c5/c6, all 192 pairs each. Validation: old Harvard c8 and Brown bm4, 64 pairs each. Train 1,000 steps, batch 8, AdamW lr .001 and weight decay .0001, gradient norm clip 1, log-depth L1 loss over query-valid positions. Five paired seeds 11,22,33,44,55. Evaluate every 100 steps including initial step 0; retain earliest minimum scene-macro validation AbsRel checkpoint. Point/fixed strong baseline is selected by mean five-seed validation AbsRel among point/point_pck/fixed before fresh test. All validation histories are retained. A short implementation smoke run is development only, never mixed with five-seed evidence. Any design revision must be separately versioned before fresh test; do not silently replace negative runs.

## New confirmation population

New SUN3D native metadata-selected 5 top-level place groups, 96 fixed pairs each, exclude every old R5/R6 place including rejected Brown bm1. See data/round7_sun3d/selection.json and independent docs/round7_protocol_review.md. Selection is 48 evenly spaced query ordinals with offsets 15/60; no performance, geometry or GT validity filtering. Fixed RGB-only place QA before any test scores; record all replacements and reasons. These are five scan/place groups, not five independent institutions.

No fresh test label export or test model execution until protocol, scene selection, source files, validation results and all selected checkpoints are hashed in a confirmation lock. Test run must verify/replay validation and all 480 pair identities; results cannot be overwritten. Export inputs first, labels second, with separate hashes.

## Analysis fixed before test

Primary AbsRel: sum absolute relative depth error / valid center count within scan and seed, then equal scan/seed macro. Also MAE, RMSE (sqrt within scan), log RMSE, delta1, boundary MAE/AbsRel and valid/boundary counts. Preserve per-pair raw sums and predicted depth arrays for independent recomputation. All methods have identical populations. Do not report pixel-level uncertainty as scan-level uncertainty.

Use 20,000 paired crossed bootstrap draws of five scans and five seeds (same indices for all methods), RNG seed 20261030, central 95% interval. Full must lower AbsRel vs validation-selected strong fusion baseline by ≥2%, difference CI upper bound <0, improve ≥4/5 seed macros, and have AbsRel difference CI upper <0 versus singleview. RMSE relative difference CI upper must be <+.02 vs strong baseline. These are research criteria, not established field thresholds. Full versus isotropic is exploratory attribution; no automatic claim that anisotropy is necessary. Show per-scan, per-seed and leave-one-scan sensitivity. If failed, retain the bounded R6 finding and explicit negative downstream conclusion.

Validate RGB/GT isolation, transformed V units, same-camera identity, all-query masks, finite gradients with missing GT, actual V/projection gradients and frozen value interventions. Independent audit recomputes raw prediction metrics and the paired analysis. Timing includes batch/model information; coarse task is not a SOTA benchmark.
