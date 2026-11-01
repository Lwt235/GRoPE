# Independent locked depth-test audit

All 31 saved archives were independently recomputed from their depth arrays and the common query sensor labels: 30 trained role/seed runs plus one raw RGB prediction. All 14,880 pair metric rows and complete scene-macro summaries agree within 7.28e-12. No original trainer metric or analysis function was called.

For 240 distinct query frames, native PNGs were independently decoded with the SUN3D circular bit rotation, converted from millimeters to meters, and sampled at half-pixel-rescaled stride-8 centers using the fixed nearest rule. Target depths, valid masks and four-neighbor boundaries match all exported test labels bitwise. Sources, the five named scan/place groups, all 480 pair identities, source hashes, checkpoint metadata, prediction hashes and shared label hashes match the frozen confirmation/data locks.

The 480 pairs contain 179,848 query-valid pair positions and 50,960 boundary pair positions. Each source query has two target pairs; these repeated positions are not independent samples. This audit does not calculate uncertainty from pixels or execute additional fresh-test model replays.

|role|five-seed / equal-place AbsRel|MAE, m|RMSE, m|delta1|
|---|---:|---:|---:|---:|
|full|0.171545594|0.404571|0.589778|0.788351|
|fixed|0.171615447|0.404727|0.589916|0.788149|
|isotropic|0.171644923|0.405100|0.590282|0.787536|
|point|0.172123030|0.406961|0.591970|0.785904|
|point_pck|0.171505405|0.404324|0.589589|0.788491|
|singleview|0.169525031|0.394109|0.581457|0.799319|
|raw|0.229686938|0.492888|0.674017|0.697685|

The validation-locked strong baseline is point. Full changes AbsRel by -0.3355% relative to it, falling short of the predeclared ≥2% reduction. Full also has worse mean AbsRel than separately retrained singleview. These independently recomputed point estimates support the negative downstream conclusion; crossed scan/seed CI and other gates are owned by the separate frozen analysis.

Implementation and population checks passed. The negative result must be retained; this audit does not license tuning on these test scores. It neither reverses the bounded Round6 area-distribution finding nor establishes general usefulness of optimized 3DGS rotation/scale.

Command, executed only after explicit root completion notification: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.round7_depth_test_audit --after-root-test-complete`. Runtime 5.56 seconds CPU1. Test-time audit scope is raw prediction/label arithmetic, independent native-sensor resampling and provenance; real-model gradient/value-path checks were performed on old validation data before this test.

Confirmation lock SHA256: 7886185a8aa39f486a0ae02079b8123a99a1c3d12eb536fe828a4f7a194e4376. All sealed source files remain unchanged.
