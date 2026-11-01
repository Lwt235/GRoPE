# Independent depth-path audit (old development validation only)

All 30 saved development prediction archives (3,840 pair rows) match independently recomputed NumPy raw sums and scene-macro summaries. Maximum numeric error is 1.82e-12. All roles/seeds use the same 46,818 query-valid positions and boundary masks. Boundaries were independently reconstructed from the query labels.

Ten fixed checkpoints were replayed on all 128 old validation pairs on CPU: six roles at seed 11 and full at the other four seeds. Maximum difference from stored GPU predictions is 7.63e-6 m. No fresh test input or label was opened.

For six seed-11 roles, four fixed real validation pairs (indices 0, 32, 64, 96) had bitwise identical outputs after injecting NaN labels and changing GT/mask fields. All model parameter gradients were finite and nonzero; actual attention and V intermediate gradients were finite and nonzero. External key depth has a nonzero gradient for fusion roles; singleview intentionally replaces it with the query self value. Missing target depths masked in the loss retained finite gradients.

Independent float64 Gaussian-energy reconstruction agreed with the actual attention probabilities within 2.31e-5. This checks the five-channel positional dot against the direct Gaussian energy while preserving the trained visual projection and temperature.

|role|Q/K weight gradient norm|V weight gradient norm|head final weight gradient norm|zero-V output change, m|reversed-V output change, m|
|---|---:|---:|---:|---:|---:|
|full|0.0009866|0.0675153|0.264545|0.060391|0.088866|
|fixed|0.000920639|0.0656688|0.257136|0.059469|0.078526|
|isotropic|0.00101163|0.0679115|0.260476|0.060566|0.081179|
|point|0.00286634|0.0778641|0.26165|0.065173|0.081632|
|point_pck|0.00109386|0.0635665|0.254869|0.058197|0.078262|
|singleview|1.75607e-05|0.0515848|0.28071|0.111649|0.066013|

These interventions establish that values affect the trained model, not that cross-view values always help. On this four-pair diagnostic, full reversed-V AbsRel is almost unchanged; zero-V even improves singleview. The separately retrained singleview comparison remains necessary. These four-pair outcomes must not be treated as new validation selection evidence.

Of the query-valid positions retained for scoring, 13,372 Harvard and 13,360 Brown positions fail the prior cross-view visibility mask; none was filtered from the depth task. Thus these measurements use query sensor validity, not GT cross-view visibility.

Commands: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests/test_round7_depth_audit.py` (7 passed); `python -m experiments.round7_depth_audit` (6.72 seconds CPU1). Raw NumPy rows, summaries, source hashes, exact population counts, gradient norms, intervention summaries, and fixed replay scope are retained.

The separate test auditor has been implemented and synthetically tested, but has not been executed. It requires explicit root completion notification and a completed source/data lock before reading any test label.
