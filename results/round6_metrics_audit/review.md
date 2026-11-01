# Round6 metric implementation review

The inspected implementation in `experiments/round6_train.py` is mathematically consistent with the declared region-area task. No model or trainer was modified. The exact inspected source is archived as `source_round6_train.py`, with source/test hashes and execution output in `verification.json`.

`per_query` computes cross entropy, target entropy and Jensen–Shannon divergence from soft target mass and softmax probabilities. `CE−entropy` is the target-to-prediction KL. Zero target entries contribute zero. The implementation matches independent probability formulas for unequal distributions, distributions with zero target entries, equal distributions and extreme finite logits. Invalid area rows have zero target mass; their per-row JS has no probability-distribution interpretation, but their zero visible mass removes them from every primary aggregate.

`measure_rows` weights primary CE/KL/JS by `sample_count * visible_fraction`: the number of visible source-image pixel samples. Because sample counts are 64 or 256 and fractions use these powers of two, the exported float32 fractions reconstruct integer visible counts exactly here. There is no predicted-depth quality mask or visibility-fraction threshold. Region-valid and point-valid populations remain independent; tests cover region-valid/point-invalid and region-invalid/point-valid rows. NaN coordinates on invalid center-point rows do not contaminate sums. Point CE handles the -1 sentinel only through its independent validity mask. PCK thresholds are in resized image pixels, including the target stride for the scale-relative secondary metric.

`macro` first sums numerator and denominator within each `(scene, grid)` condition, then averages those ratios equally. It does not silently pool all pixels across scenes or average pair metrics without their denominators. An analytic example with unequal masses distinguishes these estimands. Thus the primary validation objective is an equal-scene/equal-condition average of visible-pixel-weighted KL. Since target entropy is fixed, CE and KL give the same checkpoint ordering mathematically; KL also makes differences between target entropies easier to interpret.

The training loss uses the same visible-pixel weighting within a sampled condition/batch. Conditions are uniformly sampled; training pairs are sampled uniformly from the loaded group. This is a minibatch-normalized ratio objective, not an exactly unbiased estimator of the complete validation scene-macro objective. The declared current data use equal pair counts per scene. Methods share this sampling and weighting, so it is not a method-specific discrepancy.

The evaluator explicitly passes only the ten `FIELDS` to the matcher. A capture-model regression proves that labels, visibility fractions, GT masks and point targets do not reach `model(inputs)`. This verifies the trainer boundary; it is not a substitute for the separate RGB-cache provenance and geometry-source audits.

Two contracts should remain explicit:

- `measure_rows` takes its stride-relative PCK threshold from `ids[0]`; its batch must use one target stride. Current `evaluate` groups by grid and satisfies that contract. A future mixed-grid batch must supply per-row thresholds or validate homogeneity.
- `macro` deliberately raises when a scene/grid denominator is zero. A future dataset containing a wholly unmeasurable scene/condition needs a predeclared reporting policy; it must not silently drop the condition after inspecting method scores. Current exported train/validation groups have measurements.

Eight metric tests plus seventeen data tests pass. Tests use analytic/synthetic data only. No heldout test RGB/depth, probability files or model outputs were read in this metric review.
