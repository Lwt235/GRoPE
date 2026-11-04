# R40 region and center diagnosis of depth-trained attention

2026-11-01. Fixed before regional forwards/label decoding, while the already
frozen depth training is running. This is old-data mechanism analysis, not an
independent confirmation or a change to the depth primary/checkpoint rule.

Use all48 original R20/R38 TRAIN pairs (16 per scene), q24=48, k8=432, all2304
queries and the original1096 center labels. Evaluate all12 cross-view roles and
allfive selected depth seeds. Singleview attends query-image keys, so its matrix
cannot be scored against source-image cell labels; its depth comparison stays.
No role is chosen by depth results or regional performance.

Build central Sigma and bilinear anchor from the same cached predicted depths,
using24px query patches and8px key patches. Within-pixel support, cleanup,
metre units and camera transform match R40 training. Features are the original
R20 pooled q24 / native k8 features. For legacy controls reconstruct the R7
linearized projected central support with its original fallbacks; it is not the
R20 anchor-fixed operator. Keep original R20 strong quad/localizer scores as
external references, distinct from these retrained legacy controls.

Use the selected depth checkpoint unchanged. Shape-shuffle needs a48-query
permutation: NumPy default_rng(20261101).permutation(48); keep its original
432-key permutation. This fixed grid adaptation changes no learned parameter.
All other operators use unchanged frozen R40 sources. GPU float32 scores and
softmax, save all probabilities and native first argmax before opening labels.
No probability renormalization on export. Accept row sum residual<=2e-6;
float32 sums do not have to meet the old float64 arithmetic identity tolerance.

Compute the original g24->g8 KL/JS (1e-15 metric floor), PCK@8/RMSE, original
visible-pixel mass within scenes and equal-scene macro, separately per seed.
Use independent NumPy scalar-loop replay for region sums and center counts.
Continuous metric agreement rel1e-6/abs1e-8, counts exact. Independently compare
first-pair complete probabilities for each role/seed using existing R40 NumPy
formulas. Report differences and margins; a tolerance warning requires impact
review, not fabricated probabilities or a fresh model retrain.

For each joint-valid query define center-allowed keys by<=8px distance from GT
center. Strict conflict means the largest area count among allowed keys is
less than the global largest count. Keep joint-consistent/conflict/nonjoint
counts and group metrics, always alongside the full-population totals. Use
these groups to distinguish target conflict from general representation failure;
they never replace the primary population or justify deleting an error.

Use each model's single probability matrix for all metrics. The original eight
necessary comparisons may be displayed diagnostically for allfive primary
checkpoints, but no ensemble/checkpoint is selected from them. Success here
does not establish the full60-condition intersection, VAL/stress/cost, actual
GS or fresh confirmation. Failure here is not mathematical impossibility of
decoupled representations. Preserve predictions and diagnostics either way.
