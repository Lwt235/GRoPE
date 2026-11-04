# Round 8: routing versus source-value diagnostic

2026-10-30. Execute the four-way diagnostic proposed after Round 7. This is **old development data only**, not a new confirmation, deployable method, or mathematical upper bound. All Round 5/6/7 sealed code and results remain unchanged; no old test predictions, labels or input caches are read for this round.

## Population and fixed training

Reuse exactly the Round 7 observation/label archives: Harvard c3/c5/c6 train, 192 pairs each; Harvard c8 and Brown bm4 validation, 64 pairs each. Full query-sensor-valid stride8 center population remains the primary loss/evaluation set. The original RGB features, predicted query depth, projected predicted geometry, cameras, target depth/validity/boundaries are identical across roles. Do not filter by cross-view correspondence or key sensor coverage.

Five roles and five paired seeds 11,22,33,44,55:

| Role | Attention routing | Key depth inside V |
|---|---|---|
| pred_pred | original learned full PE | original RGB-predicted |
| gt_route | sensor-derived center correspondence where available | original RGB-predicted |
| gt_value | original learned full PE | sensor-assisted, fixed RGB fallback |
| gt_both | sensor-derived center correspondence where available | sensor-assisted, fixed RGB fallback |
| singleview | independent original query self-attention | query RGB-predicted |

The body, 41,345 stored parameters, initialization, query residual, optimizer and batch sequence are unchanged from Round 7. A single attention/V block and depth head, h=.5/T=5 for all cross-view roles. Train 1,000 steps, batch8, AdamW lr=.001, weight_decay=.0001, gradient clip1, mean valid log-depth L1. Evaluate every100 steps including0; select earliest lowest equal-scene validation AbsRel. Preserve all histories, selected predictions and hashes. For gt_route/gt_both, Q/K receive gradients only through fallback rows: stored capacity is identical but effective routing learning differs by intervention. No extra module is introduced to equalize these gradients.

## Exact oracle contracts

**Routing.** Use native query sensor depth at the fixed query center, native K/C2W and the existing independent center projection/visibility convention: target inside image and positive depth; native nearest key depth agrees within .05m+.05×key sensor depth. This defines `oracle_valid`, an *extra diagnostic input*, not a loss mask. On eligible rows use four-neighbor bilinear weights on key token centers `(3.5+8x,3.5+8y)`, clamp at the token-center convex hull, and sum duplicate edge indices. Weights do not come from any candidate Gaussian kernel, visual network or training label distribution. Preserve projected coordinates, status and clipping flags. On every ineligible row retain the original predicted softmax attention, not an oracle-selected subset or zero-valued output. Query-GT invalid rows still receive finite predictions but are excluded by the unchanged label mask.

**Source values.** For each key image, sample native sensor depth at every192×144 resized pixel center. At positive finite sensor samples replace the corresponding RGB-predicted depth; at missing samples retain that same frozen RGB prediction. Apply the same fixed .1–20m input clamp. Backproject each sample with resized K, take the complete64-position 3D patch mean, transform to query camera, then take/clamp z. Never average z and assume it belongs to the center ray. The linear transform can equivalently precede the mean. Only the log-z component of V changes; visual V features, Q/K geometry and query baseline stay predicted. Per-key sensor fraction is saved for auditing and subgroup analysis, never fed to the model. These are64 resized-grid samples, not64 native sensor pixels.

GT visibility and sensor assistance are privileged information, so the 2×2 table is a diagnostic intervention, not a clean causal factorial or a deployable ablation. Different validation-selected checkpoints add another adaptation difference. Results must retain these qualifications even if large improvements occur.

## Readout and direct transport diagnostics

Primary is all-query-valid scan-macro AbsRel, with MAE/RMSE/logRMSE/delta1/boundary metrics as in Round 7. Independently reconstruct the original metrics from full saved predictions. Report per-seed and both validation scans, along with fixed eligible/fallback subsets and their counts. Also report oracle-neighbor weighted key sensor coverage on eligible rows. Coverage subgroups are diagnostics, not candidates for the primary score.

Separately compute a **nonlearned** transport check: frozen original visual similarity plus full PE (.5,5) versus bilinear sensor route, crossed with predicted patch depth versus sensor-assisted patch depth, i.e. four `A @ z` scalars. Query GT only evaluates outputs. On ineligible rows use the same predicted route fallback. Additionally sample RGB-predicted source depth directly at the true continuous source correspondence and native sensor depth at that correspondence, backproject and transform to query z. Report these two dense correspondence checks on eligible rows only, with coverage. The latter uses the same sensor visibility tolerance that selects rows; its accuracy is therefore a geometry/quantization check, not independent prediction evidence.

The direct point sampling versus patch-token transport contrast tests loss due to patch averaging, grid quantization and boundary mixtures. It does not silently change the learned task target from center depth to regional mean.

## Predeclared decision rule

The first question is whether **gt_route with original predicted V** improves over the independently retrained singleview control. Continue investment in repairing this depth route if (a) mean validation AbsRel decreases at least2%, (b) at least4/5 paired seed macro differences favor gt_route, and (c) both validation scans favor it on seed means. This is an engineering development gate, not an independent significance claim; two repeatedly used validation scenes do not justify population confidence intervals. Report differences relative to pred_pred as well.

- If gt_route clears the gate, inspect mean-position errors, region support and visibility before changing the architecture.
- If only source sensor assistance materially helps, prioritize source-value/geometry quality; further covariance-only tuning is not justified by this diagnostic.
- If neither route nor source assistance helps, use the direct transport checks to separate token/value information loss from head/training/coverage limitations; pause this depth validation route once those checks are consistent.

gt_value and gt_both receive the same2%/4seed/2scene descriptive gate for comparison, not as replacements for the primary question. Keep negative results and all-query coverage. New independent scenes are considered only after a deployable change, selected solely on development data, passes its own frozen protocol. No new RGB reconstruction task or true-primitive method is silently substituted for this diagnostic.

## Reproducibility

Freeze code/protocol hashes before training. Supplementary oracle inputs live in separate archives from unchanged Round 7 observations and labels; preserve native sensor paths/hashes, original selections, cache hashes and pair identity joins. Verify predicted and singleview implementations against original Round 7 models, missing-source fallback and constant-plane camera transforms, bilinear border mass conservation, oracle-only field isolation, NaN-safe loss, and nonzero relevant value/head gradients. Independent audit recomputes saved predictions and subgroup masks. No output directory is overwritten.
