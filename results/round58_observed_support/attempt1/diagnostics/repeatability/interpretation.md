# Observed-support repeatability diagnostic

The prespecified diagnostic completed on old validation data while the main R58
queue was still training. It selected107710 token-pair relations from1227802
eligible relations, with at most64 per object/direction.893 of965 objects were
analyzable;144 of1930 directions had no eligible relation. Selection used only
fixed geometric correspondence rules, not model scores or evaluation outcomes.

| Relation | Mean symmetric relative change | Median | Mean sine-input RMSE |
| --- | ---: | ---: | ---: |
| Euclidean distance | .094429 | .063876 | .037324 |
| Gaussian adaptive distance | .179075 | .141294 | .207318 |
| Full Gaussian relation | .174191 | .137917 | .207944 |

On893 objects, the mean per-object changes were.094719,.178690,.174004,
respectively. Both Gaussian forms changed more than Euclidean distance on99.440%
of these objects. The full form was slightly less variable than the metric-only
form under this relative-error definition; the trace term cannot therefore be
singled out as the source of the instability.

This supports limited repeatability of these observed-support relations across
independent sampling and partial views. It does not isolate covariance estimation
error: patch centers, support membership, local geometry and nonlinear metric
sensitivity all change. Corresponding overlapping regions need not have identical
support, so exact equality would not be a valid universal requirement. The sine
RMSE additionally depends on the already-fixed argument scales. Learned relative
projections can suppress sensitive directions; these numbers are not performance
bounds and are not a new gate. R58 training, selection and comparison remain fixed.

A module-parse failure (one missing bracket) occurred before any diagnostic array
was read, was corrected, and is retained at diagnostics/preparation_failure1.
The main frozen experiment was unaffected. The successful diagnostic terminal
chunk is fe9d19. Source/plan/cache identities and all measured values are in
plan_identity.json,summary.json and values.npz in this directory.
