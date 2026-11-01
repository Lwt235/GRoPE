# Round 11: old-training-only target compatibility diagnostic

Freeze this document and the standalone diagnostic source before that program
reads any old-training label archive. No candidate source, predicted geometry,
calibration/validation score, new validation label or sealed test is accessed.
Use every one of the 48 fixed calibration pairs from Harvard c3/c5/c6. This is a
supervision diagnostic on reused training data and cannot validate a method.

Independently reconstruct native g8→g8 area counts from every already frozen
sensor-projected integer source pixel. The source raster is 192×144. Integer
pixel centers belong to g8 source cells; target containment uses physical edges
`-.5`, image extent `[-.5,191.5)×[-.5,143.5)`, and the existing `1e-9` boundary
tie guard before division by eight. Native key centers are `(8i+3.5,8j+3.5)`.
Only sensor-valid projected pixels enter the area histogram. The independent
source patch-center target and its validity are read from the frozen label
archive; they are not inferred from the histogram.

Retain a result row for every source query (48×432). Report area-valid,
center-valid, joint-valid, both one-sided-valid and neither-valid populations,
as well as source visible-pixel mass. Define the PCK/KL compatibility diagnostic
only where the regional distribution is nonempty and the independent center
target is valid. No assertion about excluded/undefined rows is implied.

An allowed key has center distance at most eight pixels from the independent
center target. For each joint-valid query report:

- Whether the first native-index argmax of the normalized area histogram lies
  in the allowed set, matching deterministic native tie selection.
- Whether **any** tied area-histogram maximum lies in the allowed set. This is
  the optimistic tie convention, and can differ from the preceding statistic.
- The minimum `KL(g || p)` over probability vectors for which some allowed key
  is a largest entry, with ties permitted. This is a relaxed constraint, not a
  guarantee that a deterministic first-argmax decoder satisfies PCK.

For a fixed allowed candidate j, the convex KL projection pools `g_j` with
entries whose original probabilities exceed the current pooled mean; repeat
in descending order until every other entry is no larger. Assign the pooled
mean to those entries and leave the others unchanged. A candidate with maximal
g among allowed keys achieves the smallest KL: swapping a maximal predicted
probability from a lower-g allowed candidate to a higher-g candidate cannot
worsen cross entropy. Exact ties use the first allowed native index. Verify the
projection independently using SciPy constrained optimization on synthetic
distributions, including zero masses and argmax ties.

Report pooled counts and scene-equal means, deterministic and optimistic
conflict fractions, query-equal and visible-mass-weighted minimum KL, quantiles,
and the number of positive bounds. The reported KL concerns the **direct g8
regional histogram**. It is not a lower bound on Round11's source-area-pooled
g16→g8 common-grid metric, and it is not a lower or upper bound on PE accuracy.

The diagnostic never supplies model inputs, changes candidate selection, tunes
parameters, alters masks in candidate experiments, or relaxes any acceptance
gate. An observed conflict explains a property of the two targets; it does not
turn a failed PCK protection condition into a pass. Zero or low conflict also
does not prove the candidate can satisfy both targets using available inputs.
