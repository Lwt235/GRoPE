# Round 10 independent protocol review

2026-10-30. This review was written before any Round 10 method score was available.
It reviews the proposed experiment, not its eventual empirical success. Only old
development sources and the existing written Round 6–9 reports/protocols were
inspected; no sealed test input archive, label, prediction, or target image was
opened for this review.

## Question and admissible conclusion

The question is whether a compact Gaussian summary of an observed token's spatial
support provides a useful accuracy/cost tradeoff when actual image regions are
split, merged, or assigned different sizes. This is distinct from directly using
optimized rendering-splat shape: the latter route failed the predeclared Round 9
development gate. Round 6 already supports a narrower regional correspondence
claim, while Round 7–8 did not establish a general depth-prediction benefit.

Passing this development probe would justify a new frozen confirmation and a
compact heterogeneous-key implementation. It would not itself pass the previous
downstream depth or RGB gates. Those outcomes must remain recorded; unit tests and
operator equivalence are implementation evidence, not empirical effectiveness.
The user's requested general solution cannot be declared achieved by selecting a
new metric that omits those remaining limits.

## Planned population and comparisons

The communicated contract uses the three old Harvard training scans for
calibration, and the old Harvard c8/Brown bm4 scans for a development gate. Eight
metadata-selected query frames, each with both old offsets 15 and 60, yield 16
pairs per scan: 48 calibration and 32 gate pairs. These scans have been used
repeatedly and are not fresh confirmation data. No population confidence interval
should be manufactured from overlapping pixels/pairs or deterministic repetitions.

The three actual rectangular partitions are uniform 8px, uniform 16px, and a
fixed mixed partition contained within the same 16px parent regions. The data
implementation cycles intact 16×16, four 8×8, two 16×8, and two 8×16 regions.
All nine query/key partition combinations are evaluated. Actual pixel membership,
area, content pooling, and support moments change with each partition. Copies of
one Gaussian are not a substitute for this experiment.

The planned main roles are point plus area, center plus native key support,
known-size fixed support, same-trace
isotropic support, full covariance, and empirical 4/16-node quadrature. Each
calibrates bandwidth from {.25,.5,1,2} in common 8px units and content temperature
from {0,5,10}, with one configuration per role selected across all calibration
scans/conditions. The 90-degree covariance rotation is an attribution diagnostic,
not another selectable candidate. Mass is actual native key pixel area with log
coefficient exactly one; it is not inferred from an arbitrary covariance
determinant. EWA/Mahalanobis equality is an algebra audit, not an accuracy control.

The implemented quadrature uses every predicted pixel, exactly as the full
candidate does. Split each source rectangle into a fixed equal-area 2×2 or 4×4
stratification. Each node is the mean projected position of all pixels in its
stratum; its covariance is the average continuous within-pixel covariance, with
within-stratum scatter deliberately omitted. The node weight is its source
pixel count divided by the region's pixel count. This is a stronger same-input
control than sampling only the stratum midpoint. It retains multimodal placement
between strata while approximating their internal spread. Count both full-support
and quadrature preparation plus operator costs. The full predicted-point hard-
transfer diagnostic remains a predicted-geometry diagnostic rather than sensor
ground truth.

## Metric and normalization audit

Independent sensor-depth projection supplies the fixed reference. Every visible
source pixel in a common 16px query parent contributes one count to a common 8px
target cell. No candidate Gaussian, predicted-geometry quality threshold, or
candidate visibility mask defines that label distribution or its population.

For the primary comparison, redistribute each predicted key probability according
to its known intersection area with common 8px cells; then average child-query
predictions inside each 16px parent using source-image area. The reference,
evaluated parent population, atomic target classes, and target entropy are then
identical for all roles and all nine conditions. Weight KL within each condition
by visible source mass, average the nine conditions, and average scans equally.
Also retain direct per-partition KL and every per-condition/scan result. A whole-
image pooled distribution would allow unrelated spatial errors to cancel and is
not an adequate primary metric.

There is an essential limit: each predicted query row is normalized before
source-area averaging, whereas the reference is conditioned on observed visible
mass. If children have unequal visibility/out-of-frame mass, the merged true
conditional distribution uses visible-count weights, not source-area weights.
The primary construction is consequently a deployable robustness test against
fixed sensor truth, not an exact mathematical invariance requirement. Sensor
weights may be used for a separately named evaluation diagnostic, never fed to
the candidate. All source children remain in the forward aggregation, including
children with no sensor-valid correspondence.

## Gate reviewed before execution

The final frozen continuation gate requires full covariance to reduce common-
grid KL by at least 2% versus the calibration-selected stronger point/center
control and improve both old
validation scans. It additionally requires one of two accuracy/cost paths versus
the strongest calibrated 4/16-sample control:

- At least 2% lower error with measured preparation-plus-operator time at most
  1.25 times the sampling control's time.
- Error no more than 1% higher with at least a twofold measured speed advantage.

The comparator selection rule must be fixed before gate scores are read. Selecting
the sampling role on calibration is natural; reporting the gate's best sampling
envelope as a conservative check avoids presenting a weak control. Bandwidth,
temperature, partition weights, denominator conventions, timing scope, and tie
rules must be in the frozen protocol and artifacts, not chosen after seeing data.
The common frozen RGB predictor may be excluded from timed costs only when that
exclusion is explicit. Use synchronized repeated operator timings, common dtype,
device, output requirements, and preparation scope; report spread rather than one
favorable timing. A speed result concerns this implementation/hardware, not an
optimality theorem.

Two protections requested from the root were included as hard gates before
freeze: preserve the prior
8→8 point-localization limit of at most 0.5 percentage-point PCK@8 loss relative
to a separately PCK-calibrated point/center control, and reject any condition with more
than 2% common-grid KL deterioration versus the strong sampling comparison.
These protections prevent a macro mean from concealing the same localization or
partition-specific losses already observed in prior rounds. The frozen protocol
and runner, with independent calibration/result audits, determine their outcome.

Before execution, 44 independent model tests passed. A calibration-only numerical
preflight compared actual float32 preparation/operator outputs against independent
NumPy float64 pixel slices and Cholesky density/mixture evaluation, retaining all
keys and up to 34 predeclared/extreme query rows. Across three scans, nine
partitions, three full/quadrature roles, four bandwidths and two temperatures,
maximum probability error was 1.12098×10⁻⁵, below the 10⁻⁴ numerical threshold,
with zero top-key disagreements. This is a bounded numerical audit, not an
accuracy result. Its frozen replay and source snapshot are in
`results/round10_numeric_preflight_frozen/`.

## Why exact subdivision invariance is not expected

First and second raw moments, with mass, can be merged exactly. Their conversion
back to one Gaussian does not preserve the entire spatial measure. For example,
equal point supports at −a and +a form a two-atom mixture after union, whereas
their moment-matched Gaussian has mean zero and variance a². Their overlaps with
an arbitrary localized key differ. Thus finite Gaussian moments cannot guarantee
exact true split/merge invariance. Exact duplicate-mass splitting is a narrower
identity and must retain a separate test/name.

Conditional row normalization is another source of nonadditivity when outgoing
mass differs. If useful, audit unnormalized mass additivity separately from
normalized attention. Do not cure this by supplying sensor visibility to the
forward pass or by changing the primary evaluated population.

## Iteration and confirmation discipline

A negative probe is retained. A subsequent compact repair is justified only by a
specific diagnosed approximation error, under a newly recorded development
contract; it is not licensed to tune repeatedly on fresh confirmation data. For
example, exact native-cell integration could diagnose errors introduced by
approximating rectangular keys with Gaussians, but it would still use established
integration mathematics and must beat strong sampling on accuracy or cost.

No number of repeatedly tried candidates guarantees a scientifically valid
positive result. A promising development candidate needs a genuinely untouched
confirmation cohort, frozen source/configuration hashes, independent metric and
forward-data audits, and relevant localization/downstream checks. Once a cohort
has informed a redesign it becomes development evidence. Round 6–9 frozen source
and result artifacts must remain unchanged throughout.
