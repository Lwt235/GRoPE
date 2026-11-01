# Round 10 — region partition value and Gaussian compression

Frozen before any candidate result is calculated. This is old-development
research, not a fresh confirmation, and it cannot erase the Round 7/9 failures.

## Question and bounded design order

Does one Gaussian summary of an observed token's spatial support retain useful
information more accurately or cheaply than a small empirical quadrature when
the same images are genuinely split/merged? Candidate A is the known directed
Gaussian density overlap. If its failure involves the Gaussian approximation to
known rectangular keys, candidate B may integrate a Gaussian query over the
actual key rectangle. B retains one query Gaussian and adds no learned module;
its integration mathematics is established, not a novelty claim. Do not open a
new confirmation set unless a candidate clears the development gates below.

## Data and partitions

Use only already used development scans Harvard c3/c5/c6 (calibration) and Harvard
c8/Brown bm4 (gate). Producer `experiments/round10_partition_data.py` chooses eight
uniform original query ordinals and the original two offset partners per scan:
48 calibration and 32 gate pairs. No score/visibility filtering. The complete
80-pair forward export precedes all sensor-depth reads. All frames are 192x144;
features and predicted depth are frozen Round 5 RGB-only predictions, with
supplied calibrated cameras. Native sensor depth is supervision only.

Every query/key partition is g8, g16, or deterministic mixed. Mixed cycles within
16x16 parent cells among intact16x16, two8x16, two16x8, and four8x8 cells. All nine
query/key combinations are evaluated. Features are area means of the same
base8 features, then L2 normalized. Native key mass is true pixel area and
covariance is the continuous rectangular footprint. Query Gaussian moments are
computed *after* dense RGB-predicted projection: projected pixel-center scatter
plus projected continuous pixel covariance. This differs from the earlier
3D-moment-then-Jacobian implementation and is identified as such. Fixed producer
fallbacks retain every pixel; predicted in-front flags are diagnostics only.

## Methods and calibration

Common relative bias: log(key area) + log N(delta;0,Cq+Ck+h²I).
Methods: point (both covariances zero, same query mean), center (zero query
covariance with the actual native key covariance preserved), fixed (isotropic native
area/12 on each side), isotropic (same trace), full, quad4, quad16. Quadrature
uses 2x2 or 4x4 equal-area source strata, each represented by its actual mean
projected position and mean within-pixel covariance, omitting within-stratum
scatter. It uses every predicted pixel, exactly as full does; no extra sensor
information. The mixture density is averaged in probability space, not by
averaging its log. All use the same native Gaussian key approximation. Full
with query covariance rotated90 degrees, preserving eigenvalues/mass/means, is
a diagnostic only. No Gaussian-generated task targets.

Each primary method gets exactly h in [.25,.5,1,2] in common8pixel units and
content temperature T in [0,5,10]. One configuration per method is selected by
calibration common-grid KL macro (scan then nine conditions), ascending h then T
break ties. Choose the stronger point/center by calibration KL as the primary
position baseline; separately maximize g8/g8 PCK@8 across both point families. Persist and
hash selection before gate labels are loaded. Rotate90 inherits full's chosen
h/T. The strong quadrature comparator is selected between quad4 and quad16 by
calibration KL; also report the better gate quadrature as a conservative
descriptive envelope. No seeds are manufactured for deterministic methods.

## Common population and metrics

The primary reference is independent sensor-projected counts from all visible
source pixels in fixed16x16 query parents into fixed8x8 target atoms. The GT
visible population is identical across methods/partitions. Each predicted key
probability is spread uniformly over the base8 atoms in that key; child query
predictions are area-averaged into their16x16 parent, using only source area,
never GT visibility. Normalize predictions within each query before pooling.
This measures deployable partition robustness against the same reference; it
does **not** assert exact equality with visible-mass-weighted child GT. Hidden
source pixels can influence all forward estimates. Coarse key discretization
error is intentional and shared by all methods in that condition.

Primary KL uses a fixed1e-15 probability floor for the log only (no smoothing
the reference), weights each parent by its sensor-visible source count, then
macro-averages conditions and scans. Report JS, direct same-partition KL,
all nine condition errors, each scan, and g8/g8 independent center PCK@8. Also
report dense predicted hard-cell transport and its common-grid KL as a diagnostic
of upstream geometry, without using it as task truth or a calibrated contender.
No whole-image mass metric can replace the query-resolved primary error.

## Development gates and timing

All gates must pass for a promotion; they are not proof of general success.
1. At least2% lower primary KL than the calibration-selected point/center, both gate scans improve.
2. Against calibration-selected strong quadrature: either at least2% lower KL
   with median complete support-preparation-plus-operator time ≤1.25x, or KL
   no more than1% worse with time ≤0.5x. No individual partition condition may
   be >2% worse than that comparator. Also disclose gate-envelope comparison.
3. g8/g8 PCK@8 loses no more than0.5 percentage point against separately
   calibrated localization point. This is a guardrail, not an area objective.
4. Finite numerical behavior and independent probability/metric/provenance
   audits pass. Review all per-scan/condition results without post-hoc exclusion.

Timing: first pair of every scan, all nine conditions, actual float32 GPU
implementation, three warmups and fifteen synchronized repeats; compare median
per-condition elapsed times, then macro scan/condition. Complete timing includes
query moments or stratum nodes, pooling both content banks, geometry score,
softmax and common-grid aggregation. All data transfers and shared frozen RGB
feature/depth predictor are excluded and explicitly labeled; this is incremental
PE cost, not end-to-end vision model latency. Record separate preparation and
operator medians and stored support/content/score tensor bytes (not allocator
peak memory). Current point/fixed preparation also computes unused full
covariance; quadrature skips that redundant calculation. Report these as
prototype implementation times, not a universal complexity/speed claim.
GPU allocation must avoid busy jobs.

## Reserve B and subsequent research

If A fails and known-key Gaussian approximation is a documented issue, B uses
positive Gaussian-to-rectangle integration with the same query mean/covariance,
h/T budget, data, metrics and gates. Include point-query isotropic integration
and 4/16 empirical-node rectangle integration as strong matched controls; do
not only compare B to a weaker key approximation. Freeze B numerical scheme,
source hashes, and calibration selection before B gate evaluation. Reusing these
old development data is disclosed. Any actual confirmation uses untouched
scenes, a new lock, inherited geometry/noise/scale/localization checks and actual
downstream utility (the earlier depth/RGB successes have not been established).
If both A and B fail, record that failure and redesign the hypothesis rather
than increasing modules or tuning against opened historical confirmation data.
