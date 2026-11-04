# R40 learned decoupled position and shape experiment

2026-11-01, prospective old-development protocol. This implements the user's
route review: test a learned PE before spending more rounds certifying a static
integrator. The unexecuted R40 half-ray files remain a separate reserve. No old
protocol, failed result, or final scientific requirement is rewritten.

## Question and fixed candidate

Can center phase and physical support shape be used separately by learned
attention, with useful gains beyond the same center information and capacity?
Alternative explanations are extra scalar capacity, scale alone, shape identity,
and mere removal of expected-RoPE attenuation. A positive depth result is not a
region/center compatibility or real-GS/NVS result.

Use the existing R7 learned Q/K, query/V projections and residual depth head.
For each RGB-predicted 8px patch, use its bilinear center-ray depth for anchor a,
but its CENTRAL point-scatter plus within-pixel footprint for covariance Sigma.
No (mu-a)(mu-a)^T correction. Transform both views into the key-camera frame;
do not drop behind/outside queries. All new roles receive both predicted depths,
anchors, covariances and identical content; old 2D strong controls are retained.
Metres are fixed by the existing depth predictor, with inherited .1..20 cleanup.

For the primary `decoupled` role:

    s(Sigma) = [svec(3 Sigma / tr Sigma), log(tr Sigma)]
    logits_ij = 10 <RoPE(a_i) normalize(Wqk f_i),
                       RoPE(a_j) normalize(Wqk f_j)>
                - softplus(beta) ||s_i-s_j||^2 / 7

svec preserves the Frobenius norm (sqrt(2) on off-diagonals), with trace floor
1e-12 m^2 only to define zero supports. One learned scalar starts at strength1.
No shape MLP, gating network, extra decoder, anchor projection or kernel search.
The scalar adds one ACTIVE parameter (41,346 vs original41,345); `center_rbf`
has the same active capacity, replacing the shape penalty by ||a_i-a_j||^2/m^2.
Do not claim all roles have equal parameter counts through inactive padding.

RoPE uses64 axis-cycling frequencies with norms logspace(0,1.3,64) rad/m, extending
the early probe schedule. No frequency/bandwidth/temperature search this round.
`center_rope` omits the penalty. `expected_rope` multiplies each rotary block by
exp(-omega^T Sigma omega/2); `normalized_rope` additionally normalizes the whole
Q/K vector after that operation. `shape_isotropic` uses tr(Sigma)I/3 only in
the shape descriptor. `shape_shuffle` independently permutes query/key Sigma
with fixed NumPy default_rng(20261101) permutations, preserving centers/features.
These are trained controls, not alternative primaries chosen after scores.
Legacy point/point_pck/fixed/full/isotropic/singleview run unchanged R7 operators.

The shape distance is a known descriptor kernel, not a new Gaussian integral or
RoPE identity. Up to a query constant its unrounded score has an8-dimensional
dot-product lift [2s_i,-1] dot [s_j,||s_j||^2] times strength/7. Finite arithmetic
clamps tiny negative squared distances to zero. Neither it nor learned attention
is an exact region-area measure; it does not guarantee the point winner. Axis
RoPE is not invariant to arbitrary rotations of the common coordinate frame.
For true GS, Sigma=R diag(s^2) R^T remains active even when a=mu.

## Data, training and bounded verification

Exactly old R7: c3/c5/c6 192 TRAIN pairs each, c8/bm4 64 VAL pairs each;
576/128 total,432 queries and keys. No old or new TEST input/label access.
Load existing R7 depth arrays; join original RGB prediction caches by the saved
frame IDs and hashes. Cache whitelist: features,predicted_depth,K,c2w. Forward
model has an explicit input allowlist; target/valid/boundary only serve loss and
metrics. Keep every sensor-valid query center in loss/evaluation including
occluded/no-correspondence/out-of-source queries. V definitions remain R7's.

Thirteen roles x five paired seeds11,22,33,44,55;1000steps,batch8,AdamW lr.001,
wd.0001,clip1,log-depth L1. Original initialization and NumPy batch sequence
shared across roles; extra scalar initialization consumes no RNG. Evaluate
steps0,100,...,1000; earliest strict minimum equal-scene VAL AbsRel checkpoint.
Save all histories, selected weights and full validation predictions. Same
checkpoint zero-V/reversed-V and shape-off interventions are diagnostics only.

GPU float32,TF32off,deterministic algorithms,CPU4. Two idle GPUs may run disjoint
seed shards with identical configs, checking occupancy before launch. A short
synthetic smoke establishes finite gradients and real CUDA execution; no score
based tuning. Freeze source hashes and this contract before tests or real-array
decoding; metadata-only cohort/cache manifest first. Ordinary pre-run bug fixes
retain failed logs and source versions in a new attempt, without a new research
round. Resource ceiling2GPU-hours per shard and15GiB allocated per GPU, with
wall/time/memory measured. Failure does not trigger silent CPU fallback.

Verification: independent NumPy score formulas on synthetic and predetermined
real inputs, float64 toy score differences<=1e-9; float32 GPU versus float64
reference logit differences<=2e-4 and P<=2e-5. Record winner margins/differences,
not automatic exact-argmax failure near ties. Errors above these trigger an
impact diagnosis, never silent acceptance or deletion. Test center translation,
shape common rotation/unit differences, repeated eigenvalues, zero-shape limit,
finite gradients, and GT-field isolation. Independently replay complete saved
VAL depth metrics with existing NumPy audit functions (rel1e-6,abs1e-8; exact
counts/identities). These are research checks, not a general accuracy theorem.

## Decision, region/center diagnosis and real GS obligations

Primary must improve AbsRel at least2% over the best retrained old position
(point/point_pck/fixed), and the best center_rope/center_rbf; strictly beat
singleview; each comparison improves both scenes and at least4/5 seeds, with
RMSE<=2% worse than the selected position controls. Compare expected/normalized,
isotropic/shuffled and shape-off effects to attribute any gain. No population CI
from two repeatedly used scenes. Failure stops promotion of this fixed design;
no post-result frequency/strength rescue. Passing is only promising development:
compression/cost, full region/center scales, robustness, actual GS and untouched
joint confirmation remain open. All applicable original final thresholds stay.

Next within this research iteration, evaluate the SAME learned attention on old
regional/center labels, keeping whole-population metrics and the existing
conflict definition; groups are diagnostic, never a new denominator. This
requires a separately specified cached-data join and inference export, not a
change in the above primary or training selection. No evidence of resolving the
region/center conflict is claimed from depth alone.

Actual-GS work is scheduled independently of the static region gates: reuse
context-only R9 optimized geometry, define legal context-only query/key content
and shape sources, retain original point/fixed/isotropic/bilinear/GS comparison,
renderer/opacity/SH0 and full-pixel populations. A primitive-to-pixel interface
must not silently replace both-side shape information with uniform pixel Sigma
and call the resulting row constant a successful shape channel. Freeze that
task adaptation before its training/target evaluation. It remains required,
not presumed to follow from the patch experiment or a=mu activity test.
