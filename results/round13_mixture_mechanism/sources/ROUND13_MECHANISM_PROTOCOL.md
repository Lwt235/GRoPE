# Round13: label-free mixture-shape mechanism diagnostic

Freeze this protocol, source/tests, and the three input identities before any
new forward result. This is a representation diagnostic, not a research gate,
candidate calibration, sensor-truth test, or authorization to resume depth.
R12 remains failed. No validation/test array or sensor label/image is opened.

Use the metadata-first old training pair in each of Harvard c3/c5/c6, exactly as
the frozen Round11 compression diagnostic. Select eight q24 native query indices
`floor(linspace(0,47,8))`, namely [0,6,13,20,26,33,40,47], and all 432 g8 keys.
No query filtering. Every query's 576 predicted pixels are available to every
method. Image geometry and parent features come only from those frozen
Round10 `inputs.npz` forward observations. Hash the input archive without
opening labels, and record the precise NPZ fields read.

Fix h=.5 (4 pixels). Compare dense (all actual predicted pixel Gaussians), full4
(four 2x2 stratum full moments), ou4 (same full4 transformed below), full1 (one
full moment Gaussian), and quad16 (sixteen stratum means with average pixel
covariance only). Same full4 bank feeds ou4; do not independently refit it.
No learned parameter and no bandwidth or rho search is allowed. Geometry is
float64, integrated over the actual native key rectangles by the frozen GL32
operator. Sum component masses first, add background density 1e-30 times native
key area once, then normalize over the complete field of view. Retain and
report all zero/tiny-FOV cases (threshold 1e-20).

For ou4 compute global mean μ and covariance Σ from the supplied component
weights, means and covariances. Set rho_squared=2/3 exactly, then
`mu_s'=mu+sqrt(2/3)*(mu_s-mu)` and `C_s'=(2/3)*C_s+(1/3)*Sigma`.
This preserves first and second moments but need not improve dense fidelity,
unimodality, center alignment or real labels. The identity 24px counterexample
in `docs/round12_failure_review.md` already shows dense-reference KL increasing
from 0.001070 to 0.010847; that negative evidence is part of the motivation to
test rather than assume this update.

For each representation evaluate T=0 and T=5. All use the same normalized mean
of the query's 3x3 base8 content features and the same normalized native key
features. In particular dense pixels do not receive individual content tokens.
The final distribution is `softmax(log(region_cell_mass)+T*q_content*k_content)`.
The dense comparator uses the same T as the candidate. Changing T is a fixed
mechanism contrast, not configuration selection.

Define an independent **predicted center**, without sensor truth: clip the
frozen q_predicted_depth field by the original nan10/+inf20/-inf.1/[.1,20] rule;
bilinearly sample it at the exact q24 half-pixel center; backproject this ray
with q_K, transform by q_to_k_transform, project with k_K using the original
z>=.1 and xy clipping fallback. Record unmodified in-front status and FOV
membership. This is not the mean of projected region pixels and is not GT.
Center-bin summaries use only in-front and in-FOV predicted centers, reporting
the denominator. All 24 queries remain in distribution-fidelity summaries.

Save complete probabilities, geometry masses/FOV, content scores, selected
query/key bounds, predicted centers and validity, input/source hashes, and each
query's metrics. For each method/T report KL(dense||method), JS, L1, agreement
with dense argmax, mode distance to predicted center, center-bin probability
and log-probability margin over every other key. The center bin is native key
containment; it is not a PCK label or a replacement decoder. Report all methods'
T0→T5 argmax changes and paired per-query changes. Verify full1/full4/ou4 moments
against the dense mixture. Preparation/integration wall times are descriptive
for eight queries only, not a paired efficiency claim or a new timing gate.

Tests before freeze use only synthetic inputs: weighted moments, no mutation,
positive semidefinite outputs, affine equivariance of the moment transform,
autograd, bad input rejection, identity continuous-vs-discrete distinction,
analytic CDF comparison, and exact synthetic predicted-center projection.
After completion an independent replay of saved arrays must recompute metric
summaries and source/input/output hashes. No parameter will be adjusted from
these results. Any later real candidate requires a new preregistered protocol
and all inherited applicable gates; this diagnostic cannot mark any gate passed.
