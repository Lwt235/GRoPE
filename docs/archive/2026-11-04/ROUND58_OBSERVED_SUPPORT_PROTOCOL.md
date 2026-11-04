# R58: observed-support relations in native 3D local correspondence

2026-11-02, prospective development experiment. R57 is complete and NOT
promoted. This changes task and Gaussian semantics; it is not confirmation of
R57 or a claim to have solved the full research objective. No fresh confirmation
data or official ModelNet test is authorized here.

## Question and decisions

H: observed local support covariances provide useful within-cloud relational
geometry for a Transformer predicting cross-cloud region correspondences.
Represent the Gaussian relation as a sinusoidal relative embedding with a
learned query interaction, rather than impose a monotone negative divergence
on the final cross-cloud match. This separates contextual region description
from the location ultimately predicted. It does not guarantee their compatibility.

Competing explanations: local features alone suffice; ordinary distance/angle
geometry suffices; the covariance metric helps but its compatibility term does
not; an unconstrained MLP of the same invariant quantities suffices; gains depend
on sampling fingerprints or fragile support estimates.

Six roles, seeds11/22/33, all trained from scratch with the same backbone,
initial shared parameters, batches, labels and validation selection:

1. gaussian (primary): phi(sqrt(2 J)), where J is Gaussian Jeffreys/2 below.
2. gaussian_distance: remove only the covariance trace term before phi.
3. distance: phi(||c_i-c_j||/.2).
4. distance_angle: pair distance plus 3-neighbor triplet angles, the applicable
   GeoTransformer-style relative-embedding control (operator adaptation only).
5. invariant_mlp: an MLP of distance, both covariance spectra, both directional
   Mahalanobis terms and both trace terms; includes every scalar used by the
   Gaussian expression and ordinary distance. Its slightly different parameter
   count is reported, not hidden by padding unused parameters.
6. nope: identical support-derived content, without explicit attention PE.

A positive result needs consistent paired advantages over the geometric and
generic strong controls in region correspondence, with compatible center metrics
and prespecified robustness. Report effect sizes, seed signs and per-object
paired uncertainty, not an arbitrary fixed pp gate. A weak-control-only win or
inconsistent strong-control differences do not justify promotion. Positive local
evidence motivates a separately frozen real-scene task; it does not erase R57's
classification or rotation limitations. A negative result ends this specific
support/divergence embedding without temperature, frequency or budget rescue.

## Semantics and operator

Each token is the mean c and population covariance C of 32 observed neighboring
points around an independently selected FPS anchor. Sigma=C+.02^2 I explicitly
models an isotropic resolution floor; it is not a numerical fiction or a fitted
3DGS covariance. This uses the center/rotation/scale Gaussian form of 3DGS;
opacity and SH color are not used. Observed moments are not optimized splats.
All roles receive the same means, covariances and invariant local content.

For P_i=Sigma_i^-1 and delta=c_i-c_j,

    J_ij = .25 [delta^T(P_i+P_j)delta
                 + tr(P_i Sigma_j+P_j Sigma_i)-6].

phi uses the standard 96-dimensional sine/cosine embedding with frequencies
10000^(-2k/96), k=0..47. Under common Sigma=sigma^2 I the primary scalar becomes
||delta||/sigma. The Gaussian-distance control has exactly the same PE parameters.
No learned covariance, Gaussian mixture, new auxiliary loss or rendering module
is added. The classic divergence is not claimed as novel theory.

All PE projections are shared between the two self-attention layers. Self scores
are (q_i dot k_j + q_i dot r_ij)/sqrt(48), using unnormalized projected q/k.
Cross attention and final correspondence use content only. Two self/cross pairs,
96 channels,2 heads, pre-LN,2x FFN. A shared invariant11D content input contains
log covariance standard deviations (relative to .2) and8 log radial quantiles.
The final 48D descriptors are normalized; a learned bounded temperature is common
to all roles. No dropout. Position projections are constructed after the common
network to pair shared initialization.
The last relative projection has no bias: a pair-independent bias would add only
a query-row constant, which softmax cancels. The MLP's hidden bias remains learned.

Distance/angle control: sinusoidal distance/.2 and angle/(pi/12), nearest3 other
centers, learned separate96x96 projections, max over neighbors after projection.
This adapts Eqs4--7 in https://arxiv.org/html/2308.03768v1#S3.SS2 . It does not
reproduce KPConv-FPN, losses, dense matching or registration from that paper.
MLP input is [distance/.2, log(std_i/.2)[3],log(std_j/.2)[3],
log1p(delta^T P_i delta),log1p(delta^T P_j delta),
log1p(tr(P_i Sigma_j)),log1p(tr(P_j Sigma_i))],11->96->96,GELU.

Independent SE(3) changes of each input cloud preserve the forward score in exact
arithmetic: input features and every used within-cloud relation are invariant.
This claim includes the explicit isotropic resolution floor, but not general
affine invariance or robustness to resampling/noise. No local eigenvector frame
is compared across views. Relative pose is NOT an input or a prealignment.

## Data and target

Reuse only R48's official-train-only ModelNet partition:7910 fit,965 val,
965 old_dev_eval. IDs and archive identities remain unchanged. Old data do not
become fresh through a new task. Class labels are not model inputs or supervision.

For each object/realization, a deterministic random permutation separates2048
source points into two disjoint1024-point pools. Each view independently keeps
the top75% along a random direction and samples512 points without replacement.
No raw point ID occurs in both views. Independent noise std.005 clipped.02 and
independent uniform SO(3) rotation plus translation in[-.5,.5]^3 are applied.
Each view independently selects64 FPS anchors and32 nearest observed neighbors.
Two fixed fit realizations alternate by epoch; one fixed validation realization.
Input point order is shuffled and no IDs, labels, pose or overlap masks enter
the forward function. Samples are not cloned patches with shared random features.

Region overlap O_ij is the mean of the two fractions of clean support points
within.05 of ANY point in the other support. Support membership comes from the
observed noisy cloud; clean source coordinates are used ONLY for labels. Labels
are measured point-set overlap, never the Gaussian score. A direction's query is
supervised/scored if max_j O_ij>=.3; normalized O_ij*1[O_ij>=.1] is its soft target.
The mask is used only in loss/metrics. Bidirectional soft CE, equal weight for
the two directions with any valid rows, is the sole loss. Nonoverlapping rows
are not supervised; this does not test overlap detection or full registration.

Primary: mean bidirectional region Recall@1 at overlap>=.3. Also report selected
overlap, center PCK@.1, hard and soft center EPE, and CE, with query counts and
eligible fractions. Center errors compare observed token centers under the
known scoring transform; soft center is the weighted mean of all target centers.
Report nearest-center and best-overlap oracle bounds to expose region/center
conflicts and sampling ceilings. No oracle centers enter the forward pass.

## Fixed fit and evaluation

60epochs,7910examples/epoch,batch32,14880updates; AdamW lr.001 to.00001,
warmup3epochs,cosine,weight_decay.01,gradient clip1. Validation CE at0 and every3
epochs; strict minimum,earlier tie. No early stopping or tuning after results.
All18 models selected before old_dev_eval model predictions are exported.

Prespecified conditions, paired random draws where applicable:
base; noise02 (std.02 clipped.08,recompute support); sparse256 (first256 of the
same selected512,recompute64x32 support); low_overlap (keep50% of each1024pool,
512points). Evaluation has one realization per old object. These are development
stress tests, not fresh confirmation. Also verify rigid-frame and token-order
equivariance on bounded actual inputs. No additional augmentation rescue.

## Correctness, resources and integrity

Before fits: bounded tests for the divergence, isotropic limit, rigid invariance,
label/membership construction, forward label isolation, paired initialization,
nonzero PE gradients and finite CUDA updates. Freeze source/config/input IDs and
inspect actual fit/validation cache coverage. Preserve any ordinary pre-run fix
as a versioned failure record; it does not create a new scientific round.
After fits: verify all selections, source/data identity and paired streams;
recompute metrics by an independent NumPy path and bounded direct Gaussian KL
samples. Tiny errors without decision impact are recorded, not universal gates.
An independent arithmetic path is not an independent laboratory replication.

Use live-confirmed idle GPUs, two observations5s apart, at most6 workers, CPU4
per worker, no other-user job interference. Cache generation is batched GPU
arithmetic. Preserve original source arrays and all negative results. No changes
to libc/drivers. Stop and retain a failed child, never silently restart a run.
