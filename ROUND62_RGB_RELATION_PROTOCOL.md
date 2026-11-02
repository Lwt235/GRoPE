# R62: reconnect Gaussian relation PE to RGB matching

2026-11-02. Prospective development experiment. No untouched confirmation data.
The full goal remains unachieved. R61 is complete; do not change frozen sources.

## Question, competing explanations, and decisions

H: RGB/camera-conditioned Gaussian centers and shapes, organized as the R58
11-dimensional relative relations, improve calibrated RGB correspondence beyond
direct learned PE and known-camera geometry. H0: depth/center alone, a generic
learned pair relation, or the known camera operator explains any gain. Another
possibility is that the encoding works with measured geometry but the learned
parameterization/interface cannot exploit it under the fixed training process.

The first test reconnects the relation map to the existing RGB task. It does not
attribute older RGB failures to bad predicted parameters, recover a physical
texture transport map, or establish generality from one task. ActualGS and
cross-task checks remain required and are deferred, not cancelled.

If learned full relations improve over both center-only and direct learned
relations plus the camera references, retain the RGB candidate for robustness
and generalization development. If only observed geometry helps, investigate
prediction/interface/optimization instead of concluding prediction quality is
the cause. If observed center alone explains its benefit, do not credit shape.
If both observed and predicted geometry fail, reconsider this task integration.
Mixed results narrow scope. No automatic budget/temperature/width rescue.

## Matched task and controls

Reuse exact R55/R51 frozen ResNet RGB features, calibrated cameras, 432tokens
per view, nine old SUN3D scenes and576fit/64val/480old_eval pairs. Model input
contains RGB features, pixel coordinates, intrinsics and relative cameras.
Sensor depth and label-derived validity/visibility are excluded from all RGB
methods. No token is filtered or masked using target labels.

Eight roles, seeds11/22/33:
- camera_only and camera_plane: exact R55 PRoPE-style and calibrated-HRoPE
  implementations/models/predictions reused by identity; six existing fits.
- camera_mlp: same P and V/output transport, learn a2D pixel-phase offset from
  RGB/camera features using96->16->2 MLP, bounded as the earlier directMLP.
- free_relation: seven unconstrained learned token coordinates, concatenate
  both ends and both normalized pixel positions (18inputs),18->96->96 MLP.
- predicted_full: one depth and six positive-Cholesky covariance coordinates
  per token, followed by the exact11->96->96 relation map.
- predicted_point: same ray/depth family, fixed isotropic covariance. The same
  11->96->96 map isolates the incremental learned-shape contribution.
- observed_full and observed_point: PRIVILEGED geometry diagnostics, with the
  same relation map and backbone, measured or fixed covariance respectively.

Six new roles x3seeds=18new fits;24models total. Same common initialization,
actual sample stream, optimization, loss and validation rule. PE parameters
are recorded, not asserted equal. In particular free_relation has a larger
first map than full, and observed groups omit the learned Gaussian head.
There is no added backbone block, geometry supervision, rendering loss or
pretrained geometry predictor. End-to-end here means learned head/PE/matcher;
the shared RGB feature extractor remains frozen as in R55.

## Integration and parameter meaning

Retain the two96D/2head cross-attention blocks and all known camera Q/K/V/output
transforms from camera_only. Add q_raw dot r_ij /sqrt(48) to each attention logit,
where q_raw is the query projection before normalization/camera transport.
The R58 formula is reused for r; its placement changes from within-view self
attention to calibrated cross-view attention. This is an explicit task adapter,
not an unchanged R58 network or a PRoPE/HRoPE full-paper reproduction.

Predict geometry from the same pre-attention normalized features at each layer.
Centers lie on each token's known camera ray. Depth in3m units is
1/30+softplus(raw_depth+inverse_softplus(29/30)),initially1 (3m).
L has diagonal (.025,.030,.010)*exp(2*tanh(raw_diag/2)) and lower off-diagonal
.03*tanh(raw_lower). Sigma=L L^T+(1/300)^2 I; floor is1cm in physical units.
The distinct initial scales avoid an exactly repeated initial eigenvalue in
the learned spectral map. This is a numerical coordinate choice, not a recovered
surface prior. A positive Cholesky factor represents the same SPD covariance
family as GS rotation/scale; no Cholesky texture-transport interpretation.
All new parameter-prediction heads start with zero weights/biases.
Point controls use Sigma=(.025^2+(1/300)^2)I. Relative inputs keep R58's distance/.2,
log standard deviations relative to.2, and log1p of two Mahalanobis/two trace
terms. Transform camera-local centers/covariance to the common query frame
using the same known cameras in every applicable role; no predicted GT pose.

Observed diagnostics sample native depth independently in each8x8 image patch
using the existing pixel-resize/nearest-depth conventions. The center is the
center-pixel ray times measured depth. Missing center depth uses patch median;
an entirely missing patch uses3m and the fixed point covariance. Covariance is
the second moment of valid patch points around that center, plus the same1cm
floor. This is an anchored Gaussian support approximation, not an optimizedGS
primitive or a ground-truth uncertainty covariance. Positive depths are clipped
to[.1,20]m; retain all tokens and report fallback counts. Observed geometry is
computed without correspondence/visibility labels and never used by RGB roles.

## Fixed execution and evaluation

18new fits use5000steps,batch8,AdamW weight_decay.0001,gradient clip1.
Learning rate and actual sampling reproduce R55: cosine.001->.0002 over first
1000steps,then.0002; RNG seed+50000. Validate every100steps including0; earliest
strict minimum validation CE selects a checkpoint. Do not claim convergence.
Common initial weights/streams must match R55; reference models are reused
only after exact data/source/configuration and implementation identity checks.
All18new selections precede any new old_eval predictions/observed caches.

Primary: five-scene-macro continuous pixel EPE of the probability expectation.
Secondary: PCK4/PCK8,CE,argmaxEPE,AUC8,continuous/argmax>16px tails and per-scene
and paired-seed effects. Retain15/60frame-gap strata, not isolated surface/zoom
claims. Baseline and fixed within-image RGB permutation diagnostics;48groups
total including12exact reference groups. Predicted heads see permuted RGB;
observed geometry stays fixed. That diagnoses RGB dependence, not its unique
causal mechanism. Use original label populations/denominators in all roles.

Correctness: independent NumPy relation/anchored-support checks, common-frame
and permutation checks, nonzero finite prediction/PE gradients and finite
differences, label/privileged-input exclusion, exact zero-addition camera
reduction and shared initial weights. Then a bounded CUDA learning/memory
preflight using fit data only, never selecting a scientific design by outcomes.
After selection: bounded independent operator arithmetic, CPU checkpoint replay,
validation-selection and actual-stream audit, separate NumPy metric sums.
Numerical tolerances address consequential discrepancies, not universal gates.
Record ordinary pre-run failures and fixes within this round before freezing.

Freeze this protocol, all concrete dependencies/configuration and original
data/model identities before training; new cache hashes carry that lock.
Only twice-observed idle GPUs are used; keep other users' processes intact.
