# R61 normal/spectrum attribution — fixed development comparison

2026-11-02. Implements docs/round61_normal_relation_plan.md. This experiment
tests whether full covariance directional relations add utility beyond normal
point-pair geometry and spectra. Competing explanations: normal relations suffice,
or nonlinear precombination rather than additional geometric information helps.
R59/R60 candidate was chosen adaptively on OLD development data. No novelty,
general effectiveness, optimized-3DGS or fresh-confirmation claim is pre-established.

Primary: exact R58 invariant_mlp. Reuse five R59/R60 reference roles:
invariant_mlp, distance_mlp, descriptor_mlp, angle_descriptor_mlp, matrix_mlp.
Add only normal_ppf_mlp and axial_mlp; each seeds11/22/33. All use the exact
R58 backbone/self-cross integration/loss. Normals come from the SAME measured
covariances; no extra pose, appearance, oriented normals or labels enter forward.

normal_ppf_mlp: d/.2, original six log-standard-deviations, u^T N_i u,
u^T N_j u, tr(N_i N_j), where N=n n^T at a simple minimum eigenvalue;
zero displacement uses u=0.10->96->96 GELU,last linear without bias,10272 PE
parameters. This is an unsigned PPF-inspired same-backbone control,not RoITr.
axial_mlp: original distance/six spectra; four original log1p metric/trace terms
computed with axisymmetrized covariance.11->96->96,10368 PE parameters exactly
as the candidate. Keep original spectra and all local content features.

FP64 eigendecomposition; minimum eigenspace includes eigenvalues within
1e-6*lambda_max of minimum. Its projector Pi has rank k. PPF uses Pi/k.
Axial tensor averages eigenvalues separately in Pi and its orthogonal complement;
k=3 uses trace(Sigma)/3 I. For k=1 this preserves the normal eigenvalue and
averages the two tangent eigenvalues. No task-selected degeneracy threshold.
Record k frequencies on fit0/fit1/val and each evaluation cache. Input supports
are fixed measurements; trainable-parameter gradients are checked,not an unused
gradient through repeated-eigenvalue eigenvectors.

Reuse exact R58 synthetic caches (7910fit,965val,965old_eval) and R60 native SUN3D
caches (576fit,64val,480old_eval). Cache NPZ and metadata identities are fixed.
No new sampling, scene choices, normalization, targets, reserved scene access
or official ModelNet test. Native depth remains public input on the real task;
pose only forms supervision/scoring. These are observed supports,not optimized splats.

Six NEW synthetic fits use exact R58 train loop/config:60epochs,14880updates,
batch32,AdamW lr.001->.00001,warmup3,weight_decay.01,clip1,validationevery3.
Common initialization matches original references per seed; actual stream uses
seed+58000 and alternating fit caches. Source validation CE selects earliest
strict minimum. Each source checkpoint initializes its OWN real-target fit:
six NEW real fits,60epochs,1080updates,lr.0001->.000001,remaining R60 config
unchanged. Target initial backbones differ between roles; compare complete
source-plus-target pipelines. Identical actual target data/order/budget.

Seal all six source selections,then run six target fits; seal all12 NEW choices
before any new old_eval export. Source validation is used only for source
selection. Source performance will not change the already fixed real fits.
Existing15source+15adapted reference checkpoints and120prediction groups are
reused by identity,not refitted or counted as independent repetitions.

Evaluate synthetic source on base/noise02/sparse256/low_overlap; evaluate real
source-direct and real-adapted on base/sparse256.48NEW groups+120reused=168.
Synthetic primary is original query-weighted regionR@1; real primary is original
five-scene macro regionR@1. Report paired seeds,scene differences,centerPCK,
softEPE,CE and query eligibility. Sparse changes supports/query populations;
compare matched roles within each condition. No query-IID confidence interval.

Bounded new checks: independent NumPy formulas,rigid/permutation behavior,
degenerate eigenspaces,axisymmetric equivalence/common initialization,actual
learning gradients/finite difference; post-fit selection/stream/checkpoint
identities,CPU replay,independent new metric arithmetic. Existing identity-checked
reference metrics need not be redundantly recomputed. Checks cover correctness,
not scientific utility. Tiny floating-point differences are not new scientific
rounds or reasons to repeat old static integrator gates. Any substantive failure
gets a preserved failure/source record before a correction.

Use GPUs only after live occupancy/CUDA checks; two idle observations before
each job,max6workers,owned process groups only. Preserve original queue handles;
never restart from a status-file delay. No budget,temperature,width or loss search.

Decisions: PPF matching/exceeding candidate weakens full-Gaussian necessity and
motivates simplification. Candidate beating PPF but matching axial points toward
precombination rather than full tangent directions. Reliable gains over BOTH
across synthetic and adapted real tasks support keeping full covariance for
cross-task/actual-GS validation. Conditional/unstable gains narrow claims and
remain evidence;no automatic optimization rescue. Interpret magnitude,seed/scene
consistency and center cost rather than require every historical metric to win.
Preserve R55/R57/R58/R59/R60 negatives and the original full research objective.
