# R64: a single depth field for Gaussian centers and supports

Prospective development experiment, 2026-11-02. Read the unchanged historical
plan in docs/round64_depth_field_plan.md for hypothesis and outcome decisions.
R62/R63 are complete. This tests RGB usefulness, not physical reconstruction.
No rendering objective, new backbone, supervised depth or new pretraining.

## Fixed model and comparisons

Retain the exact R62 camera transport, two 96D/two-head crossattention layers,
query-conditioned relation addition and 11->96->96 Gaussian relation map.
Predict one positive depth per token/layer with the original zero-initialized
96->1 head and original softplus mapping. Pixel rays/intrinsics/cameras remain
known inputs. All RGB features are the same frozen extractor output.

mu=z*r. With pixel-depth derivatives, J_u=(z/fx,0,0)+r*z_u and
J_v=(0,z/fy,0)+r*z_v. Sigma=5.25*(J_u J_u^T+J_v J_v^T)+(1/300)^2 I.
The depth unit is 3m; the physical covariance floor remains 1cm. Reconstruct
the complete 18x24 grid from pixel coordinates, use central differences /16
inside and one-sided /8 at edges, then return to token order. No smoothing.
This is a local surface approximation, not exact arbitrary patch integration.

Three new roles: depth_surface (primary), depth_footprint (same center/depth,
zero derivative terms in J), depth_local_mlp (endpoint [z,fx*z_u,fy*z_v,r_x,r_y]
concatenated into 10->96->96 MLP, same attention interaction). The last control
receives the same local field information. Initial constant-depth surface and
footprint geometry/weights agree exactly. Record 10465 versus 10369 PE
parameters; only shared backbone initialization is asserted equal to references.

Eight old RGB roles, each three seeds, reused by exact identity: R55 camera_only,
camera_plane,ray_point,ray_affine; R62 camera_mlp,free_relation,predicted_full,
predicted_point. Their scientific source and data identities remain frozen.
Learned geometry need not equal sensor geometry; independent parameter range
and coupling both change relative to R62, so no isolated-DOF attribution.

## Training, evaluation, and decisions

Nine new fits: three roles x seeds11/22/33, from scratch, 5000 updates, batch8,
AdamW weight_decay.0001, gradient clip1. Use exact R55 cosine .001->.0002 over
first1000steps, then .0002, actual sampling seed+50000. Validate at0 and every
100steps; earliest strict minimum CE selects. Same576/64/480 old RGB pairs,
three/one/five scenes. All nine selections precede any new old_eval exports.
Labels/observed depth/covariance excluded from model input; no label token mask.

33 models,66 prediction groups:18new,48reused. Baseline and original fixed
RGB permutation RNG510000+i, independently for both images. Same all-query
label population and five-scene-macro soft EPE primary; retain CE,PCK4/8,AUC8,
argmaxEPE,hard/soft>16px tails,seed/scene effects and15/60frame-gap strata.
Permutation is an RGB-dependence diagnostic, not a robustness success claim.

Keep the primary only if it improves stably over the same-information MLP,
footprint and strongest existing references. If footprint alone works, prefer
its simpler geometry. Beating only the failed R62 model is insufficient. If
all fail, reconsider RGB integration; no automatic budget/module rescue. No
fresh confirmation access. ActualGS,cross-task,applicable geometry/robustness/
generality and unused confirmation remain required by the complete goal.

## Proportionate checks

Before freezing: synthetic analytic Jacobian/covariance, canonical finite
differences including boundaries, positive floor, finite prediction gradients,
common initialization, identical surface/footprint initial output, independent
q/k token permutations and exclusion of privileged inputs. Repeated tangent
eigenvalues use eigvalsh; finite gradients matter, not an arbitrary anisotropy
rule. Then a three-update CUDA fit-only learning/memory preflight per role.

Freeze concrete sources/configuration and reference/data identities. After
training, verify actual stream, budget/selection, checkpoint/validation identities,
one-pair CPU replay and independent NumPy full-grid derivatives/geometries plus
an8x8relation sample. Recompute all66metric groups independently. Tolerances
distinguish substantive implementation defects from harmless floating error;
they are not general scientific acceptance criteria. Preserve pre-run failures
with source versions and fix them without inventing a new scientific round.

Only twice-observed idle GPUs; verify CUDA access before fits. Preserve others'
jobs. Report actual selected results and all negative findings. This is neither
a full-paper HRoPE reproduction nor evidence of novelty or general effectiveness.
