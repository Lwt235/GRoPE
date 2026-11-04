# R59: attribute the learned Gaussian-relation PE result

2026-11-02. R58's scalar Gaussian primary failed. Its invariant_mlp control was
selected for follow-up AFTER development results: this is an explicitly adaptive
development decision, not a prespecified R58 success or independent confirmation.
The full research objective remains active and unachieved.

## Hypothesis, alternatives, decisions

H: the learned relative encoding benefits from directional relations between
observed Gaussian supports, beyond a flexible distance map or paired local
descriptors. The candidate is EXACTLY the R58 invariant_mlp, not a new module.
Alternatives: generic nonlinear distance suffices; direct local-feature comparison
suffices; center triplet geometry suffices; a generic MLP of complete relative
Gaussian matrices suffices without the chosen invariant contractions.

Keep all six R58 roles x seeds11/22/33 as exact references (18 old fits and their
72 existing prediction sets). Add four controls x the same seeds (12 NEW fits).
The fourth control receives full matrix information so an information-removal
ablation is not mistaken for a same-information architecture comparison.

Primary: invariant_mlp from R58,11->96->96,GELU,10368PEparameters.
New controls, all one hidden96D layer/GELU/final96D linear without final bias:

* distance_mlp: scalar ||c_i-c_j||/.2;9408PEparameters. Isolate nonlinear distance.
* descriptor_mlp: [distance/.2,f_i,f_j],23inputs,11520parameters. f is the SAME11D
  frozen local descriptor available to every R58 backbone (3log standard
  deviations and8log radial quantiles). Isolate direct local-content comparison.
* angle_descriptor_mlp: [distance/.2,f_i,f_j,angle/(pi/12)],24inputs,11616parameters.
  For each of the query's3nearest other centers, compute the angle to the key,
  apply the SAME MLP,then channelwise max over those3 embeddings. This is a
  flexible distance/triplet-angle control with the same local information.
* matrix_mlp: [delta/.2,svec(log(Sigma_i/.04)),svec(log(Sigma_j/.04))],15inputs,
  10752parameters. Matrix log is the unique symmetric spectral function,not a
  choice of eigenvector frame. svec lists3diagonals then sqrt(2) times(10,20,21).
  This retains COMPLETE relative center/full covariance information. It is
  translation-invariant but not architecturally rotation-invariant; this is a
  declared model property,not an implementation failure. Same uniform-SO(3)
  training inputs as every other model;no extra augmentation rescue.

Candidate inputs remain R58's exact11scalars:distance,3+3log covariance standard
deviations,log1p of both directional Mahalanobis terms and both covariance trace
terms. No new bandwidth,Gaussian predictor,loss,rendering,mixture or head.
All controls share the same four attention blocks and feature/descriptor layers.
Relative position enters self Q/K scores only,with no cross-coordinate alignment.

Interpret matched effects across3seeds and4conditions with region and center
metrics and object-conditional uncertainty. If distance_mlp matches candidate,
do not attribute gains to Gaussian support. If descriptor_mlp matches,retain a
local-content explanation. If angle_descriptor_mlp matches,the chosen Gaussian
contractions are not necessary. If matrix_mlp matches,cannot claim a task gain
from this particular invariant representation;compare declared frame behavior
without treating ordinary frame sensitivity as a failed numerical check.
Only reliable gains over relevant strong controls support a further real-scene/
cross-task development study. No local result completes the full objective or
authorizes fresh confirmation yet. No post-result width/budget/temperature scans.

## Fixed data, fitting and reuse

Use EXACT R58 fit0,fit1,val,eval_base,eval_noise02,eval_sparse256,eval_low_overlap
caches,hashed before fitting. No regeneration or modified labels. Still old
ModelNet official-train-only7910fit/965val/965old_dev_eval,not fresh data.
Original R58 region eligibility and center labels/counts remain unchanged.
Source geometry is observed support moments,not optimized 3DGS. No relative
pose,world-space scoring coordinates,IDs or overlap masks enter forward.

Same60epochs/14880updates,32batch,AdamW.001->.00001,3epoch warmup/cosine,
wd.01,clip1,validation CE every3incl0,strict minimum/earlier tie. Same actual
order RNG(seed+58000) and fit0/fit1 alternation. Common initial weights match
the R58 seed exactly. PE matrices have role-appropriate shapes;no unused padding
to feign equal capacity. Report full and PE parameter counts. Final projection
bias is omitted because it is a query-row constant canceled by softmax.

Reuse the frozen R58 fit loop through a scoped adapter that changes only output
root,model factory,scope validation and cache loader. Optimizer/loss/schedule/
sampling logic is the original function. No R58 source file is edited. All new
selections precede new old_dev_eval predictions. Old reference predictions were
already examined and are explicitly reused,not relabeled as newly blinded data.

## Fixed diagnostic and correctness scope

Pre-fit: independent NumPy relation/MLP checks,reference function identity,
matched shared initialization,finite/nonzero gradients and a CUDA update on old
fit inputs. Check rigid-frame/token-order behavior of invariant new controls.
For matrix_mlp,check the actual raw relative-matrix inputs under rotation and
record frame sensitivity;do not impose invariance on a non-invariant model.

Post-fit:all30 model selections,hashes,actual streams and parameter changes;
independent NumPy metric recomputation for120groups (72reused,48new). On bounded
actual validation inputs compare stored predictions with selected checkpoints,
and test token-order behavior. Rigid-frame differences for matrix_mlp are measured
as behavior,with probability differences/winner changes;only the three invariant
controls are expected to preserve scores. No audit of tiny errors as method gates.

Also export ONE prespecified base reframe diagnostic for ALL30 selected models:
rotate view0 byZ90 and view1 byX90 and translate both by(.7,-.3,.2);transform their
covariances accordingly,keep invariant input features and scoring world centers
unchanged. This changes coordinate frames only. It is a development diagnostic,
not a fifth independent task or a substitute for baseline gains. It is computed
after all selections;candidate/reference models are evaluated,not assumed exact.

Use live-confirmed idle GPUs,two observations5s apart,max6workers,CPU4. Preserve
other jobs and system GLIBC. Ordinary fixes retain failure versions;no new round
for a typo. Same-author independent arithmetic is not second-lab replication.
