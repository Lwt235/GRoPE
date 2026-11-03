# R55: one bounded, matched extension of the original training processes

2026-11-02. Execute docs/round55_budget_diagnostic_plan.md without changing the
model classes, inputs, data, feature source, loss, geometry bounds or frequency
conventions. R52/R53/R54 are closed; their scientific sources remain immutable.
This is the single budget/feature-source diagnosis allowed after R54, choosing
budget alone. It is development evidence, not unused confirmation or general PE
success. Original PRoPE/HRoPE adaptations and their scope limitations still apply.

Hypothesis: learned local geometry needs longer joint optimization than the
fixed camera operator or ray-center control, and the original1000-step budget
underestimates its incremental utility. Alternatives: more optimization helps
all equally; local geometry overfits; or this operator/feature/task combination
offers no usable shape gain. A checkpoint at the budget endpoint is motivation
for this test, not proof of its explanation.

## Fixed comparisons and intervention

Eight unchanged roles x seeds11/22/33=24 new fits:
camera_only,camera_gs from R52;camera_offset,camera_plane from R54;
ray_point,ray_iso,ray_gaussian,ray_affine from R53.
The model factory returns these exact existing classes. All controls receive
5000 updates,batch8,AdamW lr.001/wd.0001,clip1,FP32/noTF32. Initial state and
actual batch sequence match the respective original run. Updates0..999 use
the original cosine schedule with denominator1000,ending near.0002; updates
1000..4999 use constant.0002. No schedule sweep or conditional budget extension.

The old optimizer state was not retained, so rerun from original initialization
and continue its optimizer, rather than restarting Adam from a selected model.
Every100 steps validate, selecting the earliest minimum CE over0..5000. Preserve
the0..1000 selection and validation history for comparison with original runs;
record batch/shared-init identities, selected weights and function differences.
Tiny numeric differences trigger impact inspection, not a new rejection gate.
No evaluation result influences training, model selection or the fixed budget.
All24 selections are sealed before48 baseline/RGB-permuted export groups.

Save model,optimizer,best state,RNG,sampling identity,history and next update
every100 steps with an atomic replace. A completed run cannot be overwritten.
Resuming an interrupted run must verify the original lock and preserve history.
A bounded CPU test compares continued optimization with serialized restoration.

Exact R52 arrays:576fit/64val/480old_eval pairs from five reused rooms. Preserve
source locks and original model/run identities. No unused scene or ModelNet40
official test is opened. No token true depth,normal,covariance or correspondence
label is added to model inputs. NoPE/MLP1000-step numbers are historical only;
this experiment cannot establish a matched5000-step advantage over them.

## Outcomes and decisions

Primary contrast: ray_gaussian minus ray_point in scene-macro continuous EPE,
and its change relative to the original1000-step contrast. Always compare with
ray_iso/ray_affine before attributing an effect to full Gaussian shape. Auxiliary
contrasts: plane vs P and offset; forwardGS vs P; all changes from original
1000-step models. A method improving with budget is not enough: the incremental
advantage over equally trained controls must improve or remain credible.

Report CE,PCK4/8,AUC0..8,argmax/continuous EPE,p95,>16px tails,three paired seeds,
five scene means and fixed15/60-frame gaps,RGB permutation,parameters/time/memory.
Plot fixed train/validation histories. Train interval CE is an optimization
diagnostic on changing sampled batches, not held-out evidence or whole-fit CE.
No post-hoc error strata,temperature sweeps or per-query independence claims.
All focal ratios1, so no zoom claim. Small inconsistent differences are ties;
no arbitrary1pp gate or static-integrator prerequisite is added.

If shape still offers no stable incremental utility or ties a generic affine
transform, end this framework's tuning rescue on this task. Revisit native3D
center/region structure with a separately stated falsifiable hypothesis. If
only planes gain, reconsider the geometric role of Gaussian shape. If full
Gaussian gains survive point,iso and affine controls, proceed to native3D and
cross-task development, then freeze for unused confirmation. None of these
outcomes alone achieves the original generality objective. If5000 steps still
leave optimization uncertain, report that limitation without automatic extension.

## Bounded implementation and audit checks

Reuse the existing model/operator tests; no operator has been modified. New
tests cover exact factory/common initialization, original schedule/batch prefix,
and optimizer/RNG restoration. Before freezing, verify all24 original source,
data,initialization and selected-function identities plus finite real-input
gradients for eight roles. After freezing run one CUDA smoke.

Use only GPUs observed idle twice5s apart, one worker per device,CPU4. Preserve
other users' jobs. Check current occupancy/CUDA; retain old systemGLIBC.
Audit all24 selected first-layer samples with the existing independent NumPy/
SciPy arithmetic, full CPU validation CE and all48 saved metric groups. These
are same-author independent calculations, not another lab or full-network rewrite.
Warning scales5e-5 probability/5e-6 CE prompt impact inspection. Audit effort
must remain proportional to possible effects on the conclusion.
