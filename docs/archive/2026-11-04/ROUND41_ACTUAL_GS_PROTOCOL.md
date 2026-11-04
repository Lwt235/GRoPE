# R41: learned RGB readout from actual context-trained 3DGS

2026-11-01, old development only. Freeze implementation/data identities before
new array extraction, tests and training. R40 failed; this experiment tests the
same compact family in a direct actual-GS value task, not a rescue of its depth
result or a new confirmation. Static integrator gates are not prerequisites.

Question: does separating center phase and physical shape help read real source
RGB into true GS primitives, beyond strong spatial/content controls? Alternatives
are ordinary locality, extra scalar capacity, scale alone and shape identity.
Primary remains `decoupled`, never the best observed ablation.

## Geometry, observations and direct values

Reuse only R9 context-trained Teatime/Waldo GS state, 24 context images per scene,
fixed geometry/opacity/SH0 rasterizer and the original eight target cameras.
Query a=mu and Sigma=R diag(s^2) R^T. No anchor correction or old SUN3D checkpoint.
Coordinates use R9 normalized camera-extent units, not known metres. Train from
scratch in each scene; this does not establish cross-scene weight transfer.

Frozen official local ResNet18 V1 layer2 (128 channels), ImageNet normalization
on the existing 192x144 context RGB. Bilinear upsampling to all192x144 pixels,
align_corners=False, then channel normalization. Primitive-center features/RGB
are bilinear samples in pixel-center coordinates, align_corners=True. Features
are frozen; Q/K is a shared learned128x128 linear map initialized to identity.

Source eligibility is exactly R9: projected GS center inside the closed image,
z>1e-4, bilinear rendered alpha>=.5, abs(z-expected_depth)<=.02+.03expected_depth.
All27648 source pixels remain keys. Backproject context-only GS expected depth
into the source-camera frame; replace nonfinite/nonpositive/alpha<.5 pixels by
the median eligible depth of that image, clamp to[1e-4,1e4]. No key is deleted.
Central differences of backprojected points, one-sided at image edges, define
the pixel covariance (du du^T+dv dv^T)/12. Record invalid fraction and actual
variation of key shape descriptors; constant descriptors cannot support a shape
claim. Every role receives identical observations, eligibility and fusion.

The source RGB itself is V; prediction is softmax(logits)@RGB. No learned V,
residual color path, extra color decoder, opacity update or geometry refinement.
Final per-primitive color is an equal average over eligible context views,
falling back to the original context-trained GS RGB when none are eligible.
Render every target pixel with the unchanged R9 renderer. Cameras derive from
all-image SfM, so this remains a conditional-camera experiment.

## Roles and optimization

Eleven learned roles: center_rope, expected_rope, normalized_rope, decoupled,
shape_isotropic, shape_shuffle, center_rbf, point, fixed, full, isotropic.
The first seven use R40's score definitions:64 axis-cycling frequencies with
norms logspace(0,1.3), temperature10, central7D shape descriptor, shape penalty
strength initialized1. Frequencies are per normalized scene unit. Shuffle uses
fixed independent primitive and pixel permutations from default_rng(20261101),
with all centers/features retained. center_rbf adds one learned center-distance
scalar. The four spatial roles use temperature10 plus the direct projected
Gaussian distance with query covariance respectively0,I,physicalSigma2D, or
trace(Sigma2D)I/2; common pixel covariance I/12 and bandwidth.5px. Direct
differences avoid unnecessary quadratic cancellation. Q/K has16,384 active
parameters; scalar roles16,385. No inactive capacity padding.

For each scene, context indices3,9,15,21 are selection-target views; other20
are training/source views. Training query descriptors average visible features
from the20 training views, excluding the current target view. Sources are a
different training view. For selection targets, fusion uses all20 training
views. Fixed original GS geometry used all24 contexts, so selection is only a
development proxy, not independent geometry validation. Final inference uses
all24 context features and all24 source views; target-view RGB never enters.

Train each scene/role with five paired seeds11,22,33,44,55,500steps,AdamW lr.001,
wd.0001,clip1. Each batch draws uniformly one nonempty ordered train-view pair,
then64 primitives with replacement from those visible in both views. Minimize
RGB MSE against the target context's bilinear center RGB. Evaluate steps0,100,
...500: four selection targets x source positions0,5,10,15 in the20 training
views, up to32 uniformly indexed common-visible primitives per pair. Choose
earliest strict minimum mean RGB-element MSE. Save schedules, all histories,
selected checkpoints, gradient/weight changes and full inference colors/renders.
Final inference uses256-query chunks; chunking never crops source keys.

Retain all18 original R9 readout methods as external references, including all
four fixed bandwidths, bilinear and original_gs. These have no new learned Q/K.
Compare learned roles fairly within the shared adapter and report comparisons
to the stronger external methods separately. No target RGB read until every
scene/role/seed prediction is sealed. No target-based checkpoint/bandwidth search.

## Evaluation and proportionate checks

GPU float32,TF32off;CPU4. Shared batches and source identities, finite gradients,
sample NumPy score/RGB replay, constant-V and shape-activity tests. Deterministic
training where supported; renderer is frozen and its runtime identity recorded.
Max allocated16GiB,3GPU-hours per scene/seed shard. A failure is recorded, never
silently converted to CPU training. Ordinary implementation fixes retain source
and logs in a new attempt; no new scientific round or full historical re-audit.

Save complete probabilities on eight fixed first-source-view primitives per
model, and independently replay their scores and RGB aggregation. Float32 P
error>5e-5 or RGB>2e-5 is an impact warning, not automatic scientific rejection.
Metric replay covers all saved RGB pixels; no exact-argmax gate. For the primary
also render reversed-source-V and shape-off interventions from the SAME weights.
These diagnose actual value/shape use, never replace the primary.

Report all eight reused target views, with original4 calibration/4 gate splits
separate. Primary decision uses the original gate split: at least2% mean MSE
improvement over the best learned point/fixed/center_rope/center_rbf control and
the best original R9 point/fixed/bilinear reference; strictly beat original_gs;
improve both scenes and at least4/5 paired seeds. External bandwidths are all
reported; taking their strongest gate result is a conservative comparison,
not candidate selection. No scene-population CI from two reused scenes.

Passing is only development evidence: R40's depth failure, applicable full
geometry/robustness/scale/cost, internal R/S attribution and untouched joint
confirmation still require resolution. Failing rejects promotion of this
fixed candidate; it does not prove every compact GS-derived PE impossible.
