# Round 9 — real Gaussian primitives reading observed RGB

2026-10-30. A bounded development experiment following Round 8; not a confirmed
general PE or new Gaussian filtering kernel. Freeze before fitting GS or reading
any target RGB. Earlier sealed code/results and old test image blocks stay closed.

## Question and data separation

Does the internal projected shape of a context-fitted Gaussian help read useful
observed image colors compared with stronger center/fixed/isotropic controls?
Use Teatime and Waldo kitchen, both old development scenes. By camera filename
order, retain only the first two of the three historical equal blocks. Select32
evenly spaced posed frames; every fourth is target, giving24 context and8 target.
Target order alternates calibration/gate, four of each. The split is fixed without
target pixels. No target RGB enters SIFT, triangulation, initialization, training,
visibility, color readout, or prediction generation. Given K/R/t come from prior
all-image SfM: this remains conditional on supplied calibration, not complete
end-to-end unseen-view isolation. No old points, track associations, keypoints or
Gaussian checkpoint initialize the new reconstruction.

New SIFT matching uses context images only (max edge640, max2000 features, adjacent
1/2/3 context frames, ratio.75), fixed-camera triangulation, positive depths,
at-most2px reprojection error and at-least1degree ray separation. Color comes from
the same context RGB. Normalize by context-camera extent; de-duplicate at1e-4
normalized voxel width and cap12,000 points deterministically. Actual seed counts
are799/1728. Inputs are192×144 sRGB, resized using cv2.INTER_AREA, with half-pixel
intrinsic scaling. Arrays, source files and exporter hashes are separate from
target cameras; target image bytes are not hashed until authorized evaluation.

## One shared GS reconstruction per scene

Train one context-only, view-independent RGB/SH0-capacity Gaussian reconstruction
per scene for a fixed3,000steps, seed11, one context image per step. Learn means,
three positive scales, unit quaternion, opacity and one sigmoid RGB per primitive;
Adam grouped rates .00016/.005/.001/.05/.025 respectively. Mean LR decays by100×.
Optimize full-image RGB MSE. Clamp scales to[1e-5,.2], color/opacity logits to[-10,10]
and clip gradient norm10. At steps500/1000/1500/2000, split at most
min(N,12000−N) parents ranked by opacity×largest scale: two children at±.5 of the
largest covariance-axis standard deviation, scales divided by1.6, copied color and
opacity. Reset Adam after topology changes. No candidate PE or target influences
this fixed growth rule. Use the final checkpoint, never target-selected weights.

Renderer is installed gsplat1.5.3, existing verified CUDA extension, classic alpha
compositing, eps2d=.3, black background, identical in training and all readout
roles. Integer image centers are mapped to renderer half-pixel centers by adding
.5 to K's principal point. All used intrinsics must have zero skew. Preserve
original context-trained RGB rendering as the system reference on this same
geometry. This small SH0 reconstruction is not the old million-primitive SH3 model
and is not an official full-budget 3DGS reproduction.

Report context reconstruction MSE/PSNR/alpha and growth histories. If context
quality is poor, do not attribute failure solely to PE. No target-driven geometry
budget extension, initialization repair or selection is allowed within this run.

## Fixed geometry, color readout only

Freeze means, covariance, opacity, renderer, source cameras, source set and fusion
for all controls. Project actual primitive covariance from optimized R/s into
each context image. Use double precision for readout geometry and five-dimensional
quadratic features. Keys are the regular pixel grid with common covariance I/12;
all pixels participate in softmax, with memory chunking but no spatial cutoff.

Four geometric controls each receive exactly h={.25,.5,1,2} pixels:

- full: actual projected primitive covariance C;
- isotropic: trace(C)/2 times I;
- fixed: I, a fixed one-pixel-squared query support;
- point: zero query covariance, retaining I/12+h²I.

Also include standard bilinear center sampling (no tuned h). All use actual source
RGB as V. No image content Q/K, learned head, confidence network or GT depth is
added. Geometry-only five-dimensional dot-product softmax is algebraically the
same normalized elliptical Gaussian averaging as the corresponding direct EWA
kernel. Independent direct Mahalanobis evaluation checks equality; it is not an
accuracy competitor or novelty claim. The readout kernel is not the alpha renderer
kernel: its pixel/bandwidth terms differ from renderer eps2d.

For each source view, normalize pixel weights separately. Fuse all eligible source
views with equal weight; never concatenate views after dropping view-specific
query constants. Source visibility is fixed using the original GS geometry and
opacity: projected center lies in[0,W−1]×[0,H−1], positive z, bilinearly sampled
rendered alpha≥.5, and |z−D|≤.02+.03D where D is rendered expected depth in context
normalized units. This is an approximate geometry-only visibility rule, especially
for transparency and overlapping layers. No role color or target RGB affects it.
Every role has the identical source-view eligibility. If no source qualifies,
retain the original context-trained primitive RGB. Save source-view counts and
alpha-composited target contribution of fallback primitives; do not drop such
target pixels from the primary score.

## Calibration, gate and stopping

Generate all18 role/configuration predictions for all8 target cameras before any
target RGB evaluation. The independent analyzer first opens calibration RGB only.
Select one h per role by equal-scene/equal-image full-pixel RGB MSE across the two
scenes'4calibration images; exact ties use declared h order. Save selection and
all input/prediction/source hashes before opening gate images. Then evaluate all
selected controls once on the four gate images per scene. All RGB channels and
all image pixels are included. PSNR and frame/scene scores accompany primary MSE.

Primary comparator is the lowest gate scene-macro MSE among calibrated point,
fixed,isotropic and bilinear controls (a conservative predeclared envelope, not a
post-hoc weak baseline). Use that same comparator in both per-scene checks. The
calibration-selected best control is also reported. Continue only if full lowers
primary gate MSE by at least2% and improves both scene means. Two reused scenes do
not support population confidence intervals or an independent-generalization claim.

Separately compare full with the same-geometry original GS RGB system reference.
If full only beats simpler sampling but loses to original GS, retain at most a
mechanism result, not a useful improved system. If the primary gate fails, stop
this branch without adding confidence/color/depth modules. New independent scenes
are justified only after a deployable candidate passes development and receives a
new frozen confirmation protocol. Deterministic repetitions are not training seeds.

## Verification

Freeze local source plus renderer wrapper/native-library hashes and data hashes.
Independent checks cover context-only splitting and triangulation, covariance R/s
identity, projection/half-pixel convention, color-capacity parity, full-pixel
five-dimensional versus Mahalanobis equality, shared visibility/fallback, source
view normalization, finite predictions and NumPy metric reconstruction. Source
and checkpoint changes or missing completed runs stop evaluation. Preserve failed
attempts and numerical corrections with explicit provenance.
