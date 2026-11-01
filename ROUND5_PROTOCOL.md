# Round 5: predicted Gaussian support inside two cross-attention layers

Date: 2026-10-30. Development and confirmation protocol. Test outcomes must not
select architecture, frequency, estimator, scenes, metric, or training budget.

## Claim and information condition

Test whether Gaussian support improves coarse dense RGB correspondence through
actual attention/V aggregation on held-out SUN3D places. The depth estimator
uses RGB only, but all methods also receive dataset intrinsics and supplied
RGB-D reconstructed camera poses. This is **not a strictly RGB-only system**.
Sensor depth is used only for training targets/evaluation. No GT validity mask
or sensor depth enters forward, candidate pruning, input support, or scale fit.

The operator is Gaussian expected RoPE followed by content L2 normalization;
the equality is independently tested. No novel expectation formula, full
WorldRoPE reproduction, optimized 3DGS primitive R/S contribution, or NVS claim
is made. One Gaussian is fit to each predicted-depth patch. These are Gaussian
supports in the 3DGS parameterization family, not trained rendering primitives.

## Data and split

Official SUN3D native RGB/depth/pose source. Original SUN3D toolbox defines
pre-aligned RGB/depth, uint16 depth right-circular-shift by 3 bits then /1000m,
1-based pixel intrinsics, nearest RGB/depth timestamp association, and C2W
extrinsics in full RGB-list order. Original lists, selected ordinals, hashes,
and source URLs are retained in data/round5_sun3d/selection.json and manifests.

| Split | Place/sequence | Pairs per place |
|---|---|---:|
| Train | harvard_c3/hv_c3_1; harvard_c5/hv_c5_1; harvard_c6/hv_c6_1 | 192 |
| Validation | harvard_c8/hv_c8_3 | 64 |
| Test | harvard_c11/hv_c11_2; hotel_umd/maryland_hotel3; mit_lab_hj/lab_hj_tea_nov_2_2012_scan1_erika; mit_76_studyroom/76-1studyroom2; mit_dorm_next_sj/dorm_next_sj_oct_30_2012_scan1_erika | 96 |

Place IDs are disjoint; repeated scans of a place cannot cross splits. This is
place-heldout, not building-heldout. Five test places support a limited estimate;
three MIT rooms do not establish three independent institutions. Pretraining
overlap of foundation models is not known; heldout refers to this matcher.

Pair selection is linspace query indices from 30 to N-91 and fixed offsets 15
and 60 frames (96/32/48 query indices for train/val/test), selected without RGB,
depth, or prediction scores. All selected pairs remain, including zero-valid
pairs. Sparse frames are downloaded directly to avoid multi-GB full archives.
UW capture IDs were rejected as independent-room evidence; the 3DMatch archive
server returned HTTP500, so native SUN3D is used instead.

## Inputs, geometry, labels

- RGB resized to 192x144 with BOX for the frozen ImageNet ResNet18 V1 layer2.
  Output 128D features are sampled at the 8x8 patch centers, then L2 normalized.
- Frozen official Depth Anything V2 metric Hypersim Small, input518, max20m;
  source commit a561b849ebae10a6f5ef49e26c83cbbcd36c71bf and checkpoint SHA saved.
  No test GT scale alignment, no finetuning. Predicted depth is area-resized to
  192x144, bounded [.1,20]m with fixed finite replacement, shared by all methods.
- Each 8x8 token has uniform 3D point moments from predicted depth. Native-view
  mean is its exact known image-grid center and native covariance is the exact
  discrete uniform footprint 63/(12*64) I in stride units. Cross-view means and
  covariances are first-order projected 3D moments. This approximation may be
  inaccurate near discontinuities. Cross-camera z uses a .1m denominator floor;
  projected mean is bounded +/-128 strides and covariance eigenvalues [0,256].
  Fallback counts are reported and never determine evaluation inclusion.
- 24x18=432 tokens per image, centers3.5,11.5,... in zero-based resized pixels.
  GT query center depth is nearest-sampled at native resolution; transform by
  shared cameras, require target image bounds and valid sensor depth with
  abs(z_projected-z_sensor)<=.05m+.05*z_sensor. Holes/behind/outside/inconsistent
  depth get separate reasons; these masks are supervision only.
- Correspondence labels are bilinear four-key weights; outer half-cell targets
  clamp to edge keys. No boundary/same-surface or predicted-quality filtering.
  This is matching **conditional on GT-visible correspondences**, not occlusion
  rejection. Report visibility fraction and nearest-grid PCK ceiling.

## Architecture and controls

Shared128->96 projection; two distinct bidirectional cross-attention blocks,
each with two48D heads, QKV, output, residual, LayerNorm and96->192->96 FFN.
Directions share layer weights and update simultaneously. Attention Q/K are
unit-normalized before PE with shared learned temperatures bounded60. Only
attention Q/K use PE. The final shared96->48 normalized content matching head
has no extra geometric bias or PE. All six controls have166,611 parameters and
identical seed initialization and batches:

1. `point`: projected mean RoPE, zero covariance.
2. `fixed`: normalized expected Gaussian with both supports replaced by I/12.
3. `expected`: Gaussian expectation without post-normalization.
4. `normalized`: primary Gaussian support candidate with post-normalization.
5. `isotropic`: normalized support with both sides replaced by same-trace I.
6. `resolvent`: predeclared secondary filter, not primary model selection.

Non-point methods add the same .25-stride blur standard deviation. Frequencies
are24 golden-angle directions with magnitudes logspace(-1,.6), unchanged across
methods. Both directions use the same geometry cache. Native supports are fixed
by observation; cross-projected key support is where anisotropy can contribute.
The distinct GS internal-covariance center-only question is not answered here.

## Training and sealing

Training/validation-only development first; verify finite gradients, small-set
fit and throughput. Planned confirmation budget: all six methods x five seeds
(11,22,33,44,55),1000 AdamW steps,batch8,lr.001,weight_decay.0001,gradient clip1,
cosine lr decay to20%. Validation every100 steps including initialization;
select minimum equal-place macro CE (earliest on ties). Any development-motivated
change must be written here before confirmation, and test remains unopened.

Confirmation code and inputs are hashed. Each completed checkpoint is hashed;
round5_freeze.py verifies all30 runs and chosen validation steps, selects the
primary strong baseline from point/fixed by five-seed mean validation PCK8
(point on ties), and writes an immutable confirmation lock. Test-only verifies
this lock and replays validation before opening test arrays. Test data cannot
be used for checkpoint, baseline, or hyperparameter selection.

Use currently idle GPU0 (and rechecked idle GPU1/GPU2 for independent
confirmation groups), without touching occupied cards. Estimated total budget
<=4 GPU hours; actual time/memory saved. New source files do not change frozen
Round3/4 models or results.

## Primary outcome and interpretation

Primary metric: scene-macro PCK@8 resized pixels, query-weighted within each
place then equal-place average. Report CE, PCK4, argmax EPE and soft-argmax EPE.
Gaussian normalized candidate passes only if, versus the validation-selected
point/fixed baseline: mean PCK8 gain>=1pp, paired seed/place crossed bootstrap
95% CI lower>0, at least4/5 seeds positive, and CE relative worsening<=2%.
CE guardrail is an empirical tolerance, not statistical noninferiority.

Fixed20,000 crossed seed/place draws, RNG20261030; no token-independent CI.
Report all methods, per-place and per-seed results, leave-one-place and
Harvard/Hotel/MIT group sensitivity. With only5 test places the CI has limited
population scope. Other comparisons are exploratory, no familywise guarantee.
Anisotropy requires supporting comparisons against isotropic and fixed; an
overall point-baseline gain alone is insufficient. The resolvent is secondary.

If the gate fails, retain the negative result and distinguish a conditional
matching gain from successful downstream transfer. Do not select a favorable
test subset, swap primary metric, or automatically expand to NVS.
