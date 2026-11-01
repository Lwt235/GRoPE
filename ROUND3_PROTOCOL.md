# Round 3: Gaussian parameters as per-token positional encoding

Written before third-round test evaluation, 2026-10-29. This is a research
protocol, not a promise that a positive result exists.

## Question and candidate

Does encoding spatial *support* as a normalized square-root Gaussian density
help attention beyond projected centers and a fixed isotropic footprint?
For G=N(mu,Sigma), p_G=sqrt(G) has unit L2 norm. Encode Q=q tensor p_G,
K=k tensor p_G. Their inner product is (q dot k) times the Bhattacharyya
coefficient. This is the classical probability product kernel, not new kernel
mathematics. Fourier importance quadrature produces a finite, independently
computed position vector. All coordinate/shape effects enter through this
one map; no shape bias, gate, uncertainty head, or post-test calibration.

The tensor lift expands the mathematical head dimension. Factoring the two
Gram matrices avoids materializing it but still costs O(N^2 (D+2M)). It is not
a free replacement for 48-dimensional RoPE or an orthogonal 48x48 transform.
Projection is per target camera, so keys must be encoded per view pair.

## Stage A: controlled bridge to round 2

Six predeclared methods with identical RGB encoder, temperature, dustbin head,
training batches and optimizer: 3D RoPE; projected point RoPE; expected RoPE
with fixed isotropic support; expected RoPE with projected shape; square-root
density PE with fixed support; square-root density PE with projected shape.
Two-dimensional RoPE frequencies are identical across its three variants and
include oblique directions so off-diagonal covariance is observable.

Development seeds 91,92 and data namespaces 1100/1200/1300 are exploratory.
Confirmation seeds 11,22,33,44,55 use new namespaces 2100/2200/2300.
Validation cross-entropy selects checkpoints. No test tuning. All failed
variants remain in results. Spectral feature count is fixed at 256 initially;
if development approximation checks require changing it, freeze the choice
before confirmation. Parameter counts, timing, memory, source/data hashes,
per-scene results and histories are archived.

Oracle training and balanced oracle/mixed/depth_fit training are separate.
Tests: oracle, depth noise, normal noise, scale noise, mixed, depth_fit, wide
baseline, and the SAME frozen RGB geometry predictor (round2 seed11) for all
methods/seeds. Predictor checkpoint hash must be recorded. Primary comparison
is shape versus fixed support within the SAME encoding family; comparison to
3D RoPE alone cannot establish a shape contribution. Report visible Recall@1,
unmatched AP/false acceptance and calibration together, plus paired per-scene
and per-seed differences. Center-point labels remain a limitation/negative
control for support reasoning.

## Stage B: independent coverage evidence

Construct labels from actual shared surface samples or independently observed
SfM tracks, never from the Gaussian overlap formula being evaluated. Fit each
view's region support independently, with no cross-view per-token random ID.
Use genuine trained 3DGS as a geometry source when available. Keep camera
splits disjoint for real-view training/validation/test. Sparse SfM absence is
not an unmatched/occlusion label. Real-track coverage and synthetic dense
surface coverage must be named distinctly; neither is a full NVS benchmark.

Required controls: fixed support, isotropic same-area support, covariance
shuffle, exact continuous kernel versus finite feature approximation, and
same-information absolute descriptor MLP if learned shape encoding is tested.
Any positive synthetic mechanism finding must be replicated on independent
data before calling the overall research objective achieved.

## Trust criteria

A candidate advances only if its support contribution is positive across
seeds and has a paired confidence interval excluding zero on held-out data,
without an unreported severe rejection or geometry-noise failure. A meaningful
real-data advantage and correct finite-map behavior are necessary before
claiming a strong final方案. Failure changes the hypothesis, not the labels.
Unit tests establish correctness only, not effectiveness or novelty.

## Compact projected isometric candidate: confirmation commitment

The world-space descriptor, expected, and compact isometric development runs
did not beat projected baselines. The parameter-free world BC evidence is
therefore treated as a mechanism diagnostic, not final PE success. The next
compact candidate uses per-token expected 2D rotation plus its complementary
content channels: P_G q=[a_G R(mu_G)q; sqrt(1-a_G^2)q], a_G=exp(-w^T C_G w/2).
This is one 48->96 dimensional map, without pair gates or shape biases. Its
norm is exactly preserved for fixed observed SPD supports; no claim of new
operator mathematics, true overlap/IoU, or zero-covariance gradient regularity.

Following development seed91, freeze sigma=.25 patch resolution and the shared
24 oblique frequency pairs. Confirmation uses seeds11,22,33,44,55, new rendered
namespaces2100/2200/2300 for the point bridge and4100/4200/4300 for dense regions.
The methods are projected_rope, expected_fixed, expected_shape, isometric_fixed,
isometric_shape, isometric_isotropic (same trace), isometric_area (same det),
isometric_shuffled (fixed covariance derangement), and absolute_geometry.
All comparisons use identical batches, encoder and optimization; learned
descriptor baseline has more active parameters. Fixed and shape isometric
models have the same 96D output. Validation CE alone chooses checkpoints.

Real fitting initially uses teatime train and validation cameras; no real-test
metrics or external scene outcomes select parameters/checkpoints. Include step0
and validation every20 steps (real small-data overfitting was seen in dev).
Test camera blocks and supplied GS subset swaps are evaluated only after
training. Full-cross-scene generalization must be reported alongside in-scene
performance, and strong validation-frozen geometric RBF controls are compared
with exactly the same row mask. Existing exact-kernel test results are known;
these camera blocks are an independent fitting test, not a completely untouched
benchmark of the entire research process. A further unseen-pair replication is
required if the candidate advances.

## Development numerical amendment (before confirmation)

The independent audit found that 256 Fourier samples do not approximate BC
adequately, especially for nearly singular projected supports. The mixture
quadrature now gives each proposal component a complete Sobol net. Earlier
development runs and both approximation audits remain archived. Future runs
use M=1024 and a fixed resolution sigma=.25 patch units, adding .0625 I to
both support covariances BEFORE the map. This deliberately blurs the support
kernel at a quarter-patch scale; it is not merely a numerical implementation
change. These choices follow geometry-only development approximation checks,
not confirmation outcomes. Unblurred exact kernels remain diagnostic controls.

Stage B controls additionally include trace-matched and determinant-matched
isotropic support separately, a fixed cyclic covariance derangement, a learned
absolute 5D mean/covariance descriptor MLP (more active capacity), and an
Expected-RoPE isometric completion with D->2D, as a compact alternative. Query
region covariance is known from its actual pixel rectangle and shared by all
support methods. Only the key shape differs among the support controls.

Primary kernel reference: Jebara, Kondor, Howard, Probability Product Kernels,
JMLR 5 (2004), https://www.jmlr.org/papers/volume5/jebara04a/jebara04a.pdf .


## Independent exact-kernel confirmation (frozen before test opening)

This zero-shot CPU confirmation evaluates `experiments/real_support_geometry_probe.py` on the held-out test camera blocks of `results/round3_real_support_data_adjacent`. The old development outputs remain unchanged. Method, score mode, RBF width, and softmax temperature are copied exactly from `results/round3_real_geometry_dev/summary.json`, SHA256 `8e345bc5dc4b8d41da8c6c0329ee9b1fce52861950f8895311a178e13e541ba6`; all settings are archived in `results/round3_real_geometry_confirmation/frozen_settings.json` before test NPZ files are opened. No test tuning, fitting, or learned RGB model is permitted.

The primary comparison is full 3D Gaussian exact Bhattacharyya coefficient (BC), normalized directly across valid keys, versus `world_plane_isotropic`: both retain the measured region centers, normal direction, thickness, and tangential area; the control removes tangential ellipticity and its orientation. The primary metric is row-averaged soft cross entropy against sparse observed-SfM-track transfer targets. These targets are not dense coverage truth. Report paired differences with 4,000 bootstrap samples of (scene, query-camera) clusters, pooled and separately for teatime and waldo_kitchen. Top1 target mass and tied-max-bin accuracy are secondary.

Prespecified secondary controls include 3D same-volume sphere, 3D same-trace sphere, 3D center RBF, native-query/full projected BC, fixed projected footprint, same-area/trace projected spheres, projected-query BC, projected-query same-area sphere, covariance shuffles, and the stronger 2D native/projected center RBF baselines. Every development-selected bandwidth and temperature remains frozen. `BC / sum(BC)` is the primary score mode; `softmax(temperature * BC)` with each previously selected validation temperature is reported separately, not conflated.

Masking deliberately reproduces the independent development probe: an original query row has at least four shared tracks and valid query geometry; invalid keys are removed; any row with strictly positive remaining target mass is conditioned on retained keys. Lost target mass and all exclusions are reported. This is the retained-mass >0 protocol, distinct from the main runner's >=0.95 sensitivity restriction.

The odd/even GS partition swap (`_alt`) is evaluated with exactly the same frozen settings. In addition to each partition's own validity mask, a common intersection of query/key validity is used to compare original and swapped geometry with identical rows, candidate masks, and conditional labels. Test results may validate, weaken, or reject the development finding; favorable kernel results alone do not establish a compact finite positional encoding.


## External-scene exact-kernel confirmation (zero further tuning)

Before opening external-scene NPZ data, we freeze the identical 36 method/mode/width/temperature configurations already archived in `results/round3_real_geometry_confirmation/frozen_settings.json` (SHA256 `8ebbdb3bf038ee2b97beec4b99d805a7be1baecb8b541cda4554ece35662240e`). That original freeze file is not changed. We evaluate the newly exported `results/round3_external_scenes/figurines` and `temple` geometry and sparse-track targets with these settings, without any fitting or selection on either scene. The matcher/probe has never fitted or selected configurations on either external scene.

All source train/validation/test camera blocks in each external scene are eligible as external evaluation data because none was used for the method's development. Pair rows remain correlated within a query camera; uncertainty is estimated using paired (scene, query-camera) cluster bootstrap samples and is reported separately for each external scene as well as pooled. The primary exact-BC comparison, soft cross entropy metric, positive-retained-mass masking, alternate disjoint-GS partition swap, identical-mask sensitivity, and all geometric controls remain as in the independent held-out-camera confirmation. World-center RBF normalizes coordinates using each source's existing scene-scale metadata; no target-derived scale is estimated.

The optimized 3DGS reconstructions can have used the source photographs, including their reconstruction-validation convention. This experiment tests conditional matching with an available reconstructed scene, not reconstruction from held-out photographs. Sparse SfM observation transfer remains a proxy, not dense coverage or occlusion truth. All external outcomes, including failures and disagreement with earlier scenes, must be retained.

## Normalized compact Gaussian PE: archived prospective protocol

The authoritative pre-training commitment is
`results/round3_normalized_confirmation/frozen_protocol.json`, SHA256
`f07f70c053ee654e46da634fbc4bfc4ba6fe0c09b40843ee808f06a614d754b9`.
It was written before the five-seed fitting and fresh-camera matcher
evaluation. This paragraph is an append-only index, not a replacement of
that commitment or a new success criterion.

The candidate applies per-token Q/K normalization after Gaussian expected
rotation, preserving the original 48-dimensional head and adding no learned
parameters. Fitted projected query mean/covariance is shared by all methods;
fixed/trace/area/shuffle controls intervene on key covariance. Both raw RGB
and a common frozen ResNet18 representation are predeclared, with raw RGB
the primary input. Five seeds use only Teatime train and validation. The
primary data are previously unexported camera pairs; figurines+temple form
the prespecified primary group. Old test blocks are not new discovery data.

The frozen JSON contains every method, metric, inference rule, source/data
hash and support/anisotropy gate. Existing strong RBF settings remain frozen
from the earlier geometry-development sweep. A later independent arithmetic
audit additionally groups five or ten adjacent query cameras as a
correlation sensitivity check; this is explicitly post hoc, and cannot
replace the primary result or choose a favorable partition.

The point/rejection negative-control repeat and the direct dense synthetic
mechanism verification each have a separate `frozen_protocol.json` in their
`results/round3_normalized_*_confirmation` folders. They retain the previously
used synthetic namespaces and are never described as fresh data discovery.
