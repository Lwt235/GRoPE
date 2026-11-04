# Round 6 confirmation: five-channel Gaussian support positional encoding

Version 1, 2026-10-30. This protocol is written before new test labels or method
scores are evaluated. The SHA is locked by `experiments/round6_confirm.py`.
Round 5 remains a sealed negative result for normalized Gaussian RoPE.

## Question and deliberately small model

Does projected Gaussian region support improve real visible-area correspondence
over equally calibrated center-distance/known-size baselines, across scene and
token scale, while preserving point localization? This is conditional posed
RGB-D evaluation: forward depth is predicted from RGB, but cameras are supplied
RGB-D reconstruction metadata. It is not an unposed or fully RGB-only system.

Let a source region have projected mean `mu` and covariance `C` in target-image
coordinates (common units: eight pixels). Every native target cell in the same
attention set has the same covariance `D=s_k^2/(12*8^2) I`. Define

`A=(C+D+h^2 I)^(-1)`,

`Qg=[(A mu)_x,(A mu)_y,-A_xx/2,-A_xy,-A_yy/2]`,

`Kg=[x,y,x^2,xy,y^2]`.

For conditioning, subtract the same mean target-grid origin from both means.
`Qg Kg^T` equals the log Gaussian overlap up to a query-row constant. The native
key-area factor and log determinant also cancel within each uniform target grid.
Concatenate `sqrt(T)*f` with these five channels, where `f` is the existing
128-dimensional frozen visual feature. Softmax of the resulting **unscaled**
133-dimensional dot product gives correspondence/attention weights. Applying
those weights to V is ordinary attention; no extra learned body, gate, mixture,
uncertainty predictor or geometry MLP is added. There are zero new learned
network parameters and two global calibration scalars, h and T.

This is an established quadratic Gaussian-attention construction (Cordonnier
et al., ICLR 2020, Appendix E), with per-query precision obtained from projected
region support. Neither the Gaussian kernel nor five-dimensional quadratic
lifting is a novelty claim. The finite lift requires common target covariance;
arbitrary unequal Gaussian keys require the full pairwise operator. Support
covariance is not calibrated estimation uncertainty or optimized 3DGS R/S.

## Development and fixed calibration

Development uses old Harvard c3/c5/c6 train scenes and c8 validation plus new
Brown bm4 validation. The primary model is deterministic; training seeds are
therefore not experimental replicates. An earlier two-block, 300-step body
experiment is retained as development evidence only, not a converged baseline
or a primary result. Parsimony favors the zero-additional-network solution.

Every method receives the same 35-point grid: `h in [.0625,.125,.25,.5,1,2,4]`,
`T in [0,2.5,5,10,20]`. The final calibration uses all fixed 64 pairs in each of
the two validation scans, all four token-grid conditions, and equal scan/grid
weight. No test result influences the grid, selection, or success thresholds.
Minimum area KL selects each method; exact ties retain first grid order.

- `point`: both support covariances zero; same projected means and placement.
- `fixed`: source covariance replaced by its known native cell size; target
  size retained, including the 8/16 difference.
- `isotropic`: each covariance replaced by half its trace times identity.
- `full`: full projected query support plus known native target support.
- `point_pck`: an additional point control selected by maximum validation
  8->8 center PCK@8 from the same 35 candidates. It protects against using an
  area-calibrated point baseline that unnecessarily sacrifices localization.

The area baseline is the better validation KL of point and fixed, with point
winning an exact tie. Candidate is always full; isotropic is an attribution
control, not a test-dependent fallback candidate. Current calibration gives
full/fixed/isotropic h=.5,T=5; point h=1,T=10; point_pck h=.5,T=10. The freeze
tool derives and verifies these from the saved grid records rather than trusting
this prose. It replays validation with the five-channel implementation before
authorizing test export. Optional full T=0 is a fixed counterfactual diagnostic,
not a separately tuned competitor or a selection option.

## Fresh scene selection and sampling

The immutable primary selection is `data/round6_sun3d/selection_v2.json`:

1. brown_cogsci_1/brown_cogsci_1
2. brown_cs_3/brown_cs3
3. mit_32_d507/d507_2
4. home_at/home_at_scan1_2013_jan_1
5. home_bksh/home_bksh_oct_30_2012_scan2_erika

Each has 48 linearly spaced source ordinals and offsets 15/60: 96 pairs per
scan, 480 total. Full RGB ordinal-to-pose layout is checked. Sensor depth is
associated by nearest timestamp; maximum gap for home_bksh is about 32.1 ms,
others about 16.7 ms. All fixed pairs remain, without model/label-quality pruning.

The v1 candidate Brown bm1 was removed **before any candidate score or label
evaluation**, because fixed RGB previews could not rule out shared offices or
corridors with validation Brown bm4. v1, raw downloads, rejection reason and
v2 replacement metadata are retained. Brown bm1 is excluded from confirmation.
Named scan/place groups are the statistical units. Distinct names and RGB QA
do not prove independent buildings; do not describe these as five independently
sampled buildings or a universal benchmark population.

## Forward and supervision separation

Forward receives only frozen RGB features, RGB-predicted depth, K/C2W and known
cell size. The original pretrained weights and preprocessing are unchanged.
Projected covariance is 3D sample scatter plus the continuous footprint of each
individual pixel, propagated by the projection Jacobian. Native covariance is
the continuous square-cell second moment. No sensor-depth scale alignment,
visibility mask, target labels or quality filter enters forward.

Labels count **every 192x144 source pixel center once**, nearest-sampling the
native sensor depth with the audited half-pixel coordinate convention. Project
with supplied cameras; require positive source depth, in-frame target and
target-depth agreement `abs(z_projected-z_sensor)<=.05+.05*z_sensor`. Assign to
the hard containing target cell. These are source-image visible-area samples,
not all native 640x480 pixels, physical surface area, or Gaussian-generated
labels. Normalize each row by its surviving count V; retain all V>0 rows.
Keep all candidate tokens in forward even when they have no supervision.

The four conditions are 8->8, 8->16, 16->8, 16->16. They reuse exactly the same
pixel transfer measure; coarse/fine mass conservation is independently tested.
Independent query-center sensor projection supplies point targets, not the area
label centroid. No 95%-visibility or single-surface condition defines the primary
population. Report visible coverage and fixed [0,.25),[.25,.5),[.5,.75),[.75,1]
strata as sensitivities. These strata never change model inputs or selection.

## Aggregation, uncertainty and predeclared gates

For each scan/grid, area KL/CE/JS is weighted by V (each observed visible source
pixel has equal weight). Then average scans and four grids equally. Also report
row-equal KL, entropy, point PCK4/PCK8/PCK at target stride, EPE and coverage.
KL=CE-H(y); different class counts/target entropies cannot be treated as accuracy.

Use 20,000 paired scan bootstrap draws, RNG 20261030, sharing the same five-scan
resamples across all methods, metrics and grid conditions. Report central 95%
intervals. Do not treat pixels, overlapping frame pairs or repeated deterministic
runs as independent replicates. Five groups offer limited population coverage;
include every per-scan result, leave-one-scan-out and Brown/MIT/home group-out
point estimates without manufacturing a three-cluster confidence interval.

The proposed solution passes only if ALL of the following hold:

1. Full versus the validation-selected area baseline reduces total KL by at
   least 2%, with candidate-minus-baseline difference CI upper bound <0, and
   improves at least four of five scan-level four-grid averages.
2. Each cross-scale condition separately has at least 2% KL reduction and
   a difference CI upper bound <0.
3. For each same-scale condition, the central 95% CI upper bound of relative
   KL deterioration is <+2%.
4. In 8->8 center matching, PCK@8 candidate-minus-control CI lower bound is
   >-0.005 against BOTH the area baseline and the PCK-selected point control.

Full-vs-isotropic attribution intervals are exploratory, without familywise
coverage claims. A primary gain does not itself prove that anisotropy matters.
The success gates are an intersection, not a choice of favorable endpoints.
No candidate substitution or post-test recalibration is allowed.

## Integrity and reproducibility

Before test export, freeze this protocol, analysis and execution sources,
selection v2, all calibration records/configs and input/label/manifest hashes
for validation. Verify scan disjointness and complete row identities. Replay
each selected validation configuration through the compact implementation.
After export, verify all 480 pairs in all four conditions and equality of shared
supervision across roles before analysis. Save complete per-pair metrics and a
fixed two-pair probability subset per scan/grid/method for independent NumPy
recalculation; this subset is not described as full prediction replay.

Record actual compute and numerical agreement with the exact Gaussian kernel.
The Gaussian lift and input isolation have independent analytic tests. No old
sealed Round5 source/result is rewritten. All further designs after confirmation
would require a new declared exploratory status and new confirmation data.
