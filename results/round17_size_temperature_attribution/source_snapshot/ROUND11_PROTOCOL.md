# Round 11 — restore local covariance in the existing four-node quadrature

Frozen before any Round11 sensor-label score is computed. R10 A/B remain failed
under their original gates. This is another old-development iteration, not fresh
confirmation and not evidence of general downstream gain.

## Evidence-led single change

R10 full1 improves on point but loses to16-node quadrature on coarse queries and
fails localization protection. B removes the known key Gaussian approximation
but retains those two effect failures. A forward-only diagnostic on the first
training pair of each scan compares against the dense predicted pixel measure,
without sensor/GT labels. Four local full Gaussians approximate this measure
better than full1 or quad16. These are representation errors against predicted
geometry, not sensor-task scores. Tiny-FOV sensitivity and all cases are retained
and separately reported in that diagnostic; no task rows will be removed.

Change exactly one covariance term in the existing quad4. Keep its four equal-
area strata, projected means, weights, parent content, key rectangles and h/T
search. Replace each stratum covariance by

    C_s = mean(pixel_covariance) + mean((x - mean_s)(x - mean_s)^T).

The mixture's first/second moments equal full1 by total covariance, while four
local centers preserve spatial separation. Each component uses the same B cell
integral; average masses before log, then add unchanged parent content score.
No adaptive subdivision, learned gate, extra head, optimizer or component-count
search. Components are fixed at4 for every query region.
They subdivide the current region (g8 gives4px strata, g16 gives8px strata),
not a fixed common atomization; exact query split/merge invariance is not claimed.

Primary `full4`; matched diagnostic `iso4` replaces each completed component
covariance by its same-trace isotropic covariance. Compare to frozen B full1,
quad4/16, fixed, isotropic, point/center and point_pck. This is established
mixture/integration mathematics, not a new kernel or exact five-channel Q/K map.
Supports come from RGB-predicted pixels after projection, not optimized GS R/S.

## Inputs, tuning and metrics

Reuse the exact R10 80-pair contract,48train/32oldval, all nine partition pairs,
area-weighted populations and finite fallbacks. Calibrate h in[.25,.5,1,2] and T
in[0,5,10] separately for full4 and iso4 on48train pairs. Freeze selections before
any new validation-label score. All B baselines retain their frozen configurations
and scores, verified by hashes and identical pair/partition/supervision identities.

Primary remains fixed16-query→8-key KL, sensor-visible mass weighted within
conditions, then equal conditions/scans. Query area pooling uses no sensor
visibility. Report all conditions/scans, direct KL, JS, independent g8/g8 PCK@8,
and numerical tiny-FOV counts. GL32, float64 integration, B's area-scaled1e−30
density floor and float32 prepared geometry are unchanged. Test reduction to
quad4 when scatter is zero, global moment conservation, gradients and independent
NumPy pixel-slice reconstruction.

## Unchanged acceptance gates

Full4 must pass all A/B development gates:

1. >=2% lower KL than B's calibration-selected point; both old-val scans improve.
2. Versus B's selected strong quadrature: >=2% lower KL with<=1.25x incremental
   PE time, OR <=1% worse KL with>=2x faster incremental PE.
3. No partition condition >2% worse than that quadrature.
4. g8/g8 PCK@8 loss<=.5pp against B's separately selected point_pck.
5. Independent input, numerical, saved-prediction, selection and metric audits.

Report full4 versus full1, iso4 and quad4 with costs to justify retained
information. Do not credit anisotropy if iso4 matches/exceeds it. Full4 stays the
primary even if a diagnostic appears better after validation. Also evaluate
iso4 at full4's selected h/T as an attribution diagnostic without new tuning.

Remeasure full4/iso4/quad16/full1 together on the first metadata pair of all five
scenes, nine conditions,3warmups+15 synchronized repeats. Include actual support
preparation, content pooling, positional score, softmax and common aggregation;
exclude transfers/shared frozen visual/depth predictors. Use these paired times
for the cost gate, not earlier clock samples. State tensor bytes separately from
allocator peak. If the B-selected strong quadrature is quad4, time it too; the
actual frozen B selection is quad16. Label any reused descriptive timings.

## Promotion and limits

A pass only permits new frozen confirmation on untouched scenes, inherited
scale/noise/numerical/point guardrails, and actual downstream utility checks. It
does not pass R7 depth/R9 RGB by implication. A failure remains a failure; this
protocol does not permit more components or relaxed point/partition thresholds.
