# R39 saved-output Schur diagnostic — prospective contract

2026-11-01, selected after the accepted R39 fixed-design failure (0/8), before
any new archive decode or decomposition execution. This is a diagnostic on
existing predictions, not R40, an intervention, or a candidate-selection sweep.
The original research goal and every applicable performance gate remain open.

## Exact population and read boundary

Use every saved R39 pair in lock order: 48 old TRAIN pairs, 48 q24 queries and
432 native g8 keys per pair; all three fixed roles full/point/isotropic. Thus
2304 queries per role and 6912 role-queries. No front, FOV, rank or depth-based
exclusion from the algebra audit. Invalid/zero key rays make the diagnostic
incomplete, not a silently dropped subset.

The allowlist is the R39 forward archive's five geometry fields, content, and
all five saved fields (lograw, logfloored, logp, p, winner) for each role. Read
only `k_K` from the already authenticated old input archive to locate predicted
query anchors on the image. No predicted-depth rerun, labels/targets, GT-valid
mask, model inference, new softmax, counterfactual probabilities or parameter
search. Before decoding, authenticate the R39 acceptance, post-result decision,
formal inventories, archive hashes and current/frozen source bytes. Maintain an
attempt/decoded-field access log. Data already used as old development remain
development; no new confirmation data are opened.

## Fixed decomposition

For each role use its declared Qq/Qk and S=gamma I+Qq+Qk, symmetrized only after
the R39 1e-12 normalized covariance symmetry/PSD admission. Let r be the unit
key viewing ray. Producer chooses the coordinate axis least aligned with r
(first argmin), u=normalize(cross(r,e)), v=cross(r,u), basis columns [u,v,r].
Auditor uses an independently implemented stable Householder transform:
t=+1 if r_z>=0 else -1; w=r+t ez; H=I-2wwT/(wTw), then multiply H's third row
by -t. This gives the same radial axis and a possibly different transverse
basis. Basis orthonormality and radial-row identity errors must be <=1e-10.

In that frame S=[[A,b],[bT,c]] and delta=(dT,dR). Set s=c-bT inv(A)b>0 and
e=dR-bT inv(A)dT. With no clipping, diagonal jitter or covariance repair:

    td = -.5 dT^T inv(A) dT
    tv = log(gamma) - .5 logdet(A)
    rd = -.5 e^2/s
    rv = .5 (log(gamma)-log(s))
    lograw = log(9) + td + tv + rd + rv

Save all seven matrices td,tv,rd,rv,s,residual=e,radial_delta=dR. The two
independently authored numerical modules use different bases and 2x2 solve
arithmetic. The same documented reader/row bookkeeping and summarizer are
used with each result, and receive separate static review; they are not
represented as independently authored statistical implementations.

## Anchor comparison and score bookkeeping

Keep all rows for the general stage comparisons. A predicted-anchor cell is
defined only for q_anchor z>0, finite projected coordinates, and half-open
physical FOV [-.5,191.5) x [-.5,143.5). Use floor((xy+.5)/8) to identify its
native g8 cell; no projection clipping or alternative winner. Record finite
projection, front/FOV eligibility, and exact cell-boundary status separately.
This population is descriptive and is not the original 1096 GT-valid centers.

Record raw, floored, logits, log-P and saved-P first argmax. For all five stages,
retain exact tie counts and whether the anchor is in the maximizer set, as well
as first-index misses. The complete score vectors remain in the source archive;
no tolerance changes ties. Logits/logP/P differences are recorded rather
than assumed absent. For each eligible row form two fixed comparisons: the
saved raw winner vs anchor, and the saved final winner vs anchor. Each has
the four component gaps, raw gap, floor correction gap, content gap and final
logit gap. Softmax's common row normalizer cancels in these pair differences.
Verify reconstruction; do not compute the winner after removing a component.

For each comparison record these algebraic reversals with epsilon=1e-8 log
units: T=td_gap+tv_gap is negative below -epsilon while T+rd_gap is positive
above epsilon; T is negative while T+rd_gap+rv_gap is positive; and combined
distance td_gap+rd_gap is negative while all four terms sum positive. Report
these as respectively radial-distance, full-radial and combined-volume two-cell
ranking reversals. They are not causal intervention effects or corrected PCK.
The epsilon leaves near-zero signs unresolved; it is not a tie-breaking rule.

Report each role overall and by scene, both eligible-all and raw-first-miss /
final-first-miss subsets, with explicit denominators. Save all per-row gaps and
fixed quantiles (count/min/max/mean/p50/p90/p99, NumPy linear) for rawgap,
finalgap, T, R=rd+rv, rd, rv, total volume, floor gap and content gap. Also
report stage winner changes and anchor membership counts. No result-selected
subgroups, new task metrics or causal labels.

## Freeze, tests, numerical gates and resources

Snapshot joint protocol/source closure before the first synthetic execution;
retain failed attempts. Synthetic checks cover analytic point components,
correlated SPD/PSD supports, near-axis and arbitrary key rays, rigid/scaled
camera-centered prepared geometry (rotations/scales about the ray origin;
a world translation must also move the camera origin), direct 3D score reconstruction, known radial/volume ranking
examples, finite/FOV boundaries, extreme floor and exact ties. Independently
recompute every saved component matrix and every row in the full population.

All matrices finite, s>0. Component and raw-score reconstruction errors are
normalized by max(1,abs(a),abs(b)) and must be <=1e-9. Gap reconstruction and
two-route row/summary comparisons use rel1e-9/abs1e-8; identities, population,
indices, membership, flags and counts agree exactly. A numerical violation or
incomplete population prevents interpretation; no automatic salvage or retry.

Runtime remains accepted Python, NumPy-first, four CPU threads, deterministic
Torch and TF32 off. Each real diagnostic/audit phase has a 1200s cap including
source verification and final inventory (excluding writing the final completion
seal itself). Saved component NPZ total <=512 MiB. Persist finished-pair records,
checks and partial rows, failed pair/index, and unattempted suffix on failure.
Logs are outside sealed output folders. Root verifies complete dual decomposition
and provenance before interpreting results. Performance claims, scientific gate
pass and overall goal completion stay false for every diagnostic outcome.
