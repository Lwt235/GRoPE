# R19 center-semantics design review

2026-10-31. Independent **design-only** review of
[the R19 protocol](/home/liwt/GRoPE/ROUND19_CENTER_SEMANTICS_PROTOCOL.md) and
[preparation draft](/home/liwt/GRoPE/docs/round19_center_semantics_preparation_draft.md).
No new prediction or target arrays were opened, no effects were calculated, and
no producer/auditor implementation or execution is certified by this note.

**Conditional design PASS:** this is a defensible, newly declared exploratory
TRAIN screen for whether source-center identity merits another experiment.
It does not weaken, replace or retroactively pass the failed R17 scientific
gates or R18 numerical gate. The precise new thresholds are chosen now, before
these diagnostic effects, rather than described as historical preregistration.
Previously used TRAIN data remain development data; the global forward seal
provides chronology for this new computation, not a claim of previously unseen
labels or independent confirmation.

The decision must preserve all four conditions: q24 equal-scene PCK8 gain at
least `1/200`, strictly positive q24 gain in **each** of the three scenes, and
q8 and q16 equal-scene gains each at least `-1/200`. Within each scene, pool the
original valid rows and retain integer correct/valid counts; average the three
scene Fractions equally. Test exact rational differences, never rounded
percentages, a floating epsilon, pooled scene counts, or a row-level confidence
interval. The q8/q16 protections are macro protections; they do not secretly
impose or certify per-scene noninferiority. The 0.5 pp rule is a fixed resource
screen, not a significance claim from three dependent development scenes.

The main input and label boundaries are substantive:

- Derive `a` only from the bound RGB-predicted depth, intrinsics, camera
  transform and original query bounds. Preserve the existing interpolation,
  sanitization, clamps and projection conventions. No sensor depth, target
  coordinate, old success flag or fixed R17 winner may determine `a`, its
  nearest cell, or a fallback. A predictor/cache provenance mismatch must stop
  the run, not be repaired by substituting another input.
- Use all 48 locked pairs and all q8/q16/q24 rows. Convert old reference-8
  `mu` to pixels exactly once; use the same native g8 key centers and first
  minimum tie rule for both locations, including exterior projections. The
  original center-valid mask is an evaluation mask only: front/FOV, regional
  validity and old correctness must not narrow it. Exact R13 targets, including
  g8/g16, avoid silently changing a boundary-sensitive PCK population.
- Seal every new forward array and the fixed comparator identities globally;
  independently pass the forward audit before hashing or decoding actual
  target archives. Freeze-time expected hashes are metadata, not permission
  to read targets. Then audit all counts, identities and exact decisions.
  A failed audit is not a negative scientific result. Prior accepted artifacts
  provide inherited provenance, not a new independent integration audit.

The fixed R17 winner, continuous center errors, nearest-cell disagreement,
front/FOV flags and complete benefit/harm transitions are explanatory outputs.
Neither their subgroups nor a different old configuration may rescue a failed
screen. Nearest-cell PCK can change through quantization; even a passed screen
does not establish a better content-aware point localizer or better Gaussian
probabilities. Reporting continuous errors alongside all scene/scale counts
helps expose this distinction without changing the decision rule.

A pass authorizes only consideration of a separately frozen candidate with
`mean=a` and `C_a=C+(mu-a)(mu-a)^T`, using consistent units. This is the original
support's second moment **about a**, not its centered covariance or an estimate
of anchor confidence. It changes both the mean and spread. Give point and fixed
controls the same `a`; retain strong old controls, all 14 regional conditions,
original center protections, six perturbations and complete costs. A failure
stops this anchor-information branch under the fixed diagnostic, without a
threshold, scene, center-definition or metric rescue.

Mechanistically, `a` supplies designated source-center identity that old region
moments need not retain; it is additional legal input, not information recovered
from `mu,C` alone. Its usefulness on old RGB-depth patches would support that
narrow information-loss explanation. Optimized GS already provides primitive
centers, so such a result alone neither establishes a new GS positional encoding
nor replaces learned-V utility, real-GS evidence and untouched confirmation.
