# R16: complete saved-probability mechanism diagnosis

This is an exploratory development diagnosis after the fully audited R15
training result and the separately recorded post-result futility decision.
The three hypotheses were described in
`docs/round15_pre_result_mechanism_hypotheses.md`; this detailed analysis plan
is fixed now, after the aggregate R15 outcomes are known. It is not an
independent confirmation or a newly selected candidate.

Reuse all 5,184 sealed Stage A matrices: three families, twelve original h/T
configurations, three query scales, and exactly 48 original training pairs.
These represent 28,224 static query identities and 1,016,064 query/configuration
states. There is no new probability operator, prediction, h/T search, source
center construction, validation run, training or fresh-data access. Invalid
center labels stay invalid, not incorrect. Geometry includes every query.

For every state compare its actual native winner with the same family/h T0
winner. Retain stable first-index ties, runner-up, top geometry log gap,
winner-to-mean distances, winner displacement, exact native PCK and its four
correctness transitions. For winner jT versus geometry winner j0, retain
`penalty=log P0[j0]-log P0[jT]`, the actual float32 `T*content` values converted
to double before differencing, and `advantage=log PT[jT]-log PT[j0]`.
The identity `advantage=gain-penalty` is a numerical check, not causal proof.
Check the full reconstructed PT from P0 and this bias to absolute 1e-12;
identity residual must be <=1e-10. P0 includes the already frozen native-area
floor; normalized P0 cannot identify raw tiny-FOV rows.

Preserve actual shared mu/C/content using the already frozen and previously
audited Stage A preparation and four allowed observation fields. This shared
dependency is explicit: the new independent audit independently implements
all NEW diagnostic math and aggregation, not the old moment preparation.
Record mu grid phase/FOV status, S=C+h²I condition number/major-axis angle and
mu-to-target distance on the original valid population. Equal eigenvalues
receive angle zero plus an explicit isotropic flag. No learned bins, tuned
thresholds, cherry-picked queries or new anchor are introduced.
Here C is the shared full-support geometry used as diagnostic context for all
three families. Point kernels still execute C=0; their context condition
number/angle must never be described as their actual kernel covariance.

Pair quartic_full with quartic_point and Gaussian point at the same h/T.
Quartic full versus quartic point changes covariance; quartic point versus
Gaussian point changes radial shape at C=0. There is no Gaussian full in the
complete Stage A population, so these data cannot isolate full-covariance
radial shape or prove that continuous quartic flatness caused every failure.

Save all per-query derived arrays, complete per-pair additive statistics and
paired correctness tables. Pool within each scene/scale/configuration and
then equally weight the three scenes. Keep scales separate. Report geometric
statistics on all queries and separately on valid centers; PCK always uses
the valid-center denominator. Empty derived means are null. Invalid per-query
target distances may be NaN in NPZ, with exact validity masks; JSON statistics
must remain finite. The exact PCK counts and argmax hashes must reproduce all
5,184 original Stage A evaluation rows. Harm/benefit counts must close the
integer PCK differences in both temperature and family contrasts.

A joint source lock and snapshot bind producer, separate independent math,
artifact auditor and tests BEFORE extracting any new diagnostics. This
read-only reanalysis has no new forward-before-label phase: the upstream
probabilities and labels already exist under their original complete seals.
Bind all upstream files by hash and retain R15's unchanged selection and added
futility decision. Every original probability is rehashed before analysis.
The independent audit rereads all matrices and reconstructs every new query
array/statistic, the whole population, provenance and chronological chain.
Integer/bool identities are exact; floating diagnostic arrays use 1e-10
absolute tolerance, additive sums 1e-8 absolute plus 1e-12 relative tolerance,
and derived summaries 1e-10. These are audit tolerances, not effect margins.

No scientific acceptance gate or winner is defined here. Interpret whether
temperature harms, T0 geometry ordering, and common mean error distinguish
the previously stated hypotheses; correlations and algebraic identities alone
do not establish causality. A changed design needs a separate frozen protocol
and all applicable inherited numerical, localization, regional, robustness,
cost, actual-value, actual-GS and untouched-confirmation checks.
