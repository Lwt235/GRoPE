# Round 10 iteration review: closure, normalization, and point targets

2026-10-30. Development-only reasoning after A's audited failure, before reading
any B result. No new experiment input/label/metric was opened for this review.
It uses the already reported A outcomes: full beats quad16 by 2.76% in 8→8, loses
3.91% in 16→8, loses 0.613 percentage point PCK@8, and fails the frozen combined
accuracy/cost gate. These differences suggest a mechanism to test; they do not
establish its cause. A remains failed under its original gates.

## Most useful diagnosis before another design

Test three distinct error sources on old calibration observations, with all
existing rows retained and sensor quantities confined to evaluation:

1. **Single-Gaussian closure.** A's projected pixel cloud is already available.
   Compare its one-Gaussian representation with the fixed union of base8 Gaussian
   supports and with a dense empirical projected-pixel reference, under the same
   smoothing, key model, and content settings. This isolates discarded spatial
   shape. The empirical reference is an approximation diagnostic of predicted
   geometry; only independent sensor projection remains task truth.
2. **Mean/projection error.** Separately report predicted regional mean versus
   sensor-visible regional centroid, predicted mean versus the independently
   projected query center, and projection-of-3D-moments versus dense-projection
   moments. A changed the last approximation, so coarse degradation cannot simply
   be blamed on the old Jacobian approximation. No covariance change can correct
   a systematic wrong mean by itself.
3. **Order of normalization.** Keep the same atomic scores and compare conditional
   row averaging with unnormalized-score averaging. Their difference isolates
   changing source weights, rather than claiming that an additional mixture
   automatically provides partition invariance. Also hold atomic content fixed:
   averaging content then exponentiating is another nonlinear approximation.

Depth spread, the fraction of predicted points outside the frame, and disagreement
between dense empirical transfer and its Gaussian approximation can be reported
continuously. If subgroups are used, derive their boundaries from calibration
metadata/geometry only, freeze them, and still report the complete population.
Do not infer multimodality merely from a large covariance or high anisotropy.

## One lean update worth testing if closure is the cause

Retain a fixed base8 Gaussian for each observed atomic support. A larger region
is the union of its atoms, without fitting the union back to a single Gaussian.
For the present partitions this keeps one, two, or four components per query;
there is no learned split network, confidence head, or scene-specific component
selection. This changes the representation to a bounded Gaussian mixture, which
must be stated explicitly. It does not establish that a single Gaussian was
sufficient or that its mathematics is new.

Let `κ_ab > 0` be the fixed atomic geometry/content affinity and `m_a,m_b` the
known source and key areas. Define

`Z_a = Σ_b m_b κ_ab`,  `P_ab = m_b κ_ab / Z_a`.

For query region A and key region B use

`P_AB = Σ_(a∈A) (m_a / M_A) Σ_(b∈B) P_ab`,  `M_A = Σ_(a∈A) m_a`.

Normalize once against the fixed atomic key bank, then group rows/columns. This
preserves source-area conditional aggregation exactly for unions of the retained
atoms. It does not recover detail when a coarse key probability is subsequently
redistributed uniformly across its atoms, as in the existing evaluation. It also
does not guarantee equality to sensor-visible conditional weights or correct
geometry. Changing the underlying atomization still changes the approximation.

In contrast, summing unnormalized atom scores and then normalizing a merged query
weights its atoms by `m_a Z_a`. That is a coherent joint-measure construction but
does not equal the protocol's source-area conditional average. The two operators
must not be interchanged or called the same invariance result.

This exact composition is the mechanism probe. Include the same composition of
center/point and empirical quadrature controls, with identical atomic content and
mass. Otherwise, improved pooling could be misattributed to Gaussian covariance.
Count the cost of retaining/processing the atomic bank even when the final token
bank is coarse. Reusing an atomic matrix across nine evaluation conditions is not
a legitimate speed credit for an application that requests only one condition.

## A finite positive PE is possible, but compression comes later

If a positive kernel approximation has `κ_ab ≈ φ(G_a)ᵀ ψ(G_b)` with nonnegative
features, the same conditional composition is implemented by

`S = Σ_b m_b ψ(G_b)`,

`q_A = Σ_(a∈A) (m_a/M_A) φ(G_a)/(φ(G_a)ᵀ S)`,

`k_B = Σ_(b∈B) m_b ψ(G_b)`,  `P_AB ≈ q_Aᵀ k_B`.

The query is conditioned on the key bank through S. Key mass is already in k_B;
adding another log-mass correction would double count it. Applying another
softmax or L2 normalization after grouping destroys this identity.

A concrete positive geometric feature is a density evaluated at fixed spatial
quadrature landmarks: `φ_l(G)=sqrt(w_l) N(z_l; μ, Σ+τ²I)`, with `w_l>0`.
Its dot product approximates the overlap integral through positive quadrature,
and mixtures compose linearly. This is established density/kernel quadrature,
not a new Gaussian characteristic function or new PE theorem. Sharp localized
supports may require many landmarks; translation coverage, memory and numerical
underflow are real limitations. It may be inferior to the exact small mixture.
Content interaction must also remain at atom level or receive its own justified
factorization; appending these features to newly pooled content does not preserve
the atomic operator automatically.

Accordingly, do not implement this approximation merely to obtain a finite map.
First require the exact composition to satisfy the same strong sampling,
localization and cost tests. It is not a reason to repeat the already negative
large Fourier/BC feature experiments under a different name.

## Does regional mass conflict with center localization?

Yes, in some individual rows these targets differ: a query center can lie on a
small foreground region while most query area projects elsewhere. Exact regional
transfer can then have its largest probability outside the center's PCK radius.
This explains a possible tradeoff, not permission to relax the frozen protection.

A useful calibration-only oracle diagnostic is to evaluate PCK from the argmax
of the independent area-count target itself. Also calculate the minimum extra
regional KL needed to make a PCK-correct key maximal. For a fixed acceptable key
j and truth distribution y, the KL-minimizing constrained distribution leaves
unaffected probabilities unchanged and pools `y_j` with the largest probabilities
that exceed their pooled mean. Set those pooled entries to their mean. The cost
is `Σ_(i in pool) y_i log(y_i / mean_pool)`. Minimize over acceptable keys and
report the tolerance required to make a strict, rather than tied, argmax.

This gives a diagnostic tradeoff bound, not a deployable model and not a proof
that the aggregate gates are impossible. The bound applies to the individual
fine-query area target; it is not automatically a lower bound for Round 10's
coarse-parent macro KL, where child errors can compensate. Three synthetic
three-class cases were
checked against independent constrained SLSQP optimization with agreement below
10⁻⁹; actual data have not been evaluated for this review. If severe conflict is
observed, preserve the historical fail and explain that a universal claim needs
a representation/reading rule that supports both semantics. Do not silently
change center targets to regional centroids or add a second head just to pass.

## Decision discipline and prior work

Candidate B first tests the declared key-cell approximation. If B does not clear
the gates, a new composition experiment needs a separate frozen development
protocol. Its calibration diagnosis should show a reproducible coarse-query
closure error and an improvement on all three calibration scans before a new
gate attempt. Keep the original 2%/cost/localization requirements, stronger
composition controls, complete per-condition reporting, and genuinely untouched
confirmation data. A diagnosis without a deployable gain is still a valid
negative result; no sequence of trials guarantees a positive conclusion.

The integral/discretization perspective is established by [Continuum Attention
for Neural Operators](https://www.jmlr.org/papers/v26/24-0879.html). Its theory
does not validate this Gaussian design or its real RGB-D effectiveness. [Token
Merging, Eq. 1](https://arxiv.org/html/2210.09461v3) already uses log token size
for proportional attention; that duplicate-mass correction alone does not retain
the full within-region measure or atomic score variation. The composition and
conditional-normalization identities above are direct algebraic deductions,
not claims of a newly discovered kernel. The existing [Round 6 literature
audit](round6_literature_design.md) documents the additional Gaussian-overlap,
expected-feature, and Gaussian-token precedents.
