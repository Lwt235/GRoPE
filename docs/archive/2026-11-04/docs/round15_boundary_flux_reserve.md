# Round15 boundary-flux reserve: correct identity, rejected direct replacement

2026-10-30. Pure mathematical and synthetic screening requested while the
current Cartesian GL24 operator remains fixed. No real observations, labels,
Stage B scores, or validation results were read. This is not a new candidate
or a change to any numerical threshold. The previous angular failure remains
archived. **Do not replace GL24 with the direct boundary-flux rule below.**

## The smooth identity does remove the old angular singularity

Let `d=x−μ`, `A=S⁻¹`, `q=dᵀAd`, and `D=det(S)>0`. With

\[
 p(x)=\frac{e^{-q^2/(4\pi)}}{\pi^2\sqrt D},\qquad
 F(x)=\frac{d}{2\pi\sqrt D}\frac{\operatorname{erf}(q/(2\sqrt\pi))}{q},
\]

`div(d f(q))=2f(q)+2q f′(q)`. Since
`d erf(q/(2√π))/dq = exp(−q²/(4π))/π`, this gives `div F=p`.
The exact rectangle mass is therefore the sum of four outward normal flux
integrals. It is additive under key subdivision; interior edges cancel.
This is an identity for the same quartic density, not a new position encoding.

The apparent singularity is removable:

\[
 \frac{\operatorname{erf}(q/(2\sqrt\pi))}{q}
 =\frac1\pi\left(1-\frac{q^2}{12\pi}
                    +\frac{q^4}{160\pi^2}+O(q^6)\right).
\]

Thus `F=d/(2π²√D)+O(‖d‖⁵)` is smooth at the center. Physical rectangle
edges stay fixed when μ moves: there are no ray intersections, sorted angles,
or moving angular endpoints. A safe finite series and a nonzero denominator
in the unused branch also avoid `where` producing `0/0` during autograd.

Seven synthetic gradient cases were checked against the existing independent
Cartesian adaptive reference: generic, exact edge, exact corner, either side
of an edge at distance `10⁻¹⁰`, a rotated thin covariance, and μ exactly on a
GL32 edge node (`q=0`). For μ, the three symmetric covariance directions,
and h, maximum absolute error was **9.98×10⁻¹¹**. The exact-zero-node case
was finite, with error **6.15×10⁻¹¹**. This supports the proposed repair of
the old boundary derivative mechanism on these cases; it is not a general
gradient certificate over arbitrary covariance condition numbers.

## Fixed nodes can resolve local values but cannot repair tail cancellation

The probe reused only the **synthetic case definitions and independent
reference functions** of `experiments/round15_quartic_numerical_audit.py`:
120 rectangles, point/isotropic/rotated covariances, h=.25/.5, sides1/2/3,
centers inside/on edges/on corners/outside/far, including a covariance
eigenvalue16384. No existing experiment array was opened.

Here are maximum raw errors; all values are signed and unclipped. “Segments”
splits each exterior edge into pieces of length≤1 before fixed GL integration.
It does not introduce extra keys or change the density.

| GL order | Unsegmented four edges | Segmented exterior edges |
|---:|---:|---:|
|16|1.3410×10⁻⁵|3.3127×10⁻¹³|
|24|8.7425×10⁻⁸|7.5634×10⁻¹⁶|
|32|5.1718×10⁻¹⁰|7.5807×10⁻¹⁶|
|48|5.4623×10⁻¹⁴|9.9920×10⁻¹⁶|

Segmented GL16 maximum four-child additivity defect was3.02×10⁻¹³.
This looks attractive locally: a unit square uses4n boundary evaluations
rather than24² Cartesian evaluations. For side3, segmented GL24 uses288
boundary nodes versus5184 Cartesian nodes across nine unit tiles. These are
node counts, **not measured speedups**; accuracy failed before a cost benchmark
would be meaningful.

When q is large, erf rounds to1. Each edge then integrates an algebraic
field whose signed flux is nearly zero outside the center. A positive
exponentially tiny cell mass must be recovered by subtracting comparatively
large edge contributions. Quadrature convergence cannot restore information
already lost in floating-point subtraction.

Concrete counterexample, in reference-eight units:

- `μ=(−1.35,.4)`, `C=0`, `h=.25`, `S=.0625 I`;
- rectangle `[−.0625,−.0625,.9375,.9375]`;
- independent positive Cartesian mass **9.62253985×10⁻²⁸**;
- absolute GL24 edge flux sum **.2356601265**;
- signed GL24 result **1.04083409×10⁻¹⁷**.

For μx=−1.4, true mass8.24389×10⁻³² becomes **−2.08167×10⁻¹⁷**.
Negative numerical masses also occur in remote cells when the center is
inside the FOV. The existing area floor is about10⁻³⁰ per unit cell, so it
cannot absorb noise of this size.

Four synthetic whole-FOV cases were evaluated on all432 native8 cells at
each of GL16/24/32/48. The tail-FOV reference total is6.88824524×10⁻²¹.
GL24 gives total **−1.73472348×10⁻¹⁸**, with185 negative cells. Merely
clipping negative masses before applying the unchanged floor still leaves
maximum normalized probability error **.75091679**, versus the fixed10⁻⁴
limit. Orders16/32/48 also fail. An underflow-FOV case whose correct output
is entirely the area floor produces arbitrary nonuniform probabilities;
after diagnostic clipping, GL24 maximum error is.01184. No such clipping
was adopted. All failed arrays and signed values are retained.

## Why `erfc + winding` is not a ready fix

Writing `erf=1−erfc` gives, formally,

\[
 M(R)=W_S(R,\mu)
 -\oint_{\partial R}\frac{d\cdot n}{2\pi\sqrt D\,q}
       \operatorname{erfc}(q/(2\sqrt\pi))\,ds.
\]

For μ strictly inside the rectangle `W=1`; strictly outside `W=0`.
At an edge `W=1/2`. At a corner it is the interior angle **after whitening**
divided by2π, not always1/4. At the lower-left corner, with
`ρ=Sxy/√(Sxx Syy)`, `W=acos(−ρ)/(2π)`.
These values follow by excluding a small whitened disk before applying the
divergence theorem; the singular algebraic field has unit distributional
divergence at μ. They are not an ordinary smooth indicator.

The outside-only erfc formula reproduces the example9.62254×10⁻²⁸ mass
accurately without subtracting the algebraic field. However, the new field
has a true1/‖d‖ singularity. At distance ε from an edge its normal flux
contains a peak of width proportional to|ε|. Fixed edge nodes miss that
peak as ε→0, and the missed integral is precisely what should cancel the
jump in W.

For the unit rectangle above, `C=[[.7,.18],[.18,.4]]`, h=.5, and
`μ=(−.0625+ε,.4)`, GL32 yields:

| ε | Independent mass | Correct W minus fixed-node erfc flux |
|---:|---:|---:|
|−10⁻¹⁰|.128681212056|−.371318783446|
|0|.128681212057|.128681212057|
|+10⁻¹⁰|.128681212059|.628681207561|

The smooth-erf calculation remains accurate at all three positions. The
erfc repair therefore reintroduces nonuniform near-boundary convergence;
the exact edge value alone hides an approximately half-unit error on either
side. Its finite-node derivative cannot be accepted as the derivative of
the true mass. For the same correlated S at a lower-left corner, W=.2867830;
substituting1/4 already gives a .0367830 mass error even with an otherwise
accurate erfc integral.

An analytic cancellation, a carefully bounded near/far hybrid, or adaptive
integration might eventually combine the two forms. This screening does
**not** prove such methods impossible. It does show that neither one smooth
erf boundary rule nor a global erfc-plus-winding substitution satisfies the
present positive-mass, normalized-tail, and boundary-gradient contract.
Designing and certifying extra dispatch/cancellation rules would be a new
numerical project. No hybrid, new threshold, fallback, or effect experiment
was implemented here. Stop this reserve route at this point; retain GL24.

## Reproducible evidence

`results/round15_boundary_flux_reserve_v1/synthetic_probe.py` contains the
standalone synthetic prototype, its pre-run source/reference hashes, raw
rectangle/additivity records,16 FOV records and saved432-cell arrays,
tail examples, and six original gradient records. Supplemental
`erfc_boundary_counterexample.json` and `exact_zero_node.json` record the
subsequent explicit mechanism checks described above. The latter two were
exploratory mathematical follow-ups, not falsely predeclared audit cases.
No production model, frozen experiment, or current validation draft changed.
