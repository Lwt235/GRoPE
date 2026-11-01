# Fixed Student-t4 versus ray pushforward: unselected route review

2026-10-31. Source/document review and algebra only. No real arrays, new
forward, labels or effect calculations; no R22/R23 source change. The running
R23 population must finish and be independently accepted before interpreting
its complete failure/cost profile. Nothing here selects an efficacy candidate.

## This mechanism was considered before

[R15's fixed t3 reserve](round15_student_t_reserve.md) already derived a
covariance-matched Student-t, checked six synthetic rectangles, and gave
uniform-region and bounded-content interference counterexamples. It expressly
did not enter real-data experiments. [The R18 review](round18_theoretical_options_draft.md)
reused that conclusion. Searches of current models/experiments/tests found no
Student-t production family or real-data round. Thus t4 is a new fixed shape
choice within an old reserved mechanism, not an undiscovered design principle.
R19's legal center a improves the information available relative to that old
area-mean proposal; it does not establish that the remaining shape is t-like.

## Correct formula and a useful counterexample

Let S=C_a+h²I, q=(y-a)^T S^-1(y-a), and use Student scale Sigma=S/2.
The standard multivariate-t density uses a shape matrix, as stated in the
[official SciPy definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_t.html).
In two dimensions, fixing nu=4 gives

    p4(y) = 1/(pi sqrt(det S)) * (1+q/2)^(-3).

If d² denotes Mahalanobis distance using Sigma, then d²=2q and the proposed
potential is -3 log(1+d²/4). Using S directly in d² without this factor would
double the intended covariance. Native PE must integrate the normalized
density over each original cell, then add one area floor and the declared
content bias. A center-sampled potential is a different operator. Its density
normalizer cannot be dropped before a fixed absolute floor.

Direct radial integration gives E[q]=2, hence covariance S. The center density
is twice the same-S Gaussian's; the log-density Hessian is -3 S^-1. Its
elliptical-window mass is `1-(1+R²/2)^(-2)`: at R=1 this is 5/9, above the
Gaussian value `1-exp(-1/2)`, while at R=3 it is 117/121, below
`1-exp(-9/2)`. No window-mass dominance follows from peak plus tails.

There is an exact same-covariance regional counterexample without running a
new integral. For S=I, the density ratio to Gaussian at q=2 is e/4<1. Choose
a sufficiently thin uniform annulus with squared radii 2-epsilon and
2+epsilon. Its covariance is exactly I, but throughout that annulus t4 assigns
less density than Gaussian. Its KL to t4 is therefore worse; the excess tends
to log(4)-1 as epsilon tends to zero. This does not predict our patch metrics;
it disproves a general region advantage. t4 still cannot recover asymmetric,
multimodal or curved shape discarded by one center and matrix.

Nu=4 has finite variance but divergent fourth moment (the radial fourth-moment
integrand has a logarithmically divergent tail). It has no special finite-
fourth-moment justification over the earlier nu=3 choice. Fixed nu prevents a
new scan but remains a disclosed shape prior. Increasing h²I matches covariance;
it is not convolution with independent Gaussian blur. In particular C_a=0
gives a t4 point kernel, not the old Gaussian point kernel. Finite FOV further
changes the normalized moments. Local center curvature does not guarantee a
native-cell winner, and heavy tails can strengthen remote bounded-content
competitors. A global linear-content exponential tilt is not normalizable on
the infinite plane for t4, so Gaussian linear-tilt mean laws cannot be borrowed.

## Fair comparison and numerical paths

The clean geometry contrast is t4 versus Gaussian with identical a, C_a, h,
native bounds, floor and exactly the same content bias array. Keep old strong
and legacy controls, and include a same-a t4 point control: otherwise gains
from changing the point kernel could be mislabeled as shape utility. Point
controls must not silently pay for or borrow full C_a; any shared full-shape
temperature diagnostic must declare that extra input/cost. No GT anchor or
borrowed point winner is justified. Fixed nu is not permission to sweep h/T.

The old quartic rectangle infrastructure can supply physical tiling, GL
nodes, source/seal machinery and the comparison schema. Its density-specific
reference, tested order and passing results cannot be reused as t4 validation.
The sharper t4 core may need different accuracy work even with the same h.
A mathematically different one-dimensional reference is available. Set
sx=sqrt(Sxx), u=(x-a_x)/sx, D=Syy-Syx²/Sxx. Then

    p_x(x)=3/(4 sqrt(2) sx) * (1+u²/2)^(-5/2),
    Y|x ~ t5(location=a_y+Syx/Sxx*(x-a_x), scale²=D*(2+u²)/5).

Integrate p_x times the two conditional t5 CDF boundary difference over x.
This follows by dividing joint and marginal densities and can independently
check a direct Cartesian t4 implementation. Stable tail differences, extreme
anisotropy/correlation, tiny raw FOV, native additivity, point/affine identities,
floor-normalized probabilities and actual required gradients still need a
new frozen numerical audit. Two GL orders alone are convergence evidence,
not a uniform-domain certificate. No new numerical rule or order is selected
here; no speed comparison has been executed.

## Route judgment

There is no obligation to keep optimizing an exact ray law merely because
it has a clean definition. The user's goal is a useful compact map from GS
parameters to PE, and a same-parameter elliptical density is a legitimate
alternative modeling hypothesis. It avoids the ray method's additional
depth integral and does not require recovering 3D inputs from old 2D moments.
That makes a lower-cost implementation plausible, not established against
the old quad16 cost gate. Completed R23 evidence, rather than partial rows,
must determine the status of the unchanged ray implementation.

My recommendation is to retain fixed t4 as a **cheap-to-falsify reserve**, not
promote it from the claim “sharp center plus heavy tails.” R20 already passed
center protection; R21's full4 shape benefit is not evidence for a single
radial heavy-tail family. Any later selection should state a joint region/
center hypothesis and keep every old gate. A bounded numerical check would
be justified before a real candidate comparison; another expensive ray
repair is not scientifically required first. Neither branch has demonstrated
actual V, optimized-GS or fresh-scene success. Unlike an anchor-only change,
t4 changes a GS footprint even when a=mu and T=0, but its non-Gaussian tails
are a prior rather than the exact projection of that Gaussian. This is both
its possible source of effect and its principal interpretation risk.
