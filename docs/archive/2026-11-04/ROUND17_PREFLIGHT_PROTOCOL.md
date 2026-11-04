# R17 trace-temperature numerical preflight

This is a new development candidate after R16, not a repair of any historical
scientific result. Before any new prediction, jointly snapshot the new model,
independent numerical implementation, this runner, their tests, this contract
and the transitive frozen preparation/integration dependencies. Do not edit
that snapshot or overwrite completed/partial stages. Synthetic tests may be
developed before this freeze; measured preflight artifacts start afterward.

The unique new operation is tau = T h^2 / (h^2 + trace(C)), with C the full
query-support covariance in reference-8 coordinates. Evaluate the formula in
float64 with the common scale s=max(h²,C00,C11): r=h²/s,
tau=T*r/(r+C00/s+C11/s), where the two covariance terms are summed first.
This avoids overflow without clipping or changing the mathematical formula.
Cast tau to float32, multiply float32 pooled cosine content in
float32, then add to the original float64 Gaussian GL32 native-cell log mass.
Apply the existing 1e-30 * native reference-cell area floor once, after child
mass aggregation. The inherited 24px cell split and all geometry stay fixed.
T=0 and point C=0 must preserve their respective old probability paths.
The denominator is support size, not an estimated noise confidence.

Run the separately implemented fixed synthetic suite first. Then inspect only
the first lexicographic old TRAIN pair from each of harvard_c3/c5/c6, using
the four permitted forward fields. No center/regional/sensor/VAL/fresh labels
are accessed. Use all fourteen old partition/scale conditions, four widths
(.25,.5,1,2), three temperatures (0,5,10), and eight evenly spaced native query
indices per partition. This gives 504 probability matrices / 4,032 query rows
and 168 geometry matrices / 1,344 geometry rows. Retain all selected queries,
including out-of-FOV, raw-zero and tiny-FOV cases and area floors.

The reference independently reconstructs support and content in NumPy from
the old observation fields. Integrate its Gaussian over original native key
rectangles with independent SciPy/NumPy GL64 and GL96. A second, hybrid
reference uses production float32 support/content but the independent GL96
integrator and temperature implementation, separating numerical integration
from support preparation. Reference content is cast to float32 before the
new bias calculation, matching its specified arithmetic. Covariance is not
rounded before independent-reference integration or trace calculation.

Preserve the old numerical acceptance tolerances: maximum normalized
probability absolute error <=1e-4, hybrid error <=1e-4, GL64/96 convergence
<=1e-8, relative-normalized mean/cov preparation error <=1e-4 and absolute
content preparation error <=1e-5. Check finite nonnegative normalized
probabilities at 1e-12. Record every argmax disagreement and its margin;
these numerical differences are not ground-truth PCK scores. Do not silently
relax tolerances or change the sample after a failed run. Full effective
temperature and independently reconstructed bias agree to their prescribed
arithmetic; point/T0 equivalence, derivative, rotation/unit tests are checked
in the frozen synthetic suite. Neither numerical pass grants a scientific
pass or authorizes untouched data.

Both numerical stages must pass before the separate R17 Stage A effect
experiment. All old candidate failures, R15 stopping decision and R16
diagnosis remain unchanged. Further quality, cost, robustness, actual-value,
GS and untouched-confirmation checks remain required.
