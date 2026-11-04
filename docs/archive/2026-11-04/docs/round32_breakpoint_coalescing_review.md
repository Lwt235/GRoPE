# R32 prospective partition coalescing review

2026-10-31. This is a post-failure diagnosis and proposed numerical-reference
revision. R31 remains the accepted 44/46 numerical failure. No R32 integration,
model call or implementation has been executed in this review.

The authenticated metadata diagnostic is
[diagnostic.json](/home/liwt/GRoPE/results/round31_midpoint_post_failure_diagnostic_20261031/diagnostic.json).
Its executable record extracts only the frozen `prepare` and `outer_breaks`
ASTs, verifies current and dual-snapshot hashes, reads input values from the
sealed synthetic JSON descriptor, and prohibits NPZ decoding. It binds the
complete source descriptor, all four input hashes, source archive byte hash,
accepted records and source metadata. Integration/model calls and NPZ decodes
are all zero. This is post-execution diagnosis, not a new preregistration.

For `07_radial_noncentral`, the input is mu=(1.2,-0.8,2), with the sole nonzero
factor column (0.6,-0.4,1). Independent SVD conditioning gives m=0.8944271909999159,
sigma=0.4472135954999579, sx=sy=.25, and lower/upper depth coordinates [-2,18].
There are 16 interior roots and 17 planned initial intervals. Two intervals
contain adjacent float64 numbers:

| Left | Right | Exact gap | Rounded midpoint |
|---|---|---|---|
| -1.650614378515658 | -1.6506143785156577 | 2^-52 | left endpoint |
| 9.180339887498947 | 9.180339887498949 | 2^-49 | right endpoint |

`nextafter(left,right)==right` in both cases. The alternative expression
left+(right-left)/2 has the same problem; a different midpoint formula cannot
create an interior float. The first pair comes from Y(edge=-2,level=-1) and
X(edge=-1,level=-1); the second from X(edge=1,level=8) and Y(edge=0,level=8).
Their independently evaluated root arithmetic differs by one ULP. Both
orders fail before the first pilot or charged inner node, exactly as recorded.
Other close root pairs possess strict midpoints and should remain untouched.

The proposed rule keeps the two endpoints exactly. Walk the sorted interior
roots left to right; keep a root only when it and the last retained edge have
a strict representable midpoint. Otherwise omit that partition boundary.
Before appending the upper endpoint, repeatedly pop the final interior root
if its interval with the upper endpoint has no strict midpoint. If even the
two preserved endpoints have no strict midpoint, fail. The same midpoint
formula and exactly the same retained edges must define pilots and quad
points. There is no epsilon, rank threshold or probability threshold.

For this case, the rule is expected to omit the right member of each listed
pair, giving 14 interior roots and 15 planned pilot intervals. This is a
consequence of the rule and saved float ordering, not a measured R32 outcome.
Save original, omitted and retained edges, with reason and original index;
the original source descriptor and the depth endpoints stay byte-identical.

Removing a boundary merges adjacent integration intervals. It does not remove
an interval from the integration domain: the retained partition still has
union [-2,18], and the integrand is unchanged. The existing physical depth-tail
bound therefore remains the only truncation bound. No probability mass is
declared zero and no new tail-error exemption is introduced.

This does not prove quadrature accuracy. A smooth positive function can vary
sharply across an adjacent-float neighborhood when finite coefficients are
large. Merging a hint can change adaptive subdivision and the midpoint pilot
scale; the latter remains an estimate, not a bound. Overflow, underflowed
positive final integrals, failed convergence, old-oracle disagreement or any
old tail/raw/P gate must still fail. Initial pilot count cannot increase,
and the coalescing walk is linear after sorting, but adaptive work and total
wall time can increase. Compared with R31's actual zero work for these calls,
R32 would perform new work; it must fit the unchanged whole-phase budget.

Before a separate freeze, the minimal distinct toy coverage should include:

- An adjacent-float interior pair is coalesced, while a two-ULP pair with a
  strict midpoint is retained.
- A chain of close roots, negative roots and subnormal roots terminates and
  leaves every retained interval with a strict midpoint.
- A last interior root adjacent to the upper endpoint is popped, preserving
  both original endpoints and the entire domain.
- Already valid partitions are unchanged; applying the rule twice is
  idempotent; all retained values come from the original edge set.
- An adjacent-float whole domain still fails instead of changing an endpoint.
- Saved pilots are precisely the retained-interval midpoints, and quad uses
  precisely the retained interior roots. Independent arithmetic checks the
  original/omitted/retained evidence.

All original orders, error gates, node limits, depth/interval limits and phase
budgets remain required. A future R32 pass would support only the fixed
synthetic reference validation and would not promote a geometric or efficacy
candidate or establish general real-input accuracy.
