# R35 fixed real-reference protocol

Follow [the R35 design](docs/round35_log_real_reference_design.md).

Same72 accepted R30 inputs and144 scheduled calls as R33; use unchanged R34
reference, all R33 real-phase/per-call budgets, accuracy gates and fail-fast.
No geometry/model/anchor/content choice, label or fresh-scene access.

Joint metadata/source freeze -> exact input copies -> independent72-input
audit -> globally sealed144 outcomes -> independent254 comparisons -> root
saved-arithmetic replay. Never decode old output arrays before global seal.

The original216 checks remain intact;38 additional strict comparisons use
only all19 fully successful R33 references, separately keyed by source/order.
Partial/skipped old outputs are not oracles. All failures and unattempted
outcomes remain in the fixed denominators. Input corruption is a delivery
failure. Complete delivery may accept a coherent numerical FAIL; numerical
PASS requires all checks. No scientific-goal completion follows from this
numerical preflight alone.
