# R37 fixed real-reference protocol

Follow [the design](docs/round37_component_round_real_reference_design.md).

Same72 accepted R30 inputs and144 scheduled outcomes as R35; unchanged R36
reference and helper, all original real-phase/per-call budgets and gates.
No model, geometry, anchor, content, label or fresh-scene selection.

Joint metadata/source freeze -> exact72 input copies -> independent input
audit -> globally sealed144 outcomes -> independent296 comparisons -> root
saved-arithmetic replay. Old output arrays remain unread until global seal.

Retain216 original comparisons plus80 strict source-keyed regressions from
all19 successful R33 and all21 successful R35 references. Failed partials
and skipped archives are not oracles. Fail-fast retains every outcome and
comparison denominator. IO failure is a delivery failure. A coherent FAIL
can be accepted as evidence; numerical PASS needs all gates and establishes
no scientific-goal completion by itself.
